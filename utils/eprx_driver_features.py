"""EPRX 모집량과 TEPCO 30분 실적 결합 데이터의 설명변수 생성.

통계적 관계나 원인을 계산하지 않고, 확인된 MW 평균값의 산술 파생값과
결합 품질 진단만 제공한다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


JST = "Asia/Tokyo"
JOIN_KEYS = ("delivery_date", "period_no", "period_start")
REQUIRED_COLUMNS = {
    *JOIN_KEYS,
    "procurement_volume",
    "area_demand_mw",
    "solar_generation_mw",
    "wind_generation_mw",
}
FEATURE_COLUMNS = (
    "variable_renewable_generation_mw",
    "residual_demand_mw",
    "variable_renewable_share",
    "demand_change_mw_30min",
    "solar_change_mw_30min",
    "wind_change_mw_30min",
    "variable_renewable_change_mw_30min",
    "residual_demand_change_mw_30min",
)


def _safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.ne(0)))


def _continuous_change(
    values: pd.Series, datetime_jst: pd.Series
) -> pd.Series:
    """직전 행이 정확히 30분 전일 때만 변화량을 계산한다."""
    continuous = datetime_jst.diff().eq(pd.Timedelta(minutes=30))
    return values.diff().where(continuous)


def build_eprx_driver_features(
    merged_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """결합 데이터에서 모집량 분석용 30분 설명변수와 진단을 반환한다."""
    if not isinstance(merged_df, pd.DataFrame):
        raise TypeError("merged_df는 pandas.DataFrame이어야 합니다.")
    missing = sorted(REQUIRED_COLUMNS - set(merged_df.columns))
    if missing:
        raise ValueError(f"파생변수 필수 열 누락: {', '.join(missing)}")

    result = merged_df.copy()
    result["delivery_date"] = pd.to_datetime(
        result["delivery_date"], errors="coerce"
    ).dt.normalize()
    result["period_no"] = pd.to_numeric(
        result["period_no"], errors="coerce"
    ).astype("Int64")
    local_datetime = pd.to_datetime(
        result["delivery_date"].dt.strftime("%Y-%m-%d")
        + " "
        + result["period_start"].astype("string"),
        errors="coerce",
    )
    result["datetime_jst"] = local_datetime.dt.tz_localize(JST)
    result = result.sort_values("datetime_jst", kind="stable").reset_index(drop=True)

    result["variable_renewable_generation_mw"] = (
        result["solar_generation_mw"] + result["wind_generation_mw"]
    )
    result["residual_demand_mw"] = (
        result["area_demand_mw"]
        - result["variable_renewable_generation_mw"]
    )
    result["variable_renewable_share"] = _safe_share(
        result["variable_renewable_generation_mw"], result["area_demand_mw"]
    )

    change_sources = {
        "demand_change_mw_30min": "area_demand_mw",
        "solar_change_mw_30min": "solar_generation_mw",
        "wind_change_mw_30min": "wind_generation_mw",
        "variable_renewable_change_mw_30min": "variable_renewable_generation_mw",
        "residual_demand_change_mw_30min": "residual_demand_mw",
    }
    for target, source in change_sources.items():
        result[target] = _continuous_change(
            result[source], result["datetime_jst"]
        )

    duplicate_rows = int(result.duplicated(list(JOIN_KEYS), keep=False).sum())
    join_counts = (
        result["join_status"].astype("string").value_counts().to_dict()
        if "join_status" in result.columns
        else {}
    )
    matched = int(join_counts.get("both", 0))
    diagnostics = {
        "row_count": len(result),
        "matched_rows": matched,
        "join_success_rate": matched / len(result) if len(result) else None,
        "duplicate_key_rows": duplicate_rows,
        "invalid_datetime_rows": int(result["datetime_jst"].isna().sum()),
        "timezone": str(result["datetime_jst"].dt.tz),
        "missing_value_counts": {
            column: int(result[column].isna().sum())
            for column in (
                "procurement_volume",
                "area_demand_mw",
                "solar_generation_mw",
                "wind_generation_mw",
                *FEATURE_COLUMNS,
            )
            if result[column].isna().any()
        },
    }
    return result, diagnostics


def _infer_region(data: pd.DataFrame, region: str | None) -> str:
    if region is not None:
        return region
    for column in ("area_eprx", "area", "area_tepco", "area_grid"):
        if column in data.columns:
            values = data[column].dropna().astype(str).unique()
            if len(values) == 1:
                return str(values[0])
    # 기존 Tokyo 호출자가 지역 열 없이 만든 feature frame도 동일하게 지원한다.
    return "Tokyo"


def _number_or_none(value: Any) -> int | float | None:
    if pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def _metric_summary(data: pd.DataFrame, column: str) -> dict[str, Any]:
    values = data[column]
    return {
        "mean": _number_or_none(values.mean()),
        "minimum": _number_or_none(values.min()),
        "maximum": _number_or_none(values.max()),
        "missing_count": int(values.isna().sum()),
    }


def build_eprx_driver_weekly_context(
    feature_df: pd.DataFrame,
    week_start: Any,
    region: str | None = None,
) -> dict[str, Any]:
    """선택 주의 설명변수를 JSON 직렬화 가능한 사실 요약으로 반환한다."""
    required = {"datetime_jst", "procurement_volume", *FEATURE_COLUMNS}
    missing = sorted(required - set(feature_df.columns))
    if missing:
        raise ValueError(f"주간 컨텍스트 필수 열 누락: {', '.join(missing)}")

    start = pd.Timestamp(week_start)
    start = start.tz_localize(JST) if start.tzinfo is None else start.tz_convert(JST)
    start = start.normalize()
    end = start + pd.Timedelta(days=7)
    selected = feature_df.loc[
        feature_df["datetime_jst"].ge(start)
        & feature_df["datetime_jst"].lt(end)
    ].copy()
    duplicate_rows = int(selected.duplicated(list(JOIN_KEYS), keep=False).sum())
    matched = (
        int(selected["join_status"].astype("string").eq("both").sum())
        if "join_status" in selected.columns
        else 0
    )

    metrics = {
        "procurement_volume_mw": "procurement_volume",
        "area_demand_mw": "area_demand_mw",
        "solar_generation_mw": "solar_generation_mw",
        "wind_generation_mw": "wind_generation_mw",
        "variable_renewable_generation_mw": "variable_renewable_generation_mw",
        "residual_demand_mw": "residual_demand_mw",
        "variable_renewable_share": "variable_renewable_share",
        "demand_change_mw_30min": "demand_change_mw_30min",
        "residual_demand_change_mw_30min": "residual_demand_change_mw_30min",
    }
    return {
        "context_type": "eprx_tokyo_driver_weekly",
        "region": _infer_region(feature_df, region),
        "week_start": start.date().isoformat(),
        "week_end": (end - pd.Timedelta(days=1)).date().isoformat(),
        "timezone": JST,
        "interval_minutes": 30,
        "observed_rows": len(selected),
        "expected_rows": 336,
        "matched_rows": matched,
        "complete_week": len(selected) == 336
        and matched == 336
        and duplicate_rows == 0
        and not selected[["procurement_volume", "area_demand_mw", "solar_generation_mw", "wind_generation_mw"]].isna().any().any(),
        "duplicate_key_rows": duplicate_rows,
        "metrics": {
            label: _metric_summary(selected, column)
            for label, column in metrics.items()
        },
    }
