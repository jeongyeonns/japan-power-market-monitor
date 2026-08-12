"""OpenAI 요청에 넣기 전 단계의 EPRX 모집량 분석 컨텍스트.

이 모듈은 네트워크나 파일 I/O를 수행하지 않는다. 앱이 이미 로드하고
정규화한 EPRX 데이터프레임만 입력으로 받아 JSON 직렬화 가능한 값을 만든다.
"""

from __future__ import annotations

from typing import Any

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

from utils.eprx_periods import market_regime
from utils.regional_analysis import calculate_area_kpis
from utils.weekly_aggregation import add_week_columns, create_selected_area_weekly_profile
from utils.eprx_driver_features import FEATURE_COLUMNS, build_eprx_driver_features


SUPPORTED_REGIONS = ("Tokyo", "Chubu")
REQUIRED_COLUMNS = {
    "delivery_date",
    "period_no",
    "period_start",
    "area",
    "frequency_zone",
    "procurement_volume",
    "bid_volume",
    "awarded_volume",
    "avg_price",
    "max_price",
    "min_price",
}

PROFILE_TOLERANCE = 1e-9
HISTORICAL_START = pd.Timestamp("2026-03-16")
STANDARD_DEVIATION_DDOF = 1
DRIVER_METRICS = {
    "mean_demand_mw": "area_demand_mw",
    "mean_solar_mw": "solar_generation_mw",
    "mean_wind_mw": "wind_generation_mw",
    "mean_renewable_generation_mw": "renewable_generation_mw",
    "mean_renewable_share_pct": "renewable_share_pct",
    "mean_residual_demand_proxy_mw": "residual_demand_proxy_mw",
    "mean_abs_demand_ramp_30m_mw": "abs_demand_ramp_30m_mw",
    "mean_abs_renewable_ramp_30m_mw": "abs_renewable_ramp_30m_mw",
    "mean_abs_residual_demand_ramp_30m_mw": "abs_residual_demand_ramp_30m_mw",
}


def _number_or_none(value: Any) -> int | float | None:
    """numpy/pandas 숫자를 JSON에 안전한 파이썬 숫자로 변환한다."""
    if pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def build_eprx_procurement_context(
    eprx_df: pd.DataFrame,
    region: str,
    week_start: Any,
) -> dict[str, Any]:
    """도쿄 또는 중부의 선택 주 1차 조정력 모집량 컨텍스트를 만든다.

    주간 평균과 시간대 프로파일은 앱의 기존 주간 집계 및 KPI 산식을
    재사용한다. 제도 변경일이 포함된 주는 3시간제와 30분제를 섞지 않고
    각각의 ``market_regime`` 구간으로 유지한다.
    """
    if region not in SUPPORTED_REGIONS:
        raise ValueError(
            f"지원하지 않는 EPRX 지역입니다: {region!r}. "
            f"허용 값: {', '.join(SUPPORTED_REGIONS)}"
        )
    if not isinstance(eprx_df, pd.DataFrame):
        raise TypeError("eprx_df는 pandas.DataFrame이어야 합니다.")

    missing = sorted(REQUIRED_COLUMNS - set(eprx_df.columns))
    if missing:
        raise ValueError(f"EPRX 분석 컨텍스트 필수 열 누락: {', '.join(missing)}")

    selected_start = pd.Timestamp(week_start).normalize()
    if pd.isna(selected_start):
        raise ValueError("week_start를 유효한 날짜로 변환할 수 없습니다.")

    prepared = add_week_columns(eprx_df)
    raw_week = prepared.loc[
        prepared["week_start"].eq(selected_start)
        & prepared["area"].eq(region)
    ].copy()
    profile = create_selected_area_weekly_profile(
        prepared, selected_start, [region]
    )
    kpis = calculate_area_kpis(profile, raw_week)

    dates = raw_week["delivery_date"].dropna()
    regimes = (
        dates.map(market_regime).drop_duplicates().tolist()
        if not dates.empty
        else []
    )
    statuses = (
        sorted(str(value) for value in raw_week["source_status"].dropna().unique())
        if "source_status" in raw_week.columns
        else []
    )

    period_profile = []
    for row in profile.itertuples(index=False):
        period_profile.append(
            {
                "market_regime": row.market_regime,
                "period_no": int(row.period_no),
                "period_start": row.period_start,
                "average_procurement_mw": _number_or_none(
                    row.procurement_volume
                ),
                "observation_days": int(row.observation_count),
                "expected_observation_days": int(
                    row.expected_observation_count
                ),
                "complete": row.data_status == "Complete",
            }
        )

    observed_days = int(dates.dt.normalize().nunique())
    return {
        "context_type": "eprx_primary_reserve_procurement",
        "region": region,
        "week_start": selected_start.date().isoformat(),
        "week_end": (selected_start + pd.Timedelta(days=6)).date().isoformat(),
        "unit": "MW",
        "weekly_average_procurement_mw": _number_or_none(
            kpis["평균 모집량 (MW)"]
        ),
        "observed_days": observed_days,
        "expected_days": 7,
        "complete_week": observed_days == 7 and bool(period_profile) and all(
            item["complete"] for item in period_profile
        ),
        "market_regimes": regimes,
        "source_statuses": statuses,
        "period_profile": period_profile,
    }


def to_json_safe(value: Any) -> Any:
    """Recursively convert pandas/numpy values to strict JSON-compatible values."""
    if value is None:
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [to_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _stat(value: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(value, errors="coerce")
    mean = numeric.mean()
    standard_deviation = numeric.std(ddof=STANDARD_DEVIATION_DDOF)
    return {
        "mean": _number_or_none(mean),
        "minimum": _number_or_none(numeric.min()),
        "maximum": _number_or_none(numeric.max()),
        "standard_deviation": _number_or_none(standard_deviation),
        "median": _number_or_none(numeric.median()),
        "valid_count": int(numeric.notna().sum()),
        "unique_value_count": int(numeric.dropna().nunique()),
        "coefficient_of_variation": _number_or_none(
            standard_deviation / mean if pd.notna(mean) and mean != 0 else np.nan
        ),
    }


def _region_filter(data: pd.DataFrame, region: str) -> pd.DataFrame:
    for column in ("area_eprx", "area", "area_grid", "area_tepco"):
        if column in data.columns and data[column].notna().any():
            values = data[column].dropna().astype(str)
            if region in set(values):
                return data.loc[data[column].astype(str).eq(region)].copy()
    return data.copy()


def _prepare_feature_frame(merged_df: pd.DataFrame, region: str) -> pd.DataFrame:
    selected = _region_filter(merged_df, region)
    if not set(FEATURE_COLUMNS).issubset(selected.columns):
        selected, _ = build_eprx_driver_features(selected)
    result = selected.copy()
    result["datetime_jst"] = pd.to_datetime(result["datetime_jst"], errors="coerce")
    if result["datetime_jst"].dt.tz is None:
        result["datetime_jst"] = result["datetime_jst"].dt.tz_localize("Asia/Tokyo")
    else:
        result["datetime_jst"] = result["datetime_jst"].dt.tz_convert("Asia/Tokyo")
    result["week_start"] = (
        result["datetime_jst"].dt.normalize()
        - pd.to_timedelta(result["datetime_jst"].dt.weekday, unit="D")
    )
    return result.sort_values("datetime_jst", kind="stable").reset_index(drop=True)


def _complete_modern_week(data: pd.DataFrame) -> bool:
    if len(data) != 336 or data["datetime_jst"].nunique() != 336:
        return False
    dates = data["datetime_jst"].dt.normalize()
    if dates.nunique() != 7 or not dates.value_counts().eq(48).all():
        return False
    return not data["procurement_volume"].isna().any()


def _daily_profile(data: pd.DataFrame) -> list[dict[str, Any]]:
    if data.empty:
        return []
    working = data.assign(date=data["datetime_jst"].dt.normalize())
    rows = []
    for date, group in working.groupby("date", sort=True):
        stats = _stat(group["procurement_volume"])
        rows.append({
            "date": date.date().isoformat(),
            "weekday": date.day_name(),
            "average_procurement_mw": stats["mean"],
            "minimum_procurement_mw": stats["minimum"],
            "maximum_procurement_mw": stats["maximum"],
            "standard_deviation_procurement_mw": stats["standard_deviation"],
            "valid_period_count": stats["valid_count"],
            "expected_period_count": 48,
            "complete_day": len(group) == 48
            and group["datetime_jst"].nunique() == 48
            and stats["valid_count"] == 48,
        })
    return rows


def _time_profile(data: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for time_block, group in data.groupby("period_start", sort=True):
        stats = _stat(group["procurement_volume"])
        rows.append({
            "time_block": str(time_block),
            "average_procurement_mw": stats["mean"],
            "minimum_procurement_mw": stats["minimum"],
            "maximum_procurement_mw": stats["maximum"],
            "standard_deviation_procurement_mw": stats["standard_deviation"],
            "valid_day_count": int(group.loc[group["procurement_volume"].notna(), "datetime_jst"].dt.normalize().nunique()),
        })
    return rows


def _comparison(current: Any, previous: Any, reason: str | None = None) -> dict[str, Any]:
    if reason is not None or previous is None or current is None:
        return {"current": current, "previous": None, "change": None,
                "change_pct": None, "status": "unavailable", "reason": reason or "missing_value"}
    change = float(current) - float(previous)
    return {"current": current, "previous": previous, "change": _number_or_none(change),
            "change_pct": _number_or_none(change / float(previous) * 100) if float(previous) != 0 else None,
            "status": "available", "reason": "previous_value_zero" if float(previous) == 0 else None}


def _week_metrics(data: pd.DataFrame) -> dict[str, Any]:
    procurement = _stat(data["procurement_volume"])
    result = {
        "procurement_mean_mw": procurement["mean"],
        "procurement_minimum_mw": procurement["minimum"],
        "procurement_maximum_mw": procurement["maximum"],
        "procurement_standard_deviation_mw": procurement["standard_deviation"],
        "procurement_median_mw": procurement["median"],
    }
    result.update({name: _number_or_none(data[column].mean()) for name, column in DRIVER_METRICS.items()})
    residual_ramp = pd.to_numeric(data["residual_demand_ramp_30m_mw"], errors="coerce")
    # The source values are 30-minute average MW, so interval energy is MW × 0.5 h.
    result.update({
        "total_demand_mwh": _number_or_none(pd.to_numeric(data["area_demand_mw"], errors="coerce").sum(min_count=1) * 0.5),
        "total_solar_generation_mwh": _number_or_none(pd.to_numeric(data["solar_generation_mw"], errors="coerce").sum(min_count=1) * 0.5),
        "total_wind_generation_mwh": _number_or_none(pd.to_numeric(data["wind_generation_mw"], errors="coerce").sum(min_count=1) * 0.5),
        "residual_demand_ramp_std_30m_mw": _number_or_none(residual_ramp.std(ddof=STANDARD_DEVIATION_DDOF)),
        "maximum_abs_residual_demand_ramp_30m_mw": _number_or_none(residual_ramp.abs().max()),
        "p95_abs_residual_demand_ramp_30m_mw": _number_or_none(residual_ramp.abs().quantile(0.95)),
    })
    return result


def _previous_week_comparison(frame: pd.DataFrame, selected: pd.DataFrame, start: pd.Timestamp) -> dict[str, Any]:
    previous = frame.loc[frame["week_start"].eq(start - pd.Timedelta(days=7))]
    mixed = selected["datetime_jst"].dt.tz_localize(None).dt.normalize().map(market_regime).nunique() > 1
    reason = "mixed_market_regime" if mixed else (None if _complete_modern_week(previous) else "previous_week_incomplete_or_missing")
    current_metrics = _week_metrics(selected)
    previous_metrics = _week_metrics(previous) if reason is None else {}
    return {key: _comparison(value, previous_metrics.get(key), reason) for key, value in current_metrics.items()}


def _profile_vector(data: pd.DataFrame) -> np.ndarray:
    ordered = data.assign(weekday=data["datetime_jst"].dt.weekday).sort_values(["weekday", "period_start"])
    return pd.to_numeric(ordered["procurement_volume"], errors="coerce").to_numpy(dtype=float)


def _profile_repetition(frame: pd.DataFrame, selected: pd.DataFrame, start: pd.Timestamp) -> dict[str, Any]:
    complete = [(week, group) for week, group in frame.groupby("week_start", sort=True) if _complete_modern_week(group)]
    vectors = [(week, _profile_vector(group)) for week, group in complete]
    current = _profile_vector(selected) if _complete_modern_week(selected) else None
    previous_vector = next((vector for week, vector in vectors if week == start - pd.Timedelta(days=7)), None)
    comparable = 336 if current is not None and previous_vector is not None else 0
    differences = np.abs(current - previous_vector) if comparable else np.array([])
    signatures: list[np.ndarray] = []
    counts: list[int] = []
    previous_identical_count = 0
    prior = None
    for _, vector in vectors:
        if prior is not None and np.allclose(vector, prior, rtol=0, atol=PROFILE_TOLERANCE, equal_nan=False):
            previous_identical_count += 1
        prior = vector
        match = next((index for index, known in enumerate(signatures) if np.allclose(vector, known, rtol=0, atol=PROFILE_TOLERANCE)), None)
        if match is None:
            signatures.append(vector); counts.append(1)
        else:
            counts[match] += 1
    return {
        "status": "available" if current is not None else "selected_week_incomplete",
        "same_as_previous_week": bool(comparable and np.all(differences <= PROFILE_TOLERANCE)),
        "comparable_period_count": comparable,
        "maximum_absolute_difference_mw": _number_or_none(differences.max()) if comparable else None,
        "mean_absolute_difference_mw": _number_or_none(differences.mean()) if comparable else None,
        "complete_week_count": len(vectors),
        "weeks_identical_to_previous_count": previous_identical_count,
        "unique_weekly_profile_count": len(signatures),
        "most_repeated_profile_count": max(counts, default=0),
        "tolerance_mw": PROFILE_TOLERANCE,
    }


def _historical_position(frame: pd.DataFrame, selected: pd.DataFrame, start: pd.Timestamp) -> dict[str, Any]:
    metric_names = ["procurement_mean_mw", "procurement_standard_deviation_mw", "mean_demand_mw",
                    "mean_residual_demand_proxy_mw", "mean_renewable_share_pct", "mean_abs_demand_ramp_30m_mw",
                    "mean_abs_residual_demand_ramp_30m_mw", "mean_abs_renewable_ramp_30m_mw"]
    history = []
    for week, group in frame.groupby("week_start", sort=True):
        if week >= HISTORICAL_START.tz_localize("Asia/Tokyo") and week < start and _complete_modern_week(group):
            history.append(_week_metrics(group))
    current = _week_metrics(selected)
    result = {}
    for metric in metric_names:
        values = np.array([row[metric] for row in history if row.get(metric) is not None], dtype=float)
        value = current.get(metric)
        if value is None or len(values) < 4:
            result[metric] = {"value": value, "percentile": None, "z_score": None, "historical_week_count": len(values), "status": "insufficient_history"}
            continue
        # Midrank percentile: values below + half of tied values.
        percentile = (np.sum(values < value) + 0.5 * np.sum(np.isclose(values, value, rtol=0, atol=PROFILE_TOLERANCE))) / len(values) * 100
        std = values.std(ddof=STANDARD_DEVIATION_DDOF)
        result[metric] = {"value": value, "percentile": float(percentile),
                          "z_score": float((value - values.mean()) / std) if std != 0 else None,
                          "historical_week_count": len(values), "status": "available"}
    return result


def _data_quality(data: pd.DataFrame) -> dict[str, Any]:
    dates = data["datetime_jst"].dt.normalize()
    intervals = data.sort_values("datetime_jst")["datetime_jst"].diff()
    derived = [column for column in FEATURE_COLUMNS if column in data.columns]
    numeric = data.select_dtypes(include=[np.number])
    examples = data.loc[
        data["area_demand_mw"].le(0) | data["solar_generation_mw"].lt(0) | data["wind_generation_mw"].lt(0),
        ["datetime_jst", "area_demand_mw", "solar_generation_mw", "wind_generation_mw"],
    ].head(5).to_dict("records")
    quality = {
        "expected_weekly_rows": 336, "actual_rows": len(data), "unique_date_count": int(dates.nunique()),
        "daily_period_counts": {date.date().isoformat(): int(count) for date, count in dates.value_counts().sort_index().items()},
        "duplicate_datetime_rows": int(data.duplicated("datetime_jst", keep=False).sum()),
        "duplicate_join_key_rows": int(data.duplicated([c for c in ("delivery_date", "period_no", "period_start") if c in data], keep=False).sum()),
        "missing_procurement_count": int(data["procurement_volume"].isna().sum()),
        "missing_demand_count": int(data["area_demand_mw"].isna().sum()),
        "missing_solar_count": int(data["solar_generation_mw"].isna().sum()),
        "missing_wind_count": int(data["wind_generation_mw"].isna().sum()),
        "missing_derived_value_count": int(data[derived].isna().sum().sum()),
        "non_30_minute_interval_count": int(intervals.iloc[1:].ne(pd.Timedelta(minutes=30)).sum()) if len(intervals) > 1 else 0,
        "nonpositive_demand_count": int(data["area_demand_mw"].le(0).sum()),
        "negative_solar_count": int(data["solar_generation_mw"].lt(0).sum()),
        "negative_wind_count": int(data["wind_generation_mw"].lt(0).sum()),
        "renewable_share_over_100_count": int(data["renewable_share_pct"].gt(100).sum()),
        "negative_residual_demand_count": int(data["residual_demand_proxy_mw"].lt(0).sum()),
        "infinite_value_count": int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
        "anomaly_examples": examples,
    }
    return quality


def build_eprx_analysis_context(
    merged_df: pd.DataFrame, region: str, week_start: Any, grid_source: str | None = None
) -> dict[str, Any]:
    """Build the common Tokyo/Chubu procurement-driver context without AI analysis."""
    if region not in SUPPORTED_REGIONS:
        raise ValueError(f"Unsupported region: {region}")
    frame = _prepare_feature_frame(merged_df, region)
    start = pd.Timestamp(week_start)
    start = start.tz_localize("Asia/Tokyo") if start.tzinfo is None else start.tz_convert("Asia/Tokyo")
    start = start.normalize()
    selected = frame.loc[frame["week_start"].eq(start)].copy()
    procurement = _stat(selected["procurement_volume"])
    regimes = sorted(
        selected["datetime_jst"].dt.tz_localize(None).dt.normalize().map(market_regime).unique()
    ) if not selected.empty else []
    daily = _daily_profile(selected)
    time_profile = _time_profile(selected)
    ranked = [row for row in time_profile if row["average_procurement_mw"] is not None]
    high = sorted(ranked, key=lambda row: (-row["average_procurement_mw"], row["time_block"]))[:5]
    low = sorted(ranked, key=lambda row: (row["average_procurement_mw"], row["time_block"]))[:5]
    quality = _data_quality(selected)
    context = {
        "analysis_type": "eprx_primary_reserve_driver_context", "status": "ok" if not selected.empty else "source_data_missing",
        "region": region, "grid_source": grid_source or ("TEPCO" if region == "Tokyo" else "Chuden"),
        "week": {"start": start.date().isoformat(), "end": (start + pd.Timedelta(days=6)).date().isoformat(),
                 "market_regimes": regimes, "complete": _complete_modern_week(selected)},
        "procurement": {**procurement, "unit": "MW", "week_start": start.date().isoformat(),
                        "week_end": (start + pd.Timedelta(days=6)).date().isoformat(),
                        "market_regime": regimes[0] if len(regimes) == 1 else ("mixed" if regimes else None),
                        "complete_week": _complete_modern_week(selected)},
        "daily_profile": daily, "time_profile": time_profile,
        "notable_time_blocks": {"highest": high, "lowest": low, "tie_breaker": "time_block ascending"},
        "previous_week_comparison": _previous_week_comparison(frame, selected, start),
        "profile_repetition": _profile_repetition(frame, selected, start),
        "historical_position": _historical_position(frame, selected, start),
        "driver_features_summary": {name: _number_or_none(selected[column].mean()) for name, column in DRIVER_METRICS.items()},
        "data_quality": quality,
        "limitations": [
            "Residual demand is a proxy calculated from public 30-minute demand, solar, and wind actuals.",
            "EPRX procurement is not separated into normal-time and emergency-time components.",
            "Bilateral contracts and off-market secured volumes are not included.",
            "Public 30-minute data does not reproduce TSO internal sub-second calculations.",
            "This stage does not analyze statistical relationships or causes.",
        ],
        "calculation_metadata": {"timezone": "Asia/Tokyo", "interval_minutes": 30,
                                 "standard_deviation": "sample", "standard_deviation_ddof": STANDARD_DEVIATION_DDOF,
                                 "profile_tolerance_mw": PROFILE_TOLERANCE,
                                 "percentile_method": "midrank (below + 0.5 * ties) / n",
                                 "historical_start": HISTORICAL_START.date().isoformat()},
    }
    safe = to_json_safe(context)
    safe["data_quality"]["json_serializable"] = True
    json.dumps(safe, ensure_ascii=False, allow_nan=False)
    return safe
