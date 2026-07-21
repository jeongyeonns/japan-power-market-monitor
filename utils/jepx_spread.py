"""JEPX 30분 가격의 결정론적 일별 이론 스프레드 계산 엔진."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SUPPORTED_DURATIONS = (1, 2, 4)
OPERATION_MODES = ("nem_best_case", "unconstrained", "charge_before_discharge")
DEFAULT_OPERATION_MODE = "nem_best_case"
EXPECTED_PERIODS = 48
RESULT_COLUMNS = [
    "delivery_date", "area", "duration_hours", "operation_mode", "window_type",
    "charge_start", "charge_end", "charge_average_price", "discharge_start",
    "discharge_end", "discharge_average_price", "spread", "positive_spread",
    "charge_periods", "discharge_periods", "observation_count",
    "expected_observation_count", "missing_periods", "completeness_flag",
    "calculation_status", "warning_message",
]


def _format_time(minutes: int) -> str:
    return "24:00" if minutes == 1440 else f"{minutes // 60:02d}:{minutes % 60:02d}"


def _period_start(period: int) -> str:
    return _format_time((period - 1) * 30)


def _period_end(period: int) -> str:
    return _format_time(period * 30)


def validate_daily_price_profile(
    day_area_data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """한 날짜·가격구분의 48개 코마를 검증하고 시간순으로 정렬합니다."""
    required = {"delivery_date", "area", "period_no", "price"}
    missing = sorted(required - set(day_area_data.columns))
    if missing:
        raise ValueError("JEPX 스프레드 필수 열 누락: " + ", ".join(missing))
    data = day_area_data.copy()
    data["delivery_date"] = pd.to_datetime(data["delivery_date"], errors="coerce").dt.normalize()
    data["period_no"] = pd.to_numeric(data["period_no"], errors="coerce").astype("Int64")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.sort_values("period_no", kind="mergesort").reset_index(drop=True)
    warnings = []
    if data["delivery_date"].nunique(dropna=False) != 1:
        warnings.append("delivery_date가 하나가 아닙니다.")
    if data["area"].nunique(dropna=False) != 1:
        warnings.append("area가 하나가 아닙니다.")
    if data["delivery_date"].isna().any():
        warnings.append("날짜 변환 실패가 있습니다.")
    if (~data["period_no"].between(1, EXPECTED_PERIODS)).any():
        warnings.append("시간대 번호가 1~48 범위가 아닙니다.")
    if data["period_no"].duplicated(keep=False).any():
        warnings.append("중복 시간대가 있습니다.")
    if data["price"].isna().any():
        warnings.append("결측 또는 숫자 변환 불가 가격이 있습니다.")
    observed = {
        int(value) for value in data.loc[
            data["period_no"].between(1, EXPECTED_PERIODS), "period_no"
        ].dropna()
    }
    missing_periods = tuple(sorted(set(range(1, 49)) - observed))
    if len(data) != EXPECTED_PERIODS or missing_periods:
        warnings.append(f"하루 데이터가 불완전합니다({len(observed)}/48개 시간대).")
    return data, {
        "is_valid": not warnings,
        "observation_count": len(observed),
        "expected_observation_count": EXPECTED_PERIODS,
        "missing_periods": missing_periods,
        "warnings": warnings,
        "warning_message": " ".join(warnings),
    }


def create_rolling_price_windows(
    day_area_data: pd.DataFrame, duration_hours: int
) -> pd.DataFrame:
    """자정을 넘지 않는 연속 30분 평균가격 구간을 생성합니다."""
    if duration_hours not in SUPPORTED_DURATIONS:
        raise ValueError("duration_hours는 1, 2, 4 중 하나여야 합니다.")
    data, validation = validate_daily_price_profile(day_area_data)
    columns = ["window_start", "window_end", "average_price", "periods", "start_period", "end_period"]
    if not validation["is_valid"]:
        return pd.DataFrame(columns=columns)
    count = duration_hours * 2
    prices = data.set_index("period_no")["price"]
    rows = []
    for start in range(1, EXPECTED_PERIODS - count + 2):
        periods = tuple(range(start, start + count))
        rows.append({
            "window_start": _period_start(start),
            "window_end": _period_end(periods[-1]),
            "average_price": float(prices.loc[list(periods)].mean()),
            "periods": periods, "start_period": start, "end_period": periods[-1],
        })
    return pd.DataFrame(rows, columns=columns)


def find_optimal_charge_discharge_windows(
    rolling_windows: pd.DataFrame, operation_mode: str
) -> dict[str, Any] | None:
    """연속 구간 쌍 중 제약을 만족하고 스프레드가 최대인 쌍을 찾습니다."""
    if operation_mode not in {"unconstrained", "charge_before_discharge"}:
        raise ValueError("rolling 구간에는 unconstrained 또는 charge_before_discharge를 사용하세요.")
    candidates = []
    for _, charge in rolling_windows.iterrows():
        for _, discharge in rolling_windows.iterrows():
            if not set(charge["periods"]).isdisjoint(discharge["periods"]):
                continue
            if operation_mode == "charge_before_discharge" and charge["end_period"] >= discharge["start_period"]:
                continue
            candidates.append({
                "spread": float(discharge["average_price"] - charge["average_price"]),
                "charge": charge, "discharge": discharge,
            })
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x["spread"], x["charge"]["start_period"], x["discharge"]["start_period"]))
    return candidates[0]


def _nem_best_case(data: pd.DataFrame, duration_hours: int) -> dict[str, Any] | None:
    """NEM best_case_daily를 JEPX 30분 코마에 적용한 비연속 선택 방식입니다."""
    count = duration_hours * 2
    charge = data.sort_values(["price", "period_no"], kind="mergesort").head(count)
    remaining = data.loc[~data.index.isin(charge.index)]
    discharge = remaining.sort_values(
        ["price", "period_no"], ascending=[False, True], kind="mergesort"
    ).head(count)
    if len(charge) != count or len(discharge) != count:
        return None
    cp = tuple(sorted(int(x) for x in charge["period_no"]))
    dp = tuple(sorted(int(x) for x in discharge["period_no"]))
    charge_price, discharge_price = float(charge["price"].mean()), float(discharge["price"].mean())
    return {
        "charge_start": _period_start(cp[0]), "charge_end": _period_end(cp[-1]),
        "charge_average_price": charge_price, "discharge_start": _period_start(dp[0]),
        "discharge_end": _period_end(dp[-1]), "discharge_average_price": discharge_price,
        "spread": discharge_price - charge_price, "charge_periods": cp, "discharge_periods": dp,
    }


def _nem_best_case_arrays(
    periods: np.ndarray, prices: np.ndarray, duration_hours: int
) -> dict[str, Any] | None:
    """NEM 선택 규칙을 같은 가격일 때 이른 코마 우선으로 빠르게 계산합니다."""
    count = duration_hours * 2
    ascending = np.lexsort((periods, prices))
    charge_idx = ascending[:count]
    charge_mask = np.zeros(len(periods), dtype=bool)
    charge_mask[charge_idx] = True
    available_idx = np.flatnonzero(~charge_mask)
    discharge_order = np.lexsort((periods[available_idx], -prices[available_idx]))
    discharge_idx = available_idx[discharge_order[:count]]
    if len(charge_idx) != count or len(discharge_idx) != count:
        return None
    cp = tuple(sorted(int(value) for value in periods[charge_idx]))
    dp = tuple(sorted(int(value) for value in periods[discharge_idx]))
    charge_price = float(prices[charge_idx].mean())
    discharge_price = float(prices[discharge_idx].mean())
    return {
        "charge_start": _period_start(cp[0]), "charge_end": _period_end(cp[-1]),
        "charge_average_price": charge_price, "discharge_start": _period_start(dp[0]),
        "discharge_end": _period_end(dp[-1]), "discharge_average_price": discharge_price,
        "spread": discharge_price - charge_price, "charge_periods": cp, "discharge_periods": dp,
    }


def _empty_result(data: pd.DataFrame, duration: int, mode: str, validation: dict[str, Any]) -> dict[str, Any]:
    result = {column: np.nan for column in RESULT_COLUMNS}
    result.update({
        "delivery_date": pd.to_datetime(data.get("delivery_date"), errors="coerce").min() if not data.empty else pd.NaT,
        "area": data["area"].iloc[0] if not data.empty and "area" in data else None,
        "duration_hours": duration, "operation_mode": mode,
        "window_type": "non_contiguous" if mode == "nem_best_case" else "contiguous",
        "charge_periods": tuple(), "discharge_periods": tuple(),
        "observation_count": validation["observation_count"],
        "expected_observation_count": EXPECTED_PERIODS,
        "missing_periods": validation["missing_periods"], "completeness_flag": "Incomplete",
        "calculation_status": "Not calculated", "warning_message": validation["warning_message"],
    })
    return result


def _finalize_result(
    data: pd.DataFrame, duration: int, mode: str,
    selected: dict[str, Any] | None, validation: dict[str, Any],
) -> dict[str, Any]:
    if selected is None:
        validation = dict(validation)
        validation["warning_message"] = "제약을 만족하는 충전·방전 구간이 없습니다."
        return _empty_result(data, duration, mode, validation)
    overlap = set(selected["charge_periods"]) & set(selected["discharge_periods"])
    bad_order = mode == "charge_before_discharge" and max(selected["charge_periods"]) >= min(selected["discharge_periods"])
    if overlap or bad_order:
        validation = dict(validation)
        validation["warning_message"] = "충전·방전 중복 또는 순서 제약 위반입니다."
        return _empty_result(data, duration, mode, validation)
    result = {
        "delivery_date": data["delivery_date"].iloc[0], "area": data["area"].iloc[0],
        "duration_hours": duration, "operation_mode": mode,
        "window_type": "non_contiguous" if mode == "nem_best_case" else "contiguous",
        **selected, "positive_spread": bool(selected["spread"] > 0),
        "observation_count": validation["observation_count"],
        "expected_observation_count": 48, "missing_periods": validation["missing_periods"],
        "completeness_flag": "Complete", "calculation_status": "Calculated",
        "warning_message": "",
    }
    return {column: result.get(column, np.nan) for column in RESULT_COLUMNS}


def calculate_daily_spread(
    day_area_data: pd.DataFrame,
    duration_hours: int,
    operation_mode: str = DEFAULT_OPERATION_MODE,
) -> dict[str, Any]:
    """한 날짜·가격구분·운용시간의 이론적 가격 스프레드를 계산합니다."""
    if duration_hours not in SUPPORTED_DURATIONS:
        raise ValueError("duration_hours는 1, 2, 4 중 하나여야 합니다.")
    if operation_mode not in OPERATION_MODES:
        raise ValueError("지원하지 않는 operation_mode입니다.")
    data, validation = validate_daily_price_profile(day_area_data)
    if not validation["is_valid"]:
        return _empty_result(day_area_data, duration_hours, operation_mode, validation)
    if operation_mode == "nem_best_case":
        selected = _nem_best_case(data, duration_hours)
    else:
        optimum = find_optimal_charge_discharge_windows(
            create_rolling_price_windows(data, duration_hours), operation_mode
        )
        if optimum is None:
            selected = None
        else:
            c, d = optimum["charge"], optimum["discharge"]
            selected = {
                "charge_start": c["window_start"], "charge_end": c["window_end"],
                "charge_average_price": float(c["average_price"]),
                "discharge_start": d["window_start"], "discharge_end": d["window_end"],
                "discharge_average_price": float(d["average_price"]), "spread": optimum["spread"],
                "charge_periods": tuple(c["periods"]), "discharge_periods": tuple(d["periods"]),
            }
    return _finalize_result(data, duration_hours, operation_mode, selected, validation)


def calculate_all_daily_spreads(
    normalized_long_data: pd.DataFrame,
    durations: Iterable[int] = SUPPORTED_DURATIONS,
    operation_mode: str = DEFAULT_OPERATION_MODE,
) -> pd.DataFrame:
    """모든 날짜·가격구분·운용시간 결과 또는 실패 상태를 보존합니다."""
    durations = tuple(durations)
    if set(durations) - set(SUPPORTED_DURATIONS):
        raise ValueError("durations는 1, 2, 4만 지원합니다.")
    required = {"delivery_date", "area", "period_no", "price"}
    missing = sorted(required - set(normalized_long_data.columns))
    if missing:
        raise ValueError("JEPX 스프레드 필수 열 누락: " + ", ".join(missing))
    prepared = normalized_long_data.copy()
    prepared["delivery_date"] = pd.to_datetime(prepared["delivery_date"], errors="coerce").dt.normalize()
    prepared["period_no"] = pd.to_numeric(prepared["period_no"], errors="coerce")
    prepared["price"] = pd.to_numeric(prepared["price"], errors="coerce")
    rows = []
    for _, group in prepared.groupby(["delivery_date", "area"], dropna=False, sort=True):
        data, validation = validate_daily_price_profile(group)
        periods = data["period_no"].to_numpy(dtype=int) if validation["is_valid"] else None
        prices = data["price"].to_numpy(dtype=float) if validation["is_valid"] else None
        for duration in durations:
            if not validation["is_valid"]:
                rows.append(_empty_result(group, duration, operation_mode, validation))
            elif operation_mode == "nem_best_case":
                selected = _nem_best_case_arrays(periods, prices, duration)
                rows.append(_finalize_result(data, duration, operation_mode, selected, validation))
            else:
                windows = create_rolling_price_windows(data, duration)
                optimum = find_optimal_charge_discharge_windows(windows, operation_mode)
                if optimum is None:
                    selected = None
                else:
                    c, d = optimum["charge"], optimum["discharge"]
                    selected = {
                        "charge_start": c["window_start"], "charge_end": c["window_end"],
                        "charge_average_price": float(c["average_price"]),
                        "discharge_start": d["window_start"], "discharge_end": d["window_end"],
                        "discharge_average_price": float(d["average_price"]), "spread": optimum["spread"],
                        "charge_periods": tuple(c["periods"]), "discharge_periods": tuple(d["periods"]),
                    }
                rows.append(_finalize_result(data, duration, operation_mode, selected, validation))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _aggregate(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    data = daily.copy()
    data["delivery_date"] = pd.to_datetime(data["delivery_date"])
    if period == "week":
        data["week_start"] = (data["delivery_date"] - pd.to_timedelta(data["delivery_date"].dt.weekday, unit="D")).dt.normalize()
        data["week_end"] = data["week_start"] + pd.Timedelta(days=6)
        periods = ["week_start", "week_end"]
    else:
        data["year_month"] = data["delivery_date"].dt.to_period("M").astype(str)
        periods = ["year_month"]
    keys = periods + ["area", "duration_hours", "operation_mode"]
    rows = []
    for key, group in data.groupby(keys, dropna=False, sort=True):
        valid = group[group["completeness_flag"].eq("Complete") & group["spread"].notna()]
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update({
            "average_spread": valid["spread"].mean(), "median_spread": valid["spread"].median(),
            "max_spread": valid["spread"].max(), "min_spread": valid["spread"].min(),
            "positive_spread_days": int(valid["spread"].gt(0).sum()), "total_days": len(group),
            "complete_days": len(valid), "incomplete_days": len(group) - len(valid),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_spreads_by_week(daily_spreads: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(daily_spreads, "week")


def aggregate_spreads_by_month(daily_spreads: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(daily_spreads, "month")


def save_spread_validation_results(
    daily_spreads: pd.DataFrame, validation_summary: pd.DataFrame, output_directory: str | Path
) -> tuple[Path, Path]:
    """명시적으로 호출할 때만 UTF-8-SIG 검증 파일을 저장합니다."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    daily_path, summary_path = directory / "daily_spread_results.csv", directory / "spread_validation_summary.csv"
    daily_spreads.to_csv(daily_path, index=False, encoding="utf-8-sig")
    validation_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return daily_path, summary_path
