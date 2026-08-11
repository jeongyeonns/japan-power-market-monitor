"""Statistical association diagnostics for EPRX procurement and grid actuals.

This module describes retrospective association only.  It neither predicts
procurement nor makes causal claims.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.eprx_ai_context import build_eprx_analysis_context, to_json_safe
from utils.eprx_driver_features import FEATURE_COLUMNS, build_eprx_driver_features


MIN_CORRELATION_SAMPLES = 100
MIN_REGRESSION_SAMPLES = 200
MIN_BOOTSTRAP_DAYS = 7
RUNTIME_BOOTSTRAP_ITERATIONS = 100
MAX_BOOTSTRAP_PREDICTORS = 5
FAST_CONTEXT_VERSION = "3"
FAST_PREDICTORS = (
    "demand_mw",
    "renewable_generation_mw",
    "renewable_share_pct",
    "residual_demand_proxy_mw",
    "abs_renewable_ramp_30m_mw",
    "abs_solar_ramp_30m_mw",
    "abs_demand_ramp_30m_mw",
)

SOURCE_COLUMNS = {
    "procurement_volume_mw": "procurement_volume",
    "demand_mw": "area_demand_mw",
    "solar_mw": "solar_generation_mw",
    "wind_mw": "wind_generation_mw",
    "renewable_generation_mw": "renewable_generation_mw",
    "renewable_share_pct": "renewable_share_pct",
    "residual_demand_proxy_mw": "residual_demand_proxy_mw",
    "demand_ramp_30m_mw": "demand_ramp_30m_mw",
    "abs_demand_ramp_30m_mw": "abs_demand_ramp_30m_mw",
    "renewable_ramp_30m_mw": "renewable_ramp_30m_mw",
    "abs_renewable_ramp_30m_mw": "abs_renewable_ramp_30m_mw",
    "residual_demand_ramp_30m_mw": "residual_demand_ramp_30m_mw",
    "abs_residual_demand_ramp_30m_mw": "abs_residual_demand_ramp_30m_mw",
    "solar_ramp_30m_mw": "solar_ramp_30m_mw",
    "abs_solar_ramp_30m_mw": "abs_solar_ramp_30m_mw",
    "wind_ramp_30m_mw": "wind_ramp_30m_mw",
    "abs_wind_ramp_30m_mw": "abs_wind_ramp_30m_mw",
}
PREDICTORS = tuple(key for key in SOURCE_COLUMNS if key != "procurement_volume_mw")

ANOMALY_SOURCES = {
    "procurement_anomaly_mw": "procurement_volume_mw",
    "demand_anomaly_mw": "demand_mw",
    "solar_anomaly_mw": "solar_mw",
    "wind_anomaly_mw": "wind_mw",
    "renewable_generation_anomaly_mw": "renewable_generation_mw",
    "renewable_share_anomaly_pct": "renewable_share_pct",
    "residual_demand_proxy_anomaly_mw": "residual_demand_proxy_mw",
    "abs_demand_ramp_anomaly_mw": "abs_demand_ramp_30m_mw",
    "abs_renewable_ramp_anomaly_mw": "abs_renewable_ramp_30m_mw",
    "abs_residual_demand_ramp_anomaly_mw": "abs_residual_demand_ramp_30m_mw",
    "abs_solar_ramp_anomaly_mw": "abs_solar_ramp_30m_mw",
    "abs_wind_ramp_anomaly_mw": "abs_wind_ramp_30m_mw",
}
ANOMALY_BY_PREDICTOR = {source: anomaly for anomaly, source in ANOMALY_SOURCES.items()}


def _region_filter(data: pd.DataFrame, region: str) -> pd.DataFrame:
    for column in ("area_eprx", "area", "area_grid", "area_tepco"):
        if column in data and region in set(data[column].dropna().astype(str)):
            return data.loc[data[column].astype(str).eq(region)].copy()
    return data.copy()


def prepare_statistics_data(
    feature_df: pd.DataFrame,
    region: str,
    analysis_start: Any = None,
    analysis_end: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare one region without imputing, averaging duplicates, or trimming outliers."""
    frame = _region_filter(feature_df, region)
    if not set(FEATURE_COLUMNS).issubset(frame.columns):
        frame, _ = build_eprx_driver_features(frame)
    frame = frame.copy()
    frame["datetime_jst"] = pd.to_datetime(frame["datetime_jst"], errors="coerce")
    if frame["datetime_jst"].dt.tz is None:
        frame["datetime_jst"] = frame["datetime_jst"].dt.tz_localize("Asia/Tokyo")
    else:
        frame["datetime_jst"] = frame["datetime_jst"].dt.tz_convert("Asia/Tokyo")
    frame = frame.loc[frame["datetime_jst"].ge(pd.Timestamp("2026-03-14", tz="Asia/Tokyo"))]
    if analysis_start is not None:
        start = pd.Timestamp(analysis_start)
        start = start.tz_localize("Asia/Tokyo") if start.tzinfo is None else start.tz_convert("Asia/Tokyo")
        frame = frame.loc[frame["datetime_jst"].ge(start)]
    if analysis_end is not None:
        end = pd.Timestamp(analysis_end)
        end = end.tz_localize("Asia/Tokyo") if end.tzinfo is None else end.tz_convert("Asia/Tokyo")
        frame = frame.loc[frame["datetime_jst"].le(end)]
    duplicate_mask = frame.duplicated("datetime_jst", keep=False)
    duplicate_count = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].sort_values("datetime_jst", kind="stable").reset_index(drop=True)
    for target, source in SOURCE_COLUMNS.items():
        frame[target] = pd.to_numeric(frame[source], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame["day_of_week"] = frame["datetime_jst"].dt.weekday
    frame["time_block"] = frame["datetime_jst"].dt.strftime("%H:%M")
    frame["analysis_date"] = frame["datetime_jst"].dt.normalize()
    diagnostics = {
        "input_rows": len(feature_df), "analysis_rows": len(frame),
        "duplicate_datetime_rows_excluded": duplicate_count,
        "invalid_datetime_rows": int(frame["datetime_jst"].isna().sum()),
        "infinite_values_replaced_with_missing": int(np.isinf(
            feature_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        ).sum()),
    }
    return frame, diagnostics


def add_leave_one_out_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    """Subtract weekday/time-block means that exclude the current observation."""
    result = data.copy()
    keys = ["day_of_week", "time_block"]
    for anomaly, source in ANOMALY_SOURCES.items():
        values = result[source]
        sums = values.groupby([result[key] for key in keys]).transform("sum")
        counts = values.notna().groupby([result[key] for key in keys]).transform("sum")
        baseline = (sums - values) / (counts - 1)
        result[anomaly] = (values - baseline).where(values.notna() & counts.ge(4))
    return result


def _strength(value: float | None) -> str:
    if value is None or not np.isfinite(value): return "unavailable"
    absolute = abs(value)
    if absolute < 0.20: return "negligible"
    if absolute < 0.40: return "weak"
    if absolute < 0.60: return "moderate"
    if absolute < 0.80: return "strong"
    return "very_strong"


def _direction(value: float | None) -> str:
    if value is None or not np.isfinite(value) or value == 0: return "none"
    return "positive" if value > 0 else "negative"


def _correlation(x: pd.Series, y: pd.Series, minimum: int = MIN_CORRELATION_SAMPLES) -> dict[str, Any]:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pair); x_values = pair.iloc[:, 0]; y_values = pair.iloc[:, 1]
    x_std = x_values.std(ddof=1); y_std = y_values.std(ddof=1)
    x_unique = int(x_values.nunique()); y_unique = int(y_values.nunique())
    reason = None
    if n < minimum: reason = "insufficient_samples"
    elif x_unique < 2 or y_unique < 2 or x_std == 0 or y_std == 0: reason = "constant_or_insufficient_unique_values"
    pearson = spearman = None
    if reason is None:
        pearson = float(x_values.corr(y_values, method="pearson"))
        # pandas uses average ranks for ties; Pearson of ranks is Spearman.
        spearman = float(x_values.rank(method="average").corr(y_values.rank(method="average")))
    return {
        "status": "available" if reason is None else "unavailable", "reason": reason,
        "sample_count": n, "pearson": pearson, "spearman": spearman,
        "target_standard_deviation": float(x_std) if pd.notna(x_std) else None,
        "variable_standard_deviation": float(y_std) if pd.notna(y_std) else None,
        "target_unique_value_count": int(x_unique), "variable_unique_value_count": int(y_unique),
        "pearson_direction": _direction(pearson), "pearson_strength": _strength(pearson),
        "spearman_direction": _direction(spearman), "spearman_strength": _strength(spearman),
    }


def calculate_correlations(data: pd.DataFrame, anomaly: bool = False) -> dict[str, dict[str, Any]]:
    target = "procurement_anomaly_mw" if anomaly else "procurement_volume_mw"
    output = {}
    for predictor in PREDICTORS:
        variable = ANOMALY_BY_PREDICTOR.get(predictor) if anomaly else predictor
        if variable not in data:
            continue
        output[predictor] = _correlation(data[target], data[variable])
    return output


def build_eprx_fast_context(
    feature_df: pd.DataFrame, region: str, week_start: Any,
) -> dict[str, Any]:
    """Build the default AI context without anomaly, bootstrap, or regression work."""
    base = build_eprx_analysis_context(feature_df, region, week_start)
    frame = _region_filter(feature_df, region)
    if not set(FEATURE_COLUMNS).issubset(frame.columns):
        frame, _ = build_eprx_driver_features(frame)
    timestamps = pd.to_datetime(frame["datetime_jst"], errors="coerce")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("Asia/Tokyo")
    else:
        timestamps = timestamps.dt.tz_convert("Asia/Tokyo")
    start = pd.Timestamp(week_start)
    start = start.tz_localize("Asia/Tokyo") if start.tzinfo is None else start.tz_convert("Asia/Tokyo")
    start = start.normalize()
    selected = frame.loc[timestamps.ge(start) & timestamps.lt(start + pd.Timedelta(days=7))].copy()
    selected["procurement_volume_mw"] = pd.to_numeric(selected["procurement_volume"], errors="coerce")
    selected["time_block"] = timestamps.loc[selected.index].dt.strftime("%H:%M")
    intraday_profile = (
        selected.groupby("time_block", sort=True)
        .agg(
            procurement_mw=("procurement_volume_mw", "mean"),
            demand_mw=("area_demand_mw", "mean"),
            residual_demand_mw=("residual_demand_proxy_mw", "mean"),
            renewable_generation_mw=("renewable_generation_mw", "mean"),
            observation_count=("datetime_jst", "count"),
        )
        .reset_index()
        .to_dict("records")
    )
    correlations = {}
    for predictor in FAST_PREDICTORS:
        source = SOURCE_COLUMNS[predictor]
        correlations[predictor] = _correlation(
            selected["procurement_volume_mw"], pd.to_numeric(selected[source], errors="coerce"), minimum=2)
    selected_week = {
        "week": base["week"], "procurement": base["procurement"],
        "daily_profile": base["daily_profile"], "notable_time_blocks": base["notable_time_blocks"],
        "demand_intraday_profile": intraday_profile,
        "historical_position": base["historical_position"],
        "procurement_change": base["previous_week_comparison"].get("procurement_mean_mw"),
        "driver_changes": {key: value for key, value in base["previous_week_comparison"].items()
                           if not key.startswith("procurement_")},
    }
    return to_json_safe({
        "analysis_type": "eprx_fast_weekly_association", "analysis_mode": "fast",
        "fast_context_version": FAST_CONTEXT_VERSION, "status": base["status"], "region": region,
        "selected_week": selected_week, "selected_week_correlations": correlations,
        # Compatibility alias for the fallback renderer; these are unadjusted selected-week associations.
        "time_adjusted_correlations": correlations,
        "profile_repetition": base["profile_repetition"], "data_quality": base["data_quality"],
        "limitations": [
            "This is retrospective statistical association analysis using public 30-minute actuals.",
            "Actual demand and renewable output may differ from forecasts available when EPRX procurement was decided.",
            "Residual demand is a proxy calculated from public data.",
        ],
    })


def _complete_dates(data: pd.DataFrame) -> list[pd.Timestamp]:
    counts = data.groupby("analysis_date")["datetime_jst"].nunique()
    return list(counts.loc[counts.eq(48)].index)


def block_bootstrap_interval(
    data: pd.DataFrame, target: str, variable: str, estimate: float | None,
    iterations: int = 500, random_seed: int = 42,
) -> dict[str, Any]:
    dates = _complete_dates(data)
    base = {"estimate": estimate, "ci_95_lower": None, "ci_95_upper": None,
            "bootstrap_unit": "date", "bootstrap_iterations_requested": iterations,
            "bootstrap_iterations_valid": 0, "random_seed": random_seed}
    if iterations <= 0:
        return {**base, "status": "unavailable", "reason": "bootstrap_iterations_must_be_positive"}
    if len(dates) < MIN_BOOTSTRAP_DAYS:
        return {**base, "status": "unavailable", "reason": "fewer_than_7_complete_dates"}
    rng = np.random.default_rng(random_seed); results = []
    pairs = data.loc[data["analysis_date"].isin(dates), ["analysis_date", target, variable]].dropna()
    groups = {date: group[[target, variable]].to_numpy(dtype=float, copy=False)
              for date, group in pairs.groupby("analysis_date", sort=False)}
    dates = [date for date in dates if date in groups]
    if len(dates) < MIN_BOOTSTRAP_DAYS:
        return {**base, "status": "unavailable", "reason": "fewer_than_7_complete_dates"}
    for _ in range(iterations):
        sampled = rng.choice(len(dates), size=len(dates), replace=True)
        sample = np.concatenate([groups[dates[index]] for index in sampled])
        if len(sample) >= 2 and np.unique(sample[:, 0]).size >= 2 and np.unique(sample[:, 1]).size >= 2:
            target_ranks = pd.Series(sample[:, 0], copy=False).rank(method="average").to_numpy()
            variable_ranks = pd.Series(sample[:, 1], copy=False).rank(method="average").to_numpy()
            value = np.corrcoef(target_ranks, variable_ranks)[0, 1]
            if np.isfinite(value):
                results.append(float(value))
    base["bootstrap_iterations_valid"] = len(results)
    if len(results) < iterations * 0.5:
        return {**base, "status": "unavailable", "reason": "fewer_than_half_iterations_valid"}
    return {**base, "ci_95_lower": float(np.percentile(results, 2.5)),
            "ci_95_upper": float(np.percentile(results, 97.5)), "status": "available", "reason": None}


def calculate_bootstrap_intervals(
    data: pd.DataFrame, raw: dict[str, Any], adjusted: dict[str, Any],
    iterations: int, random_seed: int,
) -> dict[str, Any]:
    output = {}
    ranked = sorted(
        (predictor for predictor in PREDICTORS if adjusted.get(predictor, {}).get("spearman") is not None
         and ANOMALY_BY_PREDICTOR.get(predictor)),
        key=lambda predictor: (-abs(adjusted[predictor]["spearman"]), predictor),
    )[:MAX_BOOTSTRAP_PREDICTORS]
    for predictor in ranked:
        anomaly_variable = ANOMALY_BY_PREDICTOR.get(predictor)
        output[predictor] = {
            "anomaly_spearman": block_bootstrap_interval(data, "procurement_anomaly_mw", anomaly_variable,
                adjusted.get(predictor, {}).get("spearman"), iterations, random_seed)
                if anomaly_variable else None,
        }
    return output


def fit_standardized_model(data: pd.DataFrame, name: str, predictors: list[str]) -> dict[str, Any]:
    target = "procurement_anomaly_mw"; excluded = []
    available = []
    for predictor in predictors:
        if predictor not in data or data[predictor].dropna().nunique() < 2 or data[predictor].std(ddof=1) == 0:
            excluded.append(predictor)
        else: available.append(predictor)
    base = {"model_name": name, "predictors": available, "excluded_predictors": excluded,
            "standardized_coefficients": {}, "warnings": []}
    if len(available) < 2:
        return {**base, "status": "unavailable", "reason": "fewer_than_2_predictors", "sample_count": 0}
    complete = data[[target, *available]].dropna()
    if len(complete) < MIN_REGRESSION_SAMPLES:
        return {**base, "status": "unavailable", "reason": "fewer_than_200_complete_cases", "sample_count": len(complete)}
    standardized = (complete - complete.mean()) / complete.std(ddof=1)
    y = standardized[target].to_numpy(); matrix = standardized[available].to_numpy()
    design = np.column_stack([np.ones(len(matrix)), matrix])
    coefficients, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients; residuals = y - fitted
    ss_res = float(np.sum(residuals ** 2)); ss_total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_total if ss_total else None
    adjusted = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - len(available) - 1) if r_squared is not None else None
    condition = float(np.linalg.cond(design)); warnings = []
    if rank < design.shape[1]: warnings.append("rank_deficient")
    if condition > 30: warnings.append("condition_number_above_30")
    return {**base, "status": "available", "reason": None, "sample_count": len(y),
            "r_squared": r_squared, "adjusted_r_squared": adjusted, "rank": int(rank),
            "condition_number": condition, "residual_std": float(np.std(residuals, ddof=len(available) + 1)),
            "standardized_coefficients": {column: float(value) for column, value in zip(available, coefficients[1:])},
            "warnings": warnings, "singular_values": singular.tolist()}


def _regression_models(data: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        fit_standardized_model(data, "model_a_demand", ["demand_anomaly_mw", "renewable_share_anomaly_pct",
            "abs_demand_ramp_anomaly_mw", "abs_renewable_ramp_anomaly_mw"]),
        fit_standardized_model(data, "model_b_residual_demand", ["residual_demand_proxy_anomaly_mw",
            "renewable_share_anomaly_pct", "abs_residual_demand_ramp_anomaly_mw", "abs_renewable_ramp_anomaly_mw"]),
    ]


def analyze_eprx_driver_statistics(
    feature_df: pd.DataFrame, region: str, analysis_start: Any = None, analysis_end: Any = None,
    bootstrap_iterations: int = 500, random_seed: int = 42,
) -> dict[str, Any]:
    data, quality = prepare_statistics_data(feature_df, region, analysis_start, analysis_end)
    data = add_leave_one_out_anomalies(data)
    raw = calculate_correlations(data); adjusted = calculate_correlations(data, anomaly=True)
    bootstrap = calculate_bootstrap_intervals(data, raw, adjusted, bootstrap_iterations, random_seed)
    warnings = []
    for variable, raw_relation in raw.items():
        adjusted_relation = adjusted.get(variable, {})
        if raw_relation.get("pearson_strength") in {"moderate", "strong", "very_strong"} and adjusted_relation.get("pearson_strength") in {"negligible", "weak"}:
            warnings.append(f"{variable}: raw association is materially reduced after weekday/time adjustment")
    if data["procurement_volume_mw"].nunique() < 5:
        warnings.append("procurement has very few unique values")
    result = {
        "analysis_type": "eprx_driver_statistical_association", "status": "ok" if len(data) else "source_data_missing",
        "region": region, "analysis_period": {"start": str(data["datetime_jst"].min()) if len(data) else None,
            "end": str(data["datetime_jst"].max()) if len(data) else None},
        "target_summary": {"sample_count": int(data["procurement_volume_mw"].notna().sum()),
            "mean": float(data["procurement_volume_mw"].mean()) if data["procurement_volume_mw"].notna().any() else None,
            "standard_deviation": float(data["procurement_volume_mw"].std(ddof=1)) if data["procurement_volume_mw"].notna().sum() > 1 else None,
            "unique_value_count": int(data["procurement_volume_mw"].nunique())},
        "raw_correlations": raw, "time_adjusted_correlations": adjusted,
        "bootstrap_intervals": bootstrap, "regression_models": _regression_models(data),
        "data_quality": quality,
        "interpretation_rules": {"strength_thresholds": [0.20, 0.40, 0.60, 0.80],
            "classification_is_descriptive_only": True},
        "warnings": warnings,
        "limitations": [
            "본 분석은 공개된 30분 실적을 이용한 사후 연관성 분석입니다.",
            "실제 수요·재생에너지 실적은 모집량 결정 시점의 예측치와 다를 수 있습니다.",
            "잔여수요는 공개자료를 이용한 추정치입니다.",
            "30분 자료로는 1차 조정력이 대응하는 초단주기 변동을 직접 재현할 수 없습니다.",
            "평상시분과 비상시분 모집량은 분리되어 있지 않습니다.",
            "자연체여력·수의계약·시장 외 조달은 포함되지 않습니다.",
            "상관계수와 회귀계수는 인과관계를 의미하지 않습니다.",
            "일자 단위 bootstrap은 시계열 의존성을 완전히 제거하지 못합니다.",
            "반복되는 모집량 프로파일은 원시 상관을 왜곡할 수 있습니다.",
            "본 분석은 모집량 예측 모델이 아닙니다.",
        ],
        "calculation_metadata": {"minimum_correlation_samples": MIN_CORRELATION_SAMPLES,
            "minimum_regression_samples": MIN_REGRESSION_SAMPLES, "minimum_bootstrap_complete_days": MIN_BOOTSTRAP_DAYS,
            "spearman_method": "Pearson correlation of average ranks", "anomaly_baseline": "leave-one-out weekday/time_block mean",
            "bootstrap_unit": "date", "bootstrap_iterations": bootstrap_iterations, "random_seed": random_seed},
    }
    return to_json_safe(result)


COMPARISON_VARIABLES = ("demand_mw", "residual_demand_proxy_mw", "renewable_generation_mw",
    "renewable_share_pct", "abs_demand_ramp_30m_mw", "abs_residual_demand_ramp_30m_mw",
    "abs_renewable_ramp_30m_mw")


def co_movement_comparison(data: pd.DataFrame) -> dict[str, Any]:
    target = data["procurement_volume_mw"].dropna()
    if target.empty: return {"status": "unavailable", "reason": "missing_procurement", "variables": {}}
    low_cut = target.quantile(0.2); high_cut = target.quantile(0.8)
    low = data["procurement_volume_mw"].le(low_cut); high = data["procurement_volume_mw"].ge(high_cut)
    tie_warning = int(low.sum() + high.sum()) > int(np.ceil(len(target) * 0.4))
    output = {}
    for variable in COMPARISON_VARIABLES:
        high_values = data.loc[high, variable].dropna(); low_values = data.loc[low, variable].dropna()
        high_mean = high_values.mean(); low_mean = low_values.mean()
        output[variable] = {"high_group_count": len(high_values), "low_group_count": len(low_values),
            "high_group_mean": high_mean, "low_group_mean": low_mean,
            "high_group_median": high_values.median(), "low_group_median": low_values.median(),
            "mean_difference": high_mean - low_mean, "median_difference": high_values.median() - low_values.median(),
            "mean_difference_pct": (high_mean - low_mean) / low_mean * 100 if pd.notna(low_mean) and low_mean != 0 else None,
            "status": "available" if len(high_values) and len(low_values) else "unavailable",
            "reason": None if len(high_values) and len(low_values) else "empty_group"}
    return to_json_safe({"status": "available", "comparison_type": "co_movement_comparison",
        "low_quantile_boundary": low_cut, "high_quantile_boundary": high_cut,
        "quantile_tie_warning": tie_warning, "variables": output})


def _association_candidates(base: dict[str, Any], statistics: dict[str, Any]) -> dict[str, Any]:
    procurement = base["previous_week_comparison"].get("procurement_mean_mw", {})
    change = procurement.get("change"); change_pct = procurement.get("change_pct")
    if change is None or (abs(change) < 1 and (change_pct is None or abs(change_pct) < 1)):
        return {"status": "no_material_procurement_change", "items": []}
    mapping = {
        "demand_mw": ("mean_demand_mw", "mean_demand_mw", "Average demand"),
        "residual_demand_proxy_mw": ("mean_residual_demand_proxy_mw", "mean_residual_demand_proxy_mw", "Average residual demand proxy"),
        "renewable_generation_mw": ("mean_renewable_generation_mw", None, "Average renewable generation"),
        "renewable_share_pct": ("mean_renewable_share_pct", "mean_renewable_share_pct", "Average renewable share"),
        "abs_demand_ramp_30m_mw": ("mean_abs_demand_ramp_30m_mw", "mean_abs_demand_ramp_30m_mw", "Average absolute demand ramp"),
        "abs_residual_demand_ramp_30m_mw": ("mean_abs_residual_demand_ramp_30m_mw", "mean_abs_residual_demand_ramp_30m_mw", "Average absolute residual-demand ramp"),
        "abs_renewable_ramp_30m_mw": ("mean_abs_renewable_ramp_30m_mw", "mean_abs_renewable_ramp_30m_mw", "Average absolute renewable ramp"),
    }
    items = []
    for variable, (comparison_key, history_key, display) in mapping.items():
        comparison = base["previous_week_comparison"].get(comparison_key, {})
        relation = statistics["time_adjusted_correlations"].get(variable, {})
        rho = relation.get("spearman")
        history = base["historical_position"].get(history_key, {}) if history_key else {}
        percentile = history.get("percentile"); zscore = history.get("z_score")
        historical_score = abs(percentile - 50) / 50 * 40 if percentile is not None else min(abs(zscore or 0) / 3, 1) * 40
        correlation_score = abs(rho or 0) * 40
        variable_change = comparison.get("change")
        consistent = bool(rho and variable_change is not None and change * variable_change * rho > 0)
        score = min(100, historical_score + correlation_score + (20 if consistent else 0))
        interval = statistics["bootstrap_intervals"].get(variable, {}).get("anomaly_spearman", {})
        lower, upper = interval.get("ci_95_lower"), interval.get("ci_95_upper")
        weak = rho is None or (lower is not None and upper is not None and lower <= 0 <= upper)
        items.append({"variable": variable, "display_name": display, "association_relevance_score": score,
            "current_week_value": comparison.get("current"), "previous_week_value": comparison.get("previous"),
            "change": variable_change, "change_pct": comparison.get("change_pct"),
            "historical_percentile": percentile, "historical_zscore": zscore, "adjusted_spearman": rho,
            "bootstrap_ci_95": [lower, upper] if lower is not None else None, "direction_consistent": consistent,
            "interpretation_status": "weak_evidence" if weak else "candidate_association"})
    items.sort(key=lambda item: (-item["association_relevance_score"], item["variable"]))
    for rank, item in enumerate(items[:5], 1): item["rank"] = rank
    return {"status": "candidate_association", "items": items[:5]}


def analyze_eprx_selected_week_statistics(
    feature_df: pd.DataFrame, region: str, week_start: Any,
    historical_statistics: dict[str, Any] | None = None,
    base_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    statistics = historical_statistics or analyze_eprx_driver_statistics(feature_df, region)
    prepared, _ = prepare_statistics_data(feature_df, region)
    start = pd.Timestamp(week_start)
    start = start.tz_localize("Asia/Tokyo") if start.tzinfo is None else start.tz_convert("Asia/Tokyo")
    end = start.normalize() + pd.Timedelta(days=7)
    selected = prepared.loc[prepared["datetime_jst"].ge(start.normalize()) & prepared["datetime_jst"].lt(end)]
    base = base_context or build_eprx_analysis_context(feature_df, region, week_start)
    return to_json_safe({"week": base["week"],
        "procurement": base["procurement"],
        "daily_profile": base["daily_profile"],
        "notable_time_blocks": base["notable_time_blocks"],
        "historical_position": base["historical_position"],
        "procurement_change": base["previous_week_comparison"].get("procurement_mean_mw"),
        "driver_changes": {key: value for key, value in base["previous_week_comparison"].items() if not key.startswith("procurement_")},
        "association_candidates": _association_candidates(base, statistics),
        "co_movement_comparison": co_movement_comparison(selected)})


def build_eprx_statistical_context(
    feature_df: pd.DataFrame, region: str, week_start: Any, analysis_start: Any = None,
    analysis_end: Any = None, bootstrap_iterations: int = RUNTIME_BOOTSTRAP_ITERATIONS,
    random_seed: int = 42,
) -> dict[str, Any]:
    base = build_eprx_analysis_context(feature_df, region, week_start)
    statistics = analyze_eprx_driver_statistics(feature_df, region, analysis_start, analysis_end,
        bootstrap_iterations, random_seed)
    selected = analyze_eprx_selected_week_statistics(feature_df, region, week_start, statistics, base)
    repetition = base["profile_repetition"]
    warnings = list(statistics["warnings"])
    complete = repetition.get("complete_week_count", 0); repeated = repetition.get("weeks_identical_to_previous_count", 0)
    if complete and (repeated / complete >= 0.30 or repetition.get("unique_weekly_profile_count", complete) <= complete * 0.5):
        warnings.append("Procurement uses repeated time-of-day profiles, so raw correlations may be strongly influenced by time patterns.")
    result = {**statistics, "profile_repetition": repetition, "selected_week": selected,
              "warnings": warnings, "calculation_metadata": {**statistics["calculation_metadata"],
              "association_relevance_score": "historical deviation 40 + abs anomaly Spearman 40 + direction consistency 20"}}
    result["profile_repetition"]["correlation_change_after_time_adjustment"] = {
        variable: {
            "raw_spearman": relation.get("spearman"),
            "anomaly_spearman": statistics["time_adjusted_correlations"].get(variable, {}).get("spearman"),
            "difference": (
                statistics["time_adjusted_correlations"].get(variable, {}).get("spearman") - relation.get("spearman")
                if relation.get("spearman") is not None
                and statistics["time_adjusted_correlations"].get(variable, {}).get("spearman") is not None
                else None
            ),
        }
        for variable, relation in statistics["raw_correlations"].items()
    }
    return to_json_safe(result)
