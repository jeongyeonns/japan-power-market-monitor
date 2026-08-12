import json

import numpy as np
import pandas as pd
import pytest
import utils.eprx_driver_statistics as statistics_module

from utils.eprx_driver_features import build_eprx_driver_features
from utils.eprx_driver_statistics import (
    ANOMALY_SOURCES,
    _correlation,
    _association_candidates,
    add_leave_one_out_anomalies,
    analyze_eprx_driver_statistics,
    block_bootstrap_interval,
    build_eprx_fast_context,
    build_eprx_statistical_context,
    co_movement_comparison,
    fit_standardized_model,
    prepare_statistics_data,
)


def _feature_history(region="Tokyo", weeks=8, anomaly_strength=2.0):
    timestamps = pd.date_range("2026-04-06", periods=336 * weeks, freq="30min")
    period = timestamps.hour * 2 + timestamps.minute // 30
    week = np.arange(len(timestamps)) // 336
    recurring = np.sin(period / 48 * 2 * np.pi)
    deviation = np.tile(np.linspace(-1, 1, 336), weeks)
    demand = 20_000 + recurring * 2_000 + deviation * 100
    procurement = 100 + recurring * 20 + week + anomaly_strength * deviation
    raw = pd.DataFrame({
        "delivery_date": timestamps.normalize(), "period_no": period + 1,
        "period_start": timestamps.strftime("%H:%M"), "procurement_volume": procurement,
        "area_demand_mw": demand, "solar_generation_mw": 500 + recurring * 200,
        "wind_generation_mw": 300 - recurring * 50 + deviation * 10,
        "area": region, "join_status": "both",
    })
    return build_eprx_driver_features(raw)[0]


def test_pearson_spearman_positive_negative_ties_and_constant():
    x = pd.Series(np.arange(120, dtype=float))
    positive = _correlation(x, x * 2)
    negative = _correlation(x, -x)
    ties = _correlation(x, pd.Series(np.repeat(np.arange(40), 3)))
    constant = _correlation(x, pd.Series(1.0, index=x.index))
    assert positive["pearson"] == pytest.approx(1)
    assert positive["spearman"] == pytest.approx(1)
    assert negative["pearson"] == pytest.approx(-1)
    assert ties["spearman"] > 0.99
    assert constant["status"] == "unavailable"
    assert _correlation(x.iloc[:99], x.iloc[:99])["reason"] == "insufficient_samples"


def test_leave_one_out_excludes_current_and_requires_four_group_values():
    frame = pd.DataFrame({
        "day_of_week": [0] * 4 + [1] * 3, "time_block": ["00:00"] * 7,
        **{source: [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0]
           for source in set(ANOMALY_SOURCES.values())},
    })
    result = add_leave_one_out_anomalies(frame)
    assert result.loc[0, "procurement_anomaly_mw"] == pytest.approx(1 - 3)
    assert pd.isna(result.loc[4, "procurement_anomaly_mw"])


def test_anomaly_correlations_preserve_positive_and_negative_relationships():
    values = np.arange(120, dtype=float)
    frame = pd.DataFrame({"day_of_week": [0] * 120, "time_block": ["00:00"] * 120})
    for source in set(ANOMALY_SOURCES.values()):
        frame[source] = values
    adjusted = add_leave_one_out_anomalies(frame)
    positive = _correlation(adjusted["procurement_anomaly_mw"], adjusted["demand_anomaly_mw"])
    negative = _correlation(adjusted["procurement_anomaly_mw"], -adjusted["demand_anomaly_mw"])
    assert positive["spearman"] == pytest.approx(1)
    assert negative["spearman"] == pytest.approx(-1)


def test_raw_time_pattern_reduces_after_leave_one_out_adjustment():
    statistics = analyze_eprx_driver_statistics(_feature_history(anomaly_strength=0), "Tokyo", bootstrap_iterations=8)
    raw = abs(statistics["raw_correlations"]["demand_mw"]["spearman"])
    adjusted = abs(statistics["time_adjusted_correlations"]["demand_mw"]["spearman"])
    assert raw > 0.7
    assert adjusted < raw


def test_date_block_bootstrap_is_reproducible_and_requires_seven_days():
    prepared, _ = prepare_statistics_data(_feature_history(weeks=2), "Tokyo")
    first = block_bootstrap_interval(prepared, "procurement_volume_mw", "demand_mw", 0.5, 20, 7)
    second = block_bootstrap_interval(prepared, "procurement_volume_mw", "demand_mw", 0.5, 20, 7)
    assert first == second
    assert first["status"] == "available"
    short = prepared.loc[prepared["analysis_date"].isin(prepared["analysis_date"].unique()[:6])]
    assert block_bootstrap_interval(short, "procurement_volume_mw", "demand_mw", 0.5, 10, 7)["status"] == "unavailable"


def test_standardized_regression_coefficients_r_squared_and_collinearity_warning():
    x = np.linspace(-2, 2, 300); z = np.sin(np.arange(300))
    data = pd.DataFrame({"procurement_anomaly_mw": 2 * x + 0.5 * z,
                         "x": x, "z": z, "x_duplicate": x})
    model = fit_standardized_model(data, "test", ["x", "z"])
    assert model["status"] == "available"
    assert model["r_squared"] == pytest.approx(1)
    assert model["adjusted_r_squared"] == pytest.approx(1)
    deficient = fit_standardized_model(data, "collinear", ["x", "x_duplicate"])
    assert "rank_deficient" in deficient["warnings"]
    assert "condition_number_above_30" in deficient["warnings"]


def test_high_low_comparison_reports_ties_and_actual_group_counts():
    data = pd.DataFrame({"procurement_volume_mw": [0] * 50 + [1] * 50})
    for column in ("demand_mw", "residual_demand_proxy_mw", "renewable_generation_mw",
                   "renewable_share_pct", "abs_demand_ramp_30m_mw",
                   "abs_residual_demand_ramp_30m_mw", "abs_renewable_ramp_30m_mw"):
        data[column] = np.arange(100)
    result = co_movement_comparison(data)
    assert result["quantile_tie_warning"] is True
    assert result["variables"]["demand_mw"]["high_group_count"] == 50
    assert result["variables"]["demand_mw"]["low_group_count"] == 50


@pytest.mark.parametrize("region", ["Tokyo", "Chubu"])
def test_common_statistical_context_is_json_safe(region):
    context = build_eprx_statistical_context(
        _feature_history(region=region), region, "2026-05-25",
        bootstrap_iterations=8, random_seed=3,
    )
    assert context["region"] == region
    assert "raw_correlations" in context
    assert "time_adjusted_correlations" in context
    assert "association_candidates" in context["selected_week"]
    assert "procurement" in context["selected_week"]
    assert "daily_profile" in context["selected_week"]
    assert "notable_time_blocks" in context["selected_week"]
    assert "historical_position" in context["selected_week"]
    assert not any(item.startswith("This is ") for item in context["limitations"])
    json.dumps(context, allow_nan=False)


def test_fast_context_builds_48_slot_demand_profile_from_seven_daily_values():
    features = _feature_history()
    context = build_eprx_fast_context(features, "Tokyo", "2026-05-25")
    profile = context["selected_week"]["demand_intraday_profile"]
    assert len(profile) == 48
    assert {row["observation_count"] for row in profile} == {7}
    slot = next(row for row in profile if row["time_block"] == "08:30")
    timestamps = pd.to_datetime(features["datetime_jst"])
    expected = features.loc[
        timestamps.dt.date >= pd.Timestamp("2026-05-25").date()
    ].loc[timestamps.dt.strftime("%H:%M").eq("08:30"), "area_demand_mw"].mean()
    assert slot["demand_mw"] == pytest.approx(expected)
    assert slot["procurement_mw"] == pytest.approx(features.loc[
        timestamps.dt.date >= pd.Timestamp("2026-05-25").date()
    ].loc[timestamps.dt.strftime("%H:%M").eq("08:30"), "procurement_volume"].mean())
    assert slot["renewable_generation_mw"] == pytest.approx(features.loc[
        timestamps.dt.date >= pd.Timestamp("2026-05-25").date()
    ].loc[timestamps.dt.strftime("%H:%M").eq("08:30"), "renewable_generation_mw"].mean())


def test_correlation_strength_uses_non_exaggerating_thresholds():
    relation = _correlation(pd.Series(range(120)), pd.Series(range(120)) * 0.51)
    assert relation["pearson_strength"] == "very_strong"
    from utils.eprx_driver_statistics import _strength
    assert _strength(0.51) == "moderate"
    assert _strength(0.61) == "strong"


def test_no_material_procurement_change_suppresses_candidates():
    features = _feature_history()
    selected = features["delivery_date"].between("2026-05-18", "2026-05-31")
    profile = np.tile(features.loc[features["delivery_date"].between("2026-05-18", "2026-05-24"), "procurement_volume"].to_numpy(), 2)
    features.loc[selected, "procurement_volume"] = profile
    context = build_eprx_statistical_context(features, "Tokyo", "2026-05-25", bootstrap_iterations=6)
    assert context["selected_week"]["association_candidates"]["status"] == "no_material_procurement_change"


def test_context_build_runs_base_statistics_and_regressions_once(monkeypatch):
    calls = {"base": 0, "statistics": 0}
    original_base = statistics_module.build_eprx_analysis_context
    original_statistics = statistics_module.analyze_eprx_driver_statistics

    def counted_base(*args, **kwargs):
        calls["base"] += 1
        return original_base(*args, **kwargs)

    def counted_statistics(*args, **kwargs):
        calls["statistics"] += 1
        return original_statistics(*args, **kwargs)

    monkeypatch.setattr(statistics_module, "build_eprx_analysis_context", counted_base)
    monkeypatch.setattr(statistics_module, "analyze_eprx_driver_statistics", counted_statistics)
    context = build_eprx_statistical_context(
        _feature_history(), "Tokyo", "2026-05-25", bootstrap_iterations=2)
    assert context["regression_models"]
    assert calls == {"base": 1, "statistics": 1}


def test_fast_context_skips_anomaly_bootstrap_and_regression(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("heavy statistics must not run in the FAST path")
    monkeypatch.setattr(statistics_module, "add_leave_one_out_anomalies", forbidden)
    monkeypatch.setattr(statistics_module, "calculate_bootstrap_intervals", forbidden)
    monkeypatch.setattr(statistics_module, "_regression_models", forbidden)
    context = build_eprx_fast_context(_feature_history(), "Tokyo", "2026-05-25")
    assert context["analysis_mode"] == "fast"
    assert len(context["selected_week_correlations"]) == 8
    assert "abs_residual_demand_ramp_30m_mw" in context["selected_week_correlations"]
    assert context["selected_week"]["procurement"]["valid_count"] == 336


def test_candidate_score_range_and_zero_crossing_ci_is_weak_evidence():
    base = {
        "previous_week_comparison": {
            "procurement_mean_mw": {"change": 10.0, "change_pct": 5.0},
            "mean_demand_mw": {"current": 110.0, "previous": 100.0, "change": 10.0, "change_pct": 10.0},
        },
        "historical_position": {
            "mean_demand_mw": {"percentile": 90.0, "z_score": 1.5},
        },
    }
    statistics = {
        "time_adjusted_correlations": {"demand_mw": {"spearman": 0.5}},
        "bootstrap_intervals": {"demand_mw": {"anomaly_spearman": {
            "ci_95_lower": -0.1, "ci_95_upper": 0.4,
        }}},
    }
    result = _association_candidates(base, statistics)
    demand = next(item for item in result["items"] if item["variable"] == "demand_mw")
    assert 0 <= demand["association_relevance_score"] <= 100
    assert demand["direction_consistent"] is True
    assert demand["interpretation_status"] == "weak_evidence"
