import json

import numpy as np
import pandas as pd
import pytest

from utils.eprx_ai_context import build_eprx_analysis_context
from utils.eprx_driver_features import build_eprx_driver_features


def _weeks(count=6, region="Tokyo", start="2026-05-04"):
    timestamps = pd.date_range(start, periods=336 * count, freq="30min")
    week_index = np.arange(len(timestamps)) // 336
    period = timestamps.hour * 2 + timestamps.minute // 30 + 1
    return pd.DataFrame({
        "delivery_date": timestamps.normalize(),
        "period_no": period,
        "period_start": timestamps.strftime("%H:%M"),
        "procurement_volume": 100.0 + (period % 4) + week_index,
        "area_demand_mw": 20_000.0 + period + week_index * 10,
        "solar_generation_mw": np.maximum(0, 1_000 - abs(period - 25) * 50),
        "wind_generation_mw": 200.0 + period,
        "area": region,
        "join_status": "both",
    })


def test_standard_features_aliases_percent_and_all_absolute_ramps():
    source = _weeks(1)
    features, _ = build_eprx_driver_features(source)
    expected = {
        "renewable_generation_mw", "renewable_share_pct", "residual_demand_proxy_mw",
        "demand_ramp_30m_mw", "abs_demand_ramp_30m_mw", "renewable_ramp_30m_mw",
        "abs_renewable_ramp_30m_mw", "residual_demand_ramp_30m_mw",
        "abs_residual_demand_ramp_30m_mw", "solar_ramp_30m_mw",
        "abs_solar_ramp_30m_mw", "wind_ramp_30m_mw", "abs_wind_ramp_30m_mw",
    }
    assert expected <= set(features)
    assert features["renewable_share_pct"].between(0, 100).all()
    for ramp in ("demand", "renewable", "residual_demand", "solar", "wind"):
        assert features[f"abs_{ramp}_ramp_30m_mw"].equals(features[f"{ramp}_ramp_30m_mw"].abs())
    assert features["variable_renewable_generation_mw"].equals(features["renewable_generation_mw"])
    assert np.allclose(features["variable_renewable_share"], features["renewable_share_pct"] / 100)


def test_ramps_cross_midnight_but_not_gap_region_change_or_missing_input():
    source = _weeks(1).iloc[46:51].copy()
    source = pd.concat([source, _weeks(1, region="Chubu").iloc[51:53]], ignore_index=True)
    source.loc[source.index[3], "solar_generation_mw"] = np.nan
    features, _ = build_eprx_driver_features(source)
    tokyo = features.loc[features["area"].eq("Tokyo")].reset_index(drop=True)
    assert tokyo.loc[2, "period_start"] == "00:00"
    assert pd.notna(tokyo.loc[2, "demand_ramp_30m_mw"])
    assert pd.isna(tokyo.loc[3, "renewable_generation_mw"])
    chubu = features.loc[features["area"].eq("Chubu")].reset_index(drop=True)
    assert pd.isna(chubu.loc[0, "demand_ramp_30m_mw"])

    nonpositive = _weeks(1).iloc[:2].copy()
    nonpositive.loc[nonpositive.index[0], "area_demand_mw"] = 0
    nonpositive.loc[nonpositive.index[1], "area_demand_mw"] = -1
    invalid_share, _ = build_eprx_driver_features(nonpositive)
    assert invalid_share["renewable_share_pct"].isna().all()


def test_complete_context_profiles_comparison_repetition_history_and_json():
    source = _weeks(6)
    # Make weeks 4 and 5 identical to exercise tolerance-based repetition.
    source.loc[source.index[336 * 4:336 * 6], "procurement_volume"] = np.tile(
        source.iloc[:336]["procurement_volume"].to_numpy(), 2
    )
    source.loc[source.index[336 * 5], "procurement_volume"] += 5e-10
    context = build_eprx_analysis_context(source, "Tokyo", "2026-06-08")
    assert context["week"]["complete"] is True
    assert context["procurement"]["valid_count"] == 336
    assert context["procurement"]["standard_deviation"] is not None
    assert len(context["daily_profile"]) == 7
    assert all(day["complete_day"] for day in context["daily_profile"])
    assert len(context["time_profile"]) == 48
    assert context["previous_week_comparison"]["procurement_mean_mw"]["status"] == "available"
    assert context["profile_repetition"]["same_as_previous_week"] is True
    assert context["profile_repetition"]["tolerance_mw"] == 1e-9
    assert context["historical_position"]["procurement_mean_mw"]["historical_week_count"] == 5
    assert context["historical_position"]["procurement_mean_mw"]["percentile"] is not None
    json.dumps(context, allow_nan=False)


def test_incomplete_previous_week_and_zero_previous_change_pct():
    source = _weeks(2)
    source.loc[source.index[:336], "procurement_volume"] = 0.0
    context = build_eprx_analysis_context(source, "Tokyo", "2026-05-11")
    comparison = context["previous_week_comparison"]["procurement_mean_mw"]
    assert comparison["previous"] == 0
    assert comparison["change_pct"] is None
    assert comparison["reason"] == "previous_value_zero"
    assert context["historical_position"]["procurement_mean_mw"]["percentile"] is None
    incomplete = build_eprx_analysis_context(source.drop(index=0), "Tokyo", "2026-05-11")
    assert incomplete["previous_week_comparison"]["procurement_mean_mw"]["status"] == "unavailable"


@pytest.mark.parametrize("region", ["Tokyo", "Chubu"])
def test_common_region_processing_and_incomplete_selected_week(region):
    source = _weeks(1, region=region).iloc[:-1]
    context = build_eprx_analysis_context(source, region, "2026-05-04")
    assert context["region"] == region
    assert context["week"]["complete"] is False
    assert context["data_quality"]["actual_rows"] == 335
    assert context["data_quality"]["json_serializable"] is True
