import json

import pandas as pd
import pytest

from utils.eprx_driver_features import (
    build_eprx_driver_features,
    build_eprx_driver_weekly_context,
)


def _merged_week() -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-13", periods=336, freq="30min")
    return pd.DataFrame(
        {
            "delivery_date": timestamps.normalize(),
            "period_no": timestamps.hour * 2 + timestamps.minute // 30 + 1,
            "period_start": timestamps.strftime("%H:%M"),
            "procurement_volume": 100.0,
            "area_demand_mw": range(20_000, 20_336),
            "solar_generation_mw": 1_000.0,
            "wind_generation_mw": 100.0,
            "join_status": "both",
        }
    )


def test_builds_confirmed_arithmetic_features_and_jst_datetime():
    features, diagnostics = build_eprx_driver_features(_merged_week())

    assert str(features["datetime_jst"].dt.tz) == "Asia/Tokyo"
    assert features.loc[0, "variable_renewable_generation_mw"] == 1100
    assert features.loc[0, "residual_demand_mw"] == 18900
    assert features.loc[0, "variable_renewable_share"] == pytest.approx(0.055)
    assert pd.isna(features.loc[0, "demand_change_mw_30min"])
    assert features.loc[1, "demand_change_mw_30min"] == 1
    assert diagnostics["join_success_rate"] == 1.0
    assert diagnostics["duplicate_key_rows"] == 0


def test_missing_input_propagates_without_zero_fill():
    merged = _merged_week().iloc[:2].copy()
    merged.loc[1, "solar_generation_mw"] = None

    features, diagnostics = build_eprx_driver_features(merged)

    assert pd.isna(features.loc[1, "variable_renewable_generation_mw"])
    assert pd.isna(features.loc[1, "residual_demand_mw"])
    assert diagnostics["missing_value_counts"]["solar_generation_mw"] == 1


def test_change_is_missing_across_non_contiguous_interval():
    merged = _merged_week().iloc[[0, 2]].copy()
    features, _ = build_eprx_driver_features(merged)
    assert pd.isna(features.loc[1, "demand_change_mw_30min"])


def test_weekly_context_is_complete_and_json_serializable():
    features, _ = build_eprx_driver_features(_merged_week())
    context = build_eprx_driver_weekly_context(features, "2026-07-13")

    assert context["complete_week"] is True
    assert context["matched_rows"] == 336
    assert context["timezone"] == "Asia/Tokyo"
    assert context["metrics"]["area_demand_mw"]["mean"] == pytest.approx(20167.5)
    json.dumps(context, ensure_ascii=False)
