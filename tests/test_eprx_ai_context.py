import pandas as pd
import pytest

from utils.eprx_ai_context import build_eprx_procurement_context
from utils.sample_data import generate_sample_data


def test_builds_json_safe_context_from_existing_normalized_frame():
    data = generate_sample_data("2026-07-20")
    data["source_status"] = "確報値"

    result = build_eprx_procurement_context(data, "Tokyo", "2026-07-13")

    assert result["region"] == "Tokyo"
    assert result["week_start"] == "2026-07-13"
    assert result["week_end"] == "2026-07-19"
    assert result["complete_week"] is True
    assert result["market_regimes"] == ["modern_30minute"]
    assert result["source_statuses"] == ["確報値"]
    assert len(result["period_profile"]) == 48
    assert result["weekly_average_procurement_mw"] == pytest.approx(
        data.loc[
            data["area"].eq("Tokyo")
            & pd.to_datetime(data["delivery_date"]).ge("2026-07-13"),
            "procurement_volume",
        ].mean()
    )


def test_missing_week_returns_explicit_empty_context():
    data = generate_sample_data("2026-07-20")

    result = build_eprx_procurement_context(data, "Chubu", "2025-01-06")

    assert result["weekly_average_procurement_mw"] is None
    assert result["observed_days"] == 0
    assert result["complete_week"] is False
    assert result["period_profile"] == []


def test_rejects_regions_outside_requested_scope():
    with pytest.raises(ValueError, match="지원하지 않는 EPRX 지역"):
        build_eprx_procurement_context(
            generate_sample_data("2026-07-20"), "Kansai", "2026-07-20"
        )


def test_reform_week_keeps_legacy_and_modern_periods_separate():
    rows = []
    for date in pd.date_range("2026-03-09", periods=7):
        count = 8 if date < pd.Timestamp("2026-03-14") else 48
        for period in range(1, count + 1):
            rows.append(
                {
                    "delivery_date": date,
                    "period_no": period,
                    "period_start": (
                        f"{(period - 1) * 3:02d}:00"
                        if count == 8
                        else f"{(period - 1) // 2:02d}:{((period - 1) % 2) * 30:02d}"
                    ),
                    "area": "Tokyo",
                    "frequency_zone": "50Hz",
                    "procurement_volume": 100.0,
                    "bid_volume": 120.0,
                    "awarded_volume": 90.0,
                    "avg_price": 5.0,
                    "max_price": 6.0,
                    "min_price": 4.0,
                }
            )

    result = build_eprx_procurement_context(
        pd.DataFrame(rows), "Tokyo", "2026-03-09"
    )

    assert result["market_regimes"] == ["legacy_3hour", "modern_30minute"]
    assert len(result["period_profile"]) == 56
    assert result["complete_week"] is True
