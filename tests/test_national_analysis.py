import numpy as np
import pandas as pd
import pytest

from utils.national_analysis import (
    calculate_national_kpis,
    calculate_national_week_over_week,
    create_national_summary_text,
    create_national_weekly_profile,
    create_regional_market_summary,
    validate_national_profile,
)
from utils.sample_data import generate_sample_data
from utils.weekly_aggregation import (
    add_week_columns,
    create_regional_weekly_profile,
)


@pytest.fixture(scope="module")
def profiles():
    data = add_week_columns(generate_sample_data("2026-07-20"))
    weeks = sorted(data["week_start"].unique())
    previous = create_regional_weekly_profile(data, weeks[-2])
    current = create_regional_weekly_profile(data, weeks[-1])
    return previous, current


def test_nine_regions_are_aggregated_to_48_periods(profiles):
    _, regional = profiles
    national = create_national_weekly_profile(regional)
    assert len(regional) == 9 * 48
    assert len(national) == 48
    assert national["area_count"].eq(9).all()
    first = national.iloc[0]
    source = regional[regional["period_no"].eq(first["period_no"])]
    for column in ("procurement_volume", "bid_volume", "awarded_volume"):
        assert first[column] == pytest.approx(source[column].sum())


def test_weighted_price_ratios_and_extremes(profiles):
    _, regional = profiles
    national = create_national_weekly_profile(regional)
    row = national.iloc[0]
    source = regional[regional["period_no"].eq(row["period_no"])]
    expected_price = (
        source["avg_price"] * source["awarded_volume"]
    ).sum() / source["awarded_volume"].sum()
    assert row["avg_price"] == pytest.approx(expected_price)
    assert row["bid_coverage_ratio"] == pytest.approx(
        row["bid_volume"] / row["procurement_volume"]
    )
    assert row["procurement_rate"] == pytest.approx(
        row["awarded_volume"] / row["procurement_volume"]
    )
    assert row["max_price"] == source["max_price"].max()
    assert row["min_price"] == source["min_price"].min()
    assert row["price_range"] == pytest.approx(
        row["max_price"] - row["min_price"]
    )


def test_zero_awarded_volume_makes_weighted_price_nan(profiles):
    _, regional = profiles
    modified = regional.copy()
    modified.loc[modified["period_no"].eq(1), "awarded_volume"] = 0
    national = create_national_weekly_profile(modified)
    assert np.isnan(national.loc[national["period_no"].eq(1), "avg_price"].iloc[0])


def test_missing_area_is_marked_incomplete(profiles):
    _, regional = profiles
    modified = regional[~regional["area"].eq("Tokyo")]
    national = create_national_weekly_profile(modified)
    assert national["area_count"].eq(8).all()
    assert national["missing_areas"].str.contains("Tokyo").all()
    assert national["completeness_flag"].eq("Incomplete").all()
    warnings = validate_national_profile(national, modified, 7)
    assert any("도쿄" in warning for warning in warnings)


def test_regional_summary_has_nine_rows(profiles):
    _, regional = profiles
    summary = create_regional_market_summary(regional)
    assert len(summary) == 9
    assert set(summary["frequency_zone"]) == {"50Hz", "60Hz"}
    assert summary["observed_period_count"].eq(48).all()


def test_week_over_week_changes_and_zero_previous(profiles):
    previous_regional, current_regional = profiles
    previous = create_national_weekly_profile(previous_regional)
    current = create_national_weekly_profile(current_regional)
    comparison = calculate_national_week_over_week(
        current, previous, current_regional, previous_regional
    )
    row = comparison.iloc[0]
    assert row["절대 변화"] == pytest.approx(row["현재 주"] - row["전주"])
    assert row["변화율"] == pytest.approx(row["절대 변화"] / row["전주"])
    rate = comparison[comparison["지표"].eq("전국 조달률 (%)")].iloc[0]
    assert rate["절대 변화 단위"] == "%p"

    zero_previous = previous.copy()
    zero_previous["procurement_volume"] = 0
    zero_comparison = calculate_national_week_over_week(
        current, zero_previous, current_regional, previous_regional
    )
    procurement = zero_comparison[
        zero_comparison["지표"].eq("평균 모집량 (MW)")
    ].iloc[0]
    assert np.isnan(procurement["변화율"])


def test_kpis_and_rule_based_summary_handle_ties_and_incomplete(profiles):
    _, regional = profiles
    national = create_national_weekly_profile(regional)
    kpis = calculate_national_kpis(national, regional)
    assert kpis["전국 조달률 (%)"] == pytest.approx(
        national["awarded_volume"].sum()
        / national["procurement_volume"].sum()
    )
    summary = create_regional_market_summary(regional)
    summary["bid_coverage_ratio"] = 1.0
    national.loc[0, "completeness_flag"] = "Incomplete"
    text = create_national_summary_text(national, summary, kpis)
    assert "불완전" in text
    assert "홋카이도" in text and "규슈" in text
    assert "전국 입찰량이 모집량보다 적었던 시간대" in text
