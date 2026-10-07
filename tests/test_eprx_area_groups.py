import numpy as np
import pandas as pd
import pytest

from utils.eprx_area_groups import AGGREGATE_REGION_GROUPS, aggregate_region_data
from utils.regional_analysis import calculate_area_kpis, calculate_previous_week_comparison
from utils.regional_charts import area_max_price_chart, area_award_rate_chart
from utils.weekly_aggregation import add_week_columns, create_selected_area_weekly_profile


GROUP = "도쿄+도호쿠"


def rows():
    return pd.DataFrame([
        dict(delivery_date="2026-07-20", period_no=1, period_start="00:00",
             product="Primary", area="Tokyo", procurement_volume=300.,
             bid_volume=200., awarded_volume=100., max_price=12., min_price=3., avg_price=8.),
        dict(delivery_date="2026-07-20", period_no=1, period_start="00:00",
             product="Primary", area="Tohoku", procurement_volume=200.,
             bid_volume=100., awarded_volume=100., max_price=15., min_price=2., avg_price=10.),
    ])


@pytest.mark.parametrize("name,expected", [
    ("전국 합산", {"Hokkaido", "Tohoku", "Tokyo", "Chubu", "Hokuriku", "Kansai", "Chugoku", "Shikoku", "Kyushu"}),
    (GROUP, {"Tokyo", "Tohoku"}),
    ("중부+호쿠리쿠+간사이", {"Chubu", "Hokuriku", "Kansai"}),
])
def test_exact_constituents(name, expected):
    assert set(AGGREGATE_REGION_GROUPS[name]) == expected
    assert len(AGGREGATE_REGION_GROUPS[name]) == len(expected)


def test_sums_ratios_and_price_not_mean_of_ratios():
    original = rows()
    data, diagnostics = aggregate_region_data(original, GROUP)
    row = data.iloc[0]
    assert (row.procurement_volume, row.bid_volume, row.awarded_volume) == (500, 300, 200)
    assert row.award_rate == pytest.approx(2 / 3)
    assert row.award_rate != .75
    profile = create_selected_area_weekly_profile(data, "2026-07-20", [GROUP])
    kpi_rate = calculate_area_kpis(profile, data)["입찰 대비 낙찰률 (%)"]
    assert kpi_rate == pytest.approx(200 / 300)
    assert kpi_rate != .75
    assert row.procurement_rate == .4
    assert (row.max_price, row.min_price, row.avg_price) == (15, 2, 9)
    assert diagnostics.iloc[0].missing_regions == ()
    pd.testing.assert_frame_equal(original, rows())


def test_unequal_award_weights_and_zero_denominators():
    source = rows()
    source.loc[1, "awarded_volume"] = 300
    data, _ = aggregate_region_data(source, GROUP)
    assert data.iloc[0].avg_price == 9.5
    source[["bid_volume", "awarded_volume", "procurement_volume"]] = 0
    data, _ = aggregate_region_data(source, GROUP)
    assert data[["avg_price", "award_rate", "procurement_rate", "bid_coverage_ratio"]].isna().all().all()


def test_dates_periods_products_and_regimes_are_separate():
    source = pd.concat([
        rows(), rows().assign(delivery_date="2026-07-21"),
        rows().assign(period_no=2, period_start="00:30"),
        rows().assign(product="Other"),
    ], ignore_index=True)
    source["market_regime"] = "modern_30minute"
    source = pd.concat([source, rows().assign(market_regime="separate")], ignore_index=True)
    data, _ = aggregate_region_data(source, GROUP)
    assert len(data) == 5
    assert data.procurement_volume.eq(500).all()


@pytest.mark.parametrize("kind", ["missing", "all_missing", "duplicate", "null_volume"])
def test_incomplete_slots_are_not_partial_totals(kind):
    source = rows()
    if kind == "missing":
        source = source.iloc[:1]
    elif kind == "all_missing":
        source = source.assign(area="Chubu")
    elif kind == "duplicate":
        source = pd.concat([source, source.iloc[:1]], ignore_index=True)
    else:
        source.loc[0, "bid_volume"] = np.nan
    data, diagnostics = aggregate_region_data(source, GROUP)
    assert not data._aggregate_complete.any()
    assert data[["procurement_volume", "bid_volume", "awarded_volume", "max_price"]].isna().all().all()
    assert diagnostics.iloc[0].expected_regions == ("Tokyo", "Tohoku")
    if kind == "missing":
        assert diagnostics.iloc[0].missing_regions == ("Tohoku",)
    if kind == "all_missing":
        assert diagnostics.iloc[0].available_regions == ()


def test_missing_one_day_masks_weekly_slot_and_kpis():
    source = pd.concat([rows(), rows().iloc[:1].assign(delivery_date="2026-07-21")])
    data, _ = aggregate_region_data(source, GROUP)
    profile = create_selected_area_weekly_profile(data, "2026-07-20", [GROUP])
    assert profile.procurement_volume.isna().all()
    assert profile.observation_count.tolist() == [1]
    assert all(pd.isna(v) for v in calculate_area_kpis(profile, data).values())


def test_regional_max_then_weekly_mean_and_raw_weekly_ratio():
    day2 = rows().assign(delivery_date="2026-07-21")
    day2["max_price"] = [20, 11]
    day2["bid_volume"] = [400, 200]
    source = pd.concat([rows(), day2, rows().assign(period_no=2, period_start="00:30")])
    data, _ = aggregate_region_data(source, GROUP)
    profile = create_selected_area_weekly_profile(data, "2026-07-20", [GROUP])
    assert profile.iloc[0].max_price == 17.5
    kpis = calculate_area_kpis(profile, data)
    assert kpis["입찰 대비 낙찰률 (%)"] == pytest.approx(600 / 1200)
    assert kpis["최고 낙찰가격"] == 20
    previous = data.assign(delivery_date=data.delivery_date - pd.Timedelta(days=7))
    combined = add_week_columns(pd.concat([data, previous], ignore_index=True))
    comparison, meta = calculate_previous_week_comparison(combined, "2026-07-20", profile, (GROUP,))
    assert comparison["절대 변화"].eq(0).all()
    assert meta["previous_week"] == pd.Timestamp("2026-07-13")
    for figure in (area_max_price_chart(profile, [GROUP], "엔", profile), area_award_rate_chart(profile, [GROUP], profile)):
        assert len(figure.data) == 2
        assert GROUP in figure.layout.title.text


def test_unknown_price_with_positive_award_is_not_zero():
    source = rows()
    source.loc[1, "avg_price"] = np.nan
    data, _ = aggregate_region_data(source, GROUP)
    assert pd.isna(data.iloc[0].avg_price)
    profile = create_selected_area_weekly_profile(data, "2026-07-20", [GROUP])
    assert pd.isna(calculate_area_kpis(profile, data)["평균 낙찰가격"])


def test_empty_input_is_safe():
    data, diagnostics = aggregate_region_data(rows().iloc[:0], GROUP)
    assert data.empty and diagnostics.empty
