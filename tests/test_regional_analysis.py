import numpy as np
import pandas as pd
import pytest

from utils.regional_analysis import (
    calculate_area_kpis,
    calculate_previous_week_comparison,
    create_area_kpi_table,
    create_regional_summary,
    safe_ratio,
    validate_area_profile,
)
from utils.sample_data import generate_sample_data
from utils.weekly_aggregation import (
    add_week_columns,
    create_selected_area_weekly_profile,
)


@pytest.fixture(scope="module")
def sample_data():
    return add_week_columns(generate_sample_data("2026-07-20"))


@pytest.fixture(scope="module")
def selected_week(sample_data):
    return sample_data["week_start"].max()


@pytest.fixture(scope="module")
def area_profile(sample_data, selected_week):
    return create_selected_area_weekly_profile(
        sample_data, selected_week, ["Tokyo", "Chubu"]
    )


def test_tokyo_and_chubu_each_have_48_rows(area_profile):
    assert len(area_profile) == 96
    assert area_profile.groupby("area").size().to_dict() == {
        "Chubu": 48,
        "Tokyo": 48,
    }


def test_ratios_price_range_and_shortage_are_calculated(area_profile):
    row = area_profile.iloc[0]
    assert row["bid_coverage_ratio"] == pytest.approx(
        row["bid_volume"] / row["procurement_volume"]
    )
    assert row["procurement_rate"] == pytest.approx(
        row["awarded_volume"] / row["procurement_volume"]
    )
    assert row["price_range"] == pytest.approx(
        row["max_price"] - row["min_price"]
    )
    expected_shortage = max(
        row["procurement_volume"] - row["awarded_volume"], 0
    )
    assert row["shortage_volume"] == pytest.approx(expected_shortage)


def test_area_kpis_include_shortage_count(
    sample_data, selected_week, area_profile
):
    tokyo = area_profile[area_profile["area"].eq("Tokyo")]
    raw = sample_data[
        sample_data["week_start"].eq(selected_week)
        & sample_data["area"].eq("Tokyo")
    ]
    kpis = calculate_area_kpis(tokyo, raw)
    assert kpis["미조달 시간대 수"] == float(
        tokyo["shortage_volume"].gt(0).sum()
    )
    assert kpis["입찰경쟁률 (배)"] == pytest.approx(
        tokyo["bid_volume"].sum() / tokyo["procurement_volume"].sum()
    )
    assert kpis["조달률 (%)"] == pytest.approx(
        tokyo["awarded_volume"].sum() / tokyo["procurement_volume"].sum()
    )


def test_kpi_table_contains_difference(sample_data, selected_week, area_profile):
    raw = sample_data[sample_data["week_start"].eq(selected_week)]
    table = create_area_kpi_table(area_profile, raw)
    assert table.shape == (12, 3)
    assert table.loc["평균 모집량 (MW)", "중부−도쿄 차이"] == pytest.approx(
        table.loc["평균 모집량 (MW)", "중부"]
        - table.loc["평균 모집량 (MW)", "도쿄"]
    )


def test_previous_week_absolute_and_percent_change(
    sample_data, selected_week, area_profile
):
    comparison, metadata = calculate_previous_week_comparison(
        sample_data, selected_week, area_profile
    )
    assert metadata["previous_week"] == selected_week - pd.Timedelta(days=7)
    assert metadata["previous_complete"]
    row = comparison.iloc[0]
    assert row["절대 변화"] == pytest.approx(row["현재 주"] - row["전주"])
    assert row["변화율"] == pytest.approx(row["절대 변화"] / row["전주"])


def test_zero_previous_value_returns_nan():
    assert np.isnan(safe_ratio(3.0, 0.0))


def test_incomplete_week_warning(sample_data, selected_week):
    incomplete = sample_data.loc[
        ~(
            sample_data["week_start"].eq(selected_week)
            & sample_data["delivery_date"].eq(selected_week)
        )
    ]
    profile = create_selected_area_weekly_profile(
        incomplete, selected_week, ["Tokyo", "Chubu"]
    )
    warnings = validate_area_profile(profile, 6)
    assert any("6/7일" in warning for warning in warnings)
    assert any("7일 관측이 아닌" in warning for warning in warnings)


def test_summary_is_rule_based_and_handles_nan(
    sample_data, selected_week, area_profile
):
    raw = sample_data[sample_data["week_start"].eq(selected_week)]
    table = create_area_kpi_table(area_profile, raw)
    summary = create_regional_summary(area_profile, table)
    assert "입찰경쟁률" in summary
    assert "최대 미조달량" in summary
    modified = area_profile.copy()
    modified.loc[modified.index[0], "avg_price"] = np.nan
    nan_summary = create_regional_summary(modified, table)
    assert "결측된 데이터" in nan_summary
