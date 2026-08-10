import numpy as np
import pandas as pd
import pytest

from utils.regional_charts import area_award_rate_chart, area_max_price_chart
from utils.sample_data import generate_sample_data
from utils.weekly_aggregation import add_week_columns, create_selected_area_weekly_profile


def _profiles(start: str = "2026-07-20"):
    data = add_week_columns(generate_sample_data(start))
    current_week = data["week_start"].max()
    previous_week = current_week - pd.Timedelta(days=7)
    current = create_selected_area_weekly_profile(data, current_week, ["Tokyo"])
    previous = create_selected_area_weekly_profile(data, previous_week, ["Tokyo"])
    return data, current_week, previous_week, current, previous


def test_week_boundaries_and_profiles_are_independent():
    data, current_week, previous_week, current, previous = _profiles()
    assert current_week - previous_week == pd.Timedelta(days=7)
    assert current_week == pd.Timestamp("2026-07-13")
    assert previous_week == pd.Timestamp("2026-07-06")

    expected_current = data.loc[
        data["week_start"].eq(current_week)
        & data["area"].eq("Tokyo")
        & data["period_no"].eq(1),
        "max_price",
    ].mean()
    expected_previous = data.loc[
        data["week_start"].eq(previous_week)
        & data["area"].eq("Tokyo")
        & data["period_no"].eq(1),
        "max_price",
    ].mean()
    assert current.iloc[0]["max_price"] == pytest.approx(expected_current)
    assert previous.iloc[0]["max_price"] == pytest.approx(expected_previous)
    assert current.iloc[0]["award_rate"] == pytest.approx(
        current.iloc[0]["awarded_volume"] / current.iloc[0]["bid_volume"]
    )
    assert previous.iloc[0]["award_rate"] == pytest.approx(
        previous.iloc[0]["awarded_volume"] / previous.iloc[0]["bid_volume"]
    )


def test_comparison_traces_share_time_axis_and_keep_current_values():
    _, _, _, current, previous = _profiles()
    price = area_max_price_chart(current, ["Tokyo"], "엔/kW", previous)
    rate = area_award_rate_chart(current, ["Tokyo"], previous)

    for figure, column in ((price, "max_price"), (rate, "award_rate")):
        assert [trace.name for trace in figure.data] == ["현재 주", "직전 주"]
        assert list(figure.data[0].x) == list(figure.data[1].x)
        assert list(figure.data[0].y) == pytest.approx(current[column].tolist())
        assert list(figure.data[1].y) == pytest.approx(previous[column].tolist())
        assert figure.data[1].opacity == pytest.approx(0.35)
        assert figure.data[1].line.dash == "dash"
        assert figure.data[1].line.width < 2


def test_missing_previous_week_and_zero_denominator_remain_safe():
    _, _, _, current, _ = _profiles()
    empty = current.iloc[0:0].copy()
    assert len(area_max_price_chart(current, ["Tokyo"], "엔/kW", empty).data) == 1
    assert len(area_award_rate_chart(current, ["Tokyo"], empty).data) == 1

    zero_bid = current.copy()
    zero_bid.loc[zero_bid.index[0], "bid_volume"] = 0
    zero_bid.loc[zero_bid.index[0], "award_rate"] = np.nan
    figure = area_award_rate_chart(current, ["Tokyo"], zero_bid)
    assert np.isnan(figure.data[1].y[0])


@pytest.mark.parametrize(
    ("current", "previous"),
    [
        ("2026-08-03", "2026-07-27"),
        ("2026-01-05", "2025-12-29"),
    ],
)
def test_month_and_year_boundaries(current: str, previous: str):
    assert pd.Timestamp(current) - pd.Timedelta(days=7) == pd.Timestamp(previous)
