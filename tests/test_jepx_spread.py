import numpy as np
import pandas as pd
import pytest

from utils.jepx_spread import (
    aggregate_spreads_by_month,
    aggregate_spreads_by_week,
    calculate_all_daily_spreads,
    calculate_daily_spread,
    create_rolling_price_windows,
    save_spread_validation_results,
    validate_daily_price_profile,
)


def one_day(prices=None, area="Tokyo", date="2026-06-01"):
    prices = list(range(48)) if prices is None else prices
    return pd.DataFrame({
        "delivery_date": pd.Timestamp(date), "area": area,
        "period_no": range(1, len(prices) + 1), "price": prices,
    })


@pytest.mark.parametrize("duration,count", [(1, 47), (2, 45), (4, 41)])
def test_rolling_window_counts_and_average(duration, count):
    windows = create_rolling_price_windows(one_day(), duration)
    assert len(windows) == count
    assert windows.iloc[0]["average_price"] == pytest.approx((duration * 2 - 1) / 2)
    assert windows.iloc[-1]["window_end"] == "24:00"


def test_unconstrained_and_charge_before_discharge_constraints():
    prices = [100, 100, 1, 1] + [10] * 40 + [90, 90, 2, 2]
    unconstrained = calculate_daily_spread(one_day(prices), 1, "unconstrained")
    ordered = calculate_daily_spread(one_day(prices), 1, "charge_before_discharge")
    assert unconstrained["charge_periods"] == (3, 4)
    assert unconstrained["discharge_periods"] == (1, 2)
    assert max(ordered["charge_periods"]) < min(ordered["discharge_periods"])
    assert set(ordered["charge_periods"]).isdisjoint(ordered["discharge_periods"])


def test_overlapping_best_windows_are_excluded():
    prices = [10] * 48
    prices[0:2] = [0, 0]
    prices[2:4] = [100, 100]
    result = calculate_daily_spread(one_day(prices), 2, "unconstrained")
    assert set(result["charge_periods"]).isdisjoint(result["discharge_periods"])


def test_tie_break_is_deterministic_and_earliest():
    prices = [5] * 48
    prices[0:2] = prices[4:6] = [0, 0]
    prices[10:12] = prices[20:22] = [10, 10]
    first = calculate_daily_spread(one_day(prices), 1, "unconstrained")
    second = calculate_daily_spread(one_day(prices), 1, "unconstrained")
    assert first == second
    assert first["charge_periods"] == (1, 2)
    assert first["discharge_periods"] == (11, 12)


def test_nem_best_case_matches_cheapest_and_dearest_n():
    result = calculate_daily_spread(one_day(), 2, "nem_best_case")
    assert result["charge_average_price"] == pytest.approx(1.5)
    assert result["discharge_average_price"] == pytest.approx(45.5)
    assert result["spread"] == pytest.approx(44.0)
    assert result["window_type"] == "non_contiguous"


def test_incomplete_duplicate_missing_and_nonnumeric_are_not_calculated():
    variants = [
        one_day().iloc[:-1],
        pd.concat([one_day(), one_day().iloc[[0]]]),
        one_day().assign(price=lambda x: x["price"].astype(object)).assign(**{"price": lambda x: x["price"].mask(x.index == 3, "x")}),
        one_day().assign(price=lambda x: x["price"].mask(x.index == 3, np.nan)),
    ]
    for data in variants:
        result = calculate_daily_spread(data, 1, "unconstrained")
        assert result["completeness_flag"] == "Incomplete"
        assert result["calculation_status"] == "Not calculated"
        assert pd.isna(result["spread"])


def test_zero_and_negative_spread_are_preserved():
    zero = calculate_daily_spread(one_day([5] * 48), 1, "charge_before_discharge")
    decreasing = calculate_daily_spread(one_day(list(range(48, 0, -1))), 1, "charge_before_discharge")
    assert zero["spread"] == 0
    assert not zero["positive_spread"]
    assert decreasing["spread"] < 0
    assert not decreasing["positive_spread"]


def test_system_and_region_all_daily_results():
    data = pd.concat([one_day(area="System"), one_day(area="Tokyo")])
    result = calculate_all_daily_spreads(data, [1, 2, 4], "nem_best_case")
    assert len(result) == 6
    assert set(result["area"]) == {"System", "Tokyo"}
    assert result["calculation_status"].eq("Calculated").all()


def test_weekly_monthly_aggregation_excludes_incomplete():
    daily = pd.DataFrame([
        calculate_daily_spread(one_day(date="2026-06-01"), 1),
        calculate_daily_spread(one_day(date="2026-06-02", prices=[5] * 48), 1),
        calculate_daily_spread(one_day(date="2026-06-03").iloc[:-1], 1),
    ])
    weekly, monthly = aggregate_spreads_by_week(daily), aggregate_spreads_by_month(daily)
    for result in (weekly.iloc[0], monthly.iloc[0]):
        assert result["average_spread"] == pytest.approx(23.0)
        assert result["median_spread"] == pytest.approx(23.0)
        assert result["positive_spread_days"] == 1
        assert result["total_days"] == 3
        assert result["complete_days"] == 2
        assert result["incomplete_days"] == 1


def test_validation_and_explicit_utf8_sig_save(tmp_path):
    _, validation = validate_daily_price_profile(one_day([-1] + [1] * 47))
    assert validation["is_valid"]  # 음수가격은 삭제·오류 처리하지 않음
    daily = pd.DataFrame([calculate_daily_spread(one_day(), 1)])
    paths = save_spread_validation_results(daily, daily.head(1), tmp_path)
    assert all(path.read_bytes().startswith(b"\xef\xbb\xbf") for path in paths)
