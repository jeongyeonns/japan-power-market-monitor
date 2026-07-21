import numpy as np
import pandas as pd

from utils.jepx_loader import AREA_DISPLAY
from utils.jepx_weekly_analysis import (
    calculate_charge_discharge_time_frequency, calculate_week_over_week,
    calculate_weekly_area_kpis, compare_area_price_series,
    create_area_price_spread_comparison, create_jepx_weekly_summary,
    filter_weekly_spreads, format_spread_change, initial_daily_spread_areas,
    order_daily_spread_areas, reconcile_daily_spread_areas,
    resolve_daily_spread_areas,
    sort_week_over_week_by_absolute_change,
)
from utils.jepx_charts import create_daily_spread_chart


def daily(area="Tokyo", values=(2, 4, -1), complete=True):
    rows = []
    for index, spread in enumerate(values):
        rows.append({"delivery_date": pd.Timestamp("2026-07-13") + pd.Timedelta(days=index), "area": area,
            "duration_hours": 2, "operation_mode": "nem_best_case", "spread": spread,
            "charge_average_price": 10 + index, "discharge_average_price": 12 + index,
            "charge_start": "01:00", "discharge_start": "18:00",
            "completeness_flag": "Complete" if complete or index else "Incomplete",
            "calculation_status": "Calculated" if complete or index else "Not calculated"})
    return pd.DataFrame(rows)


def test_monday_to_sunday_filter():
    data = pd.concat([daily(), daily().assign(delivery_date=pd.Timestamp("2026-07-20"))])
    assert len(filter_weekly_spreads(data, "2026-07-13")) == 3


def test_area_kpis_and_positive_days():
    result = calculate_weekly_area_kpis(daily()).iloc[0]
    assert result.average_spread == 5 / 3
    assert result.median_spread == 2
    assert result.max_spread == 4 and result.min_spread == -1
    assert result.positive_spread_days == 2


def test_incomplete_day_excluded():
    result = calculate_weekly_area_kpis(daily(complete=False)).iloc[0]
    assert result.complete_days == 2 and result.incomplete_days == 1
    assert result.average_spread == 1.5


def test_week_over_week_absolute_and_percent_change():
    current = calculate_weekly_area_kpis(daily(values=(4, 4)))
    previous = calculate_weekly_area_kpis(daily(values=(2, 2)))
    result = calculate_week_over_week(current, previous).iloc[0]
    assert result.average_spread_change == 2
    assert result.average_spread_change_pct == 100


def test_previous_zero_makes_percent_nan():
    current = calculate_weekly_area_kpis(daily(values=(1,)))
    previous = calculate_weekly_area_kpis(daily(values=(0,)))
    assert np.isnan(calculate_week_over_week(current, previous).iloc[0].average_spread_change_pct)


def test_missing_previous_week_is_preserved_as_nan():
    current = calculate_weekly_area_kpis(daily(values=(1,)))
    result = calculate_week_over_week(current, pd.DataFrame()).iloc[0]
    assert np.isnan(result.average_spread_previous)


def test_frequency_preserves_ties():
    data = pd.concat([daily(), daily("Chubu")], ignore_index=True)
    result = calculate_charge_discharge_time_frequency(data)
    assert result["selection_count"].sum() == 12
    assert set(result["type"]) == {"충전", "방전"}


def test_price_comparison_identical_ratio_and_difference():
    rows = []
    for period, a, b in [(1, 10, 10), (2, 10, 12)]:
        rows.extend([{"delivery_date": "2026-07-13", "period_no": period, "area": "Tokyo", "price": a},
                     {"delivery_date": "2026-07-13", "period_no": period, "area": "Chubu", "price": b}])
    result = compare_area_price_series(pd.DataFrame(rows), "2026-07-13").iloc[0]
    assert result.identical_ratio == .5
    assert result.different_count == 1
    assert result.max_price_difference == 2


def test_rule_summary_maximum_four_items_and_nan_safe():
    kpis = calculate_weekly_area_kpis(pd.concat([daily(), daily("Chubu", (np.nan,))]))
    summary = create_jepx_weekly_summary(kpis)
    assert 1 <= len(summary) <= 4
    assert any("Tokyo" in item for item in summary)


def test_area_price_spread_comparison_values_and_sorting():
    spreads = pd.concat([daily("Tokyo", (2, 4, 4)), daily("System", (1, 1, 1))], ignore_index=True)
    prices = pd.DataFrame([
        {"delivery_date": "2026-07-13", "area": area, "price": price}
        for area, values in {"Tokyo": (10, 20, 30), "System": (5, 10, 15)}.items()
        for price in values
    ])
    result = create_area_price_spread_comparison(prices, spreads, "2026-07-13", 2, "nem_best_case")
    tokyo = result[result.area.eq("Tokyo")].iloc[0]
    assert result.iloc[0].area == "Tokyo"
    assert tokyo.average_market_price == 20
    assert tokyo.minimum_market_price == 10
    assert tokyo.maximum_market_price == 30
    assert tokyo.average_charge_price == 11
    assert tokyo.average_discharge_price == 13
    assert tokyo.average_spread == 10 / 3
    assert tokyo.maximum_spread == 4
    assert tokyo.maximum_spread_date == pd.Timestamp("2026-07-14")
    assert tokyo.complete_days == 3


def test_comparison_previous_week_and_incomplete_exclusion():
    current = daily("Tokyo", (8, 10), complete=False)
    previous = daily("Tokyo", (3, 3)).assign(delivery_date=lambda x: x.delivery_date - pd.Timedelta(days=7))
    prices = pd.DataFrame([{"delivery_date": "2026-07-13", "area": "Tokyo", "price": 10}])
    result = create_area_price_spread_comparison(prices, pd.concat([current, previous]), "2026-07-13", 2, "nem_best_case").iloc[0]
    assert result.average_spread == 10
    assert result.previous_week_average_spread == 3
    assert result.week_over_week_change == 7
    assert result.excluded_days == 1


def test_change_display_negative_zero_and_nan():
    assert format_spread_change(-0.8) == "▼ 0.80"
    assert format_spread_change(1.25) == "▲ 1.25"
    assert format_spread_change(0) == "0.00"
    assert format_spread_change(np.nan) == "비교 불가"


def test_system_and_korean_area_name_mapping():
    assert AREA_DISPLAY["System"] == "시스템가격"
    assert AREA_DISPLAY["Tokyo"] == "도쿄"
    assert AREA_DISPLAY["Kyushu"] == "규슈"


def test_daily_spread_area_order_and_default_include_chubu():
    available = ["System", "Kyushu", "Tokyo", "Chubu", "Hokkaido", "Kansai"]
    assert order_daily_spread_areas(available) == [
        "Chubu", "Tokyo", "Hokkaido", "Kansai", "Kyushu", "System"
    ]
    assert initial_daily_spread_areas(available) == ["Chubu", "Tokyo", "Hokkaido", "Kyushu"]


def test_daily_default_without_chubu_and_individual_mode():
    available = ["Tokyo", "Kyushu"]
    assert initial_daily_spread_areas(available) == ["Tokyo", "Kyushu"]
    assert initial_daily_spread_areas(available, "Kyushu") == ["Kyushu"]


def test_daily_selection_reconciliation_respects_removed_chubu_and_empty_selection():
    available = ["Chubu", "Tokyo", "Kyushu"]
    assert reconcile_daily_spread_areas(["Tokyo", "Missing"], available) == ["Tokyo"]
    assert reconcile_daily_spread_areas([], available) == []
    assert resolve_daily_spread_areas([], available) == []
    assert resolve_daily_spread_areas(["Missing"], available) == ["Chubu", "Tokyo", "Kyushu"]


def test_daily_chart_supports_more_than_four_areas_and_preserves_values():
    areas = ["Chubu", "Tokyo", "Hokkaido", "Kyushu", "System"]
    data = pd.DataFrame([
        {"delivery_date": "2026-07-13", "area": area, "spread": index + 0.5,
         "charge_average_price": 1, "discharge_average_price": 2,
         "charge_start": "01:00", "discharge_start": "18:00", "completeness_flag": "Complete"}
        for index, area in enumerate(areas)
    ])
    figure = create_daily_spread_chart(data, AREA_DISPLAY)
    assert len(figure.data) == 5
    assert {trace.name for trace in figure.data} == {
        "중부", "도쿄", "홋카이도", "규슈", "시스템가격"
    }
    assert sorted(float(trace.y[0]) for trace in figure.data) == [.5, 1.5, 2.5, 3.5, 4.5]


def test_week_over_week_sort_uses_absolute_change_then_change_then_area():
    data = pd.DataFrame({
        "area": ["Zeta", "Beta", "Alpha", "Gamma", "Missing"],
        "average_spread_change": [-8.5, -7.0, 7.0, 9.23, np.nan],
        "average_spread_change_pct": [999, 5000, -9999, 1, 999999],
    })
    result = sort_week_over_week_by_absolute_change(data)
    assert result["area"].tolist() == ["Gamma", "Zeta", "Alpha", "Beta", "Missing"]
    assert result["average_spread_change"].tolist()[:4] == [9.23, -8.5, 7.0, -7.0]
    assert np.isnan(result.iloc[-1]["average_spread_change"])
    assert result.loc[result.area.eq("Beta"), "average_spread_change_pct"].iloc[0] == 5000


def test_week_over_week_sort_same_change_uses_area_name():
    data = pd.DataFrame({
        "area": ["Tokyo", "Chubu", "Kyushu"],
        "average_spread_change": [6.0, 6.0, -6.0],
    })
    assert sort_week_over_week_by_absolute_change(data)["area"].tolist() == ["Chubu", "Tokyo", "Kyushu"]
