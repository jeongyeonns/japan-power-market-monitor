"""검증된 JEPX 일별 스프레드 결과를 이용한 주간 분석 함수."""

from __future__ import annotations

import numpy as np
import pandas as pd

DAILY_SPREAD_AREA_ORDER = (
    "Chubu", "Tokyo", "Hokkaido", "Tohoku", "Hokuriku",
    "Kansai", "Chugoku", "Shikoku", "Kyushu", "System",
)
DAILY_SPREAD_PREFERRED_DEFAULTS = ("Chubu", "Tokyo", "Hokkaido", "Kyushu")


def _week_start(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values).dt.normalize()
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def format_spread_change(value) -> str:
    """변화 방향을 부호 중복 없이 표시합니다."""
    if pd.isna(value):
        return "비교 불가"
    if value > 0:
        return f"▲ {value:.2f}"
    if value < 0:
        return f"▼ {abs(value):.2f}"
    return "0.00"


def sort_week_over_week_by_absolute_change(
    comparison: pd.DataFrame,
    change_column: str = "average_spread_change",
    area_column: str = "area",
) -> pd.DataFrame:
    """변화 절댓값↓, 변화량↓, 지역명↑ 순으로 표시 결과를 정렬합니다."""
    if comparison.empty:
        return comparison.copy()
    data = comparison.copy()
    data["_absolute_change"] = data[change_column].abs()
    data = data.sort_values(
        ["_absolute_change", change_column, area_column],
        ascending=[False, False, True],
        na_position="last",
        kind="mergesort",
    )
    return data.drop(columns="_absolute_change").reset_index(drop=True)


def order_daily_spread_areas(available_areas) -> list[str]:
    """실제 존재하는 지역을 사용자 우선순위로 정렬합니다."""
    available = list(dict.fromkeys(available_areas))
    preferred = [area for area in DAILY_SPREAD_AREA_ORDER if area in available]
    return preferred + sorted(set(available) - set(preferred))


def initial_daily_spread_areas(available_areas, selected_area=None) -> list[str]:
    """최초 로드의 기본 지역을 만들며 존재하지 않는 지역은 포함하지 않습니다."""
    ordered = order_daily_spread_areas(available_areas)
    if selected_area and selected_area != "전체 지역 비교" and selected_area in ordered:
        return [selected_area]
    return [area for area in DAILY_SPREAD_PREFERRED_DEFAULTS if area in ordered]


def reconcile_daily_spread_areas(selected_areas, available_areas) -> list[str]:
    """사용자 선택 순서를 유지하며 더 이상 존재하지 않는 지역만 제거합니다."""
    available = set(available_areas)
    return [area for area in selected_areas if area in available]


def resolve_daily_spread_areas(selected_areas, available_areas, selected_area=None) -> list[str]:
    """직접 빈 선택은 유지하고, 기존 선택이 전부 무효가 된 경우에만 초기화합니다."""
    selected = list(selected_areas)
    valid = reconcile_daily_spread_areas(selected, available_areas)
    if selected and not valid:
        return initial_daily_spread_areas(available_areas, selected_area)
    return valid


def filter_weekly_spreads(daily_spreads: pd.DataFrame, week_start) -> pd.DataFrame:
    """월요일~일요일에 속한 일별 결과를 반환합니다."""
    data = daily_spreads.copy()
    data["delivery_date"] = pd.to_datetime(data["delivery_date"])
    start = pd.Timestamp(week_start).normalize()
    return data[data["delivery_date"].between(start, start + pd.Timedelta(days=6))].copy()


def calculate_weekly_area_kpis(weekly_spreads: pd.DataFrame) -> pd.DataFrame:
    """완전하고 계산된 날짜만 사용해 지역별 주간 KPI를 계산합니다."""
    rows = []
    for area, group in weekly_spreads.groupby("area", dropna=False, sort=True):
        valid = group[
            group["completeness_flag"].eq("Complete")
            & group["calculation_status"].eq("Calculated")
            & group["spread"].notna()
        ]
        rows.append({
            "area": area,
            "average_spread": valid["spread"].mean(),
            "median_spread": valid["spread"].median(),
            "max_spread": valid["spread"].max(),
            "min_spread": valid["spread"].min(),
            "positive_spread_days": int(valid["spread"].gt(0).sum()),
            "complete_days": int(valid["delivery_date"].nunique()),
            "incomplete_days": int(group.loc[~group.index.isin(valid.index), "delivery_date"].nunique()),
            "average_charge_price": valid["charge_average_price"].mean(),
            "average_discharge_price": valid["discharge_average_price"].mean(),
        })
    return pd.DataFrame(rows)


def calculate_week_over_week(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """지역별 현재 주와 전주의 KPI 차이 및 변화율을 계산합니다."""
    keys = ["area"]
    metrics = ["average_spread", "median_spread", "max_spread", "average_charge_price",
               "average_discharge_price", "positive_spread_days"]
    if current.empty:
        return pd.DataFrame()
    if previous.empty:
        previous = pd.DataFrame(columns=keys + metrics)
    merged = current[keys + metrics].merge(
        previous[keys + metrics], on="area", how="left", suffixes=("_current", "_previous")
    )
    for metric in metrics:
        merged[f"{metric}_change"] = merged[f"{metric}_current"] - merged[f"{metric}_previous"]
        denominator = merged[f"{metric}_previous"].replace(0, np.nan)
        merged[f"{metric}_change_pct"] = merged[f"{metric}_change"] / denominator * 100
    merged["direction"] = np.select(
        [merged["average_spread_change"].gt(0), merged["average_spread_change"].lt(0)],
        ["상승", "하락"], default="변화 없음",
    )
    return merged


def calculate_charge_discharge_time_frequency(weekly_spreads: pd.DataFrame) -> pd.DataFrame:
    """완전한 일별 결과의 충전·방전 시작시각 선택 빈도를 반환합니다."""
    valid = weekly_spreads[
        weekly_spreads["completeness_flag"].eq("Complete")
        & weekly_spreads["calculation_status"].eq("Calculated")
    ]
    frames = []
    for column, kind in (("charge_start", "충전"), ("discharge_start", "방전")):
        counts = valid.groupby(["area", column], dropna=False).size().rename("selection_count").reset_index()
        counts = counts.rename(columns={column: "start_time"})
        counts["type"] = kind
        frames.append(counts)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def create_area_price_spread_comparison(
    normalized_price_data: pd.DataFrame,
    daily_spread_data: pd.DataFrame,
    selected_week,
    duration_hours: int,
    operation_mode: str,
) -> pd.DataFrame:
    """지역별 주간 시장가격과 기존 일별 ESS 스프레드 결과를 결합합니다."""
    start = pd.Timestamp(selected_week).normalize()
    end = start + pd.Timedelta(days=6)
    prices = normalized_price_data.copy()
    prices["delivery_date"] = pd.to_datetime(prices["delivery_date"]).dt.normalize()
    prices = prices[prices["delivery_date"].between(start, end)]
    price_summary = prices.groupby("area", as_index=False).agg(
        average_market_price=("price", "mean"),
        minimum_market_price=("price", "min"),
        maximum_market_price=("price", "max"),
    )

    spreads = daily_spread_data.copy()
    spreads["delivery_date"] = pd.to_datetime(spreads["delivery_date"]).dt.normalize()
    spreads = spreads[
        spreads["duration_hours"].eq(duration_hours)
        & spreads["operation_mode"].eq(operation_mode)
    ]
    current = spreads[spreads["delivery_date"].between(start, end)]
    previous = spreads[spreads["delivery_date"].between(start - pd.Timedelta(days=7), start - pd.Timedelta(days=1))]
    valid = current[
        current["completeness_flag"].eq("Complete")
        & current["calculation_status"].eq("Calculated")
        & current["spread"].notna()
    ]
    previous_valid = previous[
        previous["completeness_flag"].eq("Complete")
        & previous["calculation_status"].eq("Calculated")
        & previous["spread"].notna()
    ]
    spread_rows = []
    for area, group in current.groupby("area", dropna=False, sort=True):
        area_valid = valid[valid["area"].eq(area)]
        area_previous = previous_valid[previous_valid["area"].eq(area)]
        maximum = area_valid["spread"].max()
        peak_dates = area_valid.loc[area_valid["spread"].eq(maximum), "delivery_date"]
        average = area_valid["spread"].mean()
        previous_average = area_previous["spread"].mean()
        spread_rows.append({
            "area": area,
            "average_charge_price": area_valid["charge_average_price"].mean(),
            "average_discharge_price": area_valid["discharge_average_price"].mean(),
            "average_spread": average,
            "maximum_spread": maximum,
            "maximum_spread_date": peak_dates.min() if not peak_dates.empty else pd.NaT,
            "previous_week_average_spread": previous_average,
            "week_over_week_change": average - previous_average,
            "complete_days": int(area_valid["delivery_date"].nunique()),
            "excluded_days": int(current.loc[current["area"].eq(area), "delivery_date"].nunique()
                                 - area_valid["delivery_date"].nunique()),
        })
    result = price_summary.merge(pd.DataFrame(spread_rows), on="area", how="outer")
    return result.sort_values("average_spread", ascending=False, na_position="last").reset_index(drop=True)


def calculate_tokyo_chubu_weekly_comparison(
    normalized_price_data: pd.DataFrame,
    daily_spread_data: pd.DataFrame,
    selected_week,
    duration_hours: int,
    operation_mode: str,
) -> pd.DataFrame:
    """기존 주간 집계를 재사용해 도쿄·중부 핵심 비교표를 반환합니다."""
    areas = ["Tokyo", "Chubu"]
    comparison = create_area_price_spread_comparison(
        normalized_price_data,
        daily_spread_data,
        selected_week,
        duration_hours,
        operation_mode,
    )
    weekly = filter_weekly_spreads(daily_spread_data, selected_week)
    weekly = weekly[
        weekly["area"].isin(areas)
        & weekly["duration_hours"].eq(duration_hours)
        & weekly["operation_mode"].eq(operation_mode)
    ]
    kpis = calculate_weekly_area_kpis(weekly)
    comparison = comparison[comparison["area"].isin(areas)].copy()
    if kpis.empty:
        comparison["positive_spread_days"] = np.nan
    else:
        comparison = comparison.merge(
            kpis[["area", "positive_spread_days"]], on="area", how="left"
        )
    order = pd.Categorical(comparison["area"], categories=areas, ordered=True)
    return comparison.assign(_area_order=order).sort_values("_area_order").drop(
        columns="_area_order"
    ).reset_index(drop=True)


def create_tokyo_chubu_price_profile(
    normalized_price_data: pd.DataFrame, selected_week
) -> pd.DataFrame:
    """선택 주차 도쿄·중부의 48개 시간대 가격 통계를 반환합니다."""
    start = pd.Timestamp(selected_week).normalize()
    prices = normalized_price_data.copy()
    prices["delivery_date"] = pd.to_datetime(prices["delivery_date"]).dt.normalize()
    prices = prices[
        prices["delivery_date"].between(start, start + pd.Timedelta(days=6))
        & prices["area"].isin(["Tokyo", "Chubu"])
    ]
    return prices.groupby(
        ["area", "period_no", "period_start"], as_index=False
    ).agg(
        mean_price=("price", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        observation_days=("delivery_date", "nunique"),
    )


def calculate_tokyo_chubu_week_over_week(
    normalized_price_data: pd.DataFrame,
    daily_spread_data: pd.DataFrame,
    selected_week,
    duration_hours: int,
    operation_mode: str,
) -> pd.DataFrame:
    """도쿄·중부의 현재 주·전주 핵심 지표와 절대 변화를 반환합니다."""
    current = calculate_tokyo_chubu_weekly_comparison(
        normalized_price_data,
        daily_spread_data,
        selected_week,
        duration_hours,
        operation_mode,
    )
    previous_start = pd.Timestamp(selected_week) - pd.Timedelta(days=7)
    previous_daily = filter_weekly_spreads(daily_spread_data, previous_start)
    if previous_daily.empty:
        previous = pd.DataFrame(
            columns=[
                "area",
                "average_market_price",
                "average_charge_price",
                "average_discharge_price",
                "average_spread",
                "maximum_spread",
                "positive_spread_days",
            ]
        )
    else:
        previous = calculate_tokyo_chubu_weekly_comparison(
            normalized_price_data,
            daily_spread_data,
            previous_start,
            duration_hours,
            operation_mode,
        )
    metrics = {
        "평균 전력가격": "average_market_price",
        "평균 충전가격": "average_charge_price",
        "평균 방전가격": "average_discharge_price",
        "평균 ESS 스프레드": "average_spread",
        "최대 ESS 스프레드": "maximum_spread",
        "양의 스프레드 일수": "positive_spread_days",
    }
    merged = current.merge(previous, on="area", how="left", suffixes=("_current", "_previous"))
    rows = []
    for _, area_row in merged.iterrows():
        for metric_label, column in metrics.items():
            current_value = area_row.get(f"{column}_current", np.nan)
            previous_value = area_row.get(f"{column}_previous", np.nan)
            rows.append(
                {
                    "area": area_row["area"],
                    "metric": metric_label,
                    "current": current_value,
                    "previous": previous_value,
                    "change": current_value - previous_value,
                }
            )
    return pd.DataFrame(rows)


def compare_area_price_series(long_data: pd.DataFrame, week_start) -> pd.DataFrame:
    """선택 주차의 코마별 지역가격 동일 여부를 실제 데이터로 비교합니다."""
    start = pd.Timestamp(week_start).normalize()
    data = long_data[pd.to_datetime(long_data["delivery_date"]).between(start, start + pd.Timedelta(days=6))]
    pivot = data.pivot_table(index=["delivery_date", "period_no"], columns="area", values="price", aggfunc="first")
    rows = []
    for i, left in enumerate(pivot.columns):
        for right in pivot.columns[i + 1:]:
            pair = pivot[[left, right]].dropna()
            difference = (pair[left] - pair[right]).abs()
            rows.append({"area_1": left, "area_2": right, "observation_count": len(pair),
                         "identical_count": int(difference.eq(0).sum()),
                         "identical_ratio": difference.eq(0).mean() if len(pair) else np.nan,
                         "different_count": int(difference.ne(0).sum()),
                         "max_price_difference": difference.max() if len(pair) else np.nan})
    return pd.DataFrame(rows)


def create_jepx_weekly_summary(kpis: pd.DataFrame, wow: pd.DataFrame | None = None) -> list[str]:
    """수치에 근거한 최대 4개 규칙 기반 요약 문장을 만듭니다."""
    valid = kpis.dropna(subset=["average_spread"])
    if valid.empty:
        return ["계산 가능한 완전 데이터가 없습니다."]
    high = valid.loc[valid["average_spread"].idxmax()]
    low = valid.loc[valid["average_spread"].idxmin()]
    peak = valid.loc[valid["max_spread"].idxmax()]
    messages = [
        f"평균 스프레드 최고 지역: {high['area']} ({high['average_spread']:.2f})",
        f"평균 스프레드 최저 지역: {low['area']} ({low['average_spread']:.2f})",
        f"주간 최대 스프레드: {peak['area']} ({peak['max_spread']:.2f})",
    ]
    if wow is not None and not wow.empty:
        up = int(wow["average_spread_change"].gt(0).sum())
        down = int(wow["average_spread_change"].lt(0).sum())
        messages.append(f"전주 대비 평균 스프레드: {up}개 지역 상승, {down}개 지역 하락")
    return messages[:4]
