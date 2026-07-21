"""지역 및 주파수 권역의 주간 30분 프로파일 집계 함수."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from utils.sample_data import AREA_TO_ZONE

VALUE_COLUMNS = [
    "max_price",
    "min_price",
    "avg_price",
    "awarded_volume",
    "bid_volume",
    "procurement_volume",
]


def add_week_columns(data: pd.DataFrame) -> pd.DataFrame:
    """delivery_date를 변환하고 월요일과 일요일 날짜 열을 추가합니다."""
    result = data.copy()
    result["delivery_date"] = pd.to_datetime(result["delivery_date"])
    result["week_start"] = (
        result["delivery_date"] - pd.to_timedelta(result["delivery_date"].dt.weekday, unit="D")
    ).dt.normalize()
    result["week_end"] = result["week_start"] + pd.Timedelta(days=6)
    return result


def create_regional_weekly_profile(
    data: pd.DataFrame, week_start: object
) -> pd.DataFrame:
    """선택 주의 지역별 동일 시간대 값을 평균하고 완전성을 표시합니다."""
    prepared = add_week_columns(data)
    selected_start = pd.Timestamp(week_start).normalize()
    selected = prepared.loc[prepared["week_start"] == selected_start].copy()
    columns = [
        "area",
        "frequency_zone",
        "period_no",
        "period_start",
        *VALUE_COLUMNS,
        "observation_count",
        "data_status",
    ]
    if selected.empty:
        return pd.DataFrame(columns=columns)
    missing_columns = sorted(set(VALUE_COLUMNS) - set(selected.columns))
    if missing_columns:
        raise ValueError(f"주간 집계 필수 열 누락: {', '.join(missing_columns)}")

    grouped = (
        selected.groupby(
            ["area", "frequency_zone", "period_no", "period_start"],
            as_index=False,
            observed=True,
        )
        .agg(
            **{column: (column, "mean") for column in VALUE_COLUMNS},
            observation_count=("delivery_date", "nunique"),
        )
        .sort_values(["area", "period_no"])
        .reset_index(drop=True)
    )
    grouped["data_status"] = np.where(
        grouped["observation_count"] == 7, "Complete", "Incomplete"
    )
    return grouped[columns]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """0인 분모를 NaN으로 유지하며 나눗셈합니다."""
    return numerator.div(denominator.where(denominator.ne(0)))


def create_zone_weekly_profile(regional_profile: pd.DataFrame) -> pd.DataFrame:
    """지역 프로파일을 50Hz와 60Hz 권역의 48개 시간대로 집계합니다."""
    columns = [
        "frequency_zone",
        "period_no",
        "period_start",
        "procurement_volume",
        "bid_volume",
        "awarded_volume",
        "avg_price",
        "max_price",
        "min_price",
        "bid_coverage_ratio",
        "procurement_rate",
        "award_rate",
        "excess_bid_volume",
        "shortage_volume",
    ]
    if regional_profile.empty:
        return pd.DataFrame(columns=columns)

    working = regional_profile.copy()
    working["_weighted_price"] = working["avg_price"] * working["awarded_volume"]
    grouped = (
        working.groupby(
            ["frequency_zone", "period_no", "period_start"],
            as_index=False,
            observed=True,
        )
        .agg(
            procurement_volume=("procurement_volume", "sum"),
            bid_volume=("bid_volume", "sum"),
            awarded_volume=("awarded_volume", "sum"),
            weighted_price_sum=("_weighted_price", "sum"),
            max_price=("max_price", "max"),
            min_price=("min_price", "min"),
        )
        .sort_values(["frequency_zone", "period_no"])
        .reset_index(drop=True)
    )
    zero_award = grouped["awarded_volume"].eq(0)
    if zero_award.any():
        warnings.warn(
            "총 낙찰량이 0인 권역 시간대의 평균가격을 NaN으로 유지합니다.",
            RuntimeWarning,
            stacklevel=2,
        )
    grouped["avg_price"] = _safe_divide(
        grouped["weighted_price_sum"], grouped["awarded_volume"]
    )
    grouped["bid_coverage_ratio"] = _safe_divide(
        grouped["bid_volume"], grouped["procurement_volume"]
    )
    grouped["procurement_rate"] = _safe_divide(
        grouped["awarded_volume"], grouped["procurement_volume"]
    )
    grouped["award_rate"] = _safe_divide(
        grouped["awarded_volume"], grouped["bid_volume"]
    )
    grouped["excess_bid_volume"] = (
        grouped["bid_volume"] - grouped["procurement_volume"]
    )
    grouped["shortage_volume"] = (
        grouped["procurement_volume"] - grouped["awarded_volume"]
    ).clip(lower=0)
    return grouped[columns]


def create_selected_area_weekly_profile(
    data: pd.DataFrame, week_start: object, areas: list[str]
) -> pd.DataFrame:
    """기존 지역 프로파일에서 선택 지역과 파생지표만 반환합니다."""
    regional = create_regional_weekly_profile(data, week_start)
    selected = regional.loc[regional["area"].isin(areas)].copy()
    if selected.empty:
        return selected.assign(
            completeness_flag=pd.Series(dtype="object"),
            bid_coverage_ratio=pd.Series(dtype="float64"),
            procurement_rate=pd.Series(dtype="float64"),
            award_rate=pd.Series(dtype="float64"),
            excess_bid_volume=pd.Series(dtype="float64"),
            shortage_volume=pd.Series(dtype="float64"),
            price_range=pd.Series(dtype="float64"),
        )
    selected["completeness_flag"] = np.where(
        selected["observation_count"].eq(7), "Complete", "Incomplete"
    )
    selected["bid_coverage_ratio"] = _safe_divide(
        selected["bid_volume"], selected["procurement_volume"]
    )
    selected["procurement_rate"] = _safe_divide(
        selected["awarded_volume"], selected["procurement_volume"]
    )
    selected["award_rate"] = _safe_divide(
        selected["awarded_volume"], selected["bid_volume"]
    )
    selected["excess_bid_volume"] = (
        selected["bid_volume"] - selected["procurement_volume"]
    )
    selected["shortage_volume"] = (
        selected["procurement_volume"] - selected["awarded_volume"]
    ).clip(lower=0)
    selected["price_range"] = selected["max_price"] - selected["min_price"]
    return selected.sort_values(["area", "period_no"]).reset_index(drop=True)


def find_missing_areas(regional_profile: pd.DataFrame) -> dict[str, list[str]]:
    """권역 집계에 필요한 지역 중 프로파일에 전혀 없는 지역을 반환합니다."""
    present = set(regional_profile["area"].dropna().unique())
    missing: dict[str, list[str]] = {}
    for area, zone in AREA_TO_ZONE.items():
        if area not in present:
            missing.setdefault(zone, []).append(area)
    return missing
