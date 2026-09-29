"""이미 로드된 EPRX 원본의 날짜·상품·제도·코마별 지역 합산."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.eprx_areas import EPRX_ANALYSIS_AREAS
from utils.eprx_periods import market_regime
from utils.weekly_aggregation import VALUE_COLUMNS, add_week_columns


AGGREGATE_REGION_GROUPS = {
    "전국 합산": EPRX_ANALYSIS_AREAS,
    "도쿄+도호쿠": ("Tokyo", "Tohoku"),
    "중부+호쿠리쿠+간사이": ("Chubu", "Hokuriku", "Kansai"),
}

AGGREGATE_REGION_CAPTIONS = {
    "전국 합산": "홋카이도·도호쿠·도쿄·중부·호쿠리쿠·간사이·주고쿠·시코쿠·규슈의 물량을 합산한 전국 참고지표입니다.",
    "도쿄+도호쿠": "도쿄와 도호쿠의 1차 조정력 데이터를 합산하여 광역조달권역의 모집·입찰·낙찰 상황을 확인합니다.",
    "중부+호쿠리쿠+간사이": "중부·호쿠리쿠·간사이 3개 지역의 모집·입찰·낙찰 물량을 합산하여 공급여건을 비교합니다.",
}


def aggregate_region_data(
    data: pd.DataFrame, group_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """불완전한 코마는 NaN으로 남기고 코마별 구성 지역 진단을 반환합니다.

    전체 입력의 코마를 기준으로 정렬하므로 구성 지역 전체가 빠진 코마도
    탐지합니다. 중복 지역 행 역시 임의로 더하지 않고 불완전으로 처리합니다.
    입력은 loader가 반환한 단일 상품(1차 조정력) 데이터입니다.
    """
    expected = AGGREGATE_REGION_GROUPS[group_name]
    prepared = data.copy()
    prepared["delivery_date"] = pd.to_datetime(prepared["delivery_date"])
    if "market_regime" not in prepared:
        prepared["market_regime"] = prepared["delivery_date"].map(market_regime)
    keys = ["delivery_date", "market_regime", "period_no", "period_start"]
    keys += [column for column in ("product", "price_unit", "volume_unit") if column in prepared]
    slots = prepared[keys].drop_duplicates()
    selected = prepared.loc[prepared["area"].isin(expected)].copy()
    selected["_weighted_price"] = selected["avg_price"] * selected["awarded_volume"]
    # A positive award with an unknown price must not silently become a zero price.
    selected["_unknown_price"] = selected["avg_price"].isna() & selected["awarded_volume"].gt(0)
    selected["_missing_volume"] = selected[["procurement_volume", "bid_volume", "awarded_volume"]].isna().any(axis=1)
    grouped = selected.groupby(keys, dropna=False, observed=True)
    values = grouped.agg(
        procurement_volume=("procurement_volume", lambda s: s.sum(min_count=1)),
        bid_volume=("bid_volume", lambda s: s.sum(min_count=1)),
        awarded_volume=("awarded_volume", lambda s: s.sum(min_count=1)),
        max_price=("max_price", "max"),
        min_price=("min_price", "min"),
        weighted_price=("_weighted_price", lambda s: s.sum(min_count=1)),
        unknown_price=("_unknown_price", "any"),
        missing_volume=("_missing_volume", "any"),
        available_regions=("area", lambda s: tuple(dict.fromkeys(s))),
        row_count=("area", "size"),
    ).reset_index()
    result = slots.merge(values, on=keys, how="left")
    result["available_regions"] = result["available_regions"].map(
        lambda value: value if isinstance(value, tuple) else ()
    )
    result["expected_regions"] = [expected] * len(result)
    result["missing_regions"] = result["available_regions"].map(
        lambda available: tuple(area for area in expected if area not in available)
    )
    result["_aggregate_complete"] = (
        result["missing_regions"].map(len).eq(0)
        & result["row_count"].eq(len(expected))
        & result["missing_volume"].eq(False)
    )
    result["avg_price"] = result["weighted_price"].div(
        result["awarded_volume"].where(result["awarded_volume"].ne(0))
    ).mask(result["unknown_price"].eq(True))
    result.loc[~result["_aggregate_complete"], VALUE_COLUMNS] = np.nan
    result["area"] = group_name
    result["frequency_zone"] = "Aggregate"
    for name, numerator, denominator in (
        ("award_rate", "awarded_volume", "bid_volume"),
        ("procurement_rate", "awarded_volume", "procurement_volume"),
        ("bid_coverage_ratio", "bid_volume", "procurement_volume"),
    ):
        result[name] = result[numerator].div(result[denominator].where(result[denominator].ne(0)))
    diagnostics = result[keys + ["expected_regions", "available_regions", "missing_regions", "row_count", "_aggregate_complete"]].copy()
    result = result.drop(columns=["weighted_price", "unknown_price", "missing_volume", "row_count", "expected_regions", "available_regions", "missing_regions"])
    return add_week_columns(result), diagnostics
