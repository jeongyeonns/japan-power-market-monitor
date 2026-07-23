"""도쿄·중부 지역 상세분석의 KPI, 전주 비교, 규칙형 요약."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.weekly_aggregation import (
    add_week_columns,
    create_selected_area_weekly_profile,
)

AREA_DISPLAY = {"Tokyo": "도쿄", "Chubu": "중부"}
AREA_KPI_ORDER = [
    "평균 모집량 (MW)",
    "평균 입찰량 (MW)",
    "평균 낙찰량 (MW)",
    "입찰경쟁률 (배)",
    "조달률 (%)",
    "입찰 대비 낙찰률 (%)",
    "평균 낙찰가격",
    "최고 낙찰가격",
    "최저 낙찰가격",
    "평균 가격범위",
    "미조달 시간대 수",
    "평균 미조달량 (MW)",
]


def safe_ratio(numerator: float, denominator: float) -> float:
    """분모가 0이거나 결측이면 NaN을 반환합니다."""
    if pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def calculate_area_kpis(
    area_profile: pd.DataFrame, raw_area_data: pd.DataFrame
) -> dict[str, float]:
    """한 지역의 48개 프로파일 및 선택 주 원본으로 KPI를 계산합니다."""
    if area_profile.empty:
        return {label: np.nan for label in AREA_KPI_ORDER}
    procurement_sum = area_profile["procurement_volume"].sum(min_count=1)
    bid_sum = area_profile["bid_volume"].sum(min_count=1)
    awarded_sum = area_profile["awarded_volume"].sum(min_count=1)
    raw_awarded = raw_area_data["awarded_volume"].sum(min_count=1)
    weighted_price = safe_ratio(
        (raw_area_data["avg_price"] * raw_area_data["awarded_volume"]).sum(
            min_count=1
        ),
        raw_awarded,
    )
    return {
        "평균 모집량 (MW)": area_profile["procurement_volume"].mean(),
        "평균 입찰량 (MW)": area_profile["bid_volume"].mean(),
        "평균 낙찰량 (MW)": area_profile["awarded_volume"].mean(),
        "입찰경쟁률 (배)": safe_ratio(bid_sum, procurement_sum),
        "조달률 (%)": safe_ratio(awarded_sum, procurement_sum),
        "입찰 대비 낙찰률 (%)": safe_ratio(awarded_sum, bid_sum),
        "평균 낙찰가격": weighted_price,
        "최고 낙찰가격": raw_area_data["max_price"].max(),
        "최저 낙찰가격": raw_area_data["min_price"].min(),
        "평균 가격범위": area_profile["price_range"].mean(),
        "미조달 시간대 수": float(area_profile["shortage_volume"].gt(0).sum()),
        "평균 미조달량 (MW)": area_profile["shortage_volume"].mean(),
    }


def create_area_kpi_table(
    area_profile: pd.DataFrame, raw_week_data: pd.DataFrame
) -> pd.DataFrame:
    """도쿄·중부 KPI와 중부−도쿄 차이를 표로 반환합니다."""
    values: dict[str, dict[str, float]] = {}
    for area in ("Tokyo", "Chubu"):
        values[AREA_DISPLAY[area]] = calculate_area_kpis(
            area_profile.loc[area_profile["area"].eq(area)],
            raw_week_data.loc[raw_week_data["area"].eq(area)],
        )
    table = pd.DataFrame(values).reindex(AREA_KPI_ORDER)
    table["중부−도쿄 차이"] = table["중부"] - table["도쿄"]
    return table


def validate_area_profile(
    area_profile: pd.DataFrame,
    selected_week_days: int,
    areas: tuple[str, ...] = ("Tokyo", "Chubu"),
) -> list[str]:
    """지역 존재·48개 시간대·7일 관측·필수 열을 검사합니다."""
    warnings: list[str] = []
    required = {
        "max_price",
        "min_price",
        "avg_price",
        "awarded_volume",
        "bid_volume",
        "procurement_volume",
    }
    missing_columns = sorted(required - set(area_profile.columns))
    if missing_columns:
        warnings.append("필수 가격·물량 열 누락: " + ", ".join(missing_columns))
    for area in areas:
        display = AREA_DISPLAY.get(area, area)
        selected = area_profile.loc[area_profile["area"].eq(area)]
        if selected.empty:
            warnings.append(f"{display} 데이터가 없습니다.")
        elif selected["period_no"].nunique() != 48:
            warnings.append(
                f"{display} 시간대가 {selected['period_no'].nunique()}/48개입니다."
            )
        if not selected.empty and selected["observation_count"].ne(7).any():
            count = int(selected["observation_count"].ne(7).sum())
            warnings.append(f"{display}에서 7일 관측이 아닌 시간대가 {count}개입니다.")
    if selected_week_days != 7:
        warnings.append(f"선택 주차가 {selected_week_days}/7일로 불완전합니다.")
    return warnings


def find_previous_week(
    data: pd.DataFrame, selected_week: object
) -> tuple[pd.Timestamp | None, int]:
    """선택 주 직전 데이터 보유 주차와 관측일 수를 반환합니다."""
    prepared = add_week_columns(data)
    selected = pd.Timestamp(selected_week).normalize()
    previous = sorted(
        prepared.loc[prepared["week_start"] < selected, "week_start"].unique(),
        reverse=True,
    )
    if not previous:
        return None, 0
    week = pd.Timestamp(previous[0]).normalize()
    days = int(
        prepared.loc[prepared["week_start"].eq(week), "delivery_date"].nunique()
    )
    return week, days


def calculate_previous_week_comparison(
    data: pd.DataFrame,
    selected_week: object,
    current_profile: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """도쿄·중부 현재 주와 직전 보유 주의 KPI 변화표를 반환합니다."""
    prepared = add_week_columns(data)
    selected = pd.Timestamp(selected_week).normalize()
    previous_week, previous_days = find_previous_week(prepared, selected)
    metadata = {
        "previous_week": previous_week,
        "previous_days": previous_days,
        "previous_complete": previous_days == 7,
    }
    if previous_week is None:
        return pd.DataFrame(), metadata

    current_raw = prepared.loc[prepared["week_start"].eq(selected)]
    previous_raw = prepared.loc[prepared["week_start"].eq(previous_week)]
    previous_profile = create_selected_area_weekly_profile(
        prepared, previous_week, ["Tokyo", "Chubu"]
    )
    metrics = [
        "최고 낙찰가격",
        "평균 모집량 (MW)",
        "평균 입찰량 (MW)",
        "평균 낙찰량 (MW)",
        "입찰경쟁률 (배)",
        "조달률 (%)",
        "미조달 시간대 수",
    ]
    rows: list[dict[str, Any]] = []
    for area in ("Tokyo", "Chubu"):
        current = calculate_area_kpis(
            current_profile.loc[current_profile["area"].eq(area)],
            current_raw.loc[current_raw["area"].eq(area)],
        )
        previous = calculate_area_kpis(
            previous_profile.loc[previous_profile["area"].eq(area)],
            previous_raw.loc[previous_raw["area"].eq(area)],
        )
        for metric in metrics:
            current_value = current[metric]
            previous_value = previous[metric]
            absolute = current_value - previous_value
            rows.append(
                {
                    "지역": AREA_DISPLAY[area],
                    "지표": metric,
                    "현재 주": current_value,
                    "전주": previous_value,
                    "절대 변화": absolute,
                    "변화율": safe_ratio(absolute, previous_value),
                }
            )
    return pd.DataFrame(rows), metadata


def create_regional_summary(
    area_profile: pd.DataFrame,
    kpi_table: pd.DataFrame,
    previous_comparison: pd.DataFrame | None = None,
) -> str:
    """계산 결과만 사용한 객관적인 한국어 규칙형 요약을 만듭니다."""
    if area_profile.empty:
        return "지역 상세 프로파일이 없어 요약을 생성할 수 없습니다."
    sentences: list[str] = []
    tokyo_price = kpi_table.loc["평균 낙찰가격", "도쿄"]
    chubu_price = kpi_table.loc["평균 낙찰가격", "중부"]
    if pd.notna(tokyo_price) and pd.notna(chubu_price):
        difference = abs(tokyo_price - chubu_price)
        if np.isclose(difference, 0):
            sentences.append("도쿄와 중부의 평균 낙찰가격이 같았습니다.")
        else:
            higher = "도쿄" if tokyo_price > chubu_price else "중부"
            sentences.append(
                f"{higher}의 평균 낙찰가격이 다른 지역보다 {difference:,.2f} 높았습니다."
            )
    tokyo_bid = kpi_table.loc["입찰경쟁률 (배)", "도쿄"]
    chubu_bid = kpi_table.loc["입찰경쟁률 (배)", "중부"]
    if pd.notna(tokyo_bid) and pd.notna(chubu_bid):
        if np.isclose(tokyo_bid, chubu_bid):
            sentences.append(
                f"입찰경쟁률은 도쿄와 중부가 모두 {tokyo_bid:.2f}배였습니다."
            )
        else:
            higher = "도쿄" if tokyo_bid > chubu_bid else "중부"
            sentences.append(
                f"입찰경쟁률은 {higher}가 더 높았으며 도쿄 {tokyo_bid:.2f}배, "
                f"중부 {chubu_bid:.2f}배였습니다."
            )
    counts = area_profile.groupby("area").agg(
        bid_below=("bid_coverage_ratio", lambda values: int((values < 1).sum())),
        procurement_below=("procurement_rate", lambda values: int((values < 1).sum())),
    )
    for area in ("Tokyo", "Chubu"):
        if area in counts.index:
            sentences.append(
                f"{AREA_DISPLAY[area]}는 입찰경쟁률 1.0배 미만이 "
                f"{counts.loc[area, 'bid_below']}개, 조달률 100% 미만이 "
                f"{counts.loc[area, 'procurement_below']}개 시간대였습니다."
            )
    valid_shortage = area_profile.dropna(subset=["shortage_volume"])
    if not valid_shortage.empty:
        maximum = valid_shortage.loc[valid_shortage["shortage_volume"].idxmax()]
        sentences.append(
            f"최대 미조달량은 {AREA_DISPLAY.get(maximum['area'], maximum['area'])} "
            f"{maximum['period_start']}의 {maximum['shortage_volume']:,.2f}MW였습니다."
        )
    if previous_comparison is not None and not previous_comparison.empty:
        price_rows = previous_comparison.loc[
            previous_comparison["지표"].eq("평균 낙찰가격")
        ]
        for _, row in price_rows.iterrows():
            if pd.isna(row["절대 변화"]):
                sentences.append(
                    f"{row['지역']} 평균 낙찰가격의 전주 변화는 계산할 수 없습니다."
                )
            else:
                direction = (
                    "상승"
                    if row["절대 변화"] > 0
                    else "하락"
                    if row["절대 변화"] < 0
                    else "변화 없음"
                )
                sentences.append(
                    f"{row['지역']} 평균 낙찰가격은 전주 대비 {direction}"
                    f"({row['절대 변화']:+,.2f})했습니다."
                )
    if area_profile["observation_count"].ne(7).any() or area_profile[
        ["avg_price", "procurement_volume", "bid_volume", "awarded_volume"]
    ].isna().any().any():
        sentences.append("일부 시간대에 불완전하거나 결측된 데이터가 있습니다.")
    return " ".join(sentences)
