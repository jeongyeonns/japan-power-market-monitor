"""일본 9개 지역을 합산한 전국 시장 규모 참고지표."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.regional_analysis import AREA_DISPLAY, safe_ratio
from utils.sample_data import AREAS_BY_ZONE

EXPECTED_AREAS = [area for areas in AREAS_BY_ZONE.values() for area in areas]
ALL_AREA_DISPLAY = {
    "Hokkaido": "홋카이도",
    "Tohoku": "도호쿠",
    "Tokyo": "도쿄",
    "Chubu": "중부",
    "Hokuriku": "호쿠리쿠",
    "Kansai": "간사이",
    "Chugoku": "주고쿠",
    "Shikoku": "시코쿠",
    "Kyushu": "규슈",
}
NATIONAL_KPI_LABELS = [
    "평균 모집량 (MW)",
    "평균 입찰량 (MW)",
    "평균 낙찰량 (MW)",
    "주간 모집량 합계",
    "주간 입찰량 합계",
    "주간 낙찰량 합계",
    "전국 입찰경쟁률 (배)",
    "전국 조달률 (%)",
    "전국 입찰 대비 낙찰률 (%)",
    "전국 낙찰량 가중평균 낙찰가격",
    "전국 최고 낙찰가격",
    "전국 최저 낙찰가격",
    "평균 가격범위",
    "입찰경쟁률 1.0배 미만 시간대 수",
    "조달률 100% 미만 시간대 수",
    "미조달 발생 시간대 수",
    "평균 미조달량",
    "최대 미조달량",
    "최대 미조달 발생 시간대",
]


def _sum_min_count(series: pd.Series) -> float:
    return series.sum(min_count=1)


def create_national_weekly_profile(
    regional_profile: pd.DataFrame,
) -> pd.DataFrame:
    """지역 9개 프로파일을 전국 48개 시간대 참고 프로파일로 합산합니다."""
    columns = [
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
        "price_range",
        "area_count",
        "expected_area_count",
        "missing_areas",
        "completeness_flag",
    ]
    if regional_profile.empty:
        return pd.DataFrame(columns=columns)
    working = regional_profile.copy()
    working["_weighted_price"] = working["avg_price"] * working["awarded_volume"]
    rows: list[dict[str, Any]] = []
    for (period_no, period_start), group in working.groupby(
        ["period_no", "period_start"], sort=True
    ):
        present = set(group["area"].dropna())
        missing = [area for area in EXPECTED_AREAS if area not in present]
        procurement = _sum_min_count(group["procurement_volume"])
        bid = _sum_min_count(group["bid_volume"])
        awarded = _sum_min_count(group["awarded_volume"])
        complete = (
            not missing
            and len(present) == 9
            and group["observation_count"].eq(7).all()
        )
        rows.append(
            {
                "period_no": period_no,
                "period_start": period_start,
                "procurement_volume": procurement,
                "bid_volume": bid,
                "awarded_volume": awarded,
                "avg_price": safe_ratio(
                    _sum_min_count(group["_weighted_price"]), awarded
                ),
                "max_price": group["max_price"].max(),
                "min_price": group["min_price"].min(),
                "bid_coverage_ratio": safe_ratio(bid, procurement),
                "procurement_rate": safe_ratio(awarded, procurement),
                "award_rate": safe_ratio(awarded, bid),
                "excess_bid_volume": bid - procurement,
                "shortage_volume": max(procurement - awarded, 0),
                "area_count": len(present),
                "expected_area_count": 9,
                "missing_areas": ", ".join(missing),
                "completeness_flag": "Complete" if complete else "Incomplete",
            }
        )
    result = pd.DataFrame(rows)
    result["price_range"] = result["max_price"] - result["min_price"]
    return result.reindex(columns=columns).sort_values("period_no").reset_index(drop=True)


def calculate_national_kpis(
    national_profile: pd.DataFrame, regional_profile: pd.DataFrame
) -> dict[str, Any]:
    """전국 시장 규모·경쟁·가격·수급 KPI를 계산합니다."""
    if national_profile.empty:
        return {label: np.nan for label in NATIONAL_KPI_LABELS}
    procurement = national_profile["procurement_volume"].sum(min_count=1)
    bid = national_profile["bid_volume"].sum(min_count=1)
    awarded = national_profile["awarded_volume"].sum(min_count=1)
    weighted_price = safe_ratio(
        (
            regional_profile["avg_price"] * regional_profile["awarded_volume"]
        ).sum(min_count=1),
        regional_profile["awarded_volume"].sum(min_count=1),
    )
    max_shortage_index = national_profile["shortage_volume"].idxmax()
    return {
        "평균 모집량 (MW)": national_profile["procurement_volume"].mean(),
        "평균 입찰량 (MW)": national_profile["bid_volume"].mean(),
        "평균 낙찰량 (MW)": national_profile["awarded_volume"].mean(),
        "주간 모집량 합계": procurement,
        "주간 입찰량 합계": bid,
        "주간 낙찰량 합계": awarded,
        "전국 입찰경쟁률 (배)": safe_ratio(bid, procurement),
        "전국 조달률 (%)": safe_ratio(awarded, procurement),
        "전국 입찰 대비 낙찰률 (%)": safe_ratio(awarded, bid),
        "전국 낙찰량 가중평균 낙찰가격": weighted_price,
        "전국 최고 낙찰가격": regional_profile["max_price"].max(),
        "전국 최저 낙찰가격": regional_profile["min_price"].min(),
        "평균 가격범위": national_profile["price_range"].mean(),
        "입찰경쟁률 1.0배 미만 시간대 수": int(
            national_profile["bid_coverage_ratio"].lt(1).sum()
        ),
        "조달률 100% 미만 시간대 수": int(
            national_profile["procurement_rate"].lt(1).sum()
        ),
        "미조달 발생 시간대 수": int(
            national_profile["shortage_volume"].gt(0).sum()
        ),
        "평균 미조달량": national_profile["shortage_volume"].mean(),
        "최대 미조달량": national_profile["shortage_volume"].max(),
        "최대 미조달 발생 시간대": national_profile.loc[
            max_shortage_index, "period_start"
        ],
    }


def create_regional_market_summary(
    regional_profile: pd.DataFrame,
) -> pd.DataFrame:
    """9개 지역의 규모·경쟁·가격 요약표를 만듭니다."""
    rows: list[dict[str, Any]] = []
    for area in EXPECTED_AREAS:
        group = regional_profile.loc[regional_profile["area"].eq(area)]
        if group.empty:
            continue
        procurement = group["procurement_volume"].sum(min_count=1)
        bid = group["bid_volume"].sum(min_count=1)
        awarded = group["awarded_volume"].sum(min_count=1)
        weighted = safe_ratio(
            (group["avg_price"] * group["awarded_volume"]).sum(min_count=1),
            awarded,
        )
        rows.append(
            {
                "area": area,
                "area_display": ALL_AREA_DISPLAY[area],
                "frequency_zone": group["frequency_zone"].iloc[0],
                "avg_procurement_volume": group["procurement_volume"].mean(),
                "avg_bid_volume": group["bid_volume"].mean(),
                "avg_awarded_volume": group["awarded_volume"].mean(),
                "bid_coverage_ratio": safe_ratio(bid, procurement),
                "procurement_rate": safe_ratio(awarded, procurement),
                "award_rate": safe_ratio(awarded, bid),
                "weighted_avg_price": weighted,
                "max_price": group["max_price"].max(),
                "min_price": group["min_price"].min(),
                "shortage_period_count": int(
                    (group["procurement_volume"] - group["awarded_volume"])
                    .clip(lower=0)
                    .gt(0)
                    .sum()
                ),
                "max_shortage_volume": (
                    group["procurement_volume"] - group["awarded_volume"]
                )
                .clip(lower=0)
                .max(),
                "observed_period_count": group["period_no"].nunique(),
                "completeness_flag": (
                    "Complete"
                    if group["period_no"].nunique() == 48
                    and group["observation_count"].eq(7).all()
                    else "Incomplete"
                ),
            }
        )
    return pd.DataFrame(rows)


def calculate_national_week_over_week(
    current_profile: pd.DataFrame,
    previous_profile: pd.DataFrame,
    current_regional_profile: pd.DataFrame,
    previous_regional_profile: pd.DataFrame,
) -> pd.DataFrame:
    """전국 주요 KPI의 현재·전주·절대변화·변화율을 계산합니다."""
    current = calculate_national_kpis(current_profile, current_regional_profile)
    previous = calculate_national_kpis(previous_profile, previous_regional_profile)
    metrics = [
        "평균 모집량 (MW)",
        "평균 입찰량 (MW)",
        "평균 낙찰량 (MW)",
        "전국 입찰경쟁률 (배)",
        "전국 조달률 (%)",
        "전국 낙찰량 가중평균 낙찰가격",
        "입찰경쟁률 1.0배 미만 시간대 수",
        "미조달 발생 시간대 수",
        "최대 미조달량",
    ]
    rows = []
    for metric in metrics:
        absolute = current[metric] - previous[metric]
        rows.append(
            {
                "지표": metric,
                "현재 주": current[metric],
                "전주": previous[metric],
                "절대 변화": absolute,
                "변화율": safe_ratio(absolute, previous[metric]),
                "절대 변화 단위": (
                    "%p"
                    if metric == "전국 조달률 (%)"
                    else "배"
                    if metric == "전국 입찰경쟁률 (배)"
                    else "값"
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_national_profile(
    national_profile: pd.DataFrame,
    regional_profile: pd.DataFrame,
    selected_week_days: int,
) -> list[str]:
    """전국 집계의 지역·시간대·관측·가중평균 완전성을 검사합니다."""
    warnings: list[str] = []
    present = set(regional_profile["area"].dropna())
    missing = [area for area in EXPECTED_AREAS if area not in present]
    if missing:
        warnings.append(
            "누락 지역: " + ", ".join(ALL_AREA_DISPLAY[area] for area in missing)
        )
    incomplete_areas = []
    for area in EXPECTED_AREAS:
        group = regional_profile.loc[regional_profile["area"].eq(area)]
        if not group.empty and (
            group["period_no"].nunique() != 48
            or group["observation_count"].ne(7).any()
        ):
            incomplete_areas.append(ALL_AREA_DISPLAY[area])
    if incomplete_areas:
        warnings.append("불완전한 지역: " + ", ".join(incomplete_areas))
    if len(national_profile) != 48:
        warnings.append(f"전국 시간대가 {len(national_profile)}/48개입니다.")
    incomplete_periods = national_profile["completeness_flag"].ne("Complete").sum()
    if incomplete_periods:
        warnings.append(f"포함 지역이 불완전한 전국 시간대가 {incomplete_periods}개입니다.")
    invalid_prices = national_profile["avg_price"].isna().sum()
    if invalid_prices:
        warnings.append(f"가중평균가격 계산 불가 시간대가 {invalid_prices}개입니다.")
    if selected_week_days != 7:
        warnings.append(f"선택 주차가 {selected_week_days}/7일로 불완전합니다.")
    if warnings:
        warnings.append("표시 결과는 완전한 전국 집계를 의미하지 않을 수 있습니다.")
    return warnings


def _extreme_names(
    summary: pd.DataFrame, column: str, mode: str
) -> str:
    valid = summary.dropna(subset=[column])
    if valid.empty:
        return "계산 불가"
    target = valid[column].min() if mode == "min" else valid[column].max()
    names = valid.loc[np.isclose(valid[column], target), "area_display"].tolist()
    return ", ".join(names)


def create_national_summary_text(
    national_profile: pd.DataFrame,
    regional_summary: pd.DataFrame,
    current_kpis: dict[str, Any],
    week_over_week: pd.DataFrame | None = None,
) -> str:
    """전국 시장 규모를 객관적으로 설명하는 규칙 기반 문장을 생성합니다."""
    sentences: list[str] = []
    if national_profile["completeness_flag"].ne("Complete").any():
        sentences.append("선택 주차의 전국 집계에는 불완전한 지역 또는 시간대가 있습니다.")
    if week_over_week is not None and not week_over_week.empty:
        for metric, noun in (
            ("평균 모집량 (MW)", "전국 평균 모집량"),
            (
                "전국 낙찰량 가중평균 낙찰가격",
                "전국 낙찰량 가중평균 낙찰가격 참고지표",
            ),
        ):
            row = week_over_week.loc[week_over_week["지표"].eq(metric)].iloc[0]
            if pd.notna(row["절대 변화"]):
                direction = "증가" if row["절대 변화"] > 0 else "감소" if row["절대 변화"] < 0 else "변화 없음"
                sentences.append(
                    f"{noun}은 전주 대비 {direction}했습니다"
                    f"({row['절대 변화']:+,.2f})."
                )
    sentences.append(
        "9개 지역 중 입찰경쟁률이 가장 낮은 지역은 "
        f"{_extreme_names(regional_summary, 'bid_coverage_ratio', 'min')}이고, "
        f"가장 높은 지역은 {_extreme_names(regional_summary, 'bid_coverage_ratio', 'max')}입니다."
    )
    sentences.append(
        "낙찰량 가중평균 낙찰가격 참고지표가 가장 높은 지역은 "
        f"{_extreme_names(regional_summary, 'weighted_avg_price', 'max')}이고, "
        f"가장 낮은 지역은 {_extreme_names(regional_summary, 'weighted_avg_price', 'min')}입니다."
    )
    sentences.append(
        f"전국 입찰량이 모집량보다 적었던 시간대는 "
        f"{current_kpis['입찰경쟁률 1.0배 미만 시간대 수']}개였습니다."
    )
    sentences.append(
        f"최대 미조달은 {current_kpis['최대 미조달 발생 시간대']}의 "
        f"{current_kpis['최대 미조달량']:,.2f}MW였습니다."
    )
    zone_rates = (
        regional_profile_zone_rates(regional_summary)
        if not regional_summary.empty
        else {}
    )
    if len(zone_rates) == 2:
        lower = min(zone_rates, key=zone_rates.get)
        sentences.append(
            f"권역 입찰경쟁률은 {lower}가 더 낮았습니다"
            f"(50Hz {zone_rates['50Hz']:.2f}배, 60Hz {zone_rates['60Hz']:.2f}배)."
        )
    return " ".join(sentences)


def regional_profile_zone_rates(regional_summary: pd.DataFrame) -> dict[str, float]:
    """지역 요약의 평균 물량을 이용해 권역 기간 입찰경쟁률을 계산합니다."""
    rates = {}
    for zone, group in regional_summary.groupby("frequency_zone"):
        rates[zone] = safe_ratio(
            group["avg_bid_volume"].sum(),
            group["avg_procurement_volume"].sum(),
        )
    return rates
