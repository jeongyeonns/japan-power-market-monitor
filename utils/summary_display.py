"""규칙 기반 요약 계산 결과를 화면 표시용 구조와 Markdown으로 변환합니다."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.national_analysis import ALL_AREA_DISPLAY, regional_profile_zone_rates
from utils.regional_analysis import AREA_DISPLAY


def _direction(value: object, positive: str, negative: str) -> str:
    if pd.isna(value):
        return "계산 불가"
    if np.isclose(float(value), 0):
        return "변화 없음"
    return positive if float(value) > 0 else negative


def _extreme_area(summary: pd.DataFrame, column: str, mode: str) -> str:
    valid = summary.dropna(subset=[column])
    if valid.empty:
        return "계산 불가"
    extreme = valid[column].min() if mode == "min" else valid[column].max()
    selected = valid.loc[np.isclose(valid[column], extreme)]
    names = [
        ALL_AREA_DISPLAY.get(row["area"], row.get("area_display", row["area"]))
        for _, row in selected.iterrows()
    ]
    return ", ".join(names)


def _comparison_change(
    comparison: pd.DataFrame | None, metric: str, positive: str, negative: str
) -> dict[str, Any]:
    if comparison is None or comparison.empty:
        value = np.nan
    else:
        rows = comparison.loc[comparison["지표"].eq(metric)]
        value = rows["절대 변화"].iloc[0] if not rows.empty else np.nan
    return {
        "direction": _direction(value, positive, negative),
        "absolute_change": abs(value) if pd.notna(value) else np.nan,
    }


def build_national_summary_data(
    national_profile: pd.DataFrame,
    regional_summary: pd.DataFrame,
    kpis: dict[str, Any],
    comparison: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """이미 계산된 전국 KPI를 항목형 요약 표시 구조로 변환합니다."""
    zone_ratios = (
        regional_profile_zone_rates(regional_summary)
        if not regional_summary.empty
        else {}
    )
    lower_zone = (
        min(zone_ratios, key=zone_ratios.get)
        if len(zone_ratios) == 2
        and all(pd.notna(value) for value in zone_ratios.values())
        else "계산 불가"
    )
    return {
        "market_volume_change": {
            **_comparison_change(
                comparison, "평균 모집량 (MW)", "증가", "감소"
            ),
            "unit": "MW",
        },
        "weighted_price_change": {
            **_comparison_change(
                comparison,
                "전국 낙찰량 가중평균 낙찰가격",
                "상승",
                "하락",
            ),
            "unit": "",
        },
        "lowest_competition_area": _extreme_area(
            regional_summary, "bid_coverage_ratio", "min"
        ),
        "highest_competition_area": _extreme_area(
            regional_summary, "bid_coverage_ratio", "max"
        ),
        "highest_price_area": _extreme_area(
            regional_summary, "weighted_avg_price", "max"
        ),
        "lowest_price_area": _extreme_area(
            regional_summary, "weighted_avg_price", "min"
        ),
        "shortage_period_count": kpis["입찰경쟁률 1.0배 미만 시간대 수"],
        "max_shortage_period": kpis["최대 미조달 발생 시간대"],
        "max_shortage_volume": kpis["최대 미조달량"],
        "lower_competition_zone": lower_zone,
        "zone_ratios": zone_ratios,
        "incomplete": (
            national_profile.empty
            or national_profile["completeness_flag"].ne("Complete").any()
        ),
    }


def build_regional_summary_data(
    area_profile: pd.DataFrame,
    kpi_table: pd.DataFrame,
    previous_comparison: pd.DataFrame | None = None,
    visible_areas: list[str] | None = None,
) -> dict[str, Any]:
    """이미 계산된 도쿄·중부 KPI를 항목형 요약 표시 구조로 변환합니다."""
    tokyo_price = kpi_table.loc["평균 낙찰가격", "도쿄"]
    chubu_price = kpi_table.loc["평균 낙찰가격", "중부"]
    price_difference = (
        abs(tokyo_price - chubu_price)
        if pd.notna(tokyo_price) and pd.notna(chubu_price)
        else np.nan
    )
    if pd.isna(price_difference):
        higher_price = "계산 불가"
    elif np.isclose(price_difference, 0):
        higher_price = "도쿄와 중부 동일"
    else:
        higher_price = "도쿄" if tokyo_price > chubu_price else "중부"

    competition = {
        area: kpi_table.loc["입찰경쟁률 (배)", AREA_DISPLAY[area]]
        for area in ("Tokyo", "Chubu")
    }
    if any(pd.isna(value) for value in competition.values()):
        higher_competition = "계산 불가"
    elif np.isclose(competition["Tokyo"], competition["Chubu"]):
        higher_competition = "도쿄와 중부 동일"
    else:
        higher_competition = (
            "도쿄"
            if competition["Tokyo"] > competition["Chubu"]
            else "중부"
        )

    counts = area_profile.groupby("area").agg(
        bid_below=("bid_coverage_ratio", lambda values: int((values < 1).sum())),
        procurement_below=(
            "procurement_rate",
            lambda values: int((values < 1).sum()),
        ),
    )
    below_one = {
        area: counts.loc[area, "bid_below"] if area in counts.index else np.nan
        for area in ("Tokyo", "Chubu")
    }
    below_full = {
        area: (
            counts.loc[area, "procurement_below"]
            if area in counts.index
            else np.nan
        )
        for area in ("Tokyo", "Chubu")
    }

    valid_shortage = area_profile.dropna(subset=["shortage_volume"])
    if valid_shortage.empty:
        max_shortage = {
            "area": "계산 불가",
            "period_start": "계산 불가",
            "volume": np.nan,
        }
    else:
        maximum = valid_shortage.loc[valid_shortage["shortage_volume"].idxmax()]
        max_shortage = {
            "area": AREA_DISPLAY.get(maximum["area"], maximum["area"]),
            "period_start": maximum["period_start"],
            "volume": maximum["shortage_volume"],
        }

    week_over_week = {}
    for area in ("Tokyo", "Chubu"):
        display = AREA_DISPLAY[area]
        rows = (
            previous_comparison.loc[
                previous_comparison["지표"].eq("평균 낙찰가격")
                & previous_comparison["지역"].eq(display)
            ]
            if previous_comparison is not None
            and not previous_comparison.empty
            else pd.DataFrame()
        )
        change = rows["절대 변화"].iloc[0] if not rows.empty else np.nan
        week_over_week[area] = {
            "direction": _direction(change, "상승", "하락"),
            "absolute_change": abs(change) if pd.notna(change) else np.nan,
        }

    return {
        "visible_areas": visible_areas or ["Tokyo", "Chubu"],
        "average_prices": {
            area: kpi_table.loc["평균 낙찰가격", AREA_DISPLAY[area]]
            for area in ("Tokyo", "Chubu")
        },
        "higher_price_area": higher_price,
        "price_difference": price_difference,
        "higher_competition_area": higher_competition,
        "competition_ratios": competition,
        "below_one_periods": below_one,
        "below_full_procurement_periods": below_full,
        "max_shortage": max_shortage,
        "week_over_week_price": week_over_week,
        "incomplete": (
            area_profile.empty
            or area_profile["observation_count"].ne(7).any()
            or area_profile[
                ["avg_price", "procurement_volume", "bid_volume", "awarded_volume"]
            ]
            .isna()
            .any()
            .any()
        ),
    }


def _number(value: object, suffix: str = "") -> str:
    return (
        "계산 불가"
        if pd.isna(value)
        else f"{float(value):,.2f}{suffix}"
    )


def _count(value: object) -> str:
    return "계산 불가" if pd.isna(value) else f"{int(value)}개"


def _change_text(change: dict[str, Any], unit: str = "") -> str:
    if pd.isna(change["absolute_change"]):
        return "계산 불가"
    return f"{_number(change['absolute_change'], unit)} {change['direction']}"


def national_summary_markdown(summary: dict[str, Any]) -> str:
    """전국 요약을 세로형 Markdown 목록으로 만듭니다."""
    volume = summary["market_volume_change"]
    price = summary["weighted_price_change"]
    lines = [
        "### 이번 주 전국 시장 요약",
        "",
        f"- **평균 모집량 (TSO별):** 전주 대비 **{_change_text(volume, ' MW')}**",
        f"- **낙찰량 가중평균 낙찰가격 (전원 소재지별):** 전주 대비 **{_change_text(price)}**",
        f"- **지역별 입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량):** 최저 **{summary['lowest_competition_area']}**, 최고 **{summary['highest_competition_area']}**",
        f"- **지역별 가중평균 낙찰가격 (전원 소재지별):** 최고 **{summary['highest_price_area']}**, 최저 **{summary['lowest_price_area']}**",
        f"- **입찰 부족 시간대:** **{_count(summary['shortage_period_count'])}**",
        f"- **최대 미조달:** **{summary['max_shortage_period']}**, **{_number(summary['max_shortage_volume'], ' MW')}**",
        "- **권역별 입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량)**",
    ]
    for zone in ("50Hz", "60Hz"):
        lines.append(
            f"  - {zone}: **{_number(summary['zone_ratios'].get(zone, np.nan), '배')}**"
        )
    lines.append(
        f"  - 더 낮은 권역: **{summary['lower_competition_zone']}**"
    )
    return "\n".join(lines)


def regional_summary_markdown(summary: dict[str, Any]) -> str:
    """선택된 지역의 핵심 네 항목만 Markdown 목록으로 만듭니다."""
    visible = summary.get("visible_areas", ["Tokyo", "Chubu"])
    comparing = len(visible) == 2
    title = "도쿄·중부 비교 핵심 요약" if comparing else f"{AREA_DISPLAY[visible[0]]} 핵심 요약"
    if comparing:
        if summary["higher_price_area"] == "계산 불가":
            price_description = "**계산 불가**"
        elif summary["higher_price_area"] == "도쿄와 중부 동일":
            price_description = "**도쿄와 중부 동일**"
        else:
            other = "중부" if summary["higher_price_area"] == "도쿄" else "도쿄"
            price_description = (
                f"{summary['higher_price_area']}가 {other}보다 "
                f"**{_number(summary['price_difference'])} 높음**"
            )
    else:
        price_description = f"**{_number(summary['average_prices'][visible[0]])}**"

    def competition_interpretation(value: object) -> str:
        if pd.isna(value):
            return "계산 불가"
        if np.isclose(float(value), 1.0):
            return "모집량과 입찰량이 비슷함"
        if float(value) < 1.0:
            return "모집량보다 입찰량이 적음"
        return "모집량보다 입찰량이 많음"

    availability = []
    for area in visible:
        display = AREA_DISPLAY[area]
        count = summary["below_one_periods"][area]
        if pd.isna(count):
            availability.append(f"{display}는 계산 불가")
        elif int(count) == 0:
            availability.append(
                f"{display}는 모든 시간대에서 입찰량이 모집량 이상이었습니다"
            )
        else:
            availability.append(
                f"{display}는 일부 시간대에서 입찰량이 모집량보다 적었습니다"
            )

    price_changes = []
    for area in visible:
        change = summary["week_over_week_price"][area]
        price_changes.append(
            f"{AREA_DISPLAY[area]} **{_change_text(change)}**"
        )
    lines = [
        f"### {title}",
        "",
        "입찰경쟁률은 입찰량을 모집량으로 나눈 값입니다.",
        "",
        f"- **평균 낙찰가격:** {price_description}",
        "- **모집량 대비 입찰 수준**",
    ]
    for area in visible:
        value = summary["competition_ratios"][area]
        lines.append(
            f"  - {AREA_DISPLAY[area]}: **{_number(value, '배')}** — "
            f"{competition_interpretation(value)}"
        )
    lines.extend(
        [
            f"- **입찰 여유:** {' '.join(availability)}",
            f"- **전주 대비 평균 낙찰가격:** {', '.join(price_changes)}",
            "",
            "시간대별 세부 내용은 아래 그래프와 상세 데이터 표에서 확인할 수 있습니다.",
        ]
    )
    return "\n".join(lines)


def excess_award_warning_markdown(
    period_counts: dict[str, int], source_row_count: int | None = None
) -> str:
    """낙찰량이 모집량보다 많은 사례를 쉬운 항목형 경고로 표시합니다."""
    lines = [
        "**확인이 필요한 데이터가 있습니다.**",
        "",
        "일부 시간대에서 낙찰량이 모집량보다 크게 표시되었습니다.",
        "",
    ]
    for label, count in period_counts.items():
        lines.append(f"- {label}: **{int(count):,}개 시간대**")
    if source_row_count is not None:
        lines.append(
            "- 관련 원본 데이터 행: "
            f"**{int(source_row_count):,}건**"
        )
    lines.extend(
        [
            "- 앱은 원본 EPRX 값을 수정하지 않고 그대로 표시합니다.",
            "",
            "원본 데이터의 집계 기준이나 지역 구분 방식에 따른 차이일 수 있으므로, "
            "해석 시 참고가 필요합니다.",
        ]
    )
    return "\n".join(lines)
