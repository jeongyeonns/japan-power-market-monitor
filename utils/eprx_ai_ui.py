"""Streamlit presentation helpers for the opt-in EPRX AI analysis."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, MutableMapping

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.eprx_ai_analysis import (
    build_eprx_statistical_fallback,
    calculate_eprx_context_hash,
    generate_eprx_ai_analysis,
    resolve_openai_settings,
)
from utils.eprx_ai_pipeline import (
    build_detailed_context,
    check_eprx_ai_readiness,
    load_local_eprx_grid_context,
    local_grid_file_fingerprint,
    local_grid_week_fingerprint,
)
from utils.eprx_driver_statistics import FAST_CONTEXT_VERSION, build_eprx_fast_context


DISPLAY_NAMES = {
    "procurement_volume_mw": "1차 조정력 모집량",
    "demand_mw": "전력수요",
    "solar_mw": "태양광 발전량",
    "solar_generation_mw": "태양광 발전량",
    "wind_mw": "풍력 발전량",
    "wind_generation_mw": "풍력 발전량",
    "renewable_generation_mw": "재생에너지 발전량",
    "renewable_share_pct": "재생에너지 발전 비중",
    "residual_demand_proxy_mw": "잔여수요 추정치",
    "abs_demand_ramp_30m_mw": "전력수요 30분 변동폭",
    "abs_residual_demand_ramp_30m_mw": "잔여수요 30분 변동폭",
    "abs_renewable_ramp_30m_mw": "재생에너지 30분 변동폭",
    "abs_solar_ramp_30m_mw": "태양광 30분 변동폭",
}

LIMITATION_TRANSLATIONS = {
    "This is retrospective statistical association analysis using public 30-minute actuals.":
        "본 분석은 공개된 30분 실적을 이용한 사후 연관성 분석입니다.",
    "Actual demand and renewable output may differ from forecasts available when EPRX procurement was decided.":
        "실제 수요·재생에너지 실적은 모집량 결정 시점의 예측치와 다를 수 있습니다.",
    "Residual demand is a proxy calculated from public data.":
        "잔여수요는 공개자료를 이용한 추정치입니다.",
    "Thirty-minute data cannot reproduce sub-second primary reserve variation.":
        "30분 자료로는 1차 조정력이 대응하는 초단주기 변동을 직접 재현할 수 없습니다.",
}
STATISTICS_ALGORITHM_VERSION = "3"
PRESENTATION_VERSION = "dual-axis-raw-mw-v1"


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and number not in (float("inf"), float("-inf")) else None


def format_mw(value: Any) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{number:,.1f} MW"


def format_percent(value: Any) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{number:.1f}%"


def format_correlation(value: Any) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{number:+.2f}"


def format_number(value: Any, decimals: int = 2) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{number:,.{decimals}f}"


def semantic_metric_type(metric_path: str, unit: str = "") -> str:
    path = metric_path.lower()
    if any(token in path for token in ("r_squared", "adjusted_r_squared")):
        return "r_squared"
    if any(token in path for token in ("pearson", "spearman", "correlation")):
        return "correlation"
    if any(token in path for token in ("_pct", "percent", "change_pct", "join_rate", "share")):
        return "percent"
    if any(token in path for token in ("count", "rows", "observations", "sample_count", ".n")):
        return "count"
    if any(token in path for token in ("procurement", "demand", "renewable_generation", "ramp", "_mw")):
        return "mw"
    normalized_unit = unit.strip().lower()
    if normalized_unit == "mw": return "mw"
    if normalized_unit in {"%", "percentage"}: return "percent"
    if normalized_unit in {"count", "개", "건", "행"}: return "count"
    return "number"


def correlation_strength(value: Any) -> str:
    number = _finite_number(value)
    if number is None or abs(number) < 0.20: return "뚜렷한 관계 없음"
    if abs(number) < 0.40: return "약한 관계"
    if abs(number) < 0.60: return "중간 수준의 관계"
    if abs(number) < 0.80: return "강한 관계"
    return "매우 강한 관계"


def _clock_label(value: Any) -> str:
    try:
        hour, minute = (int(part) for part in str(value).split(":")[:2])
    except (TypeError, ValueError):
        return str(value)
    period = "오전" if hour < 12 else "오후"
    display_hour = hour % 12 or 12
    return f"{period} {display_hour}시" + (f" {minute}분" if minute else "")


def _association_sentence(name: str, relation: dict[str, Any]) -> str:
    pearson = _finite_number(relation.get("pearson")); spearman = _finite_number(relation.get("spearman"))
    if pearson is None or spearman is None:
        return f"{name}와 모집량의 관계를 계산할 자료가 충분하지 않습니다."
    direction = "같이 증가하는" if pearson >= 0 else "반대로 움직이는"
    agreement = ("두 방식으로 확인해도 같은 방향의 관계가 나타났습니다." if pearson * spearman >= 0
                 else "두 방식의 관계 방향이 달라 일관된 경향으로 보기는 어렵습니다.")
    return (f"{name}가 높은 시간대일수록 모집량도 {direction} 경향이 나타났습니다. "
            f"관계의 크기는 {correlation_strength(pearson)}입니다"
            f"(r = {format_correlation(pearson)}, ρ = {format_correlation(spearman)}). {agreement}")


def _signed_mw(value: Any) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{number:+,.1f} MW"


def _signed_percent(value: Any) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{number:+.1f}%"


def _relationship_label(relation: dict[str, Any]) -> str:
    pearson = _finite_number(relation.get("pearson")); spearman = _finite_number(relation.get("spearman"))
    if pearson is None or spearman is None: return "자료 부족"
    direction = "같은 방향" if pearson >= 0 else "반대 방향"
    strength = correlation_strength(pearson).replace("중간 수준의 관계", "중간 수준")
    return (f"{strength} · {direction} "
            f"(r = {format_correlation(pearson)}, ρ = {format_correlation(spearman)})")


DIRECTION_CHANGE_THRESHOLD_PCT = 1.0


def classify_weekly_direction(procurement_change_pct: Any,
                              residual_volatility_change_pct: Any) -> str:
    """Classify week-over-week directions without implying causality."""
    procurement = _finite_number(procurement_change_pct)
    volatility = _finite_number(residual_volatility_change_pct)
    if procurement is None or volatility is None:
        return "undetermined"
    if (abs(procurement) < DIRECTION_CHANGE_THRESHOLD_PCT
            or abs(volatility) < DIRECTION_CHANGE_THRESHOLD_PCT):
        return "undetermined"
    return "same" if (procurement > 0) == (volatility > 0) else "opposite"


def _direction_symbol(value: Any) -> str:
    number = _finite_number(value)
    if number is None or abs(number) < DIRECTION_CHANGE_THRESHOLD_PCT:
        return "→"
    return "↑" if number > 0 else "↓"


def _build_direction_conclusion(procurement: dict[str, Any], comparison: dict[str, Any],
                                volatility: dict[str, Any], historical: dict[str, Any]) -> dict[str, str]:
    procurement_pct = comparison.get("change_pct")
    volatility_pct = volatility.get("change_pct")
    status = classify_weekly_direction(procurement_pct, volatility_pct)
    labels = {
        "same": ("🟢 모집량과 잔여수요 변동이 같은 방향으로 움직였습니다.", "● 같은 방향"),
        "opposite": ("🟠 모집량과 잔여수요 변동은 서로 다른 방향으로 움직였습니다.", "● 다른 방향"),
        "undetermined": ("⚪ 이번 주에는 두 지표의 방향 관계를 뚜렷하게 판단하기 어렵습니다.", "● 뚜렷한 방향 없음"),
    }


def _build_intraday_tendency(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "available":
        return {"status": "unavailable", "headline": "시간대별 경향을 계산할 자료가 충분하지 않습니다."}
    rho = _finite_number(summary.get("profile_spearman"))
    strength = summary.get("profile_strength")
    direction = summary.get("profile_direction")
    strength_labels = {"none": "뚜렷하지 않은", "weak": "약한", "moderate": "중간 수준의",
                       "strong": "강한", "very_strong": "매우 강한"}
    if direction == "none":
        headline = "⚪ 이번 주에는 모집량과 잔여수요 변동 사이에 뚜렷한 시간대 경향이 확인되지 않았습니다."
        detail = "두 지표가 높은 시간대가 일관되게 겹친다고 보기는 어렵습니다."
    elif direction == "same":
        headline = "🟢 모집량이 높은 시간대에서 잔여수요 변동도 비교적 크게 나타났습니다."
        detail = f"시간대 패턴은 {strength_labels.get(strength, '')} 같은 방향 경향을 보였습니다."
    else:
        headline = "🟠 모집량이 높은 시간대와 잔여수요 변동이 큰 시간대가 다르게 나타났습니다."
        detail = (f"시간대 패턴은 {strength_labels.get(strength, '')} 반대 방향 경향을 보였습니다. "
                  "공개 30분 변동 패턴만으로 모집량의 시간대별 차이를 설명하기 어렵습니다.")
    overlap = int(summary.get("high_slot_overlap_count") or 0)
    return {"status": "available", "headline": headline, "detail": detail,
            "rho": rho, "overlap_count": overlap, "overlap_pct": summary.get("high_slot_overlap_pct"),
            "overlap_text": f"모집량이 높은 상위 12개 시간대 중 {overlap}개 시간대에서 잔여수요 변동도 상위 25%였습니다.",
            "profile": summary.get("profile", [])}
    headline, badge = labels[status]
    current_volatility = volatility.get("current")
    previous_volatility = volatility.get("previous")
    percentile = _finite_number((historical or {}).get("percentile"))
    history_text = f" 최근 관측주 중 상위 {100 - percentile:.0f}% 수준입니다." if percentile is not None else ""
    if status == "same":
        interpretation = "공개자료 기준으로 두 지표가 같은 방향의 움직임을 보였습니다."
    elif status == "opposite":
        interpretation = ("공개 30분 잔여수요 변동만으로는 이번 주 모집량의 변화 방향을 설명하기 어렵습니다. "
                          "평상시분 외 요인의 가능성은 생각할 수 있지만 공개자료만으로 확인할 수 없습니다.")
    else:
        interpretation = "비교값이 없거나 한쪽 변화가 작아 방향 일치 여부를 무리하게 판정하지 않았습니다."
    return {
        "status": status, "headline": headline, "badge": badge,
        "procurement_line": (f"모집량  {format_mw(comparison.get('previous'))} → {format_mw(procurement.get('mean'))}  "
                             f"{_direction_symbol(procurement_pct)} {_signed_percent(procurement_pct)}"),
        "volatility_line": (f"잔여수요 변동  {format_mw(previous_volatility)} → {format_mw(current_volatility)}  "
                            f"{_direction_symbol(volatility_pct)} {_signed_percent(volatility_pct)}"),
        "absolute_text": (f"이번 주 잔여수요는 연속된 30분 구간 사이에서 평균 {format_mw(current_volatility)} 움직였습니다."
                          f"{history_text}"),
        "interpretation": interpretation,
        "comparison_line": (f"모집량 변화: {_signed_percent(procurement_pct)} {_direction_symbol(procurement_pct)} / "
                            f"잔여수요 변동: {_signed_percent(volatility_pct)} {_direction_symbol(volatility_pct)} → {badge.removeprefix('● ')}"),
        "caution": "※ 이는 전주 대비 방향 일치를 의미하며 인과관계나 모집량의 공식 산정 원인을 의미하지 않습니다.",
    }


def build_eprx_readable_presentation(context: dict[str, Any]) -> dict[str, Any]:
    selected = context.get("selected_week", {}); procurement = selected.get("procurement", {})
    comparison = selected.get("procurement_change", {}) or {}
    notable = selected.get("notable_time_blocks", {}) or {}
    high = (notable.get("highest") or [{}])[0]; low = (notable.get("lowest") or [{}])[0]
    changes = selected.get("driver_changes", {}) or {}
    historical = selected.get("historical_position", {}) or {}
    intraday_summary = selected.get("intraday_profile_summary", {}) or {}
    demand_change = changes.get("mean_demand_mw", {}) or {}
    residual_change = changes.get("mean_residual_demand_proxy_mw", {}) or {}
    renewable_change = changes.get("mean_renewable_generation_mw", {}) or {}
    solar_change = changes.get("mean_solar_mw", {}) or {}
    wind_change = changes.get("mean_wind_mw", {}) or {}
    share_change = changes.get("mean_renewable_share_pct", {}) or {}
    volatility_change = changes.get("mean_abs_residual_demand_ramp_30m_mw", {}) or {}
    volatility_std_change = changes.get("residual_demand_ramp_std_30m_mw", {}) or {}
    volatility_max_change = changes.get("maximum_abs_residual_demand_ramp_30m_mw", {}) or {}
    volatility_p95_change = changes.get("p95_abs_residual_demand_ramp_30m_mw", {}) or {}
    demand_ramp_change = changes.get("mean_abs_demand_ramp_30m_mw", {}) or {}
    renewable_ramp_change = changes.get("mean_abs_renewable_ramp_30m_mw", {}) or {}
    relations = context.get("selected_week_correlations", {}) or context.get("time_adjusted_correlations", {})
    demand = relations.get("demand_mw", {}); residual = relations.get("residual_demand_proxy_mw", {})
    renewable = relations.get("renewable_generation_mw", {})
    volatility = relations.get("abs_residual_demand_ramp_30m_mw", {})
    region = context.get("region")
    source = ("도쿄전력파워그리드(TEPCO PG)" if region == "Tokyo"
              else "중부전력파워그리드(Chubu PG)" if region == "Chubu" else "지역 계통운영기관")
    mean = procurement.get("mean"); previous = comparison.get("previous"); change = comparison.get("change")
    change_pct = comparison.get("change_pct")
    volatility_history = historical.get("mean_abs_residual_demand_ramp_30m_mw", {}) or {}
    volatility_percentile = _finite_number(volatility_history.get("percentile"))
    volatility_delta_label = (f"최근 분포 상위 {100 - volatility_percentile:.0f}%" if volatility_percentile is not None
                              else f"전주 {format_mw(volatility_change.get('previous'))}")
    direction_conclusion = _build_direction_conclusion(
        procurement, comparison, volatility_change, volatility_history)
    direction = "증가" if (_finite_number(change) or 0) >= 0 else "감소"
    procurement_lines = [
        f"평균 모집량은 {format_mw(mean)}로, 전주 {format_mw(previous)}보다 {format_mw(abs(_finite_number(change) or 0))} {direction}했습니다.",
        f"이번 주 모집량은 {format_mw(procurement.get('minimum'))}~{format_mw(procurement.get('maximum'))} 범위에서 움직였습니다.",
        f"{_clock_label(high.get('time_block'))} 평균 {format_mw(high.get('average_procurement_mw'))}로 가장 높았고, {_clock_label(low.get('time_block'))} 평균 {format_mw(low.get('average_procurement_mw'))}로 가장 낮았습니다.",
    ]
    demand_sentence = _association_sentence("전력수요", demand)
    residual_sentence = _association_sentence(
        "잔여수요 추정치(전력수요에서 태양광·풍력 발전량을 뺀 값)", residual)
    demand_r = _finite_number(demand.get("pearson")); residual_r = _finite_number(residual.get("pearson"))
    comparison_sentence = "이번 주에는 전력수요와 잔여수요의 모집량 관계 크기에 큰 차이가 없었습니다."
    if demand_r is not None and residual_r is not None and abs(abs(demand_r) - abs(residual_r)) >= 0.10:
        if abs(demand_r) > abs(residual_r):
            comparison_sentence = (f"이번 주에는 전력수요와 모집량의 관계(r = {format_correlation(demand_r)})가 "
                f"잔여수요와의 관계(r = {format_correlation(residual_r)})보다 더 크게 나타났습니다. 따라서 이번 주 "
                "자료에서는 잔여수요가 전체 수요보다 모집량을 더 잘 설명했다고 보기는 어렵습니다.")
        else:
            comparison_sentence = (f"이번 주에는 잔여수요와 모집량의 관계(r = {format_correlation(residual_r)})가 "
                f"전력수요와의 관계(r = {format_correlation(demand_r)})보다 더 크게 나타났습니다. 다만 이는 함께 "
                "움직인 정도의 비교이며 잔여수요가 모집량을 결정했다는 의미는 아닙니다.")
    demand_context = (f"주간 평균 전력수요는 {format_mw(demand_change.get('current'))}, "
                      f"잔여수요는 {format_mw(residual_change.get('current'))}였습니다. ")
    volatility_delta = _finite_number(volatility_change.get("change"))
    procurement_delta = _finite_number(change)
    if volatility_delta is None or procurement_delta is None:
        interpretation = ["전주 비교 자료가 부족해 잔여수요 변동성과 모집량의 방향을 비교할 수 없습니다."]
    elif volatility_delta > 0 and procurement_delta > 0:
        interpretation = ["이번 주에는 전주 대비 잔여수요의 30분 변동폭이 확대되었으며, 모집량도 함께 증가했습니다. 공개된 30분 자료 기준으로 평상시 조정력 필요량 증가와 같은 방향의 움직임이 확인됩니다."]
    elif volatility_delta < 0 and procurement_delta < 0:
        interpretation = ["이번 주에는 전주 대비 잔여수요 변동폭이 축소되었으며, 모집량도 함께 감소했습니다. 수요 및 재생에너지 출력의 변동이 완화된 점이 모집량 감소와 같은 방향으로 나타났습니다."]
    else:
        interpretation = ["잔여수요 변동성과 모집량의 방향이 일치하지 않아 공개된 수요·재생에너지 자료만으로 이번 주 모집량 변화를 설명하기는 어렵습니다."]
    demand_ramp = _finite_number(demand_ramp_change.get("current"))
    renewable_ramp = _finite_number(renewable_ramp_change.get("current"))
    background = "전력수요와 재생에너지 출력 변화의 크기는 자료 부족으로 비교할 수 없습니다."
    if demand_ramp is not None and renewable_ramp is not None:
        larger = "전력수요" if demand_ramp >= renewable_ramp else "재생에너지 출력"
        background = f"이번 주에는 {larger}의 30분 평균 변동폭이 다른 요소보다 크게 나타났습니다. 이는 변동의 배경 비교이며 모집량의 직접 원인을 뜻하지 않습니다."
    interpretation = [
        f"이번 주 1차 조정력 평균 모집량은 {format_mw(mean)}였습니다.",
        f"공개 30분 자료에서 잔여수요는 한 코마 사이 평균 {format_mw(volatility_change.get('current'))} 움직였습니다. 공식 평상시분 산정값은 아니지만 공개자료에서 확인되는 변동의 크기를 보여줍니다.",
        background,
    ]
    cards = [
        {"label": "이번 주 평균 모집량", "value": format_mw(mean),
         "delta": None},
        {"label": "전주 대비", "value": f"{_signed_mw(change)} ({_signed_percent(change_pct)})", "delta": None},
        {"label": "잔여수요 30분 평균 변동폭", "value": format_mw(volatility_change.get("current")),
         "delta": volatility_delta_label},
        {"label": "큰 잔여수요 변동", "value": f"P95 {format_mw(volatility_p95_change.get('current'))}", "delta": None},
    ]
    procurement_table = [
        {"항목": "이번 주 평균 모집량", "값": format_mw(mean)},
        {"항목": "전주 평균 모집량", "값": format_mw(previous)},
        {"항목": "전주 대비 변화", "값": f"{_signed_mw(change)} ({_signed_percent(change_pct)})"},
        {"항목": "주간 범위", "값": f"{format_mw(procurement.get('minimum'))} ~ {format_mw(procurement.get('maximum'))}"},
        {"항목": "최고 시간대", "값": f"{high.get('time_block') or '—'} / {format_mw(high.get('average_procurement_mw'))}"},
        {"항목": "최저 시간대", "값": f"{low.get('time_block') or '—'} / {format_mw(low.get('average_procurement_mw'))}"},
    ]
    def comparison_row(label, item, formatter=format_mw):
        return {"지표": label, "이번 주": formatter(item.get("current")),
                "전주": formatter(item.get("previous")), "증감률": _signed_percent(item.get("change_pct"))}
    comparison_table = [
        {"지표": "잔여수요 30분 평균 변동폭", "이번 주": format_mw(volatility_change.get("current")), "전주": format_mw(volatility_change.get("previous")), "참고": "1차 평상시분 관련 핵심 공개지표"},
        {"지표": "전력수요 30분 평균 변동폭", "이번 주": format_mw(demand_ramp_change.get("current")), "전주": format_mw(demand_ramp_change.get("previous")), "참고": "수요 자체의 변화"},
        {"지표": "재생에너지 30분 평균 변동폭", "이번 주": format_mw(renewable_ramp_change.get("current")), "전주": format_mw(renewable_ramp_change.get("previous")), "참고": "태양광·풍력 출력 변화"},
    ]
    detail_table = [
        comparison_row("평균 전력수요 (MW)", demand_change), comparison_row("평균 태양광 발전량 (MW)", solar_change),
        comparison_row("평균 풍력 발전량 (MW)", wind_change), comparison_row("평균 태양광+풍력 발전량 (MW)", renewable_change),
        comparison_row("재생E 비중 (%)", share_change, format_percent), comparison_row("평균 잔여수요 (MW)", residual_change),
    ]
    relationship_table = [
        {"변수": "잔여수요 변동폭 |Δ잔여수요|", "모집량과의 관계": _relationship_label(volatility), "해석": "가장 중요한 공개자료 proxy"},
        {"변수": "전력수요", "모집량과의 관계": _relationship_label(demand), "해석": "30분 실적과 모집량의 동행 정도"},
        {"변수": "잔여수요", "모집량과의 관계": _relationship_label(residual), "해석": "수요-태양광-풍력과 모집량의 동행 정도"},
        {"변수": "태양광+풍력 발전량", "모집량과의 관계": _relationship_label(renewable), "해석": "재생에너지 출력과 모집량의 동행 정도"},
    ]
    both_positive = demand_r is not None and residual_r is not None and demand_r > 0 and residual_r > 0
    relationship_intro = "잔여수요 변동폭을 우선하여 모집량과 같은 방향으로 움직인 정도를 비교합니다. 상관관계는 인과관계나 공식 산정식을 뜻하지 않습니다."
    week = selected.get("week", {}) or {}
    profile = selected.get("demand_intraday_profile", []) or []
    observations = sum(int(row.get("observation_count") or 0) for row in profile)
    return {"summary_cards": cards,
        "presentation_version": PRESENTATION_VERSION,
        "procurement_table": procurement_table, "procurement_changes": procurement_lines,
        "comparison_table": comparison_table, "relationship_intro": relationship_intro, "relationship_table": relationship_table,
        "detail_table": detail_table,
        "direction_conclusion": direction_conclusion,
        "intraday_tendency": _build_intraday_tendency(intraday_summary),
        "volatility_detail": (f"이번 주 Δ잔여수요 표준편차 {format_mw(volatility_std_change.get('current'))} / "
                              f"최대 |Δ잔여수요| {format_mw(volatility_max_change.get('current'))}"),
        "demand_intraday_profile": profile,
        "grid_source_label": source, "week_start": week.get("start"), "week_end": week.get("end"),
        "demand_observation_count": observations,
        "relationships": [demand_context + demand_sentence, residual_sentence, comparison_sentence],
        "interpretation": interpretation,
        "notes": ["※ 공개된 30분 단위 잔여수요 변동은 1차 조정력 평상시분의 공식 산정값이 아니라 주간 변화를 살펴보기 위한 보조지표입니다. 실제 모집량에는 이상시 대응분, 수의계약 물량 등 다른 요인도 영향을 줄 수 있습니다."]}


def _evidence_display_name(evidence: dict[str, Any]) -> str:
    path = str(evidence.get("metric_path", ""))
    for metric, display in sorted(DISPLAY_NAMES.items(), key=lambda item: -len(item[0])):
        if metric in path:
            return display
    return str(evidence.get("display_name") or "분석 지표")


def _format_evidence(evidence: dict[str, Any]) -> str:
    name = _evidence_display_name(evidence)
    value = evidence.get("value")
    path = str(evidence.get("metric_path", "")).lower()
    unit = str(evidence.get("unit", "")).strip()
    metric_type = semantic_metric_type(path, unit)
    if metric_type == "r_squared":
        metric = f"R² = {format_number(value)}"
    elif "spearman" in path:
        metric = f"Spearman ρ = {format_correlation(value)}"
    elif metric_type == "correlation":
        metric = f"Pearson r = {format_correlation(value)}"
    elif "percentile" in path:
        metric = f"백분위 {format_number(value, 1)}"
    elif metric_type == "mw":
        metric = format_mw(value)
    elif metric_type == "percent":
        metric = format_percent(value)
    elif metric_type == "count":
        metric = f"{format_number(value, 0)}개"
    else:
        metric = format_number(value)
    interpretation = str(evidence.get("interpretation", "")).strip()
    return f"- {name} — {metric}" + (f"\n  {interpretation}" if interpretation else "")


def _render_values(target, values: Any) -> None:
    if isinstance(values, str):
        target.write(values)
        return
    for value in values or []:
        if isinstance(value, dict) and "value" in value:
            target.markdown(_format_evidence(value))
        else:
            target.markdown(f"- {LIMITATION_TRANSLATIONS.get(str(value), value)}")


def _render_summary_cards(target, cards: list[dict[str, Any]]) -> None:
    if not cards: return
    columns = target.columns(len(cards))
    for column, card in zip(columns, cards):
        kwargs = {"label": card["label"], "value": card["value"]}
        if card.get("delta") is not None: kwargs["delta"] = card["delta"]
        column.metric(**kwargs)


def _render_relationship_table(target, rows: list[dict[str, Any]]) -> None:
    target.table(pd.DataFrame(rows))


def _render_intraday_demand_profile(target, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    frame = pd.DataFrame(rows).rename(columns={
        "time_block": "시간대", "procurement_mw": "모집량", "demand_mw": "전력수요",
        "residual_demand_mw": "잔여수요", "renewable_generation_mw": "재생에너지 출력량",
        "abs_residual_demand_ramp_30m_mw": "평균 |Δ잔여수요|",
    })
    indexed = frame.set_index("시간대")
    if "평균 |Δ잔여수요|" in indexed:
        target.line_chart(indexed[["평균 |Δ잔여수요|"]], use_container_width=True)
    target.caption("각 값은 선택 주차에서 동일한 시간대의 30분 잔여수요 변화량 절댓값을 평균한 값입니다. 어느 시간대에서 계통의 잔여수요 변화가 크게 나타났는지 보여줍니다.")


def _render_intraday_tendency(target, tendency: dict[str, Any]) -> None:
    target.markdown("### 시간대별 모집량·잔여수요 변동 경향")
    target.write("아래 그래프는 선택 주차의 7일을 동일한 30분 시간대끼리 평균하여, 모집량과 잔여수요 변동의 하루 시간대별 패턴을 비교합니다. 두 지표의 값 범위가 달라 모집량은 왼쪽 축, 잔여수요 변동폭은 오른쪽 축에 각각 실제 MW 단위로 표시합니다.")
    profile = pd.DataFrame(tendency.get("profile", []))
    if not profile.empty:
        figure = _intraday_dual_axis_figure(profile)
        target.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    target.caption("※ 두 Y축의 눈금 범위가 서로 다르므로 선의 높이나 기울기 자체를 직접 비교하기보다, 어느 시간대에서 각각 증가·감소하는지와 패턴의 방향을 중심으로 확인해 주세요.")
    target.caption("이 비교는 시간대별 경향성을 확인하기 위한 참고 분석이며 공식 1차 조정력 산정값이나 인과관계를 의미하지 않습니다.")
    target.markdown("### 시간대별 경향성")
    if tendency.get("status") != "available":
        target.info(tendency.get("headline", "시간대별 경향을 계산할 수 없습니다.")); return
    target.info(f"**{tendency.get('headline')}**\n\n{tendency.get('detail')}\n\n{tendency.get('overlap_text')}")
    target.markdown(f"**시간대 패턴 상관 ρ = {format_correlation(tendency.get('rho'))} · 높은 시간대 겹침 {tendency.get('overlap_count')}/12**")
    target.caption("두 지표 모두 하루 중 반복되는 시간대 패턴을 가질 수 있으므로, 이 결과는 48-slot 평균 일중 패턴의 유사성을 보는 참고지표입니다. 잔여수요 변동이 모집량을 결정했다는 의미가 아닙니다.")


def _intraday_dual_axis_figure(profile: pd.DataFrame) -> go.Figure:
    """Plot raw MW profiles on independent axes with a shared hover."""
    frame = profile.sort_values("time_block", kind="stable").copy()
    custom = frame[["procurement_mw", "abs_residual_demand_ramp_30m_mw"]].to_numpy()
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(
        x=frame["time_block"], y=frame["procurement_mw"], name="1차 조정력 모집량",
        mode="lines", line={"color": "#2563EB", "width": 2.5}, customdata=custom,
        hovertemplate=("시간대 %{x}<br>1차 조정력 모집량: %{customdata[0]:,.1f} MW"
                       "<br>잔여수요 30분 변동폭: %{customdata[1]:,.1f} MW<extra></extra>"),
    ), secondary_y=False)
    figure.add_trace(go.Scatter(
        x=frame["time_block"], y=frame["abs_residual_demand_ramp_30m_mw"],
        name="잔여수요 30분 변동폭", mode="lines", line={"color": "#D97706", "width": 2.5, "dash": "dot"},
        customdata=custom,
        hovertemplate=("시간대 %{x}<br>1차 조정력 모집량: %{customdata[0]:,.1f} MW"
                       "<br>잔여수요 30분 변동폭: %{customdata[1]:,.1f} MW<extra></extra>"),
    ), secondary_y=True)
    figure.update_xaxes(title_text="시간대", tickmode="array",
                        tickvals=frame["time_block"].iloc[::4].tolist())
    figure.update_yaxes(title_text="1차 조정력 모집량 (MW)", secondary_y=False,
                        title_font={"color": "#2563EB"}, tickfont={"color": "#2563EB"})
    figure.update_yaxes(title_text="잔여수요 30분 변동폭 (MW)", secondary_y=True,
                        title_font={"color": "#D97706"}, tickfont={"color": "#D97706"})
    figure.update_layout(height=420, hovermode="x unified", margin={"l": 20, "r": 20, "t": 25, "b": 20},
                         legend={"orientation": "h", "y": 1.08, "x": 0})
    return figure


def _render_level_profile(target, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    frame = pd.DataFrame(rows).rename(columns={"time_block": "시간대", "demand_mw": "전력수요",
        "residual_demand_mw": "잔여수요", "renewable_generation_mw": "재생에너지 출력량"})
    series = [name for name in ("전력수요", "잔여수요", "재생에너지 출력량") if name in frame]
    target.line_chart(frame.set_index("시간대")[series], use_container_width=True)


def _render_analysis_basis(target) -> None:
    target.info(
        "**분석 기준 — 먼저 읽어주세요**\n\n"
        "**① 무엇을 분석하나요?**  이 화면은 EPRX 1차 조정력 모집량을 분석합니다. 1차 조정력 필요량에는 평상시의 잔여수요 변동에 대응하는 부분과 발전기 탈락 등에 대비하는 이상시 대응 부분 등이 함께 반영될 수 있습니다. 현재 공개된 30분 수급실적으로 직접 살펴볼 수 있는 부분은 이 중 평상시의 잔여수요 변동과 관련된 영역입니다.\n\n"
        "**② 왜 잔여수요를 보나요?**\n\n"
        "날씨 → 전력수요 · 태양광 · 풍력 → 잔여수요 → 잔여수요의 짧은 주기 변동 → 1차 조정력 평상시분 → + 이상시분·기타 요인 → EPRX 모집량\n\n"
        "잔여수요는 전력수요에서 태양광·풍력 발전량을 제외한 값입니다. 같은 전력수요라도 재생에너지 출력이 달라지면 계통이 실제로 부담하는 잔여수요가 달라집니다. 따라서 단순한 수요 수준보다 잔여수요 변동을 핵심 참고지표로 봅니다.\n\n"
        "**③ 왜 날씨와 재생에너지를 보나요?**  날씨는 모집량에 직접 입력되는 단순 변수가 아니라 기온을 통해 수요를, 일사량·풍속을 통해 태양광·풍력 출력을 바꾸는 상위 요인입니다. 현재 날씨 자료는 분석값에 연결하지 않으며 메커니즘 설명에만 사용합니다."
    )
    target.caption("※ 본 화면의 잔여수요 변동은 30분 공개자료로 계산한 보조지표입니다. 공식 1차 조정력 평상시분은 이보다 훨씬 짧은 주기의 잔여수요 데이터를 이용하므로, 아래 값은 공식 필요량을 재현한 값이 아닙니다.")
    target.caption("※ 아래 분석은 모집량 변화의 원인을 확정하는 분석이 아니라, 공개자료에서 관찰되는 계통 변동 특성을 모집량과 함께 살펴보는 분석입니다.")


def _render_direction_conclusion(target, conclusion: dict[str, str]) -> None:
    if not conclusion: return
    target.markdown("### 이번 주 결론")
    target.info(
        f"**{conclusion.get('headline', '')}**\n\n"
        f"{conclusion.get('procurement_line', '')}\n\n"
        f"{conclusion.get('volatility_line', '')}\n\n"
        f"**{conclusion.get('badge', '')}**\n\n"
        f"{conclusion.get('absolute_text', '')}\n\n"
        f"{conclusion.get('interpretation', '')}"
    )
    target.caption(conclusion.get("caution", ""))


def _render_demand_guidance(target, presentation: dict[str, Any]) -> None:
    count = int(presentation.get("demand_observation_count") or 0)
    count_text = f"{count}개" if count else "유효한"
    source = presentation.get("grid_source_label", "지역 계통운영기관")
    start = presentation.get("week_start") or "—"; end = presentation.get("week_end") or "—"
    target.info(
        "**데이터 기준**\n\n"
        f"- 전력수요: {source} 공개 지역 수급실적, 30분 단위 실적\n"
        f"- 태양광·풍력: {source} 공개 발전실적, 30분 단위 실적\n"
        "- 잔여수요: 전력수요 − 태양광 발전량 − 풍력 발전량\n"
        f"- 주간 평균: 해당 주의 {count_text} 30분 데이터를 평균\n"
        "- 시간대별 값: 해당 주 동일 시간대의 30분 데이터를 평균\n"
        "- 잔여수요 변동폭: 시간적으로 연속된 30분 잔여수요 값의 절대 변화량"
    )
    target.caption(f"분석기간 {start}~{end} / 모든 전력·발전 실적값과 모집량은 MW 기준")


def _render_explanation_notes(target) -> None:
    target.info("**이번 화면은 어떤 조정력을 분석하나요?**\n\n이 화면은 EPRX 공개자료 중 **1차 조정력(一次調整力)만 분석**합니다. 2차 조정력과 3차 조정력은 포함하지 않습니다.")
    target.warning("**공개자료로 구분하기 어려운 항목**\n\n모집량에는 평상시분과 이상시분 등 여러 요인이 함께 반영됩니다. 공개자료만으로는 평상시분·이상시분·자연체여력 공제 효과를 완전히 분리할 수 없으므로, 공개 모집량과 수요·재생에너지 지표가 함께 움직인 경향을 참고하는 분석입니다.")


def evaluate_eprx_ai_ui_state(*, market: str, region: str, week_start: Any,
                              context_status: str, complete_week: bool,
                              market_regimes: list[str] | set[str], join_success_rate: float,
                              api_key_available: bool) -> dict[str, Any]:
    reasons = []
    start = pd.Timestamp(week_start).normalize()
    if market != "EPRX": reasons.append("unsupported_market")
    if region not in {"Tokyo", "Chubu"}: reasons.append("unsupported_region")
    if start < pd.Timestamp("2026-03-14"): reasons.append("legacy_8_block_period")
    if len(set(market_regimes)) != 1: reasons.append("mixed_market_regime")
    if not complete_week: reasons.append("incomplete_week")
    if context_status != "ok": reasons.append(context_status)
    if join_success_rate != 1.0: reasons.append("join_incomplete")
    return {"analysis_available": not reasons, "ai_button_enabled": not reasons and api_key_available,
            "reasons": list(dict.fromkeys(reasons)), "api_key_available": api_key_available}


def make_analysis_cache_key(region: str, week_start: Any, file_fingerprint: str,
                            context_hash: str, model: str) -> str:
    return "eprx_ai:" + ":".join((region, pd.Timestamp(week_start).date().isoformat(),
                                   file_fingerprint, context_hash, model))


def make_readiness_cache_key(eprx_df: pd.DataFrame, region: str, week_start: Any,
                             grid_fingerprint: str) -> str:
    start = pd.Timestamp(week_start).normalize(); end = start + pd.Timedelta(days=7)
    areas = eprx_df.get("area", pd.Series(index=eprx_df.index, dtype=str))
    selected = eprx_df.loc[areas.eq(region)].copy()
    if "delivery_date" in selected:
        dates = pd.to_datetime(selected["delivery_date"], errors="coerce").dt.normalize()
        selected = selected.loc[dates.between(start, end, inclusive="left")]
    columns = [column for column in ("delivery_date", "period_no", "period_start", "procurement_volume")
               if column in selected]
    hashed = pd.util.hash_pandas_object(selected[columns], index=False).to_numpy().tobytes() if columns else b""
    eprx_fingerprint = hashlib.sha256(hashed).hexdigest()
    return "eprx_ai_ready:" + ":".join((region, start.date().isoformat(), eprx_fingerprint,
                                          grid_fingerprint))


def make_fast_context_cache_key(eprx_df: pd.DataFrame, region: str, week_start: Any,
                                grid_fingerprint: str) -> str:
    areas = eprx_df.get("area", pd.Series(index=eprx_df.index, dtype=str))
    selected = eprx_df.loc[areas.eq(region)]
    columns = [column for column in ("delivery_date", "period_no", "period_start",
               "procurement_volume", "source_file") if column in selected]
    hashed = pd.util.hash_pandas_object(selected[columns], index=False).to_numpy().tobytes() if columns else b""
    eprx_fingerprint = hashlib.sha256(hashed).hexdigest()
    return "eprx_ai_fast:" + ":".join((region, pd.Timestamp(week_start).date().isoformat(),
        eprx_fingerprint, grid_fingerprint, FAST_CONTEXT_VERSION))


def make_fast_history_cache_key(fast_context_key: str) -> str:
    parts = fast_context_key.split(":")
    return "eprx_ai_history:" + ":".join((parts[1], parts[3], parts[4], parts[5]))


def get_or_build_fast_local_context(session_state: MutableMapping[str, Any], context_key: str,
                                    region: str, week_start: Any,
                                    loader: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    history_key = make_fast_history_cache_key(context_key)
    def build() -> dict[str, Any]:
        history = session_state.get(history_key)
        if history is not None:
            local = dict(history)
            local["analysis_context"] = build_eprx_fast_context(
                local["feature_history"], region, week_start)
            return local
        local = loader()
        if local.get("status") == "ok" and "feature_history" in local:
            session_state[history_key] = {key: value for key, value in local.items()
                                          if key != "analysis_context"}
        return local
    return get_or_build_analysis_context(session_state, context_key, build)


def _cached_analysis_result(session_state: MutableMapping[str, Any], region: str,
                            week_start: Any) -> dict[str, Any] | None:
    date = pd.Timestamp(week_start).date().isoformat()
    prefixes = (f"eprx_ai_display:{PRESENTATION_VERSION}:{region}:{date}:",
                f"eprx_ai:{region}:{date}:")
    for key in reversed(list(session_state)):
        if str(key).startswith(prefixes) and isinstance(session_state[key], dict):
            result = session_state[key]
            presentation = result.get("presentation")
            if presentation is None or (isinstance(presentation, dict)
                    and presentation.get("presentation_version") == PRESENTATION_VERSION):
                return result
    return None


def get_or_build_analysis_context(session_state: MutableMapping[str, Any], readiness_key: str,
                                  builder: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    key = f"eprx_ai_context:{STATISTICS_ALGORITHM_VERSION}:{readiness_key}"
    if key in session_state:
        return session_state[key], True
    context = builder()
    session_state[key] = context
    return context, False


def run_ai_analysis_action(*, context: dict[str, Any], region: str, week_start: Any,
                           file_fingerprint: str, model: str,
                           session_state: MutableMapping[str, Any], clicked: bool,
                           regenerate: bool = False,
                           generator: Callable[..., dict[str, Any]] = generate_eprx_ai_analysis) -> dict[str, Any] | None:
    key = make_analysis_cache_key(region, week_start, file_fingerprint,
                                  calculate_eprx_context_hash(context), model)
    if key in session_state and not regenerate:
        cached = session_state[key]
        presentation = cached.get("presentation") if isinstance(cached, dict) else None
        if not isinstance(presentation, dict) or presentation.get("presentation_version") != PRESENTATION_VERSION:
            cached = {**cached, "presentation": build_eprx_readable_presentation(context)}
            session_state[key] = cached
        return cached
    if not clicked: return None
    result = generator(context, model=model)
    if result.get("status") == "ok":
        result = {**result, "presentation": build_eprx_readable_presentation(context)}
    session_state[key] = result
    return result


def render_eprx_ai_result(target, result: dict[str, Any]) -> None:
    """Render the structured AI response used by the live Streamlit section."""
    presentation = result.get("presentation")
    if result.get("status") == "ok" and isinstance(presentation, dict):
        _render_analysis_basis(target)
        _render_summary_cards(target, presentation.get("summary_cards", []))
        target.markdown("### 잔여수요 변동의 배경")
        target.caption("잔여수요는 전력수요와 재생에너지 출력의 차이로 만들어집니다. 아래 표는 잔여수요 변동과 그 배경이 되는 수요·재생에너지 변동 수준을 보여줍니다.")
        target.table(pd.DataFrame(presentation.get("comparison_table", [])))
        _render_intraday_tendency(target, presentation.get("intraday_tendency", {}))
        target.markdown("### 이번 주 해석")
        for value in presentation.get("interpretation", []): target.markdown(f"- {value}")
        with target.expander("수요·재생에너지 상세", expanded=False):
            target.table(pd.DataFrame(presentation.get("detail_table", [])))
            _render_level_profile(target, presentation.get("demand_intraday_profile", []))
            _render_demand_guidance(target, presentation)
        with target.expander("현재 수급실적과 모집량의 시간대 관계", expanded=False):
            target.caption("아래 상관계수 분석은 위의 전주 대비 방향 비교와 별개의 분석입니다. 현재 주의 수급실적과 모집량이 시간대별로 함께 움직였는지를 보는 참고 분석이며, 모집량의 공식 산정 원인을 의미하지 않습니다.")
            _render_relationship_table(target, presentation.get("relationship_table", []))
        with target.expander("모집량 상세", expanded=False):
            target.table(pd.DataFrame(presentation.get("procurement_table", [])))
        source = presentation.get("grid_source_label", "지역 계통운영기관")
        target.caption(f"계통실적: {source} 공개 30분 수급실적 · 잔여수요 = 전력수요 - 태양광 - 풍력")
        return
    if result.get("status") == "ok" and "procurement_patterns" in result:
        target.markdown("#### AI 주간 모집량 분석")
        target.write(result["summary"])
        for title, field in (("모집량 패턴", "procurement_patterns"),
                             ("계통 변수와의 연관성", "associations"), ("주의점", "cautions")):
            values = result.get(field, [])
            if values:
                target.markdown(f"**{title}**")
                _render_values(target, values)
        return
    if result.get("status") != "ok":
        target.warning(result.get("message", "AI 분석 결과를 사용할 수 없습니다."))
        if result.get("fallback"):
            with target.expander("Python FAST 요약", expanded=False):
                target.json(result["fallback"])
        return
    target.markdown(f"#### {result['headline']}")
    target.write(result["summary"])
    sections = (("모집량 패턴", "confirmed_findings"), ("이번 주 핵심", "statistical_interpretation"),
                ("계통 변수와의 연관성", "association_candidates"),
                ("통계적으로 주의할 점", "counter_evidence"),
                ("데이터 상태", "data_quality_notes"), ("분석 한계", "limitations"))
    for title, field in sections:
        values = result.get(field, [])
        if values:
            target.markdown(f"**{title}**")
            _render_values(target, values)
    if result.get("profile_warning"): target.warning(result["profile_warning"])
    target.write(result.get("conclusion", "")); target.caption(result.get("disclaimer", ""))


def render_eprx_ai_analysis_section(target, eprx_df: pd.DataFrame, region: str, week_start: Any) -> None:
    """Render without any network activity until an enabled button is clicked."""
    if region not in {"Tokyo", "Chubu"}:
        return
    import streamlit as st
    section_started = time.perf_counter()
    timing_key = f"eprx_ai_timing:{region}:{pd.Timestamp(week_start).date()}"
    timings: dict[str, Any] = dict(st.session_state.get(timing_key, {}))
    target.divider(); target.subheader("AI 모집량 분석")
    grid_fingerprint = local_grid_week_fingerprint(region, week_start)["fingerprint"]
    full_grid_fingerprint = local_grid_file_fingerprint(region)["fingerprint"]
    readiness_key = make_readiness_cache_key(eprx_df, region, week_start, grid_fingerprint)
    fast_context_key = make_fast_context_cache_key(eprx_df, region, week_start, full_grid_fingerprint)
    readiness_started = time.perf_counter()
    if readiness_key not in st.session_state:
        st.session_state[readiness_key] = check_eprx_ai_readiness(eprx_df, region, week_start)
    readiness = st.session_state[readiness_key]
    timings["readiness_seconds"] = time.perf_counter() - readiness_started
    target.caption(f"선택 지역: {region} · 선택 주차: {pd.Timestamp(week_start):%Y-%m-%d}")
    rate = float(readiness.get("join_rate", 0.0))
    if readiness.get("latest_source_date"):
        target.caption(f"계통자료 출처: {'도쿄전력 PG' if region == 'Tokyo' else '중부전력 PG'} · 최신일: {readiness['latest_source_date']}")
    target.caption(f"선택 주차 결합 성공률: {rate:.1%}")
    api_key, model = resolve_openai_settings()
    state = evaluate_eprx_ai_ui_state(market="EPRX", region=region, week_start=week_start,
        context_status=readiness["status"], complete_week=bool(readiness.get("complete_week")),
        market_regimes=["modern_30minute"],
        join_success_rate=rate, api_key_available=bool(api_key))
    if readiness["status"] != "ok":
        target.info(readiness.get("message", "분석 가능한 완전 주차가 아닙니다."))
    if not api_key: target.info("OpenAI API 키가 설정되어 있지 않습니다. Python 통계 요약만 표시합니다.")
    clicked = target.button("AI 분석 생성", disabled=not state["ai_button_enabled"],
                            key=f"eprx_ai_generate_{region}_{pd.Timestamp(week_start).date()}")
    regenerate = target.button("다시 생성", disabled=not state["ai_button_enabled"],
                               key=f"eprx_ai_regenerate_{region}_{pd.Timestamp(week_start).date()}")
    result = _cached_analysis_result(st.session_state, region, week_start)
    if clicked or regenerate:
        with target.spinner("데이터 요약을 준비하고 있습니다..."):
            context_started = time.perf_counter()
            local, _ = get_or_build_fast_local_context(st.session_state, fast_context_key,
                region, week_start, lambda: load_local_eprx_grid_context(eprx_df, region, week_start))
            timings["context_seconds"] = time.perf_counter() - context_started
            if local["status"] != "ok":
                target.warning(local.get("message", "분석 context를 생성하지 못했습니다."))
                return
            context = local["analysis_context"]
        with target.spinner("AI가 주간 데이터를 해석하고 있습니다..."):
            generation_started = time.perf_counter()
            result = run_ai_analysis_action(context=context, region=region, week_start=week_start,
                file_fingerprint=local["file_fingerprint"], model=model, session_state=st.session_state,
                clicked=True, regenerate=regenerate)
            timings["generation_seconds"] = time.perf_counter() - generation_started
            cache_started = time.perf_counter()
            display_key = (f"eprx_ai_display:{PRESENTATION_VERSION}:{region}:{pd.Timestamp(week_start).date()}:"
                           f"{readiness['file_fingerprint']}:{model}")
            st.session_state[display_key] = result
            timings["cache_seconds"] = time.perf_counter() - cache_started
    with target.expander("상세 통계 분석", expanded=False):
        detailed_clicked = target.button("상세 통계 계산",
            key=f"eprx_ai_detailed_{region}_{pd.Timestamp(week_start).date()}")
        if detailed_clicked:
            with target.spinner("bootstrap·회귀·시간대 조정 통계를 계산하고 있습니다..."):
                local, _ = get_or_build_fast_local_context(st.session_state, fast_context_key,
                    region, week_start, lambda: load_local_eprx_grid_context(eprx_df, region, week_start))
                detailed, _ = get_or_build_analysis_context(st.session_state,
                    f"detailed:{fast_context_key}", lambda: build_detailed_context(local, region, week_start))
                target.json(build_eprx_statistical_fallback(detailed))
    if result:
        render_started = time.perf_counter()
        render_eprx_ai_result(target, result)
        timings["render_seconds"] = time.perf_counter() - render_started
        response_diagnostics = result.get("response_diagnostics", {})
        request_diagnostics = result.get("request_diagnostics", {})
        timings.update({
            "payload_seconds": request_diagnostics.get("payload_elapsed_seconds"),
            "api_seconds": response_diagnostics.get("api_elapsed_seconds"),
            "parse_seconds": response_diagnostics.get("parse_elapsed_seconds"),
            "validation_seconds": response_diagnostics.get("validation_elapsed_seconds"),
            "api_calls": request_diagnostics.get("api_calls", 0),
            "input_tokens": response_diagnostics.get("input_tokens"),
            "output_tokens": response_diagnostics.get("output_tokens"),
            "reasoning_tokens": response_diagnostics.get("reasoning_tokens"),
            "model": result.get("model") or request_diagnostics.get("model"),
        })
        timings["total_seconds"] = time.perf_counter() - section_started
        st.session_state[timing_key] = timings
    saved_timings = st.session_state.get(timing_key)
    if saved_timings:
        with target.expander("개발용 성능 진단", expanded=False):
            target.json(saved_timings)
    target.caption("본 분석은 공개된 30분 실적자료를 이용한 사후 통계분석입니다. 수요는 공개자료 기반 실적치이며 의사결정 당시 예측자료와 다를 수 있습니다. 평상시분·비상시분·수의계약량·자연체여력은 분리되어 있지 않습니다. 통계적 연관성은 인과관계를 의미하지 않으며 모집량 예측 결과가 아닙니다.")
