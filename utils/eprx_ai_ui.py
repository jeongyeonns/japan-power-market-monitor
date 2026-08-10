"""Streamlit presentation helpers for the opt-in EPRX AI analysis."""

from __future__ import annotations

from typing import Any, Callable, MutableMapping

import pandas as pd

from utils.eprx_ai_analysis import (
    build_eprx_statistical_fallback,
    calculate_eprx_context_hash,
    generate_eprx_ai_analysis,
    resolve_openai_settings,
)
from utils.eprx_ai_pipeline import load_local_eprx_grid_context


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


def _evidence_display_name(evidence: dict[str, Any]) -> str:
    path = str(evidence.get("metric_path", ""))
    for metric, display in sorted(DISPLAY_NAMES.items(), key=lambda item: -len(item[0])):
        if metric in path:
            return display
    return str(evidence.get("display_name") or "분석 지표")


def _format_evidence(evidence: dict[str, Any]) -> str:
    name = _evidence_display_name(evidence)
    value = float(evidence["value"])
    path = str(evidence.get("metric_path", "")).lower()
    unit = str(evidence.get("unit", "")).strip()
    if "r_squared" in path:
        metric = f"R² = {value:.2f}"
    elif "spearman" in path:
        metric = f"Spearman ρ = {value:+.2f}"
    elif any(token in path for token in ("pearson", "correlation", "coefficient")) or unit in {
        "coefficient", "상관계수", "unitless"
    }:
        metric = f"Pearson r = {value:+.2f}"
    elif "percentile" in path:
        metric = f"백분위 {value:.1f}"
    elif unit == "MW" or path.endswith("_mw"):
        metric = f"{value:,.1f} MW"
    elif unit in {"%", "percentage"}:
        metric = f"{value:.1f}%"
    elif unit in {"count", "개", "행", "회"}:
        metric = f"{value:,.0f}건"
    else:
        metric = f"{value:,.2f}"
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


def run_ai_analysis_action(*, context: dict[str, Any], region: str, week_start: Any,
                           file_fingerprint: str, model: str,
                           session_state: MutableMapping[str, Any], clicked: bool,
                           regenerate: bool = False,
                           generator: Callable[..., dict[str, Any]] = generate_eprx_ai_analysis) -> dict[str, Any] | None:
    key = make_analysis_cache_key(region, week_start, file_fingerprint,
                                  calculate_eprx_context_hash(context), model)
    if key in session_state and not regenerate: return session_state[key]
    if not clicked: return None
    result = generator(context, model=model)
    session_state[key] = result
    return result


def render_eprx_ai_result(target, result: dict[str, Any]) -> None:
    """Render the structured AI response used by the live Streamlit section."""
    if result.get("status") != "ok":
        target.warning(result.get("message", "AI 분석 결과를 사용할 수 없습니다.")); return
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
    target.divider(); target.subheader("AI 모집량 분석")
    local = load_local_eprx_grid_context(eprx_df, region, week_start)
    target.caption(f"선택 지역: {region} · 선택 주차: {pd.Timestamp(week_start):%Y-%m-%d}")
    if local["status"] != "ok":
        target.info(local["message"])
        target.button("AI 분석 생성", disabled=True, key=f"eprx_ai_disabled_{region}_{week_start}")
        return
    context = local["analysis_context"]
    join = local["join_diagnostics"]
    rate = join["matched_rows"] / max(join["eprx_rows"], join["tepco_rows"], 1)
    target.caption(f"계통자료 출처: {'도쿄전력 PG' if region == 'Tokyo' else '중부전력 PG'} · 최신일: {local['latest_source_date']}")
    target.caption(f"분석 기간: {context.get('analysis_period', {}).get('start')} ~ {context.get('analysis_period', {}).get('end')} · 결합 성공률: {rate:.1%}")
    fallback = build_eprx_statistical_fallback(context)
    with target.expander("Python 통계 요약", expanded=False): target.json(fallback)
    api_key, model = resolve_openai_settings()
    state = evaluate_eprx_ai_ui_state(market="EPRX", region=region, week_start=week_start,
        context_status=local["status"], complete_week=join["matched_rows"] == 336,
        market_regimes=context.get("selected_week", {}).get("week", {}).get("market_regimes", ["48_block"]),
        join_success_rate=rate, api_key_available=bool(api_key))
    if not api_key: target.info("OpenAI API 키가 설정되어 있지 않습니다. Python 통계 요약만 표시합니다.")
    clicked = target.button("AI 분석 생성", disabled=not state["ai_button_enabled"],
                            key=f"eprx_ai_generate_{region}_{pd.Timestamp(week_start).date()}")
    regenerate = target.button("다시 생성", disabled=not state["ai_button_enabled"],
                               key=f"eprx_ai_regenerate_{region}_{pd.Timestamp(week_start).date()}")
    import streamlit as st
    if clicked or regenerate:
        with target.spinner("요약 통계를 바탕으로 분석을 생성하고 있습니다..."):
            result = run_ai_analysis_action(context=context, region=region, week_start=week_start,
                file_fingerprint=local["file_fingerprint"], model=model, session_state=st.session_state,
                clicked=True, regenerate=regenerate)
    else:
        result = run_ai_analysis_action(context=context, region=region, week_start=week_start,
            file_fingerprint=local["file_fingerprint"], model=model, session_state=st.session_state, clicked=False)
    if result:
        render_eprx_ai_result(target, result)
    target.caption("본 분석은 공개된 30분 실적자료를 이용한 사후 통계분석입니다. 수요는 공개자료 기반 실적치이며 의사결정 당시 예측자료와 다를 수 있습니다. 평상시분·비상시분·수의계약량·자연체여력은 분리되어 있지 않습니다. 통계적 연관성은 인과관계를 의미하지 않으며 모집량 예측 결과가 아닙니다.")
