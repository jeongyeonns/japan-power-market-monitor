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


def _render_result(target, result: dict[str, Any]) -> None:
    if result.get("status") != "ok":
        target.warning(result.get("message", "AI 분석 결과를 사용할 수 없습니다.")); return
    target.markdown(f"#### {result['headline']}")
    target.write(result["summary"])
    sections = (("확인된 결과", "confirmed_findings"), ("통계 해석", "statistical_interpretation"),
                ("관계 후보", "association_candidates"), ("반대·제한 근거", "counter_evidence"),
                ("데이터 품질", "data_quality_notes"), ("분석 한계", "limitations"))
    for title, field in sections:
        values = result.get(field, [])
        if values:
            target.markdown(f"**{title}**")
            for value in values:
                if isinstance(value, dict) and "interpretation" in value:
                    target.markdown(f"- {value['display_name']}: {value['value']:,.4g} {value['unit']} — {value['interpretation']}")
                else:
                    target.markdown(f"- {value}")
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
    if result: _render_result(target, result)
    target.caption("본 분석은 공개된 30분 실적자료를 이용한 사후 통계분석입니다. 수요는 공개자료 기반 실적치이며 의사결정 당시 예측자료와 다를 수 있습니다. 평상시분·비상시분·수의계약량·자연체여력은 분리되어 있지 않습니다. 통계적 연관성은 인과관계를 의미하지 않으며 모집량 예측 결과가 아닙니다.")
