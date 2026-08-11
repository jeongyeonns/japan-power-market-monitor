"""Streamlit presentation helpers for the opt-in EPRX AI analysis."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, MutableMapping

import pandas as pd

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


def build_eprx_readable_presentation(context: dict[str, Any]) -> dict[str, Any]:
    selected = context.get("selected_week", {}); procurement = selected.get("procurement", {})
    comparison = selected.get("procurement_change", {}) or {}
    notable = selected.get("notable_time_blocks", {}) or {}
    high = (notable.get("highest") or [{}])[0]; low = (notable.get("lowest") or [{}])[0]
    changes = selected.get("driver_changes", {}) or {}
    demand_change = changes.get("mean_demand_mw", {}) or {}
    residual_change = changes.get("mean_residual_demand_proxy_mw", {}) or {}
    relations = context.get("selected_week_correlations", {}) or context.get("time_adjusted_correlations", {})
    demand = relations.get("demand_mw", {}); residual = relations.get("residual_demand_proxy_mw", {})
    mean = procurement.get("mean"); previous = comparison.get("previous"); change = comparison.get("change")
    change_pct = comparison.get("change_pct")
    direction = "증가" if (_finite_number(change) or 0) >= 0 else "감소"
    overview = [
        f"이번 주 평균 모집량은 {format_mw(mean)}로 전주보다 {format_mw(abs(_finite_number(change) or 0))}({format_percent(abs(_finite_number(change_pct) or 0))}) {direction}했습니다.",
        f"모집량은 {_clock_label(high.get('time_block'))}에 가장 높고 {_clock_label(low.get('time_block'))}에 가장 낮은 시간대별 패턴을 보였습니다.",
        "전력수요와 잔여수요가 높은 시간대일수록 모집량도 대체로 함께 높아지는 경향을 비교했습니다.",
    ]
    procurement_lines = [
        f"평균 모집량은 {format_mw(mean)}로, 전주 {format_mw(previous)}보다 {format_mw(abs(_finite_number(change) or 0))} {direction}했습니다.",
        f"이번 주 모집량은 {format_mw(procurement.get('minimum'))}~{format_mw(procurement.get('maximum'))} 범위에서 움직였습니다.",
        f"{_clock_label(high.get('time_block'))} 평균 {format_mw(high.get('average_procurement_mw'))}로 가장 높았고, {_clock_label(low.get('time_block'))} 평균 {format_mw(low.get('average_procurement_mw'))}로 가장 낮았습니다.",
    ]
    demand_sentence = _association_sentence("전력수요", demand)
    residual_sentence = _association_sentence(
        "잔여수요 추정치(전력수요에서 태양광·풍력 발전량을 뺀 값)", residual)
    demand_r = _finite_number(demand.get("pearson")); residual_r = _finite_number(residual.get("pearson"))
    comparison_sentence = "전력수요와 잔여수요 중 어느 하나가 모집량을 뚜렷하게 더 잘 설명한다고 보기는 어렵습니다."
    if demand_r is not None and residual_r is not None and abs(demand_r - residual_r) >= 0.10:
        stronger = "전력수요" if abs(demand_r) > abs(residual_r) else "잔여수요"
        comparison_sentence = f"이번 주 자료에서는 {stronger}와 모집량의 관계가 상대적으로 더 크게 나타났습니다."
    demand_context = (f"주간 평균 전력수요는 {format_mw(demand_change.get('current'))}, "
                      f"잔여수요는 {format_mw(residual_change.get('current'))}였습니다. ")
    interpretation = [
        "이번 주에는 모집량이 전주보다 전반적으로 높았고, 수요와 잔여수요가 높은 시간대에 모집량도 함께 높아지는 모습이 나타났습니다.",
        "다만 시간대별 모집 패턴이 반복될 수 있으므로 수요가 모집량 변화를 직접 만들었다기보다 높은 시간대가 상당 부분 겹쳤다고 보는 것이 적절합니다.",
    ]
    return {"overview": overview, "procurement_changes": procurement_lines,
        "relationships": [demand_context + demand_sentence, residual_sentence, comparison_sentence],
        "interpretation": interpretation,
        "notes": ["이번 결과는 공개된 30분 단위 실제 실적을 비교한 결과이며 인과관계를 의미하지 않습니다.",
                  "잔여수요는 공식 발표값이 아니라 전력수요에서 태양광·풍력 발전량을 제외해 계산한 추정치입니다."]}


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
    prefixes = (f"eprx_ai_display:{region}:{date}:", f"eprx_ai:{region}:{date}:")
    for key in reversed(list(session_state)):
        if str(key).startswith(prefixes) and isinstance(session_state[key], dict):
            return session_state[key]
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
    if key in session_state and not regenerate: return session_state[key]
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
        sections = (("한눈에 보기", "overview"), ("모집량 변화", "procurement_changes"),
                    ("수요·잔여수요와의 관계", "relationships"),
                    ("이렇게 해석하면 됩니다", "interpretation"), ("참고", "notes"))
        for title, field in sections:
            target.markdown(f"### {title}")
            values = presentation.get(field, [])
            if field in {"overview", "interpretation"}:
                for value in values: target.write(value)
            else:
                for value in values: target.markdown(f"- {value}")
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
            display_key = (f"eprx_ai_display:{region}:{pd.Timestamp(week_start).date()}:"
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
