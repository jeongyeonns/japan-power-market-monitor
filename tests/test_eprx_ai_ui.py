from contextlib import nullcontext

import pandas as pd
import pytest

from utils import eprx_ai_ui
from utils.eprx_ai_ui import evaluate_eprx_ai_ui_state, run_ai_analysis_action


def state(**changes):
    values = {"market": "EPRX", "region": "Tokyo", "week_start": "2026-03-14",
        "context_status": "ok", "complete_week": True, "market_regimes": ["modern"],
        "join_success_rate": 1.0, "api_key_available": True}
    values.update(changes); return evaluate_eprx_ai_ui_state(**values)


def test_supported_tokyo_and_chubu():
    assert state()["ai_button_enabled"]
    assert state(region="Chubu")["ai_button_enabled"]


def test_unsupported_states_are_disabled():
    assert not state(market="JEPX")["analysis_available"]
    assert not state(region="Kansai")["analysis_available"]
    assert not state(week_start="2026-03-07")["analysis_available"]
    assert not state(market_regimes=["old", "modern"])["analysis_available"]
    assert not state(context_status="source_data_missing")["analysis_available"]
    assert not state(join_success_rate=0.99)["analysis_available"]
    assert not state(api_key_available=False)["ai_button_enabled"]


def test_button_click_cache_rerun_context_change_and_regenerate():
    calls = []
    generator = lambda context, model: calls.append(context["value"]) or {"status": "ok", "value": context["value"]}
    session = {}; base = {"region": "Tokyo", "selected_week": {"week": {"start": "2026-03-14"}},
        "time_adjusted_correlations": {}, "limitations": [], "data_quality": {}, "value": 1}
    kwargs = dict(context=base, region="Tokyo", week_start=pd.Timestamp("2026-03-14"),
        file_fingerprint="a", model="model", session_state=session, generator=generator)
    assert run_ai_analysis_action(clicked=False, **kwargs) is None and calls == []
    assert run_ai_analysis_action(clicked=True, **kwargs)["value"] == 1 and calls == [1]
    run_ai_analysis_action(clicked=False, **kwargs); assert calls == [1]
    changed = {**base, "value": 2}
    run_ai_analysis_action(clicked=True, **{**kwargs, "context": changed}); assert calls == [1, 2]
    run_ai_analysis_action(clicked=True, regenerate=True, **kwargs); assert calls == [1, 2, 1]


def test_deterministic_context_cache_reuses_and_invalidates_by_readiness_key():
    calls = []
    session = {}
    builder = lambda: calls.append(len(calls) + 1) or {"build": len(calls)}
    first, first_hit = eprx_ai_ui.get_or_build_analysis_context(session, "source-a", builder)
    second, second_hit = eprx_ai_ui.get_or_build_analysis_context(session, "source-a", builder)
    changed, changed_hit = eprx_ai_ui.get_or_build_analysis_context(session, "source-b", builder)
    assert (first, first_hit) == ({"build": 1}, False)
    assert (second, second_hit) == ({"build": 1}, True)
    assert (changed, changed_hit) == ({"build": 2}, False)
    assert calls == [1, 2]


def test_fast_history_cache_reuses_parsed_features_across_weeks(monkeypatch):
    frame = pd.DataFrame({"area": ["Tokyo"], "delivery_date": ["2026-07-20"],
                          "period_no": [1], "period_start": ["00:00"],
                          "procurement_volume": [1.0]})
    first_key = eprx_ai_ui.make_fast_context_cache_key(frame, "Tokyo", "2026-07-20", "grid")
    second_key = eprx_ai_ui.make_fast_context_cache_key(frame, "Tokyo", "2026-07-27", "grid")
    assert eprx_ai_ui.make_fast_history_cache_key(first_key) == eprx_ai_ui.make_fast_history_cache_key(second_key)
    session = {}; loads = []
    local = {"status": "ok", "feature_history": pd.DataFrame({"x": [1]}),
             "analysis_context": {"week": "first"}}
    first, _ = eprx_ai_ui.get_or_build_fast_local_context(
        session, first_key, "Tokyo", "2026-07-20", lambda: loads.append(1) or local)
    monkeypatch.setattr(eprx_ai_ui, "build_eprx_fast_context",
                        lambda *_args: {"week": "second"})
    second, _ = eprx_ai_ui.get_or_build_fast_local_context(
        session, second_key, "Tokyo", "2026-07-27", lambda: loads.append(2) or local)
    assert first["analysis_context"]["week"] == "first"
    assert second["analysis_context"]["week"] == "second"
    assert loads == [1]


class RenderTarget:
    def __init__(self, clicked_labels=()):
        self.events = []
        self.buttons = []
        self.expanders = []
        self.markdowns = []
        self.writes = []
        self.json_values = []
        self.metrics = []
        self.tables = []
        self.infos = []
        self.warnings = []
        self.captions = []
        self.line_charts = []
        self.plotly_charts = []
        self.clicked_labels = set(clicked_labels)

    def button(self, label, **kwargs):
        self.buttons.append((label, kwargs))
        return label in self.clicked_labels

    def expander(self, label, **kwargs):
        self.expanders.append((label, kwargs))
        return nullcontext()

    def spinner(self, _label):
        return nullcontext()

    def markdown(self, value, **kwargs):
        self.events.append(("markdown", value))
        self.markdowns.append(value)

    def write(self, value, **kwargs):
        self.writes.append(value)

    def json(self, value, **kwargs):
        self.json_values.append(value)

    def columns(self, count):
        return [self] * count

    def metric(self, **kwargs):
        self.events.append(("metric", kwargs))
        self.metrics.append(kwargs)

    def table(self, value, **kwargs):
        self.events.append(("table", value))
        self.tables.append(value)

    def info(self, value, **kwargs):
        self.events.append(("info", value))
        self.infos.append(value)

    def warning(self, value, **kwargs):
        self.warnings.append(value)

    def caption(self, value, **kwargs):
        self.captions.append(value)

    def line_chart(self, value, **kwargs):
        self.line_charts.append((value, kwargs))

    def plotly_chart(self, value, **kwargs):
        self.plotly_charts.append((value, kwargs))

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _render(monkeypatch, *, api_key="key", local_status="ok", cached=False,
            cached_result=None, clicked_labels=()):
    context = {"region": "Tokyo", "selected_week": {"week": {"market_regimes": ["modern"]}}}
    local = {"status": local_status, "message": "missing", "region": "Tokyo"}
    if local_status == "ok":
        local.update({"analysis_context": context, "join_diagnostics": {
            "matched_rows": 336, "eprx_rows": 336, "tepco_rows": 336},
            "latest_source_date": "2026-08-02", "file_fingerprint": "files"})
    readiness = {"status": local_status, "ready": local_status == "ok",
        "complete_week": local_status == "ok", "join_rate": 1.0 if local_status == "ok" else 0.0,
        "latest_source_date": "2026-08-02", "file_fingerprint": "files", "message": "missing"}
    heavy_calls = []
    monkeypatch.setattr(eprx_ai_ui, "local_grid_week_fingerprint", lambda *args: {"fingerprint": "grid"})
    monkeypatch.setattr(eprx_ai_ui, "check_eprx_ai_readiness", lambda *args: readiness)
    monkeypatch.setattr(eprx_ai_ui, "load_local_eprx_grid_context",
                        lambda *args: heavy_calls.append(args) or local)
    monkeypatch.setattr(eprx_ai_ui, "build_eprx_statistical_fallback", lambda _context: {})
    monkeypatch.setattr(eprx_ai_ui, "resolve_openai_settings", lambda: (api_key, "gpt-5-mini"))
    session = {}
    if cached and local_status == "ok":
        key = eprx_ai_ui.make_analysis_cache_key(
            "Tokyo", "2026-07-27", "files",
            eprx_ai_ui.calculate_eprx_context_hash(context), "gpt-5-mini")
        session[key] = cached_result or {"status": "ok", "headline": "cached", "summary": "summary"}
    monkeypatch.setattr("streamlit.session_state", session)
    target = RenderTarget(clicked_labels)
    target.heavy_calls = heavy_calls
    eprx_ai_ui.render_eprx_ai_analysis_section(
        target, pd.DataFrame(), "Tokyo", pd.Timestamp("2026-07-27"))
    return target


def test_generate_button_visible_for_ready_and_cached_states(monkeypatch):
    for cached in (False, True):
        target = _render(monkeypatch, cached=cached)
        generate = next(item for item in target.buttons if item[0] == "AI 분석 생성")
        assert generate[1]["disabled"] is False
        expected = [("상세 통계 분석", {"expanded": False})]
        if cached:
            expected.append(("개발용 성능 진단", {"expanded": False}))
        assert target.expanders == expected
        assert target.heavy_calls == []


def test_runtime_cache_rejects_legacy_normalized_presentation():
    legacy = {"status": "ok", "presentation": {"intraday_tendency": {
        "profile": [{"procurement_index": 100, "residual_volatility_index": 100}]}}}
    current = {"status": "ok", "presentation": {
        "presentation_version": eprx_ai_ui.PRESENTATION_VERSION}}
    session = {
        "eprx_ai_display:Tokyo:2026-07-20:old": legacy,
        f"eprx_ai_display:{eprx_ai_ui.PRESENTATION_VERSION}:Tokyo:2026-07-20:new": current,
    }
    assert eprx_ai_ui._cached_analysis_result(session, "Tokyo", "2026-07-20") is current
    session.pop(next(key for key in session if eprx_ai_ui.PRESENTATION_VERSION in key))
    assert eprx_ai_ui._cached_analysis_result(session, "Tokyo", "2026-07-20") is None


def test_cached_ai_result_upgrades_runtime_presentation_without_api(monkeypatch):
    context = {"region": "Tokyo", "selected_week": {"procurement": {}, "driver_changes": {}}}
    key = eprx_ai_ui.make_analysis_cache_key(
        "Tokyo", "2026-07-20", "files",
        eprx_ai_ui.calculate_eprx_context_hash(context), "model")
    session = {key: {"status": "ok", "presentation": {
        "intraday_tendency": {"profile": [{"procurement_index": 100}]}}}}
    def forbidden(*_args, **_kwargs):
        raise AssertionError("cached presentation migration must not call the API")
    result = eprx_ai_ui.run_ai_analysis_action(
        context=context, region="Tokyo", week_start="2026-07-20",
        file_fingerprint="files", model="model", session_state=session,
        clicked=True, generator=forbidden)
    assert result["presentation"]["presentation_version"] == eprx_ai_ui.PRESENTATION_VERSION


def test_generate_button_visible_but_disabled_without_key_or_data(monkeypatch):
    no_key = _render(monkeypatch, api_key=None)
    assert next(item for item in no_key.buttons if item[0] == "AI 분석 생성")[1]["disabled"] is True
    no_data = _render(monkeypatch, local_status="source_data_missing")
    assert next(item for item in no_data.buttons if item[0] == "AI 분석 생성")[1]["disabled"] is True


def test_result_renderer_treats_interpretation_string_as_paragraph_and_formats_evidence():
    target = RenderTarget()
    eprx_ai_ui.render_eprx_ai_result(target, {
        "status": "ok", "headline": "주간 메모", "summary": "요약",
        "confirmed_findings": [
            {"metric_path": "analysis.procurement_change.change", "display_name": "change",
             "value": 36953.02976190476, "unit": "MW", "interpretation": "전주보다 증가했습니다."},
            {"metric_path": "analysis.historical.percentile", "display_name": "percentile",
             "value": 54.645, "unit": "%", "interpretation": "역사적 중간권입니다."},
            {"metric_path": "analysis.renewable_share_pct", "display_name": "renewable_share_pct",
             "value": 54.645, "unit": "%", "interpretation": "주간 평균입니다."},
        ],
        "statistical_interpretation": "피어슨·스피어만 모두 같은 방향입니다.",
        "association_candidates": [
            {"metric_path": "analysis.time_adjusted_correlations.demand_mw.pearson",
             "display_name": "demand_mw", "value": 0.5686, "unit": "coefficient",
             "interpretation": "중간 수준의 양의 관계입니다."},
        ],
        "counter_evidence": [], "data_quality_notes": [],
        "limitations": ["This is retrospective statistical association analysis using public 30-minute actuals."],
        "profile_warning": "", "conclusion": "결론", "disclaimer": "주의",
    })
    assert "피어슨·스피어만 모두 같은 방향입니다." in target.writes
    assert not any(text == "- 피" for text in target.markdowns)
    rendered = "\n".join(target.markdowns)
    assert "36,953.0 MW" in rendered
    assert "백분위 54.6" in rendered
    assert "재생에너지 발전 비중 — 54.6%" in rendered
    assert "전력수요 — Pearson r = +0.57" in rendered
    assert "0.5686 coefficient" not in rendered
    assert "demand_mw" not in rendered
    assert "e+" not in rendered.lower()
    assert "This is retrospective" not in rendered


def test_fast_result_renderer_is_immediate_and_formats_association():
    target = RenderTarget()
    eprx_ai_ui.render_eprx_ai_result(target, {
        "status": "ok", "summary": "이번 주 요약", "procurement_patterns": ["패턴 하나"],
        "associations": [{"metric_path": "analysis.selected_associations.0.spearman",
            "display_name": "평균 수요", "value": 0.5686, "unit": "coefficient",
            "interpretation": "중간 수준의 양의 관계입니다."}],
        "cautions": ["인과관계를 의미하지 않습니다."],
    })
    rendered = "\n".join(target.markdowns + target.writes + target.infos + target.warnings
                           + target.captions + [str(item) for item in target.metrics]
                           + [table.to_string(index=False) for table in target.tables])
    assert "이번 주 요약" in rendered and "패턴 하나" in rendered
    assert "+0.57" in rendered and "인과관계" in rendered


def test_live_streamlit_runtime_path_renders_cached_string_and_numbers(monkeypatch):
    fixture = {
        "status": "ok", "headline": "주간 분석", "summary": "모집량 중심 요약",
        "confirmed_findings": [
            {"metric_path": "analysis.procurement.current", "display_name": "현재 조정력(1차 조정력)",
             "value": 572.425595, "unit": "MW", "interpretation": "주간 평균입니다."},
            {"metric_path": "analysis.driver_changes.mean_demand_mw.current", "display_name": "평균 수요",
             "value": 40020.0, "unit": "MW", "interpretation": "공개 실적 평균입니다."},
            {"metric_path": "analysis.driver_changes.mean_renewable_share_pct.current",
             "display_name": "평균 재생에너지 비율", "value": 9.0114, "unit": "%",
             "interpretation": "주간 평균 비율입니다."},
        ],
        "statistical_interpretation": "피어슨·스피어만 모두 여러 변수에서 같은 방향의 관계를 보였습니다.",
        "association_candidates": [
            {"metric_path": "analysis.time_adjusted_correlations.demand_mw.pearson",
             "display_name": "평균 수요", "value": 0.5686, "unit": "coefficient",
             "interpretation": "중간 수준의 양의 관계입니다."},
        ], "counter_evidence": [], "data_quality_notes": [],
        "limitations": [], "profile_warning": "", "conclusion": "", "disclaimer": "",
    }
    target = _render(monkeypatch, cached=True, cached_result=fixture)
    rendered = "\n".join(
        target.markdowns
        + target.writes
        + target.infos
        + target.warnings
        + target.captions
        + [str(item) for item in target.metrics]
        + [table.to_string(index=False) for table in target.tables]
    )
    assert "피어슨·스피어만 모두 여러 변수에서 같은 방향의 관계를 보였습니다." in target.writes
    assert not any(item in target.markdowns for item in ("- 피", "- 어", "- 슨"))
    assert "572.4 MW" in rendered
    assert "40,020.0 MW" in rendered
    assert "9.0%" in rendered
    assert "+0.57" in rendered
    assert "4.002e+04" not in rendered
    assert "9.011 %" not in rendered
    assert "coefficient" not in rendered


def test_metric_formatters_are_nan_safe_and_never_use_scientific_notation():
    assert eprx_ai_ui.format_mw(40020.123) == "40,020.1 MW"
    assert eprx_ai_ui.format_percent(9.011) == "9.0%"
    assert eprx_ai_ui.format_correlation(0.5686) == "+0.57"
    assert eprx_ai_ui.format_number(336, 0) == "336"
    for formatter in (eprx_ai_ui.format_mw, eprx_ai_ui.format_percent,
                      eprx_ai_ui.format_correlation, eprx_ai_ui.format_number):
        assert formatter(None) == "—"
        assert formatter(float("nan")) == "—"


def test_semantic_metric_type_beats_incorrect_coefficient_unit():
    procurement = {"metric_path": "analysis.procurement_summary.mean", "display_name": "주간 평균 모집량",
                   "value": 572.4255952380952, "unit": "coefficient", "interpretation": ""}
    rendered = eprx_ai_ui._format_evidence(procurement)
    assert "572.4 MW" in rendered
    assert "Pearson" not in rendered and "coefficient" not in rendered
    assert eprx_ai_ui.semantic_metric_type("analysis.selected_associations.0.pearson", "") == "correlation"
    assert eprx_ai_ui.semantic_metric_type("analysis.week_change_pct", "") == "percent"


def test_correlation_strength_uses_plain_korean_thresholds():
    assert eprx_ai_ui.correlation_strength(0.19) == "뚜렷한 관계 없음"
    assert eprx_ai_ui.correlation_strength(-0.20) == "약한 관계"
    assert eprx_ai_ui.correlation_strength(0.40) == "중간 수준의 관계"
    assert eprx_ai_ui.correlation_strength(-0.60) == "강한 관계"
    assert eprx_ai_ui.correlation_strength(0.80) == "매우 강한 관계"


@pytest.mark.parametrize(("procurement", "volatility", "expected"), [
    (13.5, 12.6, "same"), (-8.0, -12.0, "same"),
    (10.0, -8.0, "opposite"), (-10.0, 8.0, "opposite"),
    (0.5, 8.0, "undetermined"), (None, 8.0, "undetermined"),
])
def test_weekly_direction_classification(procurement, volatility, expected):
    assert eprx_ai_ui.classify_weekly_direction(procurement, volatility) == expected


@pytest.mark.parametrize(("direction", "strength", "expected"), [
    ("same", "moderate", "같은 방향 경향"),
    ("opposite", "moderate", "반대 방향 경향"),
    ("none", "none", "뚜렷한 시간대 경향이 확인되지 않았습니다"),
])
def test_intraday_tendency_wording(direction, strength, expected):
    result = eprx_ai_ui._build_intraday_tendency({
        "status": "available", "profile_spearman": 0.5 if direction == "same" else -0.5,
        "profile_direction": direction, "profile_strength": strength,
        "high_slot_overlap_count": 6, "high_slot_overlap_pct": 50, "profile": []})
    assert expected in result["headline"] + result["detail"]


def test_live_ai_section_renders_readable_market_presentation(monkeypatch):
    context = {"region": "Tokyo", "selected_week": {
        "week": {"start": "2026-07-20", "end": "2026-07-26"},
        "procurement": {"mean": 572.4255952380952, "minimum": 544, "maximum": 599},
        "procurement_change": {"previous": 504.42261904761904, "change": 68.00297619047615,
                               "change_pct": 13.4817},
        "notable_time_blocks": {
            "highest": [{"time_block": "08:30", "average_procurement_mw": 591.1428571428571}],
            "lowest": [{"time_block": "03:00", "average_procurement_mw": 548.2857142857143}]},
            "driver_changes": {
                "mean_demand_mw": {"current": 40022.9},
                "mean_residual_demand_proxy_mw": {"current": 35877.1},
                "mean_renewable_generation_mw": {"current": 4145.8},
                "mean_solar_mw": {"current": 3900.0},
                "mean_wind_mw": {"current": 245.8},
                "mean_renewable_share_pct": {"current": 9.0114},
                    "mean_abs_residual_demand_ramp_30m_mw": {"current": 772.1, "previous": 685.6,
                                                               "change_pct": 12.6169743291},
                "mean_abs_demand_ramp_30m_mw": {"current": 640.0, "previous": 600.0},
                "mean_abs_renewable_ramp_30m_mw": {"current": 310.0, "previous": 280.0},
                "p95_abs_residual_demand_ramp_30m_mw": {"current": 1240.0}},
        "demand_intraday_profile": [
                {"time_block": "00:00", "procurement_mw": 550.0, "demand_mw": 39000.0,
                 "residual_demand_mw": 37000.0, "renewable_generation_mw": 2000.0,
                 "abs_residual_demand_ramp_30m_mw": 700.0,
                 "observation_count": 7},
            {"time_block": "08:30", "procurement_mw": 590.0, "demand_mw": 41000.0,
                 "residual_demand_mw": 36000.0, "renewable_generation_mw": 5000.0,
                 "abs_residual_demand_ramp_30m_mw": 900.0,
             "observation_count": 7},
            ],
        "intraday_profile_summary": {"status": "available", "slot_count": 48,
            "profile_spearman": 0.52, "profile_direction": "same", "profile_strength": "moderate",
            "high_slot_overlap_count": 8, "high_slot_overlap_pct": 66.6667,
            "procurement_high_slots": [f"{hour:02d}:00" for hour in range(12)],
            "residual_vol_high_slots": [f"{hour:02d}:00" for hour in range(8)] + ["12:00", "13:00", "14:00", "15:00"],
            "profile": [
                {"time_block": "00:00", "procurement_mw": 550.0,
                 "abs_residual_demand_ramp_30m_mw": 700.0,
                 "procurement_index": 95.0, "residual_volatility_index": 90.0},
                {"time_block": "08:30", "procurement_mw": 590.0,
                 "abs_residual_demand_ramp_30m_mw": 900.0,
                 "procurement_index": 105.0, "residual_volatility_index": 110.0}]},
    }, "selected_week_correlations": {
        "demand_mw": {"pearson": 0.5775835870089938, "spearman": 0.46633704875086757},
        "residual_demand_proxy_mw": {"pearson": 0.5629575833976636,
                                      "spearman": 0.45282089618210863},
        "renewable_generation_mw": {"pearson": -0.3123, "spearman": -0.2876}},
    }
    presentation = eprx_ai_ui.build_eprx_readable_presentation(context)
    result = {"status": "ok", "summary": "unused", "procurement_patterns": [],
              "associations": [], "cautions": [], "presentation": presentation}
    target = _render(monkeypatch, cached=True, cached_result=result)
    rendered = "\n".join(
        target.markdowns
        + target.writes
        + target.infos
        + target.warnings
        + target.captions
        + [str(item) for item in target.metrics]
        + [table.to_string(index=False) for table in target.tables]
    )
    for expected in ("572.4 MW", "504.4 MW", "68.0 MW", "13.5%", "544.0", "599.0",
                     "591.1 MW", "548.3 MW", "40,022.9 MW", "35,877.1 MW",
                     "4,145.8 MW", "r = +0.58", "ρ = +0.47", "r = +0.56",
                     "ρ = +0.45", "r = -0.31", "태양광+풍력 발전량", "잔여수요"):
        assert expected in rendered
    for forbidden in ("572.425595", "504.422619", "0.577583587", "Pearson r = +572",
                      "Pearson r = +544", "coefficient", "residual_demand_proxy_mw",
                      "Spearman CI ["):
        assert forbidden not in rendered
    assert "한눈에 보기" not in rendered
    assert "분석 기준 — 먼저 읽어주세요" in rendered
    assert "날씨 → 전력수요 · 태양광 · 풍력 → 잔여수요 → 잔여수요의 짧은 주기 변동 → 1차 조정력 평상시분" in rendered
    assert "공식 필요량을 재현한 값이 아닙니다" in rendered
    assert "시간대별 모집량·잔여수요 변동 경향" in rendered
    assert "중간 수준의 같은 방향 경향" in rendered
    assert "ρ = +0.52" in rendered and "높은 시간대 겹침 8/12" in rendered
    assert all(forbidden not in rendered for forbidden in ("잔여수요가 모집량을 증가", "잔여수요가 모집량을 결정", "잔여수요 변동 때문에"))
    assert "중간 수준 · 같은 방향" in rendered and "모집량의 공식 산정 원인을 의미하지 않습니다" in rendered
    assert "유의" not in rendered
    assert len(target.metrics) == 4 and len(target.tables) == 4
    assert list(target.tables[0]["지표"]) == ["잔여수요 30분 평균 변동폭", "전력수요 30분 평균 변동폭", "재생에너지 30분 평균 변동폭"]
    assert set(target.tables[2]["변수"]) == {"전력수요", "잔여수요", "태양광+풍력 발전량", "잔여수요 변동폭 |Δ잔여수요|"}
    assert len(target.plotly_charts) == 1
    figure = target.plotly_charts[0][0]
    assert [trace.name for trace in figure.data] == ["1차 조정력 모집량", "잔여수요 30분 변동폭"]
    assert list(figure.data[0].y) == [550.0, 590.0]
    assert list(figure.data[1].y) == [700.0, 900.0]
    assert figure.data[0].yaxis == "y" and figure.data[1].yaxis == "y2"
    assert figure.layout.yaxis.title.text == "1차 조정력 모집량 (MW)"
    assert figure.layout.yaxis2.title.text == "잔여수요 30분 변동폭 (MW)"
    assert "customdata[0]" in figure.data[0].hovertemplate and "customdata[1]" in figure.data[0].hovertemplate
    assert "주간 평균 = 100" not in rendered
    assert "전력수요 − 태양광 발전량 − 풍력 발전량" in rendered
    assert "시간적으로 연속된 30분 잔여수요 값의 절대 변화량" in rendered
    assert "도쿄전력파워그리드(TEPCO PG)" in rendered
    assert "이 화면은 EPRX 1차 조정력 모집량을 분석합니다" in rendered
    assert all(term in rendered for term in ("평상시분", "이상시분", "기타 요인"))
    basis_index = next(i for i, event in enumerate(target.events) if event[0] == "info" and "분석 기준" in event[1])
    metric_index = next(i for i, event in enumerate(target.events) if event[0] == "metric")
    assert basis_index < metric_index
    background_index = next(i for i, event in enumerate(target.events) if event[0] == "markdown" and event[1] == "### 잔여수요 변동의 배경")
    tendency_index = next(i for i, event in enumerate(target.events) if event[0] == "markdown" and event[1] == "### 시간대별 모집량·잔여수요 변동 경향")
    assert metric_index < background_index < tendency_index


def test_chubu_presentation_uses_chubu_grid_source():
    presentation = eprx_ai_ui.build_eprx_readable_presentation({
        "region": "Chubu", "selected_week": {"week": {
            "start": "2026-07-20", "end": "2026-07-26"}},
    })
    assert presentation["grid_source_label"] == "중부전력파워그리드(Chubu PG)"


def test_heavy_context_runs_only_after_enabled_button_click(monkeypatch):
    monkeypatch.setattr(eprx_ai_ui, "run_ai_analysis_action", lambda **_kwargs: {
        "status": "ok", "headline": "완료", "summary": "요약"})
    before = _render(monkeypatch)
    assert before.heavy_calls == []
    after = _render(monkeypatch, clicked_labels={"AI 분석 생성"})
    assert len(after.heavy_calls) == 1
    assert after.expanders == [("상세 통계 분석", {"expanded": False}),
                               ("개발용 성능 진단", {"expanded": False})]
    assert after.json_values[-1]["total_seconds"] >= 0
    assert "readiness_seconds" in after.json_values[-1]


def test_readiness_cache_key_changes_with_grid_or_selected_eprx_fingerprint():
    frame = pd.DataFrame({"area": ["Tokyo"], "delivery_date": ["2026-07-20"],
                          "period_no": [1], "period_start": ["00:00"],
                          "procurement_volume": [572.4]})
    first = eprx_ai_ui.make_readiness_cache_key(frame, "Tokyo", "2026-07-20", "grid-a")
    assert first != eprx_ai_ui.make_readiness_cache_key(frame, "Tokyo", "2026-07-20", "grid-b")
    changed = frame.copy(); changed.loc[0, "procurement_volume"] = 573.0
    assert first != eprx_ai_ui.make_readiness_cache_key(changed, "Tokyo", "2026-07-20", "grid-a")
