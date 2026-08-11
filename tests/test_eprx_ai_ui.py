from contextlib import nullcontext

import pandas as pd

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


class RenderTarget:
    def __init__(self, clicked_labels=()):
        self.buttons = []
        self.expanders = []
        self.markdowns = []
        self.writes = []
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
        self.markdowns.append(value)

    def write(self, value, **kwargs):
        self.writes.append(value)

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
        assert target.expanders == []
        assert target.heavy_calls == []


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
    rendered = "\n".join(target.markdowns + target.writes)
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


def test_heavy_context_runs_only_after_enabled_button_click(monkeypatch):
    monkeypatch.setattr(eprx_ai_ui, "run_ai_analysis_action", lambda **_kwargs: {
        "status": "ok", "headline": "완료", "summary": "요약"})
    before = _render(monkeypatch)
    assert before.heavy_calls == []
    after = _render(monkeypatch, clicked_labels={"AI 분석 생성"})
    assert len(after.heavy_calls) == 1
    assert after.expanders == [("Python 통계 요약", {"expanded": False})]


def test_readiness_cache_key_changes_with_grid_or_selected_eprx_fingerprint():
    frame = pd.DataFrame({"area": ["Tokyo"], "delivery_date": ["2026-07-20"],
                          "period_no": [1], "period_start": ["00:00"],
                          "procurement_volume": [572.4]})
    first = eprx_ai_ui.make_readiness_cache_key(frame, "Tokyo", "2026-07-20", "grid-a")
    assert first != eprx_ai_ui.make_readiness_cache_key(frame, "Tokyo", "2026-07-20", "grid-b")
    changed = frame.copy(); changed.loc[0, "procurement_volume"] = 573.0
    assert first != eprx_ai_ui.make_readiness_cache_key(changed, "Tokyo", "2026-07-20", "grid-a")
