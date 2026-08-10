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


class RenderTarget:
    def __init__(self):
        self.buttons = []
        self.expanders = []
        self.markdowns = []
        self.writes = []

    def button(self, label, **kwargs):
        self.buttons.append((label, kwargs))
        return False

    def expander(self, label, **kwargs):
        self.expanders.append((label, kwargs))
        return nullcontext()

    def markdown(self, value, **kwargs):
        self.markdowns.append(value)

    def write(self, value, **kwargs):
        self.writes.append(value)

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _render(monkeypatch, *, api_key="key", local_status="ok", cached=False):
    context = {"region": "Tokyo", "selected_week": {"week": {"market_regimes": ["modern"]}}}
    local = {"status": local_status, "message": "missing", "region": "Tokyo"}
    if local_status == "ok":
        local.update({"analysis_context": context, "join_diagnostics": {
            "matched_rows": 336, "eprx_rows": 336, "tepco_rows": 336},
            "latest_source_date": "2026-08-02", "file_fingerprint": "files"})
    monkeypatch.setattr(eprx_ai_ui, "load_local_eprx_grid_context", lambda *args: local)
    monkeypatch.setattr(eprx_ai_ui, "build_eprx_statistical_fallback", lambda _context: {})
    monkeypatch.setattr(eprx_ai_ui, "resolve_openai_settings", lambda: (api_key, "gpt-5-mini"))
    session = {}
    if cached and local_status == "ok":
        key = eprx_ai_ui.make_analysis_cache_key(
            "Tokyo", "2026-07-27", "files",
            eprx_ai_ui.calculate_eprx_context_hash(context), "gpt-5-mini")
        session[key] = {"status": "ok", "headline": "cached", "summary": "summary"}
    monkeypatch.setattr("streamlit.session_state", session)
    target = RenderTarget()
    eprx_ai_ui.render_eprx_ai_analysis_section(
        target, pd.DataFrame(), "Tokyo", pd.Timestamp("2026-07-27"))
    return target


def test_generate_button_visible_for_ready_and_cached_states(monkeypatch):
    for cached in (False, True):
        target = _render(monkeypatch, cached=cached)
        generate = next(item for item in target.buttons if item[0] == "AI 분석 생성")
        assert generate[1]["disabled"] is False
        assert target.expanders == [("Python 통계 요약", {"expanded": False})]


def test_generate_button_visible_but_disabled_without_key_or_data(monkeypatch):
    no_key = _render(monkeypatch, api_key=None)
    assert next(item for item in no_key.buttons if item[0] == "AI 분석 생성")[1]["disabled"] is True
    no_data = _render(monkeypatch, local_status="source_data_missing")
    assert next(item for item in no_data.buttons if item[0] == "AI 분석 생성")[1]["disabled"] is True


def test_result_renderer_treats_interpretation_string_as_paragraph_and_formats_evidence():
    target = RenderTarget()
    eprx_ai_ui._render_result(target, {
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
