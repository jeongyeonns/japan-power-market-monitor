import pandas as pd

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
