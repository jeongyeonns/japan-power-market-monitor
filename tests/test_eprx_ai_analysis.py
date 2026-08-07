import json
import os

import numpy as np
import pandas as pd

from utils.eprx_ai_analysis import (
    build_eprx_ai_payload, build_eprx_statistical_fallback,
    calculate_eprx_context_hash, generate_eprx_ai_analysis,
    resolve_openai_settings, validate_eprx_ai_response,
)


def context():
    return {"region": "Tokyo", "analysis_period": {"start": "2026-03-14", "end": "2026-03-20"},
        "selected_week": {"week": {"start": "2026-03-14", "market_regimes": ["modern"]},
            "procurement_change": {"current": np.float64(10), "previous": 9},
            "driver_changes": {}, "association_candidates": {"items": []},
            "co_movement_comparison": {}},
        "raw_correlations": {}, "time_adjusted_correlations": {"demand_mw": {"spearman": 0.2, "sample_count": 336}},
        "bootstrap_intervals": {}, "regression_models": [], "profile_repetition": {},
        "data_quality": {"timestamp": pd.Timestamp("2026-03-14")}, "warnings": [],
        "limitations": ["인과관계가 아님"], "calculation_metadata": {}}


def response_for(ctx, model="gpt-5-mini"):
    return {"status": "ok", "region": "Tokyo", "week_start": "2026-03-14", "headline": "요약",
        "summary": "통계 요약", "confirmed_findings": [], "statistical_interpretation": [],
        "association_candidates": [], "counter_evidence": [], "profile_warning": "주의",
        "data_quality_notes": [], "limitations": [], "conclusion": "결론", "disclaimer": "비예측",
        "model": model, "context_hash": calculate_eprx_context_hash(ctx)}


def test_settings_priority(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    assert resolve_openai_settings("argument-key", "argument-model") == ("argument-key", "argument-model")


def test_missing_key_and_import_without_sdk(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = generate_eprx_ai_analysis(context())
    assert result["status"] == "api_key_missing"
    assert result["fallback"]["status"] == "statistical_summary_only"


def test_payload_is_bounded_json_safe_and_reproducible():
    ctx = context(); built = build_eprx_ai_payload(ctx)
    assert built["diagnostics"]["input_character_count"] <= 30000
    assert "dataframe" not in json.dumps(built["payload"], ensure_ascii=False).lower()
    assert calculate_eprx_context_hash(ctx) == calculate_eprx_context_hash(ctx)
    json.dumps(built, allow_nan=False)


def test_invalid_context_never_calls_client():
    called = []
    result = generate_eprx_ai_analysis({}, api_key="fake-test-key", _client_factory=lambda **kw: called.append(kw))
    assert result["status"] == "invalid_context" and not called


def test_structured_response_validation_and_single_call():
    ctx = context(); payload = response_for(ctx)
    class Responses:
        def __init__(self): self.calls = 0
        def create(self, **kwargs):
            self.calls += 1
            return type("R", (), {"output_text": json.dumps(payload, ensure_ascii=False)})()
    responses = Responses(); client = type("C", (), {"responses": responses})()
    result = generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: client)
    assert result["status"] == "ok" and responses.calls == 1
    assert "input" not in result and "output_text" not in result


def test_bad_identity_is_rejected():
    ctx = context(); data = response_for(ctx); data["region"] = "Chubu"
    checked = validate_eprx_ai_response(data, expected_region="Tokyo")
    assert not checked["valid"] and "region_mismatch" in checked["errors"]


def test_numeric_literal_not_present_in_input_is_rejected():
    ctx = context(); data = response_for(ctx); data["summary"] = "근거에 없는 9999 값"
    checked = validate_eprx_ai_response(data, input_payload=build_eprx_ai_payload(ctx)["payload"])
    assert not checked["valid"] and "unsupported_numeric_literal" in checked["errors"]


def test_transient_error_retries_once_and_auth_does_not():
    ctx = context()
    class TemporaryError(Exception): pass
    class AuthenticationError(Exception): pass
    class Responses:
        def __init__(self, error): self.calls = 0; self.error = error
        def create(self, **kwargs): self.calls += 1; raise self.error()
    temporary = Responses(TemporaryError)
    generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: type("C", (), {"responses": temporary})())
    assert temporary.calls == 2
    auth = Responses(AuthenticationError)
    generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: type("C", (), {"responses": auth})())
    assert auth.calls == 1


def test_fallback_selects_three_relations():
    ctx = context(); ctx["time_adjusted_correlations"] = {str(i): {"spearman": i / 10, "sample_count": 100} for i in range(5)}
    assert len(build_eprx_statistical_fallback(ctx)["strongest_adjusted_spearman"]) == 3
