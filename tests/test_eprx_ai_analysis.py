import json
import os
import pytest

import numpy as np
import pandas as pd
import utils.eprx_ai_analysis as ai_module

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
    return {"status": "ok", "region": "Tokyo", "week_start": "2026-03-14",
        "summary": "통계 요약", "procurement_patterns": ["주간 패턴"],
        "associations": [{"metric_path": "analysis.selected_associations.0.spearman",
            "display_name": "수요", "value": 0.2, "unit": "coefficient", "interpretation": "약한 양의 관계"}],
        "cautions": ["인과관계를 의미하지 않는다."],
        "context_hash": calculate_eprx_context_hash(ctx)}


class Parsed:
    def __init__(self, value): self.value = value
    def model_dump(self): return self.value


def api_response(*, parsed=None, output_text="", status="completed", output=None,
                 incomplete_reason=None):
    details = type("Details", (), {"reason": incomplete_reason})() if incomplete_reason else None
    return type("Response", (), {"id": "resp_test", "status": status,
        "output_parsed": parsed, "output_text": output_text, "output": output or [],
        "incomplete_details": details})()


def client_returning(response):
    class Responses:
        def __init__(self): self.calls = 0; self.request = None
        def parse(self, **kwargs): self.calls += 1; self.request = kwargs; return response
    responses = Responses()
    return type("Client", (), {"responses": responses})(), responses


def test_settings_priority(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    assert resolve_openai_settings("argument-key", "argument-model") == ("argument-key", "argument-model")


def test_missing_key_and_import_without_sdk(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(ai_module, "_streamlit_secret", lambda name: None)
    result = generate_eprx_ai_analysis(context())
    assert result["status"] == "api_key_missing"
    assert result["fallback"]["status"] == "statistical_summary_only"


def test_payload_is_bounded_json_safe_and_reproducible():
    ctx = context(); built = build_eprx_ai_payload(ctx)
    assert built["diagnostics"]["input_character_count"] <= 30000
    assert "dataframe" not in json.dumps(built["payload"], ensure_ascii=False).lower()
    assert calculate_eprx_context_hash(ctx) == calculate_eprx_context_hash(ctx)
    json.dumps(built, allow_nan=False)
    assert "calculation_metadata" not in built["payload"]
    assert "co_movement_comparison" not in built["payload"]


def test_payload_limits_relation_detail_to_runtime_evidence_set():
    ctx = context()
    ctx["time_adjusted_correlations"] = {
        f"driver_{index}": {"spearman": index / 10, "sample_count": 336}
        for index in range(10)
    }
    ctx["raw_correlations"] = ctx["time_adjusted_correlations"]
    built = build_eprx_ai_payload(ctx)["payload"]
    assert len(built["selected_associations"]) == 5


def test_payload_supplies_display_ready_values_and_prompt_requires_them():
    ctx = context()
    ctx["selected_week"]["procurement"] = {"mean": 572.4255952380952, "minimum": 544, "maximum": 599}
    ctx["selected_week"]["procurement_change"] = {
        "previous": 504.42261904761904, "change": 68.00297619047615, "change_pct": 13.4817}
    payload = build_eprx_ai_payload(ctx)["payload"]
    assert payload["display_values"]["procurement_mean"] == "572.4 MW"
    assert payload["display_values"]["week_change"] == "68.0 MW"
    assert payload["display_values"]["week_change_pct"] == "13.5%"


def test_invalid_context_never_calls_client():
    called = []
    result = generate_eprx_ai_analysis({}, api_key="fake-test-key", _client_factory=lambda **kw: called.append(kw))
    assert result["status"] == "invalid_context" and not called


def test_structured_response_validation_and_single_call():
    ctx = context(); payload = response_for(ctx)
    client, responses = client_returning(api_response(parsed=Parsed(payload)))
    result = generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: client)
    assert result["status"] == "ok" and responses.calls == 1
    assert "text_format" in responses.request
    assert responses.request["max_output_tokens"] == 2000
    assert responses.request["reasoning"] == {"effort": "minimal"}
    assert responses.request["text"] == {"verbosity": "low"}
    assert "verbosity" not in responses.request
    assert "모집량 패턴은 최대 3개" in responses.request["instructions"]
    assert result["request_diagnostics"]["reasoning_effort"] == "minimal"
    assert result["response_diagnostics"]["structured_parsed_present"] is True
    assert "input" not in result and "output_text" not in result


def test_pydantic_schema_accepts_structured_evidence():
    parsed = ai_module._response_model().model_validate(response_for(context()))
    assert parsed.associations[0].metric_path.endswith("spearman")


def test_pydantic_schema_json_empty_multiple_and_numeric_edges():
    model = ai_module._response_model(); data = response_for(context())
    data["associations"] = []
    assert model.model_validate_json(json.dumps(data)).associations == []
    data["associations"] = [evidence("metric.zero", 0.0, "unitless"),
                                   evidence("metric.negative", -0.35)]
    parsed = model.model_validate(data)
    assert [item.value for item in parsed.associations] == [0.0, -0.35]


def test_pydantic_schema_rejects_missing_string_value_and_extra_field():
    model = ai_module._response_model()
    missing = response_for(context()); missing.pop("summary")
    with pytest.raises(Exception): model.model_validate(missing)
    string_value = response_for(context()); string_value["associations"][0]["value"] = "0.2"
    with pytest.raises(Exception): model.model_validate(string_value)
    extra = response_for(context()); extra["unexpected"] = "no"
    with pytest.raises(Exception): model.model_validate(extra)


def test_pydantic_validation_diagnostics_exclude_input_values():
    model = ai_module._response_model(); data = response_for(context())
    data["associations"][0]["value"] = "not-a-number"
    try:
        model.model_validate(data)
    except Exception as exc:
        result = ai_module._error_result(exc, {})
    assert result["diagnostics"]["error_count"] == 1
    error = result["diagnostics"]["validation_errors"][0]
    assert error["loc"][-1] == "value" and error["type"] == "float_type"
    assert "input" not in error


def test_eof_pydantic_error_is_classified_as_incomplete():
    from pydantic import ValidationError
    try:
        ai_module._response_model().model_validate_json('{"status":"ok"')
    except ValidationError as exc:
        result = ai_module._error_result(exc, {})
    assert result["status"] == "api_incomplete_response"
    assert result["diagnostics"]["validation_errors"][0]["type"] == "json_invalid"


def test_plain_and_fenced_json_are_safe_fallbacks():
    ctx = context(); payload = response_for(ctx); text = json.dumps(payload, ensure_ascii=False)
    for output in (text, f"```json\n{text}\n```"):
        client, _ = client_returning(api_response(output_text=output))
        assert generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: client)["status"] == "ok"


def test_empty_and_malformed_output_are_parse_errors():
    ctx = context()
    for output in ("", "{not-json"):
        client, _ = client_returning(api_response(output_text=output))
        result = generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: client)
        assert result["status"] == "api_response_parse_error"
        assert "JSON" not in result["message"]


def test_refusal_and_incomplete_are_not_parsed():
    ctx = context()
    refusal = {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}
    client, _ = client_returning(api_response(output=[refusal]))
    assert generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: client)["status"] == "api_refusal"
    for reason in ("max_output_tokens", "content_filter"):
        client, _ = client_returning(api_response(status="incomplete", incomplete_reason=reason))
        result = generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: client)
        assert result["status"] == "api_incomplete_response"
        assert result["diagnostics"]["incomplete_reason"] == reason


def test_bad_identity_is_rejected():
    ctx = context(); data = response_for(ctx); data["region"] = "Chubu"
    checked = validate_eprx_ai_response(data, expected_region="Tokyo")
    assert not checked["valid"] and "region_mismatch" in checked["errors"]


def evidence(path, value, unit="coefficient"):
    return {"metric_path": path, "display_name": "근거", "value": value,
            "unit": unit, "interpretation": "해석"}


def test_free_text_domain_numbers_do_not_reject_response():
    ctx = context(); data = response_for(ctx)
    data["summary"] = "2026-07-27 주차의 1차 조정력 30분 자료로 48코마·336행을 확인했다."
    data["cautions"] = ["상위 20%와 하위 20%, 95% 신뢰구간은 인과를 뜻하지 않는다."]
    payload = {"analysis": build_eprx_ai_payload(ctx)["payload"]}
    checked = validate_eprx_ai_response(data, input_payload=payload)
    assert checked["valid"]
    assert "narrative_numeric_literal_count" not in checked["diagnostics"]


def test_structured_evidence_path_value_rounding_negative_and_percentage():
    ctx = context(); data = response_for(ctx)
    payload = {"metric": {"correlation": {"spearman": -0.347812}, "change_pct": 2.386}}
    data["associations"] = [evidence("metric.correlation.spearman", -0.35),
                            evidence("metric.change_pct", 2.4, "%")]
    assert validate_eprx_ai_response(data, input_payload=payload)["valid"]


def test_structured_evidence_missing_path_and_wrong_value_are_rejected():
    ctx = context(); payload = {"metric": {"correlation": {"spearman": 0.347812}}}
    for item, reason in ((evidence("metric.missing", 0.35), "metric_path_not_found"),
                         (evidence("metric.correlation.spearman", 0.8), "evidence_value_mismatch")):
        data = response_for(ctx); data["associations"] = [item]
        checked = validate_eprx_ai_response(data, input_payload=payload)
        assert not checked["valid"] and "invalid_structured_evidence" in checked["errors"]
        assert checked["diagnostics"]["evidence_errors"][0]["reason"] == reason


def test_api_errors_do_not_trigger_implicit_second_request():
    ctx = context()
    class TemporaryError(Exception): pass
    class AuthenticationError(Exception): pass
    class Responses:
        def __init__(self, error): self.calls = 0; self.error = error
        def parse(self, **kwargs): self.calls += 1; raise self.error()
    temporary = Responses(TemporaryError)
    generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: type("C", (), {"responses": temporary})())
    assert temporary.calls == 1
    auth = Responses(AuthenticationError)
    generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: type("C", (), {"responses": auth})())
    assert auth.calls == 1


def test_timeout_is_bounded_and_not_retried():
    ctx = context()
    class APITimeoutError(Exception): pass
    class Responses:
        def __init__(self): self.calls = 0
        def parse(self, **_kwargs): self.calls += 1; raise APITimeoutError()
    responses = Responses()
    result = generate_eprx_ai_analysis(ctx, api_key="fake-test-key",
        _client_factory=lambda **_kwargs: type("Client", (), {"responses": responses})())
    assert result["status"] == "api_timeout"
    assert "90초" in result["message"] and responses.calls == 1
    assert result["request_diagnostics"]["timeout_seconds"] == 90.0


def test_raw_response_timing_and_per_request_options_are_recorded():
    ctx = context(); parsed = Parsed(response_for(ctx))
    response = api_response(parsed=parsed)
    class Raw:
        retries_taken = 0
        def parse(self): return response
    class RawResponses:
        def parse(self, **_kwargs): return Raw()
    class Responses:
        with_raw_response = RawResponses()
    class Client:
        responses = Responses()
        def __init__(self): self.options = None
        def with_options(self, **options): self.options = options; return self
    client = Client()
    result = generate_eprx_ai_analysis(ctx, api_key="fake-test-key",
        _client_factory=lambda **_kwargs: client)
    assert result["status"] == "ok"
    assert client.options == {"timeout": 90.0, "max_retries": 0}
    assert result["request_diagnostics"]["api_calls"] == 1
    assert result["response_diagnostics"]["sdk_retries_taken"] == 0
    assert result["response_diagnostics"]["api_elapsed_seconds"] >= 0
    assert result["response_diagnostics"]["parse_elapsed_seconds"] >= 0


def test_api_error_categories_and_unsupported_model():
    ctx = context()
    cases = (("AuthenticationError", 401, "api_auth_error"),
             ("RateLimitError", 429, "api_rate_limit"),
             ("BadRequestError", 404, "api_model_error"),
             ("APIError", 500, "api_call_failed"))
    for name, status_code, expected in cases:
        error = type(name, (Exception,), {"status_code": status_code})
        class Responses:
            def parse(self, **kwargs): raise error("safe error")
        client = type("Client", (), {"responses": Responses()})()
        result = generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: client)
        assert result["status"] == expected
        assert result["diagnostics"]["exception_class"] == name
        assert result["diagnostics"]["status_code"] == status_code
    assert generate_eprx_ai_analysis(ctx, api_key="fake-test-key", model="gpt-3.5-turbo")["status"] == "unsupported_model"


def test_parsed_identity_mismatches_are_rejected():
    ctx = context()
    for field, value, error in (("region", "Chubu", "region_mismatch"),
                                ("week_start", "2026-03-15", "week_start_mismatch"),
                                ("context_hash", "wrong", "context_hash_mismatch")):
        payload = response_for(ctx); payload[field] = value
        client, _ = client_returning(api_response(parsed=Parsed(payload)))
        result = generate_eprx_ai_analysis(ctx, api_key="fake-test-key", _client_factory=lambda **kw: client)
        assert result["status"] == "api_response_validation_error" and error in result["errors"]
        assert result["diagnostics"]["structured_parsed_present"] is True


def test_fallback_selects_three_relations():
    ctx = context(); ctx["time_adjusted_correlations"] = {str(i): {"spearman": i / 10, "sample_count": 100} for i in range(5)}
    assert len(build_eprx_statistical_fallback(ctx)["strongest_adjusted_spearman"]) == 3
