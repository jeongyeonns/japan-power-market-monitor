"""Cost-bounded OpenAI adapter for precomputed EPRX statistics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from functools import lru_cache
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MODEL = "gpt-5-mini"
MAX_INPUT_CHARACTERS = 30000
EPRX_AI_MAX_OUTPUT_TOKENS = 2000
EPRX_AI_REASONING_EFFORT = "minimal"
EPRX_AI_VERBOSITY = "low"
EPRX_AI_REQUEST_TIMEOUT_SECONDS = 90.0
EVIDENCE_REL_TOLERANCE = 0.02
EVIDENCE_ABS_TOLERANCE = 0.005
PAYLOAD_VERSION = "1.0"
FAST_REQUIRED_RESPONSE_FIELDS = (
    "status", "region", "week_start", "summary", "procurement_patterns",
    "associations", "cautions", "context_hash",
)
REQUIRED_RESPONSE_FIELDS = ("status", "region", "week_start", "headline", "summary",
    "confirmed_findings", "statistical_interpretation", "association_candidates",
    "counter_evidence", "profile_warning", "data_quality_notes", "limitations",
    "conclusion", "disclaimer", "context_hash")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)): return value.isoformat()
    if isinstance(value, np.generic): return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, pd.DataFrame): return {"excluded": "dataframe", "row_count": len(value)}
    return value


def _display_number(value: Any, kind: str) -> str | None:
    try: number = float(value)
    except (TypeError, ValueError): return None
    if not math.isfinite(number): return None
    if kind == "mw": return f"{number:,.1f} MW"
    if kind == "percent": return f"{number:.1f}%"
    if kind == "correlation": return f"{number:+.2f}"
    return f"{number:,.1f}"


def calculate_eprx_context_hash(analysis_context: dict[str, Any]) -> str:
    encoded = json.dumps(_json_safe(analysis_context), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_eprx_ai_context(analysis_context: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if not isinstance(analysis_context, dict): errors.append("context_must_be_object")
    else:
        if analysis_context.get("region") not in {"Tokyo", "Chubu"}: errors.append("unsupported_region")
        selected = analysis_context.get("selected_week", {})
        week = selected.get("week", {}) if isinstance(selected, dict) else {}
        if not week.get("start"): errors.append("week_start_missing")
        for field in ("time_adjusted_correlations", "limitations", "data_quality"):
            if field not in analysis_context: errors.append(f"{field}_missing")
    return {"valid": not errors, "errors": errors}


def _trim_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    excluded = []
    candidates = payload.get("association_candidates", {})
    if isinstance(candidates, dict) and len(candidates.get("items", [])) > 5:
        candidates["items"] = candidates["items"][:5]; excluded.append("association_candidates_after_5")
    for name in ("bootstrap_intervals", "regression_models", "association_candidates"):
        if len(json.dumps(payload, ensure_ascii=False, allow_nan=False)) <= MAX_INPUT_CHARACTERS: break
        if name == "bootstrap_intervals" and isinstance(payload.get(name), dict):
            payload[name] = dict(list(payload[name].items())[:6]); excluded.append("bootstrap_details_after_6")
        elif name == "regression_models" and isinstance(payload.get(name), list):
            payload[name] = [{k: v for k, v in x.items() if k != "singular_values"} for x in payload[name]]; excluded.append("regression_singular_values")
        elif name == "association_candidates" and isinstance(payload.get(name), dict):
            payload[name]["items"] = payload[name].get("items", [])[:3]; excluded.append("association_candidates_after_3")
    return payload, excluded


def build_eprx_ai_payload(analysis_context: dict[str, Any]) -> dict[str, Any]:
    validation = validate_eprx_ai_context(analysis_context)
    if not validation["valid"]: raise ValueError("Invalid EPRX context: " + ", ".join(validation["errors"]))
    selected = analysis_context["selected_week"]
    display_names = {
        "demand_mw": "평균 수요", "renewable_generation_mw": "재생에너지 발전량",
        "renewable_share_pct": "재생에너지 비율", "residual_demand_proxy_mw": "잔여수요 추정치",
        "abs_renewable_ramp_30m_mw": "재생에너지 30분 변동폭",
        "abs_solar_ramp_30m_mw": "태양광 30분 변동폭",
        "abs_demand_ramp_30m_mw": "수요 30분 변동폭",
    }
    associations = []
    for name, relation in (analysis_context.get("selected_week_correlations") or
                           analysis_context.get("time_adjusted_correlations") or {}).items():
        score = relation.get("spearman")
        if score is None:
            continue
        associations.append({"variable": name, "display_name": display_names.get(name, name),
            **{key: relation.get(key) for key in ("sample_count", "pearson", "spearman",
                "spearman_direction", "spearman_strength")}})
    associations.sort(key=lambda item: (-abs(item["spearman"]), item["variable"]))
    notable = selected.get("notable_time_blocks") or {}
    notable = {"highest": notable.get("highest", [])[:3], "lowest": notable.get("lowest", [])[:3]}
    repetition = analysis_context.get("profile_repetition") or {}
    quality = analysis_context.get("data_quality") or {}
    payload = _json_safe({
        "region": analysis_context["region"], "week": selected["week"],
        "procurement_summary": selected.get("procurement"),
        "previous_week_comparison": selected.get("procurement_change"),
        "historical_position": selected.get("historical_position"),
        "intraday_profile_summary": notable,
        "profile_repetition": {key: repetition.get(key) for key in
            ("same_as_previous_week", "maximum_absolute_difference_mw", "unique_weekly_profile_count")},
        "selected_associations": associations[:5],
        "data_quality_summary": {key: quality.get(key) for key in
            ("expected_weekly_rows", "actual_rows", "duplicate_datetime_rows", "missing_procurement_count")},
        "limitations": list(dict.fromkeys(analysis_context.get("limitations", [])))[:2],
    })
    procurement = selected.get("procurement") or {}
    previous = selected.get("procurement_change") or {}
    payload["display_values"] = {
        "procurement_mean": _display_number(procurement.get("mean"), "mw"),
        "procurement_minimum": _display_number(procurement.get("minimum"), "mw"),
        "procurement_maximum": _display_number(procurement.get("maximum"), "mw"),
        "previous_week_mean": _display_number(previous.get("previous"), "mw"),
        "week_change": _display_number(previous.get("change"), "mw"),
        "week_change_pct": _display_number(previous.get("change_pct"), "percent"),
        "associations": [{"display_name": item["display_name"],
            "pearson": _display_number(item.get("pearson"), "correlation"),
            "spearman": _display_number(item.get("spearman"), "correlation")}
            for item in associations[:5]],
    }
    payload, excluded = _trim_payload(payload)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if len(text) > MAX_INPUT_CHARACTERS: raise ValueError("AI payload exceeds the configured character limit")
    return {"payload": payload, "diagnostics": {"input_character_count": len(text),
        "included_sections": list(payload), "excluded_sections": excluded,
        "context_hash": calculate_eprx_context_hash(analysis_context), "payload_version": PAYLOAD_VERSION}}


def _streamlit_secret(name: str) -> Any:
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def resolve_openai_settings(api_key: str | None = None, model: str | None = None) -> tuple[str | None, str]:
    key = api_key or _streamlit_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    selected_model = model or _streamlit_secret("OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    return key, selected_model


def build_eprx_statistical_fallback(analysis_context: dict[str, Any]) -> dict[str, Any]:
    selected = analysis_context.get("selected_week", {})
    relations = []
    for variable, item in analysis_context.get("time_adjusted_correlations", {}).items():
        if item.get("spearman") is not None:
            relations.append({"variable": variable, "spearman": item["spearman"], "sample_count": item.get("sample_count")})
    relations.sort(key=lambda x: (-abs(x["spearman"]), x["variable"]))
    candidates = selected.get("association_candidates", {}).get("items", [])[:3]
    return {"status": "statistical_summary_only", "region": analysis_context.get("region"),
        "week_start": selected.get("week", {}).get("start"), "procurement_change": selected.get("procurement_change"),
        "strongest_adjusted_spearman": relations[:3], "association_candidates": candidates,
        "profile_warning": analysis_context.get("warnings", []), "data_quality_notes": analysis_context.get("data_quality", {}),
        "limitations": analysis_context.get("limitations", []),
        "conclusion": "공개 30분 실적을 이용한 규칙 기반 통계 요약이며 AI 생성 결과가 아닙니다."}


def validate_eprx_ai_response(response_data: dict[str, Any], *, expected_region: str | None = None,
                              expected_week_start: str | None = None, expected_context_hash: str | None = None,
                              input_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    errors = []
    diagnostics: dict[str, Any] = {}
    if not isinstance(response_data, dict): return {"valid": False, "errors": ["response_must_be_object"]}
    fast_response = isinstance(response_data, dict) and "procurement_patterns" in response_data
    required_fields = FAST_REQUIRED_RESPONSE_FIELDS if fast_response else REQUIRED_RESPONSE_FIELDS
    for field in required_fields:
        if field not in response_data: errors.append(f"missing_{field}")
    list_fields = (("procurement_patterns", "associations", "cautions") if fast_response else
                   ("confirmed_findings", "association_candidates", "counter_evidence", "data_quality_notes", "limitations"))
    for field in list_fields:
        if field in response_data and not isinstance(response_data[field], list): errors.append(f"invalid_type_{field}")
    for field in ("headline", "summary", "statistical_interpretation", "profile_warning", "conclusion", "disclaimer"):
        if field in response_data and not isinstance(response_data[field], str): errors.append(f"invalid_type_{field}")
    if expected_region and response_data.get("region") != expected_region: errors.append("region_mismatch")
    if expected_week_start and response_data.get("week_start") != expected_week_start: errors.append("week_start_mismatch")
    if expected_context_hash and response_data.get("context_hash") != expected_context_hash: errors.append("context_hash_mismatch")
    evidence_errors = []
    if input_payload is not None:
        evidence_fields = ("associations",) if fast_response else ("confirmed_findings", "association_candidates")
        for field in evidence_fields:
            for index, evidence in enumerate(response_data.get(field, [])):
                issue = _validate_evidence(evidence, input_payload)
                if issue:
                    evidence_errors.append({"response_field": field, "index": index, **issue})
        if evidence_errors:
            errors.append("invalid_structured_evidence")
            diagnostics["evidence_errors"] = evidence_errors
    return {"valid": not errors, "errors": errors,
            "diagnostics": diagnostics,
            "numeric_validation_scope": "schema and identity fields only; free-text numeric provenance cannot be proven reliably"}


def _schema() -> dict[str, Any]:
    properties = {field: {"type": "string"} for field in ("status", "region", "week_start", "headline", "summary", "profile_warning", "conclusion", "disclaimer", "model", "context_hash")}
    for field in ("confirmed_findings", "statistical_interpretation", "association_candidates", "counter_evidence", "data_quality_notes", "limitations"):
        properties[field] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": properties, "required": list(REQUIRED_RESPONSE_FIELDS), "additionalProperties": False}


@lru_cache(maxsize=1)
def _response_model():
    """Create the Pydantic schema lazily so non-AI imports remain safe."""
    from pydantic import ConfigDict, create_model

    Evidence = create_model("EprxEvidence", __config__=ConfigDict(extra="forbid", strict=True),
        metric_path=(str, ...), display_name=(str, ...), value=(float, ...),
        unit=(str, ...), interpretation=(str, ...))
    return create_model("FastEprxAiResponse", __config__=ConfigDict(extra="forbid", strict=True),
        status=(str, ...), region=(str, ...), week_start=(str, ...), summary=(str, ...),
        procurement_patterns=(list[str], ...), associations=(list[Evidence], ...),
        cautions=(list[str], ...), context_hash=(str, ...))


def _resolve_metric_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise KeyError(path)
    return current


def _expected_unit(path: str) -> str | None:
    lowered = path.lower()
    if any(token in lowered for token in ("_pct", "percent", "change_pct")): return "%"
    if any(token in lowered for token in ("spearman", "pearson", "correlation", "r_squared", "coefficient")): return "coefficient"
    if any(token in lowered for token in ("_mw", "procurement_change")): return "MW"
    if any(token in lowered for token in ("count", "rows", "iterations")): return "count"
    return None


def _validate_evidence(evidence: Any, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(evidence, dict):
        return {"reason": "evidence_must_be_object"}
    path = evidence.get("metric_path")
    if not isinstance(path, str):
        return {"reason": "metric_path_missing"}
    try:
        expected = _resolve_metric_path(payload, path)
    except KeyError:
        return {"reason": "metric_path_not_found", "metric_path": path}
    if isinstance(expected, bool) or not isinstance(expected, (int, float)) or not math.isfinite(float(expected)):
        return {"reason": "metric_path_not_numeric", "metric_path": path}
    received = evidence.get("value")
    if isinstance(received, bool) or not isinstance(received, (int, float)) or not math.isfinite(float(received)):
        return {"reason": "evidence_value_not_numeric", "metric_path": path}
    if not math.isclose(float(received), float(expected), rel_tol=EVIDENCE_REL_TOLERANCE,
                        abs_tol=EVIDENCE_ABS_TOLERANCE):
        return {"reason": "evidence_value_mismatch", "metric_path": path,
                "expected_value": float(expected), "received_value": float(received),
                "relative_tolerance": EVIDENCE_REL_TOLERANCE,
                "absolute_tolerance": EVIDENCE_ABS_TOLERANCE}
    expected_unit = _expected_unit(path)
    received_unit = str(evidence.get("unit", "")).strip()
    aliases = {"coefficient": {"coefficient", "상관계수", "unitless", ""},
               "count": {"count", "개", "행", "회"}, "MW": {"MW"}, "%": {"%", "percentage"}}
    if expected_unit and received_unit not in aliases[expected_unit]:
        return {"reason": "evidence_unit_mismatch", "metric_path": path,
                "expected_unit": expected_unit, "received_unit": received_unit}
    return None


def _numeric_metric_catalog(payload: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    def visit(value: Any, path: str) -> None:
        if len(output) >= limit: return
        if isinstance(value, dict):
            for key, child in value.items(): visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value): visit(child, f"{path}.{index}")
        elif not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)):
            output.append({"metric_path": path, "value": value, "unit": _expected_unit(path) or "unitless"})
    visit(payload, "")
    return output


def _value(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _response_diagnostics(response: Any) -> dict[str, Any]:
    output = _value(response, "output", []) or []
    item_types = [str(_value(item, "type", type(item).__name__)) for item in output]
    text = _value(response, "output_text", "") or ""
    incomplete = _value(response, "incomplete_details")
    usage = _value(response, "usage")
    output_details = _value(usage, "output_tokens_details") if usage else None
    return {"response_status": _value(response, "status"), "response_id": _value(response, "id"),
            "output_item_types": item_types, "output_text_present": bool(text.strip()),
            "output_text_length": len(text), "structured_parsed_present": _value(response, "output_parsed") is not None,
            "incomplete_reason": _value(incomplete, "reason") if incomplete else None,
            "refusal_present": bool(_find_refusal(output)),
            "input_tokens": _value(usage, "input_tokens") if usage else None,
            "output_tokens": _value(usage, "output_tokens") if usage else None,
            "reasoning_tokens": _value(output_details, "reasoning_tokens") if output_details else None,
            "total_tokens": _value(usage, "total_tokens") if usage else None}


def _find_refusal(output: Any) -> str | None:
    for item in output or []:
        for content in _value(item, "content", []) or []:
            if _value(content, "type") == "refusal" or _value(content, "refusal"):
                return str(_value(content, "refusal") or "refusal")
    return None


def _parse_json_fallback(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    if not stripped:
        raise ValueError("empty_output_text")
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("response_json_must_be_object")
    return parsed


def _error_result(exc: Exception, fallback: dict[str, Any]) -> dict[str, Any]:
    name = type(exc).__name__
    lower = name.lower()
    status_code = getattr(exc, "status_code", None)
    code = str(getattr(exc, "code", "") or "").lower()
    param = getattr(exc, "param", None)
    body = getattr(exc, "body", None)
    body_error = body.get("error", body) if isinstance(body, dict) else {}
    safe_api_type = body_error.get("type") if isinstance(body_error, dict) else None
    safe_api_code = body_error.get("code") if isinstance(body_error, dict) else None
    safe_api_param = body_error.get("param") if isinstance(body_error, dict) else None
    text = str(exc).lower()
    validation_errors = []
    candidate = exc
    seen = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        errors_method = getattr(candidate, "errors", None)
        if callable(errors_method):
            try:
                validation_errors = [{"loc": list(item.get("loc", ())), "type": item.get("type"),
                                      "msg": item.get("msg")} for item in errors_method(include_input=False)]
            except TypeError:
                validation_errors = [{"loc": list(item.get("loc", ())), "type": item.get("type"),
                                      "msg": item.get("msg")} for item in errors_method()]
            if validation_errors: break
        candidate = getattr(candidate, "__cause__", None) or getattr(candidate, "__context__", None)
    if "timeout" in lower:
        status, message = "api_timeout", "AI 분석 요청이 90초 안에 완료되지 않았습니다. 잠시 후 다시 시도해 주세요."
    elif "authentication" in lower or status_code == 401:
        status, message = "api_auth_error", "OpenAI API 인증에 실패했습니다. API 키 설정을 확인해 주세요."
    elif "ratelimit" in lower or status_code == 429:
        status, message = "api_rate_limit", "OpenAI API 사용 한도 또는 호출 제한에 도달했습니다."
    elif "model" in code or "model" in text or status_code == 404:
        status, message = "api_model_error", "설정된 OpenAI 모델을 사용할 수 없습니다."
    elif any(item.get("type") == "json_invalid" and "EOF" in str(item.get("msg")) for item in validation_errors):
        status, message = "api_incomplete_response", "AI 응답 생성이 완료되지 않았습니다. 다시 시도해 주세요."
    elif isinstance(exc, (json.JSONDecodeError, ValueError)):
        status, message = "api_response_parse_error", "AI 응답 형식을 해석하지 못했습니다. 응답 파싱 설정을 확인해 주세요."
    else:
        status, message = "api_call_failed", "OpenAI API 호출에 실패했습니다. 잠시 후 다시 시도해 주세요."
    diagnostics = {"exception_class": name, "status_code": status_code,
                   "api_error_type": safe_api_type, "api_error_code": safe_api_code or code or None,
                   "api_error_param": safe_api_param or param,
                   "error_count": len(validation_errors),
                   "validation_errors": validation_errors}
    return {"status": status, "message": message, "diagnostics": diagnostics, "fallback": fallback}


def _known_unsupported_model(model: str) -> bool:
    lowered = model.lower()
    return lowered.startswith("gpt-3.5") or lowered.startswith("gpt-4-turbo")


def generate_eprx_ai_analysis(analysis_context: dict[str, Any], api_key: str | None = None,
                              model: str | None = None, _client_factory=None,
                              _max_attempts: int = 1) -> dict[str, Any]:
    validation = validate_eprx_ai_context(analysis_context)
    fallback = build_eprx_statistical_fallback(analysis_context) if validation["valid"] else None
    if not validation["valid"]: return {"status": "invalid_context", "errors": validation["errors"]}
    key, selected_model = resolve_openai_settings(api_key, model)
    if not key: return {"status": "api_key_missing", "message": "OpenAI API 키가 설정되어 있지 않습니다.", "fallback": fallback}
    if _known_unsupported_model(selected_model):
        return {"status": "unsupported_model", "message": "설정된 OpenAI 모델은 Structured Outputs를 지원하지 않습니다.",
                "model": selected_model, "fallback": fallback}
    payload_started = time.perf_counter()
    built = build_eprx_ai_payload(analysis_context)
    payload_elapsed = time.perf_counter() - payload_started
    context_hash = built["diagnostics"]["context_hash"]
    try:
        if _client_factory is None:
            from openai import OpenAI
            _client_factory = OpenAI
        client = _client_factory(api_key=key, max_retries=0)
        request_client = client.with_options(
            timeout=EPRX_AI_REQUEST_TIMEOUT_SECONDS, max_retries=0
        ) if hasattr(client, "with_options") else client
    except (ImportError, ModuleNotFoundError) as exc:
        return {"status": "dependency_missing", "message": str(exc), "fallback": fallback}
    request_payload = {"expected_response_metadata": {"context_hash": context_hash, "model": selected_model},
                       "domain_constants": {"product": "1차 조정력", "interval_minutes": 30,
                                            "periods_per_day": 48, "days_per_week": 7,
                                            "expected_complete_week_rows": 336},
                       "analysis": built["payload"]}
    request_payload["metric_catalog"] = _numeric_metric_catalog(request_payload)
    instructions = (
        "일본 전력시장 실무자를 위한 EPRX 1차 조정력 주간 분석가로서 한국어로 답하세요. "
        "모집량 수준, 전주 변화, 시간대별 패턴을 먼저 설명한 뒤 전력수요와 잔여수요를 비교하세요. "
        "이번 주 핵심은 2~3문장, 모집량 패턴은 최대 3개, 계통 변수 연관성은 최대 3개, "
        "주의점은 최대 2개로 작성하세요. 상관계수를 나열하거나 같은 계열 변수를 반복하지 말고, "
        "인과관계로 표현하지 마세요. 유의성 검정을 하지 않았으므로 '유의'라는 표현을 쓰지 마세요. "
        "내부 변수명, 영어 한계 문구, 어려운 통계 용어는 쓰지 마세요. 숫자는 display_values의 문자열을 그대로 사용하고 "
        "재계산하거나 소수 자릿수를 늘리지 마세요. 정량 evidence는 metric_catalog의 metric_path와 값을 그대로 사용하세요."
    )
    request_text = json.dumps(request_payload, ensure_ascii=False, separators=(",", ":"))
    request_diagnostics = {"model": selected_model, "max_output_tokens": EPRX_AI_MAX_OUTPUT_TOKENS,
        "reasoning_effort": EPRX_AI_REASONING_EFFORT, "verbosity": EPRX_AI_VERBOSITY,
        "input_character_count": len(request_text), "timeout_seconds": EPRX_AI_REQUEST_TIMEOUT_SECONDS,
        "max_retries": 0, "application_max_attempts": max(1, _max_attempts),
        "payload_elapsed_seconds": payload_elapsed, "api_calls": 0}
    request = {"model": selected_model, "instructions": instructions,
        "input": request_text, "max_output_tokens": EPRX_AI_MAX_OUTPUT_TOKENS, "store": False,
        "reasoning": {"effort": EPRX_AI_REASONING_EFFORT},
        "text": {"verbosity": EPRX_AI_VERBOSITY},
        "text_format": _response_model()}
    last_error = None
    for attempt in range(max(1, _max_attempts)):
        try:
            request_diagnostics["api_calls"] += 1
            api_started = time.perf_counter()
            raw_resource = getattr(request_client.responses, "with_raw_response", None)
            if raw_resource is not None and hasattr(raw_resource, "parse"):
                raw_response = raw_resource.parse(**request)
                api_elapsed = time.perf_counter() - api_started
                parse_started = time.perf_counter()
                response = raw_response.parse()
                parse_elapsed = time.perf_counter() - parse_started
                retries_taken = getattr(raw_response, "retries_taken", 0)
            else:
                response = request_client.responses.parse(**request)
                api_elapsed = time.perf_counter() - api_started
                parse_elapsed = 0.0
                retries_taken = 0
            diagnostics = _response_diagnostics(response)
            diagnostics.update({"api_elapsed_seconds": api_elapsed,
                                "parse_elapsed_seconds": parse_elapsed,
                                "sdk_retries_taken": retries_taken})
            if diagnostics["response_status"] != "completed":
                return {"status": "api_incomplete_response",
                        "message": "AI 응답 생성이 완료되지 않았습니다. 다시 시도해 주세요.",
                        "diagnostics": diagnostics, "request_diagnostics": request_diagnostics, "fallback": fallback}
            refusal = _find_refusal(_value(response, "output", []))
            if refusal:
                return {"status": "api_refusal", "message": "OpenAI가 이 분석 요청에 응답하지 않았습니다.",
                        "diagnostics": diagnostics, "request_diagnostics": request_diagnostics, "fallback": fallback}
            parsed_object = _value(response, "output_parsed")
            if parsed_object is not None:
                parsed = parsed_object.model_dump() if hasattr(parsed_object, "model_dump") else dict(parsed_object)
            else:
                parsed = _parse_json_fallback(_value(response, "output_text", ""))
            validation_started = time.perf_counter()
            checked = validate_eprx_ai_response(parsed, expected_region=analysis_context["region"],
                expected_week_start=built["payload"]["week"]["start"], expected_context_hash=context_hash,
                input_payload=request_payload)
            diagnostics["validation_elapsed_seconds"] = time.perf_counter() - validation_started
            if not checked["valid"]:
                return {"status": "api_response_validation_error",
                        "message": "AI 응답 내용이 입력 데이터 검증을 통과하지 못했습니다. 다시 시도해 주세요.",
                        "errors": checked["errors"], "validation_diagnostics": checked.get("diagnostics", {}),
                        "diagnostics": diagnostics, "context_hash": context_hash,
                        "payload_diagnostics": built["diagnostics"], "request_diagnostics": request_diagnostics,
                        "fallback": fallback}
            return {**parsed, "status": "ok", "model": selected_model, "context_hash": context_hash,
                    "payload_diagnostics": built["diagnostics"], "request_diagnostics": request_diagnostics,
                    "response_diagnostics": diagnostics}
        except Exception as exc:
            last_error = exc
            name = type(exc).__name__.lower()
            transient = any(token in name for token in ("temporary", "timeout", "connection", "ratelimit", "internalserver"))
            if not transient or attempt + 1 >= max(1, _max_attempts): break
    result = _error_result(last_error, fallback)
    result["request_diagnostics"] = request_diagnostics
    return result
