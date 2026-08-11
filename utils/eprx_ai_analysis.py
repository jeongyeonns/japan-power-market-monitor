"""Cost-bounded OpenAI adapter for precomputed EPRX statistics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MODEL = "gpt-5-mini"
MAX_INPUT_CHARACTERS = 30000
EPRX_AI_MAX_OUTPUT_TOKENS = 3000
EPRX_AI_REASONING_EFFORT = "minimal"
EPRX_AI_VERBOSITY = "low"
EVIDENCE_REL_TOLERANCE = 0.02
EVIDENCE_ABS_TOLERANCE = 0.005
PAYLOAD_VERSION = "1.0"
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
    def compact_correlations(source: dict[str, Any] | None, limit: int) -> dict[str, Any]:
        rows = []
        for name, relation in (source or {}).items():
            score = relation.get("spearman")
            if score is None: continue
            rows.append((abs(score), name, {key: relation.get(key) for key in
                ("status", "sample_count", "pearson", "spearman", "pearson_direction",
                 "pearson_strength", "spearman_direction", "spearman_strength")}))
        rows.sort(key=lambda row: (-row[0], row[1]))
        return {name: values for _, name, values in rows[:limit]}

    adjusted = compact_correlations(analysis_context.get("time_adjusted_correlations"), 5)
    bootstrap = {}
    for name in adjusted:
        interval = analysis_context.get("bootstrap_intervals", {}).get(name, {}).get("anomaly_spearman")
        if interval:
            bootstrap[name] = {"anomaly_spearman": {key: interval.get(key) for key in
                ("estimate", "ci_95_lower", "ci_95_upper", "bootstrap_iterations_valid", "status")}}
    regressions = [{key: model.get(key) for key in ("model_name", "status", "sample_count",
                    "r_squared", "adjusted_r_squared", "standardized_coefficients", "warnings")}
                   for model in analysis_context.get("regression_models", [])]
    quality = {key: value for key, value in analysis_context.get("data_quality", {}).items()
               if not isinstance(value, (dict, list))}
    notable = selected.get("notable_time_blocks") or {}
    notable = {"highest": notable.get("highest", [])[:3], "lowest": notable.get("lowest", [])[:3]}
    repetition = analysis_context.get("profile_repetition") or {}
    repetition = {key: repetition.get(key) for key in ("status", "complete_week_count",
        "weeks_identical_to_previous_count", "unique_weekly_profile_count") if key in repetition}
    payload = _json_safe({
        "region": analysis_context["region"], "week": selected["week"],
        "analysis_period": analysis_context.get("analysis_period"),
        "procurement": selected.get("procurement"),
        "daily_profile": selected.get("daily_profile"),
        "notable_time_blocks": notable,
        "historical_position": selected.get("historical_position"),
        "procurement_change": selected.get("procurement_change"),
        "driver_changes": selected.get("driver_changes"),
        "raw_correlations": compact_correlations(analysis_context.get("raw_correlations"), 3),
        "time_adjusted_correlations": adjusted,
        "bootstrap_intervals": bootstrap,
        "regression_models": regressions,
        "association_candidates": selected.get("association_candidates"),
        "profile_repetition": repetition,
        "data_quality": quality, "warnings": list(dict.fromkeys(analysis_context.get("warnings", []))),
        "limitations": list(dict.fromkeys(analysis_context.get("limitations", [])))[:3],
    })
    if selected.get("co_movement_comparison"):
        comparison = selected["co_movement_comparison"]
        candidate_names = [item.get("variable") for item in
                           (selected.get("association_candidates") or {}).get("items", [])[:3]]
        payload["co_movement_comparison"] = _json_safe({"status": comparison.get("status"),
            "quantile_tie_warning": comparison.get("quantile_tie_warning"),
            "variables": {name: {key: comparison.get("variables", {}).get(name, {}).get(key)
                for key in ("high_group_count", "low_group_count", "high_group_mean",
                            "low_group_mean", "mean_difference", "mean_difference_pct")}
                for name in candidate_names if name in comparison.get("variables", {})}})
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
    for field in REQUIRED_RESPONSE_FIELDS:
        if field not in response_data: errors.append(f"missing_{field}")
    list_fields = ("confirmed_findings", "association_candidates", "counter_evidence", "data_quality_notes", "limitations")
    for field in list_fields:
        if field in response_data and not isinstance(response_data[field], list): errors.append(f"invalid_type_{field}")
    for field in ("headline", "summary", "statistical_interpretation", "profile_warning", "conclusion", "disclaimer"):
        if field in response_data and not isinstance(response_data[field], str): errors.append(f"invalid_type_{field}")
    if expected_region and response_data.get("region") != expected_region: errors.append("region_mismatch")
    if expected_week_start and response_data.get("week_start") != expected_week_start: errors.append("week_start_mismatch")
    if expected_context_hash and response_data.get("context_hash") != expected_context_hash: errors.append("context_hash_mismatch")
    evidence_errors = []
    if input_payload is not None:
        for field in ("confirmed_findings", "association_candidates"):
            for index, evidence in enumerate(response_data.get(field, [])):
                issue = _validate_evidence(evidence, input_payload)
                if issue:
                    evidence_errors.append({"response_field": field, "index": index, **issue})
        if evidence_errors:
            errors.append("invalid_structured_evidence")
            diagnostics["evidence_errors"] = evidence_errors
        narrative_fields = ("headline", "summary", "statistical_interpretation", "counter_evidence",
                            "profile_warning", "data_quality_notes", "limitations", "conclusion", "disclaimer")
        diagnostics["narrative_numeric_literal_count"] = sum(
            len(re.findall(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?", json.dumps(response_data.get(field), ensure_ascii=False)))
            for field in narrative_fields
        )
    return {"valid": not errors, "errors": errors,
            "diagnostics": diagnostics,
            "numeric_validation_scope": "schema and identity fields only; free-text numeric provenance cannot be proven reliably"}


def _schema() -> dict[str, Any]:
    properties = {field: {"type": "string"} for field in ("status", "region", "week_start", "headline", "summary", "profile_warning", "conclusion", "disclaimer", "model", "context_hash")}
    for field in ("confirmed_findings", "statistical_interpretation", "association_candidates", "counter_evidence", "data_quality_notes", "limitations"):
        properties[field] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": properties, "required": list(REQUIRED_RESPONSE_FIELDS), "additionalProperties": False}


def _response_model():
    """Create the Pydantic schema lazily so non-AI imports remain safe."""
    from pydantic import ConfigDict, create_model

    Evidence = create_model("EprxEvidence", __config__=ConfigDict(extra="forbid", strict=True),
        metric_path=(str, ...), display_name=(str, ...), value=(float, ...),
        unit=(str, ...), interpretation=(str, ...))
    return create_model("EprxAiResponse", __config__=ConfigDict(extra="forbid", strict=True),
        status=(str, ...), region=(str, ...), week_start=(str, ...), headline=(str, ...),
        summary=(str, ...), confirmed_findings=(list[Evidence], ...),
        statistical_interpretation=(str, ...), association_candidates=(list[Evidence], ...),
        counter_evidence=(list[str], ...), profile_warning=(str, ...),
        data_quality_notes=(list[str], ...), limitations=(list[str], ...),
        conclusion=(str, ...), disclaimer=(str, ...), context_hash=(str, ...))


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
    if "authentication" in lower or status_code == 401:
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
    built = build_eprx_ai_payload(analysis_context); context_hash = built["diagnostics"]["context_hash"]
    try:
        if _client_factory is None:
            from openai import OpenAI
            _client_factory = OpenAI
        client = _client_factory(api_key=key, max_retries=0)
    except (ImportError, ModuleNotFoundError) as exc:
        return {"status": "dependency_missing", "message": str(exc), "fallback": fallback}
    instructions = ("일본 전력시장과 EPRX 1차 조정력 주간 데이터를 검토하는 전력시장 분석가로서 한국어로 작성하라. "
        "전력시장 실무자가 30초 안에 읽을 수 있게 이번 주 모집량 자체의 평균·범위·전주 대비 변화·시간대 패턴을 먼저 설명하고, "
        "그다음 역사적 위치와 수요·재생에너지·잔여수요의 관계를 설명하라. raw correlation보다 시간대 조정 상관, "
        "Pearson·Spearman 방향 일치, bootstrap 구간, association relevance, leave-one-out 및 상·하위 모집량 비교를 우선하라. "
        "비슷한 파생변수나 같은 태양광 변수군을 중복 나열하지 말고 시장 해석에 유용한 관계만 최대 3개 고르라. "
        "제공된 계산값만 사용하고 상관관계를 영향이나 인과로 표현하지 말라. 실제 유의성 검정 결과가 없으므로 '유의', "
        "'유의한 영향', '통계적으로 유의'라는 표현을 사용하지 말라. 미래 모집량, 비공개 구성비, 자연체여력 또는 "
        "수의계약량을 추정하지 말라. 내부 영문 변수명은 쓰지 말고 자연스러운 한국어 지표명을 사용하라. "
        "수요는 공개 30분 실적이고 의사결정 당시 예측자료와 다를 수 있음을 명시하라.")
    request_payload = {"expected_response_metadata": {"context_hash": context_hash, "model": selected_model},
                       "domain_constants": {"product": "1차 조정력", "interval_minutes": 30,
                                            "periods_per_day": 48, "days_per_week": 7,
                                            "expected_complete_week_rows": 336},
                       "analysis": built["payload"]}
    request_payload["metric_catalog"] = _numeric_metric_catalog(request_payload)
    instructions += (" 정량적 근거는 metric_catalog에 있는 metric_path, value, unit을 그대로 사용해 "
        "confirmed_findings와 association_candidates에 기록하라. 새로운 계산값을 만들지 말고, "
        "각 evidence는 metric_path, display_name, value, unit, interpretation 5개 필드를 모두 포함하고 value는 숫자여야 한다. "
        "정량 근거가 없는 설명은 evidence 배열이 아니라 자유서술 필드에 기록하라. statistical_interpretation은 하나의 문자열이다. "
        "headline은 1문장, summary는 최대 2문장이며 모집량 특징부터 시작하라. confirmed_findings는 최대 3개이며 모집량 패턴만 담고, "
        "statistical_interpretation은 이번 주 핵심 해석을 최대 2문장, association_candidates는 서로 다른 변수군의 계통 연관성만 최대 3개, "
        "counter_evidence는 통계적 주의점만 최대 3개로 작성하라. 정상 데이터의 data_quality_notes는 '336/336, 결합률 100%' 수준의 1개 문장으로 제한하라. "
        "profile_warning은 최대 2문장, limitations는 현재 분석에 중요한 한국어 문장 최대 3개, conclusion은 최대 2문장, disclaimer는 1문장으로 작성하라. "
        "동일 내용을 여러 필드에서 반복하지 말고 숫자 근거는 evidence에서 한 번만 제시하며 summary에서 모두 재나열하지 말라. "
        "상관계수는 단위가 없으며 coefficient나 %를 단위처럼 설명하지 말라. 자유서술에서는 불필요하게 숫자를 반복하지 말라. "
        "|r| 0.4 이상 0.6 미만은 중간 수준, 0.6 이상 0.8 미만은 강한 관계로 표현하라. 표시 반올림은 근거 value와 의미가 같게 제한하라.")
    request_text = json.dumps(request_payload, ensure_ascii=False)
    request_diagnostics = {"model": selected_model, "max_output_tokens": EPRX_AI_MAX_OUTPUT_TOKENS,
        "reasoning_effort": EPRX_AI_REASONING_EFFORT, "verbosity": EPRX_AI_VERBOSITY,
        "input_character_count": len(request_text)}
    request = {"model": selected_model, "instructions": instructions,
        "input": request_text, "max_output_tokens": EPRX_AI_MAX_OUTPUT_TOKENS, "store": False,
        "reasoning": {"effort": EPRX_AI_REASONING_EFFORT},
        "text": {"verbosity": EPRX_AI_VERBOSITY},
        "text_format": _response_model()}
    last_error = None
    for attempt in range(max(1, _max_attempts)):
        try:
            response = client.responses.parse(**request)
            diagnostics = _response_diagnostics(response)
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
            checked = validate_eprx_ai_response(parsed, expected_region=analysis_context["region"],
                expected_week_start=built["payload"]["week"]["start"], expected_context_hash=context_hash,
                input_payload=request_payload)
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
