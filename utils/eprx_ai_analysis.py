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
PAYLOAD_VERSION = "1.0"
REQUIRED_RESPONSE_FIELDS = ("status", "region", "week_start", "headline", "summary",
    "confirmed_findings", "statistical_interpretation", "association_candidates",
    "counter_evidence", "profile_warning", "data_quality_notes", "limitations",
    "conclusion", "disclaimer", "model", "context_hash")


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
    payload = _json_safe({
        "region": analysis_context["region"], "week": selected["week"],
        "analysis_period": analysis_context.get("analysis_period"),
        "procurement_change": selected.get("procurement_change"),
        "driver_changes": selected.get("driver_changes"),
        "raw_correlations": analysis_context.get("raw_correlations"),
        "time_adjusted_correlations": analysis_context.get("time_adjusted_correlations"),
        "bootstrap_intervals": analysis_context.get("bootstrap_intervals"),
        "regression_models": analysis_context.get("regression_models"),
        "co_movement_comparison": selected.get("co_movement_comparison"),
        "association_candidates": selected.get("association_candidates"),
        "profile_repetition": analysis_context.get("profile_repetition"),
        "data_quality": analysis_context.get("data_quality"), "warnings": analysis_context.get("warnings", []),
        "limitations": analysis_context.get("limitations", []), "calculation_metadata": analysis_context.get("calculation_metadata"),
    })
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
    if not isinstance(response_data, dict): return {"valid": False, "errors": ["response_must_be_object"]}
    for field in REQUIRED_RESPONSE_FIELDS:
        if field not in response_data: errors.append(f"missing_{field}")
    list_fields = ("confirmed_findings", "statistical_interpretation", "association_candidates", "counter_evidence", "data_quality_notes", "limitations")
    for field in list_fields:
        if field in response_data and not isinstance(response_data[field], list): errors.append(f"invalid_type_{field}")
    for field in ("headline", "summary", "profile_warning", "conclusion", "disclaimer"):
        if field in response_data and not isinstance(response_data[field], str): errors.append(f"invalid_type_{field}")
    if expected_region and response_data.get("region") != expected_region: errors.append("region_mismatch")
    if expected_week_start and response_data.get("week_start") != expected_week_start: errors.append("week_start_mismatch")
    if expected_context_hash and response_data.get("context_hash") != expected_context_hash: errors.append("context_hash_mismatch")
    # This is deliberately lexical: it catches invented numeric literals but cannot
    # prove that a paraphrased qualitative statement follows from the input.
    if input_payload is not None:
        pattern = r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?"
        allowed = set(re.findall(pattern, json.dumps(input_payload, ensure_ascii=False)))
        narrative = {k: v for k, v in response_data.items() if k not in {"context_hash", "model"}}
        unsupported = sorted(set(re.findall(pattern, json.dumps(narrative, ensure_ascii=False))) - allowed)
        if unsupported: errors.append("unsupported_numeric_literal")
    return {"valid": not errors, "errors": errors,
            "numeric_validation_scope": "schema and identity fields only; free-text numeric provenance cannot be proven reliably"}


def _schema() -> dict[str, Any]:
    properties = {field: {"type": "string"} for field in ("status", "region", "week_start", "headline", "summary", "profile_warning", "conclusion", "disclaimer", "model", "context_hash")}
    for field in ("confirmed_findings", "statistical_interpretation", "association_candidates", "counter_evidence", "data_quality_notes", "limitations"):
        properties[field] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": properties, "required": list(REQUIRED_RESPONSE_FIELDS), "additionalProperties": False}


def generate_eprx_ai_analysis(analysis_context: dict[str, Any], api_key: str | None = None,
                              model: str | None = None, _client_factory=None) -> dict[str, Any]:
    validation = validate_eprx_ai_context(analysis_context)
    fallback = build_eprx_statistical_fallback(analysis_context) if validation["valid"] else None
    if not validation["valid"]: return {"status": "invalid_context", "errors": validation["errors"]}
    key, selected_model = resolve_openai_settings(api_key, model)
    if not key: return {"status": "api_key_missing", "message": "OpenAI API 키가 설정되어 있지 않습니다.", "fallback": fallback}
    built = build_eprx_ai_payload(analysis_context); context_hash = built["diagnostics"]["context_hash"]
    try:
        if _client_factory is None:
            from openai import OpenAI
            _client_factory = OpenAI
        client = _client_factory(api_key=key)
    except (ImportError, ModuleNotFoundError) as exc:
        return {"status": "dependency_missing", "message": str(exc), "fallback": fallback}
    instructions = ("한국어로 간결하게 작성하라. 제공된 계산값만 사용하고 상관관계를 인과로 표현하지 말라. "
        "미래 모집량, 비공개 구성비, 자연체여력 또는 수의계약량을 추정하지 말라. 수요는 공개 30분 실적이고 의사결정 당시 예측자료와 다를 수 있음을 명시하라.")
    request_payload = {"expected_response_metadata": {"context_hash": context_hash, "model": selected_model},
                       "analysis": built["payload"]}
    request = {"model": selected_model, "instructions": instructions,
        "input": json.dumps(request_payload, ensure_ascii=False), "max_output_tokens": 1800, "store": False,
        "text": {"format": {"type": "json_schema", "name": "eprx_analysis", "schema": _schema(), "strict": True}}}
    last_error = None
    for attempt in range(2):
        try:
            response = client.responses.create(**request)
            parsed = json.loads(response.output_text)
            checked = validate_eprx_ai_response(parsed, expected_region=analysis_context["region"],
                expected_week_start=built["payload"]["week"]["start"], expected_context_hash=context_hash,
                input_payload=request_payload)
            if not checked["valid"]: return {"status": "invalid_response", "errors": checked["errors"], "fallback": fallback}
            return {**parsed, "status": "ok", "model": selected_model, "context_hash": context_hash,
                    "payload_diagnostics": built["diagnostics"]}
        except Exception as exc:
            last_error = exc
            name = type(exc).__name__.lower()
            transient = any(token in name for token in ("temporary", "timeout", "connection", "ratelimit", "internalserver"))
            if not transient or attempt == 1: break
    return {"status": "api_call_failed", "message": f"{type(last_error).__name__}: API 호출에 실패했습니다.", "fallback": fallback}
