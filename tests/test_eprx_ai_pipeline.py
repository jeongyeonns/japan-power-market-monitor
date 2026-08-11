from pathlib import Path

import pandas as pd
import pytest

from utils import eprx_ai_pipeline
from utils.eprx_ai_pipeline import (
    check_eprx_ai_readiness,
    load_local_eprx_grid_context,
    local_grid_file_fingerprint,
    local_grid_week_fingerprint,
)


def test_missing_local_raw_files_are_safe_for_both_regions(tmp_path):
    for region, company in (("Tokyo", "도쿄전력"), ("Chubu", "중부전력")):
        result = load_local_eprx_grid_context(pd.DataFrame(), region, "2026-03-14", tmp_path)
        assert result["status"] == "source_data_missing"
        assert company in result["message"]
        assert result["source_files"] == []


def test_fingerprint_uses_name_size_and_mtime_without_absolute_path(tmp_path):
    path = tmp_path / "month.csv"; path.write_text("x", encoding="utf-8")
    value = local_grid_file_fingerprint("Tokyo", tmp_path)
    assert value["files"][0]["name"] == "month.csv"
    assert "path" not in value["files"][0]
    first = value["fingerprint"]
    path.write_text("xx", encoding="utf-8")
    assert local_grid_file_fingerprint("Tokyo", tmp_path)["fingerprint"] != first


def test_selected_week_gates_completeness_but_regression_features_use_history(tmp_path, monkeypatch):
    (tmp_path / "month.csv").write_text("fixture", encoding="utf-8")
    grid = pd.DataFrame({"source_file": ["month.csv"], "delivery_date": [pd.Timestamp("2026-07-31")]})
    diagnostics = pd.DataFrame([{"source_file": "month.csv", "status": "Loaded"}])
    monkeypatch.setattr(eprx_ai_pipeline, "load_tepco_area_data", lambda _path: (grid, diagnostics))
    calls = []

    def fake_join(_eprx, _grid, _region, week_start=None):
        calls.append(week_start)
        rows = 336 if week_start is not None else 1_000
        return pd.DataFrame({"scope": ["selected" if week_start is not None else "history"] * rows}), {
            "eprx_rows": 336, "tepco_rows": 336, "matched_rows": 336,
            "all_rows_matched": True,
        }

    monkeypatch.setattr(eprx_ai_pipeline, "join_eprx_region_with_grid", fake_join)
    monkeypatch.setattr(eprx_ai_pipeline, "build_eprx_driver_features",
                        lambda frame: (frame, {"feature_input_rows": len(frame)}))
    monkeypatch.setattr(eprx_ai_pipeline, "build_eprx_statistical_context",
                        lambda frame, *_args, **_kwargs: {"history_rows": len(frame)})
    result = load_local_eprx_grid_context(pd.DataFrame(), "Tokyo", "2026-07-20", tmp_path)
    assert result["status"] == "ok"
    assert calls == ["2026-07-20", None]
    assert result["analysis_context"]["history_rows"] == 1_000


def test_lightweight_readiness_checks_complete_week_without_heavy_statistics(tmp_path, monkeypatch):
    path = tmp_path / "eria_jukyu_202607_03.csv"; path.write_text("fixture", encoding="utf-8")
    grid = pd.DataFrame({"source_file": [path.name], "delivery_date": [pd.Timestamp("2026-07-20")]})
    diagnostics = pd.DataFrame([{"source_file": path.name, "status": "Loaded"}])
    monkeypatch.setattr(eprx_ai_pipeline, "load_tepco_area_data", lambda _paths: (grid, diagnostics))
    timestamps = pd.date_range("2026-07-20", periods=336, freq="30min")
    merged = pd.DataFrame({"delivery_date": timestamps.normalize(),
        "period_no": timestamps.hour * 2 + timestamps.minute // 30 + 1,
        "period_start": timestamps.strftime("%H:%M"), "join_status": "both"})
    monkeypatch.setattr(eprx_ai_pipeline, "join_eprx_region_with_grid", lambda *_args: (merged, {
        "eprx_rows": 336, "tepco_rows": 336, "matched_rows": 336, "all_rows_matched": True}))
    monkeypatch.setattr(eprx_ai_pipeline, "build_eprx_statistical_context",
                        lambda *_args, **_kwargs: pytest.fail("heavy statistics called during readiness"))
    result = check_eprx_ai_readiness(pd.DataFrame(), "Tokyo", "2026-07-20", tmp_path)
    assert result["ready"] is True
    assert result["eprx_rows"] == result["grid_rows"] == result["matched_timestamps"] == 336


def test_readiness_missing_grid_is_false_and_week_fingerprint_changes(tmp_path):
    missing = check_eprx_ai_readiness(pd.DataFrame(), "Tokyo", "2026-07-20", tmp_path)
    assert missing["ready"] is False and missing["status"] == "source_data_missing"
    path = tmp_path / "eria_jukyu_202607_03.csv"; path.write_text("x", encoding="utf-8")
    first = local_grid_week_fingerprint("Tokyo", "2026-07-20", tmp_path)["fingerprint"]
    path.write_text("xx", encoding="utf-8")
    assert local_grid_week_fingerprint("Tokyo", "2026-07-20", tmp_path)["fingerprint"] != first
