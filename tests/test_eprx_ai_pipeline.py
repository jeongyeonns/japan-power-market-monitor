from pathlib import Path

import pandas as pd

from utils.eprx_ai_pipeline import load_local_eprx_grid_context, local_grid_file_fingerprint


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
