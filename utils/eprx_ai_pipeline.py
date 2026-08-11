"""Local-only orchestration for the EPRX/grid analysis context."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from utils.chuden_area_loader import load_chuden_area_data
from utils.eprx_driver_features import build_eprx_driver_features
from utils.eprx_driver_statistics import (
    RUNTIME_BOOTSTRAP_ITERATIONS,
    build_eprx_fast_context,
    build_eprx_statistical_context,
)
from utils.tepco_area_loader import join_eprx_region_with_grid, load_tepco_area_data

RAW_DIRECTORIES = {
    "Tokyo": Path("data/external/tepco/raw"),
    "Chubu": Path("data/external/chuden/raw"),
}
MISSING_MESSAGES = {
    "Tokyo": "도쿄전력 PG의 30분 수급실적 원본 파일이 없습니다. 공식 사이트에서 받은 월별 CSV를 data/external/tepco/raw 폴더에 배치해 주세요.",
    "Chubu": "중부전력 PG의 30분 수급실적 원본 파일이 없습니다. 공식 사이트에서 받은 월별 CSV를 data/external/chuden/raw 폴더에 배치해 주세요.",
}


def local_grid_file_fingerprint(region: str, base_directory: Path | None = None) -> dict[str, Any]:
    if region not in RAW_DIRECTORIES:
        raise ValueError(f"Unsupported region: {region}")
    directory = Path(base_directory) if base_directory is not None else RAW_DIRECTORIES[region]
    files = sorted(directory.glob("*.csv")) if directory.exists() else []
    entries = []
    for path in files:
        stat = path.stat()
        entries.append({"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    signature = "|".join(f"{x['name']}:{x['size']}:{x['mtime_ns']}" for x in entries)
    return {"fingerprint": hashlib.sha256(signature.encode()).hexdigest(), "files": entries}


def _selected_week_grid_paths(region: str, week_start: Any,
                              base_directory: Path | None = None) -> list[Path]:
    directory = Path(base_directory) if base_directory is not None else RAW_DIRECTORIES[region]
    if not directory.exists():
        return []
    start = pd.Timestamp(week_start).normalize()
    months = {(start + pd.Timedelta(days=offset)).strftime("%Y%m") for offset in range(7)}
    paths = sorted(directory.glob("*.csv"))
    matched = [path for path in paths if any(month in path.name for month in months)]
    if any(re.search(r"20\d{4}", path.name) for path in paths):
        return matched
    return paths


def local_grid_week_fingerprint(region: str, week_start: Any,
                                base_directory: Path | None = None) -> dict[str, Any]:
    entries = []
    for path in _selected_week_grid_paths(region, week_start, base_directory):
        stat = path.stat()
        entries.append({"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    signature = "|".join(f"{x['name']}:{x['size']}:{x['mtime_ns']}" for x in entries)
    return {"fingerprint": hashlib.sha256(signature.encode()).hexdigest(), "files": entries}


def check_eprx_ai_readiness(eprx_df: pd.DataFrame, region: str, week_start: Any,
                            base_directory: Path | None = None) -> dict[str, Any]:
    """Check selected-week completeness without building statistical context."""
    if region not in RAW_DIRECTORIES:
        return {"status": "unsupported_region", "ready": False, "region": region}
    fingerprint = local_grid_week_fingerprint(region, week_start, base_directory)
    if not fingerprint["files"]:
        return {"status": "source_data_missing", "ready": False, "message": MISSING_MESSAGES[region],
                "region": region, "file_fingerprint": fingerprint["fingerprint"], "source_files": []}
    paths = _selected_week_grid_paths(region, week_start, base_directory)
    loader = load_tepco_area_data if region == "Tokyo" else load_chuden_area_data
    grid_df, diagnostics = loader(paths)
    accepted = set(diagnostics.loc[diagnostics["status"].eq("Loaded"), "source_file"].astype(str))
    grid_df = grid_df.loc[grid_df["source_file"].astype(str).isin(accepted)].copy() if not grid_df.empty else grid_df
    if grid_df.empty:
        return {"status": "source_validation_failed", "ready": False,
                "message": "검증을 통과한 계통실적 CSV가 없습니다.", "region": region,
                "file_fingerprint": fingerprint["fingerprint"], "source_files": sorted(accepted)}
    try:
        merged, join = join_eprx_region_with_grid(eprx_df, grid_df, region, week_start)
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "analysis_unavailable", "ready": False, "message": str(exc),
                "region": region, "file_fingerprint": fingerprint["fingerprint"],
                "source_files": sorted(accepted)}
    keys = ["delivery_date", "period_no", "period_start"]
    eprx_duplicates = int(merged.loc[merged["join_status"].ne("right_only")].duplicated(keys, keep=False).sum())
    grid_duplicates = int(merged.loc[merged["join_status"].ne("left_only")].duplicated(keys, keep=False).sum())
    eprx_unique = int(merged.loc[merged["join_status"].ne("right_only"), keys].drop_duplicates().shape[0])
    grid_unique = int(merged.loc[merged["join_status"].ne("left_only"), keys].drop_duplicates().shape[0])
    matched = int(join["matched_rows"])
    rate = matched / max(join["eprx_rows"], join["tepco_rows"], 1)
    complete = (join["eprx_rows"] == 336 and eprx_unique == 336 and join["tepco_rows"] == 336
                and grid_unique == 336 and matched == 336 and rate == 1.0
                and eprx_duplicates == 0 and grid_duplicates == 0)
    return {"status": "ok" if complete else "join_incomplete", "ready": complete,
            "message": "분석 가능한 완전 주차입니다." if complete else "선택 주차의 EPRX·계통실적 결합률이 100%가 아닙니다.",
            "region": region, "file_fingerprint": fingerprint["fingerprint"],
            "source_files": sorted(accepted), "latest_source_date": str(pd.to_datetime(grid_df["delivery_date"]).max().date()),
            "eprx_rows": int(join["eprx_rows"]), "eprx_unique_timestamps": eprx_unique,
            "grid_rows": int(join["tepco_rows"]), "grid_unique_timestamps": grid_unique,
            "matched_timestamps": matched, "join_rate": rate,
            "eprx_duplicate_timestamps": eprx_duplicates, "grid_duplicate_timestamps": grid_duplicates,
            "complete_week": complete}


def load_local_eprx_grid_context(
    eprx_df: pd.DataFrame, region: str, week_start: Any,
    base_directory: Path | None = None,
    bootstrap_iterations: int = RUNTIME_BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Read manually placed files; never downloads or mutates a source file."""
    fingerprint = local_grid_file_fingerprint(region, base_directory)
    if not fingerprint["files"]:
        return {"status": "source_data_missing", "message": MISSING_MESSAGES[region],
                "region": region, "file_fingerprint": fingerprint["fingerprint"], "source_files": []}
    directory = Path(base_directory) if base_directory is not None else RAW_DIRECTORIES[region]
    loader = load_tepco_area_data if region == "Tokyo" else load_chuden_area_data
    grid_df, diagnostics = loader(directory)
    accepted = set(diagnostics.loc[diagnostics["status"].eq("Loaded"), "source_file"].astype(str))
    excluded = diagnostics.loc[~diagnostics["status"].eq("Loaded")]
    grid_df = grid_df.loc[grid_df["source_file"].astype(str).isin(accepted)].copy() if not grid_df.empty else grid_df
    safe_diagnostics = diagnostics.drop(columns=["source_path"], errors="ignore").to_dict("records")
    if grid_df.empty:
        return {"status": "source_validation_failed", "message": "검증을 통과한 계통실적 CSV가 없습니다.",
                "region": region, "file_fingerprint": fingerprint["fingerprint"],
                "source_files": sorted(accepted), "file_diagnostics": safe_diagnostics,
                "excluded_file_count": len(excluded)}
    try:
        selected_merged, join_diagnostics = join_eprx_region_with_grid(
            eprx_df, grid_df, region, week_start
        )
        complete = (
            len(selected_merged) == 336
            and join_diagnostics.get("all_rows_matched") is True
        )
        # Keep one feature history so the opt-in detailed analysis can reuse it.
        historical_merged, _ = join_eprx_region_with_grid(eprx_df, grid_df, region)
        features, feature_diagnostics = build_eprx_driver_features(historical_merged)
        if not complete:
            return {"status": "join_incomplete", "message": "선택 주차의 EPRX·계통실적 결합률이 100%가 아닙니다.",
                    "region": region, "file_fingerprint": fingerprint["fingerprint"],
                    "source_files": sorted(accepted), "file_diagnostics": safe_diagnostics,
                    "join_diagnostics": join_diagnostics, "feature_diagnostics": feature_diagnostics}
        context = build_eprx_fast_context(features, region, week_start)
        return {"status": "ok", "message": "분석 컨텍스트를 생성했습니다.", "region": region,
                "file_fingerprint": fingerprint["fingerprint"], "source_files": sorted(accepted),
                "latest_source_date": str(pd.to_datetime(grid_df["delivery_date"]).max().date()),
                "file_diagnostics": safe_diagnostics, "join_diagnostics": join_diagnostics,
                "feature_diagnostics": feature_diagnostics, "analysis_context": context,
                "feature_history": features}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "analysis_unavailable", "message": str(exc), "region": region,
                "file_fingerprint": fingerprint["fingerprint"], "source_files": sorted(accepted),
                "file_diagnostics": safe_diagnostics}


def build_detailed_context(local_context: dict[str, Any], region: str, week_start: Any,
                           bootstrap_iterations: int = RUNTIME_BOOTSTRAP_ITERATIONS) -> dict[str, Any]:
    """Build opt-in heavy statistics from already parsed and joined feature history."""
    features = local_context.get("feature_history")
    if not isinstance(features, pd.DataFrame):
        raise ValueError("feature_history_missing")
    return build_eprx_statistical_context(
        features, region, week_start, bootstrap_iterations=bootstrap_iterations)
