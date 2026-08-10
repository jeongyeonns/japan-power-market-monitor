"""Local-only orchestration for the EPRX/grid analysis context."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from utils.chuden_area_loader import load_chuden_area_data
from utils.eprx_driver_features import build_eprx_driver_features
from utils.eprx_driver_statistics import build_eprx_statistical_context
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


def load_local_eprx_grid_context(
    eprx_df: pd.DataFrame, region: str, week_start: Any,
    base_directory: Path | None = None, bootstrap_iterations: int = 500,
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
        # Completeness is a selected-week gate, but weekday/time-block anomalies and
        # regressions require the full locally available history as their baseline.
        historical_merged, _ = join_eprx_region_with_grid(eprx_df, grid_df, region)
        features, feature_diagnostics = build_eprx_driver_features(historical_merged)
        if not complete:
            return {"status": "join_incomplete", "message": "선택 주차의 EPRX·계통실적 결합률이 100%가 아닙니다.",
                    "region": region, "file_fingerprint": fingerprint["fingerprint"],
                    "source_files": sorted(accepted), "file_diagnostics": safe_diagnostics,
                    "join_diagnostics": join_diagnostics, "feature_diagnostics": feature_diagnostics}
        context = build_eprx_statistical_context(features, region, week_start,
                                                 bootstrap_iterations=bootstrap_iterations)
        return {"status": "ok", "message": "분석 컨텍스트를 생성했습니다.", "region": region,
                "file_fingerprint": fingerprint["fingerprint"], "source_files": sorted(accepted),
                "latest_source_date": str(pd.to_datetime(grid_df["delivery_date"]).max().date()),
                "file_diagnostics": safe_diagnostics, "join_diagnostics": join_diagnostics,
                "feature_diagnostics": feature_diagnostics, "analysis_context": context}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "analysis_unavailable", "message": str(exc), "region": region,
                "file_fingerprint": fingerprint["fingerprint"], "source_files": sorted(accepted),
                "file_diagnostics": safe_diagnostics}
