"""Read-only diagnostics for the Tokyo/Chubu EPRX driver context pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.chuden_area_loader import load_chuden_area_data, join_eprx_chubu_with_chuden
from utils.eprx_ai_context import build_eprx_analysis_context, to_json_safe
from utils.eprx_driver_features import FEATURE_COLUMNS, build_eprx_driver_features
from utils.eprx_driver_statistics import build_eprx_statistical_context
from utils.eprx_loader import find_eprx_files, load_all_eprx_data
from utils.tepco_area_loader import load_tepco_area_data, join_eprx_tokyo_with_tepco


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=("Tokyo", "Chubu"), required=True)
    parser.add_argument("--week-start", required=True)
    parser.add_argument("--analysis-start")
    parser.add_argument("--analysis-end")
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    eprx_directory = ROOT / "data" / "eprx"
    eprx_files = find_eprx_files(eprx_directory)
    eprx, _, eprx_diagnostics = load_all_eprx_data(eprx_directory)
    if args.region == "Tokyo":
        grid, grid_diagnostics = load_tepco_area_data()
        join = join_eprx_tokyo_with_tepco
        grid_name = "TEPCO"
    else:
        grid, grid_diagnostics = load_chuden_area_data()
        join = join_eprx_chubu_with_chuden
        grid_name = "Chuden"

    if eprx.empty or grid.empty:
        result = {
            "analysis_type": "eprx_primary_reserve_driver_context",
            "status": "source_data_missing",
            "region": args.region,
            "week": {"start": args.week_start},
            "analysis_period": {"start": args.analysis_start, "end": args.analysis_end},
            "eprx_files": [str(path) for path in eprx_files],
            "grid_source": grid_name,
            "grid_files": grid_diagnostics.get("source_path", []).tolist()
            if "source_path" in grid_diagnostics else [],
            "eprx_file_diagnostics": eprx_diagnostics.to_dict("records"),
            "grid_file_diagnostics": grid_diagnostics.to_dict("records"),
            "reason": "EPRX or grid raw source data is not available.",
        }
    else:
        merged, join_diagnostics = join(eprx, grid)
        features, feature_diagnostics = build_eprx_driver_features(merged)
        procurement_context = build_eprx_analysis_context(features, args.region, args.week_start, grid_name)
        result = build_eprx_statistical_context(
            features, args.region, args.week_start,
            analysis_start=args.analysis_start, analysis_end=args.analysis_end,
            bootstrap_iterations=args.bootstrap_iterations, random_seed=args.random_seed,
        )
        result["procurement_context"] = procurement_context
        result["source_files"] = {
            "eprx": [str(path) for path in eprx_files],
            "grid": sorted(grid["source_path"].dropna().astype(str).unique().tolist())
            if "source_path" in grid else [],
        }
        result["pipeline_diagnostics"] = {
            "join": join_diagnostics,
            "features": feature_diagnostics,
            "standard_feature_columns": list(FEATURE_COLUMNS[:13]),
        }

    payload = json.dumps(to_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)
    print(payload)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
