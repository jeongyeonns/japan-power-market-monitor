"""Manual CLI for official TEPCO/Chuden monthly grid actuals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from utils.grid_data_downloader import SequentialHttpClient, atomic_write_bytes, download_validate_store, require_official_domain
import utils.tepco_area_downloader as tepco
import utils.chuden_area_downloader as chuden


PROVIDERS = {"Tokyo": tepco, "Chubu": chuden}


def _month(value: str) -> str:
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", value):
        raise argparse.ArgumentTypeError("month must be YYYY-MM")
    return value


def _write_report(path: Path | None, report: dict) -> None:
    if path:
        atomic_write_bytes(path, (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def _validate_local(module) -> list[dict]:
    results = []
    validator = module.validate_tepco_bytes if module is tepco else module.validate_chuden_bytes
    for path in sorted(module.RAW_DIRECTORY.glob("*.csv")):
        match = re.search(r"(20\d{2})[-_]?((?:0[1-9]|1[0-2]))", path.name)
        if not match:
            results.append({"file": path.name, "status": "failed", "reason": "target_month_not_in_filename"}); continue
        year_month = f"{match.group(1)}-{match.group(2)}"
        try:
            validation = validator(path.read_bytes(), path.name, year_month, "text/csv")
            results.append({"file": path.name, "status": "valid", "validation": validation})
        except Exception as exc:
            results.append({"file": path.name, "status": "failed", "reason": str(exc)})
    return results


def run(args: argparse.Namespace, client: SequentialHttpClient | None = None) -> tuple[int, dict]:
    regions = list(PROVIDERS) if args.region == "all" else [args.region]
    report = {"status": "success", "mode": "validate-only" if args.validate_only else ("dry-run" if args.dry_run else "download"), "providers": {}}
    failures = False
    if not args.dry_run and not args.validate_only and os.environ.get("GRID_DATA_DOWNLOAD_APPROVED", "").lower() != "true":
        report.update({"status": "blocked_terms_approval_required", "reason":
            "Official site terms require prior approval for use beyond personal use; set GRID_DATA_DOWNLOAD_APPROVED=true only after approval."})
        return 2, report
    client = client or SequentialHttpClient()
    for region in regions:
        module = PROVIDERS[region]
        provider_report = {"landing_page": module.LANDING_PAGE, "terms_page": module.TERMS_PAGE, "links": [], "results": []}
        report["providers"][region] = provider_report
        if args.validate_only:
            provider_report["results"] = _validate_local(module)
            failures |= any(item["status"] == "failed" for item in provider_report["results"])
            continue
        try:
            response = client.get(module.LANDING_PAGE)
            require_official_domain(response.get("url", module.LANDING_PAGE), module.ALLOWED_HOSTS)
            content_type = next((value for key, value in response["headers"].items() if key.lower() == "content-type"), "")
            if response["status"] != 200 or "html" not in content_type.lower(): raise ValueError("landing_page_not_html")
            html = response["body"].decode("utf-8", errors="replace")
            links = module.discover_tepco_links(html) if module is tepco else module.discover_chuden_links(html)
            start = args.from_month
            end = args.to_month
            if args.latest and links: start = end = max(item.target_year_month for item in links)
            links = [item for item in links if item.target_year_month >= start and (end is None or item.target_year_month <= end)]
            provider_report["links"] = [item.__dict__ for item in links]
            if args.dry_run: continue
            validator = module.validate_tepco_bytes if module is tepco else module.validate_chuden_bytes
            for link in links:
                try:
                    provider_report["results"].append(download_validate_store(
                        link, client, module.RAW_DIRECTORY, module.MANIFEST_PATH, validator,
                        args.force_check, module.ALLOWED_HOSTS))
                except Exception as exc:
                    failures = True; provider_report["results"].append({"status": "failed", "source_filename": link.source_filename, "reason": str(exc)})
        except Exception as exc:
            failures = True; provider_report["status"] = "source_access_failed"; provider_report["reason"] = str(exc)
    if failures: report["status"] = "partial_failure"
    return (1 if failures else 0), report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=("Tokyo", "Chubu", "all"), default="all")
    parser.add_argument("--from-month", type=_month, default="2026-03")
    parser.add_argument("--to-month", type=_month)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-check", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    if args.dry_run and args.validate_only:
        parser.error("--dry-run and --validate-only are mutually exclusive")
    if args.to_month and args.to_month < args.from_month:
        parser.error("--to-month must not precede --from-month")
    code, report = run(args)
    _write_report(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__": raise SystemExit(main())
