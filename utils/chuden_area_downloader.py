"""Discovery and validation for Chuden official monthly area actuals."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import urlparse

from utils.chuden_area_loader import load_chuden_area_data
from utils.grid_data_downloader import PublishedLink, completeness_diagnostics, ensure_csv_response, parse_anchors, require_official_domain


PROVIDER = "Chuden"
LANDING_PAGE = "https://powergrid.chuden.co.jp/denkiyoho/"
TERMS_PAGE = "https://powergrid.chuden.co.jp/denkiyoho/siteinfo.html"
ALLOWED_HOSTS = {"powergrid.chuden.co.jp"}
RAW_DIRECTORY = Path("data/external/chuden/raw")
MANIFEST_PATH = Path("data/external/chuden/manifest.json")
TERMS_REQUIRE_PRIOR_APPROVAL = True


def _year_month(text: str) -> str | None:
    japanese = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", text)
    if japanese: return f"{japanese.group(1)}-{int(japanese.group(2)):02d}"
    compact = re.search(r"(20\d{2})[-_/]?(0[1-9]|1[0-2])", text)
    return f"{compact.group(1)}-{compact.group(2)}" if compact else None


def discover_chuden_links(html: str) -> list[PublishedLink]:
    links = []
    for url, text in parse_anchors(html, LANDING_PAGE):
        filename = Path(urlparse(url).path).name
        if Path(filename).suffix.lower() not in {".csv", ".zip"}: continue
        year_month = _year_month(f"{text} {filename}")
        # Only a posted month link explicitly labelled as area supply-demand actuals is eligible.
        marker = f"{text} {filename}".lower()
        if not year_month or not any(token in marker for token in ("需給実績", "エリア需給", "area_jukyu", "eria_jukyu")):
            continue
        require_official_domain(url, ALLOWED_HOSTS)
        links.append(PublishedLink(PROVIDER, LANDING_PAGE, url, filename, year_month, text))
    return sorted(links, key=lambda item: (item.target_year_month, item.source_filename))


def validate_chuden_bytes(body: bytes, filename: str, target_year_month: str,
                          content_type: str | None = None, now_jst: Any = None) -> dict[str, Any]:
    ensure_csv_response(body, content_type)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / filename; path.write_bytes(body)
        data, diagnostics = load_chuden_area_data(path)
    diag = diagnostics.iloc[0].to_dict() if not diagnostics.empty else {}
    if data.empty or diag.get("status") == "Error": raise ValueError(f"chuden_loader_validation_failed: {diag.get('message', '')}")
    dates = data["delivery_date"].dropna()
    if dates.empty or set(dates.dt.strftime("%Y-%m")) != {target_year_month}:
        raise ValueError("target_month_mismatch")
    completeness = completeness_diagnostics(data, target_year_month, now_jst)
    warnings = []
    if diag.get("status") == "Review": warnings.append("loader_review")
    if completeness["status"] != "complete": warnings.append(completeness["status"])
    columns = sorted(str(column) for column in data.columns if not str(column).startswith("source_"))
    return {**completeness, "status": "valid", "warnings": warnings,
            "encoding": diag.get("encoding"), "row_count": len(data),
            "loader_name": "load_chuden_area_data",
            "schema_signature": sha256("|".join(columns).encode()).hexdigest(),
            "loader_diagnostics": diag}
