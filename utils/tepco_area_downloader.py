"""Discovery and validation for TEPCO official monthly area actuals."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import urlparse

from utils.grid_data_downloader import PublishedLink, completeness_diagnostics, ensure_csv_response, parse_anchors, require_official_domain
from utils.tepco_area_loader import load_tepco_area_data


PROVIDER = "TEPCO"
LANDING_PAGE = "https://www.tepco.co.jp/forecast/html/area_jukyu-j.html"
TERMS_PAGE = "https://www.tepco.co.jp/legal/"
ALLOWED_HOSTS = {"www.tepco.co.jp", "tepco.co.jp"}
RAW_DIRECTORY = Path("data/external/tepco/raw")
MANIFEST_PATH = Path("data/external/tepco/manifest.json")
TERMS_REQUIRE_PRIOR_APPROVAL = True


def discover_tepco_links(html: str) -> list[PublishedLink]:
    links = []
    for url, text in parse_anchors(html, LANDING_PAGE):
        filename = Path(urlparse(url).path).name
        match = re.search(r"eria_jukyu_(20\d{4})_[^/]*\.csv$", filename, re.I)
        if not match: continue
        require_official_domain(url, ALLOWED_HOSTS)
        year_month = f"{match.group(1)[:4]}-{match.group(1)[4:]}"
        links.append(PublishedLink(PROVIDER, LANDING_PAGE, url, filename, year_month, text))
    return sorted(links, key=lambda item: (item.target_year_month, item.source_filename))


def validate_tepco_bytes(body: bytes, filename: str, target_year_month: str,
                         content_type: str | None = None, now_jst: Any = None) -> dict[str, Any]:
    ensure_csv_response(body, content_type)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / filename; path.write_bytes(body)
        data, diagnostics = load_tepco_area_data(path)
    diag = diagnostics.iloc[0].to_dict() if not diagnostics.empty else {}
    if data.empty or diag.get("status") == "Error": raise ValueError(f"tepco_loader_validation_failed: {diag.get('message', '')}")
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
            "loader_name": "load_tepco_area_data",
            "schema_signature": sha256("|".join(columns).encode()).hexdigest(),
            "loader_diagnostics": diag}
