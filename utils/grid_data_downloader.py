"""Safe, sequential utilities for manually updating official grid actuals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd


USER_AGENT = "JapanMarketMonitor-grid-data-updater/1.0"
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_ZIP_MEMBERS = 20
MAX_ZIP_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MIN_REQUEST_INTERVAL_SECONDS = 2.0


@dataclass(frozen=True)
class PublishedLink:
    provider: str
    landing_page: str
    source_url: str
    source_filename: str
    target_year_month: str
    link_text: str = ""


class AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(); self.links: list[tuple[str, str]] = []
        self._href: str | None = None; self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self._href = dict(attrs).get("href"); self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None: self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None; self._text = []


def parse_anchors(html: str, landing_page: str) -> list[tuple[str, str]]:
    parser = AnchorParser(); parser.feed(html)
    return [(urljoin(landing_page, href), text) for href, text in parser.links]


def require_official_domain(url: str, allowed_hosts: set[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in allowed_hosts:
        raise ValueError(f"non_official_domain: {parsed.hostname or '<missing>'}")


class SequentialHttpClient:
    def __init__(self, timeout: float = 30, retries: int = 2,
                 request_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
                 opener: Callable[..., Any] = urlopen) -> None:
        self.timeout = timeout; self.retries = min(max(retries, 0), 2)
        self.request_interval = max(request_interval, 2.0); self.opener = opener
        self._last_request_at: float | None = None

    def get(self, url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        transient = {408, 429, 500, 502, 503, 504}
        for attempt in range(self.retries + 1):
            if self._last_request_at is not None:
                time.sleep(max(0, self.request_interval - (time.monotonic() - self._last_request_at)))
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/csv,application/zip,*/*;q=0.1", **(headers or {})})
            try:
                self._last_request_at = time.monotonic()
                with self.opener(request, timeout=self.timeout) as response:
                    status = getattr(response, "status", 200)
                    body = response.read(MAX_FILE_BYTES + 1)
                    if len(body) > MAX_FILE_BYTES: raise ValueError("file_size_limit_exceeded")
                    return {"status": status, "body": body, "headers": dict(response.headers.items()), "url": response.geturl()}
            except HTTPError as exc:
                if exc.code == 304:
                    return {"status": 304, "body": b"", "headers": dict(exc.headers.items()), "url": url}
                if exc.code not in transient or attempt >= self.retries: raise
            except (URLError, TimeoutError):
                if attempt >= self.retries: raise
        raise RuntimeError("unreachable")


def ensure_csv_response(body: bytes, content_type: str | None = None) -> None:
    prefix = body[:512].lstrip().lower()
    if not body: raise ValueError("empty_response")
    if prefix.startswith((b"<!doctype html", b"<html", b"<?xml")) or b"<html" in prefix:
        raise ValueError("html_response_rejected")
    if content_type and "html" in content_type.lower(): raise ValueError("html_content_type_rejected")
    if b"," not in body[:4096]: raise ValueError("not_csv_content")


def safe_zip_csv_candidates(body: bytes) -> dict[str, bytes]:
    if not body.startswith(b"PK\x03\x04"): raise ValueError("invalid_zip_magic")
    try:
        with ZipFile(BytesIO(body)) as archive:
            members = archive.infolist()
            if len(members) > MAX_ZIP_MEMBERS: raise ValueError("zip_member_limit_exceeded")
            if sum(item.file_size for item in members) > MAX_ZIP_UNCOMPRESSED_BYTES:
                raise ValueError("zip_uncompressed_size_limit_exceeded")
            result = {}
            for item in members:
                path = PurePosixPath(item.filename)
                if path.is_absolute() or ".." in path.parts: raise ValueError("zip_path_traversal")
                if item.is_dir(): continue
                if path.suffix.lower() != ".csv": continue
                result[path.name] = archive.read(item)
            if not result: raise ValueError("zip_contains_no_csv")
            return result
    except BadZipFile as exc:
        raise ValueError("invalid_zip") from exc


def sha256_bytes(body: bytes) -> str:
    return sha256(body).hexdigest()


def atomic_write_bytes(destination: Path, body: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_manifest(path: Path, provider: str) -> dict[str, Any]:
    if not path.exists(): return {"provider": provider, "schema_version": 1, "files": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ValueError("invalid_manifest")
    return value


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def update_manifest_entry(manifest: dict[str, Any], entry: dict[str, Any]) -> None:
    files = manifest.setdefault("files", [])
    files[:] = [item for item in files if item.get("source_filename") != entry["source_filename"]]
    files.append(entry); files.sort(key=lambda item: (item.get("target_year_month", ""), item.get("source_filename", "")))


def completeness_diagnostics(data: pd.DataFrame, target_year_month: str, now_jst: Any = None) -> dict[str, Any]:
    dates = pd.to_datetime(data["delivery_date"], errors="coerce").dt.normalize()
    periods = pd.to_numeric(data["period_no"], errors="coerce")
    month_start = pd.Timestamp(f"{target_year_month}-01")
    month_end = month_start + pd.offsets.MonthEnd(0)
    now = pd.Timestamp(now_jst or datetime.now().astimezone())
    current_month = now.strftime("%Y-%m") == target_year_month
    expected_end = min(month_end, now.tz_localize(None).normalize()) if current_month else month_end
    expected_dates = pd.date_range(month_start, expected_end, freq="D")
    counts = dates.value_counts().sort_index()
    missing_dates = [date.date().isoformat() for date in expected_dates if date not in counts.index]
    incomplete = {date.date().isoformat(): int(count) for date, count in counts.items() if count != 48}
    duplicates = int(pd.DataFrame({"date": dates, "period": periods}).duplicated(["date", "period"], keep=False).sum())
    partial_today = bool(current_month and now.tz_localize(None).normalize() in counts.index and counts[now.tz_localize(None).normalize()] < 48)
    return {"expected_rows": len(expected_dates) * 48, "actual_rows": len(data),
            "minimum_date": dates.min().date().isoformat() if dates.notna().any() else None,
            "maximum_date": dates.max().date().isoformat() if dates.notna().any() else None,
            "missing_dates": missing_dates, "incomplete_day_period_counts": incomplete,
            "duplicate_period_rows": duplicates, "partial_current_day": partial_today,
            "status": "partial_current_day" if partial_today else ("complete" if not missing_dates and not incomplete and not duplicates else "incomplete")}


def manifest_entry(link: PublishedLink, destination: Path, raw_root: Path, response: dict[str, Any],
                   validation: dict[str, Any], previous_sha: str | None = None) -> dict[str, Any]:
    headers = {str(k).lower(): str(v) for k, v in response.get("headers", {}).items()}
    return {"provider": link.provider, "official_landing_page": link.landing_page,
        "source_url": link.source_url, "source_filename": link.source_filename,
        "local_relative_path": destination.relative_to(raw_root.parent).as_posix(),
        "target_year_month": link.target_year_month,
        "retrieved_at_jst": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "etag": headers.get("etag"), "last_modified": headers.get("last-modified"),
        "content_type": headers.get("content-type"), "file_size_bytes": len(response["body"]),
        "sha256": sha256_bytes(response["body"]), "previous_sha256": previous_sha,
        "encoding": validation.get("encoding"), "row_count": validation.get("row_count"),
        "minimum_date": validation.get("minimum_date"), "maximum_date": validation.get("maximum_date"),
        "validation_status": validation.get("status"), "validation_warnings": validation.get("warnings", []),
        "loader_name": validation.get("loader_name"), "schema_signature": validation.get("schema_signature")}


def download_validate_store(
    link: PublishedLink,
    client: SequentialHttpClient,
    raw_directory: Path,
    manifest_path: Path,
    validator: Callable[[bytes, str, str, str | None], dict[str, Any]],
    force_check: bool = False,
    allowed_hosts: set[str] | None = None,
) -> dict[str, Any]:
    """Download one published link and replace a local file only after validation."""
    manifest = load_manifest(manifest_path, link.provider)
    old = next((item for item in manifest["files"] if item.get("source_filename") == link.source_filename), None)
    headers = {}
    if old and not force_check:
        if old.get("etag"): headers["If-None-Match"] = old["etag"]
        if old.get("last_modified"): headers["If-Modified-Since"] = old["last_modified"]
    response = client.get(link.source_url, headers)
    if allowed_hosts is not None:
        require_official_domain(response.get("url", link.source_url), allowed_hosts)
    if response["status"] == 304:
        return {"status": "not_modified", "source_filename": link.source_filename}
    body = response["body"]
    response_headers = {str(key).lower(): value for key, value in response.get("headers", {}).items()}
    content_type = response_headers.get("content-type")
    selected_filename = link.source_filename
    selected_body = body
    if body.startswith(b"PK\x03\x04") or Path(link.source_filename).suffix.lower() == ".zip":
        candidates = safe_zip_csv_candidates(body)
        validated = []
        for candidate_name, candidate_body in candidates.items():
            try:
                validation = validator(candidate_body, candidate_name, link.target_year_month, "text/csv")
                validated.append((candidate_name, candidate_body, validation))
            except ValueError:
                continue
        if len(validated) != 1:
            raise ValueError(f"zip_schema_candidate_count: {len(validated)}")
        selected_filename, selected_body, validation = validated[0]
    else:
        validation = validator(body, link.source_filename, link.target_year_month, content_type)
    digest = sha256_bytes(selected_body)
    destination = raw_directory / selected_filename
    previous_sha = old.get("sha256") if old else (sha256_bytes(destination.read_bytes()) if destination.exists() else None)
    if digest == previous_sha and destination.exists():
        return {"status": "unchanged_sha256", "source_filename": selected_filename, "sha256": digest}
    stored_response = {**response, "body": selected_body}
    stored_link = PublishedLink(link.provider, link.landing_page, link.source_url,
                                selected_filename, link.target_year_month, link.link_text)
    entry = manifest_entry(stored_link, destination, raw_directory, stored_response, validation, previous_sha)
    atomic_write_bytes(destination, selected_body)
    try:
        update_manifest_entry(manifest, entry); save_manifest(manifest_path, manifest)
    except Exception:
        # The validated raw file remains usable; manifest failure is explicit to the caller.
        raise
    return {"status": "stored", "source_filename": selected_filename, "sha256": digest,
            "previous_sha256": previous_sha, "validation": validation}
