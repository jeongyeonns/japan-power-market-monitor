import argparse
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

import pandas as pd
import pytest

from scripts.update_grid_data import run
import utils.tepco_area_downloader as tepco
import utils.chuden_area_downloader as chuden
from utils.grid_data_downloader import (
    PublishedLink, SequentialHttpClient, atomic_write_bytes,
    completeness_diagnostics, download_validate_store, ensure_csv_response,
    load_manifest, safe_zip_csv_candidates,
)
from utils.tepco_area_loader import COLUMN_MAP as TEPCO_COLUMNS, UNIT_LABEL as TEPCO_UNIT
from utils.chuden_area_loader import COLUMN_MAP as CHUDEN_COLUMNS, UNIT_LABEL as CHUDEN_UNIT


def _csv_bytes(columns, unit, date="2026/06/01"):
    headers = list(columns)
    values = [date if name == "DATE" else ("00:00" if name == "TIME" else "1") for name in headers]
    return (unit + "\n" + ",".join(headers) + "\n" + ",".join(values) + "\n").encode("cp932")


def _zip(entries):
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, body in entries.items(): archive.writestr(name, body)
    return stream.getvalue()


class FakeClient:
    def __init__(self, response): self.response = response; self.calls = []
    def get(self, url, headers=None): self.calls.append((url, headers or {})); return self.response


def test_relative_monthly_links_and_non_target_files():
    html = '<a href="images/eria_jukyu_202606_03.csv">2026年6月</a><a href="forecast.csv">予測</a>'
    links = tepco.discover_tepco_links(html)
    assert links[0].source_url == "https://www.tepco.co.jp/forecast/html/images/eria_jukyu_202606_03.csv"
    assert links[0].target_year_month == "2026-06"
    assert len(links) == 1


def test_chuden_uses_only_explicitly_posted_actual_link_and_rejects_official_domain_violation():
    html = '<a href="/files/area_jukyu_202606.csv">エリア需給実績 2026年6月</a><a href="/files/bg_202606.csv">BG計画 2026年6月</a>'
    assert len(chuden.discover_chuden_links(html)) == 1
    evil = '<a href="https://example.com/area_jukyu_202606.csv">エリア需給実績 2026年6月</a>'
    with pytest.raises(ValueError, match="non_official_domain"):
        chuden.discover_chuden_links(evil)


def test_html_csv_and_zip_safety_checks():
    with pytest.raises(ValueError, match="html"):
        ensure_csv_response(b"<html>error</html>", "text/html")
    ensure_csv_response(b"DATE,TIME\n2026/1/1,00:00\n", "text/csv")
    with pytest.raises(ValueError, match="invalid_zip_magic"):
        safe_zip_csv_candidates(b"not zip")
    with pytest.raises(ValueError, match="zip_path_traversal"):
        safe_zip_csv_candidates(_zip({"../bad.csv": b"a,b\n1,2\n"}))


def test_http_retry_limit_timeout_and_file_size(monkeypatch):
    monkeypatch.setattr("utils.grid_data_downloader.time.sleep", lambda _: None)
    calls = []
    def failing(request, timeout):
        calls.append(timeout); raise URLError("timeout")
    client = SequentialHttpClient(retries=2, opener=failing)
    with pytest.raises(URLError): client.get("https://www.tepco.co.jp/test")
    assert len(calls) == 3

    class Large:
        status = 200; headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size): return b"x" * size
        def geturl(self): return "https://www.tepco.co.jp/test"
    with pytest.raises(ValueError, match="file_size"):
        SequentialHttpClient(retries=0, opener=lambda *a, **k: Large()).get("https://www.tepco.co.jp/test")


def test_atomic_replace_and_manifest_create_update(tmp_path):
    destination = tmp_path / "file.csv"
    atomic_write_bytes(destination, b"old")
    atomic_write_bytes(destination, b"new")
    assert destination.read_bytes() == b"new"
    manifest = load_manifest(tmp_path / "manifest.json", "TEPCO")
    assert manifest == {"provider": "TEPCO", "schema_version": 1, "files": []}


def test_download_same_sha_skips_and_304_is_unchanged(tmp_path):
    body = b"a,b\n1,2\n"; raw = tmp_path / "raw"; raw.mkdir()
    (raw / "x.csv").write_bytes(body)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"provider": "TEPCO", "schema_version": 1, "files": [{
        "source_filename": "x.csv", "sha256": __import__("hashlib").sha256(body).hexdigest(), "etag": '"x"'}]}))
    link = PublishedLink("TEPCO", "https://www.tepco.co.jp/page", "https://www.tepco.co.jp/x.csv", "x.csv", "2026-06")
    validator = lambda *args: {"status": "valid", "row_count": 1, "loader_name": "mock", "schema_signature": "x"}
    result = download_validate_store(link, FakeClient({"status": 200, "body": body, "headers": {"Content-Type": "text/csv"}}), raw, manifest_path, validator)
    assert result["status"] == "unchanged_sha256"
    result = download_validate_store(link, FakeClient({"status": 304, "body": b"", "headers": {}}), raw, manifest_path, validator)
    assert result["status"] == "not_modified"


def test_changed_file_manifest_headers_and_validation_failure_preserves_old(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir(); destination = raw / "x.csv"; destination.write_bytes(b"old")
    link = PublishedLink("TEPCO", "https://www.tepco.co.jp/page", "https://www.tepco.co.jp/x.csv", "x.csv", "2026-06")
    response = {"status": 200, "body": b"a,b\n1,2\n", "headers": {"Content-Type": "text/csv", "ETag": '"new"', "Last-Modified": "today"}}
    manifest_path = tmp_path / "manifest.json"
    validation = {"status": "valid", "row_count": 1, "minimum_date": "2026-06-01", "maximum_date": "2026-06-01",
                  "encoding": "cp932", "warnings": [], "loader_name": "mock", "schema_signature": "sig"}
    result = download_validate_store(link, FakeClient(response), raw, manifest_path, lambda *args: validation)
    assert result["status"] == "stored" and destination.read_bytes() == response["body"]
    entry = json.loads(manifest_path.read_text())["files"][0]
    assert entry["etag"] == '"new"' and entry["last_modified"] == "today"
    assert not Path(entry["local_relative_path"]).is_absolute()
    destination.write_bytes(b"known-good")
    with pytest.raises(ValueError):
        download_validate_store(link, FakeClient(response), raw, manifest_path, lambda *args: (_ for _ in ()).throw(ValueError("bad")), True)
    assert destination.read_bytes() == b"known-good"


def test_completeness_completed_and_partial_current_month():
    full = pd.DataFrame({"delivery_date": pd.date_range("2026-06-01", "2026-06-30 23:30", freq="30min"),
                         "period_no": list(range(1, 49)) * 30})
    assert completeness_diagnostics(full, "2026-06", "2026-08-01")["status"] == "complete"
    partial = pd.DataFrame({"delivery_date": [pd.Timestamp("2026-08-01")] * 2, "period_no": [1, 2]})
    assert completeness_diagnostics(partial, "2026-08", "2026-08-01 01:00+09:00")["status"] == "partial_current_day"


def test_existing_loaders_validate_provider_bytes():
    tepco_result = tepco.validate_tepco_bytes(_csv_bytes(TEPCO_COLUMNS, TEPCO_UNIT), "eria_jukyu_202606_03.csv", "2026-06")
    chuden_result = chuden.validate_chuden_bytes(_csv_bytes(CHUDEN_COLUMNS, CHUDEN_UNIT), "area_jukyu_202606.csv", "2026-06")
    assert tepco_result["loader_name"] == "load_tepco_area_data"
    assert chuden_result["loader_name"] == "load_chuden_area_data"
    json.dumps({"tepco": tepco_result, "chuden": chuden_result}, default=str)


def test_cli_dry_run_only_uses_posted_links_and_terms_block_download(monkeypatch):
    html = '<a href="images/eria_jukyu_202606_03.csv">2026年6月</a>'.encode()
    client = FakeClient({"status": 200, "body": html, "headers": {"Content-Type": "text/html"}})
    args = argparse.Namespace(region="Tokyo", from_month="2026-03", to_month=None, latest=False,
                              dry_run=True, force_check=False, validate_only=False)
    code, report = run(args, client)
    assert code == 0 and len(report["providers"]["Tokyo"]["links"]) == 1
    args.dry_run = False
    monkeypatch.delenv("GRID_DATA_DOWNLOAD_APPROVED", raising=False)
    code, report = run(args, client)
    assert code == 2 and report["status"] == "blocked_terms_approval_required"


def test_cli_validate_only_uses_no_network(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "eria_jukyu_202606_03.csv").write_bytes(_csv_bytes(TEPCO_COLUMNS, TEPCO_UNIT))
    monkeypatch.setattr(tepco, "RAW_DIRECTORY", raw)
    args = argparse.Namespace(region="Tokyo", from_month="2026-03", to_month=None, latest=False,
                              dry_run=False, force_check=False, validate_only=True)
    client = FakeClient(None)
    code, report = run(args, client)
    assert code == 0
    assert client.calls == []
    assert report["providers"]["Tokyo"]["results"][0]["status"] == "valid"
