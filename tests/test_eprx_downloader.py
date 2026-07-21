from pathlib import Path

import pandas as pd

from utils import eprx_downloader as downloader
from utils.eprx_loader import find_eprx_files


SAMPLE_HTML = """
<html><head><title>取引実績</title></head><body><h1>取引実績</h1>
  <div>2026年7月20日 一次調整力 速報値
    <a href="/files/202607_1-0.csv">一次調整力 CSV</a>
    <a href="/files/202607_1-0.csv">중복 링크</a>
  </div>
  <div>2026/07/19 Primary Reserve
    <a href="files/primary.xlsx">Primary Reserve Excel</a>
  </div>
  <div><a href="/files/other.zip">その他商品 ZIP</a></div>
  <a href="/information/page.html">일반 페이지</a>
</body></html>
"""


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        content_type: str = "text/csv",
        url: str = "https://www.eprx.or.jp/information/results.php",
    ):
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.apparent_encoding = "utf-8"
        self.encoding = "utf-8"
        self.text = content.decode("utf-8", errors="replace")
        self.url = url
        self.status_code = 200
        self.history = []

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


class FakeSession:
    def __init__(
        self,
        content: bytes,
        content_type: str = "text/csv",
        url: str = "https://www.eprx.or.jp/information/results.php",
    ):
        self.response = FakeResponse(content, content_type, url)
        self.headers = {}
        self.cookies = {}
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return self.response


def candidate(file_name="result.csv", file_url="https://example.test/result.csv"):
    return {
        "file_name": file_name,
        "file_url": file_url,
    }


def fake_page_details(html=SAMPLE_HTML):
    return {
        "html": html,
        "source_url": downloader.EPRX_RESULTS_PAGE_URL,
        "final_url": downloader.EPRX_RESULTS_PAGE_URL,
        "status_code": 200,
        "content_type": "text/html",
        "response_size": len(html.encode()),
        "redirect_history": [],
        "cookies_set": [],
        **downloader.classify_results_page(
            html, downloader.EPRX_RESULTS_PAGE_URL, 200, "text/html"
        ),
    }


def fake_checked_result():
    links = downloader.discover_download_links(
        SAMPLE_HTML, downloader.EPRX_RESULTS_PAGE_URL
    )
    primary = downloader.identify_primary_reserve_candidates(links)
    primary = primary[primary["is_primary_reserve_candidate"]].copy()
    return {
        "all_links": links,
        "primary_candidates": primary,
        "new_candidates": primary,
        "existing_files": primary.iloc[0:0].copy(),
        "revision_candidates": primary.iloc[0:0].copy(),
        "errors": [],
        "dry_run": True,
        "page_details": {
            key: value
            for key, value in fake_page_details().items()
            if key != "html"
        },
    }


def test_discover_links_resolves_relative_urls_and_removes_duplicates():
    links = downloader.discover_download_links(
        SAMPLE_HTML, "https://www.eprx.or.jp/information/results.php"
    )
    assert len(links) == 3
    assert links["file_url"].is_unique
    assert (
        links.iloc[0]["file_url"]
        == "https://www.eprx.or.jp/files/202607_1-0.csv"
    )
    assert set(links["extension"]) == {".csv", ".xlsx", ".zip"}


def test_identify_primary_reserve_candidates():
    links = downloader.discover_download_links(
        SAMPLE_HTML, "https://www.eprx.or.jp/information/results.php"
    )
    identified = downloader.identify_primary_reserve_candidates(links)
    assert identified["is_primary_reserve_candidate"].tolist() == [
        True,
        True,
        False,
    ]
    assert identified.loc[
        identified["is_primary_reserve_candidate"], "product_hint"
    ].eq("一次調整力").all()


def test_hash_bytes_and_file_match(tmp_path):
    content = b"same-content"
    path = tmp_path / "sample.csv"
    path.write_bytes(content)
    assert downloader.calculate_file_hash(content) == downloader.calculate_file_hash(
        path
    )


def test_download_history_round_trip(tmp_path):
    history_path = tmp_path / "metadata" / "history.csv"
    history = pd.DataFrame(
        [{"saved_file_name": "a.csv", "sha256": "abc", "status": "다운로드 완료"}]
    ).reindex(columns=downloader.HISTORY_COLUMNS)
    downloader.save_download_history(history, history_path)
    assert history_path.read_bytes().startswith(b"\xef\xbb\xbf")
    loaded = downloader.load_download_history(history_path)
    assert loaded.iloc[0]["saved_file_name"] == "a.csv"
    assert loaded.iloc[0]["sha256"] == "abc"


def test_same_hash_is_not_saved_twice(tmp_path, monkeypatch):
    monkeypatch.setenv("EPRX_AUTOMATION_APPROVED", "true")
    raw = tmp_path / "raw"
    history = tmp_path / "metadata" / "history.csv"
    first = downloader.download_file(
        candidate(),
        raw,
        history_path=history,
        session=FakeSession(b"first"),
    )
    second = downloader.download_file(
        candidate(),
        raw,
        history_path=history,
        session=FakeSession(b"first"),
    )
    assert first["status"] == "다운로드 완료"
    assert second["status"] == "중복 건너뜀"
    assert second["duplicate_of"] == "result.csv"
    assert [path.name for path in raw.glob("*.csv")] == ["result.csv"]


def test_same_name_different_hash_is_saved_as_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("EPRX_AUTOMATION_APPROVED", "true")
    raw = tmp_path / "raw"
    history = tmp_path / "metadata" / "history.csv"
    downloader.download_file(
        candidate(),
        raw,
        history_path=history,
        session=FakeSession(b"first"),
    )
    revised = downloader.download_file(
        candidate(file_url="https://example.test/revised/result.csv"),
        raw,
        history_path=history,
        session=FakeSession(b"changed"),
    )
    assert revised["status"] == "다운로드 완료"
    assert revised["revision_of"] == "result.csv"
    assert "__revised_" in revised["saved_file_name"]
    assert len(list(raw.glob("*.csv"))) == 2


def test_dry_run_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.setenv("EPRX_AUTOMATION_APPROVED", "true")
    monkeypatch.setattr(
        downloader,
        "fetch_results_page_details",
        lambda session=None: fake_page_details(),
    )
    raw = tmp_path / "raw"
    history = tmp_path / "metadata" / "history.csv"
    log = tmp_path / "metadata" / "update_log.csv"
    result = downloader.update_eprx_files(
        dry_run=True,
        destination_directory=raw,
        history_path=history,
        log_path=log,
    )
    assert not result["errors"]
    assert len(result["primary_candidates"]) == 2
    assert len(result["new_candidates"]) == 2
    assert not raw.exists()
    assert not history.exists()
    assert log.exists()


def test_network_is_blocked_without_approval(monkeypatch):
    monkeypatch.delenv("EPRX_AUTOMATION_APPROVED", raising=False)
    session = FakeSession(b"<html></html>", "text/html")
    result = downloader.check_for_new_eprx_files(session=session)
    assert result["errors"]
    assert "사전 승인" in result["errors"][0]
    assert session.calls == 0


def test_approval_true_enters_request_function(monkeypatch):
    monkeypatch.setenv("EPRX_AUTOMATION_APPROVED", "TRUE")
    session = FakeSession(SAMPLE_HTML.encode(), "text/html")
    details = downloader.fetch_results_page_details(session=session)
    assert session.calls == 1
    assert details["is_results_page"]


def test_agreement_and_results_page_detection():
    agreement = """
    <html><head><title>取引実績</title></head>
    <body><h1>取引実績</h1><h2>免責事項</h2><p>ご注意</p>
    <button>同意する</button></body></html>
    """
    agreement_result = downloader.classify_results_page(
        agreement,
        "https://www.eprx.or.jp/information/agree_results.php",
        200,
        "text/html",
    )
    results_result = downloader.classify_results_page(
        SAMPLE_HTML,
        downloader.EPRX_RESULTS_PAGE_URL,
        200,
        "text/html",
    )
    assert agreement_result["is_agreement_page"]
    assert not agreement_result["is_results_page"]
    assert results_result["is_results_page"]


def test_html_response_is_not_saved_as_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("EPRX_AUTOMATION_APPROVED", "1")
    raw = tmp_path / "raw"
    history = tmp_path / "history.csv"
    record = downloader.download_file(
        candidate(),
        raw,
        history_path=history,
        session=FakeSession(b"<html>not csv</html>", "application/octet-stream"),
    )
    assert record["status"] == "실패"
    assert "HTML" in record["error_message"]
    assert not list(raw.glob("*.csv"))


def test_test_mode_downloads_only_one_and_records_parse_success(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EPRX_AUTOMATION_APPROVED", "true")
    monkeypatch.setattr(
        downloader, "check_for_new_eprx_files", lambda **kwargs: fake_checked_result()
    )
    raw = tmp_path / "raw"
    history = tmp_path / "metadata" / "history.csv"
    log = tmp_path / "metadata" / "update_log.csv"
    actual_csv = Path("data/eprx/202607_1-0_prompt.csv").read_bytes()
    result = downloader.update_eprx_files(
        dry_run=False,
        max_downloads=9,
        test_mode=True,
        session=FakeSession(actual_csv, "text/csv"),
        destination_directory=raw,
        history_path=history,
        log_path=log,
    )
    downloads = result["download_results"]
    assert len(result["selected_candidate"]) == 1
    assert len(downloads) == 1
    assert downloads.iloc[0]["parse_status"] == "파싱 성공"
    assert downloads.iloc[0]["primary_reserve_row_count"] == 9072
    assert bool(downloads.iloc[0]["required_metrics_present"])
    assert len(list(raw.glob("*.csv"))) == 1
    assert downloader.load_update_log(log).iloc[-1]["execution_mode"] == "시험 다운로드"


def test_parse_failure_preserves_downloaded_original(tmp_path, monkeypatch):
    monkeypatch.setenv("EPRX_AUTOMATION_APPROVED", "true")
    monkeypatch.setattr(
        downloader, "check_for_new_eprx_files", lambda **kwargs: fake_checked_result()
    )
    raw = tmp_path / "raw"
    result = downloader.update_eprx_files(
        test_mode=True,
        session=FakeSession(b"not,eprx,csv", "text/csv"),
        destination_directory=raw,
        history_path=tmp_path / "history.csv",
        log_path=tmp_path / "update_log.csv",
    )
    assert result["download_results"].iloc[0]["parse_status"] == "파싱 실패"
    assert len(list(raw.glob("*.csv"))) == 1


def test_loader_reads_root_and_raw_once_per_hash(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    content = b"same"
    (tmp_path / "legacy.csv").write_bytes(content)
    (raw / "downloaded.csv").write_bytes(content)
    (raw / "different.csv").write_bytes(b"different")
    files = find_eprx_files(tmp_path)
    assert len(files) == 2
    assert len({downloader.calculate_file_hash(path) for path in files}) == 2
