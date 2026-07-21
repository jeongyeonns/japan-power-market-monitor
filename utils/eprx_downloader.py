"""EPRX 거래실적 파일의 안전한 확인·다운로드 도구.

EPRX 거래실적 페이지는 자동적인 대량 취득에 사전 승인을 요구합니다. 따라서
``EPRX_AUTOMATION_APPROVED=true``가 명시되지 않으면 네트워크 요청 전에 중단합니다.
이 설정은 사용자가 EPRX의 승인을 실제로 받은 경우에만 사용해야 합니다.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils.eprx_loader import normalize_eprx_data, read_eprx_file

EPRX_RESULTS_PAGE_URL = "https://www.eprx.or.jp/information/results.php"
REQUEST_TIMEOUT_SECONDS = 20
USER_AGENT = (
    "JapanMarketMonitor/1.0 "
    "(manual-update; contact-site-owner-before-automated-collection)"
)
SUPPORTED_DOWNLOAD_EXTENSIONS = {".csv", ".zip", ".xlsx", ".xls"}
BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_DIRECTORY = BASE_DIR / "data" / "eprx" / "raw"
HISTORY_PATH = BASE_DIR / "data" / "eprx" / "metadata" / "download_history.csv"
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
MAX_DOWNLOADS_PER_RUN = 1
UPDATE_LOG_PATH = BASE_DIR / "data" / "eprx" / "metadata" / "update_log.csv"
HISTORY_COLUMNS = [
    "downloaded_at",
    "source_page",
    "file_url",
    "original_file_name",
    "saved_file_name",
    "file_size",
    "sha256",
    "status",
    "duplicate_of",
    "revision_of",
    "error_message",
    "parse_status",
    "parse_error",
    "execution_mode",
    "test_mode",
    "page_final_url",
    "page_status_code",
    "parsed_row_count",
    "primary_reserve_row_count",
    "detected_date_min",
    "detected_date_max",
    "detected_encoding",
    "detected_areas",
    "detected_zones",
    "required_metrics_present",
]
CANDIDATE_COLUMNS = [
    "link_text",
    "file_name",
    "file_url",
    "extension",
    "published_date",
    "year",
    "result_status",
    "product_hint",
    "is_primary_reserve_candidate",
    "discovered_at",
    "surrounding_text",
]
RESTRICTION_PHRASES = (
    "自動的な大量取得",
    "スクレイピング",
    "事前承諾",
)
UPDATE_LOG_COLUMNS = [
    "executed_at",
    "execution_mode",
    "approval_env_enabled",
    "source_page",
    "final_url",
    "page_status",
    "discovered_link_count",
    "primary_candidate_count",
    "selected_file",
    "download_status",
    "saved_file",
    "sha256",
    "parse_status",
    "parsed_rows",
    "primary_rows",
    "error_message",
]


class EprxDownloadError(RuntimeError):
    """EPRX 확인 또는 다운로드 실패."""


class EprxAutomationPermissionError(EprxDownloadError):
    """자동 수집 사전 승인 미확인."""


def automation_approved() -> bool:
    """승인 환경변수의 허용 값을 확인합니다."""
    return os.getenv("EPRX_AUTOMATION_APPROVED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _require_automation_approval() -> None:
    if not automation_approved():
        raise EprxAutomationPermissionError(
            "EPRX 거래실적 페이지는 자동적인 대량 취득에 사전 승인을 요구합니다. "
            "EPRX의 승인을 받은 뒤에만 EPRX_AUTOMATION_APPROVED=true를 설정하세요."
        )


def _session(session: requests.Session | None = None) -> requests.Session:
    client = session or requests.Session()
    client.headers.update({"User-Agent": USER_AGENT})
    return client


def _restriction_phrases(html: str) -> list[str]:
    return [phrase for phrase in RESTRICTION_PHRASES if phrase in html]


def _check_restrictions(html: str) -> None:
    found = _restriction_phrases(html)
    if found:
        raise EprxAutomationPermissionError(
            "페이지에서 자동수집 사전 승인 제한을 확인하여 중단했습니다: "
            + ", ".join(found)
        )


def classify_results_page(
    html: str, final_url: str, status_code: int, content_type: str
) -> dict[str, Any]:
    """URL, title, heading, 본문을 함께 사용해 응답 페이지를 판별합니다."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    headings = " ".join(
        heading.get_text(" ", strip=True)
        for heading in soup.find_all(["h1", "h2", "h3"])
    )
    text = soup.get_text(" ", strip=True)
    lowered_url = final_url.lower()
    restrictions = _restriction_phrases(html)
    is_agreement = (
        "agree_results.php" in lowered_url
        or "免責事項" in headings
        or ("同意する" in text and "ご注意" in text)
    )
    is_login = (
        "login" in lowered_url
        or "ログイン" in title
        or "ログイン" in headings
    )
    is_error = (
        status_code >= 400
        or "エラー" in title
        or "アクセス拒否" in text
        or "Forbidden" in text
    )
    html_content = "text/html" in content_type.lower()
    return {
        "page_title": title,
        "headings": headings,
        "is_agreement_page": is_agreement,
        "is_login_page": is_login,
        "is_error_page": is_error,
        "is_html": html_content,
        "restriction_phrases": restrictions,
        "is_results_page": (
            html_content
            and not is_agreement
            and not is_login
            and not is_error
            and "取引実績" in f"{title} {headings}"
        ),
    }


def fetch_results_page_details(
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """공식 거래실적 페이지를 한 번 요청하고 응답 진단 정보를 반환합니다."""
    _require_automation_approval()
    client = _session(session)
    try:
        response = client.get(
            EPRX_RESULTS_PAGE_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise EprxDownloadError(f"EPRX 거래실적 페이지 요청 실패: {exc}") from exc
    response.encoding = response.apparent_encoding or response.encoding
    html = response.text
    content_type = response.headers.get("Content-Type", "")
    page = classify_results_page(
        html, str(response.url), response.status_code, content_type
    )
    return {
        "html": html,
        "source_url": EPRX_RESULTS_PAGE_URL,
        "final_url": str(response.url),
        "status_code": response.status_code,
        "content_type": content_type,
        "response_size": len(response.content),
        "redirect_history": [
            {"status_code": item.status_code, "url": str(item.url)}
            for item in response.history
        ],
        "cookies_set": sorted(client.cookies.keys()),
        **page,
    }


def fetch_results_page(session: requests.Session | None = None) -> str:
    """유효한 거래실적 페이지인 경우에만 HTML을 반환합니다."""
    details = fetch_results_page_details(session=session)
    if details["is_agreement_page"]:
        raise EprxAutomationPermissionError(
            "거래실적 동의 페이지가 반환되었습니다. 승인 폼을 코드로 제출하거나 "
            "브라우저 쿠키를 추출하지 않습니다."
        )
    if details["restriction_phrases"]:
        _check_restrictions(details["html"])
    if not details["is_results_page"]:
        raise EprxDownloadError(
            f"거래실적 페이지로 확인되지 않았습니다: {details['final_url']}"
        )
    return str(details["html"])


def _file_name_from_link(href: str, link_text: str) -> str:
    path_name = Path(unquote(urlparse(href).path)).name
    if Path(path_name).suffix.lower() in SUPPORTED_DOWNLOAD_EXTENSIONS:
        return path_name
    text_path = Path(link_text.strip())
    if text_path.suffix.lower() in SUPPORTED_DOWNLOAD_EXTENSIONS:
        return text_path.name
    return path_name


def _published_date(text: str) -> object:
    match = re.search(
        r"(?P<year>20\d{2})[./年-](?P<month>\d{1,2})[./月-](?P<day>\d{1,2})日?",
        text,
    )
    if not match:
        return pd.NaT
    return pd.Timestamp(
        year=int(match.group("year")),
        month=int(match.group("month")),
        day=int(match.group("day")),
    )


def discover_download_links(html: str, base_url: str) -> pd.DataFrame:
    """href와 주변 문구에서 지원 파일 링크를 찾아 중복 제거합니다."""
    _check_restrictions(html)
    soup = BeautifulSoup(html, "html.parser")
    discovered_at = pd.Timestamp.now(tz="Asia/Tokyo")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", "")).strip()
        link_text = anchor.get_text(" ", strip=True)
        file_name = _file_name_from_link(href, link_text)
        extension = Path(file_name).suffix.lower()
        combined = f"{href} {link_text}".lower()
        if extension not in SUPPORTED_DOWNLOAD_EXTENSIONS:
            matching = next(
                (suffix for suffix in SUPPORTED_DOWNLOAD_EXTENSIONS if suffix in combined),
                None,
            )
            if matching is None:
                continue
            extension = matching
        file_url = urljoin(base_url, href)
        if file_url in seen:
            continue
        seen.add(file_url)
        surrounding = anchor.parent.get_text(" ", strip=True) if anchor.parent else link_text
        published = _published_date(surrounding)
        records.append(
            {
                "link_text": link_text,
                "file_name": file_name or "Unknown",
                "file_url": file_url,
                "extension": extension,
                "published_date": published,
                "year": published.year if not pd.isna(published) else pd.NA,
                "result_status": (
                    "確報値"
                    if "確報" in surrounding
                    else "速報値" if "速報" in surrounding else "Unknown"
                ),
                "product_hint": "Unknown",
                "is_primary_reserve_candidate": False,
                "discovered_at": discovered_at,
                "surrounding_text": surrounding,
            }
        )
    return pd.DataFrame(records, columns=CANDIDATE_COLUMNS)


def identify_primary_reserve_candidates(links: pd.DataFrame) -> pd.DataFrame:
    """실제 확인된 1차 조정력 표현으로 후보 여부를 기록합니다."""
    result = links.copy()
    if result.empty:
        return result.reindex(columns=CANDIDATE_COLUMNS)
    text = (
        result[["link_text", "file_name", "surrounding_text"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    primary = text.str.contains(
        r"一次調整力|Primary\s*Reserve|一次(?:[^調]|$)", case=False, regex=True
    )
    result["is_primary_reserve_candidate"] = primary
    result.loc[primary, "product_hint"] = "一次調整力"
    return result


def calculate_file_hash(file_path_or_bytes: str | Path | bytes) -> str:
    """파일 또는 bytes의 SHA-256을 계산합니다."""
    digest = hashlib.sha256()
    if isinstance(file_path_or_bytes, bytes):
        digest.update(file_path_or_bytes)
    else:
        with Path(file_path_or_bytes).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_download_history(history_path: str | Path = HISTORY_PATH) -> pd.DataFrame:
    """UTF-8-SIG 다운로드 이력을 읽거나 빈 표를 반환합니다."""
    path = Path(history_path)
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        history = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise EprxDownloadError(f"다운로드 이력을 읽지 못했습니다: {exc}") from exc
    return history.reindex(columns=HISTORY_COLUMNS)


def save_download_history(
    history: pd.DataFrame, history_path: str | Path = HISTORY_PATH
) -> None:
    """다운로드 이력을 UTF-8-SIG CSV로 원자적으로 저장합니다."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = history.reindex(columns=HISTORY_COLUMNS)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        suffix=".tmp",
        prefix="download_history_",
        dir=path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        normalized.to_csv(handle, index=False)
    temp_path.replace(path)


def load_update_log(log_path: str | Path = UPDATE_LOG_PATH) -> pd.DataFrame:
    """업데이트 실행 로그를 읽거나 빈 표를 반환합니다."""
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame(columns=UPDATE_LOG_COLUMNS)
    return pd.read_csv(path, encoding="utf-8-sig").reindex(
        columns=UPDATE_LOG_COLUMNS
    )


def save_update_log(
    log: pd.DataFrame, log_path: str | Path = UPDATE_LOG_PATH
) -> None:
    """개인정보 없이 업데이트 실행 로그를 원자적으로 저장합니다."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        suffix=".tmp",
        prefix="update_log_",
        dir=path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        log.reindex(columns=UPDATE_LOG_COLUMNS).to_csv(handle, index=False)
    temp_path.replace(path)


def _append_update_log(
    record: dict[str, Any], log_path: str | Path = UPDATE_LOG_PATH
) -> None:
    current = load_update_log(log_path)
    updated = pd.concat([current, pd.DataFrame([record])], ignore_index=True)
    save_update_log(updated, log_path)


def _safe_file_name(file_name: str) -> str:
    name = Path(file_name).name
    if not name or name in {".", ".."}:
        raise EprxDownloadError("유효한 원본 파일명이 아닙니다.")
    if Path(name).suffix.lower() not in SUPPORTED_DOWNLOAD_EXTENSIONS:
        raise EprxDownloadError(f"지원하지 않는 파일 확장자입니다: {name}")
    return name


def _content_type_allowed(content_type: str, extension: str) -> bool:
    lowered = content_type.lower().split(";", 1)[0].strip()
    allowed = {
        ".csv": {"text/csv", "text/plain", "application/csv", "application/octet-stream"},
        ".zip": {"application/zip", "application/x-zip-compressed", "application/octet-stream"},
        ".xlsx": {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/octet-stream",
        },
        ".xls": {"application/vnd.ms-excel", "application/octet-stream"},
    }
    return lowered in allowed.get(extension, set())


def _revision_name(destination: Path, original_name: str) -> str:
    original = Path(original_name)
    timestamp = pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y%m%d_%H%M%S")
    candidate = f"{original.stem}__revised_{timestamp}{original.suffix}"
    counter = 1
    while (destination / candidate).exists():
        candidate = (
            f"{original.stem}__revised_{timestamp}_{counter}{original.suffix}"
        )
        counter += 1
    return candidate


def _local_supported_files(destination: Path) -> list[Path]:
    """신규 raw 폴더와 기존 상위 폴더의 직접 저장 원본을 반환합니다."""
    files: list[Path] = []
    for directory in (destination, destination.parent):
        if not directory.exists():
            continue
        files.extend(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_DOWNLOAD_EXTENSIONS
            and not path.name.startswith((".", "~", "$", "eprx_download_"))
        )
    return files


def download_file(
    candidate: pd.Series | dict[str, Any],
    destination_directory: str | Path = DOWNLOAD_DIRECTORY,
    *,
    history_path: str | Path = HISTORY_PATH,
    session: requests.Session | None = None,
    execution_mode: str = "일반 업데이트",
    test_mode: bool = False,
    page_final_url: str = "",
    page_status_code: int | None = None,
) -> dict[str, Any]:
    """파일을 임시 저장·검사한 뒤 중복 또는 수정본 규칙에 따라 보관합니다."""
    _require_automation_approval()
    item = dict(candidate)
    original_name = _safe_file_name(str(item.get("file_name", "")))
    extension = Path(original_name).suffix.lower()
    destination = Path(destination_directory)
    destination.mkdir(parents=True, exist_ok=True)
    history = load_download_history(history_path)
    record = {
        "downloaded_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "source_page": EPRX_RESULTS_PAGE_URL,
        "file_url": item.get("file_url", ""),
        "original_file_name": original_name,
        "saved_file_name": "",
        "file_size": 0,
        "sha256": "",
        "status": "실패",
        "duplicate_of": "",
        "revision_of": "",
        "error_message": "",
        "parse_status": "미확인",
        "parse_error": "",
        "execution_mode": execution_mode,
        "test_mode": test_mode,
        "page_final_url": page_final_url,
        "page_status_code": page_status_code,
        "parsed_row_count": 0,
        "primary_reserve_row_count": 0,
        "detected_date_min": "",
        "detected_date_max": "",
        "detected_encoding": "",
        "detected_areas": "",
        "detected_zones": "",
        "required_metrics_present": False,
    }
    temp_path: Path | None = None
    try:
        response = _session(session).get(
            str(item["file_url"]),
            timeout=REQUEST_TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if not _content_type_allowed(content_type, extension):
            raise EprxDownloadError(
                f"예상하지 않은 Content-Type입니다: {content_type or '없음'}"
            )
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=extension,
            prefix="eprx_download_",
            dir=destination,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            size = 0
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_FILE_SIZE_BYTES:
                    raise EprxDownloadError(
                        f"최대 파일 크기 {MAX_FILE_SIZE_BYTES:,} bytes를 초과했습니다."
                    )
                handle.write(chunk)
        if size == 0:
            raise EprxDownloadError("다운로드한 파일 크기가 0입니다.")
        leading = temp_path.read_bytes()[:1024].lstrip().lower()
        if leading.startswith((b"<!doctype html", b"<html")):
            raise EprxDownloadError(
                "다운로드 응답이 HTML 문서이므로 원본 파일로 저장하지 않았습니다."
            )
        sha256 = calculate_file_hash(temp_path)
        record.update({"file_size": size, "sha256": sha256})
        duplicate = history.loc[history["sha256"].eq(sha256)]
        if not duplicate.empty:
            record.update(
                {
                    "status": "중복 건너뜀",
                    "duplicate_of": str(duplicate.iloc[-1]["saved_file_name"]),
                }
            )
            temp_path.unlink(missing_ok=True)
            temp_path = None
        else:
            local_duplicate = next(
                (
                    path
                    for path in _local_supported_files(destination)
                    if calculate_file_hash(path) == sha256
                ),
                None,
            )
            if local_duplicate is not None:
                record.update(
                    {
                        "status": "중복 건너뜀",
                        "duplicate_of": local_duplicate.name,
                    }
                )
                temp_path.unlink(missing_ok=True)
                temp_path = None
            final_name = original_name
            existing = next(
                (
                    path
                    for path in (
                        destination / original_name,
                        destination.parent / original_name,
                    )
                    if path.exists()
                ),
                None,
            )
            if temp_path is not None and existing is not None:
                if calculate_file_hash(existing) == sha256:
                    record.update(
                        {
                            "status": "중복 건너뜀",
                            "duplicate_of": existing.name,
                        }
                    )
                    temp_path.unlink(missing_ok=True)
                    temp_path = None
                else:
                    final_name = _revision_name(destination, original_name)
                    record["revision_of"] = original_name
            if temp_path is not None:
                final_path = destination / final_name
                temp_path.replace(final_path)
                temp_path = None
                record.update(
                    {"status": "다운로드 완료", "saved_file_name": final_name}
                )
    except Exception as exc:
        record["error_message"] = str(exc)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    updated = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
    save_download_history(updated, history_path)
    return record


def _classify_candidates(
    candidates: pd.DataFrame, history: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        empty = candidates.copy()
        return empty, empty, empty
    known_urls = set(history["file_url"].dropna().astype(str))
    known_names = set(history["original_file_name"].dropna().astype(str))
    existing_mask = candidates["file_url"].isin(known_urls)
    revision_mask = (
        candidates["file_name"].isin(known_names) & ~existing_mask
    )
    new_mask = ~existing_mask & ~revision_mask
    return (
        candidates.loc[new_mask].copy(),
        candidates.loc[existing_mask].copy(),
        candidates.loc[revision_mask].copy(),
    )


def check_for_new_eprx_files(
    *,
    session: requests.Session | None = None,
    history_path: str | Path = HISTORY_PATH,
) -> dict[str, Any]:
    """다운로드하지 않고 신규·보유·수정 가능 후보를 확인합니다."""
    result: dict[str, Any] = {
        "all_links": pd.DataFrame(columns=CANDIDATE_COLUMNS),
        "primary_candidates": pd.DataFrame(columns=CANDIDATE_COLUMNS),
        "new_candidates": pd.DataFrame(columns=CANDIDATE_COLUMNS),
        "existing_files": pd.DataFrame(columns=CANDIDATE_COLUMNS),
        "revision_candidates": pd.DataFrame(columns=CANDIDATE_COLUMNS),
        "errors": [],
        "dry_run": True,
        "page_details": {},
    }
    try:
        page_details = fetch_results_page_details(session=session)
        result["page_details"] = {
            key: value for key, value in page_details.items() if key != "html"
        }
        if page_details["is_agreement_page"]:
            raise EprxAutomationPermissionError(
                "거래실적 동의 페이지가 반환되었습니다. 브라우저의 동의 상태는 "
                "Python requests 세션에 자동 공유되지 않습니다."
            )
        if page_details["restriction_phrases"]:
            _check_restrictions(str(page_details["html"]))
        if not page_details["is_results_page"]:
            raise EprxDownloadError(
                "응답이 정상 거래실적 페이지로 확인되지 않았습니다."
            )
        links = discover_download_links(
            str(page_details["html"]), str(page_details["final_url"])
        )
        if links.empty:
            raise EprxDownloadError(
                "거래실적 페이지에서 지원 다운로드 링크를 찾지 못했습니다."
            )
        candidates = identify_primary_reserve_candidates(links)
        primary = candidates.loc[
            candidates["is_primary_reserve_candidate"].fillna(False)
        ].copy()
        history = load_download_history(history_path)
        new, existing, revisions = _classify_candidates(primary, history)
        result.update(
            {
                "all_links": links,
                "primary_candidates": primary,
                "new_candidates": new,
                "existing_files": existing,
                "revision_candidates": revisions,
            }
        )
    except Exception as exc:
        result["errors"].append(str(exc))
    return result


def update_eprx_files(
    dry_run: bool = False,
    *,
    max_downloads: int = 1,
    test_mode: bool = False,
    session: requests.Session | None = None,
    destination_directory: str | Path = DOWNLOAD_DIRECTORY,
    history_path: str | Path = HISTORY_PATH,
    log_path: str | Path = UPDATE_LOG_PATH,
) -> dict[str, Any]:
    """후보 확인 후 최대 지정 개수만 다운로드하고 파싱 결과를 기록합니다."""
    if max_downloads < 1:
        raise ValueError("max_downloads는 1 이상이어야 합니다.")
    effective_limit = 1 if test_mode else min(max_downloads, MAX_DOWNLOADS_PER_RUN)
    execution_mode = (
        "dry-run" if dry_run else "시험 다운로드" if test_mode else "일반 업데이트"
    )
    checked = check_for_new_eprx_files(
        session=session, history_path=history_path
    )
    checked["dry_run"] = dry_run
    checked["test_mode"] = test_mode
    checked["selected_candidate"] = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    checked["download_results"] = pd.DataFrame(columns=HISTORY_COLUMNS)
    page = checked.get("page_details", {})
    candidates = pd.concat(
        [checked["new_candidates"], checked["revision_candidates"]],
        ignore_index=True,
    )
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["published_date", "discovered_at"],
            ascending=False,
            na_position="last",
        )
        checked["selected_candidate"] = candidates.head(effective_limit).copy()
    if checked["errors"] or dry_run:
        _append_update_log(
            {
                "executed_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
                "execution_mode": execution_mode,
                "approval_env_enabled": automation_approved(),
                "source_page": EPRX_RESULTS_PAGE_URL,
                "final_url": page.get("final_url", ""),
                "page_status": page.get("status_code", ""),
                "discovered_link_count": len(checked["all_links"]),
                "primary_candidate_count": len(checked["primary_candidates"]),
                "selected_file": (
                    checked["selected_candidate"].iloc[0]["file_name"]
                    if not checked["selected_candidate"].empty
                    else ""
                ),
                "download_status": "미실행",
                "saved_file": "",
                "sha256": "",
                "parse_status": "미실행",
                "parsed_rows": 0,
                "primary_rows": 0,
                "error_message": " | ".join(checked["errors"]),
            },
            log_path,
        )
        return checked

    targets = checked["selected_candidate"]
    records: list[dict[str, Any]] = []
    for _, candidate in targets.iterrows():
        record = download_file(
            candidate,
            destination_directory,
            history_path=history_path,
            session=session,
            execution_mode=execution_mode,
            test_mode=test_mode,
            page_final_url=str(page.get("final_url", "")),
            page_status_code=page.get("status_code"),
        )
        if record["status"] == "다운로드 완료":
            file_path = Path(destination_directory) / record["saved_file_name"]
            try:
                raw = read_eprx_file(file_path)
                normalized = normalize_eprx_data(raw, file_path)
                if normalized.empty:
                    raise ValueError("1차 조정력 정규화 결과가 없습니다.")
                required = {
                    "max_price",
                    "min_price",
                    "avg_price",
                    "awarded_volume",
                    "bid_volume",
                    "procurement_volume",
                }
                missing = sorted(required - set(normalized.columns))
                if missing or normalized[list(required)].isna().all().any():
                    raise ValueError(
                        "필수 6개 지표 확인 실패: " + ", ".join(missing)
                    )
                dates = normalized["delivery_date"].dropna()
                record.update(
                    {
                        "parse_status": "파싱 성공",
                        "parsed_row_count": len(normalized),
                        "primary_reserve_row_count": len(normalized),
                        "detected_date_min": (
                            dates.min().strftime("%Y-%m-%d") if not dates.empty else ""
                        ),
                        "detected_date_max": (
                            dates.max().strftime("%Y-%m-%d") if not dates.empty else ""
                        ),
                        "detected_encoding": raw.attrs.get("encoding", ""),
                        "detected_areas": ", ".join(
                            sorted(normalized["area"].dropna().unique())
                        ),
                        "detected_zones": ", ".join(
                            sorted(normalized["frequency_zone"].dropna().unique())
                        ),
                        "required_metrics_present": True,
                    }
                )
            except Exception as exc:
                record["parse_status"] = "파싱 실패"
                record["parse_error"] = str(exc)
            history = load_download_history(history_path)
            history = history.astype(object)
            match = history.index[
                history["downloaded_at"].astype(str).eq(str(record["downloaded_at"]))
            ]
            if len(match):
                for column in (
                    "parse_status",
                    "parse_error",
                    "parsed_row_count",
                    "primary_reserve_row_count",
                    "detected_date_min",
                    "detected_date_max",
                    "detected_encoding",
                    "detected_areas",
                    "detected_zones",
                    "required_metrics_present",
                ):
                    history.loc[match[-1], column] = record[column]
                save_download_history(history, history_path)
        records.append(record)
    checked["download_results"] = pd.DataFrame(records, columns=HISTORY_COLUMNS)
    first_record = records[0] if records else {}
    _append_update_log(
        {
            "executed_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
            "execution_mode": execution_mode,
            "approval_env_enabled": automation_approved(),
            "source_page": EPRX_RESULTS_PAGE_URL,
            "final_url": page.get("final_url", ""),
            "page_status": page.get("status_code", ""),
            "discovered_link_count": len(checked["all_links"]),
            "primary_candidate_count": len(checked["primary_candidates"]),
            "selected_file": (
                targets.iloc[0]["file_name"] if not targets.empty else ""
            ),
            "download_status": first_record.get("status", "미실행"),
            "saved_file": first_record.get("saved_file_name", ""),
            "sha256": first_record.get("sha256", ""),
            "parse_status": first_record.get("parse_status", "미실행"),
            "parsed_rows": first_record.get("parsed_row_count", 0),
            "primary_rows": first_record.get("primary_reserve_row_count", 0),
            "error_message": first_record.get("error_message", "")
            or first_record.get("parse_error", ""),
        },
        log_path,
    )
    return checked
