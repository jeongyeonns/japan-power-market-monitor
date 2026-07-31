"""Download the approved EPRX primary-reserve prompt ZIP once per JST day."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import requests

from utils.eprx_downloader import (
    EprxDownloadError,
    REQUEST_TIMEOUT_SECONDS,
    automation_approved,
)
from utils.eprx_loader import normalize_eprx_data, read_eprx_file

EPRX_PRIMARY_PROMPT_ZIP_URL = (
    "https://www.eprx.or.jp/information/files/2026_1-0_prompt.zip"
)
USER_AGENT = (
    "LX-International-Japan-Market-Monitor/1.0 "
    "(EPRX-approved; internal-analysis; daily-download)"
)
BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_DIRECTORY = BASE_DIR / "data" / "eprx" / "raw"
STATE_PATH = (
    BASE_DIR
    / "data"
    / "eprx"
    / "metadata"
    / "eprx_prompt_auto_download_state.json"
)
MAX_ARCHIVE_SIZE_BYTES = 50 * 1024 * 1024
MAX_EXTRACTED_SIZE_BYTES = 250 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100
REQUIRED_METRICS = {
    "max_price",
    "min_price",
    "avg_price",
    "awarded_volume",
    "bid_volume",
    "procurement_volume",
}


def _tokyo_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Tokyo")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".tmp",
        prefix="eprx_prompt_state_",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    temporary.replace(path)


def load_state(path: str | Path = STATE_PATH) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EprxDownloadError(
            f"EPRX 자동 다운로드 상태를 읽지 못했습니다: {exc}"
        ) from exc


def _require_approval() -> None:
    if not automation_approved():
        raise EprxDownloadError(
            "EPRX 자동 취득 승인 확인이 필요합니다. "
            "EPRX_AUTOMATION_APPROVED=true를 설정하세요."
        )


def _session(session: requests.Session | None = None) -> requests.Session:
    client = session or requests.Session()
    client.headers.update(
        {
            "User-Agent": os.getenv("EPRX_USER_AGENT", USER_AGENT),
            "Accept": "application/zip, application/octet-stream",
        }
    )
    return client


def _is_primary_prompt_csv(member_name: str) -> bool:
    name = PurePosixPath(member_name.replace("\\", "/")).name
    lowered = name.lower()
    return (
        bool(name)
        and lowered.endswith(".csv")
        and "1-0" in lowered
        and "prompt" in lowered
        and not name.startswith((".", "~", "$"))
    )


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [item for item in archive.infolist() if not item.is_dir()]
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise EprxDownloadError(
            f"ZIP 내부 파일 수가 제한({MAX_ARCHIVE_MEMBERS}개)을 초과했습니다."
        )
    total_size = sum(item.file_size for item in members)
    if total_size > MAX_EXTRACTED_SIZE_BYTES:
        raise EprxDownloadError(
            "ZIP 압축 해제 예상 크기가 "
            f"{MAX_EXTRACTED_SIZE_BYTES:,} bytes를 초과했습니다."
        )

    selected: list[zipfile.ZipInfo] = []
    seen_names: set[str] = set()
    for item in members:
        path = PurePosixPath(item.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise EprxDownloadError(
                f"안전하지 않은 ZIP 내부 경로입니다: {item.filename}"
            )
        if not _is_primary_prompt_csv(item.filename):
            continue
        output_name = path.name
        if output_name in seen_names:
            raise EprxDownloadError(f"ZIP 내부 파일명이 중복됩니다: {output_name}")
        seen_names.add(output_name)
        selected.append(item)

    if not selected:
        raise EprxDownloadError(
            "ZIP에서 1차 조정력 속보치 CSV"
            "(파일명에 1-0 및 prompt)를 찾지 못했습니다."
        )
    return selected


def _validate_csv(path: Path) -> dict[str, Any]:
    raw = read_eprx_file(path)
    normalized = normalize_eprx_data(raw, path)
    if normalized.empty:
        raise EprxDownloadError(f"{path.name}: 1차 조정력 파싱 결과가 없습니다.")
    missing = sorted(REQUIRED_METRICS - set(normalized.columns))
    if missing or normalized[list(REQUIRED_METRICS)].isna().all().any():
        raise EprxDownloadError(
            f"{path.name}: 필수 지표 검증 실패"
            + (f" ({', '.join(missing)})" if missing else "")
        )

    dates = pd.to_datetime(normalized["delivery_date"], errors="coerce").dropna()
    return {
        "file_name": path.name,
        "rows": int(len(normalized)),
        "date_min": dates.min().date().isoformat() if not dates.empty else "",
        "date_max": dates.max().date().isoformat() if not dates.empty else "",
        "areas": sorted(normalized["area"].dropna().astype(str).unique().tolist()),
    }


def _download_archive(
    destination: Path,
    *,
    session: requests.Session | None = None,
) -> tuple[str, int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        response = _session(session).get(
            EPRX_PRIMARY_PROMPT_ZIP_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if not any(
            allowed in content_type
            for allowed in (
                "application/zip",
                "application/x-zip-compressed",
                "application/octet-stream",
            )
        ):
            raise EprxDownloadError(
                f"예상하지 않은 Content-Type입니다: {content_type or '없음'}"
            )

        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=128 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_ARCHIVE_SIZE_BYTES:
                    raise EprxDownloadError(
                        "ZIP 크기가 제한"
                        f"({MAX_ARCHIVE_SIZE_BYTES:,} bytes)을 초과했습니다."
                    )
                digest.update(chunk)
                handle.write(chunk)
    except requests.RequestException as exc:
        raise EprxDownloadError(f"EPRX 속보치 ZIP 요청 실패: {exc}") from exc

    if size == 0:
        raise EprxDownloadError("다운로드한 ZIP 파일의 크기가 0입니다.")
    if not zipfile.is_zipfile(destination):
        raise EprxDownloadError("다운로드 응답이 유효한 ZIP 파일이 아닙니다.")
    return digest.hexdigest(), size, str(response.url)


def download_primary_prompt_zip(
    *,
    destination_directory: str | Path = DOWNLOAD_DIRECTORY,
    state_path: str | Path = STATE_PATH,
    session: requests.Session | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Request the fixed URL once, then atomically publish validated CSV files."""
    _require_approval()
    now = _tokyo_now()
    run_date = (today or now.date()).isoformat()
    state_file = Path(state_path)
    previous = load_state(state_file)
    if previous.get("last_request_date_jst") == run_date:
        return {
            "status": "오늘 이미 요청함",
            "requested": False,
            "request_date_jst": run_date,
            "source_url": EPRX_PRIMARY_PROMPT_ZIP_URL,
            "files": previous.get("files", []),
        }

    state: dict[str, Any] = {
        "last_request_date_jst": run_date,
        "requested_at_jst": now.isoformat(),
        "source_url": EPRX_PRIMARY_PROMPT_ZIP_URL,
        "status": "요청 시작",
        "user_agent": os.getenv("EPRX_USER_AGENT", USER_AGENT),
        "files": [],
    }
    # Persist immediately before GET so a failed attempt is not retried that day.
    _write_json_atomic(state_file, state)

    destination = Path(destination_directory)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="eprx_prompt_") as temp_directory:
            temporary_directory = Path(temp_directory)
            archive_path = temporary_directory / "prompt.zip"
            sha256, archive_size, final_url = _download_archive(
                archive_path, session=session
            )
            staged: list[tuple[Path, Path]] = []
            validation: list[dict[str, Any]] = []

            with zipfile.ZipFile(archive_path) as archive:
                for member in _safe_members(archive):
                    output_name = PurePosixPath(
                        member.filename.replace("\\", "/")
                    ).name
                    staged_path = temporary_directory / output_name
                    with archive.open(member) as source, staged_path.open("wb") as target:
                        while chunk := source.read(128 * 1024):
                            target.write(chunk)
                    validation.append(_validate_csv(staged_path))
                    staged.append((staged_path, destination / output_name))

            changed_files: list[str] = []
            unchanged_files: list[str] = []
            for staged_path, final_path in staged:
                new_hash = hashlib.sha256(staged_path.read_bytes()).hexdigest()
                if (
                    final_path.exists()
                    and hashlib.sha256(final_path.read_bytes()).hexdigest()
                    == new_hash
                ):
                    unchanged_files.append(final_path.name)
                    continue
                staged_path.replace(final_path)
                changed_files.append(final_path.name)

        state.update(
            {
                "status": "완료",
                "final_url": final_url,
                "archive_sha256": sha256,
                "archive_size": archive_size,
                "files": validation,
                "changed_files": changed_files,
                "unchanged_files": unchanged_files,
            }
        )
        _write_json_atomic(state_file, state)
        return {
            "status": "완료",
            "requested": True,
            "request_date_jst": run_date,
            "source_url": EPRX_PRIMARY_PROMPT_ZIP_URL,
            "archive_sha256": sha256,
            "changed_files": changed_files,
            "unchanged_files": unchanged_files,
            "files": validation,
        }
    except Exception as exc:
        state.update({"status": "실패", "error": str(exc)})
        _write_json_atomic(state_file, state)
        if isinstance(exc, EprxDownloadError):
            raise
        raise EprxDownloadError(f"EPRX 속보치 자동 반영 실패: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="허가된 EPRX 1차 조정력 속보치 ZIP을 하루 한 번 반영합니다."
    )
    parser.parse_args()
    result = download_primary_prompt_zip()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
