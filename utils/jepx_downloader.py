"""JEPX 공식 공개 폼을 이용한 Day-Ahead 연도별 CSV 다운로드."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from utils.jepx_loader import DATE_COLUMN, normalize_jepx_data, read_jepx_file

BASE_URL = "https://www.jepx.jp"
SPOT_PAGE_URL = f"{BASE_URL}/electricpower/market-data/spot/"
YEAR_URL = f"{BASE_URL}/js/get_graph_year.php"
DOWNLOAD_URL = f"{BASE_URL}/_download.php"
DEFAULT_DESTINATION = Path(__file__).resolve().parents[1] / "data" / "jepx" / "raw"
REQUEST_TIMEOUT = (10, 60)
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
USER_AGENT = "japan-market-monitor/1.0 (public JEPX data downloader)"


class JepxDownloadError(RuntimeError):
    """공식 JEPX 공개 파일을 안전하게 받지 못했을 때 발생합니다."""


def _session(session: requests.Session | None = None) -> requests.Session:
    client = session or requests.Session()
    client.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Referer": SPOT_PAGE_URL,
            "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8",
        }
    )
    return client


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def fetch_available_years(
    session: requests.Session | None = None,
) -> list[int]:
    """공식 페이지가 제공하는 최신·최초 연도로 다운로드 가능 연도를 만듭니다."""
    try:
        response = _session(session).get(
            YEAR_URL,
            params={"dir": "spot_summary"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise JepxDownloadError(f"JEPX 공개 연도 조회 실패: {exc}") from exc

    match = re.fullmatch(r"\s*(\d{4}),(\d{4})\s*", response.text)
    if not match:
        raise JepxDownloadError(
            f"JEPX 공개 연도 응답 형식이 예상과 다릅니다: {response.text[:100]!r}"
        )
    latest, oldest = map(int, match.groups())
    if oldest > latest or oldest < 2000 or latest > 2100:
        raise JepxDownloadError(f"JEPX 공개 연도 범위가 유효하지 않습니다: {oldest}~{latest}")
    return list(range(latest, oldest - 1, -1))


def _validate_download(path: Path, year: int) -> dict[str, Any]:
    raw = read_jepx_file(path)
    wide, long = normalize_jepx_data(raw, path)
    if wide.empty or long.empty:
        raise JepxDownloadError("다운로드 CSV에 분석 가능한 JEPX 데이터가 없습니다.")

    dates = pd.to_datetime(raw[DATE_COLUMN], errors="coerce").dropna()
    if dates.empty:
        raise JepxDownloadError("다운로드 CSV의 受渡日을 해석할 수 없습니다.")
    fiscal_start = pd.Timestamp(year=year, month=4, day=1)
    fiscal_end = pd.Timestamp(year=year + 1, month=3, day=31)
    if not dates.between(fiscal_start, fiscal_end).all():
        raise JepxDownloadError(
            f"다운로드 CSV에 {year}년도 범위를 벗어난 날짜가 포함되어 있습니다."
        )
    return {
        "encoding": raw.attrs.get("encoding"),
        "raw_rows": len(raw),
        "normalized_rows": len(long),
        "date_min": dates.min().normalize(),
        "date_max": dates.max().normalize(),
    }


def download_spot_summary(
    year: int | None = None,
    destination_directory: str | Path = DEFAULT_DESTINATION,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """최신 또는 지정 연도의 공개 CSV를 검증 후 원자적으로 저장합니다."""
    client = _session(session)
    available_years = fetch_available_years(client)
    selected_year = available_years[0] if year is None else int(year)
    if selected_year not in available_years:
        raise JepxDownloadError(
            f"{selected_year}년도는 현재 공식 다운로드 범위에 없습니다."
        )

    filename = f"spot_summary_{selected_year}.csv"
    destination = Path(destination_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / filename
    temp_path: Path | None = None

    try:
        try:
            response = client.post(
                DOWNLOAD_URL,
                data={"dir": "spot_summary", "file": filename},
                timeout=REQUEST_TIMEOUT,
                stream=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise JepxDownloadError(f"JEPX 공개 CSV 다운로드 실패: {exc}") from exc

        content_type = response.headers.get("Content-Type", "").lower()
        disposition = response.headers.get("Content-Disposition", "")
        if "text/html" in content_type or filename not in disposition:
            raise JepxDownloadError(
                "JEPX 응답이 요청한 공개 CSV 파일 형식이 아닙니다."
            )

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{filename}.",
            suffix=".csv",
            dir=destination,
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            size = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise JepxDownloadError("JEPX 다운로드가 허용 크기 50MB를 초과했습니다.")
                stream.write(chunk)

        if not temp_path.stat().st_size:
            raise JepxDownloadError("JEPX 다운로드 파일의 크기가 0입니다.")

        validation = _validate_download(temp_path, selected_year)
        if final_path.exists():
            existing = read_jepx_file(final_path)
            existing_dates = pd.to_datetime(
                existing[DATE_COLUMN], errors="coerce"
            ).dropna()
            if (
                not existing_dates.empty
                and validation["date_max"] < existing_dates.max().normalize()
            ):
                raise JepxDownloadError(
                    "공식 다운로드의 최신 날짜가 기존 파일보다 이전이므로 교체하지 않았습니다."
                )

        new_hash = _sha256(temp_path)
        previous_hash = _sha256(final_path) if final_path.exists() else None
        if previous_hash == new_hash:
            temp_path.unlink()
            temp_path = None
            status = "변경 없음"
        else:
            os.replace(temp_path, final_path)
            temp_path = None
            status = "신규 저장" if previous_hash is None else "업데이트"

        return {
            "status": status,
            "year": selected_year,
            "file_name": filename,
            "file_path": str(final_path),
            "file_size": final_path.stat().st_size,
            "sha256": new_hash,
            "source_url": DOWNLOAD_URL,
            **validation,
        }
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main(argv: list[str] | None = None) -> int:
    """CLI에서 최신 공개 CSV를 확인하고 처리 결과에 맞는 종료 코드를 반환합니다."""
    parser = argparse.ArgumentParser(
        description="JEPX 공식 공개 Day-Ahead Spot CSV를 검증 후 갱신합니다."
    )
    parser.add_argument(
        "--year",
        type=int,
        help="다운로드할 연도입니다. 생략하면 공식 페이지의 최신 연도를 사용합니다.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="CSV 저장 폴더입니다. 기본값: data/jepx/raw",
    )
    args = parser.parse_args(argv)

    try:
        result = download_spot_summary(
            year=args.year,
            destination_directory=args.destination,
        )
    except JepxDownloadError as exc:
        print(f"JEPX 다운로드 실패: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"JEPX 다운로드 중 예상하지 못한 오류가 발생했습니다: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        f"{result['file_name']}: {result['status']} | "
        f"{result['date_min']:%Y-%m-%d} ~ {result['date_max']:%Y-%m-%d} | "
        f"정규화 {result['normalized_rows']:,}행 | SHA-256 {result['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
