"""사용자가 내려받은 EPRX 거래실적 파일을 읽고 검증합니다.

원본 파일은 읽기만 하며 수정하거나 다시 저장하지 않습니다. 202607 파일에서
확인한 행렬형 CSV 구조와 실제 일본어 표기를 기준으로 매핑합니다.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.sample_data import AREAS_BY_ZONE, AREA_TO_ZONE

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}
ENCODING_CANDIDATES = ("cp932", "shift_jis", "utf-8-sig", "utf-8")
JAPANESE_AREAS = {
    "北海道": "Hokkaido",
    "東北": "Tohoku",
    "東京": "Tokyo",
    "中部": "Chubu",
    "北陸": "Hokuriku",
    "関西": "Kansai",
    "中国": "Chugoku",
    "四国": "Shikoku",
    "九州": "Kyushu",
}
METRIC_MAP = {
    "募集量（TSO別）[MW]": "procurement_volume",
    "応札量合計（電源属地別）[MW]": "bid_volume",
    "落札量合計（電源属地別）[MW]": "awarded_volume",
    "最高落札価格（電源属地別）[円/kW・30分]": "max_price",
    "最低落札価格（電源属地別）[円/kW・30分]": "min_price",
    "平均落札価格（電源属地別）[円/kW・30分]": "avg_price",
}
REQUIRED_METRICS = tuple(METRIC_MAP.values())
NORMALIZED_COLUMNS = [
    "delivery_date",
    "period_no",
    "period_start",
    "area",
    "original_area",
    "frequency_zone",
    "product",
    "original_product",
    "max_price",
    "min_price",
    "avg_price",
    "awarded_volume",
    "bid_volume",
    "procurement_volume",
    "price_unit",
    "volume_unit",
    "source_file",
    "source_status",
    "duplicate_candidate",
]
ERROR_COLUMNS = [
    "source_file",
    "severity",
    "error_code",
    "message",
    "delivery_date",
    "area",
    "product",
    "period_no",
]


def find_eprx_files(data_directory: str | Path) -> list[Path]:
    """기존 루트와 raw에서 지원 파일을 찾고 동일 해시는 한 번만 반환합니다."""
    directory = Path(data_directory)
    if not directory.exists():
        return []
    candidates = sorted(
        path
        for path in [*directory.iterdir(), *(directory / "raw").glob("*")]
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith((".", "~", "$"))
        and not path.name.startswith("~$")
    )
    unique_files: list[Path] = []
    seen_hashes: set[str] = set()
    for path in candidates:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        file_hash = digest.hexdigest()
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)
        unique_files.append(path)
    return unique_files


def _decode_csv(file_path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    raw_bytes = file_path.read_bytes()
    attempts: list[dict[str, Any]] = []
    successful: dict[str, str] = {}
    for encoding in ENCODING_CANDIDATES:
        try:
            text = raw_bytes.decode(encoding, errors="strict")
            replacement_count = text.count("\ufffd")
            japanese_header_ok = all(
                label in text[:1000] for label in ("対象年月", "商品区分")
            )
            success = replacement_count == 0 and japanese_header_ok
            attempts.append(
                {
                    "encoding": encoding,
                    "success": success,
                    "replacement_characters": replacement_count,
                    "japanese_header_ok": japanese_header_ok,
                    "message": (
                        ""
                        if success
                        else "대체문자 포함 또는 일본어 메타데이터 확인 실패"
                    ),
                }
            )
            if success:
                successful[encoding] = text
        except UnicodeDecodeError as exc:
            attempts.append(
                {
                    "encoding": encoding,
                    "success": False,
                    "replacement_characters": np.nan,
                    "japanese_header_ok": False,
                    "message": str(exc),
                }
            )
    for encoding in ENCODING_CANDIDATES:
        if encoding in successful:
            return successful[encoding], encoding, attempts
    details = "; ".join(
        f"{item['encoding']}: {item['message']}" for item in attempts
    )
    raise ValueError(f"지원 인코딩으로 CSV를 디코딩할 수 없습니다. {details}")


def _rows_to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    width = max((len(row) for row in rows), default=0)
    padded = [list(row) + [None] * (width - len(row)) for row in rows]
    return pd.DataFrame(padded, columns=[f"column_{index}" for index in range(width)])


def read_eprx_file(file_path: str | Path) -> pd.DataFrame:
    """CSV/Excel을 구분해 읽고 확인된 메타데이터를 ``attrs``에 보존합니다."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")

    if suffix == ".csv":
        text, encoding, attempts = _decode_csv(path)
        sample = text[:8192]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error as exc:
            raise ValueError(f"CSV 구분자를 확인할 수 없습니다: {exc}") from exc
        rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
        frame = _rows_to_frame(rows)
        frame.attrs.update(
            {
                "encoding": encoding,
                "encoding_attempts": attempts,
                "delimiter": delimiter,
                "sheet_names": [],
                "file_type": "CSV",
            }
        )
        return frame

    try:
        workbook = pd.ExcelFile(path)
        sheets = {
            sheet: pd.read_excel(path, sheet_name=sheet, header=None)
            for sheet in workbook.sheet_names
        }
    except ImportError as exc:
        raise RuntimeError(
            f"{suffix} 파일을 읽는 선택 패키지가 설치되지 않았습니다: {exc}"
        ) from exc
    except Exception as exc:
        raise ValueError(f"Excel 파일을 읽지 못했습니다: {exc}") from exc
    frame = pd.concat(
        [sheet.assign(_sheet_name=name) for name, sheet in sheets.items()],
        ignore_index=True,
    )
    frame.attrs.update(
        {
            "encoding": "해당 없음",
            "encoding_attempts": [],
            "delimiter": "해당 없음",
            "sheet_names": workbook.sheet_names,
            "file_type": "Excel",
        }
    )
    return frame


def _metadata_rows(raw_data: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    rows = raw_data.where(pd.notna(raw_data), None).values.tolist()
    header = next((row for row in rows if row[0] == "H"), None)
    product = next((row for row in rows if row[0] == "P"), None)
    areas = next((row for row in rows if row[0] == "TT"), None)
    missing = [
        label
        for label, value in (("H", header), ("P", product), ("TT", areas))
        if value is None
    ]
    if missing:
        raise ValueError(f"필수 메타데이터 행이 없습니다: {', '.join(missing)}")
    return header, product, areas


def normalize_eprx_data(
    raw_data: pd.DataFrame, source_file: str | Path
) -> pd.DataFrame:
    """확인된 일본어 지표와 지역을 내부 표준 열로 변환합니다."""
    _, product_row, area_row = _metadata_rows(raw_data)
    original_status = str(product_row[1])
    original_product = str(product_row[3])
    if original_product != "一次調整力":
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    area_columns = {
        index: JAPANESE_AREAS[name]
        for index, name in enumerate(area_row)
        if name in JAPANESE_AREAS
    }
    if not area_columns:
        raise ValueError("지원하는 9개 일본어 지역 열을 찾지 못했습니다.")

    rows = raw_data.where(pd.notna(raw_data), None).values.tolist()
    metric_rows = [
        row for row in rows if len(row) > 1 and row[1] in METRIC_MAP
    ]
    found_metrics = {METRIC_MAP[row[1]] for row in metric_rows}
    missing_metrics = sorted(set(REQUIRED_METRICS) - found_metrics)
    if missing_metrics:
        raise ValueError(f"필수 6개 지표가 누락되었습니다: {', '.join(missing_metrics)}")

    records: dict[tuple[str, str], dict[str, Any]] = {}
    identifier_pattern = re.compile(r"^(?P<date>\d{8})B(?P<period>\d{2})$")
    for row in metric_rows:
        identifier = str(row[0])
        match = identifier_pattern.fullmatch(identifier)
        if not match:
            continue
        for column_index, area in area_columns.items():
            original_area = str(area_row[column_index])
            key = (identifier, original_area)
            record = records.setdefault(
                key,
                {
                    "delivery_date": match.group("date"),
                    "period_no": int(match.group("period")),
                    "period_start": (
                        f"{(int(match.group('period')) - 1) // 2:02d}:"
                        f"{((int(match.group('period')) - 1) % 2) * 30:02d}"
                    ),
                    "area": area,
                    "original_area": original_area,
                    "frequency_zone": AREA_TO_ZONE[area],
                    "product": "Primary Reserve",
                    "original_product": original_product,
                    "price_unit": "円/kW・30分",
                    "volume_unit": "MW",
                    "source_file": Path(source_file).name,
                    "source_status": original_status,
                },
            )
            record[METRIC_MAP[row[1]]] = row[column_index]

    normalized = pd.DataFrame(records.values())
    for column in REQUIRED_METRICS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["delivery_date"] = pd.to_datetime(
        normalized["delivery_date"], format="%Y%m%d", errors="coerce"
    )
    duplicate_key = ["delivery_date", "area", "original_product", "period_no"]
    normalized["duplicate_candidate"] = normalized.duplicated(
        duplicate_key, keep=False
    )
    return normalized.reindex(columns=NORMALIZED_COLUMNS)


def _error_record(
    row: pd.Series | None,
    source_file: str,
    severity: str,
    code: str,
    message: str,
    **overrides: Any,
) -> dict[str, Any]:
    record = {
        "source_file": source_file,
        "severity": severity,
        "error_code": code,
        "message": message,
        "delivery_date": None if row is None else row.get("delivery_date"),
        "area": None if row is None else row.get("area"),
        "product": None if row is None else row.get("original_product"),
        "period_no": None if row is None else row.get("period_no"),
    }
    record.update(overrides)
    return record


def validate_eprx_data(data: pd.DataFrame) -> pd.DataFrame:
    """행 수준 오류와 데이터 완전성 경고를 삭제 없이 반환합니다."""
    errors: list[dict[str, Any]] = []
    if data.empty:
        return pd.DataFrame(errors, columns=ERROR_COLUMNS)

    for _, row in data.loc[data["delivery_date"].isna()].iterrows():
        errors.append(
            _error_record(row, row["source_file"], "Fatal", "invalid_date", "날짜 변환 실패")
        )
    invalid_period = data["period_no"].isna() | ~data["period_no"].between(1, 48)
    for _, row in data.loc[invalid_period].iterrows():
        errors.append(
            _error_record(row, row["source_file"], "Fatal", "invalid_period", "시간대 번호가 1~48 범위가 아님")
        )
    for column in REQUIRED_METRICS:
        for _, row in data.loc[data[column].isna()].iterrows():
            errors.append(
                _error_record(
                    row,
                    row["source_file"],
                    "Fatal",
                    f"invalid_{column}",
                    f"{column} 값이 없거나 숫자로 변환되지 않음",
                )
            )

    duplicate_key = ["delivery_date", "area", "original_product", "period_no"]
    duplicates = data.duplicated(duplicate_key, keep=False)
    for _, row in data.loc[duplicates].iterrows():
        errors.append(
            _error_record(row, row["source_file"], "Review", "duplicate_candidate", "동일 날짜·지역·상품·시간대 중복 후보")
        )

    relations = [
        (
            (data["min_price"] > data["avg_price"]) | (data["avg_price"] > data["max_price"]),
            "price_order",
            "min_price <= avg_price <= max_price 관계 검토 필요",
        ),
        (
            data["awarded_volume"] > data["bid_volume"],
            "award_exceeds_bid",
            "낙찰량이 입찰량보다 큼",
        ),
        (
            data["awarded_volume"] > data["procurement_volume"],
            "award_exceeds_procurement",
            (
                "낙찰량（電源属地別）이 모집량（TSO別）보다 큼: "
                "원본 또는 집계기준 확인 필요"
            ),
        ),
        (
            data[["awarded_volume", "bid_volume", "procurement_volume"]].lt(0).any(axis=1),
            "negative_volume",
            "음수 물량 존재",
        ),
    ]
    for mask, code, message in relations:
        for _, row in data.loc[mask.fillna(False)].iterrows():
            errors.append(
                _error_record(row, row["source_file"], "Review", code, message)
            )

    valid_dates = data.dropna(subset=["delivery_date"])
    daily = valid_dates.groupby(
        ["source_file", "delivery_date", "area", "original_product"], dropna=False
    )["period_no"].nunique()
    for key, count in daily.loc[daily.ne(48)].items():
        source, delivery_date, area, product = key
        errors.append(
            _error_record(
                None,
                source,
                "Review",
                "incomplete_day",
                f"하루 시간대 {count}/48개",
                delivery_date=delivery_date,
                area=area,
                product=product,
            )
        )

    dated = valid_dates.copy()
    dated["week_start"] = dated["delivery_date"] - pd.to_timedelta(
        dated["delivery_date"].dt.weekday, unit="D"
    )
    weekly = dated.groupby(["source_file", "week_start"])["delivery_date"].nunique()
    for (source, week_start), count in weekly.loc[weekly.ne(7)].items():
        errors.append(
            _error_record(
                None,
                source,
                "Review",
                "incomplete_week",
                f"{week_start:%Y-%m-%d} 주차 날짜 {count}/7일",
                delivery_date=week_start,
            )
        )

    present = set(data["area"].dropna().unique())
    for zone, expected in AREAS_BY_ZONE.items():
        missing = sorted(set(expected) - present)
        if missing:
            errors.append(
                _error_record(
                    None,
                    "전체 파일",
                    "Fatal",
                    "missing_area",
                    f"{zone} 필수 지역 누락: {', '.join(missing)}",
                )
            )
    return pd.DataFrame(errors, columns=ERROR_COLUMNS)


def load_all_eprx_data(
    data_directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """모든 지원 파일을 읽어 정규화 데이터, 오류, 파일 로그를 반환합니다."""
    normalized_frames: list[pd.DataFrame] = []
    file_errors: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for path in find_eprx_files(data_directory):
        summary: dict[str, Any] = {
            "source_file": path.name,
            "file_type": path.suffix.lower().lstrip(".").upper(),
            "file_size_bytes": path.stat().st_size,
            "modified_at": pd.Timestamp(path.stat().st_mtime, unit="s", tz="Asia/Tokyo"),
            "success": False,
            "encoding": None,
            "encoding_attempts": None,
            "delimiter": None,
            "header_row": None,
            "raw_rows": 0,
            "normalized_rows": 0,
            "primary_reserve_rows": 0,
            "error_rows": 0,
            "duplicate_rows": 0,
            "missing_required_columns": "",
            "error_message": "",
        }
        try:
            raw = read_eprx_file(path)
            summary.update(
                {
                    "encoding": raw.attrs.get("encoding"),
                    "encoding_attempts": str(raw.attrs.get("encoding_attempts")),
                    "delimiter": raw.attrs.get("delimiter"),
                    "header_row": "1~3행 메타데이터(H/P/TT), 4행부터 반복 지표",
                    "raw_rows": len(raw),
                }
            )
            normalized = normalize_eprx_data(raw, path)
            summary["normalized_rows"] = len(normalized)
            summary["primary_reserve_rows"] = len(normalized)
            normalized_frames.append(normalized)
            summary["success"] = True
        except Exception as exc:
            summary["error_message"] = str(exc)
            if "필수 6개 지표가 누락" in str(exc):
                summary["missing_required_columns"] = str(exc).split(":", 1)[-1].strip()
            file_errors.append(
                _error_record(
                    None, path.name, "Fatal", "file_parse_error", str(exc)
                )
            )
        summaries.append(summary)

    combined = (
        pd.concat(normalized_frames, ignore_index=True)
        if normalized_frames
        else pd.DataFrame(columns=NORMALIZED_COLUMNS)
    )
    if not combined.empty:
        duplicate_key = ["delivery_date", "area", "original_product", "period_no"]
        combined["duplicate_candidate"] = combined.duplicated(
            duplicate_key, keep=False
        )
    validation = validate_eprx_data(combined)
    errors = pd.concat(
        [
            pd.DataFrame(file_errors, columns=ERROR_COLUMNS),
            validation,
        ],
        ignore_index=True,
    )
    summary_frame = pd.DataFrame(summaries)
    if not summary_frame.empty:
        for index, row in summary_frame.iterrows():
            file_mask = errors["source_file"].eq(row["source_file"])
            summary_frame.loc[index, "error_rows"] = int(file_mask.sum())
            if not combined.empty:
                summary_frame.loc[index, "duplicate_rows"] = int(
                    (
                        combined["source_file"].eq(row["source_file"])
                        & combined["duplicate_candidate"]
                    ).sum()
                )
    return combined, errors, summary_frame
