"""확인된 JEPX Day-Ahead CSV를 원본 보존형 wide와 분석용 long으로 변환합니다."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_SUFFIXES = {".csv"}
ENCODING_CANDIDATES = ("cp932", "shift_jis", "utf-8-sig", "utf-8")
DELIMITER = ","
PRICE_UNIT = "円/kWh"

DATE_COLUMN = "受渡日"
PERIOD_COLUMN = "時刻コード"
PRICE_COLUMNS = {
    "システムプライス(円/kWh)": "System",
    "エリアプライス北海道(円/kWh)": "Hokkaido",
    "エリアプライス東北(円/kWh)": "Tohoku",
    "エリアプライス東京(円/kWh)": "Tokyo",
    "エリアプライス中部(円/kWh)": "Chubu",
    "エリアプライス北陸(円/kWh)": "Hokuriku",
    "エリアプライス関西(円/kWh)": "Kansai",
    "エリアプライス中国(円/kWh)": "Chugoku",
    "エリアプライス四国(円/kWh)": "Shikoku",
    "エリアプライス九州(円/kWh)": "Kyushu",
}
WIDE_COLUMN_MAP = {
    DATE_COLUMN: "delivery_date",
    PERIOD_COLUMN: "period_no",
    "売り入札量(kWh)": "sell_bid_volume_kwh",
    "買い入札量(kWh)": "buy_bid_volume_kwh",
    "約定総量(kWh)": "contracted_volume_kwh",
    "システムプライス(円/kWh)": "system_price",
    "エリアプライス北海道(円/kWh)": "hokkaido_price",
    "エリアプライス東北(円/kWh)": "tohoku_price",
    "エリアプライス東京(円/kWh)": "tokyo_price",
    "エリアプライス中部(円/kWh)": "chubu_price",
    "エリアプライス北陸(円/kWh)": "hokuriku_price",
    "エリアプライス関西(円/kWh)": "kansai_price",
    "エリアプライス中国(円/kWh)": "chugoku_price",
    "エリアプライス四国(円/kWh)": "shikoku_price",
    "エリアプライス九州(円/kWh)": "kyushu_price",
    "売りブロック入札総量(kWh)": "sell_block_bid_volume_kwh",
    "売りブロック約定総量(kWh)": "sell_block_contracted_volume_kwh",
    "買いブロック入札総量(kWh)": "buy_block_bid_volume_kwh",
    "買いブロック約定総量(kWh)": "buy_block_contracted_volume_kwh",
}
REQUIRED_COLUMNS = set(WIDE_COLUMN_MAP)
AREA_DISPLAY = {
    "System": "시스템가격",
    "Hokkaido": "홋카이도",
    "Tohoku": "도호쿠",
    "Tokyo": "도쿄",
    "Chubu": "중부",
    "Hokuriku": "호쿠리쿠",
    "Kansai": "간사이",
    "Chugoku": "주고쿠",
    "Shikoku": "시코쿠",
    "Kyushu": "규슈",
}


def find_jepx_files(data_directory: str | Path) -> pd.DataFrame:
    """확인된 CSV 파일을 찾아 경로·크기·수정시각을 반환합니다."""
    directory = Path(data_directory)
    rows = []
    if not directory.exists():
        return pd.DataFrame(columns=["file_path", "file_size", "modified_at"])
    for path in sorted(directory.iterdir()):
        if (
            not path.is_file()
            or path.name.startswith((".", "~$"))
            or path.suffix.lower() not in SUPPORTED_SUFFIXES
            or path.stat().st_size == 0
        ):
            continue
        rows.append(
            {
                "file_path": path,
                "file_size": path.stat().st_size,
                "modified_at": pd.Timestamp(
                    path.stat().st_mtime, unit="s", tz="Asia/Tokyo"
                ),
            }
        )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_jepx_file(file_path: str | Path) -> pd.DataFrame:
    """원본 열을 보존하고 확인된 인코딩 순서로 JEPX CSV를 읽습니다."""
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"지원하지 않는 JEPX 파일 형식입니다: {path.suffix}")
    attempts = []
    for encoding in ENCODING_CANDIDATES:
        try:
            raw = pd.read_csv(path, encoding=encoding, sep=DELIMITER)
            if any("�" in str(column) for column in raw.columns):
                raise UnicodeError("열 이름에 대체문자가 있습니다.")
            missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
            if missing:
                raise ValueError("필수 열 누락: " + ", ".join(missing))
            raw.attrs.update(
                {
                    "encoding": encoding,
                    "delimiter": DELIMITER,
                    "header_row": 1,
                    "actual_columns": list(raw.columns),
                    "encoding_attempts": attempts + [f"{encoding}: 성공"],
                }
            )
            return raw
        except Exception as exc:
            attempts.append(f"{encoding}: 실패 ({exc})")
    raise ValueError("JEPX CSV 인코딩 판별 실패: " + "; ".join(attempts))


def _period_start(period: pd.Series) -> pd.Series:
    minutes = (period - 1) * 30
    valid = period.between(1, 48)
    hours = (minutes // 60).where(valid)
    minute_part = (minutes % 60).where(valid)
    return pd.Series(
        [
            f"{int(hour):02d}:{int(minute):02d}"
            if pd.notna(hour) and pd.notna(minute)
            else None
            for hour, minute in zip(hours, minute_part)
        ],
        index=period.index,
        dtype="object",
    )


def normalize_jepx_data(
    raw_data: pd.DataFrame, source_file: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """확인된 19개 원본 열을 wide 및 가격 long 데이터로 정규화합니다."""
    missing = sorted(REQUIRED_COLUMNS - set(raw_data.columns))
    if missing:
        raise ValueError("JEPX 정규화 필수 열 누락: " + ", ".join(missing))
    path = Path(source_file)
    source_rows = pd.Series(raw_data.index + 2, index=raw_data.index, dtype="Int64")
    wide = raw_data.rename(columns=WIDE_COLUMN_MAP).copy()
    raw_date = raw_data[DATE_COLUMN].copy()
    raw_period = raw_data[PERIOD_COLUMN].copy()
    wide["delivery_date"] = pd.to_datetime(raw_date, errors="coerce").dt.normalize()
    wide["period_no"] = pd.to_numeric(raw_period, errors="coerce").astype("Int64")
    wide["period_start"] = _period_start(wide["period_no"])
    offset = pd.to_timedelta((wide["period_no"] - 1) * 30, unit="m")
    wide["datetime_jst"] = (wide["delivery_date"] + offset).dt.tz_localize(
        "Asia/Tokyo", nonexistent="NaT", ambiguous="NaT"
    )
    for original, area in PRICE_COLUMNS.items():
        wide[WIDE_COLUMN_MAP[original]] = pd.to_numeric(
            raw_data[original], errors="coerce"
        )
    wide["price_unit"] = PRICE_UNIT
    wide["source_file"] = path.name
    wide["source_row"] = source_rows
    row_error = wide["delivery_date"].isna() | ~wide["period_no"].between(1, 48)
    wide["source_status"] = np.where(row_error, "Error", "Valid")
    wide.attrs.update(raw_data.attrs)

    id_columns = [
        "delivery_date",
        "period_no",
        "period_start",
        "datetime_jst",
        "source_file",
        "source_row",
        "source_status",
    ]
    price_wide_columns = [WIDE_COLUMN_MAP[column] for column in PRICE_COLUMNS]
    long = wide[id_columns + price_wide_columns].melt(
        id_vars=id_columns,
        value_vars=price_wide_columns,
        var_name="standard_price_column",
        value_name="price",
    )
    standard_to_original = {
        WIDE_COLUMN_MAP[original]: original for original in PRICE_COLUMNS
    }
    standard_to_area = {
        WIDE_COLUMN_MAP[original]: area for original, area in PRICE_COLUMNS.items()
    }
    long["area"] = long["standard_price_column"].map(standard_to_area)
    long["area_display"] = long["area"].map(AREA_DISPLAY)
    long["original_price_column"] = long["standard_price_column"].map(
        standard_to_original
    )
    raw_prices = raw_data[list(PRICE_COLUMNS)].rename(
        columns={column: WIDE_COLUMN_MAP[column] for column in PRICE_COLUMNS}
    )
    raw_long = raw_prices.assign(source_row=source_rows).melt(
        id_vars="source_row",
        var_name="standard_price_column",
        value_name="raw_price",
    )
    long = long.merge(
        raw_long,
        on=["source_row", "standard_price_column"],
        how="left",
        validate="one_to_one",
    )
    long["price_unit"] = PRICE_UNIT
    invalid_price = long["price"].isna() & long["raw_price"].notna()
    long.loc[invalid_price, "source_status"] = "Error"
    long.attrs.update(raw_data.attrs)
    return wide, long


def _issue(
    severity: str,
    code: str,
    message: str,
    row: pd.Series | None = None,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "issue_code": code,
        "message": message,
        "source_file": row.get("source_file") if row is not None else None,
        "source_row": row.get("source_row") if row is not None else None,
        "delivery_date": row.get("delivery_date") if row is not None else None,
        "period_no": row.get("period_no") if row is not None else None,
        "area": row.get("area") if row is not None else None,
        "price": row.get("price") if row is not None else None,
    }


def validate_jepx_data(
    normalized_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """long 데이터의 치명적 오류와 삭제하지 않을 검토 경고를 분리합니다."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if normalized_data.empty:
        errors.append(_issue("Fatal", "empty_data", "정규화 데이터가 없습니다."))
        return pd.DataFrame(errors), pd.DataFrame(warnings)

    for _, row in normalized_data.loc[normalized_data["delivery_date"].isna()].iterrows():
        errors.append(_issue("Fatal", "invalid_date", "날짜를 변환할 수 없습니다.", row))
    invalid_period = ~normalized_data["period_no"].between(1, 48)
    for _, row in normalized_data.loc[invalid_period].iterrows():
        errors.append(_issue("Fatal", "invalid_period", "시간대 번호가 1~48 범위가 아닙니다.", row))
    invalid_numeric = normalized_data["price"].isna() & normalized_data["raw_price"].notna()
    for _, row in normalized_data.loc[invalid_numeric].iterrows():
        errors.append(_issue("Fatal", "invalid_price", "가격을 숫자로 변환할 수 없습니다.", row))
    missing_price = normalized_data["price"].isna() & normalized_data["raw_price"].isna()
    for _, row in normalized_data.loc[missing_price].iterrows():
        warnings.append(_issue("Review", "missing_price", "가격이 비어 있습니다.", row))
    for _, row in normalized_data.loc[normalized_data["price"].lt(0)].iterrows():
        warnings.append(_issue("Review", "negative_price", "음수 가격이 확인되었습니다.", row))
    non_finite = normalized_data["price"].notna() & ~np.isfinite(
        normalized_data["price"].astype(float)
    )
    for _, row in normalized_data.loc[non_finite].iterrows():
        warnings.append(
            _issue("Review", "non_finite_price", "유한하지 않은 가격이 확인되었습니다.", row)
        )

    keys = ["delivery_date", "period_no", "area"]
    duplicates = normalized_data.duplicated(keys, keep=False)
    for _, row in normalized_data.loc[duplicates].iterrows():
        warnings.append(_issue("Review", "duplicate_key", "동일 날짜·시간대·지역 키가 중복됩니다.", row))
    if duplicates.any():
        conflicts = (
            normalized_data.loc[duplicates]
            .groupby(keys, dropna=False)["price"]
            .nunique(dropna=False)
        )
        for key in conflicts[conflicts.gt(1)].index:
            warnings.append(
                _issue(
                    "Review",
                    "conflicting_price",
                    f"같은 키에 서로 다른 가격이 있습니다: {key}",
                )
            )

    period_counts = normalized_data.groupby(
        ["delivery_date", "area"], dropna=False
    )["period_no"].nunique()
    for key, count in period_counts[period_counts.ne(48)].items():
        warnings.append(
            _issue(
                "Review",
                "incomplete_day",
                f"{key}의 시간대가 {int(count)}/48개입니다.",
            )
        )
    expected = set(PRICE_COLUMNS.values())
    present = set(normalized_data["area"].dropna())
    missing_areas = sorted(expected - present)
    if missing_areas:
        warnings.append(
            _issue(
                "Review",
                "missing_area",
                "확인된 원본 구조의 가격 열 누락: " + ", ".join(missing_areas),
            )
        )
    if "System" not in present:
        warnings.append(_issue("Review", "missing_system_price", "시스템가격이 없습니다."))
    date_ranges = normalized_data.groupby("area")["delivery_date"].agg(["min", "max"])
    if len(date_ranges.drop_duplicates()) > 1:
        warnings.append(
            _issue("Review", "area_date_mismatch", "지역별 데이터 기간이 서로 다릅니다.")
        )
    return pd.DataFrame(errors), pd.DataFrame(warnings)


def load_all_jepx_data(
    data_directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """모든 확인된 CSV를 로드하고 long, wide, 오류, 경고, 파일요약을 반환합니다."""
    long_frames = []
    wide_frames = []
    error_frames = []
    warning_frames = []
    summaries = []
    seen_hashes: dict[str, str] = {}
    files = find_jepx_files(data_directory)
    for _, file_info in files.iterrows():
        path = Path(file_info["file_path"])
        digest = _sha256(path)
        summary: dict[str, Any] = {
            "file_name": path.name,
            # 서버나 로컬 사용자의 절대경로를 진단 화면에 노출하지 않습니다.
            "file_path": path.name,
            "file_size": int(file_info["file_size"]),
            "modified_at": file_info["modified_at"],
            "sha256": digest,
            "encoding": None,
            "delimiter": None,
            "actual_columns": None,
            "raw_row_count": 0,
            "normalized_row_count": 0,
            "error_row_count": 0,
            "warning_row_count": 0,
            "duplicate_key_count": 0,
            "date_min": pd.NaT,
            "date_max": pd.NaT,
            "areas": "",
            "area_count": 0,
            "price_unit": PRICE_UNIT,
            "status": "실패",
            "error_message": "",
        }
        if digest in seen_hashes:
            summary["status"] = "중복 파일 제외"
            summary["error_message"] = f"동일 SHA-256 파일: {seen_hashes[digest]}"
            summaries.append(summary)
            continue
        seen_hashes[digest] = path.name
        try:
            raw = read_jepx_file(path)
            wide, long = normalize_jepx_data(raw, path)
            errors, warnings = validate_jepx_data(long)
            long_frames.append(long)
            wide_frames.append(wide)
            if not errors.empty:
                error_frames.append(errors)
            if not warnings.empty:
                warning_frames.append(warnings)
            summary.update(
                {
                    "encoding": raw.attrs.get("encoding"),
                    "delimiter": raw.attrs.get("delimiter"),
                    "actual_columns": " | ".join(raw.columns),
                    "raw_row_count": len(raw),
                    "normalized_row_count": len(long),
                    "error_row_count": len(errors),
                    "warning_row_count": len(warnings),
                    "duplicate_key_count": int(
                        long.duplicated(["delivery_date", "period_no", "area"], keep=False).sum()
                    ),
                    "date_min": long["delivery_date"].min(),
                    "date_max": long["delivery_date"].max(),
                    "areas": ", ".join(sorted(long["area"].dropna().unique())),
                    "area_count": long["area"].nunique(),
                    "status": "정상" if errors.empty else "오류 포함",
                }
            )
        except Exception as exc:
            summary["error_message"] = str(exc)
        summaries.append(summary)

    long_all = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    wide_all = pd.concat(wide_frames, ignore_index=True) if wide_frames else pd.DataFrame()
    errors_all = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    warnings_all = pd.concat(warning_frames, ignore_index=True) if warning_frames else pd.DataFrame()
    if not long_all.empty:
        duplicate_mask = long_all.duplicated(
            ["delivery_date", "period_no", "area"], keep=False
        )
        existing_duplicate_rows = set(
            zip(
                warnings_all.get("source_file", pd.Series(dtype=object)),
                warnings_all.get("source_row", pd.Series(dtype=object)),
                warnings_all.get("area", pd.Series(dtype=object)),
            )
        )
        cross_warnings = []
        for _, row in long_all.loc[duplicate_mask].iterrows():
            identity = (row["source_file"], row["source_row"], row["area"])
            if identity not in existing_duplicate_rows:
                cross_warnings.append(
                    _issue("Review", "duplicate_key", "여러 파일에 같은 키가 있습니다.", row)
                )
        if cross_warnings:
            warnings_all = pd.concat(
                [warnings_all, pd.DataFrame(cross_warnings)], ignore_index=True
            )
        conflicts = (
            long_all.loc[duplicate_mask]
            .groupby(["delivery_date", "period_no", "area"], dropna=False)["price"]
            .nunique(dropna=False)
        )
        conflict_warnings = [
            _issue(
                "Review",
                "conflicting_price",
                f"여러 파일의 같은 키에 서로 다른 가격이 있습니다: {key}",
            )
            for key in conflicts[conflicts.gt(1)].index
        ]
        if conflict_warnings:
            warnings_all = pd.concat(
                [warnings_all, pd.DataFrame(conflict_warnings)], ignore_index=True
            )
    summary_frame = pd.DataFrame(summaries)
    if not summary_frame.empty:
        for index, row in summary_frame.iterrows():
            if row["status"] == "중복 파일 제외":
                continue
            if not errors_all.empty:
                summary_frame.loc[index, "error_row_count"] = int(
                    errors_all["source_file"].eq(row["file_name"]).sum()
                )
            if not warnings_all.empty:
                summary_frame.loc[index, "warning_row_count"] = int(
                    warnings_all["source_file"].eq(row["file_name"]).sum()
                )
    return long_all, wide_all, errors_all, warnings_all, summary_frame
