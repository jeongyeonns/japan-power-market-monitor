"""수동 배치한 도쿄전력 PG 에리어 수급실적 CSV 로더.

네트워크 접근이나 자동 다운로드를 수행하지 않으며 원본 파일도 변경하지 않는다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RAW_DIRECTORY = Path("data/external/tepco/raw")
ENCODING = "cp932"
UNIT_LABEL = "単位[MW平均]"
EXPECTED_HEADER_ROW = 1

COLUMN_MAP = {
    "DATE": "delivery_date",
    "TIME": "period_start",
    "エリア需要": "area_demand_mw",
    "原子力": "nuclear_mw",
    "火力(LNG)": "thermal_lng_mw",
    "火力(石炭)": "thermal_coal_mw",
    "火力(石油)": "thermal_oil_mw",
    "火力(その他)": "thermal_other_mw",
    "水力": "hydro_mw",
    "地熱": "geothermal_mw",
    "バイオマス": "biomass_mw",
    "太陽光発電実績": "solar_generation_mw",
    "太陽光出力制御量": "solar_curtailment_mw",
    "風力発電実績": "wind_generation_mw",
    "風力出力制御量": "wind_curtailment_mw",
    "揚水": "pumped_storage_mw",
    "蓄電池": "battery_mw",
    "連系線": "interconnector_mw",
    "その他": "other_mw",
    "合計": "supply_total_mw",
}
KEY_COLUMNS = ("delivery_date", "period_no", "period_start")
NUMERIC_COLUMNS = tuple(
    normalized
    for original, normalized in COLUMN_MAP.items()
    if original not in {"DATE", "TIME"}
)


def _resolve_paths(paths: Any) -> list[Path]:
    if paths is None:
        return sorted(DEFAULT_RAW_DIRECTORY.glob("*.csv"))
    if isinstance(paths, (str, Path)):
        candidate = Path(paths)
        return sorted(candidate.glob("*.csv")) if candidate.is_dir() else [candidate]
    if isinstance(paths, Iterable):
        return [Path(path) for path in paths]
    raise TypeError("paths는 경로, 경로 목록 또는 None이어야 합니다.")


def _cache_key(path: Path) -> str:
    stat = path.stat()
    signature = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def _read_one(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    stat = path.stat()
    diagnostic: dict[str, Any] = {
        "source_file": path.name,
        "source_path": str(path.resolve()),
        "file_size_bytes": stat.st_size,
        "modified_at": pd.Timestamp(stat.st_mtime, unit="s", tz="Asia/Tokyo"),
        "cache_key": _cache_key(path),
        "encoding": ENCODING,
        "header_row": EXPECTED_HEADER_ROW,
        "status": "Error",
        "row_count": 0,
        "unmapped_columns": [],
        "numeric_conversion_failures": {},
        "negative_value_counts": {},
        "invalid_datetime_rows": 0,
        "duplicate_key_rows": 0,
        "message": "",
    }
    try:
        metadata = pd.read_csv(
            path, encoding=ENCODING, header=None, nrows=1, dtype=str
        )
        if metadata.empty or metadata.iloc[0, 0] != UNIT_LABEL:
            raise ValueError(f"첫 행 단위 표기가 {UNIT_LABEL!r}가 아닙니다.")
        original = pd.read_csv(
            path,
            encoding=ENCODING,
            header=EXPECTED_HEADER_ROW,
            dtype=str,
            keep_default_na=False,
        )
        original.columns = [str(column).strip() for column in original.columns]
        required = {"DATE", "TIME"}
        missing = sorted(required - set(original.columns))
        if missing:
            raise ValueError(f"필수 열 누락: {', '.join(missing)}")

        unmapped = [column for column in original.columns if column not in COLUMN_MAP]
        diagnostic["unmapped_columns"] = unmapped
        renamed = original.rename(columns=COLUMN_MAP).copy()
        for column in unmapped:
            renamed.rename(columns={column: f"unmapped__{column}"}, inplace=True)

        raw_date = renamed["delivery_date"].str.strip()
        raw_time = renamed["period_start"].str.strip()
        renamed["delivery_date"] = pd.to_datetime(raw_date, errors="coerce")
        interval_start = pd.to_datetime(
            raw_date + " " + raw_time, errors="coerce"
        )
        renamed["interval_start"] = interval_start
        renamed["period_start"] = interval_start.dt.strftime("%H:%M")
        renamed["period_no"] = (
            interval_start.dt.hour * 2 + interval_start.dt.minute.div(30) + 1
        ).astype("Int64")
        invalid_time_grid = interval_start.notna() & ~interval_start.dt.minute.isin([0, 30])
        invalid_datetime = interval_start.isna() | invalid_time_grid
        diagnostic["invalid_datetime_rows"] = int(invalid_datetime.sum())

        failures: dict[str, int] = {}
        negatives: dict[str, int] = {}
        for column in NUMERIC_COLUMNS:
            if column not in renamed.columns:
                continue
            raw_values = renamed[column].str.strip()
            converted = pd.to_numeric(raw_values, errors="coerce")
            failures[column] = int((raw_values.ne("") & converted.isna()).sum())
            negatives[column] = int(converted.lt(0).sum())
            renamed[column] = converted
        diagnostic["numeric_conversion_failures"] = {
            key: value for key, value in failures.items() if value
        }
        diagnostic["negative_value_counts"] = {
            key: value for key, value in negatives.items() if value
        }

        renamed["area"] = "Tokyo"
        renamed["source_file"] = path.name
        renamed["source_file_size_bytes"] = stat.st_size
        renamed["source_modified_at"] = diagnostic["modified_at"]
        renamed["source_cache_key"] = diagnostic["cache_key"]
        duplicate_mask = renamed.duplicated(list(KEY_COLUMNS), keep=False)
        renamed["duplicate_candidate"] = duplicate_mask
        diagnostic["duplicate_key_rows"] = int(duplicate_mask.sum())
        diagnostic["row_count"] = len(renamed)
        diagnostic["status"] = (
            "Review"
            if diagnostic["invalid_datetime_rows"]
            or diagnostic["numeric_conversion_failures"]
            or diagnostic["duplicate_key_rows"]
            else "Loaded"
        )
        return renamed, diagnostic
    except Exception as exc:
        diagnostic["message"] = str(exc)
        return pd.DataFrame(), diagnostic


def load_tepco_area_data(paths: Any = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """수동 배치 CSV를 정규화하고 파일별 진단 결과를 반환한다."""
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for path in _resolve_paths(paths):
        if not path.is_file():
            diagnostics.append(
                {
                    "source_file": path.name,
                    "source_path": str(path),
                    "status": "Error",
                    "message": "파일이 없습니다.",
                }
            )
            continue
        frame, diagnostic = _read_one(path)
        diagnostics.append(diagnostic)
        if not frame.empty:
            frames.append(frame)

    normalized = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not normalized.empty:
        duplicate_mask = normalized.duplicated(list(KEY_COLUMNS), keep=False)
        normalized["duplicate_candidate"] = duplicate_mask
        normalized = normalized.sort_values(
            ["delivery_date", "period_no", "source_file"], kind="stable"
        ).reset_index(drop=True)
    return normalized, pd.DataFrame(diagnostics)


def join_eprx_region_with_grid(
    eprx_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    region: str,
    week_start: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """EPRX와 지역 계통실적을 날짜·30분 시작 블록으로 outer 결합한다."""
    eprx_required = {"delivery_date", "period_no", "period_start", "area", "procurement_volume"}
    tepco_required = set(KEY_COLUMNS) | {"area_demand_mw"}
    missing_eprx = sorted(eprx_required - set(eprx_df.columns))
    missing_tepco = sorted(tepco_required - set(grid_df.columns))
    if missing_eprx or missing_tepco:
        details = []
        if missing_eprx:
            details.append("EPRX: " + ", ".join(missing_eprx))
        if missing_tepco:
            details.append("TEPCO: " + ", ".join(missing_tepco))
        raise ValueError("결합 필수 열 누락 - " + "; ".join(details))

    eprx = eprx_df.loc[eprx_df["area"].eq(region)].copy()
    tepco = grid_df.copy()
    if "area" in tepco.columns:
        tepco = tepco.loc[tepco["area"].eq(region)].copy()
    eprx["delivery_date"] = pd.to_datetime(eprx["delivery_date"]).dt.normalize()
    tepco["delivery_date"] = pd.to_datetime(tepco["delivery_date"]).dt.normalize()
    if week_start is not None:
        start = pd.Timestamp(week_start).normalize()
        end = start + pd.Timedelta(days=7)
        eprx = eprx.loc[eprx["delivery_date"].between(start, end, inclusive="left")]
        tepco = tepco.loc[tepco["delivery_date"].between(start, end, inclusive="left")]

    # EPRX의 3시간제 자료는 TEPCO 30분 실적과 동일 블록으로 간주하지 않는다.
    eprx = eprx.loc[eprx["delivery_date"].ge("2026-03-14")].copy()
    eprx["period_no"] = pd.to_numeric(eprx["period_no"], errors="coerce").astype("Int64")
    tepco["period_no"] = pd.to_numeric(tepco["period_no"], errors="coerce").astype("Int64")

    eprx_duplicates = int(eprx.duplicated(list(KEY_COLUMNS), keep=False).sum())
    tepco_duplicates = int(tepco.duplicated(list(KEY_COLUMNS), keep=False).sum())
    if eprx_duplicates or tepco_duplicates:
        raise ValueError(
            "중복 결합 키가 있어 many-to-many 결합을 중단합니다: "
            f"EPRX {eprx_duplicates}행, TEPCO {tepco_duplicates}행"
        )

    joined = eprx.merge(
        tepco,
        on=list(KEY_COLUMNS),
        how="outer",
        suffixes=("_eprx", "_tepco"),
        indicator="join_status",
        validate="one_to_one",
    ).sort_values(["delivery_date", "period_no"], kind="stable").reset_index(drop=True)
    counts = joined["join_status"].value_counts().to_dict()
    diagnostics = {
        "region": region,
        "key_columns": list(KEY_COLUMNS),
        "eprx_rows": len(eprx),
        "tepco_rows": len(tepco),
        "matched_rows": int(counts.get("both", 0)),
        "eprx_only_rows": int(counts.get("left_only", 0)),
        "tepco_only_rows": int(counts.get("right_only", 0)),
        "all_rows_matched": bool(len(joined)) and counts.get("left_only", 0) == 0 and counts.get("right_only", 0) == 0,
    }
    return joined, diagnostics


def join_eprx_tokyo_with_tepco(
    eprx_df: pd.DataFrame,
    tepco_df: pd.DataFrame,
    week_start: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """기존 Tokyo 공개 API를 유지하는 공통 결합 함수 래퍼."""
    joined, diagnostics = join_eprx_region_with_grid(
        eprx_df, tepco_df, "Tokyo", week_start
    )
    diagnostics.pop("region")
    return joined, diagnostics
