"""수동 배치한 중부전력 PG 30분 에리어 수급실적 CSV 로더."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from utils.tepco_area_loader import join_eprx_region_with_grid


DEFAULT_RAW_DIRECTORY = Path("data/external/chuden/raw")
ENCODING = "cp932"
UNIT_LABEL = "単位[MW平均]"
COLUMN_MAP = {
    "DATE": "delivery_date", "TIME": "period_start",
    "エリア需要": "area_demand_mw", "原子力": "nuclear_mw",
    "火力(LNG)": "thermal_lng_mw", "火力(石炭)": "thermal_coal_mw",
    "火力(石油)": "thermal_oil_mw", "火力(その他)": "thermal_other_mw",
    "火力出力制御量": "thermal_curtailment_mw", "水力": "hydro_mw",
    "地熱": "geothermal_mw", "バイオマス": "biomass_mw",
    "バイオマス出力制御量": "biomass_curtailment_mw",
    "太陽光発電実績": "solar_generation_mw",
    "太陽光出力制御量": "solar_curtailment_mw",
    "風力発電実績": "wind_generation_mw",
    "風力出力制御量": "wind_curtailment_mw", "揚水": "pumped_storage_mw",
    "蓄電池": "battery_mw", "連系線": "interconnector_mw",
    "その他": "other_mw", "合計": "supply_total_mw",
}
NUMERIC_COLUMNS = tuple(v for k, v in COLUMN_MAP.items() if k not in {"DATE", "TIME"})
KEYS = ["delivery_date", "period_no", "period_start"]


def _paths(paths: Any) -> list[Path]:
    if paths is None:
        return sorted(DEFAULT_RAW_DIRECTORY.glob("*.csv"))
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        return sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if isinstance(paths, Iterable):
        return [Path(path) for path in paths]
    raise TypeError("paths는 경로, 경로 목록 또는 None이어야 합니다.")


def _load(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    stat = path.stat()
    key = hashlib.sha256(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()
    diag = {"source_file": path.name, "file_size_bytes": stat.st_size,
            "modified_at": pd.Timestamp(stat.st_mtime, unit="s", tz="Asia/Tokyo"),
            "cache_key": key, "encoding": ENCODING, "header_row": 1,
            "interval_label_assumption": "30-minute interval start", "status": "Error"}
    try:
        meta = pd.read_csv(path, encoding=ENCODING, header=None, nrows=1, dtype=str)
        if meta.iloc[0, 0] != UNIT_LABEL:
            raise ValueError(f"첫 행 단위 표기가 {UNIT_LABEL!r}가 아닙니다.")
        raw = pd.read_csv(path, encoding=ENCODING, header=1, dtype=str, keep_default_na=False)
        raw.columns = [str(c).strip() for c in raw.columns]
        missing = sorted(set(COLUMN_MAP) - set(raw.columns))
        if missing:
            raise ValueError("확인된 22열 스키마 누락: " + ", ".join(missing))
        unmapped = [c for c in raw.columns if c not in COLUMN_MAP]
        data = raw.rename(columns=COLUMN_MAP).copy()
        for column in unmapped:
            data.rename(columns={column: f"unmapped__{column}"}, inplace=True)
        date_text = data["delivery_date"].str.strip()
        time_text = data["period_start"].str.strip()
        dt = pd.to_datetime(date_text + " " + time_text, errors="coerce")
        data["delivery_date"] = pd.to_datetime(date_text, errors="coerce").dt.normalize()
        data["interval_start"] = dt
        data["period_start"] = dt.dt.strftime("%H:%M")
        data["period_no"] = (dt.dt.hour * 2 + dt.dt.minute.div(30) + 1).astype("Int64")
        failures = {}
        negatives = {}
        for column in NUMERIC_COLUMNS:
            text = data[column].str.strip()
            numeric = pd.to_numeric(text, errors="coerce")
            count = int((text.ne("") & numeric.isna()).sum())
            if count: failures[column] = count
            count = int(numeric.lt(0).sum())
            if count: negatives[column] = count
            data[column] = numeric
        data["area"] = "Chubu"
        data["source_file"] = path.name
        data["source_file_size_bytes"] = stat.st_size
        data["source_modified_at"] = diag["modified_at"]
        data["source_cache_key"] = key
        duplicates = data.duplicated(KEYS, keep=False)
        data["duplicate_candidate"] = duplicates
        diag.update({"status": "Review" if failures or dt.isna().any() or duplicates.any() else "Loaded",
                     "row_count": len(data), "unmapped_columns": unmapped,
                     "numeric_conversion_failures": failures, "negative_value_counts": negatives,
                     "invalid_datetime_rows": int(dt.isna().sum()),
                     "duplicate_key_rows": int(duplicates.sum()), "contains_24_00": bool(time_text.eq("24:00").any())})
        return data, diag
    except Exception as exc:
        diag["message"] = str(exc)
        return pd.DataFrame(), diag


def load_chuden_area_data(paths: Any = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """수동 배치 중부 30분 실적 CSV와 파일별 진단을 반환한다."""
    frames, diagnostics = [], []
    for path in _paths(paths):
        if not path.is_file():
            diagnostics.append({"source_file": path.name, "status": "Error", "message": "파일이 없습니다."})
            continue
        frame, diagnostic = _load(path)
        diagnostics.append(diagnostic)
        if not frame.empty: frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not data.empty:
        data["duplicate_candidate"] = data.duplicated(KEYS, keep=False)
        data = data.sort_values(KEYS + ["source_file"], kind="stable").reset_index(drop=True)
    return data, pd.DataFrame(diagnostics)


def join_eprx_chubu_with_chuden(eprx_df: pd.DataFrame, chuden_df: pd.DataFrame,
                                week_start: Any | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """EPRX Chubu와 중부전력 PG 실적을 공통 키로 결합한다."""
    return join_eprx_region_with_grid(eprx_df, chuden_df, "Chubu", week_start)
