from pathlib import Path

import numpy as np
import pandas as pd

from utils.jepx_loader import (
    DATE_COLUMN,
    PERIOD_COLUMN,
    PRICE_COLUMNS,
    REQUIRED_COLUMNS,
    find_jepx_files,
    load_all_jepx_data,
    normalize_jepx_data,
    read_jepx_file,
    validate_jepx_data,
)


def fixture_frame() -> pd.DataFrame:
    rows = []
    for period in (1, 2):
        row = {column: 1000 + period for column in REQUIRED_COLUMNS}
        row[DATE_COLUMN] = "2026/04/01"
        row[PERIOD_COLUMN] = period
        for index, column in enumerate(PRICE_COLUMNS):
            row[column] = 10.0 + index + period / 10
        rows.append(row)
    return pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))


def write_fixture(path: Path, frame: pd.DataFrame | None = None) -> None:
    (frame if frame is not None else fixture_frame()).to_csv(
        path, index=False, encoding="cp932"
    )


def test_find_and_read_cp932_csv(tmp_path):
    valid = tmp_path / "spot.csv"
    write_fixture(valid)
    (tmp_path / "empty.csv").write_bytes(b"")
    (tmp_path / ".hidden.csv").write_bytes(valid.read_bytes())
    (tmp_path / "unsupported.xlsx").write_bytes(b"not excel")

    files = find_jepx_files(tmp_path)
    assert files["file_path"].tolist() == [valid]
    assert files.iloc[0]["file_size"] == valid.stat().st_size
    raw = read_jepx_file(valid)
    assert raw.attrs["encoding"] == "cp932"
    assert raw.attrs["delimiter"] == ","
    assert list(raw.columns) == list(REQUIRED_COLUMNS)


def test_date_period_and_wide_to_long_normalization(tmp_path):
    path = tmp_path / "spot.csv"
    write_fixture(path)
    raw = read_jepx_file(path)
    wide, long = normalize_jepx_data(raw, path)
    assert len(wide) == 2
    assert len(long) == 20
    assert wide["period_no"].tolist() == [1, 2]
    assert wide["period_start"].tolist() == ["00:00", "00:30"]
    assert str(wide["delivery_date"].iloc[0].date()) == "2026-04-01"
    assert str(wide["datetime_jst"].dt.tz) == "Asia/Tokyo"
    assert set(long["area"]) == set(PRICE_COLUMNS.values())
    assert long["price_unit"].eq("円/kWh").all()
    assert long["original_price_column"].isin(PRICE_COLUMNS).all()
    assert long["source_row"].min() == 2


def test_invalid_missing_negative_and_duplicate_prices_are_preserved(tmp_path):
    frame = fixture_frame()
    price_columns = list(PRICE_COLUMNS)
    frame[price_columns[0]] = frame[price_columns[0]].astype(object)
    frame.loc[0, price_columns[0]] = "not-a-number"
    frame.loc[0, price_columns[1]] = np.nan
    frame.loc[0, price_columns[2]] = -1.5
    frame = pd.concat([frame, frame.iloc[[1]].assign(**{price_columns[3]: 99.0})])
    path = tmp_path / "review.csv"
    write_fixture(path, frame)
    raw = read_jepx_file(path)
    _, long = normalize_jepx_data(raw, path)
    errors, warnings = validate_jepx_data(long)

    assert "invalid_price" in set(errors["issue_code"])
    assert "missing_price" in set(warnings["issue_code"])
    assert "negative_price" in set(warnings["issue_code"])
    assert "duplicate_key" in set(warnings["issue_code"])
    assert "conflicting_price" in set(warnings["issue_code"])
    invalid = long[long["raw_price"].eq("not-a-number")].iloc[0]
    assert pd.isna(invalid["price"])
    assert invalid["source_status"] == "Error"
    assert (long["price"] == -1.5).any()


def test_identical_hash_is_loaded_only_once_and_summary_is_recorded(tmp_path):
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    write_fixture(first)
    second.write_bytes(first.read_bytes())
    long, wide, errors, warnings, summary = load_all_jepx_data(tmp_path)

    assert len(long) == 20
    assert len(wide) == 2
    assert errors.empty
    assert summary["sha256"].nunique() == 1
    assert set(summary["status"]) == {"정상", "중복 파일 제외"}
    loaded = summary[summary["status"].eq("정상")].iloc[0]
    assert loaded["raw_row_count"] == 2
    assert loaded["normalized_row_count"] == 20
    assert loaded["area_count"] == 10


def test_overlapping_files_keep_rows_and_report_conflicting_price(tmp_path):
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    base = fixture_frame()
    changed = fixture_frame()
    changed.loc[0, list(PRICE_COLUMNS)[0]] = 999.0
    write_fixture(first, base)
    write_fixture(second, changed)

    long, _, errors, warnings, summary = load_all_jepx_data(tmp_path)
    assert len(long) == 40
    assert errors.empty
    assert "duplicate_key" in set(warnings["issue_code"])
    assert "conflicting_price" in set(warnings["issue_code"])
    assert summary["warning_row_count"].sum() > 0


def test_actual_jepx_file_matches_confirmed_structure():
    long, wide, errors, warnings, summary = load_all_jepx_data("data/jepx/raw")
    assert len(wide) == 5424
    assert len(long) == 54240
    assert errors.empty
    assert warnings.empty
    assert summary.iloc[0]["encoding"] == "cp932"
    assert summary.iloc[0]["date_min"] == pd.Timestamp("2026-04-01")
    assert summary.iloc[0]["date_max"] == pd.Timestamp("2026-07-22")
    assert summary.iloc[0]["area_count"] == 10
