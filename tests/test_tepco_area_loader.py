from pathlib import Path

import pandas as pd
import pytest

from utils.tepco_area_loader import (
    load_tepco_area_data,
    join_eprx_tokyo_with_tepco,
)


HEADER = (
    "単位[MW平均],,,供給力\n"
    "DATE,TIME,エリア需要,原子力,火力(LNG),火力(石炭),火力(石油),"
    "火力(その他),水力,地熱,バイオマス,太陽光発電実績,太陽光出力制御量,"
    "風力発電実績,風力出力制御量,揚水,蓄電池,連系線,その他,合計\n"
)


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_bytes((HEADER + "\n".join(rows) + "\n").encode("cp932"))


def test_loads_confirmed_cp932_two_row_header_and_preserves_source(tmp_path):
    path = tmp_path / "eria_jukyu_202607_03.csv"
    _write_csv(
        path,
        [
            "2026/7/1,0:00,26436,1301,12152,6751,114,1281,1465,0,466,0,0,39,0,0,5,2731,131,26436",
            "2026/7/1,0:30,25400,1301,11757,6752,71,1280,1468,0,466,0,0,26,0,-3,2,2101,176,25400",
        ],
    )

    data, diagnostics = load_tepco_area_data(path)

    assert diagnostics.iloc[0]["status"] == "Loaded"
    assert list(data["period_no"]) == [1, 2]
    assert list(data["period_start"]) == ["00:00", "00:30"]
    assert data.loc[1, "pumped_storage_mw"] == -3
    assert data.loc[0, "source_file"] == path.name
    assert data.loc[0, "source_cache_key"] == diagnostics.iloc[0]["cache_key"]


def test_conversion_failure_stays_missing_and_is_diagnosed(tmp_path):
    path = tmp_path / "bad.csv"
    _write_csv(
        path,
        ["2026/7/1,0:00,未公表,1301,12152,6751,114,1281,1465,0,466,0,0,39,0,0,5,2731,131,26436"],
    )

    data, diagnostics = load_tepco_area_data(path)

    assert pd.isna(data.loc[0, "area_demand_mw"])
    assert diagnostics.iloc[0]["numeric_conversion_failures"] == {"area_demand_mw": 1}
    assert diagnostics.iloc[0]["status"] == "Review"


def test_unmapped_column_is_retained_and_reported(tmp_path):
    path = tmp_path / "extra.csv"
    text = HEADER.replace("その他,合計", "その他,確認外列,合計")
    row = "2026/7/1,0:00,26436,1301,12152,6751,114,1281,1465,0,466,0,0,39,0,0,5,2731,131,abc,26436\n"
    path.write_bytes((text + row).encode("cp932"))

    data, diagnostics = load_tepco_area_data(path)

    assert data.loc[0, "unmapped__確認外列"] == "abc"
    assert diagnostics.iloc[0]["unmapped_columns"] == ["確認外列"]


def test_outer_join_verifies_exact_date_and_half_hour_keys():
    eprx = pd.DataFrame(
        {
            "delivery_date": ["2026-07-01", "2026-07-01"],
            "period_no": [1, 2],
            "period_start": ["00:00", "00:30"],
            "area": ["Tokyo", "Tokyo"],
            "procurement_volume": [100.0, 110.0],
        }
    )
    tepco = pd.DataFrame(
        {
            "delivery_date": pd.to_datetime(["2026-07-01", "2026-07-01"]),
            "period_no": pd.Series([1, 3], dtype="Int64"),
            "period_start": ["00:00", "01:00"],
            "area_demand_mw": [26436, 25000],
        }
    )

    joined, diagnostics = join_eprx_tokyo_with_tepco(eprx, tepco)

    assert list(joined["join_status"].astype(str)) == ["both", "left_only", "right_only"]
    assert diagnostics == {
        "key_columns": ["delivery_date", "period_no", "period_start"],
        "eprx_rows": 2,
        "tepco_rows": 2,
        "matched_rows": 1,
        "eprx_only_rows": 1,
        "tepco_only_rows": 1,
        "all_rows_matched": False,
    }


def test_join_rejects_duplicate_keys():
    eprx = pd.DataFrame(
        {
            "delivery_date": ["2026-07-01", "2026-07-01"],
            "period_no": [1, 1],
            "period_start": ["00:00", "00:00"],
            "area": ["Tokyo", "Tokyo"],
            "procurement_volume": [100, 100],
        }
    )
    tepco = pd.DataFrame(
        {
            "delivery_date": ["2026-07-01"],
            "period_no": [1],
            "period_start": ["00:00"],
            "area_demand_mw": [26436],
        }
    )
    with pytest.raises(ValueError, match="many-to-many"):
        join_eprx_tokyo_with_tepco(eprx, tepco)
