from pathlib import Path

import pandas as pd

from utils.chuden_area_loader import load_chuden_area_data, join_eprx_chubu_with_chuden
from utils.eprx_driver_features import build_eprx_driver_features, build_eprx_driver_weekly_context


HEADER = (
    "単位[MW平均],,,供給力,,,,,,,,,,,,,,,,,,\n"
    "DATE,TIME,エリア需要,原子力,火力(LNG),火力(石炭),火力(石油),火力(その他),"
    "火力出力制御量,水力,地熱,バイオマス,バイオマス出力制御量,太陽光発電実績,"
    "太陽光出力制御量,風力発電実績,風力出力制御量,揚水,蓄電池,連系線,その他,合計\n"
)


def test_loads_confirmed_chuden_schema_and_negative_value(tmp_path):
    path = tmp_path / "eria_jukyu_202607_04.csv"
    row = "2026/07/01,00:00,13094,0,4674,3304,0,448,0,1954,2,486,0,0,0,8,0,-258,2,1820,138,13094\n"
    path.write_bytes((HEADER + row).encode("cp932"))
    data, diagnostics = load_chuden_area_data(path)
    assert diagnostics.iloc[0].status == "Loaded"
    assert diagnostics.iloc[0].contains_24_00 == False
    assert data.loc[0, "area"] == "Chubu"
    assert data.loc[0, "pumped_storage_mw"] == -258
    assert data.loc[0, "period_no"] == 1


def test_chubu_uses_common_join_features_and_context():
    times = pd.date_range("2026-07-13", periods=336, freq="30min")
    eprx = pd.DataFrame({"delivery_date": times.normalize(), "period_no": times.hour * 2 + times.minute // 30 + 1,
                         "period_start": times.strftime("%H:%M"), "area": "Chubu", "procurement_volume": 100.0})
    grid = pd.DataFrame({"delivery_date": times.normalize(), "period_no": times.hour * 2 + times.minute // 30 + 1,
                         "period_start": times.strftime("%H:%M"), "area": "Chubu", "area_demand_mw": 15000.0,
                         "solar_generation_mw": 1000.0, "wind_generation_mw": 50.0})
    merged, join_diagnostics = join_eprx_chubu_with_chuden(eprx, grid, "2026-07-13")
    features, _ = build_eprx_driver_features(merged)
    context = build_eprx_driver_weekly_context(features, "2026-07-13")
    assert join_diagnostics["region"] == "Chubu"
    assert join_diagnostics["all_rows_matched"] is True
    assert context["region"] == "Chubu"
    assert context["complete_week"] is True
    assert features.loc[0, "residual_demand_mw"] == 13950
