from pathlib import Path

import pandas as pd

from utils.eprx_loader import METRIC_MAP
from utils.national_charts import national_price_chart, national_volume_chart
from utils.regional_charts import area_price_chart, area_volume_chart


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area": ["Tokyo", "Tokyo"],
            "period_start": ["00:00", "00:30"],
            "procurement_volume": [100.0, 110.0],
            "bid_volume": [120.0, 130.0],
            "awarded_volume": [90.0, 95.0],
            "avg_price": [4.0, 5.0],
            "area_count": [9, 9],
        }
    )


def test_original_eprx_metric_mapping_is_unchanged():
    assert METRIC_MAP["募集量（TSO別）[MW]"] == "procurement_volume"
    assert METRIC_MAP["応札量合計（電源属地別）[MW]"] == "bid_volume"
    assert METRIC_MAP["落札量合計（電源属地別）[MW]"] == "awarded_volume"
    assert METRIC_MAP["平均落札価格（電源属地別）[円/kW・30分]"] == "avg_price"
    assert METRIC_MAP["最高落札価格（電源属地別）[円/kW・30分]"] == "max_price"
    assert METRIC_MAP["最低落札価格（電源属地別）[円/kW・30分]"] == "min_price"


def test_regional_chart_labels_show_basis_without_changing_values():
    profile = _profile()
    price = area_price_chart(profile, ["Tokyo"], "엔/ΔkW·30분")
    volume = area_volume_chart(profile, "Tokyo")

    assert "전원 소재지별" in price.layout.title.text
    assert list(price.data[0].y) == [4.0, 5.0]
    assert {trace.name for trace in volume.data} == {
        "모집량 (TSO별)",
        "입찰량 (전원 소재지별)",
        "낙찰량 (전원 소재지별)",
    }


def test_national_chart_labels_show_basis_without_changing_values():
    profile = _profile().drop(columns="area")
    price = national_price_chart(profile, "엔/ΔkW·30분")
    volume = national_volume_chart(profile)

    assert "전원 소재지별" in price.layout.title.text
    assert list(price.data[0].y) == [4.0, 5.0]
    assert {trace.name for trace in volume.data} == {
        "모집량 (TSO별)",
        "입찰량 (전원 소재지별)",
        "낙찰량 (전원 소재지별)",
    }


def test_eprx_tables_and_explanations_show_the_data_basis():
    source = APP_PATH.read_text(encoding="utf-8")
    for label in (
        "평균 모집량 (TSO별, MW)",
        "평균 입찰량 (전원 소재지별, MW)",
        "평균 낙찰량 (전원 소재지별, MW)",
        "평균 낙찰가격 (전원 소재지별, {price_unit})",
        "최고 낙찰가격 (전원 소재지별, {price_unit})",
        "최저 낙찰가격 (전원 소재지별, {price_unit})",
        "본 앱의 낙찰가격은 전원 소재지별 공표값을 사용합니다.",
    ):
        assert label in source
