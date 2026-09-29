"""실제 renderer를 Streamlit runtime에서 실행하는 소규모 합산 UI 검사."""

import ast
import json
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest
from utils.eprx_area_groups import AGGREGATE_REGION_GROUPS
from utils.eprx_areas import EPRX_AREA_DISPLAY
from utils.sample_data import generate_sample_data
from utils.weekly_aggregation import add_week_columns, create_selected_area_weekly_profile


def renderer_app(missing=False):
    source = Path("app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    # Use the production functions without loading CSVs or external services.
    nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    script = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    script += "\nEPRX_REGION_OPTIONS = EPRX_AREA_OPTIONS\n"
    script += "data = add_week_columns(generate_sample_data('2026-07-20'))\n"
    if missing:
        script += "data = data.loc[data.area.ne('Tohoku')]\n"
    script += "render_regional_analysis(st, data, data.week_start.max(), 7, '엔')\n"
    return AppTest.from_string(script, default_timeout=30).run()


def test_default_group_switch_and_restore_individual_values():
    app = renderer_app()
    assert not app.exception
    assert app.radio(key="eprx_analysis_unit").value == "개별 지역"
    assert len(app.radio(key="regional_view_mode").options) == 9
    app.radio(key="regional_view_mode").set_value("도쿄").run()
    original = [item.value for item in app.markdown]
    charts = [item.proto.spec for item in app.get("plotly_chart")]
    app.radio(key="eprx_analysis_unit").set_value("광역권·합산").run()
    assert len(app.radio(key="eprx_aggregate_group").options) == 3
    for group in app.radio(key="eprx_aggregate_group").options:
        app.radio(key="eprx_aggregate_group").set_value(group).run()
        assert not app.exception
        assert f"{group} 주간 핵심지표" in [item.value for item in app.subheader]
        assert len(app.get("plotly_chart")) == 2
    app.radio(key="eprx_analysis_unit").set_value("개별 지역").run()
    assert not app.exception
    assert app.radio(key="regional_view_mode").value == "도쿄"
    assert [item.value for item in app.markdown] == original
    assert [item.proto.spec for item in app.get("plotly_chart")] == charts


def test_missing_region_warning_and_unavailable_kpi_without_exception():
    app = renderer_app(missing=True)
    app.radio(key="eprx_analysis_unit").set_value("광역권·합산").run()
    app.radio(key="eprx_aggregate_group").set_value("도쿄+도호쿠").run()
    assert not app.exception
    assert any("일부 지역 데이터가 누락되어 있습니다: 도호쿠" in item.value for item in app.warning)
    tables = [item.value for item in app.markdown if '<table' in item.value]
    assert "계산 불가" in tables[0]
    assert any(item.label == "지역 합산 데이터 진단" for item in app.expander)


def test_constituent_lines_and_heading_caption_order():
    app = renderer_app()
    data = add_week_columns(generate_sample_data("2026-07-20"))
    app.radio(key="eprx_analysis_unit").set_value("광역권·합산").run()
    assert app.radio(key="eprx_aggregate_group").options == [
        "전국 합산", "도쿄+도호쿠", "중부+호쿠리쿠+간사이"
    ]
    for name, areas in AGGREGATE_REGION_GROUPS.items():
        app.radio(key="eprx_aggregate_group").set_value(name).run()
        assert not app.exception
        headings = [item.value for item in app.subheader]
        assert "지역별 시간대별 평균 최고 낙찰가격" in headings
        assert "지역별 시간대별 입찰 대비 낙찰률" in headings
        table = next(item.value for item in app.markdown if '<table' in item.value)
        assert "입찰 대비 낙찰률" not in table
        assert table.count('class="metric-change-rate"') == 3
        for chart, column in zip(app.get("plotly_chart"), ["max_price", "award_rate"]):
            figure = json.loads(chart.proto.spec)
            assert len(figure["data"]) == len(areas)
            assert {trace["name"] for trace in figure["data"]} == {EPRX_AREA_DISPLAY[a] for a in areas}
            assert figure["layout"]["legend"]["itemclick"] == "toggle"
            assert figure["layout"]["title"]["text"] == ""
            for area in areas:
                expected = create_selected_area_weekly_profile(data, data.week_start.max(), [area])
                trace = next(t for t in figure["data"] if t["name"] == EPRX_AREA_DISPLAY[area])
                y = trace["y"]
                if isinstance(y, dict):
                    import base64
                    y = np.frombuffer(base64.b64decode(y["bdata"]), dtype=y["dtype"])
                np.testing.assert_allclose(y, expected[column], equal_nan=True)
                assert "지역=%{fullData.name}" in trace["hovertemplate"]
        children = list(app.main.children.values())
        for index, element in enumerate(children):
            if element.type == "plotly_chart":
                assert children[index-2].type == "subheader"
                assert children[index-1].type == "caption"
        expander = next(e for e in app.expander if e.label == "데이터 확인 안내")
        assert not expander.proto.expanded
        assert children[-1] is expander
        assert all(e.type != "warning" or "확인이 필요한 데이터" not in e.value for e in children)
