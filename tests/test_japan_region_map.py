import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utils.eprx_area_groups import AGGREGATE_REGION_GROUPS
from utils.eprx_areas import EPRX_AREA_DISPLAY
from utils.japan_region_map import REGION_GEOMETRY, region_map_html, selected_map_regions, render_japan_region_map


def svg_elements(html):
    svg = html[html.index('<svg '):html.index('</svg>') + 6]
    return ET.fromstring(svg)


@pytest.mark.parametrize("name,areas", AGGREGATE_REGION_GROUPS.items())
def test_group_selection_and_nine_korean_labels(name, areas):
    state = {"eprx_analysis_unit": "광역권·합산", "eprx_aggregate_group": name}
    original = state.copy()
    html = region_map_html(state)
    root = svg_elements(html)
    assert root.attrib['viewBox'] == '125 12 590 345'
    assert root.attrib['width'] == '590'
    assert root.attrib['height'] == '200'
    groups = root.findall('{http://www.w3.org/2000/svg}g')
    assert len(groups) == 9
    assert {g.attrib['data-region'] for g in groups if g.attrib['data-selected'] == 'true'} == set(areas)
    assert all(label in html for label in EPRX_AREA_DISPLAY.values())
    polygons = root.findall('.//{http://www.w3.org/2000/svg}polygon')
    assert len(polygons) == 9
    assert all(p.attrib['fill'].startswith('#') and p.attrib['stroke'].startswith('#') for p in polygons)
    assert 'height:200px' in html
    assert 'min-width:500px' not in html
    assert '선택:' not in html
    assert state == original
    assert '<script' not in html and 'src=' not in html


def test_individual_default_saved_selection_and_jepx_neutral():
    assert selected_map_regions({}) == (("Hokkaido",), "홋카이도")
    assert selected_map_regions({"regional_view_mode": "도쿄"}) == (("Tokyo",), "도쿄")
    assert selected_map_regions({"eprx_last_individual_area": "중부"}) == (("Chubu",), "중부")
    assert selected_map_regions({"market_selector": "JEPX 현물시장", "regional_view_mode": "도쿄"}) == ((), "")


def test_geographic_relative_positions():
    anchor = {area: geometry[1] for area, geometry in REGION_GEOMETRY.items()}
    assert anchor['Hokkaido'][1] < anchor['Tohoku'][1] < anchor['Tokyo'][1]
    assert anchor['Kyushu'][0] < anchor['Chugoku'][0] < anchor['Kansai'][0] < anchor['Tokyo'][0]
    assert anchor['Hokuriku'][1] < anchor['Chubu'][1]
    assert anchor['Shikoku'][1] > anchor['Chugoku'][1]


def test_map_is_between_description_and_market_selector():
    source = Path('app.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    assert tree
    assert source.index('st.caption("EPRX 조정력시장 및 JEPX 현물시장 분석")') < source.index('render_japan_region_map(header_map, st.session_state)')
    assert source.index('render_japan_region_map(header_map, st.session_state)') < source.index('with st.container(key="market_selector_container")')
    assert 'st.columns([3, 1.4]' in source
    assert '@media (max-width: 800px)' in source


def test_streamlit_map_updates_without_extra_rerun():
    # Allow cold Streamlit imports; execution counts still guard against extra reruns.
    app = AppTest.from_string('''
import streamlit as st
from utils.japan_region_map import render_japan_region_map
from utils.eprx_area_groups import AGGREGATE_REGION_GROUPS
st.session_state.map_test_runs = st.session_state.get("map_test_runs", 0) + 1
render_japan_region_map(st, st.session_state)
st.radio("분석 단위", ["개별 지역", "광역권·합산"], key="eprx_analysis_unit")
if st.session_state.eprx_analysis_unit == "개별 지역":
    st.radio("지역", ["홋카이도", "도쿄"], key="regional_view_mode")
else:
    st.radio("합산", list(AGGREGATE_REGION_GROUPS), key="eprx_aggregate_group")
''', default_timeout=10).run()
    assert not app.exception
    assert app.session_state.map_test_runs == 1
    app.radio(key='regional_view_mode').set_value('도쿄').run()
    assert app.session_state.map_test_runs == 2
    iframe = app.get('iframe')[0].proto
    assert not iframe.src
    html = iframe.srcdoc
    groups = svg_elements(html).findall('{http://www.w3.org/2000/svg}g')
    assert [g.attrib['data-region'] for g in groups if g.attrib['data-selected'] == 'true'] == ['Tokyo']
    app.radio(key='eprx_analysis_unit').set_value('광역권·합산').run()
    assert app.session_state.map_test_runs == 3
    app.radio(key='eprx_aggregate_group').set_value('중부+호쿠리쿠+간사이').run()
    assert app.session_state.map_test_runs == 4
    assert not app.exception
    html = app.get('iframe')[0].proto.srcdoc
    groups = svg_elements(html).findall('{http://www.w3.org/2000/svg}g')
    assert {g.attrib['data-region'] for g in groups if g.attrib['data-selected'] == 'true'} == {'Chubu', 'Hokuriku', 'Kansai'}


def test_renderer_uses_explicit_iframe_height_not_html_sanitizer():
    class Target:
        def iframe(self, document, *, height):
            self.document = document
            self.height = height

    target = Target()
    render_japan_region_map(target, {})
    assert target.height == 240
    assert target.document.startswith('<!doctype html>')
    assert target.document.endswith('</body></html>')
    assert '<svg ' in target.document and '</svg>' in target.document
    assert '선택:' not in target.document
    assert '분석 대상 권역의 대략적 위치를 표시한 안내도입니다.' not in target.document


def test_header_columns_and_full_width_market_selector_runtime():
    source = Path('app.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    sections = []
    for node in tree.body:
        if isinstance(node, ast.With):
            snippet = ast.get_source_segment(source, node)
            if 'key="main_header"' in snippet or 'key="market_selector_container"' in snippet:
                sections.append(snippet)
    script = 'import streamlit as st\nfrom utils.japan_region_map import render_japan_region_map\n'
    app = AppTest.from_string(script + '\n'.join(sections)).run()
    assert not app.exception
    columns = app.get('column')
    assert len(columns) == 2
    assert any(e.type == 'title' for e in columns[0])
    assert any(e.type == 'iframe' for e in columns[1])
    assert not any(e.type == 'button_group' for column in columns for e in column)
    assert app.button_group(key='market_selector').value == 'EPRX 조정력시장'
    app.button_group(key='market_selector').set_value('JEPX 현물시장').run()
    assert not app.exception
    assert app.button_group(key='market_selector').value == 'JEPX 현물시장'
