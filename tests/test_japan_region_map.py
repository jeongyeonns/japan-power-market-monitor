import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from utils.eprx_area_groups import AGGREGATE_REGION_GROUPS
from utils.eprx_areas import EPRX_AREA_DISPLAY
from utils.japan_region_map import REGION_GEOMETRY, region_map_html, selected_map_regions


def svg_elements(html):
    svg = html[html.index('<svg '):html.index('</svg>') + 6]
    return ET.fromstring(svg)


@pytest.mark.parametrize("name,areas", AGGREGATE_REGION_GROUPS.items())
def test_group_selection_and_nine_korean_labels(name, areas):
    state = {"eprx_analysis_unit": "광역권·합산", "eprx_aggregate_group": name}
    original = state.copy()
    html = region_map_html(state)
    root = svg_elements(html)
    groups = root.findall('{http://www.w3.org/2000/svg}g')
    assert len(groups) == 9
    assert {g.attrib['data-region'] for g in groups if g.attrib['data-selected'] == 'true'} == set(areas)
    assert all(label in html for label in EPRX_AREA_DISPLAY.values())
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
    assert source.index('st.caption("EPRX 조정력시장 및 JEPX 현물시장 분석")') < source.index('render_japan_region_map(st, st.session_state)')
    assert source.index('render_japan_region_map(st, st.session_state)') < source.index('with st.container(key="market_selector_container")')


def test_streamlit_map_updates_without_extra_rerun():
    app = AppTest.from_string('''
import streamlit as st
from utils.japan_region_map import render_japan_region_map
from utils.eprx_area_groups import AGGREGATE_REGION_GROUPS
render_japan_region_map(st, st.session_state)
st.radio("분석 단위", ["개별 지역", "광역권·합산"], key="eprx_analysis_unit")
if st.session_state.eprx_analysis_unit == "개별 지역":
    st.radio("지역", ["홋카이도", "도쿄"], key="regional_view_mode")
else:
    st.radio("합산", list(AGGREGATE_REGION_GROUPS), key="eprx_aggregate_group")
''').run()
    assert not app.exception
    app.radio(key='regional_view_mode').set_value('도쿄').run()
    html = app.get('html')[0].proto.body
    groups = svg_elements(html).findall('{http://www.w3.org/2000/svg}g')
    assert [g.attrib['data-region'] for g in groups if g.attrib['data-selected'] == 'true'] == ['Tokyo']
    app.radio(key='eprx_analysis_unit').set_value('광역권·합산').run()
    app.radio(key='eprx_aggregate_group').set_value('중부+호쿠리쿠+간사이').run()
    assert not app.exception
    html = app.get('html')[0].proto.body
    groups = svg_elements(html).findall('{http://www.w3.org/2000/svg}g')
    assert {g.attrib['data-region'] for g in groups if g.attrib['data-selected'] == 'true'} == {'Chubu', 'Hokuriku', 'Kansai'}
