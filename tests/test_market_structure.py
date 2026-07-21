from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_market_selector_eprx_default_and_jepx_weekly_tabs():
    app = AppTest.from_file("app.py", default_timeout=60).run()
    assert not app.exception
    assert app.radio[0].key == "market_selector"
    assert app.radio[0].value == "EPRX 조정력시장"
    assert [tab.label for tab in app.tabs] == [
        "주파수권역 분석",
        "도쿄·중부 상세분석",
        "전국 시장 요약",
    ]
    assert any(box.key == "eprx_data_source" for box in app.selectbox)
    assert any(box.key == "eprx_analysis_week" for box in app.selectbox)

    next(
        box for box in app.selectbox if box.key == "eprx_data_source"
    ).set_value("샘플 데이터").run()
    assert not app.exception
    assert len(app.tabs) == 3

    app.radio[0].set_value("JEPX 현물시장").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "주간 모니터링",
        "계산 검증",
        "데이터 품질 및 파일 진단",
    ]
    visible_text = "\n".join(
        [element.value for element in app.header]
        + [element.value for element in app.subheader]
        + [element.value for element in app.info]
        + [element.value for element in app.success]
        + [element.value for element in app.markdown]
    )
    assert "JEPX Day-Ahead 현물가격 및 ESS 스프레드 분석" in visible_text
    assert "실제 JEPX Day-Ahead 파일이 연결되었습니다" in visible_text
    assert any(box.key == "jepx_week_start" for box in app.selectbox)
    assert any(box.key == "jepx_weekly_area" for box in app.selectbox)
    assert any(box.key == "jepx_weekly_duration" for box in app.selectbox)
    assert any(box.key == "jepx_weekly_mode" for box in app.selectbox)
    assert any(box.key == "jepx_validation_date" for box in app.selectbox)
    daily_areas = next(
        box for box in app.multiselect
        if box.key == "jepx_daily_spread_selected_areas"
    )
    assert daily_areas.value == ["Chubu", "Tokyo", "Hokkaido", "Kyushu"]
    daily_areas.set_value(["Chubu", "Tokyo", "Hokkaido", "Kyushu", "System"]).run()
    assert not app.exception
    daily_areas = next(
        box for box in app.multiselect
        if box.key == "jepx_daily_spread_selected_areas"
    )
    assert len(daily_areas.value) == 5
    daily_areas.set_value([]).run()
    assert not app.exception
    assert any("표시할 지역을 하나 이상 선택하세요" in item.value for item in app.info)
    assert any(box.key == "jepx_show_all_daily_spread_areas" for box in app.checkbox)
    assert not any(box.key == "eprx_data_source" for box in app.selectbox)


def test_jepx_directory_scaffold_exists_without_assumed_parser():
    for directory in (
        Path("data/jepx"),
        Path("data/jepx/raw"),
        Path("data/jepx/metadata"),
    ):
        assert directory.is_dir()
        assert (directory / ".gitkeep").exists()
    assert Path("utils/jepx_loader.py").exists()
