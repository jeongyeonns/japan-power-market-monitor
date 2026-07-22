from pathlib import Path

from streamlit.testing.v1 import AppTest


ANALYSIS_WEEK_HELP_TEXT = (
    "분석하려는 주차를 선택해 주세요. "
    "정확한 비교를 위해 ‘데이터 완전’으로 표시된 주차를 권장합니다."
)


def test_market_selector_eprx_default_and_jepx_weekly_tabs():
    app = AppTest.from_file("app.py", default_timeout=60).run()
    assert not app.exception
    assert app.segmented_control[0].key == "market_selector"
    assert app.segmented_control[0].value == "EPRX 조정력시장"
    assert [tab.label for tab in app.tabs] == [
        "도쿄·중부 상세분석",
        "전국 시장 요약",
    ]
    assert any(box.key == "eprx_data_source" for box in app.selectbox)
    assert any(box.key == "eprx_analysis_week" for box in app.selectbox)
    week_box = next(box for box in app.selectbox if box.key == "eprx_analysis_week")
    assert str(week_box.value.date()) == "2026-07-13"
    assert any("2026-07-20" in str(value) for value in week_box.options)
    assert sum(
        item.value == ANALYSIS_WEEK_HELP_TEXT for item in app.caption
    ) == 1
    assert not any(item.key == "frequency_zone_view_mode" for item in app.radio)
    expander_labels = [item.label for item in app.expander]
    assert "데이터 출처 및 원본 파일 정보" in expander_labels
    assert "EPRX 데이터 품질 및 파일 진단" in expander_labels
    assert expander_labels.index("데이터 출처 및 원본 파일 정보") < expander_labels.index(
        "EPRX 데이터 품질 및 파일 진단"
    )

    next(
        box for box in app.selectbox if box.key == "eprx_data_source"
    ).set_value("샘플 데이터").run()
    assert not app.exception
    assert len(app.tabs) == 2
    assert any(
        "현재 가상 샘플 데이터를 사용하고 있습니다" in warning.value
        for warning in app.warning
    )

    app.segmented_control[0].set_value("JEPX 현물시장").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "도쿄·중부 분석",
        "전국 주간 모니터링",
    ]
    visible_text = "\n".join(
        [element.value for element in app.header]
        + [element.value for element in app.subheader]
        + [element.value for element in app.info]
        + [element.value for element in app.success]
        + [element.value for element in app.caption]
        + [element.value for element in app.markdown]
    )
    assert "JEPX Day-Ahead 현물가격 및 ESS 스프레드 분석" in visible_text
    assert "JEPX 데이터 연결 상태" not in visible_text
    assert "실제 JEPX Day-Ahead 파일이 연결되어 있습니다" not in visible_text
    assert not any(
        "실제 JEPX Day-Ahead 파일" in item.value for item in app.success
    )
    assert not any(item.label == "JEPX 배포 파일 진단" for item in app.expander)
    assert any(box.key == "jepx_tokyo_chubu_week" for box in app.selectbox)
    assert any(box.key == "jepx_tokyo_chubu_duration" for box in app.selectbox)
    assert any(box.key == "jepx_tokyo_chubu_operation_mode" for box in app.selectbox)
    assert sum(
        item.value == ANALYSIS_WEEK_HELP_TEXT for item in app.caption
    ) == 2
    assert any(box.key == "jepx_week_start" for box in app.selectbox)
    assert any(box.key == "jepx_weekly_area" for box in app.selectbox)
    assert any(box.key == "jepx_weekly_duration" for box in app.selectbox)
    assert any(box.key == "jepx_weekly_mode" for box in app.selectbox)
    assert not any(box.key == "jepx_validation_date" for box in app.selectbox)
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
