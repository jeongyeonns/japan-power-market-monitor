import numpy as np
import pandas as pd

from utils.summary_display import (
    build_national_summary_data,
    build_regional_summary_data,
    excess_award_warning_markdown,
    national_summary_markdown,
    regional_summary_markdown,
)


def _national_inputs(volume_change=150.37, price_change=-0.47):
    national = pd.DataFrame({"completeness_flag": ["Complete"]})
    regional = pd.DataFrame(
        {
            "area": ["Hokuriku", "Shikoku", "Tohoku"],
            "area_display": ["Hokuriku", "Shikoku", "Tohoku"],
            "frequency_zone": ["50Hz", "60Hz", "50Hz"],
            "bid_coverage_ratio": [0.9, 1.4, 1.1],
            "weighted_avg_price": [5.0, 4.0, 7.0],
            "avg_bid_volume": [90.0, 140.0, 110.0],
            "avg_procurement_volume": [100.0, 100.0, 100.0],
        }
    )
    kpis = {
        "입찰경쟁률 1.0배 미만 시간대 수": 3,
        "최대 미조달 발생 시간대": "16:00",
        "최대 미조달량": 219.85,
    }
    comparison = pd.DataFrame(
        {
            "지표": [
                "평균 모집량 (MW)",
                "전국 낙찰량 가중평균 낙찰가격",
            ],
            "절대 변화": [volume_change, price_change],
        }
    )
    return national, regional, kpis, comparison


def test_national_summary_uses_directions_korean_names_and_separate_lines():
    summary = build_national_summary_data(*_national_inputs())
    markdown = national_summary_markdown(summary)
    assert "150.37 MW 증가" in markdown
    assert "0.47 하락" in markdown
    assert "+150.37" not in markdown and "-0.47" not in markdown
    assert "**호쿠리쿠**" in markdown
    assert "**시코쿠**" in markdown
    assert "**도호쿠**" in markdown
    assert "Hokuriku" not in markdown
    assert len([line for line in markdown.splitlines() if line.startswith("- ")]) >= 7
    assert len([line for line in markdown.splitlines() if line.startswith("  - ")]) == 3


def test_zero_and_nan_changes_are_not_converted_to_numbers():
    summary = build_national_summary_data(*_national_inputs(0.0, np.nan))
    markdown = national_summary_markdown(summary)
    assert "0.00 MW 변화 없음" in markdown
    assert "계산 불가" in markdown
    assert "+0.00" not in markdown


def _regional_inputs():
    profile = pd.DataFrame(
        {
            "area": ["Tokyo", "Chubu"],
            "period_start": ["14:30", "14:30"],
            "avg_price": [5.0, 4.72],
            "procurement_volume": [100.0, 100.0],
            "bid_volume": [93.0, 179.0],
            "awarded_volume": [80.0, 100.0],
            "bid_coverage_ratio": [0.93, 1.79],
            "procurement_rate": [0.8, 1.0],
            "shortage_volume": [20.0, 0.0],
            "observation_count": [7, 7],
        }
    )
    kpis = pd.DataFrame(
        {
            "도쿄": [5.0, 0.93],
            "중부": [4.72, 1.79],
        },
        index=["평균 낙찰가격", "입찰경쟁률 (배)"],
    )
    previous = pd.DataFrame(
        {
            "지역": ["도쿄", "중부"],
            "지표": ["평균 낙찰가격", "평균 낙찰가격"],
            "절대 변화": [0.20, -1.49],
        }
    )
    return profile, kpis, previous


def test_regional_summary_has_nested_lines_and_absolute_changes():
    summary = build_regional_summary_data(*_regional_inputs())
    markdown = regional_summary_markdown(summary)
    assert "도쿄가 중부보다 **0.28 높음**" in markdown
    assert "0.20 상승" in markdown
    assert "1.49 하락" in markdown
    assert "+0.20" not in markdown and "-1.49" not in markdown
    assert "Tokyo" not in markdown and "Chubu" not in markdown
    assert "입찰경쟁률은 입찰량을 모집량으로 나눈 값" in markdown
    assert "도쿄: **0.93배** — 모집량보다 입찰량이 적음" in markdown
    assert "중부: **1.79배** — 모집량보다 입찰량이 많음" in markdown
    assert len([line for line in markdown.splitlines() if line.startswith("- ")]) == 4
    assert len([line for line in markdown.splitlines() if line.startswith("  - ")]) == 2
    assert "1.0배 미만 시간대" not in markdown
    assert "조달률 100% 미만" not in markdown
    assert "최대 미조달" not in markdown


def test_regional_single_area_summary_hides_other_area():
    inputs = _regional_inputs()
    tokyo = regional_summary_markdown(
        build_regional_summary_data(*inputs, visible_areas=["Tokyo"])
    )
    assert "### 도쿄 핵심 요약" in tokyo
    assert "도쿄: **0.93배**" in tokyo
    assert "중부" not in tokyo
    assert len([line for line in tokyo.splitlines() if line.startswith("- ")]) == 4

    chubu = regional_summary_markdown(
        build_regional_summary_data(*inputs, visible_areas=["Chubu"])
    )
    assert "### 중부 핵심 요약" in chubu
    assert "중부: **1.79배**" in chubu
    assert "도쿄" not in chubu
    assert len([line for line in chubu.splitlines() if line.startswith("- ")]) == 4


def test_competition_ratio_one_is_described_as_similar():
    profile, kpis, previous = _regional_inputs()
    kpis.loc["입찰경쟁률 (배)", "도쿄"] = 1.0
    markdown = regional_summary_markdown(
        build_regional_summary_data(
            profile, kpis, previous, visible_areas=["Tokyo"]
        )
    )
    assert "**1.00배** — 모집량과 입찰량이 비슷함" in markdown


def test_regional_nan_is_calculation_unavailable():
    profile, kpis, previous = _regional_inputs()
    kpis.loc["평균 낙찰가격", "도쿄"] = np.nan
    previous.loc[0, "절대 변화"] = np.nan
    markdown = regional_summary_markdown(
        build_regional_summary_data(profile, kpis, previous)
    )
    assert "계산 불가" in markdown
    assert "nan" not in markdown.lower()


def test_excess_award_warning_is_vertical_and_has_clear_units():
    markdown = excess_award_warning_markdown(
        {"50Hz": 1, "60Hz": 39}, 1494
    )
    assert markdown.splitlines()[0] == "**확인이 필요한 데이터가 있습니다.**"
    assert "일부 시간대에서 낙찰량이 모집량보다 크게 표시되었습니다." in markdown
    assert "- 50Hz: **1개 시간대**" in markdown
    assert "- 60Hz: **39개 시간대**" in markdown
    assert "관련 원본 데이터 행: **1,494건**" in markdown
    assert "원본 EPRX 값을 수정하지 않고 그대로 표시" in markdown
    assert "송전사업자" not in markdown
    assert "전원 소재" not in markdown
    assert "귀속 기준" not in markdown
    assert "집계기준 확인 필요" not in markdown
