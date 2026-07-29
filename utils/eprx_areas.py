"""EPRX 수급조정시장 분석 대상 지역의 공통 순서·명칭·색상."""

from __future__ import annotations

# 화면 순서는 북쪽에서 남쪽 순이며, 오키나와는 EPRX 분석 대상에 포함하지 않습니다.
EPRX_AREA_DEFINITIONS = (
    ("홋카이도", "北海道", "Hokkaido", "#17becf"),
    ("도호쿠", "東北", "Tohoku", "#ff7f0e"),
    ("도쿄", "東京", "Tokyo", "#1f77b4"),
    ("중부", "中部", "Chubu", "#d62728"),
    ("호쿠리쿠", "北陸", "Hokuriku", "#2ca02c"),
    ("간사이", "関西", "Kansai", "#9467bd"),
    ("주고쿠", "中国", "Chugoku", "#8c564b"),
    ("시코쿠", "四国", "Shikoku", "#e377c2"),
    ("규슈", "九州", "Kyushu", "#7f7f7f"),
)

EPRX_AREA_OPTIONS = {
    display: internal for display, _japanese, internal, _color in EPRX_AREA_DEFINITIONS
}
EPRX_DISPLAY_TO_JAPANESE = {
    display: japanese
    for display, japanese, _internal, _color in EPRX_AREA_DEFINITIONS
}
EPRX_JAPANESE_AREAS = {
    japanese: internal
    for _display, japanese, internal, _color in EPRX_AREA_DEFINITIONS
}
EPRX_AREA_DISPLAY = {
    internal: display
    for display, _japanese, internal, _color in EPRX_AREA_DEFINITIONS
}
EPRX_AREA_JAPANESE = {
    internal: japanese
    for _display, japanese, internal, _color in EPRX_AREA_DEFINITIONS
}
EPRX_AREA_COLORS = {
    display: color for display, _japanese, _internal, color in EPRX_AREA_DEFINITIONS
}
EPRX_ANALYSIS_AREAS = tuple(EPRX_AREA_DISPLAY)