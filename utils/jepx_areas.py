"""JEPX 상세 분석 지역과 전국 모니터링 표시 순서의 공통 설정."""

from __future__ import annotations

JEPX_DETAIL_AREA_DEFINITIONS = (
    ("도쿄", "Tokyo"),
    ("중부", "Chubu"),
    ("홋카이도", "Hokkaido"),
    ("도호쿠", "Tohoku"),
    ("호쿠리쿠", "Hokuriku"),
    ("간사이", "Kansai"),
    ("주고쿠", "Chugoku"),
    ("시코쿠", "Shikoku"),
    ("규슈", "Kyushu"),
)

JEPX_DETAIL_AREA_OPTIONS = dict(JEPX_DETAIL_AREA_DEFINITIONS)
JEPX_DETAIL_AREAS = tuple(JEPX_DETAIL_AREA_OPTIONS.values())
JEPX_AREA_DISPLAY = {
    "System": "시스템가격",
    **{internal: display for display, internal in JEPX_DETAIL_AREA_DEFINITIONS},
}

# 전국 주간 모니터링의 기존 정렬 순서입니다.
JEPX_NATIONAL_AREA_ORDER = (
    "System",
    "Hokkaido",
    "Tohoku",
    "Tokyo",
    "Chubu",
    "Hokuriku",
    "Kansai",
    "Shikoku",
    "Chugoku",
    "Kyushu",
)
JEPX_NATIONAL_AREA_DISPLAY_ORDER = {
    area: index for index, area in enumerate(JEPX_NATIONAL_AREA_ORDER)
}