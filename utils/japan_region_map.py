"""외부 파일·API 없이 표시하는 일본 전력 권역 위치 안내도."""

from __future__ import annotations

from collections.abc import Mapping

from utils.eprx_area_groups import AGGREGATE_REGION_GROUPS
from utils.eprx_areas import EPRX_AREA_DISPLAY, EPRX_AREA_OPTIONS


# Simplified shapes are only location guides, not administrative boundaries.
# Each entry contains polygon points, a region anchor, and a label anchor.
REGION_GEOMETRY = {
    "Hokkaido": ("569,38 590,49 604,27 624,48 649,58 660,79 633,88 612,113 595,96 572,106 555,94 573,75 559,57", (611,73), (661,40)),
    "Tohoku": ("548,124 559,118 568,137 560,164 557,192 539,215 521,204 531,176 533,149", (546,166), (613,158)),
    "Tokyo": ("539,215 557,192 554,219 568,238 551,252 531,248 514,237 521,221", (541,233), (613,246)),
    "Chubu": ("479,214 505,203 521,204 539,215 521,221 514,237 531,248 507,258 483,255 464,275 449,256 451,235", (489,238), (501,306)),
    "Hokuriku": ("505,203 479,214 451,235 430,224 450,205 467,203 476,179 487,176 484,196 509,190 521,204", (469,211), (425,163)),
    "Kansai": ("430,224 451,235 449,256 464,275 437,291 416,267 389,259 383,237 408,230", (421,252), (399,326)),
    "Chugoku": ("383,237 389,259 363,253 333,261 300,259 272,271 257,253 284,243 316,239 345,233", (319,250), (297,212)),
    "Shikoku": ("364,276 390,272 386,290 363,299 330,302 313,291 335,278", (353,288), (304,326)),
    "Kyushu": ("251,276 270,283 276,307 258,322 242,344 225,335 224,313 206,309 211,287 231,287 234,271", (243,306), (170,306)),
}


def selected_map_regions(state: Mapping) -> tuple[tuple[str, ...], str]:
    """EPRX widget state only affects map styling; it never changes analysis state."""
    if state.get("market_selector", "EPRX 조정력시장") != "EPRX 조정력시장":
        return (), ""
    if state.get("eprx_analysis_unit", "개별 지역") == "광역권·합산":
        name = state.get("eprx_aggregate_group", next(iter(AGGREGATE_REGION_GROUPS)))
        return tuple(AGGREGATE_REGION_GROUPS.get(name, ())), name
    name = state.get(
        "regional_view_mode", state.get("eprx_last_individual_area", "홋카이도")
    )
    area = EPRX_AREA_OPTIONS.get(name)
    return ((area,) if area else ()), (name if area else "")


def region_map_html(state: Mapping) -> str:
    """Return a small responsive SVG card with Korean labels and selected outlines."""
    selected, _ = selected_map_regions(state)
    all_selected = len(selected) == len(REGION_GEOMETRY)
    parts = []
    for area, (points, (ax, ay), (lx, ly)) in REGION_GEOMETRY.items():
        active = area in selected
        fill = "#fbe9e7" if all_selected else "#f9d7d5" if active else "#e8eef3"
        stroke = "#d98f88" if all_selected else "#d62728" if active else "#9cabb9"
        weight = "600" if active else "400"
        name = EPRX_AREA_DISPLAY[area]
        parts.append(
            f'<g data-region="{area}" data-selected="{str(active).lower()}">'
            f'<title>{name}{" · 선택됨" if active else ""}</title>'
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{2.2 if active else 1.2}" stroke-linejoin="round"/>'
            f'<path d="M {ax},{ay} L {lx},{ly - 7}" fill="none" stroke="{stroke}" stroke-width="1"/>'
            f'<circle cx="{ax}" cy="{ay}" r="2.5" fill="{stroke}"/>'
            f'<text x="{lx}" y="{ly}" text-anchor="middle" font-size="21" '
            f'font-weight="{weight}" fill="currentColor">{name}</text></g>'
        )
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<style>html,body{margin:0;padding:0;background:transparent;color:#182b49;'
        'font-family:system-ui,-apple-system,"Segoe UI",sans-serif;}'
        '*{box-sizing:border-box;}'
        '.region-map-card{background:#fff;border:1px solid rgba(128,128,128,.22);'
        'border-radius:10px;padding:8px;margin:0;}'
        '.region-map-heading{font-size:.8rem;font-weight:600;line-height:18px;}'
        '.region-map-scroll{width:100%;min-width:0;}'
        '.region-map-svg{display:block;width:100%;height:200px;max-height:200px;min-width:0;'
        'font-family:inherit;}'
        '</style></head><body>'
        '<section class="region-map-card" aria-label="일본 전력 권역 안내">'
        '<div class="region-map-heading">일본 전력 권역</div>'
        '<div class="region-map-scroll"><svg class="region-map-svg" '
        'xmlns="http://www.w3.org/2000/svg" width="590" height="200" '
        'viewBox="125 12 590 345" preserveAspectRatio="xMidYMid meet" role="img" '
        'aria-labelledby="region-map-title region-map-description">'
        '<title id="region-map-title">일본 9개 전력 권역 위치 안내도</title>'
        '<desc id="region-map-description">북동쪽 홋카이도부터 도호쿠·도쿄, 중부·호쿠리쿠·간사이, '
        '서쪽 주고쿠와 남쪽 시코쿠·규슈의 대략적인 위치입니다.</desc>'
        + "".join(parts)
        + '</svg></div></section></body></html>'
    )


def render_japan_region_map(target, state: Mapping) -> None:
    # st.html uses DOMPurify's HTML-only profile, which removes SVG elements.
    # A local srcdoc iframe preserves SVG and provides an explicit viewport.
    target.iframe(region_map_html(state), height=240)
