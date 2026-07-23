"""EPRX와 JEPX를 구분한 일본 전력시장 모니터링 Streamlit 앱."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.eprx_loader import find_eprx_files, load_all_eprx_data
from utils.eprx_downloader import automation_approved, update_eprx_files
from utils.jepx_loader import (
    AREA_DISPLAY as JEPX_AREA_DISPLAY,
    find_jepx_files,
    load_all_jepx_data,
)
from utils.jepx_spread import (
    DEFAULT_OPERATION_MODE,
    calculate_all_daily_spreads,
    calculate_daily_spread,
)
from views.jepx_view import (
    ANALYSIS_WEEK_HELP_TEXT,
    render_jepx_diagnostics,
    render_jepx_tokyo_chubu_analysis,
    render_jepx_validation,
    render_jepx_weekly_monitor,
)
from utils.sample_data import AREAS_BY_ZONE, generate_sample_data
from utils.regional_analysis import (
    AREA_DISPLAY,
    calculate_previous_week_comparison,
    create_area_kpi_table,
    validate_area_profile,
    find_previous_week,
)
from utils.national_analysis import (
    ALL_AREA_DISPLAY,
    calculate_national_kpis,
    calculate_national_week_over_week,
    create_national_weekly_profile,
    create_regional_market_summary,
    validate_national_profile,
)
from utils.national_charts import (
    national_price_chart,
    national_ratio_chart,
    national_volume_chart,
    regional_bid_bar,
    regional_price_bar,
    regional_volume_bar,
)
from utils.regional_charts import (
    area_award_rate_chart,
    area_max_price_chart,
)
from utils.summary_display import (
    build_national_summary_data,
    build_regional_summary_data,
    excess_award_warning_markdown,
    national_summary_markdown,
    regional_summary_markdown,
)
from utils.weekly_aggregation import (
    add_week_columns,
    create_regional_weekly_profile,
    create_selected_area_weekly_profile,
    create_zone_weekly_profile,
    find_missing_areas,
)

st.set_page_config(
    page_title="일본 전력시장 모니터링",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIRECTORY = BASE_DIR / "data" / "eprx"
JEPX_DATA_DIRECTORY = BASE_DIR / "data" / "jepx" / "raw"
ZONE_COLORS = {"50Hz": "#1f77b4", "60Hz": "#d62728"}

DATA_SOURCE_ACTUAL = "실제 EPRX 파일"
DATA_SOURCE_SAMPLE = "샘플 데이터"
DISPLAY_AREA_NAMES = {
    "Hokkaido": "홋카이도",
    "Tohoku": "도호쿠",
    "Tokyo": "도쿄",
    "Chubu": "주부",
    "Hokuriku": "호쿠리쿠",
    "Kansai": "간사이",
    "Chugoku": "주고쿠",
    "Shikoku": "시코쿠",
    "Kyushu": "규슈",
}


def deployment_file_diagnostics(
    label: str, directory: Path, signatures: tuple[tuple[str, int, int], ...]
) -> pd.DataFrame:
    """민감한 절대경로 없이 배포 데이터 파일 존재·읽기 상태를 표시합니다."""
    rows = []
    for file_path, _, file_size in signatures:
        path = Path(file_path)
        try:
            with path.open("rb") as stream:
                stream.read(1)
            readable = "성공"
        except OSError as exc:
            readable = f"실패 ({type(exc).__name__})"
        rows.append(
            {
                "시장": label,
                "프로젝트 기준 디렉터리": ".",
                "탐색 경로": directory.relative_to(BASE_DIR).as_posix(),
                "폴더 존재": directory.is_dir(),
                "발견 파일 수": len(signatures),
                "파일명": path.name,
                "확장자": path.suffix.lower(),
                "파일 크기 (바이트)": file_size,
                "파일 읽기": readable,
            }
        )
    if not rows:
        rows.append(
            {
                "시장": label,
                "프로젝트 기준 디렉터리": ".",
                "탐색 경로": directory.relative_to(BASE_DIR).as_posix(),
                "폴더 존재": directory.is_dir(),
                "발견 파일 수": 0,
                "파일명": "없음",
                "확장자": "EPRX: .csv/.xlsx/.xls | JEPX: .csv",
                "파일 크기 (바이트)": 0,
                "파일 읽기": "대상 없음",
            }
        )
    return pd.DataFrame(rows)


def show_deployment_file_diagnostics(
    label: str, directory: Path, signatures: tuple[tuple[str, int, int], ...]
) -> None:
    with st.expander(f"{label} 배포 파일 진단"):
        st.dataframe(
            deployment_file_diagnostics(label, directory, signatures),
            width="stretch",
        )


def display_price_unit(unit: str) -> str:
    """원본 가격 단위를 화면용 한국어 표기로 변환합니다."""
    return unit.replace("円", "엔").replace("・", "·").replace("分", "분")


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """고정 seed로 만든 샘플 데이터를 캐시합니다."""
    return add_week_columns(generate_sample_data())


@st.cache_data
def load_actual_data(
    data_directory: str, file_signatures: tuple[tuple[str, int, int], ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """경로·수정시각·크기가 바뀌면 실제 파일 캐시를 갱신합니다."""
    del file_signatures
    return load_all_eprx_data(data_directory)


def actual_file_signatures() -> tuple[tuple[str, int, int], ...]:
    """캐시 무효화에 사용할 원본 파일 서명을 반환합니다."""
    return tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
        for path in find_eprx_files(DATA_DIRECTORY)
    )


def jepx_file_signatures() -> tuple[tuple[str, int, int], ...]:
    """JEPX 캐시 무효화에 사용할 원본 파일 서명을 반환합니다."""
    files = find_jepx_files(JEPX_DATA_DIRECTORY)
    return tuple(
        (
            str(Path(row["file_path"]).resolve()),
            Path(row["file_path"]).stat().st_mtime_ns,
            Path(row["file_path"]).stat().st_size,
        )
        for _, row in files.iterrows()
    )


@st.cache_data
def load_jepx_data(
    data_directory: str, file_signatures: tuple[tuple[str, int, int], ...]
):
    """EPRX 캐시와 분리해 실제 JEPX 파일을 로드합니다."""
    del file_signatures
    return load_all_jepx_data(data_directory)


@st.cache_data
def calculate_jepx_spread_results(
    long_data: pd.DataFrame,
    duration: int = 2,
    operation_mode: str = DEFAULT_OPERATION_MODE,
) -> pd.DataFrame:
    """선택 운용시간·계산방식의 검증된 일별 결과를 별도로 캐시합니다."""
    return calculate_all_daily_spreads(
        long_data, durations=[duration], operation_mode=operation_mode
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    """분모가 0이면 NaN을 반환합니다."""
    return numerator / denominator if denominator else np.nan


def calculate_kpis(zone_data: pd.DataFrame) -> dict[str, float]:
    """권역의 선택 주 전체 KPI를 계산합니다."""
    awarded_sum = zone_data["awarded_volume"].sum(min_count=1)
    weighted_price = safe_ratio(
        float((zone_data["avg_price"] * zone_data["awarded_volume"]).sum(min_count=1)),
        float(awarded_sum),
    )
    return {
        "평균 모집량": zone_data["procurement_volume"].mean(),
        "평균 입찰량": zone_data["bid_volume"].mean(),
        "평균 낙찰량": zone_data["awarded_volume"].mean(),
        "입찰경쟁률": safe_ratio(
            float(zone_data["bid_volume"].sum(min_count=1)),
            float(zone_data["procurement_volume"].sum(min_count=1)),
        ),
        "조달률": safe_ratio(
            float(awarded_sum),
            float(zone_data["procurement_volume"].sum(min_count=1)),
        ),
        "가중평균 낙찰가격": weighted_price,
    }


def show_kpi_table(
    zone_profile: pd.DataFrame, price_unit: str, target=st
) -> None:
    """50Hz와 60Hz KPI를 나란히 표시합니다."""
    values = {
        zone: calculate_kpis(zone_profile.loc[zone_profile["frequency_zone"] == zone])
        for zone in ["50Hz", "60Hz"]
    }
    table = pd.DataFrame(values)
    table.index = [
        "평균 모집량 (MW)",
        "평균 입찰량 (MW)",
        "평균 낙찰량 (MW)",
        "입찰경쟁률 (배)",
        "조달률 (%)",
        f"가중평균 낙찰가격 ({price_unit})",
    ]
    formatted = table.copy().astype(object)
    for label in table.index:
        for zone in table.columns:
            value = table.loc[label, zone]
            if pd.isna(value):
                formatted.loc[label, zone] = "확인 불가"
            elif label == "조달률 (%)":
                formatted.loc[label, zone] = f"{value:.2%}"
            elif label == "입찰경쟁률 (배)":
                formatted.loc[label, zone] = f"{value:.2f}배"
            else:
                formatted.loc[label, zone] = f"{value:,.2f}"
    target.dataframe(formatted, width="stretch")


def line_chart(
    data: pd.DataFrame,
    y: str,
    title: str,
    zones: list[str],
    y_title: str,
    reference_line: bool = False,
) -> go.Figure:
    """권역별 30분 시간대 선 그래프를 만듭니다."""
    visible = data.loc[data["frequency_zone"].isin(zones)]
    periods = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in (0, 30)]
    figure = px.line(
        visible,
        x="period_start",
        y=y,
        color="frequency_zone",
        category_orders={"period_start": periods},
        color_discrete_map=ZONE_COLORS,
        title=title,
        labels={
            "period_start": "시간대",
            y: y_title,
            "frequency_zone": "주파수권역",
        },
    )
    if reference_line:
        figure.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="모집량 충족 기준 (1.0배)",
            annotation_position="top left",
        )
    figure.update_xaxes(title_text="시간대 (일본 표준시)")
    figure.update_xaxes(dtick=4, tickangle=-45)
    figure.update_layout(hovermode="x unified", legend_title_text="")
    return figure


def volume_chart(zone_data: pd.DataFrame, zone: str) -> go.Figure:
    """한 권역의 모집·입찰·낙찰 물량 그래프를 만듭니다."""
    long_data = zone_data.melt(
        id_vars=["period_start"],
        value_vars=["procurement_volume", "bid_volume", "awarded_volume"],
        var_name="volume_type",
        value_name="volume",
    )
    volume_names = {
        "procurement_volume": "모집량",
        "bid_volume": "입찰량",
        "awarded_volume": "낙찰량",
    }
    long_data["물량 구분"] = long_data["volume_type"].map(volume_names)
    figure = px.line(
        long_data,
        x="period_start",
        y="volume",
        color="물량 구분",
        title=f"{zone}권역 물량",
        labels={
            "period_start": "시간대",
            "volume": "물량 (MW)",
            "물량 구분": "물량 구분",
        },
    )
    figure.update_xaxes(title_text="시간대 (일본 표준시)")
    figure.update_xaxes(dtick=4, tickangle=-45)
    figure.update_layout(hovermode="x unified", legend_title_text="")
    return figure


def show_diagnostics(
    file_summary: pd.DataFrame,
    error_data: pd.DataFrame,
    regional_profile: pd.DataFrame | None = None,
    target=st,
    title: str = "데이터 품질 및 파일 진단",
    deployment_directory: Path | None = None,
    deployment_signatures: tuple[tuple[str, int, int], ...] = (),
) -> None:
    """파일 처리와 선택 주의 품질 진단을 표시합니다."""
    with target.expander(title):
        if deployment_directory is not None:
            st.caption("배포 파일 탐색 및 읽기 상태")
            st.dataframe(
                deployment_file_diagnostics(
                    "EPRX", deployment_directory, deployment_signatures
                ),
                width="stretch",
            )
            st.caption(
                "캐시 정책: 파일 경로·수정시각·크기가 바뀌면 실제 데이터 캐시를 갱신합니다."
            )
        if file_summary.empty:
            st.info("지원 가능한 실제 파일이 없습니다.")
        else:
            st.caption("파일별 처리 로그")
            summary_display = file_summary.copy()
            summary_display["success"] = summary_display["success"].map(
                {True: "성공", False: "실패"}
            )
            summary_display = summary_display.rename(
                columns={
                    "source_file": "파일명",
                    "file_type": "파일 형식",
                    "file_size_bytes": "파일 크기 (바이트)",
                    "modified_at": "파일 수정시각",
                    "success": "처리 상태",
                    "encoding": "인코딩",
                    "encoding_attempts": "인코딩 확인 결과",
                    "delimiter": "구분자",
                    "header_row": "헤더 위치",
                    "raw_rows": "원본 행 수",
                    "normalized_rows": "정규화 행 수",
                    "primary_reserve_rows": "1차 조정력 행 수",
                    "error_rows": "오류 행 수",
                    "duplicate_rows": "중복 행 수",
                    "missing_required_columns": "필수 열 누락",
                    "error_message": "오류 메시지",
                }
            )
            st.dataframe(summary_display, width="stretch")
        if regional_profile is not None and not regional_profile.empty:
            missing = find_missing_areas(regional_profile)
            missing_display = {
                zone: [DISPLAY_AREA_NAMES.get(area, area) for area in areas]
                for zone, areas in missing.items()
            }
            st.write("누락 지역:", missing_display if missing_display else "없음")
            incomplete_slots = regional_profile.loc[
                regional_profile["observation_count"] < 7,
                ["area", "period_start", "observation_count"],
            ].rename(
                columns={
                    "area": "지역",
                    "period_start": "시작시간",
                    "observation_count": "관측일 수",
                }
            )
            incomplete_slots["지역"] = incomplete_slots["지역"].replace(
                DISPLAY_AREA_NAMES
            )
            st.write("7일 미만 지역·시간대:", len(incomplete_slots))
            if not incomplete_slots.empty:
                st.dataframe(incomplete_slots, width="stretch")
        if error_data.empty:
            st.success("파싱 또는 검증 오류가 없습니다.")
        else:
            st.caption("오류 및 검토 필요 행/진단")
            error_display = error_data.copy()
            if "severity" in error_display:
                error_display["severity"] = error_display["severity"].replace(
                    {"Fatal": "치명적 오류", "Review": "검토 필요"}
                )
            if "area" in error_display:
                error_display["area"] = error_display["area"].replace(
                    DISPLAY_AREA_NAMES
                )
            error_display = error_display.rename(
                columns={
                    "source_file": "파일명",
                    "severity": "심각도",
                    "error_code": "진단 코드",
                    "message": "진단 내용",
                    "delivery_date": "대상 날짜",
                    "area": "지역",
                    "product": "상품",
                    "period_no": "시간대 번호",
                }
            )
            st.dataframe(error_display, width="stretch")


def show_eprx_source_information(
    data_source: str,
    data: pd.DataFrame,
    file_summary: pd.DataFrame,
    price_unit: str,
    volume_unit: str,
) -> None:
    """분석 아래에서 데이터 출처와 원본 파일 메타데이터를 표시합니다."""
    actual = data_source == DATA_SOURCE_ACTUAL
    successful = (
        file_summary.loc[file_summary["success"].fillna(False)].copy()
        if actual and not file_summary.empty
        else pd.DataFrame()
    )
    source_status = (
        ", ".join(map(str, data["source_status"].dropna().unique()))
        if actual and "source_status" in data
        else "가상 샘플"
    ) or "확인 불가"
    source_status = source_status.replace("速報値", "속보치").replace(
        "確報値", "확정치"
    )
    latest_modified = (
        successful["modified_at"].max() if not successful.empty else pd.NaT
    )
    with st.expander("데이터 출처 및 원본 파일 정보"):
        st.dataframe(
            pd.DataFrame(
                [
                    ("데이터 소스", "실제 EPRX 파일" if actual else "가상 샘플"),
                    ("원본 파일명", ", ".join(successful["source_file"].astype(str)) if not successful.empty else "해당 없음"),
                    ("데이터 기간", f"{data['delivery_date'].min():%Y-%m-%d} ~ {data['delivery_date'].max():%Y-%m-%d}"),
                    ("마지막 파일 수정시각", f"{latest_modified:%Y-%m-%d %H:%M:%S %Z}" if pd.notna(latest_modified) else "해당 없음"),
                    ("데이터 상태", source_status),
                    ("가격 단위", price_unit),
                    ("물량 단위", volume_unit),
                    ("정규화 행 수", f"{len(data):,}행"),
                    ("지원 파일 수", f"{len(successful):,}개" if actual else "해당 없음"),
                ],
                columns=["항목", "내용"],
            ),
            width="stretch",
            hide_index=True,
        )
        if not successful.empty:
            st.caption("원본 파일별 정보")
            st.dataframe(successful, width="stretch", hide_index=True)


def show_update_result(result: dict[str, object]) -> None:
    """다운로드 확인·실행 결과를 한국어 표와 요약으로 표시합니다."""
    page = result.get("page_details", {})
    if page:
        page_type = (
            "동의 페이지"
            if page.get("is_agreement_page")
            else "거래실적 페이지"
            if page.get("is_results_page")
            else "확인 필요"
        )
        st.write(
            f"페이지 판별: {page_type} | HTTP {page.get('status_code', '확인 불가')} | "
            f"다운로드 링크 {len(result.get('all_links', [])):,}개"
        )
        st.caption(f"최종 URL: {page.get('final_url', '확인 불가')}")
        st.caption(
            f"Content-Type: {page.get('content_type', '확인 불가')} | "
            f"응답 크기: {page.get('response_size', 0):,} bytes | "
            f"페이지 제목: {page.get('page_title') or '확인 불가'}"
        )
    errors = result.get("errors", [])
    if errors:
        for error in errors:
            st.error(str(error))
        return

    candidates = result.get("primary_candidates", pd.DataFrame())
    new_candidates = result.get("new_candidates", pd.DataFrame())
    existing = result.get("existing_files", pd.DataFrame())
    revisions = result.get("revision_candidates", pd.DataFrame())
    st.write(
        f"1차 조정력 후보 {len(candidates):,}개 | "
        f"신규 후보 {len(new_candidates):,}개 | "
        f"이미 보유 {len(existing):,}개 | "
        f"수정 가능성 {len(revisions):,}개"
    )
    candidate_display = pd.concat(
        [
            new_candidates.assign(status="신규 후보"),
            existing.assign(status="이미 보유"),
            revisions.assign(status="수정 가능성"),
        ],
        ignore_index=True,
    )
    if not candidate_display.empty:
        candidate_display = candidate_display[
            [
                "link_text",
                "file_name",
                "published_date",
                "result_status",
                "product_hint",
                "is_primary_reserve_candidate",
                "status",
            ]
        ].rename(
            columns={
                "link_text": "링크 설명",
                "file_name": "원본 파일명",
                "published_date": "게시일",
                "result_status": "데이터 상태",
                "product_hint": "상품 추정",
                "is_primary_reserve_candidate": "1차 조정력 후보",
                "status": "처리 결과",
            }
        )
        candidate_display["1차 조정력 후보"] = candidate_display[
            "1차 조정력 후보"
        ].map({True: "예", False: "아니요"})
        st.dataframe(candidate_display, width="stretch")
    selected_candidate = result.get("selected_candidate", pd.DataFrame())
    if isinstance(selected_candidate, pd.DataFrame) and not selected_candidate.empty:
        st.info(f"선택된 최신 후보: {selected_candidate.iloc[0]['file_name']}")

    downloads = result.get("download_results", pd.DataFrame())
    if isinstance(downloads, pd.DataFrame) and not downloads.empty:
        status = downloads["status"].fillna("")
        parse_status = downloads["parse_status"].fillna("")
        st.write(
            f"신규 다운로드 {(status == '다운로드 완료').sum():,}개 | "
            f"중복 건너뜀 {(status == '중복 건너뜀').sum():,}개 | "
            f"파싱 성공 {(parse_status == '파싱 성공').sum():,}개 | "
            f"파싱 실패 {(parse_status == '파싱 실패').sum():,}개 | "
            f"오류 {(status == '실패').sum():,}개"
        )
        download_display = downloads[
            [
                "saved_file_name",
                "file_size",
                "downloaded_at",
                "sha256",
                "status",
                "duplicate_of",
                "revision_of",
                "error_message",
                "parse_status",
                "parse_error",
                "execution_mode",
                "detected_encoding",
                "parsed_row_count",
                "primary_reserve_row_count",
                "detected_date_min",
                "detected_date_max",
                "detected_areas",
                "detected_zones",
                "required_metrics_present",
            ]
        ].rename(
            columns={
                "saved_file_name": "저장 파일명",
                "file_size": "파일 크기",
                "downloaded_at": "다운로드 시각",
                "sha256": "파일 해시",
                "status": "처리 결과",
                "duplicate_of": "동일 파일",
                "revision_of": "수정 전 파일",
                "error_message": "오류 내용",
                "parse_status": "파싱 결과",
                "parse_error": "파싱 오류",
                "execution_mode": "실행 방식",
                "detected_encoding": "감지 인코딩",
                "parsed_row_count": "파싱 행 수",
                "primary_reserve_row_count": "1차 조정력 행 수",
                "detected_date_min": "데이터 시작일",
                "detected_date_max": "데이터 종료일",
                "detected_areas": "포함 지역",
                "detected_zones": "포함 주파수권역",
                "required_metrics_present": "필수 6개 지표",
            }
        )
        st.dataframe(download_display, width="stretch")


def render_national_summary(summary_data: dict, target=st) -> None:
    """전국 규칙 기반 요약을 항목형 Markdown 카드로 표시합니다."""
    if summary_data.get("incomplete"):
        target.warning(
            "일부 지역 또는 시간대 데이터가 불완전하므로 아래 요약은 참고용입니다."
        )
    with target.container(border=True):
        st.markdown(national_summary_markdown(summary_data))


def render_regional_summary(summary_data: dict, target=st) -> None:
    """도쿄·중부 규칙 기반 요약을 항목형 Markdown 카드로 표시합니다."""
    if summary_data.get("incomplete"):
        target.warning(
            "일부 지역 또는 시간대 데이터가 불완전하므로 아래 요약은 참고용입니다."
        )
    with target.container(border=True):
        st.markdown(regional_summary_markdown(summary_data))
    with target.expander("입찰경쟁률이란?"):
        st.markdown(
            """
- 입찰경쟁률 = 입찰량 ÷ 모집량
- 1.0배 미만: 입찰량이 모집량보다 적음
- 1.0배 이상: 입찰량이 모집량 이상
- 값이 높을수록 모집량 대비 더 많은 물량이 입찰됨
- 이 지표만으로 사업성이나 수익성을 판단할 수는 없음
"""
        )


def render_excess_award_warning(
    target,
    period_counts: dict[str, int],
    source_row_count: int | None = None,
) -> None:
    """낙찰량이 모집량보다 많은 사례를 공통 경고 형식으로 표시합니다."""
    target.warning(
        excess_award_warning_markdown(period_counts, source_row_count)
    )
    with target.expander("왜 이런 값이 나오나요?"):
        st.markdown(
            """
- 조달률은 낙찰량을 모집량으로 나눈 값입니다.
- 일부 원본 자료에서는 낙찰량이 모집량보다 크게 표시되어 조달률이 100%를 넘을 수 있습니다.
- 원본 데이터의 집계 기준 또는 지역 구분 방식에 따라 값이 달라질 수 있습니다.
- 앱은 원본 값을 임의로 수정하거나 100%로 제한하지 않습니다.
- 정확한 공표 기준은 EPRX 원문 또는 공식 답변을 확인해야 합니다.
"""
        )


def render_hierarchical_metric_table(
    target, data: pd.DataFrame, metric_column: str = "지표"
) -> None:
    """고정 지표명의 본문과 괄호 설명을 분리한 작은 비교표를 표시합니다."""
    display = data.reset_index(names=metric_column) if metric_column not in data else data

    def label_html(value: object) -> str:
        label = str(value)
        boundary = label.find(" (")
        if boundary < 0:
            return f'<span class="metric-label-main">{escape(label)}</span>'
        main = escape(label[:boundary])
        sub = escape(label[boundary + 1 :])
        return (
            f'<span class="metric-label-main">{main}</span>'
            f'<span class="metric-label-sub">{sub}</span>'
        )

    headers = "".join(f"<th>{escape(str(column))}</th>" for column in display.columns)
    rows = []
    for _, row in display.iterrows():
        cells = []
        for column in display.columns:
            value = row[column]
            content = label_html(value) if column == metric_column else escape(str(value))
            css_class = "metric-name-cell" if column == metric_column else "metric-value-cell"
            cells.append(f'<td class="{css_class}">{content}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    target.markdown(
        """
<style>
.metric-hierarchy-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.metric-hierarchy-table th, .metric-hierarchy-table td {
  padding: 0.45rem 0.6rem; border-bottom: 1px solid rgba(128, 128, 128, 0.25);
}
.metric-hierarchy-table th { text-align: right; font-weight: 600; }
.metric-hierarchy-table th:first-child, .metric-name-cell { text-align: left; }
.metric-value-cell { text-align: right; font-variant-numeric: tabular-nums; }
.metric-label-main { color: inherit; font-size: 0.95rem; font-weight: 500; }
.metric-label-sub { color: #8a8f98; font-size: 0.75rem; margin-left: 0.25rem; }
@media (max-width: 700px) { .metric-label-sub { display: block; margin-left: 0; } }
</style>
"""
        + f'<table class="metric-hierarchy-table"><thead><tr>{headers}</tr></thead>'
        + f"<tbody>{''.join(rows)}</tbody></table>",
        unsafe_allow_html=True,
    )


def render_national_regional_table(
    target, regional_summary: pd.DataFrame, price_unit: str
) -> None:
    """전국 탭의 지역별 주간 비교표를 짧은 헤더로 표시합니다."""
    region_display = regional_summary.rename(
        columns={
            "area_display": "지역", "frequency_zone": "주파수권역",
            "avg_procurement_volume": "평균 모집량 (MW)",
            "avg_bid_volume": "평균 입찰량 (MW)",
            "avg_awarded_volume": "평균 낙찰량 (MW)",
            "bid_coverage_ratio": "입찰경쟁률 (배)",
            "procurement_rate": "조달률", "award_rate": "입찰 대비 낙찰률",
            "weighted_avg_price": f"낙찰량 가중평균 낙찰가격 ({price_unit})",
            "max_price": f"최고 낙찰가격 ({price_unit})",
            "min_price": f"최저 낙찰가격 ({price_unit})",
            "shortage_period_count": "미조달 시간대 수",
            "max_shortage_volume": "최대 미조달량 (MW)",
            "observed_period_count": "관측 시간대 수",
            "completeness_flag": "데이터 완전성",
        }
    )
    region_display["데이터 완전성"] = region_display["데이터 완전성"].replace(
        {"Complete": "데이터 완전", "Incomplete": "데이터 불완전"}
    )
    target.subheader("지역별 주간 요약")
    target.caption(
        "데이터 기준: 모집량은 TSO별, 입찰량·낙찰량·낙찰가격은 전원 소재지별입니다. "
        "입찰경쟁률과 조달률은 서로 다른 지역 귀속 기준을 비교하는 참고지표입니다."
    )
    target.dataframe(
        region_display.drop(columns=["area"]).style.format(
            {
                "평균 모집량 (MW)": "{:,.2f}", "평균 입찰량 (MW)": "{:,.2f}",
                "평균 낙찰량 (MW)": "{:,.2f}", "입찰경쟁률 (배)": "{:.2f}배",
                "조달률": "{:.2%}", "입찰 대비 낙찰률": "{:.2%}",
                f"낙찰량 가중평균 낙찰가격 ({price_unit})": "{:,.2f}",
                f"최고 낙찰가격 ({price_unit})": "{:,.2f}",
                f"최저 낙찰가격 ({price_unit})": "{:,.2f}",
                "최대 미조달량 (MW)": "{:,.2f}",
            },
            na_rep="계산 불가",
        ),
        width="stretch",
    )


def render_national_overview(
    target,
    data: pd.DataFrame,
    selected_week: object,
    selected_week_days: int,
    regional_profile: pd.DataFrame,
    price_unit: str,
    volume_unit: str,
    file_summary: pd.DataFrame,
) -> None:
    """일본 9개 지역의 전국 시장 규모 참고지표를 표시합니다."""
    def format_value(value: object, pattern: str, suffix: str = "") -> str:
        if pd.isna(value):
            return "계산 불가"
        return f"{value:{pattern}}{suffix}"

    national = create_national_weekly_profile(regional_profile)
    kpis = calculate_national_kpis(national, regional_profile)
    regional_summary = create_regional_market_summary(regional_profile)
    national_validation_messages = validate_national_profile(
        national, regional_profile, selected_week_days
    )
    target.info(
        "전국 시장 요약은 일본 9개 지역의 1차 조정력 공표자료를 합산한 "
        "참고지표입니다. 물량은 지역별 합계이며, 전국 평균가격은 지역별 평균 "
        "낙찰가격을 낙찰량으로 가중한 값입니다. 일본 전체에 공통으로 적용되는 "
        "단일 낙찰가격을 의미하지 않습니다."
    )
    target.caption(
        f"분석 주차 {pd.Timestamp(selected_week):%Y-%m-%d} 시작 | "
        f"포함 지역 {int(national['area_count'].max()) if not national.empty else 0}/9 | "
        f"데이터 완전성 "
        f"{'완전' if not national.empty and national['completeness_flag'].eq('Complete').all() else '불완전'}"
    )

    previous_week, previous_days = find_previous_week(data, selected_week)
    comparison = pd.DataFrame()
    if previous_week is not None:
        previous_regional = create_regional_weekly_profile(data, previous_week)
        previous_national = create_national_weekly_profile(previous_regional)
        comparison = calculate_national_week_over_week(
            national, previous_national, regional_profile, previous_regional
        )
        if previous_days != 7:
            target.warning(
                f"전주 비교 대상 {previous_week:%Y-%m-%d} 주차가 "
                f"{previous_days}/7일로 불완전합니다."
            )
    national_over = national.loc[national["procurement_rate"] > 1]
    source_count = None
    if not national_over.empty:
        selected_rows = data.loc[
            data["week_start"].eq(pd.Timestamp(selected_week))
        ]
        source_count = int(
            (
                selected_rows["awarded_volume"]
                > selected_rows["procurement_volume"]
            ).sum()
        )

    target.subheader("전주 대비 전국 변화")
    if previous_week is None:
        target.info("비교 가능한 이전 주차가 없습니다.")
    else:
        comparison_display = comparison.copy().astype(object)
        for index, row in comparison_display.iterrows():
            metric = row["지표"]
            is_percent = metric == "전국 조달률 (%)"
            is_ratio = metric == "전국 입찰경쟁률 (배)"
            comparison_display.loc[index, "현재 주"] = (
                format_value(row["현재 주"], ".2%" if is_percent else ",.2f")
            )
            comparison_display.loc[index, "전주"] = (
                format_value(row["전주"], ".2%" if is_percent else ",.2f")
            )
            if pd.isna(row["절대 변화"]):
                comparison_display.loc[index, "절대 변화"] = "계산 불가"
            elif is_percent:
                comparison_display.loc[index, "절대 변화"] = (
                    f"{row['절대 변화'] * 100:+.2f}%p"
                )
            elif is_ratio:
                comparison_display.loc[index, "절대 변화"] = (
                    f"{row['절대 변화']:+.2f}배"
                )
            else:
                comparison_display.loc[index, "절대 변화"] = (
                    f"{row['절대 변화']:+,.2f}"
                )
            comparison_display.loc[index, "변화율"] = (
                "계산 불가"
                if pd.isna(row["변화율"])
                else f"{row['변화율']:+.2%}"
            )
        comparison_display["지표"] = comparison_display["지표"].replace(
            {
                "평균 모집량 (MW)": "평균 모집량 (TSO별, MW)",
                "평균 입찰량 (MW)": "평균 입찰량 (전원 소재지별, MW)",
                "평균 낙찰량 (MW)": "평균 낙찰량 (전원 소재지별, MW)",
                "전국 입찰경쟁률 (배)": "전국 입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량, 배)",
                "전국 조달률 (%)": "전국 조달률 (소재지별 낙찰량 ÷ TSO별 모집량, %)",
                "전국 낙찰량 가중평균 낙찰가격": "전국 가중평균 낙찰가격 (전원 소재지별)",
            }
        )
        target.caption(f"비교 주차: {previous_week:%Y-%m-%d} 시작 주")
        render_hierarchical_metric_table(
            target, comparison_display.drop(columns=["절대 변화 단위"])
        )

    target.plotly_chart(national_price_chart(national, price_unit), width="stretch")
    target.caption(
        "이 가격은 전원 소재지별 평균 낙찰가격을 전원 소재지별 낙찰량으로 "
        "가중한 참고값이며, "
        "전국 공통 단일 낙찰가격이 아닙니다."
    )
    target.plotly_chart(national_volume_chart(national), width="stretch")
    ratio_columns = target.columns(2)
    ratio_columns[0].plotly_chart(
        national_ratio_chart(national, "bid_coverage_ratio"), width="stretch"
    )
    ratio_columns[1].plotly_chart(
        national_ratio_chart(national, "procurement_rate"), width="stretch"
    )
    target.plotly_chart(regional_volume_bar(regional_summary), width="stretch")
    price_columns = target.columns(2)
    price_columns[0].plotly_chart(
        regional_price_bar(regional_summary, price_unit), width="stretch"
    )
    price_columns[1].plotly_chart(
        regional_bid_bar(regional_summary), width="stretch"
    )
    render_national_regional_table(target, regional_summary, price_unit)

    target.subheader("전국 시간대별 상세 데이터")
    detailed = national.copy()
    detailed["missing_areas"] = detailed["missing_areas"].apply(
        lambda value: ", ".join(
            ALL_AREA_DISPLAY.get(area.strip(), area.strip())
            for area in str(value).split(",")
            if area.strip()
        )
    )
    detailed["completeness_flag"] = detailed["completeness_flag"].replace(
        {"Complete": "데이터 완전", "Incomplete": "데이터 불완전"}
    )
    detailed = detailed.rename(
        columns={
            "period_no": "시간대 번호",
            "period_start": "시작시간",
            "procurement_volume": "모집량 (TSO별, MW)",
            "bid_volume": "입찰량 (전원 소재지별, MW)",
            "awarded_volume": "낙찰량 (전원 소재지별, MW)",
            "avg_price": f"가중평균 낙찰가격 (전원 소재지별, {price_unit})",
            "max_price": f"최고 낙찰가격 (전원 소재지별, {price_unit})",
            "min_price": f"최저 낙찰가격 (전원 소재지별, {price_unit})",
            "price_range": "가격 범위",
            "bid_coverage_ratio": "입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량, 배)",
            "procurement_rate": "조달률 (소재지별 낙찰량 ÷ TSO별 모집량)",
            "award_rate": "입찰 대비 낙찰률 (전원 소재지별)",
            "excess_bid_volume": "초과입찰량 (소재지별 입찰량 − TSO별 모집량, MW)",
            "shortage_volume": "미조달량 (TSO별 모집량 − 소재지별 낙찰량, MW)",
            "area_count": "포함 지역 수",
            "missing_areas": "누락 지역",
            "completeness_flag": "데이터 완전성",
        }
    )
    detailed_display = detailed[
            [
                "시간대 번호",
                "시작시간",
                "모집량 (TSO별, MW)",
                "입찰량 (전원 소재지별, MW)",
                "낙찰량 (전원 소재지별, MW)",
                f"가중평균 낙찰가격 (전원 소재지별, {price_unit})",
                f"최고 낙찰가격 (전원 소재지별, {price_unit})",
                f"최저 낙찰가격 (전원 소재지별, {price_unit})",
                "가격 범위",
                "입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량, 배)",
                "조달률 (소재지별 낙찰량 ÷ TSO별 모집량)",
                "입찰 대비 낙찰률 (전원 소재지별)",
                "초과입찰량 (소재지별 입찰량 − TSO별 모집량, MW)",
                "미조달량 (TSO별 모집량 − 소재지별 낙찰량, MW)",
                "포함 지역 수",
                "누락 지역",
                "데이터 완전성",
            ]
        ].rename(
            columns={
                "모집량 (TSO별, MW)": "모집량 (MW)",
                "입찰량 (전원 소재지별, MW)": "입찰량 (MW)",
                "낙찰량 (전원 소재지별, MW)": "낙찰량 (MW)",
                f"가중평균 낙찰가격 (전원 소재지별, {price_unit})": f"가중평균 낙찰가격 ({price_unit})",
                f"최고 낙찰가격 (전원 소재지별, {price_unit})": f"최고 낙찰가격 ({price_unit})",
                f"최저 낙찰가격 (전원 소재지별, {price_unit})": f"최저 낙찰가격 ({price_unit})",
                "입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량, 배)": "입찰경쟁률 (배)",
                "조달률 (소재지별 낙찰량 ÷ TSO별 모집량)": "조달률",
                "입찰 대비 낙찰률 (전원 소재지별)": "입찰 대비 낙찰률",
                "초과입찰량 (소재지별 입찰량 − TSO별 모집량, MW)": "초과입찰량 (MW)",
                "미조달량 (TSO별 모집량 − 소재지별 낙찰량, MW)": "미조달량 (MW)",
            }
        )
    target.caption(
        "데이터 기준: 모집량은 TSO별, 입찰량·낙찰량·낙찰가격은 전원 소재지별입니다. "
        "입찰경쟁률과 조달률은 서로 다른 지역 귀속 기준을 비교하는 참고지표입니다."
    )
    target.dataframe(
        detailed_display.style.format(
            {
                "모집량 (MW)": "{:,.2f}",
                "입찰량 (MW)": "{:,.2f}",
                "낙찰량 (MW)": "{:,.2f}",
                f"가중평균 낙찰가격 ({price_unit})": "{:,.2f}",
                f"최고 낙찰가격 ({price_unit})": "{:,.2f}",
                f"최저 낙찰가격 ({price_unit})": "{:,.2f}",
                "가격 범위": "{:,.2f}",
                "입찰경쟁률 (배)": "{:.2f}배",
                "조달률": "{:.2%}",
                "입찰 대비 낙찰률": "{:.2%}",
                "초과입찰량 (MW)": "{:,.2f}",
                "미조달량 (MW)": "{:,.2f}",
            },
            na_rep="계산 불가",
        ),
        width="stretch",
    )

    for message in national_validation_messages:
        target.warning(message)
    if not national_over.empty:
        render_excess_award_warning(
            target,
            {"전국": len(national_over)},
            source_count,
        )

    with target.expander("데이터 기준 설명"):
        st.markdown(
            """
- 전원 소재지별: 해당 지역에 위치한 발전기·ESS 등 전원의 입찰 및 낙찰 결과
- 일반송배전사업자(TSO)별: 해당 지역 일반송배전사업자의 모집 및 조달 결과
- 낙찰가격·입찰량·낙찰량은 전원 소재지별, 모집량은 TSO별
- 전국 모집량·입찰량·낙찰량: 각 공표 기준에 따른 9개 지역 물량의 시간대별 합계
- 전국 입찰경쟁률: 전원 소재지별 전국 입찰량 ÷ TSO별 전국 모집량
- 전국 조달률: 전원 소재지별 전국 낙찰량 ÷ TSO별 전국 모집량
- 전국 가중평균 낙찰가격: 전원 소재지별 평균 낙찰가격을 전원 소재지별 낙찰량으로 가중한 참고값
- 최고·최저가격은 지역 가격을 합산하거나 평균하지 않고 각각 최댓값·최솟값 사용
- 주간 물량 합계는 30분 상품구간별 MW 합계이며 MWh 에너지량이 아님
- 두 기준은 지역 귀속 방식이 다르므로 혼합 계산 지표는 해석에 주의 필요
- 일부 원본에서는 낙찰량이 모집량보다 많은 시간대가 나타날 수 있음
- 전국 요약은 시장 규모 파악용이며, 지역 비교는 권역 및 상세분석 탭을 이용

**본 앱의 낙찰가격은 전원 소재지별 공표값을 사용합니다.**

모집량은 TSO별 공표값을 사용하며, 입찰량·낙찰량·낙찰가격은 전원 소재지별 공표값을 사용합니다.
입찰경쟁률과 조달률은 서로 다른 지역 귀속 기준을 비교하는 참고지표입니다.
"""
        )


def render_regional_analysis(
    target,
    data: pd.DataFrame,
    selected_week: object,
    selected_week_days: int,
    price_unit: str,
) -> None:
    """도쿄·중부 지역 상세분석 탭을 표시합니다."""
    view_options = {
        "도쿄": ["Tokyo"],
        "중부": ["Chubu"],
    }
    view = target.radio(
        "보기 방식",
        list(view_options),
        horizontal=True,
        key="regional_view_mode",
    )
    visible_areas = view_options[view]
    profile = create_selected_area_weekly_profile(
        data, selected_week, ["Tokyo", "Chubu"]
    )
    warnings = validate_area_profile(profile, selected_week_days)
    for message in warnings:
        target.warning(message)
    if profile.empty:
        target.info("도쿄·중부 지역 상세 데이터를 표시할 수 없습니다.")
        return

    selected_start = pd.Timestamp(selected_week).normalize()
    raw_week = data.loc[data["week_start"].eq(selected_start)].copy()
    kpi_table = create_area_kpi_table(profile, raw_week)
    previous, previous_meta = calculate_previous_week_comparison(
        data, selected_week, profile
    )
    if previous_meta["previous_week"] is None:
        target.info("비교 가능한 이전 주차가 없습니다.")
    elif not previous_meta["previous_complete"]:
        target.warning(
            f"전주 비교 대상 {previous_meta['previous_week']:%Y-%m-%d} 주차가 "
            f"{previous_meta['previous_days']}/7일로 불완전합니다."
        )

    over_rate = profile.loc[profile["procurement_rate"] > 1]

    target.subheader(f"{view} 주간 핵심지표")
    weekly_kpi_rows = [
        "평균 모집량 (MW)",
        "평균 입찰량 (MW)",
        "평균 낙찰량 (MW)",
        "입찰 대비 낙찰률 (%)",
    ]
    excluded_kpi_rows = {
        "입찰경쟁률 (배)",
        "조달률 (%)",
        "평균 낙찰가격",
        "최고 낙찰가격",
        "최저 낙찰가격",
        "평균 가격범위",
        "미조달 시간대 수",
        "평균 미조달량 (MW)",
    }
    kpi_display = kpi_table.reindex(weekly_kpi_rows)[[view]].copy().astype(object)
    percent_rows = {"입찰 대비 낙찰률 (%)"}
    for row in kpi_display.index:
        for column in kpi_display.columns:
            value = kpi_display.loc[row, column]
            if pd.isna(value):
                formatted = "계산 불가"
            elif row in percent_rows:
                formatted = f"{value:.2%}"
            else:
                formatted = f"{value:,.2f}"
            kpi_display.loc[row, column] = formatted
    kpi_display = kpi_display.rename(
        index={
            "평균 모집량 (MW)": "평균 모집량 (TSO별, MW)",
            "평균 입찰량 (MW)": "평균 입찰량 (전원 소재지별, MW)",
            "평균 낙찰량 (MW)": "평균 낙찰량 (전원 소재지별, MW)",
            "입찰 대비 낙찰률 (%)": "입찰 대비 낙찰률 (전원 소재지별, %)",
            "평균 낙찰가격": f"평균 낙찰가격 (전원 소재지별, {price_unit})",
            "최고 낙찰가격": f"최고 낙찰가격 (전원 소재지별, {price_unit})",
            "최저 낙찰가격": f"최저 낙찰가격 (전원 소재지별, {price_unit})",
        }
    )
    render_hierarchical_metric_table(target, kpi_display)

    if not over_rate.empty:
        source_over = raw_week.loc[
            raw_week["area"].isin(["Tokyo", "Chubu"])
            & (
                raw_week["awarded_volume"]
                > raw_week["procurement_volume"]
            )
        ]
        render_excess_award_warning(
            target,
            {
                AREA_DISPLAY[area]: int(count)
                for area, count in over_rate.groupby("area").size().items()
            },
            len(source_over),
        )

    target.caption("전원 소재지별 최고 낙찰가격의 선택 주차 동일 시간대 평균")
    target.plotly_chart(
        area_max_price_chart(profile, visible_areas, price_unit), width="stretch"
    )
    target.caption("입찰량과 낙찰량 모두 전원 소재지별 공표값을 사용합니다.")
    target.plotly_chart(
        area_award_rate_chart(profile, visible_areas), width="stretch"
    )

    target.subheader("전주 대비 변화")
    if not previous.empty:
        previous_display = previous.loc[
            ~previous["지표"].isin(excluded_kpi_rows)
        ].copy()
        previous_display["현재 주"] = previous_display["현재 주"].map(
            lambda value: "계산 불가" if pd.isna(value) else f"{value:,.2f}"
        )
        previous_display["전주"] = previous_display["전주"].map(
            lambda value: "계산 불가" if pd.isna(value) else f"{value:,.2f}"
        )
        previous_display["절대 변화"] = previous_display["절대 변화"].map(
            lambda value: "계산 불가" if pd.isna(value) else f"{value:+,.2f}"
        )
        previous_display["변화율"] = previous_display["변화율"].map(
            lambda value: "계산 불가" if pd.isna(value) else f"{value:+.2%}"
        )
        previous_display["지표"] = previous_display["지표"].replace(
            {
                "평균 모집량 (MW)": "평균 모집량 (TSO별, MW)",
                "평균 입찰량 (MW)": "평균 입찰량 (전원 소재지별, MW)",
                "평균 낙찰량 (MW)": "평균 낙찰량 (전원 소재지별, MW)",
                "입찰경쟁률 (배)": "입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량, 배)",
                "조달률 (%)": "조달률 (소재지별 낙찰량 ÷ TSO별 모집량, %)",
                "최고 낙찰가격": f"최고 낙찰가격 (전원 소재지별, {price_unit})",
            }
        )
        target.caption(
            f"비교 주차: {previous_meta['previous_week']:%Y-%m-%d} 시작 주"
        )
        render_hierarchical_metric_table(target, previous_display)

    target.subheader(f"{view} 시간대별 상세 데이터")
    detailed = profile.loc[profile["area"].isin(visible_areas)].copy()
    detailed["area"] = detailed["area"].map(AREA_DISPLAY)
    detailed["completeness_flag"] = detailed["completeness_flag"].replace(
        {"Complete": "데이터 완전", "Incomplete": "데이터 불완전"}
    )
    detailed = detailed[
        [
            "area",
            "period_no",
            "period_start",
            "procurement_volume",
            "bid_volume",
            "awarded_volume",
            "max_price",
            "avg_price",
            "min_price",
            "award_rate",
            "excess_bid_volume",
            "observation_count",
            "completeness_flag",
        ]
    ].rename(
        columns={
            "area": "지역",
            "period_no": "시간대 번호",
            "period_start": "시작시간",
            "procurement_volume": "모집량 (TSO별, MW)",
            "bid_volume": "입찰량 (전원 소재지별, MW)",
            "awarded_volume": "낙찰량 (전원 소재지별, MW)",
            "max_price": f"최고 낙찰가격 (전원 소재지별, {price_unit})",
            "avg_price": f"평균 낙찰가격 (전원 소재지별, {price_unit})",
            "min_price": f"최저 낙찰가격 (전원 소재지별, {price_unit})",
            "award_rate": "입찰 대비 낙찰률 (전원 소재지별)",
            "excess_bid_volume": "초과입찰량 (소재지별 입찰량 − TSO별 모집량, MW)",
            "observation_count": "관측일수",
            "completeness_flag": "데이터 완전성",
        }
    )
    detailed = detailed.rename(
        columns={
            "모집량 (TSO별, MW)": "모집량 (MW)",
            "입찰량 (전원 소재지별, MW)": "입찰량 (MW)",
            "낙찰량 (전원 소재지별, MW)": "낙찰량 (MW)",
            f"최고 낙찰가격 (전원 소재지별, {price_unit})": f"최고 낙찰가격 ({price_unit})",
            f"평균 낙찰가격 (전원 소재지별, {price_unit})": f"평균 낙찰가격 ({price_unit})",
            f"최저 낙찰가격 (전원 소재지별, {price_unit})": f"최저 낙찰가격 ({price_unit})",
            "입찰 대비 낙찰률 (전원 소재지별)": "입찰 대비 낙찰률",
            "초과입찰량 (소재지별 입찰량 − TSO별 모집량, MW)": "초과입찰량 (MW)",
        }
    )
    target.caption(
        "데이터 기준: 모집량은 TSO별, 입찰량·낙찰량·낙찰가격은 전원 소재지별입니다. "
        "입찰경쟁률과 조달률은 서로 다른 지역 귀속 기준을 비교하는 참고지표입니다."
    )
    target.caption(
        "최고·평균·최저 낙찰가격은 각 날짜에 공표된 값을 동일 시간대별로 모아 "
        "선택 주차 단위로 평균한 값입니다."
    )
    target.dataframe(
        detailed.style.format(
            {
                "모집량 (MW)": "{:,.2f}",
                "입찰량 (MW)": "{:,.2f}",
                "낙찰량 (MW)": "{:,.2f}",
                f"최고 낙찰가격 ({price_unit})": "{:,.2f}",
                f"평균 낙찰가격 ({price_unit})": "{:,.2f}",
                f"최저 낙찰가격 ({price_unit})": "{:,.2f}",
                "입찰 대비 낙찰률": "{:.2%}",
                "초과입찰량 (MW)": "{:,.2f}",
            },
            na_rep="계산 불가",
        ),
        width="stretch",
    )
    with target.expander("데이터 기준 설명"):
        st.markdown(
            """
- 도쿄: 50Hz권역에 속하는 개별 지역
- 중부: 60Hz권역에 속하는 개별 지역
- 전원 소재지별: 해당 지역에 위치한 발전기·ESS 등 전원의 입찰 및 낙찰 결과
- 일반송배전사업자(TSO)별: 해당 지역 일반송배전사업자의 모집 및 조달 결과
- 낙찰가격·입찰량·낙찰량은 전원 소재지별
- 모집량은 TSO별
- 입찰경쟁률: 전원 소재지별 입찰량 ÷ TSO별 모집량
- 조달률: 전원 소재지별 낙찰량 ÷ TSO별 모집량
- 입찰 대비 낙찰률: 전원 소재지별 낙찰량 ÷ 전원 소재지별 입찰량
- 가격 범위: 최고 낙찰가격 − 최저 낙찰가격
- 초과입찰량: 입찰량 − 모집량
- 미조달량: 모집량 − 낙찰량과 0 중 큰 값
- 전주 대비 변화: 현재 주 값과 직전 보유 주차 값의 차이
- 두 기준은 지역 귀속 방식이 다르므로 혼합 계산 지표는 해석에 주의 필요

일부 EPRX 원본 데이터에서는 낙찰량이 모집량보다 많은 시간대가 나타날 수 있습니다.
앱은 원본 EPRX 값을 수정하지 않고 그대로 표시합니다.

**본 앱의 낙찰가격은 전원 소재지별 공표값을 사용합니다.**

모집량은 TSO별 공표값을 사용하며, 입찰량·낙찰량·낙찰가격은 전원 소재지별 공표값을 사용합니다.
입찰경쟁률과 조달률은 서로 다른 지역 귀속 기준을 비교하는 참고지표입니다.
"""
        )


def render_jepx_market_placeholder() -> None:
    """JEPX 원본 연결 상태와 정규화 진단만 표시합니다."""
    st.header("JEPX Day-Ahead 현물가격 및 ESS 스프레드 분석")
    try:
        signatures = jepx_file_signatures()
        long_data, wide_data, errors, warnings, file_summary = load_jepx_data(
            str(JEPX_DATA_DIRECTORY), signatures
        )
    except Exception:
        st.error(
            "JEPX 데이터를 불러오지 못했습니다. 데이터 폴더와 파일 형식을 확인하세요."
        )
        return
    discovered = len(signatures)
    successful = (
        int(file_summary["status"].eq("정상").sum())
        if not file_summary.empty
        else 0
    )
    if long_data.empty:
        st.error(
            "JEPX 원본 파일을 읽지 못했습니다. data/jepx/raw 경로와 "
            "파일 형식을 확인해 주세요."
            if discovered
            else "배포 환경에 JEPX 데이터 파일이 없습니다. GitHub 저장소의 "
            "data/jepx/raw 경로에 지원 파일을 추가하세요."
        )
        return

    if not long_data.empty:
        tab_tokyo_chubu, tab_weekly = st.tabs([
            "도쿄·중부 분석",
            "전국 주간 모니터링",
        ])

        def spread_provider(duration: int, mode: str) -> pd.DataFrame:
            return calculate_jepx_spread_results(long_data, duration, mode)

        with tab_tokyo_chubu:
            render_jepx_tokyo_chubu_analysis(
                long_data, spread_provider, JEPX_AREA_DISPLAY
            )
        with tab_weekly:
            render_jepx_weekly_monitor(long_data, spread_provider, JEPX_AREA_DISPLAY)
        return

    date_min = long_data["delivery_date"].min() if not long_data.empty else pd.NaT
    date_max = long_data["delivery_date"].max() if not long_data.empty else pd.NaT
    date_range = (
        f"{date_min:%Y-%m-%d} ~ {date_max:%Y-%m-%d}"
        if pd.notna(date_min) and pd.notna(date_max)
        else "확인 불가"
    )
    status_values = [
        ("발견 원본 파일", f"{discovered:,}개"),
        ("정상 처리 파일", f"{successful:,}개"),
        ("데이터 날짜 범위", date_range),
        ("포함 가격 구분", f"{long_data['area'].nunique() if not long_data.empty else 0:,}개"),
        ("가격 단위", ", ".join(long_data["price_unit"].dropna().unique()) if not long_data.empty else "확인 불가"),
        ("정규화 long 데이터", f"{len(long_data):,}행"),
        ("오류", f"{len(errors):,}건"),
        ("검토 경고", f"{len(warnings):,}건"),
    ]
    for start in (0, 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, status_values[start : start + 4]):
            column.metric(label, value)

    with st.expander("JEPX 데이터 품질 및 파일 진단"):
        if file_summary.empty:
            st.info("표시할 파일 진단이 없습니다.")
        else:
            st.dataframe(file_summary, width="stretch")
        if not errors.empty:
            st.markdown("**오류 상세**")
            st.dataframe(errors.head(20), width="stretch")
        if not warnings.empty:
            st.markdown("**검토 경고 상세**")
            st.dataframe(warnings.head(20), width="stretch")

    with st.expander("JEPX 정규화 데이터 샘플"):
        if not long_data.empty:
            st.markdown("**분석용 long 데이터 첫 20행**")
            st.dataframe(long_data.head(20), width="stretch")
        if not wide_data.empty:
            st.markdown("**원본 보존형 wide 데이터 첫 20행**")
            st.dataframe(wide_data.head(20), width="stretch")
        if long_data.empty:
            st.info("표시할 정규화 데이터가 없습니다.")

    if not long_data.empty:
        st.subheader("스프레드 계산 검증")
        default_results = calculate_jepx_spread_results(long_data)
        calculated = default_results["calculation_status"].eq("Calculated")
        validation_metrics = [
            ("분석 가능 날짜", f"{long_data['delivery_date'].nunique():,}일"),
            ("분석 가능 가격 구분", f"{long_data['area'].nunique():,}개"),
            ("1시간 결과", f"{(calculated & default_results['duration_hours'].eq(1)).sum():,}건"),
            ("2시간 결과", f"{(calculated & default_results['duration_hours'].eq(2)).sum():,}건"),
            ("4시간 결과", f"{(calculated & default_results['duration_hours'].eq(4)).sum():,}건"),
            ("불완전 데이터", f"{default_results['completeness_flag'].ne('Complete').sum():,}건"),
            ("계산 실패", f"{default_results['calculation_status'].ne('Calculated').sum():,}건"),
            ("기본 계산방식", "기존 NEM 방식"),
        ]
        for start in (0, 4):
            columns = st.columns(4)
            for column, (label, value) in zip(columns, validation_metrics[start : start + 4]):
                column.metric(label, value)

        dates = sorted(long_data["delivery_date"].dropna().unique())
        areas = sorted(long_data["area"].dropna().unique())
        controls = st.columns(4)
        selected_date = controls[0].selectbox(
            "검증 날짜", dates, format_func=lambda value: f"{pd.Timestamp(value):%Y-%m-%d}", key="jepx_spread_date"
        )
        selected_area = controls[1].selectbox(
            "가격 구분", areas, format_func=lambda value: JEPX_AREA_DISPLAY.get(value, value), key="jepx_spread_area"
        )
        duration = controls[2].selectbox(
            "운용시간", [1, 2, 4], format_func=lambda value: f"{value}시간", key="jepx_spread_duration"
        )
        mode_labels = {
            "기존 NEM 방식": "nem_best_case",
            "순서 무관 연속구간": "unconstrained",
            "충전 후 방전 연속구간": "charge_before_discharge",
        }
        mode_label = controls[3].selectbox(
            "계산방식", list(mode_labels), key="jepx_spread_operation_mode"
        )
        selected_profile = long_data.loc[
            long_data["delivery_date"].eq(pd.Timestamp(selected_date))
            & long_data["area"].eq(selected_area)
        ].sort_values("period_no")
        result = calculate_daily_spread(
            selected_profile, duration, mode_labels[mode_label]
        )
        result_display = pd.DataFrame([result]).rename(columns={
            "charge_start": "충전 시작", "charge_end": "충전 종료",
            "charge_average_price": "충전 평균가격", "discharge_start": "방전 시작",
            "discharge_end": "방전 종료", "discharge_average_price": "방전 평균가격",
            "spread": "스프레드", "completeness_flag": "데이터 완전성",
        })
        st.dataframe(result_display[[
            "충전 시작", "충전 종료", "충전 평균가격", "방전 시작", "방전 종료",
            "방전 평균가격", "스프레드", "데이터 완전성",
        ]].style.format({
            "충전 평균가격": "{:,.2f}", "방전 평균가격": "{:,.2f}", "스프레드": "{:,.2f}",
        }, na_rep="계산 불가"), width="stretch")

        if result["calculation_status"] == "Calculated":
            if result["window_type"] == "non_contiguous":
                st.caption(
                    "기존 NEM 방식은 연속 구간이 아니라 하루 중 선택된 30분 코마를 사용합니다. "
                    "아래 시작·종료 시각은 선택 코마 전체의 범위입니다."
                )
            st.markdown(
                f"- **충전 구간:** {result['charge_start']}~{result['charge_end']}  "
                f"\n- **평균 충전가격:** {result['charge_average_price']:,.2f}엔/kWh  "
                f"\n- **방전 구간:** {result['discharge_start']}~{result['discharge_end']}  "
                f"\n- **평균 방전가격:** {result['discharge_average_price']:,.2f}엔/kWh  "
                f"\n- **이론적 가격 스프레드:** {result['spread']:,.2f}엔/kWh"
            )
            if result["window_type"] == "non_contiguous":
                st.caption(
                    f"충전 선택 코마: {', '.join(map(str, result['charge_periods']))} · "
                    f"방전 선택 코마: {', '.join(map(str, result['discharge_periods']))}"
                )
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=selected_profile["period_start"], y=selected_profile["price"],
                mode="lines+markers", name="30분 가격",
            ))
            for periods, name, color in (
                (result["charge_periods"], "충전 선택", "#1f77b4"),
                (result["discharge_periods"], "방전 선택", "#d62728"),
            ):
                marked = selected_profile[selected_profile["period_no"].isin(periods)]
                figure.add_trace(go.Scatter(
                    x=marked["period_start"], y=marked["price"], mode="markers",
                    marker={"size": 11, "color": color}, name=name,
                ))
            figure.update_layout(
                title="선택일 가격 및 최적 충·방전 구간",
                xaxis_title="시간", yaxis_title="가격 (엔/kWh)", hovermode="x unified",
            )
            st.plotly_chart(figure, width="stretch")
        else:
            st.warning(result["warning_message"])
        st.info(
            "이 결과는 배터리 효율, 수수료, 계통비용 및 열화비용을 반영하지 않은 "
            "단순 가격 스프레드입니다. 실제 ESS 순수익을 의미하지 않습니다."
        )
        with st.expander("선택 날짜·가격 구분의 30분 원가격"):
            st.dataframe(
                selected_profile[["period_no", "period_start", "price", "price_unit"]],
                width="stretch",
            )


st.title("일본 전력시장 모니터링")
st.caption("EPRX 조정력시장 및 JEPX 현물시장 분석")
st.markdown(
    """
<style>
.st-key-market_selector_container { margin: 0.35rem 0 1rem; }
.st-key-market_selector_container [data-testid="stSegmentedControl"] { width: 100%; }
.st-key-market_selector_container [data-testid="stSegmentedControl"] > div {
  display: flex; width: 100%; gap: 0.5rem;
}
.st-key-market_selector_container [data-testid="stSegmentedControl"] button {
  flex: 1 1 50%; min-height: 48px; padding: 0.65rem 1rem;
  border-radius: 9px; font-size: 1.05rem; font-weight: 600;
}
.st-key-market_selector_container [data-testid="stSegmentedControl"] button[aria-pressed="true"],
.st-key-market_selector_container [data-testid="stSegmentedControl"] button[aria-checked="true"] {
  background: #d62728; border-color: #d62728; color: #fff;
}
@media (max-width: 520px) {
  .st-key-market_selector_container [data-testid="stSegmentedControl"] button {
    min-height: 44px; padding: 0.5rem 0.35rem; font-size: 0.92rem;
  }
}
</style>
""",
    unsafe_allow_html=True,
)
with st.container(key="market_selector_container"):
    selected_market = st.segmented_control(
        "분석 시장 선택",
        ["EPRX 조정력시장", "JEPX 현물시장"],
        default="EPRX 조정력시장",
        key="market_selector",
        width="stretch",
    )

if selected_market == "JEPX 현물시장":
    render_jepx_market_placeholder()
    st.stop()

st.header("EPRX 1차 조정력 주간 모니터")

with st.sidebar.expander("EPRX 데이터 업데이트"):
    approval_enabled = automation_approved()
    st.write(
        "자동수집 승인 상태: "
        + ("승인 환경 활성화" if approval_enabled else "비활성화")
    )
    st.caption(
        "이 설정은 사용자가 EPRX 이용 승인을 받았음을 앱에 알리는 용도이며, "
        "공식 승인 자체를 대신하지 않습니다."
    )
    if "eprx_update_running" not in st.session_state:
        st.session_state.eprx_update_running = False
    check_clicked = st.button(
        "신규 파일 확인",
        disabled=st.session_state.eprx_update_running,
        width="stretch",
    )
    confirm_test_download = st.checkbox(
        "EPRX 이용 승인을 받았으며, 최신 후보 파일 1개를 시험 다운로드합니다."
    )
    test_download_clicked = st.button(
        "시험 다운로드 1개",
        disabled=st.session_state.eprx_update_running,
        width="stretch",
    )
    st.button(
        "신규 파일 다운로드 및 반영",
        disabled=True,
        help="시험 다운로드와 파싱 성공 후 사용 가능하도록 준비 중입니다.",
        width="stretch",
    )
    if test_download_clicked and not confirm_test_download:
        st.warning("시험 다운로드 확인 항목을 먼저 선택하세요.")
    if check_clicked or (test_download_clicked and confirm_test_download):
        st.session_state.eprx_update_running = True
        try:
            if test_download_clicked:
                st.info(
                    "EPRX 공식 거래실적 페이지에서 최신 1차 조정력 후보 1개를 "
                    "시험 다운로드하고 파싱을 확인합니다."
                )
            with st.spinner("EPRX 거래실적 페이지를 확인하고 있습니다..."):
                update_result = update_eprx_files(
                    dry_run=check_clicked,
                    max_downloads=1,
                    test_mode=test_download_clicked,
                )
            st.session_state.eprx_update_result = update_result
            downloads = update_result.get("download_results", pd.DataFrame())
            if (
                test_download_clicked
                and isinstance(downloads, pd.DataFrame)
                and not downloads.empty
                and downloads["parse_status"].eq("파싱 성공").any()
            ):
                st.cache_data.clear()
        finally:
            st.session_state.eprx_update_running = False
    if "eprx_update_result" in st.session_state:
        show_update_result(st.session_state.eprx_update_result)

data_source = st.sidebar.selectbox(
    "데이터 소스",
    [DATA_SOURCE_ACTUAL, DATA_SOURCE_SAMPLE],
    index=0,
    key="eprx_data_source",
)

file_summary = pd.DataFrame()
error_data = pd.DataFrame()
if data_source == DATA_SOURCE_ACTUAL:
    try:
        signatures = actual_file_signatures()
        data, error_data, file_summary = load_actual_data(
            str(DATA_DIRECTORY), signatures
        )
    except Exception as exc:
        st.error(
            "EPRX 데이터를 불러오지 못했습니다. 데이터 폴더와 파일 형식을 확인하세요."
        )
        with st.expander("EPRX 데이터 로딩 상세 오류"):
            st.code(f"{type(exc).__name__}: {exc}")
        st.stop()
    if not signatures:
        st.error(
            "배포 환경에 EPRX 데이터 파일이 없습니다. GitHub 저장소의 "
            "data/eprx 경로에 지원 파일을 추가하거나 왼쪽의 데이터 업데이트 "
            "기능을 사용하세요."
        )
        show_deployment_file_diagnostics("EPRX", DATA_DIRECTORY, signatures)
        show_diagnostics(file_summary, error_data)
        st.stop()
    failed = file_summary.loc[~file_summary["success"].fillna(False)]
    if data.empty:
        st.error("실제 EPRX 파일을 정규화하지 못했습니다. 아래 진단을 확인하세요.")
        show_diagnostics(file_summary, error_data)
        st.stop()
    if not failed.empty:
        st.error("일부 실제 파일을 처리하지 못했습니다. 진단 영역을 확인하세요.")

    data = add_week_columns(data)
    price_units = ", ".join(
        display_price_unit(str(unit))
        for unit in data["price_unit"].dropna().unique()
    ) or "확인 불가"
    volume_units = (
        ", ".join(map(str, data["volume_unit"].dropna().unique())) or "확인 불가"
    )
    statuses = (
        ", ".join(map(str, data["source_status"].dropna().unique())) or "확인 불가"
    )
    statuses = statuses.replace("速報値", "속보치").replace("確報値", "확정치")
else:
    data = load_sample_data()
    price_units = "샘플 가격 단위"
    volume_units = "MW"
    st.warning(
        "현재 가상 샘플 데이터를 사용하고 있습니다. 실제 EPRX 거래 결과가 아닙니다."
    )

week_days = data.groupby("week_start")["delivery_date"].nunique().to_dict()
week_options = sorted(data["week_start"].drop_duplicates(), reverse=True)
if not week_options:
    st.warning("표시할 분석 주차가 없습니다.")
    if data_source == DATA_SOURCE_ACTUAL:
        show_diagnostics(file_summary, error_data)
    st.stop()

selected_week = st.selectbox(
    "분석 주차",
    week_options,
    index=next(
        (index for index, week in enumerate(week_options) if week_days[week] == 7),
        0,
    ),
    key="eprx_analysis_week",
    format_func=lambda value: (
        f"{pd.Timestamp(value):%Y-%m-%d} ~ "
        f"{pd.Timestamp(value) + pd.Timedelta(days=6):%Y-%m-%d} "
        f"({'데이터 완전' if week_days[value] == 7 else f'데이터 불완전: {week_days[value]}/7일'})"
    ),
)
st.caption(ANALYSIS_WEEK_HELP_TEXT)

if week_days[selected_week] < 7:
    st.warning(
        f"선택 주차는 {week_days[selected_week]}/7일만 포함합니다. "
        "누락 데이터를 임의로 보완하지 않습니다."
    )

regional_profile = create_regional_weekly_profile(data, selected_week)
if regional_profile.empty:
    st.warning("선택한 주차에 데이터가 없습니다.")
    if data_source == DATA_SOURCE_ACTUAL:
        show_diagnostics(file_summary, error_data)
    st.stop()

incomplete = regional_profile.loc[regional_profile["observation_count"] < 7]
if not incomplete.empty:
    st.warning(
        f"7개 미만 관측값을 가진 지역·시간대가 {len(incomplete)}개 있습니다."
    )

render_regional_analysis(
    st,
    data,
    selected_week,
    week_days[selected_week],
    price_units,
)

show_eprx_source_information(
    data_source, data, file_summary, price_units, volume_units
)
if data_source == DATA_SOURCE_ACTUAL:
    show_diagnostics(
        file_summary,
        error_data,
        regional_profile,
        title="EPRX 데이터 품질 및 파일 진단",
        deployment_directory=DATA_DIRECTORY,
        deployment_signatures=signatures,
    )
