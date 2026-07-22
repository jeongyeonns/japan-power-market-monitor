"""JEPX 주간 모니터링 화면."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from utils.jepx_charts import (
    create_area_spread_bar_chart, create_charge_discharge_price_chart,
    create_daily_spread_chart, create_time_frequency_chart,
    create_tokyo_chubu_charge_discharge_chart,
    create_tokyo_chubu_daily_spread_chart,
    create_tokyo_chubu_price_profile_chart, create_weekly_price_profile_chart,
)
from utils.jepx_weekly_analysis import (
    calculate_charge_discharge_time_frequency, calculate_week_over_week,
    calculate_tokyo_chubu_week_over_week,
    calculate_tokyo_chubu_weekly_comparison,
    calculate_weekly_area_kpis, compare_area_price_series,
    create_area_price_spread_comparison, create_tokyo_chubu_price_profile,
    filter_weekly_spreads,
    format_spread_change, initial_daily_spread_areas,
    order_daily_spread_areas, resolve_daily_spread_areas,
    sort_week_over_week_by_absolute_change,
)

MODE_LABELS = {
    "일별 최저·최고 비연속 코마 선택": "nem_best_case",
    "순서 무관": "unconstrained",
    "충전 후 방전": "charge_before_discharge",
}
MODE_DISPLAY = {value: label for label, value in MODE_LABELS.items()}
ANALYSIS_WEEK_HELP_TEXT = (
    "분석하려는 주차를 선택해 주세요. "
    "정확한 비교를 위해 ‘데이터 완전’으로 표시된 주차를 권장합니다."
)
JEPX_AREA_DISPLAY_ORDER = {
    "System": 0,
    "Hokkaido": 1,
    "Tohoku": 2,
    "Tokyo": 3,
    "Chubu": 4,
    "Hokuriku": 5,
    "Kansai": 6,
    "Shikoku": 7,
    "Chugoku": 8,
    "Kyushu": 9,
}


def render_jepx_validation(long_data, spread_results_provider, area_names):
    """일별 계산 결과를 한 건씩 확인하는 검증 화면."""
    st.subheader("스프레드 계산 검증")
    controls = st.columns(4)
    date = controls[0].selectbox("검증 날짜", sorted(long_data["delivery_date"].unique()), key="jepx_validation_date")
    area = controls[1].selectbox("가격 구분", sorted(long_data["area"].unique()),
                                format_func=lambda x: area_names.get(x, x), key="jepx_validation_area")
    duration = controls[2].selectbox("운용시간", [1, 2, 4], index=1, format_func=lambda x: f"{x}시간", key="jepx_validation_duration")
    mode_label = controls[3].selectbox("계산방식", list(MODE_LABELS), key="jepx_validation_mode")
    results = spread_results_provider(duration, MODE_LABELS[mode_label])
    result = results[results["delivery_date"].eq(pd.Timestamp(date)) & results["area"].eq(area)]
    if result.empty:
        st.warning("선택 조건의 계산 결과가 없습니다.")
        return
    row = result.iloc[0]
    st.dataframe(result, width="stretch")
    if row["calculation_status"] != "Calculated": st.warning(row["warning_message"])
    st.info("배터리 효율, 수수료, 계통비용 및 열화비용을 반영하지 않은 단순 가격 스프레드이며 실제 ESS 순수익이 아닙니다.")


def render_jepx_diagnostics(long_data, wide_data, errors, warnings, file_summary):
    """원본 로드·정규화·품질 상태를 표시합니다."""
    st.subheader("데이터 품질 및 파일 진단")
    metrics = [("정규화 long 데이터", f"{len(long_data):,}행"), ("원본 보존 wide 데이터", f"{len(wide_data):,}행"),
               ("오류", f"{len(errors):,}건"), ("검토 경고", f"{len(warnings):,}건")]
    for col, (label, value) in zip(st.columns(4), metrics): col.metric(label, value)
    st.markdown("**파일 처리 로그**"); st.dataframe(file_summary, width="stretch")
    if not errors.empty: st.markdown("**오류 상세**"); st.dataframe(errors, width="stretch")
    if not warnings.empty: st.markdown("**검토 경고 상세**"); st.dataframe(warnings, width="stretch")
    with st.expander("정규화 데이터 샘플"):
        st.dataframe(long_data.head(20), width="stretch")
        st.dataframe(wide_data.head(20), width="stretch")


def _fmt(value):
    return "계산 불가" if pd.isna(value) else f"{value:,.2f}"


def _render_metric_hierarchy_table(data: pd.DataFrame) -> None:
    """JEPX 비교표의 지표 본문과 괄호 설명을 분리해 표시합니다."""
    display = data.reset_index(names="지표")
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in display.columns)
    body = []
    for _, row in display.iterrows():
        label = str(row["지표"])
        boundary = label.find(" (")
        if boundary < 0:
            label_html = f'<span class="jepx-metric-main">{escape(label)}</span>'
        else:
            label_html = (
                f'<span class="jepx-metric-main">{escape(label[:boundary])}</span>'
                f'<span class="jepx-metric-sub">{escape(label[boundary + 1:])}</span>'
            )
        values = "".join(
            f'<td class="jepx-metric-value">{escape(str(row[column]))}</td>'
            for column in display.columns[1:]
        )
        body.append(f'<tr><td class="jepx-metric-name">{label_html}</td>{values}</tr>')
    st.markdown(
        """
<style>
.jepx-metric-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.jepx-metric-table th, .jepx-metric-table td {
  padding: 0.45rem 0.6rem; border-bottom: 1px solid rgba(128, 128, 128, 0.25);
}
.jepx-metric-table th { text-align: right; font-weight: 600; }
.jepx-metric-table th:first-child, .jepx-metric-name { text-align: left; }
.jepx-metric-value { text-align: right; font-variant-numeric: tabular-nums; }
.jepx-metric-main { color: inherit; font-size: 0.95rem; font-weight: 500; }
.jepx-metric-sub { color: #8a8f98; font-size: 0.75rem; font-weight: 400; margin-left: 0.25rem; }
@media (max-width: 700px) { .jepx-metric-sub { display: block; margin-left: 0; } }
</style>
"""
        + f'<table class="jepx-metric-table"><thead><tr>{headers}</tr></thead>'
        + f"<tbody>{''.join(body)}</tbody></table>",
        unsafe_allow_html=True,
    )


def _jepx_week_options(long_data):
    dates = pd.to_datetime(long_data["delivery_date"].dropna()).dt.normalize()
    observed = set(dates)
    weeks = sorted(
        {date - pd.Timedelta(days=date.weekday()) for date in observed}, reverse=True
    )
    complete = {
        week: all(week + pd.Timedelta(days=offset) in observed for offset in range(7))
        for week in weeks
    }
    default = next((index for index, week in enumerate(weeks) if complete[week]), 0)
    return weeks, complete, default


def _format_week_option(week, complete):
    status = "데이터 완전" if complete[week] else "일부 데이터"
    return f"{week:%Y-%m-%d} ~ {(week + pd.Timedelta(days=6)):%Y-%m-%d} ({status})"


def _most_frequent_times(frequency, area, kind):
    if frequency.empty:
        return "계산 불가", 0
    selected = frequency[
        frequency["area"].eq(area) & frequency["type"].eq(kind)
    ]
    if selected.empty:
        return "계산 불가", 0
    maximum = int(selected["selection_count"].max())
    times = sorted(
        selected.loc[
            selected["selection_count"].eq(maximum), "start_time"
        ].dropna().astype(str).str.slice(0, 5).unique()
    )
    return ", ".join(times) if times else "계산 불가", maximum


def render_jepx_tokyo_chubu_analysis(long_data, spread_results_provider, area_names):
    """기존 JEPX 계산 결과를 이용한 도쿄·중부 전용 비교 화면."""
    weeks, complete_weeks, default_week = _jepx_week_options(long_data)
    controls = st.columns(3)
    week = controls[0].selectbox(
        "분석 주차",
        weeks,
        index=default_week,
        format_func=lambda value: _format_week_option(value, complete_weeks),
        key="jepx_tokyo_chubu_week",
    )
    controls[0].caption(ANALYSIS_WEEK_HELP_TEXT)
    duration = controls[1].selectbox(
        "ESS 운용시간",
        [1, 2, 4],
        index=1,
        format_func=lambda value: f"{value}시간",
        key="jepx_tokyo_chubu_duration",
    )
    mode_label = controls[2].selectbox(
        "계산방식", list(MODE_LABELS), key="jepx_tokyo_chubu_operation_mode"
    )
    mode = MODE_LABELS[mode_label]
    all_daily = spread_results_provider(duration, mode)
    weekly = filter_weekly_spreads(all_daily, week)
    weekly = weekly[weekly["area"].isin(["Tokyo", "Chubu"])].copy()
    comparison = calculate_tokyo_chubu_weekly_comparison(
        long_data, all_daily, week, duration, mode
    )
    wow = calculate_tokyo_chubu_week_over_week(
        long_data, all_daily, week, duration, mode
    )

    st.subheader("도쿄·중부 주간 핵심지표 비교")
    if comparison.empty:
        st.info("선택 조건에서 도쿄·중부 비교 결과를 계산할 수 없습니다.")
        return
    rows = [
        ("주간 평균 전력가격 (엔/kWh)", "average_market_price", "number"),
        ("주간 최저 전력가격 (엔/kWh)", "minimum_market_price", "number"),
        ("주간 최고 전력가격 (엔/kWh)", "maximum_market_price", "number"),
        ("평균 충전가격 (엔/kWh)", "average_charge_price", "number"),
        ("평균 방전가격 (엔/kWh)", "average_discharge_price", "number"),
        ("평균 ESS 스프레드 (엔/kWh)", "average_spread", "number"),
        ("최대 스프레드 발생일", "maximum_spread_date", "date"),
        ("전주 대비 평균 스프레드 변화 (엔/kWh)", "week_over_week_change", "change"),
    ]
    core = pd.DataFrame(index=[label for label, _, _ in rows])
    for area in ("Tokyo", "Chubu"):
        selected = comparison[comparison["area"].eq(area)]
        values = []
        for _, column, kind in rows:
            value = selected.iloc[0][column] if not selected.empty else np.nan
            if kind == "date":
                value = "계산 불가" if pd.isna(value) else f"{pd.Timestamp(value):%Y-%m-%d}"
            elif kind == "integer":
                value = "계산 불가" if pd.isna(value) else f"{int(value)}"
            elif kind == "change":
                value = format_spread_change(value)
            else:
                value = _fmt(value)
            values.append(value)
        core[area_names.get(area, area)] = values
    _render_metric_hierarchy_table(core)

    incomplete = weekly[
        weekly["completeness_flag"].ne("Complete")
        | weekly["calculation_status"].ne("Calculated")
    ]
    if not incomplete.empty:
        st.warning(
            f"선택 주차에서 불완전하거나 계산하지 못한 도쿄·중부 결과가 "
            f"{len(incomplete):,}건 있습니다. 해당 결과는 주간 KPI에서 제외됩니다."
        )
    if not complete_weeks.get(week, False):
        observed_days = int(pd.to_datetime(weekly["delivery_date"]).dt.normalize().nunique())
        st.warning(
            f"선택 주차는 일부 데이터만 포함합니다 ({observed_days}/7일). "
            "현재 확보된 완전한 일별 결과만 KPI에 반영합니다."
        )

    st.plotly_chart(
        create_tokyo_chubu_daily_spread_chart(weekly, area_names), width="stretch"
    )
    profile = create_tokyo_chubu_price_profile(long_data, week)
    st.plotly_chart(
        create_tokyo_chubu_price_profile_chart(profile, area_names), width="stretch"
    )
    st.caption(
        "이 그래프는 선택 주차의 시간대별 평균 가격 패턴을 보여주는 참고 그래프이며, "
        "일별 최적 ESS 스프레드 계산 자체를 나타내지 않습니다."
    )
    st.plotly_chart(
        create_tokyo_chubu_charge_discharge_chart(weekly, area_names), width="stretch"
    )

    st.markdown("### 주요 충전·방전 시간대 비교")
    frequency = calculate_charge_discharge_time_frequency(weekly)
    frequency_rows = []
    for area in ("Tokyo", "Chubu"):
        charge_time, charge_count = _most_frequent_times(frequency, area, "충전")
        discharge_time, discharge_count = _most_frequent_times(frequency, area, "방전")
        frequency_rows.append(
            {
                "지역": area_names.get(area, area),
                "최빈 충전 시작시간": charge_time,
                "충전 선택 횟수": charge_count,
                "최빈 방전 시작시간": discharge_time,
                "방전 선택 횟수": discharge_count,
            }
        )
    st.dataframe(pd.DataFrame(frequency_rows), width="stretch", hide_index=True)

    st.markdown("### 도쿄·중부 전주 대비")
    if wow.empty:
        st.info("비교 가능한 직전 주차 데이터가 없습니다. 현재 주 값은 위 표에서 확인할 수 있습니다.")
    else:
        previous_week = week - pd.Timedelta(days=7)
        if previous_week not in complete_weeks:
            st.info("직전 주차 원본 데이터가 없어 전주 값은 계산 불가로 표시합니다.")
        elif not complete_weeks[previous_week]:
            st.info("직전 주차가 불완전합니다. 완전한 일별 결과만 전주 값에 반영합니다.")
        wow_table = wow.copy()
        wow_table["지역"] = wow_table["area"].map(area_names).fillna(wow_table["area"])
        wow_table["현재 주"] = wow_table.apply(
            lambda row: f"{int(row['current'])}" if row["metric"] == "양의 스프레드 일수" and pd.notna(row["current"]) else _fmt(row["current"]), axis=1
        )
        wow_table["전주"] = wow_table.apply(
            lambda row: f"{int(row['previous'])}" if row["metric"] == "양의 스프레드 일수" and pd.notna(row["previous"]) else _fmt(row["previous"]), axis=1
        )
        wow_table["변화"] = wow_table["change"].map(format_spread_change)
        st.dataframe(
            wow_table[["metric", "지역", "현재 주", "전주", "변화"]].rename(
                columns={"metric": "지표"}
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown("### 도쿄·중부 일별 ESS 스프레드 상세 데이터")
    detail = weekly.copy()
    detail["지역"] = detail["area"].map(area_names).fillna(detail["area"])
    detail["날짜"] = pd.to_datetime(detail["delivery_date"]).dt.strftime("%Y-%m-%d")
    detail["ESS 운용시간"] = detail["duration_hours"].map(lambda value: f"{value}시간")
    detail["계산방식"] = detail["operation_mode"].map(MODE_DISPLAY).fillna(detail["operation_mode"])
    detail["양의 스프레드 여부"] = detail["positive_spread"].map({True: "예", False: "아니오"})
    detail = detail.sort_values(
        ["delivery_date", "area"],
        key=lambda values: values.map({"Tokyo": 0, "Chubu": 1}) if values.name == "area" else values,
    )
    detail = detail.rename(
        columns={
            "charge_start": "충전 시작", "charge_end": "충전 종료",
            "charge_average_price": "충전 평균가격 (엔/kWh)",
            "discharge_start": "방전 시작", "discharge_end": "방전 종료",
            "discharge_average_price": "방전 평균가격 (엔/kWh)",
            "spread": "스프레드 (엔/kWh)", "completeness_flag": "데이터 완전성",
            "warning_message": "경고 내용",
        }
    )
    detail_columns = [
        "날짜", "지역", "ESS 운용시간", "계산방식", "충전 시작", "충전 종료",
        "충전 평균가격 (엔/kWh)", "방전 시작", "방전 종료",
        "방전 평균가격 (엔/kWh)", "스프레드 (엔/kWh)", "양의 스프레드 여부",
        "데이터 완전성", "경고 내용",
    ]
    st.dataframe(
        detail[detail_columns].style.format(
            {
                "충전 평균가격 (엔/kWh)": "{:,.2f}",
                "방전 평균가격 (엔/kWh)": "{:,.2f}",
                "스프레드 (엔/kWh)": "{:,.2f}",
            },
            na_rep="계산 불가",
        ),
        width="stretch",
        hide_index=True,
    )
    with st.expander("도쿄·중부 분석 지표 설명"):
        st.markdown(
            """
- 주간 평균·최저·최고 전력가격: 선택 주차의 해당 지역 30분 가격 전체의 평균·최솟값·최댓값
- 충전 평균가격: 일별 계산에서 선택된 충전 코마 또는 충전구간 평균가격의 주간 평균
- 방전 평균가격: 일별 계산에서 선택된 방전 코마 또는 방전구간 평균가격의 주간 평균
- ESS 스프레드: 방전 평균가격 − 충전 평균가격
- 1시간·2시간·4시간: 각각 30분 코마 2개·4개·8개
- 일별 최저·최고 비연속 코마 선택: 가장 낮은 코마를 충전에, 겹치지 않는 가장 높은 코마를 방전에 선택
- 순서 무관: 충전·방전 연속구간의 시간 순서를 제한하지 않음
- 충전 후 방전: 충전 연속구간 종료 후 방전 연속구간을 선택
- 불완전 날짜는 상세표에 보존하고 주간 KPI에서는 제외
- 전주 대비 변화: 현재 주 값 − 직전 주 값

**현재 ESS 스프레드는 배터리 효율, 수수료, 계통비용 및 열화비용을 반영하지 않은 이론적 가격 차이입니다.**
"""
        )


def render_jepx_weekly_monitor(long_data, spread_results_provider, area_names):
    st.subheader("JEPX 주간 통합지표")
    dates = pd.to_datetime(long_data["delivery_date"].dropna().unique())
    week_starts = sorted({date.normalize() - pd.Timedelta(days=date.weekday()) for date in dates}, reverse=True)
    controls = st.columns(4)
    week = controls[0].selectbox(
        "분석 주차", week_starts,
        format_func=lambda d: f"{d:%Y-%m-%d} ~ {(d + pd.Timedelta(days=6)):%Y-%m-%d} "
                              f"({'데이터 완전' if all((d + pd.Timedelta(days=i)) in set(dates) for i in range(7)) else '일부 데이터'})",
        key="jepx_week_start",
    )
    controls[0].caption(ANALYSIS_WEEK_HELP_TEXT)
    areas = sorted(long_data["area"].dropna().unique())
    area_label = controls[1].selectbox("분석 지역", ["전체 지역 비교"] + areas,
        format_func=lambda x: x if x == "전체 지역 비교" else area_names.get(x, x), key="jepx_weekly_area")
    duration = controls[2].selectbox("ESS 운용시간", [1, 2, 4], index=1,
        format_func=lambda x: f"{x}시간", key="jepx_weekly_duration")
    mode_label = controls[3].selectbox("계산방식", list(MODE_LABELS), key="jepx_weekly_mode")
    mode = MODE_LABELS[mode_label]
    all_daily = spread_results_provider(duration, mode)
    weekly = filter_weekly_spreads(all_daily, week)
    previous = filter_weekly_spreads(all_daily, week - pd.Timedelta(days=7))
    if area_label != "전체 지역 비교":
        weekly = weekly[weekly["area"].eq(area_label)]
        previous = previous[previous["area"].eq(area_label)]
    kpis = calculate_weekly_area_kpis(weekly)
    previous_kpis = calculate_weekly_area_kpis(previous)
    wow = calculate_week_over_week(kpis, previous_kpis) if not kpis.empty else pd.DataFrame()
    comparison = create_area_price_spread_comparison(
        long_data, all_daily, week, duration, mode
    )
    if area_label != "전체 지역 비교":
        comparison = comparison[comparison["area"].eq(area_label)].copy()

    expected_dates = {week + pd.Timedelta(days=index) for index in range(7)}
    observed_dates = set(pd.to_datetime(weekly["delivery_date"]).dt.normalize())
    missing_dates = len(expected_dates - observed_dates)
    incomplete = int(weekly["completeness_flag"].ne("Complete").sum())
    failed = int(weekly["calculation_status"].ne("Calculated").sum())
    if missing_dates or incomplete or failed:
        affected = weekly.loc[
            weekly["completeness_flag"].ne("Complete")
            | weekly["calculation_status"].ne("Calculated"), "area"
        ].dropna().map(area_names).unique().tolist()
        affected_text = ", ".join(affected) if affected else "주차 전체"
        st.warning(f"**확인이 필요한 데이터가 있습니다.**\n\n- 불완전 날짜: **{missing_dates}일**\n- 제외된 결과: **{max(incomplete, failed)}건**\n- 영향을 받은 지역: **{affected_text}**\n\n완전한 데이터만 집계에 반영했습니다.")

    if kpis.empty or kpis["average_spread"].isna().all():
        st.info("선택 조건에서 계산 가능한 완전 데이터가 없습니다.")
        return
    high = comparison.iloc[0]
    peak = comparison.loc[comparison["maximum_spread"].idxmax()]
    st.markdown("### 이번 주 JEPX 핵심 요약")
    st.markdown(
        f"- **평균 스프레드 최고 지역:** {area_names.get(high['area'], high['area'])}, "
        f"**{high['average_spread']:.2f}엔/kWh**\n"
        f"- **주간 최대 스프레드:** {area_names.get(peak['area'], peak['area'])}에서 "
        f"**{peak['maximum_spread']:.2f}엔/kWh**\n"
        + (f"- **전주 대비:** {int(wow['average_spread_change'].gt(0).sum())}개 상승, "
           f"{int(wow['average_spread_change'].lt(0).sum())}개 하락"
           if not wow.empty and wow["average_spread_previous"].notna().any()
           else "- **전주 대비:** 비교 가능한 데이터가 없습니다.")
    )

    table = comparison.assign(
        _area_order=comparison["area"].map(JEPX_AREA_DISPLAY_ORDER),
        _area_name=comparison["area"].astype(str),
    ).sort_values(
        ["_area_order", "_area_name"],
        ascending=[True, True],
        na_position="last",
    ).drop(columns=["_area_order", "_area_name"])
    table["지역"] = table["area"].map(area_names).fillna(table["area"])
    table["전주 대비 변화 (엔/kWh)"] = table["week_over_week_change"].map(format_spread_change)
    table["최대 스프레드 발생일"] = pd.to_datetime(table["maximum_spread_date"]).dt.strftime("%Y-%m-%d").fillna("계산 불가")
    table = table.rename(columns={
        "average_market_price": "주간 평균 전력가격 (엔/kWh)",
        "minimum_market_price": "주간 최저 전력가격 (엔/kWh)",
        "maximum_market_price": "주간 최고 전력가격 (엔/kWh)",
        "average_charge_price": "평균 충전가격 (엔/kWh)",
        "average_discharge_price": "평균 방전가격 (엔/kWh)",
        "average_spread": "평균 ESS 스프레드 (엔/kWh)",
        "maximum_spread": "최대 ESS 스프레드 (엔/kWh)",
        "complete_days": "완전 데이터 일수",
    })
    columns = ["지역", "주간 평균 전력가격 (엔/kWh)", "주간 최저 전력가격 (엔/kWh)",
               "주간 최고 전력가격 (엔/kWh)", "평균 충전가격 (엔/kWh)", "평균 방전가격 (엔/kWh)",
               "평균 ESS 스프레드 (엔/kWh)", "최대 ESS 스프레드 (엔/kWh)", "최대 스프레드 발생일",
               "전주 대비 변화 (엔/kWh)", "완전 데이터 일수"]
    st.markdown("### 지역별 전력가격 및 ESS 스프레드 비교" if area_label == "전체 지역 비교" else "### 선택 지역 주간 가격 및 스프레드")
    numeric_formats = {column: "{:,.2f}" for column in columns if "(엔/kWh)" in column and column != "전주 대비 변화 (엔/kWh)"}
    styled_table = table[columns].style.format(numeric_formats, na_rep="계산 불가")
    if area_label == "전체 지역 비교":
        styled_table = styled_table.highlight_max(
            subset=["평균 ESS 스프레드 (엔/kWh)", "최대 ESS 스프레드 (엔/kWh)"],
            props="font-weight: 600; background-color: #fff4cc;",
        )
    st.dataframe(styled_table, width="stretch", hide_index=True)
    st.caption("평균 ESS 스프레드는 각 날짜의 최적 충전·방전 가격 차이를 주간 평균한 값입니다.")
    st.caption("현재 스프레드는 배터리 효율, 수수료, 계통비용 및 열화비용을 반영하지 않은 이론적 가격 차이입니다.")
    if not (missing_dates or incomplete or failed):
        st.caption("모든 지역의 데이터가 완전합니다.")

    metrics = [
        ("평균 스프레드 최고 지역", area_names.get(high["area"], high["area"])),
        ("해당 평균 스프레드", f"{high['average_spread']:.2f} 엔/kWh"),
        ("주간 최대 스프레드", f"{peak['maximum_spread']:.2f} 엔/kWh"),
        ("최대 발생 지역·날짜", f"{area_names.get(peak['area'], peak['area'])} · {peak['maximum_spread_date']:%Y-%m-%d}"),
    ]
    for col, (label, value) in zip(st.columns(4), metrics):
        col.metric(label, value)

    if area_label == "전체 지역 비교":
        st.plotly_chart(create_area_spread_bar_chart(kpis, area_names), width="stretch")
        available_chart_areas = order_daily_spread_areas(kpis["area"].tolist())
        selection_key = "jepx_daily_spread_selected_areas"
        if selection_key not in st.session_state:
            st.session_state[selection_key] = initial_daily_spread_areas(
                available_chart_areas, area_label
            )
        else:
            st.session_state[selection_key] = resolve_daily_spread_areas(
                st.session_state[selection_key], available_chart_areas, area_label
            )
        show_all = st.checkbox(
            "전체 지역 표시", key="jepx_show_all_daily_spread_areas"
        )
        selected_chart_areas = st.multiselect(
            "일별 추이 표시 지역",
            available_chart_areas,
            format_func=lambda x: area_names.get(x, x),
            key=selection_key,
        )
        chart_areas = available_chart_areas if show_all else selected_chart_areas
        if show_all:
            st.caption(
                "전체 지역이 표시되고 있습니다. 범례를 클릭하면 특정 지역을 "
                "숨기거나 다시 표시할 수 있습니다."
            )
        daily_chart_data = weekly[weekly["area"].isin(chart_areas)]
    else:
        chart_areas = [area_label]
        daily_chart_data = weekly
    if daily_chart_data.empty:
        st.info("표시할 지역을 하나 이상 선택하세요.")
    else:
        st.plotly_chart(create_daily_spread_chart(daily_chart_data, area_names), width="stretch")

    profile_areas = areas[:4] if area_label == "전체 지역 비교" else [area_label]
    if area_label == "전체 지역 비교":
        profile_areas = st.multiselect("가격 프로파일 표시 지역", areas, default=profile_areas,
                                       format_func=lambda x: area_names.get(x, x), max_selections=4, key="jepx_profile_areas")
    raw_week = long_data[pd.to_datetime(long_data["delivery_date"]).between(week, week + pd.Timedelta(days=6))]
    raw_week = raw_week[raw_week["area"].isin(profile_areas)]
    profile = raw_week.groupby(["area", "period_no", "period_start"], as_index=False).agg(
        mean_price=("price", "mean"), min_price=("price", "min"), max_price=("price", "max"),
        observation_days=("delivery_date", "nunique"))
    st.plotly_chart(create_weekly_price_profile_chart(profile, area_names), width="stretch")
    st.caption("이 차트는 선택 주차의 시간대별 평균 가격 패턴을 보여주며, 일별 최적 스프레드를 직접 계산하는 차트가 아닙니다.")

    selected_daily = weekly if area_label != "전체 지역 비교" else weekly[weekly["area"].isin(chart_areas[:1])]
    if not selected_daily.empty:
        st.plotly_chart(create_charge_discharge_price_chart(selected_daily), width="stretch")
        frequency = calculate_charge_discharge_time_frequency(selected_daily)
        st.plotly_chart(create_time_frequency_chart(frequency), width="stretch")
        for kind in ("충전", "방전"):
            part = frequency[frequency["type"].eq(kind)]
            if not part.empty:
                maximum = part["selection_count"].max(); times = sorted(part.loc[part["selection_count"].eq(maximum), "start_time"].astype(str).unique())
                st.caption(f"가장 자주 선택된 {kind} 시작시간: {', '.join(times)} ({maximum}회)")

    st.markdown("### 전주 대비")
    st.caption("전주 대비 평균 스프레드 변화량의 절댓값이 큰 순서로 정렬됩니다.")
    if previous.empty:
        st.info("비교 가능한 직전 주차 데이터가 없습니다.")
    else:
        wow_display = sort_week_over_week_by_absolute_change(wow)
        wow_display["area"] = wow_display["area"].map(area_names).fillna(wow_display["area"])
        wow_display["average_spread_change_display"] = wow_display["average_spread_change"].map(format_spread_change)
        wow_display.loc[wow_display["average_spread_change"].isna(), "direction"] = "비교 불가"
        st.dataframe(wow_display[["area", "average_spread_current", "average_spread_previous", "average_spread_change", "average_spread_change_pct", "direction"]]
            .assign(average_spread_change=wow_display["average_spread_change_display"])
            .rename(columns={"area":"지역", "average_spread_current":"현재 주 평균", "average_spread_previous":"전주 평균", "average_spread_change":"변화량", "average_spread_change_pct":"변화율(%)", "direction":"방향"})
            .style.format({"현재 주 평균":"{:,.2f}", "전주 평균":"{:,.2f}", "변화율(%)":"{:,.2f}"}, na_rep="계산 불가"), width="stretch", hide_index=True)

    with st.expander("지역별 가격 동일 여부"):
        comparison = compare_area_price_series(long_data, week)
        comparison["area_1"] = comparison["area_1"].map(area_names).fillna(comparison["area_1"])
        comparison["area_2"] = comparison["area_2"].map(area_names).fillna(comparison["area_2"])
        st.dataframe(comparison, width="stretch")

    st.markdown("### JEPX 일별 ESS 스프레드 상세 데이터")
    detail = weekly.copy().rename(columns={"delivery_date":"날짜", "area":"지역", "duration_hours":"ESS 운용시간",
        "operation_mode":"계산방식", "charge_start":"충전 시작", "charge_end":"충전 종료", "charge_average_price":"충전 평균가격",
        "discharge_start":"방전 시작", "discharge_end":"방전 종료", "discharge_average_price":"방전 평균가격", "spread":"스프레드",
        "positive_spread":"양의 스프레드 여부", "observation_count":"관측 시간대 수", "expected_observation_count":"예상 시간대 수",
        "completeness_flag":"데이터 완전성", "warning_message":"경고 내용"})
    detail["지역"] = detail["지역"].map(area_names).fillna(detail["지역"])
    st.dataframe(detail, width="stretch")
    with st.expander("JEPX 스프레드 지표 설명"):
        st.markdown("""
- 충전 평균가격: 선택된 충전 코마 또는 연속 충전구간의 평균가격
- 방전 평균가격: 선택된 방전 코마 또는 연속 방전구간의 평균가격
- 스프레드: 방전 평균가격 − 충전 평균가격
- 1시간·2시간·4시간: 각각 30분 코마 2개·4개·8개
- 순서 무관: 충전·방전 연속구간의 순서는 제한하지 않음
- 충전 후 방전: 충전 연속구간이 끝난 뒤 방전 연속구간을 선택
- 모든 방식에서 충전·방전 코마의 중복을 금지
- 불완전한 일별 데이터는 KPI에서 제외하고 상세표에 상태를 보존

**표현된 스프레드는 시장가격 차이만 계산한 값이며, 배터리 효율, 수수료, 계통비용 및 열화비용을 반영한 실제 ESS 순수익이 아닙니다.**
""")
