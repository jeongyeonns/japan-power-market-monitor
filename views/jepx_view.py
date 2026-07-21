"""JEPX 주간 모니터링 화면."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.jepx_charts import (
    create_area_spread_bar_chart, create_charge_discharge_price_chart,
    create_daily_spread_chart, create_time_frequency_chart,
    create_weekly_price_profile_chart,
)
from utils.jepx_weekly_analysis import (
    calculate_charge_discharge_time_frequency, calculate_week_over_week,
    calculate_weekly_area_kpis, compare_area_price_series,
    create_area_price_spread_comparison, filter_weekly_spreads,
    format_spread_change, initial_daily_spread_areas,
    order_daily_spread_areas, resolve_daily_spread_areas,
    sort_week_over_week_by_absolute_change,
)

MODE_LABELS = {
    "기존 NEM 방식": "nem_best_case",
    "순서 무관": "unconstrained",
    "충전 후 방전": "charge_before_discharge",
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

    table = comparison.copy()
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
