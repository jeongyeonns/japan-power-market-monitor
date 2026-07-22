"""JEPX 주간 모니터링 Plotly 차트 생성 함수."""

import plotly.express as px
import plotly.graph_objects as go


def create_area_spread_bar_chart(kpis, area_names):
    data = kpis.sort_values("average_spread", ascending=False).copy()
    data["지역"] = data["area"].map(area_names).fillna(data["area"])
    return px.bar(data, x="지역", y="average_spread", title="지역별 주간 평균 ESS 스프레드",
                  labels={"average_spread": "평균 스프레드 (엔/kWh)"},
                  hover_data=["median_spread", "max_spread", "positive_spread_days", "complete_days"])


def create_daily_spread_chart(daily, area_names):
    data = daily.copy(); data["지역"] = data["area"].map(area_names).fillna(data["area"])
    figure = px.line(data, x="delivery_date", y="spread", color="지역", markers=True,
                     title="일별 ESS 스프레드 추이", labels={"delivery_date": "날짜", "spread": "스프레드 (엔/kWh)"},
                     hover_data=["charge_average_price", "discharge_average_price", "charge_start", "discharge_start", "completeness_flag"])
    figure.update_traces(line={"width": 1.8}, marker={"size": 5})
    figure.update_layout(
        height=520,
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "xanchor": "left", "x": 0},
        hovermode="x unified",
    )
    return figure


def create_weekly_price_profile_chart(profile, area_names):
    data = profile.copy(); data["지역"] = data["area"].map(area_names).fillna(data["area"])
    return px.line(data, x="period_start", y="mean_price", color="지역", markers=True,
                   title="선택 주차 시간대별 평균 가격", labels={"period_start": "시간대", "mean_price": "평균 가격 (엔/kWh)"},
                   hover_data=["min_price", "max_price", "observation_days"])


def create_charge_discharge_price_chart(daily):
    fig = go.Figure()
    fig.add_scatter(x=daily["delivery_date"], y=daily["charge_average_price"], name="충전 평균가격", mode="lines+markers")
    fig.add_scatter(x=daily["delivery_date"], y=daily["discharge_average_price"], name="방전 평균가격", mode="lines+markers")
    fig.update_layout(title="일별 평균 충전가격과 방전가격", xaxis_title="날짜", yaxis_title="가격 (엔/kWh)", hovermode="x unified")
    return fig


def create_time_frequency_chart(frequency):
    return px.bar(frequency, x="start_time", y="selection_count", color="type", barmode="group",
                  facet_row="area" if frequency["area"].nunique() > 1 else None,
                  title="주요 충전·방전 시작시간대", labels={"start_time": "시작시간", "selection_count": "선택 횟수"})


def create_tokyo_chubu_daily_spread_chart(daily, area_names):
    """도쿄·중부 일별 스프레드와 계산 근거를 함께 표시합니다."""
    data = daily[daily["area"].isin(["Tokyo", "Chubu"])].copy()
    data["지역"] = data["area"].map(area_names).fillna(data["area"])
    figure = px.line(
        data,
        x="delivery_date",
        y="spread",
        color="지역",
        markers=True,
        title="도쿄·중부 일별 ESS 스프레드 추이",
        labels={"delivery_date": "날짜", "spread": "스프레드 (엔/kWh)"},
        hover_data={
            "charge_average_price": ":.2f",
            "discharge_average_price": ":.2f",
            "charge_start": True,
            "discharge_start": True,
            "completeness_flag": True,
        },
    )
    figure.update_traces(connectgaps=False, line={"width": 2}, marker={"size": 6})
    figure.update_layout(hovermode="x unified")
    return figure


def create_tokyo_chubu_price_profile_chart(profile, area_names):
    """도쿄·중부의 선택 주차 시간대별 가격 범위를 표시합니다."""
    data = profile.copy()
    data["지역"] = data["area"].map(area_names).fillna(data["area"])
    return px.line(
        data,
        x="period_start",
        y="mean_price",
        color="지역",
        markers=True,
        title="도쿄·중부 시간대별 평균 전력가격",
        labels={"period_start": "시간대", "mean_price": "평균 전력가격 (엔/kWh)"},
        hover_data={
            "min_price": ":.2f",
            "max_price": ":.2f",
            "observation_days": True,
        },
    )


def create_tokyo_chubu_charge_discharge_chart(daily, area_names):
    """두 지역의 일별 충전·방전 평균가격을 네 개 선으로 비교합니다."""
    data = daily[daily["area"].isin(["Tokyo", "Chubu"])].copy()
    data["지역"] = data["area"].map(area_names).fillna(data["area"])
    long = data.melt(
        id_vars=["delivery_date", "지역", "spread", "charge_start", "charge_end", "discharge_start", "discharge_end"],
        value_vars=["charge_average_price", "discharge_average_price"],
        var_name="price_type",
        value_name="price",
    )
    long["가격 구분"] = long["price_type"].map(
        {"charge_average_price": "충전가격", "discharge_average_price": "방전가격"}
    )
    long["계열"] = long["지역"] + " " + long["가격 구분"]
    return px.line(
        long,
        x="delivery_date",
        y="price",
        color="계열",
        markers=True,
        title="도쿄·중부 일별 충전·방전 가격",
        labels={"delivery_date": "날짜", "price": "가격 (엔/kWh)"},
        hover_data=[
            "지역", "가격 구분", "spread", "charge_start", "charge_end",
            "discharge_start", "discharge_end",
        ],
    )
