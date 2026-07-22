"""도쿄·중부 상세분석용 Plotly 그래프."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

AREA_NAMES = {"Tokyo": "도쿄", "Chubu": "중부"}
AREA_COLORS = {"도쿄": "#1f77b4", "중부": "#d62728"}
PERIOD_ORDER = [
    f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in (0, 30)
]


def _display_data(profile: pd.DataFrame, areas: list[str]) -> pd.DataFrame:
    data = profile.loc[profile["area"].isin(areas)].copy()
    data["지역"] = data["area"].map(AREA_NAMES)
    return data


def area_price_chart(
    profile: pd.DataFrame, areas: list[str], price_unit: str
) -> go.Figure:
    data = _display_data(profile, areas)
    figure = px.line(
        data,
        x="period_start",
        y="avg_price",
        color="지역",
        custom_data=["awarded_volume"],
        category_orders={"period_start": PERIOD_ORDER},
        color_discrete_map=AREA_COLORS,
        title="도쿄·중부 시간대별 평균 낙찰가격 (전원 소재지별)",
        labels={
            "period_start": "시간대",
            "avg_price": f"평균 낙찰가격 (전원 소재지별, {price_unit})",
            "지역": "지역",
        },
    )
    figure.update_traces(
        hovertemplate=(
            "지역=%{fullData.name}<br>시간대=%{x}<br>"
            f"평균 낙찰가격(전원 소재지별)=%{{y:,.2f}} {price_unit}<br>"
            "낙찰량(전원 소재지별)=%{customdata[0]:,.2f} MW<extra></extra>"
        )
    )
    return _finish(figure, f"평균 낙찰가격 (전원 소재지별, {price_unit})")


def area_price_range_chart(
    profile: pd.DataFrame, areas: list[str], price_unit: str
) -> go.Figure:
    data = _display_data(profile, areas)
    figure = px.line(
        data,
        x="period_start",
        y="price_range",
        color="지역",
        custom_data=["max_price", "avg_price", "min_price"],
        category_orders={"period_start": PERIOD_ORDER},
        color_discrete_map=AREA_COLORS,
        title="도쿄·중부 시간대별 낙찰가격 범위 (전원 소재지별)",
        labels={
            "period_start": "시간대",
            "price_range": f"가격 범위 (전원 소재지별, {price_unit})",
            "지역": "지역",
        },
    )
    figure.update_traces(
        hovertemplate=(
            "지역=%{fullData.name}<br>시간대=%{x}<br>"
            "최고 낙찰가격(전원 소재지별)=%{customdata[0]:,.2f}<br>"
            "평균 낙찰가격(전원 소재지별)=%{customdata[1]:,.2f}<br>"
            "최저 낙찰가격(전원 소재지별)=%{customdata[2]:,.2f}<br>"
            "가격 범위=%{y:,.2f}<extra></extra>"
        )
    )
    return _finish(figure, f"가격 범위 (전원 소재지별, {price_unit})")


def area_volume_chart(profile: pd.DataFrame, area: str) -> go.Figure:
    data = profile.loc[profile["area"].eq(area)].copy()
    long_data = data.melt(
        id_vars=["period_start"],
        value_vars=["procurement_volume", "bid_volume", "awarded_volume"],
        var_name="volume_type",
        value_name="volume",
    )
    names = {
        "procurement_volume": "모집량 (TSO별)",
        "bid_volume": "입찰량 (전원 소재지별)",
        "awarded_volume": "낙찰량 (전원 소재지별)",
    }
    long_data["물량 구분"] = long_data["volume_type"].map(names)
    figure = px.line(
        long_data,
        x="period_start",
        y="volume",
        color="물량 구분",
        category_orders={"period_start": PERIOD_ORDER},
        title=f"{AREA_NAMES[area]} 물량 프로파일",
        labels={
            "period_start": "시간대",
            "volume": "물량 (MW)",
            "물량 구분": "물량 구분",
        },
    )
    return _finish(figure, "물량 (MW)")


def area_bid_coverage_chart(
    profile: pd.DataFrame, areas: list[str]
) -> go.Figure:
    data = _display_data(profile, areas)
    figure = px.line(
        data,
        x="period_start",
        y="bid_coverage_ratio",
        color="지역",
        category_orders={"period_start": PERIOD_ORDER},
        color_discrete_map=AREA_COLORS,
        title="도쿄·중부 시간대별 입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량)",
        labels={
            "period_start": "시간대",
            "bid_coverage_ratio": "입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량, 배)",
            "지역": "지역",
        },
    )
    below = data.loc[data["bid_coverage_ratio"] < 1]
    for area, group in below.groupby("지역"):
        figure.add_scatter(
            x=group["period_start"],
            y=group["bid_coverage_ratio"],
            mode="markers",
            marker={"size": 7, "symbol": "circle-open", "color": AREA_COLORS[area]},
            name=f"{area} 1.0배 미만",
            showlegend=False,
            hovertemplate=f"지역={area}<br>시간대=%{{x}}<br>입찰경쟁률=%{{y:.2f}}배<extra></extra>",
        )
    figure.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="모집량 충족 기준 (1.0배)",
        annotation_position="top left",
    )
    return _finish(figure, "입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량, 배)")


def area_procurement_rate_chart(
    profile: pd.DataFrame, areas: list[str]
) -> go.Figure:
    data = _display_data(profile, areas)
    figure = px.line(
        data,
        x="period_start",
        y="procurement_rate",
        color="지역",
        custom_data=["procurement_volume", "awarded_volume"],
        category_orders={"period_start": PERIOD_ORDER},
        color_discrete_map=AREA_COLORS,
        title="도쿄·중부 시간대별 조달률 (소재지별 낙찰량 ÷ TSO별 모집량)",
        labels={
            "period_start": "시간대",
            "procurement_rate": "조달률 (소재지별 낙찰량 ÷ TSO별 모집량)",
            "지역": "지역",
        },
    )
    figure.update_traces(
        hovertemplate=(
            "지역=%{fullData.name}<br>시간대=%{x}<br>"
            "모집량(TSO별)=%{customdata[0]:,.2f} MW<br>"
            "낙찰량(전원 소재지별)=%{customdata[1]:,.2f} MW<br>"
            "조달률=%{y:.2%}<extra></extra>"
        )
    )
    figure.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="조달률 100% 기준",
        annotation_position="top left",
    )
    figure.update_yaxes(tickformat=".0%")
    return _finish(figure, "조달률 (소재지별 낙찰량 ÷ TSO별 모집량)")


def _finish(figure: go.Figure, y_title: str) -> go.Figure:
    figure.update_xaxes(title_text="시간대 (일본 표준시)", dtick=4, tickangle=-45)
    figure.update_yaxes(title_text=y_title)
    figure.update_layout(hovermode="x unified", legend_title_text="")
    return figure
