"""전국 시장 요약용 Plotly 그래프."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PERIOD_ORDER = [
    f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in (0, 30)
]


def national_price_chart(profile: pd.DataFrame, price_unit: str) -> go.Figure:
    figure = px.line(
        profile,
        x="period_start",
        y="avg_price",
        custom_data=["awarded_volume", "area_count"],
        category_orders={"period_start": PERIOD_ORDER},
        title="전국 낙찰량 가중평균 낙찰가격 (전원 소재지별)",
        labels={
            "period_start": "시간대",
            "avg_price": f"가중평균 낙찰가격 (전원 소재지별, {price_unit})",
        },
    )
    figure.update_traces(
        name="전국 가중평균 참고값",
        hovertemplate=(
            "시간대=%{x}<br>가중평균 낙찰가격(전원 소재지별)=%{y:,.2f}<br>"
            "전국 낙찰량(전원 소재지별)=%{customdata[0]:,.2f} MW<br>"
            "포함 지역 수=%{customdata[1]}<extra></extra>"
        ),
    )
    return _finish(figure, f"가중평균 낙찰가격 (전원 소재지별, {price_unit})")


def national_volume_chart(profile: pd.DataFrame) -> go.Figure:
    data = profile.melt(
        id_vars=["period_start"],
        value_vars=["procurement_volume", "bid_volume", "awarded_volume"],
        var_name="type",
        value_name="volume",
    )
    data["물량 구분"] = data["type"].map(
        {
            "procurement_volume": "모집량 (TSO별)",
            "bid_volume": "입찰량 (전원 소재지별)",
            "awarded_volume": "낙찰량 (전원 소재지별)",
        }
    )
    figure = px.line(
        data,
        x="period_start",
        y="volume",
        color="물량 구분",
        category_orders={"period_start": PERIOD_ORDER},
        title="전국 시간대별 모집·입찰·낙찰 물량",
        labels={"period_start": "시간대", "volume": "물량 (MW)"},
    )
    return _finish(figure, "물량 (MW)")


def national_ratio_chart(
    profile: pd.DataFrame, metric: str
) -> go.Figure:
    if metric == "bid_coverage_ratio":
        title, y_title, annotation = (
            "전국 시간대별 입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량)",
            "입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량, 배)",
            "모집량 충족 기준 (1.0배)",
        )
    else:
        title, y_title, annotation = (
            "전국 시간대별 조달률 (소재지별 낙찰량 ÷ TSO별 모집량)",
            "조달률 (소재지별 낙찰량 ÷ TSO별 모집량)",
            "모집량 대비 100%",
        )
    figure = px.line(
        profile,
        x="period_start",
        y=metric,
        category_orders={"period_start": PERIOD_ORDER},
        title=title,
        labels={"period_start": "시간대", metric: y_title},
    )
    if metric == "bid_coverage_ratio":
        below = profile.loc[profile[metric] < 1]
        figure.add_scatter(
            x=below["period_start"],
            y=below[metric],
            mode="markers",
            marker={"size": 7, "symbol": "circle-open"},
            showlegend=False,
            hovertemplate="시간대=%{x}<br>입찰경쟁률=%{y:.2f}배<extra></extra>",
        )
    figure.add_hline(
        y=1,
        line_dash="dash",
        line_color="gray",
        annotation_text=annotation,
        annotation_position="top left",
    )
    if metric == "procurement_rate":
        figure.update_yaxes(tickformat=".0%")
    return _finish(figure, y_title)


def regional_volume_bar(summary: pd.DataFrame) -> go.Figure:
    data = summary.sort_values("avg_procurement_volume", ascending=False)
    figure = px.bar(
        data,
        x="area_display",
        y="avg_procurement_volume",
        color="frequency_zone",
        custom_data=["avg_bid_volume", "avg_awarded_volume"],
        title="지역별 주간 평균 모집량 비교 (TSO별)",
        labels={
            "area_display": "지역",
            "avg_procurement_volume": "평균 모집량 (TSO별, MW)",
            "frequency_zone": "주파수권역",
        },
    )
    figure.update_traces(
        hovertemplate=(
            "지역=%{x}<br>주파수권역=%{fullData.name}<br>"
            "평균 모집량(TSO별)=%{y:,.2f} MW<br>"
            "평균 입찰량(전원 소재지별)=%{customdata[0]:,.2f} MW<br>"
            "평균 낙찰량(전원 소재지별)=%{customdata[1]:,.2f} MW<extra></extra>"
        )
    )
    return figure


def regional_price_bar(summary: pd.DataFrame, price_unit: str) -> go.Figure:
    data = summary.sort_values("weighted_avg_price", ascending=False)
    return px.bar(
        data,
        x="area_display",
        y="weighted_avg_price",
        color="frequency_zone",
        title="지역별 낙찰량 가중평균 낙찰가격 참고지표 (전원 소재지별)",
        labels={
            "area_display": "지역",
            "weighted_avg_price": f"가중평균 낙찰가격 (전원 소재지별, {price_unit})",
            "frequency_zone": "주파수권역",
        },
    )


def regional_bid_bar(summary: pd.DataFrame) -> go.Figure:
    data = summary.sort_values("bid_coverage_ratio", ascending=False)
    figure = px.bar(
        data,
        x="area_display",
        y="bid_coverage_ratio",
        color="frequency_zone",
        title="지역별 입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량)",
        labels={
            "area_display": "지역",
            "bid_coverage_ratio": "입찰경쟁률 (소재지별 입찰량 ÷ TSO별 모집량, 배)",
            "frequency_zone": "주파수권역",
        },
    )
    figure.add_hline(y=1, line_dash="dash", line_color="gray")
    return figure


def _finish(figure: go.Figure, y_title: str) -> go.Figure:
    figure.update_xaxes(title_text="시간대 (일본 표준시)", dtick=4, tickangle=-45)
    figure.update_yaxes(title_text=y_title)
    figure.update_layout(hovermode="x unified", legend_title_text="")
    return figure
