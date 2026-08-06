"""OpenAI 요청에 넣기 전 단계의 EPRX 모집량 분석 컨텍스트.

이 모듈은 네트워크나 파일 I/O를 수행하지 않는다. 앱이 이미 로드하고
정규화한 EPRX 데이터프레임만 입력으로 받아 JSON 직렬화 가능한 값을 만든다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.eprx_periods import market_regime
from utils.regional_analysis import calculate_area_kpis
from utils.weekly_aggregation import add_week_columns, create_selected_area_weekly_profile


SUPPORTED_REGIONS = ("Tokyo", "Chubu")
REQUIRED_COLUMNS = {
    "delivery_date",
    "period_no",
    "period_start",
    "area",
    "frequency_zone",
    "procurement_volume",
    "bid_volume",
    "awarded_volume",
    "avg_price",
    "max_price",
    "min_price",
}


def _number_or_none(value: Any) -> int | float | None:
    """numpy/pandas 숫자를 JSON에 안전한 파이썬 숫자로 변환한다."""
    if pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def build_eprx_procurement_context(
    eprx_df: pd.DataFrame,
    region: str,
    week_start: Any,
) -> dict[str, Any]:
    """도쿄 또는 중부의 선택 주 1차 조정력 모집량 컨텍스트를 만든다.

    주간 평균과 시간대 프로파일은 앱의 기존 주간 집계 및 KPI 산식을
    재사용한다. 제도 변경일이 포함된 주는 3시간제와 30분제를 섞지 않고
    각각의 ``market_regime`` 구간으로 유지한다.
    """
    if region not in SUPPORTED_REGIONS:
        raise ValueError(
            f"지원하지 않는 EPRX 지역입니다: {region!r}. "
            f"허용 값: {', '.join(SUPPORTED_REGIONS)}"
        )
    if not isinstance(eprx_df, pd.DataFrame):
        raise TypeError("eprx_df는 pandas.DataFrame이어야 합니다.")

    missing = sorted(REQUIRED_COLUMNS - set(eprx_df.columns))
    if missing:
        raise ValueError(f"EPRX 분석 컨텍스트 필수 열 누락: {', '.join(missing)}")

    selected_start = pd.Timestamp(week_start).normalize()
    if pd.isna(selected_start):
        raise ValueError("week_start를 유효한 날짜로 변환할 수 없습니다.")

    prepared = add_week_columns(eprx_df)
    raw_week = prepared.loc[
        prepared["week_start"].eq(selected_start)
        & prepared["area"].eq(region)
    ].copy()
    profile = create_selected_area_weekly_profile(
        prepared, selected_start, [region]
    )
    kpis = calculate_area_kpis(profile, raw_week)

    dates = raw_week["delivery_date"].dropna()
    regimes = (
        dates.map(market_regime).drop_duplicates().tolist()
        if not dates.empty
        else []
    )
    statuses = (
        sorted(str(value) for value in raw_week["source_status"].dropna().unique())
        if "source_status" in raw_week.columns
        else []
    )

    period_profile = []
    for row in profile.itertuples(index=False):
        period_profile.append(
            {
                "market_regime": row.market_regime,
                "period_no": int(row.period_no),
                "period_start": row.period_start,
                "average_procurement_mw": _number_or_none(
                    row.procurement_volume
                ),
                "observation_days": int(row.observation_count),
                "expected_observation_days": int(
                    row.expected_observation_count
                ),
                "complete": row.data_status == "Complete",
            }
        )

    observed_days = int(dates.dt.normalize().nunique())
    return {
        "context_type": "eprx_primary_reserve_procurement",
        "region": region,
        "week_start": selected_start.date().isoformat(),
        "week_end": (selected_start + pd.Timedelta(days=6)).date().isoformat(),
        "unit": "MW",
        "weekly_average_procurement_mw": _number_or_none(
            kpis["평균 모집량 (MW)"]
        ),
        "observed_days": observed_days,
        "expected_days": 7,
        "complete_week": observed_days == 7 and bool(period_profile) and all(
            item["complete"] for item in period_profile
        ),
        "market_regimes": regimes,
        "source_statuses": statuses,
        "period_profile": period_profile,
    }
