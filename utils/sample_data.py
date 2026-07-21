"""실제 시장 데이터가 아닌 EPRX 1차 조정력 샘플 데이터를 생성합니다."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

AREAS_BY_ZONE = {
    # TODO: 향후 Hokkaido의 별도 계통 특성을 검토하되 현재는 50Hz에 포함합니다.
    "50Hz": ["Hokkaido", "Tohoku", "Tokyo"],
    "60Hz": ["Chubu", "Hokuriku", "Kansai", "Chugoku", "Shikoku", "Kyushu"],
}
AREA_TO_ZONE = {
    area: zone for zone, areas in AREAS_BY_ZONE.items() for area in areas
}


def _last_completed_sunday(reference_date: date) -> date:
    """기준일 이전에 완전히 끝난 가장 최근 일요일을 반환합니다."""
    current_week_monday = reference_date - timedelta(days=reference_date.weekday())
    return current_week_monday - timedelta(days=1)


def generate_sample_data(
    reference_date: date | str | None = None, seed: int = 20250714
) -> pd.DataFrame:
    """최근 완료된 8주간의 재현 가능한 가상 EPRX 샘플을 생성합니다.

    이 함수의 결과는 실제 EPRX 시장 데이터가 아니며 화면과 집계 기능을
    개발·검증하기 위한 샘플입니다. 날짜는 Asia/Tokyo 달력 기준입니다.
    """
    if reference_date is None:
        reference = pd.Timestamp.now(tz="Asia/Tokyo").date()
    else:
        reference = pd.Timestamp(reference_date).date()

    end_date = _last_completed_sunday(reference)
    start_date = end_date - timedelta(weeks=8) + timedelta(days=1)
    delivery_dates = pd.date_range(start_date, end_date, freq="D")
    rng = np.random.default_rng(seed)

    area_factors = {
        "Hokkaido": (1.16, 0.80),
        "Tohoku": (1.07, 0.96),
        "Tokyo": (1.13, 1.22),
        "Chubu": (1.00, 1.10),
        "Hokuriku": (0.94, 0.70),
        "Kansai": (1.04, 1.18),
        "Chugoku": (0.98, 0.82),
        "Shikoku": (0.93, 0.66),
        "Kyushu": (0.96, 1.04),
    }
    rows: list[dict[str, object]] = []

    for delivery_date in delivery_dates:
        weekday_effect = 0.96 if delivery_date.weekday() >= 5 else 1.0
        for area, zone in AREA_TO_ZONE.items():
            price_factor, volume_factor = area_factors[area]
            for period_no in range(1, 49):
                half_hour = period_no - 1
                hour = half_hour / 2
                peak = (
                    np.exp(-((hour - 9) / 3.4) ** 2)
                    + 1.15 * np.exp(-((hour - 18) / 3.0) ** 2)
                )
                procurement = max(
                    0.0,
                    (68 + 28 * peak)
                    * volume_factor
                    * weekday_effect
                    + rng.normal(0, 3.0),
                )
                bid = max(0.0, procurement * rng.uniform(1.08, 1.55))
                awarded = max(
                    0.0, min(procurement, bid) * rng.uniform(0.72, 0.98)
                )
                avg_price = max(
                    0.0,
                    (7.5 + 4.5 * peak) * price_factor + rng.normal(0, 0.7),
                )
                min_price = max(0.0, avg_price - rng.uniform(0.4, 2.2))
                max_price = avg_price + rng.uniform(0.5, 3.2)
                rows.append(
                    {
                        "delivery_date": delivery_date.date(),
                        "period_no": period_no,
                        "period_start": f"{half_hour // 2:02d}:{(half_hour % 2) * 30:02d}",
                        "area": area,
                        "frequency_zone": zone,
                        "max_price": round(max_price, 2),
                        "min_price": round(min_price, 2),
                        "avg_price": round(avg_price, 2),
                        "awarded_volume": round(awarded, 2),
                        "bid_volume": round(bid, 2),
                        "procurement_volume": round(procurement, 2),
                    }
                )

    return pd.DataFrame(rows)
