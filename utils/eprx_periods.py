"""EPRX 1차 조정력 거래제도별 시간 블록 규칙."""

from __future__ import annotations

from typing import Any

import pandas as pd

EPRX_REFORM_DATE = pd.Timestamp("2026-03-14")
LEGACY_REGIME = "legacy_3hour"
MODERN_REGIME = "modern_30minute"
LEGACY_PERIOD_COUNT = 8
MODERN_PERIOD_COUNT = 48
LEGACY_PERIOD_NOTICE = (
    "해당 기간은 3시간 단위 8개 블록으로 거래된 과거 제도 자료입니다."
)


def market_regime(value: Any) -> str:
    """거래일을 구형 3시간제 또는 신형 30분제로 분류합니다."""
    timestamp = pd.Timestamp(value).normalize()
    return LEGACY_REGIME if timestamp < EPRX_REFORM_DATE else MODERN_REGIME


def expected_period_count(value: Any) -> int:
    """거래일에 적용되는 정상 시간 블록 수를 반환합니다."""
    return (
        LEGACY_PERIOD_COUNT
        if market_regime(value) == LEGACY_REGIME
        else MODERN_PERIOD_COUNT
    )


def period_start_label(value: Any, period_no: int) -> str:
    """원본 블록 번호를 해당 제도의 실제 시작시각으로 표시합니다."""
    if market_regime(value) == LEGACY_REGIME:
        return f"{(int(period_no) - 1) * 3:02d}:00"
    half_hour = int(period_no) - 1
    return f"{half_hour // 2:02d}:{(half_hour % 2) * 30:02d}"