"""How much history to fetch so every indicator computed on it is defined where it is used."""

from __future__ import annotations

from stock_dashboard.logic.backtest import MOMENTUM_PERIODS, RATIO_MA_PERIODS, RS_IMPULSE_WINDOWS
from stock_dashboard.logic.events import EVENT_REQUIRED_BARS
from stock_dashboard.logic.patterns import PATTERN_REQUIRED_BARS
from stock_dashboard.logic.technical import (
    DEFAULT_MOMENTUM_PERIOD,
    DEFAULT_RATIO_MA_PERIOD,
    ta_required_bars,
)
from stock_dashboard.logic.weinstein import LONG_MA_PERIOD as WEINSTEIN_LONG_MA
from stock_dashboard.logic.weinstein import TREND_PERIOD as WEINSTEIN_TREND_PERIOD


def screener_lookback_bars(
    momentum_period: int = DEFAULT_MOMENTUM_PERIOD,
    ratio_to_ma_period: int = DEFAULT_RATIO_MA_PERIOD,
    timeframe: str = "D",
    ma_touch_period: int | None = None,
) -> int:
    """Daily bars per ticker for the screener's TA filters, event flags and pattern scans."""
    return max(
        ta_required_bars(momentum_period, ratio_to_ma_period, timeframe, ma_touch_period),
        EVENT_REQUIRED_BARS,
        PATTERN_REQUIRED_BARS,
    )


def chart_warmup_bars() -> int:
    """Daily bars to load before a chart's first bar so that every preset-overlay
    indicator (see :mod:`stock_dashboard.logic.backtest`) is defined from it."""
    return max(
        screener_lookback_bars(max(MOMENTUM_PERIODS), max(RATIO_MA_PERIODS)),
        WEINSTEIN_LONG_MA + WEINSTEIN_TREND_PERIOD,
        max(RS_IMPULSE_WINDOWS) + 1,
    )
