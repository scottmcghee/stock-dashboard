"""Weinstein stage classification: pure calculation, no DB or UI.

Stan Weinstein's four stages, approximated with two moving averages:
Stage 1 basing, Stage 2 advancing, Stage 3 topping, Stage 4 declining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHORT_MA_PERIOD = 50
LONG_MA_PERIOD = 150  # about 30 weeks, Weinstein's trend line
TREND_PERIOD = 20  # an MA is "rising" if it is higher than TREND_PERIOD bars ago
BASING_STD_WINDOW = 50
BASING_MAX_STD_FRACTION = 0.1  # basing: 50-bar close std below 10% of price
# Bars to fetch for a stage at the latest bar: the long MA, its trend period, and a margin.
WEINSTEIN_LOOKBACK_BARS = 200


def calculate_weinstein_stages(
    df: pd.DataFrame,
    short_ma: int = SHORT_MA_PERIOD,
    long_ma: int = LONG_MA_PERIOD,
    trend_period: int = TREND_PERIOD,
) -> pd.DataFrame:
    """Identify Weinstein market stages from OHLCV data.

    Returns a date-sorted copy of the input with an added 'stage' column. A bar
    is "Undetermined" until the long MA and its ``trend_period``-bar change are
    defined (its first ``long_ma + trend_period - 1`` bars), like every other
    indicator, which is NaN until its window is full.
    """
    df = df.sort_values("date").copy()
    close = df["close"]

    ma_short = close.rolling(window=short_ma, min_periods=short_ma).mean()
    ma_long = close.rolling(window=long_ma, min_periods=long_ma).mean()
    short_change = ma_short.diff(periods=trend_period)
    long_change = ma_long.diff(periods=trend_period)

    short_rising = short_change > 0
    long_rising = long_change > 0
    above_short = close > ma_short
    above_long = close > ma_long
    higher_high = df["high"] > df["high"].shift(1)
    lower_low = df["low"] < df["low"].shift(1)
    basing_range = (
        close.rolling(window=BASING_STD_WINDOW, min_periods=BASING_STD_WINDOW).std()
        < BASING_MAX_STD_FRACTION * close
    )

    conditions = [
        # Stage 4: declining
        ~long_rising & ~above_long & lower_low,
        # Stage 2: advancing
        long_rising & short_rising & above_short & higher_high,
        # Stage 3: topping
        long_rising & ~short_rising & ~above_short,
        # Stage 1: basing
        ~long_rising & basing_range,
    ]
    choices = ["Stage 4", "Stage 2", "Stage 3", "Stage 1"]
    stage = np.select(conditions, choices, default="Undetermined")
    defined = (long_change.notna() & short_change.notna()).to_numpy()
    df["stage"] = np.where(defined, stage, "Undetermined")
    return df


def classify_bulk_stages(ohlcv_df: pd.DataFrame) -> dict[str, str]:
    """Classify Weinstein stages for multiple tickers from a single DataFrame.

    Args:
        ohlcv_df: DataFrame with columns [ticker, date, open, high, low, close, volume].

    Returns:
        Dict mapping ticker -> latest stage label.
    """
    result: dict[str, str] = {}
    for ticker, group in ohlcv_df.groupby("ticker", observed=True):
        staged = calculate_weinstein_stages(group)
        result[str(ticker)] = str(staged["stage"].iloc[-1]) if not staged.empty else "Undetermined"
    return result
