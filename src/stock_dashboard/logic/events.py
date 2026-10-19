"""Event-based TA signals: binary "fired within the last K bars" flags.

Each event is a discrete crossing (RSI through 30/70, MACD histogram through
zero, close through its 200-bar MA, a 20-bar breakout on heavy volume). A
crossing is a single bar, so flags are held for ``lookback`` bars to make
them usable as screen filters. Pure functions; no DB or UI dependencies.
"""

from __future__ import annotations

import pandas as pd

from stock_dashboard.logic.technical import (
    RSI_PERIOD,
    compute_macd,
    compute_ratio_to_ma_series,
    compute_rsi,
    compute_volume_z,
)

EVENT_COLUMNS = [
    "evt_rsi_cross_up_30",
    "evt_rsi_cross_down_70",
    "evt_macd_bullish_cross",
    "evt_macd_bearish_cross",
    "evt_breakout_20d_high_vol",
    "evt_regain_200ma",
    "evt_lose_200ma",
]

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BREAKOUT_WINDOW = 20
BREAKOUT_MIN_PERIODS = 10
BREAKOUT_MIN_VOLUME_Z = 1.5
LONG_MA_PERIOD = 200
EVENT_LOOKBACK = 3
# The 200-bar MA, the bar before a crossing, the hold window, and a small margin.
EVENT_REQUIRED_BARS = LONG_MA_PERIOD + 10


def _rolling_any(series: pd.Series, lookback: int) -> pd.Series:
    """True where any of the last ``lookback`` bars is True."""
    flags = series.astype("boolean").fillna(False).astype(int)
    return flags.rolling(lookback, min_periods=1).max().astype(bool)


def detect_events_for_ticker(
    close: pd.Series,
    rsi_14: pd.Series,
    macd_histogram: pd.Series,
    volume_z: pd.Series,
    ratio_to_200ma: pd.Series,
    lookback: int = EVENT_LOOKBACK,
) -> dict[str, pd.Series]:
    """Compute the full event flag set for one ticker.

    Inputs must be same-length series. Indices are reset internally to avoid
    alignment surprises when inputs come from different frames.
    """
    close = close.reset_index(drop=True)
    rsi_14 = rsi_14.reset_index(drop=True)
    macd_histogram = macd_histogram.reset_index(drop=True)
    volume_z = volume_z.reset_index(drop=True)
    ratio_to_200ma = ratio_to_200ma.reset_index(drop=True)

    prior_high = close.shift(1).rolling(BREAKOUT_WINDOW, min_periods=BREAKOUT_MIN_PERIODS).max()
    raw = {
        "evt_rsi_cross_up_30": (rsi_14 > RSI_OVERSOLD) & (rsi_14.shift(1) <= RSI_OVERSOLD),
        "evt_rsi_cross_down_70": (rsi_14 < RSI_OVERBOUGHT) & (rsi_14.shift(1) >= RSI_OVERBOUGHT),
        "evt_macd_bullish_cross": (macd_histogram > 0) & (macd_histogram.shift(1) <= 0),
        "evt_macd_bearish_cross": (macd_histogram < 0) & (macd_histogram.shift(1) >= 0),
        "evt_breakout_20d_high_vol": (close >= prior_high) & (volume_z >= BREAKOUT_MIN_VOLUME_Z),
        "evt_regain_200ma": (ratio_to_200ma >= 1.0) & (ratio_to_200ma.shift(1) < 1.0),
        "evt_lose_200ma": (ratio_to_200ma < 1.0) & (ratio_to_200ma.shift(1) >= 1.0),
    }
    return {name: _rolling_any(flags, lookback) for name, flags in raw.items()}


def latest_event_flags_bulk(
    ohlcv_df: pd.DataFrame, tickers: list[str], lookback: int = EVENT_LOOKBACK
) -> dict[str, dict[str, bool]]:
    """Event flags as of each ticker's latest bar: ``{ticker: {event_name: fired}}``.

    Tickers with fewer than :data:`EVENT_REQUIRED_BARS` bars are omitted.
    """
    out: dict[str, dict[str, bool]] = {}
    for ticker in tickers:
        g = ohlcv_df[ohlcv_df["ticker"] == ticker].sort_values("date")
        if len(g) < EVENT_REQUIRED_BARS:
            continue
        close = g["close"].reset_index(drop=True)
        _, _, macd_hist = compute_macd(close)
        events = detect_events_for_ticker(
            close=close,
            rsi_14=compute_rsi(close, RSI_PERIOD),
            macd_histogram=macd_hist,
            volume_z=compute_volume_z(g["volume"].reset_index(drop=True)),
            ratio_to_200ma=compute_ratio_to_ma_series(close, LONG_MA_PERIOD),
            lookback=lookback,
        )
        out[ticker] = {name: bool(series.iloc[-1]) for name, series in events.items()}
    return out
