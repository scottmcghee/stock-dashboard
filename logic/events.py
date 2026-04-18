"""Event-based TA signals — rolling binary "fired in last K days" flags.

Backed by empirical analysis (ta_signal_analysis, 800-ticker / 4-year study)
showing per-regime post-event forward-return effects from ±37 bps to +136 bps
with BH-FDR p<0.05. Pure functions; no DB or UI dependencies.
"""

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "evt_rsi_cross_up_30",
    "evt_rsi_cross_down_70",
    "evt_macd_bullish_cross",
    "evt_macd_bearish_cross",
    "evt_breakout_20d_high_vol",
    "evt_regain_200ma",
    "evt_lose_200ma",
]


def _rolling_or(series: pd.Series, lookback: int) -> pd.Series:
    """Rolling-OR: True if any of the last `lookback` bars were True."""
    return series.fillna(False).astype(bool).rolling(lookback, min_periods=1).max().astype(bool)


def detect_events_for_ticker(
    close: pd.Series,
    rsi_14: pd.Series,
    macd_histogram: pd.Series,
    volume_z: pd.Series,
    ratio_to_200ma: pd.Series,
    lookback: int = 3,
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

    rsi_cross_up_30   = (rsi_14 > 30) & (rsi_14.shift(1) <= 30)
    rsi_cross_down_70 = (rsi_14 < 70) & (rsi_14.shift(1) >= 70)
    macd_bull         = (macd_histogram > 0) & (macd_histogram.shift(1) <= 0)
    macd_bear         = (macd_histogram < 0) & (macd_histogram.shift(1) >= 0)
    high_20_prior     = close.shift(1).rolling(20, min_periods=10).max()
    breakout          = (close >= high_20_prior) & (volume_z >= 1.5)
    regain_200ma      = (ratio_to_200ma >= 1.0) & (ratio_to_200ma.shift(1) < 1.0)
    lose_200ma        = (ratio_to_200ma < 1.0)  & (ratio_to_200ma.shift(1) >= 1.0)

    return {
        "evt_rsi_cross_up_30":       _rolling_or(rsi_cross_up_30,   lookback),
        "evt_rsi_cross_down_70":     _rolling_or(rsi_cross_down_70, lookback),
        "evt_macd_bullish_cross":    _rolling_or(macd_bull,         lookback),
        "evt_macd_bearish_cross":    _rolling_or(macd_bear,         lookback),
        "evt_breakout_20d_high_vol": _rolling_or(breakout,          lookback),
        "evt_regain_200ma":          _rolling_or(regain_200ma,      lookback),
        "evt_lose_200ma":            _rolling_or(lose_200ma,        lookback),
    }


def latest_event_flags_bulk(
    ohlcv_df: pd.DataFrame, tickers: list[str], lookback: int = 3
) -> dict[str, dict[str, bool]]:
    """Compute event fired-in-last-K-days flags as of the latest bar per ticker.

    Returns: {ticker: {event_name: True/False}}. Tickers without enough history
    are omitted. Imports heavy pandas/numpy only once.
    """
    from logic.technical import compute_macd, compute_rsi, compute_ratio_to_ma_series

    out: dict[str, dict[str, bool]] = {}
    for ticker in tickers:
        g = ohlcv_df[ohlcv_df["ticker"] == ticker].sort_values("date")
        if len(g) < 210:  # need 200MA lookback + some history
            continue
        close = g["close"].reset_index(drop=True)
        volume = g["volume"].astype(float).reset_index(drop=True)
        rsi = compute_rsi(close, 14)
        _, _, macd_hist = compute_macd(close)
        vol_avg = volume.rolling(20, min_periods=20).mean()
        vol_std = volume.rolling(20, min_periods=20).std()
        volume_z = (volume - vol_avg) / vol_std.replace(0, pd.NA)
        ratio_200 = compute_ratio_to_ma_series(close, 200)
        events = detect_events_for_ticker(
            close=close, rsi_14=rsi, macd_histogram=macd_hist,
            volume_z=volume_z.astype(float), ratio_to_200ma=ratio_200,
            lookback=lookback,
        )
        out[ticker] = {name: bool(series.iloc[-1]) for name, series in events.items()}
    return out


def compute_pivot_proximity(
    close: pd.Series, window: int, exclude_recent: int = 5
) -> tuple[pd.Series, pd.Series]:
    """Trailing-window max/min proximity (as pct from current close).

    Uses close.shift(exclude_recent) so pivots are "confirmed" (no lookahead).
    Returns (dist_to_high, dist_to_low) where:
        dist_to_high = close / max(close[t - window : t - exclude_recent]) - 1
        dist_to_low  = close / min(same window) - 1
    Positive high-distance means above recent peak; negative means below.
    """
    close = close.reset_index(drop=True)
    shifted = close.shift(exclude_recent)
    high = shifted.rolling(window, min_periods=max(20, window // 3)).max()
    low  = shifted.rolling(window, min_periods=max(20, window // 3)).min()
    return (close / high - 1.0, close / low - 1.0)
