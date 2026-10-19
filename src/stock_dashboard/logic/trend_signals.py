"""Trend-readiness scans: washout reversals and volatility compression ("coiled spring").

Each scan is a fixed rule on indicators at the latest bar:

- Washout bull: oversold, well below the 50-bar MA, wide Bollinger bands and
  volume at or above its 20-bar mean. The idea is seller exhaustion: a sharp,
  high-volume decline that has stretched price far from its trend.
- Washout bear: the mirror image, an extended high-volatility rally.
- Coiled spring: narrow bands and a low daily range with price near its
  200-bar MA, i.e. a quiet base. It is direction-agnostic; a breakout decides.

The default thresholds are hand-chosen starting points exposed as parameters,
not fitted values. They equal the Signal Scans page defaults
(``scan_settings.ScanSettings()``), which the screener's pre-filter also uses.
Each result has a scale-free ``strength`` (larger = more extreme) for the
composite score.
"""

from __future__ import annotations

import pandas as pd

from stock_dashboard.logic.technical import (
    RSI_PERIOD,
    compute_adr_series,
    compute_rsi,
    compute_sma,
)

MOMENTUM_PERIOD = 20
SHORT_MA_PERIOD = 50
LONG_MA_PERIOD = 200
BB_PERIOD = 20
BB_STD_MULTIPLE = 2
VOLUME_AVG_PERIOD = 20
BB_WIDTH_CHANGE_PERIOD = 20
WASHOUT_MIN_VOLUME_RATIO = 1.0  # washout bull: volume at or above its 20-bar mean
# The longest look-back above (the 200-bar MA) sets the history every scan needs.
TREND_REQUIRED_BARS = LONG_MA_PERIOD


def compute_trend_indicators(
    ohlcv: pd.DataFrame, min_bars: int = TREND_REQUIRED_BARS
) -> pd.DataFrame:
    """Compute trend-readiness indicators at the latest bar for each ticker.

    Args:
        ohlcv: DataFrame with columns: ticker, date, open, high, low, close, volume.
        min_bars: Minimum history required per ticker.

    Returns:
        One row per ticker with: symbol, rsi, mom_20, ratio_50ma, ratio_200ma,
        bb_width (band width as % of the middle band), adr_pct, vol_ratio
        (volume / 20-bar mean), bb_width_chg (change in bb_width over 20 bars).
    """
    if ohlcv.empty:
        return pd.DataFrame()

    rows = []
    for ticker, g in ohlcv.groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < min_bars:
            continue
        c = g["close"]
        sma_bb = c.rolling(BB_PERIOD).mean()
        bb_width = (2 * BB_STD_MULTIPLE * c.rolling(BB_PERIOD).std() / sma_bb) * 100
        volume = g["volume"].astype(float)
        rows.append(
            {
                "symbol": ticker,
                "rsi": compute_rsi(c, RSI_PERIOD).iloc[-1],
                "mom_20": (c.pct_change(MOMENTUM_PERIOD) * 100).iloc[-1],
                "ratio_50ma": (c / compute_sma(c, SHORT_MA_PERIOD)).iloc[-1],
                "ratio_200ma": (c / compute_sma(c, LONG_MA_PERIOD)).iloc[-1],
                "bb_width": bb_width.iloc[-1],
                "adr_pct": compute_adr_series(g["high"], g["low"], c).iloc[-1],
                "vol_ratio": (volume / volume.rolling(VOLUME_AVG_PERIOD).mean()).iloc[-1],
                "bb_width_chg": bb_width.diff(BB_WIDTH_CHANGE_PERIOD).iloc[-1],
            }
        )

    return pd.DataFrame(rows)


def scan_washout_bull(
    indicators: pd.DataFrame,
    rsi_max: float = 40.0,
    mom_max: float = -5.0,
    ratio_50ma_max: float = 0.94,
    bb_width_min: float = 12.0,
    vol_ratio_min: float = WASHOUT_MIN_VOLUME_RATIO,
) -> pd.DataFrame:
    """Tickers in a bullish washout setup: oversold, below the 50MA, volatile, on volume."""
    if indicators.empty:
        return pd.DataFrame()

    mask = (
        (indicators["rsi"] <= rsi_max)
        & (indicators["mom_20"] <= mom_max)
        & (indicators["ratio_50ma"] <= ratio_50ma_max)
        & (indicators["bb_width"] >= bb_width_min)
        & (indicators["vol_ratio"] >= vol_ratio_min)
    )
    result = indicators[mask].copy()
    if result.empty:
        return result
    result["signal_value"] = result["mom_20"]
    result["strength"] = -result["mom_20"]  # the deeper the decline, the stronger
    result["signal_label"] = "Washout Bull"
    return result


def scan_washout_bear(
    indicators: pd.DataFrame,
    rsi_min: float = 60.0,
    mom_min: float = 5.0,
    ratio_50ma_min: float = 1.06,
    bb_width_min: float = 12.0,
) -> pd.DataFrame:
    """Tickers in a bearish washout setup: overbought, extended above the 50MA, volatile."""
    if indicators.empty:
        return pd.DataFrame()

    mask = (
        (indicators["rsi"] >= rsi_min)
        & (indicators["mom_20"] >= mom_min)
        & (indicators["ratio_50ma"] >= ratio_50ma_min)
        & (indicators["bb_width"] >= bb_width_min)
    )
    result = indicators[mask].copy()
    if result.empty:
        return result
    result["signal_value"] = result["mom_20"]
    result["strength"] = result["mom_20"]
    result["signal_label"] = "Washout Bear"
    return result


def scan_coiled_spring(
    indicators: pd.DataFrame,
    bb_width_max: float = 7.0,
    adr_max: float = 1.5,
    ratio_200ma_min: float = 0.85,
    ratio_200ma_max: float = 1.15,
) -> pd.DataFrame:
    """Tickers with compressed volatility near the 200MA (a base; direction-agnostic)."""
    if indicators.empty:
        return pd.DataFrame()

    mask = (
        (indicators["bb_width"] <= bb_width_max)
        & (indicators["adr_pct"] <= adr_max)
        & (indicators["ratio_200ma"] >= ratio_200ma_min)
        & (indicators["ratio_200ma"] <= ratio_200ma_max)
    )
    result = indicators[mask].copy()
    if result.empty:
        return result
    result["signal_value"] = result["bb_width"]
    result["strength"] = -result["bb_width"]  # tighter bands score higher
    result["signal_label"] = result["ratio_200ma"].map(
        lambda r: "Coiled Spring (above 200MA)" if r >= 1.0 else "Coiled Spring (below 200MA)"
    )
    return result
