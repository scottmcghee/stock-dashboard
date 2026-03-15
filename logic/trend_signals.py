"""Trend signal scans — washout reversal and coiled spring detection.

Based on empirical analysis of 1,562 ETF trends (2020-2026):
- Washout: oversold + elevated volatility → median +20.8% / 53 days
- Coiled Spring: volatility compression near 200MA → median +17.6% / 95 days
"""

import numpy as np
import pandas as pd


def compute_trend_indicators(ohlcv: pd.DataFrame, min_bars: int = 200) -> pd.DataFrame:
    """Compute trend-readiness indicators at the latest bar for each ticker.

    Args:
        ohlcv: DataFrame with columns: ticker, date, open, high, low, close, volume.
        min_bars: Minimum history required per ticker.

    Returns:
        One row per ticker with: symbol, rsi, mom_20, ratio_50ma, ratio_200ma,
        bb_width, adr_pct, vol_ratio, bb_width_chg.
    """
    if ohlcv.empty:
        return pd.DataFrame()

    frames = []
    for ticker, g in ohlcv.groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < min_bars:
            continue

        c = g["close"]

        # RSI (14) — Wilder exponential smoothing
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Momentum (20d ROC %)
        mom_20 = c.pct_change(20) * 100

        # Moving averages
        sma_20 = c.rolling(20).mean()
        sma_50 = c.rolling(50).mean()
        sma_200 = c.rolling(200).mean()

        # Ratios to MAs
        ratio_50ma = c / sma_50
        ratio_200ma = c / sma_200

        # Bollinger Band Width (20d, 2 std each side) as percentage
        bb_std = c.rolling(20).std()
        bb_width = (4 * bb_std / sma_20) * 100

        # ADR% (14-day average of daily range)
        adr_pct = ((g["high"] / g["low"] - 1) * 100).rolling(14).mean()

        # Volume ratio (vs 20d average)
        vol_avg = g["volume"].rolling(20).mean()
        vol_ratio = g["volume"] / vol_avg

        # BB Width change over 20 days (positive = expanding, negative = contracting)
        bb_width_chg = bb_width.diff(20)

        frames.append({
            "symbol": ticker,
            "rsi": rsi.iloc[-1],
            "mom_20": mom_20.iloc[-1],
            "ratio_50ma": ratio_50ma.iloc[-1],
            "ratio_200ma": ratio_200ma.iloc[-1],
            "bb_width": bb_width.iloc[-1],
            "adr_pct": adr_pct.iloc[-1],
            "vol_ratio": vol_ratio.iloc[-1],
            "bb_width_chg": bb_width_chg.iloc[-1],
        })

    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


def scan_washout_bull(
    indicators: pd.DataFrame,
    rsi_max: float = 40.0,
    mom_max: float = -5.0,
    ratio_50ma_max: float = 0.94,
    bb_width_min: float = 12.0,
    vol_ratio_min: float = 1.0,
) -> pd.DataFrame:
    """Find tickers showing washout reversal setup (bullish).

    Oversold, below 50MA, elevated volatility and volume — exhausted sellers,
    buyers arriving. Empirical median: +20.8% over 53 days.
    """
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
    result["signal_label"] = "Washout Bull"
    return result


def scan_washout_bear(
    indicators: pd.DataFrame,
    rsi_min: float = 60.0,
    mom_min: float = 5.0,
    ratio_50ma_min: float = 1.05,
    bb_width_min: float = 12.0,
) -> pd.DataFrame:
    """Find tickers showing washout reversal setup (bearish).

    Overbought, above 50MA, elevated volatility — extended rally vulnerable
    to reversal. Mirror image of washout bull.
    """
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
    result["signal_label"] = "Washout Bear"
    return result


def scan_coiled_spring(
    indicators: pd.DataFrame,
    bb_width_max: float = 7.0,
    adr_max: float = 1.5,
    ratio_200ma_min: float = 0.90,
    ratio_200ma_max: float = 1.15,
) -> pd.DataFrame:
    """Find tickers showing coiled spring / volatility compression setup.

    Tight Bollinger Bands, low ADR%, price near 200MA — basing pattern
    (Weinstein Stage 1 or 3). Direction agnostic; breakout determines trade.
    Empirical median: 17.6% move over 95 days.
    """
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

    # Tighter bands = stronger signal (negate so higher abs z-score = tighter)
    result["signal_value"] = -result["bb_width"]
    result["signal_label"] = result["ratio_200ma"].apply(
        lambda r: "Coiled Spring (above 200MA)" if r >= 1.0
        else "Coiled Spring (below 200MA)"
    )
    return result
