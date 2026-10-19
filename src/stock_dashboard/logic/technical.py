"""Technical analysis indicators: pure functions, no DB or UI.

This module holds the single definition of each shared indicator (RSI,
volume z-score, ADR%, pivot proximity, ...). The screener, trend scans,
event flags, backtester and preset overlay all call these, so a preset's
filters pass on a chart's bars as the backtest computed them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stock_dashboard.logic.chart_helpers import resample_ohlcv

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_CROSS_LOOKBACK = 3
ADR_PERIOD = 14
VOLUME_Z_WINDOW = 20
DEFAULT_MOMENTUM_PERIOD = 63
DEFAULT_RATIO_MA_PERIOD = 200
DEFAULT_GAP_LOOKBACK = 5

# Support/resistance proximity: pivots come from the trailing window ending
# PIVOT_EXCLUDE_RECENT bars ago, so the most recent bars cannot form their own pivot.
PIVOT_EXCLUDE_RECENT = 5
PIVOT_MIN_PERIODS_FLOOR = 20
SCREENER_PIVOT_WINDOW = 126

# Bars fetched for indicator-history charts: about 20 months of sessions.
INDICATOR_HISTORY_BARS = 420

# Daily bars per resampled bar, used to size look-backs for weekly/monthly TA.
BARS_PER_TIMEFRAME = {"D": 1, "W": 5, "M": 21}


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window=period, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI: gains and losses smoothed with ``ewm(alpha=1/period, adjust=False)``.

    100 when the window holds only gains, 0 when only losses, NaN when flat.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    return rsi.where(avg_loss != 0, other=np.where(avg_gain > 0, 100.0, np.nan))


def get_latest_rsi(close: pd.Series, period: int = RSI_PERIOD) -> float | None:
    """Return the most recent RSI value, or None if insufficient data."""
    if len(close) < period + 1:
        return None
    val = compute_rsi(close, period).iloc[-1]
    return None if pd.isna(val) else float(val)


def compute_volume_z(volume: pd.Series, window: int = VOLUME_Z_WINDOW) -> pd.Series:
    """Volume z-score against the trailing ``window`` sessions, current bar included."""
    vol = volume.astype(float)
    mean = vol.rolling(window, min_periods=window).mean()
    std = vol.rolling(window, min_periods=window).std()
    return (vol - mean) / std.replace(0, np.nan)


def compute_adr_series(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADR_PERIOD
) -> pd.Series:
    """Average daily range in percent: rolling mean of ``(high - low) / close * 100``."""
    daily_range_pct = (high - low) / close * 100
    return daily_range_pct.rolling(period, min_periods=period).mean()


def compute_adr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADR_PERIOD
) -> float | None:
    """Latest ADR% (see :func:`compute_adr_series`), or None with fewer than ``period`` bars."""
    if len(close) < period:
        return None
    val = compute_adr_series(high, low, close, period).iloc[-1]
    return None if pd.isna(val) else float(val)


def compute_macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD: returns (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def detect_macd_crossover(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
    lookback: int = MACD_CROSS_LOOKBACK,
) -> str | None:
    """Detect a MACD/signal-line crossover within the last ``lookback`` bars.

    Returns 'bullish', 'bearish', or None.
    """
    if len(close) < slow + signal + lookback:
        return None
    macd_line, signal_line, _ = compute_macd(close, fast, slow, signal)
    recent = (macd_line - signal_line).iloc[-(lookback + 1) :]
    if recent.isna().any():
        return None
    for i in range(1, len(recent)):
        if recent.iloc[i - 1] < 0 <= recent.iloc[i]:
            return "bullish"
        if recent.iloc[i - 1] > 0 >= recent.iloc[i]:
            return "bearish"
    return None


def check_price_touches_ma(ohlcv: pd.DataFrame, period: int) -> bool:
    """True if the latest candle's [low, high] range contains the SMA value."""
    if len(ohlcv) < period:
        return False
    sma_val = compute_sma(ohlcv["close"], period).iloc[-1]
    if pd.isna(sma_val):
        return False
    return bool(float(ohlcv["low"].iloc[-1]) <= sma_val <= float(ohlcv["high"].iloc[-1]))


def compute_momentum(close: pd.Series, period: int = DEFAULT_MOMENTUM_PERIOD) -> float | None:
    """Rate of change over `period` bars: (close - close_n) / close_n * 100."""
    if len(close) < period + 1:
        return None
    current = close.iloc[-1]
    past = close.iloc[-period - 1]
    if pd.isna(past) or past == 0:
        return None
    return float((current - past) / past * 100)


def compute_ratio_to_ma(close: pd.Series, period: int = 126) -> float | None:
    """Ratio of latest close to its SMA: close / SMA(period)."""
    if len(close) < period:
        return None
    sma_val = compute_sma(close, period).iloc[-1]
    if pd.isna(sma_val) or sma_val == 0:
        return None
    return float(close.iloc[-1] / sma_val)


def compute_momentum_series(close: pd.Series, period: int = DEFAULT_MOMENTUM_PERIOD) -> pd.Series:
    """Rate of change series: (close - close_shifted) / close_shifted * 100."""
    shifted = close.shift(period)
    return ((close - shifted) / shifted.replace(0, np.nan)) * 100


def compute_ratio_to_ma_series(close: pd.Series, period: int = 126) -> pd.Series:
    """Ratio of close to its SMA over time: close / SMA(period)."""
    return close / compute_sma(close, period).replace(0, np.nan)


def compute_pivot_proximity(
    close: pd.Series, window: int, exclude_recent: int = PIVOT_EXCLUDE_RECENT
) -> tuple[pd.Series, pd.Series]:
    """Distance of each close from the trailing-window high and low, as a fraction.

    The window ends ``exclude_recent`` bars ago (no lookahead, and the latest
    bars cannot be their own pivot). Returns ``(dist_to_high, dist_to_low)``:

        dist_to_high = close / max(close[t - window - exclude_recent : t - exclude_recent]) - 1
        dist_to_low  = close / min(same window) - 1
    """
    close = close.reset_index(drop=True)
    shifted = close.shift(exclude_recent)
    min_periods = max(PIVOT_MIN_PERIODS_FLOOR, window // 3)
    high = shifted.rolling(window, min_periods=min_periods).max()
    low = shifted.rolling(window, min_periods=min_periods).min()
    return close / high - 1.0, close / low - 1.0


def ta_required_bars(
    momentum_period: int = DEFAULT_MOMENTUM_PERIOD,
    ratio_to_ma_period: int = DEFAULT_RATIO_MA_PERIOD,
    timeframe: str = "D",
    ma_touch_period: int | None = None,
) -> int:
    """Daily bars :func:`apply_ta_filters` needs so every indicator has a value at the last bar."""
    bars = max(
        RSI_PERIOD + 1,
        MACD_SLOW + MACD_SIGNAL + MACD_CROSS_LOOKBACK,
        ADR_PERIOD,
        momentum_period + 1,
        ratio_to_ma_period,
        ma_touch_period or 0,
        SCREENER_PIVOT_WINDOW + PIVOT_EXCLUDE_RECENT,
    )
    per_bar = BARS_PER_TIMEFRAME[timeframe]
    # One extra resampled bar covers a partial week or month at either end.
    return (bars + (1 if per_bar > 1 else 0)) * per_bar


def compute_indicator_history(
    ohlcv: pd.DataFrame,
    rsi_period: int = RSI_PERIOD,
    momentum_period: int = DEFAULT_MOMENTUM_PERIOD,
    ma_period: int = 126,
) -> pd.DataFrame:
    """Compute RSI, Momentum, and Ratio-to-MA time series for a single ticker's OHLCV.

    Returns DataFrame with columns: date, rsi, momentum, ratio_to_ma.
    """
    ohlcv = ohlcv.sort_values("date")
    close = ohlcv["close"]
    result = pd.DataFrame({"date": ohlcv["date"].to_numpy()})
    result["rsi"] = compute_rsi(close, rsi_period).to_numpy()
    result["momentum"] = compute_momentum_series(close, momentum_period).to_numpy()
    result["ratio_to_ma"] = compute_ratio_to_ma_series(close, ma_period).to_numpy()
    return result


def compute_technical_zscore_table(
    ohlcv_df: pd.DataFrame,
    tickers: list[str],
    rsi_period: int = RSI_PERIOD,
    momentum_period: int = DEFAULT_MOMENTUM_PERIOD,
    ma_period: int = 126,
) -> pd.DataFrame:
    """Compute RSI, Momentum, Ratio-to-MA for each ticker, then z-score across the universe.

    Returns DataFrame with columns:
        ticker, rsi, momentum, ratio_to_ma, rsi_z, momentum_z, ratio_z, composite_z
    """
    rows = []
    for ticker in tickers:
        group = ohlcv_df[ohlcv_df["ticker"] == ticker]
        if group.empty:
            continue
        close = group.sort_values("date")["close"]
        rsi_val = get_latest_rsi(close, rsi_period)
        mom_val = compute_momentum(close, momentum_period)
        ratio_val = compute_ratio_to_ma(close, ma_period)
        if rsi_val is None and mom_val is None and ratio_val is None:
            continue
        rows.append(
            {"ticker": ticker, "rsi": rsi_val, "momentum": mom_val, "ratio_to_ma": ratio_val}
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col, z_col in [("rsi", "rsi_z"), ("momentum", "momentum_z"), ("ratio_to_ma", "ratio_z")]:
        values = df[col].astype(float)
        valid = values.dropna()
        if len(valid) > 1:
            std = valid.std()
            df[z_col] = (values - valid.mean()) / std if std > 0 else 0.0
        else:
            df[z_col] = np.nan

    # Composite = mean of the z-scores available in each row.
    df["composite_z"] = df[["rsi_z", "momentum_z", "ratio_z"]].mean(axis=1)
    return df.sort_values("composite_z", ascending=False).reset_index(drop=True)


def detect_gap(
    ohlcv: pd.DataFrame, min_gap_pct: float = 0.0, lookback: int = DEFAULT_GAP_LOOKBACK
) -> dict[str, Any] | None:
    """Detect the most recent price gap within lookback bars.

    Bull gap: current low > previous high (gap up).
    Bear gap: current high < previous low (gap down).
    min_gap_pct: minimum gap size as % of previous close.

    Returns dict with 'type' ('bull'/'bear'), 'gap_pct', 'date', or None.
    """
    if len(ohlcv) < 2:
        return None
    start = max(1, len(ohlcv) - lookback)
    for i in range(len(ohlcv) - 1, start - 1, -1):
        prev = ohlcv.iloc[i - 1]
        curr = ohlcv.iloc[i]
        prev_close = prev["close"]
        if prev_close == 0:
            continue
        if curr["low"] > prev["high"]:
            gap_type, gap_pct = "bull", (curr["low"] - prev["high"]) / prev_close * 100
        elif curr["high"] < prev["low"]:
            gap_type, gap_pct = "bear", (prev["low"] - curr["high"]) / prev_close * 100
        else:
            continue
        if gap_pct >= min_gap_pct:
            on_index = isinstance(ohlcv.index, pd.DatetimeIndex)
            return {
                "type": gap_type,
                "gap_pct": round(float(gap_pct), 2),
                "date": ohlcv.index[i] if on_index else curr.get("date"),
            }
    return None


def _outside(value: float | None, low: float | None, high: float | None) -> bool:
    """True when an active [low, high] bound rejects ``value`` (a missing value fails any bound)."""
    if low is None and high is None:
        return False
    if value is None:
        return True
    return (low is not None and value < low) or (high is not None and value > high)


def apply_ta_filters(
    ohlcv_df: pd.DataFrame,
    tickers: list[str],
    rsi_min: float | None = None,
    rsi_max: float | None = None,
    macd_crossover: str | None = None,
    ma_touch_period: int | None = None,
    adr_min: float | None = None,
    gap_type: str | None = None,
    gap_min_pct: float = 0.0,
    gap_lookback: int = DEFAULT_GAP_LOOKBACK,
    momentum_min: float | None = None,
    momentum_max: float | None = None,
    momentum_period: int = DEFAULT_MOMENTUM_PERIOD,
    ratio_to_ma_min: float | None = None,
    ratio_to_ma_max: float | None = None,
    ratio_to_ma_period: int = DEFAULT_RATIO_MA_PERIOD,
    dist_recent_low_126_min: float | None = None,
    dist_recent_low_126_max: float | None = None,
    timeframe: str = "D",
) -> dict[str, dict[str, Any]]:
    """Apply TA filters to bulk OHLCV data, return passing tickers with indicator values.

    Args:
        ohlcv_df: DataFrame with columns [ticker, date, open, high, low, close, volume].
            Fetch at least :func:`ta_required_bars` bars per ticker.
        tickers: List of tickers to evaluate.
        rsi_min/rsi_max: RSI range filter (0-100).
        macd_crossover: 'bullish' or 'bearish' to require that crossover; 'not bullish' /
            'not bearish' to exclude it.
        ma_touch_period: SMA period for price-touch check.
        adr_min: Minimum average daily range percentage (e.g. 2 = 2%).
        gap_type: 'bull', 'bear', or 'any' to require a recent gap.
        gap_min_pct: Minimum gap size as % of previous close.
        gap_lookback: Number of bars to search for a gap.
        momentum_min/momentum_max: Rate-of-change filter (%).
        momentum_period: Lookback for momentum.
        ratio_to_ma_min/ratio_to_ma_max: Close/SMA ratio filter.
        ratio_to_ma_period: SMA period for ratio.
        dist_recent_low_126_min/max: Fractional distance above the trailing 126-bar low.
        timeframe: 'D', 'W', or 'M' for resampling.

    Returns:
        Dict of ticker -> {'rsi', 'macd_cross', 'ma_touch', 'adr', 'momentum',
        'ratio_to_ma', 'gap', 'dist_recent_low_126'} for tickers that pass all active filters.
    """
    no_filters = all(
        f is None
        for f in (
            rsi_min,
            rsi_max,
            macd_crossover,
            ma_touch_period,
            adr_min,
            gap_type,
            momentum_min,
            momentum_max,
            ratio_to_ma_min,
            ratio_to_ma_max,
            dist_recent_low_126_min,
            dist_recent_low_126_max,
        )
    )
    if no_filters:
        return {t: {} for t in tickers}

    results: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        ticker_data = ohlcv_df[ohlcv_df["ticker"] == ticker]
        if ticker_data.empty:
            continue
        if timeframe != "D":
            ticker_data = resample_ohlcv(ticker_data.set_index("date"), timeframe).reset_index()

        close = ticker_data["close"]
        info: dict[str, Any] = {}

        info["rsi"] = get_latest_rsi(close)
        if _outside(info["rsi"], rsi_min, rsi_max):
            continue

        cross = detect_macd_crossover(close)
        info["macd_cross"] = cross
        if macd_crossover is not None:
            if macd_crossover.startswith("not "):
                if cross == macd_crossover.removeprefix("not "):
                    continue
            elif cross != macd_crossover:
                continue

        info["ma_touch"] = None
        if ma_touch_period is not None:
            info["ma_touch"] = check_price_touches_ma(ticker_data, ma_touch_period)
            if not info["ma_touch"]:
                continue

        info["adr"] = compute_adr(ticker_data["high"], ticker_data["low"], close)
        if _outside(info["adr"], adr_min, None):
            continue

        info["momentum"] = compute_momentum(close, momentum_period)
        if _outside(info["momentum"], momentum_min, momentum_max):
            continue

        info["ratio_to_ma"] = compute_ratio_to_ma(close, ratio_to_ma_period)
        if _outside(info["ratio_to_ma"], ratio_to_ma_min, ratio_to_ma_max):
            continue

        gap = detect_gap(ticker_data, min_gap_pct=gap_min_pct, lookback=gap_lookback)
        info["gap"] = gap
        if gap_type is not None and (gap is None or gap_type not in ("any", gap["type"])):
            continue

        info["dist_recent_low_126"] = None
        if dist_recent_low_126_min is not None or dist_recent_low_126_max is not None:
            _, dist_lo = compute_pivot_proximity(close, window=SCREENER_PIVOT_WINDOW)
            last = dist_lo.iloc[-1] if not dist_lo.empty else np.nan
            info["dist_recent_low_126"] = None if pd.isna(last) else float(last)
            if _outside(
                info["dist_recent_low_126"], dist_recent_low_126_min, dist_recent_low_126_max
            ):
                continue

        results[ticker] = info

    return results
