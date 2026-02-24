"""Technical analysis indicators — pure functions, no DB or UI."""

import numpy as np
import pandas as pd

from logic.chart_helpers import resample_ohlcv


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window=period, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI using exponential moving average of gains/losses."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # All gains (avg_loss=0) → rs=inf → rsi=100; all losses (avg_gain=0) → rs=0 → rsi=0
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    rsi = rsi.where(avg_loss != 0, other=np.where(avg_gain > 0, 100.0, np.nan))
    return rsi


def get_latest_rsi(close: pd.Series, period: int = 14) -> float | None:
    """Return the most recent RSI value, or None if insufficient data."""
    if len(close) < period + 1:
        return None
    rsi = compute_rsi(close, period)
    val = rsi.iloc[-1]
    return None if pd.isna(val) else float(val)


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD: returns (macd_line, signal_line, histogram)."""
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def detect_macd_crossover(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    lookback: int = 3,
) -> str | None:
    """Detect recent MACD crossover within lookback bars.

    Returns 'bullish', 'bearish', or None.
    """
    min_required = slow + signal + lookback
    if len(close) < min_required:
        return None
    macd_line, signal_line, _ = compute_macd(close, fast, slow, signal)
    diff = macd_line - signal_line
    recent = diff.iloc[-(lookback + 1):]
    if recent.isna().any():
        return None
    for i in range(1, len(recent)):
        if recent.iloc[i - 1] < 0 and recent.iloc[i] >= 0:
            return "bullish"
        if recent.iloc[i - 1] > 0 and recent.iloc[i] <= 0:
            return "bearish"
    return None


def check_price_touches_ma(
    ohlcv: pd.DataFrame, period: int
) -> bool:
    """True if the latest candle's [low, high] range contains the SMA value."""
    if len(ohlcv) < period:
        return False
    sma = compute_sma(ohlcv["close"], period)
    sma_val = sma.iloc[-1]
    if pd.isna(sma_val):
        return False
    return float(ohlcv["low"].iloc[-1]) <= sma_val <= float(ohlcv["high"].iloc[-1])


def apply_ta_filters(
    ohlcv_df: pd.DataFrame,
    tickers: list[str],
    rsi_min: float | None = None,
    rsi_max: float | None = None,
    macd_crossover: str | None = None,
    ma_touch_period: int | None = None,
    timeframe: str = "D",
) -> dict[str, dict]:
    """Apply TA filters to bulk OHLCV data, return passing tickers with indicator values.

    Args:
        ohlcv_df: DataFrame with columns [ticker, date, open, high, low, close, volume].
        tickers: List of tickers to evaluate.
        rsi_min/rsi_max: RSI range filter (0-100).
        macd_crossover: 'bullish' or 'bearish' to require that crossover.
        ma_touch_period: SMA period for price-touch check.
        timeframe: 'D', 'W', or 'M' for resampling.

    Returns:
        Dict of ticker -> {'rsi': float, 'macd_cross': str|None, 'ma_touch': bool}
        for tickers that pass all active filters.
    """
    no_filters = (
        rsi_min is None
        and rsi_max is None
        and macd_crossover is None
        and ma_touch_period is None
    )
    if no_filters:
        return {t: {} for t in tickers}

    results = {}
    for ticker in tickers:
        ticker_data = ohlcv_df[ohlcv_df["ticker"] == ticker].copy()
        if ticker_data.empty:
            continue

        # Resample if needed
        if timeframe != "D":
            ticker_data = ticker_data.set_index("date")
            ticker_data = resample_ohlcv(ticker_data, timeframe)
            ticker_data = ticker_data.reset_index()

        close = ticker_data["close"]
        info: dict = {}

        # RSI filter
        rsi_val = get_latest_rsi(close)
        info["rsi"] = rsi_val
        if rsi_min is not None and (rsi_val is None or rsi_val < rsi_min):
            continue
        if rsi_max is not None and (rsi_val is None or rsi_val > rsi_max):
            continue

        # MACD crossover filter
        cross = detect_macd_crossover(close)
        info["macd_cross"] = cross
        if macd_crossover is not None and cross != macd_crossover:
            continue

        # MA touch filter
        if ma_touch_period is not None:
            touches = check_price_touches_ma(ticker_data, ma_touch_period)
            info["ma_touch"] = touches
            if not touches:
                continue
        else:
            info["ma_touch"] = None

        results[ticker] = info

    return results
