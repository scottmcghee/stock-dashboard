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


def compute_adr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> float | None:
    """Average Daily Range as a percentage of close over the last `period` bars.

    ADR% = mean((high - low) / close * 100) over the period.
    Returns None if insufficient data.
    """
    if len(close) < period:
        return None
    daily_range_pct = (high - low) / close * 100
    val = daily_range_pct.iloc[-period:].mean()
    return None if pd.isna(val) else float(val)


def compute_momentum(close: pd.Series, period: int = 63) -> float | None:
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
    sma = compute_sma(close, period)
    sma_val = sma.iloc[-1]
    if pd.isna(sma_val) or sma_val == 0:
        return None
    return float(close.iloc[-1] / sma_val)


def compute_technical_zscore_table(
    ohlcv_df: pd.DataFrame,
    tickers: list[str],
    rsi_period: int = 14,
    momentum_period: int = 63,
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
        rows.append({
            "ticker": ticker,
            "rsi": rsi_val,
            "momentum": mom_val,
            "ratio_to_ma": ratio_val,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Z-score each indicator across the universe
    for col, z_col in [("rsi", "rsi_z"), ("momentum", "momentum_z"), ("ratio_to_ma", "ratio_z")]:
        valid = df[col].dropna()
        if len(valid) > 1:
            mean = valid.mean()
            std = valid.std()
            if std > 0:
                df[z_col] = (df[col] - mean) / std
            else:
                df[z_col] = 0.0
        else:
            df[z_col] = np.nan

    # Composite = average of available z-scores per row
    z_cols = ["rsi_z", "momentum_z", "ratio_z"]
    df["composite_z"] = df[z_cols].mean(axis=1)

    return df.sort_values("composite_z", ascending=False).reset_index(drop=True)


def detect_gap(
    ohlcv: pd.DataFrame, min_gap_pct: float = 0.0, lookback: int = 5
) -> dict | None:
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
        # Bull gap
        if curr["low"] > prev["high"]:
            gap_pct = (curr["low"] - prev["high"]) / prev_close * 100
            if gap_pct >= min_gap_pct:
                return {
                    "type": "bull",
                    "gap_pct": round(gap_pct, 2),
                    "date": ohlcv.index[i] if isinstance(ohlcv.index, pd.DatetimeIndex) else ohlcv.iloc[i].get("date"),
                }
        # Bear gap
        if curr["high"] < prev["low"]:
            gap_pct = (prev["low"] - curr["high"]) / prev_close * 100
            if gap_pct >= min_gap_pct:
                return {
                    "type": "bear",
                    "gap_pct": round(gap_pct, 2),
                    "date": ohlcv.index[i] if isinstance(ohlcv.index, pd.DatetimeIndex) else ohlcv.iloc[i].get("date"),
                }
    return None


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
    gap_lookback: int = 5,
    timeframe: str = "D",
) -> dict[str, dict]:
    """Apply TA filters to bulk OHLCV data, return passing tickers with indicator values.

    Args:
        ohlcv_df: DataFrame with columns [ticker, date, open, high, low, close, volume].
        tickers: List of tickers to evaluate.
        rsi_min/rsi_max: RSI range filter (0-100).
        macd_crossover: 'bullish' or 'bearish' to require that crossover.
        ma_touch_period: SMA period for price-touch check.
        adr_min: Minimum average daily range percentage (e.g. 2 = 2%).
        gap_type: 'bull', 'bear', or 'any' to require a recent gap.
        gap_min_pct: Minimum gap size as % of previous close.
        gap_lookback: Number of bars to search for a gap.
        timeframe: 'D', 'W', or 'M' for resampling.

    Returns:
        Dict of ticker -> {'rsi': float, 'macd_cross': str|None, 'ma_touch': bool, 'adr': float,
                           'gap': dict|None}
        for tickers that pass all active filters.
    """
    no_filters = (
        rsi_min is None
        and rsi_max is None
        and macd_crossover is None
        and ma_touch_period is None
        and adr_min is None
        and gap_type is None
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

        # ADR filter
        adr_val = compute_adr(ticker_data["high"], ticker_data["low"], close)
        info["adr"] = adr_val
        if adr_min is not None and (adr_val is None or adr_val < adr_min):
            continue

        # Gap filter
        gap = detect_gap(ticker_data, min_gap_pct=gap_min_pct, lookback=gap_lookback)
        info["gap"] = gap
        if gap_type is not None:
            if gap is None:
                continue
            if gap_type != "any" and gap["type"] != gap_type:
                continue

        results[ticker] = info

    return results
