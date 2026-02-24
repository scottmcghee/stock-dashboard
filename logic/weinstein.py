"""Weinstein stage classification — pure calculation, no DB or UI."""

import numpy as np
import pandas as pd


def calculate_weinstein_stages(
    df: pd.DataFrame,
    short_ma: int = 50,
    long_ma: int = 150,
    trend_period: int = 20,
    vol_lookback: int = 21,
) -> pd.DataFrame:
    """Identify Weinstein market stages from OHLCV data.

    Returns the input DataFrame with added 'stage' column.
    """
    if len(df) < long_ma:
        result = df.copy()
        result["stage"] = "Undetermined"
        return result

    df = df.sort_values("date").copy()

    # Moving averages
    df["MA50"] = df["close"].rolling(window=short_ma, min_periods=1).mean()
    df["MA150"] = df["close"].rolling(window=long_ma, min_periods=1).mean()

    # Trends
    df["MA50_trend"] = df["MA50"].diff(periods=trend_period) > 0
    df["MA150_trend"] = df["MA150"].diff(periods=trend_period) > 0

    # Price position
    df["Above_MA50"] = df["close"] > df["MA50"]
    df["Above_MA150"] = df["close"] > df["MA150"]

    # Momentum
    df["Higher_High"] = df["high"].rolling(window=2).apply(
        lambda x: x.iloc[-1] > x.iloc[0] if len(x) == 2 else False
    )
    df["Lower_Low"] = df["low"].rolling(window=2).apply(
        lambda x: x.iloc[-1] < x.iloc[0] if len(x) == 2 else False
    )

    # Volume
    df["Vol_MA"] = df["volume"].rolling(window=vol_lookback, min_periods=1).mean()
    df["High_Vol"] = df["volume"] > 1.5 * df["Vol_MA"]

    # Stage classification
    conditions = [
        # Stage 4: Declining
        (~df["MA150_trend"]) & (~df["Above_MA150"]) & (df["Lower_Low"] == True),
        # Stage 2: Advancing
        (df["MA150_trend"]) & (df["MA50_trend"]) & (df["Above_MA50"]) & (df["Higher_High"] == True),
        # Stage 3: Topping
        (df["MA150_trend"]) & (~df["MA50_trend"]) & (~df["Above_MA50"]),
        # Stage 1: Basing
        (~df["MA150_trend"]) & (df["close"].rolling(window=50, min_periods=1).std() < 0.1 * df["close"]),
    ]
    choices = ["Stage 4", "Stage 2", "Stage 3", "Stage 1"]
    df["stage"] = np.select(conditions, choices, default="Undetermined")

    return df


def classify_bulk_stages(ohlcv_df: pd.DataFrame) -> dict[str, str]:
    """Classify Weinstein stages for multiple tickers from a single DataFrame.

    Args:
        ohlcv_df: DataFrame with columns [ticker, date, open, high, low, close, volume].

    Returns:
        Dict mapping ticker -> latest stage label.
    """
    result: dict[str, str] = {}
    for ticker, group in ohlcv_df.groupby("ticker"):
        staged = calculate_weinstein_stages(group)
        if not staged.empty and "stage" in staged.columns:
            result[ticker] = staged["stage"].iloc[-1]
        else:
            result[ticker] = "Undetermined"
    return result
