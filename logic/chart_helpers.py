"""Chart utilities — pure functions, no DB or UI."""

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample OHLCV data to weekly (W) or monthly (M) intervals.

    Expects df with a DatetimeIndex and columns: open, high, low, close, volume.
    Returns original df if interval is 'D'.
    """
    if interval == "D":
        return df
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df.resample(interval, label="right", closed="right").agg(agg_dict)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    return resampled


def get_non_trading_days(dates: pd.Series) -> pd.DatetimeIndex:
    """Compute non-trading days (weekends + holidays) for rangebreak gaps."""
    all_dates = pd.date_range(dates.min(), dates.max())
    weekends = all_dates[all_dates.weekday >= 5]
    all_bdays = pd.bdate_range(dates.min(), dates.max())
    holidays = all_bdays.difference(dates)
    return weekends.union(holidays)


def compute_rs_moving_averages(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add 150-day and 252-day RS moving averages."""
    df = metrics_df.copy()
    df["MA252"] = df["relative_strength"].rolling(window=252, min_periods=1).mean()
    df["MA150"] = df["relative_strength"].rolling(window=150, min_periods=1).mean()
    return df
