"""Chart utilities — pure functions, no DB or UI."""

import numpy as np
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


_IMPULSE_WINDOWS = [5, 10, 20]


def compute_etf_impulse(rank_history: pd.DataFrame) -> pd.DataFrame:
    """Compute z-scored RS rank impulse for each ETF.

    For each ETF, computes the change in rs_rank_index over N-day windows,
    then z-scores each impulse against that ETF's own impulse history.
    A high z-score means the ETF is accelerating unusually fast for its
    own typical behavior.

    Args:
        rank_history: DataFrame with columns [date, etf_ticker, rs_rank_index],
                      sorted by etf_ticker, date.

    Returns:
        DataFrame with one row per ETF (latest date), columns:
        etf_ticker, impulse_Nd, impulse_Nd_z for each window N.
    """
    if rank_history.empty:
        return pd.DataFrame()

    rows = []
    for ticker, group in rank_history.groupby("etf_ticker"):
        g = group.sort_values("date")
        rs = g["rs_rank_index"].astype(float)
        row: dict = {"etf_ticker": ticker}
        for w in _IMPULSE_WINDOWS:
            impulse = rs - rs.shift(w)
            latest = impulse.iloc[-1] if len(impulse) > w else np.nan
            mu = impulse.mean()
            sigma = impulse.std()
            z = (latest - mu) / sigma if sigma and sigma > 0 else 0.0
            row[f"impulse_{w}d"] = round(float(latest), 4) if not np.isnan(latest) else np.nan
            row[f"impulse_{w}d_z"] = round(float(z), 2) if not np.isnan(latest) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def compute_52w_highs_lows(
    df: pd.DataFrame,
    display_start: pd.Timestamp | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute daily 52W high/low counts and current lists.

    Args:
        df: DataFrame with columns [date, ticker, high, low].
        display_start: Only return counts/lists from this date onward
                       (earlier data is used for lookback).

    Returns:
        (daily_counts, highs_list, lows_list):
        - daily_counts: date, new_highs, new_lows
        - highs_list: tickers at 52W high on latest date
        - lows_list: tickers at 52W low on latest date
    """
    if df.empty:
        empty_counts = pd.DataFrame(columns=["date", "new_highs", "new_lows"])
        empty_list = pd.DataFrame(columns=["ticker"])
        return empty_counts, empty_list, empty_list

    display_start = pd.Timestamp(display_start)
    df = df.sort_values(["ticker", "date"])

    # Rolling 252-day max(high) and min(low) per ticker
    grouped = df.groupby("ticker")
    df["rolling_high"] = grouped["high"].transform(
        lambda s: s.rolling(252, min_periods=200).max()
    )
    df["rolling_low"] = grouped["low"].transform(
        lambda s: s.rolling(252, min_periods=200).min()
    )

    df["is_52w_high"] = df["high"] >= df["rolling_high"]
    df["is_52w_low"] = df["low"] <= df["rolling_low"]

    # Filter to display range
    display = df[df["date"] >= display_start].copy()

    # Daily counts
    daily_counts = display.groupby("date").agg(
        new_highs=("is_52w_high", "sum"),
        new_lows=("is_52w_low", "sum"),
    ).reset_index()
    daily_counts = daily_counts.sort_values("date")

    # Latest date lists
    latest_date = display["date"].max()
    latest = display[display["date"] == latest_date]
    highs_list = latest[latest["is_52w_high"]][["ticker"]].reset_index(drop=True)
    lows_list = latest[latest["is_52w_low"]][["ticker"]].reset_index(drop=True)

    return daily_counts, highs_list, lows_list
