"""Chart utilities: pure functions, no DB or UI."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

# A 52-week high/low is the extreme of the trailing 252 sessions; a ticker needs
# at least 200 of them before its first new high or low is reported.
FIFTY_TWO_WEEK_BARS = 252
FIFTY_TWO_WEEK_MIN_BARS = 200
RS_MA_SHORT = 150
RS_MA_LONG = 252


# Bar intervals offered in the UI, as pandas resampling rules.
RESAMPLE_RULES = {"W": "W", "M": "ME"}


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
    resampled = df.resample(RESAMPLE_RULES[interval], label="right", closed="right").agg(agg_dict)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    return resampled


def get_non_trading_days(dates: pd.Series) -> pd.DatetimeIndex:
    """Compute non-trading days (weekends + holidays) for rangebreak gaps."""
    all_dates = pd.date_range(dates.min(), dates.max())
    weekends = all_dates[all_dates.weekday >= 5]
    all_bdays = pd.bdate_range(dates.min(), dates.max())
    holidays = all_bdays.difference(pd.DatetimeIndex(dates))
    return weekends.union(holidays)


def compute_rs_moving_averages(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add the 150- and 252-bar moving averages of relative strength as MA150 and MA252.

    ``metrics_df`` is one series (one ticker and ETF) sorted by date; an average
    is NaN until its window is full, so pass history from before the first
    displayed date.
    """
    df = metrics_df.copy()
    rs = df["relative_strength"]
    df[f"MA{RS_MA_LONG}"] = rs.rolling(window=RS_MA_LONG).mean()
    df[f"MA{RS_MA_SHORT}"] = rs.rolling(window=RS_MA_SHORT).mean()
    return df


def trim_to_range(
    df: pd.DataFrame,
    start: date | str | None = None,
    end: date | str | None = None,
    column: str = "date",
) -> pd.DataFrame:
    """Rows of ``df`` whose ``column`` lies within [start, end] (None = unbounded)."""
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df[column] >= pd.Timestamp(start)
    if end is not None:
        mask &= df[column] <= pd.Timestamp(end)
    return df[mask]


def best_etf(metrics: pd.DataFrame) -> str | None:
    """The ETF in which the stock ranks highest (``rs_rank_index``) on its latest date.

    Ties and missing ranks are broken by ETF ticker; None when there are no rows.
    """
    rows = metrics.dropna(subset=["etf_ticker"])
    if rows.empty:
        return None
    latest = rows[rows["date"] == rows["date"].max()]
    ordered = latest.sort_values(
        ["rs_rank_index", "etf_ticker"], ascending=[False, True], na_position="last"
    )
    return str(ordered["etf_ticker"].iloc[0])


# Signal markers sit just outside the bar so they do not hide it.
MARKER_OFFSET = 0.005


def signal_marker_prices(bars: pd.DataFrame, bullish: bool) -> pd.Series:
    """Marker y-values for signal bars: just below the low (bullish) or above the high."""
    if bullish:
        return bars["low"] * (1 - MARKER_OFFSET)
    return bars["high"] * (1 + MARKER_OFFSET)


SIGMA = "\N{GREEK SMALL LETTER SIGMA}"
SIGMA_BAND_MULTIPLES = (1, 2)


def sigma_bands(
    values: pd.Series, multiples: tuple[int, ...] = SIGMA_BAND_MULTIPLES
) -> list[tuple[str, float]]:
    """(label, level) pairs at mean +/- k standard deviations of ``values``, NaNs ignored.

    Empty when fewer than two values are present.
    """
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return []
    mean, std = float(clean.mean()), float(clean.std())
    bands: list[tuple[str, float]] = []
    for k in multiples:
        bands.append((f"+{k}{SIGMA}", mean + k * std))
        bands.append((f"-{k}{SIGMA}", mean - k * std))
    return bands


IMPULSE_WINDOWS = (5, 10, 20)


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
        row: dict[str, Any] = {"etf_ticker": ticker}
        for w in IMPULSE_WINDOWS:
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
        lambda s: s.rolling(FIFTY_TWO_WEEK_BARS, min_periods=FIFTY_TWO_WEEK_MIN_BARS).max()
    )
    df["rolling_low"] = grouped["low"].transform(
        lambda s: s.rolling(FIFTY_TWO_WEEK_BARS, min_periods=FIFTY_TWO_WEEK_MIN_BARS).min()
    )

    df["is_52w_high"] = df["high"] >= df["rolling_high"]
    df["is_52w_low"] = df["low"] <= df["rolling_low"]

    # Filter to display range
    display = df[df["date"] >= display_start].copy()

    # Daily counts
    daily_counts = (
        display.groupby("date")
        .agg(
            new_highs=("is_52w_high", "sum"),
            new_lows=("is_52w_low", "sum"),
        )
        .reset_index()
    )
    daily_counts = daily_counts.sort_values("date")

    # Latest date lists
    latest_date = display["date"].max()
    latest = display[display["date"] == latest_date]
    highs_list = latest[latest["is_52w_high"]][["ticker"]].reset_index(drop=True)
    lows_list = latest[latest["is_52w_low"]][["ticker"]].reset_index(drop=True)

    return daily_counts, highs_list, lows_list
