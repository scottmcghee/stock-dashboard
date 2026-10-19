"""Per-contract snapshots from option_chain_data: OI by strike, contract OHLC, OI weights."""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_dashboard import db


def get_tickers_with_option_data() -> list[str]:
    """Distinct underlying symbols present in option_chain_data."""
    query = "SELECT DISTINCT symbol FROM option_chain_data ORDER BY symbol"
    return db.read_df(db.Database.OPTIONS, query)["symbol"].tolist()


def fetch_most_open_interest_symbol() -> str | None:
    """The underlying with the most total open interest on the chain's latest date."""
    query = """
        SELECT symbol FROM option_chain_data
        WHERE collection_date = (SELECT MAX(collection_date) FROM option_chain_data)
        GROUP BY symbol
        ORDER BY SUM(COALESCE(oi, 0)) DESC, symbol
        LIMIT 1
    """
    df = db.read_df(db.Database.OPTIONS, query)
    return None if df.empty else str(df["symbol"].iloc[0])


def get_latest_collection_date(symbol: str) -> pd.Timestamp | None:
    """The newest collection_date with chain rows for ``symbol``."""
    df = db.read_df(
        db.Database.OPTIONS,
        "SELECT MAX(collection_date) AS latest FROM option_chain_data WHERE symbol = %(symbol)s",
        {"symbol": symbol},
    )
    latest = df["latest"].iloc[0] if not df.empty else None
    return None if latest is None or pd.isna(latest) else pd.Timestamp(latest)


def get_expirations(symbol: str) -> list[pd.Timestamp]:
    """Distinct expiration_dates for a symbol, ascending."""
    df = db.read_df(
        db.Database.OPTIONS,
        "SELECT DISTINCT expiration_date FROM option_chain_data "
        "WHERE symbol = %(symbol)s ORDER BY expiration_date",
        {"symbol": symbol},
    )
    return [pd.Timestamp(d) for d in df["expiration_date"]]


def fetch_oi_by_strike(symbol: str, expiration_date: date | str) -> pd.DataFrame:
    """Latest snapshot by strike/right for (symbol, expiry).

    Columns: collection_date (the snapshot's date), strike, option_type, oi, iv.
    """
    query = """
        SELECT collection_date, strike, option_type, COALESCE(oi, 0) AS oi, volatility AS iv
        FROM option_chain_data
        WHERE symbol = %(symbol)s
          AND expiration_date = %(expiration)s
          AND collection_date = (
              SELECT MAX(collection_date) FROM option_chain_data
              WHERE symbol = %(symbol)s AND expiration_date = %(expiration)s
          )
        ORDER BY strike, option_type
    """
    params = {"symbol": symbol, "expiration": expiration_date}
    return db.read_df(db.Database.OPTIONS, query, params)


def fetch_contract_ohlc_history(
    symbol: str, expiration_date: date | str, strike: float, option_type: str
) -> pd.DataFrame:
    """Daily OHLC history for one contract: collection_date, open, high, low, close, oi, day_volume.

    ``price`` (the last trade at the snapshot) is the close (the contract has
    no separate close column). Rows without a positive price are
    dropped, and a missing or non-positive open/high/low falls back to the close.
    """
    df = db.read_df(
        db.Database.OPTIONS,
        """
        SELECT collection_date,
               day_open_price AS open,
               day_high_price AS high,
               day_low_price  AS low,
               price          AS close,
               oi,
               day_volume
        FROM option_chain_data
        WHERE symbol = %(symbol)s
          AND expiration_date = %(expiration)s
          AND strike = %(strike)s
          AND option_type = %(option_type)s
        ORDER BY collection_date
        """,
        {
            "symbol": symbol,
            "expiration": expiration_date,
            "strike": strike,
            "option_type": option_type,
        },
    )
    if df.empty:
        return df
    df = df[df["close"].notna() & (df["close"] > 0)].reset_index(drop=True)
    for col in ("open", "high", "low"):
        missing = df[col].isna() | (df[col] <= 0)
        df.loc[missing, col] = df.loc[missing, "close"]
    return df


def fetch_oi_weights_over_time(symbol: str, expiration_date: date | str) -> pd.DataFrame:
    """Per-day OI by strike/right for (symbol, expiration).

    Columns: collection_date, strike, option_type, weight.
    """
    return db.read_df(
        db.Database.OPTIONS,
        """
        SELECT collection_date, strike, option_type, COALESCE(oi, 0)::float AS weight
        FROM option_chain_data
        WHERE symbol = %(symbol)s AND expiration_date = %(expiration)s
        ORDER BY collection_date, strike, option_type
        """,
        {"symbol": symbol, "expiration": expiration_date},
    )
