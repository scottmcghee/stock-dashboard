"""Data access for per-contract OHLC + OI snapshots from option_chain_data."""

import pandas as pd

from db import option_connection


def get_tickers_with_option_data() -> list[str]:
    """Distinct underlying symbols present in option_chain_data."""
    with option_connection() as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT symbol FROM option_chain_data ORDER BY symbol",
            conn,
        )
        return df["symbol"].tolist()


def get_expirations(symbol: str) -> list[pd.Timestamp]:
    """Distinct expiration_dates for a symbol, ascending."""
    with option_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT expiration_date
            FROM option_chain_data
            WHERE symbol = %s
            ORDER BY expiration_date
            """,
            conn,
            params=[symbol],
        )
        if df.empty:
            return []
        return pd.to_datetime(df["expiration_date"]).tolist()


def fetch_oi_by_strike(symbol: str, expiration_date: str) -> pd.DataFrame:
    """Open interest by strike/right for the latest snapshot of (symbol, expiry).

    Returns columns: strike, option_type, oi. Empty frame if no data.
    """
    with option_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MAX(collection_date) FROM option_chain_data
                WHERE symbol = %s AND expiration_date = %s
                """,
                (symbol, expiration_date),
            )
            row = cur.fetchone()
            latest = row[0] if row else None
        if latest is None:
            return pd.DataFrame(columns=["strike", "option_type", "oi"])

        df = pd.read_sql_query(
            """
            SELECT strike, option_type, COALESCE(oi, 0) AS oi
            FROM option_chain_data
            WHERE symbol = %s
              AND expiration_date = %s
              AND collection_date = %s
            ORDER BY strike, option_type
            """,
            conn,
            params=[symbol, expiration_date, latest],
        )
        return df


def fetch_contract_ohlc_history(
    symbol: str, expiration_date: str, strike: float, option_type: str
) -> pd.DataFrame:
    """Full OHLC history for one contract, ordered by collection_date.

    Returns columns: collection_date, open, high, low, close, oi, day_volume.
    Uses `price` (last trade at snapshot) as close — the vendor's
    `day_close_price` is near-universally 0/NULL and unreliable.
    Drops rows lacking a usable price.
    """
    with option_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT collection_date,
                   day_open_price AS open,
                   day_high_price AS high,
                   day_low_price  AS low,
                   price          AS close,
                   oi,
                   day_volume
            FROM option_chain_data
            WHERE symbol = %s
              AND expiration_date = %s
              AND strike = %s
              AND option_type = %s
            ORDER BY collection_date
            """,
            conn,
            params=[symbol, expiration_date, strike, option_type],
        )
        if df.empty:
            return df
        df["collection_date"] = pd.to_datetime(df["collection_date"])
        df = df[df["close"].notna() & (df["close"] > 0)].reset_index(drop=True)
        for col in ("open", "high", "low"):
            mask = df[col].isna() | (df[col] <= 0)
            df.loc[mask, col] = df.loc[mask, "close"]
        return df


def fetch_oi_weights_over_time(symbol: str, expiration_date: str) -> pd.DataFrame:
    """Per-day OI by strike/right for (symbol, expiration). OI-basis weights."""
    with option_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT collection_date, strike, option_type,
                   COALESCE(oi, 0)::float AS weight
            FROM option_chain_data
            WHERE symbol = %s AND expiration_date = %s
            ORDER BY collection_date, strike, option_type
            """,
            conn,
            params=[symbol, expiration_date],
        )
        if not df.empty:
            df["collection_date"] = pd.to_datetime(df["collection_date"])
        return df


