"""Data access layer for the option_data database (SQL-injection-safe)."""

import numpy as np
import pandas as pd
from psycopg2 import sql

from db import option_connection

# Whitelist of allowed column names for dynamic queries
_ALLOWED_COLUMNS: set[str] | None = None


def _get_allowed_columns() -> set[str]:
    """Fetch and cache the set of valid numerical columns from underlying_data."""
    global _ALLOWED_COLUMNS
    if _ALLOWED_COLUMNS is not None:
        return _ALLOWED_COLUMNS
    _ALLOWED_COLUMNS = set(get_option_numerical_columns())
    return _ALLOWED_COLUMNS


def _validate_column(column: str) -> str:
    """Validate that a column name is in the whitelist."""
    allowed = _get_allowed_columns()
    if column not in allowed:
        raise ValueError(f"Invalid column name: {column!r}")
    return column


def get_all_option_symbols() -> list[str]:
    """Get all unique symbols from underlying_data."""
    with option_connection() as conn:
        query = "SELECT DISTINCT symbol FROM underlying_data ORDER BY symbol"
        df = pd.read_sql_query(query, conn)
        return df["symbol"].tolist()


def get_option_numerical_columns() -> list[str]:
    """Get all numerical column names from underlying_data."""
    with option_connection() as conn:
        query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'underlying_data'
          AND table_schema = 'public'
          AND data_type IN ('integer', 'bigint', 'numeric', 'double precision', 'real')
          AND column_name NOT IN ('collection_date', 'symbol', 'created_at')
        ORDER BY column_name
        """
        df = pd.read_sql_query(query, conn)
        return df["column_name"].tolist()


def fetch_option_last_30_days(symbol: str) -> pd.DataFrame:
    """Fetch last 30 rows of data for a symbol from underlying_data."""
    with option_connection() as conn:
        query = """
        SELECT * FROM underlying_data
        WHERE symbol = %s
        ORDER BY collection_date DESC
        LIMIT 30
        """
        df = pd.read_sql_query(query, conn, params=[symbol])
        if not df.empty:
            df["collection_date"] = pd.to_datetime(df["collection_date"])
            df = df.sort_values("collection_date", ascending=False)
        return df


def fetch_option_latest_comparison(symbol: str) -> dict:
    """Fetch latest value/avg ratio for each numeric column."""
    with option_connection() as conn:
        # Get latest date for this symbol
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(collection_date) FROM underlying_data WHERE symbol = %s",
                (symbol,),
            )
            latest_date = cur.fetchone()[0]
        if not latest_date:
            return {}

        df_data = pd.read_sql_query(
            "SELECT * FROM underlying_data WHERE symbol = %s AND collection_date = %s",
            conn,
            params=[symbol, latest_date],
        )
        df_avg = pd.read_sql_query(
            "SELECT * FROM underlying_data_avg WHERE symbol = %s AND date = %s",
            conn,
            params=[symbol, latest_date],
        )
        if df_data.empty or df_avg.empty:
            return {}

        data_row = df_data.iloc[0]
        avg_row = df_avg.iloc[0]
        ratio_dict: dict = {}
        for col in df_data.columns:
            if col in ("collection_date", "symbol", "created_at"):
                continue
            avg_col = col + "_avg"
            if avg_col in df_avg.columns:
                try:
                    avg_val = avg_row[avg_col]
                    if avg_val not in (0, None) and not (isinstance(avg_val, float) and np.isnan(avg_val)):
                        ratio_dict[col] = data_row[col] / avg_val
                    else:
                        ratio_dict[col] = np.nan
                except Exception:
                    ratio_dict[col] = np.nan
        return ratio_dict


def fetch_option_column_history(
    symbol: str, column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch full history for a symbol/column using safe SQL identifiers."""
    base_col = column.replace("_avg", "")
    _validate_column(base_col)
    avg_col = base_col + "_avg"

    with option_connection() as conn:
        data_query = sql.SQL(
            "SELECT collection_date, {col} FROM underlying_data "
            "WHERE symbol = %(symbol)s ORDER BY collection_date"
        ).format(col=sql.Identifier(base_col))

        avg_query = sql.SQL(
            "SELECT date, {col} FROM underlying_data_avg "
            "WHERE symbol = %(symbol)s ORDER BY date"
        ).format(col=sql.Identifier(avg_col))

        with conn.cursor() as cur:
            cur.execute(data_query, {"symbol": symbol})
            cols_data = [desc[0] for desc in cur.description]
            df_data = pd.DataFrame(cur.fetchall(), columns=cols_data)

            cur.execute(avg_query, {"symbol": symbol})
            cols_avg = [desc[0] for desc in cur.description]
            df_avg = pd.DataFrame(cur.fetchall(), columns=cols_avg)

        if not df_data.empty:
            df_data["collection_date"] = pd.to_datetime(df_data["collection_date"])
        if not df_avg.empty:
            df_avg["date"] = pd.to_datetime(df_avg["date"])
        return df_data, df_avg


def fetch_option_screener_results(
    column: str, min_multiple: float
) -> pd.DataFrame:
    """Get symbols where latest value / 30d avg exceeds min_multiple.

    Uses psycopg2.sql.Identifier for column names (SQL-injection safe).
    """
    _validate_column(column)
    col_id = sql.Identifier(column)
    avg_col_id = sql.Identifier(column + "_avg")

    with option_connection() as conn:
        # Get latest date
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(collection_date) FROM underlying_data")
            latest_date = cur.fetchone()[0]
        if not latest_date:
            return pd.DataFrame()

        query = sql.SQL("""
        WITH latest_data AS (
            SELECT symbol, {col} FROM underlying_data
            WHERE collection_date = %(d1)s
        ),
        latest_avg AS (
            SELECT symbol, {avg_col} FROM underlying_data_avg
            WHERE date = %(d2)s
        )
        SELECT
            d.symbol,
            d.{col} as current_value,
            a.{avg_col} as avg_value,
            CASE WHEN a.{avg_col} IS NOT NULL AND a.{avg_col} != 0
                 THEN d.{col} / a.{avg_col}
                 ELSE NULL
            END as multiple
        FROM latest_data d
        JOIN latest_avg a ON d.symbol = a.symbol
        WHERE a.{avg_col} IS NOT NULL
          AND a.{avg_col} != 0
          AND d.{col} / a.{avg_col} >= %(min_mult)s
        ORDER BY multiple DESC
        """).format(col=col_id, avg_col=avg_col_id)

        with conn.cursor() as cur:
            cur.execute(query, {
                "d1": latest_date,
                "d2": latest_date,
                "min_mult": min_multiple,
            })
            cols = [desc[0] for desc in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)
