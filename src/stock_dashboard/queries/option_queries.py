"""Read queries against the per-underlying option tables.

``underlying_data`` holds one row per (symbol, collection_date);
``underlying_data_avg`` holds the trailing average of every numeric column as
``<column>_avg``, keyed by (symbol, date). A value is only ever compared with
the average row of the same symbol and date. Column names chosen in the UI are
checked against the table's numeric columns and quoted as identifiers.

A multiple of the average ("5x its average") only means something for a
column that cannot go negative: :func:`get_option_multiple_columns` lists those.
"""

from __future__ import annotations

import functools
import math
import time
from typing import Any

import pandas as pd
from psycopg2 import sql

from stock_dashboard import db
from stock_dashboard.config import CACHE_TTL_SHORT, get_settings
from stock_dashboard.demo.contract import UNDERLYING_VALUE_COLUMNS

KEY_COLUMNS = frozenset({"collection_date", "symbol", "created_at"})
RECENT_ROWS = 30

# Contract columns that are never negative (docs/DATA_CONTRACT.md): implied
# volatilities, volumes and ratios of non-negative quantities. The signed ones
# (flownet, vflownet, deltas, gammas and the *xoi exposures) are left out: a
# multiple of a signed average can flip sign or explode near zero.
NON_NEGATIVE_COLUMNS = frozenset(
    {
        "volatility",
        "front_volatility",
        "back_volatility",
        "call_volume",
        "put_volume",
        "option_volume",
        "put_call_ratio",
        "flowratio",
        "vflowratio",
        "volm_call_ratio",
        "volm_put_ratio",
    }
)


def get_all_option_symbols() -> list[str]:
    """Distinct symbols in underlying_data, sorted."""
    query = "SELECT DISTINCT symbol FROM underlying_data ORDER BY symbol"
    return db.read_df(db.Database.OPTIONS, query)["symbol"].tolist()


def get_option_numerical_columns() -> list[str]:
    """Numeric column names of underlying_data (excluding keys), sorted."""
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'underlying_data'
          AND table_schema = ANY (current_schemas(false))
          AND data_type IN ('smallint', 'integer', 'bigint', 'numeric', 'double precision', 'real')
          AND column_name <> ALL (%(keys)s)
        ORDER BY column_name
    """
    df = db.read_df(db.Database.OPTIONS, query, {"keys": sorted(KEY_COLUMNS)})
    return df["column_name"].tolist()


def get_option_average_columns() -> list[str]:
    """Numeric columns of underlying_data that have a ``<col>_avg`` partner, sorted."""
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'underlying_data_avg'
          AND table_schema = ANY (current_schemas(false))
    """
    averaged = set(db.read_df(db.Database.OPTIONS, query)["column_name"])
    return [c for c in get_option_numerical_columns() if f"{c}_avg" in averaged]


def has_underlying_rows() -> bool:
    """True when underlying_data holds at least one row."""
    found = db.read_df(
        db.Database.OPTIONS, "SELECT EXISTS (SELECT 1 FROM underlying_data) AS has_rows"
    )
    return bool(found["has_rows"].iloc[0])


def get_option_multiple_columns() -> list[str]:
    """Averaged columns for which "N x its average" means something, sorted.

    The contract's :data:`NON_NEGATIVE_COLUMNS`, plus any column a provider adds
    (not in the contract) that has an ``_avg`` partner and no negative value on
    the latest session.
    """
    averaged = get_option_average_columns()
    known = [c for c in averaged if c in NON_NEGATIVE_COLUMNS]
    extras = [c for c in averaged if c not in UNDERLYING_VALUE_COLUMNS]
    if not extras:
        return known
    query = sql.SQL("""
        SELECT {minima} FROM underlying_data
        WHERE collection_date = (SELECT MAX(collection_date) FROM underlying_data)
    """).format(
        minima=sql.SQL(", ").join(
            sql.SQL("MIN({col}) AS {col}").format(col=sql.Identifier(c)) for c in extras
        )
    )
    minima = db.read_df(db.Database.OPTIONS, query)
    if minima.empty:
        return known
    row = minima.iloc[0]
    non_negative = [c for c in extras if pd.notna(row[c]) and float(row[c]) >= 0]
    return sorted([*known, *non_negative])


# The column whitelist is re-read after this many seconds, so a column added
# while the app runs (or a switch of OPTION_DB_NAME) is picked up without a restart.
ALLOWED_COLUMNS_TTL = CACHE_TTL_SHORT


@functools.lru_cache(maxsize=8)
def _allowed_columns(source: tuple[str, int, str], period: int) -> frozenset[str]:
    """Numeric columns of ``source`` (host, port, database) during one TTL ``period``."""
    return frozenset(get_option_numerical_columns())


def validate_column(column: str) -> str:
    """Return ``column`` if it is a numeric underlying_data column, else raise ValueError."""
    s = get_settings()
    period = int(time.monotonic() // ALLOWED_COLUMNS_TTL)
    if column not in _allowed_columns((s.db_host, s.db_port, s.option_db), period):
        raise ValueError(f"Invalid column name: {column!r}")
    return column


def fetch_option_last_30_days(symbol: str) -> pd.DataFrame:
    """The symbol's most recent RECENT_ROWS underlying_data rows, newest first."""
    query = """
        SELECT * FROM underlying_data
        WHERE symbol = %(symbol)s
        ORDER BY collection_date DESC
        LIMIT %(limit)s
    """
    return db.read_df(db.Database.OPTIONS, query, {"symbol": symbol, "limit": RECENT_ROWS})


def _ratio(value: Any, average: Any) -> float:
    try:
        value, average = float(value), float(average)
    except (TypeError, ValueError):
        return math.nan
    return value / average if average != 0 and not math.isnan(average) else math.nan


def fetch_option_latest_comparison(symbol: str) -> dict[str, float]:
    """Latest value / average for each numeric column, on the symbol's latest date with both rows.

    Returns {} when the symbol has no date with an average row.
    """
    data = db.read_df(
        db.Database.OPTIONS,
        """
        SELECT u.* FROM underlying_data u
        WHERE u.symbol = %(symbol)s
          AND EXISTS (
              SELECT 1 FROM underlying_data_avg a
              WHERE a.symbol = u.symbol AND a.date = u.collection_date
          )
        ORDER BY u.collection_date DESC
        LIMIT 1
        """,
        {"symbol": symbol},
    )
    if data.empty:
        return {}
    avg = db.read_df(
        db.Database.OPTIONS,
        "SELECT * FROM underlying_data_avg WHERE symbol = %(symbol)s AND date = %(date)s",
        {"symbol": symbol, "date": data["collection_date"].iloc[0].date()},
    )
    if avg.empty:
        return {}
    data_row, avg_row = data.iloc[0], avg.iloc[0]
    return {
        col: _ratio(data_row[col], avg_row[f"{col}_avg"])
        for col in data.columns
        if col not in KEY_COLUMNS and f"{col}_avg" in avg.columns
    }


def fetch_option_column_history(symbol: str, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full history of one underlying_data column and of its average.

    Returns (collection_date, col) and (date, col_avg) frames; the average frame
    is empty when the column has no ``_avg`` partner.
    """
    base_col = validate_column(column)
    data_query = sql.SQL(
        "SELECT collection_date, {col} FROM underlying_data "
        "WHERE symbol = %(symbol)s ORDER BY collection_date"
    ).format(col=sql.Identifier(base_col))
    params = {"symbol": symbol}
    values = db.read_df(db.Database.OPTIONS, data_query, params)
    if base_col not in get_option_average_columns():
        return values, pd.DataFrame(columns=["date", f"{base_col}_avg"])
    avg_query = sql.SQL(
        "SELECT date, {col} FROM underlying_data_avg WHERE symbol = %(symbol)s ORDER BY date"
    ).format(col=sql.Identifier(f"{base_col}_avg"))
    return values, db.read_df(db.Database.OPTIONS, avg_query, params)


def fetch_option_screener_results(column: str, min_multiple: float) -> pd.DataFrame:
    """Symbols whose latest ``column`` / same-day average >= ``min_multiple``, highest first.

    Only rows with a positive average count (meant for :func:`get_option_multiple_columns`).
    Columns: symbol, current_value, avg_value, multiple. Raises ValueError for a
    column that is not numeric or has no ``_avg`` partner.
    """
    if validate_column(column) not in get_option_average_columns():
        raise ValueError(f"Column {column!r} has no {column}_avg average")
    col = sql.Identifier(column)
    avg_col = sql.Identifier(f"{column}_avg")
    query = sql.SQL("""
        SELECT d.symbol,
               d.{col} AS current_value,
               a.{avg_col} AS avg_value,
               d.{col}::float / a.{avg_col} AS multiple
        FROM underlying_data d
        JOIN underlying_data_avg a ON a.symbol = d.symbol AND a.date = d.collection_date
        WHERE d.collection_date = (SELECT MAX(collection_date) FROM underlying_data)
          AND a.{avg_col} > 0
          AND d.{col}::float / a.{avg_col} >= %(min_multiple)s
        ORDER BY multiple DESC
    """).format(col=col, avg_col=avg_col)
    return db.read_df(db.Database.OPTIONS, query, {"min_multiple": min_multiple})
