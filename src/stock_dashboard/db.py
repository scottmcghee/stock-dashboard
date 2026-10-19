"""PostgreSQL access for the two databases the dashboard reads.

Every query goes through :func:`read_df`, which ends its transaction before
returning, so a pooled connection never sits "idle in transaction" between
Streamlit reruns. NUMERIC values arrive as floats and DATE/TIMESTAMP columns
as pandas datetimes, so callers get ready-to-use frames.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, cast

import pandas as pd
import psycopg2
import psycopg2.extensions
from psycopg2 import sql
from psycopg2.pool import ThreadedConnectionPool

from stock_dashboard.config import get_settings

logger = logging.getLogger(__name__)


class Database(StrEnum):
    STOCKS = "stocks"
    OPTIONS = "options"


# The option-data contract: pages that need option data check for all three.
OPTION_CONTRACT_TABLES = ("option_chain_data", "underlying_data", "underlying_data_avg")

# Streamlit serves each session on its own thread, so the pool must be thread-safe.
# No idle connections are kept (minconn=0): a connection the server dropped while
# the app sat idle is never handed out again.
MAX_CONNECTIONS = 8

_DATETIME_TYPE_CODES = frozenset(
    psycopg2.extensions.DATE.values  # noqa: PD011 - psycopg2 type OIDs, not pandas
    + psycopg2.extensions.PYDATETIME.values  # noqa: PD011
)
_NUMBER_TYPE_CODES = frozenset(
    psycopg2.extensions.INTEGER.values  # noqa: PD011
    + psycopg2.extensions.LONGINTEGER.values  # noqa: PD011
    + psycopg2.extensions.FLOAT.values  # noqa: PD011
    + psycopg2.extensions.DECIMAL.values  # noqa: PD011
)


def _numeric_to_float(value: str | bytes | None, _cursor: object) -> float | None:
    return None if value is None else float(value)


_NUMERIC_AS_FLOAT = psycopg2.extensions.new_type(
    psycopg2.extensions.DECIMAL.values, "NUMERIC_AS_FLOAT", _numeric_to_float
)

_Params = Sequence[Any] | Mapping[str, Any] | None

_pools: dict[Database, ThreadedConnectionPool] = {}
_pools_lock = threading.Lock()


def _database_name(database: Database) -> str:
    settings = get_settings()
    return settings.stocks_db if database is Database.STOCKS else settings.option_db


def _get_pool(database: Database) -> ThreadedConnectionPool:
    with _pools_lock:
        pool = _pools.get(database)
        if pool is None or pool.closed:
            kwargs = get_settings().db_kwargs(_database_name(database))
            pool = ThreadedConnectionPool(0, MAX_CONNECTIONS, **kwargs)
            _pools[database] = pool
        return pool


@contextmanager
def connection(database: Database) -> Iterator[psycopg2.extensions.connection]:
    """Lend a connection to ``database`` for the duration of a ``with`` block."""
    pool = _get_pool(database)
    conn = pool.getconn()
    try:
        psycopg2.extensions.register_type(_NUMERIC_AS_FLOAT, conn)
        yield conn
    finally:
        pool.putconn(conn, close=bool(conn.closed))


def read_df(database: Database, query: str | sql.Composable, params: object = None) -> pd.DataFrame:
    """Run a read query and return a DataFrame; always ends the transaction.

    The transaction is rolled back after the rows are fetched, so the pooled
    connection is never left idle in transaction.
    """
    with connection(database) as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(query, cast(_Params, params))
                description: Any = cur.description or ()
                columns = [col[0] for col in description]
                type_codes = [col[1] for col in description]
                rows = cur.fetchall()
        finally:
            if not conn.closed:
                conn.rollback()
    df = pd.DataFrame.from_records(rows, columns=columns)
    for name, type_code in zip(columns, type_codes, strict=True):
        if type_code in _DATETIME_TYPE_CODES:
            df[name] = pd.to_datetime(df[name])
        elif type_code in _NUMBER_TYPE_CODES and df[name].dtype == object:
            df[name] = pd.to_numeric(df[name])  # an all-NULL column arrives as None objects
    return df


def option_data_available() -> bool:
    """True when the option database is reachable and has the three contract tables."""
    query = """
        SELECT COUNT(DISTINCT table_name)
        FROM information_schema.tables
        WHERE table_schema = ANY (current_schemas(false))
          AND table_name = ANY (%s)
    """
    try:
        found = read_df(Database.OPTIONS, query, (list(OPTION_CONTRACT_TABLES),))
    except psycopg2.Error as exc:
        logger.info("Option database unavailable: %s", exc)
        return False
    return bool(found.iloc[0, 0] == len(OPTION_CONTRACT_TABLES))


def close_all_pools() -> None:
    """Close every connection pool. Call on app shutdown."""
    with _pools_lock:
        for pool in _pools.values():
            if not pool.closed:
                pool.closeall()
        _pools.clear()
