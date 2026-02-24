"""Connection pooling for stocks and option_data databases."""

from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.pool import SimpleConnectionPool

from config import STOCKS_DB_CONFIG, OPTION_DB_CONFIG

_stocks_pool: SimpleConnectionPool | None = None
_option_pool: SimpleConnectionPool | None = None


def _get_stocks_pool() -> SimpleConnectionPool:
    global _stocks_pool
    if _stocks_pool is None or _stocks_pool.closed:
        _stocks_pool = SimpleConnectionPool(minconn=1, maxconn=5, **STOCKS_DB_CONFIG)
    return _stocks_pool


def _get_option_pool() -> SimpleConnectionPool:
    global _option_pool
    if _option_pool is None or _option_pool.closed:
        _option_pool = SimpleConnectionPool(minconn=1, maxconn=5, **OPTION_DB_CONFIG)
    return _option_pool


@contextmanager
def stocks_connection() -> Generator:
    """Context manager that checks out a connection from the stocks pool."""
    pool = _get_stocks_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


@contextmanager
def option_connection() -> Generator:
    """Context manager that checks out a connection from the option_data pool."""
    pool = _get_option_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def close_all_pools() -> None:
    """Close both connection pools. Call on app shutdown."""
    global _stocks_pool, _option_pool
    if _stocks_pool is not None and not _stocks_pool.closed:
        _stocks_pool.closeall()
    if _option_pool is not None and not _option_pool.closed:
        _option_pool.closeall()
