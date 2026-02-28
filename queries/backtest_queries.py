"""Data loading for the backtesting engine with parquet caching."""

import hashlib
import time
from pathlib import Path

import pandas as pd

from db import stocks_connection

CACHE_DIR = Path("~/.cache/stock_dashboard").expanduser()
CACHE_TTL_SECONDS = 86400  # 1 day


def _cache_path(name: str, etf_filter: str | None) -> Path:
    """Build a cache file path keyed by name and etf_filter."""
    key = f"{name}_{etf_filter or 'all'}"
    hashed = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{name}_{hashed}.parquet"


def _is_cache_fresh(path: Path) -> bool:
    """Check if a cache file exists and is younger than CACHE_TTL_SECONDS."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < CACHE_TTL_SECONDS


def load_ohlcv_universe(etf_filter: str | None = None) -> pd.DataFrame:
    """Load full OHLCV history for all stocks (or one ETF's components).

    Returns: ticker, date, open, high, low, close, volume
    Caches to parquet keyed by etf_filter. Cache TTL: 1 day.
    """
    cache = _cache_path("ohlcv", etf_filter)
    if _is_cache_fresh(cache):
        return pd.read_parquet(cache)

    with stocks_connection() as conn:
        if etf_filter:
            query = """
            SELECT o.ticker, o.date, o.open, o.high, o.low, o.close, o.volume
            FROM fact_commons_ohlc o
            JOIN (
                SELECT DISTINCT ticker
                FROM fact_stock_metrics
                WHERE etf_ticker = %(etf)s
            ) m ON o.ticker = m.ticker
            ORDER BY o.ticker, o.date
            """
            df = pd.read_sql_query(query, conn, params={"etf": etf_filter})
        else:
            query = """
            SELECT ticker, date, open, high, low, close, volume
            FROM fact_commons_ohlc
            ORDER BY ticker, date
            """
            df = pd.read_sql_query(query, conn)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def load_metrics_universe(etf_filter: str | None = None) -> pd.DataFrame:
    """Load full fact_stock_metrics history.

    Returns: date, ticker, etf_ticker, rs_rank_index, rs_rank_overall, rs_z_score
    Caches to parquet keyed by etf_filter. Cache TTL: 1 day.
    """
    cache = _cache_path("metrics", etf_filter)
    if _is_cache_fresh(cache):
        return pd.read_parquet(cache)

    with stocks_connection() as conn:
        if etf_filter:
            query = """
            SELECT date, ticker, etf_ticker, rs_rank_index, rs_rank_overall, rs_z_score
            FROM fact_stock_metrics
            WHERE etf_ticker = %(etf)s
            ORDER BY date, ticker
            """
            df = pd.read_sql_query(query, conn, params={"etf": etf_filter})
        else:
            query = """
            SELECT date, ticker, etf_ticker, rs_rank_index, rs_rank_overall, rs_z_score
            FROM fact_stock_metrics
            ORDER BY date, ticker
            """
            df = pd.read_sql_query(query, conn)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def load_spy_ohlcv() -> pd.DataFrame:
    """Load SPY OHLCV from fact_index_ohlc for regime classification."""
    cache = _cache_path("spy_ohlcv", None)
    if _is_cache_fresh(cache):
        return pd.read_parquet(cache)

    with stocks_connection() as conn:
        query = """
        SELECT date, open, high, low, close, volume
        FROM fact_index_ohlc
        WHERE ticker = 'SPY'
        ORDER BY date
        """
        df = pd.read_sql_query(query, conn)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def clear_cache() -> int:
    """Remove all cached parquet files. Returns count of files removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.parquet"):
        f.unlink()
        count += 1
    return count
