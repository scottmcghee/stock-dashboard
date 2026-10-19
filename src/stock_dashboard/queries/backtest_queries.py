"""Data loading for the backtesting engine, cached as parquet files.

Full-history loads take a while, so each result is written to ``CACHE_DIR``
and reused for :data:`CACHE_TTL_SECONDS`. Cache keys include the database
host, port and name and the benchmark, so switching databases (e.g. demo vs
real data) or benchmark never serves another source's cache.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from psycopg2 import sql

from stock_dashboard import db
from stock_dashboard.config import get_settings
from stock_dashboard.queries.stocks_queries import fetch_ohlcv

CACHE_TTL_SECONDS = 24 * 60 * 60
# Bump when a loader's output changes shape so stale files are not reused.
CACHE_VERSION = 3


def cache_path(name: str, etf_filter: str | None = None) -> Path:
    """Cache file for loader ``name``, keyed by data source, benchmark and ETF filter."""
    s = get_settings()
    key = "|".join(
        [
            str(CACHE_VERSION),
            name,
            s.db_host,
            str(s.db_port),
            s.stocks_db,
            s.benchmark,
            etf_filter or "all",
        ]
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return s.cache_dir / f"{name}_{digest}.parquet"


def _is_cache_fresh(path: Path) -> bool:
    """True if ``path`` exists and is younger than CACHE_TTL_SECONDS."""
    return path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL_SECONDS


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Store tickers as categories and float64 columns as float32."""
    if df.empty:
        return df
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype("category")
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def _cached(path: Path, load: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    if _is_cache_fresh(path):
        return pd.read_parquet(path)
    df = _optimize_dtypes(load())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _etf_members(etf: str) -> list[str]:
    df = db.read_df(
        db.Database.STOCKS,
        "SELECT DISTINCT ticker FROM fact_stock_metrics WHERE etf_ticker = %(etf)s",
        {"etf": etf},
    )
    return df["ticker"].tolist()


def load_ohlcv_universe(etf_filter: str | None = None) -> pd.DataFrame:
    """Full OHLCV history for every stock, or for one ETF's members.

    Returns: ticker, date, open, high, low, close, volume.
    """

    def load() -> pd.DataFrame:
        tickers = _etf_members(etf_filter) if etf_filter else None
        return fetch_ohlcv("stock", tickers)

    return _cached(cache_path("ohlcv", etf_filter), load)


def load_metrics_universe(etf_filter: str | None = None) -> pd.DataFrame:
    """fact_stock_metrics aggregated to one row per (date, ticker).

    A stock has a row per ETF that holds it. ``rs_rank_overall`` is the same in
    each; ``rs_z_score`` is averaged across them.
    Returns: date, ticker, rs_rank_overall, rs_z_score.
    """

    def load() -> pd.DataFrame:
        where = sql.SQL("WHERE etf_ticker = %(etf)s" if etf_filter else "")
        query = sql.SQL("""
            SELECT date, ticker,
                   AVG(rs_rank_overall) AS rs_rank_overall,
                   AVG(rs_z_score) AS rs_z_score
            FROM fact_stock_metrics
            {where}
            GROUP BY date, ticker
            ORDER BY date, ticker
        """).format(where=where)
        return db.read_df(db.Database.STOCKS, query, {"etf": etf_filter} if etf_filter else None)

    return _cached(cache_path("metrics", etf_filter), load)


def load_benchmark_ohlcv() -> pd.DataFrame:
    """OHLCV of the configured benchmark (``BENCHMARK_TICKER``), for regime classification."""
    benchmark = get_settings().benchmark
    return _cached(
        cache_path("benchmark_ohlcv"),
        lambda: fetch_ohlcv("etf", [benchmark]).drop(columns="ticker"),
    )


def clear_cache() -> int:
    """Remove all cached parquet files. Returns count of files removed."""
    cache_dir = get_settings().cache_dir
    if not cache_dir.exists():
        return 0
    files = list(cache_dir.glob("*.parquet"))
    for f in files:
        f.unlink()
    return len(files)
