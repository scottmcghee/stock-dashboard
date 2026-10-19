"""Read queries against the stocks database (rs-pipeline's schema).

"Latest" always means the newest date in the table being read, never the
calendar date, and every ``fact_etf_metrics`` read is restricted to the
configured benchmark (``BENCHMARK_TICKER``).

A stock belongs to every ETF that holds it, so ``fact_stock_metrics`` has one
row per (date, ticker, ETF). Functions that return one row per ticker keep
the ETF in which the stock ranks highest (``rs_rank_index``) and list all of
its ETFs in an ``etfs`` column; ``rs_rank_overall`` is the same in every row.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal

import pandas as pd
from psycopg2 import sql

from stock_dashboard import db
from stock_dashboard.config import get_settings
from stock_dashboard.logic.weinstein import WEINSTEIN_LOOKBACK_BARS, classify_bulk_stages

OhlcvKind = Literal["stock", "etf"]

OHLCV_TABLES: dict[OhlcvKind, str] = {"stock": "fact_commons_ohlc", "etf": "fact_index_ohlc"}
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

# Volume statistics in the screener, in trading sessions ending on the latest bar.
VOLUME_STATS_BARS = 252  # about a year: mean and std for the volume z-score
RECENT_VOLUME_BARS = 30  # "avg_30d_volume"

# last_bars scans at most this many times as many recent sessions as it returns.
LAST_BARS_PREFILTER_FACTOR = 2

METRIC_COLUMNS = ("relative_strength", "rs_rank_index", "rs_rank_overall", "rs_z_score")


def get_all_tickers() -> list[str]:
    """All stock tickers in dim_stock, sorted."""
    df = db.read_df(db.Database.STOCKS, "SELECT ticker FROM dim_stock ORDER BY ticker")
    return df["ticker"].tolist()


def get_all_etfs() -> list[str]:
    """ETF tickers that have stock members in fact_stock_metrics, sorted."""
    query = "SELECT DISTINCT etf_ticker FROM fact_stock_metrics ORDER BY etf_ticker"
    return db.read_df(db.Database.STOCKS, query)["etf_ticker"].tolist()


def _date_filters(
    start: date | None, end: date | None, column: str = "date"
) -> tuple[list[sql.Composable], dict[str, object]]:
    clauses: list[sql.Composable] = []
    params: dict[str, object] = {}
    if start is not None:
        clauses.append(sql.SQL("{} >= %(start)s").format(sql.Identifier(column)))
        params["start"] = start
    if end is not None:
        clauses.append(sql.SQL("{} <= %(end)s").format(sql.Identifier(column)))
        params["end"] = end
    return clauses, params


def _where(clauses: Sequence[sql.Composable]) -> sql.Composable:
    if not clauses:
        return sql.SQL("")
    return sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses)


def _session_floor(
    table: sql.Identifier, limit_placeholder: str, condition: sql.Composable | None = None
) -> sql.Composable:
    """The earliest of the table's most recent N distinct dates (optionally filtered).

    N is bound through ``limit_placeholder``; the expression is NULL when no date qualifies.
    """
    where = sql.SQL(" WHERE ") + condition if condition is not None else sql.SQL("")
    return sql.SQL(
        "(SELECT MIN(date) FROM (SELECT DISTINCT date FROM {table}{where} "
        "ORDER BY date DESC LIMIT {limit}) sessions)"
    ).format(table=table, where=where, limit=sql.SQL(limit_placeholder))


def fetch_ohlcv(
    kind: OhlcvKind,
    tickers: Sequence[str] | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
    last_bars: int | None = None,
    warmup_bars: int = 0,
    columns: Sequence[str] = OHLCV_COLUMNS,
) -> pd.DataFrame:
    """Daily bars for stocks (``fact_commons_ohlc``) or ETFs and benchmarks (``fact_index_ohlc``).

    Args:
        kind: "stock" or "etf".
        tickers: Restrict to these tickers (None = every ticker; an empty list = none).
        start, end: Inclusive date bounds.
        last_bars: Keep only each ticker's last N bars (within the date bounds), the
            way indicator look-backs are sized.
        warmup_bars: Also return the N trading sessions of the table before ``start``,
            so rolling indicators are defined from ``start`` on.
        columns: Price/volume columns to return besides ticker and date.

    Returns:
        ticker, date and ``columns``, sorted by ticker and date.
    """
    unknown = set(columns) - set(OHLCV_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown OHLCV columns: {sorted(unknown)}")
    if tickers is not None and not tickers:
        return pd.DataFrame(columns=["ticker", "date", *columns])

    table = sql.Identifier(OHLCV_TABLES[kind])
    clauses, params = _date_filters(None, end)
    if tickers is not None:
        clauses.append(sql.SQL("ticker = ANY(%(tickers)s)"))
        params["tickers"] = list(tickers)
    if start is not None:
        params["start"] = start
        if warmup_bars > 0:
            floor = _session_floor(table, "%(warmup)s", sql.SQL("date < %(start)s"))
            clauses.append(sql.SQL("date >= COALESCE({}, %(start)s)").format(floor))
            params["warmup"] = warmup_bars
        else:
            clauses.append(sql.SQL("date >= %(start)s"))

    value_cols = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    if last_bars is None:
        query = sql.SQL(
            "SELECT ticker, date, {cols} FROM {table}{where} ORDER BY ticker, date"
        ).format(cols=value_cols, table=table, where=_where(clauses))
    else:
        # Bound the window scan to recent sessions; the margin lets a ticker
        # with missing sessions still reach last_bars rows.
        end_bound = sql.SQL("date <= %(end)s") if end is not None else None
        floor = _session_floor(table, "%(prefilter)s", end_bound)
        clauses.append(sql.SQL("date >= {}").format(floor))
        params["prefilter"] = LAST_BARS_PREFILTER_FACTOR * last_bars
        query = sql.SQL(
            "SELECT ticker, date, {cols} FROM ("
            "SELECT ticker, date, {cols}, "
            "ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn "
            "FROM {table}{where}) bars "
            "WHERE rn <= %(last_bars)s ORDER BY ticker, date"
        ).format(cols=value_cols, table=table, where=_where(clauses))
        params["last_bars"] = last_bars
    return db.read_df(db.Database.STOCKS, query, params)


def fetch_symbol_ohlcv(
    symbol: str, *, start: date | None = None, end: date | None = None
) -> pd.DataFrame:
    """Bars for ``symbol`` from whichever table has it: stocks first, then ETFs/benchmarks."""
    bars = fetch_ohlcv("stock", [symbol], start=start, end=end)
    if bars.empty:
        bars = fetch_ohlcv("etf", [symbol], start=start, end=end)
    return bars


def fetch_stock_metrics(
    ticker: str, start: date | None = None, end: date | None = None
) -> pd.DataFrame:
    """RS metrics for a stock: one row per (date, ETF it belongs to)."""
    clauses, params = _date_filters(start, end)
    clauses.insert(0, sql.SQL("ticker = %(ticker)s"))
    params["ticker"] = ticker
    query = sql.SQL(
        "SELECT date, ticker, etf_ticker, relative_strength, rs_rank_index, "
        "rs_rank_overall, rs_z_score FROM fact_stock_metrics{where} ORDER BY date, etf_ticker"
    ).format(where=_where(clauses))
    return db.read_df(db.Database.STOCKS, query, params)


def fetch_latest_etf_grid() -> pd.DataFrame:
    """One row per ETF on the latest ETF-metrics date: etf_ticker, etf_name, rank, z-score.

    The name prefers the sector that maps to the ETF, then the alphabetically
    first industry, then ``dim_etf.etf_name``; several industries can share an ETF.
    """
    query = """
        WITH latest AS (
            SELECT MAX(date) AS date FROM fact_etf_metrics WHERE ref_ticker = %(benchmark)s
        ),
        sector AS (SELECT etf_ticker, MIN(name) AS name FROM dim_sector GROUP BY etf_ticker),
        industry AS (SELECT etf_ticker, MIN(name) AS name FROM dim_industry GROUP BY etf_ticker)
        SELECT f.etf_ticker,
               COALESCE(s.name, i.name, e.etf_name) AS etf_name,
               f.rs_rank_index, f.rs_z_score
        FROM fact_etf_metrics f
        JOIN latest l ON f.date = l.date
        LEFT JOIN sector s ON s.etf_ticker = f.etf_ticker
        LEFT JOIN industry i ON i.etf_ticker = f.etf_ticker
        LEFT JOIN dim_etf e ON e.etf_ticker = f.etf_ticker
        WHERE f.ref_ticker = %(benchmark)s
        ORDER BY f.rs_rank_index DESC NULLS LAST, f.etf_ticker
    """
    return db.read_df(db.Database.STOCKS, query, {"benchmark": get_settings().benchmark})


def fetch_etf_metrics(
    etf_ticker: str, start: date | None = None, end: date | None = None
) -> pd.DataFrame:
    """RS metrics for an ETF against the benchmark."""
    clauses, params = _date_filters(start, end)
    clauses[:0] = [sql.SQL("etf_ticker = %(etf)s"), sql.SQL("ref_ticker = %(benchmark)s")]
    params.update(etf=etf_ticker, benchmark=get_settings().benchmark)
    query = sql.SQL(
        "SELECT date, etf_ticker, ref_ticker, relative_strength, rs_rank_index, "
        "rs_rank_overall, rs_z_score FROM fact_etf_metrics{where} ORDER BY date"
    ).format(where=_where(clauses))
    return db.read_df(db.Database.STOCKS, query, params)


def fetch_all_etf_rank_history(start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """rs_rank_index history of every ETF against the benchmark."""
    clauses, params = _date_filters(start, end)
    clauses.insert(0, sql.SQL("ref_ticker = %(benchmark)s"))
    params["benchmark"] = get_settings().benchmark
    query = sql.SQL(
        "SELECT date, etf_ticker, rs_rank_index FROM fact_etf_metrics{where} "
        "ORDER BY etf_ticker, date"
    ).format(where=_where(clauses))
    return db.read_df(db.Database.STOCKS, query, params)


def fetch_exchange_map(tickers: Sequence[str]) -> dict[str, str | None]:
    """Vendor exchange code from dim_stock for each ticker (None when unknown)."""
    if not tickers:
        return {}
    df = db.read_df(
        db.Database.STOCKS,
        "SELECT ticker, exchange FROM dim_stock WHERE ticker = ANY(%(tickers)s)",
        {"tickers": list(tickers)},
    )
    result: dict[str, str | None] = dict.fromkeys(tickers)
    for ticker, exchange in zip(df["ticker"], df["exchange"], strict=True):
        result[ticker] = exchange if isinstance(exchange, str) else None
    return result


def fetch_top_ranked_ticker() -> str | None:
    """The stock with the highest rs_rank_overall on the latest metrics date (ties by ticker)."""
    query = """
        SELECT ticker FROM fact_stock_metrics
        WHERE date = (SELECT MAX(date) FROM fact_stock_metrics)
        ORDER BY rs_rank_overall DESC NULLS LAST, ticker
        LIMIT 1
    """
    df = db.read_df(db.Database.STOCKS, query)
    return None if df.empty else str(df["ticker"].iloc[0])


def fetch_known_symbols(symbols: Sequence[str]) -> set[str]:
    """The subset of ``symbols`` the stocks database knows: stocks, ETFs and the benchmark."""
    if not symbols:
        return set()
    query = """
        SELECT ticker AS symbol FROM dim_stock WHERE ticker = ANY(%(symbols)s)
        UNION SELECT etf_ticker FROM dim_etf WHERE etf_ticker = ANY(%(symbols)s)
        UNION SELECT etf_ticker FROM dim_sector WHERE etf_ticker = ANY(%(symbols)s)
        UNION SELECT etf_ticker FROM dim_industry WHERE etf_ticker = ANY(%(symbols)s)
    """
    known = db.read_df(db.Database.STOCKS, query, {"symbols": list(symbols)})
    found = set(known["symbol"]) if not known.empty else set()
    benchmark = get_settings().benchmark
    return found | ({benchmark} & set(symbols))


def collapse_etf_rows(df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
    """One row per ticker from per-ETF metric rows.

    Keeps the row of the ETF where the ticker ranks highest by ``rs_rank_index``
    (ties and missing ranks broken by ETF ticker) and adds ``etfs``: every ETF
    of the ticker, sorted and comma-separated.
    """
    if df.empty:
        return df.assign(etfs=pd.Series(dtype=object))
    etfs = (
        df.groupby(ticker_col)["etf_ticker"]
        .agg(lambda s: ", ".join(sorted(set(s.dropna()))))
        .rename("etfs")
    )
    best = (
        df.sort_values(
            [ticker_col, "rs_rank_index", "etf_ticker"],
            ascending=[True, False, True],
            na_position="last",
        )
        .drop_duplicates(subset=ticker_col, keep="first")
        .set_index(ticker_col)
    )
    return best.join(etfs).reset_index()


def _latest_stock_metrics(tickers: Sequence[str] | None) -> pd.DataFrame:
    """Per-ETF rows of fact_stock_metrics on its latest date."""
    clauses = [sql.SQL("m.date = (SELECT MAX(date) FROM fact_stock_metrics)")]
    params: dict[str, object] = {}
    if tickers is not None:
        clauses.append(sql.SQL("m.ticker = ANY(%(tickers)s)"))
        params["tickers"] = list(tickers)
    query = sql.SQL(
        "SELECT m.ticker, m.etf_ticker, m.relative_strength, m.rs_rank_index, "
        "m.rs_rank_overall, m.rs_z_score FROM fact_stock_metrics m{where}"
    ).format(where=_where(clauses))
    return db.read_df(db.Database.STOCKS, query, params)


def enrich_with_rs(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    """Left-join latest RS metrics onto ``df`` by ``symbol_col``, one metrics row per symbol.

    Adds etf_ticker (the ETF where the symbol ranks highest), etfs,
    rs_rank_index, rs_rank_overall and rs_z_score.
    """
    if df.empty or symbol_col not in df.columns:
        return df
    symbols = df[symbol_col].dropna().unique().tolist()
    rs = collapse_etf_rows(_latest_stock_metrics(symbols))
    rs = rs[["ticker", "etf_ticker", "etfs", "rs_rank_index", "rs_rank_overall", "rs_z_score"]]
    return df.merge(rs.rename(columns={"ticker": symbol_col}), on=symbol_col, how="left")


def _volume_stats(tickers: Sequence[str] | None) -> pd.DataFrame:
    """Volume mean/std over the last year of bars, recent mean, and the latest bar's volume.

    Only tickers with a bar on the table's latest date are returned.
    """
    table = sql.Identifier(OHLCV_TABLES["stock"])
    params: dict[str, object] = {
        "stats_bars": VOLUME_STATS_BARS,
        "recent_bars": RECENT_VOLUME_BARS,
        "prefilter": LAST_BARS_PREFILTER_FACTOR * VOLUME_STATS_BARS,
    }
    clauses: list[sql.Composable] = [
        sql.SQL("date >= {}").format(_session_floor(table, "%(prefilter)s"))
    ]
    if tickers is not None:
        clauses.append(sql.SQL("ticker = ANY(%(tickers)s)"))
        params["tickers"] = list(tickers)
    query = sql.SQL("""
        WITH bars AS (
            SELECT ticker, date, volume,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM fact_commons_ohlc{where}
        )
        SELECT ticker,
               AVG(volume) AS yearly_avg_volume,
               STDDEV(volume) AS yearly_std_volume,
               AVG(volume) FILTER (WHERE rn <= %(recent_bars)s) AS avg_30d_volume,
               MAX(volume) FILTER (WHERE rn = 1) AS current_volume
        FROM bars
        WHERE rn <= %(stats_bars)s
        GROUP BY ticker
        HAVING MAX(date) = (SELECT MAX(date) FROM fact_commons_ohlc)
    """).format(where=_where(clauses))
    stats = db.read_df(db.Database.STOCKS, query, params)
    std = stats["yearly_std_volume"].astype(float)
    z = (stats["current_volume"] - stats["yearly_avg_volume"]) / std.where(std > 0)
    stats["volume_z_score"] = z.fillna(0.0)
    return stats


def _between(series: pd.Series, low: float | None, high: float | None) -> pd.Series:
    mask = pd.Series(True, index=series.index)
    if low is not None:
        mask &= series >= low
    if high is not None:
        mask &= series <= high
    return mask


def search_stocks_by_metrics(
    rs_rank_min: float | None = None,
    rs_rank_max: float | None = None,
    rs_rank_overall_min: float | None = None,
    rs_rank_overall_max: float | None = None,
    rs_z_score_min: float | None = None,
    rs_z_score_max: float | None = None,
    volume_std_dev_min: float | None = None,
    avg_volume_min: float | None = None,
    weinstein_stage: str | None = None,
    etf_ticker: str | None = None,
    restrict_tickers: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Screen stocks on their latest RS metrics, volume and (optionally) Weinstein stage.

    Returns one row per ticker (see :func:`collapse_etf_rows`); with
    ``etf_ticker`` the row is that ETF's and only its members are returned.
    Rank bounds are fractions (0-1). Sorted by rs_rank_index, best first.
    """
    if restrict_tickers is not None and not restrict_tickers:
        return pd.DataFrame()
    metrics = _latest_stock_metrics(restrict_tickers)
    if metrics.empty:
        return pd.DataFrame()

    etfs = collapse_etf_rows(metrics)[["ticker", "etfs"]]
    if etf_ticker:
        rows = metrics[metrics["etf_ticker"] == etf_ticker].merge(etfs, on="ticker")
    else:
        rows = collapse_etf_rows(metrics)

    volume = _volume_stats(rows["ticker"].tolist())
    df = rows.merge(volume, on="ticker")
    mask = (
        _between(df["rs_rank_index"], rs_rank_min, rs_rank_max)
        & _between(df["rs_rank_overall"], rs_rank_overall_min, rs_rank_overall_max)
        & _between(df["rs_z_score"], rs_z_score_min, rs_z_score_max)
        & _between(df["avg_30d_volume"], avg_volume_min, None)
    )
    if volume_std_dev_min is not None:
        mask &= (df["yearly_std_volume"] > 0) & (df["volume_z_score"] >= volume_std_dev_min)
    df = df[mask]

    if weinstein_stage and not df.empty:
        bars = fetch_ohlcv("stock", df["ticker"].tolist(), last_bars=WEINSTEIN_LOOKBACK_BARS)
        stages = classify_bulk_stages(bars)
        df = df.assign(stage=df["ticker"].map(stages))
        df = df[df["stage"] == weinstein_stage]

    return df.sort_values("rs_rank_index", ascending=False, na_position="last").reset_index(
        drop=True
    )
