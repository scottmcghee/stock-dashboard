"""Pre-filter queries for the stock screener.

Each pre-filter returns a set of tickers; the screener intersects them before
running :func:`stock_dashboard.queries.stocks_queries.search_stocks_by_metrics`.
Option-data reads compare ``underlying_data`` with the ``underlying_data_avg``
row of the same symbol and date.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from stock_dashboard import db
from stock_dashboard.config import get_settings
from stock_dashboard.logic.scan_settings import TREND_SCANS, ScanSettings
from stock_dashboard.logic.trend_signals import TREND_REQUIRED_BARS, compute_trend_indicators
from stock_dashboard.queries import signal_queries
from stock_dashboard.queries.stocks_queries import fetch_ohlcv

# A symbol "has weeklies" when its latest chain lists at least this many
# expirations within the next WEEKLY_HORIZON_DAYS days (monthly-only lists about one).
WEEKLY_MIN_EXPIRATIONS = 3
WEEKLY_HORIZON_DAYS = 30

OPTION_SCANS: dict[str, Callable[..., pd.DataFrame]] = {
    "vanna_extreme": signal_queries.scan_vanna_extreme,
    "flownet_extreme": signal_queries.scan_flownet_extreme,
    "volume_spike": signal_queries.scan_volume_spike,
    "dxoi_change": signal_queries.scan_dxoi_change,
    "gex_level": signal_queries.scan_gex_level,
}


def _symbols(df: pd.DataFrame, column: str = "symbol") -> set[str]:
    return set(df[column]) if not df.empty else set()


def fetch_etf_rs_tickers(min_rs_rank: float) -> set[str]:
    """Stocks that belong to an ETF whose latest rs_rank_index (vs the benchmark) >= threshold."""
    query = """
        WITH strong_etf AS (
            SELECT etf_ticker
            FROM fact_etf_metrics
            WHERE ref_ticker = %(benchmark)s
              AND rs_rank_index >= %(min_rank)s
              AND date = (SELECT MAX(date) FROM fact_etf_metrics WHERE ref_ticker = %(benchmark)s)
        )
        SELECT DISTINCT m.ticker
        FROM fact_stock_metrics m
        JOIN strong_etf e ON m.etf_ticker = e.etf_ticker
        WHERE m.date = (SELECT MAX(date) FROM fact_stock_metrics)
    """
    params = {"benchmark": get_settings().benchmark, "min_rank": min_rs_rank}
    return _symbols(db.read_df(db.Database.STOCKS, query, params), "ticker")


def fetch_option_volume_spike_symbols(min_multiple: float) -> set[str]:
    """Symbols whose latest option_volume is >= ``min_multiple`` x its average on that date."""
    query = """
        SELECT u.symbol
        FROM underlying_data u
        JOIN underlying_data_avg a ON a.symbol = u.symbol AND a.date = u.collection_date
        WHERE u.collection_date = (SELECT MAX(collection_date) FROM underlying_data)
          AND a.option_volume_avg > 0
          AND u.option_volume::float / a.option_volume_avg >= %(min_multiple)s
    """
    return _symbols(db.read_df(db.Database.OPTIONS, query, {"min_multiple": min_multiple}))


def fetch_weekly_options_symbols() -> set[str]:
    """Symbols whose latest option chain has weekly expirations (see WEEKLY_MIN_EXPIRATIONS)."""
    query = """
        SELECT symbol
        FROM option_chain_data
        WHERE collection_date = (SELECT MAX(collection_date) FROM option_chain_data)
        GROUP BY symbol
        HAVING COUNT(DISTINCT expiration_date) FILTER (
            WHERE expiration_date BETWEEN collection_date AND collection_date + %(horizon)s
        ) >= %(min_expirations)s
    """
    params = {"horizon": WEEKLY_HORIZON_DAYS, "min_expirations": WEEKLY_MIN_EXPIRATIONS}
    return _symbols(db.read_df(db.Database.OPTIONS, query, params))


def fetch_tracked_tickers() -> list[str]:
    """Stock tickers with RS metrics on the latest metrics date, sorted."""
    query = """
        SELECT DISTINCT ticker
        FROM fact_stock_metrics
        WHERE date = (SELECT MAX(date) FROM fact_stock_metrics)
        ORDER BY ticker
    """
    df = db.read_df(db.Database.STOCKS, query)
    return df["ticker"].tolist() if not df.empty else []


def fetch_trend_indicators(tickers: Sequence[str], kind: str = "stock") -> pd.DataFrame:
    """Latest trend-scan indicators for ``tickers``, fetched with exactly the history they need."""
    ohlcv = fetch_ohlcv("etf" if kind == "etf" else "stock", tickers, last_bars=TREND_REQUIRED_BARS)
    return compute_trend_indicators(ohlcv) if not ohlcv.empty else pd.DataFrame()


def fetch_signal_scan_symbols(scan_names: Sequence[str]) -> set[str]:
    """Run the named scans and return the symbols flagged by all of them.

    Valid names: the keys of OPTION_SCANS and TREND_SCANS; unknown names are ignored.
    Option scans share one history fetch. Every scan runs with its default
    thresholds (``ScanSettings()``), the Signal Scans page's defaults.
    """
    result_sets: list[set[str]] = []

    option_names = [n for n in scan_names if n in OPTION_SCANS]
    if option_names:
        history = signal_queries.fetch_option_signal_history()
        for name in option_names:
            flagged = _symbols(OPTION_SCANS[name](history=history, enrich=False))
            if not flagged:
                return set()
            result_sets.append(flagged)

    trend_names = [n for n in scan_names if n in TREND_SCANS]
    if trend_names:
        indicators = fetch_trend_indicators(fetch_tracked_tickers())
        if indicators.empty:
            return set()
        defaults = ScanSettings()
        for name in trend_names:
            flagged = _symbols(TREND_SCANS[name](indicators, defaults))
            if not flagged:
                return set()
            result_sets.append(flagged)

    if not result_sets:
        return set()
    return set.intersection(*result_sets)


def fetch_option_metrics_for_tickers(tickers: Sequence[str]) -> pd.DataFrame:
    """Latest option metrics for ``tickers``, vs the average row of the same date.

    Columns: symbol, option_volume, option_volume_avg, vol_multiple,
    put_call_ratio, gxoi, dxoi, vannaxoi, flownet.
    """
    if not tickers:
        return pd.DataFrame()
    query = """
        SELECT u.symbol,
               u.option_volume,
               a.option_volume_avg,
               CASE WHEN a.option_volume_avg > 0
                    THEN u.option_volume::float / a.option_volume_avg
               END AS vol_multiple,
               u.put_call_ratio, u.gxoi, u.dxoi, u.vannaxoi, u.flownet
        FROM underlying_data u
        LEFT JOIN underlying_data_avg a ON a.symbol = u.symbol AND a.date = u.collection_date
        WHERE u.collection_date = (SELECT MAX(collection_date) FROM underlying_data)
          AND u.symbol = ANY(%(tickers)s)
    """
    return db.read_df(db.Database.OPTIONS, query, {"tickers": list(tickers)})
