"""Pre-filter queries for the enhanced stock screener."""

import pandas as pd

from db import option_connection, stocks_connection


def fetch_etf_rs_tickers(min_rs_rank: float) -> set[str]:
    """Return stock tickers belonging to ETFs whose latest rs_rank_index >= threshold.

    Joins fact_etf_metrics (ETF-level rank) → fact_stock_metrics (stock membership).
    """
    with stocks_connection() as conn:
        query = """
        WITH latest_etf AS (
            SELECT etf_ticker, rs_rank_index
            FROM fact_etf_metrics
            WHERE date = (SELECT MAX(date) FROM fact_etf_metrics)
              AND rs_rank_index >= %s
        )
        SELECT DISTINCT m.ticker
        FROM fact_stock_metrics m
        JOIN latest_etf e ON m.etf_ticker = e.etf_ticker
        WHERE m.date = (SELECT MAX(date) FROM fact_stock_metrics)
        """
        df = pd.read_sql_query(query, conn, params=[min_rs_rank])
    return set(df["ticker"].tolist()) if not df.empty else set()


def fetch_option_volume_spike_symbols(min_multiple: float) -> set[str]:
    """Return symbols where latest option_volume / option_volume_avg >= min_multiple."""
    with option_connection() as conn:
        query = """
        WITH latest AS (
            SELECT MAX(collection_date) as max_date FROM underlying_data
        )
        SELECT u.symbol
        FROM underlying_data u
        JOIN underlying_data_avg a
            ON u.symbol = a.symbol AND a.date = (SELECT MAX(date) FROM underlying_data_avg)
        JOIN latest l ON u.collection_date = l.max_date
        WHERE a.option_volume_avg > 0
          AND u.option_volume::float / a.option_volume_avg >= %s
        """
        df = pd.read_sql_query(query, conn, params=[min_multiple])
    return set(df["symbol"].tolist()) if not df.empty else set()


def fetch_signal_scan_symbols(scan_names: list[str]) -> set[str]:
    """Run selected signal scans and return the intersection of their symbol sets.

    Valid scan_names: vanna_extreme, flownet_extreme, volume_spike, dxoi_change, gex_level.
    """
    from queries.signal_queries import (
        scan_vanna_extreme,
        scan_flownet_extreme,
        scan_volume_spike,
        scan_dxoi_change,
        scan_gex_level,
    )

    SCAN_MAP = {
        "vanna_extreme": scan_vanna_extreme,
        "flownet_extreme": scan_flownet_extreme,
        "volume_spike": scan_volume_spike,
        "dxoi_change": scan_dxoi_change,
        "gex_level": scan_gex_level,
    }

    result_sets: list[set[str]] = []
    for name in scan_names:
        fn = SCAN_MAP.get(name)
        if fn is None:
            continue
        df = fn()
        if not df.empty and "symbol" in df.columns:
            result_sets.append(set(df["symbol"].tolist()))
        else:
            # Scan returned nothing — intersection will be empty
            return set()

    if not result_sets:
        return set()
    return result_sets[0].intersection(*result_sets[1:])


def fetch_option_metrics_for_tickers(tickers: list[str]) -> pd.DataFrame:
    """Fetch latest option metrics for a list of tickers.

    Returns DataFrame with columns: symbol, option_volume, option_volume_avg,
    vol_multiple, put_call_ratio, gxoi, dxoi, vannaxoi, flownet.
    """
    if not tickers:
        return pd.DataFrame()
    with option_connection() as conn:
        placeholders = ",".join(["%s"] * len(tickers))
        query = f"""
        WITH latest AS (
            SELECT MAX(collection_date) as max_date FROM underlying_data
        )
        SELECT
            u.symbol,
            u.option_volume,
            a.option_volume_avg,
            CASE WHEN a.option_volume_avg > 0
                 THEN u.option_volume::float / a.option_volume_avg
                 ELSE NULL END AS vol_multiple,
            u.put_call_ratio,
            u.gxoi,
            u.dxoi,
            u.vannaxoi,
            u.flownet
        FROM underlying_data u
        JOIN underlying_data_avg a
            ON u.symbol = a.symbol AND a.date = (SELECT MAX(date) FROM underlying_data_avg)
        JOIN latest l ON u.collection_date = l.max_date
        WHERE u.symbol IN ({placeholders})
        """
        df = pd.read_sql_query(query, conn, params=tickers)
    return df


def fetch_bulk_ohlcv_for_ta(
    tickers: list[str], lookback_days: int = 300
) -> pd.DataFrame:
    """Fetch recent OHLCV for multiple tickers in a single query.

    Returns DataFrame with columns: ticker, date, open, high, low, close, volume.
    """
    if not tickers:
        return pd.DataFrame()
    with stocks_connection() as conn:
        placeholders = ",".join(["%s"] * len(tickers))
        query = f"""
        SELECT ticker, date, open, high, low, close, volume
        FROM fact_commons_ohlc
        WHERE ticker IN ({placeholders})
          AND date >= NOW() - INTERVAL '{int(lookback_days)} days'
        ORDER BY ticker, date
        """
        df = pd.read_sql_query(query, conn, params=tickers)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_bulk_etf_ohlcv_for_ta(
    etf_tickers: list[str], lookback_days: int = 300
) -> pd.DataFrame:
    """Fetch recent OHLCV for multiple ETFs from fact_index_ohlc.

    Returns DataFrame with columns: ticker, date, open, high, low, close, volume.
    """
    if not etf_tickers:
        return pd.DataFrame()
    with stocks_connection() as conn:
        placeholders = ",".join(["%s"] * len(etf_tickers))
        query = f"""
        SELECT ticker, date, open, high, low, close, volume
        FROM fact_index_ohlc
        WHERE ticker IN ({placeholders})
          AND date >= NOW() - INTERVAL '{int(lookback_days)} days'
        ORDER BY ticker, date
        """
        df = pd.read_sql_query(query, conn, params=etf_tickers)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
