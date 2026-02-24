"""Cross-DB signal scan queries for option_data + stocks enrichment."""

import pandas as pd

from db import option_connection, stocks_connection


def scan_vanna_extreme(percentile_threshold: float = 20.0) -> pd.DataFrame:
    """Find symbols where vannaxoi is in the top/bottom Nth percentile of 90-day history.

    High vanna = bullish (vol drop -> delta increase). Tier 1 signal.
    """
    with option_connection() as conn:
        query = """
        WITH latest AS (
            SELECT MAX(collection_date) as max_date FROM underlying_data
        ),
        history AS (
            SELECT symbol, vannaxoi, collection_date,
                   PERCENT_RANK() OVER (PARTITION BY symbol ORDER BY vannaxoi) as pct_rank
            FROM underlying_data
            WHERE collection_date >= (SELECT max_date - INTERVAL '90 days' FROM latest)
              AND vannaxoi IS NOT NULL
        ),
        current_vals AS (
            SELECT h.symbol, h.vannaxoi, h.pct_rank
            FROM history h
            JOIN latest l ON h.collection_date = l.max_date
        )
        SELECT symbol, vannaxoi as signal_value, pct_rank,
               CASE WHEN pct_rank >= (1 - %(threshold)s / 100.0) THEN 'High Vanna'
                    WHEN pct_rank <= %(threshold)s / 100.0 THEN 'Low Vanna'
               END as signal_label
        FROM current_vals
        WHERE pct_rank >= (1 - %(threshold)s / 100.0)
           OR pct_rank <= %(threshold)s / 100.0
        ORDER BY pct_rank DESC
        """
        df = pd.read_sql_query(query, conn, params={"threshold": percentile_threshold})
    return _enrich_with_rs(df) if not df.empty else df


def scan_flownet_extreme(percentile_threshold: float = 10.0) -> pd.DataFrame:
    """Find symbols where flownet is in the bottom Nth percentile (contrarian buy).

    Extreme negative flow -> mean reversion. Tier 1 signal.
    """
    with option_connection() as conn:
        query = """
        WITH latest AS (
            SELECT MAX(collection_date) as max_date FROM underlying_data
        ),
        history AS (
            SELECT symbol, flownet, collection_date,
                   PERCENT_RANK() OVER (PARTITION BY symbol ORDER BY flownet) as pct_rank
            FROM underlying_data
            WHERE collection_date >= (SELECT max_date - INTERVAL '90 days' FROM latest)
              AND flownet IS NOT NULL
        ),
        current_vals AS (
            SELECT h.symbol, h.flownet, h.pct_rank
            FROM history h
            JOIN latest l ON h.collection_date = l.max_date
        )
        SELECT symbol, flownet as signal_value, pct_rank,
               'Extreme Negative Flow' as signal_label
        FROM current_vals
        WHERE pct_rank <= %(threshold)s / 100.0
        ORDER BY pct_rank ASC
        """
        df = pd.read_sql_query(query, conn, params={"threshold": percentile_threshold})
    return _enrich_with_rs(df) if not df.empty else df


def scan_volume_spike(multiple_threshold: float = 2.0) -> pd.DataFrame:
    """Find symbols where option volume > Nx 20-day average.

    Predicts higher absolute moves. Tier 2 signal.
    """
    with option_connection() as conn:
        query = """
        WITH latest AS (
            SELECT MAX(collection_date) as max_date FROM underlying_data
        ),
        vol_stats AS (
            SELECT symbol, option_volume, collection_date,
                   AVG(option_volume) OVER (
                       PARTITION BY symbol ORDER BY collection_date
                       ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                   ) as avg_20d_volume
            FROM underlying_data
            WHERE collection_date >= (SELECT max_date - INTERVAL '30 days' FROM latest)
              AND option_volume IS NOT NULL
        ),
        current_vals AS (
            SELECT v.symbol, v.option_volume, v.avg_20d_volume,
                   v.option_volume::float / NULLIF(v.avg_20d_volume, 0) as vol_ratio
            FROM vol_stats v
            JOIN latest l ON v.collection_date = l.max_date
            WHERE v.avg_20d_volume > 0
        )
        SELECT symbol, option_volume as signal_value, vol_ratio,
               avg_20d_volume,
               CASE WHEN vol_ratio >= 3.0 THEN 'Extreme Spike (>3x)'
                    WHEN vol_ratio >= 2.0 THEN 'Major Spike (2-3x)'
                    ELSE 'Elevated (>threshold)'
               END as signal_label
        FROM current_vals
        WHERE vol_ratio >= %(threshold)s
        ORDER BY vol_ratio DESC
        """
        df = pd.read_sql_query(query, conn, params={"threshold": multiple_threshold})
    return _enrich_with_rs(df) if not df.empty else df


def scan_dxoi_change(pct_threshold: float = 20.0) -> pd.DataFrame:
    """Find symbols where dxoi changed > N% from prior day.

    Big positioning shifts precede price moves. Tier 2 signal.
    """
    with option_connection() as conn:
        query = """
        WITH latest AS (
            SELECT MAX(collection_date) as max_date FROM underlying_data
        ),
        with_lag AS (
            SELECT symbol, dxoi, collection_date,
                   LAG(dxoi) OVER (PARTITION BY symbol ORDER BY collection_date) as prev_dxoi
            FROM underlying_data
            WHERE collection_date >= (SELECT max_date - INTERVAL '5 days' FROM latest)
              AND dxoi IS NOT NULL
        ),
        current_vals AS (
            SELECT w.symbol, w.dxoi, w.prev_dxoi,
                   CASE WHEN ABS(w.prev_dxoi) > 0
                        THEN (w.dxoi - w.prev_dxoi) / ABS(w.prev_dxoi) * 100
                        ELSE NULL
                   END as pct_change
            FROM with_lag w
            JOIN latest l ON w.collection_date = l.max_date
            WHERE w.prev_dxoi IS NOT NULL
        )
        SELECT symbol, dxoi as signal_value, prev_dxoi, pct_change,
               CASE WHEN pct_change >= %(threshold)s THEN 'Big Increase'
                    WHEN pct_change <= -%(threshold)s THEN 'Big Decrease'
               END as signal_label
        FROM current_vals
        WHERE ABS(pct_change) >= %(threshold)s
        ORDER BY ABS(pct_change) DESC
        """
        df = pd.read_sql_query(query, conn, params={"threshold": pct_threshold})
    return _enrich_with_rs(df) if not df.empty else df


def scan_gex_level() -> pd.DataFrame:
    """Classify current GEX regime for all symbols.

    Negative GEX -> higher volatility. Tier 2 signal.
    """
    with option_connection() as conn:
        query = """
        WITH latest AS (
            SELECT MAX(collection_date) as max_date FROM underlying_data
        ),
        current_vals AS (
            SELECT u.symbol, u.gxoi
            FROM underlying_data u
            JOIN latest l ON u.collection_date = l.max_date
            WHERE u.gxoi IS NOT NULL
        )
        SELECT symbol, gxoi as signal_value,
               CASE WHEN gxoi < 0 THEN 'Negative GEX'
                    WHEN gxoi < 5000 THEN 'Low GEX'
                    WHEN gxoi < 15000 THEN 'Mid GEX'
                    WHEN gxoi < 50000 THEN 'High GEX'
                    ELSE 'Very High GEX'
               END as signal_label
        FROM current_vals
        ORDER BY gxoi ASC
        """
        df = pd.read_sql_query(query, conn)
    return _enrich_with_rs(df) if not df.empty else df


def _enrich_with_rs(df: pd.DataFrame) -> pd.DataFrame:
    """Add RS metrics from stocks DB to scan results."""
    if df.empty or "symbol" not in df.columns:
        return df

    symbols = df["symbol"].unique().tolist()
    if not symbols:
        return df

    with stocks_connection() as conn:
        placeholders = ",".join(["%s"] * len(symbols))
        query = f"""
        SELECT m.ticker as symbol, m.etf_ticker,
               m.rs_rank_index, m.rs_rank_overall, m.rs_z_score
        FROM fact_stock_metrics m
        WHERE m.date = (SELECT MAX(date) FROM fact_stock_metrics)
          AND m.ticker IN ({placeholders})
        """
        rs_df = pd.read_sql_query(query, conn, params=symbols)

    if rs_df.empty:
        return df

    # Take first row per symbol (in case of multiple ETF entries)
    rs_df = rs_df.drop_duplicates(subset="symbol", keep="first")
    return df.merge(rs_df, on="symbol", how="left")
