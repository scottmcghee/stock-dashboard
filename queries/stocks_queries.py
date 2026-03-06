"""Data access layer for the stocks database (pure SQL -> DataFrames)."""

import pandas as pd

from db import stocks_connection


def get_all_tickers() -> list[str]:
    """Get all distinct stock tickers from dim_stock."""
    with stocks_connection() as conn:
        query = "SELECT DISTINCT ticker FROM dim_stock ORDER BY ticker"
        df = pd.read_sql_query(query, conn)
        return df["ticker"].tolist()


def get_all_etfs() -> list[str]:
    """Get all distinct ETF tickers from fact_stock_metrics."""
    with stocks_connection() as conn:
        query = """
        SELECT DISTINCT etf_ticker
        FROM fact_stock_metrics
        WHERE etf_ticker IS NOT NULL
        ORDER BY etf_ticker
        """
        df = pd.read_sql_query(query, conn)
        return df["etf_ticker"].tolist()


def fetch_stock_ohlcv(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data for a stock."""
    with stocks_connection() as conn:
        query = """
        SELECT date, open, high, low, close, volume
        FROM fact_commons_ohlc
        WHERE ticker = %s
        """
        params: list = [ticker]
        if start_date and end_date:
            query += " AND date BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        query += " ORDER BY date"
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df


def fetch_stock_metrics(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch RS metrics for a stock."""
    with stocks_connection() as conn:
        query = """
        SELECT date, ticker, etf_ticker, relative_strength,
               rs_rank_index, rs_rank_overall, rs_z_score
        FROM fact_stock_metrics
        WHERE ticker = %s
        """
        params: list = [ticker]
        if start_date and end_date:
            query += " AND date BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        query += " ORDER BY date"
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df


def fetch_latest_etf_grid() -> pd.DataFrame:
    """Fetch latest ETF grid: ticker, name, rs_rank_index, rs_z_score."""
    with stocks_connection() as conn:
        latest_query = "SELECT MAX(date) FROM fact_etf_metrics"
        with conn.cursor() as cur:
            cur.execute(latest_query)
            latest_date = cur.fetchone()[0]
        if not latest_date:
            return pd.DataFrame()
        query = """
        SELECT f.etf_ticker,
               COALESCE(s.name, i.name, e.etf_name) AS etf_name,
               f.rs_rank_index, f.rs_z_score
        FROM fact_etf_metrics f
        LEFT JOIN dim_sector s ON f.etf_ticker = s.etf_ticker
        LEFT JOIN dim_industry i ON f.etf_ticker = i.etf_ticker
        LEFT JOIN dim_etf e ON f.etf_ticker = e.etf_ticker
        WHERE f.date = %s
        ORDER BY f.rs_rank_index DESC
        """
        return pd.read_sql_query(query, conn, params=[latest_date])


def fetch_etf_metrics(
    etf_ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch RS metrics for an ETF."""
    with stocks_connection() as conn:
        query = """
        SELECT date, etf_ticker, ref_ticker, relative_strength,
               rs_rank_index, rs_rank_overall, rs_z_score
        FROM fact_etf_metrics
        WHERE etf_ticker = %s
        """
        params: list = [etf_ticker]
        if start_date and end_date:
            query += " AND date BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        query += " ORDER BY date"
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df


def fetch_etf_ohlcv(
    etf_ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data for an ETF from fact_index_ohlc."""
    with stocks_connection() as conn:
        query = """
        SELECT date, open, high, low, close, volume
        FROM fact_index_ohlc
        WHERE ticker = %s
        """
        params: list = [etf_ticker]
        if start_date and end_date:
            query += " AND date BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        query += " ORDER BY date"
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df


def fetch_all_etf_rank_history(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch rs_rank_index history for all ETFs."""
    with stocks_connection() as conn:
        query = """
        SELECT date, etf_ticker, rs_rank_index
        FROM fact_etf_metrics
        WHERE 1=1
        """
        params: list = []
        if start_date and end_date:
            query += " AND date BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        query += " ORDER BY etf_ticker, date"
        df = pd.read_sql_query(query, conn, params=params or None)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df


def fetch_52w_data(start_date: str) -> pd.DataFrame:
    """Fetch date/ticker/high/low for 52W high/low computation.

    start_date should include lookback (display_start - 370 days).
    """
    with stocks_connection() as conn:
        query = """
        SELECT date, ticker, high, low
        FROM fact_commons_ohlc
        WHERE date >= %s
        """
        df = pd.read_sql_query(query, conn, params=[start_date])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
        return df


def fetch_52w_etf_data(start_date: str) -> pd.DataFrame:
    """Fetch date/ticker/high/low for ETFs (fact_index_ohlc)."""
    with stocks_connection() as conn:
        query = """
        SELECT date, ticker, high, low
        FROM fact_index_ohlc
        WHERE date >= %s
        """
        df = pd.read_sql_query(query, conn, params=[start_date])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
        return df


def fetch_exchange_map(tickers: list[str]) -> dict[str, str | None]:
    """Lookup exchange codes from dim_stock for a list of tickers."""
    if not tickers:
        return {}
    with stocks_connection() as conn:
        placeholders = ",".join(["%s"] * len(tickers))
        query = f"SELECT ticker, exchange FROM dim_stock WHERE ticker IN ({placeholders})"
        df = pd.read_sql_query(query, conn, params=tickers)
    result: dict[str, str | None] = {t: None for t in tickers}
    for _, row in df.iterrows():
        result[row["ticker"]] = row["exchange"]
    return result


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
    restrict_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Search stocks by various metrics criteria.

    Uses a single bulk query + bulk Weinstein classification (no N+1).
    """
    with stocks_connection() as conn:
        # Get latest date
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM fact_stock_metrics")
            latest_date = cur.fetchone()[0]
        if not latest_date:
            return pd.DataFrame()

        query = """
        WITH avg_vol AS (
            SELECT
                ticker,
                AVG(volume) as yearly_avg_volume,
                STDDEV(volume) as yearly_std_volume,
                AVG(volume) FILTER (WHERE date >= NOW() - INTERVAL '30 days') as avg_30d_volume
            FROM fact_commons_ohlc
            WHERE date >= NOW() - INTERVAL '1 year'
            GROUP BY ticker
        ),
        current_vol AS (
            SELECT ticker, volume as current_volume
            FROM fact_commons_ohlc
            WHERE date = (SELECT MAX(date) FROM fact_commons_ohlc)
        )
        SELECT
            m.ticker, m.etf_ticker, m.relative_strength,
            m.rs_rank_index, m.rs_rank_overall, m.rs_z_score,
            v.yearly_avg_volume, v.yearly_std_volume, v.avg_30d_volume,
            cv.current_volume,
            CASE WHEN v.yearly_std_volume > 0
                 THEN (cv.current_volume - v.yearly_avg_volume) / v.yearly_std_volume
                 ELSE 0
            END as volume_z_score
        FROM fact_stock_metrics m
        JOIN avg_vol v ON m.ticker = v.ticker
        JOIN current_vol cv ON m.ticker = cv.ticker
        WHERE m.date = %s
        """
        params: list = [latest_date]

        if restrict_tickers is not None:
            placeholders = ",".join(["%s"] * len(restrict_tickers))
            query += f" AND m.ticker IN ({placeholders})"
            params.extend(restrict_tickers)

        if etf_ticker:
            query += " AND m.etf_ticker = %s"
            params.append(etf_ticker)
        if rs_rank_min is not None:
            query += " AND m.rs_rank_index >= %s"
            params.append(rs_rank_min)
        if rs_rank_max is not None:
            query += " AND m.rs_rank_index <= %s"
            params.append(rs_rank_max)
        if rs_rank_overall_min is not None:
            query += " AND m.rs_rank_overall >= %s"
            params.append(rs_rank_overall_min)
        if rs_rank_overall_max is not None:
            query += " AND m.rs_rank_overall <= %s"
            params.append(rs_rank_overall_max)
        if rs_z_score_min is not None:
            query += " AND m.rs_z_score >= %s"
            params.append(rs_z_score_min)
        if rs_z_score_max is not None:
            query += " AND m.rs_z_score <= %s"
            params.append(rs_z_score_max)
        if volume_std_dev_min is not None:
            query += " AND (cv.current_volume - v.yearly_avg_volume) / NULLIF(v.yearly_std_volume, 0) >= %s"
            params.append(volume_std_dev_min)
        if avg_volume_min is not None:
            query += " AND v.avg_30d_volume >= %s"
            params.append(avg_volume_min)

        query += " ORDER BY m.rs_rank_index DESC"
        df = pd.read_sql_query(query, conn, params=params)

        # Bulk Weinstein classification if requested
        if weinstein_stage and not df.empty:
            from logic.weinstein import classify_bulk_stages
            tickers = df["ticker"].unique().tolist()
            stages = fetch_bulk_weinstein_data(conn, tickers)
            if not stages.empty:
                stage_map = classify_bulk_stages(stages)
                df["stage"] = df["ticker"].map(stage_map)
                df = df[df["stage"] == weinstein_stage]

        return df


def fetch_bulk_weinstein_data(conn, tickers: list[str]) -> pd.DataFrame:
    """Fetch last 200 days of OHLCV for multiple tickers in a single query."""
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join(["%s"] * len(tickers))
    query = f"""
    SELECT ticker, date, open, high, low, close, volume
    FROM (
        SELECT ticker, date, open, high, low, close, volume,
               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
        FROM fact_commons_ohlc
        WHERE ticker IN ({placeholders})
    ) sub
    WHERE rn <= 200
    ORDER BY ticker, date
    """
    df = pd.read_sql_query(query, conn, params=tickers)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
