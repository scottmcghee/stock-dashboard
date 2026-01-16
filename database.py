import pandas as pd
import psycopg2
from psycopg2 import sql
from config import DB_CONFIG, OPTION_DB_CONFIG
import numpy as np

def get_db_connection():
    """Create a database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def fetch_stock_data(ticker, start_date=None, end_date=None):
    """Fetch OHLCV data for a given stock."""
    conn = get_db_connection()
    try:
        query = """
        SELECT date, open, high, low, close, volume
        FROM fact_commons_ohlc
        WHERE ticker = %s
        """
        if start_date and end_date:
            query += " AND date BETWEEN %s AND %s"
        query += " ORDER BY date"
        
        params = [ticker]
        if start_date and end_date:
            params.extend([start_date, end_date])
            
        df = pd.read_sql_query(query, conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_stock_metrics(ticker, start_date=None, end_date=None):
    """Fetch relative strength metrics for a given stock."""
    conn = get_db_connection()
    try:
        query = """
        SELECT date, ticker, etf_ticker, relative_strength, rs_rank_index, 
               rs_rank_overall, rs_z_score
        FROM fact_stock_metrics
        WHERE ticker = %s
        """
        if start_date and end_date:
            query += " AND date BETWEEN %s AND %s"
        query += " ORDER BY date"
        
        params = [ticker]
        if start_date and end_date:
            params.extend([start_date, end_date])
            
        df = pd.read_sql_query(query, conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error fetching stock metrics: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def calculate_weinstein_stages(df, 
                              short_ma=50, 
                              long_ma=150, 
                              trend_period=20,
                              vol_lookback=21):
    """
    Identifies Weinstein's market stages using daily OHLC data
    Returns DataFrame with stage labels and indicators
    """
    try:
        # Ensure we have enough data
        if len(df) < long_ma:
            return pd.DataFrame({'stage': ['Undetermined']})
            
        # Sort by date ascending for proper calculation
        df = df.sort_values('date')
        
        # Calculate moving averages
        df['MA50'] = df['close'].rolling(window=short_ma, min_periods=1).mean()
        df['MA150'] = df['close'].rolling(window=long_ma, min_periods=1).mean()
        
        # Moving average trends (using simple difference)
        df['MA50_trend'] = df['MA50'].diff(periods=trend_period) > 0
        df['MA150_trend'] = df['MA150'].diff(periods=trend_period) > 0
        
        # Price position relative to MAs
        df['Above_MA50'] = df['close'] > df['MA50']
        df['Above_MA150'] = df['close'] > df['MA150']
        
        # Price momentum analysis (simplified to avoid rolling window issues)
        df['Higher_High'] = df['high'].rolling(window=2).apply(lambda x: x.iloc[-1] > x.iloc[0] if len(x) == 2 else False)
        df['Lower_Low'] = df['low'].rolling(window=2).apply(lambda x: x.iloc[-1] < x.iloc[0] if len(x) == 2 else False)
        
        # Volume analysis
        df['Vol_MA'] = df['volume'].rolling(window=vol_lookback, min_periods=1).mean()
        df['High_Vol'] = df['volume'] > 1.5 * df['Vol_MA']
        
        # Stage classification
        conditions = [
            # Stage 4: Declining
            (df['MA150_trend'] == False) & 
            (df['Above_MA150'] == False) &
            (df['Lower_Low'] == True),
            
            # Stage 2: Advancing
            (df['MA150_trend'] == True) &
            (df['MA50_trend'] == True) &
            (df['Above_MA50'] == True) &
            (df['Higher_High'] == True),
            
            # Stage 3: Topping
            (df['MA150_trend'] == True) &
            (df['MA50_trend'] == False) &
            (df['Above_MA50'] == False),
            
            # Stage 1: Basing
            (df['MA150_trend'] == False) &
            (df['close'].rolling(window=50, min_periods=1).std() < 0.1 * df['close'])
        ]
        
        choices = ['Stage 4', 'Stage 2', 'Stage 3', 'Stage 1']
        df['stage'] = np.select(conditions, choices, default='Undetermined')
        
        return df
    except Exception as e:
        print(f"Error in calculate_weinstein_stages: {str(e)}")
        return pd.DataFrame({'stage': ['Undetermined']})

def get_stock_stage(ticker, conn):
    """Get the current Weinstein stage for a stock."""
    try:
        # Get the last 200 days of data to ensure we have enough for calculations
        query = """
        SELECT date, open, high, low, close, volume
        FROM fact_commons_ohlc
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT 200
        """
        df = pd.read_sql_query(query, conn, params=[ticker])
        if df.empty:
            print(f"No data found for {ticker}")
            return None
            
        # Calculate stages
        df = calculate_weinstein_stages(df)
        if df.empty or 'stage' not in df.columns:
            print(f"Could not calculate stage for {ticker}")
            return None
            
        # Get the most recent stage
        latest_stage = df['stage'].iloc[-1]
        print(f"Calculated stage for {ticker}: {latest_stage}")
        return latest_stage
    except Exception as e:
        print(f"Error calculating stage for {ticker}: {str(e)}")
        return None

def search_stocks_by_metrics(rs_rank_min=None, rs_rank_max=None, 
                           rs_rank_overall_min=None, rs_rank_overall_max=None,
                           rs_z_score_min=None, rs_z_score_max=None,
                           volume_std_dev_min=None, avg_volume_min=None,
                           weinstein_stage=None, etf_ticker=None):
    """Search for stocks based on various metrics criteria."""
    conn = None
    try:
        print("\nAttempting database connection...")
        conn = get_db_connection()
        if conn is None:
            print("Failed to establish database connection")
            return pd.DataFrame()

        # First, get the latest date
        latest_date_query = "SELECT MAX(date) FROM fact_stock_metrics"
        print("\nExecuting latest date query...")
        with conn.cursor() as cur:
            cur.execute(latest_date_query)
            latest_date = cur.fetchone()[0]
            print(f"Latest date in database: {latest_date}")

        # Build the base query with volume metrics
        base_query = """
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
            SELECT 
                ticker,
                volume as current_volume
            FROM fact_commons_ohlc
            WHERE date = (SELECT MAX(date) FROM fact_commons_ohlc)
        )
        SELECT 
            m.ticker,
            m.etf_ticker,
            m.relative_strength,
            m.rs_rank_index,
            m.rs_rank_overall,
            m.rs_z_score,
            v.yearly_avg_volume,
            v.yearly_std_volume,
            v.avg_30d_volume,
            cv.current_volume,
            CASE 
                WHEN v.yearly_std_volume > 0 THEN 
                    (cv.current_volume - v.yearly_avg_volume) / v.yearly_std_volume
                ELSE 0 
            END as volume_z_score
        FROM fact_stock_metrics m
        JOIN avg_vol v ON m.ticker = v.ticker
        JOIN current_vol cv ON m.ticker = cv.ticker
        WHERE m.date = %s
        """
        
        params = [latest_date]
        
        # Add ETF filter if specified
        if etf_ticker:
            base_query += " AND m.etf_ticker = %s"
            params.append(etf_ticker)
            
        # Add other filters
        if rs_rank_min is not None:
            base_query += " AND m.rs_rank_index >= %s"
            params.append(rs_rank_min)
        if rs_rank_max is not None:
            base_query += " AND m.rs_rank_index <= %s"
            params.append(rs_rank_max)
        if rs_rank_overall_min is not None:
            base_query += " AND m.rs_rank_overall >= %s"
            params.append(rs_rank_overall_min)
        if rs_rank_overall_max is not None:
            base_query += " AND m.rs_rank_overall <= %s"
            params.append(rs_rank_overall_max)
        if rs_z_score_min is not None:
            base_query += " AND m.rs_z_score >= %s"
            params.append(rs_z_score_min)
        if rs_z_score_max is not None:
            base_query += " AND m.rs_z_score <= %s"
            params.append(rs_z_score_max)
        if volume_std_dev_min is not None:
            base_query += " AND (cv.current_volume - v.yearly_avg_volume) / NULLIF(v.yearly_std_volume, 0) >= %s"
            params.append(volume_std_dev_min)
        if avg_volume_min is not None:
            base_query += " AND v.avg_30d_volume >= %s"
            params.append(avg_volume_min)
            
        base_query += " ORDER BY m.rs_rank_index DESC"
        
        print("\nExecuting main query...")
        df = pd.read_sql_query(base_query, conn, params=params)
        
        if weinstein_stage:
            print("\nCalculating Weinstein stages...")
            df['stage'] = df['ticker'].apply(lambda x: get_stock_stage(x, conn))
            df = df[df['stage'] == weinstein_stage]
            
        return df
        
    except Exception as e:
        print(f"Error in search_stocks_by_metrics: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_all_tickers():
    """Get list of all available stock tickers."""
    conn = get_db_connection()
    try:
        query = """
        SELECT DISTINCT ticker 
        FROM dim_stock 
        ORDER BY ticker
        """
        df = pd.read_sql_query(query, conn)
        return df['ticker'].tolist()
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return []
    finally:
        conn.close()

def get_all_etfs():
    """Get all available ETFs from the database."""
    conn = get_db_connection()
    try:
        query = """
        SELECT DISTINCT etf_ticker 
        FROM fact_stock_metrics 
        WHERE etf_ticker IS NOT NULL 
        ORDER BY etf_ticker
        """
        df = pd.read_sql_query(query, conn)
        return df['etf_ticker'].tolist()
    except Exception as e:
        print(f"Error fetching ETFs: {e}")
        return []
    finally:
        conn.close() 

def fetch_latest_etf_metrics():
    """Fetch the latest rs_rank_index and rs_z_score for each ETF from fact_etf_metrics, including ETF name from dim_sector or dim_industry."""
    conn = get_db_connection()
    try:
        # Get the latest date in fact_etf_metrics
        latest_date_query = "SELECT MAX(date) FROM fact_etf_metrics"
        with conn.cursor() as cur:
            cur.execute(latest_date_query)
            latest_date = cur.fetchone()[0]
        if not latest_date:
            return pd.DataFrame()
        # Join with dim_sector and dim_industry for ETF names
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
        df = pd.read_sql_query(query, conn, params=[latest_date])
        return df
    except Exception as e:
        print(f"Error fetching latest ETF metrics: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_etf_ohlcv(etf_ticker, start_date=None, end_date=None):
    """Fetch OHLCV data for a given ETF from fact_index_ohlc."""
    conn = get_db_connection()
    try:
        query = """
        SELECT date, open, high, low, close, volume
        FROM fact_index_ohlc
        WHERE ticker = %s
        """
        params = [etf_ticker]
        if start_date and end_date:
            query += " AND date BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        query += " ORDER BY date"
        df = pd.read_sql_query(query, conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error fetching ETF OHLCV: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_etf_metrics(etf_ticker, start_date=None, end_date=None):
    """Fetch relative strength metrics for a given ETF from fact_etf_metrics."""
    conn = get_db_connection()
    try:
        query = """
        SELECT date, etf_ticker, ref_ticker, relative_strength, rs_rank_index, rs_rank_overall, rs_z_score
        FROM fact_etf_metrics
        WHERE etf_ticker = %s
        """
        params = [etf_ticker]
        if start_date and end_date:
            query += " AND date BETWEEN %s AND %s"
            params.extend([start_date, end_date])
        query += " ORDER BY date"
        df = pd.read_sql_query(query, conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error fetching ETF metrics: {e}")
        return pd.DataFrame()
    finally:
        conn.close() 

def get_option_db_connection():
    """Create a database connection to option_data."""
    try:
        conn = psycopg2.connect(**OPTION_DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to option_data database: {e}")
        return None


def get_all_option_symbols():
    """Get all unique symbols from underlying_data in option_data."""
    conn = get_option_db_connection()
    try:
        query = """
        SELECT DISTINCT symbol FROM underlying_data ORDER BY symbol
        """
        df = pd.read_sql_query(query, conn)
        return df['symbol'].tolist()
    except Exception as e:
        print(f"Error fetching option symbols: {e}")
        return []
    finally:
        conn.close()


def fetch_option_last_30_days(symbol):
    """Fetch last 30 calendar days of data for a symbol from underlying_data."""
    conn = get_option_db_connection()
    try:
        query = """
        SELECT * FROM underlying_data
        WHERE symbol = %s
        ORDER BY collection_date DESC
        LIMIT 30
        """
        df = pd.read_sql_query(query, conn, params=[symbol])
        df['collection_date'] = pd.to_datetime(df['collection_date'])
        return df.sort_values('collection_date', ascending=False)
    except Exception as e:
        print(f"Error fetching last 30 days for {symbol}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def fetch_option_latest_comparison(symbol):
    """Fetch the most recent row for a symbol from underlying_data and underlying_data_avg, and return a dict of {col: value/avg_value}."""
    conn = get_option_db_connection()
    try:
        # Get the most recent date for this symbol
        date_query = """
        SELECT MAX(collection_date) FROM underlying_data WHERE symbol = %s
        """
        with conn.cursor() as cur:
            cur.execute(date_query, (symbol,))
            latest_date = cur.fetchone()[0]
        if not latest_date:
            return {}
        # Get the row from both tables
        data_query = f"""
        SELECT * FROM underlying_data WHERE symbol = %s AND collection_date = %s
        """
        avg_query = f"""
        SELECT * FROM underlying_data_avg WHERE symbol = %s AND date = %s
        """
        df_data = pd.read_sql_query(data_query, conn, params=[symbol, latest_date])
        df_avg = pd.read_sql_query(avg_query, conn, params=[symbol, latest_date])
        if df_data.empty or df_avg.empty:
            return {}
        data_row = df_data.iloc[0]
        avg_row = df_avg.iloc[0]
        ratio_dict = {}
        for col in df_data.columns:
            if col in ['collection_date', 'symbol', 'created_at']:
                continue
            avg_col = col + '_avg' if col not in ['collection_date', 'symbol', 'created_at'] else col
            if avg_col in df_avg.columns:
                try:
                    ratio = data_row[col] / avg_row[avg_col] if avg_row[avg_col] not in [0, None, np.nan] else np.nan
                except Exception:
                    ratio = np.nan
                ratio_dict[col] = ratio
        return ratio_dict
    except Exception as e:
        print(f"Error fetching latest comparison for {symbol}: {e}")
        return {}
    finally:
        conn.close()


def fetch_option_column_history(symbol, column):
    """Fetch full history for a symbol/column from underlying_data and underlying_data_avg."""
    conn = get_option_db_connection()
    try:
        # Remove _avg if present for base column
        base_col = column.replace('_avg', '')
        avg_col = base_col + '_avg'
        # Query both tables for the column
        data_query = sql.SQL("""
            SELECT collection_date, {col} FROM underlying_data
            WHERE symbol = %s
            ORDER BY collection_date
        """).format(col=sql.Identifier(base_col))
        avg_query = sql.SQL("""
            SELECT date, {col} FROM underlying_data_avg
            WHERE symbol = %s
            ORDER BY date
        """).format(col=sql.Identifier(avg_col))
        df_data = pd.read_sql_query(data_query.as_string(conn), conn, params=[symbol])
        df_avg = pd.read_sql_query(avg_query.as_string(conn), conn, params=[symbol])
        df_data['collection_date'] = pd.to_datetime(df_data['collection_date'])
        df_avg['date'] = pd.to_datetime(df_avg['date'])
        return df_data, df_avg
    except Exception as e:
        print(f"Error fetching column history for {symbol}, {column}: {e}")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        conn.close() 

def fetch_option_screener_results(column, min_multiple):
    """Get all symbols and their latest multiples for a given column, filtered by minimum multiple threshold."""
    conn = get_option_db_connection()
    try:
        # Get the most recent date available
        latest_date_query = "SELECT MAX(collection_date) FROM underlying_data"
        with conn.cursor() as cur:
            cur.execute(latest_date_query)
            latest_date = cur.fetchone()[0]
        if not latest_date:
            return pd.DataFrame()
        
        # Query to get all symbols with their latest data and averages
        query = f"""
        WITH latest_data AS (
            SELECT symbol, {column}
            FROM underlying_data
            WHERE collection_date = %s
        ),
        latest_avg AS (
            SELECT symbol, {column}_avg
            FROM underlying_data_avg
            WHERE date = %s
        )
        SELECT 
            d.symbol,
            d.{column} as current_value,
            a.{column}_avg as avg_value,
            CASE 
                WHEN a.{column}_avg IS NOT NULL AND a.{column}_avg != 0 
                THEN d.{column} / a.{column}_avg 
                ELSE NULL 
            END as multiple
        FROM latest_data d
        JOIN latest_avg a ON d.symbol = a.symbol
        WHERE a.{column}_avg IS NOT NULL 
        AND a.{column}_avg != 0
        AND d.{column} / a.{column}_avg >= %s
        ORDER BY multiple DESC
        """
        
        df = pd.read_sql_query(query, conn, params=[latest_date, latest_date, min_multiple])
        return df
    except Exception as e:
        print(f"Error fetching option screener results for {column}: {e}")
        return pd.DataFrame()
    finally:
        conn.close() 

def get_option_numerical_columns():
    """Get all numerical columns from underlying_data table."""
    conn = get_option_db_connection()
    try:
        query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'underlying_data' 
        AND table_schema = 'public'
        AND data_type IN ('integer', 'bigint', 'numeric', 'double precision', 'real')
        AND column_name NOT IN ('collection_date', 'symbol', 'created_at')
        ORDER BY column_name
        """
        df = pd.read_sql_query(query, conn)
        return df['column_name'].tolist()
    except Exception as e:
        print(f"Error fetching option numerical columns: {e}")
        return []
    finally:
        conn.close() 