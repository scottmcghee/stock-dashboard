"""ENV-based configuration with .env fallback defaults."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# PostgreSQL credentials
_PG_USER = os.getenv("POSTGRES_USER", "postgres")
_PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
_PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
_PG_PORT = os.getenv("POSTGRES_PORT", "5432")

STOCKS_DB_CONFIG = {
    "dbname": os.getenv("STOCKS_DB_NAME", "stocks"),
    "user": _PG_USER,
    "password": _PG_PASSWORD,
    "host": _PG_HOST,
    "port": _PG_PORT,
}

OPTION_DB_CONFIG = {
    "dbname": os.getenv("OPTION_DB_NAME", "option_data"),
    "user": _PG_USER,
    "password": _PG_PASSWORD,
    "host": _PG_HOST,
    "port": _PG_PORT,
}

# Cache TTL (seconds) for Streamlit @st.cache_data
CACHE_TTL_SHORT = 60       # Dropdown lists, metadata
CACHE_TTL_DEFAULT = 300    # OHLCV, metrics
CACHE_TTL_LONG = 900       # ETF grid, screener results
