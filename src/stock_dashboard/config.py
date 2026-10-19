"""Runtime settings, read from environment variables.

Nothing secret has a default: the database password is empty unless
``POSTGRES_PASSWORD`` is set, and it is kept out of ``repr``.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path

# Streamlit ``st.cache_data`` lifetimes, in seconds.
CACHE_TTL_SHORT = 60  # dropdown lists, metadata
CACHE_TTL_DEFAULT = 300  # OHLCV, metrics
CACHE_TTL_LONG = 900  # ETF grid and impulse, 52-week highs and lows

# The app links to the docs on GitHub: an installed package or the Docker image has no docs/.
REPO_URL = "https://github.com/scottmcghee/stock-dashboard"
README_URL = f"{REPO_URL}#readme"
ENV_EXAMPLE_URL = f"{REPO_URL}/blob/main/.env.example"
DOCS_URL = f"{REPO_URL}/blob/main/docs"
LIMITATIONS_URL = f"{DOCS_URL}/DESIGN.md#limitations"
DATA_CONTRACT_URL = f"{DOCS_URL}/DATA_CONTRACT.md"


def _env(name: str, default: str) -> str:
    """``name`` from the environment; unset or empty means ``default``."""
    return os.getenv(name) or default


def _env_path(name: str, default: str) -> Path:
    return Path(_env(name, default)).expanduser()


@dataclass(frozen=True)
class Settings:
    db_host: str = field(default_factory=lambda: _env("POSTGRES_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(_env("POSTGRES_PORT", "5432")))
    db_user: str = field(default_factory=lambda: _env("POSTGRES_USER", "postgres"))
    db_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""), repr=False)
    stocks_db: str = field(default_factory=lambda: _env("STOCKS_DB_NAME", "stocks"))
    option_db: str = field(default_factory=lambda: _env("OPTION_DB_NAME", "option_data"))
    # Ticker every ETF's relative strength is measured against (fact_etf_metrics.ref_ticker);
    # the backtester also classifies market regimes from its prices.
    benchmark: str = field(
        default_factory=lambda: _env("BENCHMARK_TICKER", "SPY").strip().upper() or "SPY"
    )
    preset_dir: Path = field(
        default_factory=lambda: _env_path("PRESET_DIR", "~/.config/stock_dashboard/presets")
    )
    cache_dir: Path = field(
        default_factory=lambda: _env_path("CACHE_DIR", "~/.cache/stock_dashboard")
    )

    def db_kwargs(self, dbname: str) -> dict[str, object]:
        """Keyword arguments for ``psycopg2.connect`` to database ``dbname``."""
        return {
            "dbname": dbname,
            "user": self.db_user,
            "password": self.db_password,
            "host": self.db_host,
            "port": self.db_port,
        }


@functools.cache
def get_settings() -> Settings:
    """Settings read from the environment once per process (``cache_clear()`` to re-read)."""
    return Settings()
