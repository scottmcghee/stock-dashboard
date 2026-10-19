"""Shared fixtures for the unit tests: no database, isolated settings, sample frames."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest
from psycopg2 import sql

from stock_dashboard.config import get_settings

_SETTINGS_ENV = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "STOCKS_DB_NAME",
    "OPTION_DB_NAME",
    "BENCHMARK_TICKER",
)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    """Default settings, with cache and preset directories under tmp_path."""
    for name in _SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PRESET_DIR", str(tmp_path / "presets"))
    get_settings.cache_clear()
    from stock_dashboard.queries import option_queries

    option_queries._allowed_columns.cache_clear()
    yield
    get_settings.cache_clear()
    option_queries._allowed_columns.cache_clear()


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """Make any attempt to open a real database connection fail loudly."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("unit tests must not connect to a database")

    monkeypatch.setattr("stock_dashboard.db.ThreadedConnectionPool", refuse)


# ---------------------------------------------------------------------------
# Fake read_df
# ---------------------------------------------------------------------------


def render_sql(query: str | sql.Composable) -> str:
    """Render a query to text without a connection, with whitespace collapsed."""

    def render(part: Any) -> str:
        if isinstance(part, str):
            return part
        if isinstance(part, sql.Composed):
            return "".join(render(p) for p in part.seq)
        if isinstance(part, sql.SQL):
            return part.string
        if isinstance(part, sql.Identifier):
            return ".".join(f'"{s}"' for s in part.strings)
        if isinstance(part, sql.Placeholder):
            return f"%({part.name})s" if part.name else "%s"
        if isinstance(part, sql.Literal):
            return repr(part.wrapped)
        raise TypeError(f"Cannot render {part!r}")

    return " ".join(render(query).split())


@dataclass
class Call:
    database: str
    sql: str
    params: Any


@dataclass
class FakeDb:
    """Stands in for ``db.read_df``: records calls, returns queued frames.

    A queued item may be a DataFrame or a callable taking the Call.
    Once the queue is empty, ``default`` answers (an empty frame by default).
    """

    calls: list[Call] = field(default_factory=list)
    queue: list[pd.DataFrame | Callable[[Call], pd.DataFrame]] = field(default_factory=list)
    default: Callable[[Call], pd.DataFrame] = lambda call: pd.DataFrame()

    def returns(self, *items: pd.DataFrame | Callable[[Call], pd.DataFrame]) -> FakeDb:
        self.queue.extend(items)
        return self

    def __call__(self, database: str, query: str | sql.Composable, params: Any = None):
        call = Call(str(database), render_sql(query), params)
        self.calls.append(call)
        item = self.queue.pop(0) if self.queue else self.default
        return item(call) if callable(item) else item.copy()

    @property
    def last(self) -> Call:
        return self.calls[-1]


@pytest.fixture
def fake_db(monkeypatch) -> FakeDb:
    fake = FakeDb()
    monkeypatch.setattr("stock_dashboard.db.read_df", fake)
    return fake


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """200 rows of synthetic OHLCV data for one ticker."""
    np.random.seed(42)
    dates = pd.bdate_range("2025-06-01", periods=200)
    close = 100 + np.cumsum(np.random.randn(200) * 0.5)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - np.random.rand(200) * 0.5,
            "high": close + np.random.rand(200) * 1.0,
            "low": close - np.random.rand(200) * 1.0,
            "close": close,
            "volume": np.random.randint(100_000, 5_000_000, size=200),
        }
    )


@pytest.fixture
def sample_ohlcv_short() -> pd.DataFrame:
    """30 rows: too short for the 150-day MA (Undetermined stage)."""
    dates = pd.bdate_range("2025-06-01", periods=30)
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(100, 110, 30),
            "high": np.linspace(101, 111, 30),
            "low": np.linspace(99, 109, 30),
            "close": np.linspace(100, 110, 30),
            "volume": [1_000_000] * 30,
        }
    )


@pytest.fixture
def sample_metrics() -> pd.DataFrame:
    """300 rows of synthetic RS metrics."""
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-01", periods=300)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "etf_ticker": "XLK",
            "relative_strength": 1.0 + np.cumsum(np.random.randn(300) * 0.01),
            "rs_rank_index": np.random.rand(300),
            "rs_rank_overall": np.random.rand(300),
            "rs_z_score": np.random.randn(300),
        }
    )


@pytest.fixture
def sample_multi_ticker_ohlcv() -> pd.DataFrame:
    """OHLCV for 3 tickers, 200 days each."""
    np.random.seed(42)
    frames = []
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        dates = pd.bdate_range("2025-06-01", periods=200)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": np.random.randint(100_000, 5_000_000, size=200),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
