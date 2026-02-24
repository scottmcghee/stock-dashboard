"""Shared fixtures for stock_dashboard tests."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# Ensure tests never hit a real database
@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """Prevent any real DB connections during tests."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_only")


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """200 rows of synthetic OHLCV data for one ticker."""
    np.random.seed(42)
    dates = pd.bdate_range("2025-06-01", periods=200)
    close = 100 + np.cumsum(np.random.randn(200) * 0.5)
    return pd.DataFrame({
        "date": dates,
        "open": close - np.random.rand(200) * 0.5,
        "high": close + np.random.rand(200) * 1.0,
        "low": close - np.random.rand(200) * 1.0,
        "close": close,
        "volume": np.random.randint(100_000, 5_000_000, size=200),
    })


@pytest.fixture
def sample_ohlcv_short() -> pd.DataFrame:
    """30 rows — too short for 150-day MA (Undetermined stage)."""
    dates = pd.bdate_range("2025-06-01", periods=30)
    return pd.DataFrame({
        "date": dates,
        "open": np.linspace(100, 110, 30),
        "high": np.linspace(101, 111, 30),
        "low": np.linspace(99, 109, 30),
        "close": np.linspace(100, 110, 30),
        "volume": [1_000_000] * 30,
    })


@pytest.fixture
def sample_metrics() -> pd.DataFrame:
    """300 rows of synthetic RS metrics."""
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-01", periods=300)
    return pd.DataFrame({
        "date": dates,
        "ticker": "AAPL",
        "etf_ticker": "XLK",
        "relative_strength": 1.0 + np.cumsum(np.random.randn(300) * 0.01),
        "rs_rank_index": np.random.rand(300),
        "rs_rank_overall": np.random.rand(300),
        "rs_z_score": np.random.randn(300),
    })


@pytest.fixture
def sample_multi_ticker_ohlcv() -> pd.DataFrame:
    """OHLCV for 3 tickers, 200 days each."""
    np.random.seed(42)
    frames = []
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        dates = pd.bdate_range("2025-06-01", periods=200)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "date": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.random.randint(100_000, 5_000_000, size=200),
        }))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Mock DB pools
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_stocks_pool():
    """Mock that patches db._get_stocks_pool."""
    mock_conn = MagicMock()
    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    with patch("db._get_stocks_pool", return_value=mock_pool):
        yield mock_conn


@pytest.fixture
def mock_option_pool():
    """Mock that patches db._get_option_pool."""
    mock_conn = MagicMock()
    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    with patch("db._get_option_pool", return_value=mock_pool):
        yield mock_conn
