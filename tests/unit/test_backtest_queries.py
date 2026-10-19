"""Tests for queries/backtest_queries.py: cache keying, TTL and loaders (read_df faked)."""

import os
import time

import pandas as pd
import pytest

from stock_dashboard.config import get_settings
from stock_dashboard.queries import backtest_queries as bq


def ohlcv(ticker="AAA"):
    return pd.DataFrame(
        {
            "ticker": [ticker, ticker],
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "volume": [100, 200],
        }
    )


def reconfigure(monkeypatch, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()


class TestCachePath:
    def test_lives_in_configured_cache_dir(self, tmp_path):
        assert bq.cache_path("ohlcv").parent == tmp_path / "cache"

    @pytest.mark.parametrize(
        "env",
        [
            {"BENCHMARK_TICKER": "MKT"},
            {"STOCKS_DB_NAME": "demo"},
            {"POSTGRES_HOST": "elsewhere"},
            {"POSTGRES_PORT": "6543"},
        ],
    )
    def test_key_changes_with_source_and_benchmark(self, monkeypatch, env):
        """The old key was name + ETF filter only, so demo and real data shared files."""
        before = bq.cache_path("ohlcv")
        reconfigure(monkeypatch, **env)
        assert bq.cache_path("ohlcv") != before

    def test_key_changes_with_name_and_filter(self):
        assert bq.cache_path("ohlcv") != bq.cache_path("metrics")
        assert bq.cache_path("ohlcv") != bq.cache_path("ohlcv", "SEMX")
        assert bq.cache_path("ohlcv", None) == bq.cache_path("ohlcv")

    def test_password_not_part_of_key(self, monkeypatch):
        before = bq.cache_path("ohlcv")
        reconfigure(monkeypatch, POSTGRES_PASSWORD="secret")
        assert bq.cache_path("ohlcv") == before


class TestLoaders:
    def test_ohlcv_cached_after_first_load(self, fake_db):
        fake_db.returns(ohlcv())
        first = bq.load_ohlcv_universe()
        second = bq.load_ohlcv_universe()
        assert len(fake_db.calls) == 1
        assert isinstance(first["ticker"].dtype, pd.CategoricalDtype)
        assert first["open"].dtype == "float32"
        pd.testing.assert_frame_equal(first, second)
        assert bq.cache_path("ohlcv").exists()

    def test_stale_cache_reloads(self, fake_db):
        fake_db.returns(ohlcv(), ohlcv())
        bq.load_ohlcv_universe()
        path = bq.cache_path("ohlcv")
        old = time.time() - bq.CACHE_TTL_SECONDS - 1
        os.utime(path, (old, old))
        bq.load_ohlcv_universe()
        assert len(fake_db.calls) == 2

    def test_etf_filter_restricts_to_members(self, fake_db):
        fake_db.returns(pd.DataFrame({"ticker": ["AAA", "BBB"]}), ohlcv())
        bq.load_ohlcv_universe("SEMX")
        members, prices = fake_db.calls
        assert members.params == {"etf": "SEMX"}
        assert prices.params["tickers"] == ["AAA", "BBB"]

    def test_metrics_aggregates_per_date_and_ticker(self, fake_db):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-02"]),
                "ticker": ["AAA"],
                "rs_rank_overall": [0.5],
                "rs_z_score": [1.0],
            }
        )
        fake_db.returns(frame, frame)
        bq.load_metrics_universe()
        assert "GROUP BY date, ticker" in fake_db.last.sql
        assert "WHERE etf_ticker" not in fake_db.last.sql
        bq.load_metrics_universe("SEMX")
        assert "WHERE etf_ticker = %(etf)s" in fake_db.last.sql
        assert fake_db.last.params == {"etf": "SEMX"}

    def test_benchmark_comes_from_settings(self, fake_db, monkeypatch):
        """The old loader hardcoded SPY."""
        reconfigure(monkeypatch, BENCHMARK_TICKER="MKT")
        fake_db.returns(ohlcv("MKT"))
        df = bq.load_benchmark_ohlcv()
        assert fake_db.last.params["tickers"] == ["MKT"]
        assert '"fact_index_ohlc"' in fake_db.last.sql
        assert "ticker" not in df.columns
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]


def test_clear_cache(fake_db):
    assert bq.clear_cache() == 0
    fake_db.returns(ohlcv(), ohlcv())
    bq.load_ohlcv_universe()
    bq.load_ohlcv_universe("SEMX")
    assert bq.clear_cache() == 2
    assert not list(get_settings().cache_dir.glob("*.parquet"))
