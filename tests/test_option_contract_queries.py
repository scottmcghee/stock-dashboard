"""Tests for queries/option_contract_queries.py — all DB calls mocked."""

from unittest.mock import MagicMock, patch

import pandas as pd


class TestGetTickersWithOptionData:
    def test_returns_list(self, mock_option_pool):
        mock_df = pd.DataFrame({"symbol": ["AAPL", "SPY", "TSLA"]})
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.option_contract_queries import get_tickers_with_option_data
            result = get_tickers_with_option_data()
            assert result == ["AAPL", "SPY", "TSLA"]


class TestGetExpirations:
    def test_returns_timestamps_sorted(self, mock_option_pool):
        mock_df = pd.DataFrame({
            "expiration_date": ["2026-05-16", "2026-06-20", "2026-07-18"],
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.option_contract_queries import get_expirations
            result = get_expirations("AAPL")
            assert len(result) == 3
            assert all(isinstance(r, pd.Timestamp) for r in result)

    def test_empty_when_no_data(self, mock_option_pool):
        with patch("pandas.read_sql_query", return_value=pd.DataFrame({"expiration_date": []})):
            from queries.option_contract_queries import get_expirations
            result = get_expirations("NONEXIST")
            assert result == []


class TestFetchOiByStrike:
    def test_returns_oi_snapshot(self, mock_option_pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("2026-04-17",)
        mock_option_pool.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_option_pool.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_df = pd.DataFrame({
            "strike": [100.0, 100.0, 105.0, 105.0],
            "option_type": ["call", "put", "call", "put"],
            "oi": [500, 300, 200, 400],
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.option_contract_queries import fetch_oi_by_strike
            result = fetch_oi_by_strike("AAPL", "2026-05-16")
            assert len(result) == 4
            assert set(result.columns) == {"strike", "option_type", "oi"}

    def test_empty_when_no_latest_date(self, mock_option_pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (None,)
        mock_option_pool.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_option_pool.cursor.return_value.__exit__ = MagicMock(return_value=False)

        from queries.option_contract_queries import fetch_oi_by_strike
        result = fetch_oi_by_strike("AAPL", "2026-05-16")
        assert result.empty
        assert list(result.columns) == ["strike", "option_type", "oi"]


class TestFetchContractOhlcHistory:
    def test_returns_ohlc_frame(self, mock_option_pool):
        mock_df = pd.DataFrame({
            "collection_date": ["2026-04-01", "2026-04-02", "2026-04-03"],
            "open":  [1.20, 1.25, 1.30],
            "high":  [1.35, 1.40, 1.45],
            "low":   [1.15, 1.20, 1.25],
            "close": [1.30, 1.35, 1.42],
            "oi":    [500, 520, 530],
            "day_volume": [100, 150, 80],
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.option_contract_queries import fetch_contract_ohlc_history
            result = fetch_contract_ohlc_history("AAPL", "2026-05-16", 100.0, "call")
            assert len(result) == 3
            assert pd.api.types.is_datetime64_any_dtype(result["collection_date"])

    def test_drops_rows_without_usable_close(self, mock_option_pool):
        mock_df = pd.DataFrame({
            "collection_date": ["2026-04-01", "2026-04-02", "2026-04-03"],
            "open":  [1.20, 1.25, 1.30],
            "high":  [1.35, 1.40, 1.45],
            "low":   [1.15, 1.20, 1.25],
            "close": [1.30, None, 0.0],
            "oi":    [500, 520, 530],
            "day_volume": [100, 0, 0],
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.option_contract_queries import fetch_contract_ohlc_history
            result = fetch_contract_ohlc_history("AAPL", "2026-05-16", 100.0, "call")
            assert len(result) == 1
            assert result.iloc[0]["close"] == 1.30

    def test_fills_missing_ohl_with_close(self, mock_option_pool):
        """If day_open/high/low are 0/NULL, fall back to the close (price)."""
        mock_df = pd.DataFrame({
            "collection_date": ["2026-04-01"],
            "open":  [0.0],
            "high":  [None],
            "low":   [0.0],
            "close": [1.42],
            "oi":    [500],
            "day_volume": [10],
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.option_contract_queries import fetch_contract_ohlc_history
            result = fetch_contract_ohlc_history("AAPL", "2026-05-16", 100.0, "call")
            row = result.iloc[0]
            assert row["open"] == 1.42
            assert row["high"] == 1.42
            assert row["low"] == 1.42

    def test_empty_when_no_data(self, mock_option_pool):
        empty = pd.DataFrame(columns=[
            "collection_date", "open", "high", "low", "close", "oi", "day_volume",
        ])
        with patch("pandas.read_sql_query", return_value=empty):
            from queries.option_contract_queries import fetch_contract_ohlc_history
            result = fetch_contract_ohlc_history("AAPL", "2026-05-16", 100.0, "call")
            assert result.empty
