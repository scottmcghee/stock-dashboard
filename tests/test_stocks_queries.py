"""Tests for queries/stocks_queries.py — all DB calls mocked."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


class TestGetAllTickers:
    def test_returns_sorted_list(self, mock_stocks_pool):
        mock_df = pd.DataFrame({"ticker": ["AAPL", "MSFT", "TSLA"]})
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.stocks_queries import get_all_tickers
            result = get_all_tickers()
            assert result == ["AAPL", "MSFT", "TSLA"]

    def test_empty_table(self, mock_stocks_pool):
        with patch("pandas.read_sql_query", return_value=pd.DataFrame({"ticker": []})):
            from queries.stocks_queries import get_all_tickers
            result = get_all_tickers()
            assert result == []


class TestGetAllEtfs:
    def test_returns_list(self, mock_stocks_pool):
        mock_df = pd.DataFrame({"etf_ticker": ["XLK", "XLF"]})
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.stocks_queries import get_all_etfs
            result = get_all_etfs()
            assert result == ["XLK", "XLF"]


class TestFetchStockOhlcv:
    def test_returns_dataframe_with_dates(self, mock_stocks_pool):
        mock_df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 2000],
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.stocks_queries import fetch_stock_ohlcv
            result = fetch_stock_ohlcv("AAPL", "2025-01-01", "2025-01-02")
            assert len(result) == 2
            assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_no_date_range(self, mock_stocks_pool):
        mock_df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"]),
            "open": [100.0], "high": [102.0], "low": [99.0],
            "close": [101.0], "volume": [1000],
        })
        with patch("pandas.read_sql_query", return_value=mock_df) as mock_read:
            from queries.stocks_queries import fetch_stock_ohlcv
            fetch_stock_ohlcv("AAPL")
            # Should not include BETWEEN clause
            call_sql = mock_read.call_args[0][0]
            assert "BETWEEN" not in call_sql

    def test_empty_result(self, mock_stocks_pool):
        with patch("pandas.read_sql_query", return_value=pd.DataFrame()):
            from queries.stocks_queries import fetch_stock_ohlcv
            result = fetch_stock_ohlcv("INVALID")
            assert result.empty


class TestFetchStockMetrics:
    def test_returns_metrics(self, mock_stocks_pool):
        mock_df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"]),
            "ticker": ["AAPL"],
            "etf_ticker": ["XLK"],
            "relative_strength": [1.05],
            "rs_rank_index": [0.85],
            "rs_rank_overall": [0.90],
            "rs_z_score": [1.2],
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.stocks_queries import fetch_stock_metrics
            result = fetch_stock_metrics("AAPL", "2025-01-01", "2025-01-02")
            assert "relative_strength" in result.columns


class TestFetchLatestEtfGrid:
    def test_returns_grid(self, mock_stocks_pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("2025-01-01",)
        mock_stocks_pool.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_stocks_pool.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_df = pd.DataFrame({
            "etf_ticker": ["XLK"],
            "etf_name": ["Technology"],
            "rs_rank_index": [0.85],
            "rs_z_score": [1.2],
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.stocks_queries import fetch_latest_etf_grid
            result = fetch_latest_etf_grid()
            assert "etf_ticker" in result.columns

    def test_no_data_returns_empty(self, mock_stocks_pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (None,)
        mock_stocks_pool.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_stocks_pool.cursor.return_value.__exit__ = MagicMock(return_value=False)

        from queries.stocks_queries import fetch_latest_etf_grid
        result = fetch_latest_etf_grid()
        assert result.empty


class TestFetchEtfMetrics:
    def test_with_date_range(self, mock_stocks_pool):
        mock_df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"]),
            "etf_ticker": ["XLK"],
            "ref_ticker": ["SPY"],
            "relative_strength": [1.05],
            "rs_rank_index": [0.85],
            "rs_rank_overall": [0.90],
            "rs_z_score": [1.2],
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.stocks_queries import fetch_etf_metrics
            result = fetch_etf_metrics("XLK", "2025-01-01", "2025-06-01")
            assert len(result) == 1


class TestFetchBulkWeinsteinData:
    def test_returns_multi_ticker_data(self, mock_stocks_pool):
        mock_df = pd.DataFrame({
            "ticker": ["AAPL"] * 5 + ["MSFT"] * 5,
            "date": pd.to_datetime(["2025-01-01"] * 5 + ["2025-01-01"] * 5),
            "open": [100.0] * 10,
            "high": [102.0] * 10,
            "low": [99.0] * 10,
            "close": [101.0] * 10,
            "volume": [1000] * 10,
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.stocks_queries import fetch_bulk_weinstein_data
            result = fetch_bulk_weinstein_data(mock_stocks_pool, ["AAPL", "MSFT"])
            assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}

    def test_empty_tickers_list(self, mock_stocks_pool):
        from queries.stocks_queries import fetch_bulk_weinstein_data
        result = fetch_bulk_weinstein_data(mock_stocks_pool, [])
        assert result.empty
