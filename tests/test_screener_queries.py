"""Tests for queries/screener_queries.py — all DB calls mocked."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


class TestFetchEtfRsTickers:
    def test_returns_matching_tickers(self, mock_stocks_pool):
        mock_df = pd.DataFrame({"ticker": ["AAPL", "MSFT", "NVDA"]})
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.screener_queries import fetch_etf_rs_tickers
            result = fetch_etf_rs_tickers(0.7)
            assert result == {"AAPL", "MSFT", "NVDA"}

    def test_empty_result(self, mock_stocks_pool):
        with patch("pandas.read_sql_query", return_value=pd.DataFrame({"ticker": []})):
            from queries.screener_queries import fetch_etf_rs_tickers
            result = fetch_etf_rs_tickers(0.99)
            assert result == set()


class TestFetchOptionVolumeSpikeSymbols:
    def test_returns_symbols(self, mock_option_pool):
        mock_df = pd.DataFrame({"symbol": ["AAPL", "TSLA"]})
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.screener_queries import fetch_option_volume_spike_symbols
            result = fetch_option_volume_spike_symbols(2.0)
            assert result == {"AAPL", "TSLA"}

    def test_empty_result(self, mock_option_pool):
        with patch("pandas.read_sql_query", return_value=pd.DataFrame({"symbol": []})):
            from queries.screener_queries import fetch_option_volume_spike_symbols
            result = fetch_option_volume_spike_symbols(5.0)
            assert result == set()


class TestFetchSignalScanSymbols:
    def test_intersection_of_two_scans(self):
        scan1_df = pd.DataFrame({"symbol": ["AAPL", "MSFT", "NVDA"], "signal_value": [1, 2, 3]})
        scan2_df = pd.DataFrame({"symbol": ["MSFT", "NVDA", "TSLA"], "signal_value": [4, 5, 6]})
        with patch("queries.signal_queries.scan_vanna_extreme", return_value=scan1_df), \
             patch("queries.signal_queries.scan_volume_spike", return_value=scan2_df):
            from queries.screener_queries import fetch_signal_scan_symbols
            result = fetch_signal_scan_symbols(["vanna_extreme", "volume_spike"])
            assert result == {"MSFT", "NVDA"}

    def test_empty_scan_returns_empty(self):
        scan1_df = pd.DataFrame({"symbol": ["AAPL"], "signal_value": [1]})
        with patch("queries.signal_queries.scan_vanna_extreme", return_value=scan1_df), \
             patch("queries.signal_queries.scan_volume_spike", return_value=pd.DataFrame()):
            from queries.screener_queries import fetch_signal_scan_symbols
            result = fetch_signal_scan_symbols(["vanna_extreme", "volume_spike"])
            assert result == set()

    def test_empty_names_returns_empty(self):
        from queries.screener_queries import fetch_signal_scan_symbols
        result = fetch_signal_scan_symbols([])
        assert result == set()


class TestFetchBulkOhlcvForTa:
    def test_returns_dataframe(self, mock_stocks_pool):
        mock_df = pd.DataFrame({
            "ticker": ["AAPL"] * 5,
            "date": pd.to_datetime(["2025-01-01"] * 5),
            "open": [100.0] * 5,
            "high": [102.0] * 5,
            "low": [99.0] * 5,
            "close": [101.0] * 5,
            "volume": [1000] * 5,
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.screener_queries import fetch_bulk_ohlcv_for_ta
            result = fetch_bulk_ohlcv_for_ta(["AAPL"])
            assert len(result) == 5
            assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_empty_tickers(self):
        from queries.screener_queries import fetch_bulk_ohlcv_for_ta
        result = fetch_bulk_ohlcv_for_ta([])
        assert result.empty


class TestSearchStocksRestrictTickers:
    """Test the restrict_tickers parameter added to search_stocks_by_metrics."""

    def test_restrict_tickers_adds_in_clause(self, mock_stocks_pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("2025-01-01",)
        mock_stocks_pool.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_stocks_pool.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_df = pd.DataFrame({
            "ticker": ["AAPL"],
            "etf_ticker": ["XLK"],
            "relative_strength": [1.05],
            "rs_rank_index": [0.85],
            "rs_rank_overall": [0.90],
            "rs_z_score": [1.2],
            "yearly_avg_volume": [1000000],
            "yearly_std_volume": [500000],
            "avg_30d_volume": [1200000],
            "current_volume": [2000000],
            "volume_z_score": [2.0],
        })
        with patch("pandas.read_sql_query", return_value=mock_df) as mock_read:
            from queries.stocks_queries import search_stocks_by_metrics
            search_stocks_by_metrics(restrict_tickers=["AAPL", "MSFT"])
            call_sql = mock_read.call_args[0][0]
            assert "m.ticker IN" in call_sql

    def test_no_restrict_tickers_no_in_clause(self, mock_stocks_pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("2025-01-01",)
        mock_stocks_pool.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_stocks_pool.cursor.return_value.__exit__ = MagicMock(return_value=False)

        mock_df = pd.DataFrame({
            "ticker": ["AAPL"],
            "etf_ticker": ["XLK"],
            "relative_strength": [1.05],
            "rs_rank_index": [0.85],
            "rs_rank_overall": [0.90],
            "rs_z_score": [1.2],
            "yearly_avg_volume": [1000000],
            "yearly_std_volume": [500000],
            "avg_30d_volume": [1200000],
            "current_volume": [2000000],
            "volume_z_score": [2.0],
        })
        with patch("pandas.read_sql_query", return_value=mock_df) as mock_read:
            from queries.stocks_queries import search_stocks_by_metrics
            search_stocks_by_metrics()
            call_sql = mock_read.call_args[0][0]
            assert "m.ticker IN" not in call_sql
