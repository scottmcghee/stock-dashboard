"""Tests for queries/option_queries.py — all DB calls mocked."""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


class TestGetAllOptionSymbols:
    def test_returns_list(self, mock_option_pool):
        mock_df = pd.DataFrame({"symbol": ["AAPL", "TSLA"]})
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.option_queries import get_all_option_symbols
            result = get_all_option_symbols()
            assert result == ["AAPL", "TSLA"]


class TestGetOptionNumericalColumns:
    def test_returns_column_names(self, mock_option_pool):
        mock_df = pd.DataFrame({"column_name": ["dxoi", "gxoi", "flownet"]})
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.option_queries import get_option_numerical_columns
            result = get_option_numerical_columns()
            assert "dxoi" in result
            assert "gxoi" in result


class TestFetchOptionLast30Days:
    def test_returns_sorted_descending(self, mock_option_pool):
        dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
        mock_df = pd.DataFrame({
            "collection_date": dates,
            "symbol": ["AAPL"] * 3,
            "dxoi": [100, 200, 300],
        })
        with patch("pandas.read_sql_query", return_value=mock_df):
            from queries.option_queries import fetch_option_last_30_days
            result = fetch_option_last_30_days("AAPL")
            assert result["collection_date"].is_monotonic_decreasing


class TestFetchOptionLatestComparison:
    def test_returns_ratios(self, mock_option_pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("2025-01-03",)
        mock_option_pool.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_option_pool.cursor.return_value.__exit__ = MagicMock(return_value=False)

        df_data = pd.DataFrame({
            "collection_date": ["2025-01-03"],
            "symbol": ["AAPL"],
            "dxoi": [200.0],
        })
        df_avg = pd.DataFrame({
            "date": ["2025-01-03"],
            "symbol": ["AAPL"],
            "dxoi_avg": [100.0],
        })

        with patch("pandas.read_sql_query", side_effect=[df_data, df_avg]):
            from queries.option_queries import fetch_option_latest_comparison
            result = fetch_option_latest_comparison("AAPL")
            assert abs(result["dxoi"] - 2.0) < 0.01

    def test_returns_empty_when_no_data(self, mock_option_pool):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (None,)
        mock_option_pool.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_option_pool.cursor.return_value.__exit__ = MagicMock(return_value=False)

        from queries.option_queries import fetch_option_latest_comparison
        result = fetch_option_latest_comparison("NONEXIST")
        assert result == {}


class TestFetchOptionColumnHistory:
    def test_returns_two_dataframes(self, mock_option_pool):
        import queries.option_queries as oq
        oq._ALLOWED_COLUMNS = {"dxoi", "gxoi", "flownet"}

        mock_cursor = MagicMock()
        exec_count = [0]

        def execute_side_effect(*args, **kwargs):
            exec_count[0] += 1
            if exec_count[0] == 1:
                mock_cursor.description = [("collection_date",), ("dxoi",)]
            else:
                mock_cursor.description = [("date",), ("dxoi_avg",)]

        mock_cursor.execute.side_effect = execute_side_effect
        mock_cursor.fetchall.side_effect = [
            [("2025-01-01", 100.0)],
            [("2025-01-01", 90.0)],
        ]
        mock_option_pool.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_option_pool.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result_data, result_avg = oq.fetch_option_column_history("AAPL", "dxoi")
        assert "dxoi" in result_data.columns
        assert "dxoi_avg" in result_avg.columns


class TestFetchOptionScreenerResults:
    def test_validates_column_name(self, mock_option_pool):
        """SQL injection attempt should raise ValueError."""
        # Reset the whitelist cache
        import queries.option_queries as oq
        oq._ALLOWED_COLUMNS = {"dxoi", "gxoi", "flownet"}

        with pytest.raises(ValueError, match="Invalid column name"):
            oq.fetch_option_screener_results("dxoi; DROP TABLE--", 5.0)

    def test_valid_column_succeeds(self, mock_option_pool):
        import queries.option_queries as oq
        oq._ALLOWED_COLUMNS = {"dxoi", "gxoi", "flownet"}

        mock_cursor = MagicMock()
        # First call: fetchone for latest_date
        mock_cursor.fetchone.return_value = ("2025-01-03",)
        # Second call: fetchall for results
        mock_cursor.description = [
            ("symbol",), ("current_value",), ("avg_value",), ("multiple",)
        ]
        mock_cursor.fetchall.return_value = [("AAPL", 500.0, 100.0, 5.0)]
        mock_option_pool.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_option_pool.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = oq.fetch_option_screener_results("dxoi", 5.0)
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "AAPL"

    def test_empty_when_no_latest_date(self, mock_option_pool):
        import queries.option_queries as oq
        oq._ALLOWED_COLUMNS = {"dxoi"}

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (None,)
        mock_option_pool.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_option_pool.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = oq.fetch_option_screener_results("dxoi", 5.0)
        assert result.empty
