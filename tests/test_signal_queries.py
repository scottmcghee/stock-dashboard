"""Tests for queries/signal_queries.py — all DB calls mocked."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


@pytest.fixture
def mock_both_pools():
    """Mock both DB pools."""
    mock_opt_conn = MagicMock()
    mock_opt_pool = MagicMock()
    mock_opt_pool.getconn.return_value = mock_opt_conn

    mock_stk_conn = MagicMock()
    mock_stk_pool = MagicMock()
    mock_stk_pool.getconn.return_value = mock_stk_conn

    with patch("db._get_option_pool", return_value=mock_opt_pool), \
         patch("db._get_stocks_pool", return_value=mock_stk_pool):
        yield mock_opt_conn, mock_stk_conn


class TestScanVannaExtreme:
    def test_returns_results(self, mock_both_pools):
        opt_conn, stk_conn = mock_both_pools
        scan_df = pd.DataFrame({
            "symbol": ["AAPL", "NVDA"],
            "signal_value": [50000.0, -30000.0],
            "pct_rank": [0.95, 0.05],
            "signal_label": ["High Vanna", "Low Vanna"],
        })
        rs_df = pd.DataFrame({
            "symbol": ["AAPL", "NVDA"],
            "etf_ticker": ["XLK", "XLK"],
            "rs_rank_index": [0.85, 0.90],
            "rs_rank_overall": [0.80, 0.88],
            "rs_z_score": [1.2, 1.5],
        })
        with patch("pandas.read_sql_query", side_effect=[scan_df, rs_df]):
            from queries.signal_queries import scan_vanna_extreme
            result = scan_vanna_extreme(20.0)
            assert len(result) == 2
            assert "rs_rank_index" in result.columns

    def test_empty_when_no_extremes(self, mock_both_pools):
        with patch("pandas.read_sql_query", return_value=pd.DataFrame()):
            from queries.signal_queries import scan_vanna_extreme
            result = scan_vanna_extreme(20.0)
            assert result.empty


class TestScanFlownetExtreme:
    def test_returns_contrarian_signals(self, mock_both_pools):
        opt_conn, stk_conn = mock_both_pools
        scan_df = pd.DataFrame({
            "symbol": ["TSLA"],
            "signal_value": [-74600000.0],
            "pct_rank": [0.03],
            "signal_label": ["Extreme Negative Flow"],
        })
        rs_df = pd.DataFrame({
            "symbol": ["TSLA"],
            "etf_ticker": ["ARKK"],
            "rs_rank_index": [0.70],
            "rs_rank_overall": [0.65],
            "rs_z_score": [0.5],
        })
        with patch("pandas.read_sql_query", side_effect=[scan_df, rs_df]):
            from queries.signal_queries import scan_flownet_extreme
            result = scan_flownet_extreme(10.0)
            assert len(result) == 1
            assert result.iloc[0]["signal_label"] == "Extreme Negative Flow"


class TestScanVolumeSpike:
    def test_returns_spikes(self, mock_both_pools):
        scan_df = pd.DataFrame({
            "symbol": ["AMD"],
            "signal_value": [500000],
            "vol_ratio": [3.5],
            "avg_20d_volume": [142857.0],
            "signal_label": ["Extreme Spike (>3x)"],
        })
        rs_df = pd.DataFrame({
            "symbol": ["AMD"],
            "etf_ticker": ["XLK"],
            "rs_rank_index": [0.75],
            "rs_rank_overall": [0.72],
            "rs_z_score": [0.8],
        })
        with patch("pandas.read_sql_query", side_effect=[scan_df, rs_df]):
            from queries.signal_queries import scan_volume_spike
            result = scan_volume_spike(2.0)
            assert len(result) == 1
            assert "vol_ratio" in result.columns


class TestScanDxoiChange:
    def test_returns_big_changes(self, mock_both_pools):
        scan_df = pd.DataFrame({
            "symbol": ["META"],
            "signal_value": [15000.0],
            "prev_dxoi": [10000.0],
            "pct_change": [50.0],
            "signal_label": ["Big Increase"],
        })
        rs_df = pd.DataFrame({
            "symbol": ["META"],
            "etf_ticker": ["XLC"],
            "rs_rank_index": [0.88],
            "rs_rank_overall": [0.85],
            "rs_z_score": [1.8],
        })
        with patch("pandas.read_sql_query", side_effect=[scan_df, rs_df]):
            from queries.signal_queries import scan_dxoi_change
            result = scan_dxoi_change(20.0)
            assert len(result) == 1
            assert result.iloc[0]["signal_label"] == "Big Increase"


class TestScanGexLevel:
    def test_returns_all_regimes(self, mock_both_pools):
        scan_df = pd.DataFrame({
            "symbol": ["AAPL", "TSLA", "NVDA"],
            "signal_value": [-5000.0, 3000.0, 60000.0],
            "signal_label": ["Negative GEX", "Low GEX", "Very High GEX"],
        })
        rs_df = pd.DataFrame({
            "symbol": ["AAPL", "TSLA", "NVDA"],
            "etf_ticker": ["XLK", "ARKK", "XLK"],
            "rs_rank_index": [0.85, 0.70, 0.95],
            "rs_rank_overall": [0.80, 0.65, 0.92],
            "rs_z_score": [1.2, 0.5, 2.0],
        })
        with patch("pandas.read_sql_query", side_effect=[scan_df, rs_df]):
            from queries.signal_queries import scan_gex_level
            result = scan_gex_level()
            assert len(result) == 3


class TestEnrichWithRs:
    def test_handles_no_rs_data(self, mock_both_pools):
        """If stocks DB has no matching tickers, columns still appear."""
        scan_df = pd.DataFrame({
            "symbol": ["AAPL"],
            "signal_value": [50000.0],
            "pct_rank": [0.95],
            "signal_label": ["High Vanna"],
        })
        rs_df = pd.DataFrame(columns=["symbol", "etf_ticker", "rs_rank_index", "rs_rank_overall", "rs_z_score"])
        with patch("pandas.read_sql_query", side_effect=[scan_df, rs_df]):
            from queries.signal_queries import scan_vanna_extreme
            result = scan_vanna_extreme(20.0)
            # Should still have symbol column, just no RS enrichment
            assert "symbol" in result.columns
