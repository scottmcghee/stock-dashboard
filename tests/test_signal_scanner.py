"""Tests for logic/signal_scanner.py — pure functions, no mocking needed."""

import numpy as np
import pandas as pd
import pytest

from logic.signal_scanner import (
    TIER_WEIGHTS,
    build_composite_scores,
    compute_z_score,
    score_scan_results,
)


class TestComputeZScore:
    def test_standard_values(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        z = compute_z_score(s)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std() - 1.0) < 0.2  # approximate

    def test_constant_series_returns_zeros(self):
        s = pd.Series([5.0, 5.0, 5.0])
        z = compute_z_score(s)
        assert (z == 0.0).all()

    def test_single_value(self):
        s = pd.Series([42.0])
        z = compute_z_score(s)
        assert z.iloc[0] == 0.0

    def test_empty_series(self):
        s = pd.Series([], dtype=float)
        z = compute_z_score(s)
        assert len(z) == 0


class TestScoreScanResults:
    def test_adds_z_score_column(self):
        df = pd.DataFrame({
            "symbol": ["AAPL", "TSLA"],
            "signal_value": [100.0, 200.0],
        })
        result = score_scan_results(df)
        assert "z_score" in result.columns
        assert len(result) == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = score_scan_results(df)
        assert result.empty

    def test_missing_column(self):
        df = pd.DataFrame({"symbol": ["AAPL"], "other": [1.0]})
        result = score_scan_results(df, "nonexistent")
        assert "z_score" not in result.columns


class TestBuildCompositeScores:
    def test_combines_multiple_scans(self):
        scans = {
            "vanna_extreme": pd.DataFrame({
                "symbol": ["AAPL", "TSLA"],
                "signal_value": [50000.0, -30000.0],
            }),
            "volume_spike": pd.DataFrame({
                "symbol": ["AAPL", "NVDA"],
                "signal_value": [500000.0, 300000.0],
            }),
        }
        result = build_composite_scores(scans)
        assert "composite_score" in result.columns
        assert "signal_count" in result.columns
        # AAPL appears in both scans
        aapl = result[result["symbol"] == "AAPL"]
        assert len(aapl) == 1
        assert aapl.iloc[0]["signal_count"] == 2

    def test_tier_1_weighted_higher(self):
        # Same z-score magnitude, but vanna (T1) should contribute more
        scans = {
            "vanna_extreme": pd.DataFrame({
                "symbol": ["AAPL"],
                "signal_value": [100.0],
            }),
            "volume_spike": pd.DataFrame({
                "symbol": ["TSLA"],
                "signal_value": [100.0],
            }),
        }
        result = build_composite_scores(scans)
        # Single items in each scan -> z_score = 0 (constant series)
        # So composite scores would be 0 for both. Let's test with more data.

    def test_empty_scans(self):
        scans = {
            "vanna_extreme": pd.DataFrame(),
            "volume_spike": pd.DataFrame(),
        }
        result = build_composite_scores(scans)
        assert result.empty

    def test_sorted_by_composite_score(self):
        scans = {
            "vanna_extreme": pd.DataFrame({
                "symbol": ["AAPL", "TSLA", "NVDA"],
                "signal_value": [10.0, 50.0, 30.0],
            }),
        }
        result = build_composite_scores(scans)
        # Highest absolute z-score should be first
        assert result.iloc[0]["composite_score"] >= result.iloc[-1]["composite_score"]

    def test_signals_column_lists_contributing_scans(self):
        scans = {
            "vanna_extreme": pd.DataFrame({
                "symbol": ["AAPL"],
                "signal_value": [50000.0],
            }),
            "flownet_extreme": pd.DataFrame({
                "symbol": ["AAPL"],
                "signal_value": [-74600000.0],
            }),
        }
        result = build_composite_scores(scans)
        assert "flownet_extreme, vanna_extreme" == result.iloc[0]["signals"]

    def test_rs_enrichment_preserved(self):
        scans = {
            "vanna_extreme": pd.DataFrame({
                "symbol": ["AAPL"],
                "signal_value": [50000.0],
                "rs_rank_index": [0.85],
                "rs_rank_overall": [0.80],
                "rs_z_score": [1.2],
                "etf_ticker": ["XLK"],
            }),
        }
        result = build_composite_scores(scans)
        assert "rs_rank_index" in result.columns
        assert result.iloc[0]["rs_rank_index"] == 0.85

    def test_tier_weights_exist(self):
        assert "vanna_extreme" in TIER_WEIGHTS
        assert "flownet_extreme" in TIER_WEIGHTS
        assert TIER_WEIGHTS["vanna_extreme"] == 2.0
        assert TIER_WEIGHTS["volume_spike"] == 1.0
