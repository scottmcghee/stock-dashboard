"""Tests for logic/preset_signal.py — signal mask and direction."""

import numpy as np
import pandas as pd
import pytest

from logic.preset_signal import compute_preset_signal_mask, determine_signal_direction


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """200 bars of OHLCV with trending data."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.3,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.random.randint(500_000, 2_000_000, size=n),
    })


class TestComputePresetSignalMask:
    def test_empty_filters_all_true(self, sample_ohlcv):
        mask = compute_preset_signal_mask(sample_ohlcv, None, [])
        assert mask.all()
        assert len(mask) == len(sample_ohlcv)

    def test_rsi_min_filter(self, sample_ohlcv):
        filters = [{"name": "rsi_min", "value": 50}]
        mask = compute_preset_signal_mask(sample_ohlcv, None, filters)
        assert len(mask) == len(sample_ohlcv)
        # Should have some True and some False (not all True)
        assert not mask.all()
        assert mask.any()

    def test_rsi_max_filter(self, sample_ohlcv):
        filters = [{"name": "rsi_max", "value": 50}]
        mask = compute_preset_signal_mask(sample_ohlcv, None, filters)
        assert not mask.all()
        assert mask.any()

    def test_momentum_filter(self, sample_ohlcv):
        filters = [{"name": "momentum_63_min", "value": 0.0}]
        mask = compute_preset_signal_mask(sample_ohlcv, None, filters)
        assert len(mask) == len(sample_ohlcv)
        # First 63 bars should be False (NaN momentum)
        assert not mask.iloc[0]

    def test_ratio_to_ma_filter(self, sample_ohlcv):
        filters = [{"name": "ratio_to_200ma_min", "value": 0.5}]
        mask = compute_preset_signal_mask(sample_ohlcv, None, filters)
        assert len(mask) == len(sample_ohlcv)

    def test_weinstein_stage_filter(self, sample_ohlcv):
        filters = [{"name": "weinstein_stage", "value": "Stage 2"}]
        mask = compute_preset_signal_mask(sample_ohlcv, None, filters)
        assert len(mask) == len(sample_ohlcv)

    def test_combined_filters_and_behavior(self, sample_ohlcv):
        """Multiple filters should be AND'd — result is subset of each individual filter."""
        f1 = [{"name": "rsi_min", "value": 40}]
        f2 = [{"name": "rsi_max", "value": 60}]
        combined = [
            {"name": "rsi_min", "value": 40},
            {"name": "rsi_max", "value": 60},
        ]
        mask1 = compute_preset_signal_mask(sample_ohlcv, None, f1)
        mask2 = compute_preset_signal_mask(sample_ohlcv, None, f2)
        mask_combined = compute_preset_signal_mask(sample_ohlcv, None, combined)
        # Combined should be subset of each individual
        assert (mask_combined & ~mask1).sum() == 0
        assert (mask_combined & ~mask2).sum() == 0

    def test_unknown_filter_ignored(self, sample_ohlcv):
        """Unknown filters not in FILTER_INDICATOR are skipped, not causing failure."""
        filters = [{"name": "totally_unknown_filter", "value": 42}]
        mask = compute_preset_signal_mask(sample_ohlcv, None, filters)
        # All unknown → all True (no valid filters to apply)
        assert mask.all()

    def test_metrics_merge_rs_rank(self, sample_ohlcv):
        dates = sample_ohlcv["date"]
        metrics = pd.DataFrame({
            "date": dates,
            "rs_rank_overall": np.linspace(0, 1, len(dates)),
            "etf_ticker": "SPY",
        })
        filters = [{"name": "rs_rank_overall_min", "value": 0.8}]
        mask = compute_preset_signal_mask(sample_ohlcv, metrics, filters)
        # Only top ~20% of dates should pass
        assert mask.sum() < len(dates) * 0.5
        assert mask.sum() > 0

    def test_volume_z_filter(self, sample_ohlcv):
        filters = [{"name": "volume_z_min", "value": 0.0}]
        mask = compute_preset_signal_mask(sample_ohlcv, None, filters)
        assert len(mask) == len(sample_ohlcv)

    def test_adr_filter(self, sample_ohlcv):
        filters = [{"name": "adr_14_min", "value": 1.0}]
        mask = compute_preset_signal_mask(sample_ohlcv, None, filters)
        assert len(mask) == len(sample_ohlcv)


class TestDetermineSignalDirection:
    def test_bullish_from_t20(self):
        stats = {"T+20": {"mean_return": 2.5, "win_rate": 0.6}}
        assert determine_signal_direction(stats) == "bullish"

    def test_bearish_from_t20(self):
        stats = {"T+20": {"mean_return": -1.5, "win_rate": 0.4}}
        assert determine_signal_direction(stats) == "bearish"

    def test_fallback_to_t10(self):
        stats = {"T+10": {"mean_return": 1.0}}
        assert determine_signal_direction(stats) == "bullish"

    def test_fallback_to_t5(self):
        stats = {"T+5": {"mean_return": -0.5}}
        assert determine_signal_direction(stats) == "bearish"

    def test_fallback_to_t63(self):
        stats = {"T+63": {"mean_return": 3.0}}
        assert determine_signal_direction(stats) == "bullish"

    def test_empty_stats_defaults_bullish(self):
        assert determine_signal_direction({}) == "bullish"

    def test_zero_return_is_bearish(self):
        stats = {"T+20": {"mean_return": 0}}
        assert determine_signal_direction(stats) == "bearish"
