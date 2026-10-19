"""Tests for logic/patterns.py — pure function tests, no DB needed."""

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.logic.patterns import (
    PATTERN_DETECTORS,
    detect_bear_flag,
    detect_bull_flag,
    detect_cup_and_handle,
    detect_high_tight_flag,
    detect_v_bottom,
    rolling_detect_patterns,
    scan_patterns,
)


def _make_ohlcv(close_values: list[float], spread: float = 1.0) -> pd.DataFrame:
    """Helper to create OHLCV DataFrame from close values."""
    close = np.array(close_values, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=len(close)),
            "open": close - 0.5 * spread,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": [1_000_000] * len(close),
        }
    )


class TestDetectHighTightFlag:
    def test_classic_htf(self):
        # 100% rally from 50 to 100, then consolidation near the top
        rally = list(np.linspace(50, 100, 60))
        flag = [98, 97, 96, 95, 96, 97, 98, 97, 96, 95, 96, 97, 98, 97, 96]
        ohlcv = _make_ohlcv(rally + flag)
        result = detect_high_tight_flag(ohlcv, min_rally_pct=80.0)
        assert result is not None
        assert result["pattern"] == "high_tight_flag"

    def test_no_htf_weak_rally(self):
        rally = list(np.linspace(90, 110, 60))  # only ~22% rally
        flag = [109, 108, 107, 108, 109] * 3
        ohlcv = _make_ohlcv(rally + flag)
        result = detect_high_tight_flag(ohlcv, min_rally_pct=100.0)
        assert result is None

    def test_no_htf_deep_pullback(self):
        rally = list(np.linspace(50, 100, 60))
        flag = [70, 65, 60, 55, 50, 55, 60, 65, 70, 65, 60, 55, 50, 55, 60]  # 50% pullback
        ohlcv = _make_ohlcv(rally + flag)
        result = detect_high_tight_flag(ohlcv, max_pullback_pct=25.0)
        assert result is None

    def test_insufficient_data(self):
        ohlcv = _make_ohlcv([100.0] * 10)
        assert detect_high_tight_flag(ohlcv) is None


class TestDetectBullFlag:
    def test_classic_bull_flag(self):
        # Strong pole up, then slight drift down
        base = [100.0] * 10
        pole = list(np.linspace(100, 115, 15))
        flag = list(np.linspace(114, 112, 10))
        ohlcv = _make_ohlcv(base + pole + flag)
        result = detect_bull_flag(ohlcv, min_pole_pct=10.0)
        assert result is not None
        assert result["pattern"] == "bull_flag"

    def test_no_flag_strong_retrace(self):
        base = [100.0] * 10
        pole = list(np.linspace(100, 115, 15))
        flag = list(np.linspace(115, 100, 10))  # full retrace
        ohlcv = _make_ohlcv(base + pole + flag)
        result = detect_bull_flag(ohlcv, max_flag_retrace_pct=50.0)
        assert result is None

    def test_insufficient_data(self):
        ohlcv = _make_ohlcv([100.0] * 5)
        assert detect_bull_flag(ohlcv) is None


class TestDetectBearFlag:
    def test_classic_bear_flag(self):
        base = [120.0] * 10
        pole = list(np.linspace(120, 105, 15))
        flag = list(np.linspace(106, 108, 10))
        ohlcv = _make_ohlcv(base + pole + flag)
        result = detect_bear_flag(ohlcv, min_pole_pct=10.0)
        assert result is not None
        assert result["pattern"] == "bear_flag"

    def test_no_flag_strong_bounce(self):
        base = [120.0] * 10
        pole = list(np.linspace(120, 105, 15))
        flag = list(np.linspace(105, 120, 10))  # full bounce
        ohlcv = _make_ohlcv(base + pole + flag)
        result = detect_bear_flag(ohlcv, max_flag_retrace_pct=50.0)
        assert result is None

    def test_insufficient_data(self):
        ohlcv = _make_ohlcv([100.0] * 5)
        assert detect_bear_flag(ohlcv) is None


class TestDetectCupAndHandle:
    def test_classic_cup_and_handle(self):
        # Left rim at 110, dip to 90, recovery to 108, handle dip to 105
        left_rim = list(np.linspace(100, 110, 20))
        cup_bottom = list(np.linspace(110, 90, 15)) + list(np.linspace(90, 108, 25))
        handle = list(np.linspace(108, 105, 8)) + list(np.linspace(105, 107, 7))
        ohlcv = _make_ohlcv(left_rim + cup_bottom + handle)
        result = detect_cup_and_handle(ohlcv, cup_window=60, handle_window=15)
        assert result is not None
        assert result["pattern"] == "cup_and_handle"

    def test_no_cup_too_deep(self):
        left_rim = list(np.linspace(100, 110, 20))
        cup_bottom = list(np.linspace(110, 60, 20)) + list(np.linspace(60, 108, 20))
        handle = [107] * 15
        ohlcv = _make_ohlcv(left_rim + cup_bottom + handle)
        result = detect_cup_and_handle(ohlcv, max_cup_depth_pct=35.0)
        assert result is None

    def test_insufficient_data(self):
        ohlcv = _make_ohlcv([100.0] * 10)
        assert detect_cup_and_handle(ohlcv) is None


class TestDetectVBottom:
    def test_classic_v_bottom(self):
        decline = list(np.linspace(120, 95, 15))
        recovery = list(np.linspace(95, 118, 15))
        ohlcv = _make_ohlcv(decline + recovery)
        result = detect_v_bottom(ohlcv, window=30, min_decline_pct=15.0, min_recovery_pct=70.0)
        assert result is not None
        assert result["pattern"] == "v_bottom"

    def test_no_v_shallow_decline(self):
        mild = list(np.linspace(100, 97, 15)) + list(np.linspace(97, 100, 15))
        ohlcv = _make_ohlcv(mild)
        result = detect_v_bottom(ohlcv, min_decline_pct=10.0)
        assert result is None

    def test_insufficient_data(self):
        ohlcv = _make_ohlcv([100.0] * 5)
        assert detect_v_bottom(ohlcv) is None


class TestScanPatterns:
    @pytest.fixture
    def bulk_ohlcv(self) -> pd.DataFrame:
        np.random.seed(42)
        frames = []
        for ticker in ["AAPL", "MSFT", "NVDA"]:
            dates = pd.bdate_range("2025-01-01", periods=200)
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

    def test_returns_dict_of_lists(self, bulk_ohlcv):
        result = scan_patterns(bulk_ohlcv, ["AAPL", "MSFT", "NVDA"])
        assert isinstance(result, dict)
        for patterns in result.values():
            assert isinstance(patterns, list)
            for p in patterns:
                assert "pattern" in p

    def test_subset_of_patterns(self, bulk_ohlcv):
        result = scan_patterns(bulk_ohlcv, ["AAPL"], patterns=["bull_flag"])
        # Should only detect bull_flag, not others
        for patterns in result.values():
            for p in patterns:
                assert p["pattern"] == "bull_flag"

    def test_empty_tickers(self, bulk_ohlcv):
        result = scan_patterns(bulk_ohlcv, [])
        assert result == {}

    def test_unknown_ticker(self, bulk_ohlcv):
        result = scan_patterns(bulk_ohlcv, ["FAKE"])
        assert result == {}

    def test_all_detectors_registered(self):
        expected = {"high_tight_flag", "bull_flag", "bear_flag", "cup_and_handle", "v_bottom"}
        assert set(PATTERN_DETECTORS.keys()) == expected


class TestRollingDetectPatterns:
    def test_output_keys_and_lengths(self):
        """Result should have 5 pattern keys with arrays matching input length."""
        ohlcv = _make_ohlcv(list(np.linspace(100, 150, 100)))
        result = rolling_detect_patterns(ohlcv)
        expected_keys = {
            "has_high_tight_flag",
            "has_bull_flag",
            "has_bear_flag",
            "has_cup_and_handle",
            "has_v_bottom",
        }
        assert set(result.keys()) == expected_keys
        for arr in result.values():
            assert len(arr) == 100
            assert arr.dtype == np.int8

    def test_short_data_returns_all_zeros(self):
        """Data shorter than any window should produce all-zero arrays."""
        ohlcv = _make_ohlcv([100.0] * 5)
        result = rolling_detect_patterns(ohlcv)
        for arr in result.values():
            assert len(arr) == 5
            assert arr.sum() == 0

    def test_empty_data(self):
        """Empty DataFrame should return all-zero arrays of length 0."""
        ohlcv = _make_ohlcv([])
        result = rolling_detect_patterns(ohlcv)
        for arr in result.values():
            assert len(arr) == 0

    def test_bull_flag_detected_at_correct_bar(self):
        """A synthetic bull flag should trigger at the bar where the flag ends."""
        # Build: 20 bars of flat base, 15-bar pole, 10-bar flag, 10 bars of noise
        base = [100.0] * 20
        pole = list(np.linspace(100, 115, 15))
        flag = list(np.linspace(114, 112, 10))
        noise = [112.0] * 10
        ohlcv = _make_ohlcv(base + pole + flag + noise)

        result = rolling_detect_patterns(ohlcv)
        arr = result["has_bull_flag"]
        # The bull flag should be detected somewhere around bars 44 (end of flag)
        # At minimum it is detected at bar 44 (0-indexed: base 20 + pole 15 + flag 10 - 1)
        assert arr[44] == 1

    def test_values_are_binary(self):
        """All values should be 0 or 1."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        ohlcv = _make_ohlcv(close.tolist())
        result = rolling_detect_patterns(ohlcv)
        for arr in result.values():
            assert set(np.unique(arr)).issubset({0, 1})
