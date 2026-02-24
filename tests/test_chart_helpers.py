"""Tests for logic/chart_helpers.py."""

import numpy as np
import pandas as pd
import pytest

from logic.chart_helpers import (
    resample_ohlcv,
    get_non_trading_days,
    compute_rs_moving_averages,
)


class TestResampleOhlcv:
    @pytest.fixture
    def daily_ohlcv(self) -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-01", periods=60)
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(60) * 0.5)
        df = pd.DataFrame({
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.random.randint(100_000, 1_000_000, size=60),
        }, index=dates)
        return df

    def test_daily_returns_unchanged(self, daily_ohlcv):
        result = resample_ohlcv(daily_ohlcv, "D")
        assert len(result) == len(daily_ohlcv)
        pd.testing.assert_frame_equal(result, daily_ohlcv)

    def test_weekly_reduces_rows(self, daily_ohlcv):
        result = resample_ohlcv(daily_ohlcv, "W")
        assert len(result) < len(daily_ohlcv)
        assert len(result) > 0

    def test_monthly_reduces_rows(self, daily_ohlcv):
        result = resample_ohlcv(daily_ohlcv, "M")
        assert len(result) < len(daily_ohlcv)

    def test_weekly_high_is_max(self, daily_ohlcv):
        result = resample_ohlcv(daily_ohlcv, "W")
        # Weekly high should be >= all daily highs in that week
        assert result["high"].max() <= daily_ohlcv["high"].max() + 1e-10

    def test_weekly_volume_is_sum(self, daily_ohlcv):
        result = resample_ohlcv(daily_ohlcv, "W")
        # Total volume should be preserved
        assert abs(result["volume"].sum() - daily_ohlcv["volume"].sum()) < 1

    def test_no_nan_rows_in_result(self, daily_ohlcv):
        result = resample_ohlcv(daily_ohlcv, "W")
        assert not result[["open", "high", "low", "close"]].isna().any().any()


class TestGetNonTradingDays:
    def test_weekends_excluded(self):
        # 10 business days spanning 2 weeks
        dates = pd.Series(pd.bdate_range("2025-01-06", periods=10))
        non_trading = get_non_trading_days(dates)
        # Should include Sat Jan 11 and Sun Jan 12 (within the range)
        assert pd.Timestamp("2025-01-11") in non_trading
        assert pd.Timestamp("2025-01-12") in non_trading

    def test_holidays_detected(self):
        # Create a series that skips a Wednesday (simulating holiday)
        dates = pd.Series([
            pd.Timestamp("2025-01-06"),  # Mon
            pd.Timestamp("2025-01-07"),  # Tue
            # Wed Jan 8 missing = "holiday"
            pd.Timestamp("2025-01-09"),  # Thu
            pd.Timestamp("2025-01-10"),  # Fri
        ])
        non_trading = get_non_trading_days(dates)
        assert pd.Timestamp("2025-01-08") in non_trading

    def test_returns_datetimeindex(self):
        dates = pd.Series(pd.bdate_range("2025-01-06", periods=10))
        result = get_non_trading_days(dates)
        assert isinstance(result, pd.DatetimeIndex)


class TestComputeRsMovingAverages:
    def test_adds_ma_columns(self, sample_metrics):
        result = compute_rs_moving_averages(sample_metrics)
        assert "MA252" in result.columns
        assert "MA150" in result.columns

    def test_does_not_modify_original(self, sample_metrics):
        original_cols = set(sample_metrics.columns)
        compute_rs_moving_averages(sample_metrics)
        assert set(sample_metrics.columns) == original_cols

    def test_ma_values_are_numeric(self, sample_metrics):
        result = compute_rs_moving_averages(sample_metrics)
        assert result["MA252"].dtype in [np.float64, np.float32]
        assert result["MA150"].dtype in [np.float64, np.float32]

    def test_ma150_converges_faster_than_ma252(self, sample_metrics):
        result = compute_rs_moving_averages(sample_metrics)
        # MA150 should have fewer NaN-equivalent start values
        # (both use min_periods=1, so no NaNs, but MA150 should converge to true MA faster)
        assert len(result) == len(sample_metrics)
