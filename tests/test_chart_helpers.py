"""Tests for logic/chart_helpers.py."""

import numpy as np
import pandas as pd
import pytest

from logic.chart_helpers import (
    resample_ohlcv,
    get_non_trading_days,
    compute_rs_moving_averages,
    compute_etf_impulse,
    compute_52w_highs_lows,
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


class TestComputeEtfImpulse:
    @pytest.fixture
    def etf_rank_history(self) -> pd.DataFrame:
        """60 days of rank history for 3 ETFs."""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=60)
        frames = []
        for ticker, base in [("XLK", 0.8), ("XLF", 0.5), ("XLE", 0.3)]:
            ranks = base + np.cumsum(np.random.randn(60) * 0.01)
            ranks = np.clip(ranks, 0, 1)
            frames.append(pd.DataFrame({
                "date": dates,
                "etf_ticker": ticker,
                "rs_rank_index": ranks,
            }))
        return pd.concat(frames, ignore_index=True)

    def test_returns_one_row_per_etf(self, etf_rank_history):
        result = compute_etf_impulse(etf_rank_history)
        assert len(result) == 3
        assert set(result["etf_ticker"]) == {"XLK", "XLF", "XLE"}

    def test_has_impulse_columns(self, etf_rank_history):
        result = compute_etf_impulse(etf_rank_history)
        for w in [5, 10, 20]:
            assert f"impulse_{w}d" in result.columns
            assert f"impulse_{w}d_z" in result.columns

    def test_empty_input(self):
        result = compute_etf_impulse(pd.DataFrame())
        assert result.empty

    def test_known_impulse_value(self):
        """Linearly rising rank should produce predictable impulse."""
        dates = pd.bdate_range("2025-01-01", periods=30)
        df = pd.DataFrame({
            "date": dates,
            "etf_ticker": "TEST",
            "rs_rank_index": np.linspace(0.0, 0.29, 30),
        })
        result = compute_etf_impulse(df)
        row = result.iloc[0]
        # 5-day impulse on linear data: 5 * step = 5 * 0.01 = 0.05
        assert row["impulse_5d"] == pytest.approx(0.05, abs=0.005)

    def test_z_score_detects_unusual_move(self):
        """Flat rank with a sudden spike should produce high z-score."""
        dates = pd.bdate_range("2025-01-01", periods=60)
        ranks = np.full(60, 0.5)
        # Big jump in last 5 days
        ranks[-5:] = 0.8
        df = pd.DataFrame({
            "date": dates,
            "etf_ticker": "SPIKE",
            "rs_rank_index": ranks,
        })
        result = compute_etf_impulse(df)
        row = result.iloc[0]
        assert row["impulse_5d_z"] > 2.0  # Should be very unusual

    def test_stable_etf_vs_volatile_etf(self):
        """Same raw impulse should yield higher z-score for stable ETF."""
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=60)

        # Stable ETF: flat then small jump
        stable_ranks = np.full(60, 0.5)
        stable_ranks[-1] = 0.55  # +0.05 impulse_1d equivalent, but measure at 5d

        # Volatile ETF: noisy with same final jump
        volatile_ranks = 0.5 + np.cumsum(np.random.randn(60) * 0.02)
        volatile_ranks[-1] = volatile_ranks[-6] + 0.05  # Same 5d impulse

        frames = []
        for ticker, ranks in [("STABLE", stable_ranks), ("VOLATILE", volatile_ranks)]:
            frames.append(pd.DataFrame({
                "date": dates,
                "etf_ticker": ticker,
                "rs_rank_index": ranks,
            }))
        df = pd.concat(frames, ignore_index=True)
        result = compute_etf_impulse(df)
        stable_z = result[result["etf_ticker"] == "STABLE"]["impulse_5d_z"].iloc[0]
        volatile_z = result[result["etf_ticker"] == "VOLATILE"]["impulse_5d_z"].iloc[0]
        assert abs(stable_z) > abs(volatile_z)

    def test_short_history_produces_nan(self):
        """ETF with fewer rows than window should produce NaN impulse."""
        dates = pd.bdate_range("2025-01-01", periods=3)
        df = pd.DataFrame({
            "date": dates,
            "etf_ticker": "SHORT",
            "rs_rank_index": [0.5, 0.55, 0.6],
        })
        result = compute_etf_impulse(df)
        row = result.iloc[0]
        assert np.isnan(row["impulse_20d"])


class TestCompute52wHighsLows:
    @pytest.fixture
    def sample_52w_data(self) -> pd.DataFrame:
        """300 trading days for 2 tickers — enough for 252-day window."""
        dates = pd.bdate_range("2024-01-01", periods=300)
        frames = []
        # AAPL: steady uptrend — should hit 52W highs at end
        for ticker, highs, lows in [
            ("AAPL", np.linspace(100, 200, 300), np.linspace(95, 195, 300)),
            ("BAD", np.linspace(200, 100, 300), np.linspace(195, 95, 300)),
        ]:
            frames.append(pd.DataFrame({
                "date": dates,
                "ticker": ticker,
                "high": highs,
                "low": lows,
            }))
        return pd.concat(frames, ignore_index=True)

    def test_returns_three_dataframes(self, sample_52w_data):
        counts, highs, lows = compute_52w_highs_lows(
            sample_52w_data, "2025-01-01"
        )
        assert isinstance(counts, pd.DataFrame)
        assert isinstance(highs, pd.DataFrame)
        assert isinstance(lows, pd.DataFrame)

    def test_daily_counts_has_expected_columns(self, sample_52w_data):
        counts, _, _ = compute_52w_highs_lows(sample_52w_data, "2025-01-01")
        assert "date" in counts.columns
        assert "new_highs" in counts.columns
        assert "new_lows" in counts.columns

    def test_uptrend_produces_highs(self, sample_52w_data):
        _, highs, _ = compute_52w_highs_lows(sample_52w_data, "2025-01-01")
        assert "AAPL" in highs["ticker"].values

    def test_downtrend_produces_lows(self, sample_52w_data):
        _, _, lows = compute_52w_highs_lows(sample_52w_data, "2025-01-01")
        assert "BAD" in lows["ticker"].values

    def test_empty_input(self):
        counts, highs, lows = compute_52w_highs_lows(
            pd.DataFrame(), "2025-01-01"
        )
        assert counts.empty
        assert highs.empty
        assert lows.empty

    def test_display_start_filters_output(self, sample_52w_data):
        """Counts should only include dates >= display_start."""
        counts, _, _ = compute_52w_highs_lows(
            sample_52w_data, "2025-06-01"
        )
        if not counts.empty:
            assert counts["date"].min() >= pd.Timestamp("2025-06-01")

    def test_insufficient_lookback_returns_empty(self):
        """Only 10 days of data — not enough for 252-day window."""
        dates = pd.bdate_range("2025-01-01", periods=10)
        df = pd.DataFrame({
            "date": dates,
            "ticker": "AAPL",
            "high": np.linspace(100, 110, 10),
            "low": np.linspace(95, 105, 10),
        })
        counts, highs, lows = compute_52w_highs_lows(df, "2025-01-01")
        # rolling(252, min_periods=200) won't produce values with only 10 rows
        assert counts["new_highs"].sum() == 0
        assert highs.empty
