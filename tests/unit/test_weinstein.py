"""Tests for logic/weinstein.py."""

import numpy as np
import pandas as pd

from stock_dashboard.logic.weinstein import (
    LONG_MA_PERIOD,
    TREND_PERIOD,
    calculate_weinstein_stages,
    classify_bulk_stages,
)


class TestCalculateWeinsteinStages:
    def test_short_data_returns_undetermined(self, sample_ohlcv_short):
        result = calculate_weinstein_stages(sample_ohlcv_short)
        assert "stage" in result.columns
        assert (result["stage"] == "Undetermined").all()

    def test_returns_all_original_columns(self, sample_ohlcv):
        result = calculate_weinstein_stages(sample_ohlcv)
        for col in ["date", "open", "high", "low", "close", "volume"]:
            assert col in result.columns
        assert "stage" in result.columns

    def test_stage_values_are_valid(self, sample_ohlcv):
        result = calculate_weinstein_stages(sample_ohlcv)
        valid_stages = {"Stage 1", "Stage 2", "Stage 3", "Stage 4", "Undetermined"}
        assert set(result["stage"].unique()).issubset(valid_stages)

    def test_output_length_matches_input(self, sample_ohlcv):
        result = calculate_weinstein_stages(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_sorted_by_date(self, sample_ohlcv):
        # shuffle input
        shuffled = sample_ohlcv.sample(frac=1, random_state=0)
        result = calculate_weinstein_stages(shuffled)
        assert result["date"].is_monotonic_increasing

    def test_strong_uptrend_gets_stage_2(self):
        """Synthetic strong uptrend should classify as Stage 2."""
        dates = pd.bdate_range("2024-01-01", periods=200)
        close = np.linspace(50, 150, 200)  # steady climb
        df = pd.DataFrame(
            {
                "date": dates,
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1_000_000] * 200,
            }
        )
        result = calculate_weinstein_stages(df)
        assert result["stage"].iloc[-1] == "Stage 2"

    def test_strong_downtrend_gets_stage_4(self):
        """Synthetic downtrend should classify as Stage 4."""
        dates = pd.bdate_range("2024-01-01", periods=200)
        close = np.linspace(150, 50, 200)  # steady decline
        df = pd.DataFrame(
            {
                "date": dates,
                "open": close + 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1_000_000] * 200,
            }
        )
        result = calculate_weinstein_stages(df)
        assert result["stage"].iloc[-1] == "Stage 4"

    def test_undetermined_until_long_ma_trend_is_defined(self):
        """No stage from partial MAs: the first LONG_MA + TREND - 1 bars are Undetermined."""
        n = 260
        close = np.linspace(50, 150, n)
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=n),
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1_000_000] * n,
            }
        )
        stages = calculate_weinstein_stages(df)["stage"].reset_index(drop=True)
        first_defined = LONG_MA_PERIOD + TREND_PERIOD - 1
        assert (stages.iloc[:first_defined] == "Undetermined").all()
        assert (stages.iloc[first_defined:] == "Stage 2").all()

    def test_custom_ma_periods(self, sample_ohlcv):
        result = calculate_weinstein_stages(sample_ohlcv, short_ma=30, long_ma=100)
        assert "stage" in result.columns
        assert len(result) == len(sample_ohlcv)


class TestClassifyBulkStages:
    def test_returns_dict_of_stages(self, sample_multi_ticker_ohlcv):
        result = classify_bulk_stages(sample_multi_ticker_ohlcv)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"AAPL", "MSFT", "GOOGL"}

    def test_all_values_are_valid_stages(self, sample_multi_ticker_ohlcv):
        result = classify_bulk_stages(sample_multi_ticker_ohlcv)
        valid = {"Stage 1", "Stage 2", "Stage 3", "Stage 4", "Undetermined"}
        for stage in result.values():
            assert stage in valid

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
        result = classify_bulk_stages(df)
        assert result == {}

    def test_single_ticker(self, sample_ohlcv):
        sample_ohlcv["ticker"] = "AAPL"
        result = classify_bulk_stages(sample_ohlcv)
        assert len(result) == 1
        assert "AAPL" in result
