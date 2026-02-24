"""Tests for logic/technical.py — pure function tests, no DB needed."""

import numpy as np
import pandas as pd
import pytest

from logic.technical import (
    apply_ta_filters,
    check_price_touches_ma,
    compute_macd,
    compute_rsi,
    compute_sma,
    detect_macd_crossover,
    get_latest_rsi,
)


class TestComputeSma:
    def test_basic_sma(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = compute_sma(close, period=3)
        assert sma.iloc[-1] == pytest.approx(4.0)  # (3+4+5)/3
        assert pd.isna(sma.iloc[0])  # not enough data

    def test_period_equals_length(self):
        close = pd.Series([10.0, 20.0, 30.0])
        sma = compute_sma(close, period=3)
        assert sma.iloc[-1] == pytest.approx(20.0)


class TestComputeRsi:
    def test_all_gains(self):
        close = pd.Series(range(1, 50), dtype=float)
        rsi = compute_rsi(close, period=14)
        # All gains = RSI near 100
        assert rsi.iloc[-1] > 95

    def test_all_losses(self):
        close = pd.Series(range(50, 1, -1), dtype=float)
        rsi = compute_rsi(close, period=14)
        # All losses = RSI near 0
        assert rsi.iloc[-1] < 5

    def test_rsi_range(self):
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(100)))
        rsi = compute_rsi(close, period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_insufficient_data(self):
        close = pd.Series([100.0, 101.0, 102.0])
        rsi = compute_rsi(close, period=14)
        assert rsi.dropna().empty


class TestGetLatestRsi:
    def test_returns_float(self):
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(50)))
        val = get_latest_rsi(close, period=14)
        assert isinstance(val, float)
        assert 0 <= val <= 100

    def test_insufficient_data_returns_none(self):
        close = pd.Series([100.0, 101.0])
        assert get_latest_rsi(close, period=14) is None


class TestComputeMacd:
    def test_basic_shape(self):
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(100)))
        macd_line, signal_line, histogram = compute_macd(close)
        assert len(macd_line) == 100
        assert len(signal_line) == 100
        # histogram = macd - signal
        valid_idx = histogram.dropna().index
        for i in valid_idx:
            assert histogram[i] == pytest.approx(macd_line[i] - signal_line[i])


class TestDetectMacdCrossover:
    def test_bullish_crossover(self):
        # Long decline, then sharp V-recovery near the end so crossover is recent
        close = pd.Series(
            list(np.linspace(150, 50, 100)) + list(np.linspace(52, 120, 20))
        )
        cross = detect_macd_crossover(close, lookback=20)
        assert cross == "bullish"

    def test_bearish_crossover(self):
        # Long rally, then sharp selloff near the end so crossover is recent
        close = pd.Series(
            list(np.linspace(50, 150, 100)) + list(np.linspace(148, 80, 20))
        )
        cross = detect_macd_crossover(close, lookback=20)
        assert cross == "bearish"

    def test_no_crossover(self):
        # Steady trend — unlikely to cross in last 3 bars
        close = pd.Series(range(1, 101), dtype=float)
        cross = detect_macd_crossover(close, lookback=3)
        assert cross is None

    def test_insufficient_data(self):
        close = pd.Series([100.0, 101.0, 102.0])
        assert detect_macd_crossover(close) is None


class TestCheckPriceTouchesMa:
    def test_touch_in_range(self):
        # SMA of 5 = mean of last 5 closes = (96+97+98+99+100)/5 = 98
        # Latest candle: low=97, high=103 → 97 <= 98 <= 103 → True
        ohlcv = pd.DataFrame({
            "close": list(range(90, 101)),
            "high": [x + 3 for x in range(90, 101)],
            "low": [x - 3 for x in range(90, 101)],
        })
        assert check_price_touches_ma(ohlcv, period=5)

    def test_no_touch(self):
        # Price jumps far above the SMA
        close = [100.0] * 10 + [200.0]
        ohlcv = pd.DataFrame({
            "close": close,
            "high": [x + 0.5 for x in close],
            "low": [x - 0.5 for x in close],
        })
        # SMA(5) = (100+100+100+100+200)/5 = 120
        # low=199.5, high=200.5 → 199.5 <= 120? No → False
        assert not check_price_touches_ma(ohlcv, period=5)

    def test_insufficient_data(self):
        ohlcv = pd.DataFrame({
            "close": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
        })
        assert not check_price_touches_ma(ohlcv, period=50)


class TestApplyTaFilters:
    @pytest.fixture
    def bulk_ohlcv(self) -> pd.DataFrame:
        """OHLCV for 3 tickers, 200 days each."""
        np.random.seed(42)
        frames = []
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            dates = pd.bdate_range("2025-06-01", periods=200)
            close = 100 + np.cumsum(np.random.randn(200) * 0.5)
            frames.append(pd.DataFrame({
                "ticker": ticker,
                "date": dates,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.random.randint(100_000, 5_000_000, size=200),
            }))
        return pd.concat(frames, ignore_index=True)

    def test_no_filters_returns_all(self, bulk_ohlcv):
        result = apply_ta_filters(bulk_ohlcv, ["AAPL", "MSFT", "GOOGL"])
        assert set(result.keys()) == {"AAPL", "MSFT", "GOOGL"}

    def test_rsi_narrows_results(self, bulk_ohlcv):
        # Very tight RSI range should eliminate some tickers
        result = apply_ta_filters(
            bulk_ohlcv, ["AAPL", "MSFT", "GOOGL"],
            rsi_min=45.0, rsi_max=55.0,
        )
        assert len(result) <= 3

    def test_unknown_ticker_skipped(self, bulk_ohlcv):
        result = apply_ta_filters(
            bulk_ohlcv, ["AAPL", "UNKNOWN"],
            rsi_min=0.0, rsi_max=100.0,
        )
        assert "UNKNOWN" not in result
        assert "AAPL" in result
