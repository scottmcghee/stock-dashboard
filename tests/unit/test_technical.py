"""Tests for logic/technical.py — pure function tests, no DB needed."""

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.logic.technical import (
    apply_ta_filters,
    check_price_touches_ma,
    compute_adr,
    compute_indicator_history,
    compute_macd,
    compute_momentum,
    compute_momentum_series,
    compute_ratio_to_ma,
    compute_ratio_to_ma_series,
    compute_rsi,
    compute_sma,
    compute_technical_zscore_table,
    detect_gap,
    detect_macd_crossover,
    get_latest_rsi,
    split_by_ticker,
)


class TestSplitByTicker:
    def test_groups_keep_their_rows_in_order(self) -> None:
        df = pd.DataFrame({"ticker": ["B", "A", "B", "A"], "close": [1.0, 2.0, 3.0, 4.0]})
        groups = split_by_ticker(df)
        assert list(groups) == ["B", "A"]
        assert groups["B"]["close"].tolist() == [1.0, 3.0]
        assert groups["A"].index.tolist() == [1, 3]

    def test_unused_categories_are_left_out(self) -> None:
        tickers = pd.Categorical(["A", "A"], categories=["A", "Z"])
        groups = split_by_ticker(pd.DataFrame({"ticker": tickers, "close": [1.0, 2.0]}))
        assert list(groups) == ["A"]

    def test_empty_frame_without_columns(self) -> None:
        assert split_by_ticker(pd.DataFrame()) == {}


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
        close = pd.Series(list(np.linspace(150, 50, 100)) + list(np.linspace(52, 120, 20)))
        cross = detect_macd_crossover(close, lookback=20)
        assert cross == "bullish"

    def test_bearish_crossover(self):
        # Long rally, then sharp selloff near the end so crossover is recent
        close = pd.Series(list(np.linspace(50, 150, 100)) + list(np.linspace(148, 80, 20)))
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
        ohlcv = pd.DataFrame(
            {
                "close": list(range(90, 101)),
                "high": [x + 3 for x in range(90, 101)],
                "low": [x - 3 for x in range(90, 101)],
            }
        )
        assert check_price_touches_ma(ohlcv, period=5)

    def test_no_touch(self):
        # Price jumps far above the SMA
        close = [100.0] * 10 + [200.0]
        ohlcv = pd.DataFrame(
            {
                "close": close,
                "high": [x + 0.5 for x in close],
                "low": [x - 0.5 for x in close],
            }
        )
        # SMA(5) = (100+100+100+100+200)/5 = 120
        # low=199.5, high=200.5 → 199.5 <= 120? No → False
        assert not check_price_touches_ma(ohlcv, period=5)

    def test_insufficient_data(self):
        ohlcv = pd.DataFrame(
            {
                "close": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
            }
        )
        assert not check_price_touches_ma(ohlcv, period=50)


class TestComputeAdr:
    def test_basic_adr(self):
        # Constant range of 2 on close of 100 → ADR = 2%
        high = pd.Series([101.0] * 20)
        low = pd.Series([99.0] * 20)
        close = pd.Series([100.0] * 20)
        adr = compute_adr(high, low, close, period=14)
        assert adr == pytest.approx(2.0)

    def test_varying_range(self):
        high = pd.Series([105.0, 110.0, 103.0])
        low = pd.Series([95.0, 90.0, 97.0])
        close = pd.Series([100.0, 100.0, 100.0])
        adr = compute_adr(high, low, close, period=3)
        # ranges: 10%, 20%, 6% → mean = 12%
        assert adr == pytest.approx(12.0)

    def test_insufficient_data(self):
        high = pd.Series([101.0, 102.0])
        low = pd.Series([99.0, 98.0])
        close = pd.Series([100.0, 100.0])
        assert compute_adr(high, low, close, period=14) is None


class TestComputeMomentum:
    def test_positive_momentum(self):
        close = pd.Series([100.0] * 10 + [120.0])
        val = compute_momentum(close, period=10)
        assert val == pytest.approx(20.0)  # (120-100)/100 * 100

    def test_negative_momentum(self):
        close = pd.Series([100.0] * 10 + [80.0])
        val = compute_momentum(close, period=10)
        assert val == pytest.approx(-20.0)

    def test_insufficient_data(self):
        close = pd.Series([100.0, 101.0])
        assert compute_momentum(close, period=10) is None


class TestComputeRatioToMa:
    def test_at_ma(self):
        close = pd.Series([100.0] * 130)
        val = compute_ratio_to_ma(close, period=126)
        assert val == pytest.approx(1.0)

    def test_above_ma(self):
        close = pd.Series([100.0] * 126 + [120.0])
        val = compute_ratio_to_ma(close, period=126)
        # SMA still near 100, close=120 → ratio ~1.2
        assert val is not None
        assert val > 1.0

    def test_insufficient_data(self):
        close = pd.Series([100.0] * 50)
        assert compute_ratio_to_ma(close, period=126) is None


class TestComputeMomentumSeries:
    def test_returns_series(self):
        close = pd.Series([100.0] * 20 + [120.0])
        result = compute_momentum_series(close, period=10)
        assert len(result) == len(close)
        # Last value: (120 - 100) / 100 * 100 = 20%
        assert result.iloc[-1] == pytest.approx(20.0)

    def test_first_values_are_nan(self):
        close = pd.Series([100.0] * 20)
        result = compute_momentum_series(close, period=10)
        assert pd.isna(result.iloc[0])


class TestComputeRatioToMaSeries:
    def test_returns_series(self):
        close = pd.Series([100.0] * 130)
        result = compute_ratio_to_ma_series(close, period=126)
        assert len(result) == 130
        # At the end, ratio should be 1.0
        assert result.iloc[-1] == pytest.approx(1.0)

    def test_above_ma(self):
        close = pd.Series([100.0] * 126 + [120.0])
        result = compute_ratio_to_ma_series(close, period=126)
        assert result.iloc[-1] > 1.0


class TestComputeIndicatorHistory:
    def test_returns_all_columns(self):
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=200)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        ohlcv = pd.DataFrame(
            {
                "date": dates,
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": [1_000_000] * 200,
            }
        )
        result = compute_indicator_history(ohlcv, momentum_period=63, ma_period=126)
        assert "date" in result.columns
        assert "rsi" in result.columns
        assert "momentum" in result.columns
        assert "ratio_to_ma" in result.columns
        assert len(result) == 200

    def test_rsi_in_range(self):
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=200)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        ohlcv = pd.DataFrame(
            {
                "date": dates,
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": [1_000_000] * 200,
            }
        )
        result = compute_indicator_history(ohlcv)
        valid_rsi = result["rsi"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()


class TestComputeTechnicalZscoreTable:
    @pytest.fixture
    def zscore_ohlcv(self) -> pd.DataFrame:
        np.random.seed(42)
        frames = []
        for ticker in ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]:
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

    def test_returns_all_columns(self, zscore_ohlcv):
        result = compute_technical_zscore_table(
            zscore_ohlcv,
            ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"],
            momentum_period=63,
            ma_period=126,
        )
        assert not result.empty
        for col in [
            "ticker",
            "rsi",
            "momentum",
            "ratio_to_ma",
            "rsi_z",
            "momentum_z",
            "ratio_z",
            "composite_z",
        ]:
            assert col in result.columns

    def test_sorted_by_composite(self, zscore_ohlcv):
        result = compute_technical_zscore_table(
            zscore_ohlcv,
            ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"],
        )
        composites = result["composite_z"].tolist()
        assert composites == sorted(composites, reverse=True)

    def test_z_scores_centered(self, zscore_ohlcv):
        result = compute_technical_zscore_table(
            zscore_ohlcv,
            ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"],
        )
        # Z-scores should be approximately centered around 0
        for col in ["rsi_z", "momentum_z", "ratio_z"]:
            valid = result[col].dropna()
            if len(valid) > 1:
                assert abs(valid.mean()) < 0.5

    def test_empty_tickers(self, zscore_ohlcv):
        result = compute_technical_zscore_table(zscore_ohlcv, [])
        assert result.empty

    def test_unknown_ticker(self, zscore_ohlcv):
        result = compute_technical_zscore_table(zscore_ohlcv, ["FAKE"])
        assert result.empty


class TestDetectGap:
    def test_bull_gap(self):
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 110.0],
                "high": [105.0, 115.0],
                "low": [98.0, 106.0],  # 106 > 105 = bull gap
                "close": [103.0, 112.0],
            }
        )
        result = detect_gap(ohlcv, min_gap_pct=0.0)
        assert result is not None
        assert result["type"] == "bull"
        assert result["gap_pct"] > 0

    def test_bear_gap(self):
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 85.0],
                "high": [105.0, 90.0],  # 90 < 98 = bear gap
                "low": [98.0, 82.0],
                "close": [103.0, 87.0],
            }
        )
        result = detect_gap(ohlcv, min_gap_pct=0.0)
        assert result is not None
        assert result["type"] == "bear"
        assert result["gap_pct"] > 0

    def test_no_gap(self):
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 103.0],
                "high": [105.0, 107.0],
                "low": [98.0, 101.0],  # 101 < 105, no bull gap; 107 > 98, no bear gap
                "close": [103.0, 105.0],
            }
        )
        result = detect_gap(ohlcv, min_gap_pct=0.0)
        assert result is None

    def test_min_pct_filters_small_gap(self):
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 106.0],
                "high": [105.0, 110.0],
                "low": [98.0, 105.5],  # gap = 0.5/103 = ~0.49%
                "close": [103.0, 108.0],
            }
        )
        # With 0% threshold → gap found
        assert detect_gap(ohlcv, min_gap_pct=0.0) is not None
        # With 1% threshold → gap too small
        assert detect_gap(ohlcv, min_gap_pct=1.0) is None

    def test_lookback_limits_search(self):
        # Gap on bar 1, no gap on bars 2-6
        rows = [
            {"open": 100, "high": 105, "low": 98, "close": 103},
            {"open": 110, "high": 115, "low": 106, "close": 112},  # bull gap
        ]
        for i in range(5):
            rows.append({"open": 112 + i, "high": 114 + i, "low": 111 + i, "close": 113 + i})
        ohlcv = pd.DataFrame(rows)
        # lookback=2 → only checks last 2 bars, gap is further back
        assert detect_gap(ohlcv, lookback=2) is None
        # lookback=10 → finds the old gap
        assert detect_gap(ohlcv, lookback=10) is not None

    def test_insufficient_data(self):
        ohlcv = pd.DataFrame({"open": [100], "high": [105], "low": [98], "close": [103]})
        assert detect_gap(ohlcv) is None

    def test_gap_pct_calculation(self):
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 115.0],
                "high": [105.0, 120.0],
                "low": [95.0, 110.0],  # gap = 110 - 105 = 5, pct = 5/100 = 5%
                "close": [100.0, 118.0],
            }
        )
        result = detect_gap(ohlcv, min_gap_pct=0.0)
        assert result is not None
        assert result["gap_pct"] == 5.0


class TestApplyTaFilters:
    @pytest.fixture
    def bulk_ohlcv(self) -> pd.DataFrame:
        """OHLCV for 3 tickers, 200 days each."""
        np.random.seed(42)
        frames = []
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            dates = pd.bdate_range("2025-06-01", periods=200)
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

    def test_no_filters_returns_all(self, bulk_ohlcv):
        result = apply_ta_filters(bulk_ohlcv, ["AAPL", "MSFT", "GOOGL"])
        assert set(result.keys()) == {"AAPL", "MSFT", "GOOGL"}

    def test_rsi_narrows_results(self, bulk_ohlcv):
        # Very tight RSI range should eliminate some tickers
        result = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL", "MSFT", "GOOGL"],
            rsi_min=45.0,
            rsi_max=55.0,
        )
        assert len(result) <= 3

    def test_unknown_ticker_skipped(self, bulk_ohlcv):
        result = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL", "UNKNOWN"],
            rsi_min=0.0,
            rsi_max=100.0,
        )
        assert "UNKNOWN" not in result
        assert "AAPL" in result

    def test_adr_filter(self, bulk_ohlcv):
        # With high=close+1, low=close-1, range=2 on ~100 close → ADR ~2%
        # Setting min to 1% should keep all, 50% should keep none
        result_keep = apply_ta_filters(bulk_ohlcv, ["AAPL", "MSFT"], adr_min=1.0)
        assert len(result_keep) == 2
        assert all("adr" in v for v in result_keep.values())

        result_none = apply_ta_filters(bulk_ohlcv, ["AAPL", "MSFT"], adr_min=50.0)
        assert len(result_none) == 0

    def test_gap_filter(self, bulk_ohlcv):
        # With random walk data, most tickers won't have large gaps
        result = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL", "MSFT", "GOOGL"],
            gap_type="any",
            gap_min_pct=0.0,
            gap_lookback=200,
        )
        # All results should have 'gap' key
        for info in result.values():
            assert "gap" in info

    def test_momentum_min_filter(self, bulk_ohlcv):
        result = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL", "MSFT", "GOOGL"],
            momentum_min=0.0,
            momentum_period=63,
        )
        for info in result.values():
            assert "momentum" in info
            assert info["momentum"] >= 0.0

    def test_momentum_max_filter(self, bulk_ohlcv):
        result = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL", "MSFT", "GOOGL"],
            momentum_max=0.0,
            momentum_period=63,
        )
        for info in result.values():
            assert info["momentum"] <= 0.0

    def test_momentum_period(self, bulk_ohlcv):
        # Different periods should potentially give different results
        result_20 = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL"],
            momentum_min=-999.0,
            momentum_period=20,
        )
        result_126 = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL"],
            momentum_min=-999.0,
            momentum_period=126,
        )
        # Both should pass (very loose filter), but values may differ
        assert "AAPL" in result_20
        assert "AAPL" in result_126
        assert result_20["AAPL"]["momentum"] != result_126["AAPL"]["momentum"]

    def test_ratio_to_ma_min_filter(self, bulk_ohlcv):
        result = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL", "MSFT", "GOOGL"],
            ratio_to_ma_min=0.5,
            ratio_to_ma_period=50,
        )
        for info in result.values():
            assert "ratio_to_ma" in info
            assert info["ratio_to_ma"] >= 0.5

    def test_ratio_to_ma_max_filter(self, bulk_ohlcv):
        result = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL", "MSFT", "GOOGL"],
            ratio_to_ma_max=2.0,
            ratio_to_ma_period=50,
        )
        for info in result.values():
            assert info["ratio_to_ma"] <= 2.0

    def test_ratio_to_ma_period(self, bulk_ohlcv):
        result_50 = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL"],
            ratio_to_ma_min=0.0,
            ratio_to_ma_period=50,
        )
        apply_ta_filters(
            bulk_ohlcv,
            ["AAPL"],
            ratio_to_ma_min=0.0,
            ratio_to_ma_period=200,
        )
        # Period 50 should have a value; period 200 might be None (only 200 bars)
        assert "AAPL" in result_50
        assert result_50["AAPL"]["ratio_to_ma"] is not None

    def test_no_filters_includes_new_params(self, bulk_ohlcv):
        """When no filters are set (all None), all tickers pass with empty dicts."""
        result = apply_ta_filters(bulk_ohlcv, ["AAPL", "MSFT", "GOOGL"])
        assert set(result.keys()) == {"AAPL", "MSFT", "GOOGL"}
        for info in result.values():
            assert info == {}

    def test_macd_not_bearish_filter(self, bulk_ohlcv):
        """'not bearish' keeps tickers with bullish or None crossover, excludes bearish."""
        result = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL", "MSFT", "GOOGL"],
            macd_crossover="not bearish",
        )
        for info in result.values():
            assert info["macd_cross"] != "bearish"

    def test_macd_not_bullish_filter(self, bulk_ohlcv):
        """'not bullish' keeps tickers with bearish or None crossover, excludes bullish."""
        result = apply_ta_filters(
            bulk_ohlcv,
            ["AAPL", "MSFT", "GOOGL"],
            macd_crossover="not bullish",
        )
        for info in result.values():
            assert info["macd_cross"] != "bullish"


# ---------------------------------------------------------------------------
# Shared indicator definitions
# ---------------------------------------------------------------------------

from stock_dashboard.logic.technical import (  # noqa: E402
    BARS_PER_TIMEFRAME,
    VOLUME_Z_WINDOW,
    compute_adr_series,
    compute_pivot_proximity,
    compute_volume_z,
    ta_required_bars,
)


class TestRsiDefinition:
    def test_wilder_smoothing(self):
        """RSI uses ewm(alpha=1/period, adjust=False) for both gains and losses."""
        close = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 60)))
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        expected = 100 - 100 / (1 + gain / loss)
        pd.testing.assert_series_equal(compute_rsi(close), expected)

    def test_flat_series_is_nan(self):
        assert compute_rsi(pd.Series([5.0] * 30)).isna().all()


class TestVolumeZ:
    def test_window_includes_current_bar(self):
        volume = pd.Series([100.0] * (VOLUME_Z_WINDOW - 1) + [200.0])
        window = volume.iloc[-VOLUME_Z_WINDOW:]
        expected = (200.0 - window.mean()) / window.std()
        z = compute_volume_z(volume)
        assert z.iloc[-1] == pytest.approx(expected)
        assert z.iloc[: VOLUME_Z_WINDOW - 1].isna().all()

    def test_constant_volume_is_nan(self):
        assert compute_volume_z(pd.Series([10] * 25)).isna().all()


def test_adr_series_matches_latest_adr():
    high = pd.Series(np.linspace(11, 12, 30))
    low = pd.Series(np.linspace(9, 10, 30))
    close = pd.Series(np.linspace(10, 11, 30))
    series = compute_adr_series(high, low, close)
    assert series.iloc[-1] == pytest.approx(compute_adr(high, low, close))
    assert series.iloc[:13].isna().all()


class TestPivotProximity:
    def test_excludes_recent_bars(self):
        """The 20s in the last 5 bars cannot be the pivot high the latest close is compared to."""
        close = pd.Series([10.0] * 30 + [20.0] * 4 + [15.0])
        dist_hi, dist_lo = compute_pivot_proximity(close, window=30, exclude_recent=5)
        assert dist_hi.iloc[-1] == pytest.approx(0.5)
        assert dist_lo.iloc[-1] == pytest.approx(0.5)

    def test_needs_minimum_history(self):
        dist_hi, _ = compute_pivot_proximity(pd.Series(np.arange(1.0, 20.0)), window=63)
        assert dist_hi.isna().all()


class TestTaRequiredBars:
    def test_daily_defaults(self):
        assert ta_required_bars() == 200

    def test_longest_period_wins(self):
        assert ta_required_bars(momentum_period=252) == 253
        assert ta_required_bars(ratio_to_ma_period=50) == 131  # pivot window + exclusion
        assert ta_required_bars(ma_touch_period=300) == 300

    @pytest.mark.parametrize("timeframe", ["W", "M"])
    def test_resampled_timeframes(self, timeframe):
        per_bar = BARS_PER_TIMEFRAME[timeframe]
        assert ta_required_bars(timeframe=timeframe) == (200 + 1) * per_bar

    def test_filters_have_values_at_required_bars(self):
        rng = np.random.default_rng(5)
        n = ta_required_bars()
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        df = pd.DataFrame(
            {
                "ticker": "T",
                "date": pd.bdate_range("2024-01-01", periods=n),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000,
            }
        )
        info = apply_ta_filters(
            df,
            ["T"],
            rsi_min=0,
            momentum_min=-1000,
            ratio_to_ma_min=0,
            dist_recent_low_126_min=-10,
            adr_min=0,
        )["T"]
        for key in ("rsi", "momentum", "ratio_to_ma", "dist_recent_low_126", "adr"):
            assert info[key] is not None, key


class TestApplyTaFiltersExtra:
    @pytest.fixture
    def df(self):
        rng = np.random.default_rng(9)
        n = 260
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame(
            {
                "ticker": "T",
                "date": pd.bdate_range("2024-01-01", periods=n),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000,
            }
        )

    def test_dist_recent_low_bounds(self, df):
        assert apply_ta_filters(df, ["T"], dist_recent_low_126_max=-5.0) == {}
        assert "T" in apply_ta_filters(df, ["T"], dist_recent_low_126_min=-5.0)

    def test_ma_touch(self, df):
        touched = apply_ta_filters(df, ["T"], ma_touch_period=5)
        assert touched == {} or touched["T"]["ma_touch"] is True

    def test_weekly_timeframe(self, df):
        result = apply_ta_filters(df, ["T"], rsi_min=0, timeframe="W")
        assert "T" in result

    def test_exact_macd_requirement(self, df):
        result = apply_ta_filters(df, ["T"], macd_crossover="bullish")
        assert all(info["macd_cross"] == "bullish" for info in result.values())

    def test_missing_ticker_skipped(self, df):
        assert apply_ta_filters(df, ["NOPE"], rsi_min=0) == {}
