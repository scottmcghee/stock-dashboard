"""Tests for logic.trend_signals — washout reversal and coiled spring detection."""

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.logic.trend_signals import (
    compute_trend_indicators,
    scan_coiled_spring,
    scan_washout_bear,
    scan_washout_bull,
)


def _make_ohlcv(
    ticker: str = "TEST",
    n_bars: int = 250,
    base_price: float = 100.0,
    daily_return: float = 0.0,
    volatility: float = 0.01,
    volume: int = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_bars)
    returns = daily_return + rng.normal(0, volatility, n_bars)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.001, 0.02, n_bars))
    low = close * (1 - rng.uniform(0.001, 0.02, n_bars))
    open_ = close * (1 + rng.normal(0, 0.005, n_bars))
    volumes = rng.randint(int(volume * 0.5), int(volume * 1.5), n_bars)

    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volumes,
        }
    )


def _make_selloff_ohlcv(ticker: str = "SOLD", seed: int = 99) -> pd.DataFrame:
    """Generate data with a recent selloff — should trigger washout bull."""
    rng = np.random.RandomState(seed)
    n_bars = 250
    dates = pd.bdate_range("2025-01-01", periods=n_bars)

    # Flat for 200 bars, then sharp selloff for 50 bars
    returns = np.concatenate(
        [
            rng.normal(0.0005, 0.008, 200),  # gentle uptrend
            rng.normal(-0.008, 0.015, 50),  # sharp selloff with high vol
        ]
    )
    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.005, 0.03, n_bars))
    low = close * (1 - rng.uniform(0.005, 0.03, n_bars))
    open_ = close * (1 + rng.normal(0, 0.005, n_bars))

    # Volume spikes during selloff
    volumes = np.concatenate(
        [
            rng.randint(800_000, 1_200_000, 200),
            rng.randint(1_500_000, 3_000_000, 50),
        ]
    )

    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volumes,
        }
    )


def _make_rally_ohlcv(ticker: str = "RALLY", seed: int = 77) -> pd.DataFrame:
    """Generate data with a recent rally — should trigger washout bear."""
    rng = np.random.RandomState(seed)
    n_bars = 250
    dates = pd.bdate_range("2025-01-01", periods=n_bars)

    # Flat for 200 bars, then sharp rally for 50 bars
    returns = np.concatenate(
        [
            rng.normal(0.0, 0.008, 200),
            rng.normal(0.008, 0.015, 50),  # sharp rally
        ]
    )
    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.005, 0.03, n_bars))
    low = close * (1 - rng.uniform(0.005, 0.03, n_bars))
    open_ = close * (1 + rng.normal(0, 0.005, n_bars))
    volumes = rng.randint(800_000, 1_500_000, n_bars)

    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volumes,
        }
    )


def _make_tight_range_ohlcv(ticker: str = "COIL", seed: int = 55) -> pd.DataFrame:
    """Generate data with very tight, low-volatility range — should trigger coiled spring."""
    rng = np.random.RandomState(seed)
    n_bars = 250
    dates = pd.bdate_range("2025-01-01", periods=n_bars)

    # Very tight range for the last 50 bars
    returns = np.concatenate(
        [
            rng.normal(0.0003, 0.01, 200),
            rng.normal(0.0, 0.002, 50),  # extremely tight
        ]
    )
    close = 100.0 * np.cumprod(1 + returns)
    # Tight high-low range
    high = close * (
        1
        + np.concatenate(
            [
                rng.uniform(0.005, 0.02, 200),
                rng.uniform(0.001, 0.005, 50),
            ]
        )
    )
    low = close * (
        1
        - np.concatenate(
            [
                rng.uniform(0.005, 0.02, 200),
                rng.uniform(0.001, 0.005, 50),
            ]
        )
    )
    open_ = close * (1 + rng.normal(0, 0.002, n_bars))
    volumes = rng.randint(800_000, 1_200_000, n_bars)

    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volumes,
        }
    )


# ---- compute_trend_indicators ----


class TestComputeTrendIndicators:
    def test_returns_one_row_per_ticker(self):
        ohlcv = pd.concat(
            [
                _make_ohlcv("AAA", seed=1),
                _make_ohlcv("BBB", seed=2),
            ]
        )
        result = compute_trend_indicators(ohlcv)
        assert len(result) == 2
        assert set(result["symbol"]) == {"AAA", "BBB"}

    def test_skips_tickers_with_insufficient_history(self):
        short = _make_ohlcv("SHORT", n_bars=50)
        long = _make_ohlcv("LONG", n_bars=250)
        ohlcv = pd.concat([short, long])
        result = compute_trend_indicators(ohlcv)
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "LONG"

    def test_empty_input(self):
        result = compute_trend_indicators(pd.DataFrame())
        assert result.empty

    def test_all_indicators_present(self):
        ohlcv = _make_ohlcv("TEST")
        result = compute_trend_indicators(ohlcv)
        expected_cols = {
            "symbol",
            "rsi",
            "mom_20",
            "ratio_50ma",
            "ratio_200ma",
            "bb_width",
            "adr_pct",
            "vol_ratio",
            "bb_width_chg",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_rsi_in_valid_range(self):
        ohlcv = _make_ohlcv("TEST")
        result = compute_trend_indicators(ohlcv)
        rsi = result.iloc[0]["rsi"]
        assert 0 <= rsi <= 100

    def test_ratio_50ma_near_one_for_flat_series(self):
        ohlcv = _make_ohlcv("FLAT", volatility=0.001, daily_return=0.0)
        result = compute_trend_indicators(ohlcv)
        ratio = result.iloc[0]["ratio_50ma"]
        assert 0.95 <= ratio <= 1.05

    def test_bb_width_positive(self):
        ohlcv = _make_ohlcv("TEST")
        result = compute_trend_indicators(ohlcv)
        assert result.iloc[0]["bb_width"] > 0

    def test_custom_min_bars(self):
        ohlcv = _make_ohlcv("TEST", n_bars=150)
        assert compute_trend_indicators(ohlcv, min_bars=200).empty
        assert len(compute_trend_indicators(ohlcv, min_bars=100)) == 1


# ---- scan_washout_bull ----


class TestScanWashoutBull:
    def test_empty_indicators(self):
        result = scan_washout_bull(pd.DataFrame())
        assert result.empty

    def test_selloff_triggers_signal(self):
        ohlcv = _make_selloff_ohlcv()
        indicators = compute_trend_indicators(ohlcv)
        # Use relaxed thresholds to ensure the synthetic data triggers
        result = scan_washout_bull(
            indicators,
            rsi_max=50.0,
            mom_max=-1.0,
            ratio_50ma_max=1.0,
            bb_width_min=3.0,
            vol_ratio_min=0.5,
        )
        assert len(result) == 1
        assert result.iloc[0]["signal_label"] == "Washout Bull"
        assert "signal_value" in result.columns

    def test_flat_data_no_signal(self):
        ohlcv = _make_ohlcv("FLAT", volatility=0.001, daily_return=0.0)
        indicators = compute_trend_indicators(ohlcv)
        result = scan_washout_bull(indicators)
        assert result.empty

    def test_signal_value_is_momentum(self):
        ohlcv = _make_selloff_ohlcv()
        indicators = compute_trend_indicators(ohlcv)
        result = scan_washout_bull(
            indicators,
            rsi_max=60,
            mom_max=0,
            ratio_50ma_max=1.1,
            bb_width_min=1.0,
            vol_ratio_min=0.1,
        )
        assert not result.empty
        assert result.iloc[0]["signal_value"] == result.iloc[0]["mom_20"]
        assert result.iloc[0]["strength"] == -result.iloc[0]["mom_20"]

    def test_threshold_filtering(self):
        ohlcv = pd.concat(
            [
                _make_selloff_ohlcv("A", seed=10),
                _make_selloff_ohlcv("B", seed=20),
                _make_ohlcv("C", seed=30),
            ]
        )
        indicators = compute_trend_indicators(ohlcv)
        # Very strict thresholds
        strict = scan_washout_bull(
            indicators,
            rsi_max=15,
            mom_max=-30,
            ratio_50ma_max=0.7,
            bb_width_min=50,
            vol_ratio_min=5.0,
        )
        assert strict.empty


# ---- scan_washout_bear ----


class TestScanWashoutBear:
    def test_empty_indicators(self):
        result = scan_washout_bear(pd.DataFrame())
        assert result.empty

    def test_rally_triggers_signal(self):
        ohlcv = _make_rally_ohlcv()
        indicators = compute_trend_indicators(ohlcv)
        result = scan_washout_bear(
            indicators,
            rsi_min=50.0,
            mom_min=1.0,
            ratio_50ma_min=1.0,
            bb_width_min=3.0,
        )
        assert len(result) == 1
        assert result.iloc[0]["signal_label"] == "Washout Bear"

    def test_flat_data_no_signal(self):
        ohlcv = _make_ohlcv("FLAT", volatility=0.001, daily_return=0.0)
        indicators = compute_trend_indicators(ohlcv)
        result = scan_washout_bear(indicators)
        assert result.empty


# ---- scan_coiled_spring ----


class TestScanCoiledSpring:
    def test_empty_indicators(self):
        result = scan_coiled_spring(pd.DataFrame())
        assert result.empty

    def test_tight_range_triggers_signal(self):
        ohlcv = _make_tight_range_ohlcv()
        indicators = compute_trend_indicators(ohlcv)
        result = scan_coiled_spring(
            indicators,
            bb_width_max=10.0,
            adr_max=3.0,
            ratio_200ma_min=0.80,
            ratio_200ma_max=1.20,
        )
        assert len(result) == 1
        assert "Coiled Spring" in result.iloc[0]["signal_label"]

    def test_strength_is_negative_bb_width(self):
        ohlcv = _make_tight_range_ohlcv()
        indicators = compute_trend_indicators(ohlcv)
        result = scan_coiled_spring(
            indicators,
            bb_width_max=20,
            adr_max=5,
            ratio_200ma_min=0.5,
            ratio_200ma_max=1.5,
        )
        assert not result.empty
        assert result.iloc[0]["strength"] == -result.iloc[0]["bb_width"]

    def test_above_below_200ma_labels(self):
        ohlcv = _make_tight_range_ohlcv()
        indicators = compute_trend_indicators(ohlcv)
        result = scan_coiled_spring(
            indicators,
            bb_width_max=20,
            adr_max=5,
            ratio_200ma_min=0.5,
            ratio_200ma_max=1.5,
        )
        assert not result.empty
        label = result.iloc[0]["signal_label"]
        ratio = result.iloc[0]["ratio_200ma"]
        assert ("above" if ratio >= 1.0 else "below") in label

    def test_volatile_data_no_signal(self):
        ohlcv = _make_ohlcv("VOL", volatility=0.03)
        indicators = compute_trend_indicators(ohlcv)
        result = scan_coiled_spring(indicators)
        assert result.empty

    def test_200ma_range_filter(self):
        ohlcv = _make_tight_range_ohlcv()
        indicators = compute_trend_indicators(ohlcv)
        # Very narrow range around 200MA — may exclude if price has drifted
        result_narrow = scan_coiled_spring(
            indicators,
            bb_width_max=20,
            adr_max=5,
            ratio_200ma_min=0.999,
            ratio_200ma_max=1.001,
        )
        result_wide = scan_coiled_spring(
            indicators,
            bb_width_max=20,
            adr_max=5,
            ratio_200ma_min=0.5,
            ratio_200ma_max=1.5,
        )
        assert len(result_wide) >= len(result_narrow)


# ---- Integration: multiple tickers ----


class TestMultiTicker:
    def test_mixed_universe(self):
        """A mix of selloff, rally, tight, and flat tickers."""
        ohlcv = pd.concat(
            [
                _make_selloff_ohlcv("SELL"),
                _make_rally_ohlcv("RALLY"),
                _make_tight_range_ohlcv("COIL"),
                _make_ohlcv("FLAT", volatility=0.001),
            ]
        )
        indicators = compute_trend_indicators(ohlcv)
        assert len(indicators) == 4

        # Relaxed thresholds for synthetic data
        bulls = scan_washout_bull(
            indicators,
            rsi_max=50,
            mom_max=-1,
            ratio_50ma_max=1.0,
            bb_width_min=3,
            vol_ratio_min=0.5,
        )
        bears = scan_washout_bear(
            indicators,
            rsi_min=50,
            mom_min=1,
            ratio_50ma_min=1.0,
            bb_width_min=3,
        )
        scan_coiled_spring(
            indicators,
            bb_width_max=10,
            adr_max=3,
            ratio_200ma_min=0.8,
            ratio_200ma_max=1.2,
        )

        # Selloff ticker should appear in bull, rally in bear
        bull_symbols = set(bulls["symbol"]) if not bulls.empty else set()
        bear_symbols = set(bears["symbol"]) if not bears.empty else set()

        assert "SELL" in bull_symbols
        assert "RALLY" in bear_symbols

    def test_no_overlap_bull_bear_strict(self):
        """With default thresholds, bull and bear shouldn't overlap."""
        ohlcv = pd.concat([_make_selloff_ohlcv("A", seed=i) for i in range(5)])
        indicators = compute_trend_indicators(ohlcv)
        bulls = scan_washout_bull(indicators)
        bears = scan_washout_bear(indicators)

        bull_set = set(bulls["symbol"]) if not bulls.empty else set()
        bear_set = set(bears["symbol"]) if not bears.empty else set()
        assert bull_set.isdisjoint(bear_set)


def test_uses_shared_rsi_and_adr_definitions():
    """Trend scans read the same RSI (Wilder, adjust=False) and ADR% as every other module."""
    from stock_dashboard.logic.technical import compute_adr_series, compute_rsi

    ohlcv = _make_ohlcv(n_bars=220)
    row = compute_trend_indicators(ohlcv).iloc[0]
    assert row["rsi"] == pytest.approx(compute_rsi(ohlcv["close"]).iloc[-1])
    adr = compute_adr_series(ohlcv["high"], ohlcv["low"], ohlcv["close"]).iloc[-1]
    assert row["adr_pct"] == pytest.approx(adr)
