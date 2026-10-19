"""Tests for logic/events.py: event crossings and the bars they need."""

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.logic.events import (
    EVENT_COLUMNS,
    EVENT_REQUIRED_BARS,
    detect_events_for_ticker,
    latest_event_flags_bulk,
)


def ohlcv(n: int, ticker: str = "T", seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.bdate_range("2024-01-01", periods=n),
            "close": close,
            "volume": rng.integers(1_000, 2_000, n),
        }
    )


def series(values):
    return pd.Series(values, dtype=float)


class TestDetectEvents:
    def test_crossings_are_held_for_lookback_bars(self):
        n = 8
        rsi = series([25, 25, 35, 35, 35, 35, 35, 35])  # crosses up through 30 at bar 2
        flat = series([0.0] * n)
        events = detect_events_for_ticker(
            close=series(range(n)),
            rsi_14=rsi,
            macd_histogram=flat,
            volume_z=flat,
            ratio_to_200ma=series([1.0] * n),
            lookback=3,
        )
        assert set(events) == set(EVENT_COLUMNS)
        assert (
            events["evt_rsi_cross_up_30"].tolist() == [False, False, True, True, True] + [False] * 3
        )

    def test_macd_and_200ma_crossings(self):
        events = detect_events_for_ticker(
            close=series([1, 1, 1, 1]),
            rsi_14=series([50] * 4),
            macd_histogram=series([-1, 1, 1, -1]),
            volume_z=series([0] * 4),
            ratio_to_200ma=series([0.9, 1.1, 0.9, 0.9]),
            lookback=1,
        )
        assert events["evt_macd_bullish_cross"].tolist() == [False, True, False, False]
        assert events["evt_macd_bearish_cross"].tolist() == [False, False, False, True]
        assert events["evt_regain_200ma"].tolist() == [False, True, False, False]
        assert events["evt_lose_200ma"].tolist() == [False, False, True, False]

    def test_breakout_requires_volume(self):
        close = series([10.0] * 15 + [12.0, 13.0])
        vol_z = series([0.0] * 16 + [2.0])
        flat = series([0.0] * 17)
        events = detect_events_for_ticker(close, flat + 50, flat, vol_z, flat + 1, lookback=1)
        assert events["evt_breakout_20d_high_vol"].tolist() == [False] * 16 + [True]

    def test_nan_inputs_do_not_fire(self):
        nan = series([np.nan] * 5)
        events = detect_events_for_ticker(nan, nan, nan, nan, nan)
        assert not any(s.any() for s in events.values())


class TestLatestEventFlagsBulk:
    def test_flags_at_required_bars(self):
        flags = latest_event_flags_bulk(ohlcv(EVENT_REQUIRED_BARS), ["T"])
        assert set(flags["T"]) == set(EVENT_COLUMNS)
        assert all(isinstance(v, bool) for v in flags["T"].values())

    def test_omits_ticker_with_too_few_bars(self):
        assert latest_event_flags_bulk(ohlcv(EVENT_REQUIRED_BARS - 1), ["T"]) == {}

    def test_unknown_ticker_omitted(self):
        assert latest_event_flags_bulk(ohlcv(EVENT_REQUIRED_BARS), ["X"]) == {}

    @pytest.mark.parametrize("n", [EVENT_REQUIRED_BARS, 400])
    def test_uses_shared_volume_z(self, n):
        """The breakout event and the backtest/overlay use one 20-session volume z-score."""
        from stock_dashboard.logic.backtest import compute_ticker_indicators

        bars = ohlcv(n)
        latest = latest_event_flags_bulk(bars, ["T"])["T"]
        panel = compute_ticker_indicators(bars.assign(open=1.0, high=1.0, low=1.0), EVENT_COLUMNS)
        assert latest == {c: bool(panel[c].iloc[-1]) for c in EVENT_COLUMNS}
