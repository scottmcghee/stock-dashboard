"""Tests for logic/lookback.py."""

from stock_dashboard.logic.backtest import RS_IMPULSE_WINDOWS
from stock_dashboard.logic.events import EVENT_REQUIRED_BARS
from stock_dashboard.logic.lookback import chart_warmup_bars, screener_lookback_bars
from stock_dashboard.logic.patterns import PATTERN_REQUIRED_BARS
from stock_dashboard.logic.technical import ta_required_bars
from stock_dashboard.logic.weinstein import LONG_MA_PERIOD, TREND_PERIOD


def test_screener_lookback_covers_every_consumer():
    bars = screener_lookback_bars(126, 200, "W", 50)
    assert bars >= ta_required_bars(126, 200, "W", 50)
    assert bars >= EVENT_REQUIRED_BARS
    assert bars >= PATTERN_REQUIRED_BARS


def test_chart_warmup_covers_the_overlay_indicators():
    bars = chart_warmup_bars()
    assert bars >= screener_lookback_bars(126, 200)
    assert bars >= LONG_MA_PERIOD + TREND_PERIOD
    assert bars > max(RS_IMPULSE_WINDOWS)
