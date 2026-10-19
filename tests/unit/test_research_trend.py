"""trend_precursor_analysis: trend detection, shared indicators and the baseline sample."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research import trend_precursor_analysis as tpa
from stock_dashboard.logic.technical import compute_rsi

RULE = tpa.TrendRule(min_move_pct=15, min_duration=5, max_duration=30, max_adverse_pct=5)


def _bars(closes: list[float] | np.ndarray, ticker: str = "ETF") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.bdate_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": 1000.0,
        }
    )


def test_a_steady_rise_is_one_uptrend_ending_at_its_best_bar() -> None:
    closes = [100.0] * 10 + list(np.linspace(100, 130, 21)) + [130.0] * 10
    trends = tpa.identify_trends(_bars(closes), RULE, warmup=5)
    up = trends[trends["direction"] == "up"]
    assert len(up) >= 1
    first = up.iloc[0]
    assert first["move_pct"] == pytest.approx(30.0)
    assert first["trend_end"] == _bars(closes)["date"].iloc[30]
    assert trends[trends["direction"] == "down"].empty


def test_an_adverse_move_beyond_the_limit_ends_the_search() -> None:
    closes = [100.0] * 6 + [110.0, 103.0, 120.0, 125.0, 125.0, 125.0, 125.0, 125.0]
    trends = tpa.identify_trends(_bars(closes), RULE, warmup=5)
    starts = trends.loc[trends["direction"] == "up", "trend_start"]
    assert _bars(closes)["date"].iloc[5] not in set(starts)  # the 6.4% pullback breaks it


def test_downtrends_are_reported_with_negative_moves() -> None:
    closes = [100.0] * 10 + list(np.linspace(100, 80, 11)) + [80.0] * 10
    down = tpa.identify_trends(_bars(closes), RULE, warmup=5).query("direction == 'down'")
    assert down["move_pct"].iloc[0] == pytest.approx(-20.0)


def test_indicators_use_the_package_rsi() -> None:
    closes = 100 + np.cumsum(np.random.default_rng(0).normal(size=80))
    bars = _bars(closes)
    metrics = pd.DataFrame(columns=["ticker", "date", "rs_rank", "rs_z_score"])
    out = tpa.compute_indicators(bars, metrics.astype({"date": "datetime64[ns]"}))
    pd.testing.assert_series_equal(
        out["rsi"], compute_rsi(bars["close"]), check_names=False, check_index=False
    )


def test_baseline_excludes_bars_near_trend_starts() -> None:
    indicators = _bars(np.linspace(100, 120, 60))
    for col in tpa.INDICATORS:
        indicators[col] = 1.0
    start = indicators["date"].iloc[30]
    trends = pd.DataFrame({"ticker": ["ETF"], "trend_start": [start]})
    baseline = tpa.sample_baseline(indicators, trends, n_samples=1000, exclusion=3, seed=0)
    assert len(baseline) == 60 - 7
    positions = indicators.index[indicators["date"].isin(baseline["date"])]
    assert not set(range(27, 34)) & set(positions)


def test_reports_handle_no_trends() -> None:
    empty = tpa.profile_precursors(
        tpa.identify_trends(_bars([100.0] * 40), RULE, warmup=5), _bars([100.0] * 40), 5
    )
    baseline = pd.DataFrame(columns=["ticker", "date", *tpa.INDICATORS])
    assert empty.empty
    assert len(tpa.comparison_table(empty, empty, baseline)) == len(tpa.INDICATORS)
    assert tpa.distribution_table(empty, empty).empty
    assert (tpa.regime_split(empty, baseline)["n"] == 0).all()
