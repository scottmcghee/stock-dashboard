"""Tests for logic/preset_signal.py: signal mask and direction."""

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.logic.backtest import (
    FILTER_TO_COLUMN,
    FilterSpec,
    compute_rs_impulse,
    precompute_indicators,
    signal_mask,
)
from stock_dashboard.logic.preset_signal import (
    available_metric_columns,
    compute_preset_signal_mask,
    determine_signal_direction,
    unsupported_filters,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """260 bars of OHLCV with trending data."""
    rng = np.random.default_rng(42)
    n = 260
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.3,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.integers(500_000, 2_000_000, size=n),
        }
    )


@pytest.fixture
def sample_metrics(sample_ohlcv) -> pd.DataFrame:
    """Two ETF rows per date, as fact_stock_metrics stores them."""
    dates = sample_ohlcv["date"]
    rank = np.linspace(0, 1, len(dates))
    rows = [
        pd.DataFrame({"date": dates, "etf_ticker": etf, "rs_rank_overall": rank, "rs_z_score": z})
        for etf, z in [("AAA", np.zeros(len(dates))), ("BBB", np.ones(len(dates)))]
    ]
    return pd.concat(rows, ignore_index=True)


def _mask(ohlcv, metrics, filters):
    return compute_preset_signal_mask(ohlcv, metrics, filters).mask


class TestComputePresetSignalMask:
    def test_empty_filters_all_true(self, sample_ohlcv):
        mask, unsupported, _ = compute_preset_signal_mask(sample_ohlcv, None, [])
        assert mask.all()
        assert len(mask) == len(sample_ohlcv)
        assert unsupported == []

    def test_rsi_min_filter(self, sample_ohlcv):
        mask = _mask(sample_ohlcv, None, [{"name": "rsi_min", "value": 50}])
        assert len(mask) == len(sample_ohlcv)
        assert not mask.all()
        assert mask.any()

    def test_rsi_max_filter(self, sample_ohlcv):
        mask = _mask(sample_ohlcv, None, [{"name": "rsi_max", "value": 50}])
        assert not mask.all()
        assert mask.any()

    def test_momentum_filter(self, sample_ohlcv):
        mask = _mask(sample_ohlcv, None, [{"name": "momentum_63_min", "value": 0.0}])
        assert len(mask) == len(sample_ohlcv)
        assert not mask.iloc[0]  # momentum undefined on the first bars

    def test_ratio_to_ma_filter(self, sample_ohlcv):
        mask = _mask(sample_ohlcv, None, [{"name": "ratio_to_200ma_min", "value": 0.5}])
        assert not mask.iloc[:199].any()
        assert mask.iloc[199:].all()

    def test_weinstein_stage_filter(self, sample_ohlcv):
        mask = _mask(sample_ohlcv, None, [{"name": "weinstein_stage", "value": "Stage 2"}])
        assert len(mask) == len(sample_ohlcv)

    def test_combined_filters_and_behavior(self, sample_ohlcv):
        """Multiple filters are AND'd: the result is a subset of each individual filter."""
        mask1 = _mask(sample_ohlcv, None, [{"name": "rsi_min", "value": 40}])
        mask2 = _mask(sample_ohlcv, None, [{"name": "rsi_max", "value": 60}])
        combined = _mask(
            sample_ohlcv,
            None,
            [{"name": "rsi_min", "value": 40}, {"name": "rsi_max", "value": 60}],
        )
        assert (combined & ~mask1).sum() == 0
        assert (combined & ~mask2).sum() == 0

    def test_unknown_filter_is_reported_and_matches_nothing(self, sample_ohlcv):
        """Old behavior: unknown filters were silently dropped, so every bar matched."""
        filters = [{"name": "rsi_min", "value": 0}, {"name": "totally_unknown", "value": 42}]
        mask, unsupported, _ = compute_preset_signal_mask(sample_ohlcv, None, filters)
        assert unsupported == ["totally_unknown=42"]
        assert not mask.any()

    def test_unknown_filter_is_logged(self, sample_ohlcv, caplog):
        compute_preset_signal_mask(sample_ohlcv, None, [{"name": "nope", "value": 1}])
        assert "nope=1" in caplog.text

    def test_metric_filter_without_metrics_is_reported(self, sample_ohlcv):
        mask, unsupported, _ = compute_preset_signal_mask(
            sample_ohlcv, None, [{"name": "rs_rank_overall_min", "value": 0.5}]
        )
        assert unsupported == ["rs_rank_overall_min=0.5"]
        assert not mask.any()

    def test_metrics_merge_rs_rank(self, sample_ohlcv, sample_metrics):
        mask = _mask(sample_ohlcv, sample_metrics, [{"name": "rs_rank_overall_min", "value": 0.8}])
        assert 0 < mask.sum() < len(sample_ohlcv) * 0.5

    def test_rs_z_score_averages_etf_rows_like_backtest(self, sample_ohlcv, sample_metrics):
        """The backtest averages rs_z_score over a stock's ETF rows (here 0 and 1 -> 0.5)."""
        at_half = _mask(sample_ohlcv, sample_metrics, [{"name": "rs_z_score_min", "value": 0.5}])
        above = _mask(sample_ohlcv, sample_metrics, [{"name": "rs_z_score_min", "value": 0.51}])
        assert at_half.all()
        assert not above.any()

    def test_volume_z_filter(self, sample_ohlcv):
        mask = _mask(sample_ohlcv, None, [{"name": "volume_z_min", "value": 0.0}])
        assert len(mask) == len(sample_ohlcv)
        assert mask.any()

    def test_adr_filter(self, sample_ohlcv):
        mask = _mask(sample_ohlcv, None, [{"name": "adr_14_min", "value": 1.0}])
        assert len(mask) == len(sample_ohlcv)

    @pytest.mark.parametrize(
        "name", ["macd_histogram_min", "dist_recent_low_126_min", "has_evt_regain_200ma"]
    )
    def test_filters_the_old_overlay_dropped_are_supported(self, sample_ohlcv, name):
        _, unsupported, _ = compute_preset_signal_mask(
            sample_ohlcv, None, [{"name": name, "value": 0}]
        )
        assert unsupported == []

    def test_keeps_input_index_when_unsorted(self, sample_ohlcv):
        shuffled = sample_ohlcv.sample(frac=1.0, random_state=1)
        filters = [{"name": "rsi_min", "value": 50}]
        expected = _mask(sample_ohlcv, None, filters)
        mask = _mask(shuffled, None, filters)
        assert mask.index.equals(shuffled.index)
        pd.testing.assert_series_equal(mask.sort_index(), expected, check_names=False)


class TestOverlayMatchesBacktest:
    """The overlay must mark exactly the bars the backtest counts as signals."""

    @pytest.mark.parametrize(
        "filters",
        [
            [{"name": "volume_z_min", "value": 1.0}],  # old overlay used a 30-bar window
            [{"name": "rsi_max", "value": 45}, {"name": "ratio_to_50ma_min", "value": 1.0}],
            [{"name": "has_evt_macd_bullish_cross", "value": True}],
            [{"name": "rs_impulse_10_min", "value": 0.0}],
        ],
    )
    def test_same_mask_as_backtest(self, sample_ohlcv, sample_metrics, filters):
        panel = precompute_indicators(sample_ohlcv.assign(ticker="T"))
        per_date = sample_metrics.groupby("date", as_index=False)[
            ["rs_rank_overall", "rs_z_score"]
        ].mean()
        panel = compute_rs_impulse(panel.merge(per_date.assign(ticker="T"), on=["date", "ticker"]))
        specs = [FilterSpec(f["name"], f["value"]) for f in filters]
        expected = signal_mask(panel, specs).to_numpy()
        overlay = _mask(sample_ohlcv, sample_metrics, filters).to_numpy()
        np.testing.assert_array_equal(overlay, expected)


class TestUnsupportedFilters:
    def test_every_backtest_filter_is_supported_with_metrics(self, sample_metrics):
        filters = [{"name": name, "value": 0} for name in FILTER_TO_COLUMN]
        assert unsupported_filters(filters, available_metric_columns(sample_metrics)) == []

    def test_impulse_needs_rank_column(self):
        metrics = pd.DataFrame({"date": [pd.Timestamp("2025-01-01")], "rs_z_score": [1.0]})
        assert available_metric_columns(metrics) == {"rs_z_score"}
        assert unsupported_filters(
            [{"name": "rs_impulse_10_min", "value": 0.1}], available_metric_columns(metrics)
        ) == ["rs_impulse_10_min=0.1"]

    def test_no_metrics(self):
        assert available_metric_columns(None) == set()
        assert available_metric_columns(pd.DataFrame()) == set()


class TestDetermineSignalDirection:
    def test_bullish_from_t20(self):
        stats = {"T+20": {"mean_return": 2.5, "win_rate": 0.6}}
        assert determine_signal_direction(stats) == "bullish"

    def test_bearish_from_t20(self):
        stats = {"T+20": {"mean_return": -1.5, "win_rate": 0.4}}
        assert determine_signal_direction(stats) == "bearish"

    def test_fallback_to_t10(self):
        assert determine_signal_direction({"T+10": {"mean_return": 1.0}}) == "bullish"

    def test_fallback_to_t5(self):
        assert determine_signal_direction({"T+5": {"mean_return": -0.5}}) == "bearish"

    def test_fallback_to_t63(self):
        assert determine_signal_direction({"T+63": {"mean_return": 3.0}}) == "bullish"

    def test_empty_stats_defaults_bullish(self):
        assert determine_signal_direction({}) == "bullish"

    def test_zero_return_is_bearish(self):
        assert determine_signal_direction({"T+20": {"mean_return": 0}}) == "bearish"


class TestPresetRegimeAndEtf:
    """The overlay replays the preset's regime and the ETF it was backtested within."""

    def test_regime_masks_bars_outside_it(self, sample_ohlcv):
        dates = sample_ohlcv["date"]
        regimes = pd.Series(np.where(np.arange(len(dates)) % 2, "bull", "bear"), index=dates)
        filters = [{"name": "rsi_min", "value": 0}]
        unconditional = _mask(sample_ohlcv, None, filters)
        signal = compute_preset_signal_mask(
            sample_ohlcv, None, filters, regime="bull", regimes=regimes
        )
        assert signal.unsupported == []
        assert signal.mask.any()
        assert (dates[signal.mask].map(regimes) == "bull").all()
        assert signal.mask.equals(unconditional & dates.map(regimes).eq("bull"))

    def test_regime_without_regime_series_matches_nothing(self, sample_ohlcv):
        signal = compute_preset_signal_mask(
            sample_ohlcv, None, [{"name": "rsi_min", "value": 0}], regime="bear"
        )
        assert signal.unsupported == ["regime=bear"]
        assert not signal.mask.any()

    def test_regime_alone_marks_its_bars(self, sample_ohlcv):
        dates = sample_ohlcv["date"]
        regimes = pd.Series("bear", index=dates)
        regimes.iloc[:100] = "bull"
        signal = compute_preset_signal_mask(sample_ohlcv, None, [], regime="bull", regimes=regimes)
        assert int(signal.mask.sum()) == 100

    @pytest.mark.parametrize(("etf", "expected"), [(None, True), ("AAA", False), ("BBB", True)])
    def test_etf_selects_that_etfs_metric_rows(self, sample_ohlcv, sample_metrics, etf, expected):
        """AAA's z-score is 0, BBB's 1; their mean 0.5 is what an all-stocks backtest sees."""
        filters = [{"name": "rs_z_score_min", "value": 0.5}]
        mask = compute_preset_signal_mask(sample_ohlcv, sample_metrics, filters, etf=etf).mask
        assert mask.all() if expected else not mask.any()

    @pytest.mark.parametrize(
        "filters",
        [
            [{"name": "rs_rank_overall_min", "value": 0.0}],  # same in every ETF
            [{"name": "rsi_min", "value": 0}],  # price-only
        ],
    )
    def test_ticker_outside_the_preset_etf_has_no_signals(
        self, sample_ohlcv, sample_metrics, filters
    ):
        """Its backtest never saw this ticker; once reported as a filter it 'cannot evaluate'
        (RS filters) or marked signals anyway (price-only filters)."""
        signal = compute_preset_signal_mask(sample_ohlcv, sample_metrics, filters, etf="ZZZ")
        assert signal.outside_etf
        assert signal.unsupported == []
        assert not signal.mask.any()
