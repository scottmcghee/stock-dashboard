"""Tests for logic/backtest_report.py: the backtester page's tables and helpers."""

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.logic.backtest import (
    FilterSpec,
    HorizonStats,
    Strategy,
    StrategyResult,
    evaluate_strategy,
)
from stock_dashboard.logic.backtest_report import (
    ANY_REGIME_LABEL,
    HORIZONS,
    cumulative_signal_returns,
    format_p_value,
    horizon_column,
    horizon_stats_table,
    preset_from_result,
    preset_stats,
    regime_blocks,
    regime_breakdown,
    regime_choices,
    results_column_config_formats,
    results_table,
    sort_results,
    strategy_signals,
)
from stock_dashboard.logic.preset import parse_preset, save_preset


def _hs(mean: float, significant: bool = True, **kw) -> HorizonStats:
    return HorizonStats(
        mean_return=mean,
        median_return=kw.get("median", mean),
        win_rate=kw.get("win_rate", 0.6),
        sharpe=kw.get("sharpe", 0.2),
        p_value=0.01,
        p_value_adj=kw.get("p_adj", 0.02),
        significant=significant,
    )


def _result(name: str, n: int, horizons: dict, regime: str | None = None) -> StrategyResult:
    strategy = Strategy(name=name, filters=[FilterSpec("rsi_min", 60)], regime=regime)
    return StrategyResult(strategy=strategy, n_signals=n, horizons=horizons)


def test_regime_choices():
    assert regime_choices([ANY_REGIME_LABEL, "bull"]) == [None, "bull"]


class TestRegimeBlocks:
    def test_contiguous_runs_touch(self):
        dates = pd.bdate_range("2026-01-01", periods=6)
        regimes = pd.Series([np.nan, "bull", "bull", "bear", "bear", "bull"], index=dates)
        blocks = regime_blocks(regimes)
        assert [b.regime for b in blocks] == ["bull", "bear", "bull"]
        assert blocks[0].start == dates[1]
        assert blocks[0].end == blocks[1].start == dates[3]
        assert blocks[-1].start == blocks[-1].end == dates[5]

    def test_empty(self):
        assert regime_blocks(pd.Series([np.nan], dtype=object)) == []


RESULTS = (
    _result("sig", 50, {"T+5": _hs(0.01), "T+20": _hs(0.03)}),
    _result("not_sig", 50, {"T+5": _hs(0.02, significant=False)}),
    _result("few", 5, {"T+5": _hs(0.05)}),
)


class TestResultsTable:
    def test_keeps_only_reportable_strategies(self):
        table = results_table(RESULTS)
        assert table["strategy"].tolist() == ["sig"]

    def test_show_all(self):
        assert len(results_table(RESULTS, show_all=True)) == 3

    def test_values_in_percent_and_missing_horizons(self):
        row = results_table(RESULTS).iloc[0]
        assert row["T+5_mean"] == pytest.approx(1.0)
        assert row["T+5_wr"] == pytest.approx(60.0)
        assert row["T+20_mean"] == pytest.approx(3.0)
        assert pd.isna(row["T+10_mean"])
        assert not row["T+10_sig"]
        assert row["regime"] == "any"

    def test_empty(self):
        assert results_table([]).empty

    def test_sort_results(self):
        table = results_table(RESULTS, show_all=True)
        ordered = sort_results(table, "T+5", "mean_return")
        assert ordered["strategy"].tolist() == ["few", "not_sig", "sig"]
        assert sort_results(pd.DataFrame(), "T+5", "sharpe").empty

    def test_formats_cover_every_numeric_column(self):
        table = results_table(RESULTS)
        formats = results_column_config_formats()
        numeric = [c for c in table.columns if c not in ("strategy", "regime")]
        assert all(c in formats for c in numeric if not c.endswith("_sig"))
        assert formats["T+5_pval"] == ("T+5 p (BH)", None)  # shown by format_p_value


@pytest.mark.parametrize(
    ("p", "text"), [(0.00001, "<0.0001"), (0.0001, "0.0001"), (0.03456, "0.0346"), (None, "")]
)
def test_format_p_value(p, text):
    assert format_p_value(p) == text


@pytest.fixture
def indicators() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=8)
    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["A"] * 8 + ["B"] * 8,
            "rsi_14": [70, 50, 65, 80, 40, 75, 61, 30] * 2,
            "fwd_5": [0.01, 0.02, -0.01, 0.03, 0.0, 0.02, np.nan, 0.01] * 2,
        }
    )


@pytest.fixture
def regimes(indicators) -> pd.Series:
    dates = sorted(indicators["date"].unique())
    return pd.Series(["bull", "bull", "bear", "bear", "bull", "bull", "bear", "bull"], index=dates)


class TestStrategySignals:
    def test_matches_evaluate_strategy(self, indicators, regimes):
        result = _result("rsi", 0, {}, regime="bull")
        signals = strategy_signals(indicators, result, regimes)
        assert len(signals) == evaluate_strategy(indicators, result.strategy, regimes).n_signals
        assert (signals["rsi_14"] >= 60).all()
        assert set(signals["date"].map(regimes)) == {"bull"}

    def test_regime_strategy_without_regimes_has_no_signals(self, indicators):
        result = _result("rsi", 0, {}, regime="bull")
        assert strategy_signals(indicators, result, None).empty

    def test_unconditional_strategy_without_regimes(self, indicators):
        result = _result("rsi", 0, {})
        assert len(strategy_signals(indicators, result, None)) == 10


def test_horizon_column():
    assert horizon_column("T+20") == "fwd_20"
    assert all(horizon_column(h).startswith("fwd_") for h in HORIZONS)


class TestCumulativeSignalReturns:
    def test_compounds_daily_means(self):
        signals = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-05"]),
                "fwd_5": [0.10, 0.0, 0.10],
            }
        )
        cumulative = cumulative_signal_returns(signals, "T+5")
        assert cumulative.tolist() == pytest.approx([5.0, 15.5])

    def test_missing_column_or_rows(self):
        assert cumulative_signal_returns(pd.DataFrame({"date": []}), "T+5").empty


class TestRegimeBreakdown:
    def test_per_regime_rows_above_minimum(self, indicators, regimes):
        table = regime_breakdown(indicators, regimes, min_signals=5)
        assert table["regime"].tolist() == ["bull", "bear"]
        bull = table.set_index("regime").loc["bull"]
        assert bull["n_signals"] == 10
        assert bull["T+5 Mean %"] == pytest.approx(1.2)
        assert bull["T+5 Win %"] == pytest.approx(80.0)
        assert "T+20 Mean %" not in table.columns

    def test_too_few_signals(self, indicators, regimes):
        assert regime_breakdown(indicators, regimes, min_signals=100).empty


def test_horizon_stats_table():
    table = horizon_stats_table(_result("s", 20, {"T+5": _hs(0.01, p_adj=0.03)}))
    row = table.iloc[0]
    assert row["Horizon"] == "T+5"
    assert row["Mean Return %"] == pytest.approx(1.0)
    assert row["p-value (BH)"] == pytest.approx(0.03)


class TestPresetFromResult:
    def test_stats_format(self):
        stats = preset_stats(_result("s", 20, {"T+5": _hs(0.012345, win_rate=0.61234)}))
        assert stats == {
            "T+5": {"mean_return": 1.23, "win_rate": 0.612, "sharpe": 0.2, "p_value_adj": 0.02}
        }

    def test_round_trip_through_save_and_parse(self, tmp_path, monkeypatch):
        from stock_dashboard.config import get_settings

        monkeypatch.setenv("PRESET_DIR", str(tmp_path))
        get_settings.cache_clear()
        result = _result("rsi_min=60 & regime=bull", 20, {"T+20": _hs(0.02)}, regime="bull")
        preset = preset_from_result(result, "rsi_min=60 & regime=bull")
        assert preset["filters"] == [{"name": "rsi_min", "value": 60}]
        assert preset["regime"] == "bull"
        parsed = parse_preset(save_preset(preset).stem)
        assert parsed["rsi_min"] == 60
        assert parsed["_regime"] == "bull"
        assert parsed["_stats"]["T+20"]["mean_return"] == 2.0

    def test_unconditional_strategy_has_no_regime(self):
        preset = preset_from_result(_result("s", 20, {}), "s")
        assert preset["regime"] is None

    def test_records_the_etf_of_an_etf_filtered_run(self):
        preset = preset_from_result(_result("s", 20, {}), "s", etf="SEMX")
        assert preset["etf"] == "SEMX"
        assert "etf" not in preset_from_result(_result("s", 20, {}), "s")
