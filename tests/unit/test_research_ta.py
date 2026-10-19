"""ta_signal_analysis: exportable filters, universe selection, residuals, events, presets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import ta_signal_analysis as ta
from stock_dashboard.logic.backtest import FILTER_TO_COLUMN, METRIC_COLUMNS, RS_IMPULSE_COLUMNS
from stock_dashboard.logic.preset_signal import unsupported_filters


def test_exportable_signals_are_exactly_those_the_backtester_filters() -> None:
    for signal, (lo, hi) in ta.SIGNAL_TO_FILTER.items():
        assert FILTER_TO_COLUMN[lo] == FILTER_TO_COLUMN[hi] == signal
    assert set(ta.SIGNAL_TO_FILTER) | set(ta.NOT_EXPORTABLE) == set(ta.SIGNAL_COLS)
    # Features computed only in the research script cannot be replayed.
    assert {"bb_width", "bb_pctb", "dist_52w_high", "atr_over_price"} <= set(ta.NOT_EXPORTABLE)


def test_every_exportable_filter_replays_in_the_chart_overlay() -> None:
    filters = [{"name": n, "value": 1.0} for pair in ta.SIGNAL_TO_FILTER.values() for n in pair]
    metric_columns = {*METRIC_COLUMNS, *RS_IMPULSE_COLUMNS}
    assert unsupported_filters(filters, metric_columns) == []


def test_liquid_universe_ranks_on_the_latest_sessions_only() -> None:
    dates = pd.bdate_range("2025-01-01", periods=100)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"ticker": "OLD", "date": d, "close": 10.0, "volume": 1e9 if i < 30 else 1})
        rows.append({"ticker": "NEW", "date": d, "close": 10.0, "volume": 1e6})
    ohlcv = pd.DataFrame(rows)
    assert ta.select_liquid_universe(ohlcv, 1, bars=63) == ["NEW"]
    assert ta.select_liquid_universe(ohlcv.iloc[:0], 5) == []


def test_residual_returns_remove_beta_times_benchmark() -> None:
    panel = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-02"] * 2), "ticker": ["A", "B"]}
        | {f"fwd_{h}": [0.10, 0.10] for h in ta.HORIZONS}
    )
    bench = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"])})
    for h in ta.HORIZONS:
        bench[f"bench_fwd_{h}"] = 0.05
    beta = pd.DataFrame({"date": panel["date"], "ticker": ["A", "B"], "beta": [2.0, np.nan]})
    out = ta.residualize_forward_returns(panel, bench, beta)
    assert out["resid_fwd_10"].tolist() == pytest.approx([0.0, 0.05])  # NaN beta -> 1


def test_rolling_beta_recovers_a_known_beta() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2025-01-01", periods=120)
    mkt = rng.normal(scale=0.01, size=len(dates))
    bench = pd.DataFrame({"date": dates, "close": 100 * np.exp(np.cumsum(mkt))})
    stock = pd.DataFrame({"ticker": "A", "date": dates, "close": 50 * np.exp(np.cumsum(2 * mkt))})
    beta = ta.rolling_beta(stock, bench)
    assert beta["beta"].iloc[-1] == pytest.approx(2.0)
    assert beta["beta"].iloc[: ta.BETA_MIN_PERIODS - 1].isna().all()


def test_events_are_compared_with_the_rows_where_they_did_not_fire() -> None:
    n = 200
    panel = pd.DataFrame(
        {"y": np.r_[np.ones(50), np.zeros(n - 50)], "evt": [True] * 50 + [False] * 150}
    )
    panel["y"] += np.random.default_rng(0).normal(scale=0.01, size=n)
    out = ta.event_conditional_analysis(panel, "y", ["evt"], min_n=20)
    row = out.iloc[0]
    assert row["not_fired_mean"] == pytest.approx(0.0, abs=0.01)
    assert row["lift_bps"] == pytest.approx(10_000, rel=0.01)
    assert row["p_fdr"] < 0.05
    assert ta.event_conditional_analysis(panel, "y", ["evt"], min_n=60).empty


def _pair_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "signal_a": "rsi_14",
        "dir_a": "bottom",
        "signal_b": "momentum_63",
        "dir_b": "top",
        "n": 120,
        "cond_hit_rate": 0.3,
        "lift_vs_overall_bps": 50.0,
        "p_fdr": 0.01,
        "exportable": True,
    }
    return row | overrides


def _export_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    return pd.DataFrame(
        {
            "date": np.repeat(pd.bdate_range("2024-01-01", periods=n // 4), 4),
            "ticker": np.tile(list("ABCD"), n // 4),
            "rsi_14": rng.uniform(0, 100, n),
            "momentum_63": rng.normal(0, 10, n),
            "bb_width": rng.uniform(1, 20, n),
        }
        | {f"fwd_{h}": rng.normal(0.01, 0.02, n) for h in ta.HORIZONS}
    )


def test_export_writes_replayable_presets_and_skips_unexportable_pairs(tmp_path: Path) -> None:
    panel = _export_panel()
    pairs = pd.DataFrame([_pair_row(), _pair_row(signal_b="bb_width", exportable=False)])
    written = ta.export_presets(panel, pairs, pd.Series(dtype=object), tmp_path)

    assert len(written) == 1
    preset = json.loads(written[0].read_text())
    names = [f["name"] for f in preset["filters"]]
    assert names == ["rsi_max", "momentum_63_min"]
    assert preset["filters"][0]["value"] == pytest.approx(panel["rsi_14"].quantile(0.10), abs=1e-6)
    assert unsupported_filters(preset["filters"]) == []
    assert set(preset["stats"]) == {"T+5", "T+10", "T+20", "T+63"}
    assert preset["discovery"]["n_discovery"] == 120


def test_export_adjusts_p_values_across_all_exported_pairs(tmp_path: Path) -> None:
    panel = _export_panel()
    pairs = pd.DataFrame([_pair_row(), _pair_row(dir_a="top", dir_b="bottom")])
    no_regimes = pd.Series(dtype=object)
    written = ta.export_presets(panel, pairs, no_regimes, tmp_path)

    strategies = [
        ta.Strategy(
            name="",
            filters=[
                ta.decile_filter(panel, "rsi_14", dir_a),
                ta.decile_filter(panel, "momentum_63", dir_b),
            ],
        )
        for dir_a, dir_b in (("bottom", "top"), ("top", "bottom"))
    ]
    joint = ta.evaluate_all_strategies(panel, strategies, no_regimes)
    presets = [json.loads(path.read_text()) for path in written]
    stats_by_filters = {
        tuple((f["name"], f["value"]) for f in preset["filters"]): preset["stats"]
        for preset in presets
    }
    assert len(stats_by_filters) == 2
    for result in joint:
        stats = stats_by_filters[tuple((f.name, f.value) for f in result.strategy.filters)]
        for label, hs in result.horizons.items():
            assert stats[label]["p_value_adj"] == pytest.approx(hs.p_value_adj, abs=1e-4)
