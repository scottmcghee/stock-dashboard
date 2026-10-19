"""Shared research helpers: statistics, SQL building and the signal-bin report."""

from __future__ import annotations

import importlib
import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from research import common, signal_analysis
from tests.unit.conftest import render_sql


def test_fdr_adjust_matches_benjamini_hochberg_and_skips_nan() -> None:
    p = [0.01, np.nan, 0.04, 0.03]
    adjusted = common.fdr_adjust(p)
    # m = 3 (the NaN is not a test): sorted 0.01, 0.03, 0.04 -> 0.03, 0.04, 0.04
    np.testing.assert_allclose(adjusted[[0, 2, 3]], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[1])
    assert common.fdr_adjust([]).size == 0


def test_where_clause_binds_every_value() -> None:
    where, params = common.where_clause(
        date_col="collection_date",
        symbol_col="symbol",
        symbols=["ABC"],
        start=date(2025, 1, 2),
        end=None,
    )
    text = render_sql(where)
    assert text == 'WHERE TRUE AND "symbol" = ANY(%(symbols)s) AND "collection_date" >= %(start)s'
    assert params == {"symbols": ["ABC"], "start": date(2025, 1, 2)}


def _signal_frame(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {"good": x, "noise": rng.normal(size=n), "y": x + rng.normal(scale=0.5, size=n)}
    )


def test_univariate_scan_ranks_the_informative_signal_first() -> None:
    out = common.univariate_scan(_signal_frame(), "y", ["noise", "good", "missing"], min_n=100)
    assert out["signal"].tolist() == ["good", "noise"]
    good = out.iloc[0]
    assert good["spearman_rho"] > 0.5
    assert good["lift_bps"] > 0
    assert good["top_dec_hit_rate"] > good["overall_hit_rate"]
    assert {"p_value", "p_fdr"} <= set(out.columns)


def test_univariate_scan_skips_signals_below_min_n() -> None:
    assert common.univariate_scan(_signal_frame(50), "y", ["good"], min_n=100).empty


def test_pair_scan_tests_each_pair_and_direction_once() -> None:
    df = _signal_frame(5000)
    df["good2"] = df["good"] + np.random.default_rng(1).normal(scale=0.1, size=len(df))
    out = common.pair_scan(df, "y", ["good", "good2", "noise"], min_n=10)
    assert len(out.drop_duplicates(["signal_a", "dir_a", "signal_b", "dir_b"])) == len(out)
    best = out.iloc[0]
    assert {best["signal_a"], best["signal_b"]} == {"good", "good2"}
    assert best["dir_a"] == best["dir_b"]


def test_replication_summary_counts_agreeing_groups() -> None:
    rep = pd.DataFrame(
        {
            "g": ["a", "b", "c", "a", "b"],
            "signal": ["s", "s", "s", "t", "t"],
            "n": 100,
            "rho": [0.2, 0.1, -0.05, 0.1, -0.1],
            "p": [0.01, 0.02, 0.5, 0.01, 0.01],
        }
    )
    summary = common.summarize_replication(rep, ["s", "t"]).set_index("signal")
    assert summary.loc["s", "n_groups"] == 3
    assert summary.loc["s", "pct_sig_same_sign"] == pytest.approx(2 / 3)
    assert common.replicating_signals(rep, need=2) == {"s"}


def test_forward_returns_stay_within_each_ticker() -> None:
    prices = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "B", "B"],
            "date": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-02", "2025-01-03"]
            ),
            "close": [10.0, 11.0, 12.1, 50.0, 25.0],
        }
    )
    out = common.forward_returns(prices, [1, 2])
    a = out[out["ticker"] == "A"]
    np.testing.assert_allclose(a["fwd_1"].iloc[:2], [0.1, 0.1])
    assert a["fwd_2"].iloc[0] == pytest.approx(0.21)
    b = out[out["ticker"] == "B"]
    assert b["fwd_1"].iloc[0] == pytest.approx(-0.5)
    assert b["fwd_1"].isna().iloc[-1]
    assert b["fwd_2"].isna().all()


def test_bollinger_width_matches_its_definition() -> None:
    close = pd.Series(np.linspace(10, 20, 40))
    width = common.bollinger_width(close)
    window = close.iloc[-20:]
    expected = 4 * window.std() / window.mean() * 100
    assert width.iloc[-1] == pytest.approx(expected)
    assert width.iloc[:19].isna().all()


# ---------------------------------------------------------------------------
# signal_analysis
# ---------------------------------------------------------------------------


def test_volume_ratio_uses_only_prior_sessions() -> None:
    n = signal_analysis.VOLUME_AVG_SESSIONS + 1
    df = pd.DataFrame(
        {
            "symbol": "ABC",
            "date": pd.bdate_range("2025-01-01", periods=n),
            "flownet": np.arange(n, dtype=float),
            "dxoi": 1.0,
            "gxoi": 1.0,
            "option_volume": [100.0] * (n - 1) + [300.0],
            "put_call_ratio": 1.0,
            "vannaxoi": 1.0,
        }
    )
    out = signal_analysis.add_features(df)
    assert out["vol_ratio"].iloc[-1] == pytest.approx(3.0)
    assert out["vol_ratio"].iloc[:-1].isna().all()


def test_signal_report_adjusts_p_values_over_all_cells() -> None:
    rng = np.random.default_rng(0)
    n = 400
    panel = pd.DataFrame(
        {
            "symbol": "ABC",
            "flownet_decile": rng.integers(0, 10, n),
            "vanna_quintile": rng.integers(0, 5, n),
            "dxoi_pct_change": rng.normal(scale=0.2, size=n),
            "gxoi": rng.normal(10_000, 20_000, n),
            "vol_ratio": rng.uniform(0.5, 3, n),
            "put_call_ratio_pct_change": rng.normal(scale=0.3, size=n),
            "gxoi_pct_change": rng.normal(scale=0.5, size=n),
            "fwd_1": rng.normal(size=n),
            "fwd_5": rng.normal(size=n),
        }
    )
    out = signal_analysis.run(panel, [1, 5], min_bin_n=10)
    assert set(out["signal"]) == {s.name for s in signal_analysis.SIGNALS}
    assert {"p_1", "p_5", "p_fdr_1", "p_fdr_5"} <= set(out.columns)
    tested = out[["p_1", "p_5"]].notna()
    assert (out[["p_fdr_1", "p_fdr_5"]].notna() == tested.to_numpy()).all().all()
    assert (out["p_fdr_1"].dropna() >= out.loc[tested["p_1"], "p_1"]).all()


@pytest.mark.parametrize(
    "module",
    ["signal_analysis", "ta_signal_analysis", "option_signal_analysis", "trend_precursor_analysis"],
)
def test_help_describes_every_option(module: str, capsys: pytest.CaptureFixture[str]) -> None:
    """The README defers to --help, which once listed options with no description or default."""
    with pytest.raises(SystemExit):
        importlib.import_module(f"research.{module}").main(["--help"])
    lines = capsys.readouterr().out.split("options:")[1].splitlines()
    options = [i for i, line in enumerate(lines) if line.startswith("  -")]
    assert options
    for i in options:
        described = len(re.split(r"\s{2,}", lines[i].strip())) > 1 or (
            i + 1 < len(lines)
            and lines[i + 1].startswith(" " * 6)
            and not lines[i + 1].strip().startswith("-")
        )
        assert described, lines[i]
    text = " ".join(" ".join(lines).split())
    assert "(default: None)" not in text
    assert "(default: False)" not in text
    assert "logging level (default: INFO)" in text


def test_help_formatter_skips_missing_and_stated_defaults() -> None:
    parser = common.make_parser("demo")
    parser.add_argument("--n", type=int, default=30, help="rows")
    parser.add_argument("--start", help="first date (default: all history)")
    parser.add_argument("--flag", action="store_true", help="a switch")
    parser.add_argument("--k", type=int, default=5, help="top K (default: 5)")
    text = " ".join(parser.format_help().split())
    assert "--n N rows (default: 30)" in text
    assert "first date (default: all history)" in text
    assert "a switch" in text
    assert "(default: False)" not in text
    assert text.count("(default: 5)") == 1
