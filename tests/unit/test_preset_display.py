"""Tests for logic/preset_display.py — pure rendering."""

from typing import ClassVar

import pandas as pd

from stock_dashboard.config import LIMITATIONS_URL
from stock_dashboard.logic.preset_display import (
    describe_filter,
    interpret_preset,
    render_preset_summary,
    screener_differences,
    screener_regime_note,
)


class TestDescribeFilter:
    def test_rsi_min(self):
        assert "RSI ≥ 60" in describe_filter("rsi_min", 60)

    def test_ratio_to_200ma_min(self):
        assert "1.27" in describe_filter("ratio_to_200ma_min", 1.27)

    def test_pattern_label(self):
        assert describe_filter("has_bull_flag", True) == "Bull flag"

    def test_unknown_falls_back(self):
        assert describe_filter("mystery_filter", 42) == "mystery_filter = 42"


class TestRenderPresetSummary:
    def _preset(self, **overrides):
        base = {
            "_name": "TA-disc_test",
            "_regime": None,
            "_raw_filters": [
                {"name": "adr_14_min", "value": 5.5},
                {"name": "ratio_to_200ma_min", "value": 1.27},
            ],
            "_stats": {
                "T+5": {"mean_return": 1.4, "win_rate": 0.50, "sharpe": 0.09, "p_value_adj": 0.01},
                "T+10": {"mean_return": 2.9, "win_rate": 0.50, "sharpe": 0.13, "p_value_adj": 0.0},
                "T+20": {"mean_return": 5.2, "win_rate": 0.48, "sharpe": 0.14, "p_value_adj": 0.0},
            },
            "_discovery": {
                "n_discovery": 22170,
                "cond_hit_rate_disc": 0.298,
                "lift_vs_overall_bps_disc": 192.8,
                "p_fdr_disc": 0.0,
            },
        }
        base.update(overrides)
        return base

    def test_includes_filter_description(self):
        md = render_preset_summary(self._preset())
        assert "ADR%" in md
        assert "200MA" in md

    def test_includes_stats_table(self):
        md = render_preset_summary(self._preset())
        assert "T+10" in md
        assert "+2.90%" in md

    def test_labels_mean_sd_and_states_the_in_sample_caveat(self):
        md = render_preset_summary(self._preset())
        assert "Sharpe" not in md
        assert "Mean/SD" in md
        assert "In-sample" in md
        assert "✓" not in md

    def test_includes_discovery_block_when_present(self):
        md = render_preset_summary(self._preset())
        assert "29.8%" in md
        assert "+193" in md or "+192" in md
        assert "**29.8%**" not in md  # numbers are stated, not highlighted
        assert "BH p <0.0001" in md

    def test_omits_discovery_when_absent(self):
        md = render_preset_summary(self._preset(_discovery=None))
        assert "Discovery" not in md

    def test_empty_preset_returns_empty(self):
        assert render_preset_summary({}) == ""


class TestInterpretPreset:
    STATS: ClassVar[dict[str, dict[str, float]]] = {
        "T+5": {"mean_return": 9.0, "win_rate": 0.7, "sharpe": 0.5, "p_value_adj": 0.001},
        "T+20": {"mean_return": -3.0, "win_rate": 0.40, "sharpe": -0.3, "p_value_adj": 0.2},
    }

    def test_quotes_t20_not_the_best_looking_horizon(self):
        msg = interpret_preset({"_stats": self.STATS})
        assert msg.startswith("In-sample backtest, T+20: mean -3.00%, win rate 40%")
        assert "mean/SD -0.30, BH p 0.2000" in msg

    def test_falls_back_to_the_next_fixed_horizon(self):
        msg = interpret_preset({"_stats": {"T+5": self.STATS["T+5"]}})
        assert "T+5" in msg

    def test_neutral_wording(self):
        for stats in (self.STATS, {"T+5": self.STATS["T+5"]}):
            msg = interpret_preset({"_stats": stats}).lower()
            for word in ("outperformance", "significant", "sharpe", "strong", "best"):
                assert word not in msg
            assert "optimistic" in msg
            assert "design.md#limitations" in msg

    def test_links_to_the_limitations_on_github(self):
        """A bare docs/ path goes nowhere in an installed package or the Docker image."""
        msg = interpret_preset({"_stats": self.STATS})
        assert f"]({LIMITATIONS_URL})" in msg
        assert LIMITATIONS_URL.startswith(
            "https://github.com/scottmcghee/stock-dashboard/blob/main/"
        )

    def test_empty_stats(self):
        assert interpret_preset({"_stats": {}}) == "This preset has no backtest stats."

    def test_malformed_stats(self):
        msg = interpret_preset({"_stats": {"T+20": {"mean_return": "n/a"}}})
        assert "incomplete" in msg


class TestScreenerRegimeNote:
    REGIMES = pd.Series(
        ["bull", "bear", None], index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    )

    def test_no_regime_no_note(self):
        assert screener_regime_note(None, self.REGIMES, "MKT") == ""

    def test_matching_regime(self):
        note = screener_regime_note("bear", self.REGIMES, "MKT")
        assert "MKT is in the bear regime on 2026-01-05" in note
        assert "matches" in note

    def test_differing_regime_is_said_plainly(self):
        note = screener_regime_note("bull", self.REGIMES, "MKT")
        assert "backtested in the bull regime, but MKT is in the bear regime" in note
        assert "does not hold" in note

    def test_without_regimes(self):
        note = screener_regime_note("bull", None, "SPY")
        assert "SPY has no classified sessions" in note


def test_screener_differences_name_the_filters_it_measures_differently():
    notes = screener_differences(
        [{"name": "volume_z_min", "value": 2}, {"name": "rsi_min", "value": 30}]
    )
    assert len(notes) == 1
    assert "252 sessions" in notes[0]
    notes = screener_differences([{"name": "rs_z_score_max", "value": 1}])
    assert "ranks highest" in notes[0]
    assert screener_differences([{"name": "rsi_min", "value": 30}]) == []
