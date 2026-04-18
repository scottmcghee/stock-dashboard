"""Tests for logic/preset_display.py — pure rendering."""

from logic.preset_display import (
    describe_filter,
    interpret_preset,
    render_preset_summary,
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
                "T+5":  {"mean_return": 1.4, "win_rate": 0.50, "sharpe": 0.09, "p_value_adj": 0.01},
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
        assert "ADR%" in md and "200MA" in md

    def test_includes_stats_table(self):
        md = render_preset_summary(self._preset())
        assert "T+10" in md and "+2.90%" in md

    def test_marks_significant_p(self):
        md = render_preset_summary(self._preset())
        assert "✓" in md  # T+10, T+20 have p < 0.05

    def test_includes_discovery_block_when_present(self):
        md = render_preset_summary(self._preset())
        assert "29.8%" in md
        assert "+193" in md or "+192" in md

    def test_omits_discovery_when_absent(self):
        md = render_preset_summary(self._preset(_discovery=None))
        assert "Discovery" not in md

    def test_empty_preset_returns_empty(self):
        assert render_preset_summary({}) == ""


class TestInterpretPreset:
    def test_outperformance_language_for_positive(self):
        preset = {"_stats": {"T+10": {"mean_return": 2.5, "win_rate": 0.52, "sharpe": 0.25, "p_value_adj": 0.01}}}
        msg = interpret_preset(preset)
        assert "outperformance" in msg
        assert "T+10" in msg

    def test_underperformance_for_negative(self):
        preset = {"_stats": {"T+20": {"mean_return": -3.0, "win_rate": 0.40, "sharpe": -0.3, "p_value_adj": 0.001}}}
        msg = interpret_preset(preset)
        assert "underperformance" in msg

    def test_significance_flag(self):
        sig = interpret_preset({"_stats": {"T+5": {"mean_return": 1.0, "win_rate": 0.5, "sharpe": 0.2, "p_value_adj": 0.001}}})
        nonsig = interpret_preset({"_stats": {"T+5": {"mean_return": 1.0, "win_rate": 0.5, "sharpe": 0.2, "p_value_adj": 0.5}}})
        assert "significant" in sig
        assert "not significant" in nonsig

    def test_empty_stats(self):
        assert "stats" in interpret_preset({"_stats": {}}).lower() or "behavior" in interpret_preset({"_stats": {}}).lower()
