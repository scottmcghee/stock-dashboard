"""Tests for logic/signal_scanner.py — pure functions, no mocking needed."""

import pandas as pd
import pytest

from stock_dashboard.logic.signal_scanner import (
    TIER_1_WEIGHT,
    TIER_2_WEIGHT,
    TIER_WEIGHTS,
    build_composite_scores,
)


class TestBuildCompositeScores:
    def test_combines_multiple_scans(self):
        scans = {
            "vanna_extreme": pd.DataFrame(
                {
                    "symbol": ["AAPL", "TSLA"],
                    "strength": [50000.0, -30000.0],
                }
            ),
            "volume_spike": pd.DataFrame(
                {
                    "symbol": ["AAPL", "NVDA"],
                    "strength": [500000.0, 300000.0],
                }
            ),
        }
        result = build_composite_scores(scans)
        assert "composite_score" in result.columns
        assert "signal_count" in result.columns
        # AAPL appears in both scans
        aapl = result[result["symbol"] == "AAPL"]
        assert len(aapl) == 1
        assert aapl.iloc[0]["signal_count"] == 2

    def test_tier_1_weighted_higher(self):
        """Same strengths in a Tier-1 and a Tier-2 scan: the Tier-1 contributions are doubled."""
        strengths = [1.0, 2.0, 3.0]
        scans = {
            "vanna_extreme": pd.DataFrame({"symbol": ["V1", "V2", "V3"], "strength": strengths}),
            "volume_spike": pd.DataFrame({"symbol": ["S1", "S2", "S3"], "strength": strengths}),
        }
        result = build_composite_scores(scans).set_index("symbol")["composite_score"]
        assert result.index[:2].tolist() == ["V3", "V2"]
        assert result.index[-1] == "S1"
        assert result["V3"] == pytest.approx(2 * result["S3"])
        assert result["V3"] == pytest.approx(TIER_1_WEIGHT * 1.0)  # the strongest ranks 1
        assert result["V1"] == pytest.approx(TIER_1_WEIGHT / 3)

    def test_scores_strength_not_underlying_size(self):
        """A huge raw value with a weak strength ranks below a small but extreme symbol."""
        scans = {
            "volume_spike": pd.DataFrame(
                {
                    "symbol": ["HUGE", "MID", "SMALL"],
                    "signal_value": [5e7, 1e5, 300.0],
                    "strength": [2.0, 2.5, 6.0],
                }
            ),
        }
        result = build_composite_scores(scans)
        assert result["symbol"].tolist() == ["SMALL", "MID", "HUGE"]

    def test_confluence_adds_up(self):
        scans = {
            "vanna_extreme": pd.DataFrame({"symbol": ["A", "B", "C"], "strength": [0.5, 0.4, 0.3]}),
            "volume_spike": pd.DataFrame({"symbol": ["A", "B", "C"], "strength": [5.0, 3.0, 2.0]}),
        }
        result = build_composite_scores(scans).set_index("symbol")
        assert result.loc["A", "signal_count"] == 2
        assert result.loc["A", "composite_score"] == pytest.approx(TIER_1_WEIGHT + TIER_2_WEIGHT)
        assert result.loc["C", "composite_score"] == pytest.approx(
            (TIER_1_WEIGHT + TIER_2_WEIGHT) / 3
        )

    def test_agreement_outranks_a_single_strong_hit(self):
        """A symbol three scans flag with mediocre strengths is not pushed below one-scan hits."""
        scans = {
            "vanna_extreme": pd.DataFrame(
                {"symbol": ["A", "B", "C"], "strength": [0.45, 0.5, 0.5]}
            ),
            "flownet_extreme": pd.DataFrame(
                {"symbol": ["A", "D", "E"], "strength": [0.45, 0.5, 0.5]}
            ),
            "volume_spike": pd.DataFrame({"symbol": ["A", "F", "G"], "strength": [2.1, 5.0, 6.0]}),
            "gex_level": pd.DataFrame({"symbol": ["H"], "strength": [3.0]}),
        }
        result = build_composite_scores(scans)
        assert result.iloc[0]["symbol"] == "A"
        assert result.iloc[0]["signal_count"] == 3
        one_scan = result[result["signal_count"] == 1]
        assert (result.iloc[0]["composite_score"] >= one_scan["composite_score"]).all()
        assert (result["composite_score"] > 0).all()

    def test_weak_hits_can_score_below_one_strong_hit(self):
        """Agreement only adds: two bottom-ranked hits can still trail one top hit."""
        many = [f"S{i}" for i in range(9)]
        scans = {
            "dxoi_change": pd.DataFrame({"symbol": ["A", *many], "strength": range(10)}),
            "gex_level": pd.DataFrame({"symbol": ["A", *many], "strength": range(10)}),
            "volume_spike": pd.DataFrame({"symbol": ["B"], "strength": [2.1]}),
        }
        result = build_composite_scores(scans).set_index("symbol")
        assert result.loc["A", "composite_score"] == pytest.approx(0.2)
        assert result.loc["B", "composite_score"] == pytest.approx(TIER_2_WEIGHT)

    def test_single_symbol_scan_scores_its_full_weight(self):
        scans = {"gex_level": pd.DataFrame({"symbol": ["H"], "strength": [-7.0]})}
        result = build_composite_scores(scans)
        assert result.iloc[0]["composite_score"] == pytest.approx(TIER_2_WEIGHT)

    def test_ties_share_the_average_rank(self):
        scans = {
            "volume_spike": pd.DataFrame({"symbol": ["A", "B", "C"], "strength": [2.0, 2.0, 5.0]})
        }
        result = build_composite_scores(scans).set_index("symbol")["composite_score"]
        assert result["A"] == pytest.approx(0.5)
        assert result["B"] == pytest.approx(0.5)
        assert result["C"] == pytest.approx(1.0)

    def test_equal_scores_sort_by_signal_count(self):
        scans = {
            "volume_spike": pd.DataFrame({"symbol": ["ONE"], "strength": [5.0]}),
            "dxoi_change": pd.DataFrame({"symbol": ["TWO", "X"], "strength": [1.0, 2.0]}),
            "gex_level": pd.DataFrame({"symbol": ["TWO", "Y"], "strength": [1.0, 2.0]}),
        }
        result = build_composite_scores(scans)
        assert result["symbol"].tolist()[:2] == ["TWO", "ONE"]
        assert result["composite_score"].iloc[0] == result["composite_score"].iloc[1]

    def test_missing_strength_ranks_lowest_but_still_counts(self):
        scans = {
            "volume_spike": pd.DataFrame(
                {"symbol": ["NAN", "A", "B"], "strength": [float("nan"), 2.0, 3.0]}
            )
        }
        result = build_composite_scores(scans).set_index("symbol")["composite_score"]
        assert result.idxmin() == "NAN"
        assert result["NAN"] == pytest.approx(1 / 3)

    def test_empty_scans(self):
        scans = {
            "vanna_extreme": pd.DataFrame(),
            "volume_spike": pd.DataFrame(),
        }
        result = build_composite_scores(scans)
        assert result.empty

    def test_sorted_by_composite_score(self):
        scans = {
            "vanna_extreme": pd.DataFrame(
                {
                    "symbol": ["AAPL", "TSLA", "NVDA"],
                    "strength": [10.0, 50.0, 30.0],
                }
            ),
        }
        result = build_composite_scores(scans)
        assert result["symbol"].tolist() == ["TSLA", "NVDA", "AAPL"]

    def test_signals_column_lists_contributing_scans(self):
        scans = {
            "vanna_extreme": pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "strength": [50000.0],
                }
            ),
            "flownet_extreme": pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "strength": [-74600000.0],
                }
            ),
        }
        result = build_composite_scores(scans)
        assert result.iloc[0]["signals"] == "flownet_extreme, vanna_extreme"

    def test_rs_enrichment_preserved(self):
        scans = {
            "vanna_extreme": pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "strength": [50000.0],
                    "rs_rank_index": [0.85],
                    "rs_rank_overall": [0.80],
                    "rs_z_score": [1.2],
                    "etf_ticker": ["XLK"],
                }
            ),
        }
        result = build_composite_scores(scans)
        assert "rs_rank_index" in result.columns
        assert result.iloc[0]["rs_rank_index"] == 0.85

    def test_tier_weights_exist(self):
        assert "vanna_extreme" in TIER_WEIGHTS
        assert "flownet_extreme" in TIER_WEIGHTS
        assert TIER_WEIGHTS["vanna_extreme"] == 2.0
        assert TIER_WEIGHTS["volume_spike"] == 1.0


def test_rs_columns_come_from_every_enriched_scan():
    """The old merge used only the first enriched scan, leaving other symbols without RS."""
    rs = {"rs_rank_overall": [0.5], "rs_z_score": [0.1], "etf_ticker": ["X"], "etfs": ["X, Y"]}
    scans = {
        "vanna_extreme": pd.DataFrame(
            {"symbol": ["AAA"], "strength": [1.0], "rs_rank_index": [0.9], **rs}
        ),
        "volume_spike": pd.DataFrame(
            {"symbol": ["BBB"], "strength": [2.0], "rs_rank_index": [0.4], **rs}
        ),
    }
    result = build_composite_scores(scans).set_index("symbol")
    assert result.loc["AAA", "rs_rank_index"] == 0.9
    assert result.loc["BBB", "rs_rank_index"] == 0.4
    assert result.loc["BBB", "etfs"] == "X, Y"
