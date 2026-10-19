"""Tests for logic/max_pain.py — pure compute."""

import numpy as np
import pandas as pd

from stock_dashboard.logic.max_pain import compute_max_pain_at, compute_max_pain_series


def _w(rows):
    return pd.DataFrame(rows, columns=["strike", "option_type", "weight"])


class TestComputeMaxPainAt:
    def test_single_call_dot_on_strike(self):
        # Only a call at K=100 with weight 100.
        # pain(S) = 100 * max(S-100, 0). Min at S=100 (or below). argmin picks first.
        weights = _w([(100, "call", 100)])
        result = compute_max_pain_at(weights, np.array([90, 100, 110]))
        assert result in (90.0, 100.0)  # both have zero pain; argmin takes lowest index

    def test_naked_call_gets_max_pain_below(self):
        weights = _w([(100, "call", 100)])
        result = compute_max_pain_at(weights, np.array([90, 100, 110]))
        assert result <= 100.0

    def test_balanced_call_and_put_pins_in_middle(self):
        weights = _w([(110, "call", 100), (90, "put", 100)])
        candidates = np.arange(85, 116, 1)
        result = compute_max_pain_at(weights, candidates)
        assert 90 <= result <= 110

    def test_canonical_example(self):
        # Textbook: OI of 1000 calls at 100, 500 puts at 95.
        weights = _w([(100, "call", 1000), (95, "put", 500)])
        candidates = np.array([90.0, 95.0, 100.0, 105.0])
        result = compute_max_pain_at(weights, candidates)
        assert result in (90.0, 95.0)

    def test_empty_weights_returns_nan(self):
        result = compute_max_pain_at(_w([]), np.array([100.0]))
        assert np.isnan(result)

    def test_empty_candidates_returns_nan(self):
        weights = _w([(100, "call", 100)])
        result = compute_max_pain_at(weights, np.array([]))
        assert np.isnan(result)

    def test_all_puts_no_calls(self):
        weights = _w([(100, "put", 500), (105, "put", 300)])
        result = compute_max_pain_at(weights, np.array([95.0, 100.0, 105.0, 110.0]))
        # Puts only → pain minimized above the highest put strike
        assert result >= 105.0


class TestComputeMaxPainSeries:
    def test_per_day_output(self):
        df = pd.DataFrame(
            {
                "collection_date": pd.to_datetime(["2026-04-01"] * 2 + ["2026-04-02"] * 2),
                "strike": [100, 105, 100, 105],
                "option_type": ["call", "put", "call", "put"],
                "weight": [500, 300, 400, 500],
            }
        )
        result = compute_max_pain_series(df)
        assert len(result) == 2
        assert list(result.columns) == ["collection_date", "max_pain"]
        assert result["collection_date"].is_monotonic_increasing

    def test_empty_returns_empty_frame(self):
        df = pd.DataFrame(columns=["collection_date", "strike", "option_type", "weight"])
        result = compute_max_pain_series(df)
        assert result.empty
        assert list(result.columns) == ["collection_date", "max_pain"]
