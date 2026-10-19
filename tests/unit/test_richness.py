"""Tests for logic/richness.py."""

import pandas as pd

from stock_dashboard.logic.richness import compute_surface_richness


def _chain(rows):
    return pd.DataFrame(rows, columns=["strike", "option_type", "iv"])


class TestComputeSurfaceRichness:
    def test_picks_atm_as_nearest_to_spot(self):
        chain = _chain(
            [
                (95, "call", 0.30),
                (100, "call", 0.25),
                (105, "call", 0.28),
                (95, "put", 0.31),
                (100, "put", 0.26),
                (105, "put", 0.29),
            ]
        )
        # spot=101 → ATM is 100 (closest)
        r = compute_surface_richness(chain, spot=101.0, strike=105, option_type="call")
        assert r is not None
        assert r.atm_iv == 0.25
        assert r.iv == 0.28
        assert abs(r.iv_vs_atm - 0.03) < 1e-9

    def test_percentile_for_richest_strike(self):
        chain = _chain(
            [
                (95, "call", 0.20),
                (100, "call", 0.22),
                (105, "call", 0.25),
                (110, "call", 0.40),
            ]
        )
        r = compute_surface_richness(chain, spot=100.0, strike=110, option_type="call")
        assert r is not None
        assert r.percentile_within_right == 1.0  # richest
        assert r.is_rich is True
        assert r.is_very_rich is True

    def test_percentile_for_cheapest_strike(self):
        chain = _chain(
            [
                (95, "call", 0.20),
                (100, "call", 0.22),
                (105, "call", 0.25),
                (110, "call", 0.40),
            ]
        )
        r = compute_surface_richness(chain, spot=100.0, strike=95, option_type="call")
        assert r is not None
        assert r.percentile_within_right == 0.25  # 1 of 4 ≤ self
        assert r.is_rich is False

    def test_returns_none_when_selected_missing(self):
        chain = _chain([(100, "call", 0.25), (105, "call", 0.28)])
        r = compute_surface_richness(chain, spot=100.0, strike=999, option_type="call")
        assert r is None

    def test_returns_none_with_too_few_peers(self):
        chain = _chain([(100, "call", 0.25), (105, "call", 0.28)])
        r = compute_surface_richness(chain, spot=100.0, strike=100, option_type="call")
        # Only 2 same-right strikes → below min of 3
        assert r is None

    def test_returns_none_when_iv_missing(self):
        chain = _chain(
            [
                (95, "call", None),
                (100, "call", None),
                (105, "call", None),
            ]
        )
        r = compute_surface_richness(chain, spot=100.0, strike=100, option_type="call")
        assert r is None

    def test_calls_and_puts_ranked_independently(self):
        chain = _chain(
            [
                (95, "call", 0.20),
                (100, "call", 0.40),
                (105, "call", 0.22),
                (95, "put", 0.45),
                (100, "put", 0.30),
                (105, "put", 0.25),
            ]
        )
        # Call strike 100 is rich among calls.
        r = compute_surface_richness(chain, spot=100.0, strike=100, option_type="call")
        assert r is not None
        assert r.percentile_within_right == 1.0
        # Put strike 105 is cheap among puts.
        r = compute_surface_richness(chain, spot=100.0, strike=105, option_type="put")
        assert r is not None
        assert r.percentile_within_right < 0.5
