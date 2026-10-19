"""Tests for logic/option_views.py."""

import math

import pandas as pd

from stock_dashboard.logic.option_views import (
    default_expiration,
    latest_vs_average_row,
    nearest_strike,
    spot_on,
    strike_profile,
    underlying_formats,
)


def test_latest_vs_average_row():
    row = latest_vs_average_row(
        {"option_volume": 2.0, "volatility": 0.5},
        ["volatility", "option_volume", "put_volume"],
        {"option_volume", "volatility", "put_volume"},
    )
    assert list(row.columns) == ["volatility", "option_volume", "put_volume"]
    assert row.iloc[0]["option_volume"] == 2.0
    assert math.isnan(row.iloc[0]["put_volume"])
    assert len(row) == 1


def test_latest_vs_average_row_blanks_signed_columns():
    """-5000 against an average of -100 once showed as a 50x multiple."""
    row = latest_vs_average_row(
        {"flownet": 50.0, "dxoi": -5.0, "option_volume": 3.0},
        ["flownet", "dxoi", "option_volume"],
        {"option_volume"},
    )
    assert math.isnan(row.iloc[0]["flownet"])
    assert math.isnan(row.iloc[0]["dxoi"])
    assert row.iloc[0]["option_volume"] == 3.0


class TestStrikeProfile:
    def test_both_rights_aligned_on_sorted_strikes(self):
        chain = pd.DataFrame(
            {
                "strike": [110.0, 100.0, 100.0, 90.0],
                "option_type": ["call", "call", "put", "put"],
                "oi": [5, 10, 20, 7],
            }
        )
        profile = strike_profile(chain)
        assert profile.strikes == [90.0, 100.0, 110.0]
        assert profile.call_oi == [0, 10, 5]
        assert profile.put_oi == [7, 20, 0]

    def test_only_calls(self):
        chain = pd.DataFrame({"strike": [50.0], "option_type": ["call"], "oi": [3]})
        profile = strike_profile(chain)
        assert profile.put_oi == [0]
        assert profile.call_oi == [3]


class TestSpotOn:
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "close": [10.0, 11.0, None],
        }
    )

    def test_last_close_on_or_before_the_date(self):
        assert spot_on(self.bars, pd.Timestamp("2026-01-05")) == 11.0
        assert spot_on(self.bars, pd.Timestamp("2026-01-04")) == 10.0

    def test_latest_usable_close_without_a_date(self):
        assert spot_on(self.bars) == 11.0

    def test_none_before_history_or_without_bars(self):
        assert spot_on(self.bars, pd.Timestamp("2025-12-31")) is None
        assert spot_on(pd.DataFrame(columns=["date", "close"])) is None


EXPIRATIONS = [pd.Timestamp(d) for d in ("2026-10-10", "2026-10-17", "2026-10-23", "2026-11-20")]


class TestDefaultExpiration:
    as_of = pd.Timestamp("2026-10-16")

    def test_nearest_at_least_a_week_out(self):
        assert default_expiration(EXPIRATIONS, self.as_of) == pd.Timestamp("2026-10-23")

    def test_falls_back_to_the_furthest_future_then_the_last(self):
        near = EXPIRATIONS[:2]
        assert default_expiration(near, self.as_of) == pd.Timestamp("2026-10-17")
        assert default_expiration(near[:1], self.as_of) == pd.Timestamp("2026-10-10")
        assert default_expiration(EXPIRATIONS, None) == pd.Timestamp("2026-11-20")
        assert default_expiration([], self.as_of) is None


def test_nearest_strike():
    strikes = [90.0, 100.0, 110.0]
    assert nearest_strike(strikes, 104.0) == 100.0
    assert nearest_strike(strikes, 105.0) == 100.0  # ties go to the lower strike
    assert nearest_strike(strikes, None) == 100.0
    assert nearest_strike([], 100.0) is None


def test_underlying_formats_show_counts_and_exposures_as_whole_numbers():
    formats = underlying_formats(["option_volume", "flownet", "gxoi", "volatility", "extra"])
    assert formats == {
        "option_volume": "%,.0f",
        "flownet": "%,.0f",
        "gxoi": "%,.0f",
        "volatility": "%,.3f",
        "extra": "%,.3f",
    }
