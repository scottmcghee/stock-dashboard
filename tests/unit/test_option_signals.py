"""Tests for logic/option_signals.py: the option-positioning scan rules."""

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.logic import option_signals as os_

DATES = pd.bdate_range("2025-01-01", periods=70)


def history(values: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    """Build history rows: {symbol: {column: values ending on the latest date}}."""
    frames = []
    for symbol, columns in values.items():
        n = max(len(v) for v in columns.values())
        frame = pd.DataFrame({"symbol": symbol, "collection_date": DATES[-n:]})
        for col in os_.SIGNAL_COLUMNS:
            series = columns.get(col, [np.nan] * n)
            frame[col] = [np.nan] * (n - len(series)) + list(series)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class TestLatestPercentRank:
    def test_matches_sql_percent_rank(self):
        # Latest value 3 among [1, 2, 3, 3, 5]: 2 values below, n = 5 -> 2 / 4.
        h = history({"A": {"vannaxoi": [1, 2, 3, 5, 3]}})
        ranked = os_.latest_percent_rank(h, "vannaxoi")
        assert ranked["pct_rank"].tolist() == [0.5]
        assert ranked["value"].tolist() == [3]

    def test_single_value_ranks_zero(self):
        ranked = os_.latest_percent_rank(history({"A": {"flownet": [7.0]}}), "flownet")
        assert ranked["pct_rank"].tolist() == [0.0]

    def test_window_limited_to_recent_sessions(self):
        values = [100.0] * 10 + [1.0] * 5 + [2.0]
        ranked = os_.latest_percent_rank(history({"A": {"vannaxoi": values}}), "vannaxoi", 6)
        assert ranked["pct_rank"].tolist() == [1.0]

    def test_symbol_without_latest_value_is_skipped(self):
        h = history({"A": {"vannaxoi": [1, 2, np.nan]}, "B": {"vannaxoi": [1, 2, 3]}})
        assert os_.latest_percent_rank(h, "vannaxoi")["symbol"].tolist() == ["B"]

    def test_empty_history(self):
        empty = history({"A": {"vannaxoi": [1.0]}}).iloc[0:0]
        assert os_.latest_percent_rank(empty, "vannaxoi").empty


class TestVannaAndFlownet:
    def test_vanna_high_and_low_tails(self):
        rising = list(range(1, 21))
        falling = list(range(20, 0, -1))
        middle = [*range(1, 21), 10]
        h = history(
            {"HI": {"vannaxoi": rising}, "LO": {"vannaxoi": falling}, "MID": {"vannaxoi": middle}}
        )
        out = os_.classify_vanna_extreme(h, 20.0)
        assert dict(zip(out["symbol"], out["signal_label"], strict=True)) == {
            "HI": "High Vanna",
            "LO": "Low Vanna",
        }
        assert out["symbol"].iloc[0] == "HI"  # sorted by pct_rank, high first
        assert out.set_index("symbol").loc["HI", "signal_value"] == 20

    def test_flownet_bottom_tail_only(self):
        h = history(
            {"SELL": {"flownet": list(range(20, 0, -1))}, "BUY": {"flownet": list(range(1, 21))}}
        )
        out = os_.classify_flownet_extreme(h, 10.0)
        assert out["symbol"].tolist() == ["SELL"]
        assert out["signal_label"].tolist() == ["Unusually Bearish Flow"]


class TestVolumeSpike:
    def test_ratio_against_prior_sessions(self):
        h = history(
            {"A": {"option_volume": [100.0] * 20 + [350.0]}, "B": {"option_volume": [100.0] * 21}}
        )
        out = os_.classify_volume_spike(h, 2.0)
        assert out["symbol"].tolist() == ["A"]
        assert out["vol_ratio"].iloc[0] == pytest.approx(3.5)
        assert out["avg_20d_volume"].iloc[0] == pytest.approx(100.0)
        assert out["signal_label"].iloc[0] == "Extreme Spike (≥3x)"

    @pytest.mark.parametrize(
        ("latest", "label"),
        [(250.0, "Major Spike (≥2x)"), (160.0, "Elevated (≥ threshold)")],
    )
    def test_labels(self, latest, label):
        h = history({"A": {"option_volume": [100.0] * 20 + [latest]}})
        assert os_.classify_volume_spike(h, 1.5)["signal_label"].tolist() == [label]

    def test_needs_full_average_window(self):
        h = history({"A": {"option_volume": [100.0] * 10 + [900.0]}})
        assert os_.classify_volume_spike(h).empty

    def test_empty(self):
        assert os_.classify_volume_spike(history({"A": {"gxoi": [1.0]}}).iloc[0:0]).empty


class TestDxoiChange:
    def test_increase_and_decrease(self):
        h = history(
            {
                "UP": {"dxoi": [100.0, 150.0]},
                "DOWN": {"dxoi": [-100.0, -180.0]},
                "FLAT": {"dxoi": [100.0, 105.0]},
            }
        )
        out = os_.classify_dxoi_change(h, 20.0).set_index("symbol")
        assert sorted(out.index) == ["DOWN", "UP"]
        assert out.loc["UP", "pct_change"] == pytest.approx(50.0)
        assert out.loc["UP", "signal_label"] == "Big Increase"
        # Change is relative to |previous|, so a more negative dxoi is a decrease.
        assert out.loc["DOWN", "pct_change"] == pytest.approx(-80.0)
        assert out.loc["DOWN", "signal_label"] == "Big Decrease"
        assert os_.classify_dxoi_change(h, 20.0)["symbol"].iloc[0] == "DOWN"  # largest |change|

    def test_requires_value_on_previous_session(self):
        h = history({"GAP": {"dxoi": [100.0, np.nan, 300.0]}, "ZERO": {"dxoi": [0.0, 50.0]}})
        assert os_.classify_dxoi_change(h).empty

    def test_needs_two_sessions(self):
        assert os_.classify_dxoi_change(history({"A": {"dxoi": [1.0]}})).empty


class TestNegativeGex:
    def test_only_negative_symbols(self):
        """The old GEX scan returned every symbol with a bucket label."""
        h = history(
            {
                "NEG": {"gxoi": [5.0, -10.0]},
                "POS": {"gxoi": [-5.0, 60_000.0]},
                "NEG2": {"gxoi": [-1.0]},
            }
        )
        out = os_.classify_negative_gex(h)
        assert out["symbol"].tolist() == ["NEG", "NEG2"]
        assert set(out["signal_label"]) == {"Negative GEX"}

    def test_empty(self):
        assert os_.classify_negative_gex(history({"A": {"gxoi": [1.0]}}).iloc[0:0]).empty


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (-1.0, "Negative GEX"),
        (0.0, "Low GEX"),
        (4_999.0, "Low GEX"),
        (5_000.0, "Mid GEX"),
        (15_000.0, "High GEX"),
        (50_000.0, "Very High GEX"),
        (None, None),
        (np.nan, None),
    ],
)
def test_gex_bucket_label(value, label):
    assert os_.gex_bucket_label(value) == label


def test_history_window_covers_every_scan():
    assert os_.SIGNAL_HISTORY_SESSIONS >= os_.PERCENTILE_WINDOW_SESSIONS
    assert os_.SIGNAL_HISTORY_SESSIONS > os_.VOLUME_AVG_SESSIONS


class TestStrength:
    """Every scan scores a scale-free strength, so a symbol's size doesn't decide its rank."""

    def test_vanna_and_flownet_distance_from_median_rank(self):
        # Same percent ranks at a thousand times the size: same strength.
        small = list(range(1, 22))
        h = history({"S": {"vannaxoi": small}, "BIG": {"vannaxoi": [v * 1000 for v in small]}})
        out = os_.classify_vanna_extreme(h, 20.0).set_index("symbol")
        assert out.loc["S", "strength"] == out.loc["BIG", "strength"] == pytest.approx(0.5)
        h = history({"F": {"flownet": list(range(20, -1, -1))}})
        assert os_.classify_flownet_extreme(h, 10.0)["strength"].iloc[0] == pytest.approx(0.5)

    def test_volume_spike_strength_is_the_multiple(self):
        h = history(
            {
                "SMALL": {"option_volume": [10.0] * 20 + [40.0]},
                "BIG": {"option_volume": [1e6] * 20 + [2.5e6]},
            }
        )
        out = os_.classify_volume_spike(h, 2.0).set_index("symbol")
        assert out.loc["SMALL", "strength"] == pytest.approx(4.0)
        assert out.loc["BIG", "strength"] == pytest.approx(2.5)

    def test_dxoi_strength_is_absolute_change(self):
        h = history({"DOWN": {"dxoi": [-100.0, -180.0]}})
        assert os_.classify_dxoi_change(h, 20.0)["strength"].iloc[0] == pytest.approx(80.0)

    def test_negative_gex_strength_is_low_rank_in_own_history(self):
        # BIG's gxoi is the most negative in absolute terms but typical for it.
        h = history(
            {
                "BIG": {"gxoi": [-9e6, -1e6, -5e6]},
                "SMALL": {"gxoi": [10.0, 5.0, -1.0]},
            }
        )
        out = os_.classify_negative_gex(h).set_index("symbol")
        assert out.loc["SMALL", "strength"] == pytest.approx(1.0)
        assert out.loc["BIG", "strength"] == pytest.approx(0.5)
