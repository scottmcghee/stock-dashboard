"""Tests for logic/scan_settings.py: slider values -> scan arguments, and rule text."""

import inspect
import re

import pandas as pd
import pytest

from stock_dashboard.logic.scan_settings import (
    COMPOSITE_RULE,
    OPTION_SCAN_TITLES,
    TREND_SCAN_TITLES,
    TREND_SCANS,
    CoiledSpringSettings,
    ScanSettings,
    WashoutSettings,
    scan_rule,
)
from stock_dashboard.logic.trend_signals import (
    scan_coiled_spring,
    scan_washout_bear,
    scan_washout_bull,
)


class TestWashoutSettings:
    def test_bull_kwargs(self):
        kwargs = WashoutSettings(rsi=35, momentum_pct=8, ma_distance_pct=6, min_bb_width=10)
        assert kwargs.bull_kwargs() == pytest.approx(
            {
                "rsi_max": 35,
                "mom_max": -8,
                "ratio_50ma_max": 0.94,
                "bb_width_min": 10,
                "vol_ratio_min": 1.0,
            }
        )

    def test_bear_mirrors_bull(self):
        kwargs = WashoutSettings(rsi=35, momentum_pct=8, ma_distance_pct=6, min_bb_width=10)
        assert kwargs.bear_kwargs() == pytest.approx(
            {"rsi_min": 65, "mom_min": 8, "ratio_50ma_min": 1.06, "bb_width_min": 10}
        )


@pytest.mark.parametrize(
    ("scan", "kwargs"),
    [
        (scan_washout_bull, lambda s: s.washout.bull_kwargs()),
        (scan_washout_bear, lambda s: s.washout.bear_kwargs()),
        (scan_coiled_spring, lambda s: s.coiled_spring.kwargs()),
    ],
)
def test_page_defaults_match_the_scan_function_defaults(scan, kwargs):
    """One source of truth: ScanSettings() equals every trend scan's own defaults."""
    params = inspect.signature(scan).parameters
    given = kwargs(ScanSettings())
    defaults = {name: p.default for name, p in params.items() if name != "indicators"}
    assert given == pytest.approx(defaults)


def test_trend_scan_registry_applies_the_settings():
    indicators = pd.DataFrame(
        {
            "symbol": ["BEAR"],
            "rsi": [70.0],
            "mom_20": [10.0],
            "ratio_50ma": [1.055],  # 5.5% above: below the 6% default, above 5%
            "ratio_200ma": [1.2],
            "bb_width": [15.0],
            "adr_pct": [3.0],
            "vol_ratio": [1.0],
        }
    )
    assert TREND_SCANS["washout_bear"](indicators, ScanSettings()).empty
    looser = ScanSettings(washout=WashoutSettings(ma_distance_pct=5.0))
    assert TREND_SCANS["washout_bear"](indicators, looser)["symbol"].tolist() == ["BEAR"]
    assert set(TREND_SCANS) == set(TREND_SCAN_TITLES)


def test_coiled_spring_kwargs():
    kwargs = CoiledSpringSettings(max_bb_width=6, max_adr_pct=1.0, ma_range_pct=10).kwargs()
    assert kwargs == pytest.approx(
        {"bb_width_max": 6, "adr_max": 1.0, "ratio_200ma_min": 0.9, "ratio_200ma_max": 1.1}
    )


def test_kwargs_are_accepted_by_the_scans():
    indicators = pd.DataFrame(
        {
            "symbol": ["A"],
            "rsi": [50.0],
            "mom_20": [0.0],
            "ratio_50ma": [1.0],
            "ratio_200ma": [1.0],
            "bb_width": [10.0],
            "adr_pct": [1.0],
            "vol_ratio": [1.0],
        }
    )
    s = ScanSettings()
    scan_washout_bull(indicators, **s.washout.bull_kwargs())
    scan_washout_bear(indicators, **s.washout.bear_kwargs())
    assert len(scan_coiled_spring(indicators, **s.coiled_spring.kwargs())) == 0


class TestScanRule:
    @pytest.mark.parametrize("name", [*OPTION_SCAN_TITLES, *TREND_SCAN_TITLES])
    def test_every_scan_has_a_rule_without_performance_claims(self, name):
        rule = scan_rule(name, ScanSettings())
        assert rule.endswith(".")
        assert not re.search(r"win rate|median|over \d+ days|bps", rule, re.IGNORECASE)

    def test_thresholds_are_filled_in(self):
        s = ScanSettings(vanna_percentile=15, washout=WashoutSettings(rsi=30))
        assert "15%" in scan_rule("vanna_extreme", s)
        assert "at or below 30" in scan_rule("washout_bull", s)
        assert "at or above 70" in scan_rule("washout_bear", s)
        assert "volume at or above its 20-day mean" in scan_rule("washout_bull", s)

    def test_flownet_rule_is_relative_bearish_flow_not_selling(self):
        rule = scan_rule("flownet_extreme", ScanSettings())
        assert "unusually bearish for this symbol" in rule
        assert "selling" not in rule

    def test_unknown_scan(self):
        with pytest.raises(KeyError):
            scan_rule("nope", ScanSettings())


def test_composite_rule_names_the_weights():
    assert "weighted 2" in COMPOSITE_RULE
    assert "not fitted" in COMPOSITE_RULE
