"""Signal-scan thresholds as entered on the scans page, and a one-line rule per scan.

The page's sliders speak in user terms (e.g. "distance from the 50-day MA,
%"); these classes translate them into the keyword arguments of the trend
scans in :mod:`stock_dashboard.logic.trend_signals`. ``ScanSettings()`` holds
the defaults: the Signal Scans page starts from it and the screener's
pre-filter uses it, through :data:`TREND_SCANS`. Descriptions state each
scan's rule only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from stock_dashboard.logic.option_signals import (
    DEFAULT_DXOI_CHANGE_PCT,
    DEFAULT_FLOWNET_PERCENTILE,
    DEFAULT_VANNA_PERCENTILE,
    DEFAULT_VOLUME_MULTIPLE,
    PERCENTILE_WINDOW_SESSIONS,
    VOLUME_AVG_SESSIONS,
)
from stock_dashboard.logic.signal_scanner import TIER_1_WEIGHT, TIER_2_WEIGHT
from stock_dashboard.logic.technical import ADR_PERIOD, RSI_PERIOD
from stock_dashboard.logic.trend_signals import (
    BB_PERIOD,
    LONG_MA_PERIOD,
    MOMENTUM_PERIOD,
    SHORT_MA_PERIOD,
    VOLUME_AVG_PERIOD,
    WASHOUT_MIN_VOLUME_RATIO,
    scan_coiled_spring,
    scan_washout_bear,
    scan_washout_bull,
)

PERCENT = 100.0
RSI_SCALE = 100.0  # RSI runs 0-100, so the bear threshold mirrors the bull one


@dataclass(frozen=True)
class WashoutSettings:
    """Washout thresholds; the bear scan mirrors the bull scan around neutral."""

    rsi: float = 40.0  # bull: RSI <= rsi; bear: RSI >= 100 - rsi
    momentum_pct: float = 5.0  # bull: momentum <= -N%; bear: >= +N%
    ma_distance_pct: float = 6.0  # bull: N% below the 50-day MA; bear: N% above
    min_bb_width: float = 12.0  # Bollinger band width, % of the middle band

    def bull_kwargs(self) -> dict[str, float]:
        return {
            "rsi_max": self.rsi,
            "mom_max": -self.momentum_pct,
            "ratio_50ma_max": 1.0 - self.ma_distance_pct / PERCENT,
            "bb_width_min": self.min_bb_width,
            "vol_ratio_min": WASHOUT_MIN_VOLUME_RATIO,
        }

    def bear_kwargs(self) -> dict[str, float]:
        return {
            "rsi_min": RSI_SCALE - self.rsi,
            "mom_min": self.momentum_pct,
            "ratio_50ma_min": 1.0 + self.ma_distance_pct / PERCENT,
            "bb_width_min": self.min_bb_width,
        }


@dataclass(frozen=True)
class CoiledSpringSettings:
    max_bb_width: float = 7.0  # Bollinger band width, % of the middle band
    max_adr_pct: float = 1.5  # average daily range, % of close
    ma_range_pct: float = 15.0  # close within +/- N% of the 200-day MA

    def kwargs(self) -> dict[str, float]:
        return {
            "bb_width_max": self.max_bb_width,
            "adr_max": self.max_adr_pct,
            "ratio_200ma_min": 1.0 - self.ma_range_pct / PERCENT,
            "ratio_200ma_max": 1.0 + self.ma_range_pct / PERCENT,
        }


@dataclass(frozen=True)
class ScanSettings:
    vanna_percentile: float = DEFAULT_VANNA_PERCENTILE
    flownet_percentile: float = DEFAULT_FLOWNET_PERCENTILE
    volume_multiple: float = DEFAULT_VOLUME_MULTIPLE
    dxoi_change_pct: float = DEFAULT_DXOI_CHANGE_PCT
    washout: WashoutSettings = field(default_factory=WashoutSettings)
    coiled_spring: CoiledSpringSettings = field(default_factory=CoiledSpringSettings)


OPTION_SCAN_TITLES: dict[str, str] = {
    "vanna_extreme": "Vanna Extreme",
    "flownet_extreme": "Flownet Extreme",
    "volume_spike": "Volume Spike",
    "dxoi_change": "dxoi Change",
    "gex_level": "Negative GEX",
}
TREND_SCAN_TITLES: dict[str, str] = {
    "washout_bull": "Washout Bull",
    "washout_bear": "Washout Bear",
    "coiled_spring": "Coiled Spring",
}

# Each trend scan run with a ScanSettings; the single place a scan meets its thresholds.
TREND_SCANS: dict[str, Callable[[pd.DataFrame, ScanSettings], pd.DataFrame]] = {
    "washout_bull": lambda ind, s: scan_washout_bull(ind, **s.washout.bull_kwargs()),
    "washout_bear": lambda ind, s: scan_washout_bear(ind, **s.washout.bear_kwargs()),
    "coiled_spring": lambda ind, s: scan_coiled_spring(ind, **s.coiled_spring.kwargs()),
}


def scan_rule(name: str, s: ScanSettings) -> str:
    """One-sentence rule of scan ``name`` with the current thresholds filled in."""
    w, c = s.washout, s.coiled_spring
    window = f"the symbol's last {PERCENTILE_WINDOW_SESSIONS} sessions"
    rules = {
        "vanna_extreme": (
            f"Latest vanna exposure (vannaxoi) in the top or bottom {s.vanna_percentile:g}% "
            f"of {window}."
        ),
        "flownet_extreme": (
            f"Latest flownet (call minus put net customer premium) in the bottom "
            f"{s.flownet_percentile:g}% of {window}: customer flow unusually bearish for "
            "this symbol, read contrarian."
        ),
        "volume_spike": (
            f"Option volume at least {s.volume_multiple:g}x its mean over the previous "
            f"{VOLUME_AVG_SESSIONS} sessions."
        ),
        "dxoi_change": (
            f"Delta exposure (dxoi) changed by at least {s.dxoi_change_pct:g}% "
            "since the previous session."
        ),
        "gex_level": (
            "Gamma exposure (gxoi) below zero on the latest session: dealers are short "
            "gamma and hedge in the direction of the move."
        ),
        "washout_bull": (
            f"RSI({RSI_PERIOD}) at or below {w.rsi:g}, "
            f"{MOMENTUM_PERIOD}-day momentum at or below -{w.momentum_pct:g}%"
            f", close at least {w.ma_distance_pct:g}% below the "
            f"{SHORT_MA_PERIOD}-day MA, {BB_PERIOD}-day Bollinger width at least "
            f"{w.min_bb_width:g}% and volume at or above its {VOLUME_AVG_PERIOD}-day mean."
        ),
        "washout_bear": (
            f"RSI({RSI_PERIOD}) at or above {RSI_SCALE - w.rsi:g}, "
            f"{MOMENTUM_PERIOD}-day momentum at least +{w.momentum_pct:g}%, "
            f"close at least {w.ma_distance_pct:g}% above the "
            f"{SHORT_MA_PERIOD}-day MA and {BB_PERIOD}-day Bollinger width at least "
            f"{w.min_bb_width:g}%."
        ),
        "coiled_spring": (
            f"{BB_PERIOD}-day Bollinger width at most {c.max_bb_width:g}%, "
            f"{ADR_PERIOD}-day ADR at most {c.max_adr_pct:g}% "
            f"and close within ±{c.ma_range_pct:g}% of the "
            f"{LONG_MA_PERIOD}-day MA; direction-agnostic."
        ),
    }
    return rules[name]


COMPOSITE_RULE = (
    "Each scan scores how extreme a flagged symbol is on a scale-free strength (percent "
    "rank in its own history, volume multiple, dxoi change, or the trend scan's momentum or "
    "band width). Within each scan that strength becomes a percentile rank among the flagged "
    "symbols, from just above 0 to 1 (the strongest; a lone symbol scores 1). A symbol's "
    f"composite is the sum of its ranks, weighted {TIER_1_WEIGHT:g} for the two percentile "
    f"scans (vanna, flownet) and {TIER_2_WEIGHT:g} for the rest, so every hit adds and the "
    "Scans Hit column shows how many scans agree. Ties are sorted by Scans Hit. The weights "
    "are a choice, not fitted."
)
