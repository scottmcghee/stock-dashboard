"""Shaping option data for the option pages: latest-vs-average row, OI profile, spot."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd


def latest_vs_average_row(
    ratios: Mapping[str, float], columns: Sequence[str], multiple_columns: Collection[str]
) -> pd.DataFrame:
    """One numeric row: latest value / average for each of ``columns``.

    NaN when the ratio is unknown or the column is not in ``multiple_columns``
    (a multiple of a signed average can flip sign or explode near zero).
    """
    return pd.DataFrame(
        [{col: ratios.get(col, np.nan) if col in multiple_columns else np.nan for col in columns}],
        index=["latest / avg"],
    )


WHOLE_NUMBER_FORMAT = "%,.0f"
DECIMAL_FORMAT = "%,.3f"
# underlying_data columns that are counts, shares or dollars (docs/DATA_CONTRACT.md),
# shown as whole numbers; the *xoi exposures are matched by name.
WHOLE_NUMBER_COLUMNS = frozenset(
    {"call_volume", "put_volume", "option_volume", "flownet", "vflownet", "deltas", "gammas"}
)
_EXPOSURE = re.compile(r"[a-z]+xoi")


def underlying_formats(columns: Iterable[str]) -> dict[str, str]:
    """Display format per numeric ``underlying_data`` column: whole numbers for counts,
    flows and exposures, three decimals for IVs and ratios (and unknown columns)."""
    return {
        c: WHOLE_NUMBER_FORMAT
        if c in WHOLE_NUMBER_COLUMNS or _EXPOSURE.fullmatch(c)
        else DECIMAL_FORMAT
        for c in columns
    }


class StrikeProfile(NamedTuple):
    strikes: list[float]
    call_oi: list[int]
    put_oi: list[int]


def strike_profile(chain: pd.DataFrame) -> StrikeProfile:
    """Open interest by strike for calls and puts (0 where a right has no row).

    ``chain`` has strike, option_type ('call'/'put') and oi for one snapshot.
    """
    by_right = chain.pivot_table(
        index="strike", columns="option_type", values="oi", aggfunc="sum", fill_value=0
    )
    by_right = by_right.reindex(columns=["call", "put"], fill_value=0).sort_index()
    return StrikeProfile(
        strikes=[float(s) for s in by_right.index],
        call_oi=[int(v) for v in by_right["call"]],
        put_oi=[int(v) for v in by_right["put"]],
    )


def spot_on(bars: pd.DataFrame, as_of: pd.Timestamp | None = None) -> float | None:
    """The underlying's close on the last bar on or before ``as_of`` (latest bar if None)."""
    if bars.empty:
        return None
    usable = bars.dropna(subset=["close"])
    if as_of is not None:
        usable = usable[usable["date"] <= as_of]
    if usable.empty:
        return None
    return float(usable.sort_values("date")["close"].iloc[-1])


# The default expiration leaves at least this long to trade, so its chain is not near-empty.
MIN_DAYS_TO_EXPIRATION = 7


def default_expiration(
    expirations: Sequence[pd.Timestamp],
    as_of: pd.Timestamp | None,
    min_days: int = MIN_DAYS_TO_EXPIRATION,
) -> pd.Timestamp | None:
    """The nearest expiration at least ``min_days`` after ``as_of``.

    Falls back to the furthest expiration after ``as_of``, then to the last one listed.
    """
    if not expirations:
        return None
    ordered = sorted(expirations)
    if as_of is None:
        return ordered[-1]
    later = [e for e in ordered if e > as_of]
    far_enough = [e for e in later if e >= as_of + pd.Timedelta(days=min_days)]
    if far_enough:
        return far_enough[0]
    return later[-1] if later else ordered[-1]


def nearest_strike(strikes: Sequence[float], spot: float | None) -> float | None:
    """The strike closest to ``spot`` (the middle strike when spot is unknown)."""
    if not strikes:
        return None
    if spot is None:
        return sorted(strikes)[len(strikes) // 2]
    return min(strikes, key=lambda s: (abs(s - spot), s))
