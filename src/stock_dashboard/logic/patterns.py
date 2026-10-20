"""Chart pattern detection: pure functions, no DB or UI.

Detects: High Tight Flag (HTF), Bull Flag, Bear Flag, Cup-and-Handle, V-Bottom.
All functions take OHLCV DataFrames and return pattern info or None.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from stock_dashboard.logic.technical import split_by_ticker

PatternMatch = dict[str, Any]


def detect_high_tight_flag(
    ohlcv: pd.DataFrame,
    min_rally_pct: float = 100.0,
    rally_window: int = 60,
    max_pullback_pct: float = 25.0,
    flag_window: int = 15,
) -> PatternMatch | None:
    """Detect a High Tight Flag: massive rally followed by shallow consolidation.

    Criteria:
    - Price rallied >= min_rally_pct% within rally_window bars
    - Then pulled back <= max_pullback_pct% from the high within flag_window bars
    """
    if len(ohlcv) < rally_window + flag_window:
        return None

    close = ohlcv["close"].to_numpy()
    high = ohlcv["high"].to_numpy()

    # Look at the recent rally_window + flag_window bars
    rally_end = len(ohlcv) - flag_window
    rally_start = rally_end - rally_window

    if rally_start < 0:
        return None

    # Find the low in the rally window and the high after it
    rally_section = close[rally_start:rally_end]
    low_idx = np.argmin(rally_section)
    rally_low = rally_section[low_idx]

    if rally_low <= 0:
        return None

    # High must come after the low
    high_section = high[rally_start + low_idx : rally_end]
    if len(high_section) == 0:
        return None
    rally_high = np.max(high_section)
    rally_pct = (rally_high - rally_low) / rally_low * 100

    if rally_pct < min_rally_pct:
        return None

    # Check the flag: pullback from high within flag_window
    flag_section = close[rally_end:]
    flag_low = np.min(flag_section)
    pullback_pct = (rally_high - flag_low) / rally_high * 100

    if pullback_pct > max_pullback_pct:
        return None

    return {
        "pattern": "high_tight_flag",
        "rally_pct": round(rally_pct, 1),
        "pullback_pct": round(pullback_pct, 1),
    }


def detect_bull_flag(
    ohlcv: pd.DataFrame,
    min_pole_pct: float = 5.0,
    pole_window: int = 15,
    flag_window: int = 10,
    max_flag_retrace_pct: float = 50.0,
) -> PatternMatch | None:
    """Detect a Bull Flag: strong up-move (pole) followed by a shallow downward drift (flag).

    Criteria:
    - Pole: price rose >= min_pole_pct% within pole_window bars
    - Flag: price drifted down but retraced < max_flag_retrace_pct% of the pole
    - Flag close values should trend slightly downward (negative slope)
    """
    if len(ohlcv) < pole_window + flag_window:
        return None

    close = ohlcv["close"].to_numpy()

    flag_start = len(ohlcv) - flag_window
    pole_start = flag_start - pole_window

    pole_low = np.min(close[pole_start:flag_start])
    pole_high = np.max(close[pole_start:flag_start])

    if pole_low <= 0:
        return None

    pole_pct = (pole_high - pole_low) / pole_low * 100
    if pole_pct < min_pole_pct:
        return None

    # The pole high should be near the end of the pole (not at the start)
    pole_high_idx = np.argmax(close[pole_start:flag_start])
    if pole_high_idx < pole_window // 2:
        return None

    # Flag section
    flag_section = close[flag_start:]
    flag_low = np.min(flag_section)
    retrace = pole_high - flag_low
    pole_range = pole_high - pole_low

    if pole_range == 0:
        return None

    retrace_pct = retrace / pole_range * 100
    if retrace_pct > max_flag_retrace_pct:
        return None

    # Flag should drift slightly downward (negative slope)
    if len(flag_section) > 2:
        x = np.arange(len(flag_section))
        slope = np.polyfit(x, flag_section, 1)[0]
        if slope > 0:
            return None

    return {
        "pattern": "bull_flag",
        "pole_pct": round(pole_pct, 1),
        "retrace_pct": round(retrace_pct, 1),
    }


def detect_bear_flag(
    ohlcv: pd.DataFrame,
    min_pole_pct: float = 5.0,
    pole_window: int = 15,
    flag_window: int = 10,
    max_flag_retrace_pct: float = 50.0,
) -> PatternMatch | None:
    """Detect a Bear Flag: strong down-move (pole) followed by a shallow upward drift (flag).

    Mirror of bull flag with inverted direction.
    """
    if len(ohlcv) < pole_window + flag_window:
        return None

    close = ohlcv["close"].to_numpy()

    flag_start = len(ohlcv) - flag_window
    pole_start = flag_start - pole_window

    pole_high = np.max(close[pole_start:flag_start])
    pole_low = np.min(close[pole_start:flag_start])

    if pole_high <= 0:
        return None

    pole_pct = (pole_high - pole_low) / pole_high * 100
    if pole_pct < min_pole_pct:
        return None

    # The pole low should be near the end of the pole
    pole_low_idx = np.argmin(close[pole_start:flag_start])
    if pole_low_idx < pole_window // 2:
        return None

    # Flag section
    flag_section = close[flag_start:]
    flag_high = np.max(flag_section)
    retrace = flag_high - pole_low
    pole_range = pole_high - pole_low

    if pole_range == 0:
        return None

    retrace_pct = retrace / pole_range * 100
    if retrace_pct > max_flag_retrace_pct:
        return None

    # Flag should drift slightly upward (positive slope)
    if len(flag_section) > 2:
        x = np.arange(len(flag_section))
        slope = np.polyfit(x, flag_section, 1)[0]
        if slope < 0:
            return None

    return {
        "pattern": "bear_flag",
        "pole_pct": round(pole_pct, 1),
        "retrace_pct": round(retrace_pct, 1),
    }


def detect_cup_and_handle(
    ohlcv: pd.DataFrame,
    cup_window: int = 60,
    handle_window: int = 15,
    max_cup_depth_pct: float = 35.0,
    max_handle_depth_pct: float = 15.0,
) -> PatternMatch | None:
    """Detect a Cup-and-Handle: U-shaped recovery with a shallow handle pullback.

    Criteria:
    - Cup: left rim high, dip, recovery to near left rim within cup_window bars
    - Handle: shallow pullback within handle_window bars, < max_handle_depth_pct%
    """
    total = cup_window + handle_window
    if len(ohlcv) < total:
        return None

    close = ohlcv["close"].to_numpy()

    handle_start = len(ohlcv) - handle_window
    cup_start = handle_start - cup_window

    cup_section = close[cup_start:handle_start]

    # Left rim = max of first third, right rim = max of last third
    third = max(cup_window // 3, 1)
    left_rim = np.max(cup_section[:third])
    right_rim = np.max(cup_section[-third:])
    cup_bottom = (
        np.min(cup_section[third:-third]) if cup_window > 3 * third else np.min(cup_section)
    )

    if left_rim <= 0:
        return None

    # Cup depth
    cup_depth_pct = (left_rim - cup_bottom) / left_rim * 100
    if cup_depth_pct > max_cup_depth_pct or cup_depth_pct < 5.0:
        return None

    # Right rim should be within 10% of left rim
    rim_diff_pct = abs(right_rim - left_rim) / left_rim * 100
    if rim_diff_pct > 15.0:
        return None

    # Handle
    handle_section = close[handle_start:]
    handle_low = np.min(handle_section)

    if right_rim <= 0:
        return None

    handle_depth_pct = (right_rim - handle_low) / right_rim * 100
    if handle_depth_pct > max_handle_depth_pct:
        return None

    return {
        "pattern": "cup_and_handle",
        "cup_depth_pct": round(cup_depth_pct, 1),
        "handle_depth_pct": round(handle_depth_pct, 1),
    }


def detect_v_bottom(
    ohlcv: pd.DataFrame,
    window: int = 30,
    min_decline_pct: float = 10.0,
    min_recovery_pct: float = 70.0,
) -> PatternMatch | None:
    """Detect a V-Bottom: sharp decline followed by equally sharp recovery.

    Criteria:
    - Price declined >= min_decline_pct% in the first half of window
    - Recovered >= min_recovery_pct% of the decline in the second half
    """
    if len(ohlcv) < window:
        return None

    close = ohlcv["close"].to_numpy()[-window:]
    half = window // 2

    first_half = close[:half]
    second_half = close[half:]

    peak = np.max(first_half)
    trough_idx = np.argmin(close)
    trough = close[trough_idx]

    if peak <= 0:
        return None

    decline_pct = (peak - trough) / peak * 100
    if decline_pct < min_decline_pct:
        return None

    # Trough should be near the middle (not at the very start or end)
    if trough_idx < half // 2 or trough_idx > half + half // 2:
        return None

    # Recovery
    recovery_high = np.max(second_half)
    decline_amount = peak - trough
    if decline_amount == 0:
        return None

    recovery_pct = (recovery_high - trough) / decline_amount * 100
    if recovery_pct < min_recovery_pct:
        return None

    return {
        "pattern": "v_bottom",
        "decline_pct": round(decline_pct, 1),
        "recovery_pct": round(recovery_pct, 1),
    }


# Pattern windows: minimum bars needed for each detector
_PATTERN_WINDOWS = {
    "high_tight_flag": 75,  # 60 rally + 15 flag
    "bull_flag": 25,  # 15 pole + 10 flag
    "bear_flag": 25,  # 15 pole + 10 flag
    "cup_and_handle": 75,  # 60 cup + 15 handle
    "v_bottom": 30,  # single window
}

# Bars a pattern scan needs at the latest bar (the longest detector window).
PATTERN_REQUIRED_BARS = max(_PATTERN_WINDOWS.values())

# All pattern detectors in scanning order
PATTERN_DETECTORS: dict[str, Callable[[pd.DataFrame], PatternMatch | None]] = {
    "high_tight_flag": detect_high_tight_flag,
    "bull_flag": detect_bull_flag,
    "bear_flag": detect_bear_flag,
    "cup_and_handle": detect_cup_and_handle,
    "v_bottom": detect_v_bottom,
}


def scan_patterns(
    ohlcv_df: pd.DataFrame,
    tickers: list[str],
    patterns: list[str] | None = None,
) -> dict[str, list[PatternMatch]]:
    """Scan multiple tickers for chart patterns.

    Args:
        ohlcv_df: Bulk OHLCV with 'ticker' column.
        tickers: Tickers to scan.
        patterns: Subset of pattern names to scan (default: all).

    Returns:
        Dict of ticker -> list of detected pattern dicts.
    """
    if patterns is None:
        patterns = list(PATTERN_DETECTORS.keys())

    detectors = {k: v for k, v in PATTERN_DETECTORS.items() if k in patterns}
    results: dict[str, list[PatternMatch]] = {}

    by_ticker = split_by_ticker(ohlcv_df)
    for ticker in tickers:
        if ticker not in by_ticker:
            continue
        group = by_ticker[ticker].sort_values("date")
        detected = []
        for fn in detectors.values():
            result = fn(group)
            if result is not None:
                detected.append(result)
        if detected:
            results[ticker] = detected

    return results


def rolling_detect_patterns(ohlcv: pd.DataFrame) -> dict[str, np.ndarray]:
    """Run all pattern detectors at every bar of a single ticker's OHLCV.

    For each bar i, slices ohlcv[max(0, i+1-window):i+1] and calls the
    detector on that slice.  Returns dict mapping "has_<pattern>" to int8
    arrays of length len(ohlcv).

    Args:
        ohlcv: Single-ticker OHLCV sorted by date.

    Returns:
        Dict mapping column names (e.g. "has_bull_flag") to int8 arrays.
    """
    n = len(ohlcv)
    result: dict[str, np.ndarray] = {}

    for pattern_name, detector in PATTERN_DETECTORS.items():
        col_name = f"has_{pattern_name}"
        arr = np.zeros(n, dtype=np.int8)
        window = _PATTERN_WINDOWS[pattern_name]

        if n < window:
            result[col_name] = arr
            continue

        for i in range(window - 1, n):
            start = max(0, i + 1 - window)
            slc = ohlcv.iloc[start : i + 1]
            if detector(slc) is not None:
                arr[i] = 1

        result[col_name] = arr

    return result
