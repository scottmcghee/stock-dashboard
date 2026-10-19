"""Composite scoring of the signal scans' results: pure functions, no DB or UI.

Every scan result carries ``strength``: a scale-free measure of how extreme the
symbol is (see :mod:`stock_dashboard.logic.option_signals` and
:mod:`stock_dashboard.logic.trend_signals`), larger = more extreme. The
composite turns it into a percentile rank within the symbols one scan flags,
in (0, 1], times the scan's weight, so every hit adds to a symbol's score and
a further scan agreeing on a symbol can only raise it. Several weak hits can
still score below one strong hit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Composite weights. Tier 1 (weight 2) holds the two scans that rank a symbol
# against its own history (percentile extremes); every other scan is Tier 2
# (weight 1). This is a weighting choice, not a fitted or validated parameter.
TIER_1_WEIGHT = 2.0
TIER_2_WEIGHT = 1.0
TIER_WEIGHTS = {
    "vanna_extreme": TIER_1_WEIGHT,
    "flownet_extreme": TIER_1_WEIGHT,
    "volume_spike": TIER_2_WEIGHT,
    "dxoi_change": TIER_2_WEIGHT,
    "gex_level": TIER_2_WEIGHT,
    "washout_bull": TIER_2_WEIGHT,
    "washout_bear": TIER_2_WEIGHT,
    "coiled_spring": TIER_2_WEIGHT,
}
RS_COLUMNS = ["symbol", "etf_ticker", "etfs", "rs_rank_index", "rs_rank_overall", "rs_z_score"]


STRENGTH_COLUMN = "strength"
SCORE_DECIMALS = 9


def strength_rank(strength: pd.Series) -> pd.Series:
    """Percentile rank of each strength among one scan's symbols, in (0, 1].

    Ties share their average rank and a single symbol ranks 1.0. A missing
    strength ranks lowest (tied with any other missing ones), so the hit still
    counts for something.
    """
    return strength.astype(float).fillna(-np.inf).rank(method="average", pct=True)


def build_composite_scores(scan_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple scan results into a composite score per symbol.

    Args:
        scan_results: Dict mapping scan name -> DataFrame with 'symbol' and 'strength'.

    Returns:
        DataFrame with symbol, composite_score (sum over scans of the scan's weight
        times the symbol's :func:`strength_rank`), signal_count, contributing
        signals, and RS enrichment; sorted by score, then signal count.
    """
    all_scores: list[pd.DataFrame] = []

    for scan_name, df in scan_results.items():
        if df.empty or STRENGTH_COLUMN not in df.columns:
            continue

        weight = TIER_WEIGHTS.get(scan_name, TIER_2_WEIGHT)
        scored = df[["symbol", STRENGTH_COLUMN]].copy()
        scored["weighted_score"] = strength_rank(scored[STRENGTH_COLUMN]) * weight
        scored["scan_name"] = scan_name
        all_scores.append(scored)

    if not all_scores:
        return pd.DataFrame(columns=["symbol", "composite_score", "signal_count", "signals"])

    combined = pd.concat(all_scores, ignore_index=True)

    # Aggregate per symbol
    composite = (
        combined.groupby("symbol")
        .agg(
            composite_score=("weighted_score", "sum"),
            signal_count=("scan_name", "count"),
            signals=("scan_name", lambda x: ", ".join(sorted(x))),
        )
        .reset_index()
    )

    # RS columns come from whichever scans were enriched; every scan row for a
    # symbol carries the same values, so the first one found is kept.
    rs_frames = [
        df[[c for c in RS_COLUMNS if c in df.columns]]
        for df in scan_results.values()
        if not df.empty and "rs_rank_index" in df.columns
    ]
    if rs_frames:
        rs_data = pd.concat(rs_frames, ignore_index=True).drop_duplicates(subset="symbol")
        composite = composite.merge(rs_data, on="symbol", how="left")

    # Rounded so that sums equal on paper tie exactly and the signal count decides.
    composite["composite_score"] = composite["composite_score"].round(SCORE_DECIMALS)
    return composite.sort_values(["composite_score", "signal_count"], ascending=False).reset_index(
        drop=True
    )
