"""Signal evaluation + composite scoring — pure functions, no DB or UI."""

import numpy as np
import pandas as pd


# Tier weights for composite scoring
TIER_WEIGHTS = {
    "vanna_extreme": 2.0,    # Tier 1
    "flownet_extreme": 2.0,  # Tier 1
    "volume_spike": 1.0,     # Tier 2
    "dxoi_change": 1.0,      # Tier 2
    "gex_level": 1.0,        # Tier 2
}


def compute_z_score(series: pd.Series) -> pd.Series:
    """Compute z-score for a numeric series."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def score_scan_results(df: pd.DataFrame, signal_col: str = "signal_value") -> pd.DataFrame:
    """Add z_score column to scan results based on signal_value."""
    if df.empty or signal_col not in df.columns:
        return df
    result = df.copy()
    result["z_score"] = compute_z_score(result[signal_col].astype(float))
    return result


def build_composite_scores(scan_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple scan results into a composite score per symbol.

    Args:
        scan_results: Dict mapping scan name -> DataFrame with 'symbol' and 'signal_value'.

    Returns:
        DataFrame with symbol, composite_score, contributing signals, and RS enrichment.
    """
    all_scores: list[pd.DataFrame] = []

    for scan_name, df in scan_results.items():
        if df.empty or "signal_value" not in df.columns:
            continue

        weight = TIER_WEIGHTS.get(scan_name, 1.0)
        scored = df[["symbol", "signal_value"]].copy()
        scored["z_score"] = compute_z_score(scored["signal_value"].astype(float))
        scored["weighted_score"] = scored["z_score"].abs() * weight
        scored["scan_name"] = scan_name
        all_scores.append(scored)

    if not all_scores:
        return pd.DataFrame(columns=["symbol", "composite_score", "signal_count", "signals"])

    combined = pd.concat(all_scores, ignore_index=True)

    # Aggregate per symbol
    composite = combined.groupby("symbol").agg(
        composite_score=("weighted_score", "sum"),
        signal_count=("scan_name", "count"),
        signals=("scan_name", lambda x: ", ".join(sorted(x))),
    ).reset_index()

    # Merge RS data from the first scan that has it
    rs_cols = ["symbol", "etf_ticker", "rs_rank_index", "rs_rank_overall", "rs_z_score"]
    for df in scan_results.values():
        if not df.empty and "rs_rank_index" in df.columns:
            rs_data = df[rs_cols].drop_duplicates(subset="symbol", keep="first")
            composite = composite.merge(rs_data, on="symbol", how="left")
            break

    return composite.sort_values("composite_score", ascending=False).reset_index(drop=True)
