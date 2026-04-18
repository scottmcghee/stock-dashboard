"""Surface-richness helpers — how "rich" is one contract's IV vs its peers?

Backed by empirical finding (option_signal_analysis): contracts trading rich
vs the local vol surface underperform on a 5-day horizon with consistent
cross-ticker replication (|ρ| ~ 0.02-0.06 but same sign across AAPL/MSFT/
NVDA/TSLA). Intended as a "don't chase the rich strike" UX warning.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SurfaceRichness:
    iv: float
    atm_iv: float
    iv_vs_atm: float
    percentile_within_right: float  # 0..1; higher = richer
    n_peers: int

    @property
    def is_rich(self) -> bool:
        """Top 20% of same-right strikes on this expiration."""
        return self.percentile_within_right >= 0.80

    @property
    def is_very_rich(self) -> bool:
        """Top 10%."""
        return self.percentile_within_right >= 0.90


def compute_surface_richness(
    chain: pd.DataFrame, spot: float, strike: float, option_type: str
) -> SurfaceRichness | None:
    """Evaluate one contract's richness vs same-expiration peers.

    `chain` rows: strike, option_type, iv (volatility). Must already be
    filtered to one (symbol, expiration_date, collection_date).
    Returns None if the selected contract or ATM peer isn't available.
    """
    if chain.empty:
        return None
    needed = {"strike", "option_type", "iv"}
    if not needed.issubset(chain.columns):
        return None

    selected = chain[
        (chain["strike"] == strike) & (chain["option_type"] == option_type)
    ]
    if selected.empty or pd.isna(selected.iloc[0]["iv"]):
        return None
    iv = float(selected.iloc[0]["iv"])

    same_right = chain[chain["option_type"] == option_type].dropna(subset=["iv"])
    if len(same_right) < 3:
        return None

    # ATM = strike closest to spot (any right for reference).
    both_rights = chain.dropna(subset=["iv"])
    if both_rights.empty:
        return None
    atm_row = both_rights.iloc[(both_rights["strike"] - spot).abs().argsort()[:1]]
    atm_iv = float(atm_row.iloc[0]["iv"])

    iv_vs_atm = iv - atm_iv
    # Percentile within same right's iv_vs_atm distribution
    peer_vs_atm = same_right["iv"].astype(float) - atm_iv
    # Rank of our value among peers (inclusive of self)
    rank = (peer_vs_atm <= iv_vs_atm).sum()
    pct = rank / len(peer_vs_atm)
    return SurfaceRichness(
        iv=iv,
        atm_iv=atm_iv,
        iv_vs_atm=iv_vs_atm,
        percentile_within_right=float(pct),
        n_peers=int(len(same_right)),
    )
