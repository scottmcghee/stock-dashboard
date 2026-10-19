"""Surface rank: where one contract's IV sits on its expiration's smile.

The contract's IV minus the at-the-money IV is ranked against the same
quantity for every other strike of the same right and expiration. Since every
strike is compared with the same ATM IV, this ranks the contract's IV along
the expiration's smile: skew alone makes far out-of-the-money puts rank high.
It says where the contract sits on the smile, not that it is mispriced; the
option-contract page shows it as context, not a trade signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RICH_PERCENTILE = 0.80
VERY_RICH_PERCENTILE = 0.90
MIN_PEERS = 3


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
        return self.percentile_within_right >= RICH_PERCENTILE

    @property
    def is_very_rich(self) -> bool:
        """Top 10%."""
        return self.percentile_within_right >= VERY_RICH_PERCENTILE


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

    selected = chain[(chain["strike"] == strike) & (chain["option_type"] == option_type)]
    if selected.empty or pd.isna(selected.iloc[0]["iv"]):
        return None
    iv = float(selected.iloc[0]["iv"])

    same_right = chain[chain["option_type"] == option_type].dropna(subset=["iv"])
    if len(same_right) < MIN_PEERS:
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
        n_peers=len(same_right),
    )
