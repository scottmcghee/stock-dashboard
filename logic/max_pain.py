"""Max pain computation — pure functions, no DB or UI dependencies.

Max pain = the strike S that minimizes total holder payout at expiration:

    pain(S) = sum over calls: oi_call(K) * max(S - K, 0)
            + sum over puts : oi_put(K)  * max(K - S, 0)

Uses raw OI as the weight (naive assumption that customers are long all OI).
"""

import numpy as np
import pandas as pd


def compute_max_pain_at(weights: pd.DataFrame, candidate_strikes: np.ndarray) -> float:
    """Single-snapshot max pain.

    `weights` must have columns: strike, option_type ('call'/'put'), weight.
    Returns the candidate strike that minimizes total pain.
    """
    if weights.empty or len(candidate_strikes) == 0:
        return float("nan")

    calls = weights[weights["option_type"] == "call"]
    puts = weights[weights["option_type"] == "put"]

    call_K = calls["strike"].to_numpy(dtype=float)
    call_w = calls["weight"].to_numpy(dtype=float)
    put_K = puts["strike"].to_numpy(dtype=float)
    put_w = puts["weight"].to_numpy(dtype=float)

    S = np.asarray(candidate_strikes, dtype=float)

    call_pain = np.clip(S[:, None] - call_K[None, :], 0, None) @ call_w if call_K.size else np.zeros_like(S)
    put_pain = np.clip(put_K[None, :] - S[:, None], 0, None) @ put_w if put_K.size else np.zeros_like(S)
    pain = call_pain + put_pain

    return float(S[int(np.argmin(pain))])


def compute_max_pain_series(weights_over_time: pd.DataFrame) -> pd.DataFrame:
    """Per-date max pain time series.

    `weights_over_time` has columns: collection_date, strike, option_type, weight.
    Returns DataFrame with columns: collection_date, max_pain.
    """
    if weights_over_time.empty:
        return pd.DataFrame(columns=["collection_date", "max_pain"])

    out = []
    for date, group in weights_over_time.groupby("collection_date", sort=True):
        candidates = np.sort(group["strike"].unique())
        mp = compute_max_pain_at(group, candidates)
        out.append({"collection_date": date, "max_pain": mp})
    return pd.DataFrame(out)
