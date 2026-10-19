"""Option-positioning scans over a per-symbol daily history: pure pandas, no DB or UI.

Input is one frame of ``underlying_data`` rows (symbol, collection_date and the
scanned columns) covering the last :data:`SIGNAL_HISTORY_SESSIONS` collection
dates. Each scan looks at the latest collection date only and returns one row
per flagged symbol with ``signal_value`` (the raw metric), ``strength`` and
``signal_label`` columns. ``strength`` is scale-free, so large and small
underlyings compare (the composite score ranks it); larger means more extreme.

The rules (all but negative GEX, a sign test, compare a symbol with its own recent history):

- Vanna extreme: ``vannaxoi`` in the top or bottom N percent of the window.
  Vanna exposure says how dealer delta hedges change when implied volatility
  moves, so an extreme reading marks unusual hedging sensitivity.
  Strength: distance of the percent rank from 0.5.
- Flownet extreme: ``flownet`` (call minus put net customer premium) in the
  bottom N percent of the window: customer flow unusually bearish for this
  symbol (calls sold or puts bought; the value itself can still be positive),
  read contrarian. Strength: distance of the percent rank from 0.5.
- Volume spike: option volume at least a multiple of its trailing average.
  Strength: that multiple.
- dxoi change: delta exposure moved at least N percent since the previous
  session. Strength: the absolute percent change.
- Negative GEX: ``gxoi`` (gamma exposure) below zero. Dealers short gamma hedge
  in the direction of the move, which tends to amplify volatility. Strength:
  1 minus the percent rank of gxoi in the window, i.e. how low it is for this
  symbol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The window each symbol is ranked against, in sessions (about three months).
PERCENTILE_WINDOW_SESSIONS = 63
MEDIAN_RANK = 0.5
VOLUME_AVG_SESSIONS = 20
SIGNAL_HISTORY_SESSIONS = max(PERCENTILE_WINDOW_SESSIONS, VOLUME_AVG_SESSIONS + 1)
SIGNAL_COLUMNS = ("vannaxoi", "flownet", "option_volume", "dxoi", "gxoi")

DEFAULT_VANNA_PERCENTILE = 20.0
DEFAULT_FLOWNET_PERCENTILE = 10.0
DEFAULT_VOLUME_MULTIPLE = 2.0
DEFAULT_DXOI_CHANGE_PCT = 20.0
VOLUME_SPIKE_EXTREME = 3.0
VOLUME_SPIKE_MAJOR = 2.0

# Display buckets for gxoi, in the feed's units; only the sign carries meaning
# for the scan, the positive cut-offs are presentation choices.
GEX_BUCKETS: tuple[tuple[float, str], ...] = (
    (0.0, "Negative GEX"),
    (5_000.0, "Low GEX"),
    (15_000.0, "Mid GEX"),
    (50_000.0, "High GEX"),
)
GEX_TOP_LABEL = "Very High GEX"


def gex_bucket_label(gxoi: float | None) -> str | None:
    """Display bucket for a gamma-exposure value (None when the value is missing)."""
    if gxoi is None or pd.isna(gxoi):
        return None
    for upper, label in GEX_BUCKETS:
        if gxoi < upper:
            return label
    return GEX_TOP_LABEL


def _window(history: pd.DataFrame, column: str, sessions: int) -> pd.DataFrame:
    """Non-null ``column`` rows from the last ``sessions`` collection dates."""
    rows = history.loc[history[column].notna(), ["symbol", "collection_date", column]]
    dates = np.sort(history["collection_date"].unique())[-sessions:]
    return rows[rows["collection_date"].isin(dates)]


def _latest_date(history: pd.DataFrame) -> pd.Timestamp | None:
    return None if history.empty else pd.Timestamp(history["collection_date"].max())


def latest_percent_rank(
    history: pd.DataFrame, column: str, sessions: int = PERCENTILE_WINDOW_SESSIONS
) -> pd.DataFrame:
    """Percent rank of each symbol's latest value within its own window.

    Matches SQL ``PERCENT_RANK``: (number of window values below the latest) / (n - 1),
    0 when the window holds a single value. Only symbols with a value on the
    latest collection date are returned: columns symbol, value, pct_rank.
    """
    latest = _latest_date(history)
    if latest is None:
        return pd.DataFrame(columns=["symbol", "value", "pct_rank"])
    window = _window(history, column, sessions)
    current = window.loc[window["collection_date"] == latest, ["symbol", column]]
    current = current.rename(columns={column: "value"})
    merged = window.merge(current, on="symbol")
    below = (merged[column] < merged["value"]).groupby(merged["symbol"]).sum()
    count = merged.groupby("symbol")[column].count()
    pct = (below / (count - 1)).where(count > 1, 0.0)
    out = current.set_index("symbol")
    out["pct_rank"] = pct.astype(float)
    return out.reset_index()


def classify_vanna_extreme(
    history: pd.DataFrame, percentile_threshold: float = DEFAULT_VANNA_PERCENTILE
) -> pd.DataFrame:
    """Symbols whose latest ``vannaxoi`` is in the top or bottom ``percentile_threshold`` %."""
    ranked = latest_percent_rank(history, "vannaxoi")
    cutoff = percentile_threshold / 100.0
    high = ranked["pct_rank"] >= 1 - cutoff
    low = ranked["pct_rank"] <= cutoff
    out = ranked[high | low].rename(columns={"value": "signal_value"})
    out["strength"] = (out["pct_rank"] - MEDIAN_RANK).abs()
    out["signal_label"] = np.where(high[high | low], "High Vanna", "Low Vanna")
    return out.sort_values("pct_rank", ascending=False).reset_index(drop=True)


def classify_flownet_extreme(
    history: pd.DataFrame, percentile_threshold: float = DEFAULT_FLOWNET_PERCENTILE
) -> pd.DataFrame:
    """Symbols whose latest ``flownet`` is in the bottom ``percentile_threshold`` %."""
    ranked = latest_percent_rank(history, "flownet")
    out = ranked[ranked["pct_rank"] <= percentile_threshold / 100.0]
    out = out.rename(columns={"value": "signal_value"})
    out["strength"] = (out["pct_rank"] - MEDIAN_RANK).abs()
    out["signal_label"] = "Unusually Bearish Flow"
    return out.sort_values("pct_rank").reset_index(drop=True)


def _volume_spike_label(ratio: float) -> str:
    if ratio >= VOLUME_SPIKE_EXTREME:
        return f"Extreme Spike (≥{VOLUME_SPIKE_EXTREME:g}x)"
    if ratio >= VOLUME_SPIKE_MAJOR:
        return f"Major Spike (≥{VOLUME_SPIKE_MAJOR:g}x)"
    return "Elevated (≥ threshold)"


def classify_volume_spike(
    history: pd.DataFrame, multiple_threshold: float = DEFAULT_VOLUME_MULTIPLE
) -> pd.DataFrame:
    """Symbols whose latest option volume is >= ``multiple_threshold`` x the prior-session mean.

    The mean covers the :data:`VOLUME_AVG_SESSIONS` sessions before the latest
    one; symbols without that much history are skipped.
    """
    columns = ["symbol", "signal_value", "vol_ratio", "avg_20d_volume", "strength", "signal_label"]
    latest = _latest_date(history)
    if latest is None:
        return pd.DataFrame(columns=columns)
    window = _window(history, "option_volume", VOLUME_AVG_SESSIONS + 1).sort_values(
        ["symbol", "collection_date"]
    )
    volume = window["option_volume"].astype(float)
    prior_mean = volume.groupby(window["symbol"]).transform(
        lambda s: s.shift(1).rolling(VOLUME_AVG_SESSIONS, min_periods=VOLUME_AVG_SESSIONS).mean()
    )
    frame = pd.DataFrame(
        {
            "symbol": window["symbol"],
            "collection_date": window["collection_date"],
            "signal_value": volume,
            "avg_20d_volume": prior_mean,
        }
    )
    frame = frame[(frame["collection_date"] == latest) & (frame["avg_20d_volume"] > 0)]
    frame = frame.assign(vol_ratio=frame["signal_value"] / frame["avg_20d_volume"])
    frame = frame[frame["vol_ratio"] >= multiple_threshold]
    frame = frame.assign(
        strength=frame["vol_ratio"], signal_label=frame["vol_ratio"].map(_volume_spike_label)
    )
    return frame[columns].sort_values("vol_ratio", ascending=False).reset_index(drop=True)


def classify_dxoi_change(
    history: pd.DataFrame, pct_threshold: float = DEFAULT_DXOI_CHANGE_PCT
) -> pd.DataFrame:
    """Symbols whose ``dxoi`` moved at least ``pct_threshold`` % since the previous session.

    The change is measured against ``abs(previous)`` so its sign is the
    direction of the move; both values must exist on the latest two sessions.
    """
    columns = ["symbol", "signal_value", "prev_dxoi", "pct_change", "strength", "signal_label"]
    dates = np.sort(history["collection_date"].unique())
    if len(dates) < 2:
        return pd.DataFrame(columns=columns)
    window = _window(history, "dxoi", 2)
    current = window[window["collection_date"] == dates[-1]].set_index("symbol")["dxoi"]
    previous = window[window["collection_date"] == dates[-2]].set_index("symbol")["dxoi"]
    frame = pd.DataFrame({"signal_value": current, "prev_dxoi": previous}).dropna()
    frame = frame[frame["prev_dxoi"] != 0]
    frame["pct_change"] = (frame["signal_value"] - frame["prev_dxoi"]) / frame["prev_dxoi"].abs()
    frame["pct_change"] *= 100
    frame = frame[frame["pct_change"].abs() >= pct_threshold]
    frame["strength"] = frame["pct_change"].abs()
    frame["signal_label"] = np.where(frame["pct_change"] > 0, "Big Increase", "Big Decrease")
    frame = frame.reset_index(names="symbol")
    order = frame["pct_change"].abs().sort_values(ascending=False).index
    return frame.loc[order, columns].reset_index(drop=True)


def classify_negative_gex(
    history: pd.DataFrame, sessions: int = PERCENTILE_WINDOW_SESSIONS
) -> pd.DataFrame:
    """Symbols whose latest ``gxoi`` is below zero, most negative first.

    ``pct_rank`` is the latest gxoi's percent rank in the symbol's last
    ``sessions`` sessions; ``strength`` = 1 - pct_rank.
    """
    columns = ["symbol", "signal_value", "pct_rank", "strength", "signal_label"]
    ranked = latest_percent_rank(history, "gxoi", sessions)
    out = ranked[ranked["value"] < 0].rename(columns={"value": "signal_value"})
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = out.assign(
        signal_value=out["signal_value"].astype(float),
        strength=1.0 - out["pct_rank"],
        signal_label=GEX_BUCKETS[0][1],
    )
    return out[columns].sort_values("signal_value").reset_index(drop=True)
