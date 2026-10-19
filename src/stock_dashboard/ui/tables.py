"""``st.dataframe`` with readable column labels, number formats and blank missing values."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler

# Labels for the column names queries and logic return; anything not listed is
# humanized by :func:`column_label`.
COLUMN_LABELS: dict[str, str] = {
    "symbol": "Symbol",
    "ticker": "Ticker",
    "etf": "ETF",
    "etf_ticker": "ETF",
    "etf_name": "Name",
    "etfs": "All ETFs",
    "collection_date": "Date",
    "date": "Date",
    "relative_strength": "Relative Strength",
    "rs_rank_index": "RS Rank in ETF %",
    "rs_rank_overall": "RS Rank Overall %",
    "rs_z_score": "RS Z-Score",
    "signal_label": "Signal",
    "signal_value": "Signal Value",
    "strength": "Strength",
    "pct_rank": "Percentile %",
    "vol_ratio": "Volume / Avg",
    "pct_change": "Change %",
    "prev_dxoi": "Previous dxoi",
    "avg_20d_volume": "Avg Volume 20d",
    "rsi": "RSI",
    "mom_20": "Momentum 20d %",
    "ratio_50ma": "Close / 50d MA",
    "ratio_200ma": "Close / 200d MA",
    "bb_width": "Bollinger Width %",
    "adr_pct": "ADR %",
    "bb_width_chg": "Bollinger Width Change",
    "composite_score": "Composite Score",
    "signal_count": "Scans Hit",
    "signals": "Scans",
    "strategy": "Strategy",
    "regime": "Regime",
    "n_signals": "Signals",
    "current_value": "Latest",
    "avg_value": "30-Session Avg",
    "multiple": "Multiple",
    # underlying_data columns (docs/DATA_CONTRACT.md)
    "volatility": "IV (30d ATM)",
    "front_volatility": "Front IV",
    "back_volatility": "Back IV",
    "call_volume": "Call Volume",
    "put_volume": "Put Volume",
    "option_volume": "Option Volume",
    "put_call_ratio": "P/C Ratio",
    "flownet": "Flownet $",
    "vflownet": "Flownet (contracts)",
    "flowratio": "Flow Ratio $",
    "vflowratio": "Flow Ratio (contracts)",
    "volm_call_ratio": "Call Buy/Sell",
    "volm_put_ratio": "Put Buy/Sell",
    "deltas": "Customer Delta",
    "gammas": "Customer Gamma",
}

DATE_FORMAT = "YYYY-MM-DD"

# Exposure columns (dxoi, gxoi, vannaxoi, ...) keep their documented names.
_EXPOSURE = re.compile(r"[a-z]+xoi")


def column_label(name: str) -> str:
    """Readable header: a known label, an exposure column's own name, else humanized."""
    if name in COLUMN_LABELS:
        return COLUMN_LABELS[name]
    if _EXPOSURE.fullmatch(name):
        return name
    if "_" in name or name.islower():
        return name.replace("_", " ").strip().capitalize()
    return name


def column_config(
    columns: Iterable[str],
    formats: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    dates: Collection[str] = (),
) -> dict[str, Any]:
    """``column_config`` labeling every column; those in ``formats`` also get a number
    format (printf style or a Streamlit preset such as "localized"), and those in
    ``dates`` show as plain dates (no time of day)."""
    formats, labels = formats or {}, labels or {}
    config: dict[str, Any] = {}
    for name in columns:
        label = labels.get(name) or column_label(name)
        if name in dates:
            config[name] = st.column_config.DateColumn(label, format=DATE_FORMAT)
        elif name in formats:
            config[name] = st.column_config.NumberColumn(label, format=formats[name])
        else:
            config[name] = st.column_config.Column(label)
    return config


def show_table(
    df: pd.DataFrame | Styler,
    *,
    formats: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    dates: Collection[str] = (),
    **kwargs: Any,
) -> Any:
    """``st.dataframe`` with a readable label on every column and missing values blank.

    ``df`` may be a Styler: its formatting shows in the columns not in ``formats``.
    Other keyword arguments (``key``, ``on_select``, ...) go to ``st.dataframe``;
    the index is hidden unless ``hide_index=False`` is passed.
    """
    kwargs.setdefault("hide_index", True)
    return st.dataframe(
        df,
        column_config=column_config(df.columns, formats, labels, dates),
        placeholder="",
        **kwargs,
    )


def selection_key(prefix: str, row_ids: Iterable[object]) -> str:
    """A table key that changes whenever its rows (or their order) change.

    Streamlit keeps a keyed table's row selection across reruns even when the
    data changes, so a stale row number would point at another row, or past the
    end. Keying the table on its row ids resets the selection instead.
    """
    digest = hashlib.sha256("\x1f".join(map(str, row_ids)).encode()).hexdigest()[:16]
    return f"{prefix}_{digest}"


def selected_row(event: Any, n_rows: int) -> int | None:
    """The selected row of a single-row ``show_table`` event, if it is within ``n_rows``."""
    rows = event.selection.rows if event else []
    return rows[0] if rows and 0 <= rows[0] < n_rows else None
