"""Preset picker, loader and regime source shared by the pages that load backtester presets."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from stock_dashboard.config import CACHE_TTL_DEFAULT
from stock_dashboard.logic.backtest import classify_regimes
from stock_dashboard.logic.preset import PRESET_ERRORS, list_presets, parse_preset, preset_label
from stock_dashboard.queries.stocks_queries import fetch_ohlcv

NO_PRESET = "(none)"


def preset_selectbox(label: str, *, key: str) -> str | None:
    """A selectbox over every preset (user and bundled samples); None for "(none)".

    Renders nothing and returns None when there are no presets.
    """
    names = list_presets()
    if not names:
        return None
    choice = st.selectbox(
        label,
        [NO_PRESET, *names],
        format_func=lambda stem: stem if stem == NO_PRESET else preset_label(stem),
        key=key,
    )
    return None if choice == NO_PRESET else choice


def load_preset(stem: str) -> dict[str, Any] | None:
    """:func:`parse_preset` for a chosen stem; shows ``st.error`` and returns None if unreadable.

    ``list_presets`` already skips unreadable files, so this catches a file
    that changed or broke after the list was built.
    """
    try:
        return parse_preset(stem)
    except PRESET_ERRORS as exc:
        st.error(f"Cannot read preset {preset_label(stem)!r}: {exc}")
        return None


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def benchmark_regimes(benchmark: str) -> pd.Series | None:
    """Market regime per date from the benchmark's full history, as the backtester classifies
    it; None when the benchmark has no bars."""
    bars = fetch_ohlcv("etf", [benchmark])
    return classify_regimes(bars) if not bars.empty else None
