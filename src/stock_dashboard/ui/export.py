"""The TradingView watchlist download button used by every page that lists symbols."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from stock_dashboard.config import CACHE_TTL_SHORT
from stock_dashboard.logic.export import build_tradingview_watchlist
from stock_dashboard.queries.stocks_queries import fetch_exchange_map, fetch_known_symbols

EXPORT_LABEL = "Export TradingView Watchlist"


@st.cache_data(ttl=CACHE_TTL_SHORT, show_spinner=False)
def _watchlist(tickers: tuple[str, ...]) -> tuple[str, int]:
    """Watchlist text for the known tickers among ``tickers``, and how many were dropped."""
    known = fetch_known_symbols(tickers)
    exportable = [t for t in tickers if t in known]
    text = build_tradingview_watchlist(exportable, fetch_exchange_map(exportable))
    return text, len(tickers) - len(exportable)


def tradingview_export(
    tickers: Iterable[str], *, filename: str, key: str, label: str = EXPORT_LABEL
) -> None:
    """A download button for a TradingView watchlist of ``tickers``.

    Only symbols the stocks database knows (stocks, ETFs, the benchmark) are
    exported, each with its TradingView exchange prefix when known; duplicates
    keep their first position.
    """
    unique = tuple(dict.fromkeys(t for t in tickers if isinstance(t, str) and t))
    if not unique:
        return
    text, dropped = _watchlist(unique)
    if not text:
        st.caption("None of these symbols is a known stock or ETF, so there is nothing to export.")
        return
    st.download_button(label, text, file_name=filename, mime="text/plain", key=key)
    if dropped:
        st.caption(
            f"{dropped} symbol(s) unknown to the stocks database are left out of the export."
        )
