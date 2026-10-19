"""Option screener: underlyings whose latest metric is a multiple of its 30-session average."""

from __future__ import annotations

import streamlit as st

from stock_dashboard.config import CACHE_TTL_SHORT
from stock_dashboard.queries.option_queries import (
    fetch_option_screener_results,
    get_option_multiple_columns,
)
from stock_dashboard.ui.export import tradingview_export
from stock_dashboard.ui.option_guard import require_option_data
from stock_dashboard.ui.tables import column_label, show_table

DEFAULT_METRIC = "option_volume"
DEFAULT_MIN_MULTIPLE = 5.0

st.header("Option Screener")
st.caption(
    "Underlyings whose latest value of a metric is at least a multiple of its average over "
    "the 30 sessions before it. Only metrics that cannot be negative (volumes, implied "
    "volatilities, ratios) are offered, and only rows with a positive average count: a "
    "multiple of a signed average such as flownet or an exposure has no clear meaning."
)
require_option_data()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _columns() -> list[str]:
    return get_option_multiple_columns()


with st.form("option_screener"):
    col1, col2 = st.columns(2)
    with col1:
        columns = _columns()
        metric = st.selectbox(
            "Metric",
            columns,
            index=columns.index(DEFAULT_METRIC) if DEFAULT_METRIC in columns else None,
            format_func=lambda c: f"{column_label(c)} ({c})",
            placeholder="Select a metric...",
        )
    with col2:
        min_multiple = st.number_input(
            "Minimum multiple of average", min_value=0.0, value=DEFAULT_MIN_MULTIPLE, step=0.1
        )
    # Runs on every load with the last submitted values (the defaults at first).
    st.form_submit_button("Search", type="primary")

if not metric:
    st.stop()

with st.spinner("Searching..."):
    results = fetch_option_screener_results(metric, min_multiple)
if results.empty:
    st.warning("No underlyings meet the threshold.")
    st.stop()

show_table(
    results, formats={"current_value": "localized", "avg_value": "localized", "multiple": "%.2f"}
)
st.caption(f"{len(results)} results")
tradingview_export(results["symbol"], filename="option_screener_watchlist.txt", key="opt_export")
