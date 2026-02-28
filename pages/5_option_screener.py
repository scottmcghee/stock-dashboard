"""Option Screener — find symbols where current value / 30d avg exceeds a threshold."""

import pandas as pd
import streamlit as st

from config import CACHE_TTL_LONG, CACHE_TTL_SHORT
from queries.option_queries import (
    fetch_option_screener_results,
    get_option_numerical_columns,
)

st.header("Option Data Screener")
st.caption("Find symbols where current values are significantly above their 30-day averages.")


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _get_columns() -> list[str]:
    return get_option_numerical_columns()


columns = _get_columns()

# --- Controls ---
col1, col2, col3 = st.columns([4, 4, 2])
with col1:
    metric = st.selectbox("Select Metric", columns, index=None, placeholder="Select metric to screen...")
with col2:
    min_multiple = st.number_input("Minimum Multiple", min_value=0.0, value=5.0, step=0.1, format="%.1f")
with col3:
    st.write("")  # spacer
    search = st.button("Search", type="primary")

if not search or not metric:
    st.stop()

# --- Results ---
with st.spinner("Searching..."):
    results = fetch_option_screener_results(metric, min_multiple)

if results.empty:
    st.warning("No results found.")
    st.stop()

# Format for display
display_df = results.copy()
display_df["current_value"] = display_df["current_value"].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
display_df["avg_value"] = display_df["avg_value"].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
display_df["multiple"] = display_df["multiple"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
display_df.columns = ["Symbol", "Current Value", "30-Day Avg", "Multiple"]

st.dataframe(display_df, use_container_width=True, hide_index=True)
st.caption(f"{len(display_df)} results")

# --- TradingView Watchlist Export ---
from logic.export import build_tradingview_watchlist
from queries.stocks_queries import fetch_exchange_map

_export_tickers = display_df["Symbol"].tolist()
_exchange_map = fetch_exchange_map(_export_tickers)
_watchlist = build_tradingview_watchlist(_export_tickers, _exchange_map)
st.download_button("Export TradingView Watchlist", _watchlist, "watchlist.txt", "text/plain")
