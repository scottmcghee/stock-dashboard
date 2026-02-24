"""Option Data — underlying metrics table + history graph."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import CACHE_TTL_DEFAULT, CACHE_TTL_SHORT
from queries.option_queries import (
    fetch_option_column_history,
    fetch_option_last_30_days,
    fetch_option_latest_comparison,
    get_all_option_symbols,
)

st.header("Option Data: Underlying Metrics")


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _get_symbols() -> list[str]:
    return get_all_option_symbols()


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _get_30d(symbol: str) -> pd.DataFrame:
    return fetch_option_last_30_days(symbol)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _get_comparison(symbol: str) -> dict:
    return fetch_option_latest_comparison(symbol)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _get_history(symbol: str, column: str):
    return fetch_option_column_history(symbol, column)


# --- Symbol selector ---
symbols = _get_symbols()
symbol = st.selectbox("Select Symbol", symbols, index=None, placeholder="Select option symbol...")

if not symbol:
    st.info("Select a symbol to view option data.")
    st.stop()

# --- Data table ---
df = _get_30d(symbol)
if df.empty:
    st.warning(f"No data found for {symbol}.")
    st.stop()

# Build ratio row
ratio_dict = _get_comparison(symbol)
if ratio_dict:
    ratio_row = {}
    for col in df.columns:
        if col == "collection_date":
            ratio_row[col] = "Latest/Avg Ratio"
        elif col == "symbol":
            ratio_row[col] = symbol
        elif col == "created_at":
            ratio_row[col] = ""
        elif col in ratio_dict:
            val = ratio_dict[col]
            ratio_row[col] = f"{val:.2f}" if pd.notnull(val) else ""
        else:
            ratio_row[col] = ""

    ratio_df = pd.DataFrame([ratio_row])
    display_df = pd.concat([ratio_df, df.astype(str)], ignore_index=True)
else:
    display_df = df

# Drop created_at from display
display_cols = [c for c in display_df.columns if c != "created_at"]
st.dataframe(display_df[display_cols], use_container_width=True, hide_index=True)

# --- Column selector for history chart ---
numeric_cols = [c for c in df.columns if c not in ("collection_date", "symbol", "created_at")]
selected_col = st.selectbox("Select metric to chart", numeric_cols, index=None, placeholder="Click a column to see history...")

if selected_col:
    df_hist, df_avg = _get_history(symbol, selected_col)
    if not df_hist.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["collection_date"], y=df_hist[selected_col],
            mode="lines", name=f"{selected_col} (actual)",
        ))
        if not df_avg.empty:
            avg_col = selected_col + "_avg"
            if avg_col in df_avg.columns:
                fig.add_trace(go.Scatter(
                    x=df_avg["date"], y=df_avg[avg_col],
                    mode="lines", name=f"{selected_col} (30d avg)",
                    line=dict(dash="dash", color="orange"),
                ))
        fig.update_layout(
            title=f"{symbol}: {selected_col} vs 30d Avg",
            yaxis_title=selected_col, xaxis_title="Date",
            legend=dict(orientation="h"), height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No history data for {selected_col}.")
