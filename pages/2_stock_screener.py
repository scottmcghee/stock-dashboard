"""Stock Screener — multi-criteria search with bulk Weinstein classification."""

import pandas as pd
import streamlit as st

from config import CACHE_TTL_LONG, CACHE_TTL_SHORT
from queries.stocks_queries import get_all_etfs, search_stocks_by_metrics

st.header("Stock Screener")
st.caption(
    "Volume Z-Score: shows stocks where current volume is X standard deviations above the 1-year average."
)


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _get_etfs() -> list[str]:
    return get_all_etfs()


# --- Filter Form ---
with st.form("screener_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        etfs = _get_etfs()
        etf_ticker = st.selectbox("ETF Filter", [None] + etfs, format_func=lambda x: "All ETFs" if x is None else x)
        rs_rank_min = st.number_input("RS Rank Min (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, format="%.1f")
        rs_rank_max = st.number_input("RS Rank Max (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, format="%.1f")
    with col2:
        rs_rank_overall_min = st.number_input("RS Rank Overall Min (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, format="%.1f")
        rs_rank_overall_max = st.number_input("RS Rank Overall Max (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, format="%.1f")
        rs_z_score_min = st.number_input("RS Z-Score Min", value=None, step=0.1, format="%.2f")
        rs_z_score_max = st.number_input("RS Z-Score Max", value=None, step=0.1, format="%.2f")
    with col3:
        volume_std_dev_min = st.number_input("Volume Z-Score Min", value=None, step=0.1, format="%.1f")
        avg_volume_min = st.number_input("Avg 30-Day Volume Min", value=None, step=10000.0, format="%.0f")
        weinstein_stage = st.selectbox(
            "Weinstein Stage",
            [None, "Stage 1", "Stage 2", "Stage 3", "Stage 4"],
            format_func=lambda x: "All Stages" if x is None else x,
        )

    submitted = st.form_submit_button("Search", type="primary")

if not submitted:
    st.stop()

# --- Run Search ---
with st.spinner("Searching..."):
    results = search_stocks_by_metrics(
        rs_rank_min=rs_rank_min,
        rs_rank_max=rs_rank_max,
        rs_rank_overall_min=rs_rank_overall_min,
        rs_rank_overall_max=rs_rank_overall_max,
        rs_z_score_min=rs_z_score_min,
        rs_z_score_max=rs_z_score_max,
        volume_std_dev_min=volume_std_dev_min,
        avg_volume_min=avg_volume_min,
        weinstein_stage=weinstein_stage,
        etf_ticker=etf_ticker,
    )

if results.empty:
    st.warning("No stocks found matching the criteria.")
    st.stop()

# --- Format & Display ---
display_df = results[["ticker", "etf_ticker", "rs_rank_index", "rs_rank_overall", "rs_z_score", "volume_z_score", "avg_30d_volume"]].copy()
display_df.columns = ["Ticker", "ETF", "RS Rank", "RS Rank Overall", "RS Z-Score", "Volume Z-Score", "Avg 30d Volume"]

# Format percentages
for col in ["RS Rank", "RS Rank Overall"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else "N/A")
for col in ["RS Z-Score", "Volume Z-Score"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
display_df["Avg 30d Volume"] = display_df["Avg 30d Volume"].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")

if "stage" in results.columns:
    display_df["Weinstein Stage"] = results["stage"].values

st.dataframe(display_df, use_container_width=True, hide_index=True)
st.caption(f"{len(display_df)} results")
