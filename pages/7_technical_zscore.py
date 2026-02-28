"""Technical Z-Score Composite — RSI, Momentum, Ratio-to-6MMA z-scores across the stock universe."""

import pandas as pd
import streamlit as st

from config import CACHE_TTL_SHORT
from queries.stocks_queries import get_all_etfs

st.header("Technical Z-Score Composite")
st.caption(
    "Each indicator is z-scored across the selected universe. "
    "Composite = average of RSI, Momentum, and Ratio-to-6MMA z-scores."
)

# --- Controls ---
with st.form("zscore_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        @st.cache_data(ttl=CACHE_TTL_SHORT)
        def _get_etfs() -> list[str]:
            return get_all_etfs()

        etf_filter = st.selectbox(
            "ETF Universe",
            [None] + _get_etfs(),
            format_func=lambda x: "All Stocks" if x is None else x,
        )
    with col2:
        momentum_period = st.selectbox(
            "Momentum Period (days)",
            [10, 20, 63, 126, 252],
            index=2,
            format_func=lambda x: {10: "10d", 20: "20d (1mo)", 63: "63d (3mo)", 126: "126d (6mo)", 252: "252d (1yr)"}[x],
        )
    with col3:
        ma_period = st.selectbox(
            "MA Period (days)",
            [50, 100, 126, 200, 252],
            index=2,
            format_func=lambda x: f"{x}-day SMA",
        )
    submitted = st.form_submit_button("Compute", type="primary")

if not submitted:
    st.stop()

# --- Fetch tickers from the selected universe ---
with st.spinner("Loading tickers..."):
    from queries.stocks_queries import search_stocks_by_metrics

    results = search_stocks_by_metrics(etf_ticker=etf_filter)
    if results.empty:
        st.warning("No stocks found in the selected universe.")
        st.stop()
    tickers = results["ticker"].unique().tolist()

st.info(f"Computing indicators for {len(tickers)} stocks...")

# --- Fetch OHLCV and compute z-scores ---
with st.spinner("Fetching OHLCV and computing z-scores..."):
    from queries.screener_queries import fetch_bulk_ohlcv_for_ta
    from logic.technical import compute_technical_zscore_table

    ohlcv = fetch_bulk_ohlcv_for_ta(tickers, lookback_days=max(300, ma_period + 50))
    if ohlcv.empty:
        st.warning("No OHLCV data available.")
        st.stop()

    zscore_df = compute_technical_zscore_table(
        ohlcv, tickers,
        momentum_period=momentum_period,
        ma_period=ma_period,
    )

if zscore_df.empty:
    st.warning("Insufficient data to compute z-scores.")
    st.stop()

# --- Merge ETF info ---
if "etf_ticker" in results.columns:
    etf_map = results.drop_duplicates("ticker").set_index("ticker")["etf_ticker"]
    zscore_df["etf"] = zscore_df["ticker"].map(etf_map)

# --- Format & Display ---
display_df = zscore_df.copy()
display_df = display_df.rename(columns={
    "ticker": "Ticker",
    "etf": "ETF",
    "rsi": "RSI",
    "momentum": "Momentum %",
    "ratio_to_ma": f"Close/{ma_period}MA",
    "rsi_z": "RSI Z",
    "momentum_z": "Mom Z",
    "ratio_z": "Ratio Z",
    "composite_z": "Composite Z",
})

# Format values
for col in ["RSI"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
for col in ["Momentum %"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
for col in [f"Close/{ma_period}MA"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
for col in ["RSI Z", "Mom Z", "Ratio Z", "Composite Z"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")

# Select display columns
show_cols = ["Ticker"]
if "ETF" in display_df.columns:
    show_cols.append("ETF")
show_cols += ["RSI", "RSI Z", "Momentum %", "Mom Z", f"Close/{ma_period}MA", "Ratio Z", "Composite Z"]
show_cols = [c for c in show_cols if c in display_df.columns]

st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)
st.caption(f"{len(display_df)} stocks")
