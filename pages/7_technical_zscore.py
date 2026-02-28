"""Technical Z-Score Composite — RSI, Momentum, Ratio-to-6MMA z-scores across the stock universe."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import CACHE_TTL_SHORT
from queries.stocks_queries import get_all_etfs

st.header("Technical Z-Score Composite")
st.caption(
    "Each indicator is z-scored across the selected universe. "
    "Composite = average of RSI, Momentum, and Ratio-to-MA z-scores. "
    "Click a row to see indicator history charts."
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

if not submitted and "zscore_df" not in st.session_state:
    st.stop()

# --- Fetch and compute (only on form submit) ---
if submitted:
    with st.spinner("Loading tickers..."):
        from queries.stocks_queries import search_stocks_by_metrics

        results = search_stocks_by_metrics(etf_ticker=etf_filter)
        if results.empty:
            st.warning("No stocks found in the selected universe.")
            st.stop()
        tickers = results["ticker"].unique().tolist()

    st.info(f"Computing indicators for {len(tickers)} stocks...")

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

    # Merge ETF info
    if "etf_ticker" in results.columns:
        etf_map = results.drop_duplicates("ticker").set_index("ticker")["etf_ticker"]
        zscore_df["etf"] = zscore_df["ticker"].map(etf_map)

    # Store in session state for row selection
    st.session_state["zscore_df"] = zscore_df
    st.session_state["ohlcv"] = ohlcv
    st.session_state["momentum_period"] = momentum_period
    st.session_state["ma_period"] = ma_period

# --- Retrieve from session state ---
zscore_df = st.session_state.get("zscore_df")
if zscore_df is None or zscore_df.empty:
    st.stop()

ohlcv = st.session_state.get("ohlcv", pd.DataFrame())
momentum_period = st.session_state.get("momentum_period", 63)
ma_period = st.session_state.get("ma_period", 126)

# --- Format display DataFrame ---
display_df = zscore_df.copy()
ratio_col_name = f"Close/{ma_period}MA"
display_df = display_df.rename(columns={
    "ticker": "Ticker",
    "etf": "ETF",
    "rsi": "RSI",
    "momentum": "Momentum %",
    "ratio_to_ma": ratio_col_name,
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
for col in [ratio_col_name]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
for col in ["RSI Z", "Mom Z", "Ratio Z", "Composite Z"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")

show_cols = ["Ticker"]
if "ETF" in display_df.columns:
    show_cols.append("ETF")
show_cols += ["RSI", "RSI Z", "Momentum %", "Mom Z", ratio_col_name, "Ratio Z", "Composite Z"]
show_cols = [c for c in show_cols if c in display_df.columns]

# --- Selectable dataframe ---
event = st.dataframe(
    display_df[show_cols],
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="zscore_table",
)

st.caption(f"{len(display_df)} stocks — click a row to see indicator charts")

# --- Charts for selected row ---
selected_rows = event.selection.rows if event and event.selection else []
if not selected_rows:
    st.stop()

selected_idx = selected_rows[0]
selected_ticker = zscore_df.iloc[selected_idx]["ticker"]

st.divider()
st.subheader(f"Indicator History: {selected_ticker}")

# Compute time-series indicators for the selected ticker
ticker_ohlcv = ohlcv[ohlcv["ticker"] == selected_ticker].sort_values("date")
if ticker_ohlcv.empty:
    st.warning(f"No OHLCV data for {selected_ticker}.")
    st.stop()

from logic.technical import compute_indicator_history

history = compute_indicator_history(
    ticker_ohlcv,
    rsi_period=14,
    momentum_period=momentum_period,
    ma_period=ma_period,
)

# --- RSI Chart ---
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(
    x=history["date"], y=history["rsi"],
    mode="lines", name="RSI",
    line=dict(color="royalblue", width=1.5),
))
# Overbought/oversold reference lines
fig_rsi.add_hline(y=70, line_dash="dash", line_color="gray", annotation_text="Overbought (70)")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="gray", annotation_text="Oversold (30)")
fig_rsi.update_layout(
    title=f"RSI (14) — {selected_ticker}",
    yaxis_title="RSI",
    xaxis_title="Date",
    height=300,
    margin=dict(t=40, b=30),
    yaxis=dict(range=[0, 100]),
)
st.plotly_chart(fig_rsi, use_container_width=True)

# --- Momentum Chart ---
mom_clean = history.dropna(subset=["momentum"])
fig_mom = go.Figure()
fig_mom.add_trace(go.Scatter(
    x=mom_clean["date"], y=mom_clean["momentum"],
    mode="lines", name="Momentum",
    line=dict(color="royalblue", width=1.5),
))
fig_mom.add_hline(y=0, line_dash="dash", line_color="gray")
fig_mom.update_layout(
    title=f"Momentum ({momentum_period}d ROC%) — {selected_ticker}",
    yaxis_title="Rate of Change %",
    xaxis_title="Date",
    height=300,
    margin=dict(t=40, b=30),
)
st.plotly_chart(fig_mom, use_container_width=True)

# --- Ratio to MA Chart ---
ratio_clean = history.dropna(subset=["ratio_to_ma"])
fig_ratio = go.Figure()
fig_ratio.add_trace(go.Scatter(
    x=ratio_clean["date"], y=ratio_clean["ratio_to_ma"],
    mode="lines", name=f"Close / {ma_period}MA",
    line=dict(color="royalblue", width=1.5),
))
fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="At MA")
# Standard deviation bands for context
ratio_vals = ratio_clean["ratio_to_ma"]
if len(ratio_vals) > 1:
    mean_r = ratio_vals.mean()
    std_r = ratio_vals.std()
    fig_ratio.add_hline(y=mean_r + std_r, line_dash="dot", line_color="rgba(255,0,0,0.4)",
                        annotation_text="+1σ")
    fig_ratio.add_hline(y=mean_r - std_r, line_dash="dot", line_color="rgba(255,0,0,0.4)",
                        annotation_text="-1σ")
    fig_ratio.add_hline(y=mean_r + 2 * std_r, line_dash="dot", line_color="rgba(255,0,0,0.2)",
                        annotation_text="+2σ")
    fig_ratio.add_hline(y=mean_r - 2 * std_r, line_dash="dot", line_color="rgba(255,0,0,0.2)",
                        annotation_text="-2σ")
fig_ratio.update_layout(
    title=f"Ratio to {ma_period}-day MA — {selected_ticker}",
    yaxis_title=f"Close / {ma_period}MA",
    xaxis_title="Date",
    height=300,
    margin=dict(t=40, b=30),
)
st.plotly_chart(fig_ratio, use_container_width=True)
