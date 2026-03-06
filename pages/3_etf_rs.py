"""ETF Relative Strength — grid + RS/Rank/Z-Score charts + RS Impulse."""

from datetime import datetime, timedelta

import plotly.graph_objects as go
import streamlit as st

from config import CACHE_TTL_DEFAULT, CACHE_TTL_LONG
from logic.chart_helpers import compute_etf_impulse
from queries.stocks_queries import (
    fetch_all_etf_rank_history,
    fetch_etf_metrics,
    fetch_latest_etf_grid,
)

st.header("ETF Relative Strength")


@st.cache_data(ttl=CACHE_TTL_LONG)
def _get_grid():
    return fetch_latest_etf_grid()


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _get_metrics(etf: str, start: str, end: str):
    return fetch_etf_metrics(etf, start, end)


@st.cache_data(ttl=CACHE_TTL_LONG)
def _get_impulse():
    """Fetch 1 year of rank history, compute z-scored impulse."""
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    history = fetch_all_etf_rank_history(start, end)
    return compute_etf_impulse(history)


# --- Date Range ---
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=(datetime.now() - timedelta(days=365)).date(), key="etf_start")
with col2:
    end_date = st.date_input("End Date", value=datetime.now().date(), key="etf_end")

# --- ETF Grid with Impulse ---
grid_df = _get_grid()
if grid_df.empty:
    st.warning("No ETF metrics data available.")
    st.stop()

impulse_df = _get_impulse()
if not impulse_df.empty:
    grid_df = grid_df.merge(impulse_df, on="etf_ticker", how="left")

st.subheader("ETF Grid")
selected = st.dataframe(
    grid_df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

# --- TradingView Watchlist Export ---
from logic.export import build_tradingview_watchlist
from queries.stocks_queries import fetch_exchange_map

_etf_tickers = grid_df["etf_ticker"].tolist()
_exchange_map = fetch_exchange_map(_etf_tickers)
_watchlist = build_tradingview_watchlist(_etf_tickers, _exchange_map)
st.download_button("Export TradingView Watchlist", _watchlist, "etf_watchlist.txt", "text/plain")

# --- RS Impulse Bar Chart ---
if not impulse_df.empty:
    st.subheader("RS Impulse (Normalized)")
    window_label = st.selectbox(
        "Impulse Window",
        options=["5d", "10d", "20d"],
        index=1,
        key="impulse_window",
    )
    z_col = f"impulse_{window_label}_z"
    raw_col = f"impulse_{window_label}"

    chart_df = impulse_df[["etf_ticker", z_col, raw_col]].dropna().copy()
    chart_df = chart_df.sort_values(z_col, ascending=True)

    colors = ["#ef5350" if v < 0 else "#26a69a" for v in chart_df[z_col]]

    impulse_fig = go.Figure(go.Bar(
        x=chart_df[z_col],
        y=chart_df["etf_ticker"],
        orientation="h",
        marker_color=colors,
        customdata=chart_df[raw_col],
        hovertemplate="%{y}<br>Z-Score: %{x:.2f}<br>Raw: %{customdata:.4f}<extra></extra>",
    ))
    impulse_fig.update_layout(
        title=f"RS Rank Impulse Z-Score ({window_label})",
        xaxis_title="Z-Score (unusual acceleration →)",
        yaxis_title="",
        height=max(400, len(chart_df) * 22),
        margin=dict(l=80),
    )
    st.plotly_chart(impulse_fig, use_container_width=True)

# --- Charts for selected ETF ---
selected_rows = selected.selection.rows if selected.selection else []
if not selected_rows:
    st.info("Select an ETF row above to view detailed charts.")
    st.stop()

row = grid_df.iloc[selected_rows[0]]
etf_ticker = row["etf_ticker"]
etf_name = row.get("etf_name", "")

st.subheader(f"{etf_ticker} ({etf_name})")

metrics = _get_metrics(etf_ticker, str(start_date), str(end_date))
if metrics.empty:
    st.warning(f"No metrics data for {etf_ticker}.")
    st.stop()

# RS chart with 30-week MA
rs_fig = go.Figure()
rs_fig.add_trace(go.Scatter(
    x=metrics["date"], y=metrics["relative_strength"],
    mode="lines", name="Relative Strength",
))
if len(metrics) >= 150:
    rs_ma = metrics["relative_strength"].rolling(window=150, min_periods=1).mean()
    rs_fig.add_trace(go.Scatter(
        x=metrics["date"], y=rs_ma,
        mode="lines", name="30-Week MA",
        line=dict(dash="dash", color="orange"),
    ))
rs_fig.update_layout(
    title=f"{etf_ticker} Relative Strength",
    yaxis_title="Relative Strength", height=350,
    legend=dict(orientation="h"),
)
st.plotly_chart(rs_fig, use_container_width=True)

# RS Rank chart
rank_fig = go.Figure()
rank_fig.add_trace(go.Scatter(
    x=metrics["date"], y=metrics["rs_rank_index"],
    mode="lines", name="RS Rank Index",
))
rank_fig.update_layout(
    title=f"{etf_ticker} RS Rank Index",
    yaxis_title="RS Rank Index", height=350,
    legend=dict(orientation="h"),
)
st.plotly_chart(rank_fig, use_container_width=True)

# Z-Score chart
z_fig = go.Figure()
z_fig.add_trace(go.Scatter(
    x=metrics["date"], y=metrics["rs_z_score"],
    mode="lines", name="RS Z-Score",
))
z_fig.update_layout(
    title=f"{etf_ticker} RS Z-Score",
    yaxis_title="RS Z-Score", height=350,
    legend=dict(orientation="h"),
)
st.plotly_chart(z_fig, use_container_width=True)
