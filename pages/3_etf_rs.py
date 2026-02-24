"""ETF Relative Strength — grid + RS/Rank/Z-Score charts."""

from datetime import datetime, timedelta

import plotly.graph_objects as go
import streamlit as st

from config import CACHE_TTL_DEFAULT, CACHE_TTL_LONG
from queries.stocks_queries import fetch_etf_metrics, fetch_latest_etf_grid

st.header("ETF Relative Strength")


@st.cache_data(ttl=CACHE_TTL_LONG)
def _get_grid():
    return fetch_latest_etf_grid()


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _get_metrics(etf: str, start: str, end: str):
    return fetch_etf_metrics(etf, start, end)


# --- Date Range ---
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=(datetime.now() - timedelta(days=365)).date(), key="etf_start")
with col2:
    end_date = st.date_input("End Date", value=datetime.now().date(), key="etf_end")

# --- ETF Grid ---
grid_df = _get_grid()
if grid_df.empty:
    st.warning("No ETF metrics data available.")
    st.stop()

st.subheader("ETF Grid")
selected = st.dataframe(
    grid_df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

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
