"""ETF relative strength: latest ranks and RS impulse per ETF, charts for a selected ETF."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_dashboard.config import CACHE_TTL_DEFAULT, CACHE_TTL_LONG, get_settings
from stock_dashboard.logic.chart_helpers import (
    IMPULSE_WINDOWS,
    compute_etf_impulse,
    compute_rs_moving_averages,
    trim_to_range,
)
from stock_dashboard.queries.stocks_queries import (
    fetch_all_etf_rank_history,
    fetch_etf_metrics,
    fetch_latest_etf_grid,
)
from stock_dashboard.ui.charts import rs_figure
from stock_dashboard.ui.export import tradingview_export
from stock_dashboard.ui.tables import selected_row, selection_key, show_table

DEFAULT_RANGE_DAYS = 365
# Each ETF's impulse is z-scored against its own last year of sessions.
IMPULSE_HISTORY_BARS = 252
PERCENT = 100.0
CHART_HEIGHT = 350
BAR_HEIGHT_PER_ETF = 22


@st.cache_data(ttl=CACHE_TTL_LONG)
def _grid() -> pd.DataFrame:
    return fetch_latest_etf_grid()


@st.cache_data(ttl=CACHE_TTL_LONG)
def _impulse() -> pd.DataFrame:
    history = fetch_all_etf_rank_history()
    recent = history.groupby("etf_ticker").tail(IMPULSE_HISTORY_BARS)
    return compute_etf_impulse(recent)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _metrics(etf: str) -> pd.DataFrame:
    """Full history, so the moving averages are defined from the first displayed day."""
    return compute_rs_moving_averages(fetch_etf_metrics(etf))


benchmark = get_settings().benchmark
st.header("ETF Relative Strength")
st.caption(
    f"Each ETF's relative strength against {benchmark}, its rank among the ETFs (%), the "
    "change in that rank over N sessions (percentage points), and its ETF RS impulse: that "
    f"change z-scored against the ETF's own last {IMPULSE_HISTORY_BARS} sessions."
)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date.today() - timedelta(DEFAULT_RANGE_DAYS))
with col2:
    end_date = st.date_input("End Date", value=date.today())

grid = _grid()
if grid.empty:
    st.warning(f"No ETF metrics against {benchmark} found.")
    st.stop()

# Ranks and their changes are stored as fractions (0-1) and shown in percent /
# percentage points.
impulse = _impulse()
if not impulse.empty:
    impulse[[f"impulse_{w}d" for w in IMPULSE_WINDOWS]] *= PERCENT
    grid = grid.merge(impulse, on="etf_ticker", how="left")
grid["rs_rank_index"] = grid["rs_rank_index"] * PERCENT

formats = {"rs_rank_index": "%.1f", "rs_z_score": "%.2f"}
labels = {"rs_rank_index": "RS Rank %"}
for w in IMPULSE_WINDOWS:
    formats[f"impulse_{w}d"], labels[f"impulse_{w}d"] = "%+.1f", f"Rank Change {w}d (pts)"
    formats[f"impulse_{w}d_z"], labels[f"impulse_{w}d_z"] = "%+.2f", f"ETF RS Impulse {w}d (z)"

st.subheader("ETF Grid")
selection = show_table(
    grid,
    formats=formats,
    labels=labels,
    on_select="rerun",
    selection_mode="single-row",
    key=selection_key("etf_grid", grid["etf_ticker"]),
)
tradingview_export(grid["etf_ticker"], filename="etf_watchlist.txt", key="etf_export")

if not impulse.empty and st.checkbox("Show ETF RS Impulse Chart", value=True):
    st.subheader("ETF RS Impulse (z)")
    window = st.selectbox("Impulse Window", IMPULSE_WINDOWS, index=1, format_func=lambda w: f"{w}d")
    z_col, raw_col = f"impulse_{window}d_z", f"impulse_{window}d"
    bars = impulse[["etf_ticker", z_col, raw_col]].dropna().sort_values(z_col)
    fig = go.Figure(
        go.Bar(
            x=bars[z_col],
            y=bars["etf_ticker"],
            orientation="h",
            marker_color=["#ef5350" if v < 0 else "#26a69a" for v in bars[z_col]],
            customdata=bars[raw_col],
            hovertemplate="%{y}<br>ETF RS impulse (z): %{x:.2f}"
            "<br>Rank change: %{customdata:+.1f} pts<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"ETF RS Impulse (z), {window}d",
        xaxis_title="ETF RS impulse (z): unusual acceleration →",
        height=max(400, len(bars) * BAR_HEIGHT_PER_ETF),
        margin={"l": 80},
    )
    st.plotly_chart(fig)

etfs = grid["etf_ticker"].tolist()
# A clicked grid row preselects the ETF; the selectbox also works on its own.
row = selected_row(selection, len(etfs))
etf = st.selectbox(
    "ETF to chart",
    etfs,
    index=row if row is not None else 0,  # the grid is sorted by rank, best first
)
if etf is None:
    st.info("Select an ETF to view its charts.")
    st.stop()

name = grid.loc[grid["etf_ticker"] == etf, "etf_name"].iloc[0]
st.subheader(f"{etf} ({name})")

metrics = trim_to_range(_metrics(etf), start_date, end_date)
if metrics.empty:
    st.warning(f"No metrics for {etf} in this date range.")
    st.stop()

st.plotly_chart(rs_figure(metrics, f"{etf} Relative Strength vs {benchmark}"))
for column, title, scale in (
    ("rs_rank_index", "RS Rank %", PERCENT),
    ("rs_z_score", "RS Z-Score", 1.0),
):
    fig = go.Figure(go.Scatter(x=metrics["date"], y=metrics[column] * scale, name=title))
    fig.update_layout(title=f"{etf} {title}", yaxis_title=title, height=CHART_HEIGHT)
    st.plotly_chart(fig)
