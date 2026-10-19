"""52-week highs and lows: daily breadth counts and the current lists."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_dashboard.config import CACHE_TTL_LONG
from stock_dashboard.logic.chart_helpers import FIFTY_TWO_WEEK_BARS, compute_52w_highs_lows
from stock_dashboard.queries.stocks_queries import OhlcvKind, fetch_ohlcv
from stock_dashboard.ui.export import tradingview_export
from stock_dashboard.ui.tables import show_table

DEFAULT_RANGE_DAYS = 365
UP_COLOR, DOWN_COLOR = "#26a69a", "#ef5350"


@st.cache_data(ttl=CACHE_TTL_LONG, show_spinner="Loading price data...")
def _highs_lows(start: date, with_etfs: bool) -> pd.DataFrame:
    """Daily highs and lows from ``start``, plus the 52-week window of sessions before it."""
    kinds: list[OhlcvKind] = ["stock", "etf"] if with_etfs else ["stock"]
    frames = [
        fetch_ohlcv(kind, start=start, warmup_bars=FIFTY_TWO_WEEK_BARS, columns=("high", "low"))
        for kind in kinds
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


st.header("52-Week Highs & Lows")
st.caption(
    f"A new 52-week high (low) is a session whose high (low) reaches the extreme of the "
    f"trailing {FIFTY_TWO_WEEK_BARS} sessions."
)

col1, col2 = st.columns(2)
with col1:
    display_start = st.date_input(
        "Display Start", value=date.today() - timedelta(DEFAULT_RANGE_DAYS), key="52w_start"
    )
with col2:
    include_etfs = st.checkbox("Include ETFs", value=True, key="52w_etfs")

if not isinstance(display_start, date):
    st.stop()
prices = _highs_lows(display_start, include_etfs)
if prices.empty:
    st.warning("No price data available.")
    st.stop()

daily, highs, lows = compute_52w_highs_lows(prices, display_start.isoformat())
if daily.empty:
    st.warning("Not enough history for 52-week extremes in this range.")
    st.stop()

latest = daily.iloc[-1]
col_h, col_l, col_r = st.columns(3)
col_h.metric("New 52W Highs", int(latest["new_highs"]))
col_l.metric("New 52W Lows", int(latest["new_lows"]))
col_r.metric(
    "High/Low Ratio",
    f"{latest['new_highs'] / latest['new_lows']:.1f}" if latest["new_lows"] > 0 else "n/a",
)

st.subheader("New 52W Highs vs Lows")
fig = go.Figure()
fig.add_trace(
    go.Bar(x=daily["date"], y=daily["new_highs"], name="New Highs", marker_color=UP_COLOR)
)
fig.add_trace(
    go.Bar(x=daily["date"], y=-daily["new_lows"], name="New Lows", marker_color=DOWN_COLOR)
)
fig.update_layout(
    barmode="relative",
    height=400,
    yaxis_title="Count (lows shown negative)",
    legend={"orientation": "h"},
    hovermode="x unified",
)
st.plotly_chart(fig)

st.subheader("Net New Highs (Highs - Lows)")
net = daily["new_highs"] - daily["new_lows"]
net_fig = go.Figure(
    go.Bar(x=daily["date"], y=net, marker_color=[UP_COLOR if v >= 0 else DOWN_COLOR for v in net])
)
net_fig.update_layout(height=300, yaxis_title="Net New Highs", hovermode="x unified")
st.plotly_chart(net_fig)

st.subheader(f"On the latest session ({pd.Timestamp(latest['date']).date()})")
tab_h, tab_l = st.tabs([f"52W Highs ({len(highs)})", f"52W Lows ({len(lows)})"])
for tab, listing, kind in ((tab_h, highs, "highs"), (tab_l, lows, "lows")):
    with tab:
        if listing.empty:
            st.info(f"No symbols at 52-week {kind} on the latest session.")
            continue
        show_table(listing)
        tradingview_export(
            listing["ticker"],
            filename=f"52w_{kind}.txt",
            key=f"export_52w_{kind}",
            label=f"Export {kind.capitalize()} Watchlist",
        )
