"""52-Week Highs & Lows — breadth indicator and current lists."""

from datetime import datetime, timedelta

import plotly.graph_objects as go
import streamlit as st

from config import CACHE_TTL_LONG
from logic.chart_helpers import compute_52w_highs_lows
from queries.stocks_queries import fetch_52w_data, fetch_52w_etf_data

st.header("52-Week Highs & Lows")

# --- Controls ---
col1, col2 = st.columns(2)
with col1:
    display_start = st.date_input(
        "Display Start",
        value=(datetime.now() - timedelta(days=365)).date(),
        key="52w_start",
    )
with col2:
    include_etfs = st.checkbox("Include ETFs", value=True, key="52w_etfs")

# Lookback: 370 calendar days before display start for the 252 trading day window
lookback_start = (display_start - timedelta(days=370)).strftime("%Y-%m-%d")


@st.cache_data(ttl=CACHE_TTL_LONG)
def _load_data(lb_start: str, with_etfs: bool):
    df = fetch_52w_data(lb_start)
    if with_etfs:
        etf_df = fetch_52w_etf_data(lb_start)
        if not etf_df.empty:
            df = df if df.empty else df._append(etf_df, ignore_index=True)
    return df


with st.spinner("Loading price data..."):
    raw_df = _load_data(lookback_start, include_etfs)

if raw_df.empty:
    st.warning("No price data available.")
    st.stop()

daily_counts, highs_list, lows_list = compute_52w_highs_lows(
    raw_df, str(display_start)
)

if daily_counts.empty:
    st.warning("Not enough data for 52-week calculation in this range.")
    st.stop()

# --- Summary Metrics ---
latest = daily_counts.iloc[-1]
col_h, col_l, col_r = st.columns(3)
col_h.metric("New 52W Highs", int(latest["new_highs"]))
col_l.metric("New 52W Lows", int(latest["new_lows"]))
ratio = (
    latest["new_highs"] / latest["new_lows"]
    if latest["new_lows"] > 0
    else float("inf")
)
col_r.metric("High/Low Ratio", f"{ratio:.1f}" if ratio != float("inf") else "N/A")

# --- Time Series Chart ---
st.subheader("New 52W Highs vs Lows Over Time")
fig = go.Figure()
fig.add_trace(go.Bar(
    x=daily_counts["date"],
    y=daily_counts["new_highs"],
    name="New Highs",
    marker_color="#26a69a",
))
fig.add_trace(go.Bar(
    x=daily_counts["date"],
    y=-daily_counts["new_lows"],
    name="New Lows",
    marker_color="#ef5350",
))
fig.update_layout(
    barmode="relative",
    height=400,
    yaxis_title="Count (lows shown negative)",
    legend=dict(orientation="h"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# --- Net New Highs ---
st.subheader("Net New Highs (Highs - Lows)")
daily_counts["net"] = daily_counts["new_highs"] - daily_counts["new_lows"]
net_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in daily_counts["net"]]
net_fig = go.Figure(go.Bar(
    x=daily_counts["date"],
    y=daily_counts["net"],
    marker_color=net_colors,
))
net_fig.update_layout(
    height=300,
    yaxis_title="Net New Highs",
    hovermode="x unified",
)
st.plotly_chart(net_fig, use_container_width=True)

# --- Current Lists ---
st.subheader("Current 52W Highs & Lows")
tab_h, tab_l = st.tabs([
    f"52W Highs ({len(highs_list)})",
    f"52W Lows ({len(lows_list)})",
])
with tab_h:
    if highs_list.empty:
        st.info("No stocks at 52-week highs today.")
    else:
        st.dataframe(highs_list, use_container_width=True, hide_index=True)

        # TradingView export
        from logic.export import build_tradingview_watchlist
        from queries.stocks_queries import fetch_exchange_map

        _tickers = highs_list["ticker"].tolist()
        _exchange_map = fetch_exchange_map(_tickers)
        _wl = build_tradingview_watchlist(_tickers, _exchange_map)
        st.download_button(
            "Export Highs Watchlist", _wl, "52w_highs.txt", "text/plain"
        )

with tab_l:
    if lows_list.empty:
        st.info("No stocks at 52-week lows today.")
    else:
        st.dataframe(lows_list, use_container_width=True, hide_index=True)

        from logic.export import build_tradingview_watchlist
        from queries.stocks_queries import fetch_exchange_map

        _tickers_l = lows_list["ticker"].tolist()
        _exchange_map_l = fetch_exchange_map(_tickers_l)
        _wl_l = build_tradingview_watchlist(_tickers_l, _exchange_map_l)
        st.download_button(
            "Export Lows Watchlist", _wl_l, "52w_lows.txt", "text/plain"
        )
