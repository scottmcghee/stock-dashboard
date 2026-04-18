"""Option Contract — select a ticker + expiry, see OI by strike, drill into per-contract OHLC."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import CACHE_TTL_DEFAULT, CACHE_TTL_SHORT
from logic.chart_helpers import get_non_trading_days
from queries.option_contract_queries import (
    fetch_contract_ohlc_history,
    fetch_oi_by_strike,
    get_expirations,
    get_tickers_with_option_data,
)

st.header("Option Contract Explorer")


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _get_tickers() -> list[str]:
    return get_tickers_with_option_data()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _get_expirations(symbol: str) -> list[pd.Timestamp]:
    return get_expirations(symbol)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _get_oi(symbol: str, expiration_date: str) -> pd.DataFrame:
    return fetch_oi_by_strike(symbol, expiration_date)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _get_history(symbol: str, expiration_date: str, strike: float, option_type: str) -> pd.DataFrame:
    return fetch_contract_ohlc_history(symbol, expiration_date, strike, option_type)


# --- Ticker + expiration selectors ---
col1, col2 = st.columns(2)
with col1:
    tickers = _get_tickers()
    symbol = st.selectbox(
        "Underlying",
        tickers,
        index=None,
        placeholder="Select a ticker with option data...",
    )

if not symbol:
    st.info("Select an underlying ticker to continue.")
    st.stop()

with col2:
    expirations = _get_expirations(symbol)
    if not expirations:
        st.warning(f"No option data found for {symbol}.")
        st.stop()
    expiration = st.selectbox(
        "Expiration",
        options=expirations,
        index=None,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        placeholder="Select an expiration...",
    )

if expiration is None:
    st.info("Select an expiration to see the OI profile.")
    st.stop()

expiration_str = expiration.strftime("%Y-%m-%d")

# --- OI by strike chart ---
oi_df = _get_oi(symbol, expiration_str)
if oi_df.empty:
    st.warning(f"No OI data for {symbol} {expiration_str}.")
    st.stop()

calls = oi_df[oi_df["option_type"] == "call"].set_index("strike")["oi"]
puts = oi_df[oi_df["option_type"] == "put"].set_index("strike")["oi"]
strikes = sorted(set(calls.index) | set(puts.index))
call_vals = [int(calls.get(s, 0)) for s in strikes]
put_vals = [int(puts.get(s, 0)) for s in strikes]

oi_fig = go.Figure()
oi_fig.add_trace(go.Bar(
    y=strikes, x=[-v for v in put_vals], name="Puts", orientation="h",
    marker_color="#e15759",
    customdata=[[s, "put"] for s in strikes],
    hovertemplate="Strike %{y}<br>Put OI: %{customdata[0]}<extra></extra>",
    text=put_vals,
))
oi_fig.add_trace(go.Bar(
    y=strikes, x=call_vals, name="Calls", orientation="h",
    marker_color="#4e79a7",
    customdata=[[s, "call"] for s in strikes],
    hovertemplate="Strike %{y}<br>Call OI: %{x}<extra></extra>",
    text=call_vals,
))
oi_fig.update_layout(
    title=f"{symbol} {expiration_str} — OI by Strike (click a bar to select)",
    barmode="overlay",
    bargap=0.1,
    xaxis_title="Puts ← OI → Calls",
    yaxis_title="Strike",
    height=max(400, min(900, 20 * len(strikes))),
    legend=dict(orientation="h"),
    clickmode="event+select",
)

selection = st.plotly_chart(
    oi_fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
    key="oi_chart",
)

# --- Strike + right dropdowns (synced with chart click) ---
clicked_strike: float | None = None
clicked_type: str | None = None
points = (selection or {}).get("selection", {}).get("points", [])
if points:
    custom = points[0].get("customdata")
    if custom and len(custom) == 2:
        clicked_strike = float(custom[0])
        clicked_type = str(custom[1])

strike_options = strikes
type_options = ["call", "put"]

col3, col4 = st.columns(2)
with col3:
    strike_idx = strike_options.index(clicked_strike) if clicked_strike in strike_options else None
    strike = st.selectbox(
        "Strike",
        strike_options,
        index=strike_idx,
        placeholder="Pick a strike...",
    )
with col4:
    type_idx = type_options.index(clicked_type) if clicked_type in type_options else None
    option_type = st.selectbox(
        "Right",
        type_options,
        index=type_idx,
        placeholder="Call or put...",
    )

if strike is None or option_type is None:
    st.info("Pick a strike and right (or click a bar above) to see the contract's OHLC history.")
    st.stop()

# --- Candlestick chart for selected contract ---
hist = _get_history(symbol, expiration_str, float(strike), option_type)
if hist.empty:
    st.warning(f"No price history for {symbol} {expiration_str} {strike} {option_type}.")
    st.stop()

candle_fig = go.Figure()
candle_fig.add_trace(go.Candlestick(
    x=hist["collection_date"],
    open=hist["open"],
    high=hist["high"],
    low=hist["low"],
    close=hist["close"],
    name="OHLC",
))
non_trading = get_non_trading_days(hist["collection_date"])
candle_fig.update_layout(
    title=f"{symbol} {expiration_str} {strike:g} {option_type.upper()} — OHLC History",
    xaxis_title="Date",
    yaxis_title="Price",
    height=500,
    xaxis_rangeslider_visible=False,
    xaxis=dict(rangebreaks=[dict(values=non_trading)]),
)
st.plotly_chart(candle_fig, use_container_width=True)

with st.expander("OI & Volume history"):
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(
        x=hist["collection_date"], y=hist["oi"], mode="lines", name="Open Interest",
    ))
    vol_fig.add_trace(go.Bar(
        x=hist["collection_date"], y=hist["day_volume"], name="Daily Volume", yaxis="y2", opacity=0.5,
    ))
    vol_fig.update_layout(
        height=300,
        yaxis=dict(title="OI"),
        yaxis2=dict(title="Volume", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        xaxis=dict(rangebreaks=[dict(values=non_trading)]),
    )
    st.plotly_chart(vol_fig, use_container_width=True)
