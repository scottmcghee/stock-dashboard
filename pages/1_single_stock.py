"""Single Stock Analysis — candlestick chart + RS metrics + 30-week MA."""

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import CACHE_TTL_DEFAULT, CACHE_TTL_SHORT
from logic.chart_helpers import (
    compute_rs_moving_averages,
    get_non_trading_days,
    resample_ohlcv,
)
from queries.stocks_queries import (
    fetch_stock_metrics,
    fetch_stock_ohlcv,
    get_all_tickers,
)

st.header("Single Stock Analysis")


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _get_tickers() -> list[str]:
    return get_all_tickers()


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _get_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    return fetch_stock_ohlcv(ticker, start, end)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _get_metrics(ticker: str, start: str, end: str) -> pd.DataFrame:
    return fetch_stock_metrics(ticker, start, end)


# --- Controls ---
col1, col2, col3, col4 = st.columns([3, 3, 3, 3])

tickers = _get_tickers()
with col1:
    ticker = st.selectbox("Stock Ticker", tickers, index=None, placeholder="Select a ticker...")
with col2:
    etf_filter = st.selectbox("ETF Filter", [None], index=0, placeholder="Select ETF (optional)")
with col3:
    default_start = (datetime.now() - timedelta(days=365)).date()
    start_date = st.date_input("Start Date", value=default_start)
with col4:
    end_date = st.date_input("End Date", value=datetime.now().date())

interval = st.radio("Interval", ["D", "W", "M"], horizontal=True, index=0)

if not ticker:
    st.info("Select a stock ticker to view charts.")
    st.stop()

# --- Load ETFs for this ticker and update filter ---
metrics_data = _get_metrics(ticker, str(start_date), str(end_date))
if not metrics_data.empty:
    etf_options = sorted(metrics_data["etf_ticker"].dropna().unique().tolist())
    if len(etf_options) > 1:
        etf_filter = st.selectbox(
            "ETF Filter",
            [None] + etf_options,
            format_func=lambda x: "All ETFs" if x is None else x,
            key="etf_filter_real",
        )
    elif len(etf_options) == 1:
        etf_filter = etf_options[0]

if etf_filter and not metrics_data.empty:
    metrics_data = metrics_data[metrics_data["etf_ticker"] == etf_filter]

# --- Price chart ---
stock_data = _get_ohlcv(ticker, str(start_date), str(end_date))
if stock_data.empty:
    st.warning("No price data found for this ticker/range.")
    st.stop()

stock_data = stock_data.set_index("date")
stock_data = resample_ohlcv(stock_data, interval)
stock_data = stock_data.reset_index()

volume_colors = [
    "green" if c >= o else "red"
    for o, c in zip(stock_data["open"], stock_data["close"])
]

fig1 = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=[0.7, 0.3],
)
fig1.add_trace(
    go.Candlestick(
        x=stock_data["date"], open=stock_data["open"],
        high=stock_data["high"], low=stock_data["low"],
        close=stock_data["close"], name="OHLC",
    ), row=1, col=1,
)
fig1.add_trace(
    go.Bar(
        x=stock_data["date"], y=stock_data["volume"],
        marker_color=volume_colors, name="Volume",
    ), row=2, col=1,
)

non_trading = get_non_trading_days(stock_data["date"])
fig1.update_layout(
    title=f"{ticker} Price and Volume",
    xaxis_rangeslider_visible=False,
    height=600,
    xaxis=dict(rangebreaks=[dict(values=non_trading)]),
)
st.plotly_chart(fig1, use_container_width=True)

# --- Metrics chart ---
if not metrics_data.empty:
    metrics_data = compute_rs_moving_averages(metrics_data)

    fig2 = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Relative Strength", "RS Ranks", "RS Z-Score"),
    )
    fig2.add_trace(go.Scatter(
        x=metrics_data["date"], y=metrics_data["relative_strength"],
        name="RS", line=dict(color="blue"),
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=metrics_data["date"], y=metrics_data["MA252"],
        name="252d MA", line=dict(color="red", dash="dash"),
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=metrics_data["date"], y=metrics_data["MA150"],
        name="150d MA", line=dict(color="green", dash="dash"),
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=metrics_data["date"], y=metrics_data["rs_rank_index"],
        name="RS Rank (Index)", line=dict(color="purple"),
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=metrics_data["date"], y=metrics_data["rs_rank_overall"],
        name="RS Rank (Overall)", line=dict(color="orange"),
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=metrics_data["date"], y=metrics_data["rs_z_score"],
        name="RS Z-Score", line=dict(color="brown"),
    ), row=3, col=1)

    non_trading_m = get_non_trading_days(metrics_data["date"])
    fig2.update_layout(
        height=800, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangebreaks=[dict(values=non_trading_m)]),
    )
    fig2.update_yaxes(title_text="Relative Strength", row=1, col=1)
    fig2.update_yaxes(title_text="Rank %", row=2, col=1)
    fig2.update_yaxes(title_text="Z-Score", row=3, col=1)
    st.plotly_chart(fig2, use_container_width=True)

    # --- RS with 30-week MA ---
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=metrics_data["date"], y=metrics_data["relative_strength"],
        mode="lines", name="Relative Strength",
    ))
    if len(metrics_data) >= 150:
        rs_ma = metrics_data["relative_strength"].rolling(window=150, min_periods=1).mean()
        fig3.add_trace(go.Scatter(
            x=metrics_data["date"], y=rs_ma,
            mode="lines", name="30-Week MA",
            line=dict(dash="dash", color="orange"),
        ))
    fig3.update_layout(
        title="Relative Strength (with 30-Week MA)",
        yaxis_title="Relative Strength", xaxis_title="Date",
        legend=dict(orientation="h"), height=350,
    )
    st.plotly_chart(fig3, use_container_width=True)
