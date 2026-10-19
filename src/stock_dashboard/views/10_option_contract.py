"""Option contract: OI by strike, max pain over time, one contract's history."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stock_dashboard.config import CACHE_TTL_DEFAULT, CACHE_TTL_SHORT
from stock_dashboard.logic.chart_helpers import get_non_trading_days
from stock_dashboard.logic.max_pain import compute_max_pain_series
from stock_dashboard.logic.option_views import (
    default_expiration,
    nearest_strike,
    spot_on,
    strike_profile,
)
from stock_dashboard.logic.richness import compute_surface_richness
from stock_dashboard.queries.option_contract_queries import (
    fetch_contract_ohlc_history,
    fetch_most_open_interest_symbol,
    fetch_oi_by_strike,
    fetch_oi_weights_over_time,
    get_expirations,
    get_latest_collection_date,
    get_tickers_with_option_data,
)
from stock_dashboard.queries.stocks_queries import fetch_symbol_ohlcv
from stock_dashboard.ui.option_guard import require_option_data

PUT_COLOR, CALL_COLOR = "#e15759", "#4e79a7"
OPTION_TYPES = ["call", "put"]
DEFAULT_OPTION_TYPE = "call"
MIN_CHART_HEIGHT, MAX_CHART_HEIGHT, PIXELS_PER_STRIKE = 400, 900, 20

st.header("Option Contract")
require_option_data()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _symbols() -> list[str]:
    return get_tickers_with_option_data()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _default_symbol() -> str | None:
    return fetch_most_open_interest_symbol()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _expirations(symbol: str) -> tuple[list[pd.Timestamp], pd.Timestamp | None]:
    """The symbol's expirations and its latest collection date."""
    return get_expirations(symbol), get_latest_collection_date(symbol)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _chain(symbol: str, expiration: date) -> pd.DataFrame:
    return fetch_oi_by_strike(symbol, expiration)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _contract_history(symbol: str, expiration: date, strike: float, right: str) -> pd.DataFrame:
    return fetch_contract_ohlc_history(symbol, expiration, strike, right)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _max_pain(symbol: str, expiration: date) -> pd.DataFrame:
    weights = fetch_oi_weights_over_time(symbol, expiration)
    return compute_max_pain_series(weights)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _underlying(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Daily bars of the underlying, stock or ETF."""
    return fetch_symbol_ohlcv(symbol, start=start, end=end)


col1, col2 = st.columns(2)
with col1:
    symbols = _symbols()
    default_symbol = _default_symbol()
    symbol = st.selectbox(
        "Underlying",
        symbols,
        index=symbols.index(default_symbol) if default_symbol in symbols else None,
        placeholder="Select an underlying...",
        help="Defaults to the underlying with the most open interest on the latest date.",
    )
if not symbol:
    st.info("Select an underlying to continue.")
    st.stop()
with col2:
    expirations, latest = _expirations(symbol)
    default_expiry = default_expiration(expirations, latest)
    expiration = st.selectbox(
        "Expiration",
        expirations,
        index=expirations.index(default_expiry) if default_expiry in expirations else None,
        help="Defaults to the nearest expiration at least a week after the latest data.",
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        placeholder="Select an expiration...",
    )
if expiration is None:
    st.info("Select an expiration to see its open interest by strike.")
    st.stop()

expiry = expiration.date()
chain = _chain(symbol, expiry)
if chain.empty:
    st.warning(f"No open interest for {symbol} {expiry}.")
    st.stop()
snapshot = pd.Timestamp(chain["collection_date"].iloc[0])
profile = strike_profile(chain)

# --- OI by strike (puts left, calls right) ---
oi_fig = go.Figure()
oi_fig.add_trace(
    go.Bar(
        y=profile.strikes,
        x=[-v for v in profile.put_oi],
        name="Puts",
        orientation="h",
        marker_color=PUT_COLOR,
        customdata=[[s, "put", oi] for s, oi in zip(profile.strikes, profile.put_oi, strict=True)],
        hovertemplate="Strike %{y}<br>Put OI: %{customdata[2]:,}<extra></extra>",
    )
)
oi_fig.add_trace(
    go.Bar(
        y=profile.strikes,
        x=profile.call_oi,
        name="Calls",
        orientation="h",
        marker_color=CALL_COLOR,
        customdata=[
            [s, "call", oi] for s, oi in zip(profile.strikes, profile.call_oi, strict=True)
        ],
        hovertemplate="Strike %{y}<br>Call OI: %{customdata[2]:,}<extra></extra>",
    )
)
chart_height = max(
    MIN_CHART_HEIGHT, min(MAX_CHART_HEIGHT, PIXELS_PER_STRIKE * len(profile.strikes))
)
oi_fig.update_layout(
    title=f"{symbol} {expiry} OI by strike ({snapshot.date()})",
    barmode="overlay",
    bargap=0.1,
    xaxis_title="Puts ← OI → Calls",
    yaxis_title="Strike",
    height=chart_height,
    legend={"orientation": "h"},
    clickmode="event+select",
)

# --- Max pain with the underlying's close ---
max_pain = _max_pain(symbol, expiry)
left, right = st.columns(2)
with left:
    selection = st.plotly_chart(oi_fig, on_select="rerun", selection_mode="points", key="oi_chart")
    st.caption("Click a bar to chart that contract below.")
with right:
    if max_pain.empty:
        st.info("Not enough open interest history to compute max pain.")
    else:
        dates = max_pain["collection_date"]
        spot = _underlying(symbol, dates.min().date(), dates.max().date())
        mp_fig = go.Figure(
            go.Scatter(
                x=dates,
                y=max_pain["max_pain"],
                mode="lines+markers",
                name="Max Pain",
                line={"color": PUT_COLOR, "width": 2},
            )
        )
        if not spot.empty:
            mp_fig.add_trace(
                go.Scatter(
                    x=spot["date"],
                    y=spot["close"],
                    mode="lines",
                    name=f"{symbol} close",
                    line={"color": CALL_COLOR, "width": 1, "dash": "dot"},
                )
            )
        mp_fig.update_layout(
            title="Max pain (open-interest weighted)",
            yaxis={"title": "Strike", "range": [min(profile.strikes), max(profile.strikes)]},
            height=chart_height,
            legend={"orientation": "h"},
            hovermode="x unified",
        )
        st.plotly_chart(mp_fig, key="mp_chart")
        st.caption(
            "Max pain: the strike at which the expiration's open interest would pay option "
            "holders the least, assuming customers are long all of it. It describes open "
            "interest, not a price target."
        )

# --- Strike and right, preselected by a clicked bar ---
clicked_strike: float | None = None
clicked_type: str | None = None
points = selection.selection.points if selection else []
if points and len(points[0].get("customdata") or []) == 3:
    clicked_strike, clicked_type = (
        float(points[0]["customdata"][0]),
        str(points[0]["customdata"][1]),
    )

spot_price = spot_on(_underlying(symbol, snapshot.date(), snapshot.date()), snapshot)
default_strike = (
    clicked_strike if clicked_strike is not None else nearest_strike(profile.strikes, spot_price)
)
col3, col4 = st.columns(2)
with col3:
    strike = st.selectbox(
        "Strike",
        profile.strikes,
        index=profile.strikes.index(default_strike) if default_strike in profile.strikes else None,
        help="Defaults to the strike nearest the underlying's close on the snapshot date.",
        format_func=lambda s: f"{s:g}",
        placeholder="Pick a strike...",
    )
with col4:
    option_type = st.selectbox(
        "Right",
        OPTION_TYPES,
        index=OPTION_TYPES.index(clicked_type or DEFAULT_OPTION_TYPE),
        placeholder="Call or put...",
    )
if strike is None or option_type is None:
    st.info("Pick a strike and right (or click a bar) to see the contract's history.")
    st.stop()

# --- Surface richness, against the underlying's close on the chain's snapshot date ---
if spot_price is None:
    st.caption(f"No {symbol} close on {snapshot.date()}, so the surface rank is not shown.")
else:
    richness = compute_surface_richness(chain, spot_price, float(strike), option_type)
    if richness is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Contract IV", f"{richness.iv:.3f}")
        c2.metric("ATM IV", f"{richness.atm_iv:.3f}")
        c3.metric("IV - ATM", f"{richness.iv_vs_atm:+.3f}")
        c4.metric(
            "Surface rank",
            f"{richness.percentile_within_right:.0%}",
            help=f"Among {richness.n_peers} {option_type} strikes of this expiration; "
            "100% = highest IV relative to at-the-money.",
        )
        st.caption(
            "Surface rank: where this contract's IV minus the at-the-money IV ranks among the "
            "expiration's strikes of the same right, i.e. its place on the smile. Skew alone "
            "makes far out-of-the-money puts rank high; a high rank is not a mispricing."
        )
        if richness.is_rich:
            level = "Very rich" if richness.is_very_rich else "Rich"
            st.info(
                f"**{level} on the smile**: IV relative to at-the-money ranks at "
                f"{richness.percentile_within_right:.0%} among the {richness.n_peers} "
                f"{option_type} strikes on this expiration (spot {spot_price:,.2f} on "
                f"{snapshot.date()})."
            )

# --- Contract candlesticks with OI and volume ---
history = _contract_history(symbol, expiry, float(strike), option_type)
if history.empty:
    st.warning(f"No price history for {symbol} {expiry} {strike:g} {option_type}.")
    st.stop()

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3],
    specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    subplot_titles=("", "Open Interest & Volume"),  # the price panel's y-axis names it
)
fig.add_trace(
    go.Candlestick(
        x=history["collection_date"],
        open=history["open"],
        high=history["high"],
        low=history["low"],
        close=history["close"],
        name="Price",
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=history["collection_date"],
        y=history["oi"],
        name="Open Interest",
        line={"color": CALL_COLOR},
    ),
    row=2,
    col=1,
    secondary_y=False,
)
fig.add_trace(
    go.Bar(
        x=history["collection_date"],
        y=history["day_volume"],
        name="Daily Volume",
        marker_color="#f28e2b",
        opacity=0.6,
    ),
    row=2,
    col=1,
    secondary_y=True,
)
fig.update_layout(
    title=f"{symbol} {expiry} {strike:g} {option_type.upper()}",
    height=700,
    xaxis_rangeslider_visible=False,
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    hovermode="x unified",
)
fig.update_xaxes(rangebreaks=[{"values": get_non_trading_days(history["collection_date"])}])
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="OI", row=2, col=1, secondary_y=False)
# Own round ticks rather than ones aligned to the OI axis's grid; no second grid.
fig.update_yaxes(
    title_text="Volume", row=2, col=1, secondary_y=True, tickmode="auto", showgrid=False
)
st.plotly_chart(fig)
