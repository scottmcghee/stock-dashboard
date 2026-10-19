"""Technical z-scores: RSI, momentum and close/MA, z-scored across a stock or ETF universe."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_dashboard.config import CACHE_TTL_DEFAULT, CACHE_TTL_SHORT
from stock_dashboard.logic.chart_helpers import sigma_bands
from stock_dashboard.logic.technical import (
    INDICATOR_HISTORY_BARS,
    RSI_PERIOD,
    compute_indicator_history,
    compute_technical_zscore_table,
)
from stock_dashboard.queries.stocks_queries import (
    OhlcvKind,
    fetch_ohlcv,
    get_all_etfs,
    search_stocks_by_metrics,
)
from stock_dashboard.ui.export import tradingview_export
from stock_dashboard.ui.tables import selected_row, selection_key, show_table

MOMENTUM_PERIODS: dict[int, str] = {
    10: "10d",
    20: "20d (1mo)",
    63: "63d (3mo)",
    126: "126d (6mo)",
    252: "252d (1yr)",
}
MA_PERIODS = [50, 100, 126, 200, 252]
CHART_BARS: dict[int, str] = {63: "3 Months", 126: "6 Months", 252: "1 Year", 504: "2 Years"}
RSI_OVERBOUGHT, RSI_OVERSOLD = 70, 30
CHART_HEIGHT = 300


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _etfs() -> list[str]:
    return get_all_etfs()


@st.cache_data(ttl=CACHE_TTL_DEFAULT, show_spinner="Loading the universe...")
def _universe(kind: OhlcvKind, etf: str | None) -> pd.DataFrame:
    """Tickers to score (column ticker) with their ETF (column etf) for stocks."""
    if kind == "etf":
        return pd.DataFrame({"ticker": [etf] if etf else _etfs()})
    members = search_stocks_by_metrics(etf_ticker=etf)
    if members.empty:
        return pd.DataFrame(columns=["ticker", "etf"])
    return members[["ticker", "etf_ticker"]].rename(columns={"etf_ticker": "etf"})


@st.cache_data(ttl=CACHE_TTL_DEFAULT, show_spinner="Loading prices...")
def _bars(kind: OhlcvKind, tickers: tuple[str, ...], ma_period: int) -> pd.DataFrame:
    return fetch_ohlcv(kind, list(tickers), last_bars=max(INDICATOR_HISTORY_BARS, 2 * ma_period))


st.header("Technical Z-Scores")
st.caption(
    "RSI, momentum and close/MA ratio are each z-scored across the chosen universe; the "
    "composite is the mean of the three z-scores. Select a row for indicator history."
)

with st.form("zscore_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        etf = st.selectbox(
            "ETF universe", [None, *_etfs()], format_func=lambda x: "All" if x is None else x
        )
    with col2:
        momentum_period = st.selectbox(
            "Momentum period",
            list(MOMENTUM_PERIODS),
            index=2,
            format_func=MOMENTUM_PERIODS.__getitem__,
        )
    with col3:
        ma_period = st.selectbox(
            "MA period", MA_PERIODS, index=2, format_func=lambda x: f"{x}-day SMA"
        )
    etf_mode = st.checkbox("Score the ETFs themselves instead of their member stocks")
    # Computed on every load with the last submitted values (the defaults at first).
    st.form_submit_button("Compute", type="primary")

kind: OhlcvKind = "etf" if etf_mode else "stock"
universe = _universe(kind, etf)
tickers = universe["ticker"].tolist()
bars = _bars(kind, tuple(tickers), ma_period) if tickers else pd.DataFrame()
table = (
    compute_technical_zscore_table(
        bars, tickers, momentum_period=momentum_period, ma_period=ma_period
    )
    if not bars.empty
    else pd.DataFrame()
)
if table.empty:
    st.warning("Not enough price history to compute z-scores for this universe.")
    st.stop()
if "etf" in universe.columns:
    table.insert(1, "etf", table["ticker"].map(universe.set_index("ticker")["etf"]))

ratio_label = f"Close/{ma_period}MA"
labels = {
    "ticker": "Ticker",
    "etf": "ETF",
    "rsi": "RSI",
    "rsi_z": "RSI Z",
    "momentum": "Momentum %",
    "momentum_z": "Mom Z",
    "ratio_to_ma": ratio_label,
    "ratio_z": "Ratio Z",
    "composite_z": "Composite Z",
}
shown = table[[c for c in labels if c in table.columns]]
formats = {"rsi": "%.1f", "momentum": "%+.1f", "ratio_to_ma": "%.3f"}
formats.update(dict.fromkeys(["rsi_z", "momentum_z", "ratio_z", "composite_z"], "%+.2f"))
event = show_table(
    shown,
    formats=formats,
    labels=labels,
    on_select="rerun",
    selection_mode="single-row",
    key=selection_key("zscore_table", shown["ticker"]),
)
st.caption(f"{len(shown)} tickers")
tradingview_export(table["ticker"], filename="zscore_watchlist.txt", key="zscore_export")

row = selected_row(event, len(shown))
ticker = st.selectbox(
    "Ticker to chart",
    shown["ticker"].tolist(),
    index=row if row is not None else 0,  # the table is sorted by composite z-score
)

st.divider()
st.subheader(f"Indicator History: {ticker}")
chart_bars = st.selectbox(
    "Chart range", list(CHART_BARS), index=2, format_func=CHART_BARS.__getitem__
)

history = compute_indicator_history(
    bars[bars["ticker"] == ticker],
    rsi_period=RSI_PERIOD,
    momentum_period=momentum_period,
    ma_period=ma_period,
).tail(chart_bars)


def _line_chart(column: str, title: str, y_title: str) -> go.Figure:
    data = history.dropna(subset=[column])
    fig = go.Figure(go.Scatter(x=data["date"], y=data[column], name=title))
    fig.update_layout(
        title=f"{title}: {ticker}",
        yaxis_title=y_title,
        height=CHART_HEIGHT,
        margin={"t": 40, "b": 30},
    )
    return fig


rsi_fig = _line_chart("rsi", f"RSI ({RSI_PERIOD})", "RSI")
rsi_fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dash", line_color="gray", annotation_text="70")
rsi_fig.add_hline(y=RSI_OVERSOLD, line_dash="dash", line_color="gray", annotation_text="30")
rsi_fig.update_yaxes(range=[0, 100])
st.plotly_chart(rsi_fig)

mom_fig = _line_chart("momentum", f"Momentum ({momentum_period}-day % change)", "% change")
mom_fig.add_hline(y=0, line_dash="dash", line_color="gray")
st.plotly_chart(mom_fig)

ratio_fig = _line_chart("ratio_to_ma", f"Close / {ma_period}-day MA", ratio_label)
ratio_fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="At MA")
for label, level in sigma_bands(history["ratio_to_ma"]):
    ratio_fig.add_hline(
        y=level, line_dash="dot", line_color="rgba(255,0,0,0.4)", annotation_text=label
    )
st.plotly_chart(ratio_fig)
st.caption("Dotted lines: mean ± 1 and 2 standard deviations of the ratio over the chart range.")
