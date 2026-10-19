"""Single stock: candlesticks and volume, RS metrics within one ETF, optional preset overlay."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stock_dashboard.config import CACHE_TTL_DEFAULT, CACHE_TTL_SHORT, get_settings
from stock_dashboard.logic.chart_helpers import (
    best_etf,
    compute_rs_moving_averages,
    get_non_trading_days,
    resample_ohlcv,
    signal_marker_prices,
    trim_to_range,
)
from stock_dashboard.logic.lookback import chart_warmup_bars
from stock_dashboard.logic.preset_signal import (
    compute_preset_signal_mask,
    determine_signal_direction,
)
from stock_dashboard.queries.stocks_queries import (
    fetch_ohlcv,
    fetch_stock_metrics,
    fetch_top_ranked_ticker,
    get_all_tickers,
)
from stock_dashboard.ui.charts import LEGEND_ABOVE, rs_figure
from stock_dashboard.ui.presets import benchmark_regimes, load_preset, preset_selectbox

DEFAULT_RANGE_DAYS = 365
INTERVALS = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
PERCENT = 100.0
RANK_AXIS_RANGE = [0, 102]  # a little headroom, so a rank of 100 isn't drawn on the edge


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _tickers() -> list[str]:
    return get_all_tickers()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _default_ticker() -> str | None:
    return fetch_top_ranked_ticker()


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _bars(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Daily bars from ``start`` to ``end`` plus the warm-up the overlay indicators need."""
    return fetch_ohlcv("stock", [ticker], start=start, end=end, warmup_bars=chart_warmup_bars())


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _metrics(ticker: str, end: date) -> pd.DataFrame:
    """Full metrics history, so moving averages are defined from the first displayed day."""
    return fetch_stock_metrics(ticker, end=end)


st.header("Single Stock")

tickers = _tickers()
default_ticker = _default_ticker()
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.selectbox(
        "Stock Ticker",
        tickers,
        index=tickers.index(default_ticker) if default_ticker in tickers else None,
        placeholder="Select a ticker...",
        help="Defaults to the stock with the highest overall RS rank on the latest date.",
    )
with col2:
    start_date = st.date_input("Start Date", value=date.today() - timedelta(DEFAULT_RANGE_DAYS))
with col3:
    end_date = st.date_input("End Date", value=date.today())

interval = st.radio("Interval", list(INTERVALS), format_func=INTERVALS.__getitem__, horizontal=True)
preset_stem = preset_selectbox("Signal Overlay (backtester preset)", key="single_stock_preset")

if not ticker:
    st.info("Select a stock ticker to view charts.")
    st.stop()
if not isinstance(start_date, date) or not isinstance(end_date, date) or start_date > end_date:
    st.warning("Pick a start date on or before the end date.")
    st.stop()

all_metrics = _metrics(ticker, end_date)
etf_options = sorted(all_metrics["etf_ticker"].dropna().unique())
etf = None
if etf_options:
    default_etf = best_etf(all_metrics)
    etf = st.selectbox(
        "ETF (relative strength and its rank are measured within one ETF)",
        etf_options,
        index=etf_options.index(default_etf) if default_etf in etf_options else 0,
        help="Defaults to the ETF in which the stock ranks highest on the latest date.",
    )
metrics = all_metrics[all_metrics["etf_ticker"] == etf].sort_values("date")

daily = _bars(ticker, start_date, end_date)
shown = trim_to_range(daily, start_date, end_date)
if shown.empty:
    st.warning("No price data found for this ticker and range.")
    st.stop()

# --- Price and volume ---
bars = resample_ohlcv(shown.set_index("date"), interval).reset_index()
up_day = bars["close"] >= bars["open"]

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3]
)
fig.add_trace(
    go.Candlestick(
        x=bars["date"],
        open=bars["open"],
        high=bars["high"],
        low=bars["low"],
        close=bars["close"],
        name="OHLC",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=bars["date"],
        y=bars["volume"],
        marker_color=up_day.map({True: "green", False: "red"}).tolist(),
        name="Volume",
    ),
    row=2,
    col=1,
)

preset = load_preset(preset_stem) if preset_stem else None
if preset_stem and preset is not None:
    regime = preset.get("_regime")
    if interval != "D":
        st.caption("The preset overlay is drawn on daily bars only.")
    elif preset.get("_raw_filters") or regime:
        # Evaluated over the warm-up bars as well, so every indicator is defined
        # from the first displayed bar; only displayed signals are drawn.
        benchmark = get_settings().benchmark
        regimes = benchmark_regimes(benchmark) if regime else None
        signal = compute_preset_signal_mask(
            daily,
            all_metrics,
            preset["_raw_filters"],
            etf=preset.get("_etf"),
            regime=regime,
            regimes=regimes,
        )
        if signal.outside_etf:
            st.warning(
                f"{ticker} is not a member of {preset['_etf']}, the ETF the preset was "
                "backtested within, so the overlay shows no signals."
            )
        elif signal.unsupported:
            reason = (
                f" (the benchmark {benchmark} has no price bars to classify regimes)"
                if regime and regimes is None
                else ""
            )
            st.warning(
                "The overlay cannot evaluate these preset conditions, so it shows no signals: "
                + ", ".join(signal.unsupported)
                + reason
            )
        else:
            applied = []
            if regime and regimes is not None:
                applied.append(f"regime ({regime}, classified from {benchmark})")
            if preset.get("_etf"):
                applied.append(f"ETF ({preset['_etf']}: RS metrics within it)")
            if applied:
                st.caption(
                    f"The overlay applies the preset's {' and '.join(applied)}, "
                    "as its backtest did."
                )
        hits = trim_to_range(daily[signal.mask], start_date, end_date)
        stats = preset.get("_stats", {})
        bullish = determine_signal_direction(stats) == "bullish"
        if not hits.empty:
            fig.add_trace(
                go.Scatter(
                    x=hits["date"],
                    y=signal_marker_prices(hits, bullish),
                    mode="markers",
                    marker={
                        "symbol": "triangle-up" if bullish else "triangle-down",
                        "size": 10,
                        "color": "green" if bullish else "red",
                    },
                    name=f"Signal: {preset['_name']}",
                ),
                row=1,
                col=1,
            )
        direction = (
            "markers point in the direction of its in-sample mean return"
            if stats
            else "markers point up because the preset has no backtest stats"
        )
        st.caption(
            f"{len(hits)} signal(s) from preset **{preset['_name']}** in this range; {direction}."
        )

rangebreaks = [{"values": get_non_trading_days(bars["date"])}] if interval == "D" else []
fig.update_layout(
    title=f"{ticker} Price and Volume",
    xaxis_rangeslider_visible=False,
    height=600,
    xaxis={"rangebreaks": rangebreaks},
)
st.plotly_chart(fig)

# --- Relative strength ---
if metrics.empty:
    st.info(f"No relative-strength metrics for {ticker}.")
    st.stop()

shown_metrics = trim_to_range(compute_rs_moving_averages(metrics), start_date, end_date)
if shown_metrics.empty:
    st.info("No relative-strength metrics in this date range.")
    st.stop()

st.plotly_chart(rs_figure(shown_metrics, f"{ticker} Relative Strength vs {etf}"))

ranks = make_subplots(
    rows=2, cols=1, shared_xaxes=True, subplot_titles=("RS Ranks (%)", "RS Z-Score")
)
for column, label, color in (
    ("rs_rank_index", f"RS Rank in {etf}", "purple"),
    ("rs_rank_overall", "RS Rank (overall)", "orange"),
):
    ranks.add_trace(
        go.Scatter(
            x=shown_metrics["date"],
            y=shown_metrics[column] * PERCENT,
            name=label,
            line={"color": color},
        ),
        row=1,
        col=1,
    )
ranks.add_trace(
    go.Scatter(
        x=shown_metrics["date"],
        y=shown_metrics["rs_z_score"],
        name="RS Z-Score",
        line={"color": "brown"},
    ),
    row=2,
    col=1,
)
ranks.update_layout(
    height=600,
    legend=LEGEND_ABOVE,
    xaxis={"rangebreaks": [{"values": get_non_trading_days(shown_metrics["date"])}]},
)
ranks.update_yaxes(title_text="Rank %", range=RANK_AXIS_RANGE, row=1, col=1)
ranks.update_yaxes(title_text="Z-Score", row=2, col=1)
st.plotly_chart(ranks)
