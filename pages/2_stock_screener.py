"""Stock Screener — multi-criteria search with bulk Weinstein classification and advanced filters."""

import pandas as pd
import streamlit as st

from config import CACHE_TTL_LONG, CACHE_TTL_SHORT
from queries.stocks_queries import get_all_etfs, search_stocks_by_metrics

st.header("Stock Screener")
st.caption(
    "Volume Z-Score: shows stocks where current volume is X standard deviations above the 1-year average."
)


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _get_etfs() -> list[str]:
    return get_all_etfs()


SIGNAL_SCAN_OPTIONS = [
    "vanna_extreme",
    "flownet_extreme",
    "volume_spike",
    "dxoi_change",
    "gex_level",
]

MA_PERIOD_OPTIONS = [None, 5, 8, 10, 20, 21, 50, 100, 200]

# --- Filter Form ---
with st.form("screener_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        etfs = _get_etfs()
        etf_ticker = st.selectbox("ETF Filter", [None] + etfs, format_func=lambda x: "All ETFs" if x is None else x)
        rs_rank_min = st.number_input("RS Rank Min (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, format="%.1f")
        rs_rank_max = st.number_input("RS Rank Max (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, format="%.1f")
    with col2:
        rs_rank_overall_min = st.number_input("RS Rank Overall Min (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, format="%.1f")
        rs_rank_overall_max = st.number_input("RS Rank Overall Max (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, format="%.1f")
        rs_z_score_min = st.number_input("RS Z-Score Min", value=None, step=0.1, format="%.2f")
        rs_z_score_max = st.number_input("RS Z-Score Max", value=None, step=0.1, format="%.2f")
    with col3:
        volume_std_dev_min = st.number_input("Volume Z-Score Min", value=None, step=0.1, format="%.1f")
        avg_volume_min = st.number_input("Avg 30-Day Volume Min", value=None, step=10000.0, format="%.0f")
        weinstein_stage = st.selectbox(
            "Weinstein Stage",
            [None, "Stage 1", "Stage 2", "Stage 3", "Stage 4"],
            format_func=lambda x: "All Stages" if x is None else x,
        )

    # --- Advanced Filters ---
    with st.expander("Advanced Filters"):
        adv1, adv2 = st.columns(2)
        with adv1:
            etf_rs_rank_min = st.number_input(
                "ETF RS Rank Min", min_value=0.0, max_value=1.0,
                value=None, step=0.05, format="%.2f",
                help="Only stocks in ETFs with rs_rank_index >= this value",
            )
            option_vol_spike = st.number_input(
                "Option Volume Spike (x avg)", min_value=1.0, max_value=10.0,
                value=None, step=0.5, format="%.1f",
                help="Option volume multiple vs 30-day average",
            )
            signal_scans = st.multiselect(
                "Signal Scans (intersection)",
                options=SIGNAL_SCAN_OPTIONS,
                help="Stocks must appear in ALL selected scans",
            )
        with adv2:
            rsi_min = st.number_input("RSI Min", min_value=0, max_value=100, value=None, step=5)
            rsi_max = st.number_input("RSI Max", min_value=0, max_value=100, value=None, step=5)
            ma_touch_period = st.selectbox(
                "Price Touching MA",
                MA_PERIOD_OPTIONS,
                format_func=lambda x: "None" if x is None else f"{x}-day SMA",
            )
            ta_timeframe = st.selectbox("TA Timeframe", ["D", "W", "M"], format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x])
            macd_crossover = st.selectbox(
                "MACD Crossover",
                [None, "bullish", "bearish"],
                format_func=lambda x: "Any" if x is None else x.title(),
            )
            adr_min = st.number_input(
                "ADR% Min", min_value=0, max_value=100, value=None, step=1,
                help="Minimum average daily range as % of close (14-day)",
            )

    submitted = st.form_submit_button("Search", type="primary")

if not submitted:
    st.stop()

# --- Phase 1: Pre-filters (build ticker restriction sets, intersect) ---
restrict_sets: list[set[str]] = []

with st.spinner("Applying pre-filters..."):
    if etf_rs_rank_min is not None:
        from queries.screener_queries import fetch_etf_rs_tickers
        restrict_sets.append(fetch_etf_rs_tickers(etf_rs_rank_min))

    if option_vol_spike is not None:
        from queries.screener_queries import fetch_option_volume_spike_symbols
        restrict_sets.append(fetch_option_volume_spike_symbols(option_vol_spike))

    if signal_scans:
        from queries.screener_queries import fetch_signal_scan_symbols
        restrict_sets.append(fetch_signal_scan_symbols(signal_scans))

# Intersect all pre-filter sets
restrict_tickers: list[str] | None = None
if restrict_sets:
    combined = restrict_sets[0]
    for s in restrict_sets[1:]:
        combined = combined & s
    if not combined:
        st.warning("No stocks found matching the pre-filter criteria.")
        st.stop()
    restrict_tickers = sorted(combined)

# --- Phase 2: SQL query (existing search + restrict_tickers) ---
with st.spinner("Searching..."):
    results = search_stocks_by_metrics(
        rs_rank_min=rs_rank_min,
        rs_rank_max=rs_rank_max,
        rs_rank_overall_min=rs_rank_overall_min,
        rs_rank_overall_max=rs_rank_overall_max,
        rs_z_score_min=rs_z_score_min,
        rs_z_score_max=rs_z_score_max,
        volume_std_dev_min=volume_std_dev_min,
        avg_volume_min=avg_volume_min,
        weinstein_stage=weinstein_stage,
        etf_ticker=etf_ticker,
        restrict_tickers=restrict_tickers,
    )

if results.empty:
    st.warning("No stocks found matching the criteria.")
    st.stop()

# --- Phase 3: TA post-filters (only on Phase 2 survivors) ---
ta_active = (
    rsi_min is not None
    or rsi_max is not None
    or macd_crossover is not None
    or ma_touch_period is not None
    or adr_min is not None
)
ta_info: dict[str, dict] = {}

if ta_active:
    with st.spinner("Computing technical indicators..."):
        from queries.screener_queries import fetch_bulk_ohlcv_for_ta
        from logic.technical import apply_ta_filters

        surviving_tickers = results["ticker"].unique().tolist()
        ohlcv = fetch_bulk_ohlcv_for_ta(surviving_tickers)

        if not ohlcv.empty:
            ta_info = apply_ta_filters(
                ohlcv_df=ohlcv,
                tickers=surviving_tickers,
                rsi_min=float(rsi_min) if rsi_min is not None else None,
                rsi_max=float(rsi_max) if rsi_max is not None else None,
                macd_crossover=macd_crossover,
                ma_touch_period=ma_touch_period,
                adr_min=float(adr_min) if adr_min is not None else None,
                timeframe=ta_timeframe,
            )
            results = results[results["ticker"].isin(ta_info.keys())]

if results.empty:
    st.warning("No stocks found after applying technical filters.")
    st.stop()

# --- Format & Display ---
display_df = results[["ticker", "etf_ticker", "rs_rank_index", "rs_rank_overall", "rs_z_score", "volume_z_score", "avg_30d_volume"]].copy()
display_df.columns = ["Ticker", "ETF", "RS Rank", "RS Rank Overall", "RS Z-Score", "Volume Z-Score", "Avg 30d Volume"]

# Format percentages
for col in ["RS Rank", "RS Rank Overall"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else "N/A")
for col in ["RS Z-Score", "Volume Z-Score"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
display_df["Avg 30d Volume"] = display_df["Avg 30d Volume"].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A")

if "stage" in results.columns:
    display_df["Weinstein Stage"] = results["stage"].values

# Add TA columns when active
if ta_active and ta_info:
    tickers = display_df["Ticker"].tolist()
    if any(info.get("rsi") is not None for info in ta_info.values()):
        display_df["RSI"] = [
            f"{ta_info.get(t, {}).get('rsi', None):.1f}" if ta_info.get(t, {}).get("rsi") is not None else "N/A"
            for t in tickers
        ]
    if any(info.get("macd_cross") is not None for info in ta_info.values()):
        display_df["MACD Cross"] = [
            ta_info.get(t, {}).get("macd_cross") or "N/A"
            for t in tickers
        ]
    if any(info.get("ma_touch") is not None for info in ta_info.values()):
        display_df["MA Touch"] = [
            str(ta_info.get(t, {}).get("ma_touch", "N/A"))
            for t in tickers
        ]
    if any(info.get("adr") is not None for info in ta_info.values()):
        display_df["ADR%"] = [
            f"{ta_info.get(t, {}).get('adr', None):.1f}%" if ta_info.get(t, {}).get("adr") is not None else "N/A"
            for t in tickers
        ]

st.dataframe(display_df, use_container_width=True, hide_index=True)
st.caption(f"{len(display_df)} results")
