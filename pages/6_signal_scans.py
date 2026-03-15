"""Signal Scans — live signal detection from option metrics + technical trend signals."""

import pandas as pd
import streamlit as st

from config import CACHE_TTL_DEFAULT, CACHE_TTL_SHORT
from logic.signal_scanner import build_composite_scores, score_scan_results
from logic.trend_signals import scan_coiled_spring, scan_washout_bear, scan_washout_bull
from queries.signal_queries import (
    _enrich_with_rs,
    scan_dxoi_change,
    scan_flownet_extreme,
    scan_gex_level,
    scan_vanna_extreme,
    scan_volume_spike,
)

st.header("Signal Scans")
st.caption(
    "Live scans based on option metrics and technical trend signals. "
    "Tier 1 signals (Vanna, Flownet) are weighted 2x in composite scoring."
)


# --- Option Scan Functions (cached) ---
@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _run_vanna(threshold: float) -> pd.DataFrame:
    return scan_vanna_extreme(threshold)

@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _run_flownet(threshold: float) -> pd.DataFrame:
    return scan_flownet_extreme(threshold)

@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _run_volume(threshold: float) -> pd.DataFrame:
    return scan_volume_spike(threshold)

@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _run_dxoi(threshold: float) -> pd.DataFrame:
    return scan_dxoi_change(threshold)

@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _run_gex() -> pd.DataFrame:
    return scan_gex_level()


# --- Trend indicator computation (cached) ---
@st.cache_data(ttl=CACHE_TTL_SHORT)
def _fetch_trend_indicators(etf_mode: bool) -> pd.DataFrame:
    """Fetch OHLCV and compute trend indicators for the scan universe."""
    from logic.trend_signals import compute_trend_indicators

    if etf_mode:
        from queries.screener_queries import fetch_bulk_etf_ohlcv_for_ta
        from queries.stocks_queries import get_all_etfs

        tickers = get_all_etfs()
        ohlcv = fetch_bulk_etf_ohlcv_for_ta(tickers, lookback_days=350)
    else:
        from queries.screener_queries import (
            fetch_bulk_ohlcv_for_ta,
            fetch_tracked_tickers,
        )

        tickers = fetch_tracked_tickers()
        ohlcv = fetch_bulk_ohlcv_for_ta(tickers, lookback_days=350)

    if ohlcv.empty:
        return pd.DataFrame()
    return compute_trend_indicators(ohlcv)


# --- Shared threshold controls ---
with st.sidebar:
    st.subheader("Option Scan Thresholds")
    vanna_pct = st.slider("Vanna Percentile", 5, 40, 20, help="Top/bottom N percentile of 90-day history")
    flownet_pct = st.slider("Flownet Percentile", 5, 30, 10, help="Bottom N percentile (contrarian)")
    vol_mult = st.slider("Volume Multiple", 1.5, 5.0, 2.0, 0.5, help="Nx 20-day average")
    dxoi_pct = st.slider("dxoi Change %", 10.0, 50.0, 20.0, 5.0, help="Absolute % change threshold")

    st.divider()
    st.subheader("Trend Scan Settings")
    trend_etf_mode = st.checkbox("Scan ETFs (instead of stocks)", key="trend_etf_mode")

    with st.expander("Washout Thresholds"):
        wo_rsi = st.slider("RSI Threshold", 20.0, 50.0, 40.0, 5.0,
                           help="Bull: RSI below / Bear: RSI above")
        wo_mom = st.slider("Momentum % Threshold", 2.0, 15.0, 5.0, 1.0,
                           help="Bull: mom below -N% / Bear: mom above +N%")
        wo_ratio = st.slider("Distance from 50MA %", 2.0, 12.0, 6.0, 1.0,
                             help="Bull: below by N% / Bear: above by N%")
        wo_bb = st.slider("Min BB Width %", 8.0, 20.0, 12.0, 1.0,
                          help="Minimum Bollinger Band width")

    with st.expander("Coiled Spring Thresholds"):
        cs_bb = st.slider("Max BB Width %", 4.0, 10.0, 7.0, 0.5,
                          help="Maximum Bollinger Band width (tighter = more coiled)")
        cs_adr = st.slider("Max ADR %", 0.5, 3.0, 1.5, 0.25,
                           help="Maximum Average Daily Range %")
        cs_200ma_range = st.slider("200MA Range %", 5.0, 20.0, 15.0, 5.0,
                                   help="Price within ±N% of 200MA")


def _format_rs_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Format RS columns for display."""
    display = df.copy()
    if "rs_rank_index" in display.columns:
        display["rs_rank_index"] = display["rs_rank_index"].apply(
            lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else ""
        )
    if "rs_rank_overall" in display.columns:
        display["rs_rank_overall"] = display["rs_rank_overall"].apply(
            lambda x: f"{x * 100:.1f}%" if pd.notnull(x) else ""
        )
    if "rs_z_score" in display.columns:
        display["rs_z_score"] = display["rs_z_score"].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else ""
        )
    return display


def _display_scan_tab(df: pd.DataFrame, value_col: str = "signal_value", tab_name: str = "scan") -> None:
    """Display a scan result table with z-scores."""
    if df.empty:
        st.info("No signals found with current thresholds.")
        return
    scored = score_scan_results(df, value_col)
    display = _format_rs_columns(scored)
    # Select display columns
    cols = ["symbol", "signal_value", "signal_label"]
    if "z_score" in display.columns:
        cols.append("z_score")
    if "pct_rank" in display.columns:
        cols.append("pct_rank")
    if "vol_ratio" in display.columns:
        cols.append("vol_ratio")
    if "pct_change" in display.columns:
        cols.append("pct_change")
    for rc in ["rs_rank_index", "rs_z_score"]:
        if rc in display.columns:
            cols.append(rc)
    available = [c for c in cols if c in display.columns]
    st.dataframe(display[available], use_container_width=True, hide_index=True)
    st.caption(f"{len(display)} results")

    # --- TradingView Watchlist Export ---
    from logic.export import build_tradingview_watchlist
    from queries.stocks_queries import fetch_exchange_map

    _tickers = display["symbol"].tolist()
    _exchange_map = fetch_exchange_map(_tickers)
    _watchlist = build_tradingview_watchlist(_tickers, _exchange_map)
    st.download_button("Export TradingView Watchlist", _watchlist, f"{tab_name}_watchlist.txt", "text/plain", key=f"export_{tab_name}")


def _display_trend_tab(df: pd.DataFrame, tab_name: str = "trend") -> None:
    """Display a trend scan result with indicator columns."""
    if df.empty:
        st.info("No signals found with current thresholds.")
        return

    scored = score_scan_results(df)
    display = _format_rs_columns(scored)

    # Format indicator columns for readability
    fmt = display.copy()
    for col in ["rsi", "mom_20", "bb_width", "adr_pct"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    for col in ["ratio_50ma", "ratio_200ma", "vol_ratio"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    if "bb_width_chg" in fmt.columns:
        fmt["bb_width_chg"] = fmt["bb_width_chg"].apply(
            lambda x: f"{x:+.2f}" if pd.notnull(x) else ""
        )

    cols = ["symbol", "signal_label", "rsi", "mom_20", "ratio_50ma", "ratio_200ma",
            "bb_width", "adr_pct", "vol_ratio", "bb_width_chg"]
    for rc in ["rs_rank_index", "rs_z_score"]:
        if rc in fmt.columns:
            cols.append(rc)
    available = [c for c in cols if c in fmt.columns]
    st.dataframe(fmt[available], use_container_width=True, hide_index=True)
    st.caption(f"{len(fmt)} results")

    # --- TradingView Watchlist Export ---
    from logic.export import build_tradingview_watchlist
    from queries.stocks_queries import fetch_exchange_map

    _tickers = display["symbol"].tolist()
    _exchange_map = fetch_exchange_map(_tickers)
    _watchlist = build_tradingview_watchlist(_tickers, _exchange_map)
    st.download_button("Export TradingView Watchlist", _watchlist, f"{tab_name}_watchlist.txt", "text/plain", key=f"export_{tab_name}")


# --- Compute trend scans upfront (shared across tabs) ---
with st.spinner("Computing trend indicators..."):
    trend_indicators = _fetch_trend_indicators(trend_etf_mode)

wo_bull_df = pd.DataFrame()
wo_bear_df = pd.DataFrame()
coiled_df = pd.DataFrame()

if not trend_indicators.empty:
    wo_bull_df = _enrich_with_rs(scan_washout_bull(
        trend_indicators,
        rsi_max=wo_rsi,
        mom_max=-wo_mom,
        ratio_50ma_max=1.0 - wo_ratio / 100,
        bb_width_min=wo_bb,
    ))
    wo_bear_df = _enrich_with_rs(scan_washout_bear(
        trend_indicators,
        rsi_min=100 - wo_rsi,
        mom_min=wo_mom,
        ratio_50ma_min=1.0 + wo_ratio / 100,
        bb_width_min=wo_bb,
    ))
    coiled_df = _enrich_with_rs(scan_coiled_spring(
        trend_indicators,
        bb_width_max=cs_bb,
        adr_max=cs_adr,
        ratio_200ma_min=1.0 - cs_200ma_range / 100,
        ratio_200ma_max=1.0 + cs_200ma_range / 100,
    ))


# --- Tabs ---
tabs = st.tabs([
    "Vanna Extreme (T1)",
    "Flownet Extreme (T1)",
    "Volume Spike (T2)",
    "dxoi Change (T2)",
    "GEX Level (T2)",
    "Washout Bull (T2)",
    "Washout Bear (T2)",
    "Coiled Spring (T2)",
    "Composite",
])

with tabs[0]:
    st.subheader("Vanna Exposure Extremes")
    st.caption("High vanna = when vol drops, delta increases (bullish). Spread: +3.38% over 10d.")
    with st.spinner("Running vanna scan..."):
        vanna_df = _run_vanna(float(vanna_pct))
    _display_scan_tab(vanna_df, tab_name="vanna")

with tabs[1]:
    st.subheader("Flownet Extremes (Contrarian)")
    st.caption("Extreme negative flow -> contrarian buy. +1.25% 10d, 56.4% win rate.")
    with st.spinner("Running flownet scan..."):
        flownet_df = _run_flownet(float(flownet_pct))
    _display_scan_tab(flownet_df, tab_name="flownet")

with tabs[2]:
    st.subheader("Volume Spikes")
    st.caption("Volume > Nx 20d average. Predicts higher absolute moves.")
    with st.spinner("Running volume scan..."):
        volume_df = _run_volume(float(vol_mult))
    _display_scan_tab(volume_df, tab_name="volume")

with tabs[3]:
    st.subheader("dxoi Change")
    st.caption("Big positioning shifts precede price moves. +2.01% at 5d.")
    with st.spinner("Running dxoi scan..."):
        dxoi_df = _run_dxoi(float(dxoi_pct))
    _display_scan_tab(dxoi_df, tab_name="dxoi")

with tabs[4]:
    st.subheader("GEX Regime")
    st.caption("Negative GEX -> higher volatility. Very high GEX -> strong trending.")
    with st.spinner("Running GEX scan..."):
        gex_df = _run_gex()
    _display_scan_tab(gex_df, tab_name="gex")

with tabs[5]:
    st.subheader("Washout Bull — Reversal Long Setup")
    st.caption(
        "Oversold + below 50MA + elevated volatility + volume. "
        "Empirical median: +20.8% over 53 days."
    )
    if trend_indicators.empty:
        st.warning("No OHLCV data available for trend scans.")
    else:
        _display_trend_tab(wo_bull_df, tab_name="washout_bull")

with tabs[6]:
    st.subheader("Washout Bear — Reversal Short Setup")
    st.caption(
        "Overbought + above 50MA + elevated volatility. "
        "Mirror of washout bull — extended rallies vulnerable to reversal."
    )
    if trend_indicators.empty:
        st.warning("No OHLCV data available for trend scans.")
    else:
        _display_trend_tab(wo_bear_df, tab_name="washout_bear")

with tabs[7]:
    st.subheader("Coiled Spring — Volatility Compression")
    st.caption(
        "Tight Bollinger Bands + low ADR% + near 200MA. "
        "Basing pattern (Weinstein Stage 1/3). Direction agnostic. "
        "Empirical median: 17.6% move over 95 days."
    )
    if trend_indicators.empty:
        st.warning("No OHLCV data available for trend scans.")
    else:
        _display_trend_tab(coiled_df, tab_name="coiled_spring")

# --- Composite ---
with tabs[8]:
    st.subheader("Composite Signal Score")
    st.caption("Tier 1 signals weighted 2x. Symbols appearing in multiple scans score highest.")

    scan_results = {
        "vanna_extreme": vanna_df,
        "flownet_extreme": flownet_df,
        "volume_spike": volume_df,
        "dxoi_change": dxoi_df,
        "gex_level": gex_df,
    }
    if not wo_bull_df.empty:
        scan_results["washout_bull"] = wo_bull_df
    if not wo_bear_df.empty:
        scan_results["washout_bear"] = wo_bear_df
    if not coiled_df.empty:
        scan_results["coiled_spring"] = coiled_df

    composite = build_composite_scores(scan_results)
    if composite.empty:
        st.info("No composite signals found.")
    else:
        display = _format_rs_columns(composite)
        cols = ["symbol", "composite_score", "signal_count", "signals"]
        for rc in ["rs_rank_index", "rs_z_score"]:
            if rc in display.columns:
                cols.append(rc)
        available = [c for c in cols if c in display.columns]
        st.dataframe(display[available], use_container_width=True, hide_index=True)
        st.caption(f"{len(display)} symbols with active signals")

        # --- TradingView Watchlist Export ---
        from logic.export import build_tradingview_watchlist
        from queries.stocks_queries import fetch_exchange_map

        _tickers = display["symbol"].tolist()
        _exchange_map = fetch_exchange_map(_tickers)
        _watchlist = build_tradingview_watchlist(_tickers, _exchange_map)
        st.download_button("Export TradingView Watchlist", _watchlist, "composite_watchlist.txt", "text/plain", key="export_composite")
