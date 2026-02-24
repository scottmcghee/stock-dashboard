"""Signal Scans — live signal detection from option metrics."""

import pandas as pd
import streamlit as st

from config import CACHE_TTL_DEFAULT
from logic.signal_scanner import build_composite_scores, score_scan_results
from queries.signal_queries import (
    scan_dxoi_change,
    scan_flownet_extreme,
    scan_gex_level,
    scan_vanna_extreme,
    scan_volume_spike,
)

st.header("Signal Scans")
st.caption(
    "Live scans based on option metrics. "
    "Tier 1 signals (Vanna, Flownet) are weighted 2x in composite scoring."
)


# --- Scan Functions (cached) ---
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


# --- Shared threshold controls ---
with st.sidebar:
    st.subheader("Scan Thresholds")
    vanna_pct = st.slider("Vanna Percentile", 5, 40, 20, help="Top/bottom N percentile of 90-day history")
    flownet_pct = st.slider("Flownet Percentile", 5, 30, 10, help="Bottom N percentile (contrarian)")
    vol_mult = st.slider("Volume Multiple", 1.5, 5.0, 2.0, 0.5, help="Nx 20-day average")
    dxoi_pct = st.slider("dxoi Change %", 10.0, 50.0, 20.0, 5.0, help="Absolute % change threshold")


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


def _display_scan_tab(df: pd.DataFrame, value_col: str = "signal_value") -> None:
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


# --- Tabs ---
tabs = st.tabs([
    "Vanna Extreme (T1)",
    "Flownet Extreme (T1)",
    "Volume Spike (T2)",
    "dxoi Change (T2)",
    "GEX Level (T2)",
    "Composite",
])

with tabs[0]:
    st.subheader("Vanna Exposure Extremes")
    st.caption("High vanna = when vol drops, delta increases (bullish). Spread: +3.38% over 10d.")
    with st.spinner("Running vanna scan..."):
        vanna_df = _run_vanna(float(vanna_pct))
    _display_scan_tab(vanna_df)

with tabs[1]:
    st.subheader("Flownet Extremes (Contrarian)")
    st.caption("Extreme negative flow -> contrarian buy. +1.25% 10d, 56.4% win rate.")
    with st.spinner("Running flownet scan..."):
        flownet_df = _run_flownet(float(flownet_pct))
    _display_scan_tab(flownet_df)

with tabs[2]:
    st.subheader("Volume Spikes")
    st.caption("Volume > Nx 20d average. Predicts higher absolute moves.")
    with st.spinner("Running volume scan..."):
        volume_df = _run_volume(float(vol_mult))
    _display_scan_tab(volume_df)

with tabs[3]:
    st.subheader("dxoi Change")
    st.caption("Big positioning shifts precede price moves. +2.01% at 5d.")
    with st.spinner("Running dxoi scan..."):
        dxoi_df = _run_dxoi(float(dxoi_pct))
    _display_scan_tab(dxoi_df)

with tabs[4]:
    st.subheader("GEX Regime")
    st.caption("Negative GEX -> higher volatility. Very high GEX -> strong trending.")
    with st.spinner("Running GEX scan..."):
        gex_df = _run_gex()
    _display_scan_tab(gex_df)

with tabs[5]:
    st.subheader("Composite Signal Score")
    st.caption("Tier 1 signals weighted 2x. Symbols appearing in multiple scans score highest.")

    scan_results = {
        "vanna_extreme": vanna_df,
        "flownet_extreme": flownet_df,
        "volume_spike": volume_df,
        "dxoi_change": dxoi_df,
        "gex_level": gex_df,
    }
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
