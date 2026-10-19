"""Signal scans: option-positioning scans, trend-readiness scans and a composite score.

Without option data only the trend scans (and a composite of them) are shown.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
import streamlit as st

from stock_dashboard.config import CACHE_TTL_DEFAULT
from stock_dashboard.logic.scan_settings import (
    COMPOSITE_RULE,
    OPTION_SCAN_TITLES,
    TREND_SCAN_TITLES,
    TREND_SCANS,
    CoiledSpringSettings,
    ScanSettings,
    WashoutSettings,
    scan_rule,
)
from stock_dashboard.logic.signal_scanner import build_composite_scores
from stock_dashboard.queries import signal_queries
from stock_dashboard.queries.screener_queries import fetch_tracked_tickers, fetch_trend_indicators
from stock_dashboard.queries.stocks_queries import enrich_with_rs, get_all_etfs
from stock_dashboard.ui.export import tradingview_export
from stock_dashboard.ui.option_guard import note_missing_option_data
from stock_dashboard.ui.tables import show_table

PERCENT = 100.0
RS_FORMATS = {"rs_rank_index": "%.1f", "rs_rank_overall": "%.1f", "rs_z_score": "%.2f"}
SCAN_FORMATS = {
    "signal_value": "%.4g",
    "strength": "%.3f",
    "pct_rank": "%.1f",
    "vol_ratio": "%.2f",
    "pct_change": "%+.1f",
    "rsi": "%.1f",
    "mom_20": "%+.1f",
    "ratio_50ma": "%.3f",
    "ratio_200ma": "%.3f",
    "bb_width": "%.1f",
    "adr_pct": "%.2f",
    "bb_width_chg": "%+.2f",
    "composite_score": "%.2f",
    **RS_FORMATS,
}
OPTION_COLUMNS = ["symbol", "signal_label", "signal_value", "pct_rank", "vol_ratio",
                  "pct_change", "strength"]  # fmt: skip
TREND_COLUMNS = ["symbol", "signal_label", "rsi", "mom_20", "ratio_50ma", "ratio_200ma",
                 "bb_width", "adr_pct", "vol_ratio", "bb_width_chg",
                 "strength"]  # fmt: skip
# Columns stored as fractions (0-1) and shown as percentages, like the sliders.
PERCENT_COLUMNS = ("rs_rank_index", "rs_rank_overall", "pct_rank")
RS_COLUMNS = ["etf_ticker", "rs_rank_index", "rs_rank_overall", "rs_z_score"]


@st.cache_data(ttl=CACHE_TTL_DEFAULT, show_spinner="Running option scans...")
def _option_scans(
    vanna_percentile: float, flownet_percentile: float, volume_multiple: float, dxoi_pct: float
) -> dict[str, pd.DataFrame]:
    """Every option scan on one shared history fetch, enriched with RS metrics.

    Keyed on the option thresholds only, so moving a trend slider does not re-run them.
    """
    history = signal_queries.fetch_option_signal_history()
    scans: dict[str, Callable[[], pd.DataFrame]] = {
        "vanna_extreme": lambda: signal_queries.scan_vanna_extreme(
            vanna_percentile, history=history
        ),
        "flownet_extreme": lambda: signal_queries.scan_flownet_extreme(
            flownet_percentile, history=history
        ),
        "volume_spike": lambda: signal_queries.scan_volume_spike(volume_multiple, history=history),
        "dxoi_change": lambda: signal_queries.scan_dxoi_change(dxoi_pct, history=history),
        "gex_level": lambda: signal_queries.scan_gex_level(history=history),
    }
    return {name: run() for name, run in scans.items()}


@st.cache_data(ttl=CACHE_TTL_DEFAULT, show_spinner="Computing trend indicators...")
def _trend_indicators(etf_mode: bool) -> pd.DataFrame:
    if etf_mode:
        return fetch_trend_indicators(get_all_etfs(), kind="etf")
    return fetch_trend_indicators(fetch_tracked_tickers())


def _trend_scans(indicators: pd.DataFrame, s: ScanSettings) -> dict[str, pd.DataFrame]:
    if indicators.empty:
        return dict.fromkeys(TREND_SCANS, pd.DataFrame())
    return {name: enrich_with_rs(scan(indicators, s)) for name, scan in TREND_SCANS.items()}


def _show_table(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if df.empty:
        st.info("No symbols match with the current thresholds.")
        return
    shown = df.copy()
    for col in PERCENT_COLUMNS:
        if col in shown.columns:
            shown[col] = shown[col] * PERCENT
    available = [c for c in (*columns, *RS_COLUMNS) if c in shown.columns]
    show_table(shown[available], formats=SCAN_FORMATS)
    st.caption(f"{len(shown)} results. Ranks and percentiles are percentages.")
    tradingview_export(shown["symbol"], filename=f"{name}_watchlist.txt", key=f"export_{name}")


st.header("Signal Scans")
st.caption(
    "Rule-based scans of the latest session. Each tab states its rule; thresholds are in "
    "the sidebar."
)
has_options = note_missing_option_data("the option-positioning scans")

defaults = ScanSettings()
with st.sidebar:
    if has_options:
        st.subheader("Option Scan Thresholds")
        vanna_pct = st.slider("Vanna percentile", 5.0, 40.0, defaults.vanna_percentile, 1.0)
        flownet_pct = st.slider("Flownet percentile", 5.0, 30.0, defaults.flownet_percentile, 1.0)
        volume_multiple = st.slider("Volume multiple", 1.5, 5.0, defaults.volume_multiple, 0.5)
        dxoi_pct = st.slider("dxoi change %", 10.0, 50.0, defaults.dxoi_change_pct, 5.0)
        st.divider()
    st.subheader("Trend Scan Settings")
    etf_mode = st.checkbox("Scan ETFs instead of stocks", key="trend_etf_mode")
    wo, cs = defaults.washout, defaults.coiled_spring
    with st.expander("Washout thresholds"):
        wo_rsi = st.slider(
            "RSI threshold", 20.0, 50.0, wo.rsi, 5.0, help="Bull: at or below; bear: 100 minus it"
        )
        wo_mom = st.slider("Momentum %", 2.0, 15.0, wo.momentum_pct, 1.0)
        wo_dist = st.slider("Distance from 50-day MA %", 2.0, 12.0, wo.ma_distance_pct, 1.0)
        wo_bb = st.slider("Min Bollinger width %", 8.0, 20.0, wo.min_bb_width, 1.0)
    with st.expander("Coiled spring thresholds"):
        cs_bb = st.slider("Max Bollinger width %", 4.0, 10.0, cs.max_bb_width, 0.5)
        cs_adr = st.slider("Max ADR %", 0.5, 3.0, cs.max_adr_pct, 0.25)
        cs_range = st.slider("Range around 200-day MA %", 5.0, 20.0, cs.ma_range_pct, 5.0)

settings = ScanSettings(
    **(
        {
            "vanna_percentile": vanna_pct,
            "flownet_percentile": flownet_pct,
            "volume_multiple": volume_multiple,
            "dxoi_change_pct": dxoi_pct,
        }
        if has_options
        else {}
    ),
    washout=WashoutSettings(wo_rsi, wo_mom, wo_dist, wo_bb),
    coiled_spring=CoiledSpringSettings(cs_bb, cs_adr, cs_range),
)

option_results = (
    _option_scans(
        settings.vanna_percentile,
        settings.flownet_percentile,
        settings.volume_multiple,
        settings.dxoi_change_pct,
    )
    if has_options
    else {}
)
indicators = _trend_indicators(etf_mode)
trend_results = _trend_scans(indicators, settings)

titles = {**(OPTION_SCAN_TITLES if has_options else {}), **TREND_SCAN_TITLES}
tabs = st.tabs([*titles.values(), "Composite"])
for tab, (name, title) in zip(tabs, titles.items(), strict=False):
    with tab:
        st.subheader(title)
        st.caption(scan_rule(name, settings))
        if name in option_results:
            _show_table(option_results[name], OPTION_COLUMNS, name)
        elif indicators.empty:
            st.warning("No price history available for the trend scans.")
        else:
            _show_table(trend_results[name], TREND_COLUMNS, name)

with tabs[-1]:
    st.subheader("Composite Signal Score")
    st.caption(COMPOSITE_RULE)
    composite = build_composite_scores({**option_results, **trend_results})
    _show_table(composite, ["symbol", "composite_score", "signal_count", "signals"], "composite")
