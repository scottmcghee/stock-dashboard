"""Stock screener: RS and volume search, pre-filters, technical post-filters, events, patterns.

Option-based filters and columns appear only when option data is available.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from stock_dashboard.config import CACHE_TTL_SHORT, get_settings
from stock_dashboard.logic.events import latest_event_flags_bulk
from stock_dashboard.logic.patterns import PATTERN_DETECTORS, scan_patterns
from stock_dashboard.logic.preset_display import (
    interpret_preset,
    render_preset_summary,
    screener_differences,
    screener_regime_note,
)
from stock_dashboard.logic.scan_settings import TREND_SCANS
from stock_dashboard.logic.screener import (
    EVENT_LABELS,
    RESULT_FORMATS,
    TaFilters,
    build_results_table,
    intersect_prefilters,
    option_signal_table,
    percent_to_fraction,
    tickers_with_events,
)
from stock_dashboard.logic.technical import (
    DEFAULT_GAP_LOOKBACK,
    PIVOT_EXCLUDE_RECENT,
    SCREENER_PIVOT_WINDOW,
    apply_ta_filters,
)
from stock_dashboard.queries.screener_queries import (
    OPTION_SCANS,
    WEEKLY_HORIZON_DAYS,
    WEEKLY_MIN_EXPIRATIONS,
    fetch_etf_rs_tickers,
    fetch_option_metrics_for_tickers,
    fetch_option_volume_spike_symbols,
    fetch_signal_scan_symbols,
    fetch_weekly_options_symbols,
)
from stock_dashboard.queries.signal_queries import (
    fetch_option_signal_history,
    scan_flownet_extreme,
    scan_vanna_extreme,
)
from stock_dashboard.queries.stocks_queries import (
    fetch_ohlcv,
    get_all_etfs,
    search_stocks_by_metrics,
)
from stock_dashboard.ui.export import tradingview_export
from stock_dashboard.ui.option_guard import note_missing_option_data
from stock_dashboard.ui.presets import benchmark_regimes, load_preset, preset_selectbox
from stock_dashboard.ui.tables import show_table

MA_TOUCH_PERIODS = [None, 5, 8, 10, 20, 21, 50, 100, 200]
MOMENTUM_PERIODS = [20, 63, 126]
RATIO_MA_PERIODS = [50, 200]
STAGES = [None, "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
TIMEFRAMES = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
MACD_CHOICES = [None, "bullish", "bearish", "not bullish", "not bearish"]
GAP_CHOICES: dict[str | None, str] = {
    None: "None",
    "bull": "Bull gap (up)",
    "bear": "Bear gap (down)",
    "any": "Any gap",
}
RESET_KEY = "_screener_reset"
PRESET_KEY = "_screener_preset"


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _etfs() -> list[str]:
    return get_all_etfs()


def _reset_form() -> None:
    st.session_state[RESET_KEY] = st.session_state.get(RESET_KEY, 0) + 1


def _index(options: list[Any], value: Any, default: int = 0) -> int:
    return options.index(value) if value in options else default


st.header("Stock Screener")
st.caption(
    "Ranks are percentages (0-100). Volume Z-Score: the latest volume's distance from "
    "its one-year mean, in standard deviations."
)
has_options = note_missing_option_data("option-based filters and columns")

# --- Preset ---
preset: dict[str, Any] = st.session_state.get(PRESET_KEY, {})
selected = preset_selectbox("Load backtester preset", key="_screener_preset_select")
if selected != preset.get("_stem"):
    loaded = load_preset(selected) if selected else {}
    if loaded is not None:  # None: unreadable, the error is shown and the form is kept
        st.session_state[PRESET_KEY] = loaded
        _reset_form()  # widgets pick up the preset's values only in a new form
        st.rerun()
if preset:
    if preset.get("_description"):
        st.caption(preset["_description"])
    if preset.get("_unmapped"):
        st.info(f"Preset filters the screener does not apply: {', '.join(preset['_unmapped'])}")
    for note in screener_differences(preset.get("_raw_filters") or []):
        st.caption(note)
    if preset.get("_regime"):
        benchmark = get_settings().benchmark
        st.caption(screener_regime_note(preset["_regime"], benchmark_regimes(benchmark), benchmark))
    st.caption(interpret_preset(preset))
    with st.expander("Preset details"):
        st.markdown(render_preset_summary(preset))

if st.button("Reset Filters"):
    st.session_state.pop(PRESET_KEY, None)
    st.session_state.pop("_screener_preset_select", None)
    _reset_form()
    st.rerun()

scan_options = [*(OPTION_SCANS if has_options else ()), *TREND_SCANS]

with st.form(f"screener_form_{st.session_state.get(RESET_KEY, 0)}"):
    col1, col2, col3 = st.columns(3)
    with col1:
        etf_ticker = st.selectbox(
            "ETF", [None, *_etfs()], format_func=lambda x: "All ETFs" if x is None else x
        )
        rs_rank_min = st.number_input("RS Rank in ETF Min (%)", 0.0, 100.0, value=None, step=1.0)
        rs_rank_max = st.number_input("RS Rank in ETF Max (%)", 0.0, 100.0, value=None, step=1.0)
    with col2:
        rs_rank_overall_min = st.number_input(
            "RS Rank Overall Min (%)", 0.0, 100.0, value=preset.get("rs_rank_overall_min"), step=1.0
        )
        rs_rank_overall_max = st.number_input(
            "RS Rank Overall Max (%)", 0.0, 100.0, value=preset.get("rs_rank_overall_max"), step=1.0
        )
        rs_z_score_min = st.number_input(
            "RS Z-Score Min", value=preset.get("rs_z_score_min"), step=0.1, format="%.2f"
        )
        rs_z_score_max = st.number_input(
            "RS Z-Score Max", value=preset.get("rs_z_score_max"), step=0.1, format="%.2f"
        )
    with col3:
        volume_z_min = st.number_input(
            "Volume Z-Score Min", value=preset.get("volume_std_dev_min"), step=0.1, format="%.1f"
        )
        avg_volume_min = st.number_input(
            "Avg 30-Day Volume Min", value=None, step=10000.0, format="%.0f"
        )
        weinstein_stage = st.selectbox(
            "Weinstein Stage",
            STAGES,
            index=_index(STAGES, preset.get("weinstein_stage")),
            format_func=lambda x: "All Stages" if x is None else x,
        )

    with st.expander("Advanced Filters"):
        adv1, adv2 = st.columns(2)
        with adv1:
            etf_rs_rank_min = st.number_input(
                "ETF RS Rank Min (%)",
                0.0,
                100.0,
                value=None,
                step=5.0,
                help="Only stocks in an ETF whose rank against the benchmark is at least this.",
            )
            option_vol_spike = (
                st.number_input(
                    "Option Volume Spike (x avg)",
                    1.0,
                    10.0,
                    value=None,
                    step=0.5,
                    help="Latest option volume as a multiple of its 30-session average.",
                )
                if has_options
                else None
            )
            weekly_options_only = has_options and st.checkbox(
                "Weekly options only",
                help=f"Only stocks with at least {WEEKLY_MIN_EXPIRATIONS} expirations within "
                f"{WEEKLY_HORIZON_DAYS} days of the latest option snapshot.",
            )
            signal_scans = st.multiselect(
                "Signal Scans (all must match)",
                scan_options,
                help="See the Signal Scans page for each scan's rule.",
            )
            momentum_period = st.selectbox(
                "Momentum Period",
                MOMENTUM_PERIODS,
                index=_index(MOMENTUM_PERIODS, preset.get("momentum_period"), 1),
                format_func=lambda x: f"{x}-day",
            )
            momentum_min = st.number_input(
                "Momentum Min (%)", value=preset.get("momentum_min"), step=5.0, format="%.1f"
            )
            momentum_max = st.number_input(
                "Momentum Max (%)", value=preset.get("momentum_max"), step=5.0, format="%.1f"
            )
            ratio_to_ma_period = st.selectbox(
                "Ratio to MA Period",
                RATIO_MA_PERIODS,
                index=_index(RATIO_MA_PERIODS, preset.get("ratio_to_ma_period"), 1),
                format_func=lambda x: f"{x}-day SMA",
            )
            ratio_to_ma_min = st.number_input(
                "Ratio to MA Min",
                value=preset.get("ratio_to_ma_min"),
                step=0.05,
                format="%.3f",
                help="Close / SMA (1.0 = at the MA).",
            )
            ratio_to_ma_max = st.number_input(
                "Ratio to MA Max", value=preset.get("ratio_to_ma_max"), step=0.05, format="%.3f"
            )
        with adv2:
            rsi_min = st.number_input(
                "RSI Min", 0.0, 100.0, value=preset.get("rsi_min"), step=5.0, format="%.1f"
            )
            rsi_max = st.number_input(
                "RSI Max", 0.0, 100.0, value=preset.get("rsi_max"), step=5.0, format="%.1f"
            )
            ma_touch_period = st.selectbox(
                "Price Touching MA",
                MA_TOUCH_PERIODS,
                format_func=lambda x: "None" if x is None else f"{x}-day SMA",
            )
            ta_timeframe = st.selectbox(
                "TA Timeframe", list(TIMEFRAMES), format_func=TIMEFRAMES.__getitem__
            )
            macd_crossover = st.selectbox(
                "MACD Crossover",
                MACD_CHOICES,
                format_func=lambda x: "Any" if x is None else x.capitalize(),
            )
            adr_min = st.number_input(
                "ADR% Min",
                0.0,
                100.0,
                value=preset.get("adr_min"),
                step=0.5,
                format="%.2f",
                help="Minimum 14-day average daily range, % of close.",
            )
            gap_type = st.selectbox(
                "Gap Detection", list(GAP_CHOICES), format_func=GAP_CHOICES.__getitem__
            )
            gap_min_pct = st.number_input("Gap Min %", 0.0, 20.0, value=1.0, step=0.5)
            gap_lookback = st.number_input(
                "Gap Lookback (bars)", 1, 20, value=DEFAULT_GAP_LOOKBACK, step=1
            )
            dist_low_min = st.number_input(
                "Dist from 126-day Low Min (%)",
                value=preset.get("dist_recent_low_126_min"),
                step=1.0,
                format="%.1f",
                help=f"How far the close is above the lowest close of the {SCREENER_PIVOT_WINDOW} "
                f"bars ending {PIVOT_EXCLUDE_RECENT} bars ago (10 = 10% above).",
            )
            dist_low_max = st.number_input(
                "Dist from 126-day Low Max (%)",
                value=preset.get("dist_recent_low_126_max"),
                step=1.0,
                format="%.1f",
            )
        pattern_scans = st.multiselect(
            "Chart Patterns (any may match)",
            list(PATTERN_DETECTORS),
            default=[p for p in preset.get("pattern_scans", []) if p in PATTERN_DETECTORS],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        event_scans = st.multiselect(
            "TA Events in the last 3 bars (any may match)",
            list(EVENT_LABELS),
            format_func=EVENT_LABELS.__getitem__,
        )

    # The search runs on every load with the last submitted values (the defaults at first).
    st.form_submit_button("Search", type="primary")

ta = TaFilters(
    rsi_min=rsi_min,
    rsi_max=rsi_max,
    macd_crossover=macd_crossover,
    ma_touch_period=ma_touch_period,
    adr_min=adr_min,
    gap_type=gap_type,
    momentum_min=momentum_min,
    momentum_max=momentum_max,
    ratio_to_ma_min=ratio_to_ma_min,
    ratio_to_ma_max=ratio_to_ma_max,
    dist_low_126_min_pct=dist_low_min,
    dist_low_126_max_pct=dist_low_max,
    gap_min_pct=float(gap_min_pct),
    gap_lookback=int(gap_lookback),
    momentum_period=int(momentum_period),
    ratio_to_ma_period=int(ratio_to_ma_period),
    timeframe=ta_timeframe,
)

# --- 1. Pre-filters, intersected ---
with st.spinner("Applying pre-filters..."):
    prefilters: list[set[str]] = []
    if etf_rs_rank_min is not None:
        prefilters.append(fetch_etf_rs_tickers(percent_to_fraction(etf_rs_rank_min) or 0.0))
    if option_vol_spike is not None:
        prefilters.append(fetch_option_volume_spike_symbols(option_vol_spike))
    if signal_scans:
        prefilters.append(fetch_signal_scan_symbols(signal_scans))
    if weekly_options_only:
        prefilters.append(fetch_weekly_options_symbols())
    restrict = intersect_prefilters(prefilters)

if restrict is not None and not restrict:
    st.warning("No stocks pass the pre-filters.")
    st.stop()

# --- 2. RS and volume search ---
with st.spinner("Searching..."):
    results = search_stocks_by_metrics(
        rs_rank_min=percent_to_fraction(rs_rank_min),
        rs_rank_max=percent_to_fraction(rs_rank_max),
        rs_rank_overall_min=percent_to_fraction(rs_rank_overall_min),
        rs_rank_overall_max=percent_to_fraction(rs_rank_overall_max),
        rs_z_score_min=rs_z_score_min,
        rs_z_score_max=rs_z_score_max,
        volume_std_dev_min=volume_z_min,
        avg_volume_min=avg_volume_min,
        weinstein_stage=weinstein_stage,
        etf_ticker=etf_ticker,
        restrict_tickers=restrict,
    )
if results.empty:
    st.warning("No stocks match the criteria.")
    st.stop()

# --- 3. Technical filters, events and patterns on one OHLCV fetch ---
ta_info: dict[str, dict[str, Any]] = {}
event_flags: dict[str, dict[str, bool]] = {}
pattern_info: dict[str, list[dict[str, Any]]] = {}
if ta.active or event_scans or pattern_scans:
    with st.spinner("Computing technical indicators..."):
        survivors = results["ticker"].tolist()
        ohlcv = fetch_ohlcv("stock", survivors, last_bars=ta.lookback_bars())
        if ta.active:
            ta_info = apply_ta_filters(ohlcv, survivors, **ta.apply_kwargs())
            survivors = [t for t in survivors if t in ta_info]
        if event_scans:
            event_flags = latest_event_flags_bulk(ohlcv, survivors)
            passing = tickers_with_events(event_flags, event_scans)
            survivors = [t for t in survivors if t in passing]
        if pattern_scans:
            pattern_info = scan_patterns(ohlcv, survivors, patterns=pattern_scans)
            survivors = [t for t in survivors if t in pattern_info]
        results = results[results["ticker"].isin(survivors)]
if results.empty:
    st.warning("No stocks pass the technical, event and pattern filters.")
    st.stop()

# --- 4. Option columns ---
option_table: pd.DataFrame | None = None
if has_options:
    with st.spinner("Loading option metrics..."):
        history = fetch_option_signal_history()
        option_table = option_signal_table(
            fetch_option_metrics_for_tickers(results["ticker"].tolist()),
            {
                "Vanna": scan_vanna_extreme(history=history, enrich=False),
                "Flow": scan_flownet_extreme(history=history, enrich=False),
            },
        )

table = build_results_table(
    results,
    ta_info=ta_info,
    event_flags=event_flags,
    events=event_scans,
    option_table=option_table,
    pattern_info=pattern_info,
)
show_table(table, formats=RESULT_FORMATS)
st.caption(f"{len(table)} results")
tradingview_export(table["Ticker"], filename="screener_watchlist.txt", key="screener_export")
