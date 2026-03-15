"""Stock Screener — multi-criteria search with bulk Weinstein classification and advanced filters."""

import pandas as pd
import streamlit as st

from config import CACHE_TTL_LONG, CACHE_TTL_SHORT
from logic.preset import list_presets, parse_preset, PRESET_DIR
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
    "washout_bull",
    "washout_bear",
    "coiled_spring",
]

MA_PERIOD_OPTIONS = [None, 5, 8, 10, 20, 21, 50, 100, 200]

# Preset selector UI
preset: dict = st.session_state.get("_preset", {})
_preset_names = list_presets()

if _preset_names:
    _selected = st.selectbox(
        "Load Backtester Preset",
        ["(none)"] + _preset_names,
        format_func=lambda x: x.replace("_", " "),
        key="_preset_select",
    )
    if _selected != "(none)" and _selected != preset.get("_stem"):
        preset = parse_preset(_selected)
        st.session_state["_preset"] = preset
        st.session_state["_screener_reset"] = st.session_state.get("_screener_reset", 0) + 1
        st.rerun()
    elif _selected == "(none)" and preset:
        st.session_state.pop("_preset", None)
        preset = {}
        st.session_state["_screener_reset"] = st.session_state.get("_screener_reset", 0) + 1
        st.rerun()

    if preset.get("_regime"):
        st.caption(f"Backtest regime: **{preset['_regime'].title()}**")
    if preset.get("_unmapped"):
        st.info(f"Unmapped filters (not in screener): {', '.join(preset['_unmapped'])}")

# --- Reset Button ---
if st.button("Reset Filters"):
    st.session_state["_screener_reset"] = st.session_state.get("_screener_reset", 0) + 1
    st.session_state.pop("_preset", None)
    st.session_state.pop("_preset_select", None)
    st.rerun()

_form_key = f"screener_form_{st.session_state.get('_screener_reset', 0)}"

# --- Filter Form ---
with st.form(_form_key):
    col1, col2, col3 = st.columns(3)
    with col1:
        etfs = _get_etfs()
        etf_ticker = st.selectbox("ETF Filter", [None] + etfs, format_func=lambda x: "All ETFs" if x is None else x)
        rs_rank_min = st.number_input("RS Rank Min (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, format="%.1f")
        rs_rank_max = st.number_input("RS Rank Max (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, format="%.1f")
    with col2:
        rs_rank_overall_min = st.number_input("RS Rank Overall Min (%)", min_value=0.0, max_value=100.0, value=preset.get("rs_rank_overall_min"), step=1.0, format="%.1f")
        rs_rank_overall_max = st.number_input("RS Rank Overall Max (%)", min_value=0.0, max_value=100.0, value=preset.get("rs_rank_overall_max"), step=1.0, format="%.1f")
        rs_z_score_min = st.number_input("RS Z-Score Min", value=preset.get("rs_z_score_min"), step=0.1, format="%.2f")
        rs_z_score_max = st.number_input("RS Z-Score Max", value=preset.get("rs_z_score_max"), step=0.1, format="%.2f")
    with col3:
        volume_std_dev_min = st.number_input("Volume Z-Score Min", value=preset.get("volume_std_dev_min"), step=0.1, format="%.1f")
        avg_volume_min = st.number_input("Avg 30-Day Volume Min", value=None, step=10000.0, format="%.0f")
        _stages = [None, "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
        _stage_val = preset.get("weinstein_stage")
        _stage_idx = _stages.index(_stage_val) if _stage_val in _stages else 0
        weinstein_stage = st.selectbox(
            "Weinstein Stage",
            _stages,
            index=_stage_idx,
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
            _mom_periods = [20, 63, 126]
            _mom_default = preset.get("momentum_period", 63)
            _mom_idx = _mom_periods.index(_mom_default) if _mom_default in _mom_periods else 1
            momentum_period = st.selectbox(
                "Momentum Period", _mom_periods, index=_mom_idx,
                format_func=lambda x: f"{x}-day",
            )
            momentum_min = st.number_input(
                "Momentum Min (%)", value=preset.get("momentum_min"), step=5.0, format="%.1f",
                help="Minimum rate-of-change over the momentum period",
            )
            momentum_max = st.number_input(
                "Momentum Max (%)", value=preset.get("momentum_max"), step=5.0, format="%.1f",
                help="Maximum rate-of-change over the momentum period",
            )
            _ratio_periods = [50, 200]
            _ratio_default = preset.get("ratio_to_ma_period", 200)
            _ratio_idx = _ratio_periods.index(_ratio_default) if _ratio_default in _ratio_periods else 1
            ratio_to_ma_period = st.selectbox(
                "Ratio to MA Period", _ratio_periods, index=_ratio_idx,
                format_func=lambda x: f"{x}-day SMA",
            )
            ratio_to_ma_min = st.number_input(
                "Ratio to MA Min", value=preset.get("ratio_to_ma_min"), step=0.05, format="%.3f",
                help="Minimum close / SMA ratio (e.g. 1.0 = at the MA)",
            )
            ratio_to_ma_max = st.number_input(
                "Ratio to MA Max", value=preset.get("ratio_to_ma_max"), step=0.05, format="%.3f",
                help="Maximum close / SMA ratio",
            )
        with adv2:
            rsi_min = st.number_input("RSI Min", min_value=0, max_value=100, value=preset.get("rsi_min"), step=5)
            rsi_max = st.number_input("RSI Max", min_value=0, max_value=100, value=preset.get("rsi_max"), step=5)
            ma_touch_period = st.selectbox(
                "Price Touching MA",
                MA_PERIOD_OPTIONS,
                format_func=lambda x: "None" if x is None else f"{x}-day SMA",
            )
            ta_timeframe = st.selectbox("TA Timeframe", ["D", "W", "M"], format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x])
            macd_crossover = st.selectbox(
                "MACD Crossover",
                [None, "bullish", "bearish", "not bullish", "not bearish"],
                format_func=lambda x: "Any" if x is None else x.replace("not ", "Not ").title(),
            )
            adr_min = st.number_input(
                "ADR% Min", min_value=0, max_value=100, value=preset.get("adr_min"), step=1,
                help="Minimum average daily range as % of close (14-day)",
            )
            gap_type = st.selectbox(
                "Gap Detection",
                [None, "bull", "bear", "any"],
                format_func=lambda x: "None" if x is None else {"bull": "Bull Gap (up)", "bear": "Bear Gap (down)", "any": "Any Gap"}[x],
                help="Detect recent price gaps within lookback window",
            )
            gap_min_pct = st.number_input(
                "Gap Min %", min_value=0.0, max_value=20.0, value=1.0, step=0.5, format="%.1f",
                help="Minimum gap size as % of previous close",
            )
            gap_lookback = st.number_input(
                "Gap Lookback (bars)", min_value=1, max_value=20, value=5, step=1,
                help="Number of bars to search for a gap",
            )
        pattern_scans = st.multiselect(
            "Chart Patterns",
            options=["high_tight_flag", "bull_flag", "bear_flag", "cup_and_handle", "v_bottom"],
            default=preset.get("pattern_scans", []),
            format_func=lambda x: x.replace("_", " ").title(),
            help="Only show stocks matching at least one selected pattern",
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
        rs_rank_min=rs_rank_min / 100.0 if rs_rank_min is not None else None,
        rs_rank_max=rs_rank_max / 100.0 if rs_rank_max is not None else None,
        rs_rank_overall_min=rs_rank_overall_min / 100.0 if rs_rank_overall_min is not None else None,
        rs_rank_overall_max=rs_rank_overall_max / 100.0 if rs_rank_overall_max is not None else None,
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
    or gap_type is not None
    or momentum_min is not None
    or momentum_max is not None
    or ratio_to_ma_min is not None
    or ratio_to_ma_max is not None
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
                gap_type=gap_type,
                gap_min_pct=float(gap_min_pct),
                gap_lookback=int(gap_lookback),
                momentum_min=float(momentum_min) if momentum_min is not None else None,
                momentum_max=float(momentum_max) if momentum_max is not None else None,
                momentum_period=int(momentum_period),
                ratio_to_ma_min=float(ratio_to_ma_min) if ratio_to_ma_min is not None else None,
                ratio_to_ma_max=float(ratio_to_ma_max) if ratio_to_ma_max is not None else None,
                ratio_to_ma_period=int(ratio_to_ma_period),
                timeframe=ta_timeframe,
            )
            results = results[results["ticker"].isin(ta_info.keys())]

if results.empty:
    st.warning("No stocks found after applying technical filters.")
    st.stop()

# --- Fetch option metrics + signal labels for surviving tickers ---
option_metrics: dict[str, dict] = {}
signal_labels: dict[str, dict[str, str]] = {}  # ticker -> {scan_name: label}
surviving_tickers_for_opts = results["ticker"].unique().tolist()
if surviving_tickers_for_opts:
    from queries.screener_queries import fetch_option_metrics_for_tickers
    from queries.signal_queries import scan_gex_level, scan_vanna_extreme, scan_flownet_extreme

    opt_df = fetch_option_metrics_for_tickers(surviving_tickers_for_opts)
    if not opt_df.empty:
        for _, row in opt_df.iterrows():
            option_metrics[row["symbol"]] = row.to_dict()

    # Build signal label maps from scan results
    _survivor_set = set(surviving_tickers_for_opts)
    for scan_fn, col_name in [
        (scan_gex_level, "gex"),
        (scan_vanna_extreme, "vanna"),
        (scan_flownet_extreme, "flow"),
    ]:
        scan_df = scan_fn()
        if not scan_df.empty:
            for _, row in scan_df.iterrows():
                sym = row["symbol"]
                if sym in _survivor_set:
                    signal_labels.setdefault(sym, {})[col_name] = row["signal_label"]

# --- Phase 4: Pattern scanning (only on survivors) ---
pattern_info: dict[str, list[dict]] = {}
if pattern_scans:
    with st.spinner("Scanning for chart patterns..."):
        from queries.screener_queries import fetch_bulk_ohlcv_for_ta
        from logic.patterns import scan_patterns

        surviving_tickers = results["ticker"].unique().tolist()
        if "ohlcv" not in dir() or ohlcv.empty:
            ohlcv = fetch_bulk_ohlcv_for_ta(surviving_tickers)

        if not ohlcv.empty:
            pattern_info = scan_patterns(ohlcv, surviving_tickers, patterns=pattern_scans)
            results = results[results["ticker"].isin(pattern_info.keys())]

    if results.empty:
        st.warning("No stocks found matching the selected chart patterns.")
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
    if any(info.get("gap") is not None for info in ta_info.values()):
        display_df["Gap"] = [
            f"{ta_info.get(t, {}).get('gap', {}).get('type', '').title()} {ta_info.get(t, {}).get('gap', {}).get('gap_pct', '')}%"
            if ta_info.get(t, {}).get("gap") is not None else "N/A"
            for t in tickers
        ]
    if any(info.get("momentum") is not None for info in ta_info.values()):
        display_df["Momentum%"] = [
            f"{ta_info.get(t, {}).get('momentum', None):.1f}%" if ta_info.get(t, {}).get("momentum") is not None else "N/A"
            for t in tickers
        ]
    if any(info.get("ratio_to_ma") is not None for info in ta_info.values()):
        display_df["Ratio-MA"] = [
            f"{ta_info.get(t, {}).get('ratio_to_ma', None):.3f}" if ta_info.get(t, {}).get("ratio_to_ma") is not None else "N/A"
            for t in tickers
        ]

# Add option metric columns when available
if option_metrics or signal_labels:
    tickers = display_df["Ticker"].tolist()
    # Numeric option columns
    _opt_cols = [
        ("vol_multiple", "Opt Vol (x)", lambda v: f"{v:.1f}x"),
        ("put_call_ratio", "P/C Ratio", lambda v: f"{v:.2f}"),
        ("dxoi", "dxOI", lambda v: f"{v:,.0f}"),
    ]
    for key, label, fmt in _opt_cols:
        if any(
            option_metrics.get(t, {}).get(key) is not None
            and pd.notnull(option_metrics.get(t, {}).get(key))
            for t in tickers
        ):
            display_df[label] = [
                fmt(option_metrics[t][key])
                if t in option_metrics and option_metrics[t].get(key) is not None and pd.notnull(option_metrics[t][key])
                else "N/A"
                for t in tickers
            ]
    # Signal label columns
    for scan_key, col_label in [("gex", "GEX"), ("vanna", "Vanna"), ("flow", "Flow")]:
        if any(signal_labels.get(t, {}).get(scan_key) for t in tickers):
            display_df[col_label] = [
                signal_labels.get(t, {}).get(scan_key, "—")
                for t in tickers
            ]

# Add pattern columns when active
if pattern_info:
    tickers = display_df["Ticker"].tolist()
    display_df["Patterns"] = [
        ", ".join(p["pattern"].replace("_", " ").title() for p in pattern_info.get(t, []))
        if pattern_info.get(t) else "N/A"
        for t in tickers
    ]

st.dataframe(display_df, use_container_width=True, hide_index=True)
st.caption(f"{len(display_df)} results")

# --- TradingView Watchlist Export ---
from logic.export import build_tradingview_watchlist
from queries.stocks_queries import fetch_exchange_map

_export_tickers = display_df["Ticker"].tolist()
_exchange_map = fetch_exchange_map(_export_tickers)
_watchlist = build_tradingview_watchlist(_export_tickers, _exchange_map)
st.download_button("Export TradingView Watchlist", _watchlist, "watchlist.txt", "text/plain")
