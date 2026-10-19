"""Backtester: replay screener filter combinations over history and test forward returns."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_dashboard.config import CACHE_TTL_SHORT, LIMITATIONS_URL, get_settings
from stock_dashboard.logic.backtest import (
    MIN_SIGNALS,
    StrategyResult,
    build_backtest_dataframe,
    classify_regimes,
    evaluate_all_strategies,
    generate_strategies,
)
from stock_dashboard.logic.backtest_report import (
    ANY_REGIME_LABEL,
    HORIZONS,
    REGIME_LABELS,
    SORT_METRIC_LABELS,
    SORT_METRICS,
    cumulative_signal_returns,
    format_p_value,
    horizon_column,
    horizon_stats_table,
    preset_from_result,
    regime_blocks,
    regime_breakdown,
    regime_choices,
    results_column_config_formats,
    results_table,
    sort_results,
    strategy_signals,
)
from stock_dashboard.logic.preset import preset_dir, save_preset
from stock_dashboard.queries.backtest_queries import (
    clear_cache,
    load_benchmark_ohlcv,
    load_metrics_universe,
    load_ohlcv_universe,
)
from stock_dashboard.queries.stocks_queries import get_all_etfs
from stock_dashboard.ui.tables import selected_row, selection_key, show_table

STATE_KEYS = ("bt_indicators", "bt_results", "bt_regimes", "bt_benchmark", "bt_etf")
REGIME_COLORS = {
    "bull": "rgba(0,200,0,0.15)",
    "bear": "rgba(200,0,0,0.15)",
    "volatile": "rgba(255,165,0,0.15)",
    "correction": "rgba(200,0,0,0.30)",
}
LEGEND_OPACITY = 0.6
HISTOGRAM_BINS = 50
CHART_HEIGHT = 400
PRESET_NAME_LENGTH = 60


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _etfs() -> list[str]:
    return get_all_etfs()


benchmark = get_settings().benchmark
st.header("Backtester")
st.caption(
    "Replays filter combinations over the stored history and measures forward returns "
    f"(T+{HORIZONS[0].removeprefix('T+')} to {HORIZONS[-1]} sessions). Each strategy and "
    "horizon is tested against a zero mean (one-sample t-test), with the Benjamini-Hochberg "
    f"false-discovery-rate correction across all tests. Market regimes come from {benchmark}. "
    "Mean/SD is per signal, not annualized."
)
st.caption(
    "Results are in-sample; signals overlap, so p-values are optimistic. Entry is at the "
    "signal bar's close, with no costs; the universe and ETF memberships are as of today. "
    f"See [the limitations]({LIMITATIONS_URL})."
)

col1, col2, col3 = st.columns(3)
with col1:
    etf_filter = st.selectbox(
        "ETF",
        [None, *_etfs()],
        format_func=lambda x: "All stocks" if x is None else x,
        help="Backtest one ETF's members or every stock.",
    )
with col2:
    max_filters = st.slider(
        "Max filters per strategy",
        1,
        3,
        1,
        help="Each extra filter multiplies the number of combinations (and the run time).",
    )
with col3:
    regime_labels = st.multiselect(
        "Regimes to test",
        [ANY_REGIME_LABEL, *REGIME_LABELS],
        default=[ANY_REGIME_LABEL],
        help=f"'{ANY_REGIME_LABEL}' tests without a regime condition.",
    )

col4, col5, col6 = st.columns(3)
with col4:
    show_all = st.checkbox("Show all strategies (including non-significant)")
with col5:
    sort_horizon = st.selectbox("Sort by horizon", HORIZONS, index=2)
with col6:
    sort_metric = st.selectbox(
        "Sort metric", SORT_METRICS, format_func=SORT_METRIC_LABELS.__getitem__
    )

btn1, btn2, btn3 = st.columns(3)
with btn1:
    if st.button("Clear data cache"):
        st.toast(f"Cleared {clear_cache()} cached files.")
with btn2:
    run = st.button("Run Backtest", type="primary")
with btn3:
    if st.button("Clear Results"):
        for key in STATE_KEYS:
            st.session_state.pop(key, None)
        st.rerun()

if run:
    progress = st.progress(0, text="Loading prices...")
    ohlcv = load_ohlcv_universe(etf_filter)
    if ohlcv.empty:
        st.error("No price data found for this selection.")
        st.stop()
    progress.progress(15, text="Loading RS metrics...")
    metrics = load_metrics_universe(etf_filter)
    progress.progress(30, text=f"Loading {benchmark} for regime classification...")
    benchmark_bars = load_benchmark_ohlcv()
    regimes = classify_regimes(benchmark_bars) if not benchmark_bars.empty else None
    progress.progress(40, text="Computing indicators...")
    indicators = build_backtest_dataframe(ohlcv, metrics if not metrics.empty else None)
    del ohlcv, metrics  # the indicator frame replaces them

    wanted = regime_choices(regime_labels)
    if regimes is None:
        wanted = [None]  # a regime condition cannot be evaluated without regimes
    strategies = [s for s in generate_strategies(max_filters) if s.regime in wanted]
    progress.progress(70, text=f"Evaluating {len(strategies):,} strategies...")
    evaluated = evaluate_all_strategies(indicators, strategies, regimes)
    progress.progress(100, text="Done")
    st.session_state.update(
        bt_indicators=indicators,
        bt_results=evaluated,
        bt_regimes=regimes,
        bt_benchmark=benchmark_bars,
        bt_etf=etf_filter,
    )

if "bt_results" not in st.session_state:
    st.info("Choose the settings above and click **Run Backtest**.")
    st.stop()

indicators = st.session_state["bt_indicators"]
results: list[StrategyResult] = st.session_state["bt_results"]
regimes = st.session_state["bt_regimes"]
benchmark_bars = st.session_state["bt_benchmark"]

# --- Regime context ---
if regimes is None:
    st.warning(
        f"No price bars for the benchmark {benchmark} (BENCHMARK_TICKER), so market regimes "
        "cannot be classified: only unconditional strategies were evaluated."
    )
elif not benchmark_bars.empty:
    with st.expander(f"{benchmark} and market regimes"):
        bars = benchmark_bars.sort_values("date")
        fig = go.Figure(go.Scatter(x=bars["date"], y=bars["close"], name=benchmark))
        for block in regime_blocks(regimes):
            fig.add_vrect(
                x0=block.start,
                x1=block.end,
                fillcolor=REGIME_COLORS.get(block.regime, "rgba(0,0,0,0)"),
                line_width=0,
                layer="below",
            )
        for regime, color in REGIME_COLORS.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker={"size": 10, "color": color, "opacity": LEGEND_OPACITY},
                    name=regime.capitalize(),
                )
            )
        fig.update_layout(
            title=f"{benchmark} with market regime classification",
            yaxis_title=f"{benchmark} close",
            height=350,
            legend={"orientation": "h", "y": 1.12},
        )
        st.plotly_chart(fig)

# --- Results ---
table = sort_results(results_table(results, show_all=show_all), sort_horizon, sort_metric)
if table.empty:
    st.warning("No strategy met the significance criteria. Try 'Show all strategies'.")
    st.stop()

st.subheader(f"Results ({len(table):,} strategies)")
result_formats = results_column_config_formats()
p_columns = [col for col, (_, fmt) in result_formats.items() if fmt is None]
selection = show_table(
    # Styled, so p-values stay numbers (and sort as numbers) but show as "<0.0001".
    table.style.format(format_p_value, subset=pd.Index(p_columns)),
    formats={col: fmt for col, (_, fmt) in result_formats.items() if fmt is not None},
    labels={
        **{col: label for col, (label, _) in result_formats.items()},
        **{f"{h}_sig": f"{h} Significant" for h in HORIZONS},
    },
    on_select="rerun",
    selection_mode="single-row",
    key=selection_key("bt_table", table["strategy"]),
)
strategy_name = st.selectbox(
    "Strategy to inspect",
    table["strategy"].tolist(),
    index=selected_row(selection, len(table)),
    placeholder="Select a strategy (or click a row)...",
)
if strategy_name is None:
    st.stop()

selected = next(r for r in results if r.strategy.name == strategy_name)
if selected.n_signals < MIN_SIGNALS:
    st.warning(f"Fewer than {MIN_SIGNALS} signals: no detail view.")
    st.stop()

st.subheader(f"Strategy detail: {strategy_name}")
signals = strategy_signals(indicators, selected, regimes)
tab1, tab2, tab3 = st.tabs(["Cumulative Returns", "Distribution", "Regime Breakdown"])
with tab1:
    cumulative = cumulative_signal_returns(signals, sort_horizon)
    if cumulative.empty:
        st.info("No forward returns at this horizon.")
    else:
        fig = go.Figure(go.Scatter(x=cumulative.index, y=cumulative.to_numpy(), mode="lines"))
        fig.update_layout(
            title=f"Compounded mean {sort_horizon} return of each signal date",
            yaxis_title="Cumulative return (%)",
            height=CHART_HEIGHT,
        )
        st.plotly_chart(fig)
with tab2:
    returns = signals[horizon_column(sort_horizon)].dropna() * 100
    if returns.empty:
        st.info("No forward returns at this horizon.")
    else:
        fig = go.Figure(go.Histogram(x=returns, nbinsx=HISTOGRAM_BINS))
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.add_vline(
            x=returns.mean(),
            line_dash="dot",
            line_color="green",
            annotation_text=f"Mean: {returns.mean():.2f}%",
        )
        fig.update_layout(
            title=f"{sort_horizon} forward returns",
            xaxis_title="Return (%)",
            yaxis_title="Count",
            height=CHART_HEIGHT,
        )
        st.plotly_chart(fig)
with tab3:
    breakdown = regime_breakdown(signals, regimes) if regimes is not None else None
    if breakdown is None or breakdown.empty:
        st.info("Not enough signals in any single regime for a breakdown.")
    else:
        formats = {c: "%.2f" for c in breakdown.columns if c.endswith("%")}
        show_table(breakdown, formats=formats)

st.subheader("Horizon Statistics")
stats_formats = {"Mean/SD": "%.3f"}
stats_formats.update(dict.fromkeys(("Mean Return %", "Median Return %", "Win Rate %"), "%.2f"))
show_table(
    horizon_stats_table(selected).style.format(format_p_value, subset=["p-value (BH)"]),
    formats=stats_formats,
)

st.subheader("Save as Preset")
preset_name = st.text_input("Preset name", value=strategy_name[:PRESET_NAME_LENGTH])
if st.button("Save Preset"):
    try:
        preset = preset_from_result(selected, preset_name, etf=st.session_state.get("bt_etf"))
        path = save_preset(preset, preset_dir())
    except (ValueError, OSError) as exc:
        st.error(f"Could not save the preset: {exc}")
    else:
        st.success(f"Saved preset to `{path}`")
