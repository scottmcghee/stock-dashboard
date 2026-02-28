"""Backtester — replay screener filters over historical data and measure forward returns."""

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import CACHE_TTL_SHORT
from logic.backtest import (
    FILTER_RANGES,
    FILTER_TO_COLUMN,
    REGIMES,
    FilterSpec,
    Strategy,
    StrategyResult,
    _apply_filter,
    build_backtest_dataframe,
    classify_regimes,
    evaluate_all_strategies,
    generate_strategies,
)
from queries.backtest_queries import (
    clear_cache,
    load_metrics_universe,
    load_ohlcv_universe,
    load_spy_ohlcv,
)
from queries.stocks_queries import get_all_etfs

st.header("Screener Backtester")
st.caption(
    "Replay screener filter combinations over historical data. "
    "Measures forward returns (T+5 to T+63) and tests statistical significance."
)

PRESET_DIR = Path("~/.config/stock_dashboard/presets").expanduser()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _get_etfs() -> list[str]:
    return get_all_etfs()


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    etfs = _get_etfs()
    etf_filter = st.selectbox(
        "ETF Filter",
        [None] + etfs,
        format_func=lambda x: "All ETFs" if x is None else x,
        help="Filter stocks to a single ETF's components, or test the full universe.",
    )

with col2:
    max_filters = st.slider(
        "Max filters per strategy",
        min_value=1,
        max_value=3,
        value=2,
        help="Higher = more combos but slower. 1 → ~90, 2 → ~800, 3 → ~3000+.",
    )

with col3:
    regime_options = st.multiselect(
        "Regimes to test",
        ["None (unconditional)", "bull", "bear", "volatile", "correction"],
        default=["None (unconditional)", "bull", "bear"],
        help="Which market regimes to include. 'None' tests without regime filter.",
    )

col4, col5 = st.columns(2)
with col4:
    show_all = st.checkbox("Show all strategies (including non-significant)", value=False)
with col5:
    sort_by_horizon = st.selectbox("Sort by horizon", ["T+5", "T+10", "T+20", "T+63"], index=2)

sort_metric = st.radio(
    "Sort metric", ["mean_return", "median_return", "sharpe", "win_rate"], horizontal=True
)

btn_col1, btn_col2, btn_col3 = st.columns(3)
with btn_col1:
    if st.button("Clear data cache"):
        n = clear_cache()
        st.toast(f"Cleared {n} cached files.")
with btn_col2:
    run = st.button("Run Backtest", type="primary")
with btn_col3:
    if st.button("Clear Results"):
        for key in ["bt_indicators_df", "bt_results", "bt_results_df", "bt_regime_series", "bt_spy_ohlcv"]:
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------------------------------------------------------
# Data loading & computation (only on Run click)
# ---------------------------------------------------------------------------

if run:
    progress = st.progress(0, text="Loading OHLCV data...")

    ohlcv = load_ohlcv_universe(etf_filter)
    if ohlcv.empty:
        st.error("No OHLCV data found. Check your ETF filter and database.")
        st.stop()

    progress.progress(15, text="Loading RS metrics...")
    metrics = load_metrics_universe(etf_filter)

    progress.progress(30, text="Loading SPY for regime classification...")
    spy = load_spy_ohlcv()

    # Build indicator dataframe
    progress.progress(40, text="Pre-computing indicators (this may take a minute)...")
    indicators_df = build_backtest_dataframe(ohlcv, metrics if not metrics.empty else None)

    # Free raw data — no longer needed after merge
    del ohlcv, metrics, spy
    gc.collect()

    progress.progress(55, text="Reloading SPY for regime classification...")
    spy = load_spy_ohlcv()

    progress.progress(65, text="Classifying market regimes...")
    regime_series = classify_regimes(spy) if not spy.empty else None
    st.session_state["bt_spy_ohlcv"] = spy
    del spy
    gc.collect()

    # Generate & evaluate strategies
    progress.progress(70, text="Generating strategies...")

    selected_regimes: list[str | None] = []
    for r in regime_options:
        if r == "None (unconditional)":
            selected_regimes.append(None)
        else:
            selected_regimes.append(r)

    all_strategies = generate_strategies(max_filters=max_filters)
    strategies = [s for s in all_strategies if s.regime in selected_regimes]

    n_total = len(strategies)
    st.text(f"Evaluating {n_total:,} strategies...")

    progress.progress(75, text=f"Evaluating {n_total:,} strategies...")
    results = evaluate_all_strategies(indicators_df, strategies, regime_series)

    progress.progress(95, text="Ranking results...")

    # Store in session state
    st.session_state["bt_indicators_df"] = indicators_df
    st.session_state["bt_results"] = results
    st.session_state["bt_regime_series"] = regime_series

    progress.progress(100, text="Done!")

# ---------------------------------------------------------------------------
# Results display (from session state)
# ---------------------------------------------------------------------------

if "bt_results" not in st.session_state:
    st.info("Configure filters above and click **Run Backtest** to start.")
    st.stop()

indicators_df = st.session_state["bt_indicators_df"]
results: list[StrategyResult] = st.session_state["bt_results"]
regime_series = st.session_state.get("bt_regime_series")
spy_ohlcv = st.session_state.get("bt_spy_ohlcv")

# ---------------------------------------------------------------------------
# Regime context chart (SPY + regime shading)
# ---------------------------------------------------------------------------

if regime_series is not None and spy_ohlcv is not None and not spy_ohlcv.empty:
    with st.expander("SPY / Market Regime Context", expanded=False):
        spy_sorted = spy_ohlcv.sort_values("date").set_index("date")
        fig_regime = go.Figure()
        fig_regime.add_trace(go.Scatter(
            x=spy_sorted.index,
            y=spy_sorted["close"],
            mode="lines",
            name="SPY",
            line=dict(color="white", width=1.5),
        ))

        # Shade regime periods
        regime_colors = {
            "bull": "rgba(0,200,0,0.15)",
            "bear": "rgba(200,0,0,0.15)",
            "volatile": "rgba(255,165,0,0.15)",
            "correction": "rgba(200,0,0,0.30)",
        }
        rs = regime_series.dropna()
        if not rs.empty:
            # Build contiguous regime blocks
            prev_regime = rs.iloc[0]
            block_start = rs.index[0]
            for dt, regime in rs.items():
                if regime != prev_regime:
                    if prev_regime in regime_colors:
                        fig_regime.add_vrect(
                            x0=block_start, x1=dt,
                            fillcolor=regime_colors[prev_regime],
                            line_width=0, layer="below",
                        )
                    prev_regime = regime
                    block_start = dt
            # Final block
            if prev_regime in regime_colors:
                fig_regime.add_vrect(
                    x0=block_start, x1=rs.index[-1],
                    fillcolor=regime_colors[prev_regime],
                    line_width=0, layer="below",
                )

        fig_regime.update_layout(
            title="SPY with Market Regime Classification",
            xaxis_title="Date",
            yaxis_title="SPY Close",
            height=350,
            legend=dict(orientation="h", y=1.12),
            margin=dict(t=50, b=40),
        )
        # Add legend entries for regime colors
        for regime, color in regime_colors.items():
            fig_regime.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color.replace("0.15", "0.6").replace("0.30", "0.6")),
                name=regime.capitalize(),
                showlegend=True,
            ))
        st.plotly_chart(fig_regime, use_container_width=True)

# Filter to significant or show all
display_results = results
if not show_all:
    display_results = [
        r for r in display_results
        if r.n_signals >= 10
        and any(hs.significant for hs in r.horizons.values())
    ]

if not display_results:
    st.warning("No strategies met significance criteria. Try enabling 'Show all'.")
    st.stop()

# Build results table
table_rows = []
for r in display_results:
    row: dict = {
        "strategy": r.strategy.name,
        "regime": r.strategy.regime or "any",
        "n_signals": r.n_signals,
    }
    for label in ["T+5", "T+10", "T+20", "T+63"]:
        if label in r.horizons:
            hs = r.horizons[label]
            row[f"{label}_mean"] = round(hs.mean_return * 100, 2)
            row[f"{label}_med"] = round(hs.median_return * 100, 2)
            row[f"{label}_wr"] = round(hs.win_rate * 100, 1)
            row[f"{label}_sharpe"] = round(hs.sharpe, 3)
            row[f"{label}_pval"] = round(hs.p_value_adj, 4)
            row[f"{label}_sig"] = hs.significant
        else:
            row[f"{label}_mean"] = None
            row[f"{label}_med"] = None
            row[f"{label}_wr"] = None
            row[f"{label}_sharpe"] = None
            row[f"{label}_pval"] = None
            row[f"{label}_sig"] = False
    table_rows.append(row)

results_df = pd.DataFrame(table_rows)

# Sort
_METRIC_TO_COL_SUFFIX = {
    "mean_return": "mean",
    "median_return": "med",
    "sharpe": "sharpe",
    "win_rate": "wr",
}
sort_col = f"{sort_by_horizon}_{_METRIC_TO_COL_SUFFIX.get(sort_metric, sort_metric)}"
if sort_col in results_df.columns:
    results_df = results_df.sort_values(sort_col, ascending=False, na_position="last")
    results_df = results_df.reset_index(drop=True)

st.subheader(f"Results ({len(results_df):,} strategies)")
st.caption("Click a row to inspect strategy details below.")

# Clickable row selection
event = st.dataframe(
    results_df,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "strategy": st.column_config.TextColumn("Strategy", width="large"),
        "regime": st.column_config.TextColumn("Regime"),
        "n_signals": st.column_config.NumberColumn("Signals", format="%d"),
        "T+5_mean": st.column_config.NumberColumn("T+5 Mean%", format="%.2f"),
        "T+5_med": st.column_config.NumberColumn("T+5 Med%", format="%.2f"),
        "T+5_wr": st.column_config.NumberColumn("T+5 WR%", format="%.1f"),
        "T+5_sharpe": st.column_config.NumberColumn("T+5 Sharpe", format="%.3f"),
        "T+10_mean": st.column_config.NumberColumn("T+10 Mean%", format="%.2f"),
        "T+10_med": st.column_config.NumberColumn("T+10 Med%", format="%.2f"),
        "T+10_wr": st.column_config.NumberColumn("T+10 WR%", format="%.1f"),
        "T+10_sharpe": st.column_config.NumberColumn("T+10 Sharpe", format="%.3f"),
        "T+20_mean": st.column_config.NumberColumn("T+20 Mean%", format="%.2f"),
        "T+20_med": st.column_config.NumberColumn("T+20 Med%", format="%.2f"),
        "T+20_wr": st.column_config.NumberColumn("T+20 WR%", format="%.1f"),
        "T+20_sharpe": st.column_config.NumberColumn("T+20 Sharpe", format="%.3f"),
        "T+63_mean": st.column_config.NumberColumn("T+63 Mean%", format="%.2f"),
        "T+63_med": st.column_config.NumberColumn("T+63 Med%", format="%.2f"),
        "T+63_wr": st.column_config.NumberColumn("T+63 WR%", format="%.1f"),
        "T+63_sharpe": st.column_config.NumberColumn("T+63 Sharpe", format="%.3f"),
    },
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Detail view (driven by row selection)
# ---------------------------------------------------------------------------

selected_rows = event.selection.rows if event.selection else []
if not selected_rows:
    st.info("Select a row above to see strategy details.")
    st.stop()

selected_idx = selected_rows[0]
selected_strategy_name = results_df.iloc[selected_idx]["strategy"]

# Find the matching StrategyResult
selected_result: StrategyResult | None = None
for r in display_results:
    if r.strategy.name == selected_strategy_name:
        selected_result = r
        break

if not selected_result or selected_result.n_signals < 10:
    st.warning("Not enough signals for detail view.")
    st.stop()

st.subheader(f"Strategy Detail: {selected_strategy_name}")

# Reconstruct the signal mask
mask = pd.Series(True, index=indicators_df.index)
for fspec in selected_result.strategy.filters:
    col = FILTER_TO_COLUMN.get(fspec.name)
    if col and col in indicators_df.columns:
        mask = mask & _apply_filter(indicators_df[col], fspec)

if selected_result.strategy.regime and regime_series is not None:
    date_regime = indicators_df["date"].map(regime_series)
    mask = mask & (date_regime == selected_result.strategy.regime)

signal_data = indicators_df.loc[mask]

tab1, tab2, tab3 = st.tabs(["Cumulative Returns", "Distribution", "Regime Breakdown"])

with tab1:
    horizon_col = f"fwd_{sort_by_horizon.replace('T+', '')}"
    if horizon_col in signal_data.columns:
        daily_signals = (
            signal_data.groupby("date")[horizon_col]
            .mean()
            .sort_index()
            .dropna()
        )
        if not daily_signals.empty:
            cumulative = (1 + daily_signals).cumprod() - 1
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative.values * 100,
                mode="lines",
                name=f"Cumulative {sort_by_horizon} Return",
            ))
            fig.update_layout(
                title=f"Cumulative Signal Returns ({sort_by_horizon})",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    if horizon_col in signal_data.columns:
        returns = signal_data[horizon_col].dropna() * 100
        if not returns.empty:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name=f"{sort_by_horizon} Returns",
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.add_vline(x=returns.mean(), line_dash="dot", line_color="green",
                         annotation_text=f"Mean: {returns.mean():.2f}%")
            fig.update_layout(
                title=f"{sort_by_horizon} Forward Return Distribution",
                xaxis_title="Return (%)",
                yaxis_title="Count",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    if regime_series is not None:
        date_regime = signal_data["date"].map(regime_series)
        regime_rows = []
        for regime_label in ["bull", "bear", "volatile", "correction"]:
            regime_mask = date_regime == regime_label
            regime_signals = signal_data.loc[regime_mask]
            if len(regime_signals) < 5:
                continue
            regime_row = {"regime": regime_label, "n_signals": len(regime_signals)}
            for h_col, h_label in [("fwd_5", "T+5"), ("fwd_10", "T+10"),
                                   ("fwd_20", "T+20"), ("fwd_63", "T+63")]:
                if h_col in regime_signals.columns:
                    rets = regime_signals[h_col].dropna()
                    if len(rets) > 0:
                        regime_row[f"{h_label}_mean%"] = round(float(rets.mean()) * 100, 2)
                        regime_row[f"{h_label}_wr%"] = round(float((rets > 0).mean()) * 100, 1)
            regime_rows.append(regime_row)

        if regime_rows:
            st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough signals in individual regimes for breakdown.")

# Horizon stats summary
st.subheader("Horizon Statistics")
stats_rows = []
for label, hs in selected_result.horizons.items():
    stats_rows.append({
        "Horizon": label,
        "Mean Return %": round(hs.mean_return * 100, 2),
        "Median Return %": round(hs.median_return * 100, 2),
        "Win Rate %": round(hs.win_rate * 100, 1),
        "Sharpe": round(hs.sharpe, 3),
        "p-value (adj)": round(hs.p_value_adj, 4),
        "Significant": hs.significant,
    })
if stats_rows:
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Preset export
# ---------------------------------------------------------------------------

st.subheader("Export as Preset")
preset_name = st.text_input(
    "Preset name",
    value=selected_result.strategy.name[:50],
)

if st.button("Save Preset"):
    preset = {
        "name": preset_name,
        "filters": [
            {"name": f.name, "value": f.value}
            for f in selected_result.strategy.filters
        ],
        "regime": selected_result.strategy.regime,
        "stats": {
            label: {
                "mean_return": round(hs.mean_return * 100, 2),
                "win_rate": round(hs.win_rate, 3),
                "sharpe": round(hs.sharpe, 3),
                "p_value_adj": round(hs.p_value_adj, 4),
            }
            for label, hs in selected_result.horizons.items()
        },
    }
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    filename = preset_name.replace(" ", "_").replace("/", "_")[:50] + ".json"
    preset_path = PRESET_DIR / filename
    preset_path.write_text(json.dumps(preset, indent=2))
    st.success(f"Saved preset to `{preset_path}`")
