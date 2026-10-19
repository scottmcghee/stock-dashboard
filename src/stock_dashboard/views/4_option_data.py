"""Option data: one underlying's recent option metrics, their ratio to average, and history."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_dashboard.config import CACHE_TTL_DEFAULT, CACHE_TTL_SHORT
from stock_dashboard.logic.option_views import latest_vs_average_row, underlying_formats
from stock_dashboard.queries.option_contract_queries import fetch_most_open_interest_symbol
from stock_dashboard.queries.option_queries import (
    KEY_COLUMNS,
    RECENT_ROWS,
    fetch_option_column_history,
    fetch_option_last_30_days,
    fetch_option_latest_comparison,
    get_all_option_symbols,
    get_option_multiple_columns,
    get_option_numerical_columns,
)
from stock_dashboard.ui.charts import LEGEND_ABOVE
from stock_dashboard.ui.option_guard import require_option_data
from stock_dashboard.ui.tables import column_label, show_table

DEFAULT_METRIC = "option_volume"
RATIO_FORMAT = "%.2f"

st.header("Option Data")
st.caption("One underlying's per-session option metrics against their 30-session averages.")
require_option_data()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _symbols() -> list[str]:
    return get_all_option_symbols()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _default_symbol() -> str | None:
    return fetch_most_open_interest_symbol()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _numeric_columns() -> list[str]:
    return get_option_numerical_columns()


@st.cache_data(ttl=CACHE_TTL_SHORT)
def _multiple_columns() -> list[str]:
    return get_option_multiple_columns()


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _recent(symbol: str) -> pd.DataFrame:
    return fetch_option_last_30_days(symbol)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _ratios(symbol: str) -> dict[str, float]:
    return fetch_option_latest_comparison(symbol)


@st.cache_data(ttl=CACHE_TTL_DEFAULT)
def _history(symbol: str, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_option_column_history(symbol, column)


symbols = _symbols()
default_symbol = _default_symbol()
symbol = st.selectbox(
    "Symbol",
    symbols,
    index=symbols.index(default_symbol) if default_symbol in symbols else None,
    placeholder="Select a symbol...",
    help="Defaults to the underlying with the most open interest on the latest date.",
)
if not symbol:
    st.info("Select a symbol to view its option data.")
    st.stop()

recent = _recent(symbol)
if recent.empty:
    st.warning(f"No option data for {symbol}.")
    st.stop()

# Only numeric columns can be charted or compared with an average; a provider's
# extra text or timestamp columns still appear in the table of recent rows.
numeric = set(_numeric_columns())
metric_columns = [c for c in recent.columns if c in numeric]
ratios = _ratios(symbol)
if ratios:
    st.subheader("Latest value / 30-session average")
    row = latest_vs_average_row(ratios, metric_columns, set(_multiple_columns()))
    show_table(row, formats=dict.fromkeys(metric_columns, RATIO_FORMAT), hide_index=False)
    st.caption(
        "Blank: no average, or a signed column (flows, customer greeks, exposures), where a "
        "multiple of the average can flip sign or explode near zero."
    )

st.subheader(f"Last {RECENT_ROWS} sessions")
show_table(
    recent[["collection_date", *(c for c in recent.columns if c not in KEY_COLUMNS)]],
    formats=underlying_formats(metric_columns),
    dates={"collection_date"},
)

column = st.selectbox(
    "Metric to chart",
    metric_columns,
    index=metric_columns.index(DEFAULT_METRIC) if DEFAULT_METRIC in metric_columns else None,
    format_func=column_label,
    placeholder="Select a metric...",
)
if column:
    values, averages = _history(symbol, column)
    if values.empty:
        st.warning(f"No history for {column_label(column)}.")
        st.stop()
    label = column_label(column)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=values["collection_date"], y=values[column], name=label))
    has_average = f"{column}_avg" in averages.columns and not averages.empty
    if has_average:
        fig.add_trace(
            go.Scatter(
                x=averages["date"],
                y=averages[f"{column}_avg"],
                name=f"{label} (30-session avg)",
                line={"dash": "dash", "color": "orange"},
            )
        )
    else:
        st.caption(f"No 30-session average for {label}: only its values are drawn.")
    fig.update_layout(
        title=f"{symbol}: {label}" + (" vs its 30-session average" if has_average else ""),
        yaxis_title=label,
        legend=LEGEND_ABOVE,
        height=400,
    )
    st.plotly_chart(fig)
