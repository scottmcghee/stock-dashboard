"""Option-positioning scans: one history fetch, rules in ``logic.option_signals``.

Every scan reads the last :data:`SIGNAL_HISTORY_SESSIONS` collection dates of
``underlying_data`` (anchored on the table's newest date), classifies the
latest session, and by default joins the latest RS metrics from the stocks
database. Pass ``history`` to share one fetch between scans.
"""

from __future__ import annotations

import pandas as pd
from psycopg2 import sql

from stock_dashboard import db
from stock_dashboard.logic.option_signals import (
    DEFAULT_DXOI_CHANGE_PCT,
    DEFAULT_FLOWNET_PERCENTILE,
    DEFAULT_VANNA_PERCENTILE,
    DEFAULT_VOLUME_MULTIPLE,
    SIGNAL_COLUMNS,
    SIGNAL_HISTORY_SESSIONS,
    classify_dxoi_change,
    classify_flownet_extreme,
    classify_negative_gex,
    classify_vanna_extreme,
    classify_volume_spike,
)
from stock_dashboard.queries.stocks_queries import enrich_with_rs


def fetch_option_signal_history(sessions: int = SIGNAL_HISTORY_SESSIONS) -> pd.DataFrame:
    """symbol, collection_date and the scanned columns for the last ``sessions`` dates."""
    query = sql.SQL("""
        WITH sessions AS (
            SELECT DISTINCT collection_date
            FROM underlying_data
            ORDER BY collection_date DESC
            LIMIT %(sessions)s
        )
        SELECT symbol, collection_date, {columns}
        FROM underlying_data
        WHERE collection_date >= (SELECT MIN(collection_date) FROM sessions)
        ORDER BY symbol, collection_date
    """).format(columns=sql.SQL(", ").join(sql.Identifier(c) for c in SIGNAL_COLUMNS))
    return db.read_df(db.Database.OPTIONS, query, {"sessions": sessions})


def _finish(result: pd.DataFrame, enrich: bool) -> pd.DataFrame:
    return enrich_with_rs(result) if enrich and not result.empty else result


def _history(history: pd.DataFrame | None) -> pd.DataFrame:
    return fetch_option_signal_history() if history is None else history


def scan_vanna_extreme(
    percentile_threshold: float = DEFAULT_VANNA_PERCENTILE,
    *,
    history: pd.DataFrame | None = None,
    enrich: bool = True,
) -> pd.DataFrame:
    """Symbols whose vannaxoi is in the top/bottom N percent of their own recent history."""
    return _finish(classify_vanna_extreme(_history(history), percentile_threshold), enrich)


def scan_flownet_extreme(
    percentile_threshold: float = DEFAULT_FLOWNET_PERCENTILE,
    *,
    history: pd.DataFrame | None = None,
    enrich: bool = True,
) -> pd.DataFrame:
    """Symbols whose flownet is in the bottom N percent of their own recent history."""
    return _finish(classify_flownet_extreme(_history(history), percentile_threshold), enrich)


def scan_volume_spike(
    multiple_threshold: float = DEFAULT_VOLUME_MULTIPLE,
    *,
    history: pd.DataFrame | None = None,
    enrich: bool = True,
) -> pd.DataFrame:
    """Symbols whose option volume is at least N x its trailing 20-session mean."""
    return _finish(classify_volume_spike(_history(history), multiple_threshold), enrich)


def scan_dxoi_change(
    pct_threshold: float = DEFAULT_DXOI_CHANGE_PCT,
    *,
    history: pd.DataFrame | None = None,
    enrich: bool = True,
) -> pd.DataFrame:
    """Symbols whose dxoi moved at least N percent since the previous session."""
    return _finish(classify_dxoi_change(_history(history), pct_threshold), enrich)


def scan_gex_level(*, history: pd.DataFrame | None = None, enrich: bool = True) -> pd.DataFrame:
    """Symbols with negative gamma exposure (gxoi < 0) on the latest session."""
    return _finish(classify_negative_gex(_history(history)), enrich)
