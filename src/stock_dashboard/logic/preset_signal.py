"""Replay a backtester preset on one ticker's chart as a boolean signal mask.

Indicators and filter semantics come from :mod:`stock_dashboard.logic.backtest`
(``compute_ticker_indicators``, ``FILTER_TO_COLUMN``, ``signal_mask``). Given the
preset's regime with the benchmark's regime series, and the ETF it was backtested
within (if any), the overlay marks the bars the backtest counted as signals for
this ticker. A condition the overlay cannot evaluate (unknown filter, RS metrics
not supplied, or a regime without a regime series) is reported back and, as in
the backtest, matches no bars. So does a ticker outside the preset's ETF, which
that backtest's universe did not contain.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any, NamedTuple

import pandas as pd

from stock_dashboard.logic.backtest import (
    FILTER_TO_COLUMN,
    INDICATOR_COLUMNS,
    METRIC_COLUMNS,
    RS_IMPULSE_COLUMNS,
    RS_IMPULSE_WINDOWS,
    FilterSpec,
    compute_ticker_indicators,
    signal_mask,
)

logger = logging.getLogger(__name__)

_METRIC_DERIVED = frozenset(METRIC_COLUMNS + RS_IMPULSE_COLUMNS)


class PresetSignal(NamedTuple):
    mask: pd.Series  # indexed like the input OHLCV; True where every filter passes
    unsupported: list[str]  # filters that could not be evaluated, as "name=value"
    outside_etf: bool = False  # the ticker has no metric rows in the preset's ETF


def available_metric_columns(metrics: pd.DataFrame | None) -> set[str]:
    """Metric-derived filter columns that ``metrics`` can supply."""
    if metrics is None or metrics.empty:
        return set()
    present = {c for c in METRIC_COLUMNS if c in metrics.columns}
    if "rs_rank_overall" in present:
        present.update(RS_IMPULSE_COLUMNS)
    return present


def unsupported_filters(
    raw_filters: list[dict[str, Any]], metric_columns: Collection[str] = ()
) -> list[str]:
    """Filters the overlay cannot evaluate, as ``name=value`` strings.

    ``metric_columns`` lists the metric-derived columns available
    (see :func:`available_metric_columns`).
    """
    out = []
    for filt in raw_filters:
        col = FILTER_TO_COLUMN.get(filt["name"])
        if col is None or (col in _METRIC_DERIVED and col not in metric_columns):
            out.append(f"{filt['name']}={filt['value']}")
    return out


def _metric_columns(
    dates: pd.Series, metrics: pd.DataFrame, needed: set[str]
) -> dict[str, pd.Series]:
    """RS metric and impulse columns aligned to ``dates``, aggregated per date like the backtest."""
    present = [c for c in METRIC_COLUMNS if c in metrics.columns]
    per_date = metrics.groupby("date")[present].mean()
    out = {col: dates.map(per_date[col]) for col in present}
    if "rs_rank_overall" in out:
        rank = out["rs_rank_overall"]
        for window, col in zip(RS_IMPULSE_WINDOWS, RS_IMPULSE_COLUMNS, strict=True):
            if col in needed:
                out[col] = rank - rank.shift(window)
    return out


def compute_preset_signal_mask(
    ohlcv: pd.DataFrame,
    metrics: pd.DataFrame | None,
    raw_filters: list[dict[str, Any]],
    *,
    etf: str | None = None,
    regime: str | None = None,
    regimes: pd.Series | None = None,
) -> PresetSignal:
    """Evaluate a preset's filters (and regime) on every bar of one ticker.

    Args:
        ohlcv: Single-ticker OHLCV DataFrame with date, open, high, low, close, volume.
        metrics: Optional fact_stock_metrics rows (date, rs_rank_overall, rs_z_score, ...),
            possibly one row per ETF and date. They are averaged per date, as in a
            backtest over all stocks.
        raw_filters: List of {"name": str, "value": ...} dicts from preset JSON.
        etf: The ETF the preset was backtested within; only that ETF's metric rows
            are used (needs an ``etf_ticker`` column), as in that backtest.
        regime: The preset's market regime, or None for "any".
        regimes: Date-indexed regime labels (``backtest.classify_regimes``); a
            ``regime`` without them cannot be evaluated.

    Returns:
        ``PresetSignal(mask, unsupported, outside_etf)``. When any condition is
        unsupported, or the ticker is not a member of ``etf``, the mask is all
        False, matching how the backtest treats it.
    """
    if metrics is not None and etf is not None and "etf_ticker" in metrics.columns:
        metrics = metrics[metrics["etf_ticker"] == etf]
        if metrics.empty:
            return PresetSignal(pd.Series(False, index=ohlcv.index), [], outside_etf=True)

    unsupported = unsupported_filters(raw_filters, available_metric_columns(metrics))
    if regime and regimes is None:
        unsupported.append(f"regime={regime}")
    if unsupported:
        logger.warning("Preset overlay cannot evaluate: %s", ", ".join(unsupported))
        return PresetSignal(pd.Series(False, index=ohlcv.index), unsupported)

    specs = [FilterSpec(name=f["name"], value=f["value"]) for f in raw_filters]
    needed = {FILTER_TO_COLUMN[s.name] for s in specs}

    ordered = ohlcv.sort_values("date")
    indicators = compute_ticker_indicators(ordered, needed & set(INDICATOR_COLUMNS))
    if metrics is not None and needed & _METRIC_DERIVED:
        for col, values in _metric_columns(ordered["date"], metrics, needed).items():
            indicators[col] = values

    mask = signal_mask(indicators, specs)
    if regime and regimes is not None:
        mask &= ordered["date"].map(regimes).eq(regime).to_numpy()
    return PresetSignal(mask.reindex(ohlcv.index, fill_value=False), [])


def determine_signal_direction(stats: dict[str, dict[str, Any]]) -> str:
    """Determine if a preset represents a bullish or bearish signal from its backtest stats.

    Checks T+20 mean_return first, then falls back to T+10, T+5, T+63.
    Returns 'bullish' if mean_return > 0, else 'bearish'.
    """
    for horizon in ["T+20", "T+10", "T+5", "T+63"]:
        if horizon in stats:
            mean_ret = stats[horizon].get("mean_return", 0)
            return "bullish" if mean_ret > 0 else "bearish"
    return "bullish"
