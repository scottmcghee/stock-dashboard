"""Presentation of backtest results: tables, regime shading, signal detail, presets.

Pure functions over :mod:`stock_dashboard.logic.backtest` results, used by the
backtester page. Returns are fractions in the engine and percentages here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import pandas as pd

from stock_dashboard.logic.backtest import (
    HORIZON_LABELS,
    MIN_SIGNALS,
    REGIMES,
    StrategyResult,
    strategy_mask,
)
from stock_dashboard.logic.preset import build_preset

PERCENT = 100.0
HORIZONS = tuple(HORIZON_LABELS.values())  # "T+5", ...
SORT_METRICS = ("mean_return", "median_return", "sharpe", "win_rate")
# "sharpe" is mean / SD of the per-signal forward returns: not annualized, no risk-free rate.
SORT_METRIC_LABELS = {
    "mean_return": "Mean return",
    "median_return": "Median return",
    "sharpe": "Mean/SD (per signal)",
    "win_rate": "Win rate",
}
_METRIC_SUFFIX = {
    "mean_return": "mean",
    "median_return": "med",
    "sharpe": "sharpe",
    "win_rate": "wr",
}

ANY_REGIME_LABEL = "None (unconditional)"
REGIME_LABELS = [r for r in REGIMES if r is not None]
# A regime needs this many signals before the breakdown reports it.
MIN_REGIME_SIGNALS = 5


def regime_choices(labels: Iterable[str]) -> list[str | None]:
    """Regimes from the page's multiselect labels (``ANY_REGIME_LABEL`` -> None)."""
    return [None if label == ANY_REGIME_LABEL else label for label in labels]


class RegimeBlock(NamedTuple):
    start: pd.Timestamp
    end: pd.Timestamp
    regime: str


def regime_blocks(regimes: pd.Series) -> list[RegimeBlock]:
    """Contiguous runs of one regime label (date-indexed series; NaN days skipped).

    Each block ends where the next one starts, so shaded blocks leave no gaps;
    the last ends on the last classified date.
    """
    labeled = regimes.dropna().sort_index()
    if labeled.empty:
        return []
    change = labeled.ne(labeled.shift())
    starts = list(labeled.index[change])
    labels = list(labeled[change])
    ends = [*starts[1:], labeled.index[-1]]
    return [
        RegimeBlock(pd.Timestamp(s), pd.Timestamp(e), str(r))
        for s, e, r in zip(starts, ends, labels, strict=True)
    ]


def _is_reportable(result: StrategyResult, min_signals: int) -> bool:
    return result.n_signals >= min_signals and any(
        hs.significant for hs in result.horizons.values()
    )


def results_table(
    results: Sequence[StrategyResult],
    *,
    show_all: bool = False,
    min_signals: int = MIN_SIGNALS,
) -> pd.DataFrame:
    """One row per strategy with per-horizon mean/median/win rate (%), mean/SD and adjusted p.

    Unless ``show_all``, only strategies with ``min_signals`` signals and at
    least one significant horizon are kept.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        if not show_all and not _is_reportable(r, min_signals):
            continue
        row: dict[str, Any] = {
            "strategy": r.strategy.name,
            "regime": r.strategy.regime or "any",
            "n_signals": r.n_signals,
        }
        for label in HORIZONS:
            hs = r.horizons.get(label)
            row[f"{label}_mean"] = None if hs is None else hs.mean_return * PERCENT
            row[f"{label}_med"] = None if hs is None else hs.median_return * PERCENT
            row[f"{label}_wr"] = None if hs is None else hs.win_rate * PERCENT
            row[f"{label}_sharpe"] = None if hs is None else hs.sharpe
            row[f"{label}_pval"] = None if hs is None else hs.p_value_adj
            row[f"{label}_sig"] = False if hs is None else hs.significant
        rows.append(row)
    return pd.DataFrame(rows)


def sort_results(table: pd.DataFrame, horizon: str, metric: str) -> pd.DataFrame:
    """``table`` sorted best first by ``metric`` (one of SORT_METRICS) at ``horizon``."""
    column = f"{horizon}_{_METRIC_SUFFIX[metric]}"
    if table.empty or column not in table.columns:
        return table
    return table.sort_values(column, ascending=False, na_position="last").reset_index(drop=True)


P_VALUE_FLOOR = 1e-4  # smaller p-values show as "<0.0001" rather than "0.0000"


def format_p_value(p: object) -> str:
    """A p-value with four decimals, "<0.0001" below that, blank when missing."""
    if not isinstance(p, int | float) or pd.isna(p):
        return ""
    return f"<{P_VALUE_FLOOR:.4f}" if p < P_VALUE_FLOOR else f"{p:.4f}"


def results_column_config_formats() -> dict[str, tuple[str, str | None]]:
    """Column -> (header, printf format) for the results table's numeric columns.

    The p-value columns have no printf format: :func:`format_p_value` displays them.
    """
    formats: dict[str, tuple[str, str | None]] = {"n_signals": ("Signals", "%d")}
    for label in HORIZONS:
        formats[f"{label}_mean"] = (f"{label} Mean %", "%.2f")
        formats[f"{label}_med"] = (f"{label} Median %", "%.2f")
        formats[f"{label}_wr"] = (f"{label} Win %", "%.1f")
        formats[f"{label}_sharpe"] = (f"{label} Mean/SD", "%.3f")
        formats[f"{label}_pval"] = (f"{label} p (BH)", None)
    return formats


def strategy_signals(
    indicators: pd.DataFrame, result: StrategyResult, regimes: pd.Series | None
) -> pd.DataFrame:
    """The (date, ticker) rows the strategy counted as signals, as in ``evaluate_strategy``."""
    return indicators.loc[strategy_mask(indicators, result.strategy, regimes)]


def horizon_column(horizon: str) -> str:
    """Forward-return column for a horizon label: "T+20" -> "fwd_20"."""
    return f"fwd_{horizon.removeprefix('T+')}"


def cumulative_signal_returns(signals: pd.DataFrame, horizon: str) -> pd.Series:
    """Compounded mean forward return of each signal date, in percent, by date."""
    column = horizon_column(horizon)
    if column not in signals.columns or signals.empty:
        return pd.Series(dtype=float)
    daily = signals.groupby("date")[column].mean().sort_index().dropna().astype(float)
    return ((1 + daily).cumprod() - 1) * PERCENT


def regime_breakdown(
    signals: pd.DataFrame, regimes: pd.Series, min_signals: int = MIN_REGIME_SIGNALS
) -> pd.DataFrame:
    """Signal count, mean forward return (%) and win rate (%) per market regime."""
    labels = signals["date"].map(regimes)
    rows: list[dict[str, Any]] = []
    for regime in REGIME_LABELS:
        subset = signals[labels == regime]
        if len(subset) < min_signals:
            continue
        row: dict[str, Any] = {"regime": regime, "n_signals": len(subset)}
        for column, label in HORIZON_LABELS.items():
            returns = subset[column].dropna() if column in subset.columns else pd.Series()
            if len(returns):
                row[f"{label} Mean %"] = float(returns.mean()) * PERCENT
                row[f"{label} Win %"] = float((returns > 0).mean()) * PERCENT
        rows.append(row)
    return pd.DataFrame(rows)


def horizon_stats_table(result: StrategyResult) -> pd.DataFrame:
    """One row per evaluated horizon of a strategy."""
    return pd.DataFrame(
        [
            {
                "Horizon": label,
                "Mean Return %": hs.mean_return * PERCENT,
                "Median Return %": hs.median_return * PERCENT,
                "Win Rate %": hs.win_rate * PERCENT,
                "Mean/SD": hs.sharpe,
                "p-value (BH)": hs.p_value_adj,
                "Significant": hs.significant,
            }
            for label, hs in result.horizons.items()
        ]
    )


# Rounding of the stats stored in a preset file.
_STAT_DECIMALS = {"mean_return": 2, "win_rate": 3, "sharpe": 3, "p_value_adj": 4}


def preset_stats(result: StrategyResult) -> dict[str, dict[str, float]]:
    """Per-horizon stats in the preset format: mean_return in %, win_rate as a fraction."""
    return {
        label: {
            "mean_return": round(hs.mean_return * PERCENT, _STAT_DECIMALS["mean_return"]),
            "win_rate": round(hs.win_rate, _STAT_DECIMALS["win_rate"]),
            "sharpe": round(hs.sharpe, _STAT_DECIMALS["sharpe"]),
            "p_value_adj": round(hs.p_value_adj, _STAT_DECIMALS["p_value_adj"]),
        }
        for label, hs in result.horizons.items()
    }


def preset_from_result(
    result: StrategyResult, name: str, *, etf: str | None = None
) -> dict[str, object]:
    """A preset document for a backtested strategy; ``etf`` is the run's ETF filter, if any."""
    strategy = result.strategy
    return build_preset(
        name,
        {f.name: f.value for f in strategy.filters},
        regimes=[strategy.regime] if strategy.regime else [],
        stats=preset_stats(result),
        etf=etf,
    )
