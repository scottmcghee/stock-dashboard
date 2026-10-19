"""Screener backtesting engine: pure logic, no DB or UI.

Replays screener filters over historical data, measures forward returns,
and tests each (strategy, horizon) mean against zero with a one-sample
t-test, controlling the false discovery rate across all tests with
Benjamini-Hochberg.

:func:`compute_ticker_indicators` and :data:`FILTER_TO_COLUMN` are the single
definition of every filterable indicator; the preset chart overlay reuses
both, so an exported preset's filters replay as they were backtested. The
overlay also applies the preset's regime and, for a preset exported from an
ETF-filtered run, that ETF's RS metrics; the screener applies neither (see
docs/DESIGN.md).
"""

from __future__ import annotations

import itertools
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from stock_dashboard.logic.events import (
    EVENT_COLUMNS,
    EVENT_LOOKBACK,
    LONG_MA_PERIOD,
    detect_events_for_ticker,
)
from stock_dashboard.logic.patterns import PATTERN_DETECTORS, rolling_detect_patterns
from stock_dashboard.logic.technical import (
    RSI_PERIOD,
    compute_adr_series,
    compute_macd,
    compute_momentum_series,
    compute_pivot_proximity,
    compute_ratio_to_ma_series,
    compute_rsi,
    compute_sma,
    compute_volume_z,
)
from stock_dashboard.logic.weinstein import calculate_weinstein_stages

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FilterSpec:
    """A single filter criterion."""

    name: str  # e.g. "rsi_min", "rs_rank_overall_min"
    value: float | str | bool  # threshold value


@dataclass
class Strategy:
    """A named combination of filters + optional regime constraint."""

    name: str
    filters: list[FilterSpec]
    regime: str | None = None  # None, "bull", "bear", "volatile", "correction"


@dataclass
class HorizonStats:
    """Statistical evaluation for one forward-return horizon."""

    mean_return: float
    median_return: float
    win_rate: float  # fraction > 0
    sharpe: float  # mean / std of per-signal returns if std > 0 (not annualized)
    p_value: float  # t-test vs zero
    p_value_adj: float = 1.0  # BH-adjusted (set after batch correction)
    significant: bool = False  # p_value_adj < SIGNIFICANCE_LEVEL


@dataclass
class StrategyResult:
    """Statistical evaluation of a strategy."""

    strategy: Strategy
    n_signals: int  # total (date, ticker) pairs passing filters
    horizons: dict[str, HorizonStats] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

REGIME_TREND_MA = 200
REALIZED_VOL_WINDOW = 20
TRADING_DAYS_PER_YEAR = 252
VOL_PERCENTILE_WINDOW = 252
VOL_PERCENTILE_MIN_PERIODS = 63
VOLATILE_QUANTILE = 0.80
DRAWDOWN_WINDOW = 63
CORRECTION_DRAWDOWN = -0.10


def classify_regimes(benchmark_ohlcv: pd.DataFrame) -> pd.Series:
    """Classify each trading day into a market regime from the benchmark's closes.

    Returns Series indexed by date with values:
    - "bull": close above its 200-day SMA
    - "bear": close below its 200-day SMA
    - "volatile": 20-day annualized realized vol above the 80th percentile of the trailing year
    - "correction": drawdown from the 63-day high worse than 10%

    Priority: correction > volatile > bear > bull. NaN until the SMA is defined.
    """
    df = benchmark_ohlcv.sort_values("date").set_index("date")
    close = df["close"]

    sma = compute_sma(close, REGIME_TREND_MA)
    log_returns = (close / close.shift(1)).apply(np.log)
    realized_vol = log_returns.rolling(
        REALIZED_VOL_WINDOW, min_periods=REALIZED_VOL_WINDOW
    ).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol_threshold = realized_vol.rolling(
        VOL_PERCENTILE_WINDOW, min_periods=VOL_PERCENTILE_MIN_PERIODS
    ).quantile(VOLATILE_QUANTILE)
    rolling_high = close.rolling(DRAWDOWN_WINDOW, min_periods=1).max()
    drawdown = (close - rolling_high) / rolling_high

    regime = pd.Series("bull", index=df.index, dtype="object")
    regime[close < sma] = "bear"
    regime[realized_vol > vol_threshold] = "volatile"
    regime[drawdown < CORRECTION_DRAWDOWN] = "correction"
    return regime.where(sma.notna())


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

# Weinstein stage string -> numeric for vectorized filtering
STAGE_MAP = {
    "Stage 1": 1,
    "Stage 2": 2,
    "Stage 3": 3,
    "Stage 4": 4,
    "Undetermined": 0,
}

MOMENTUM_PERIODS = (20, 63, 126)
RATIO_MA_PERIODS = (50, LONG_MA_PERIOD)
PIVOT_WINDOWS = (63, 126)

PATTERN_COLUMNS = tuple(f"has_{name}" for name in PATTERN_DETECTORS)
PIVOT_COLUMNS = tuple(
    f"dist_recent_{side}_{window}" for window in PIVOT_WINDOWS for side in ("high", "low")
)
INDICATOR_COLUMNS: tuple[str, ...] = (
    "rsi_14",
    *(f"momentum_{p}" for p in MOMENTUM_PERIODS),
    *(f"ratio_to_{p}ma" for p in RATIO_MA_PERIODS),
    "macd_histogram",
    "volume_z",
    "adr_14",
    "weinstein_stage",
    "weinstein_stage_num",
    *PATTERN_COLUMNS,
    *PIVOT_COLUMNS,
    *EVENT_COLUMNS,
)

# Columns that come from fact_stock_metrics rather than prices.
METRIC_COLUMNS = ("rs_rank_overall", "rs_z_score")
RS_IMPULSE_WINDOWS = (5, 10, 20, 42)
RS_IMPULSE_COLUMNS = tuple(f"rs_impulse_{w}" for w in RS_IMPULSE_WINDOWS)


def compute_ticker_indicators(
    ohlcv: pd.DataFrame, columns: Collection[str] | None = None
) -> pd.DataFrame:
    """Price-derived indicator columns for one ticker's OHLCV, which must be sorted by date.

    ``columns`` limits the work to the named :data:`INDICATOR_COLUMNS` (all by
    default); inputs an event needs are included with it. The result keeps
    ``ohlcv``'s index.
    """
    want = set(INDICATOR_COLUMNS if columns is None else columns)
    g = ohlcv.reset_index(drop=True)
    close = g["close"]
    out = pd.DataFrame(index=g.index)

    def needed(*cols: str) -> bool:
        return not want.isdisjoint(cols)

    events = needed(*EVENT_COLUMNS)
    if events or needed("rsi_14"):
        out["rsi_14"] = compute_rsi(close, RSI_PERIOD)
    for period in MOMENTUM_PERIODS:
        if needed(f"momentum_{period}"):
            out[f"momentum_{period}"] = compute_momentum_series(close, period)
    for period in RATIO_MA_PERIODS:
        if needed(f"ratio_to_{period}ma") or (events and period == LONG_MA_PERIOD):
            out[f"ratio_to_{period}ma"] = compute_ratio_to_ma_series(close, period)
    if events or needed("macd_histogram"):
        out["macd_histogram"] = compute_macd(close)[2]
    if events or needed("volume_z"):
        out["volume_z"] = compute_volume_z(g["volume"])
    if needed("adr_14"):
        out["adr_14"] = compute_adr_series(g["high"], g["low"], close)
    if needed("weinstein_stage", "weinstein_stage_num"):
        stages = calculate_weinstein_stages(g)["stage"].to_numpy()
        out["weinstein_stage"] = stages
        out["weinstein_stage_num"] = pd.Series(stages, index=out.index).map(STAGE_MAP)
    if needed(*PATTERN_COLUMNS):
        for col_name, flags in rolling_detect_patterns(g).items():
            out[col_name] = flags
    for window in PIVOT_WINDOWS:
        if needed(f"dist_recent_high_{window}", f"dist_recent_low_{window}"):
            dist_hi, dist_lo = compute_pivot_proximity(close, window=window)
            out[f"dist_recent_high_{window}"] = dist_hi
            out[f"dist_recent_low_{window}"] = dist_lo
    if events:
        event_flags = detect_events_for_ticker(
            close=close,
            rsi_14=out["rsi_14"],
            macd_histogram=out["macd_histogram"],
            volume_z=out["volume_z"],
            ratio_to_200ma=out[f"ratio_to_{LONG_MA_PERIOD}ma"],
            lookback=EVENT_LOOKBACK,
        )
        for col_name, series in event_flags.items():
            out[col_name] = series

    out.index = ohlcv.index
    return out


def precompute_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Every :data:`INDICATOR_COLUMNS` value for each (date, ticker), from trailing data only.

    Args:
        ohlcv: DataFrame with columns [ticker, date, open, high, low, close, volume]

    Returns:
        DataFrame with date, ticker and the indicator columns (floats as float32).
    """
    frames = []
    for ticker, group in ohlcv.groupby("ticker", observed=True):
        g = group.sort_values("date").reset_index(drop=True)
        indicators = compute_ticker_indicators(g)
        indicators.insert(0, "ticker", ticker)
        indicators.insert(0, "date", g["date"])
        frames.append(indicators)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

HORIZONS = (5, 10, 20, 63)
HORIZON_COLUMNS = [f"fwd_{h}" for h in HORIZONS]
HORIZON_LABELS = {f"fwd_{h}": f"T+{h}" for h in HORIZONS}


def compute_forward_returns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute forward returns for each (date, ticker).

    Returns DataFrame with: date, ticker, fwd_5, fwd_10, fwd_20, fwd_63
    where fwd_N = (close[date+N] - close[date]) / close[date]
    """
    frames = []
    for ticker, group in ohlcv.groupby("ticker", observed=True):
        g = group.sort_values("date")
        close = g["close"].to_numpy(dtype=float)
        row = pd.DataFrame({"date": g["date"].to_numpy(), "ticker": ticker})
        for horizon in HORIZONS:
            shifted = g["close"].shift(-horizon).to_numpy(dtype=float)
            row[f"fwd_{horizon}"] = (shifted - close) / close
        frames.append(row)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df[HORIZON_COLUMNS] = df[HORIZON_COLUMNS].astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------

MIN_SIGNALS = 10
SIGNIFICANCE_LEVEL = 0.05


def apply_filter(series: pd.Series, fspec: FilterSpec) -> pd.Series:
    """Return boolean mask for rows passing a single filter."""
    name, value = fspec.name, fspec.value
    if name.startswith("has_"):
        return series.fillna(0).astype(bool)
    if name.endswith("_max"):
        return series <= float(value)
    if name == "weinstein_stage":
        return series == str(value)
    return series >= float(value)


# Mapping from filter name to indicator column. The single source of truth for
# what a filter means; the preset overlay derives its columns from this too.
FILTER_TO_COLUMN: dict[str, str] = {
    "rsi_min": "rsi_14",
    "rsi_max": "rsi_14",
    **{
        f"{col}_{bound}": col
        for col in (
            "rs_rank_overall",
            "rs_z_score",
            *(f"momentum_{p}" for p in MOMENTUM_PERIODS),
            *(f"ratio_to_{p}ma" for p in RATIO_MA_PERIODS),
            "macd_histogram",
            "volume_z",
            "adr_14",
            "rs_impulse_10",
            "rs_impulse_20",
            # S/R proximity (negative = below trailing peak / above trailing trough)
            *PIVOT_COLUMNS,
        )
        for bound in ("min", "max")
    },
    "weinstein_stage": "weinstein_stage",
    **{col: col for col in PATTERN_COLUMNS},
    # Events (has_-prefix -> boolean filter)
    **{f"has_{col}": col for col in EVENT_COLUMNS},
}


def signal_mask(indicators_df: pd.DataFrame, filters: Iterable[FilterSpec]) -> pd.Series:
    """Rows passing every filter. A filter whose column is missing passes no rows."""
    mask = pd.Series(True, index=indicators_df.index)
    for fspec in filters:
        col = FILTER_TO_COLUMN.get(fspec.name)
        if col is None or col not in indicators_df.columns:
            return pd.Series(False, index=indicators_df.index)
        mask &= apply_filter(indicators_df[col], fspec).fillna(False).astype(bool)
    return mask


def strategy_mask(
    indicators_df: pd.DataFrame, strategy: Strategy, regime_series: pd.Series | None
) -> pd.Series:
    """Rows passing the strategy's filters and, if it has one, its regime.

    No row passes a regime strategy when ``regime_series`` is None.
    """
    mask = signal_mask(indicators_df, strategy.filters)
    if strategy.regime:
        if regime_series is None:
            return pd.Series(False, index=indicators_df.index)
        mask &= indicators_df["date"].map(regime_series) == strategy.regime
    return mask


def evaluate_strategy(
    indicators_df: pd.DataFrame,
    strategy: Strategy,
    regime_series: pd.Series | None = None,
) -> StrategyResult:
    """Apply strategy filters to indicator DataFrame, compute stats on forward returns.

    Args:
        indicators_df: DataFrame with indicator columns + forward return columns
                       (fwd_5, fwd_10, fwd_20, fwd_63) and a 'date' column.
        strategy: Strategy with filters and optional regime.
        regime_series: Series indexed by date with regime labels. A strategy
            with a regime and no regime series has no signals: its condition
            cannot be evaluated, like a filter whose column is missing.

    Returns:
        StrategyResult with per-horizon statistics (p_value_adj not yet set).
        Horizons with fewer than MIN_SIGNALS observations are left out.
    """
    mask = strategy_mask(indicators_df, strategy, regime_series)

    signals = indicators_df.loc[mask]
    result = StrategyResult(strategy=strategy, n_signals=len(signals))
    if len(signals) < MIN_SIGNALS:
        return result

    for col, label in HORIZON_LABELS.items():
        if col not in signals.columns:
            continue
        returns = signals[col].dropna()
        if len(returns) < MIN_SIGNALS:
            continue

        mean_ret = float(returns.mean())
        std_ret = float(returns.std())
        p_val = float(stats.ttest_1samp(returns, 0.0).pvalue) if std_ret > 0 else 1.0
        result.horizons[label] = HorizonStats(
            mean_return=mean_ret,
            median_return=float(returns.median()),
            win_rate=float((returns > 0).mean()),
            sharpe=mean_ret / std_ret if std_ret > 0 else 0.0,
            p_value=p_val,
        )

    return result


def apply_bh_correction(results: list[StrategyResult]) -> list[StrategyResult]:
    """Apply Benjamini-Hochberg FDR correction across all p-values.

    Modifies results in-place, setting p_value_adj and significant on each HorizonStats.
    """
    entries = [
        (i, label, hs.p_value)
        for i, res in enumerate(results)
        for label, hs in res.horizons.items()
    ]
    if not entries:
        return results

    entries.sort(key=lambda x: x[2])
    m = len(entries)
    # p_adj[k] = min over j >= k of p[j] * m / rank(j), capped at 1.
    adjusted = [p * m / (rank + 1) for rank, (_, _, p) in enumerate(entries)]
    for k in range(m - 2, -1, -1):
        adjusted[k] = min(adjusted[k], adjusted[k + 1])

    for (res_i, label, _), adj in zip(entries, adjusted, strict=True):
        hs = results[res_i].horizons[label]
        hs.p_value_adj = min(adj, 1.0)
        hs.significant = hs.p_value_adj < SIGNIFICANCE_LEVEL

    return results


def evaluate_all_strategies(
    indicators_df: pd.DataFrame,
    strategies: list[Strategy],
    regime_series: pd.Series | None = None,
) -> list[StrategyResult]:
    """Evaluate multiple strategies and apply BH FDR correction."""
    results = [evaluate_strategy(indicators_df, s, regime_series) for s in strategies]
    apply_bh_correction(results)
    return results


# ---------------------------------------------------------------------------
# Strategy generation
# ---------------------------------------------------------------------------

# Threshold grid for generated strategies. These are round values spanning
# each indicator's usual range, chosen by hand; the BH correction accounts for
# testing all of them.
FILTER_RANGES: dict[str, list[float | str | bool]] = {
    "rsi_max": [30, 40],
    "rsi_min": [60, 70],
    "rs_rank_overall_min": [0.7, 0.8, 0.9],
    "rs_z_score_min": [1.0, 1.5, 2.0],
    "momentum_63_min": [5.0, 10.0, 20.0],
    "ratio_to_200ma_min": [1.0, 1.05],
    "ratio_to_200ma_max": [0.95, 0.90],
    "volume_z_min": [1.5, 2.0, 3.0],
    "adr_14_min": [3.0, 5.0, 7.0],
    "weinstein_stage": ["Stage 2", "Stage 4"],
    "has_bull_flag": [True],
    "has_bear_flag": [True],
    "has_high_tight_flag": [True],
    "has_cup_and_handle": [True],
    "has_v_bottom": [True],
    "rs_impulse_10_min": [0.05, 0.10, 0.15],
    "rs_impulse_10_max": [-0.05, -0.10, -0.15],
    "rs_impulse_20_min": [0.05, 0.10, 0.15],
    "rs_impulse_20_max": [-0.05, -0.10, -0.15],
    # Only one S/R column is gridded, to keep the strategy count manageable;
    # the others remain available as manual filters through FILTER_TO_COLUMN.
    "dist_recent_low_126_min": [0.10, 0.20],  # >= 10%/20% above trailing low
    "dist_recent_low_126_max": [-0.05, 0.0],
    "has_evt_rsi_cross_down_70": [True],
    "has_evt_macd_bullish_cross": [True],
    "has_evt_macd_bearish_cross": [True],
    "has_evt_breakout_20d_high_vol": [True],
    "has_evt_regain_200ma": [True],
    "has_evt_lose_200ma": [True],
}

REGIMES: list[str | None] = [None, "bull", "bear", "volatile", "correction"]

# Filter names that conflict (should not appear together in a strategy)
CONFLICTING_FILTERS = [
    {"rsi_min", "rsi_max"},
    {"ratio_to_200ma_min", "ratio_to_200ma_max"},
    {"rs_impulse_10_min", "rs_impulse_10_max"},
    {"rs_impulse_20_min", "rs_impulse_20_max"},
]


def _has_conflict(filter_names: list[str]) -> bool:
    """Check if a set of filter names contains conflicting pairs."""
    name_set = set(filter_names)
    return any(pair.issubset(name_set) for pair in CONFLICTING_FILTERS)


def generate_strategies(max_filters: int = 3) -> list[Strategy]:
    """Generate all 1-to-max_filters combinations from FILTER_RANGES x REGIMES.

    Prunes nonsensical combos (e.g., rsi_min + rsi_max both set).
    """
    options = [FilterSpec(name, val) for name, values in FILTER_RANGES.items() for val in values]
    strategies: list[Strategy] = []
    for n_filters in range(1, max_filters + 1):
        for combo in itertools.combinations(options, n_filters):
            names = [f.name for f in combo]
            if _has_conflict(names) or len(set(names)) != len(names):
                continue
            filters = list(combo)
            for regime in REGIMES:
                parts = [f"{f.name}={f.value}" for f in filters]
                if regime:
                    parts.append(f"regime={regime}")
                strategies.append(Strategy(name=" & ".join(parts), filters=filters, regime=regime))
    return strategies


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def compute_rs_impulse(df: pd.DataFrame) -> pd.DataFrame:
    """Add RS rank change over 5, 10, 20 and 42 trading days per ticker (in place).

    Positive values mean the ticker is rising through the ranks.
    """
    ordered = df.sort_values(["ticker", "date"])
    rank = ordered.groupby("ticker", observed=True)["rs_rank_overall"]
    for window, col_name in zip(RS_IMPULSE_WINDOWS, RS_IMPULSE_COLUMNS, strict=True):
        df[col_name] = (ordered["rs_rank_overall"] - rank.shift(window)).astype(np.float32)
    return df


def build_backtest_dataframe(
    ohlcv: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the unified indicator + forward returns DataFrame.

    Merges pre-computed indicators and forward returns with optional RS metrics
    (one row per date and ticker).
    """
    indicators = precompute_indicators(ohlcv)
    fwd_returns = compute_forward_returns(ohlcv)
    df = indicators.merge(fwd_returns, on=["date", "ticker"], how="left")

    if metrics is not None and not metrics.empty:
        metric_cols = [c for c in ("date", "ticker", *METRIC_COLUMNS) if c in metrics.columns]
        df = df.merge(metrics[metric_cols], on=["date", "ticker"], how="left")

    if "rs_rank_overall" in df.columns:
        df = compute_rs_impulse(df)

    return df
