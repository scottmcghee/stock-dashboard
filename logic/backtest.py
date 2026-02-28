"""Screener backtesting engine — pure logic, no DB or UI.

Replays screener filters over historical data, measures forward returns,
and tests statistical significance with BH FDR correction.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from logic.technical import (
    compute_adr,
    compute_macd,
    compute_momentum_series,
    compute_ratio_to_ma_series,
    compute_rsi,
    compute_sma,
)
from logic.patterns import rolling_detect_patterns
from logic.weinstein import calculate_weinstein_stages


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
    sharpe: float  # mean / std if std > 0
    p_value: float  # t-test vs zero
    p_value_adj: float = 1.0  # BH-adjusted (set after batch correction)
    significant: bool = False  # p_value_adj < 0.05


@dataclass
class StrategyResult:
    """Statistical evaluation of a strategy."""

    strategy: Strategy
    n_signals: int  # total (date, ticker) pairs passing filters
    horizons: dict[str, HorizonStats] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


def classify_regimes(spy_ohlcv: pd.DataFrame) -> pd.Series:
    """Classify each trading day into a regime based on SPY.

    Returns Series indexed by date with values:
    - "bull": SPY close > 200-day SMA
    - "bear": SPY close < 200-day SMA
    - "volatile": 20-day realized vol > 80th percentile of 1-year rolling
    - "correction": SPY drawdown from 63-day high > 10%

    Priority: correction > volatile > bear > bull.
    """
    df = spy_ohlcv.sort_values("date").copy()
    df = df.set_index("date")

    sma_200 = compute_sma(df["close"], 200)

    # Realized vol: 20-day std of log returns
    log_returns = np.log(df["close"] / df["close"].shift(1))
    realized_vol = log_returns.rolling(20, min_periods=20).std() * np.sqrt(252)
    vol_pct80 = realized_vol.rolling(252, min_periods=63).quantile(0.80)

    # Drawdown from 63-day high
    rolling_high = df["close"].rolling(63, min_periods=1).max()
    drawdown = (df["close"] - rolling_high) / rolling_high

    # Classification with priority
    regime = pd.Series("bull", index=df.index, dtype="object")
    regime[df["close"] < sma_200] = "bear"
    regime[realized_vol > vol_pct80] = "volatile"
    regime[drawdown < -0.10] = "correction"

    # NaN out early period where we can't compute indicators
    regime[sma_200.isna()] = np.nan

    return regime


# ---------------------------------------------------------------------------
# Indicator pre-computation
# ---------------------------------------------------------------------------

# Weinstein stage string → numeric for vectorized filtering
STAGE_MAP = {
    "Stage 1": 1,
    "Stage 2": 2,
    "Stage 3": 3,
    "Stage 4": 4,
    "Undetermined": 0,
}


def precompute_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute indicator values for each (date, ticker) using trailing data.

    Indicators:
    - rsi_14: Wilder's RSI(14)
    - momentum_20, momentum_63, momentum_126: rate of change (%)
    - ratio_to_50ma, ratio_to_200ma: close / SMA
    - macd_histogram: MACD(12,26,9) histogram
    - volume_z: (volume - 20d_avg) / 20d_std
    - adr_14: rolling 14-day ADR%
    - weinstein_stage_num: numeric stage (1-4, 0=undetermined)

    Args:
        ohlcv: DataFrame with columns [ticker, date, open, high, low, close, volume]

    Returns:
        DataFrame with date, ticker, + indicator columns
    """
    results = []

    for ticker, group in ohlcv.groupby("ticker"):
        g = group.sort_values("date").copy()
        close = g["close"]
        high = g["high"]
        low = g["low"]
        volume = g["volume"].astype(float)

        row = pd.DataFrame({"date": g["date"].values, "ticker": ticker})

        # RSI
        row["rsi_14"] = compute_rsi(close, 14).values

        # Momentum at multiple horizons
        row["momentum_20"] = compute_momentum_series(close, 20).values
        row["momentum_63"] = compute_momentum_series(close, 63).values
        row["momentum_126"] = compute_momentum_series(close, 126).values

        # Ratio to moving averages
        row["ratio_to_50ma"] = compute_ratio_to_ma_series(close, 50).values
        row["ratio_to_200ma"] = compute_ratio_to_ma_series(close, 200).values

        # MACD histogram
        _, _, hist = compute_macd(close)
        row["macd_histogram"] = hist.values

        # Volume z-score
        vol_avg = volume.rolling(20, min_periods=20).mean()
        vol_std = volume.rolling(20, min_periods=20).std()
        row["volume_z"] = ((volume - vol_avg) / vol_std.replace(0, np.nan)).values

        # ADR%
        daily_range_pct = (high - low) / close * 100
        row["adr_14"] = daily_range_pct.rolling(14, min_periods=14).mean().values

        # Weinstein stage
        staged = calculate_weinstein_stages(g)
        row["weinstein_stage"] = staged["stage"].values
        row["weinstein_stage_num"] = row["weinstein_stage"].map(STAGE_MAP).values

        # Chart patterns (rolling detection)
        patterns = rolling_detect_patterns(g)
        for col_name, arr in patterns.items():
            row[col_name] = arr

        results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)

    # Downcast indicator columns to float32 to reduce memory
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------


def compute_forward_returns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute forward returns for each (date, ticker).

    Returns DataFrame with: date, ticker, fwd_5, fwd_10, fwd_20, fwd_63
    where fwd_N = (close[date+N] - close[date]) / close[date]
    """
    results = []

    for ticker, group in ohlcv.groupby("ticker"):
        g = group.sort_values("date").copy()
        close = g["close"]

        row = pd.DataFrame({"date": g["date"].values, "ticker": ticker})
        for horizon in [5, 10, 20, 63]:
            shifted = close.shift(-horizon)
            row[f"fwd_{horizon}"] = ((shifted.values - close.values) / close.values)

        results.append(row)

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)

    # Downcast forward return columns to float32
    fwd_cols = [c for c in df.columns if c.startswith("fwd_")]
    df[fwd_cols] = df[fwd_cols].astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------


def _apply_filter(series: pd.Series, fspec: FilterSpec) -> pd.Series:
    """Return boolean mask for rows passing a single filter."""
    name = fspec.name
    value = fspec.value

    if name.startswith("has_"):
        return series.astype(bool)
    elif name.endswith("_min"):
        return series >= float(value)
    elif name.endswith("_max"):
        return series <= float(value)
    elif name == "weinstein_stage":
        return series == str(value)
    else:
        return series >= float(value)


# Mapping from filter name to indicator column
FILTER_TO_COLUMN = {
    "rsi_min": "rsi_14",
    "rsi_max": "rsi_14",
    "rs_rank_overall_min": "rs_rank_overall",
    "rs_rank_overall_max": "rs_rank_overall",
    "rs_z_score_min": "rs_z_score",
    "rs_z_score_max": "rs_z_score",
    "momentum_20_min": "momentum_20",
    "momentum_20_max": "momentum_20",
    "momentum_63_min": "momentum_63",
    "momentum_63_max": "momentum_63",
    "momentum_126_min": "momentum_126",
    "momentum_126_max": "momentum_126",
    "ratio_to_50ma_min": "ratio_to_50ma",
    "ratio_to_50ma_max": "ratio_to_50ma",
    "ratio_to_200ma_min": "ratio_to_200ma",
    "ratio_to_200ma_max": "ratio_to_200ma",
    "macd_histogram_min": "macd_histogram",
    "macd_histogram_max": "macd_histogram",
    "volume_z_min": "volume_z",
    "volume_z_max": "volume_z",
    "adr_14_min": "adr_14",
    "adr_14_max": "adr_14",
    "weinstein_stage": "weinstein_stage",
    "has_bull_flag": "has_bull_flag",
    "has_bear_flag": "has_bear_flag",
    "has_high_tight_flag": "has_high_tight_flag",
    "has_cup_and_handle": "has_cup_and_handle",
    "has_v_bottom": "has_v_bottom",
}

HORIZON_COLUMNS = ["fwd_5", "fwd_10", "fwd_20", "fwd_63"]
HORIZON_LABELS = {"fwd_5": "T+5", "fwd_10": "T+10", "fwd_20": "T+20", "fwd_63": "T+63"}


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
        regime_series: Series indexed by date with regime labels.

    Returns:
        StrategyResult with per-horizon statistics (p_value_adj not yet set).
    """
    mask = pd.Series(True, index=indicators_df.index)

    for fspec in strategy.filters:
        col = FILTER_TO_COLUMN.get(fspec.name)
        if col is None or col not in indicators_df.columns:
            # Unknown filter — no rows pass
            mask[:] = False
            break
        col_data = indicators_df[col]
        mask = mask & _apply_filter(col_data, fspec)

    # Regime filter
    if strategy.regime and regime_series is not None:
        dates = indicators_df["date"]
        date_regime = dates.map(regime_series)
        mask = mask & (date_regime == strategy.regime)

    signals = indicators_df.loc[mask]
    n_signals = len(signals)

    result = StrategyResult(strategy=strategy, n_signals=n_signals)

    if n_signals < 10:
        # Not enough signals for meaningful stats
        return result

    for col, label in HORIZON_LABELS.items():
        if col not in signals.columns:
            continue
        returns = signals[col].dropna()
        if len(returns) < 10:
            continue

        mean_ret = float(returns.mean())
        median_ret = float(returns.median())
        std_ret = float(returns.std())
        win_rate = float((returns > 0).mean())
        sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

        # One-sample t-test: H0 mean return = 0
        if std_ret > 0:
            _, p_val = stats.ttest_1samp(returns, 0.0)
            p_val = float(p_val)
        else:
            p_val = 1.0

        result.horizons[label] = HorizonStats(
            mean_return=mean_ret,
            median_return=median_ret,
            win_rate=win_rate,
            sharpe=sharpe,
            p_value=p_val,
        )

    return result


def apply_bh_correction(results: list[StrategyResult]) -> list[StrategyResult]:
    """Apply Benjamini-Hochberg FDR correction across all p-values.

    Modifies results in-place, setting p_value_adj and significant on each HorizonStats.
    """
    # Collect all (result_idx, horizon_label, p_value) triples
    entries: list[tuple[int, str, float]] = []
    for i, res in enumerate(results):
        for label, hs in res.horizons.items():
            entries.append((i, label, hs.p_value))

    if not entries:
        return results

    # Sort by p-value
    entries.sort(key=lambda x: x[2])
    m = len(entries)

    # BH adjustment: p_adj[i] = min(p[i] * m / (rank), 1.0)
    # Process in reverse to enforce monotonicity
    adjusted = [0.0] * m
    for rank_idx in range(m):
        rank = rank_idx + 1  # 1-based rank
        adjusted[rank_idx] = entries[rank_idx][2] * m / rank

    # Enforce monotonicity (descending pass)
    for k in range(m - 2, -1, -1):
        adjusted[k] = min(adjusted[k], adjusted[k + 1])

    # Cap at 1.0
    adjusted = [min(a, 1.0) for a in adjusted]

    # Write back
    for idx, (res_i, label, _) in enumerate(entries):
        hs = results[res_i].horizons[label]
        hs.p_value_adj = adjusted[idx]
        hs.significant = adjusted[idx] < 0.05

    return results


def evaluate_all_strategies(
    indicators_df: pd.DataFrame,
    strategies: list[Strategy],
    regime_series: pd.Series | None = None,
) -> list[StrategyResult]:
    """Evaluate multiple strategies and apply BH FDR correction."""
    results = [
        evaluate_strategy(indicators_df, s, regime_series)
        for s in strategies
    ]
    apply_bh_correction(results)
    return results


# ---------------------------------------------------------------------------
# Strategy generation
# ---------------------------------------------------------------------------

FILTER_RANGES: dict[str, list] = {
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
}

REGIMES: list[str | None] = [None, "bull", "bear", "volatile", "correction"]

# Filter names that conflict (should not appear together in a strategy)
_CONFLICTS = [
    {"rsi_min", "rsi_max"},
    {"ratio_to_200ma_min", "ratio_to_200ma_max"},
]


def _has_conflict(filter_names: list[str]) -> bool:
    """Check if a set of filter names contains conflicting pairs."""
    name_set = set(filter_names)
    for conflict_pair in _CONFLICTS:
        if conflict_pair.issubset(name_set):
            return True
    return False


def generate_strategies(max_filters: int = 3) -> list[Strategy]:
    """Generate all 1-to-max_filters combinations from FILTER_RANGES x REGIMES.

    Prunes nonsensical combos (e.g., rsi_min + rsi_max both set).
    """
    # Build all individual FilterSpec options
    all_filter_options: list[FilterSpec] = []
    for name, values in FILTER_RANGES.items():
        for val in values:
            all_filter_options.append(FilterSpec(name=name, value=val))

    strategies: list[Strategy] = []

    for n_filters in range(1, max_filters + 1):
        for combo in itertools.combinations(all_filter_options, n_filters):
            # Check for conflicts
            names = [f.name for f in combo]
            if _has_conflict(names):
                continue

            # Check for duplicate filter names (same name, different values)
            if len(set(names)) != len(names):
                continue

            filters = list(combo)
            for regime in REGIMES:
                # Build descriptive name
                parts = [f"{f.name}={f.value}" for f in filters]
                if regime:
                    parts.append(f"regime={regime}")
                name = " & ".join(parts)

                strategies.append(Strategy(
                    name=name,
                    filters=filters,
                    regime=regime,
                ))

    return strategies


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def build_backtest_dataframe(
    ohlcv: pd.DataFrame,
    metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the unified indicator + forward returns DataFrame.

    Merges pre-computed indicators and forward returns with optional RS metrics.
    """
    indicators = precompute_indicators(ohlcv)
    fwd_returns = compute_forward_returns(ohlcv)

    # Merge indicators with forward returns on (date, ticker)
    df = indicators.merge(fwd_returns, on=["date", "ticker"], how="left")

    # Merge RS metrics if provided (already aggregated per date/ticker)
    if metrics is not None and not metrics.empty:
        metric_cols = [c for c in ["date", "ticker", "rs_rank_overall", "rs_z_score"]
                       if c in metrics.columns]
        df = df.merge(metrics[metric_cols], on=["date", "ticker"], how="left")

    return df
