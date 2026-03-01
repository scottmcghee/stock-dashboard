"""Compute boolean signal masks from backtester preset filters for chart overlay."""

import re

import numpy as np
import pandas as pd

from logic.technical import (
    compute_momentum_series,
    compute_ratio_to_ma_series,
    compute_rsi,
)

# Maps filter name → indicator column name (mirrors backtest.py FILTER_TO_COLUMN)
FILTER_INDICATOR: dict[str, str] = {
    "rsi_min": "rsi_14",
    "rsi_max": "rsi_14",
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
    "volume_z_min": "volume_z",
    "volume_z_max": "volume_z",
    "adr_14_min": "adr_14",
    "adr_14_max": "adr_14",
    "rs_rank_overall_min": "rs_rank_overall",
    "rs_rank_overall_max": "rs_rank_overall",
    "rs_z_score_min": "rs_z_score",
    "rs_z_score_max": "rs_z_score",
    "weinstein_stage": "weinstein_stage",
    "has_bull_flag": "has_bull_flag",
    "has_bear_flag": "has_bear_flag",
    "has_high_tight_flag": "has_high_tight_flag",
    "has_cup_and_handle": "has_cup_and_handle",
    "has_v_bottom": "has_v_bottom",
}


def _compute_indicators(
    ohlcv: pd.DataFrame,
    metrics: pd.DataFrame | None,
    needed: set[str],
) -> pd.DataFrame:
    """Lazily compute only the indicator columns that the preset requires."""
    close = ohlcv["close"]
    result = pd.DataFrame(index=ohlcv.index)

    if "rsi_14" in needed:
        result["rsi_14"] = compute_rsi(close, period=14).values

    # Momentum variants
    for period in [20, 63, 126]:
        col = f"momentum_{period}"
        if col in needed:
            result[col] = compute_momentum_series(close, period=period).values

    # Ratio-to-MA variants
    for period in [50, 200]:
        col = f"ratio_to_{period}ma"
        if col in needed:
            result[col] = compute_ratio_to_ma_series(close, period=period).values

    if "volume_z" in needed:
        vol = ohlcv["volume"].astype(float)
        rolling_mean = vol.rolling(30, min_periods=1).mean()
        rolling_std = vol.rolling(30, min_periods=1).std()
        result["volume_z"] = ((vol - rolling_mean) / rolling_std.replace(0, np.nan)).values

    if "adr_14" in needed:
        daily_range_pct = (ohlcv["high"] - ohlcv["low"]) / close * 100
        result["adr_14"] = daily_range_pct.rolling(14, min_periods=1).mean().values

    if "weinstein_stage" in needed:
        from logic.weinstein import calculate_weinstein_stages
        staged = calculate_weinstein_stages(ohlcv.reset_index(drop=True))
        result["weinstein_stage"] = staged["stage"].values

    # Pattern indicators — only compute if any has_* filters are needed
    pattern_cols = {c for c in needed if c.startswith("has_")}
    if pattern_cols:
        from logic.patterns import rolling_detect_patterns
        patterns = rolling_detect_patterns(ohlcv.reset_index(drop=True))
        for col_name in pattern_cols:
            if col_name in patterns:
                result[col_name] = patterns[col_name]
            else:
                result[col_name] = 0

    # Metrics-based indicators (rs_rank_overall, rs_z_score)
    metrics_cols = {c for c in needed if c in ("rs_rank_overall", "rs_z_score")}
    if metrics_cols and metrics is not None and not metrics.empty:
        # Deduplicate metrics by date (may have multiple ETF rows)
        metrics_dedup = metrics.drop_duplicates(subset=["date"], keep="first")
        ohlcv_dates = ohlcv["date"] if "date" in ohlcv.columns else ohlcv.index
        date_series = pd.Series(ohlcv_dates.values, index=ohlcv.index)
        for col_name in metrics_cols:
            if col_name in metrics_dedup.columns:
                lookup = metrics_dedup.set_index("date")[col_name]
                result[col_name] = date_series.map(lookup).values
            else:
                result[col_name] = np.nan

    return result


def compute_preset_signal_mask(
    ohlcv: pd.DataFrame,
    metrics: pd.DataFrame | None,
    raw_filters: list[dict],
) -> pd.Series:
    """Compute a boolean mask indicating where all preset filters are satisfied.

    Args:
        ohlcv: Single-ticker OHLCV DataFrame with date, open, high, low, close, volume.
        metrics: Optional metrics DataFrame with date, rs_rank_overall, rs_z_score, etc.
        raw_filters: List of {"name": str, "value": ...} dicts from preset JSON.

    Returns:
        Boolean Series indexed like ohlcv, True where all filters pass.
    """
    if not raw_filters:
        return pd.Series(True, index=ohlcv.index)

    # Determine which indicator columns we need
    needed: set[str] = set()
    valid_filters: list[dict] = []
    for filt in raw_filters:
        fname = filt["name"]
        if fname in FILTER_INDICATOR:
            needed.add(FILTER_INDICATOR[fname])
            valid_filters.append(filt)

    if not valid_filters:
        return pd.Series(True, index=ohlcv.index)

    indicators = _compute_indicators(ohlcv, metrics, needed)

    mask = pd.Series(True, index=ohlcv.index)
    for filt in valid_filters:
        fname = filt["name"]
        fvalue = filt["value"]
        col = FILTER_INDICATOR[fname]

        if col not in indicators.columns:
            mask = mask & False
            continue

        series = indicators[col]

        if fname.startswith("has_"):
            mask = mask & series.astype(bool)
        elif fname.endswith("_min"):
            mask = mask & (series >= float(fvalue))
        elif fname.endswith("_max"):
            mask = mask & (series <= float(fvalue))
        elif fname == "weinstein_stage":
            mask = mask & (series == str(fvalue))
        else:
            mask = mask & (series >= float(fvalue))

    return mask.fillna(False)


def determine_signal_direction(stats: dict) -> str:
    """Determine if a preset represents a bullish or bearish signal from its backtest stats.

    Checks T+20 mean_return first, then falls back to T+10, T+5, T+63.
    Returns 'bullish' if mean_return > 0, else 'bearish'.
    """
    for horizon in ["T+20", "T+10", "T+5", "T+63"]:
        if horizon in stats:
            mean_ret = stats[horizon].get("mean_return", 0)
            return "bullish" if mean_ret > 0 else "bearish"
    return "bullish"
