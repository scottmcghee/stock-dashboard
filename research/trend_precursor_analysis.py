"""Trend precursors: what did ETFs look like just before their sustained trends began?

Finds sustained up- and downtrends in ETF closes after the fact (a minimum
move within a duration range and a bounded adverse excursion), records the
technical and relative-strength readings at each trend's first bar and over
the bars before it, and compares them with randomly sampled bars away from
any trend start. This profiles trends that are known to have happened; it is
not a forward test of any entry rule.

Usage:
    python -m research.trend_precursor_analysis --help
    python -m research.trend_precursor_analysis --start 2020-01-01 --min-move 15 --focus ABC
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2 import sql
from scipy import stats

from research.common import (
    add_common_args,
    bollinger_width,
    configure_logging,
    fdr_adjust,
    make_parser,
    print_table,
    symbol_list,
    where_clause,
    write_csv,
)
from stock_dashboard import db
from stock_dashboard.config import get_settings
from stock_dashboard.logic.technical import (
    RSI_PERIOD,
    compute_adr_series,
    compute_momentum_series,
    compute_ratio_to_ma_series,
    compute_rsi,
)
from stock_dashboard.logic.trend_signals import (
    LONG_MA_PERIOD,
    SHORT_MA_PERIOD,
    VOLUME_AVG_PERIOD,
)
from stock_dashboard.queries.stocks_queries import fetch_ohlcv

logger = logging.getLogger(__name__)

MOMENTUM_PERIODS = (10, 20, 63)
RSI_CHANGE_BARS = 5
RS_CHANGE_WINDOWS = (5, 10, 20)
DEFAULT_LOOKBACK = 20
DEFAULT_BASELINE_SAMPLES = 2000
DEFAULT_SEED = 42
MIN_SLOPE_POINTS = 5
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
# Close / 200-bar MA bands for the stage-proxy table.
BELOW_200MA = 0.95
ABOVE_200MA = 1.05

INDICATORS = {
    "rsi": f"RSI ({RSI_PERIOD})",
    **{f"mom_{p}": f"Momentum {p} bars %" for p in MOMENTUM_PERIODS},
    "ratio_50ma": f"Close / {SHORT_MA_PERIOD}-bar MA",
    "ratio_200ma": f"Close / {LONG_MA_PERIOD}-bar MA",
    "vol_ratio": f"Volume / {VOLUME_AVG_PERIOD}-bar mean",
    "adr_pct": "ADR %",
    "bb_width": "Bollinger width %",
    "rsi_roc_5": f"RSI change over {RSI_CHANGE_BARS} bars",
    "rs_rank": "RS rank (within ETFs)",
    "rs_z_score": "RS z-score",
    **{f"rs_rank_chg_{w}": f"RS rank change {w} bars" for w in RS_CHANGE_WINDOWS},
}
LOOKBACK_FEATURES = {
    "rsi_lookback_mean": ("rsi", "mean"),
    "rsi_lookback_trend": ("rsi", "slope"),
    "mom_20_lookback_trend": ("mom_20", "slope"),
    "rs_rank_lookback_mean": ("rs_rank", "mean"),
    "rs_rank_lookback_trend": ("rs_rank", "slope"),
    "vol_ratio_lookback_mean": ("vol_ratio", "mean"),
    "bb_width_lookback_mean": ("bb_width", "mean"),
    "bb_width_lookback_trend": ("bb_width", "slope"),
}
DISTRIBUTION_COLUMNS = ("rsi", "rs_rank", "ratio_50ma", "bb_width", "rs_rank_chg_10")
SPLIT_COLUMNS = (
    "rsi",
    "mom_20",
    "mom_63",
    "ratio_50ma",
    "ratio_200ma",
    "vol_ratio",
    "adr_pct",
    "bb_width",
    "rsi_roc_5",
    "rs_rank",
    "rs_z_score",
    "rs_rank_chg_10",
    "move_pct",
    "duration",
    *LOOKBACK_FEATURES,
)


@dataclass(frozen=True)
class TrendRule:
    """What counts as a sustained trend, in percent and bars."""

    min_move_pct: float = 15.0
    min_duration: int = 20
    max_duration: int = 120
    max_adverse_pct: float = 8.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_etf_metrics(
    etfs: Sequence[str] | None, start: date | None, end: date | None
) -> pd.DataFrame:
    """ETF relative-strength rank and z-score against the configured benchmark."""
    where, params = where_clause(
        date_col="date", symbol_col="etf_ticker", symbols=etfs, start=start, end=end
    )
    params["benchmark"] = get_settings().benchmark
    query = sql.SQL(
        "SELECT etf_ticker AS ticker, date, rs_rank_index AS rs_rank, rs_z_score "
        "FROM fact_etf_metrics {where} AND ref_ticker = %(benchmark)s ORDER BY ticker, date"
    ).format(where=where)
    return db.read_df(db.Database.STOCKS, query, params)


def compute_indicators(ohlcv: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-bar indicators (package definitions) and RS metrics for each ETF."""
    frames = []
    for _, g in ohlcv.groupby("ticker", observed=True):
        g = g.sort_values("date").copy()
        close = g["close"].astype(float)
        volume = g["volume"].astype(float)
        g["rsi"] = compute_rsi(close, RSI_PERIOD)
        for period in MOMENTUM_PERIODS:
            g[f"mom_{period}"] = compute_momentum_series(close, period)
        g["ratio_50ma"] = compute_ratio_to_ma_series(close, SHORT_MA_PERIOD)
        g["ratio_200ma"] = compute_ratio_to_ma_series(close, LONG_MA_PERIOD)
        g["vol_ratio"] = volume / volume.rolling(VOLUME_AVG_PERIOD).mean()
        g["adr_pct"] = compute_adr_series(g["high"].astype(float), g["low"].astype(float), close)
        g["bb_width"] = bollinger_width(close)
        g["rsi_roc_5"] = g["rsi"].diff(RSI_CHANGE_BARS)
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", *INDICATORS])
    out = pd.concat(frames, ignore_index=True).merge(metrics, on=["ticker", "date"], how="left")
    by_ticker = out.groupby("ticker", observed=True)["rs_rank"]
    for window in RS_CHANGE_WINDOWS:
        out[f"rs_rank_chg_{window}"] = by_ticker.diff(window)
    return out


# ---------------------------------------------------------------------------
# Trend identification and profiles
# ---------------------------------------------------------------------------


def _best_end(closes: np.ndarray, i: int, rule: TrendRule, direction: str) -> tuple[int, float]:
    """Bar giving the largest qualifying move from bar ``i`` before the adverse limit is hit.

    Returns ``(end_index, move_pct)``, or ``(-1, 0.0)`` when no bar qualifies.
    """
    start = closes[i]
    extreme = start
    best_j, best_move = -1, 0.0
    for j in range(i + 1, min(i + rule.max_duration + 1, len(closes))):
        price = closes[j]
        if direction == "up":
            extreme = max(extreme, price)
            adverse = (extreme - price) / extreme * 100
            move = (price - start) / start * 100
        else:
            extreme = min(extreme, price)
            adverse = (price - extreme) / extreme * 100 if extreme > 0 else 0.0
            move = (start - price) / start * 100
        if adverse > rule.max_adverse_pct:
            break
        if move >= rule.min_move_pct and j - i >= rule.min_duration and move > best_move:
            best_j, best_move = j, move
    return best_j, best_move


def identify_trends(ohlcv: pd.DataFrame, rule: TrendRule, warmup: int) -> pd.DataFrame:
    """Sustained up- and downtrends per ticker, scanned greedily from bar ``warmup`` on.

    From each bar, the trend runs to the bar with the largest move of at least
    ``min_move_pct`` within ``max_duration`` bars, provided the adverse move from
    the running extreme never exceeds ``max_adverse_pct``. The scan then resumes
    after the trend ends. Up- and downtrends are scanned independently.
    """
    trends = []
    for ticker, g in ohlcv.groupby("ticker", observed=True):
        g = g.sort_values("date")
        closes = g["close"].to_numpy(dtype=float)
        dates = g["date"].to_numpy()
        for direction in ("up", "down"):
            i = warmup
            while i < len(closes) - rule.min_duration:
                j, move = _best_end(closes, i, rule, direction)
                if j < 0:
                    i += 1
                    continue
                trends.append(
                    {
                        "ticker": ticker,
                        "trend_start": dates[i],
                        "trend_end": dates[j],
                        "direction": direction,
                        "move_pct": move if direction == "up" else -move,
                        "duration": j - i,
                    }
                )
                i = j + 1
    columns = ["ticker", "trend_start", "trend_end", "direction", "move_pct", "duration"]
    return pd.DataFrame(trends, columns=columns).astype({"move_pct": float, "duration": int})


def _slope(values: pd.Series) -> float:
    """Least-squares slope per bar, ignoring missing values."""
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) < MIN_SLOPE_POINTS:
        return np.nan
    return float(np.polyfit(np.arange(len(vals)), vals, 1)[0])


def profile_precursors(
    trends: pd.DataFrame, indicators: pd.DataFrame, lookback: int
) -> pd.DataFrame:
    """Indicator values at each trend's first bar and summaries of the ``lookback`` bars to it."""
    by_ticker = {
        t: g.sort_values("date").reset_index(drop=True)
        for t, g in indicators.groupby("ticker", observed=True)
    }
    rows = []
    for trend in trends.itertuples(index=False):
        bars = by_ticker.get(trend.ticker)
        if bars is None:
            continue
        upto = bars[bars["date"] <= trend.trend_start]
        if len(upto) < lookback:
            continue
        window = upto.tail(lookback)
        at_start = window.iloc[-1]
        row: dict[str, object] = {
            "ticker": trend.ticker,
            "trend_start": trend.trend_start,
            "direction": trend.direction,
            "move_pct": trend.move_pct,
            "duration": trend.duration,
            **{col: at_start[col] for col in INDICATORS},
        }
        for name, (col, how) in LOOKBACK_FEATURES.items():
            row[name] = window[col].mean() if how == "mean" else _slope(window[col])
        rows.append(row)
    columns = ["ticker", "trend_start", "direction", "move_pct", "duration"]
    return pd.DataFrame(rows, columns=[*columns, *INDICATORS, *LOOKBACK_FEATURES])


def sample_baseline(
    indicators: pd.DataFrame, trends: pd.DataFrame, n_samples: int, exclusion: int, seed: int
) -> pd.DataFrame:
    """Random bars at least ``exclusion`` bars from any trend start of the same ticker."""
    frames = []
    starts = trends.groupby("ticker", observed=True)["trend_start"].apply(set).to_dict()
    for ticker, g in indicators.groupby("ticker", observed=True):
        g = g.sort_values("date").reset_index(drop=True)
        start_positions = np.flatnonzero(g["date"].isin(starts.get(ticker, set())).to_numpy())
        positions = np.arange(len(g))
        near = np.zeros(len(g), dtype=bool)
        for p in start_positions:
            near |= np.abs(positions - p) <= exclusion
        frames.append(g[~near])
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", *INDICATORS])
    valid = pd.concat(frames, ignore_index=True).dropna(subset=["rsi", "ratio_50ma", "rs_rank"])
    sample = valid if len(valid) <= n_samples else valid.sample(n=n_samples, random_state=seed)
    return sample[["ticker", "date", *INDICATORS]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _median(df: pd.DataFrame, col: str) -> float:
    return float(df[col].median()) if col in df.columns and df[col].notna().any() else np.nan


def comparison_table(up: pd.DataFrame, down: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Medians at trend start vs baseline, with Mann-Whitney tests (BH over all indicators)."""
    rows = []
    for col, label in INDICATORS.items():
        row: dict[str, object] = {
            "indicator": col,
            "label": label,
            "n_up": int(up[col].notna().sum()),
            "n_down": int(down[col].notna().sum()),
            "median_up": _median(up, col),
            "median_down": _median(down, col),
            "median_baseline": _median(baseline, col),
        }
        for name, group in (("up", up), ("down", down)):
            a, b = group[col].dropna(), baseline[col].dropna()
            row[f"p_{name}_vs_baseline"] = (
                float(stats.mannwhitneyu(a, b).pvalue) if len(a) and len(b) else np.nan
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    p_cols = ["p_up_vs_baseline", "p_down_vs_baseline"]
    adjusted = fdr_adjust(out[p_cols].to_numpy(dtype=float).ravel()).reshape(len(out), -1)
    out["p_fdr_up"], out["p_fdr_down"] = adjusted[:, 0], adjusted[:, 1]
    return out


def lookback_table(up: pd.DataFrame, down: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature": f, "median_up": _median(up, f), "median_down": _median(down, f)}
            for f in LOOKBACK_FEATURES
        ]
    )


def distribution_table(up: pd.DataFrame, down: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of selected indicators at trend start."""
    rows = []
    for col in DISTRIBUTION_COLUMNS:
        for direction, group in (("up", up), ("down", down)):
            vals = group[col].dropna()
            if vals.empty:
                continue
            q = vals.quantile(list(QUANTILES)).to_numpy()
            rows.append(
                {
                    "indicator": col,
                    "direction": direction,
                    "n": len(vals),
                    **{f"q{round(k * 100)}": v for k, v in zip(QUANTILES, q, strict=True)},
                }
            )
    return pd.DataFrame(rows)


def regime_split(profiles: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Medians for trends starting with wide vs narrow bands (split at the baseline median width).

    ``move_pct`` and ``duration`` describe trends selected after the fact; they
    are not the forward returns of any entry rule.
    """
    split = _median(baseline, "bb_width")
    rows = []
    for direction in ("up", "down"):
        group = profiles[profiles["direction"] == direction]
        for name, part in (
            ("wide_bands", group[group["bb_width"] >= split]),
            ("narrow_bands", group[group["bb_width"] < split]),
        ):
            vals = part["ratio_200ma"].dropna()
            row: dict[str, object] = {
                "direction": direction,
                "group": name,
                "bb_width_split": split,
                "n": len(part),
                "share_below_200ma": (vals < BELOW_200MA).mean() if len(vals) else np.nan,
                "share_near_200ma": vals.between(BELOW_200MA, ABOVE_200MA).mean()
                if len(vals)
                else np.nan,
                "share_above_200ma": (vals > ABOVE_200MA).mean() if len(vals) else np.nan,
            }
            row.update({f"median_{c}": _median(part, c) for c in SPLIT_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    defaults = TrendRule()
    parser = make_parser(__doc__.split("\n")[0])
    parser.add_argument(
        "--etfs",
        type=symbol_list,
        help="comma-separated (default: every ETF with RS metrics against the benchmark)",
    )
    parser.add_argument(
        "--min-move", type=float, default=defaults.min_move_pct, help="trend move, in percent"
    )
    parser.add_argument(
        "--min-duration", type=int, default=defaults.min_duration, help="shortest trend, in bars"
    )
    parser.add_argument(
        "--max-duration", type=int, default=defaults.max_duration, help="longest trend, in bars"
    )
    parser.add_argument(
        "--max-adverse",
        type=float,
        default=defaults.max_adverse_pct,
        help="largest move against the trend from its running extreme, in percent",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help="bars profiled before each trend start, and the baseline exclusion radius",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=DEFAULT_BASELINE_SAMPLES,
        help="random non-trend bars to compare the trend starts with",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="random seed for the baseline sample"
    )
    parser.add_argument(
        "--focus", type=symbol_list, help="also list the trends and profiles of these ETFs"
    )
    add_common_args(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    out_dir: Path = args.output_dir
    rule = TrendRule(args.min_move, args.min_duration, args.max_duration, args.max_adverse)

    metrics = load_etf_metrics(args.etfs, args.start, args.end)
    etfs = args.etfs or sorted(metrics["ticker"].unique())
    ohlcv = fetch_ohlcv("etf", etfs, start=args.start, end=args.end)
    logger.info("ETF bars: %d rows, %d ETFs", len(ohlcv), ohlcv["ticker"].nunique())
    indicators = compute_indicators(ohlcv, metrics)
    trends = identify_trends(ohlcv, rule, warmup=args.lookback)
    profiles = profile_precursors(trends, indicators, args.lookback)
    baseline = (
        sample_baseline(indicators, trends, args.baseline_samples, args.lookback, args.seed)
        if not indicators.empty
        else pd.DataFrame(columns=["ticker", "date", *INDICATORS])
    )
    up, down = (profiles[profiles["direction"] == d] for d in ("up", "down"))
    logger.info("%d uptrends, %d downtrends; %d baseline bars", len(up), len(down), len(baseline))
    if profiles.empty:
        logger.warning("no trends matched the rule; loosen --min-move/--max-adverse or add history")

    tables = {
        "trends": trends,
        "profiles": profiles,
        "baseline": baseline,
        "comparison": comparison_table(up, down, baseline),
        "lookback": lookback_table(up, down),
        "distribution": distribution_table(up, down),
        "regime_split": regime_split(profiles, baseline),
    }
    print_table("Largest uptrends", trends[trends["direction"] == "up"].nlargest(10, "move_pct"))
    print_table(
        "Largest downtrends", trends[trends["direction"] == "down"].nsmallest(10, "move_pct")
    )
    print_table("Median readings at trend start vs baseline", tables["comparison"])
    print_table(f"Lookback window ({args.lookback} bars) before trend start", tables["lookback"])
    print_table("Distribution at trend start", tables["distribution"])
    print_table("Wide vs narrow Bollinger bands at trend start", tables["regime_split"])
    for ticker in args.focus or []:
        print_table(f"{ticker}: trends", trends[trends["ticker"] == ticker])
        print_table(f"{ticker}: profiles", profiles[profiles["ticker"] == ticker])
    for name, table in tables.items():
        write_csv(table, out_dir / f"trend_{name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
