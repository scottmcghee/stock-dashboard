"""Technical-indicator signal discovery on the stock universe.

Scans the backtester's indicator library (plus a few extra scalar features)
for monotone association with forward returns, checks whether each top
signal replicates across market regimes, sectors and the two halves of the
sample, tests event flags against the rows where they did not fire, and
searches pairs of top signals in their extreme deciles. Pairs whose filters
the backtester can replay can optionally be exported as presets.

Usage:
    python -m research.ta_signal_analysis --help
    python -m research.ta_signal_analysis --universe-size 500 --start 2021-01-01
    python -m research.ta_signal_analysis --target fwd_20 --export-presets
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.common import (
    BOTTOM_QUANTILE,
    BPS,
    HIT_QUANTILE,
    SIGNIFICANCE_LEVEL,
    TOP_QUANTILE,
    add_common_args,
    bollinger_bands,
    configure_logging,
    fdr_adjust,
    group_replication,
    make_parser,
    pair_scan,
    print_table,
    replicating_signals,
    summarize_replication,
    univariate_scan,
    write_csv,
)
from stock_dashboard import db
from stock_dashboard.config import get_settings
from stock_dashboard.logic.backtest import (
    FILTER_TO_COLUMN,
    HORIZONS,
    FilterSpec,
    Strategy,
    build_backtest_dataframe,
    classify_regimes,
    evaluate_all_strategies,
)
from stock_dashboard.logic.backtest_report import preset_stats
from stock_dashboard.logic.events import EVENT_COLUMNS
from stock_dashboard.logic.preset import build_preset, save_preset
from stock_dashboard.queries.backtest_queries import (
    load_benchmark_ohlcv,
    load_metrics_universe,
    load_ohlcv_universe,
)

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE_SIZE = 2000
LIQUIDITY_BARS = 63  # sessions of median dollar volume that rank the universe
BETA_WINDOW = 63
BETA_MIN_PERIODS = 30
WEEKS_52_BARS = 252
WEEKS_52_MIN_PERIODS = 40
ATR_PERIOD = 14
DEFAULT_TARGET = "resid_fwd_10"
TARGETS = [f"{prefix}fwd_{h}" for prefix in ("resid_", "") for h in HORIZONS]

DEFAULT_MIN_N = 500
DEFAULT_MIN_GROUP_N = 500
DEFAULT_MIN_TIME_HALF_N = 200
DEFAULT_MIN_EVENT_N = 50
DEFAULT_MIN_PAIR_N = 100
DEFAULT_TOP_K = 8
DEFAULT_MIN_HIT_RATE = 0.25
MAX_PAIRS_REPORTED = 50
PRESET_PREFIX = "TA-disc"

SIGNAL_COLS = [
    # Overbought/oversold
    "rsi_14",
    "bb_pctb",
    "dist_52w_high",
    "dist_52w_low",
    # Momentum
    "momentum_20",
    "momentum_63",
    "momentum_126",
    "macd_histogram",
    "rs_impulse_5",
    "rs_impulse_10",
    "rs_impulse_20",
    "rs_impulse_42",
    # Trend / position
    "ratio_to_50ma",
    "ratio_to_200ma",
    "weinstein_stage_num",
    "rs_rank_overall",
    "rs_z_score",
    # Volatility / range
    "adr_14",
    "bb_width",
    "atr_over_price",
    # Volume
    "volume_z",
    # Support/resistance proximity
    "dist_recent_high_63",
    "dist_recent_high_126",
    "dist_recent_low_63",
    "dist_recent_low_126",
]


def _filter_pairs() -> dict[str, tuple[str, str]]:
    """Indicator column -> (min filter, max filter), for columns the backtester filters both ways.

    Derived from ``backtest.FILTER_TO_COLUMN`` so a preset can only use filters
    that the backtest and the chart overlay evaluate.
    """
    bounds: dict[str, dict[str, str]] = {}
    for name, col in FILTER_TO_COLUMN.items():
        for suffix in ("min", "max"):
            if name.endswith(f"_{suffix}"):
                bounds.setdefault(col, {})[suffix] = name
    return {col: (b["min"], b["max"]) for col, b in bounds.items() if len(b) == 2}


SIGNAL_TO_FILTER = {col: names for col, names in _filter_pairs().items() if col in SIGNAL_COLS}
NOT_EXPORTABLE = [col for col in SIGNAL_COLS if col not in SIGNAL_TO_FILTER]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def select_liquid_universe(
    ohlcv: pd.DataFrame, top_n: int, bars: int = LIQUIDITY_BARS
) -> list[str]:
    """Top-N tickers by median dollar volume over the last ``bars`` sessions of the data."""
    sessions = np.sort(ohlcv["date"].unique())
    if len(sessions) == 0:
        return []
    window = ohlcv[ohlcv["date"] >= sessions[-min(bars, len(sessions))]]
    dollar_volume = (window["close"].astype(float) * window["volume"].astype(float)).groupby(
        window["ticker"], observed=True
    )
    ranked = dollar_volume.median().sort_values(ascending=False)
    return [str(t) for t in ranked.head(top_n).index]


def load_sector_map(tickers: Sequence[str]) -> dict[str, str]:
    """Sector per ticker from dim_stock ("Unknown" when missing)."""
    df = db.read_df(
        db.Database.STOCKS,
        "SELECT ticker, sector FROM dim_stock WHERE ticker = ANY(%(tickers)s)",
        {"tickers": list(tickers)},
    )
    return dict(zip(df["ticker"], df["sector"].fillna("Unknown"), strict=True))


def _date_window(df: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    if start is not None:
        keep &= df["date"] >= pd.Timestamp(start)
    if end is not None:
        keep &= df["date"] <= pd.Timestamp(end)
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Residualization against the benchmark
# ---------------------------------------------------------------------------


def benchmark_forward_returns(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Benchmark close-to-close returns over each backtest horizon: date, bench_fwd_<h>."""
    b = benchmark.sort_values("date")
    close = b["close"].astype(float)
    out = b[["date"]].copy()
    for h in HORIZONS:
        out[f"bench_fwd_{h}"] = close.shift(-h) / close - 1.0
    return out


def rolling_beta(ohlcv: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, date) beta of daily log returns on the benchmark's, over the trailing window."""
    bench = benchmark.sort_values("date").set_index("date")["close"].astype(float)
    bench_ret = (bench / bench.shift(1)).apply(np.log)
    frames = []
    for ticker, g in ohlcv.groupby("ticker", observed=True):
        g = g.sort_values("date")
        close = pd.Series(g["close"].astype(float).to_numpy(), index=g["date"])
        ret = (close / close.shift(1)).apply(np.log)
        mkt = bench_ret.reindex(g["date"])
        cov = ret.rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).cov(mkt)
        var = mkt.rolling(BETA_WINDOW, min_periods=BETA_MIN_PERIODS).var()
        frames.append(
            pd.DataFrame(
                {"date": g["date"].to_numpy(), "ticker": ticker, "beta": (cov / var).to_numpy()}
            )
        )
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "beta"])
    return pd.concat(frames, ignore_index=True)


def residualize_forward_returns(
    panel: pd.DataFrame, bench_fwd: pd.DataFrame, beta: pd.DataFrame
) -> pd.DataFrame:
    """``resid_fwd_h = fwd_h - beta * bench_fwd_h``; beta defaults to 1 before it is estimable."""
    panel = panel.merge(bench_fwd, on="date", how="left").merge(
        beta, on=["date", "ticker"], how="left"
    )
    b = panel["beta"].fillna(1.0)
    for h in HORIZONS:
        panel[f"resid_fwd_{h}"] = (panel[f"fwd_{h}"] - b * panel[f"bench_fwd_{h}"]).astype(
            np.float32
        )
    return panel


# ---------------------------------------------------------------------------
# Extra scalar features
# ---------------------------------------------------------------------------


def add_extra_features(ohlcv: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """52-week distance, Bollinger %B and width, ATR as % of price."""
    frames = []
    for ticker, g in ohlcv.groupby("ticker", observed=True):
        g = g.sort_values("date")
        close, high, low = (g[c].astype(float) for c in ("close", "high", "low"))
        sma, upper, lower = bollinger_bands(close)
        prev_close = close.shift(1)
        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = true_range.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
        rolling_52w = close.rolling(WEEKS_52_BARS, min_periods=WEEKS_52_MIN_PERIODS)
        frames.append(
            pd.DataFrame(
                {
                    "date": g["date"].to_numpy(),
                    "ticker": ticker,
                    "dist_52w_high": (close / rolling_52w.max() - 1.0).to_numpy(),
                    "dist_52w_low": (close / rolling_52w.min() - 1.0).to_numpy(),
                    "bb_pctb": ((close - lower) / (upper - lower).replace(0, np.nan)).to_numpy(),
                    "bb_width": ((upper - lower) / sma.replace(0, np.nan) * 100).to_numpy(),
                    "atr_over_price": (atr / close.replace(0, np.nan) * 100).to_numpy(),
                }
            )
        )
    if not frames:
        return panel
    extra = pd.concat(frames, ignore_index=True)
    value_cols = extra.columns.difference(["date", "ticker"])
    extra[value_cols] = extra[value_cols].astype(np.float32)
    return panel.merge(extra, on=["date", "ticker"], how="left")


# ---------------------------------------------------------------------------
# Event analysis
# ---------------------------------------------------------------------------


def event_conditional_analysis(
    panel: pd.DataFrame,
    target: str,
    event_cols: Sequence[str],
    min_n: int,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Forward ``target`` when each event fired vs when it did not (Welch t-test).

    One row per event (or per (group, event)); ``p_fdr`` is BH-adjusted over all rows.
    """
    hit_threshold = panel[target].quantile(HIT_QUANTILE)
    groups = [(None, panel)] if group_col is None else panel.groupby(group_col, observed=True)
    rows = []
    for key, g in groups:
        y = g[target]
        for ev in event_cols:
            if ev not in g.columns:
                continue
            fired_mask = g[ev].fillna(False).astype(bool)
            fired, quiet = y[fired_mask].dropna(), y[~fired_mask].dropna()
            if min(len(fired), len(quiet)) < min_n:
                continue
            t, p = stats.ttest_ind(fired, quiet, equal_var=False)
            row: dict[str, object] = {
                "event": ev,
                "n": len(fired),
                "cond_mean": fired.mean(),
                "not_fired_mean": quiet.mean(),
                "lift_bps": (fired.mean() - quiet.mean()) * BPS,
                "cond_hit_rate": (fired >= hit_threshold).mean(),
                "not_fired_hit_rate": (quiet >= hit_threshold).mean(),
                "t_stat": float(t),
                "p_value": float(p),
            }
            if group_col is not None:
                row = {group_col: key, **row}
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = fdr_adjust(out["p_value"])
    return out.sort_values("p_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pairs and preset export
# ---------------------------------------------------------------------------


def discover_pairs(
    panel: pd.DataFrame,
    target: str,
    top_signals: Sequence[str],
    per_regime: pd.DataFrame,
    per_time: pd.DataFrame,
    *,
    min_n: int,
    min_hit_rate: float,
    min_regimes: int,
    min_time_halves: int,
) -> pd.DataFrame:
    """Signal pairs, both in an extreme decile, that pass FDR, hit-rate and replication checks.

    Replication: each signal of the pair must be individually significant with
    a consistent sign in at least ``min_regimes`` regimes and ``min_time_halves``
    halves of the sample.
    """
    pairs = pair_scan(panel, target, top_signals, min_n)
    if pairs.empty:
        return pairs
    replicated = replicating_signals(per_regime, min_regimes) & replicating_signals(
        per_time, min_time_halves
    )
    keep = (
        (pairs["p_fdr"] < SIGNIFICANCE_LEVEL)
        & (pairs["cond_hit_rate"] >= min_hit_rate)
        & pairs["signal_a"].isin(replicated)
        & pairs["signal_b"].isin(replicated)
    )
    out = pairs[keep].head(MAX_PAIRS_REPORTED).reset_index(drop=True)
    out["exportable"] = out["signal_a"].isin(SIGNAL_TO_FILTER) & out["signal_b"].isin(
        SIGNAL_TO_FILTER
    )
    return out


def decile_filter(panel: pd.DataFrame, signal: str, direction: str) -> FilterSpec:
    """Backtester filter selecting ``signal``'s top or bottom decile (cut-off from ``panel``)."""
    min_name, max_name = SIGNAL_TO_FILTER[signal]
    if direction == "top":
        return FilterSpec(min_name, round(float(panel[signal].quantile(TOP_QUANTILE)), 6))
    return FilterSpec(max_name, round(float(panel[signal].quantile(BOTTOM_QUANTILE)), 6))


def export_presets(
    panel: pd.DataFrame, pairs: pd.DataFrame, regimes: pd.Series, preset_dir: Path | None
) -> list[Path]:
    """Backtest the exportable pairs over the panel and save each as a preset.

    All pairs are evaluated in one :func:`evaluate_all_strategies` call, so
    each preset's ``p_value_adj`` is BH-adjusted across every exported pair
    and horizon (not across the pair search that found them).
    """
    rows = pairs[pairs["exportable"]].to_dict("records")
    strategies = []
    for row in rows:
        a, dir_a, b, dir_b = (str(row[k]) for k in ("signal_a", "dir_a", "signal_b", "dir_b"))
        filters = [decile_filter(panel, a, dir_a), decile_filter(panel, b, dir_b)]
        strategies.append(
            Strategy(name=f"{PRESET_PREFIX}_{a}_{dir_a}__{b}_{dir_b}", filters=filters)
        )
    results = evaluate_all_strategies(panel, strategies, regimes) if strategies else []

    written = []
    for row, result in zip(rows, results, strict=True):
        name = result.strategy.name
        if not result.horizons:
            logger.info("skipping preset %s: too few signals to backtest", name)
            continue
        a, dir_a, b, dir_b = (str(row[k]) for k in ("signal_a", "dir_a", "signal_b", "dir_b"))
        preset = build_preset(
            name,
            {f.name: f.value for f in result.strategy.filters},
            description=(
                f"Discovered by ta_signal_analysis: {a} in its {dir_a} decile and "
                f"{b} in its {dir_b} decile (in-sample cut-offs)."
            ),
            stats=preset_stats(result),
        )
        preset["discovery"] = {
            "source": "ta_signal_analysis",
            "n_discovery": int(row["n"]),
            "cond_hit_rate_disc": round(float(row["cond_hit_rate"]), 3),
            "lift_vs_overall_bps_disc": round(float(row["lift_vs_overall_bps"]), 1),
            "p_fdr_disc": round(float(row["p_fdr"]), 4),
        }
        written.append(save_preset(preset, preset_dir))
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_panel(
    universe_size: int, start: date | None, end: date | None
) -> tuple[pd.DataFrame, pd.Series]:
    """Indicators, forward and residual returns, regime, sector and time half per (date, ticker).

    Also returns the regime of each date (empty when there is no data).
    """
    ohlcv = _date_window(load_ohlcv_universe(), start, end)
    metrics = _date_window(load_metrics_universe(), start, end)
    benchmark = _date_window(load_benchmark_ohlcv(), start, end)
    logger.info("stocks: %d tickers, %d rows", ohlcv["ticker"].nunique(), len(ohlcv))

    liquid = select_liquid_universe(ohlcv, universe_size)
    ohlcv = ohlcv[ohlcv["ticker"].isin(liquid)].reset_index(drop=True)
    metrics = metrics[metrics["ticker"].isin(liquid)].reset_index(drop=True)
    logger.info("universe: top %d by median dollar volume, %d rows", len(liquid), len(ohlcv))
    if ohlcv.empty or benchmark.empty:
        return pd.DataFrame(), pd.Series(dtype=object)

    panel = build_backtest_dataframe(ohlcv, metrics)
    panel = residualize_forward_returns(
        panel, benchmark_forward_returns(benchmark), rolling_beta(ohlcv, benchmark)
    )
    panel = add_extra_features(ohlcv, panel)

    regimes = classify_regimes(benchmark)
    panel["regime"] = panel["date"].map(regimes)
    panel["sector"] = panel["ticker"].map(load_sector_map(liquid)).fillna("Unknown")
    midpoint = panel["date"].quantile(0.5)
    panel["time_half"] = np.where(panel["date"] < midpoint, "first_half", "second_half")
    return panel, regimes


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser(__doc__.split("\n")[0])
    parser.add_argument(
        "--universe-size",
        type=int,
        default=DEFAULT_UNIVERSE_SIZE,
        help=f"most liquid N stocks (median dollar volume, last {LIQUIDITY_BARS} sessions)",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=TARGETS,
        help="forward return to explain: raw fwd_<h> or benchmark-residual resid_fwd_<h>",
    )
    parser.add_argument(
        "--top-k-for-pairs",
        type=int,
        default=DEFAULT_TOP_K,
        help="the strongest K single signals are replicated and paired",
    )
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N, help="rows per signal scan")
    parser.add_argument(
        "--min-group-n", type=int, default=DEFAULT_MIN_GROUP_N, help="rows per regime/sector"
    )
    parser.add_argument(
        "--min-time-half-n",
        type=int,
        default=DEFAULT_MIN_TIME_HALF_N,
        help="rows per half of the sample",
    )
    parser.add_argument(
        "--min-event-n", type=int, default=DEFAULT_MIN_EVENT_N, help="fired rows per event test"
    )
    parser.add_argument(
        "--min-pair-n", type=int, default=DEFAULT_MIN_PAIR_N, help="rows per signal pair"
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=DEFAULT_MIN_HIT_RATE,
        help="pairs need this share of rows in the target's top decile (baseline ~0.10)",
    )
    parser.add_argument(
        "--min-regimes-replicate",
        type=int,
        default=1,
        help="regimes in which each signal of a pair must be significant with the same sign",
    )
    parser.add_argument(
        "--min-time-halves-replicate",
        type=int,
        default=1,
        help="halves of the sample in which each signal of a pair must be significant",
    )
    parser.add_argument("--skip-events", action="store_true", help="skip the event tests")
    parser.add_argument(
        "--export-presets", action="store_true", help="save exportable pairs as presets"
    )
    parser.add_argument(
        "--preset-dir", type=Path, help="where to save presets (default: PRESET_DIR)"
    )
    add_common_args(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    out_dir: Path = args.output_dir
    target: str = args.target
    stem = f"universe{args.universe_size}_{target}"

    logger.info("benchmark: %s", get_settings().benchmark)
    panel, regimes = build_panel(args.universe_size, args.start, args.end)
    if panel.empty or panel[target].notna().sum() == 0:
        logger.warning("no stock or benchmark data in the selected range; nothing to analyze")
        return 0
    if not args.skip_events:
        rates = panel[[c for c in EVENT_COLUMNS if c in panel.columns]].astype(float).mean()
        print_table(
            "Event fire rates", rates.rename("fire_rate").rename_axis("event").reset_index()
        )
    panel = panel.dropna(subset=[target]).reset_index(drop=True)
    logger.info("panel: %d rows with a %s value", len(panel), target)

    uni = univariate_scan(panel, target, SIGNAL_COLS, args.min_n)
    print_table(f"Univariate scan (target={target}, m={len(uni)} signals)", uni)
    write_csv(uni, out_dir / f"ta_univariate_{stem}.csv")
    top_signals = uni.head(args.top_k_for_pairs)["signal"].tolist() if not uni.empty else []

    reps = {
        "regime": group_replication(panel, target, top_signals, "regime", args.min_group_n),
        "sector": group_replication(panel, target, top_signals, "sector", args.min_group_n),
        "time": group_replication(panel, target, top_signals, "time_half", args.min_time_half_n),
    }
    for name, rep in reps.items():
        print_table(f"Per-{name} replication", summarize_replication(rep, top_signals))
        write_csv(rep, out_dir / f"ta_per_{name}_{stem}.csv")

    if not args.skip_events:
        overall = event_conditional_analysis(panel, target, EVENT_COLUMNS, args.min_event_n)
        print_table("Events: fired vs not fired (Welch t-test)", overall)
        write_csv(overall, out_dir / f"ta_events_overall_{stem}.csv")
        per_regime = event_conditional_analysis(
            panel, target, EVENT_COLUMNS, args.min_event_n, group_col="regime"
        )
        print_table("Events per regime", per_regime)
        write_csv(per_regime, out_dir / f"ta_events_per_regime_{stem}.csv")

    pairs = discover_pairs(
        panel,
        target,
        top_signals,
        reps["regime"],
        reps["time"],
        min_n=args.min_pair_n,
        min_hit_rate=args.min_hit_rate,
        min_regimes=args.min_regimes_replicate,
        min_time_halves=args.min_time_halves_replicate,
    )
    print_table(f"Pairs passing FDR, hit-rate and replication (top {len(top_signals)})", pairs)
    write_csv(pairs, out_dir / f"ta_pairs_{stem}.csv")

    if args.export_presets and not pairs.empty:
        skipped = pairs[~pairs["exportable"]]
        if not skipped.empty:
            missing = (set(skipped["signal_a"]) | set(skipped["signal_b"])) - set(SIGNAL_TO_FILTER)
            print(
                f"{len(skipped)} pairs not exported: the backtester has no filter for "
                + ", ".join(sorted(missing))
            )
        for path in export_presets(panel, pairs, regimes, args.preset_dir):
            print(f"preset written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
