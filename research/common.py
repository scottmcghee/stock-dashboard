"""Helpers shared by the research scripts: CLI plumbing, output and the statistical scans."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2 import sql
from scipy import stats

from stock_dashboard.logic.trend_signals import BB_PERIOD, BB_STD_MULTIPLE
from stock_dashboard.queries.stocks_queries import fetch_ohlcv

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("research_results")
SIGNIFICANCE_LEVEL = 0.05
HIT_QUANTILE = 0.90  # a "hit" is a target value in the top decile of the whole sample
DECILES = 10
TOP_QUANTILE = 0.90
BOTTOM_QUANTILE = 0.10
MIN_DISTINCT_VALUES = 4  # fewer distinct signal values cannot form deciles
BPS = 10_000


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def symbol_list(text: str) -> list[str]:
    """Parse a comma-separated ticker list (argparse ``type``)."""
    symbols = [s.strip().upper() for s in text.split(",") if s.strip()]
    if not symbols:
        raise argparse.ArgumentTypeError("expected at least one ticker")
    return symbols


def int_list(text: str) -> list[int]:
    """Parse a comma-separated list of positive integers (argparse ``type``)."""
    try:
        values = [int(v) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected positive integers, e.g. 1,5,10")
    return values


def iso_date(text: str) -> date:
    """Parse a YYYY-MM-DD date (argparse ``type``)."""
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Appends each option's default to its help, unless it has none or the help names it."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default in (None, False) or "default:" in (action.help or ""):
            return action.help
        return super()._get_help_string(action)


def make_parser(description: str) -> argparse.ArgumentParser:
    """An argument parser whose ``--help`` shows the defaults."""
    return argparse.ArgumentParser(description=description, formatter_class=HelpFormatter)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add --start/--end, --output-dir and --log-level."""
    parser.add_argument("--start", type=iso_date, help="first date (default: all history)")
    parser.add_argument("--end", type=iso_date, help="last date (default: latest data)")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for the CSV outputs"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"], help="logging level"
    )


def where_clause(
    *,
    date_col: str,
    symbol_col: str,
    symbols: Sequence[str] | None,
    start: date | None,
    end: date | None,
) -> tuple[sql.Composable, dict[str, object]]:
    """A ``WHERE`` clause (identifiers quoted, values bound) restricting symbols and dates."""
    clauses: list[sql.Composable] = [sql.SQL("TRUE")]
    params: dict[str, object] = {}
    if symbols is not None:
        clauses.append(sql.SQL("{} = ANY(%(symbols)s)").format(sql.Identifier(symbol_col)))
        params["symbols"] = list(symbols)
    if start is not None:
        clauses.append(sql.SQL("{} >= %(start)s").format(sql.Identifier(date_col)))
        params["start"] = start
    if end is not None:
        clauses.append(sql.SQL("{} <= %(end)s").format(sql.Identifier(date_col)))
        params["end"] = end
    return sql.SQL("WHERE ") + sql.SQL(" AND ").join(clauses), params


def load_closes(
    symbols: Sequence[str] | None, start: date | None, end: date | None
) -> pd.DataFrame:
    """Daily closes (ticker, date, close) of stocks and ETFs, from both price tables."""
    frames = [
        fetch_ohlcv(kind, symbols, start=start, end=end, columns=("close",))
        for kind in ("stock", "etf")
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close"])
    closes = pd.concat(frames, ignore_index=True)
    closes["date"] = pd.to_datetime(closes["date"])
    return closes.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path`` (creating its directory) and log it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("wrote %s (%d rows)", path, len(df))
    return path


def print_table(title: str, df: pd.DataFrame) -> None:
    """Print a titled report table, or a note when it is empty."""
    print(f"\n=== {title} ===")
    if df.empty:
        print("  (no rows: too little data, or nothing passed the thresholds)")
    else:
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def fdr_adjust(p_values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaN inputs stay NaN and are not counted in m."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    if valid.any():
        out[valid] = stats.false_discovery_control(p[valid], method="bh")
    return out


def univariate_scan(
    df: pd.DataFrame, target: str, signal_cols: Sequence[str], min_n: int
) -> pd.DataFrame:
    """Spearman correlation and decile lift of each signal against ``target``.

    For each signal with at least ``min_n`` joint observations: Spearman rho
    and its p-value, the mean target in the signal's top and bottom deciles,
    their difference (``lift_bps``), and the share of top-decile rows whose
    target lands in the sample's top decile (``top_dec_hit_rate``, baseline
    ``overall_hit_rate`` ~0.10). ``p_fdr`` is BH-adjusted over the signals tested.
    """
    y = df[target]
    hit_threshold = y.quantile(HIT_QUANTILE)
    rows = []
    for col in signal_cols:
        if col not in df.columns:
            continue
        mask = df[col].notna() & y.notna()
        if mask.sum() < min_n:
            continue
        x, yv = df.loc[mask, col].astype(float), y[mask]
        if x.nunique() < MIN_DISTINCT_VALUES:
            continue
        rho, p = stats.spearmanr(x, yv)
        # Tied values share a bin, so a discrete signal can have fewer than ten.
        deciles = pd.qcut(x, DECILES, labels=False, duplicates="drop")
        top, bottom = yv[deciles == deciles.max()], yv[deciles == deciles.min()]
        rows.append(
            {
                "signal": col,
                "n": int(mask.sum()),
                "spearman_rho": float(rho),
                "p_value": float(p),
                "top_dec_mean": float(top.mean()),
                "bot_dec_mean": float(bottom.mean()),
                "overall_mean": float(yv.mean()),
                "lift_bps": float(top.mean() - bottom.mean()) * BPS,
                "top_dec_hit_rate": float((top >= hit_threshold).mean()),
                "overall_hit_rate": float((yv >= hit_threshold).mean()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = fdr_adjust(out["p_value"])
    return out.sort_values("p_value").reset_index(drop=True)


def group_replication(
    df: pd.DataFrame, target: str, signals: Sequence[str], group_col: str, min_n: int
) -> pd.DataFrame:
    """Spearman rho of each signal against ``target`` within each value of ``group_col``."""
    rows = []
    for key, g in df.groupby(group_col, observed=True):
        if len(g) < min_n:
            continue
        for col in signals:
            if col not in g.columns:
                continue
            mask = g[col].notna() & g[target].notna()
            if mask.sum() < min_n or g.loc[mask, col].nunique() < MIN_DISTINCT_VALUES:
                continue
            rho, p = stats.spearmanr(g.loc[mask, col], g.loc[mask, target])
            rows.append({group_col: key, "signal": col, "n": int(mask.sum()), "rho": rho, "p": p})
    return pd.DataFrame(rows, columns=[group_col, "signal", "n", "rho", "p"])


def summarize_replication(rep: pd.DataFrame, signals: Sequence[str]) -> pd.DataFrame:
    """Per signal: groups tested, mean rho, and the share of groups agreeing in sign.

    ``pct_sig_same_sign`` counts groups that are individually significant
    (unadjusted p < 0.05) with the same sign as the mean rho.
    """
    rows = []
    for sig in signals:
        sub = rep[rep["signal"] == sig]
        if sub.empty:
            continue
        mean_rho = sub["rho"].mean()
        same_sign = (sub["rho"] > 0) == (mean_rho > 0)
        rows.append(
            {
                "signal": sig,
                "n_groups": len(sub),
                "mean_rho": mean_rho,
                "pct_same_sign": same_sign.mean(),
                "pct_sig_same_sign": (same_sign & (sub["p"] < SIGNIFICANCE_LEVEL)).mean(),
                "min_rho": sub["rho"].min(),
                "max_rho": sub["rho"].max(),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("pct_sig_same_sign", ascending=False).reset_index(drop=True)


def replicating_signals(rep: pd.DataFrame, need: int) -> set[str]:
    """Signals significant (p < 0.05), with their mean rho's sign, in at least ``need`` groups."""
    out = set()
    for sig, sub in rep.groupby("signal"):
        mean_rho = sub["rho"].mean()
        agree = ((sub["rho"] > 0) == (mean_rho > 0)) & (sub["p"] < SIGNIFICANCE_LEVEL)
        if int(agree.sum()) >= need:
            out.add(str(sig))
    return out


def decile_mask(x: pd.Series, direction: str) -> pd.Series:
    """Rows in the top (>= 90th percentile) or bottom (<= 10th) decile of ``x``."""
    if direction == "top":
        return x >= x.quantile(TOP_QUANTILE)
    return x <= x.quantile(BOTTOM_QUANTILE)


def pair_scan(
    df: pd.DataFrame,
    target: str,
    signals: Sequence[str],
    min_n: int,
    directions: Sequence[str] = ("top", "bottom"),
) -> pd.DataFrame:
    """Mean ``target`` when two signals are both in an extreme decile, against the overall mean.

    Every unordered pair of ``signals`` and every combination of ``directions``
    is one test: a z-test of the conditional mean against the full-sample
    mean (treated as known). ``p_fdr`` is BH-adjusted over all tests run.
    """
    y = df[target]
    hit_threshold = y.quantile(HIT_QUANTILE)
    overall_mean = y.mean()
    masks = {
        (sig, d): decile_mask(df[sig], d)
        for sig in signals
        if sig in df.columns
        for d in directions
    }
    rows = []
    for a, b in combinations([s for s in signals if s in df.columns], 2):
        for dir_a in directions:
            for dir_b in directions:
                mask = masks[a, dir_a] & masks[b, dir_b] & y.notna()
                n = int(mask.sum())
                if n < min_n:
                    continue
                yv = y[mask]
                se = yv.std() / np.sqrt(n)
                z = (yv.mean() - overall_mean) / se if se > 0 else np.nan
                rows.append(
                    {
                        "signal_a": a,
                        "dir_a": dir_a,
                        "signal_b": b,
                        "dir_b": dir_b,
                        "n": n,
                        "cond_mean": yv.mean(),
                        "lift_vs_overall_bps": (yv.mean() - overall_mean) * BPS,
                        "cond_hit_rate": (yv >= hit_threshold).mean(),
                        "z": z,
                        "p_value": 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = fdr_adjust(out["p_value"])
    return out.sort_values("p_value").reset_index(drop=True)


def forward_returns(
    prices: pd.DataFrame, horizons: Sequence[int], key: str = "ticker"
) -> pd.DataFrame:
    """Close-to-close forward returns ``fwd_<h>`` per ``key``, ``h`` rows (sessions) ahead.

    ``prices`` needs ``key``, ``date`` and ``close``; the result adds one column per horizon.
    """
    out = prices.sort_values([key, "date"]).reset_index(drop=True)
    close = out["close"].astype(float).where(out["close"] > 0)
    grouped = close.groupby(out[key], observed=True)
    for h in horizons:
        out[f"fwd_{h}"] = grouped.shift(-h) / close - 1.0
    return out


def bollinger_bands(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(middle, upper, lower) bands with the trend scans' period and width.

    Band width as a percentage of the middle band, ``(upper - lower) / middle * 100``,
    is the ``bb_width`` of :mod:`stock_dashboard.logic.trend_signals`.
    """
    middle = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    spread = BB_STD_MULTIPLE * close.rolling(BB_PERIOD, min_periods=BB_PERIOD).std()
    return middle, middle + spread, middle - spread


def bollinger_width(close: pd.Series) -> pd.Series:
    """Bollinger band width in percent of the middle band (see :func:`bollinger_bands`)."""
    middle, upper, lower = bollinger_bands(close)
    return (upper - lower) / middle.where(middle != 0) * 100
