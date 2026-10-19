"""Underlying-level option metrics against the forward returns of the underlying.

Bins each symbol-day by an option-positioning reading (net flow, delta and
gamma exposure and their changes, option volume against its average,
put/call ratio change, vanna exposure) and reports the mean forward
close-to-close return of the underlying per bin, with a Welch t-test of each
bin against all other rows and Benjamini-Hochberg control over every test.

Usage:
    python -m research.signal_analysis --help
    python -m research.signal_analysis --symbols ABC,XYZ --start 2025-07-01 --horizons 1,5,10
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import date
from typing import NamedTuple

import numpy as np
import pandas as pd
from psycopg2 import sql
from scipy import stats

from research.common import (
    BPS,
    add_common_args,
    configure_logging,
    fdr_adjust,
    forward_returns,
    int_list,
    load_closes,
    make_parser,
    print_table,
    symbol_list,
    where_clause,
    write_csv,
)
from stock_dashboard import db
from stock_dashboard.logic.option_signals import VOLUME_AVG_SESSIONS, gex_bucket_label

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = (1, 5, 10)
DEFAULT_MIN_BIN_N = 30
OUTPUT_FILE = "signal_bins.csv"

UNDERLYING_COLUMNS = ("flownet", "dxoi", "gxoi", "option_volume", "put_call_ratio", "vannaxoi")

FLOW_DECILES = 10
VANNA_QUINTILES = 5
HIGH_FLOW_DECILE = 8  # deciles 8-9 of a symbol's flownet history
LOW_FLOW_DECILE = 1  # deciles 0-1
VOLUME_SPIKE_MAJOR = 2.0
VOLUME_SPIKE_MINOR = 1.5
# Day-over-day changes beyond these bounds are treated as artefacts of a near-zero base.
PCR_CHANGE_BOUND = 2.0
GXOI_CHANGE_BOUND = 5.0

Bins = tuple[list[float], list[str]]
INF = np.inf
DXOI_CHANGE_BINS: Bins = (
    [-INF, -0.20, -0.05, 0.05, 0.20, INF],
    ["big_decrease", "moderate_decrease", "flat", "moderate_increase", "big_increase"],
)
VOL_RATIO_BINS: Bins = (
    [-INF, 0.8, 1.2, VOLUME_SPIKE_MINOR, VOLUME_SPIKE_MAJOR, INF],
    ["low", "normal", "elevated", "spike_1.5-2x", "spike_>2x"],
)
PCR_CHANGE_BINS: Bins = (
    [-INF, -0.30, -0.10, 0.10, 0.30, INF],
    ["big_call_shift", "call_shift", "stable", "put_shift", "big_put_shift"],
)
GXOI_CHANGE_BINS: Bins = (
    [-INF, -0.50, -0.10, 0.10, 0.50, INF],
    ["gex_collapse", "gex_decrease", "gex_stable", "gex_increase", "gex_surge"],
)

Labeler = Callable[[pd.DataFrame], pd.Series]


class SignalSpec(NamedTuple):
    name: str
    description: str
    label: Labeler


def load_underlying(
    symbols: Sequence[str] | None, start: date | None, end: date | None
) -> pd.DataFrame:
    """underlying_data rows: symbol, date and :data:`UNDERLYING_COLUMNS`."""
    where, params = where_clause(
        date_col="collection_date", symbol_col="symbol", symbols=symbols, start=start, end=end
    )
    query = sql.SQL(
        "SELECT symbol, collection_date AS date, {cols} FROM underlying_data {where} "
        "ORDER BY symbol, collection_date"
    ).format(cols=sql.SQL(", ").join(map(sql.Identifier, UNDERLYING_COLUMNS)), where=where)
    return db.read_df(db.Database.OPTIONS, query, params)


def _pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    base = previous.abs()
    return (current - previous) / base.where(base > 0)


def _symbol_bins(df: pd.DataFrame, col: str, bins: int) -> pd.Series:
    """Quantile bin (0 = lowest) of ``col`` within each symbol's own history."""
    return df.groupby("symbol", observed=True)[col].transform(
        lambda x: pd.qcut(x, bins, labels=False, duplicates="drop")
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-over-day changes, volume against its trailing mean, and per-symbol quantile bins."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    by_symbol = df.groupby("symbol", observed=True)
    for col in ("dxoi", "gxoi", "put_call_ratio"):
        df[f"{col}_pct_change"] = _pct_change(df[col], by_symbol[col].shift(1))
    prior_avg = by_symbol["option_volume"].transform(
        lambda v: v.rolling(VOLUME_AVG_SESSIONS, min_periods=VOLUME_AVG_SESSIONS).mean().shift(1)
    )
    df["vol_ratio"] = df["option_volume"] / prior_avg.where(prior_avg > 0)
    df["flownet_decile"] = _symbol_bins(df, "flownet", FLOW_DECILES)
    nonzero_vanna = df[df["vannaxoi"].fillna(0) != 0]
    df["vanna_quintile"] = _symbol_bins(nonzero_vanna, "vannaxoi", VANNA_QUINTILES)
    return df


def _cut(col: str, bins: Bins, bound: float | None = None) -> Labeler:
    edges, labels = bins

    def label(df: pd.DataFrame) -> pd.Series:
        values = df[col]
        if bound is not None:
            values = values.where(values.between(-bound, bound))
        return pd.Series(pd.cut(values, bins=edges, labels=labels), index=df.index).astype("string")

    return label


def _quantile_label(col: str) -> Labeler:
    return lambda df: df[col].astype("Int64").astype("string")


def _volume_flow(df: pd.DataFrame) -> pd.Series:
    spike = np.select(
        [df["vol_ratio"] > VOLUME_SPIKE_MAJOR, df["vol_ratio"] > VOLUME_SPIKE_MINOR],
        ["spike_>2x", "spike_1.5-2x"],
        "normal",
    )
    flow = np.select(
        [df["flownet_decile"] >= HIGH_FLOW_DECILE, df["flownet_decile"] <= LOW_FLOW_DECILE],
        ["high_flow", "low_flow"],
        "mid_flow",
    )
    label = pd.Series(spike, index=df.index) + " x " + pd.Series(flow, index=df.index)
    return label.where(df["vol_ratio"].notna() & df["flownet_decile"].notna()).astype("string")


def _gex_level(df: pd.DataFrame) -> pd.Series:
    return df["gxoi"].map(gex_bucket_label).astype("string")


SIGNALS = (
    SignalSpec(
        "flownet_decile",
        "decile of net customer premium within the symbol's own history",
        _quantile_label("flownet_decile"),
    ),
    SignalSpec(
        "dxoi_change", "day-over-day change in dxoi", _cut("dxoi_pct_change", DXOI_CHANGE_BINS)
    ),
    SignalSpec("gex_level", "gxoi bucket (the dashboard's display buckets)", _gex_level),
    SignalSpec(
        "volume_ratio",
        f"option volume / mean of the prior {VOLUME_AVG_SESSIONS} sessions",
        _cut("vol_ratio", VOL_RATIO_BINS),
    ),
    SignalSpec(
        "pcr_change",
        "day-over-day change in put/call ratio",
        _cut("put_call_ratio_pct_change", PCR_CHANGE_BINS, PCR_CHANGE_BOUND),
    ),
    SignalSpec(
        "vanna_quintile",
        "quintile of nonzero vannaxoi within the symbol's own history",
        _quantile_label("vanna_quintile"),
    ),
    SignalSpec("volume_x_flow", "volume-spike bucket x flownet regime", _volume_flow),
    SignalSpec(
        "gex_change",
        "day-over-day change in gxoi",
        _cut("gxoi_pct_change", GXOI_CHANGE_BINS, GXOI_CHANGE_BOUND),
    ),
)


def bin_table(
    df: pd.DataFrame, labels: pd.Series, horizons: Sequence[int], min_bin_n: int
) -> pd.DataFrame:
    """Per bin: count, mean forward return per horizon, and a Welch test of the bin vs the rest."""
    rows = []
    valid = labels.notna()
    for bin_label in sorted(labels[valid].unique()):
        in_bin = valid & (labels == bin_label)
        row: dict[str, object] = {"bin": bin_label, "n": int(in_bin.sum())}
        for h in horizons:
            col = f"fwd_{h}"
            inside = df.loc[in_bin, col].dropna()
            outside = df.loc[valid & ~in_bin, col].dropna()
            row[f"mean_fwd_{h}_bps"] = inside.mean() * BPS
            row[f"pct_positive_{h}"] = (inside > 0).mean() if len(inside) else np.nan
            testable = min(len(inside), len(outside)) >= min_bin_n
            row[f"p_{h}"] = (
                float(stats.ttest_ind(inside, outside, equal_var=False).pvalue)
                if testable
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run(panel: pd.DataFrame, horizons: Sequence[int], min_bin_n: int) -> pd.DataFrame:
    """All signals' bin tables, stacked, with p-values BH-adjusted over every (bin, horizon)."""
    tables = []
    for spec in SIGNALS:
        table = bin_table(panel, spec.label(panel), horizons, min_bin_n)
        if table.empty:
            logger.warning("signal %s: no rows with a value", spec.name)
            continue
        table.insert(0, "signal", spec.name)
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    out = pd.concat(tables, ignore_index=True)
    p_cols = [f"p_{h}" for h in horizons]
    adjusted = fdr_adjust(out[p_cols].to_numpy(dtype=float).ravel()).reshape(len(out), -1)
    for i, h in enumerate(horizons):
        out[f"p_fdr_{h}"] = adjusted[:, i]
    return out


def build_panel(
    symbols: Sequence[str] | None, start: date | None, end: date | None, horizons: Sequence[int]
) -> pd.DataFrame:
    """Option features joined to the underlying's forward returns on (symbol, date)."""
    options = load_underlying(symbols, start, end)
    if options.empty:
        return options
    logger.info("underlying_data: %d rows, %d symbols", len(options), options["symbol"].nunique())
    # Prices run past --end so the last option dates still get forward returns.
    prices = load_closes(sorted(options["symbol"].unique()), start, None)
    logger.info("prices: %d rows, %d tickers", len(prices), prices["ticker"].nunique())
    fwd = forward_returns(prices, horizons).drop(columns="close")
    panel = add_features(options).merge(
        fwd.rename(columns={"ticker": "symbol"}), on=["symbol", "date"], how="inner"
    )
    logger.info("merged panel: %d rows", len(panel))
    return panel


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser(__doc__.split("\n")[0])
    parser.add_argument(
        "--symbols", type=symbol_list, help="comma-separated (default: every symbol with data)"
    )
    parser.add_argument(
        "--horizons",
        type=int_list,
        default=list(DEFAULT_HORIZONS),
        help="comma-separated forward-return horizons in sessions (default: "
        + ",".join(map(str, DEFAULT_HORIZONS))
        + ")",
    )
    parser.add_argument(
        "--min-bin-n",
        type=int,
        default=DEFAULT_MIN_BIN_N,
        help="rows needed in a bin, and outside it, for its t-test",
    )
    add_common_args(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    panel = build_panel(args.symbols, args.start, args.end, args.horizons)
    results = run(panel, args.horizons, args.min_bin_n) if not panel.empty else pd.DataFrame()
    if results.empty:
        logger.warning("no option rows joined to prices; nothing to report")
    for spec in SIGNALS:
        section = (
            results[results["signal"] == spec.name].drop(columns="signal")
            if not results.empty
            else results
        )
        print_table(f"{spec.name}: {spec.description}", section)
    write_csv(results, args.output_dir / OUTPUT_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
