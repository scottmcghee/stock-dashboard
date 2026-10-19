"""Option-contract signals against forward option returns relative to similar contracts.

Builds a per-contract daily panel from ``option_chain_data``, measures each
contract's forward log return over a horizon, subtracts the mean return of
similar contracts on the same date (same symbol, expiry bucket, moneyness
bucket and right, so the same underlying move), and scans candidate signals
(implied volatility dynamics, customer flow, volume, open interest, skew,
underlying-level aggregates) for association with that demeaned return.

Usage:
    python -m research.option_signal_analysis --help
    python -m research.option_signal_analysis --symbols ABC,XYZ --horizon 5
    python -m research.option_signal_analysis --universe-size 10 --sessions 250 --min-oi 200
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2 import sql

from research.common import (
    add_common_args,
    configure_logging,
    group_replication,
    load_closes,
    make_parser,
    pair_scan,
    print_table,
    summarize_replication,
    symbol_list,
    univariate_scan,
    where_clause,
    write_csv,
)
from stock_dashboard import db

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE_SIZE = 5
UNIVERSE_SESSIONS = 20  # recent sessions of option volume that rank the default universe
DEFAULT_SESSIONS = 120
DEFAULT_HORIZON = 5
DEFAULT_MIN_OI = 100
DEFAULT_MIN_PRICE = 0.10
DEFAULT_MIN_DTE = 5
DEFAULT_TOP_K = 6
DEFAULT_MIN_N = 100
DEFAULT_MIN_GROUP_N = 500
DEFAULT_MIN_TICKER_N = 2000
DEFAULT_MIN_PAIR_N = 30
MAX_PAIRS_REPORTED = 20
MAX_SYMBOLS_IN_FILENAME = 5

CHANGE_LAG = 5  # sessions for the "5-day" changes
AVG_WINDOW = 20  # prior sessions in the trailing averages
RESIDUAL_CLIP = (0.01, 0.99)  # winsorize the target at these quantiles
OTM_MONEYNESS = 0.05

DTE_BINS = [-1, 14, 45, 120, 10_000]
DTE_LABELS = ["dte_0_14", "dte_15_45", "dte_46_120", "dte_120plus"]
MNY_BINS = [-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf]
MNY_LABELS = ["mny_lt_-10", "mny_-10_-3", "atm", "mny_3_10", "mny_gt_10"]

TARGETS = ("residual_ret", "fwd_opt_ret")
CONTRACT_KEYS = ["symbol", "expiration_date", "strike", "option_type"]
EXPIRY_KEYS = ["symbol", "expiration_date", "collection_date", "option_type"]

SIGNAL_COLS = [
    # IV dynamics (changes, own-history z-score)
    "iv_change_1d",
    "iv_change_5d",
    "iv_zscore_20d",
    # Customer flow (contract-level)
    "net_flow",
    "net_flow_norm_volume",
    "net_flow_norm_oi",
    # Volume intensity (contract-level)
    "vol_over_oi",
    "vol_vs_avg",
    "log_volume",
    # Vanna dynamics
    "vanna_change_1d",
    "vanna_change_5d",
    # OI dynamics (contract-level)
    "oi_change_1d_pct",
    "oi_change_5d_pct",
    # Cross-strike (same expiration, same right)
    "expiry_skew_slope",
    "skew_slope_chg5",
    "expiry_oi_growth_5d_pct",
    "iv_vs_atm",
    "iv_vs_otm5",
    "oi_share_of_expiry",
    "vol_share_of_expiry",
    # Underlying-level (IV term structure, flow)
    "und_term_slope",
    "und_term_slope_chg5",
    "und_iv_change5",
    "und_pcr",
    "und_pcr_change5",
    "und_flownet",
    "und_flownet_change5",
    "und_vflownet",
    "und_flowratio",
    "und_vflowratio",
    "und_volm_call_ratio",
    "und_volm_put_ratio",
    "und_opt_vol_vs_avg",
]
# Structural features the bucketing should already control for: a sanity check.
STRUCTURAL_COLS = [
    "volatility",
    "gamma_level",
    "vega_level",
    "vanna_level",
    "charm_level",
    "dte",
    "moneyness",
    "moneyness_abs",
    "log_oi",
    "is_call",
]

UNDERLYING_COLUMNS = (
    "volatility",
    "front_volatility",
    "back_volatility",
    "put_call_ratio",
    "option_volume",
    "flownet",
    "vflownet",
    "flowratio",
    "vflowratio",
    "volm_call_ratio",
    "volm_put_ratio",
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def default_universe(size: int) -> list[str]:
    """The ``size`` symbols with the highest median option volume over the latest sessions."""
    query = """
        WITH recent AS (
            SELECT DISTINCT collection_date FROM underlying_data
            ORDER BY collection_date DESC LIMIT %(sessions)s
        )
        SELECT symbol
        FROM underlying_data JOIN recent USING (collection_date)
        GROUP BY symbol
        ORDER BY PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY option_volume) DESC NULLS LAST, symbol
        LIMIT %(size)s
    """
    df = db.read_df(db.Database.OPTIONS, query, {"sessions": UNIVERSE_SESSIONS, "size": size})
    return [str(s) for s in df["symbol"]]


def default_start(sessions: int, end: date | None) -> date | None:
    """The collection date ``sessions`` sessions before the latest chain data (or ``end``)."""
    query = """
        SELECT MIN(collection_date) AS start FROM (
            SELECT DISTINCT collection_date FROM underlying_data
            WHERE collection_date <= COALESCE(
                %(end)s, (SELECT MAX(collection_date) FROM option_chain_data))
            ORDER BY collection_date DESC LIMIT %(sessions)s
        ) recent
    """
    df = db.read_df(db.Database.OPTIONS, query, {"sessions": sessions, "end": end})
    value = df["start"].iloc[0] if not df.empty else None
    return None if pd.isna(value) else pd.Timestamp(value).date()


def load_contract_panel(
    symbols: Sequence[str],
    start: date | None,
    end: date | None,
    *,
    horizon: int,
    min_oi: int,
    min_price: float,
    min_dte: int,
) -> pd.DataFrame:
    """Per-contract daily rows with the price ``horizon`` observations later and lagged values.

    Lags, leads and trailing averages run over each contract's full series;
    the liquidity filters apply afterwards, so they do not create gaps.
    """
    where, params = where_clause(
        date_col="collection_date", symbol_col="symbol", symbols=symbols, start=start, end=None
    )
    params.update(
        horizon=horizon,
        lag=CHANGE_LAG,
        min_oi=min_oi,
        min_price=min_price,
        min_dte=min_dte,
        end=end,
    )
    query = sql.SQL("""
        SELECT * FROM (
            SELECT collection_date, symbol, expiration_date, strike, option_type,
                   price, oi, day_volume, volatility, delta, gamma, vega, vanna, charm,
                   volm_buy, volm_sell,
                   LEAD(price, %(horizon)s) OVER w AS fwd_price,
                   LEAD(collection_date, %(horizon)s) OVER w AS fwd_date,
                   LAG(volatility, 1) OVER w AS iv_lag_1,
                   LAG(volatility, %(lag)s) OVER w AS iv_lag_5,
                   LAG(vanna, 1) OVER w AS vanna_lag_1,
                   LAG(vanna, %(lag)s) OVER w AS vanna_lag_5,
                   LAG(oi, 1) OVER w AS oi_lag_1,
                   LAG(oi, %(lag)s) OVER w AS oi_lag_5,
                   AVG(day_volume) OVER prior AS vol_avg_20d,
                   AVG(volatility) OVER prior AS iv_avg_20d,
                   STDDEV(volatility) OVER prior AS iv_std_20d
            FROM option_chain_data
            {where}
            WINDOW w AS (PARTITION BY symbol, expiration_date, strike, option_type
                         ORDER BY collection_date),
                   prior AS (w ROWS BETWEEN {avg_window} PRECEDING AND 1 PRECEDING)
        ) contracts
        WHERE price > %(min_price)s
          AND oi >= %(min_oi)s
          AND volatility IS NOT NULL
          AND delta IS NOT NULL
          AND fwd_price IS NOT NULL
          AND expiration_date - collection_date >= %(min_dte)s
          AND (%(end)s::date IS NULL OR collection_date <= %(end)s::date)
    """).format(where=where, avg_window=sql.Literal(AVG_WINDOW))
    return db.read_df(db.Database.OPTIONS, query, params)


def load_underlying_features(symbols: Sequence[str], end: date | None) -> pd.DataFrame:
    """Underlying-level IV term structure, put/call ratio, flow and volume features per date."""
    where, params = where_clause(
        date_col="collection_date", symbol_col="symbol", symbols=symbols, start=None, end=end
    )
    query = sql.SQL(
        "SELECT symbol, collection_date, {cols} FROM underlying_data {where} "
        "ORDER BY symbol, collection_date"
    ).format(cols=sql.SQL(", ").join(map(sql.Identifier, UNDERLYING_COLUMNS)), where=where)
    df = db.read_df(db.Database.OPTIONS, query, params)
    by_symbol = df.groupby("symbol", observed=True)

    def lagged(col: str) -> pd.Series:
        return by_symbol[col].shift(CHANGE_LAG)

    out = df[["symbol", "collection_date"]].copy()
    term_slope = df["front_volatility"] - df["back_volatility"]
    out["und_term_slope"] = term_slope
    out["und_term_slope_chg5"] = term_slope - (
        lagged("front_volatility") - lagged("back_volatility")
    )
    out["und_iv_change5"] = df["volatility"] - lagged("volatility")
    out["und_pcr"] = df["put_call_ratio"]
    out["und_pcr_change5"] = df["put_call_ratio"] - lagged("put_call_ratio")
    out["und_flownet"] = df["flownet"]
    out["und_flownet_change5"] = df["flownet"] - lagged("flownet")
    for col in ("vflownet", "flowratio", "vflowratio", "volm_call_ratio", "volm_put_ratio"):
        out[f"und_{col}"] = df[col]
    prior_avg = by_symbol["option_volume"].transform(
        lambda v: v.rolling(AVG_WINDOW, min_periods=AVG_WINDOW).mean().shift(1)
    )
    out["und_opt_vol_vs_avg"] = (df["option_volume"] - prior_avg) / prior_avg.where(prior_avg > 0)
    return out


def load_cross_strike_aggregates(
    symbols: Sequence[str], start: date | None, end: date | None, spot: pd.DataFrame
) -> pd.DataFrame:
    """Per (symbol, expiration, date, right): total OI and volume, OI growth, ATM/OTM IV, skew."""
    where, params = where_clause(
        date_col="collection_date", symbol_col="symbol", symbols=symbols, start=start, end=end
    )
    query = sql.SQL("""
        SELECT symbol, collection_date, expiration_date, strike, option_type,
               COALESCE(oi, 0) AS oi, COALESCE(day_volume, 0) AS day_volume, volatility
        FROM option_chain_data
        {where} AND volatility IS NOT NULL
    """).format(where=where)
    chain = db.read_df(db.Database.OPTIONS, query, params)
    chain = chain.merge(spot, on=["symbol", "collection_date"], how="inner")
    if chain.empty:
        return pd.DataFrame(columns=EXPIRY_KEYS)
    chain["moneyness"] = chain["strike"] / chain["spot"] - 1.0

    agg = (
        chain.groupby(EXPIRY_KEYS, observed=True)
        .agg(expiry_total_oi=("oi", "sum"), expiry_total_volume=("day_volume", "sum"))
        .reset_index()
        .sort_values(EXPIRY_KEYS)
    )
    series = agg.groupby(["symbol", "expiration_date", "option_type"], observed=True)
    oi_lag = series["expiry_total_oi"].shift(CHANGE_LAG)
    agg["expiry_oi_growth_5d_pct"] = (agg["expiry_total_oi"] - oi_lag) / oi_lag.where(oi_lag > 0)

    def nearest_iv(target_moneyness: pd.Series, name: str) -> pd.DataFrame:
        distance = (chain["moneyness"] - target_moneyness).abs()
        nearest = distance.groupby([chain[k] for k in EXPIRY_KEYS], observed=True).idxmin()
        return chain.loc[nearest.to_numpy(), [*EXPIRY_KEYS, "volatility"]].rename(
            columns={"volatility": name}
        )

    atm = pd.Series(0.0, index=chain.index)
    otm = pd.Series(
        np.where(chain["option_type"] == "call", OTM_MONEYNESS, -OTM_MONEYNESS), index=chain.index
    )
    agg = agg.merge(nearest_iv(atm, "atm_iv"), on=EXPIRY_KEYS, how="left")
    agg = agg.merge(nearest_iv(otm, "otm5_iv"), on=EXPIRY_KEYS, how="left")
    agg["expiry_skew_slope"] = agg["otm5_iv"] - agg["atm_iv"]
    agg = agg.sort_values(EXPIRY_KEYS)
    skew_lag = agg.groupby(["symbol", "expiration_date", "option_type"], observed=True)[
        "expiry_skew_slope"
    ].shift(CHANGE_LAG)
    agg["skew_slope_chg5"] = agg["expiry_skew_slope"] - skew_lag
    return agg


def attach_spot(panel: pd.DataFrame, closes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add the underlying's close and its close ``horizon`` sessions later.

    Rows whose option forward price is not from that same later session (the
    contract was missing from some chains) are dropped, so option and
    underlying returns cover the same interval.
    """
    spot = closes.rename(columns={"ticker": "symbol", "date": "collection_date", "close": "spot"})
    spot = spot.sort_values(["symbol", "collection_date"])
    by_symbol = spot.groupby("symbol", observed=True)
    spot["fwd_spot"] = by_symbol["spot"].shift(-horizon)
    spot["fwd_spot_date"] = by_symbol["collection_date"].shift(-horizon)
    df = panel.merge(spot, on=["symbol", "collection_date"], how="inner")
    aligned = df["fwd_date"] == df["fwd_spot_date"]
    if (~aligned).any():
        logger.info("dropping %d rows whose forward option price is misaligned", (~aligned).sum())
    return df[aligned & df["fwd_spot"].notna()].reset_index(drop=True)


# --------------------------------------------------------------------------
# Targets and signals
# --------------------------------------------------------------------------


def demean_within_bucket_date(df: pd.DataFrame) -> pd.Series:
    """``fwd_opt_ret`` minus its mean within each (bucket, date) group.

    The bucket includes the symbol, so every row of a group shares the same
    underlying move; subtracting the group mean removes the part of that move
    the group's contracts have in common.
    """
    keys = [df["bucket"], df["collection_date"]]
    y = df["fwd_opt_ret"]
    return y - y.groupby(keys, observed=True).transform("mean")


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Forward option log return, buckets, and the winsorized demeaned return.

    The bucket (symbol, DTE bucket, moneyness bucket, right) and the date
    absorb time decay, call/put asymmetry and the underlying's move, to the
    extent that the contracts of a bucket respond to them alike.
    """
    df = df.copy()
    df["fwd_opt_ret"] = np.log(df["fwd_price"] / df["price"])
    df["dte"] = (df["expiration_date"] - df["collection_date"]).dt.days
    df["moneyness"] = df["strike"] / df["spot"] - 1.0
    dte_bucket = pd.cut(df["dte"], bins=DTE_BINS, labels=DTE_LABELS).astype(str)
    mny_bucket = pd.cut(df["moneyness"], bins=MNY_BINS, labels=MNY_LABELS).astype(str)
    df["bucket"] = (
        df["symbol"].astype(str) + "|" + dte_bucket + "|" + mny_bucket + "|" + df["option_type"]
    )
    residual = demean_within_bucket_date(df)
    lo, hi = residual.quantile(list(RESIDUAL_CLIP))
    df["residual_ret"] = residual.clip(lo, hi)
    return df


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate predictors, all measured at the row's date (no look-ahead)."""
    df = df.copy()
    oi = df["oi"].where(df["oi"] > 0)
    volume = df["day_volume"].where(df["day_volume"] > 0)

    df["iv_change_1d"] = df["volatility"] - df["iv_lag_1"]
    df["iv_change_5d"] = df["volatility"] - df["iv_lag_5"]
    df["iv_zscore_20d"] = (df["volatility"] - df["iv_avg_20d"]) / df["iv_std_20d"].where(
        df["iv_std_20d"] > 0
    )
    df["net_flow"] = df["volm_buy"].fillna(0) - df["volm_sell"].fillna(0)
    df["net_flow_norm_volume"] = df["net_flow"] / volume
    df["net_flow_norm_oi"] = df["net_flow"] / oi
    df["vol_over_oi"] = df["day_volume"] / oi
    df["vol_vs_avg"] = (df["day_volume"] - df["vol_avg_20d"]) / df["vol_avg_20d"].where(
        df["vol_avg_20d"] > 0
    )
    df["vanna_change_1d"] = df["vanna"] - df["vanna_lag_1"]
    df["vanna_change_5d"] = df["vanna"] - df["vanna_lag_5"]
    for lag in ("1", "5"):
        prev = df[f"oi_lag_{lag}"]
        df[f"oi_change_{lag}d_pct"] = (df["oi"] - prev) / prev.where(prev > 0)

    df["moneyness_abs"] = df["moneyness"].abs()
    df["is_call"] = (df["option_type"] == "call").astype(int)
    df["log_oi"] = np.log1p(df["oi"])
    df["log_volume"] = np.log1p(df["day_volume"].clip(lower=0))
    for greek in ("gamma", "vanna", "charm", "vega"):
        df[f"{greek}_level"] = df[greek]

    df["iv_vs_atm"] = df["volatility"] - df["atm_iv"]
    df["iv_vs_otm5"] = df["volatility"] - df["otm5_iv"]
    df["oi_share_of_expiry"] = df["oi"] / df["expiry_total_oi"].where(df["expiry_total_oi"] > 0)
    df["vol_share_of_expiry"] = df["day_volume"] / df["expiry_total_volume"].where(
        df["expiry_total_volume"] > 0
    )
    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_panel(args: argparse.Namespace, symbols: list[str], start: date | None) -> pd.DataFrame:
    """Contract rows with targets and every signal column."""
    panel = load_contract_panel(
        symbols,
        start,
        args.end,
        horizon=args.horizon,
        min_oi=args.min_oi,
        min_price=args.min_price,
        min_dte=args.min_dte,
    )
    logger.info("contract-days with a forward price: %d", len(panel))
    if panel.empty:
        return panel
    closes = load_closes(symbols, start, None)
    panel = attach_spot(panel, closes, args.horizon)
    spot = closes.rename(columns={"ticker": "symbol", "date": "collection_date", "close": "spot"})
    panel = panel.merge(
        load_underlying_features(symbols, args.end), on=["symbol", "collection_date"], how="left"
    )
    xs = load_cross_strike_aggregates(symbols, start, args.end, spot)
    panel = panel.merge(xs, on=EXPIRY_KEYS, how="left")
    if panel.empty:
        return panel
    panel = build_signals(build_targets(panel))
    return panel.dropna(subset=[args.target]).reset_index(drop=True)


def output_stem(symbols: Sequence[str], horizon: int, target: str) -> str:
    tag = "_".join(symbols) if len(symbols) <= MAX_SYMBOLS_IN_FILENAME else f"{len(symbols)}symbols"
    return f"{tag}_h{horizon}_{target}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser(__doc__.split("\n")[0])
    parser.add_argument(
        "--symbols",
        type=symbol_list,
        help="comma-separated (default: the --universe-size most active option symbols)",
    )
    parser.add_argument(
        "--universe-size",
        type=int,
        default=DEFAULT_UNIVERSE_SIZE,
        help="symbols with the highest median option volume, without --symbols",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=DEFAULT_SESSIONS,
        help="without --start, analyze this many latest sessions",
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON, help="forward return, in sessions"
    )
    parser.add_argument(
        "--min-oi", type=int, default=DEFAULT_MIN_OI, help="minimum open interest of a contract"
    )
    parser.add_argument(
        "--min-price", type=float, default=DEFAULT_MIN_PRICE, help="contract price must exceed this"
    )
    parser.add_argument(
        "--min-dte", type=int, default=DEFAULT_MIN_DTE, help="minimum calendar days to expiration"
    )
    parser.add_argument(
        "--target",
        default=TARGETS[0],
        choices=TARGETS,
        help="demeaned (residual_ret) or raw forward option log return",
    )
    parser.add_argument(
        "--top-k-for-pairs",
        type=int,
        default=DEFAULT_TOP_K,
        help="the strongest K single signals are paired",
    )
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N, help="rows per signal scan")
    parser.add_argument(
        "--min-group-n", type=int, default=DEFAULT_MIN_GROUP_N, help="rows per bucket"
    )
    parser.add_argument(
        "--min-ticker-n", type=int, default=DEFAULT_MIN_TICKER_N, help="rows per symbol"
    )
    parser.add_argument(
        "--min-pair-n", type=int, default=DEFAULT_MIN_PAIR_N, help="rows per signal pair"
    )
    add_common_args(parser)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    out_dir: Path = args.output_dir
    target: str = args.target

    symbols = args.symbols or default_universe(args.universe_size)
    start = args.start or default_start(args.sessions, args.end)
    logger.info("symbols=%s start=%s horizon=%d", ",".join(symbols), start, args.horizon)
    panel = build_panel(args, symbols, start) if symbols else pd.DataFrame()
    stem = output_stem(symbols, args.horizon, target)
    if panel.empty:
        logger.warning("no contract rows with forward prices in the selected range")
        for name in ("structural", "univariate", "per_ticker", "replication", "pairwise"):
            write_csv(pd.DataFrame(), out_dir / f"{name}_{stem}.csv")
        return 0
    logger.info("panel: %d rows, %d buckets", len(panel), panel["bucket"].nunique())
    print_table(
        "Target summary",
        panel[["fwd_opt_ret", "residual_ret"]].describe().T.reset_index(names="target"),
    )

    structural = univariate_scan(panel, target, STRUCTURAL_COLS, args.min_n)
    print_table("Structural features (should be near zero after demeaning)", structural)
    write_csv(structural, out_dir / f"structural_{stem}.csv")

    uni = univariate_scan(panel, target, SIGNAL_COLS, args.min_n)
    print_table(f"Univariate scan (target={target}, m={len(uni)} signals)", uni)
    write_csv(uni, out_dir / f"univariate_{stem}.csv")
    top = uni.head(args.top_k_for_pairs)["signal"].tolist() if not uni.empty else []

    per_ticker = group_replication(panel, target, top, "symbol", args.min_ticker_n)
    print_table("Per-symbol replication of the top signals", summarize_replication(per_ticker, top))
    write_csv(per_ticker, out_dir / f"per_ticker_{stem}.csv")

    per_bucket = group_replication(panel, target, top, "bucket", args.min_group_n)
    print_table("Per-bucket replication of the top signals", summarize_replication(per_bucket, top))
    write_csv(per_bucket, out_dir / f"replication_{stem}.csv")

    pairs = pair_scan(panel, target, top, args.min_pair_n, directions=("top",))
    pairs = pairs.head(MAX_PAIRS_REPORTED)
    print_table("Pairs: both signals in their top decile", pairs)
    write_csv(pairs, out_dir / f"pairwise_{stem}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
