"""Option signal analysis: what predicts big option price rises *beyond* the underlying move?

Workflow:
1. Pull per-contract daily panel (option_chain_data) + underlying close (fact_commons_ohlc).
2. Compute forward option return over N days.
3. Residualize: subtract delta-expected return implied by the underlying's move.
4. Build candidate signal columns (IV, flow, greeks, OI, moneyness, ...).
5. Univariate scan: Spearman ρ vs residual forward return + top-decile lift, BH-FDR corrected.
6. Pairwise scan: combine top-K univariate signals; report conditional lift.

Usage:
    python option_signal_analysis.py --symbols AAPL --start 2025-10-01 --horizon 5
    python option_signal_analysis.py --symbols AAPL,SPY --horizon 10 --min-oi 200
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy import stats

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv(Path(__file__).parent.parent / "option_data_collector" / ".env")

DB_PARAMS = dict(
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", ""),
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_underlying_features(
    symbols: list[str], start_date: str, end_date: str | None,
) -> pd.DataFrame:
    """Underlying-level aggregates: IV term structure, P/C ratio, flow z-scores."""
    symbol_list = ",".join(f"'{s}'" for s in symbols)
    end_filter = f"AND collection_date <= '{end_date}'" if end_date else ""
    sql = f"""
    SELECT
        symbol, collection_date,
        volatility              AS und_iv,
        front_volatility        AS und_front_iv,
        back_volatility         AS und_back_iv,
        put_call_ratio          AS und_pcr,
        option_volume           AS und_total_opt_vol,
        flownet                 AS und_flownet,
        vflownet                AS und_vflownet,
        flowratio               AS und_flowratio,
        vflowratio              AS und_vflowratio,
        volm_call_ratio         AS und_volm_call_ratio,
        volm_put_ratio          AS und_volm_put_ratio,
        LAG(volatility, 5)       OVER w AS und_iv_lag5,
        LAG(front_volatility, 5) OVER w AS und_front_lag5,
        LAG(back_volatility, 5)  OVER w AS und_back_lag5,
        LAG(put_call_ratio, 5)   OVER w AS und_pcr_lag5,
        LAG(flownet, 5)          OVER w AS und_flownet_lag5,
        AVG(option_volume) OVER (PARTITION BY symbol ORDER BY collection_date
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS und_opt_vol_avg20d
    FROM underlying_data
    WHERE symbol IN ({symbol_list})
      AND collection_date >= '{start_date}'
      {end_filter}
    WINDOW w AS (PARTITION BY symbol ORDER BY collection_date)
    """
    with psycopg2.connect(dbname="option_data", **DB_PARAMS) as conn:
        df = pd.read_sql_query(sql, conn)
    df["collection_date"] = pd.to_datetime(df["collection_date"])

    df["und_term_slope"] = df["und_front_iv"] - df["und_back_iv"]
    df["und_term_slope_lag5"] = df["und_front_lag5"] - df["und_back_lag5"]
    df["und_term_slope_chg5"] = df["und_term_slope"] - df["und_term_slope_lag5"]
    df["und_iv_change5"] = df["und_iv"] - df["und_iv_lag5"]
    df["und_pcr_change5"] = df["und_pcr"] - df["und_pcr_lag5"]
    df["und_flownet_change5"] = df["und_flownet"] - df["und_flownet_lag5"]
    df["und_opt_vol_zscore"] = (
        (df["und_total_opt_vol"] - df["und_opt_vol_avg20d"])
        / df["und_opt_vol_avg20d"].replace(0, np.nan)
    )
    keep = [
        "symbol", "collection_date",
        "und_iv", "und_term_slope", "und_term_slope_chg5",
        "und_iv_change5", "und_pcr", "und_pcr_change5",
        "und_flownet", "und_flownet_change5", "und_vflownet",
        "und_flowratio", "und_vflowratio",
        "und_volm_call_ratio", "und_volm_put_ratio",
        "und_opt_vol_zscore",
    ]
    return df[keep]


def load_cross_strike_aggregates(
    symbols: list[str], start_date: str, end_date: str | None,
) -> pd.DataFrame:
    """For each (symbol, expiration, collection_date, option_type), compute:
       - expiry_total_oi, expiry_total_volume
       - expiry_total_oi_lag5 (for neighborhood OI growth)
       - atm_iv (IV of the contract whose strike is closest to spot)
       - otm5_iv (IV of the strike ~5% OTM — above for calls, below for puts)
       Spot is joined in from fact_commons_ohlc.
    """
    symbol_list = ",".join(f"'{s}'" for s in symbols)
    end_filter_opt = f"AND collection_date <= '{end_date}'" if end_date else ""
    end_filter_px = f"AND date <= '{end_date}'" if end_date else ""

    sql_chain = f"""
    SELECT
        symbol, collection_date, expiration_date, strike, option_type,
        COALESCE(oi, 0) AS oi,
        COALESCE(day_volume, 0) AS day_volume,
        volatility
    FROM option_chain_data
    WHERE symbol IN ({symbol_list})
      AND collection_date >= '{start_date}'
      {end_filter_opt}
      AND volatility IS NOT NULL
    """
    sql_px = f"""
    SELECT ticker AS symbol, date AS collection_date, close AS spot
    FROM fact_commons_ohlc
    WHERE ticker IN ({symbol_list})
      AND date >= '{start_date}'
      {end_filter_px}
    """
    with psycopg2.connect(dbname="option_data", **DB_PARAMS) as oconn:
        chain = pd.read_sql_query(sql_chain, oconn)
    with psycopg2.connect(dbname="stocks", **DB_PARAMS) as sconn:
        px = pd.read_sql_query(sql_px, sconn)

    chain["collection_date"] = pd.to_datetime(chain["collection_date"])
    px["collection_date"] = pd.to_datetime(px["collection_date"])
    chain = chain.merge(px, on=["symbol", "collection_date"], how="inner")
    chain["moneyness"] = chain["strike"] / chain["spot"] - 1.0

    # --- Per-(symbol, expiry, date, right) aggregates ---
    group_keys = ["symbol", "expiration_date", "collection_date", "option_type"]
    agg = chain.groupby(group_keys).agg(
        expiry_total_oi=("oi", "sum"),
        expiry_total_volume=("day_volume", "sum"),
    ).reset_index()

    # 5-day lag of expiry_total_oi (for neighborhood OI growth)
    agg = agg.sort_values(group_keys)
    agg["expiry_total_oi_lag5"] = agg.groupby(
        ["symbol", "expiration_date", "option_type"]
    )["expiry_total_oi"].shift(5)
    agg["expiry_oi_growth_5d_pct"] = (
        (agg["expiry_total_oi"] - agg["expiry_total_oi_lag5"])
        / agg["expiry_total_oi_lag5"].replace(0, np.nan)
    )

    # --- ATM IV and OTM-5% IV ---
    def _nearest_row_iv(group: pd.DataFrame, target_moneyness: float) -> float:
        d = (group["moneyness"] - target_moneyness).abs()
        if d.empty:
            return np.nan
        idx = d.idxmin()
        return group.loc[idx, "volatility"]

    # ATM = strike closest to spot (moneyness 0)
    # OTM-5%: for calls, strike ~5% above spot (moneyness +0.05)
    #         for puts,  strike ~5% below spot (moneyness -0.05)
    atm_iv_rows = []
    otm_iv_rows = []
    for keys, group in chain.groupby(group_keys):
        atm_iv_rows.append((*keys, _nearest_row_iv(group, 0.0)))
        target = 0.05 if keys[3] == "call" else -0.05
        otm_iv_rows.append((*keys, _nearest_row_iv(group, target)))

    atm_iv_df = pd.DataFrame(atm_iv_rows, columns=[*group_keys, "atm_iv"])
    otm_iv_df = pd.DataFrame(otm_iv_rows, columns=[*group_keys, "otm5_iv"])

    agg = agg.merge(atm_iv_df, on=group_keys, how="left")
    agg = agg.merge(otm_iv_df, on=group_keys, how="left")
    agg["expiry_skew_slope"] = agg["otm5_iv"] - agg["atm_iv"]

    # 5d change in skew slope
    agg["skew_slope_lag5"] = agg.groupby(
        ["symbol", "expiration_date", "option_type"]
    )["expiry_skew_slope"].shift(5)
    agg["skew_slope_chg5"] = agg["expiry_skew_slope"] - agg["skew_slope_lag5"]

    keep = [
        *group_keys,
        "expiry_total_oi", "expiry_total_volume", "expiry_oi_growth_5d_pct",
        "atm_iv", "otm5_iv", "expiry_skew_slope", "skew_slope_chg5",
    ]
    return agg[keep]


def load_contract_panel(
    symbols: list[str],
    start_date: str,
    end_date: str | None,
    horizon: int,
    min_oi: int,
    min_price: float,
    min_dte: int,
) -> pd.DataFrame:
    """Load per-contract daily panel with forward price, underlying close, and
    lagged values for change-based signals.
    """
    symbol_list = ",".join(f"'{s}'" for s in symbols)
    end_filter = f"AND collection_date <= '{end_date}'" if end_date else ""

    sql_opts = f"""
    SELECT
        collection_date, symbol, expiration_date, strike, option_type,
        price, oi, day_volume, volatility,
        delta, gamma, theta, vega, vanna, charm, vomma,
        volm_buy, volm_sell,
        gxoi, dxoi, vxoi, vannaxoi, charmxoi,
        LEAD(price, {horizon}) OVER w AS fwd_price,
        LAG(volatility, 1)     OVER w AS iv_lag_1,
        LAG(volatility, 5)     OVER w AS iv_lag_5,
        LAG(vanna, 1)          OVER w AS vanna_lag_1,
        LAG(vanna, 5)          OVER w AS vanna_lag_5,
        LAG(oi, 1)             OVER w AS oi_lag_1,
        LAG(oi, 5)             OVER w AS oi_lag_5,
        AVG(day_volume) OVER (PARTITION BY symbol, expiration_date, strike, option_type
            ORDER BY collection_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS vol_avg_20d,
        AVG(volatility) OVER (PARTITION BY symbol, expiration_date, strike, option_type
            ORDER BY collection_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS iv_avg_20d,
        STDDEV(volatility) OVER (PARTITION BY symbol, expiration_date, strike, option_type
            ORDER BY collection_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS iv_std_20d
    FROM option_chain_data
    WHERE symbol IN ({symbol_list})
      AND collection_date >= '{start_date}'
      {end_filter}
      AND price IS NOT NULL AND price > {min_price}
      AND oi >= {min_oi}
      AND volatility IS NOT NULL
      AND (expiration_date - collection_date) >= {min_dte}
    WINDOW w AS (
        PARTITION BY symbol, expiration_date, strike, option_type
        ORDER BY collection_date
    )
    """

    end_filter_px = f"AND date <= '{end_date}'" if end_date else ""
    sql_px = f"""
    SELECT ticker AS symbol, date AS collection_date, close AS spot
    FROM fact_commons_ohlc
    WHERE ticker IN ({symbol_list}) AND date >= '{start_date}' {end_filter_px}
    UNION ALL
    SELECT ticker AS symbol, date AS collection_date, close AS spot
    FROM fact_index_ohlc
    WHERE ticker IN ({symbol_list}) AND date >= '{start_date}' {end_filter_px}
    """

    with psycopg2.connect(dbname="option_data", **DB_PARAMS) as oconn:
        df_opt = pd.read_sql_query(sql_opts, oconn)
    with psycopg2.connect(dbname="stocks", **DB_PARAMS) as sconn:
        df_px = pd.read_sql_query(sql_px, sconn)

    df_opt["collection_date"] = pd.to_datetime(df_opt["collection_date"])
    df_px["collection_date"] = pd.to_datetime(df_px["collection_date"])

    # Merge current spot
    df = df_opt.merge(df_px, on=["symbol", "collection_date"], how="inner")

    # Forward spot: T+horizon close per symbol
    df_px = df_px.sort_values(["symbol", "collection_date"])
    df_px["fwd_spot"] = df_px.groupby("symbol")["spot"].shift(-horizon)
    df = df.merge(
        df_px[["symbol", "collection_date", "fwd_spot"]],
        on=["symbol", "collection_date"], how="left",
    )

    df = df.dropna(subset=["fwd_price", "fwd_spot", "spot", "delta"]).copy()
    return df


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

DTE_BINS = [-1, 14, 45, 120, 10_000]
DTE_LABELS = ["dte_0_14", "dte_15_45", "dte_46_120", "dte_120plus"]
MNY_BINS = [-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf]
MNY_LABELS = ["oom_deep", "oom_near", "atm", "itm_near", "itm_deep"]


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw forward option + underlying log-returns, assign buckets,
    then residualize option returns within (bucket, date) via OLS on
    underlying return.

    "Bucket" = (DTE bucket, moneyness bucket, right). This controls for
    structural factors (time decay, convexity, call/put asymmetry) and
    same-day market regime without needing greek-sign assumptions.
    """
    df = df.copy()
    df["fwd_opt_ret"] = np.log(df["fwd_price"] / df["price"])
    df["fwd_und_ret"] = np.log(df["fwd_spot"] / df["spot"])

    dte = (pd.to_datetime(df["expiration_date"]) - df["collection_date"]).dt.days
    df["dte_bucket"] = pd.cut(dte, bins=DTE_BINS, labels=DTE_LABELS)
    moneyness = df["strike"] / df["spot"] - 1.0
    df["mny_bucket"] = pd.cut(moneyness, bins=MNY_BINS, labels=MNY_LABELS)
    df["bucket"] = (
        df["symbol"].astype(str) + "|" +
        df["dte_bucket"].astype(str) + "|" +
        df["mny_bucket"].astype(str) + "|" +
        df["option_type"].astype(str)
    )

    df["residual_ret"] = _residualize_within_bucket_date(df)

    lo, hi = df["residual_ret"].quantile([0.01, 0.99])
    df["residual_ret"] = df["residual_ret"].clip(lo, hi)
    return df


def _residualize_within_bucket_date(df: pd.DataFrame) -> pd.Series:
    """For each (bucket, date) group, regress fwd_opt_ret on fwd_und_ret and
    return the residuals. Groups with <5 rows or degenerate x-variance fall
    back to simple demeaning.
    """
    x = df["fwd_und_ret"].to_numpy()
    y = df["fwd_opt_ret"].to_numpy()
    resid = np.empty(len(df))
    resid.fill(np.nan)
    group_ids = df.groupby(["bucket", "collection_date"]).ngroup().to_numpy()
    n_groups = group_ids.max() + 1
    for g in range(n_groups):
        idx = np.where(group_ids == g)[0]
        if len(idx) < 5:
            yi = y[idx]
            resid[idx] = yi - yi.mean()
            continue
        xi, yi = x[idx], y[idx]
        xm = xi - xi.mean()
        ym = yi - yi.mean()
        xx = (xm * xm).sum()
        if xx < 1e-10:
            resid[idx] = yi - yi.mean()
        else:
            slope = (xm * ym).sum() / xx
            intercept = yi.mean() - slope * xi.mean()
            resid[idx] = yi - (intercept + slope * xi)
    return pd.Series(resid, index=df.index)


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candidate predictor columns. All measured strictly at time T."""
    df = df.copy()
    px = df["price"].replace(0, np.nan)
    oi = df["oi"].replace(0, np.nan)

    # IV dynamics
    df["iv_change_1d"] = df["volatility"] - df["iv_lag_1"]
    df["iv_change_5d"] = df["volatility"] - df["iv_lag_5"]
    df["iv_zscore_20d"] = (df["volatility"] - df["iv_avg_20d"]) / df["iv_std_20d"].replace(0, np.nan)

    # Flow (customer perspective: volm_buy - volm_sell)
    df["net_flow"] = df["volm_buy"].fillna(0) - df["volm_sell"].fillna(0)
    df["net_flow_norm_volume"] = df["net_flow"] / df["day_volume"].replace(0, np.nan)
    df["net_flow_norm_oi"] = df["net_flow"] / oi
    df["vol_over_oi"] = df["day_volume"] / oi
    df["vol_zscore"] = (df["day_volume"] - df["vol_avg_20d"]) / df["vol_avg_20d"].replace(0, np.nan)

    # Greek dynamics
    df["vanna_change_1d"] = df["vanna"] - df["vanna_lag_1"]
    df["vanna_change_5d"] = df["vanna"] - df["vanna_lag_5"]

    # OI dynamics
    df["oi_change_1d_pct"] = (df["oi"] - df["oi_lag_1"]) / df["oi_lag_1"].replace(0, np.nan)
    df["oi_change_5d_pct"] = (df["oi"] - df["oi_lag_5"]) / df["oi_lag_5"].replace(0, np.nan)

    # Structural
    df["dte"] = (pd.to_datetime(df["expiration_date"]) - df["collection_date"]).dt.days
    df["moneyness"] = df["strike"] / df["spot"] - 1.0
    df["moneyness_abs"] = df["moneyness"].abs()
    df["is_call"] = (df["option_type"] == "call").astype(int)
    df["log_oi"] = np.log1p(df["oi"])
    df["log_volume"] = np.log1p(df["day_volume"].clip(lower=0))

    # Greek levels
    df["gamma_level"] = df["gamma"]
    df["vanna_level"] = df["vanna"]
    df["charm_level"] = df["charm"]
    df["vega_level"] = df["vega"]

    # Cross-strike derived (require cross-strike aggregates joined in):
    if "atm_iv" in df.columns:
        df["iv_vs_atm"] = df["volatility"] - df["atm_iv"]
    if "otm5_iv" in df.columns:
        df["iv_vs_otm5"] = df["volatility"] - df["otm5_iv"]
    if "expiry_total_oi" in df.columns:
        df["oi_share_of_expiry"] = df["oi"] / df["expiry_total_oi"].replace(0, np.nan)
    if "expiry_total_volume" in df.columns:
        df["vol_share_of_expiry"] = df["day_volume"] / df["expiry_total_volume"].replace(0, np.nan)

    return df


SIGNAL_COLS = [
    # IV dynamics (level, changes, own-history z-score)
    "iv_change_1d", "iv_change_5d", "iv_zscore_20d",
    # Customer flow (contract-level)
    "net_flow", "net_flow_norm_volume", "net_flow_norm_oi",
    # Volume intensity (contract-level)
    "vol_over_oi", "vol_zscore", "log_volume",
    # Vanna/charm dynamics
    "vanna_change_1d", "vanna_change_5d",
    # OI dynamics (contract-level)
    "oi_change_1d_pct", "oi_change_5d_pct",
    # Cross-strike (same expiration, same right)
    "expiry_skew_slope", "skew_slope_chg5", "expiry_oi_growth_5d_pct",
    "iv_vs_atm", "iv_vs_otm5",
    "oi_share_of_expiry", "vol_share_of_expiry",
    # Underlying-level (IV term structure + flow)
    "und_term_slope", "und_term_slope_chg5",
    "und_iv_change5",
    "und_pcr", "und_pcr_change5",
    "und_flownet", "und_flownet_change5", "und_vflownet",
    "und_flowratio", "und_vflowratio",
    "und_volm_call_ratio", "und_volm_put_ratio",
    "und_opt_vol_zscore",
]
# Structural signals (bucket-stratified out, kept as sanity check):
STRUCTURAL_COLS = [
    "volatility", "gamma_level", "vega_level", "vanna_level", "charm_level",
    "dte", "moneyness", "moneyness_abs", "log_oi", "is_call",
]


# --------------------------------------------------------------------------
# Scans
# --------------------------------------------------------------------------

def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def univariate_scan(df: pd.DataFrame, target: str, signal_cols: list[str]) -> pd.DataFrame:
    """Spearman ρ + decile lift per signal. Returns a sorted DataFrame."""
    rows = []
    y = df[target]
    for col in signal_cols:
        x = df[col]
        mask = x.notna() & y.notna()
        if mask.sum() < 100:
            continue
        xv, yv = x[mask], y[mask]
        if xv.nunique() < 4:
            continue
        rho, p = stats.spearmanr(xv, yv)

        # Decile lift: top decile mean vs bottom decile mean of target
        try:
            deciles = pd.qcut(xv, 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        top = yv[deciles == deciles.max()].mean()
        bot = yv[deciles == deciles.min()].mean()
        overall = yv.mean()

        # Hit rate in top decile: fraction above 90th percentile of y
        y_hit_thresh = yv.quantile(0.90)
        top_hit = (yv[deciles == deciles.max()] >= y_hit_thresh).mean()
        overall_hit = (yv >= y_hit_thresh).mean()

        rows.append({
            "signal": col,
            "n": int(mask.sum()),
            "spearman_rho": rho,
            "p_value": p,
            "top_dec_mean": top,
            "bot_dec_mean": bot,
            "lift_bps": (top - bot) * 10_000,
            "overall_mean": overall,
            "top_dec_hit_rate": top_hit,
            "overall_hit_rate": overall_hit,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = bh_fdr(out["p_value"].to_numpy())
    out = out.sort_values("p_value").reset_index(drop=True)
    return out


def per_bucket_replication(
    df: pd.DataFrame, target: str, signals: list[str], min_n: int = 500
) -> pd.DataFrame:
    """For each signal, compute Spearman ρ within each bucket. A signal that
    replicates cross-bucket (consistent sign, similar magnitude) is more
    likely real than bucket-specific noise.
    """
    rows = []
    for bucket, g in df.groupby("bucket"):
        if len(g) < min_n:
            continue
        for col in signals:
            x = g[col]
            y = g[target]
            mask = x.notna() & y.notna()
            if mask.sum() < min_n:
                continue
            if x[mask].nunique() < 4:
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            rows.append({
                "bucket": bucket, "signal": col,
                "n": int(mask.sum()), "rho": rho, "p": p,
            })
    return pd.DataFrame(rows)


def per_ticker_replication(
    df: pd.DataFrame, target: str, signals: list[str], min_n: int = 2000
) -> pd.DataFrame:
    """Per-symbol Spearman ρ for each signal. Answers: does the signal
    replicate across tickers?
    """
    rows = []
    for symbol, g in df.groupby("symbol"):
        if len(g) < min_n:
            continue
        for col in signals:
            x = g[col]
            y = g[target]
            mask = x.notna() & y.notna()
            if mask.sum() < min_n or x[mask].nunique() < 4:
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            rows.append({
                "symbol": symbol, "signal": col,
                "n": int(mask.sum()), "rho": rho, "p": p,
            })
    return pd.DataFrame(rows)


def summarize_per_ticker(rep: pd.DataFrame, top_signals: list[str]) -> pd.DataFrame:
    """Aggregate per-ticker replication: how consistent is each signal
    across symbols?
    """
    rows = []
    for sig in top_signals:
        sub = rep[rep["signal"] == sig]
        if sub.empty:
            continue
        mean_rho = sub["rho"].mean()
        n_symbols = len(sub)
        same_sign = ((sub["rho"] > 0) == (mean_rho > 0)).sum() / n_symbols
        sig_same = ((sub["p"] < 0.05) & ((sub["rho"] > 0) == (mean_rho > 0))).sum() / n_symbols
        rows.append({
            "signal": sig,
            "n_symbols": n_symbols,
            "mean_rho": mean_rho,
            "pct_same_sign": same_sign,
            "pct_sig_same_sign": sig_same,
            "min_rho": sub["rho"].min(),
            "max_rho": sub["rho"].max(),
            "rho_by_symbol": dict(zip(sub["symbol"], sub["rho"].round(3))),
        })
    return pd.DataFrame(rows).sort_values("pct_sig_same_sign", ascending=False)


def summarize_replication(rep: pd.DataFrame, top_signals: list[str]) -> pd.DataFrame:
    """For each top signal, summarize its ρ across buckets: mean ρ, fraction of
    buckets where ρ is same-signed and significant, # buckets tested.
    """
    rows = []
    for sig in top_signals:
        sub = rep[rep["signal"] == sig]
        if sub.empty:
            continue
        mean_rho = sub["rho"].mean()
        n_buckets = len(sub)
        # Consistency: fraction same-signed as mean
        consistent = ((sub["rho"] > 0) == (mean_rho > 0)).sum() / n_buckets
        # Significance: fraction with p<0.05 AND same sign
        sig_mask = (sub["p"] < 0.05) & ((sub["rho"] > 0) == (mean_rho > 0))
        frac_sig = sig_mask.sum() / n_buckets
        rows.append({
            "signal": sig,
            "n_buckets": n_buckets,
            "mean_rho": mean_rho,
            "pct_same_sign": consistent,
            "pct_sig_same_sign": frac_sig,
            "min_rho": sub["rho"].min(),
            "max_rho": sub["rho"].max(),
        })
    return pd.DataFrame(rows).sort_values("pct_sig_same_sign", ascending=False)


def pairwise_scan(
    df: pd.DataFrame, target: str, top_signals: list[str], n_pairs: int = 20
) -> pd.DataFrame:
    """Evaluate pairwise AND conditions on top-decile of each signal pair."""
    rows = []
    y = df[target]
    y_hit_thresh = y.quantile(0.90)
    overall_mean = y.mean()
    overall_hit = (y >= y_hit_thresh).mean()

    for i in range(len(top_signals)):
        for j in range(i + 1, len(top_signals)):
            a, b = top_signals[i], top_signals[j]
            xa, xb = df[a], df[b]
            mask = xa.notna() & xb.notna() & y.notna()
            if mask.sum() < 200:
                continue
            try:
                da = pd.qcut(xa[mask], 10, labels=False, duplicates="drop")
                db = pd.qcut(xb[mask], 10, labels=False, duplicates="drop")
            except ValueError:
                continue
            # Both in top decile
            cond = (da == da.max()) & (db == db.max())
            if cond.sum() < 30:
                continue
            yv = y[mask]
            cond_mean = yv[cond].mean()
            cond_hit = (yv[cond] >= y_hit_thresh).mean()
            # Permutation z-score for mean vs overall (approximate)
            n = cond.sum()
            se = yv.std() / np.sqrt(n)
            z = (cond_mean - overall_mean) / se if se > 0 else np.nan
            p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else 1.0
            rows.append({
                "signal_a": a, "signal_b": b,
                "n": int(n),
                "cond_mean": cond_mean,
                "lift_vs_overall_bps": (cond_mean - overall_mean) * 10_000,
                "cond_hit_rate": cond_hit,
                "overall_hit_rate": overall_hit,
                "z": z, "p_value": p,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = bh_fdr(out["p_value"].to_numpy())
    out = out.sort_values("p_value").reset_index(drop=True).head(n_pairs)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--symbols", default="AAPL", help="comma-separated tickers")
    p.add_argument("--start", default="2025-10-01")
    p.add_argument("--end", default=None)
    p.add_argument("--horizon", type=int, default=5, help="forward return days")
    p.add_argument("--min-oi", type=int, default=100)
    p.add_argument("--min-price", type=float, default=0.10)
    p.add_argument("--min-dte", type=int, default=5)
    p.add_argument("--target", default="residual_ret",
                   choices=["residual_ret", "fwd_opt_ret"],
                   help="target: residualized (default) or raw option return")
    p.add_argument("--output-dir", default="./signal_results")
    p.add_argument("--top-k-for-pairs", type=int, default=6)
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading panel: symbols={symbols} start={args.start} horizon={args.horizon}d")
    df = load_contract_panel(
        symbols, args.start, args.end, args.horizon,
        args.min_oi, args.min_price, args.min_dte,
    )
    print(f"  loaded {len(df):,} contract-days")

    print("  loading underlying features...")
    und = load_underlying_features(symbols, args.start, args.end)
    df = df.merge(und, on=["symbol", "collection_date"], how="left")
    print(f"  after underlying merge: {len(df):,} rows, {df.columns.size} columns")

    print("  loading cross-strike aggregates...")
    xs = load_cross_strike_aggregates(symbols, args.start, args.end)
    df = df.merge(
        xs, on=["symbol", "expiration_date", "collection_date", "option_type"],
        how="left",
    )
    print(f"  after cross-strike merge: {len(df):,} rows, {df.columns.size} columns")

    df = build_targets(df)
    df = build_signals(df)
    df = df.dropna(subset=[args.target]).copy()
    print(f"  after target/signal build: {len(df):,} rows")
    print(f"  forward option return stats: mean={df['fwd_opt_ret'].mean():.4f}, "
          f"std={df['fwd_opt_ret'].std():.4f}")
    print(f"  residual return stats:       mean={df['residual_ret'].mean():.4f}, "
          f"std={df['residual_ret'].std():.4f}")
    print()

    print(f"  bucket distribution (top 10):")
    print(df["bucket"].value_counts().head(10).to_string())
    print()

    # --- Univariate: structural sanity check first ---
    print("=== Structural signals (sanity check — should be ~0 if residualization worked) ===")
    sc = univariate_scan(df, args.target, STRUCTURAL_COLS)
    display = sc[["signal", "n", "spearman_rho", "p_value", "lift_bps", "top_dec_hit_rate"]].copy()
    for col in ("spearman_rho", "p_value"):
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    display["lift_bps"] = display["lift_bps"].map(lambda v: f"{v:+.0f}")
    display["top_dec_hit_rate"] = display["top_dec_hit_rate"].map(lambda v: f"{v:.3f}")
    print(display.to_string(index=False))
    print()

    # --- Univariate ---
    uni = univariate_scan(df, args.target, SIGNAL_COLS)
    print(f"=== Univariate scan of behavioral signals (target={args.target}) ===")
    display = uni[[
        "signal", "n", "spearman_rho", "p_value", "p_fdr",
        "lift_bps", "top_dec_hit_rate", "overall_hit_rate",
    ]].copy()
    for col in ("spearman_rho", "p_value", "p_fdr"):
        display[col] = display[col].map(lambda v: f"{v:.4f}")
    display["lift_bps"] = display["lift_bps"].map(lambda v: f"{v:+.0f}")
    display["top_dec_hit_rate"] = display["top_dec_hit_rate"].map(lambda v: f"{v:.3f}")
    display["overall_hit_rate"] = display["overall_hit_rate"].map(lambda v: f"{v:.3f}")
    print(display.to_string(index=False))

    stem = "_".join(symbols) + f"_h{args.horizon}_{args.target}"
    uni.to_csv(out_dir / f"univariate_{stem}.csv", index=False)

    # --- Per-ticker replication (if multiple symbols) ---
    top_behavioral = uni.head(args.top_k_for_pairs)["signal"].tolist()
    if df["symbol"].nunique() > 1:
        print("\n=== Per-ticker replication of top behavioral signals ===")
        per_sym = per_ticker_replication(df, args.target, top_behavioral)
        if not per_sym.empty:
            summary = summarize_per_ticker(per_sym, top_behavioral)
            d = summary.drop(columns=["rho_by_symbol"]).copy()
            d["mean_rho"] = d["mean_rho"].map(lambda v: f"{v:+.4f}")
            d["pct_same_sign"] = d["pct_same_sign"].map(lambda v: f"{v:.2f}")
            d["pct_sig_same_sign"] = d["pct_sig_same_sign"].map(lambda v: f"{v:.2f}")
            d["min_rho"] = d["min_rho"].map(lambda v: f"{v:+.3f}")
            d["max_rho"] = d["max_rho"].map(lambda v: f"{v:+.3f}")
            print(d.to_string(index=False))
            print("\nSpearman ρ by ticker (top signals):")
            wide = per_sym.pivot(index="signal", columns="symbol", values="rho")
            # Reindex to match ordered top_behavioral
            wide = wide.loc[[s for s in top_behavioral if s in wide.index]]
            print(wide.round(3).to_string())
            per_sym.to_csv(out_dir / f"per_ticker_{stem}.csv", index=False)

    # --- Per-bucket replication on top behavioral signals ---
    print("\n=== Per-bucket replication of top behavioral signals ===")
    rep = per_bucket_replication(df, args.target, top_behavioral)
    if not rep.empty:
        summary = summarize_replication(rep, top_behavioral)
        d = summary.copy()
        d["mean_rho"] = d["mean_rho"].map(lambda v: f"{v:+.4f}")
        d["pct_same_sign"] = d["pct_same_sign"].map(lambda v: f"{v:.2f}")
        d["pct_sig_same_sign"] = d["pct_sig_same_sign"].map(lambda v: f"{v:.2f}")
        d["min_rho"] = d["min_rho"].map(lambda v: f"{v:+.3f}")
        d["max_rho"] = d["max_rho"].map(lambda v: f"{v:+.3f}")
        print(d.to_string(index=False))
        rep.to_csv(out_dir / f"replication_{stem}.csv", index=False)
    else:
        print("  (no buckets had enough rows)")

    # --- Pairwise on top-K univariate signals ---
    top_signals = uni.head(args.top_k_for_pairs)["signal"].tolist()
    if len(top_signals) >= 2:
        print(f"\n=== Pairwise scan (top {len(top_signals)} univariate signals) ===")
        pairs = pairwise_scan(df, args.target, top_signals)
        if not pairs.empty:
            d = pairs.copy()
            d["cond_mean"] = d["cond_mean"].map(lambda v: f"{v:+.4f}")
            d["lift_vs_overall_bps"] = d["lift_vs_overall_bps"].map(lambda v: f"{v:+.0f}")
            d["cond_hit_rate"] = d["cond_hit_rate"].map(lambda v: f"{v:.3f}")
            d["overall_hit_rate"] = d["overall_hit_rate"].map(lambda v: f"{v:.3f}")
            d["z"] = d["z"].map(lambda v: f"{v:+.2f}")
            d["p_value"] = d["p_value"].map(lambda v: f"{v:.4f}")
            d["p_fdr"] = d["p_fdr"].map(lambda v: f"{v:.4f}")
            print(d.to_string(index=False))
            pairs.to_csv(out_dir / f"pairwise_{stem}.csv", index=False)
        else:
            print("  (no pairs met the minimum-sample threshold)")

    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()
