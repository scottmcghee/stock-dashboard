"""TA signal discovery on the underlying stock universe.

Scans the full indicator library (reused from backtester) for predictive
power vs forward returns, with per-sector / per-regime / per-time-half
replication and BH-FDR correction. Top validated pairs auto-export as
preset JSONs compatible with the stock screener + backtester.

Reuses (do not duplicate):
    queries.backtest_queries.load_ohlcv_universe / load_metrics_universe
                                / load_spy_ohlcv   (cached parquet)
    logic.backtest.build_backtest_dataframe        (indicators + fwd returns)
    logic.backtest.classify_regimes                (bull/bear/volatile/correction)
    logic.backtest.evaluate_all_strategies         (validation for preset stats)

Usage:
    python ta_signal_analysis.py                                   # defaults
    python ta_signal_analysis.py --universe-size 2000 --start 2021-01-01
    python ta_signal_analysis.py --no-export-presets               # just report
"""

import argparse
import gc
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy import stats

from logic.backtest import (
    FilterSpec,
    Strategy,
    build_backtest_dataframe,
    classify_regimes,
    evaluate_all_strategies,
)
from logic.preset import PRESET_DIR
from queries.backtest_queries import (
    load_metrics_universe,
    load_ohlcv_universe,
    load_spy_ohlcv,
)

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv(Path(__file__).parent.parent / "option_data_collector" / ".env")


# ---------------------------------------------------------------------------
# Universe + sector enrichment
# ---------------------------------------------------------------------------

def select_liquid_universe(ohlcv: pd.DataFrame, top_n: int, lookback_days: int = 90) -> list[str]:
    """Top-N tickers by median daily dollar volume over the trailing window."""
    last_date = ohlcv["date"].max()
    window_start = last_date - pd.Timedelta(days=lookback_days * 2)  # buffer for weekends/holidays
    window = ohlcv[ohlcv["date"] >= window_start].copy()
    window["dollar_volume"] = window["close"] * window["volume"]
    median_dv = (
        window.groupby("ticker")["dollar_volume"].median().sort_values(ascending=False)
    )
    return median_dv.head(top_n).index.tolist()


def load_sector_map(tickers: list[str]) -> dict[str, str]:
    """Fetch sector per ticker from dim_stock."""
    db_params = dict(
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )
    symbol_list = ",".join(f"'{s}'" for s in tickers)
    sql = f"""
    SELECT ticker, sector
    FROM dim_stock
    WHERE ticker IN ({symbol_list})
    """
    with psycopg2.connect(dbname="stocks", **db_params) as conn:
        df = pd.read_sql_query(sql, conn)
    return dict(zip(df["ticker"], df["sector"].fillna("Unknown")))


# ---------------------------------------------------------------------------
# Residualization vs SPY (β-adjusted per-ticker, rolling)
# ---------------------------------------------------------------------------

def compute_spy_fwd_returns(spy: pd.DataFrame) -> pd.DataFrame:
    """Forward SPY returns at 5/10/20 horizons — for residualization targets."""
    s = spy.sort_values("date").copy()
    out = s[["date"]].copy()
    for h in (5, 10, 20):
        out[f"spy_fwd_{h}"] = (s["close"].shift(-h).values - s["close"].values) / s["close"].values
    return out


def compute_rolling_beta(ohlcv: pd.DataFrame, spy: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Rolling β per (ticker, date) against SPY over a 63-day window of daily log-returns."""
    spy_px = spy.sort_values("date")[["date", "close"]].rename(columns={"close": "spy_close"})
    merged = ohlcv.merge(spy_px, on="date", how="left").sort_values(["ticker", "date"])
    merged["ret"] = merged.groupby("ticker")["close"].transform(
        lambda c: np.log(c / c.shift(1))
    )
    merged["spy_ret"] = np.log(merged["spy_close"] / merged.groupby("ticker")["spy_close"].shift(1))

    def _beta(g: pd.DataFrame) -> pd.Series:
        cov = g["ret"].rolling(window, min_periods=30).cov(g["spy_ret"])
        var = g["spy_ret"].rolling(window, min_periods=30).var()
        return cov / var.replace(0, np.nan)

    merged["beta"] = merged.groupby("ticker", group_keys=False).apply(_beta)
    out = merged[["date", "ticker", "beta"]].copy()
    out["beta"] = out["beta"].astype(np.float32)
    return out


def residualize_forward_returns(df: pd.DataFrame, spy_fwd: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    """residual_fwd_H = fwd_H - β * spy_fwd_H, per (ticker, date).

    If β is NaN (insufficient history), falls back to fwd_H - spy_fwd_H (β=1).
    """
    df = df.merge(spy_fwd, on="date", how="left")
    df = df.merge(beta, on=["date", "ticker"], how="left")
    b = df["beta"].fillna(1.0).astype(np.float32)
    for h in (5, 10, 20):
        fwd_col = f"fwd_{h}"
        spy_col = f"spy_fwd_{h}"
        if fwd_col in df.columns and spy_col in df.columns:
            df[f"resid_fwd_{h}"] = (df[fwd_col] - b * df[spy_col]).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Additional scalar features not in precompute_indicators
# ---------------------------------------------------------------------------

def add_extra_features(ohlcv: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    """Add 52w distance, BB %B, ATR/price — computed per ticker then merged."""
    rows = []
    for ticker, g in ohlcv.groupby("ticker"):
        g = g.sort_values("date").copy()
        close = g["close"]
        high = g["high"]
        low = g["low"]

        high_252 = close.rolling(252, min_periods=40).max()
        low_252 = close.rolling(252, min_periods=40).min()
        dist_52w_high = (close / high_252 - 1.0)  # negative when below high
        dist_52w_low = (close / low_252 - 1.0)    # positive when above low

        sma20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        bb_pctb = (close - lower) / (upper - lower).replace(0, np.nan)
        bb_width = (upper - lower) / sma20.replace(0, np.nan) * 100

        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=14).mean()
        atr_over_price = (atr_14 / close.replace(0, np.nan)) * 100

        rows.append(pd.DataFrame({
            "date": g["date"].values,
            "ticker": ticker,
            "dist_52w_high": dist_52w_high.values,
            "dist_52w_low": dist_52w_low.values,
            "bb_pctb": bb_pctb.values,
            "bb_width": bb_width.values,
            "atr_over_price": atr_over_price.values,
        }))
    if not rows:
        return indicators
    extra = pd.concat(rows, ignore_index=True)
    float_cols = [c for c in extra.columns if c not in ("date", "ticker")]
    extra[float_cols] = extra[float_cols].astype(np.float32)
    return indicators.merge(extra, on=["date", "ticker"], how="left")


# Signal columns to scan (all scalar — "events" are Phase 2)
SIGNAL_COLS = [
    # Overbought/oversold (scalar)
    "rsi_14", "bb_pctb", "dist_52w_high", "dist_52w_low",
    # Momentum
    "momentum_20", "momentum_63", "momentum_126",
    "macd_histogram",
    "rs_impulse_5", "rs_impulse_10", "rs_impulse_20", "rs_impulse_42",
    # Trend / position
    "ratio_to_50ma", "ratio_to_200ma", "weinstein_stage_num",
    "rs_rank_overall", "rs_z_score",
    # Volatility / range
    "adr_14", "bb_width", "atr_over_price",
    # Volume (supply/demand)
    "volume_z",
]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def univariate_scan(df: pd.DataFrame, target: str, signal_cols: list[str]) -> pd.DataFrame:
    """Spearman ρ + decile lift (+ top/bottom hit rates) per signal."""
    rows = []
    y = df[target]
    y_hit = y.quantile(0.90)
    for col in signal_cols:
        if col not in df.columns:
            continue
        x = df[col]
        mask = x.notna() & y.notna()
        if mask.sum() < 500:
            continue
        xv, yv = x[mask], y[mask]
        if xv.nunique() < 4:
            continue
        rho, p = stats.spearmanr(xv, yv)
        try:
            deciles = pd.qcut(xv, 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        top = yv[deciles == deciles.max()].mean()
        bot = yv[deciles == deciles.min()].mean()
        top_hit = (yv[deciles == deciles.max()] >= y_hit).mean()
        rows.append({
            "signal": col,
            "n": int(mask.sum()),
            "spearman_rho": rho,
            "p_value": p,
            "lift_bps": (top - bot) * 10_000,
            "top_dec_hit_rate": top_hit,
            "overall_hit_rate": (yv >= y_hit).mean(),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = bh_fdr(out["p_value"].to_numpy())
    return out.sort_values("p_value").reset_index(drop=True)


def per_group_replication(
    df: pd.DataFrame, target: str, signals: list[str], group_col: str, min_n: int = 500
) -> pd.DataFrame:
    """Spearman ρ per (group, signal). Used for per-sector and per-regime reports."""
    rows = []
    for gkey, g in df.groupby(group_col):
        if len(g) < min_n:
            continue
        for col in signals:
            if col not in g.columns:
                continue
            x = g[col]
            y = g[target]
            mask = x.notna() & y.notna()
            if mask.sum() < min_n or x[mask].nunique() < 4:
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            rows.append({group_col: gkey, "signal": col, "n": int(mask.sum()),
                         "rho": rho, "p": p})
    return pd.DataFrame(rows)


def summarize_replication(rep: pd.DataFrame, group_col: str, top_signals: list[str]) -> pd.DataFrame:
    rows = []
    for sig in top_signals:
        sub = rep[rep["signal"] == sig]
        if sub.empty:
            continue
        mean_rho = sub["rho"].mean()
        n_groups = len(sub)
        same_sign = ((sub["rho"] > 0) == (mean_rho > 0)).sum() / n_groups
        sig_same = ((sub["p"] < 0.05) & ((sub["rho"] > 0) == (mean_rho > 0))).sum() / n_groups
        rows.append({
            "signal": sig, "n_groups": n_groups,
            "mean_rho": mean_rho,
            "pct_same_sign": same_sign,
            "pct_sig_same_sign": sig_same,
            "min_rho": sub["rho"].min(),
            "max_rho": sub["rho"].max(),
            "by_group": dict(zip(sub[group_col], sub["rho"].round(3))),
        })
    return pd.DataFrame(rows).sort_values("pct_sig_same_sign", ascending=False)


# ---------------------------------------------------------------------------
# Map scan signals → backtester filter specs (for preset export)
# ---------------------------------------------------------------------------

# Map from signal_col → (filter_name_template, decile_direction)
# When the signal's TOP decile is what we want, use filter_name with "_min";
# when the BOTTOM decile is what we want, use "_max".
# Only includes signals the backtester already has filters for.
SIGNAL_TO_FILTER = {
    "rsi_14":           ("rsi_min", "rsi_max"),
    "momentum_20":      ("momentum_20_min", "momentum_20_max"),
    "momentum_63":      ("momentum_63_min", "momentum_63_max"),
    "momentum_126":     ("momentum_126_min", "momentum_126_max"),
    "macd_histogram":   ("macd_histogram_min", "macd_histogram_max"),
    "ratio_to_50ma":    ("ratio_to_50ma_min", "ratio_to_50ma_max"),
    "ratio_to_200ma":   ("ratio_to_200ma_min", "ratio_to_200ma_max"),
    "volume_z":         ("volume_z_min", "volume_z_max"),
    "adr_14":           ("adr_14_min", "adr_14_max"),
    "rs_rank_overall":  ("rs_rank_overall_min", "rs_rank_overall_max"),
    "rs_z_score":       ("rs_z_score_min", "rs_z_score_max"),
    "rs_impulse_10":    ("rs_impulse_10_min", "rs_impulse_10_max"),
    "rs_impulse_20":    ("rs_impulse_20_min", "rs_impulse_20_max"),
}


def build_filter_spec(df: pd.DataFrame, signal: str, direction: str) -> FilterSpec | None:
    """Convert a (signal, direction) into a backtester FilterSpec using the
    signal's decile cutoff on the full panel. direction: 'top' or 'bottom'.
    """
    if signal not in SIGNAL_TO_FILTER or signal not in df.columns:
        return None
    fmin, fmax = SIGNAL_TO_FILTER[signal]
    if direction == "top":
        cutoff = df[signal].quantile(0.90)
        return FilterSpec(name=fmin, value=float(cutoff))
    elif direction == "bottom":
        cutoff = df[signal].quantile(0.10)
        return FilterSpec(name=fmax, value=float(cutoff))
    return None


# ---------------------------------------------------------------------------
# Pairwise combo scan + preset export
# ---------------------------------------------------------------------------

def discover_pair_combos(
    df: pd.DataFrame,
    target: str,
    top_signals: list[str],
    per_regime: pd.DataFrame,
    per_time: pd.DataFrame,
    min_n: int = 100,
    min_top_dec_hit: float = 0.55,
    min_regimes_replicate: int = 1,
    min_time_halves_replicate: int = 2,
) -> pd.DataFrame:
    """For each ordered pair of top signals × each (top/bottom) direction pair,
    evaluate the "both in their target decile" conditional forward return.
    Keep pairs passing replication thresholds.
    """
    y = df[target]
    y_hit = y.quantile(0.90)
    overall_mean = y.mean()

    candidates = []
    for a, b in combinations(top_signals, 2):
        if a not in df.columns or b not in df.columns:
            continue
        if a not in SIGNAL_TO_FILTER or b not in SIGNAL_TO_FILTER:
            continue
        try:
            da_top = df[a] >= df[a].quantile(0.90)
            da_bot = df[a] <= df[a].quantile(0.10)
            db_top = df[b] >= df[b].quantile(0.90)
            db_bot = df[b] <= df[b].quantile(0.10)
        except Exception:
            continue
        for dir_a, mask_a in [("top", da_top), ("bottom", da_bot)]:
            for dir_b, mask_b in [("top", db_top), ("bottom", db_bot)]:
                mask = mask_a & mask_b & y.notna()
                n = int(mask.sum())
                if n < min_n:
                    continue
                yv = y[mask]
                cond_mean = yv.mean()
                cond_hit = (yv >= y_hit).mean()
                if cond_hit < min_top_dec_hit:
                    continue
                se = yv.std() / np.sqrt(n)
                z = (cond_mean - overall_mean) / se if se > 0 else np.nan
                p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else 1.0
                candidates.append({
                    "signal_a": a, "dir_a": dir_a,
                    "signal_b": b, "dir_b": dir_b,
                    "n": n, "cond_mean": cond_mean,
                    "cond_hit_rate": cond_hit,
                    "lift_vs_overall_bps": (cond_mean - overall_mean) * 10_000,
                    "z": z, "p_value": p,
                })

    if not candidates:
        return pd.DataFrame()

    out = pd.DataFrame(candidates)
    out["p_fdr"] = bh_fdr(out["p_value"].to_numpy())
    out = out[out["p_fdr"] < 0.05].copy()
    if out.empty:
        return out

    # Replication filters: for each pair signal, require same-sign significant
    # replication in enough regimes and time halves.
    def _replicates(sig: str, rep: pd.DataFrame, group_col: str, need: int) -> bool:
        sub = rep[rep["signal"] == sig]
        if sub.empty:
            return False
        mean_rho = sub["rho"].mean()
        same_sig = ((sub["p"] < 0.05) & ((sub["rho"] > 0) == (mean_rho > 0))).sum()
        return same_sig >= need

    def _both_replicate(row) -> bool:
        return (
            _replicates(row["signal_a"], per_regime, "regime", min_regimes_replicate)
            and _replicates(row["signal_b"], per_regime, "regime", min_regimes_replicate)
            and _replicates(row["signal_a"], per_time, "time_half", min_time_halves_replicate)
            and _replicates(row["signal_b"], per_time, "time_half", min_time_halves_replicate)
        )

    out = out[out.apply(_both_replicate, axis=1)].reset_index(drop=True)
    return out.sort_values("p_fdr").head(50)


def export_preset(
    df: pd.DataFrame,
    row: pd.Series,
    indicators_df: pd.DataFrame,
    regime_series: pd.Series,
) -> Path | None:
    """Build a Strategy from a winning pair, evaluate on the full panel for
    BH-corrected backtester stats, write as preset JSON.
    """
    fa = build_filter_spec(df, row["signal_a"], row["dir_a"])
    fb = build_filter_spec(df, row["signal_b"], row["dir_b"])
    if fa is None or fb is None:
        return None
    strategy_name = f"TA-disc_{row['signal_a']}_{row['dir_a']}__{row['signal_b']}_{row['dir_b']}"
    strategy = Strategy(name=strategy_name, filters=[fa, fb], regime=None)

    results = evaluate_all_strategies(indicators_df, [strategy], regime_series)
    if not results:
        return None
    res = results[0]

    preset = {
        "name": strategy_name,
        "filters": [{"name": f.name, "value": f.value} for f in strategy.filters],
        "regime": strategy.regime,
        "stats": {
            label: {
                "mean_return": round(hs.mean_return * 100, 2),
                "win_rate": round(hs.win_rate, 3),
                "sharpe": round(hs.sharpe, 3),
                "p_value_adj": round(hs.p_value_adj, 4),
            }
            for label, hs in res.horizons.items()
        },
        "discovery": {
            "source": "ta_signal_analysis",
            "n_discovery": int(row["n"]),
            "cond_hit_rate_disc": round(float(row["cond_hit_rate"]), 3),
            "lift_vs_overall_bps_disc": round(float(row["lift_vs_overall_bps"]), 1),
            "p_fdr_disc": round(float(row["p_fdr"]), 4),
        },
    }
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    filename = strategy_name.replace(" ", "_").replace("/", "_")[:80] + ".json"
    path = PRESET_DIR / filename
    path.write_text(json.dumps(preset, indent=2))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--universe-size", type=int, default=2000)
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=None, help="defaults to latest available")
    p.add_argument("--target", default="resid_fwd_10",
                   choices=["resid_fwd_5", "resid_fwd_10", "resid_fwd_20",
                            "fwd_5", "fwd_10", "fwd_20"],
                   help="target column for the primary univariate scan")
    p.add_argument("--output-dir", default="./signal_results")
    p.add_argument("--top-k-for-pairs", type=int, default=8)
    p.add_argument("--export-presets", action="store_true", default=True)
    p.add_argument("--no-export-presets", dest="export_presets", action="store_false")
    p.add_argument("--min-hit-rate", type=float, default=0.25,
                   help="conditional top-decile hit rate threshold for pair export (baseline ~0.10)")
    p.add_argument("--min-regimes-replicate", type=int, default=1)
    p.add_argument("--min-time-halves-replicate", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading OHLCV universe + SPY (parquet cache)...")
    ohlcv = load_ohlcv_universe()
    metrics = load_metrics_universe()
    spy = load_spy_ohlcv()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    metrics["date"] = pd.to_datetime(metrics["date"])
    spy["date"] = pd.to_datetime(spy["date"])

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) if args.end else ohlcv["date"].max()
    ohlcv = ohlcv[(ohlcv["date"] >= start) & (ohlcv["date"] <= end)]
    metrics = metrics[(metrics["date"] >= start) & (metrics["date"] <= end)]
    spy = spy[(spy["date"] >= start) & (spy["date"] <= end)]
    print(f"  raw universe: {ohlcv['ticker'].nunique():,} tickers, {len(ohlcv):,} rows")

    liquid = select_liquid_universe(ohlcv, args.universe_size)
    ohlcv = ohlcv[ohlcv["ticker"].isin(liquid)].reset_index(drop=True)
    metrics = metrics[metrics["ticker"].isin(liquid)].reset_index(drop=True)
    print(f"  filtered to top {len(liquid):,} by 90d median $-volume: {len(ohlcv):,} rows")

    print("Computing indicators + forward returns (reused from backtester)...")
    panel = build_backtest_dataframe(ohlcv, metrics)
    del metrics
    gc.collect()
    print(f"  panel: {len(panel):,} rows × {panel.columns.size} cols")

    print("Residualizing forward returns vs SPY (rolling β)...")
    spy_fwd = compute_spy_fwd_returns(spy)
    beta = compute_rolling_beta(ohlcv, spy, window=63)
    panel = residualize_forward_returns(panel, spy_fwd, beta)
    del spy_fwd, beta
    gc.collect()

    print("Adding 52w distance, BB %B, ATR/price, BB width...")
    panel = add_extra_features(ohlcv, panel)
    del ohlcv
    gc.collect()

    # Regime + sector + time-half
    print("Classifying regimes, attaching sectors, splitting time halves...")
    regimes = classify_regimes(spy)
    panel["regime"] = panel["date"].map(regimes)
    sectors = load_sector_map(liquid)
    panel["sector"] = panel["ticker"].map(sectors).fillna("Unknown")
    midpoint = panel["date"].quantile(0.5)
    panel["time_half"] = np.where(panel["date"] < midpoint, "first_half", "second_half")

    panel = panel.dropna(subset=[args.target]).reset_index(drop=True)
    print(f"  final: {len(panel):,} rows after target dropna")
    print()

    # --- Univariate scan ---
    uni = univariate_scan(panel, args.target, SIGNAL_COLS)
    print(f"=== Univariate scan (target={args.target}) ===")
    d = uni[["signal", "n", "spearman_rho", "p_value", "p_fdr",
             "lift_bps", "top_dec_hit_rate", "overall_hit_rate"]].copy()
    for col in ("spearman_rho", "p_value", "p_fdr"):
        d[col] = d[col].map(lambda v: f"{v:.4f}")
    d["lift_bps"] = d["lift_bps"].map(lambda v: f"{v:+.0f}")
    d["top_dec_hit_rate"] = d["top_dec_hit_rate"].map(lambda v: f"{v:.3f}")
    d["overall_hit_rate"] = d["overall_hit_rate"].map(lambda v: f"{v:.3f}")
    print(d.to_string(index=False))
    stem = f"universe{args.universe_size}_{args.target}"
    uni.to_csv(out_dir / f"ta_univariate_{stem}.csv", index=False)

    top_signals = uni.head(args.top_k_for_pairs)["signal"].tolist()

    # --- Per-regime replication ---
    print("\n=== Per-regime replication ===")
    per_regime = per_group_replication(panel, args.target, top_signals, "regime")
    reg_summary = summarize_replication(per_regime, "regime", top_signals)
    if not reg_summary.empty:
        print(reg_summary.drop(columns=["by_group"]).to_string(index=False))

    # --- Per-sector replication ---
    print("\n=== Per-sector replication ===")
    per_sector = per_group_replication(panel, args.target, top_signals, "sector")
    sec_summary = summarize_replication(per_sector, "sector", top_signals)
    if not sec_summary.empty:
        print(sec_summary.drop(columns=["by_group"]).to_string(index=False))

    # --- Time-stability ---
    print("\n=== Time-half replication (first vs second) ===")
    per_time = per_group_replication(panel, args.target, top_signals, "time_half", min_n=200)
    time_summary = summarize_replication(per_time, "time_half", top_signals)
    if not time_summary.empty:
        print(time_summary.drop(columns=["by_group"]).to_string(index=False))

    per_regime.to_csv(out_dir / f"ta_per_regime_{stem}.csv", index=False)
    per_sector.to_csv(out_dir / f"ta_per_sector_{stem}.csv", index=False)
    per_time.to_csv(out_dir / f"ta_per_time_{stem}.csv", index=False)

    # --- Pairwise discovery + preset export ---
    print(f"\n=== Pairwise combo discovery (top-{args.top_k_for_pairs} signals) ===")
    pairs = discover_pair_combos(
        panel, args.target, top_signals,
        per_regime=per_regime, per_time=per_time,
        min_top_dec_hit=args.min_hit_rate,
        min_regimes_replicate=args.min_regimes_replicate,
        min_time_halves_replicate=args.min_time_halves_replicate,
    )
    if pairs.empty:
        print("  (no pairs passed replication + hit-rate thresholds)")
    else:
        print(f"  {len(pairs)} pairs passed all filters:")
        d = pairs.copy()
        d["cond_mean"] = d["cond_mean"].map(lambda v: f"{v:+.4f}")
        d["cond_hit_rate"] = d["cond_hit_rate"].map(lambda v: f"{v:.3f}")
        d["lift_vs_overall_bps"] = d["lift_vs_overall_bps"].map(lambda v: f"{v:+.0f}")
        d["p_fdr"] = d["p_fdr"].map(lambda v: f"{v:.4f}")
        print(d[["signal_a", "dir_a", "signal_b", "dir_b", "n",
                 "cond_mean", "cond_hit_rate", "lift_vs_overall_bps",
                 "p_fdr"]].to_string(index=False))
        pairs.to_csv(out_dir / f"ta_pairs_{stem}.csv", index=False)

        if args.export_presets:
            print("\n=== Exporting validated presets (runs backtester for stats) ===")
            exported = 0
            for _, row in pairs.iterrows():
                path = export_preset(panel, row, panel, regimes)
                if path:
                    print(f"  wrote {path.name}")
                    exported += 1
            print(f"\n  {exported} presets written to {PRESET_DIR}")

    print(f"\nCSVs written to {out_dir}/")


if __name__ == "__main__":
    main()
