"""Cross-database signal analysis: option metrics (option_data DB) → price returns (stocks DB).

Joins underlying_data with fact_commons_ohlc to test whether option flow,
positioning, and greek exposure metrics predict forward stock returns.

Usage:
    python signal_analysis.py                         # default 30 liquid names
    python signal_analysis.py --symbols AAPL,TSLA     # specific symbols
    python signal_analysis.py --start-date 2025-10-01 # custom start date
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load credentials from .env (same pattern as main collector)
# Try local .env first, then option_data_collector's .env for DB credentials
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

DEFAULT_SYMBOLS = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "JPM", "BAC",
    "XOM", "CVX", "JNJ", "PFE", "DIS", "NFLX", "CRM", "ORCL", "INTC", "COST",
    "HD", "WMT", "TGT", "LOW", "MCD", "SBUX", "NKE", "ABBV", "LLY", "UNH",
)


def load_data(
    symbols: tuple[str, ...], start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load option metrics and price data from their respective databases."""
    symbol_list = ",".join(f"'{s}'" for s in symbols)

    option_conn = psycopg2.connect(dbname="option_data", **DB_PARAMS)
    stocks_conn = psycopg2.connect(dbname="stocks", **DB_PARAMS)

    option_sql = f"""
    SELECT
        symbol, collection_date, dxoi, gxoi, flownet, vflownet,
        deltas, gammas, option_volume, volatility, vannaxoi, put_call_ratio,
        LAG(dxoi) OVER w as prev_dxoi,
        LAG(gxoi) OVER w as prev_gxoi,
        LAG(flownet) OVER w as prev_flownet,
        LAG(option_volume) OVER w as prev_volume,
        LAG(put_call_ratio) OVER w as prev_pcr,
        AVG(option_volume) OVER (PARTITION BY symbol ORDER BY collection_date
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as avg_20d_volume
    FROM underlying_data
    WHERE symbol IN ({symbol_list})
      AND collection_date >= '{start_date}'
    WINDOW w AS (PARTITION BY symbol ORDER BY collection_date)
    ORDER BY symbol, collection_date
    """

    price_sql = f"""
    SELECT
        ticker as symbol, date as collection_date, close,
        LEAD(close) OVER w as next_close,
        LEAD(close, 5) OVER w as close_5d,
        LEAD(close, 10) OVER w as close_10d
    FROM fact_commons_ohlc
    WHERE date >= '{start_date}'
      AND ticker IN ({symbol_list})
    WINDOW w AS (PARTITION BY ticker ORDER BY date)
    """

    options_df = pd.read_sql(option_sql, option_conn)
    prices_df = pd.read_sql(price_sql, stocks_conn)
    option_conn.close()
    stocks_conn.close()

    return options_df, prices_df


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Derive signal columns from raw option metrics."""
    df = df.copy()

    df["dxoi_pct_change"] = np.where(
        df["prev_dxoi"].abs() > 0,
        (df["dxoi"] - df["prev_dxoi"]) / df["prev_dxoi"].abs(),
        np.nan,
    )
    df["gxoi_pct_change"] = np.where(
        df["prev_gxoi"].abs() > 0,
        (df["gxoi"] - df["prev_gxoi"]) / df["prev_gxoi"].abs(),
        np.nan,
    )
    df["vol_ratio"] = np.where(
        df["avg_20d_volume"] > 0,
        df["option_volume"] / df["avg_20d_volume"],
        np.nan,
    )
    df["pcr_pct_change"] = np.where(
        df["prev_pcr"] > 0,
        (df["put_call_ratio"] - df["prev_pcr"]) / df["prev_pcr"],
        np.nan,
    )
    df["flownet_decile"] = df.groupby("symbol")["flownet"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    return df


def analyze_signal(
    data: pd.DataFrame, signal_col: str, bins: list, labels: list, title: str,
) -> pd.DataFrame:
    """Bin a signal column and compute forward return stats per bin."""
    df = data.copy()
    df["signal_bin"] = pd.cut(df[signal_col], bins=bins, labels=labels)
    result = df.groupby("signal_bin", observed=True).agg(
        n=("fwd_1d", "count"),
        avg_fwd_1d=("fwd_1d", "mean"),
        avg_fwd_5d=("fwd_5d", "mean"),
        avg_fwd_10d=("fwd_10d", "mean"),
        pct_positive_1d=("fwd_1d", lambda x: (x > 0).mean()),
        avg_abs_fwd_1d=("fwd_1d", lambda x: x.abs().mean()),
    ).round(6)
    _print_header(title)
    print(result.to_string())
    return result


def _print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Option signal → price return analysis")
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols (default: 30 liquid names)",
    )
    parser.add_argument(
        "--start-date", type=str, default="2025-07-01",
        help="Start date for analysis (default: 2025-07-01)",
    )
    args = parser.parse_args()

    symbols = tuple(args.symbols.split(",")) if args.symbols else DEFAULT_SYMBOLS

    print("Loading data from both databases...")
    options_df, prices_df = load_data(symbols, args.start_date)
    print(f"  Options: {options_df.shape[0]} rows, {options_df['symbol'].nunique()} symbols")
    print(f"  Prices:  {prices_df.shape[0]} rows, {prices_df['symbol'].nunique()} symbols")

    # Compute forward returns
    prices_df["fwd_1d"] = np.where(
        prices_df["close"] > 0,
        (prices_df["next_close"] - prices_df["close"]) / prices_df["close"],
        np.nan,
    )
    prices_df["fwd_5d"] = np.where(
        prices_df["close"] > 0,
        (prices_df["close_5d"] - prices_df["close"]) / prices_df["close"],
        np.nan,
    )
    prices_df["fwd_10d"] = np.where(
        prices_df["close"] > 0,
        (prices_df["close_10d"] - prices_df["close"]) / prices_df["close"],
        np.nan,
    )

    # Derive option signals
    options_df = compute_signals(options_df)

    # Merge on (symbol, date)
    merged = options_df.merge(
        prices_df[["symbol", "collection_date", "fwd_1d", "fwd_5d", "fwd_10d"]],
        on=["symbol", "collection_date"],
        how="inner",
    )
    print(f"  Merged: {merged.shape[0]} rows")

    # --- Signal A: Flownet decile ---
    _print_header("SIGNAL A: Flownet Decile → Forward Returns")
    flow_result = merged.groupby("flownet_decile", observed=True).agg(
        n=("fwd_1d", "count"),
        avg_flownet=("flownet", "mean"),
        avg_fwd_1d=("fwd_1d", "mean"),
        avg_fwd_5d=("fwd_5d", "mean"),
        avg_fwd_10d=("fwd_10d", "mean"),
        pct_positive_1d=("fwd_1d", lambda x: (x > 0).mean()),
    ).round(6)
    print(flow_result.to_string())

    # --- Signal B: dxoi % change ---
    analyze_signal(
        merged.dropna(subset=["dxoi_pct_change"]),
        "dxoi_pct_change",
        bins=[-np.inf, -0.20, -0.05, 0.05, 0.20, np.inf],
        labels=["big_decrease", "moderate_decrease", "flat", "moderate_increase", "big_increase"],
        title="SIGNAL B: dxoi % Change → Forward Returns",
    )

    # --- Signal C: GEX level ---
    analyze_signal(
        merged.dropna(subset=["gxoi"]),
        "gxoi",
        bins=[-np.inf, 0, 5000, 15000, 50000, np.inf],
        labels=["negative_gex", "low_gex", "mid_gex", "high_gex", "very_high_gex"],
        title="SIGNAL C: GEX Level → Forward Returns",
    )

    # --- Signal D: Volume spike ---
    analyze_signal(
        merged.dropna(subset=["vol_ratio"]),
        "vol_ratio",
        bins=[-np.inf, 0.8, 1.2, 1.5, 2.0, np.inf],
        labels=["low_vol", "normal", "elevated", "spike_1.5-2x", "spike_>2x"],
        title="SIGNAL D: Volume Spike (vs 20d avg) → Forward Returns",
    )

    # --- Signal E: PCR % change ---
    valid = merged.dropna(subset=["pcr_pct_change"])
    valid = valid[valid["pcr_pct_change"].between(-2, 2)]
    analyze_signal(
        valid, "pcr_pct_change",
        bins=[-np.inf, -0.30, -0.10, 0.10, 0.30, np.inf],
        labels=["big_call_surge", "call_increase", "stable", "put_increase", "big_put_surge"],
        title="SIGNAL E: Put/Call Ratio Change → Forward Returns",
    )

    # --- Signal F: Vanna exposure quintile ---
    valid = merged.dropna(subset=["vannaxoi"])
    valid = valid[valid["vannaxoi"] != 0].copy()
    valid["vanna_quintile"] = valid.groupby("symbol")["vannaxoi"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
    )
    _print_header("SIGNAL F: Vanna Exposure Quintile → Forward Returns")
    vanna_result = valid.groupby("vanna_quintile", observed=True).agg(
        n=("fwd_1d", "count"),
        avg_vannaxoi=("vannaxoi", "mean"),
        avg_fwd_1d=("fwd_1d", "mean"),
        avg_fwd_5d=("fwd_5d", "mean"),
        avg_fwd_10d=("fwd_10d", "mean"),
        pct_positive_1d=("fwd_1d", lambda x: (x > 0).mean()),
        avg_abs_fwd_1d=("fwd_1d", lambda x: x.abs().mean()),
    ).round(6)
    print(vanna_result.to_string())

    # --- Signal G: Volume spike × flow interaction ---
    valid = merged.dropna(subset=["flownet_decile", "vol_ratio"]).copy()
    valid["vol_spike"] = np.where(
        valid["vol_ratio"] > 2.0, "spike_>2x",
        np.where(valid["vol_ratio"] > 1.5, "spike_1.5-2x", "normal"),
    )
    valid["flow_regime"] = np.where(
        valid["flownet_decile"] >= 8, "high_flow",
        np.where(valid["flownet_decile"] <= 2, "low_flow", "mid_flow"),
    )
    _print_header("SIGNAL G: Volume Spike × Flow Regime → Forward Returns")
    combo = valid.groupby(["vol_spike", "flow_regime"], observed=True).agg(
        n=("fwd_1d", "count"),
        avg_fwd_1d=("fwd_1d", "mean"),
        avg_fwd_5d=("fwd_5d", "mean"),
        pct_positive_1d=("fwd_1d", lambda x: (x > 0).mean()),
    ).round(6)
    print(combo.to_string())

    # --- Signal H: GEX change ---
    valid = merged.dropna(subset=["gxoi_pct_change"])
    valid = valid[valid["gxoi_pct_change"].between(-5, 5)]
    analyze_signal(
        valid, "gxoi_pct_change",
        bins=[-np.inf, -0.50, -0.10, 0.10, 0.50, np.inf],
        labels=["gex_collapse", "gex_decrease", "gex_stable", "gex_increase", "gex_surge"],
        title="SIGNAL H: GEX Change → Forward Returns",
    )

    print("\n\nDone.")


if __name__ == "__main__":
    main()
