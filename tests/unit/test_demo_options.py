"""Tests for the synthetic option-data generator (stock_dashboard.demo)."""

import re
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.demo.bsm import bsm_greeks, bsm_price
from stock_dashboard.demo.contract import (
    TABLES,
    UNDERLYING_VALUE_COLUMNS,
)
from stock_dashboard.demo.load import option_data_ddl
from stock_dashboard.demo.options import (
    AVG_WINDOW,
    OptionDemoConfig,
    OptionFrames,
    PlantedEvents,
    compute_underlying_avg,
    generate_option_data,
    monthly_expirations,
    nice_strike_step,
    weekly_expirations,
)

PLANTED = PlantedEvents()
N_SYMBOLS = 50
SESSIONS = 260
HISTORY = 400


def make_closes(n_symbols: int = N_SYMBOLS, n_days: int = HISTORY, seed: int = 1) -> pd.DataFrame:
    """Random-walk closes for the planted symbols plus filler symbols."""
    rng = np.random.default_rng(seed)
    planted = [
        PLANTED.vanna_high,
        PLANTED.vanna_low,
        PLANTED.flownet_low,
        PLANTED.volume_spike,
        PLANTED.dxoi_jump,
        *PLANTED.negative_gex,
    ]
    symbols = planted + [f"S{i:03d}" for i in range(n_symbols - len(planted))]
    dates = pd.bdate_range(end="2026-10-16", periods=n_days)
    frames = []
    for symbol in symbols:
        vol = rng.uniform(0.012, 0.03)
        start = rng.uniform(15, 2000)
        close = start * np.exp(np.cumsum(rng.normal(0.0003, vol, n_days)))
        frames.append(pd.DataFrame({"date": dates, "ticker": symbol, "close": close}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def closes() -> pd.DataFrame:
    return make_closes()


@pytest.fixture(scope="module")
def frames(closes: pd.DataFrame) -> OptionFrames:
    return generate_option_data(closes, OptionDemoConfig(sessions=SESSIONS))


@pytest.fixture(scope="module")
def underlying(frames: OptionFrames) -> pd.DataFrame:
    return frames.underlying_data


@pytest.fixture(scope="module")
def latest(underlying: pd.DataFrame) -> pd.Timestamp:
    return underlying["collection_date"].max()


# ---------------------------------------------------------------------------
# Pandas replicas of the dashboard's SQL (queries/signal_queries.py,
# queries/screener_queries.py, queries/option_queries.py)
# ---------------------------------------------------------------------------


def _since(df: pd.DataFrame, latest: pd.Timestamp, days: int) -> pd.DataFrame:
    return df[df["collection_date"] >= latest - pd.Timedelta(days=days)]


def _percent_rank(df: pd.DataFrame, column: str, latest: pd.Timestamp) -> pd.Series:
    """PERCENT_RANK() OVER (PARTITION BY symbol ORDER BY column) over 90 days, on latest."""
    hist = _since(df.dropna(subset=[column]), latest, 90).copy()
    grouped = hist.groupby("symbol")[column]
    rank = grouped.rank(method="min")
    count = grouped.transform("count")
    hist["pct_rank"] = ((rank - 1) / (count - 1)).fillna(0.0)
    return hist[hist["collection_date"] == latest].set_index("symbol")["pct_rank"]


def scan_vanna_extreme(
    df: pd.DataFrame, latest: pd.Timestamp, threshold: float = 20.0
) -> pd.Series:
    pct = _percent_rank(df, "vannaxoi", latest)
    return pct[(pct >= 1 - threshold / 100) | (pct <= threshold / 100)]


def scan_flownet_extreme(
    df: pd.DataFrame, latest: pd.Timestamp, threshold: float = 10.0
) -> pd.Series:
    pct = _percent_rank(df, "flownet", latest)
    return pct[pct <= threshold / 100]


def scan_volume_spike(df: pd.DataFrame, latest: pd.Timestamp, multiple: float = 2.0) -> pd.Series:
    hist = _since(df, latest, 30).sort_values(["symbol", "collection_date"]).copy()
    hist["avg"] = hist.groupby("symbol")["option_volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean()
    )
    today = hist[(hist["collection_date"] == latest) & (hist["avg"] > 0)].set_index("symbol")
    ratio = today["option_volume"] / today["avg"]
    return ratio[ratio >= multiple]


def scan_dxoi_change(df: pd.DataFrame, latest: pd.Timestamp, pct: float = 20.0) -> pd.Series:
    hist = _since(df, latest, 5).sort_values(["symbol", "collection_date"]).copy()
    hist["prev"] = hist.groupby("symbol")["dxoi"].shift(1)
    today = hist[(hist["collection_date"] == latest) & hist["prev"].notna()].set_index("symbol")
    change = (today["dxoi"] - today["prev"]) / today["prev"].abs() * 100
    return change[change.abs() >= pct]


def gex_buckets(df: pd.DataFrame, latest: pd.Timestamp) -> pd.Series:
    gxoi = df[df["collection_date"] == latest].set_index("symbol")["gxoi"]
    labels = ["Negative GEX", "Low GEX", "Mid GEX", "High GEX", "Very High GEX"]
    return pd.cut(gxoi, [-np.inf, 0, 5000, 15000, 50000, np.inf], right=False, labels=labels)


def screener_multiples(frames: OptionFrames, latest: pd.Timestamp, column: str) -> pd.Series:
    today = frames.underlying_data[frames.underlying_data["collection_date"] == latest]
    avg = frames.underlying_data_avg[frames.underlying_data_avg["date"] == latest]
    merged = today.merge(avg, on="symbol")
    merged = merged[merged[f"{column}_avg"].notna() & (merged[f"{column}_avg"] != 0)]
    return (merged[column] / merged[f"{column}_avg"]).set_axis(merged["symbol"])


def weekly_symbols(chain: pd.DataFrame) -> set[str]:
    today = chain[chain["collection_date"] == chain["collection_date"].max()]
    in_30 = today[today["expiration_date"] <= today["collection_date"] + pd.Timedelta(days=30)]
    counts = in_30.groupby("symbol")["expiration_date"].nunique()
    return set(counts[counts >= 3].index)


# ---------------------------------------------------------------------------
# Black-Scholes-Merton
# ---------------------------------------------------------------------------


def _arr(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def test_bsm_textbook_values() -> None:
    # Hull, Options, Futures and Other Derivatives, example 15.6: c = 4.76, p = 0.81.
    spot, strike, t, sigma = _arr(42.0), _arr(40.0), _arr(0.5), _arr(0.2)
    call = bsm_price(spot, strike, t, sigma, np.array([True]), r=0.10, q=0.0)
    put = bsm_price(spot, strike, t, sigma, np.array([False]), r=0.10, q=0.0)
    assert call[0] == pytest.approx(4.76, abs=0.005)
    assert put[0] == pytest.approx(0.81, abs=0.005)


def test_put_call_parity_and_bounds() -> None:
    rng = np.random.default_rng(0)
    n = 500
    spot = rng.uniform(10, 500, n)
    strike = spot * rng.uniform(0.7, 1.3, n)
    t = rng.uniform(1 / 365, 1.0, n)
    sigma = rng.uniform(0.1, 1.0, n)
    r, q = 0.04, 0.01
    call = bsm_greeks(spot, strike, t, sigma, np.ones(n, dtype=bool), r, q)
    put = bsm_greeks(spot, strike, t, sigma, np.zeros(n, dtype=bool), r, q)
    parity = spot * np.exp(-q * t) - strike * np.exp(-r * t)
    np.testing.assert_allclose(call.price - put.price, parity, atol=1e-9)
    assert np.all((call.delta >= 0) & (call.delta <= 1))
    assert np.all((put.delta >= -1) & (put.delta <= 0))
    np.testing.assert_allclose(call.delta - put.delta, np.exp(-q * t), atol=1e-12)
    assert np.all(call.gamma > 0)
    np.testing.assert_allclose(call.gamma, put.gamma)
    assert np.all(call.vega > 0)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(is_call: bool) -> None:
    spot, strike, t, sigma = _arr(100.0), _arr(105.0), _arr(0.25), _arr(0.3)
    flag = np.array([is_call])
    r, q = 0.04, 0.01
    g = bsm_greeks(spot, strike, t, sigma, flag, r, q)

    def price(s: float = 100.0, tt: float = 0.25, vol: float = 0.3) -> float:
        return float(bsm_price(_arr(s), strike, _arr(tt), _arr(vol), flag, r, q)[0])

    def delta(s: float = 100.0, tt: float = 0.25, vol: float = 0.3) -> float:
        return float(bsm_greeks(_arr(s), strike, _arr(tt), _arr(vol), flag, r, q).delta[0])

    def vega(vol: float) -> float:
        return float(bsm_greeks(spot, strike, t, _arr(vol), flag, r, q).vega[0])

    hs, ht, hv = 0.01, 1e-5, 1e-5
    assert g.delta[0] == pytest.approx((price(100 + hs) - price(100 - hs)) / (2 * hs), rel=1e-5)
    assert g.gamma[0] == pytest.approx((delta(100 + hs) - delta(100 - hs)) / (2 * hs), rel=1e-4)
    # theta and charm: per calendar day, i.e. the change as T shrinks by 1/365.
    theta = -(price(tt=0.25 + ht) - price(tt=0.25 - ht)) / (2 * ht) / 365
    assert g.theta[0] == pytest.approx(theta, rel=1e-4)
    charm = -(delta(tt=0.25 + ht) - delta(tt=0.25 - ht)) / (2 * ht) / 365
    assert g.charm[0] == pytest.approx(charm, rel=1e-4)
    # vega, vanna, vomma: per volatility point (0.01).
    assert g.vega[0] == pytest.approx(
        (price(vol=0.3 + hv) - price(vol=0.3 - hv)) / (2 * hv) / 100, rel=1e-5
    )
    vanna = (delta(vol=0.3 + hv) - delta(vol=0.3 - hv)) / (2 * hv) / 100
    assert g.vanna[0] == pytest.approx(vanna, rel=1e-4)
    vomma = (vega(0.3 + hv) - vega(0.3 - hv)) / (2 * hv) / 100
    assert g.vomma[0] == pytest.approx(vomma, rel=1e-4)


# ---------------------------------------------------------------------------
# Calendar and strike helpers
# ---------------------------------------------------------------------------


def test_expiration_calendars() -> None:
    friday = date(2026, 10, 16)  # the third Friday of October 2026
    assert weekly_expirations(friday, 2) == (date(2026, 10, 23), date(2026, 10, 30))
    assert monthly_expirations(friday, 2) == (date(2026, 11, 20), date(2026, 12, 18))
    assert monthly_expirations(date(2026, 10, 15), 1) == (friday,)
    assert weekly_expirations(date(2026, 10, 14), 1) == (friday,)


def test_nice_strike_step() -> None:
    steps = nice_strike_step(np.array([15.0, 100.0, 180.0, 3600.0]), 0.025)
    np.testing.assert_array_equal(steps, [0.5, 2.5, 5.0, 100.0])


# ---------------------------------------------------------------------------
# Generator: determinism and schema
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_frames() -> None:
    closes = make_closes(n_symbols=8, n_days=120)
    config = OptionDemoConfig(sessions=60, chain_sessions=None)
    first, second = generate_option_data(closes, config), generate_option_data(closes, config)
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_symbol_data_does_not_depend_on_other_symbols() -> None:
    closes = make_closes(n_symbols=8, n_days=120)
    config = OptionDemoConfig(sessions=60)
    full = generate_option_data(closes, config).underlying_data
    one = generate_option_data(closes[closes["ticker"] == "S000"], config).underlying_data
    pd.testing.assert_frame_equal(
        full[full["symbol"] == "S000"].reset_index(drop=True), one.reset_index(drop=True)
    )


def test_different_seed_changes_data() -> None:
    closes = make_closes(n_symbols=3, n_days=80)
    a = generate_option_data(closes, OptionDemoConfig(sessions=40, seed=1)).underlying_data
    b = generate_option_data(closes, OptionDemoConfig(sessions=40, seed=2)).underlying_data
    assert not a["option_volume"].equals(b["option_volume"])


def _parse_ddl(ddl: str) -> dict[str, dict[str, str]]:
    tables: dict[str, dict[str, str]] = {}
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", ddl, re.S):
        columns = {}
        for line in body.strip().splitlines():
            match = re.match(r"\s*([a-z_]+)\s+(DATE|VARCHAR|INTEGER|DOUBLE PRECISION)\b", line)
            if match:
                columns[match.group(1)] = match.group(2).lower()
        tables[name] = columns
    return tables


def test_registry_matches_ddl() -> None:
    assert _parse_ddl(option_data_ddl()) == TABLES


def test_frames_match_registry(frames: OptionFrames) -> None:
    kinds = {"date": "M", "integer": "i", "double precision": "f"}
    for table, df in frames._asdict().items():
        registry = TABLES[table]
        assert list(df.columns) == list(registry), table
        for column, sql_type in registry.items():
            if sql_type == "varchar":
                assert pd.api.types.is_string_dtype(df[column]), (table, column)
            else:
                assert df[column].dtype.kind == kinds[sql_type], (table, column)
    chain_key = ["collection_date", "symbol", "expiration_date", "strike", "option_type"]
    assert not frames.option_chain_data.duplicated(chain_key).any()
    assert not frames.underlying_data.duplicated(["symbol", "collection_date"]).any()
    assert not frames.underlying_data_avg.duplicated(["symbol", "date"]).any()


# ---------------------------------------------------------------------------
# Generator: chain and aggregate consistency
# ---------------------------------------------------------------------------


def test_chain_rows_are_consistent(frames: OptionFrames) -> None:
    chain = frames.option_chain_data
    assert set(chain["option_type"]) == {"call", "put"}
    assert (chain["expiration_date"] > chain["collection_date"]).all()
    assert (chain["day_low_price"] <= chain["price"]).all()
    assert (chain["price"] <= chain["day_high_price"]).all()
    assert (chain["day_low_price"] <= chain["day_open_price"]).all()
    assert (chain["day_open_price"] <= chain["day_high_price"]).all()
    assert (chain["volm_buy"] + chain["volm_sell"] == chain["day_volume"]).all()
    assert (chain[["oi", "day_volume", "volm_buy", "volm_sell"]] >= 0).all().all()
    assert chain["volatility"].between(0.05, 3.0).all()
    calls = chain["option_type"] == "call"
    assert (chain.loc[calls, "gxoi"] >= 0).all()
    assert (chain.loc[~calls, "gxoi"] <= 0).all()
    sign = np.where(calls, 1.0, -1.0)
    np.testing.assert_allclose(chain["dxoi"], sign * chain["delta"] * chain["oi"] * 100)
    assert (chain["dxoi"] >= 0).all()


def test_chain_history_is_trimmed(frames: OptionFrames) -> None:
    chain_dates = frames.option_chain_data["collection_date"].unique()
    assert len(chain_dates) == OptionDemoConfig().chain_sessions
    assert frames.underlying_data["collection_date"].nunique() == SESSIONS


def test_open_interest_is_persistent(frames: OptionFrames) -> None:
    chain = frames.option_chain_data
    key = ["symbol", "expiration_date", "strike", "option_type"]
    ordered = chain[chain["oi"] >= 100].sort_values([*key, "collection_date"])
    change = ordered.groupby(key)["oi"].pct_change().dropna()
    assert change.abs().median() < 0.1


def test_open_interest_concentrates_near_the_money(
    frames: OptionFrames, closes: pd.DataFrame
) -> None:
    chain = frames.option_chain_data
    today = chain[chain["collection_date"] == chain["collection_date"].max()]
    spot = closes[closes["date"] == closes["date"].max()].set_index("ticker")["close"]
    moneyness = (today["strike"] / today["symbol"].map(spot) - 1).abs()
    near = today.loc[moneyness < 0.05, "oi"].mean()
    far = today.loc[moneyness > 0.15, "oi"].mean()
    assert near > 2 * far


def test_aggregates_match_chain(frames: OptionFrames) -> None:
    chain = frames.option_chain_data
    sums = chain.groupby(["symbol", "collection_date"]).agg(
        option_volume=("day_volume", "sum"), gxoi=("gxoi", "sum"), dxoi=("dxoi", "sum")
    )
    und = frames.underlying_data.set_index(["symbol", "collection_date"]).loc[sums.index]
    np.testing.assert_array_equal(und["option_volume"], sums["option_volume"])
    np.testing.assert_allclose(und["gxoi"], sums["gxoi"], rtol=1e-9)
    np.testing.assert_allclose(und["dxoi"], sums["dxoi"], rtol=1e-9)
    ratio = und["put_call_ratio"] * und["call_volume"]
    np.testing.assert_allclose(ratio, und["put_volume"], rtol=1e-9)


def test_avg_is_trailing_mean_of_prior_sessions(frames: OptionFrames) -> None:
    und = frames.underlying_data
    avg = frames.underlying_data_avg
    symbol_rows = und[und["symbol"] == "S000"].sort_values("collection_date").reset_index(drop=True)
    for i in (AVG_WINDOW, 100, len(symbol_rows) - 1):
        day = symbol_rows.loc[i, "collection_date"]
        expected = symbol_rows.loc[i - AVG_WINDOW : i - 1, list(UNDERLYING_VALUE_COLUMNS)].mean()
        row = avg[(avg["symbol"] == "S000") & (avg["date"] == day)].iloc[0]
        actual = row[[f"{c}_avg" for c in UNDERLYING_VALUE_COLUMNS]].astype(float).to_numpy()
        np.testing.assert_allclose(actual, expected.to_numpy(dtype=float), rtol=1e-9)
    assert avg.groupby("symbol").size().max() == SESSIONS - AVG_WINDOW
    assert avg["date"].max() == und["collection_date"].max()


def test_avg_of_empty_frame_is_empty(frames: OptionFrames) -> None:
    out = compute_underlying_avg(frames.underlying_data.iloc[:0])
    assert out.empty
    assert list(out.columns) == list(TABLES["underlying_data_avg"])


def test_term_structure_and_smile(frames: OptionFrames) -> None:
    und = frames.underlying_data
    assert und[["volatility", "front_volatility", "back_volatility"]].gt(0).all().all()
    assert not np.allclose(und["front_volatility"], und["back_volatility"])
    chain = frames.option_chain_data
    puts = chain[
        (chain["option_type"] == "put")
        & (chain["collection_date"] == chain["collection_date"].max())
    ]
    first = puts.groupby(["symbol", "expiration_date"])
    low_strike_iv = first.apply(lambda g: g.loc[g["strike"].idxmin(), "volatility"])
    mid_iv = first["volatility"].median()
    assert (low_strike_iv > mid_iv).mean() > 0.9  # downside skew


# ---------------------------------------------------------------------------
# Planted events: each dashboard scan finds them on the latest date
# ---------------------------------------------------------------------------


def test_vanna_extremes(underlying: pd.DataFrame, latest: pd.Timestamp) -> None:
    hits = scan_vanna_extreme(underlying, latest)
    assert hits[PLANTED.vanna_high] == 1.0
    assert hits[PLANTED.vanna_low] == 0.0


def test_flownet_extreme(underlying: pd.DataFrame, latest: pd.Timestamp) -> None:
    hits = scan_flownet_extreme(underlying, latest)
    assert hits[PLANTED.flownet_low] == 0.0


def test_volume_spike(underlying: pd.DataFrame, latest: pd.Timestamp) -> None:
    hits = scan_volume_spike(underlying, latest)
    assert hits[PLANTED.volume_spike] >= 3.0
    assert len(hits) < N_SYMBOLS / 4


def test_dxoi_change(underlying: pd.DataFrame, latest: pd.Timestamp) -> None:
    hits = scan_dxoi_change(underlying, latest)
    assert hits[PLANTED.dxoi_jump] >= 20.0
    assert len(hits) < N_SYMBOLS / 4


def test_gex_buckets_spread(underlying: pd.DataFrame, latest: pd.Timestamp) -> None:
    buckets = gex_buckets(underlying, latest)
    for symbol in PLANTED.negative_gex:
        assert buckets[symbol] == "Negative GEX"
    assert buckets.nunique() >= 4


def test_option_screener_hit(frames: OptionFrames, latest: pd.Timestamp) -> None:
    multiples = screener_multiples(frames, latest, "option_volume")
    assert multiples[PLANTED.volume_spike] >= 5.0
    assert (multiples >= 5.0).sum() <= 3


def test_some_but_not_all_symbols_have_weeklies(frames: OptionFrames) -> None:
    weekly = weekly_symbols(frames.option_chain_data)
    assert 0 < len(weekly) < N_SYMBOLS


def test_most_symbols_are_unremarkable(underlying: pd.DataFrame, latest: pd.Timestamp) -> None:
    flagged = set(scan_flownet_extreme(underlying, latest).index)
    flagged |= set(scan_volume_spike(underlying, latest).index)
    flagged |= set(scan_dxoi_change(underlying, latest).index)
    assert len(flagged) < N_SYMBOLS / 2


def test_planted_symbols_missing_from_input_are_skipped() -> None:
    closes = make_closes(n_symbols=10, n_days=80)
    closes = closes[~closes["ticker"].isin([PLANTED.vanna_high, PLANTED.volume_spike])]
    frames = generate_option_data(closes, OptionDemoConfig(sessions=40))
    assert PLANTED.vanna_high not in set(frames.underlying_data["symbol"])


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_runtime_for_demo_size(closes: pd.DataFrame) -> None:
    start = time.perf_counter()
    generate_option_data(closes, OptionDemoConfig(sessions=SESSIONS))
    assert time.perf_counter() - start < 20.0
