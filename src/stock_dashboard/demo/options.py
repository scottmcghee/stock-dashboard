"""Deterministic synthetic option chains and per-underlying aggregates.

Given daily closes for a set of underlyings, `generate_option_data` builds the
three contract tables (see docs/DATA_CONTRACT.md):

- option_chain_data: for every underlying and session, monthly expirations
  (third Fridays) and, for some underlyings, weekly ones; strikes on a grid
  around spot; Black-Scholes-Merton prices and greeks; open interest that is
  concentrated near the money and at round strikes and drifts slowly; customer
  buy/sell volume consistent with open interest.
- underlying_data: per (symbol, session) aggregates of that session's chain.
- underlying_data_avg: trailing 30-session means of underlying_data, over the
  30 sessions before the row's date.

Implied volatility follows the underlying's realized volatility, with a term
structure that mean-reverts toward its longer-run level and a skewed smile.

A few underlyings get planted events on the latest session (negative GEX:
every session; see `PlantedEvents`) so every option scan returns rows on the
demo data. All randomness comes from numpy Generators keyed by (seed, symbol), so one
symbol's data does not depend on which other symbols are generated.
"""

import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cache
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from stock_dashboard.demo.bsm import DAYS_PER_YEAR, FloatArray, bsm_greeks, bsm_price
from stock_dashboard.demo.contract import (
    OPTION_CHAIN_COLUMNS,
    UNDERLYING_AVG_COLUMNS,
    UNDERLYING_COLUMNS,
    UNDERLYING_VALUE_COLUMNS,
)

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

RISK_FREE_RATE: Final = 0.04
DIVIDEND_YIELD: Final = 0.01
CONTRACT_MULTIPLIER: Final = 100
AVG_WINDOW: Final = 30
"""Sessions in the underlying_data_avg window (the sessions before the row's date)."""
ATM_TENOR_DAYS: Final = 30
FRONT_MIN_DAYS: Final = 7
"""front_volatility uses the nearest expiration at least this many days out."""

TRADING_DAYS_PER_YEAR: Final = 252
REALIZED_VOL_WINDOW: Final = 20
REALIZED_VOL_MIN_BARS: Final = 5
FALLBACK_VOL: Final = 0.3
"""Used when a symbol has too few bars to estimate realized volatility."""
LONG_RUN_VOL_WINDOW: Final = 120
MIN_VOL: Final = 0.08
MAX_VOL: Final = 1.5
IV_PREMIUM: Final = 1.1
"""Implied over realized volatility."""
TERM_DECAY_YEARS: Final = 0.25
SKEW: Final = -0.06
CURVATURE: Final = 0.015
SMILE_CLIP: Final = 4.0
"""Bound on standardized moneyness in the smile, so far wings stay finite."""

NICE_STEPS: Final = np.array([0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500])
ROUND_STRIKE_OI_BOOST: Final = 1.8
MONTHLY_OI_WEIGHT: Final = 2.0
OI_TERM_DAYS: Final = 45.0
OI_MIN_WIDTH: Final = 0.04
OI_MAX_WIDTH: Final = 0.12
"""Bounds on the log-moneyness spread of open interest (one standard deviation)."""
OI_OTM_SHIFT: Final = 0.02
"""Calls cluster slightly above spot, puts slightly below (log-moneyness)."""
OI_REFERENCE_WINDOW: Final = 10
"""Open interest centers on a moving average of spot, so it is sticky."""
OI_DRIFT_PERIOD: Final = 40.0
OI_DRIFT_AMPLITUDE: Final = 0.12
LISTING_RAMP_DAYS: Final = 25.0
LISTING_RAMP_OFFSET: Final = 5.0
LISTING_RAMP_FLOOR: Final = 0.2
DAYS_PER_WEEK: Final = 7
DAYS_PER_MONTH: Final = 31
OI_SCALE_MEDIAN: Final = 1500.0
OI_SCALE_DISPERSION: Final = 0.5
CONTRACT_OI_DISPERSION: Final = 0.6
"""Spread of the fixed per-contract OI level (log units, uniform)."""
CALL_SHARE_RANGE: Final = (0.55, 0.68)
"""Share of open interest in calls, drawn per symbol."""
ROLL_OFF_DAYS: Final = 10.0
"""Open interest winds down over the last days before expiration."""
INTRADAY_WICK: Final = 0.004
NEAR_EXPIRY_DAYS: Final = 10
NEAR_EXPIRY_VOLUME_BOOST: Final = 2.5
NEAR_MONEY_WIDTH: Final = 0.05
NEAR_MONEY_FLOOR: Final = 0.3
"""Volume per unit OI far from the money, relative to at the money."""
MIN_VOLUME_OI: Final = 2.0
"""Added to OI in the volume mean so contracts with no OI can still trade."""
TURNOVER_RANGE: Final = (0.08, 0.15)
"""Daily volume / OI near the money, drawn per symbol."""
VOLUME_NOISE: Final = 0.4
MOVE_ACTIVITY: Final = 10.0
"""Activity rises by this much per unit of absolute log return."""
VOLUME_AVG_SESSIONS: Final = 20
# (persistence, shock sd) of the AR(1) processes driving activity, buy bias and IV.
ACTIVITY_AR: Final = (0.6, 0.25)
BUY_BIAS_AR: Final = (0.7, 0.05)
IV_NOISE_AR: Final = (0.8, 0.03)
BUY_FRACTION_NOISE: Final = 0.08
BUY_FRACTION_BOUNDS: Final = (0.03, 0.97)
MIN_PRICE: Final = 0.01

DEMO_SEED: Final = 7


@dataclass(frozen=True)
class PlantedEvents:
    """Underlyings that get a deliberate signal on the latest session (negative GEX:
    every session).

    Defaults name stocks from the rs-pipeline synthetic market; symbols absent
    from the input (or without a row on the latest date) are skipped.
    """

    vanna_high: str = "RENI"
    """Out-of-the-money OI triples: vannaxoi at its 63-session high."""
    vanna_low: str = "VETO"
    """Most out-of-the-money OI is closed: vannaxoi at its 63-session low."""
    flownet_low: str = "DODO"
    """Heavy customer call selling and put buying: flownet at its 63-session low."""
    volume_spike: str = "PAVE"
    """Option volume at VOLUME_SPIKE_MULTIPLE x its recent average."""
    dxoi_jump: str = "NIFE"
    """Call OI up by DXOI_JUMP_OI_MULTIPLE: dxoi up sharply from the prior session."""
    negative_gex: tuple[str, ...] = ("ZUFE", "SAQI")
    """Put-heavy open interest on every session, so gxoi < 0."""


VANNA_HIGH_OI_MULTIPLE: Final = 3.0
VANNA_LOW_OI_MULTIPLE: Final = 0.25
DXOI_JUMP_OI_MULTIPLE: Final = 1.6
FLOW_ACTIVITY_MULTIPLE: Final = 3.0
FLOW_SELL_CALL_BUY_FRACTION: Final = 0.15
VOLUME_SPIKE_MULTIPLE: Final = 10.0
NEGATIVE_GEX_CALL_SHARE: Final = 0.3


@dataclass(frozen=True)
class OptionDemoConfig:
    sessions: int = 260
    """Trailing sessions (distinct input dates) to generate."""
    chain_sessions: int | None = 80
    """Trailing sessions whose chain rows are kept (None = all). Aggregates always
    cover all `sessions`. 80 covers the full life of every expiration listed on
    the latest session."""
    weekly_expirations: int = 5
    monthly_expirations: int = 3
    weekly_fraction: float = 0.4
    """Share of underlyings (chosen per symbol from the seed) with weekly expirations."""
    strike_range: float = 0.22
    """Strikes span spot x (1 +/- strike_range)."""
    strike_spacing: float = 0.025
    """Target strike step as a fraction of spot, rounded to a nice number."""
    seed: int = DEMO_SEED
    planted: PlantedEvents = field(default_factory=PlantedEvents)


class OptionFrames(NamedTuple):
    option_chain_data: pd.DataFrame
    underlying_data: pd.DataFrame
    underlying_data_avg: pd.DataFrame


def generate_option_data(
    closes: pd.DataFrame, config: OptionDemoConfig | None = None
) -> OptionFrames:
    """Build the three contract tables from daily closes (columns date, ticker, close).

    Pass the full close history: realized volatility uses the bars before the
    generated window. Each symbol is generated on its own dates within the
    trailing `config.sessions` distinct dates of the input.
    """
    config = config or OptionDemoConfig()
    frame = closes.loc[:, ["date", "ticker", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["close"] = frame["close"].astype(float)
    frame = frame.dropna().sort_values(["ticker", "date"])

    all_dates = np.sort(frame["date"].unique())
    window = all_dates[-config.sessions :].astype("datetime64[D]")
    latest = window[-1]
    kept = window[-config.chain_sessions :] if config.chain_sessions else window
    chain_start = kept[0]

    chains: list[pd.DataFrame] = []
    underlyings: list[pd.DataFrame] = []
    for symbol, group in frame.groupby("ticker", sort=True):
        history = _SymbolHistory.from_closes(group, window)
        if history is None:
            continue
        chain, underlying = _generate_symbol(str(symbol), history, latest, config)
        chains.append(chain[chain["collection_date"] >= chain_start])
        underlyings.append(underlying)

    chain_df = _conform(chains, OPTION_CHAIN_COLUMNS)
    underlying_df = _conform(underlyings, UNDERLYING_COLUMNS)
    return OptionFrames(chain_df, underlying_df, compute_underlying_avg(underlying_df))


def compute_underlying_avg(underlying: pd.DataFrame) -> pd.DataFrame:
    """Trailing AVG_WINDOW-session mean of each value column, per symbol.

    The row for `date` averages the AVG_WINDOW sessions before it (not the
    session itself), so a value can be compared with its own recent baseline.
    NULLs are skipped. Rows start once a symbol has a full window of history.
    """
    if underlying.empty:
        return _conform([], UNDERLYING_AVG_COLUMNS)
    ordered = underlying.sort_values(["symbol", "collection_date"])
    values = ordered[list(UNDERLYING_VALUE_COLUMNS)].astype(float)
    grouped = values.groupby(ordered["symbol"], sort=False)
    means = grouped.transform(lambda s: s.rolling(AVG_WINDOW, min_periods=1).mean().shift(1))
    position = ordered.groupby("symbol", sort=False).cumcount()
    out = means.add_suffix("_avg")
    out.insert(0, "date", ordered["collection_date"])
    out.insert(0, "symbol", ordered["symbol"])
    out = out[position.to_numpy() >= AVG_WINDOW]
    return _conform([out], UNDERLYING_AVG_COLUMNS)


def nice_strike_step(spot: FloatArray, spacing: float) -> FloatArray:
    """Strike step nearest (in log terms) to spot x spacing, from NICE_STEPS."""
    target = np.log(spot * spacing)[:, None]
    nearest = np.abs(np.log(NICE_STEPS)[None, :] - target).argmin(axis=1)
    return np.asarray(NICE_STEPS[nearest], dtype=np.float64)


@cache
def weekly_expirations(session: date, count: int) -> tuple[date, ...]:
    """The next `count` Fridays strictly after `session`."""
    days_ahead = (4 - session.weekday()) % 7 or 7
    first = session + timedelta(days=days_ahead)
    return tuple(first + timedelta(weeks=i) for i in range(count))


@cache
def monthly_expirations(session: date, count: int) -> tuple[date, ...]:
    """The next `count` third Fridays strictly after `session`."""
    out: list[date] = []
    year, month = session.year, session.month
    while len(out) < count:
        third_friday = _third_friday(year, month)
        if third_friday > session:
            out.append(third_friday)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return tuple(out)


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def atm_volatility(short_level: FloatArray, long_level: FloatArray, t: FloatArray) -> FloatArray:
    """ATM implied vol at maturity t: short-term level decaying to the long-run level."""
    return long_level + (short_level - long_level) * np.exp(-t / TERM_DECAY_YEARS)


def smile_volatility(
    atm: FloatArray, strike: FloatArray, forward: FloatArray, t: FloatArray
) -> FloatArray:
    """Skewed quadratic smile in standardized log-moneyness ln(K/F) / (atm sqrt(t))."""
    x = np.clip(np.log(strike / forward) / (atm * np.sqrt(t)), -SMILE_CLIP, SMILE_CLIP)
    return atm * (1.0 + SKEW * x + CURVATURE * x * x)


@dataclass(frozen=True)
class _SymbolHistory:
    """Per-session inputs for one symbol, restricted to the generated window."""

    dates: npt.NDArray[np.datetime64]
    spot: FloatArray
    prev_spot: FloatArray
    gap_days: IntArray
    abs_return: FloatArray
    realized_vol: FloatArray
    long_run_vol: FloatArray
    spot_reference: FloatArray

    @classmethod
    def from_closes(
        cls, group: pd.DataFrame, window: npt.NDArray[np.datetime64]
    ) -> "_SymbolHistory | None":
        close = group.set_index("date")["close"]
        dates = close.index.to_numpy().astype("datetime64[D]")
        log_ret = np.log(close).diff()
        realized = (
            (
                log_ret.rolling(REALIZED_VOL_WINDOW, min_periods=REALIZED_VOL_MIN_BARS).std()
                * np.sqrt(TRADING_DAYS_PER_YEAR)
            )
            .bfill()
            .fillna(FALLBACK_VOL)
            .clip(MIN_VOL, MAX_VOL)
        )
        long_run = realized.rolling(LONG_RUN_VOL_WINDOW, min_periods=1).mean()
        reference = close.rolling(OI_REFERENCE_WINDOW, min_periods=1).mean()
        prev = close.shift(1).fillna(close)
        prev_dates = np.concatenate([dates[:1], dates[:-1]])

        keep = np.isin(dates, window)
        if not keep.any():
            return None
        gap = (dates - prev_dates).astype(np.int64)
        return cls(
            dates=dates[keep],
            spot=close.to_numpy(dtype=float)[keep],
            prev_spot=prev.to_numpy(dtype=float)[keep],
            gap_days=np.maximum(gap[keep], 1),
            abs_return=log_ret.abs().fillna(0.0).to_numpy(dtype=float)[keep],
            realized_vol=realized.to_numpy(dtype=float)[keep],
            long_run_vol=long_run.to_numpy(dtype=float)[keep],
            spot_reference=reference.to_numpy(dtype=float)[keep],
        )


def _symbol_rng(seed: int, symbol: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(symbol.encode())])


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> FloatArray:
    shocks = rng.normal(0.0, sd, n)
    out = np.empty(n)
    level = 0.0
    for i, shock in enumerate(shocks):
        level = phi * level + shock
        out[i] = level
    return out


def _hash_uniform(*keys: IntArray) -> FloatArray:
    """Stateless uniform(0, 1) per element from integer keys (splitmix64 mixing)."""
    h = np.zeros(keys[0].shape, dtype=np.uint64)
    for key in keys:
        h ^= key.astype(np.uint64)
        h += np.uint64(0x9E3779B97F4A7C15)
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        h ^= h >> np.uint64(31)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _expiration_lists(
    dates: npt.NDArray[np.datetime64], weekly: bool, config: OptionDemoConfig
) -> list[tuple[npt.NDArray[np.datetime64], BoolArray]]:
    out = []
    for d in dates:
        session = pd.Timestamp(d).date()
        monthly = monthly_expirations(session, config.monthly_expirations)
        weeks = weekly_expirations(session, config.weekly_expirations) if weekly else ()
        expiries = sorted(set(monthly) | set(weeks))
        out.append(
            (
                np.array(expiries, dtype="datetime64[D]"),
                np.array([e in monthly for e in expiries], dtype=bool),
            )
        )
    return out


@dataclass(frozen=True)
class _Profile:
    """One symbol's random constants and per-session state."""

    has_weeklies: bool
    oi_scale: float
    call_share: float
    turnover: float
    activity: FloatArray
    buy_bias: FloatArray
    iv_short: FloatArray
    iv_long: FloatArray


@dataclass(frozen=True)
class _Grid:
    """Chain rows (session x strike x expiration x right) as flat arrays."""

    session: IntArray
    strike: FloatArray
    strike_step: FloatArray
    expiry: npt.NDArray[np.datetime64]
    is_monthly: BoolArray
    is_call: BoolArray
    dte: IntArray
    listed_expiries: npt.NDArray[np.datetime64]
    """Every session's expirations, concatenated session by session."""
    expiry_offsets: IntArray
    """Start of each session's block in `listed_expiries`."""


def _generate_symbol(
    symbol: str, h: _SymbolHistory, latest: np.datetime64, config: OptionDemoConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = _symbol_rng(config.seed, symbol)
    planted = config.planted
    last = len(h.dates) - 1
    plant_today = h.dates[last] == latest
    profile = _draw_profile(rng, symbol, h, plant_today, config)
    g = _contract_grid(h, profile.has_weeklies, config)

    t = g.dte / DAYS_PER_YEAR
    spot = h.spot[g.session]
    atm = atm_volatility(profile.iv_short[g.session], profile.iv_long[g.session], t)
    forward = spot * np.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * t)
    iv = smile_volatility(atm, g.strike, forward, t)
    greeks = bsm_greeks(spot, g.strike, t, iv, g.is_call, RISK_FREE_RATE, DIVIDEND_YIELD)

    base_oi = _open_interest(symbol, h, g, atm, profile, config)
    today = (g.session == last) & plant_today
    call_buy_override = FLOW_SELL_CALL_BUY_FRACTION if symbol == planted.flownet_low else None
    day_volume, volm_buy = _volume_and_flow(
        rng, g, base_oi, spot, profile, today, call_buy_override
    )

    otm = np.where(g.is_call, g.strike > spot, g.strike < spot)
    oi_multiple = np.ones(len(base_oi))
    if symbol == planted.vanna_high:
        oi_multiple[today & otm] = VANNA_HIGH_OI_MULTIPLE
    if symbol == planted.vanna_low:
        oi_multiple[today & otm] = VANNA_LOW_OI_MULTIPLE
    if symbol == planted.dxoi_jump:
        oi_multiple[today & g.is_call] = DXOI_JUMP_OI_MULTIPLE
    oi = np.round(base_oi * oi_multiple).astype(np.int64)

    close, open_, high, low = _day_prices(h, g, iv, greeks.price)
    # Dealer book: long calls, short puts (see docs/DATA_CONTRACT.md).
    dealer_shares = np.where(g.is_call, 1.0, -1.0) * oi * float(CONTRACT_MULTIPLIER)
    chain = pd.DataFrame(
        {
            "collection_date": h.dates[g.session],
            "symbol": symbol,
            "expiration_date": g.expiry,
            "strike": g.strike,
            "option_type": np.where(g.is_call, "call", "put"),
            "price": close,
            "day_open_price": open_,
            "day_high_price": high,
            "day_low_price": low,
            "oi": oi,
            "day_volume": day_volume,
            "volatility": iv,
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "theta": greeks.theta,
            "vega": greeks.vega,
            "vanna": greeks.vanna,
            "charm": greeks.charm,
            "vomma": greeks.vomma,
            "volm_buy": volm_buy,
            "volm_sell": day_volume - volm_buy,
            "gxoi": greeks.gamma * dealer_shares,
            "dxoi": greeks.delta * dealer_shares,
            "vxoi": greeks.vega * dealer_shares,
            "vannaxoi": greeks.vanna * dealer_shares,
            "charmxoi": greeks.charm * dealer_shares,
        }
    )
    underlying = _aggregate(chain, greeks.theta * dealer_shares, g.session, len(h.dates), g.is_call)
    underlying.insert(0, "collection_date", h.dates)
    underlying.insert(0, "symbol", symbol)
    for name, values in _session_volatility(h, g, profile).items():
        underlying[name] = values
    return chain, underlying


def _draw_profile(
    rng: np.random.Generator,
    symbol: str,
    h: _SymbolHistory,
    plant_today: bool,
    config: OptionDemoConfig,
) -> _Profile:
    planted = config.planted
    n = len(h.dates)
    has_weeklies = bool(rng.random() < config.weekly_fraction)
    oi_scale = float(rng.lognormal(np.log(OI_SCALE_MEDIAN), OI_SCALE_DISPERSION))
    call_share = float(rng.uniform(*CALL_SHARE_RANGE))
    if symbol in planted.negative_gex:
        call_share = NEGATIVE_GEX_CALL_SHARE
    turnover = float(rng.uniform(*TURNOVER_RANGE))
    activity = np.exp(_ar1(rng, n, *ACTIVITY_AR)) * (1.0 + MOVE_ACTIVITY * h.abs_return)
    buy_bias = 0.5 + _ar1(rng, n, *BUY_BIAS_AR)
    iv_noise = np.exp(_ar1(rng, n, *IV_NOISE_AR))
    last = n - 1
    if plant_today and symbol == planted.volume_spike:
        baseline = activity[max(0, last - VOLUME_AVG_SESSIONS) : last].mean()
        activity[last] = VOLUME_SPIKE_MULTIPLE * baseline
    if plant_today and symbol == planted.flownet_low:
        activity[last] *= FLOW_ACTIVITY_MULTIPLE
    return _Profile(
        has_weeklies=has_weeklies,
        oi_scale=oi_scale,
        call_share=call_share,
        turnover=turnover,
        activity=activity,
        buy_bias=buy_bias,
        iv_short=h.realized_vol * IV_PREMIUM * iv_noise,
        iv_long=h.long_run_vol * IV_PREMIUM,
    )


def _contract_grid(h: _SymbolHistory, weekly: bool, config: OptionDemoConfig) -> _Grid:
    """Expand sessions into strike, expiration and call/put rows (no Python loop over rows)."""
    n = len(h.dates)
    step = nice_strike_step(h.spot, config.strike_spacing)
    lo = np.ceil(h.spot * (1 - config.strike_range) / step).astype(np.int64)
    hi = np.floor(h.spot * (1 + config.strike_range) / step).astype(np.int64)
    n_strikes = hi - lo + 1
    s_session = np.repeat(np.arange(n), n_strikes)
    strike_offsets = np.concatenate([[0], np.cumsum(n_strikes)[:-1]])
    strike_index = lo[s_session] + np.arange(len(s_session)) - strike_offsets[s_session]
    strike_step = step[s_session]
    strike_values = np.round(strike_index * strike_step, 4)

    exp_lists = _expiration_lists(h.dates, weekly, config)
    n_exp = np.array([len(e) for e, _ in exp_lists], dtype=np.int64)
    exp_flat = np.concatenate([e for e, _ in exp_lists])
    monthly_flat = np.concatenate([m for _, m in exp_lists])
    exp_offsets = np.concatenate([[0], np.cumsum(n_exp)[:-1]])
    se_strike = np.repeat(np.arange(len(s_session)), n_exp[s_session])
    se_session = s_session[se_strike]
    se_first = np.concatenate([[0], np.cumsum(n_exp[s_session])[:-1]])
    se_exp = exp_offsets[se_session] + np.arange(len(se_strike)) - se_first[se_strike]

    row = np.repeat(np.arange(len(se_strike)), 2)
    session = se_session[row]
    expiry = exp_flat[se_exp[row]]
    return _Grid(
        session=session,
        strike=strike_values[se_strike[row]],
        strike_step=strike_step[se_strike[row]],
        expiry=expiry,
        is_monthly=monthly_flat[se_exp[row]],
        is_call=np.tile(np.array([True, False]), len(se_strike)),
        dte=(expiry - h.dates[session]).astype(np.int64),
        listed_expiries=exp_flat,
        expiry_offsets=exp_offsets,
    )


def _open_interest(
    symbol: str,
    h: _SymbolHistory,
    g: _Grid,
    atm: FloatArray,
    profile: _Profile,
    config: OptionDemoConfig,
) -> FloatArray:
    """Open interest before planted events: sticky, near the money, at round strikes."""
    t = g.dte / DAYS_PER_YEAR
    log_m = np.log(g.strike / h.spot_reference[g.session])
    width = np.clip(atm * np.sqrt(t), OI_MIN_WIDTH, OI_MAX_WIDTH)
    center = np.where(g.is_call, OI_OTM_SHIFT, -OI_OTM_SHIFT)
    shape = np.exp(-0.5 * ((log_m - center) / width) ** 2)
    side = 2.0 * np.where(g.is_call, profile.call_share, 1.0 - profile.call_share)
    term = np.where(g.is_monthly, MONTHLY_OI_WEIGHT, 1.0) / (1.0 + g.dte / OI_TERM_DAYS)
    round_unit = 10.0 ** (np.floor(np.log10(g.strike_step)) + 1)
    remainder = np.mod(g.strike, round_unit)
    is_round = np.isclose(remainder, 0.0) | np.isclose(remainder, round_unit)
    boost = np.where(is_round, ROUND_STRIKE_OI_BOOST, 1.0)

    listing_window = np.where(
        g.is_monthly,
        DAYS_PER_MONTH * config.monthly_expirations,
        DAYS_PER_WEEK * config.weekly_expirations,
    )
    age = listing_window - g.dte
    ramp = np.clip((age + LISTING_RAMP_OFFSET) / LISTING_RAMP_DAYS, LISTING_RAMP_FLOOR, 1.0)
    ramp *= np.minimum(1.0, g.dte / ROLL_OFF_DAYS)

    keys = (
        np.full(len(g.session), zlib.crc32(symbol.encode()) ^ config.seed, dtype=np.int64),
        g.expiry.astype(np.int64),
        np.round(g.strike * 100).astype(np.int64),
        g.is_call.astype(np.int64),
    )
    level = np.exp(CONTRACT_OI_DISPERSION * (_hash_uniform(*keys) - 0.5))
    phase = _hash_uniform(*keys[::-1])
    session_ordinal = h.dates[g.session].astype(np.int64)
    drift = np.exp(
        OI_DRIFT_AMPLITUDE * np.sin(2 * np.pi * (session_ordinal / OI_DRIFT_PERIOD + phase))
    )
    oi: FloatArray = profile.oi_scale * side * shape * term * boost * ramp * level * drift
    return oi


def _volume_and_flow(
    rng: np.random.Generator,
    g: _Grid,
    base_oi: FloatArray,
    spot: FloatArray,
    profile: _Profile,
    today: BoolArray,
    call_buy_override: float | None,
) -> tuple[IntArray, IntArray]:
    """Day volume (Poisson around an OI-proportional mean) and the customer-bought part."""
    n = len(g.session)
    spot_m = np.log(g.strike / spot)
    near_money = NEAR_MONEY_FLOOR + (1 - NEAR_MONEY_FLOOR) * np.exp(
        -0.5 * (spot_m / NEAR_MONEY_WIDTH) ** 2
    )
    near_expiry = np.where(g.dte <= NEAR_EXPIRY_DAYS, NEAR_EXPIRY_VOLUME_BOOST, 1.0)
    mean = (
        (base_oi + MIN_VOLUME_OI)
        * profile.turnover
        * profile.activity[g.session]
        * near_money
        * near_expiry
    )
    day_volume = rng.poisson(mean * rng.lognormal(0.0, VOLUME_NOISE, n)).astype(np.int64)

    call_buy = profile.buy_bias[g.session]
    put_buy = 1.0 - call_buy
    if call_buy_override is not None:
        call_buy = np.where(today, call_buy_override, call_buy)
        put_buy = np.where(today, 1.0 - call_buy_override, put_buy)
    noise = rng.normal(0.0, BUY_FRACTION_NOISE, n)
    buy_fraction = np.clip(np.where(g.is_call, call_buy, put_buy) + noise, *BUY_FRACTION_BOUNDS)
    volm_buy = rng.binomial(day_volume, buy_fraction).astype(np.int64)
    return day_volume, volm_buy


def _day_prices(
    h: _SymbolHistory, g: _Grid, iv: FloatArray, model_close: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Close, open, high and low in cents: open at the prior close, high/low at a widened range."""
    t = g.dte / DAYS_PER_YEAR
    spot, prev = h.spot[g.session], h.prev_spot[g.session]
    t_open = t + h.gap_days[g.session] / DAYS_PER_YEAR
    r, q = RISK_FREE_RATE, DIVIDEND_YIELD
    open_ = _cents(bsm_price(prev, g.strike, t_open, iv, g.is_call, r, q))
    s_high = np.maximum(prev, spot) * (1 + INTRADAY_WICK)
    s_low = np.minimum(prev, spot) * (1 - INTRADAY_WICK)
    at_high = _cents(bsm_price(s_high, g.strike, t, iv, g.is_call, r, q))
    at_low = _cents(bsm_price(s_low, g.strike, t, iv, g.is_call, r, q))
    close = _cents(model_close)
    high = np.maximum.reduce([at_high, at_low, open_, close])
    low = np.minimum.reduce([at_high, at_low, open_, close])
    return close, open_, high, low


def _session_volatility(h: _SymbolHistory, g: _Grid, profile: _Profile) -> dict[str, FloatArray]:
    """ATM IV at 30 days, at the front expiration (>= FRONT_MIN_DAYS) and at the back one."""
    n_exp = np.diff(np.append(g.expiry_offsets, len(g.listed_expiries)))
    exp_t = (g.listed_expiries - np.repeat(h.dates, n_exp)).astype(np.int64) / DAYS_PER_YEAR
    back_t = np.maximum.reduceat(exp_t, g.expiry_offsets)
    eligible = np.where(exp_t >= FRONT_MIN_DAYS / DAYS_PER_YEAR, exp_t, np.inf)
    front_t = np.minimum.reduceat(eligible, g.expiry_offsets)
    front_t = np.where(np.isfinite(front_t), front_t, back_t)
    tenor = np.full(len(h.dates), ATM_TENOR_DAYS / DAYS_PER_YEAR)
    return {
        "volatility": atm_volatility(profile.iv_short, profile.iv_long, tenor),
        "front_volatility": atm_volatility(profile.iv_short, profile.iv_long, front_t),
        "back_volatility": atm_volatility(profile.iv_short, profile.iv_long, back_t),
    }


def _cents(price: FloatArray) -> FloatArray:
    return np.maximum(np.round(price, 2), MIN_PRICE)


def _aggregate(
    chain: pd.DataFrame, txoi: FloatArray, session: IntArray, n: int, is_call: BoolArray
) -> pd.DataFrame:
    """Per-session sums and ratios over one symbol's chain rows."""

    def total(values: FloatArray | IntArray, mask: BoolArray | None = None) -> FloatArray:
        weights = values if mask is None else np.where(mask, values, 0)
        return np.bincount(session, weights=weights.astype(np.float64), minlength=n)

    def ratio(num: FloatArray, den: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    is_put = ~is_call
    volume = chain["day_volume"].to_numpy()
    buy = chain["volm_buy"].to_numpy()
    sell = chain["volm_sell"].to_numpy()
    net = (buy - sell).astype(np.float64)
    premium = chain["price"].to_numpy() * CONTRACT_MULTIPLIER
    call_volume, put_volume = total(volume, is_call), total(volume, is_put)
    call_buy, call_sell = total(buy, is_call), total(sell, is_call)
    put_buy, put_sell = total(buy, is_put), total(sell, is_put)
    signed_right = np.where(is_call, 1.0, -1.0)

    out = {
        "call_volume": call_volume.astype(np.int64),
        "put_volume": put_volume.astype(np.int64),
        "option_volume": (call_volume + put_volume).astype(np.int64),
        "put_call_ratio": ratio(put_volume, call_volume),
        "flownet": total(signed_right * net * premium),
        "vflownet": total(signed_right * net),
        "flowratio": ratio(total(buy * premium), total(sell * premium)),
        "vflowratio": ratio(call_buy + put_buy, call_sell + put_sell),
        "volm_call_ratio": ratio(call_buy, call_sell),
        "volm_put_ratio": ratio(put_buy, put_sell),
        "deltas": total(net * chain["delta"].to_numpy() * CONTRACT_MULTIPLIER),
        "gammas": total(net * chain["gamma"].to_numpy() * CONTRACT_MULTIPLIER),
        **{c: total(chain[c].to_numpy()) for c in ("dxoi", "gxoi", "vxoi", "vannaxoi", "charmxoi")},
        "txoi": total(txoi),
    }
    return pd.DataFrame(out)


def _conform(frames: Iterable[pd.DataFrame], columns: dict[str, str]) -> pd.DataFrame:
    """Concatenate and coerce to the registry's column order and dtypes."""
    parts = [f for f in frames if not f.empty]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(columns))
    df = df.loc[:, list(columns)]
    dtypes = {
        "date": "datetime64[s]",
        "varchar": "str",
        "integer": "int64",
        "double precision": "float64",
    }
    return df.astype({c: dtypes[t] for c, t in columns.items()})
