# Option-data contract

The dashboard reads option data from a second PostgreSQL database (`OPTION_DB_NAME`,
default `option_data`). This document defines the subset of that database the
dashboard depends on: three tables, their keys, and what every column means. The
DDL is [`src/stock_dashboard/schema/option_data.sql`](../src/stock_dashboard/schema/option_data.sql).
It is idempotent plain PostgreSQL and needs no extensions.

A provider-specific database can hold the same tables with more columns, and the
dashboard works against it as long as the columns below exist with these meanings
(see [Extra columns](#extra-columns)).

Pages that need option data show a message instead of failing when the database
or these tables are missing.

## Conventions

- **Dates.** `collection_date` is the trading session a row describes, as of that
  session's close. There is one snapshot per session; intraday snapshots are out
  of scope.
- **"Latest".** Queries anchor on the data, never on the wall clock. "The latest
  session" is `MAX(collection_date)` of the table being read, taken across all
  symbols. A symbol with no row on that date has no current reading, so it drops
  out of the latest-day scans. The page that looks at a single symbol (Option Data)
  uses that symbol's own latest date.
- **Averages.** The `underlying_data_avg` row for `(symbol, date)` holds that
  symbol's averages for the session `date`, so it joins on
  `underlying_data_avg.date = underlying_data.collection_date`. Both tables must
  therefore have the same latest date.
- **Contracts.** One contract covers 100 shares (`CONTRACT_MULTIPLIER`). Prices
  are quoted per share.
- **Greek units.** delta is per $1 move in the underlying, and gamma is the change
  in delta per $1. theta (in $) and charm (in delta) are per calendar day. vega
  (in $) and vanna (in delta) are per volatility point, i.e. a 0.01 change in
  implied volatility. vomma is the change in vega per volatility point.
- **Exposure sign (`*xoi` columns).** Exposures are computed from the dealers'
  side, under the usual convention that **dealers are long the calls and short
  the puts** that make up open interest. Each one is

  `s × greek × oi × 100`, where s = +1 for calls and −1 for puts.

  This is a modeling assumption, not an observed position. With it, `gxoi < 0`
  means dealers are net short gamma: their hedging trades go with the market's
  moves, which tends to amplify volatility. `gxoi > 0` means their hedging leans
  against the moves. Under the same assumption `dxoi` is never negative per
  contract, because a long call and a short put are both long delta.
- **Customer flow.** `volm_buy` / `volm_sell` count contracts that customers
  bought and sold. Flow columns take the customer's side: `deltas` and
  `gammas` are the delta and gamma customers net bought (negative when net
  sold), and `flownet` / `vflownet` are positive for bullish flow (calls
  bought, puts sold) and negative for bearish flow (calls sold, puts bought).
- **Max pain** (Option Contract page) uses raw `oi` as if customers held all of it long. This
  is a separate, naive assumption, documented in `logic/max_pain.py`.

## `option_chain_data`

One row per contract per session.
Primary key: `(collection_date, symbol, expiration_date, strike, option_type)`.
There is a secondary index on `(symbol, expiration_date, collection_date)`.

| Column | Type | Meaning and units |
|---|---|---|
| `collection_date` | date | Session the snapshot describes |
| `symbol` | varchar | Underlying ticker |
| `expiration_date` | date | Contract expiration. The demo lists only expirations after `collection_date`; a real feed may include same-day expirations, which the pages show like any other |
| `strike` | double | Strike price, $ |
| `option_type` | varchar | `'call'` or `'put'` |
| `price` | double | Last trade price, $ per share. The dashboard treats it as the contract's close |
| `day_open_price`, `day_high_price`, `day_low_price` | double | The contract's session open, high and low, $ per share |
| `oi` | integer | Open interest, contracts |
| `day_volume` | integer | Contracts traded in the session |
| `volatility` | double | Implied volatility, annualized fraction (0.25 = 25%) |
| `delta`, `gamma`, `theta`, `vega`, `vanna`, `charm`, `vomma` | double | Greeks per share, in the units under Conventions |
| `volm_buy`, `volm_sell` | integer | Contracts bought and sold by customers in the session |
| `gxoi` | double | s × gamma × oi × 100: dealer shares of delta gained per $1 rise in the underlying |
| `dxoi` | double | s × delta × oi × 100: dealer delta, in shares |
| `vxoi` | double | s × vega × oi × 100: dealer $ P&L per volatility point |
| `vannaxoi` | double | s × vanna × oi × 100: dealer shares of delta gained per volatility point |
| `charmxoi` | double | s × charm × oi × 100: dealer shares of delta gained per calendar day |

## `underlying_data`

One row per underlying per session. Each row aggregates that session's
`option_chain_data` rows for the symbol, plus implied-volatility levels.
Primary key: `(symbol, collection_date)`. There is an index on `(collection_date)`.

| Column | Type | Meaning and units |
|---|---|---|
| `symbol`, `collection_date` | varchar, date | Key |
| `volatility` | double | At-the-money implied volatility at a constant 30-day maturity |
| `front_volatility` | double | ATM implied volatility of the nearest expiration at least 7 days out |
| `back_volatility` | double | ATM implied volatility of the furthest listed expiration. `front − back > 0` means an inverted term structure |
| `call_volume`, `put_volume`, `option_volume` | integer | Σ `day_volume` over calls, over puts, and over both |
| `put_call_ratio` | double | `put_volume / call_volume`; NULL when there's no call volume |
| `flownet` | double | Net customer premium in $, with directional sign: Σ_calls (buy − sell) × price × 100 − Σ_puts (buy − sell) × price × 100. Positive = net bullish customer flow |
| `vflownet` | double | The same in contracts: Σ_calls (buy − sell) − Σ_puts (buy − sell) |
| `flowratio` | double | Premium customers bought / premium they sold, both rights |
| `vflowratio` | double | Contracts customers bought / contracts they sold, both rights |
| `volm_call_ratio`, `volm_put_ratio` | double | Customer buy / sell contracts, for calls and for puts |
| `deltas` | double | Net customer delta traded, in shares: Σ (buy − sell) × delta × 100 |
| `gammas` | double | Net customer gamma traded: Σ (buy − sell) × gamma × 100 |
| `dxoi`, `gxoi`, `vxoi`, `vannaxoi`, `charmxoi` | double | Σ of the chain's exposure columns (dealer sign) |
| `txoi` | double | Σ s × theta × oi × 100: dealer $ theta per calendar day |

A ratio is NULL when its denominator is zero.

The columns that can't be negative are the implied volatilities, the volumes and
the ratios (`put_call_ratio`, `flowratio`, `vflowratio`, `volm_call_ratio`,
`volm_put_ratio`): `option_queries.NON_NEGATIVE_COLUMNS`. Only for these is "N ×
its average" meaningful; flownet, vflownet, deltas, gammas and the exposures are
signed, and a multiple of a signed average can flip sign or explode near zero.

## `underlying_data_avg`

One row per underlying per session.
Primary key: `(symbol, date)`. There is an index on `(date)`.

For every value column `c` of `underlying_data`, `c_avg` (double precision) is the
mean of `c` over the **30 sessions before `date`, not including `date`**, with NULLs
skipped. The baseline leaves out the session being compared, so a ratio like
`option_volume / option_volume_avg` asks how unusual today is relative to the
previous 30 sessions. This matches the volume-spike scan's own trailing average,
which also excludes the current session. A symbol gets avg rows once it has 30
earlier sessions.

## Which pages read what

| Page | Tables | Use |
|---|---|---|
| 2 Stock Screener | `underlying_data`, `underlying_data_avg`, `option_chain_data` | Option pre-filters (volume vs. average, weekly expirations, signal scans) and the option columns in the results grid |
| 4 Option Data | `underlying_data`, `underlying_data_avg` | A symbol's recent rows, the latest / average ratio of every numeric column, and the history of one column against its average |
| 5 Option Screener | `underlying_data`, `underlying_data_avg` | Symbols whose latest value is at least N × its average, for a chosen column |
| 6 Signal Scans | `underlying_data` | Vanna extreme, flownet extreme, volume spike, dxoi change and negative-GEX scans on the latest session |
| 10 Option Contract | `option_chain_data` | OI by strike, max pain over time, surface richness (the contract's IV relative to ATM, ranked along its expiration's smile; see [DESIGN.md](DESIGN.md#option-contract-page)), and contract OHLC with OI/volume |

"Weekly expirations" means at least 3 distinct expirations within 30 days of the
latest `collection_date`.

Page 5 offers only the non-negative columns (above) that have an `_avg` partner,
and counts a row only when its average is positive.

## Extra columns

- **Page 4** finds the numeric `underlying_data` columns through
  `information_schema`, so an extra numeric column can be charted. Text or
  timestamp columns appear in its table of recent rows only. An extra column
  with a `<column>_avg` partner in `underlying_data_avg` also gets the ratio to
  its average and an average line; one without is charted alone.
- **Page 5** offers an extra column only if it has an `_avg` partner and no
  negative value on the latest session.
- The list of numeric columns, used to validate a chosen column, is re-read
  every 60 seconds, so a column added while the app runs appears without a
  restart.

## Demo data

The demo database is filled by `stock_dashboard.demo`. The generator is
deterministic: the same closes and seed always give identical frames, and all of
the data is fictional.

```python
from stock_dashboard.demo.load import read_closes, write_option_data
from stock_dashboard.demo.options import OptionDemoConfig, generate_option_data

# date, ticker, close from fact_commons_ohlc and fact_index_ohlc
closes = read_closes(stocks_conn)
frames = generate_option_data(closes, OptionDemoConfig(sessions=260))
# applies the DDL, then replaces these symbols' rows
write_option_data(option_conn, frames)
```

### Input and output

- **Input.** Daily closes (columns `date`, `ticker`, `close`) for every
  underlying. In the demo these are the rs-pipeline synthetic market's stocks and
  ETFs. Pass the full history: realized volatility uses bars from before the
  generated window.
- **Window.** The trailing `sessions` distinct input dates (260 by default).
  `underlying_data` covers the whole window. `underlying_data_avg` starts 30
  sessions in.
- **Chain history.** Only the last `chain_sessions` sessions (80 by default) of
  chain rows are kept. That covers the full life of every expiration listed on
  the latest session, which is what the Option Contract page's max-pain history needs. Aggregates
  are still computed from full chains for every session. Pass
  `chain_sessions=None` to keep everything: about 3.3× the rows.

### Model

- **Expirations.** The next 3 monthly expirations (third Fridays). About 40% of
  underlyings, chosen per symbol from the seed, also get the next 5 weekly
  Fridays. Expirations fall strictly after the session.
- **Strikes.** A grid over spot × (1 ± 22%). The step is the "nice" number
  (0.5, 1, 2.5, 5, 10, 25, …) closest to 2.5% of spot.
- **Implied volatility.** The ATM level follows the underlying's 20-session
  realized volatility × 1.1, with a little persistent noise. Across maturities it
  decays toward a 120-session long-run level, so the term structure inverts after
  volatile stretches. A quadratic smile in standardized log-moneyness adds
  downside skew.
- **Prices and greeks.** Black–Scholes–Merton with r = 4% and a continuous
  dividend yield q = 1%. The open is priced at the prior close. The high and low
  are priced at the session's underlying range widened by 0.4%.
- **Open interest.** Lognormal in strike, centered slightly out of the money on a
  10-session moving average of spot, so it is sticky. Round strikes get more OI,
  front and monthly expirations are weighted up, OI builds after listing and
  winds down over the last 10 days before expiration, and each contract has a
  fixed level plus a slow drift. The typical day-to-day change is under 10%.
- **Volume and flow.** Poisson volume proportional to OI, scaled by a persistent
  per-symbol activity level that rises on large underlying moves. Volume is
  heavier near the money and in the last 10 days before expiration. A persistent
  daily bias splits customer buys and sells: bullish days buy calls and sell
  puts. In the demo, `volm_buy + volm_sell = day_volume`; a real feed may leave
  some volume unclassified.
- **Scale.** gxoi is in shares per $1, so the dashboard's presentational GEX
  buckets (0 / 5,000 / 15,000 / 50,000) spread the demo's underlyings across all
  five buckets. A provider's feed may scale gxoi differently. Only its sign has a
  fixed meaning.

### Planted events

A few underlyings carry deliberate signals, on the latest session (negative
GEX: every session), so every option scan has something to show. The symbols are the defaults of `PlantedEvents` in
`demo/options.py`, and all of them are stocks in the synthetic market. Any that
are missing from the input are skipped. The other symbols follow the model above.
The percentile scans can still flag some of them by chance, because a percentile
rank always puts some symbols in the tails.

| Event | Default symbol | What is changed | Scan it satisfies |
|---|---|---|---|
| Vanna high | RENI | Latest session: out-of-the-money OI × 3 | `vannaxoi` at the top of its last 63 sessions |
| Vanna low | VETO | Latest session: out-of-the-money OI × 0.25 | `vannaxoi` at the bottom of its last 63 sessions |
| Flownet low | DODO | Latest session: activity × 3; customers sell 85% of call volume and buy 85% of put volume | `flownet` at the bottom of its last 63 sessions |
| Volume spike | PAVE | Latest session: activity set to 10 × its 20-session mean | Option volume ≥ 2 × the prior-20-session mean. On Option Screener, ≥ 5 × `option_volume_avg` |
| dxoi jump | NIFE | Latest session: call OI × 1.6 | `dxoi` up ≥ 20% from the prior session |
| Negative GEX | ZUFE, SAQI | Put-heavy OI (30% calls) on every session | `gxoi < 0` |

Planted OI changes don't feed back into the generated volume, which comes from
the un-planted open interest. `tests/unit/test_demo_options.py` reproduces each
scan's rule in pandas and checks that the planted symbol is flagged.

### Size

With 48 underlyings (the synthetic market's 40 stocks, 7 ETFs and the
benchmark `MKT`) × 260 sessions, the output is about 622k chain rows, 12.5k
`underlying_data` rows and 11k `underlying_data_avg` rows; with full chain history it is about 2.05M chain
rows. The chain table uses about 200 MB on disk, including indexes. Generation
and load times are in [REQUIREMENTS.md](REQUIREMENTS.md#performance).

### Loading

`write_option_data(conn, frames)` runs the DDL, deletes the rows of the symbols
in `frames` from all three tables, and COPYs the new rows, all in one
transaction committed on success. Re-running it is safe and gives the same
result. It leaves other symbols' rows alone, so it can't wipe a database it
didn't write.
