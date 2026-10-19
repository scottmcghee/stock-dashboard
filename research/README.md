# Research scripts

Studies of the dashboard's indicators and option metrics. They read
the same two databases as the dashboard and write CSV tables (and, for
`ta_signal_analysis`, optional backtester presets); none of them changes a
database.

This document describes methods and their limits only. It quotes no results:
any result depends on the dataset, the date range and the parameters it was
run with. The demo database (see the main README) is a **synthetic** market
with generated option data. The scripts run on it so that they can be tested,
and whatever they report there means nothing about real markets.

## Running

```bash
uv run python -m research.signal_analysis --help
uv run python -m research.ta_signal_analysis --help
uv run python -m research.option_signal_analysis --help
uv run python -m research.trend_precursor_analysis --help
```

Run them from the repository root, as modules. Configuration comes from the
same environment variables as the dashboard: `POSTGRES_HOST`, `POSTGRES_PORT`,
`POSTGRES_USER`, `POSTGRES_PASSWORD`, `STOCKS_DB_NAME`, `OPTION_DB_NAME`,
`BENCHMARK_TICKER`, plus `CACHE_DIR` (the parquet cache that
`ta_signal_analysis` shares with the backtester) and `PRESET_DIR` (where
exported presets go). Every script takes `--start`/`--end` (default: all the
data), `--output-dir` (default `./research_results`, which git ignores) and
`--log-level`. Tables are printed to stdout and written as CSV; progress goes
to the log.

## Shared definitions

- **Session, bar**: one row of a ticker's daily data. Look-backs and horizons
  are counted in rows, not calendar days.
- **Forward return** `fwd_h`: close-to-close return from the close of day *t*
  to the close *h* sessions later, `close[t+h] / close[t] - 1`.
- **Univariate scan**: for each signal with enough rows, the Spearman rank
  correlation with the target and its p-value; the mean target in the
  signal's top and bottom deciles and their difference (`lift_bps`, in basis
  points); and the **hit rate**: the share of top-decile rows whose target is
  in the top decile of the whole sample (about 0.10 when the signal carries no
  information).
- **Replication**: the same Spearman correlation computed within each group
  (regime, sector, half of the sample, symbol or contract bucket).
  `pct_same_sign` is the share of groups whose correlation has the sign of the
  mean correlation, and `pct_sig_same_sign` the share that is also
  individually significant at an unadjusted 5% level.
- **Multiple testing**: wherever a table has a `p_fdr` column, its p-values
  are Benjamini-Hochberg adjusted (false discovery rate) over every test in
  that table. Each script section below says what *m*, the number of tests,
  is. Tests that are skipped for lack of data are not counted.

## Caveats that apply to every script

- **Overlapping forward windows.** A forward return over *h* sessions is
  computed for every day, so consecutive rows share *h - 1* days of the same
  price path. Rows for different tickers on the same date share the market's
  move. The t-tests, z-tests, Mann-Whitney tests and Spearman p-values all
  assume independent observations, so they overstate significance, by a
  wide margin for long horizons and large cross-sections. Treat p-values as a
  ranking device, not as error rates.
- **Multiple testing beyond the adjusted table.** BH controls the false
  discovery rate within one table. It does not account for the other tables,
  horizons, targets, parameter settings or date ranges that were tried
  before a result was looked at.
- **In-sample thresholds.** Decile cut-offs, quantile bins and the band-width
  split are computed on the same data that is then used to judge them. No
  script holds out data or walks forward.
- **Close-to-close entry.** Returns start at the close of the day the signal
  is observed. That assumes the signal is known before that close and that a
  trade fills at it, with no spread, slippage, fees or borrow costs. For
  option data captured after the close, the realistic entry is the next
  session, and short horizons are optimistic.
- **Survivorship.** The universe is whatever the database holds today:
  tickers that were delisted and removed are missing, and stock
  relative-strength metrics are computed for the current ETF memberships.
  Sectors come from today's `dim_stock`.
- **Regimes.** Market regimes come from `logic.backtest.classify_regimes`
  applied to the configured benchmark. Its thresholds are hand-chosen, and a
  regime is a handful of long episodes, so per-regime results rest on few
  independent periods.

## signal_analysis.py

**Question.** Do underlying-level option positioning readings (net customer
premium, delta and gamma exposure, option volume, put/call ratio, vanna
exposure) line up with the underlying's forward returns?

**Data.** `underlying_data` (`flownet`, `dxoi`, `gxoi`, `option_volume`,
`put_call_ratio`, `vannaxoi`) for every symbol (or `--symbols`), and daily
closes from `fact_commons_ohlc` and `fact_index_ohlc`.

**Method.**
1. Per symbol, compute day-over-day percentage changes of `dxoi`, `gxoi` and
   the put/call ratio (relative to the absolute previous value), option
   volume over the mean of the prior 20 sessions, and decile (flownet) and
   quintile (nonzero vannaxoi) ranks within the symbol's own history.
2. Join to forward returns at each horizon in `--horizons` (default 1, 5, 10
   sessions) on (symbol, date).
3. Bin each row by eight signals: flownet decile; dxoi change (±5%, ±20%
   edges); the dashboard's gamma-exposure display buckets; volume ratio
   (0.8, 1.2, 1.5, 2.0); put/call ratio change (±10%, ±30%, changes beyond
   ±200% dropped); vanna quintile; volume spike × flownet regime; gamma
   exposure change (±10%, ±50%, changes beyond ±500% dropped).
4. For each bin and horizon: row count, mean forward return, share positive,
   and a Welch t-test of the bin's returns against all other rows with a
   value for that signal (bins or complements under `--min-bin-n` rows are
   not tested).
5. BH adjustment over every tested (signal, bin, horizon) cell: *m* is the
   number of non-missing p-values in the output.

**Outputs.** `signal_bins.csv`: one row per (signal, bin) with
`mean_fwd_<h>_bps`, `pct_positive_<h>`, `p_<h>` and `p_fdr_<h>`.

**Specific caveats.**
- The decile and quintile edges use each symbol's whole sample, including
  dates after the row: a mild look-ahead in the bin assignment.
- Bin edges were chosen by hand; the gamma-exposure buckets are in the data
  feed's units and only their sign has an interpretation.
- Every bin is tested against the rest of the same sample, and the eight
  signals partition the same rows, so tests within a signal are not
  independent of each other.
- The returns are raw, not adjusted for the market or the benchmark.

## ta_signal_analysis.py

**Question.** Which of the backtester's technical and relative-strength
indicators have a monotone relation with forward (optionally
benchmark-adjusted) stock returns, and does that relation hold across
regimes, sectors and time? Which pairs of extreme readings stand out?

**Data.** The backtester's loaders in `queries.backtest_queries`:
`load_ohlcv_universe` (all stocks), `load_metrics_universe` (relative-strength
rank and z-score, one row per date and ticker) and `load_benchmark_ohlcv`
(`BENCHMARK_TICKER`); sectors from `dim_stock`.

**Method.**
1. Restrict to `--start`/`--end` and keep the `--universe-size` stocks
   (default 2000) with the highest median dollar volume over the last 63
   sessions of that window.
2. Build the backtester's panel with `logic.backtest.build_backtest_dataframe`:
   every filterable indicator (RSI, momentum, MA ratios, MACD histogram,
   volume z-score, ADR, Weinstein stage, chart-pattern and event flags,
   support/resistance distances), relative-strength impulse, and forward
   returns at 5, 10, 20 and 63 sessions. These are the same definitions the
   backtester and the screener use.
3. Benchmark-adjusted target: `resid_fwd_h = fwd_h - beta * bench_fwd_h`, with
   beta the rolling 63-session regression slope of the stock's daily log
   returns on the benchmark's (at least 30 sessions; beta = 1 before that).
4. Add features the backtester does not filter on: distance from the 52-week
   high and low, Bollinger %B and width, 14-session ATR as a percentage of
   price.
5. Label each date's regime, each ticker's sector, and each row's half of the
   sample (split at the median date).
6. Univariate scan of 25 signals against `--target` (default `resid_fwd_10`);
   *m* is the number of signals with at least `--min-n` rows.
7. Replication of the top `--top-k-for-pairs` signals (default 8) per regime,
   sector and time half.
8. Event flags (RSI crossing 30/70, MACD crosses, a 20-session breakout on
   volume, regaining or losing the 200-session MA, each within the last 3
   sessions): Welch t-test of the target when the flag is set against when it
   is not, overall (*m* = events tested) and per regime (*m* = (regime, event)
   cells tested).
9. Pairs: for every pair of top signals and each of the four top/bottom
   decile combinations, a z-test of the mean target when both are in their
   decile against the full-sample mean. *m* is the number of pair tests run
   (at most 112 for eight signals). A pair is kept when `p_fdr < 0.05`, its
   hit rate is at least `--min-hit-rate` (default 0.25), and both signals
   replicate (significant, consistent sign) in at least
   `--min-regimes-replicate` regimes and `--min-time-halves-replicate` halves.
10. `--export-presets` (off by default) saves each kept pair whose two
    signals both have a backtester filter as a preset (`logic.preset`), with
    the in-sample 10th/90th percentile as the threshold. Its stats come from
    running all exported pairs through one
    `logic.backtest.evaluate_all_strategies` call on the same panel, and a
    `discovery` block records the pair-scan statistics.
    Presets go to `--preset-dir` or `PRESET_DIR`.

Signals without a backtester filter, which are reported but never exported:
`bb_pctb`, `bb_width`, `atr_over_price`, `dist_52w_high`, `dist_52w_low`,
`rs_impulse_5`, `rs_impulse_42` and `weinstein_stage_num`. The exportable set
is derived from `logic.backtest.FILTER_TO_COLUMN`, so every exported filter
replays in the backtester and in the chart's preset overlay.

**Outputs** (`<stem>` is `universe<N>_<target>`): `ta_univariate_<stem>.csv`,
`ta_per_regime_<stem>.csv`, `ta_per_sector_<stem>.csv`,
`ta_per_time_<stem>.csv`, `ta_events_overall_<stem>.csv`,
`ta_events_per_regime_<stem>.csv`, `ta_pairs_<stem>.csv`, and preset JSON
files when exporting.

**Specific caveats.**
- The universe is chosen by liquidity at the end of the window, which is
  information the earlier dates did not have.
- Residualization removes only the benchmark's move scaled by a trailing
  beta: no sector, size or other factor adjustment. Beta is estimated from
  trailing data only, but a 63-session beta is noisy.
- The time-half check is a split of the same in-sample window, not an
  out-of-sample test.
- An exported preset's thresholds and its backtest stats come from the data
  the pair was discovered on. Its `p_value_adj` is BH-adjusted over the
  exported pairs and their horizons (*m* ≤ 4 × pairs exported), not over the
  search that found them, so it is not a confirmation. Re-test a preset on a later period before reading
  anything into it.

## option_signal_analysis.py

**Question.** What, measured on an option contract or its underlying at day
*t*, is associated with the contract's forward return relative to similar
contracts of the same underlying on the same day?

**Data.** `option_chain_data` (price, open interest, volume, implied
volatility, greeks, customer buy/sell volume), `underlying_data` (IV term
structure, put/call ratio, flow aggregates) and daily closes of the
underlyings. Universe: by default, the 5 symbols (`--universe-size`) with the
highest median option volume over the last 20 sessions. History: the last
120 sessions (`--sessions`) of chain data.

**Method.**
1. Over each contract's full series: the price `--horizon` observations later
   (default 5), lags of IV, vanna and open interest at 1 and 5 observations,
   and the mean and standard deviation of volume and IV over the prior 20
   observations. Liquidity filters apply afterwards: price above
   `--min-price`, open interest at least `--min-oi`, at least `--min-dte`
   days to expiry, IV and delta present.
2. Keep a row only if its forward option price comes from the same session as
   the underlying's close `--horizon` sessions later (a contract missing from
   some chains would otherwise stretch the horizon).
3. Targets: the forward log return of the option, and of the underlying.
   Bucket each row by symbol, days to expiry (0-14, 15-45, 46-120, >120),
   moneyness (strike/spot - 1, edges ±3%, ±10%) and right. Subtract the
   mean return of its (bucket, date) group from the option's return. The bucket
   includes the symbol, so all rows of a group share the same underlying
   move, and demeaning removes the part of it the group's contracts have in
   common. Winsorize the result (`residual_ret`) at its 1st and 99th
   percentiles. `--target fwd_opt_ret` uses the raw return instead.
4. Signals (33): IV changes and z-score; net customer flow, absolute and
   relative to volume and open interest; volume relative to open interest and
   to its own average; vanna and open-interest changes; per expiry and right:
   skew (IV of the strike nearest ±5% moneyness minus the at-the-money IV),
   its 5-observation change, open-interest growth, the contract's IV relative
   to at-the-money and to the 5% out-of-the-money strike, and its share of
   the expiry's open interest and volume; underlying-level IV term slope and
   changes, put/call ratio, flow measures, and the underlying's option volume
   against its prior 20-session mean (`und_opt_vol_vs_avg`, (volume - mean) /
   mean; the contract-level `vol_vs_avg` is the same for the contract's
   volume).
5. A structural scan (IV level, greeks, days to expiry, moneyness, open
   interest, call/put) as a sanity check: the bucketing is meant to absorb
   these, so strong correlations there point at the demeaning, not at
   a signal.
6. Univariate scan (*m* = signals tested), replication of the top
   `--top-k-for-pairs` (default 6) per symbol and per bucket, and pairs of top
   signals both in their top decile (*m* = pairs tested, at most 15).

**Outputs** (`<stem>` is the symbols, or `<N>symbols`, plus horizon and
target): `structural_<stem>.csv`, `univariate_<stem>.csv`,
`per_ticker_<stem>.csv`, `replication_<stem>.csv`, `pairwise_<stem>.csv`.

**Specific caveats.**
- The demeaned return compares a contract with its peers over the same
  window, so it is a decomposition of what happened, not a tradable return.
  A signal that predicts it still needs a way to hedge the underlying's move
  and the peers' return.
- Demeaning removes only the move the contracts of a bucket share. Within a
  bucket, delta and gamma still vary with strike and days to expiry, so part
  of the underlying's move stays in the target; a large move leaves more.
  Winsorizing changes the target's tails.
- A (bucket, date) group with one contract demeans to zero, so it adds rows
  but no information.
- Returns use `price`, the contract's last trade price, with no bid-ask
  spread. On an illiquid contract the last trade can be stale, and for cheap
  contracts the spread can dwarf any effect.
- Rows are far from independent: all strikes and expiries of a symbol move
  together on a given day, underlying-level signals repeat across every
  contract of a symbol-date, and the same contract appears on consecutive
  days with overlapping windows. The effective sample is much smaller than
  the row count.
- The default universe is picked by recent option volume, which is
  information the earlier dates did not have.
- Many feeds report open interest as of the previous session's settlement.
  `docs/DATA_CONTRACT.md` does not fix that timing, so the open-interest
  signals may describe the day before the row's date.

## trend_precursor_analysis.py

**Question.** What did sector and thematic ETFs look like at the start of the
sustained trends they went on to make, compared with ordinary days?

**Data.** Daily bars from `fact_index_ohlc` and relative-strength rank and
z-score from `fact_etf_metrics` against `BENCHMARK_TICKER`, for every ETF that
has those metrics (or `--etfs`). The benchmark itself has none, so it is not
included.

**Method.**
1. Indicators per ETF, using the package's definitions: RSI(14) from
   `logic.technical.compute_rsi` (Wilder smoothing, the one RSI used
   everywhere in the dashboard), momentum over 10, 20 and 63 sessions, close
   relative to the 50- and 200-session moving averages, volume relative to its
   20-session mean, ADR%, Bollinger band width (as in `logic.trend_signals`),
   the 5-session RSI change, and the relative-strength rank change over 5, 10
   and 20 sessions.
2. Trends: starting `--lookback` bars in (default 20), scan each bar. From a
   bar, an uptrend runs to the bar with the largest gain of at least
   `--min-move` percent (default 15) between `--min-duration` and
   `--max-duration` bars later (default 20-120), provided the price never
   falls more than `--max-adverse` percent (default 8) below its running peak
   on the way. The scan resumes after the trend ends. Downtrends mirror this
   (a bounce from the running low ends the search); the two are scanned
   independently.
3. Profiles: the indicators on each trend's first bar, and means or
   least-squares slopes of RSI, momentum, relative-strength rank, volume ratio
   and band width over the `--lookback` bars ending there.
4. Baseline: up to `--baseline-samples` (default 2000) random bars
   (`--seed`, default 42) that lie more than `--lookback` bars from any trend
   start of the same ETF.
5. Medians at trend start for uptrends, downtrends and the baseline, with
   Mann-Whitney tests of each against the baseline. *m* is the number of
   tests (two per indicator, 15 indicators).
6. Quantiles of selected indicators at trend start, and a split of the trends
   at the baseline's median band width into wide-band and narrow-band starts,
   with medians of every profile column (including the trends' move and
   duration) and the share of starts below, near or above the 200-session MA.
7. `--focus` lists the trends and profiles of chosen ETFs.

**Outputs.** `trend_trends.csv`, `trend_profiles.csv`, `trend_baseline.csv`,
`trend_comparison.csv`, `trend_lookback.csv`, `trend_distribution.csv`,
`trend_regime_split.csv`.

**Specific caveats.**
- **The trends are selected after the fact.** A trend's start, end, move and
  duration are known only from the prices that follow. The profiles describe
  conditions at the start of trends that did happen. They do not give the
  probability that a trend follows a condition, nor the return of acting on
  it.
- **The start is a local extreme by construction.** An uptrend's first bar is
  the first from which a qualifying rise follows, which tends to come right
  after a decline. Low RSI and negative short-term momentum at uptrend starts
  (and the mirror for downtrends) are partly produced by the definition
  itself.
- **This is not a test of the dashboard's trend scans.** The washout and
  coiled-spring scans in `logic/trend_signals.py` are fixed rules whose
  thresholds were set by hand. This script does not fit those thresholds and
  does not measure what follows when a scan fires. The wide/narrow-band
  split, and the move and duration medians reported for it, describe the
  selected trends only; they are not the forward returns of either scan.
  Answering that question needs a forward test of the scan rules, which
  neither this script nor the backtester (it has no Bollinger-width filter)
  performs.
- Baseline bars may fall inside trends; only the neighborhood of trend
  starts is excluded.
- ETFs overlap in holdings (industry ETFs sit inside sector ETFs) and move
  with the market, so their trends are not independent events; the number
  of distinct episodes is much smaller than the number of trends.
