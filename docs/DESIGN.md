# Design

This document defines what the dashboard computes, lists the invariants the
code relies on, and records the main design decisions, each with its reason.
Module names are relative to `src/stock_dashboard/`.

## Terms

- **Bar, session.** One daily row of a ticker: a trading session. "N-day"
  or "Nd" in a UI label (a 20-day MA, "Momentum 20d %") means N bars, not N
  calendar days.
- **Relative strength (RS).** rs-pipeline's measure of a stock's performance
  against an ETF it belongs to (and of an ETF's against the benchmark).
  `rs_rank_index` ranks a stock within one ETF, `rs_rank_overall` across the
  whole universe, and `rs_z_score` is the stock's RS against that ETF as a
  z-score against its own trailing mean and standard deviation
  ([rs-pipeline's definition](https://github.com/scottmcghee/rs-pipeline/blob/main/docs/DESIGN.md)).
  Ranks are stored as fractions 0–1 and shown as percentages in the UI;
  preset files keep the fractions, so `"rs_rank_overall_min": 0.8` means the
  top fifth, and a value above 1 is rejected when the preset loads.
- **RS impulse** names two different things. On the ETF page, the *ETF RS
  impulse* is the change in an ETF's rank among the ETFs over 5, 10 or 20
  sessions (shown in percentage points), z-scored against that ETF's own last
  252 sessions. In the backtester, the *stock RS impulse* `rs_impulse_N` is
  the change in a stock's `rs_rank_overall` over N = 5, 10, 20 or 42 bars, as
  a fraction (0.10 = ten rank points).
- **Option exposures** (the `*xoi` columns), from the dealers' side under the
  convention in [DATA_CONTRACT.md](DATA_CONTRACT.md#conventions): *gxoi*
  (GEX, gamma exposure), *dxoi* (delta exposure), *vannaxoi* (vanna
  exposure: how dealer delta changes when implied volatility moves).
- **flownet.** Call net customer premium minus put net customer premium, in
  dollars: positive is bullish customer flow, negative bearish.
- **Weinstein stage.** Stan Weinstein's four stages (1 basing, 2 advancing,
  3 topping, 4 declining), approximated from the 50- and 150-bar moving
  averages, their 20-bar change, the close's position and the last bar's
  range (`logic/weinstein.py`). A bar is "Undetermined" until the 150-bar MA
  and its 20-bar change are defined, i.e. for a ticker's first 169 bars, and on
  any later bar that matches none of the stage rules.
- **Max pain** and **surface richness**: see [Option contract
  page](#option-contract-page).

## Invariants

1. **"Latest" is the newest data date, never the wall clock.** A query that
   wants the latest session compares with `MAX(date)` (or
   `MAX(collection_date)`) of the table it reads. Taken across all symbols, a
   symbol with no row on that date has no current reading and drops out of
   the latest-day screens. *Reason:* the stocks database is filled by a batch
   job, and the demo's data ends at its last generated session. Anchoring on
   the data means a page shows the same thing whether it is opened on a
   trading day, over a weekend or a month later, and a late pipeline run
   shows yesterday's data rather than nothing. The one place that uses the
   calendar is the default date range of the chart pages, which is a UI
   default the user can change. A test scans every query module (and
   `research/common.py`) for the wall clock. [`test_no_query_uses_wall_clock`,
   `test_history_query_anchors_on_latest_sessions`, `test_latest_dates_agree`]

2. **Look-backs are counted in trading bars, and each consumer fetches what
   its indicators need.** Calendar windows hold a varying number of sessions
   (holidays, halts, a ticker's own gaps), so an indicator could be undefined
   at the latest bar without any error. Instead, each indicator module
   declares its requirement, and the queries fetch that many rows per ticker
   with `fetch_ohlcv(last_bars=N)` (each ticker's last `N` rows) or
   `fetch_ohlcv(start=..., warmup_bars=N)` (the table's `N` sessions before
   `start`). [`test_last_bars_counts_rows_per_ticker`,
   `test_warmup_reaches_back_n_sessions_before_start`,
   `test_volume_stats_anchor_on_bars_not_calendar`,
   `test_filters_have_values_at_required_bars`,
   `test_screener_lookback_covers_every_consumer`,
   `test_chart_warmup_covers_the_overlay_indicators`]

   | Consumer | Constant | Bars (daily) |
   |---|---|---|
   | Screener TA filters | `technical.ta_required_bars()`: the longest of RSI(14)+1, MACD 26+9+3, ADR 14, momentum+1, the ratio-to-MA period, the MA-touch period and the 126-bar pivot window + 5 | 200 by default; ×5 or ×21 daily bars, plus one resampled bar, on weekly or monthly bars |
   | Event flags | `events.EVENT_REQUIRED_BARS` = 200-bar MA + 10 | 210 |
   | Chart patterns | `patterns.PATTERN_REQUIRED_BARS`, the longest detector window | 75 |
   | Screener, all of the above | `lookback.screener_lookback_bars()` | 210 |
   | Trend scans | `trend_signals.TREND_REQUIRED_BARS` (the 200-bar MA) | 200 |
   | Weinstein stage filter | `weinstein.WEINSTEIN_LOOKBACK_BARS` (150-bar MA, 20-bar trend, margin) | 200 |
   | Single-stock overlay warm-up | `lookback.chart_warmup_bars()`: the screener's need at the longest momentum and MA periods, the Weinstein MA and trend, and the longest RS impulse + 1 | 210 |
   | Screener volume statistics | `stocks_queries.VOLUME_STATS_BARS` | 252 |
   | Option scans | `option_signals.SIGNAL_HISTORY_SESSIONS` = max(63, 20 + 1) | 63 sessions |
   | 52-week highs and lows | `chart_helpers.FIFTY_TWO_WEEK_BARS` as warm-up; 200 bars minimum | 252 |

   RSI and MACD are exponential averages, so their value depends very
   slightly on where the history starts. At these depths the starting
   point's weight has decayed below 10⁻⁶.

3. **One row per ticker.** rs-pipeline stores a metrics row per (date, stock,
   ETF), and a stock belongs to several ETFs (the ETFs that hold it, plus its
   sector and industry ETFs). Every list of stocks shows one row per ticker:
   `stocks_queries.collapse_etf_rows` keeps the row of the ETF in which the
   stock ranks highest by `rs_rank_index` (a missing rank loses; ties go to
   the alphabetically first ETF) and lists all of its ETFs in an `etfs`
   column. The screener's ETF-rank bounds apply to that row. When the
   screener is limited to one ETF, that ETF's row is kept instead.
   *Reason:* `rs_rank_overall` is the same in every row of a stock, so the
   only choice is which ETF's context to show. "Where is this stock
   strongest" is the most useful one for a momentum screen, and the
   tie-breaks make it deterministic. [`test_keeps_best_ranked_etf_and_lists_all`,
   `test_missing_rank_loses_to_ranked_row`, `test_tie_broken_by_etf_ticker`,
   `test_rank_filter_applies_to_best_etf_row`,
   `test_etf_filter_keeps_that_etfs_row`, `test_one_row_per_symbol`,
   `test_stock_screener_returns_one_row_per_ticker`]

   The backtester and the chart overlay need one value per (date, ticker)
   for the metric filters instead. They average the ETF rows:
   `rs_rank_overall` is unchanged by that, and `rs_z_score` becomes the mean
   across the stock's ETFs. A backtest limited to one ETF uses only that
   ETF's rows, and so does the overlay for a preset exported from such a run
   (its `etf` key). [`test_metrics_aggregates_per_date_and_ticker`,
   `test_rs_z_score_averages_etf_rows_like_backtest`,
   `test_etf_selects_that_etfs_metric_rows`]

4. **ETF metrics are read for the configured benchmark only.**
   `fact_etf_metrics` is keyed by (date, ETF, `ref_ticker`), so one database
   can hold ETF metrics against several benchmarks. Every read filters on
   `ref_ticker = BENCHMARK_TICKER`; without that filter the ETF grid could
   show an ETF twice, and "latest" could be another benchmark's date.
   [`test_latest_date_and_rows_use_benchmark`,
   `test_etf_metrics_filter_benchmark`, `test_rank_history_filters_benchmark`,
   `test_filters_benchmark`, `test_etf_grid_has_one_row_per_etf`]

5. **One definition of each indicator.** `logic/technical.py` defines the
   shared indicators once, and the screener, the trend scans, the event
   flags, the backtester and the preset overlay call them:

   | Indicator | Definition |
   |---|---|
   | RSI(14) | Wilder's: gains and losses smoothed with `ewm(alpha=1/14, adjust=False)`, `100 − 100 / (1 + gain/loss)`; 100 with only gains, undefined when flat |
   | Volume z-score | `(v − mean) / std` over the trailing 20 sessions, the current bar included; undefined when the volume is constant |
   | ADR% (14) | Mean over 14 bars of `(high − low) / close × 100` |
   | Momentum (n) | `(close / close[n bars ago] − 1) × 100` |
   | Ratio to MA (n) | `close / SMA(n)` |
   | MACD | EMA(12) − EMA(26), signal EMA(9), histogram = MACD − signal |
   | Pivot proximity (w) | `close / max(close)` − 1 and `close / min(close)` − 1, over the `w` bars ending 5 bars ago |

   *Reason:* a backtest only means something if the filter it tested is the
   filter the screener and the chart apply. The backtester's
   `compute_ticker_indicators` and `FILTER_TO_COLUMN` are also the overlay's,
   so a preset's filters pass on the same bars of a chart as in the backtest;
   the overlay also applies the preset's regime and ETF (see
   [Presets](#presets)).
   [`test_wilder_smoothing`, `test_window_includes_current_bar`,
   `test_uses_shared_volume_z`, `test_uses_shared_rsi_and_adr_definitions`,
   `test_indicators_use_the_package_rsi`, `test_same_mask_as_backtest`,
   `test_subset_matches_full_computation`]

   Two screener filters are deliberately different, and a preset loaded into
   the screener is therefore a starting point, not an exact replay: the
   screener's *Volume Z-Score* is the latest volume against the last 252
   bars (computed in SQL, `stocks_queries.VOLUME_STATS_BARS`), and its *RS
   Z-Score* is the kept ETF row's (invariant 3), not the mean across ETFs.
   The screener also filters the latest session only and does not apply a
   preset's regime (see [Presets](#presets)). When a loaded preset uses
   either filter, the screener says so under the preset.
   [`test_screener_differences_name_the_filters_it_measures_differently`]

## Presets

A preset is a JSON file: a strategy from the backtester, or from
`research/ta_signal_analysis.py`, that other pages can load.

```json
{
  "name": "Sample: RS leaders in an uptrend",
  "filters": [
    {"name": "rs_rank_overall_min", "value": 0.8},
    {"name": "ratio_to_200ma_min", "value": 1.0}
  ],
  "regime": null,
  "stats": {},
  "description": "Stocks in the top fifth of the universe by relative strength that trade above their 200-day moving average."
}
```

Filter names are the backtester's (`backtest.FILTER_TO_COLUMN`). `regime` is
null or one regime name. A preset saved from a backtest limited to one ETF
also records it as `"etf"`. A preset saved from the backtester fills `stats`
with one entry per horizon (`"T+20"`, ...) holding `mean_return` in percent,
`win_rate` as a fraction, `sharpe` (mean / SD of the per-signal returns, not
annualized; the UI labels it Mean/SD) and `p_value_adj` (BH-adjusted). A
research export adds a `discovery` block describing how the pair was found.

- **Two directories.** Presets are listed from `PRESET_DIR` (user presets,
  where the backtester saves) and from the samples shipped in `presets/`. A
  user preset with the same file name shadows a sample. The samples have
  empty `stats`: they illustrate the format and are not backtest results.
- **Unreadable files are skipped.** A file that isn't valid preset JSON is
  left out of the list with a logged warning; one that breaks after the list
  was built shows an error where it is loaded. So does a value outside the
  range of the screener input it fills (an RSI above 100, or a rank written
  as a percentage, 80, instead of the fraction 0.8), and a numeric value
  that is not finite (JSON's `Infinity` or `NaN`).
  [`test_list_skips_unreadable_files_with_a_warning`,
  `test_load_preset_shows_an_error_for_an_unreadable_file`,
  `test_value_outside_its_widget_range_raises`, `test_non_finite_value_raises`,
  `test_stock_screener_preset_value_outside_a_widget_range`]
- **Where they are used.** The single-stock page draws a preset as signal
  markers (`logic/preset_signal.py`); the marker direction comes from the
  sign of the preset's mean return at T+20 (else T+10, T+5, T+63), and is
  up (bullish) when the preset has no stats, as with the samples. The
  screener fills its widgets from the filters it has inputs for
  (`preset.PRESET_FILTER_MAP`, and the pattern filters) and lists the rest
  as not applied. It has one momentum and one ratio-to-MA period, so a
  preset's first period is mapped and filters on a second period are listed
  as not applied rather than overwriting it. RSI and ADR thresholds keep
  their decimals, and rank fractions convert to percentages and back
  exactly. [`test_second_momentum_period_is_not_applied`,
  `test_fractional_rsi_and_adr_are_kept`, `test_rank_round_trips_exactly`]
- **Only filters that can be replayed.** The overlay evaluates a filter
  through `FILTER_TO_COLUMN`. A filter it can't evaluate (an unknown name,
  or a metric filter with no metrics loaded) is reported on the page and
  matches no bars, which is also how the backtest treats it. The research
  script exports only pairs whose filters are in `FILTER_TO_COLUMN`, so
  every exported preset can be replayed.
  [`test_unknown_filter_is_reported_and_matches_nothing`,
  `test_every_backtest_filter_is_supported_with_metrics`,
  `test_every_exportable_filter_replays_in_the_chart_overlay`,
  `test_export_writes_replayable_presets_and_skips_unexportable_pairs`]
- **The overlay replays the regime and the ETF; the screener can't.** The
  overlay keeps only bars in the preset's regime, classified from the
  benchmark's full history as in the backtest; if the benchmark has no bars
  it says the regime couldn't be applied and marks nothing. For a preset
  with an `etf`, it reads that ETF's metric rows. The screener filters the
  latest session only, so it states the preset's regime and the benchmark's
  regime on its latest date, and says plainly when they differ.
  [`test_regime_masks_bars_outside_it`,
  `test_regime_without_regime_series_matches_nothing`,
  `test_single_stock_overlay_applies_the_preset_regime_and_etf`,
  `test_differing_regime_is_said_plainly`]
- **A neutral summary.** The screener's one-line summary quotes the stats at
  a fixed horizon (T+20, else T+10, T+5, T+63) and says they are in-sample
  with optimistic p-values; it doesn't grade or pick the best-looking
  horizon. [`test_quotes_t20_not_the_best_looking_horizon`,
  `test_neutral_wording`]

## Option scans

`logic/option_signals.py` holds the rules, over one fetch of the last 63
sessions of `underlying_data`. Each scan looks at the latest session only
and compares a symbol with **its own** recent history, so a large and a small
underlying are judged on the same scale.

| Scan | Rule (defaults) | Rationale | Strength |
|---|---|---|---|
| Vanna extreme | Latest `vannaxoi` in the top or bottom 20% of the symbol's last 63 sessions | Vanna exposure says how dealer hedges change when implied volatility moves; an extreme marks unusual hedging sensitivity | \|percent rank − 0.5\| |
| Flownet extreme | Latest `flownet` in the bottom 10% of the symbol's last 63 sessions | Customer flow unusually bearish for this symbol (calls sold or puts bought; the value itself can be positive), read contrarian | \|percent rank − 0.5\| |
| Volume spike | Option volume ≥ 2 × its mean over the previous 20 sessions (the latest excluded); labeled major from ≥ 2× and extreme from ≥ 3× | Unusual activity | the multiple |
| dxoi change | `(dxoi − previous) / abs(previous)` at least 20% in either direction, against the previous session | A sudden shift in dealer delta | \|% change\| |
| Negative GEX | Latest `gxoi` < 0 | Dealers short gamma hedge with the move, which tends to amplify volatility | 1 − percent rank of `gxoi` in the last 63 sessions |

The percentile is SQL's `PERCENT_RANK`: the number of window values below
the latest one, divided by (n − 1), and 0 for a single value. A symbol needs
a value on the latest session, and the volume spike needs all 20 prior
sessions. The GEX buckets shown beside `gxoi` (0 / 5,000 / 15,000 / 50,000)
are presentation only; only the sign has a fixed meaning (see the
[data contract](DATA_CONTRACT.md#conventions)). [`TestLatestPercentRank`,
`test_vanna_high_and_low_tails`, `test_flownet_bottom_tail_only`,
`test_ratio_against_prior_sessions`, `test_needs_full_average_window`,
`test_increase_and_decrease`, `test_only_negative_symbols`,
`test_history_window_covers_every_scan`, `TestStrength`]

The trend scans (`logic/trend_signals.py`) are fixed rules at the latest bar:
*washout bull* (RSI ≤ 40, 20-day momentum ≤ −5%, close ≥ 6% below the 50-day
MA, 20-day Bollinger width ≥ 12% of the middle band, volume at or above its
20-day mean), *washout bear* (the mirror image: RSI ≥ 60, momentum ≥ +5%,
close ≥ 6% above the 50-day MA, the same width, without the volume
condition), and *coiled spring* (Bollinger width ≤ 7%, ADR ≤ 1.5%, close
within ±15% of the 200-day MA). Their thresholds are hand-chosen starting
points, adjustable on the page. `scan_settings.ScanSettings()` is the one
source of these defaults: the page starts from it, the screener's scan
pre-filter runs with it, and the scan functions' own defaults equal it.
Their strengths are −momentum (washout bull), momentum (washout bear) and
−Bollinger width (coiled spring).
[`test_page_defaults_match_the_scan_function_defaults`,
`test_trend_scans_use_the_signal_scans_page_defaults`]

**The composite is a weighting choice.** Each scan gives every symbol it
flags a scale-free *strength* (the last column above), so a large and a
small underlying compare. Within one scan, a symbol's strength becomes its
percentile rank among the symbols that scan flags, in (0, 1]: the strongest
scores 1, ties share their average rank, a scan that flags a single symbol
gives it 1, and a missing strength ranks lowest (still above 0). A symbol's
composite is the sum of its ranks, weighted 2 for the two percentile scans
(vanna, flownet) and 1 for the rest (`signal_scanner.TIER_WEIGHTS`). The
weights are not fitted. Every hit adds, so a symbol that several scans agree
on is not pushed down by a mediocre strength in one of them (several weak
hits can still score below one strong hit); equal scores
are ordered by the *Scans Hit* column, which shows how many scans agree. The
page prints this rule next to the table. [`test_tier_1_weighted_higher`,
`test_scores_strength_not_underlying_size`, `test_confluence_adds_up`,
`test_agreement_outranks_a_single_strong_hit`]

**Exposure sign.** All `*xoi` columns take the dealers' side under the
assumption that dealers are long the calls and short the puts in open
interest; see [DATA_CONTRACT.md](DATA_CONTRACT.md#conventions). The scans'
descriptions follow that convention, and a provider whose feed uses another
one must convert it.

## Graceful degradation

Option data is optional. `db.option_data_available()` checks that the option
database is reachable and has the three contract tables, and
`ui.option_guard.option_data_ready()` also requires rows in
`underlying_data`; the answer is cached for 60 s. There is no flag to set:
the pages detect the situation.

- Pages built on option data (Option Data, Option Screener, Option Contract)
  call `require_option_data()`, which explains what is missing and stops the
  page.
- Pages that use it in part (Stock Screener, Signal Scans) call
  `note_missing_option_data()`, name what is hidden, and hide only that: the
  option pre-filters, option columns and option scans.
- The home page says the same once.

*Reason:* the stocks side is useful on its own, and someone running
rs-pipeline may have no option feed. A missing database, missing tables and
empty tables all look the same to the user.
[`test_option_pages_explain_missing_option_data`,
`test_partly_option_pages_render_without_option_data`,
`test_screener_hides_option_filters_without_option_data`,
`test_signal_scans_keep_trend_tabs_without_option_data`,
`test_app_home_explains_missing_option_data`]

The stocks database is required. `app.py` runs every page inside
`ui.errors.database_errors()`: an unreachable server and a database without
rs-pipeline's tables each get a message (for the latter, "run rs-pipeline
first") instead of a traceback. [`test_missing_table_says_to_run_rs_pipeline`,
`test_unreachable_database_names_the_server`,
`test_stocks_database_without_rs_pipeline_tables`]

## Option contract page

- **Max pain** is the strike at which the expiration's open interest would
  pay option holders the least at expiry: it minimizes
  Σ_calls oi × max(S − K, 0) + Σ_puts oi × max(K − S, 0) over the listed
  strikes S (`logic/max_pain.py`). It weights by raw open interest, as if
  customers held all of it long, and describes open interest, not a price
  target.
- **Surface richness** (`logic/richness.py`) takes the contract's implied
  volatility minus the at-the-money IV of its expiration and ranks it among
  the expiration's strikes of the same right ("surface rank"); 80% and above
  is shown as rich, 90% as very rich. Every strike is compared with the same
  ATM IV, so this is the contract's place on the smile: with skew, far
  out-of-the-money puts rank rich by construction. It is context, not a
  mispricing signal.

## Selections that follow the data

Streamlit keeps a keyed table's selected row across reruns, even when the
table's rows change, so a remembered row number could point at another row
or past the end. The selectable tables (ETF grid, technical z-scores,
backtest results) are keyed on their row ids (`ui.tables.selection_key`), so
a change of universe, filter or sort resets the selection, and the selected
index is clamped to the table (`selected_row`).
[`test_stale_selection_resets_when_the_rows_change`,
`test_technical_zscore_stale_selection_resets`,
`test_backtester_stale_selection_resets`]

## Backtester

`logic/backtest.py` replays filter combinations over the stored history and
asks, for each one, whether the mean forward return after a signal differs
from zero.

### Method

1. **Universe.** Every stock's full price history, or that of one ETF's
   members (every ticker that has metric rows for the ETF), with the per-date
   metrics of invariant 3. With an ETF selected, the metrics are that ETF's
   rows only, so `rs_z_score` is the stock's z-score within that ETF.
2. **Indicators.** For each ticker, from trailing data only: RSI(14),
   momentum (20, 63, 126), ratio to the 50- and 200-day MA, MACD histogram,
   volume z, ADR(14), Weinstein stage, the five chart patterns detected at
   every bar, pivot proximity (63 and 126 bars), the event flags, and the
   stock RS impulse (see [Terms](#terms)): the change in `rs_rank_overall`
   over 5, 10, 20 and 42 bars.
3. **Forward returns.** `fwd_h = close[t+h] / close[t] − 1` for h = 5, 10,
   20 and 63 bars of the same ticker.
4. **Regimes**, from the benchmark's closes (`classify_regimes`): *bull*
   above its 200-day SMA, *bear* below it, *volatile* when the 20-day
   annualized realized volatility is above the 80th percentile of the
   trailing 252 days (at least 63), and *correction* when the close is more
   than 10% below its 63-day high. Priority: correction, volatile, bear,
   bull. Undefined until the SMA is; a regime-conditioned strategy skips
   those dates. When the benchmark has no bars there are no regimes: a
   regime strategy then has no signals (`backtest.strategy_mask`), and the
   page warns and evaluates only the unconditional strategies.
   [`test_regime_strategy_without_regime_series_has_no_signals`,
   `test_backtester_without_benchmark_bars_runs_only_unconditional_strategies`]
5. **Strategies.** Every combination of 1 to *k* filters (*k* = 1–3 on the
   page) from a hand-chosen threshold grid (`FILTER_RANGES`), skipping
   repeated filter names and conflicting pairs (`rsi_min` with `rsi_max`,
   ...), times each selected regime. With the page's defaults (*k* = 1, no
   regime condition) that is 52 strategies; *k* = 2 with regimes none, bull
   and bear gives 3,954.
6. **Signals.** Every (date, ticker) row that passes all of a strategy's
   filters and, if it has one, is in its regime on that date.
7. **Statistics.** For each horizon with at least 10 non-missing returns:
   mean, median, win rate (share > 0), mean / standard deviation (a
   per-signal ratio, labeled Mean/SD, not annualized), and a two-sided
   one-sample t-test of mean = 0.
8. **Multiple testing.** Benjamini-Hochberg over every (strategy, horizon)
   p-value of the run; a horizon is significant when its adjusted p-value is
   below 0.05. By default the table lists only strategies with at least one
   significant horizon.

[`TestClassifyRegimes`, `test_forward_return_arithmetic`,
`test_min_signal_threshold`, `test_multiple_pvalues_adjusted`,
`test_adjusted_values_capped_at_one`, `test_no_conflicting_pairs`,
`test_matches_evaluate_strategy`,
`test_backtester_runs_a_small_backtest_and_saves_a_preset`]

### Limitations

These are properties of the method, stated so results are read correctly:

- **Signals overlap, so the t-test overstates significance.** A forward
  return is computed for every day a filter passes, and a filter usually
  passes on runs of consecutive days. Neighboring signals of one ticker
  share *h* − 1 days of the same price path, and signals of different
  tickers on one date share the market's move. The t-test assumes
  independent observations, so its p-values are too small, by a wide margin
  for long horizons and broad filters. BH corrects for the number of tests,
  not for this dependence. Use the p-values to rank strategies, not as error
  rates.
- **The number of tests depends on the run.** BH's *m* is the number of
  (strategy, horizon) pairs with enough signals *in this run*, so it
  depends on *k*, the selected regimes and the ETF filter. The same strategy can be
  significant in one run and not in another.
- **Close-to-close entry.** Returns start at the close of the signal bar,
  which assumes the signal is known before that close and the trade fills
  at it, with no spread, slippage or fees.
- **Regimes come from the benchmark alone.** A date's regime uses the
  benchmark's closes up to that date, so it has no look-ahead, but its
  thresholds are hand-chosen, and one benchmark labels every stock.
- **Survivorship.** The universe is what the stocks database holds today,
  and RS metrics use today's ETF memberships (see rs-pipeline's
  [limitations](https://github.com/scottmcghee/rs-pipeline#limitations)).
- **In-sample.** The threshold grid was chosen by hand, every strategy is
  judged on the whole history, and nothing is held out. A preset's stats are
  in-sample too.
- **The cumulative chart is not an equity curve.** It compounds each signal
  date's mean *h*-day return, date after date, although those returns
  overlap.

`research/` applies the same caveats to its studies; see
[research/README.md](../research/README.md).

### Memory

The full-history panel is the largest object the app builds. Loads are cached
as parquet (one file per database, benchmark and ETF filter), tickers are
stored as categories, and every float column, including the indicators and
forward returns, is float32. The page keeps the indicator panel in
`st.session_state`, so selecting a result doesn't rerun the backtest.
[`test_ohlcv_cached_after_first_load`, `test_key_changes_with_source_and_benchmark`,
`test_float32_dtype`]

## Security

- **No secrets in code.** Credentials come only from the environment, nothing
  secret has a default, and the password is kept out of `Settings`' `repr`.
  gitleaks runs in pre-commit and CI. [`test_config.py::test_defaults`,
  `test_password_hidden_from_repr`, `test_password_not_part_of_key`]
- **Parameterized SQL.** Values are always bound parameters; identifiers are
  constants or `psycopg2.sql.Identifier`. A column name the user picks
  (Option Data and Option Screener) must be one of `underlying_data`'s numeric columns, read
  from `information_schema`, before it is quoted. OHLCV column names are
  checked against a fixed list. [`test_rejects_unknown_column`,
  `test_rejects_injection`, `test_quotes_identifiers`,
  `test_unknown_column_rejected`]
- **Preset files stay in their directory.** A preset's file name is reduced
  to `[A-Za-z0-9._-]`, without leading dots and at most 80 characters, and a
  write through a symlink that leads elsewhere is refused. Listing and
  loading use only the files found in the two preset directories.
  [`test_refuses_path_traversal`, `test_refuses_to_write_through_a_symlink`,
  `test_traversal_is_not_a_preset`]
- **Exposure.** The dashboard has no login. The demo publishes its ports on
  127.0.0.1 only, and the container runs as a non-root user that can't
  modify the installed code. The dashboard only reads; a read-only database
  role is enough for it.

## Decisions

**Two databases, one contract.** Option data comes from whatever feed the
user has, so the dashboard defines the minimal tables it reads
([DATA_CONTRACT.md](DATA_CONTRACT.md)) instead of depending on one
provider's schema. Columns are found through `information_schema` where the
UI lists them, so a provider's extra numeric columns show up on Option Data
without code changes (charted without an average line if they have no
`_avg` partner), and on Option Screener if they can serve as a multiple (see
[DATA_CONTRACT.md](DATA_CONTRACT.md#extra-columns)). The column whitelist
is re-read every 60 s, so a column added while the app runs is picked up.
[`test_option_data_with_provider_extra_columns`,
`test_column_without_average_skips_the_average_query`,
`test_whitelist_is_reread_after_its_ttl`]

**Queries, logic and pages as separate layers.** SQL lives in `queries/`,
calculations in `logic/` as pure functions, Streamlit only in `views/` and
`ui/`. The logic is then unit-tested without a database or Streamlit, and
the queries are tested on their SQL text and parameters; the page tests
check the wiring.

**Demo data from rs-pipeline itself.** The demo and the integration tests
run rs-pipeline's own demo, built from a pinned commit, rather than a copy
of its schema with handmade rows. The dashboard is therefore tested against
the real schema and the real metric semantics, and a schema change in
rs-pipeline shows up when the pin moves. The option generator derives its
chains from those prices, so the stock and option pages describe the same
market.

**Planted events instead of tuned randomness.** Every option scan's demo hit
is planted deliberately (`demo.options.PlantedEvents`), on the latest session
except negative GEX, which holds on every session; the trend scans run on
rs-pipeline's synthetic prices as they are. So the demo and the page tests
show each option scan working without relying on chance.
