# Requirements

Each requirement names the tests that verify it, or says how it is met when
it isn't directly testable. Page tests are in
`tests/integration/test_pages.py` and run each page against the seeded demo.

## Functional

| ID | Requirement | Verified by |
|---|---|---|
| FR-1 | List every page on the home page and navigate to each. | `test_app_home_lists_every_page`, `test_app_switches_to_a_page`, `test_every_page_renders` |
| FR-2 | Single Stock: candlesticks and volume on daily, weekly or monthly bars, and the stock's relative strength within one of its ETFs (by default the one where it ranks highest) with its 150- and 252-day moving averages. | `test_single_stock_defaults_to_best_etf_and_draws_overlay`, `test_highest_rank_on_latest_date`, `test_weekly_reduces_rows`, `test_rs_figure_draws_the_moving_averages_present` |
| FR-3 | Draw a backtester preset on the single-stock chart as signal markers, on the bars where its filters pass as computed by the backtest, within the preset's regime and, for a preset from an ETF-filtered run, on that ETF's RS metrics (no signals for a stock outside that ETF); report conditions it can't evaluate. | `test_single_stock_defaults_to_best_etf_and_draws_overlay`, `test_single_stock_overlay_applies_the_preset_regime_and_etf`, `test_single_stock_regime_preset_without_benchmark_bars`, `test_same_mask_as_backtest`, `test_regime_masks_bars_outside_it`, `test_etf_selects_that_etfs_metric_rows`, `test_unknown_filter_is_reported_and_matches_nothing`, `test_metric_filter_without_metrics_is_reported`, `test_single_stock_preset_etf_the_stock_is_not_in`, `test_ticker_outside_the_preset_etf_has_no_signals` |
| FR-4 | Stock Screener: intersect the chosen pre-filters (ETF rank, option volume, signal scans, weekly expirations), search on RS rank, RS z-score, volume and Weinstein stage, then apply technical filters, recent events and chart patterns, and show one row per ticker with option columns when available. | `test_stock_screener_returns_one_row_per_ticker`, `test_stock_screener_signal_scan_prefilter_and_ta_filters`, `test_stock_screener_event_filter_finds_events`, `test_intersection_is_sorted`, `test_option_scans_share_one_history_and_intersect`, `test_metrics_gex_bucket_and_scan_labels`, `test_one_row_per_ticker_in_input_order` |
| FR-5 | Load a preset into the screener's inputs; list the preset filters the screener does not apply (no input, or a second momentum or ratio-to-MA period); state the preset's regime against the benchmark's latest regime; skip unreadable preset files and reject values outside the inputs' ranges, or not finite, with an error. | `test_stock_screener_loads_a_sample_preset`, `test_stock_screener_states_the_preset_regime_and_skipped_filters`, `test_unreadable_preset_is_skipped`, `test_basic_mapping`, `test_unmapped_filters`, `test_second_momentum_period_is_not_applied`, `test_fractional_rsi_and_adr_are_kept`, `test_rank_round_trips_exactly`, `test_sample_maps_onto_screener_widgets`, `test_value_outside_its_widget_range_raises`, `test_non_finite_value_raises`, `test_stock_screener_preset_value_outside_a_widget_range` |
| FR-6 | ETF Relative Strength: one row per ETF with its rank against the benchmark and its RS impulse z-scored against its own history, and charts for a selected (or clicked) ETF. | `test_etf_grid_has_one_row_per_etf_and_charts_the_selection`, `test_etf_grid_row_click_selects_the_etf`, `test_known_impulse_value`, `test_z_score_detects_unusual_move` |
| FR-7 | Option Data: a symbol's recent `underlying_data` rows, each numeric metric's latest value as a multiple of its 30-session average (the `underlying_data_avg` row of the same date; blank for signed metrics), and one numeric metric's history against its average (or alone, if it has none). | `test_option_data_shows_ratio_row_and_history`, `test_option_data_with_provider_extra_columns`, `test_ratios_from_same_date`, `test_empty_when_no_dated_pair`, `test_column_without_average_skips_the_average_query`, `test_latest_vs_average_row_blanks_signed_columns`, `test_column_ending_in_avg_is_read_as_given` |
| FR-8 | Option Screener: underlyings whose latest value of a chosen non-negative metric is at least N times its positive 30-session average (the `underlying_data_avg` row of the same date). | `test_option_screener_finds_the_volume_spike`, `test_same_day_average`, `test_get_option_average_columns_needs_an_avg_partner`, `test_contract_columns_only_non_negative_ones`, `test_provider_column_needs_avg_and_no_negative_latest_value`, `test_rejects_a_column_without_an_average` |
| FR-9 | Signal Scans: five option-positioning scans, three trend scans (on stocks or on ETFs) with the same defaults as the screener's scan pre-filter, each with its rule shown, and a weighted composite of scale-free strengths. | `test_signal_scans_find_the_planted_symbols`, `test_signal_scans_etf_mode`, `test_signal_scans_show_percentiles_in_percent`, `test_every_option_scan_finds_the_planted_symbols`, `test_combines_multiple_scans`, `test_tier_1_weighted_higher`, `test_scores_strength_not_underlying_size`, `test_page_defaults_match_the_scan_function_defaults`, `test_trend_scans_use_the_signal_scans_page_defaults`, `test_thresholds_are_filled_in`, `test_agreement_outranks_a_single_strong_hit`, `test_confluence_adds_up` |
| FR-10 | Technical Z-Scores: RSI, momentum and close/MA for a stock or ETF universe, z-scored across it, with indicator charts. | `test_technical_zscore_computes_and_charts`, `test_technical_zscore_stale_selection_resets`, `test_z_scores_centered`, `test_sorted_by_composite` |
| FR-11 | Backtester: generate filter combinations for the chosen regimes (only unconditional ones, with a warning, when the benchmark has no bars), measure forward returns at T+5, T+10, T+20 and T+63, test each against zero with a Benjamini-Hochberg correction, show the details of a chosen strategy, and save it as a preset that records an ETF filter. | `test_backtester_runs_a_small_backtest_and_saves_a_preset`, `test_backtester_without_benchmark_bars_runs_only_unconditional_strategies`, `test_backtester_stale_selection_resets`, `test_multiple_pvalues_adjusted`, `test_no_conflicting_pairs`, `test_regime_filter`, `test_regime_strategy_without_regime_series_has_no_signals`, `test_matches_evaluate_strategy`, `test_round_trip_through_save_and_parse`, `test_records_the_etf_of_an_etf_filtered_run` |
| FR-12 | 52-Week Highs & Lows: daily counts of new 52-week highs and lows, and the current lists. | `test_52_week_highs_lows`, `test_uptrend_produces_highs`, `test_downtrend_produces_lows`, `test_insufficient_lookback_returns_empty` |
| FR-13 | Option Contract: open interest by strike for an expiration, max pain over time, the contract's IV against the surface, and its OHLC with OI and volume. | `test_option_contract_picks_an_expiration_and_contract`, `test_canonical_example`, `test_per_day_output`, `test_percentile_for_richest_strike`, `test_latest_snapshot_in_one_query` |
| FR-14 | Export the symbols a page lists as a TradingView watchlist: known symbols only, in order, without duplicates, with exchange prefixes where known. | `test_keeps_known_symbols_in_order_with_prefixes`, `test_download_button_for_known_symbols`, `test_deduplicates_tickers`, `test_stocks_etfs_and_the_benchmark` |
| FR-15 | A self-contained demo with no market-data feed or API keys: `docker compose up` seeds both databases (stocks through rs-pipeline's demo, options through the generator) and serves the app; the generator plants an event for every option scan. The first build downloads images and packages. | CI `docker` job, `test_option_data_is_available`, `test_vanna_extremes`, `test_flownet_extreme`, `test_volume_spike`, `test_dxoi_change`, `test_option_screener_hit` |
| FR-16 | The research scripts run against the demo databases and write their reports, including when there is no data in range; `--help` describes every option and its default. | `test_signal_analysis`, `test_ta_signal_analysis`, `test_option_signal_analysis`, `test_trend_precursor_analysis`, `test_ta_signal_analysis_with_no_data_in_range`, `test_trend_precursor_analysis_without_trends`, `test_help_describes_every_option` |

## Nonfunctional

| ID | Requirement | How it is met / verified |
|---|---|---|
| NFR-1 **Correctness** | Each indicator has one definition, used by the screener, the scans, the backtester and the overlay; statistics match their textbook definitions. | `test_wilder_smoothing`, `test_window_includes_current_bar`, `test_uses_shared_volume_z`, `test_uses_shared_rsi_and_adr_definitions`, `test_matches_sql_percent_rank`, `test_forward_return_arithmetic`, `test_multiple_pvalues_adjusted`, `test_same_mask_as_backtest`, `test_greeks_match_finite_differences`, `test_put_call_parity_and_bounds`, `test_aggregates_match_chain`, `test_avg_is_trailing_mean_of_prior_sessions` |
| NFR-2 **"Latest" semantics** | "Latest" is the newest date in the table read, never the wall clock; a value is compared only with the average row of the same symbol and date; the stock and option tables agree on the latest date in the demo. | `test_no_query_uses_wall_clock`, `test_history_query_anchors_on_latest_sessions`, `test_latest_date_and_rows_use_benchmark`, `test_ratios_from_same_date`, `test_average_row_from_same_date`, `test_same_date_join_and_bound_tickers`, `test_latest_dates_agree` |
| NFR-3 **Enough history** | Indicator look-backs are counted in trading bars, and every consumer fetches at least what its indicators need, so values at the latest bar are defined. | `test_last_bars_counts_rows_per_ticker`, `test_warmup_reaches_back_n_sessions_before_start`, `test_volume_stats_anchor_on_bars_not_calendar`, `test_filters_have_values_at_required_bars`, `test_screener_lookback_covers_every_consumer`, `test_chart_warmup_covers_the_overlay_indicators`, `test_trend_scan_fetches_required_bars`, `test_weinstein_stage_fetches_lookback_bars`, `test_history_window_covers_every_scan`, `test_event_flags_defined_with_fetched_bars` |
| NFR-4 **Data freshness** | Query results are cached for 60 s, 5 min or 15 min (`CACHE_TTL_*`), and the backtester's disk cache for 24 h, keyed by database and benchmark. | `test_cache_ttl_constants`, `test_stale_cache_reloads`, `test_key_changes_with_source_and_benchmark` |
| NFR-5 **Graceful degradation** | Without the option database, its tables, or rows in them, the option pages explain what is missing and stop, the partly-option pages hide only their option parts, and nothing raises. An unreachable server or a stocks database without rs-pipeline's tables gets a message, not a traceback. | `test_option_pages_explain_missing_option_data`, `test_partly_option_pages_render_without_option_data`, `test_screener_hides_option_filters_without_option_data`, `test_stocks_database_without_rs_pipeline_tables`, `test_missing_table_says_to_run_rs_pipeline`, `test_unreachable_database_names_the_server`, `test_signal_scans_keep_trend_tabs_without_option_data`, `test_app_home_explains_missing_option_data`, `test_false_when_database_unreachable`, `test_false_when_tables_missing`, `test_needs_rows`, `test_stops_with_an_explanation`, `test_names_what_is_hidden` |
| NFR-6 **No idle transactions** | Every read ends its transaction, also when it fails, and a connection the server closed is never handed out again. | `test_ends_transaction`, `test_ends_transaction_when_query_fails`, `test_skips_rollback_on_dead_connection`, `test_discards_closed_connection`, `test_read_closes_ends_its_transaction` |
| NFR-7 **Performance** | Pages render in under a second on the demo; generating the demo option data takes seconds, and loading it depends on disk speed. | Measured: see [Performance](#performance) |
| NFR-8 **Bounded memory** | The backtester's full-history panel stores tickers as categories and floats as float32, and loads are cached as parquet. | `test_float32_dtype`, `test_ohlcv_cached_after_first_load`; measured peak in [Performance](#performance) |
| NFR-9 **Security** | No secrets in code or defaults, and the password stays out of `repr` and cache keys. Values are bound parameters; identifiers are constants or `psycopg2.sql`; user-chosen columns pass a whitelist. Preset files can't be written outside their directory. | `test_config.py::test_defaults`, `test_password_hidden_from_repr`, `test_password_not_part_of_key`, `test_rejects_injection`, `test_rejects_unknown_column`, `test_quotes_identifiers`, `test_unknown_column_rejected`, `test_date_bounds_are_parameters`, `test_refuses_path_traversal`, `test_refuses_to_write_through_a_symlink`, `test_traversal_is_not_a_preset`, `test_where_clause_binds_every_value`, gitleaks in CI and pre-commit |
| NFR-10 **Honest claims** | Scans are described by their rules, with no performance claims; the composite's weights are stated as a choice. | `test_every_scan_has_a_rule_without_performance_claims`, `test_composite_rule_names_the_weights`; the backtester's limits in [DESIGN.md](DESIGN.md#limitations) |
| NFR-11 **Reproducible demo** | The demo option data is deterministic and one symbol's data doesn't depend on which others are generated; re-seeding replaces only the generated symbols' rows. | `test_same_seed_gives_identical_frames`, `test_symbol_data_does_not_depend_on_other_symbols`, `test_write_option_data_replaces_symbols_and_copies_every_table`, `test_ddl_is_packaged_and_idempotent` |
| NFR-12 **Portability** | Python 3.12 to 3.14 and PostgreSQL 18. The container runs as a non-root user that can't modify the installed code. | CI matrix (3.12, 3.13, 3.14); integration tests on the compose file's `postgres:18` image; Dockerfile |
| NFR-13 **Maintainability** | Strict typing of `src/` and `research/`, lint and format checks, a 90% coverage floor (lines and branches), and pinned CI actions, base images and the rs-pipeline commit, kept current by Dependabot. | CI `lint` and `test` jobs; `.github/dependabot.yml` |

## Performance

Measured in October 2026 on a laptop (AMD Ryzen 5 7530U, 6 cores, 12
threads) with PostgreSQL 18 in Docker. Other containers and test runs were
using the machine at the same time, so the numbers are indicative; disk
contention affected the database loads most.

**Demo data.** The synthetic market has 40 stocks, 7 ETFs and the benchmark
`MKT` (48 underlyings), from January 2022 (about 1,250 sessions per stock).

| Operation | Time |
|---|---|
| Generate the option data (48 underlyings × 260 sessions, 622k chain rows) | 2.5 s |
| Load it with COPY into a fresh database | 8.7 s uncontended; up to about 2 min while other containers competed for the disk |
| Integration-test setup: start PostgreSQL, run rs-pipeline's demo, seed the option data | 31 s |
| `docker compose build --no-cache` (base images already pulled) | 3 min 38 s |
| First `docker compose up --wait` after the build, from an empty volume | 3 min 4 s, of which the option load took 1 min 48 s |

**Pages** (`uv run pytest tests/integration/test_pages.py --durations=0`,
the `AppTest` run of each test after the shared setup). Rendering a page
with its defaults takes 0.1–0.6 s; a page's main interaction, such as a
screener search or a scan run, takes 0.2–0.7 s. The page test's small
backtest (one filter per strategy, no regime condition, one industry ETF's
members) takes 6 s.

**Backtest on the demo universe** (all 40 stocks, 50,000 stock-days). The
page's default is 1 filter and no regime condition: 52 strategies.

| Setting | Indicator panel | Evaluation | Peak memory |
|---|---|---|---|
| 1 filter, regimes none / bull / bear (156 strategies) | 17 s | 1.1 s | 253 MB |
| 2 filters, regimes none / bull / bear (3,954 strategies) | 16 s | 20 s | 278 MB |

The indicator panel is dominated by the chart-pattern detectors, which run
at every bar of every ticker. Its frame takes 9 MB. Loading the data from the
database takes about 4 s the first time and nothing once the parquet cache
holds it. Time and memory grow linearly with the number of stock-days;
evaluation also grows with the number of strategies, which rises steeply with
the filters per strategy (102,620 strategies at 3 filters with every regime).
