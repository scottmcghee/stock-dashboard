# Contributing

## Setup

```bash
uv sync                    # Python 3.12+; installs the package and dev tools into .venv
uv run pre-commit install  # lint, format, type-check and secret-scan on each commit
```

Docker is needed for the integration tests and the demo. Without Docker the
integration tests are skipped locally (and coverage then falls below the 90%
floor). In CI, where `CI` is set, a missing Docker fails the run instead.

## Checks

```bash
uv run ruff check .                  # lint
uv run ruff format .                 # format (CI runs --check)
uv run mypy                          # strict type check of src/ and research/
uv run pytest -m "not integration"   # fast unit tests, no database
uv run pytest --cov                  # everything, with the coverage floor
docker compose up --build            # the demo (no market-data feed or API keys)
```

CI runs the same checks on every push to `main` and on every pull request.
ruff also formats the Python code blocks in Markdown files, so a snippet that
isn't valid, formatted Python belongs in a `text` or `bash` fence.

### Unit and integration tests

| | Unit (`tests/unit`) | Integration (`tests/integration`) |
|---|---|---|
| Needs | nothing | Docker |
| Database | none: `db.read_df` is replaced by a fake that records each query's SQL and parameters, and opening a real connection fails the test | a throwaway PostgreSQL 18 container |
| Covers | indicators, scans, screener and backtest logic, query text and bound parameters, settings, presets, the demo generator, the Streamlit components | real SQL, every page, the demo data, the research scripts |

The integration session starts the PostgreSQL image named in
`docker-compose.yml`, runs rs-pipeline's demo into it (rs-pipeline is a dev
dependency, pinned to a commit in `pyproject.toml`), then runs this project's
option-data generator: the same steps as `docker compose up`, run in-process
instead of through compose. That setup runs
once per session; its time is in
[REQUIREMENTS.md](docs/REQUIREMENTS.md#performance). The tests only read, so
they share the seeded server (a few create extra databases on it).

Page tests (`tests/integration/test_pages.py`) run each page script with
Streamlit's `AppTest`, drive its main interaction, and check the result
against the demo data and its planted events. They also point
`OPTION_DB_NAME` at a missing, an empty and a provider-style database (extra
columns), and `STOCKS_DB_NAME` at a database without rs-pipeline's tables, to
check that the pages explain the situation instead of failing. Most tests run
a page script directly; the ones that need `app.py`'s navigation, such as the
missing-tables test, which relies on the `ui.errors.database_errors` boundary
`app.py` wraps every page in, start from `app.py` and use `switch_page`.

### Coverage

The floor is 90% of lines and branches (`[tool.coverage]` in
`pyproject.toml`). The page scripts, `app.py` and the `__main__` modules are
left out of the measurement: coverage can't trace scripts that run inside
`AppTest`, and the entry points only hand over to Streamlit or to the demo
seeder. Every page is still run by the page tests.

## Conventions

- **"Latest" is a data date.** A query that wants the latest session anchors
  on `MAX(date)` (or `MAX(collection_date)`) of the table it reads, never on
  `NOW()` or `CURRENT_DATE`.
- **Look-backs are trading bars.** Fetch indicator history with
  `fetch_ohlcv(..., last_bars=N)` or `warmup_bars=N`, where `N` comes from the
  indicator's own `*_REQUIRED_BARS` constant or from `logic/lookback.py`, not
  from a calendar interval.
- **SQL is parameterized.** Values are bound parameters. Identifiers are
  constants, or are composed with `psycopg2.sql.Identifier`; a column name
  chosen in the UI must first pass the table's whitelist
  (`option_queries.validate_column`). Every query goes through `db.read_df`,
  which ends its transaction.
- **Logic is pure and tested.** `logic/` takes and returns DataFrames and
  plain values, with no database or Streamlit imports, and has unit tests.
  Indicators are defined once, in `logic/technical.py` or next to the scan
  that owns them, and reused everywhere else.
- **Pages are thin.** A page reads widgets, calls cached query wrappers and
  logic functions, and renders the result. Shared widgets go in `ui/`.
- **Named constants, not magic numbers**, and no performance claims in UI
  text, docstrings or comments: describe a scan by its rule.
- Logging goes through `logging`, never `print`, in library code. Catch
  specific exceptions; a broad `except` belongs only at a UI boundary that
  shows the error.

## Adding things

**A page.** Add `src/stock_dashboard/views/<n>_<name>.py` and an entry
(file, title, icon, one-line purpose) to `PAGES` in `app.py`. Wrap its
queries in `st.cache_data` with one of the TTLs in `config.py`. A page that
needs option data calls `ui.option_guard.require_option_data()` first; one
that only partly needs it calls `note_missing_option_data()` and hides those
parts. `test_every_page_renders` picks the page up automatically; add a test
in `test_pages.py` that drives its main interaction. A page that needs option
data goes into `OPTION_PAGES` there; one that partly needs it goes into
`PARTLY_OPTION_PAGES` and gets a test with the `no_option_data` fixture, like
`test_screener_hides_option_filters_without_option_data`. A table with
row selection uses `ui.tables.selection_key` and `selected_row`, so a stale
selection resets when the rows change.

**A query.** Add a function to the matching module in `queries/` that calls
`db.read_df(db.Database.STOCKS | OPTIONS, query, params)` and returns a
DataFrame. Test it in `tests/unit` with the `fake_db` fixture, which checks
the SQL text and parameters without a database, and through a page test if a
page uses it. A new option-data column must also go into
`schema/option_data.sql` and [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md).

**A scan.**
- *Option scan:* a pure `classify_*` function in `logic/option_signals.py`
  that looks at the latest session and returns `symbol`, `signal_value`, a
  scale-free `strength` (larger = more extreme; the composite ranks it within
  the scan) and `signal_label`. Then: its column in `SIGNAL_COLUMNS`; a
  wrapper in `queries/signal_queries.py`; an entry in `OPTION_SCANS`
  (`queries/screener_queries.py`), in `OPTION_SCAN_TITLES` and `scan_rule`
  (`logic/scan_settings.py`), in `TIER_WEIGHTS` (`logic/signal_scanner.py`),
  and in `_option_scans` on the scans page. If it has a threshold, also give
  it a `DEFAULT_*` constant (`option_signals.py`), a `ScanSettings` field and
  a sidebar slider on the scans page, passed to `_option_scans` as a
  parameter. Plant an event for it in `demo/options.py` (`PlantedEvents`), so
  the demo and the page tests have a hit.
- *Trend scan:* a function over `compute_trend_indicators`' output in
  `logic/trend_signals.py` that adds a `strength` column, with defaults equal
  to its settings in `scan_settings.py` (`ScanSettings`); register it in
  `TREND_SCANS`, `TREND_SCAN_TITLES` and `scan_rule` (`scan_settings.py`) and
  in `TIER_WEIGHTS` (`logic/signal_scanner.py`). The scans page and the
  screener's pre-filter both run it through `TREND_SCANS`.
