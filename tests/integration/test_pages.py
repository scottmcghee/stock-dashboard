"""Every page and the app entry point, run with Streamlit's AppTest against the seeded demo.

Each test drives a page's main interaction and checks what it shows against the
synthetic market (benchmark MKT, seven ETFs) and the option data's planted
events (``stock_dashboard.demo.options.PlantedEvents``). The last tests point
OPTION_DB_NAME at a missing, an empty and a provider-style database, and
STOCKS_DB_NAME at one without rs-pipeline's tables, and check that the pages
explain the situation instead of failing.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg2
import pytest
import streamlit as st
from psycopg2 import sql
from streamlit.testing.v1 import AppTest

from stock_dashboard import db
from stock_dashboard.config import Settings, get_settings
from stock_dashboard.demo.options import PlantedEvents
from stock_dashboard.logic.backtest import classify_regimes
from stock_dashboard.logic.chart_helpers import (
    FIFTY_TWO_WEEK_BARS,
    best_etf,
    compute_52w_highs_lows,
    trim_to_range,
)
from stock_dashboard.logic.preset import build_preset, parse_preset, save_preset
from stock_dashboard.logic.preset_signal import compute_preset_signal_mask
from stock_dashboard.logic.screener import EVENT_LABELS
from stock_dashboard.queries import option_contract_queries, option_queries, stocks_queries
from stock_dashboard.ui.tables import column_label, selection_key

APP_DIR = Path(__file__).parents[2] / "src" / "stock_dashboard"
PAGES = sorted(p.name for p in (APP_DIR / "views").glob("*.py"))
OPTION_PAGES = ("4_option_data.py", "5_option_screener.py", "10_option_contract.py")
PARTLY_OPTION_PAGES = ("2_stock_screener.py", "6_signal_scans.py")
SCHEMA = APP_DIR / "schema" / "option_data.sql"
TIMEOUT = 120
PLANTED = PlantedEvents()
GUARD_TEXT = "Option data is not available"
# page file -> its navigation title in app.PAGES
NAV_TITLES = dict(re.findall(r'\("(\d+_\w+\.py)", "([^"]+)"', (APP_DIR / "app.py").read_text()))


def page(name: str) -> AppTest:
    return AppTest.from_file(str(APP_DIR / "views" / name), default_timeout=TIMEOUT)


def run(at: AppTest) -> AppTest:
    """Run (or rerun) and fail on any exception the script raised or error it showed."""
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


def by_label(widgets, label: str):
    return next(w for w in widgets if w.label == label)


def symbols(frame: pd.DataFrame, column: str = "symbol") -> set[str]:
    return set(frame[column]) if column in frame.columns else set()


def test_app_home_lists_every_page(settings: Settings) -> None:
    at = run(AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=TIMEOUT))
    listed = at.markdown[0].value
    assert listed.count("- **") == len(PAGES)
    assert not at.info  # option data is available


def test_app_switches_to_a_page(settings: Settings) -> None:
    at = run(AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=TIMEOUT))
    at.switch_page("views/9_52w_highs_lows.py")
    run(at)
    assert at.header[0].value == "52-Week Highs & Lows"


@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders(settings: Settings, name: str) -> None:
    at = run(page(name))
    assert at.header[0].value == NAV_TITLES[name]  # the header matches the navigation
    assert not any(GUARD_TEXT in i.value for i in at.info)


# page -> (plotly charts, tables) shown on first load, before any interaction
FIRST_RENDER = {
    "1_single_stock.py": (3, 0),
    "2_stock_screener.py": (0, 1),
    "3_etf_rs.py": (4, 1),
    "4_option_data.py": (1, 2),
    "5_option_screener.py": (0, 1),
    "6_signal_scans.py": (0, 7),
    "7_technical_zscore.py": (3, 1),
    "9_52w_highs_lows.py": (2, 1),
    "10_option_contract.py": (3, 0),
}


def displayed_labels(element) -> list[str]:
    """The header shown for each column of a rendered st.dataframe."""
    config = json.loads(element.proto.columns or "{}")
    return [config.get(c, {}).get("label") or c for c in element.value.columns]


def assert_readable(labels: list[str]) -> None:
    for label in labels:
        assert "_" not in label, label
        assert not label.islower() or label.endswith("xoi"), label


@pytest.mark.parametrize(("name", "expected"), FIRST_RENDER.items())
def test_first_load_shows_content(settings: Settings, name: str, expected: tuple[int, int]) -> None:
    charts, tables = expected
    at = run(page(name))
    assert len(at.get("plotly_chart")) >= charts
    assert len(at.dataframe) >= tables
    for table in at.dataframe:
        assert_readable(displayed_labels(table))
        assert table.proto.placeholder == ""  # missing values render blank


def test_first_load_defaults(settings: Settings) -> None:
    at = run(page("1_single_stock.py"))
    assert at.selectbox[0].value == stocks_queries.fetch_top_ranked_ticker()

    at = run(page("10_option_contract.py"))
    symbol = option_contract_queries.fetch_most_open_interest_symbol()
    assert by_label(at.selectbox, "Underlying").value == symbol
    latest = option_contract_queries.get_latest_collection_date(symbol)
    assert by_label(at.selectbox, "Expiration").value >= latest + pd.Timedelta(days=7)
    assert by_label(at.selectbox, "Right").value == "call"
    assert len(at.metric) == 4  # surface richness for the near-the-money strike

    at = run(page("4_option_data.py"))
    assert at.selectbox[0].value == symbol

    at = run(page("5_option_screener.py"))
    assert PLANTED.volume_spike in symbols(at.dataframe[0].value)

    at = run(page("8_backtester.py"))
    assert at.slider[0].value == 1
    assert at.multiselect[0].value == ["None (unconditional)"]


# A fixed display range inside the demo history (from 2022-01-03, with room for the
# overlay's warm-up), so the expected signals don't move with today's date.
PINNED_START, PINNED_END = date(2023, 1, 2), date(2024, 12, 31)


def test_single_stock_defaults_to_best_etf_and_draws_overlay(settings: Settings) -> None:
    ticker = "RENI"
    at = run(page("1_single_stock.py"))
    at.selectbox[0].set_value(ticker)
    at.date_input[0].set_value(PINNED_START)
    at.date_input[1].set_value(PINNED_END)
    run(at)
    etf_box = next(s for s in at.selectbox if s.label.startswith("ETF"))
    # The default is the stock's best ETF as of the displayed range's end, not of today.
    assert etf_box.value == best_etf(stocks_queries.fetch_stock_metrics(ticker, end=PINNED_END))
    assert len(at.get("plotly_chart")) == 3  # price, RS with its MAs, ranks

    # The overlay must count exactly the signals a full-history evaluation finds in the
    # displayed range: indicators need warm-up bars before the first displayed day.
    preset = "sample_oversold_in_uptrend"
    at.selectbox(key="single_stock_preset").set_value(preset)
    run(at)
    full = stocks_queries.fetch_ohlcv("stock", [ticker])
    mask = compute_preset_signal_mask(
        full, stocks_queries.fetch_stock_metrics(ticker), parse_preset(preset)["_raw_filters"]
    ).mask
    expected = trim_to_range(full[mask], PINNED_START, PINNED_END)
    caption = next(c.value for c in at.caption if "signal(s) from preset" in c.value)
    assert int(caption.split()[0]) == len(expected) > 0
    assert "**Sample: oversold in an uptrend**" in caption  # the preset's name, not its file
    assert not at.warning


def test_single_stock_overlay_applies_the_preset_regime_and_etf(settings: Settings) -> None:
    """A regime preset marks only bars in that regime; an ETF preset uses that ETF's rows."""
    ticker = "RENI"
    metrics = stocks_queries.fetch_stock_metrics(ticker)
    etf = sorted(metrics["etf_ticker"].unique())[-1]
    filters = {"rsi_max": 40, "rs_z_score_min": -10.0}
    save_preset(build_preset("Oversold in bull", filters, regimes=["bull"], etf=etf))
    at = run(page("1_single_stock.py"))
    at.selectbox[0].set_value(ticker)
    at.date_input[0].set_value(PINNED_START)
    at.date_input[1].set_value(PINNED_END)
    at.selectbox(key="single_stock_preset").set_value("Oversold_in_bull")
    run(at)

    full = stocks_queries.fetch_ohlcv("stock", [ticker])
    regimes = classify_regimes(stocks_queries.fetch_ohlcv("etf", [settings.benchmark]))
    raw = parse_preset("Oversold_in_bull")["_raw_filters"]
    with_regime = compute_preset_signal_mask(
        full, metrics, raw, etf=etf, regime="bull", regimes=regimes
    ).mask
    unconditional = compute_preset_signal_mask(full, metrics, raw).mask
    expected = trim_to_range(full[with_regime], PINNED_START, PINNED_END)
    assert set(expected["date"].map(regimes)) <= {"bull"}
    assert len(expected) < len(trim_to_range(full[unconditional], PINNED_START, PINNED_END))
    caption = next(c.value for c in at.caption if "signal(s) from preset" in c.value)
    assert int(caption.split()[0]) == len(expected) > 0
    applied = next(c.value for c in at.caption if "The overlay applies" in c.value)
    assert "regime (bull" in applied
    assert etf in applied


def test_single_stock_preset_etf_the_stock_is_not_in(settings: Settings) -> None:
    """Once: 'cannot evaluate rs_z_score_min=-10' with no reason, next to 'applies the ETF'."""
    ticker = "RENI"
    members = set(stocks_queries.fetch_stock_metrics(ticker)["etf_ticker"])
    other = next(e for e in sorted(stocks_queries.get_all_etfs()) if e not in members)
    save_preset(build_preset("Other ETF", {"rs_z_score_min": -10.0}, etf=other))
    at = run(page("1_single_stock.py"))
    at.selectbox[0].set_value(ticker)
    at.selectbox(key="single_stock_preset").set_value("Other_ETF")
    run(at)
    assert [w.value for w in at.warning] == [
        f"{ticker} is not a member of {other}, the ETF the preset was backtested within, "
        "so the overlay shows no signals."
    ]
    assert not any("The overlay applies" in c.value for c in at.caption)
    caption = next(c.value for c in at.caption if "signal(s) from preset" in c.value)
    assert caption.startswith("0 ")


def test_single_stock_regime_preset_without_benchmark_bars(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_preset(build_preset("Any bull day", {"rsi_min": 0}, regimes=["bull"]))
    monkeypatch.setenv("BENCHMARK_TICKER", "NOPE")
    get_settings.cache_clear()
    at = run(page("1_single_stock.py"))
    at.selectbox(key="single_stock_preset").set_value("Any_bull_day")
    run(at)
    assert "regime=bull" in at.warning[0].value
    assert "NOPE has no price bars" in at.warning[0].value
    caption = next(c.value for c in at.caption if "signal(s) from preset" in c.value)
    assert caption.startswith("0 ")


def test_stock_screener_returns_one_row_per_ticker(settings: Settings) -> None:
    at = run(page("2_stock_screener.py"))
    by_label(at.button, "Search").click()
    run(at)
    table = at.dataframe[0].value
    assert table["Ticker"].is_unique
    assert set(table["Ticker"]) == set(stocks_queries.search_stocks_by_metrics()["ticker"])
    assert {"GEX", "P/C Ratio"} <= set(table.columns)  # option columns shown
    assert len(at.get("download_button")) == 1


def test_stock_screener_signal_scan_prefilter_and_ta_filters(settings: Settings) -> None:
    at = run(page("2_stock_screener.py"))
    by_label(at.multiselect, "Signal Scans (all must match)").set_value(["volume_spike"])
    by_label(at.number_input, "RSI Min").set_value(0)
    by_label(at.button, "Search").click()
    run(at)
    table = at.dataframe[0].value
    assert PLANTED.volume_spike in set(table["Ticker"])
    assert "RSI" in table.columns
    assert table["RS Rank in ETF %"].between(0, 100).all()


def test_stock_screener_event_filter_finds_events(settings: Settings) -> None:
    """Regression: the screener fetched too few bars for the event flags, so this was empty."""
    at = run(page("2_stock_screener.py"))
    by_label(at.multiselect, "TA Events in the last 3 bars (any may match)").set_value(
        list(EVENT_LABELS)
    )
    by_label(at.button, "Search").click()
    run(at)
    table = at.dataframe[0].value
    assert not table.empty
    assert table["TA Events"].notna().all()


def test_stock_screener_loads_a_sample_preset(settings: Settings) -> None:
    at = run(page("2_stock_screener.py"))
    at.selectbox(key="_screener_preset_select").set_value("sample_rs_leaders")
    run(at)
    assert by_label(at.number_input, "RS Rank Overall Min (%)").value == pytest.approx(80.0)
    by_label(at.button, "Search").click()
    run(at)
    table = at.dataframe[0].value
    assert not table.empty
    assert (table["RS Rank Overall %"] >= 80).all()


def test_stock_screener_states_the_preset_regime_and_skipped_filters(settings: Settings) -> None:
    filters = {"momentum_20_min": 0.0, "momentum_126_max": 500.0, "rsi_min": 0.5}
    save_preset(build_preset("Two momenta in bear", filters, regimes=["bear"]))
    at = run(page("2_stock_screener.py"))
    at.selectbox(key="_screener_preset_select").set_value("Two_momenta_in_bear")
    run(at)
    assert "momentum_126_max=500.0" in at.info[0].value
    assert by_label(at.number_input, "Momentum Min (%)").value == 0.0
    assert by_label(at.number_input, "Momentum Max (%)").value is None
    assert by_label(at.number_input, "RSI Min").value == pytest.approx(0.5)
    regimes = classify_regimes(stocks_queries.fetch_ohlcv("etf", [settings.benchmark]))
    latest = regimes.dropna().sort_index()
    note = next(c.value for c in at.caption if "backtested in the bear regime" in c.value)
    assert f"{settings.benchmark} is in the {latest.iloc[-1]} regime" in note
    assert f"{latest.index[-1]:%Y-%m-%d}" in note


def test_stock_screener_preset_value_outside_a_widget_range(settings: Settings) -> None:
    """Once an uncaught StreamlitValueAboveMaxError, on every rerun until Reset Filters."""
    save_preset(build_preset("Percent ranks", {"rs_rank_overall_min": 80, "rsi_min": 150}))
    at = page("2_stock_screener.py").run()
    at.selectbox(key="_screener_preset_select").set_value("Percent_ranks")
    at.run()
    assert not at.exception
    assert "rs_rank_overall_min=80 is outside 0 to 1" in at.error[0].value
    assert "ranks are fractions" in at.error[0].value
    by_label(at.button, "Search").click()
    at.run()
    assert not at.exception
    assert len(at.dataframe) == 1  # the form still works, without the preset


def test_unreadable_preset_is_skipped(settings: Settings) -> None:
    (settings.preset_dir / "truncated.json").parent.mkdir(parents=True, exist_ok=True)
    (settings.preset_dir / "truncated.json").write_text('{"name": "cut')
    for name, key in (("1_single_stock.py", "single_stock_preset"),
                      ("2_stock_screener.py", "_screener_preset_select")):  # fmt: skip
        at = run(page(name))
        assert "truncated" not in at.selectbox(key=key).options


def test_etf_grid_has_one_row_per_etf_and_charts_the_selection(settings: Settings) -> None:
    at = run(page("3_etf_rs.py"))
    grid = at.dataframe[0].value
    assert grid["etf_ticker"].is_unique
    assert set(grid["etf_ticker"]) == set(stocks_queries.fetch_latest_etf_grid()["etf_ticker"])
    labels = displayed_labels(at.dataframe[0])
    assert "Rank Change 10d (pts)" in labels
    assert "ETF RS Impulse 10d (z)" in labels  # one name for the concept, as in the caption
    assert grid["impulse_10d"].abs().max() > 1  # percentage points, not a 0-1 fraction
    by_label(at.selectbox, "ETF to chart").set_value("TCHX")
    run(at)
    assert any(s.value.startswith("TCHX") for s in at.subheader)
    assert len(at.get("plotly_chart")) == 4  # impulse, RS, rank, z-score


def select_row(at: AppTest, prefix: str, row_ids: pd.Series, row: int) -> None:
    """Send the selection a browser would send for ``row`` of a content-keyed table."""
    at.session_state[selection_key(prefix, row_ids)] = {"selection": {"rows": [row], "columns": []}}


def test_etf_grid_row_click_selects_the_etf(settings: Settings) -> None:
    at = run(page("3_etf_rs.py"))
    etfs = at.dataframe[0].value["etf_ticker"]
    select_row(at, "etf_grid", etfs, len(etfs) - 1)
    run(at)
    assert by_label(at.selectbox, "ETF to chart").value == etfs.iloc[-1]


def test_option_data_shows_ratio_row_and_history(settings: Settings) -> None:
    """Also: the ratio row and the chart list only numeric columns."""
    at = run(page("4_option_data.py"))
    at.selectbox[0].set_value(PLANTED.volume_spike)
    run(at)
    ratio_row = at.dataframe[0].value
    assert ratio_row["option_volume"].iloc[0] > 1  # the planted spike
    assert ratio_row[["flownet", "dxoi", "gxoi"]].isna().all(axis=None)  # signed: no multiple
    assert len(at.dataframe[1].value) == 30
    numeric = set(option_queries.get_option_numerical_columns())
    assert set(by_label(at.selectbox, "Metric to chart").options) <= {
        column_label(c) for c in numeric
    }
    recent = json.loads(at.dataframe[1].proto.columns)
    assert recent["collection_date"]["type_config"]["format"] == "YYYY-MM-DD"
    assert recent["flownet"]["type_config"]["format"] == "%,.0f"
    by_label(at.selectbox, "Metric to chart").set_value("option_volume")
    run(at)
    assert len(at.get("plotly_chart")) == 1
    layout = json.loads(at.get("plotly_chart")[0].proto.spec)["layout"]
    assert layout["yaxis"]["title"]["text"] == column_label("option_volume")


def test_option_screener_finds_the_volume_spike(settings: Settings) -> None:
    at = run(page("5_option_screener.py"))
    columns = option_queries.get_option_multiple_columns()
    assert at.selectbox[0].options == [f"{column_label(c)} ({c})" for c in columns]
    assert "option_volume" in columns
    assert not {"flownet", "dxoi", "gxoi", "vannaxoi"} & set(columns)  # signed
    at.selectbox[0].set_value("option_volume")
    at.number_input[0].set_value(5.0)
    at.button[0].click()
    run(at)
    assert PLANTED.volume_spike in symbols(at.dataframe[0].value)


def test_signal_scans_find_the_planted_symbols(settings: Settings) -> None:
    at = run(page("6_signal_scans.py"))
    tabs = {t.label: t for t in at.tabs}
    expected = {
        "Vanna Extreme": {PLANTED.vanna_high, PLANTED.vanna_low},
        "Flownet Extreme": {PLANTED.flownet_low},
        "Volume Spike": {PLANTED.volume_spike},
        "dxoi Change": {PLANTED.dxoi_jump},
        "Negative GEX": set(PLANTED.negative_gex),
    }
    for label, planted in expected.items():
        found = symbols(tabs[label].dataframe[0].value)
        assert planted <= found, (label, found)
    composite = symbols(tabs["Composite"].dataframe[0].value)
    assert set().union(*expected.values()) <= composite
    assert "Washout Bull" in tabs
    # Strength is the per-scan measure; no table adds a z-score of it.
    for label in (*expected, "Composite"):
        assert "z_score" not in tabs[label].dataframe[0].value.columns, label


def test_signal_scans_etf_mode(settings: Settings) -> None:
    at = run(page("6_signal_scans.py"))
    at.checkbox(key="trend_etf_mode").check()
    run(at)
    tabs = {t.label: t for t in at.tabs}
    assert {"Washout Bull", "Washout Bear", "Coiled Spring"} <= set(tabs)
    assert not at.warning  # the ETFs have the history the trend scans need
    etfs = set(stocks_queries.get_all_etfs())
    for label in ("Washout Bull", "Washout Bear", "Coiled Spring"):
        tab = tabs[label]
        if tab.dataframe:
            assert symbols(tab.dataframe[0].value) <= etfs
        else:
            assert "No symbols match" in tab.info[0].value


def test_signal_scans_show_percentiles_in_percent(settings: Settings) -> None:
    at = run(page("6_signal_scans.py"))
    vanna = next(t for t in at.tabs if t.label == "Vanna Extreme").dataframe[0]
    assert "Percentile %" in displayed_labels(vanna)
    ranks = vanna.value["pct_rank"]
    assert ranks.max() > 1  # 0-100, like the sidebar's percentile sliders
    assert ranks.between(0, 100).all()


def test_technical_zscore_stale_selection_resets(settings: Settings) -> None:
    """A row selected under one universe must not carry over to another universe.

    Row 1 exists in both tables, so only the content-keyed table resets it; a
    constant key would keep row 1 selected and chart a different ticker.
    """
    at = run(page("7_technical_zscore.py"))
    tickers = at.dataframe[0].value["ticker"]
    select_row(at, "zscore_table", tickers, 1)
    run(at)
    assert by_label(at.selectbox, "Ticker to chart").value == tickers.iloc[1]
    by_label(at.selectbox, "ETF universe").set_value("SEMX")
    by_label(at.button, "Compute").click()
    run(at)
    smaller = at.dataframe[0].value["ticker"]
    assert 1 < len(smaller) < len(tickers)
    assert by_label(at.selectbox, "Ticker to chart").value == smaller.iloc[0]


def test_technical_zscore_computes_and_charts(settings: Settings) -> None:
    at = run(page("7_technical_zscore.py"))
    by_label(at.button, "Compute").click()
    run(at)
    table = at.dataframe[0].value
    assert table["ticker"].is_unique
    assert {"RSI Z", "Composite Z", "ETF"} <= set(displayed_labels(at.dataframe[0]))
    box = by_label(at.selectbox, "Ticker to chart")
    box.set_value(box.options[0])
    run(at)
    assert len(at.get("plotly_chart")) == 3

    at.checkbox[0].check()
    by_label(at.button, "Compute").click()
    run(at)
    assert set(at.dataframe[0].value["ticker"]) <= set(stocks_queries.get_all_etfs())


def test_backtester_runs_a_small_backtest_and_saves_a_preset(settings: Settings) -> None:
    at = run(page("8_backtester.py"))
    at.selectbox[0].set_value("SEMX")
    at.slider[0].set_value(1)
    at.multiselect[0].set_value(["None (unconditional)"])
    by_label(at.checkbox, "Show all strategies (including non-significant)").check()
    by_label(at.button, "Run Backtest").click()
    run(at)
    table = at.dataframe[0].value
    assert not table.empty
    assert table["strategy"].is_unique
    assert_readable(displayed_labels(at.dataframe[0]))
    assert (table["regime"] == "any").all()
    # Styled to show "<0.0001", but still numbers, so the column sorts numerically.
    assert pd.api.types.is_numeric_dtype(table["T+5_pval"])
    assert f"{settings.benchmark} and market regimes" in [e.label for e in at.expander]

    box = by_label(at.selectbox, "Strategy to inspect")
    box.set_value(box.options[0])
    run(at)
    assert at.subheader[-1].value == "Save as Preset"
    by_label(at.button, "Save Preset").click()
    run(at)
    saved = Path(at.success[0].value.split("`")[1])
    assert saved.parent == settings.preset_dir.resolve()
    assert parse_preset(saved.stem)["_raw_filters"]
    assert parse_preset(saved.stem)["_etf"] == "SEMX"


def _run_backtest(at: AppTest, regimes: list[str], etf: str | None = "SEMX") -> AppTest:
    at.selectbox[0].set_value(etf)
    at.multiselect[0].set_value(regimes)
    by_label(at.checkbox, "Show all strategies (including non-significant)").check()
    by_label(at.button, "Run Backtest").click()
    return run(at)


def test_backtester_stale_selection_resets(settings: Settings) -> None:
    """Show all, select the last row, then hide the non-significant rows: no stale index."""
    at = _run_backtest(run(page("8_backtester.py")), ["None (unconditional)"], etf="TCHX")
    strategies = at.dataframe[0].value["strategy"]
    select_row(at, "bt_table", strategies, len(strategies) - 1)
    run(at)
    assert by_label(at.selectbox, "Strategy to inspect").value == strategies.iloc[-1]
    by_label(at.checkbox, "Show all strategies (including non-significant)").uncheck()
    run(at)
    assert 0 < len(at.dataframe[0].value) < len(strategies)
    assert by_label(at.selectbox, "Strategy to inspect").value is None
    # Re-sorting reorders the rows: the selection resets instead of picking another strategy.
    by_label(at.checkbox, "Show all strategies (including non-significant)").check()
    run(at)
    strategies = at.dataframe[0].value["strategy"]
    select_row(at, "bt_table", strategies, 0)
    run(at)
    by_label(at.selectbox, "Sort by horizon").set_value("T+5")
    run(at)
    assert by_label(at.selectbox, "Strategy to inspect").value is None


def test_backtester_without_benchmark_bars_runs_only_unconditional_strategies(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHMARK_TICKER", "SPY")  # not in the synthetic market
    get_settings.cache_clear()
    at = _run_backtest(run(page("8_backtester.py")), ["None (unconditional)", "bull"])
    assert "only unconditional strategies were evaluated" in at.warning[0].value
    table = at.dataframe[0].value
    assert not table.empty
    assert (table["regime"] == "any").all()


def test_52_week_highs_lows(settings: Settings) -> None:
    at = run(page("9_52w_highs_lows.py"))
    assert [m.label for m in at.metric] == ["New 52W Highs", "New 52W Lows", "High/Low Ratio"]
    assert len(at.get("plotly_chart")) == 2
    by_label(at.checkbox, "Include ETFs").uncheck()
    run(at)
    start = at.date_input(key="52w_start").value
    prices = stocks_queries.fetch_ohlcv(
        "stock", start=start, warmup_bars=FIFTY_TWO_WEEK_BARS, columns=("high", "low")
    )
    daily, highs, lows = compute_52w_highs_lows(prices, start.isoformat())
    assert not daily.empty
    latest = daily.iloc[-1]
    assert [m.value for m in at.metric[:2]] == [str(latest["new_highs"]), str(latest["new_lows"])]
    shown = set().union(*(set(f.value["ticker"]) for f in at.dataframe))
    assert shown == set(highs["ticker"]) | set(lows["ticker"])
    assert not shown & set(stocks_queries.get_all_etfs())


def test_option_contract_picks_an_expiration_and_contract(settings: Settings) -> None:
    at = run(page("10_option_contract.py"))
    at.selectbox[0].set_value(PLANTED.vanna_high)
    run(at)
    expirations = by_label(at.selectbox, "Expiration")
    expirations.set_value(pd.Timestamp(expirations.options[-1]))
    run(at)
    # OI by strike, max pain, and the default near-the-money contract
    assert len(at.get("plotly_chart")) == 3
    # Regression: the put bars' hover showed the strike as "Put OI".
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    title = spec["layout"]["title"]["text"]
    assert title.startswith(f"{PLANTED.vanna_high} ")
    assert "OI by strike" in title
    assert "click" not in title  # was cut off in the chart; the hint is a caption now
    assert any("Click a bar" in c.value for c in at.caption)
    traces = spec["data"]
    puts = next(t for t in traces if t["name"] == "Puts")
    assert "customdata[2]" in puts["hovertemplate"]
    assert [row[2] for row in puts["customdata"]] == [-x for x in puts["x"]]
    strikes = by_label(at.selectbox, "Strike")
    strikes.select_index(len(strikes.options) // 2)
    by_label(at.selectbox, "Right").set_value("put")
    run(at)
    assert [m.label for m in at.metric] == ["Contract IV", "ATM IV", "IV - ATM", "Surface rank"]
    assert len(at.get("plotly_chart")) == 3
    layout = json.loads(at.get("plotly_chart")[2].proto.spec)["layout"]
    assert "Price" not in [a["text"] for a in layout["annotations"]]  # the y-axis says it
    assert layout["yaxis3"]["tickmode"] == "auto"  # not synced to the OI axis


# --- Without option data ---------------------------------------------------


@pytest.fixture
def empty_option_db(seeded: Settings) -> str:
    """A database holding the option-data contract tables, all empty."""
    name = "empty_options"
    conn = psycopg2.connect(**seeded.db_kwargs("postgres"))
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        if cur.fetchone() is None:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    conn.close()
    with psycopg2.connect(**seeded.db_kwargs(name)) as schema_conn:
        schema_conn.cursor().execute(SCHEMA.read_text())
    schema_conn.close()
    return name


@pytest.fixture(params=["missing", "empty"])
def no_option_data(
    request: pytest.FixtureRequest, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    dbname = (
        "no_such_database"
        if request.param == "missing"
        else request.getfixturevalue("empty_option_db")
    )
    monkeypatch.setenv("OPTION_DB_NAME", dbname)
    get_settings.cache_clear()
    db.close_all_pools()
    st.cache_data.clear()
    assert db.option_data_available() == (request.param == "empty")


@pytest.mark.parametrize("name", OPTION_PAGES)
def test_option_pages_explain_missing_option_data(no_option_data: None, name: str) -> None:
    at = run(page(name))
    assert GUARD_TEXT in at.info[0].value
    assert not at.dataframe
    assert not at.selectbox


@pytest.mark.parametrize("name", PARTLY_OPTION_PAGES)
def test_partly_option_pages_render_without_option_data(no_option_data: None, name: str) -> None:
    """They note what is missing and still show their stock-based tables."""
    at = run(page(name))
    assert GUARD_TEXT in at.info[0].value
    assert at.dataframe
    assert not at.dataframe[0].value.empty


def test_screener_hides_option_filters_without_option_data(no_option_data: None) -> None:
    at = run(page("2_stock_screener.py"))
    assert GUARD_TEXT in at.info[0].value
    assert "Weekly options only" not in [c.label for c in at.checkbox]
    scans = by_label(at.multiselect, "Signal Scans (all must match)")
    assert set(scans.options) == {"washout_bull", "washout_bear", "coiled_spring"}
    by_label(at.button, "Search").click()
    run(at)
    table = at.dataframe[0].value
    assert table["Ticker"].is_unique
    assert not {"GEX", "P/C Ratio", "Vanna"} & set(table.columns)


def test_signal_scans_keep_trend_tabs_without_option_data(no_option_data: None) -> None:
    at = run(page("6_signal_scans.py"))
    assert GUARD_TEXT in at.info[0].value
    assert [t.label for t in at.tabs] == [
        "Washout Bull",
        "Washout Bear",
        "Coiled Spring",
        "Composite",
    ]


def test_app_home_explains_missing_option_data(no_option_data: None) -> None:
    at = run(AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=TIMEOUT))
    assert GUARD_TEXT in at.info[0].value


# --- Databases that are not what the dashboard expects ------------------------


def _create_database(seeded: Settings, name: str, template: str | None = None) -> None:
    db.close_all_pools()  # a template database must have no open connections
    conn = psycopg2.connect(**seeded.db_kwargs("postgres"))
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        if cur.fetchone() is None:
            query = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name))
            if template:
                query += sql.SQL(" TEMPLATE {}").format(sql.Identifier(template))
            cur.execute(query)
    conn.close()


@pytest.fixture
def provider_option_db(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> str:
    """The demo option data plus provider-style extra columns: text, timestamp, and a
    numeric one with no _avg partner."""
    name = "provider_options"
    _create_database(settings, name, template=settings.option_db)
    with psycopg2.connect(**settings.db_kwargs(name)) as conn, conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE underlying_data
                ADD COLUMN IF NOT EXISTS source varchar DEFAULT 'feed',
                ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now(),
                ADD COLUMN IF NOT EXISTS skew_extra double precision DEFAULT -0.1
        """)
    conn.close()
    monkeypatch.setenv("OPTION_DB_NAME", name)
    get_settings.cache_clear()
    db.close_all_pools()
    st.cache_data.clear()
    return name


def test_option_data_with_provider_extra_columns(provider_option_db: str) -> None:
    """Text columns are not chartable; a numeric column without an average is charted alone."""
    at = run(page("4_option_data.py"))
    at.selectbox[0].set_value(PLANTED.volume_spike)
    run(at)
    metric = by_label(at.selectbox, "Metric to chart")
    assert not {"Source", "Updated at"} & set(metric.options)
    assert "Skew extra" in metric.options
    assert "source" in at.dataframe[1].value.columns  # still in the table of recent rows
    metric.set_value("skew_extra")
    run(at)
    assert len(at.get("plotly_chart")) == 1
    assert any("No 30-session average for Skew extra" in c.value for c in at.caption)

    at = run(page("5_option_screener.py"))
    assert not any("skew_extra" in o for o in at.selectbox[0].options)


def test_stocks_database_without_rs_pipeline_tables(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real UndefinedTable from a database rs-pipeline never filled, through app.py.

    The page is reached with ``switch_page``, so the error is caught by the
    ``database_errors`` boundary app.py wraps every page in.
    """
    _create_database(settings, "not_rs_pipeline")
    monkeypatch.setenv("STOCKS_DB_NAME", "not_rs_pipeline")
    get_settings.cache_clear()
    db.close_all_pools()
    st.cache_data.clear()
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=TIMEOUT).run()
    at.switch_page("views/1_single_stock.py").run()
    assert not at.exception
    assert "run rs-pipeline first" in at.error[0].value
    assert "dim_stock" in at.error[0].value
