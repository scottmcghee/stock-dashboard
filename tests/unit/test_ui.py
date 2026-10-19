"""Tests for the shared Streamlit components in ui/, run through Streamlit's AppTest.

AppTest runs each small script in this process, so the monkeypatched query
functions are the ones the components call.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import psycopg2
import pytest
from streamlit.testing.v1 import AppTest

from stock_dashboard.config import get_settings
from stock_dashboard.ui import export, option_guard
from stock_dashboard.ui.charts import rs_figure
from stock_dashboard.ui.tables import (
    COLUMN_LABELS,
    column_config,
    column_label,
    selected_row,
    selection_key,
)

TIMEOUT = 30


@pytest.fixture(autouse=True)
def _clear_component_caches():
    option_guard.option_data_ready.clear()
    export._watchlist.clear()
    yield
    option_guard.option_data_ready.clear()
    export._watchlist.clear()


@pytest.fixture
def options_available(monkeypatch):
    def set_state(contract: bool, rows: bool = True) -> None:
        monkeypatch.setattr(option_guard.db, "option_data_available", lambda: contract)
        monkeypatch.setattr(option_guard, "has_underlying_rows", lambda: rows)

    return set_state


class TestOptionDataReady:
    def test_needs_the_contract_tables(self, options_available):
        options_available(contract=False)
        assert option_guard.option_data_ready() is False

    def test_needs_rows(self, options_available):
        options_available(contract=True, rows=False)
        assert option_guard.option_data_ready() is False

    def test_ready(self, options_available):
        options_available(contract=True)
        assert option_guard.option_data_ready() is True

    def test_unreadable_tables(self, monkeypatch):
        def fail() -> bool:
            raise psycopg2.errors.InsufficientPrivilege("permission denied")

        monkeypatch.setattr(option_guard.db, "option_data_available", lambda: True)
        monkeypatch.setattr(option_guard, "has_underlying_rows", fail)
        assert option_guard.option_data_ready() is False


def test_message_names_the_database_and_tables(monkeypatch):
    monkeypatch.setenv("OPTION_DB_NAME", "my_options")
    option_guard.get_settings.cache_clear()
    message = option_guard.missing_option_data_message()
    assert "`my_options`" in message
    assert "option_chain_data, underlying_data, underlying_data_avg" in message
    assert "against a database without real option data" in message  # the seeder replaces rows
    assert "(https://github.com/scottmcghee/stock-dashboard/blob/main/docs/DATA_CONTRACT.md)" in (
        message
    )


def _require_script() -> None:
    import streamlit as st

    from stock_dashboard.ui.option_guard import require_option_data

    require_option_data()
    st.write("page body")


def _note_script() -> None:
    import streamlit as st

    from stock_dashboard.ui.option_guard import note_missing_option_data

    ready = note_missing_option_data("the option columns")
    st.write(f"ready={ready}")


class TestRequireOptionData:
    def test_stops_with_an_explanation(self, options_available):
        options_available(contract=False)
        at = AppTest.from_function(_require_script, default_timeout=TIMEOUT).run()
        assert not at.exception
        assert "Option data is not available" in at.info[0].value
        assert not [m for m in at.markdown if m.value == "page body"]

    def test_passes_through(self, options_available):
        options_available(contract=True)
        at = AppTest.from_function(_require_script, default_timeout=TIMEOUT).run()
        assert not at.info
        assert at.markdown[0].value == "page body"


class TestNoteMissingOptionData:
    def test_names_what_is_hidden(self, options_available):
        options_available(contract=False)
        at = AppTest.from_function(_note_script, default_timeout=TIMEOUT).run()
        assert at.info[0].value.endswith("Hidden on this page: the option columns.")
        assert at.markdown[0].value == "ready=False"

    def test_silent_when_ready(self, options_available):
        options_available(contract=True)
        at = AppTest.from_function(_note_script, default_timeout=TIMEOUT).run()
        assert not at.info
        assert at.markdown[0].value == "ready=True"


@pytest.fixture
def known(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fetch_known(symbols):
        calls.append(tuple(symbols))
        return {"AAA", "TCHX"} & set(symbols)

    monkeypatch.setattr(export, "fetch_known_symbols", fetch_known)
    monkeypatch.setattr(
        export, "fetch_exchange_map", lambda tickers: {t: "NMS" for t in tickers if t == "AAA"}
    )
    return calls


class TestWatchlist:
    def test_keeps_known_symbols_in_order_with_prefixes(self, known):
        text, dropped = export._watchlist(("TCHX", "ZZZ", "AAA"))
        assert text == "TCHX\nNASDAQ:AAA"
        assert dropped == 1


def _export_script() -> None:
    from stock_dashboard.ui.export import tradingview_export

    tradingview_export(["AAA", "ZZZ", "AAA", None], filename="w.txt", key="k")


def _export_unknown_script() -> None:
    from stock_dashboard.ui.export import tradingview_export

    tradingview_export(["ZZZ"], filename="w.txt", key="k")


def _export_empty_script() -> None:
    import streamlit as st

    from stock_dashboard.ui.export import tradingview_export

    tradingview_export([], filename="w.txt", key="k")
    st.write("done")


class TestTradingviewExport:
    def test_download_button_for_known_symbols(self, known):
        at = AppTest.from_function(_export_script, default_timeout=TIMEOUT).run()
        assert not at.exception
        assert len(at.get("download_button")) == 1
        assert known == [("AAA", "ZZZ")]  # deduplicated, non-strings dropped
        assert "1 symbol(s) unknown" in at.caption[0].value

    def test_nothing_to_export(self, known):
        at = AppTest.from_function(_export_unknown_script, default_timeout=TIMEOUT).run()
        assert not at.get("download_button")
        assert "nothing to export" in at.caption[0].value

    def test_no_tickers_renders_nothing(self, known):
        at = AppTest.from_function(_export_empty_script, default_timeout=TIMEOUT).run()
        assert not at.get("download_button")
        assert not at.caption
        assert known == []


def _preset_script() -> None:
    import streamlit as st

    from stock_dashboard.ui.presets import preset_selectbox

    st.write(f"chosen={preset_selectbox('Preset', key='p')}")


class TestPresetSelectbox:
    def test_none_by_default_then_the_chosen_stem(self):
        at = AppTest.from_function(_preset_script, default_timeout=TIMEOUT).run()
        box = at.selectbox(key="p")
        assert box.options[0] == "(none)"
        assert "sample rs leaders" in box.options
        assert at.markdown[0].value == "chosen=None"
        box.set_value("sample_rs_leaders").run()
        assert at.markdown[0].value == "chosen=sample_rs_leaders"

    def test_hidden_without_presets(self, monkeypatch):
        from stock_dashboard.ui import presets

        monkeypatch.setattr(presets, "list_presets", list)
        at = AppTest.from_function(_preset_script, default_timeout=TIMEOUT).run()
        assert not at.selectbox
        assert at.markdown[0].value == "chosen=None"


def test_rs_figure_draws_the_moving_averages_present():
    metrics = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=3),
            "relative_strength": [1.0, 1.1, 1.2],
            "MA150": [None, 1.05, 1.1],
        }
    )
    fig = rs_figure(metrics, "title")
    assert [t.name for t in fig.data] == ["Relative Strength", "150-day MA"]
    assert fig.layout.title.text == "title"
    # Above the plot, so it doesn't cover the date labels under it.
    assert fig.layout.legend.yanchor == "bottom"
    assert fig.layout.legend.y > 1


class TestColumnLabels:
    @pytest.mark.parametrize(
        ("name", "label"),
        [
            ("rs_rank_index", "RS Rank in ETF %"),
            ("signal_label", "Signal"),
            ("vannaxoi", "vannaxoi"),  # exposure columns keep their documented names
            ("some_new_column", "Some new column"),
            ("plain", "Plain"),
            ("RSI Z", "RSI Z"),  # already a label
        ],
    )
    def test_column_label(self, name, label):
        assert column_label(name) == label

    def test_known_labels_are_readable(self):
        assert all("_" not in label for label in COLUMN_LABELS.values())

    def test_column_config_labels_every_column_and_formats_some(self):
        config = column_config(["ticker", "rsi", "x"], {"rsi": "%.1f"}, {"x": "Custom"})
        assert config["ticker"]["label"] == "Ticker"
        assert config["rsi"]["type_config"]["format"] == "%.1f"
        assert config["x"]["label"] == "Custom"
        assert "type_config" not in config["ticker"] or not config["ticker"]["type_config"]

    def test_column_config_shows_date_columns_without_a_time(self):
        config = column_config(["collection_date", "x"], dates={"collection_date"})
        assert config["collection_date"]["type_config"]["format"] == "YYYY-MM-DD"
        assert config["collection_date"]["label"] == "Date"


def _table_script() -> None:
    import pandas as pd

    from stock_dashboard.ui.tables import show_table

    show_table(
        pd.DataFrame({"symbol": ["A", "B"], "rs_z_score": [1.0, None]}),
        formats={"rs_z_score": "%.2f"},
    )


def test_show_table_labels_columns_and_blanks_missing_values():
    import json

    at = AppTest.from_function(_table_script, default_timeout=TIMEOUT).run()
    element = at.dataframe[0]
    config = json.loads(element.proto.columns)
    assert config["symbol"]["label"] == "Symbol"
    assert config["rs_z_score"]["label"] == "RS Z-Score"
    assert element.proto.placeholder == ""


class TestSelectionHelpers:
    def test_key_changes_with_rows_and_their_order(self):
        assert selection_key("t", ["A", "B"]) == selection_key("t", ["A", "B"])
        assert selection_key("t", ["A", "B"]) != selection_key("t", ["B", "A"])
        assert selection_key("t", ["A", "B"]) != selection_key("t", ["A"])
        assert selection_key("t", ["A"]).startswith("t_")

    @pytest.mark.parametrize(
        ("rows", "n_rows", "expected"), [([2], 5, 2), ([5], 5, None), ([], 5, None), ([0], 0, None)]
    )
    def test_selected_row_is_clamped(self, rows, n_rows, expected):
        event = SimpleNamespace(selection=SimpleNamespace(rows=rows))
        assert selected_row(event, n_rows) == expected
        assert selected_row(None, n_rows) is None


def _selection_script() -> None:
    import pandas as pd
    import streamlit as st

    from stock_dashboard.ui.tables import selected_row, selection_key, show_table

    n = st.number_input("n", 1, 50, 10)
    df = pd.DataFrame({"t": [f"T{i}" for i in range(int(n))]})
    event = show_table(
        df, on_select="rerun", selection_mode="single-row", key=selection_key("tbl", df["t"])
    )
    row = selected_row(event, len(df))
    st.selectbox("pick", df["t"].tolist(), index=row if row is not None else 0)


def test_stale_selection_resets_when_the_rows_change():
    """A browser keeps sending the old selection; it must not index the new rows."""
    at = AppTest.from_function(_selection_script, default_timeout=TIMEOUT).run()
    key = selection_key("tbl", [f"T{i}" for i in range(10)])
    at.session_state[key] = {"selection": {"rows": [9], "columns": []}}
    at.run()
    assert at.selectbox[0].value == "T9"
    at.number_input[0].set_value(3).run()
    assert not at.exception
    assert at.selectbox[0].value == "T0"


def _load_preset_script() -> None:
    import streamlit as st

    from stock_dashboard.ui.presets import load_preset

    st.markdown(f"loaded={load_preset('broken') is not None}")


def test_load_preset_shows_an_error_for_an_unreadable_file(tmp_path, monkeypatch):
    (tmp_path / "broken.json").write_text('{"name": "cut')
    monkeypatch.setenv("PRESET_DIR", str(tmp_path))
    get_settings.cache_clear()
    at = AppTest.from_function(_load_preset_script, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert "Cannot read preset 'broken'" in at.error[0].value
    assert at.markdown[0].value == "loaded=False"


class TestBenchmarkRegimes:
    def test_classifies_the_benchmark(self, monkeypatch):
        from stock_dashboard.ui import presets

        dates = pd.bdate_range("2024-01-01", periods=260)
        bars = pd.DataFrame({"date": dates, "close": np.linspace(100, 200, 260)})
        monkeypatch.setattr(presets, "fetch_ohlcv", lambda kind, tickers: bars)
        presets.benchmark_regimes.clear()
        regimes = presets.benchmark_regimes("MKT")
        assert regimes is not None
        assert regimes.iloc[-1] == "bull"

    def test_none_without_bars(self, monkeypatch):
        from stock_dashboard.ui import presets

        monkeypatch.setattr(presets, "fetch_ohlcv", lambda kind, tickers: pd.DataFrame())
        presets.benchmark_regimes.clear()
        assert presets.benchmark_regimes("NOPE") is None


def _raising_script(error: str) -> None:
    import psycopg2
    import psycopg2.errors

    from stock_dashboard.ui.errors import database_errors

    errors = {
        "undefined_table": psycopg2.errors.UndefinedTable('relation "dim_stock" does not exist'),
        "operational": psycopg2.OperationalError("connection refused"),
    }
    with database_errors():
        raise errors[error]


def test_missing_table_says_to_run_rs_pipeline():
    at = AppTest.from_function(_raising_script, args=("undefined_table",), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert "run rs-pipeline first" in at.error[0].value
    assert 'relation "dim_stock" does not exist' in at.error[0].value
    assert "(https://github.com/scottmcghee/stock-dashboard#readme)" in at.error[0].value


def test_unreachable_database_names_the_server():
    at = AppTest.from_function(_raising_script, args=("operational",), default_timeout=TIMEOUT)
    at.run()
    assert "Cannot reach the database on localhost:5432" in at.error[0].value
    assert "/blob/main/.env.example)" in at.error[0].value  # a link, not a bare file name
    assert "connection refused" in at.exception[0].message  # st.exception shows the cause
