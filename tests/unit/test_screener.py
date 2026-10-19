"""Tests for logic/screener.py: the screener page's pure steps."""

import pandas as pd
import pytest

from stock_dashboard.logic.lookback import screener_lookback_bars
from stock_dashboard.logic.screener import (
    RANK_COLUMNS,
    RESULT_FORMATS,
    TaFilters,
    build_results_table,
    intersect_prefilters,
    option_signal_table,
    percent_to_fraction,
    tickers_with_events,
)


def test_percent_to_fraction():
    assert percent_to_fraction(None) is None
    assert percent_to_fraction(75) == pytest.approx(0.75)
    assert percent_to_fraction(0.0) == 0.0


class TestIntersectPrefilters:
    def test_no_prefilter_means_no_restriction(self):
        assert intersect_prefilters([]) is None

    def test_intersection_is_sorted(self):
        assert intersect_prefilters([{"B", "A", "C"}, {"C", "A"}]) == ["A", "C"]

    def test_empty_intersection_is_an_empty_list(self):
        assert intersect_prefilters([{"A"}, {"B"}]) == []


class TestTaFilters:
    def test_inactive_by_default_even_with_settings(self):
        assert not TaFilters(momentum_period=20, timeframe="W", gap_lookback=3).active

    @pytest.mark.parametrize(
        "field",
        [
            "rsi_min",
            "rsi_max",
            "macd_crossover",
            "ma_touch_period",
            "adr_min",
            "gap_type",
            "momentum_min",
            "momentum_max",
            "ratio_to_ma_min",
            "ratio_to_ma_max",
            "dist_low_126_min_pct",
            "dist_low_126_max_pct",
        ],
    )
    def test_any_filter_activates(self, field):
        value = "bull" if field in ("macd_crossover", "gap_type") else 1
        assert TaFilters(**{field: value}).active

    def test_kwargs_convert_distance_percent_to_fraction(self):
        kwargs = TaFilters(rsi_min=30.0, dist_low_126_min_pct=10.0).apply_kwargs()
        assert kwargs["rsi_min"] == 30.0
        assert kwargs["dist_recent_low_126_min"] == pytest.approx(0.10)
        assert kwargs["dist_recent_low_126_max"] is None
        assert kwargs["timeframe"] == "D"

    def test_kwargs_match_apply_ta_filters_signature(self):
        import inspect

        from stock_dashboard.logic.technical import apply_ta_filters

        params = set(inspect.signature(apply_ta_filters).parameters) - {"ohlcv_df", "tickers"}
        assert set(TaFilters().apply_kwargs()) == params

    def test_lookback_covers_events_and_patterns(self):
        filters = TaFilters(momentum_period=126, ratio_to_ma_period=200, timeframe="W")
        assert filters.lookback_bars() == screener_lookback_bars(126, 200, "W", None)
        assert filters.lookback_bars() >= screener_lookback_bars()


def test_tickers_with_events():
    flags = {
        "A": {"evt_regain_200ma": True, "evt_lose_200ma": False},
        "B": {"evt_regain_200ma": False},
        "C": {},
    }
    assert tickers_with_events(flags, ["evt_regain_200ma", "evt_lose_200ma"]) == {"A"}
    assert tickers_with_events(flags, []) == set()


class TestOptionSignalTable:
    def test_metrics_gex_bucket_and_scan_labels(self):
        metrics = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "option_volume": [10, 20],
                "vol_multiple": [1.5, None],
                "put_call_ratio": [0.7, 1.2],
                "dxoi": [100.0, -50.0],
                "gxoi": [-10.0, 20_000.0],
            }
        )
        scans = {
            "Vanna": pd.DataFrame({"symbol": ["B", "Z"], "signal_label": ["High Vanna", "Low"]}),
            "Flow": pd.DataFrame(),
        }
        table = option_signal_table(metrics, scans).set_index("symbol")
        assert list(table.index) == ["A", "B", "Z"]
        assert table.loc["A", "GEX"] == "Negative GEX"
        assert table.loc["B", "GEX"] == "High GEX"
        assert table.loc["B", "Vanna"] == "High Vanna"
        assert pd.isna(table.loc["A", "Vanna"])
        assert table.loc["A", "P/C Ratio"] == 0.7
        assert "Flow" not in table.columns

    def test_no_metrics(self):
        scans = {"Vanna": pd.DataFrame({"symbol": ["A"], "signal_label": ["Low Vanna"]})}
        table = option_signal_table(pd.DataFrame(), scans)
        assert table.to_dict("records") == [{"symbol": "A", "Vanna": "Low Vanna"}]


def _values(column: pd.Series) -> list:
    return [None if pd.isna(v) else v for v in column]


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "etf_ticker": ["TCHX", "FINX"],
            "etfs": ["SEMX, TCHX", "FINX"],
            "rs_rank_index": [0.9, 0.5],
            "rs_rank_overall": [0.8, 0.25],
            "rs_z_score": [1.5, -0.2],
            "volume_z_score": [2.0, 0.1],
            "avg_30d_volume": [1e6, 2e6],
            "yearly_std_volume": [1.0, 1.0],
        }
    )


class TestBuildResultsTable:
    def test_base_columns_with_percent_ranks(self):
        table = build_results_table(_results())
        assert list(table.columns) == [
            "Ticker",
            "ETF",
            "All ETFs",
            *RANK_COLUMNS,
            "RS Z-Score",
            "Volume Z-Score",
            "Avg 30d Volume",
        ]
        assert table[RANK_COLUMNS[0]].tolist() == pytest.approx([90.0, 50.0])
        assert table[RANK_COLUMNS[1]].tolist() == pytest.approx([80.0, 25.0])

    def test_stage_column(self):
        table = build_results_table(_results().assign(stage=["Stage 2", "Stage 4"]))
        assert table["Weinstein Stage"].tolist() == ["Stage 2", "Stage 4"]

    def test_ta_columns_only_when_they_hold_values(self):
        ta_info = {
            "AAA": {
                "rsi": 55.0,
                "macd_cross": None,
                "ma_touch": None,
                "adr": 3.0,
                "gap": {"type": "bull", "gap_pct": 2.5},
                "momentum": None,
                "ratio_to_ma": None,
                "dist_recent_low_126": 0.15,
            },
            "BBB": {"rsi": 40.0, "gap": None, "adr": None, "dist_recent_low_126": None},
        }
        table = build_results_table(_results(), ta_info=ta_info)
        assert "MACD Cross" not in table.columns
        assert "Momentum %" not in table.columns
        assert table["RSI"].tolist() == [55.0, 40.0]
        assert _values(table["Gap"]) == ["Bull 2.5%", None]
        assert table["Dist 126d Low %"].tolist()[0] == pytest.approx(15.0)

    def test_event_column_lists_fired_selected_events(self):
        flags = {"AAA": {"evt_regain_200ma": True, "evt_lose_200ma": False}, "BBB": {}}
        table = build_results_table(
            _results(), event_flags=flags, events=["evt_regain_200ma", "evt_lose_200ma"]
        )
        assert _values(table["TA Events"]) == ["Reclaimed 200-day MA", None]

    def test_option_columns_drop_all_empty_ones(self):
        options = pd.DataFrame(
            {"symbol": ["AAA"], "P/C Ratio": [0.9], "dxoi": [None], "Vanna": ["High Vanna"]}
        )
        table = build_results_table(_results(), option_table=options)
        assert _values(table["P/C Ratio"]) == [0.9, None]
        assert _values(table["Vanna"]) == ["High Vanna", None]
        assert "dxoi" not in table.columns

    def test_empty_option_table_adds_nothing(self):
        table = build_results_table(_results(), option_table=pd.DataFrame())
        assert "GEX" not in table.columns

    def test_pattern_column(self):
        patterns = {"AAA": [{"pattern": "bull_flag"}, {"pattern": "cup_and_handle"}]}
        table = build_results_table(_results(), pattern_info=patterns)
        assert _values(table["Patterns"]) == ["Bull Flag, Cup And Handle", None]

    def test_one_row_per_ticker_in_input_order(self):
        table = build_results_table(_results())
        assert table["Ticker"].tolist() == ["AAA", "BBB"]


def test_volumes_and_exposures_show_as_whole_numbers():
    """The "localized" format kept fractional digits, e.g. 1,338,390.767 shares."""
    assert RESULT_FORMATS["Avg 30d Volume"] == RESULT_FORMATS["dxoi"] == "%,.0f"
