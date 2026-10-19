"""Tests for queries/stocks_queries.py: read_df is faked, SQL text and params are checked."""

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.logic.weinstein import WEINSTEIN_LOOKBACK_BARS
from stock_dashboard.queries import stocks_queries as sq


def dispatch(**frames):
    """Answer each query with the frame whose key appears in its SQL."""

    def answer(call):
        for key, frame in frames.items():
            if key in call.sql:
                return frame.copy()
        return pd.DataFrame()

    return answer


def bars(ticker="AAA", n=3):
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.bdate_range("2025-01-01", periods=n),
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 100,
        }
    )


@pytest.fixture
def benchmark_mkt(monkeypatch):
    monkeypatch.setenv("BENCHMARK_TICKER", "MKT")
    sq.get_settings.cache_clear()


class TestListings:
    def test_get_all_tickers(self, fake_db):
        fake_db.returns(pd.DataFrame({"ticker": ["AAA", "BBB"]}))
        assert sq.get_all_tickers() == ["AAA", "BBB"]
        assert "FROM dim_stock" in fake_db.last.sql

    def test_get_all_etfs(self, fake_db):
        fake_db.returns(pd.DataFrame({"etf_ticker": ["SEMX", "TCHX"]}))
        assert sq.get_all_etfs() == ["SEMX", "TCHX"]


class TestFetchOhlcv:
    def test_stock_table_and_bound_tickers(self, fake_db):
        fake_db.returns(bars())
        df = sq.fetch_ohlcv("stock", ["AAA"])
        assert len(df) == 3
        assert fake_db.last.database == "stocks"
        assert 'FROM "fact_commons_ohlc"' in fake_db.last.sql
        assert "ticker = ANY(%(tickers)s)" in fake_db.last.sql
        assert fake_db.last.params["tickers"] == ["AAA"]

    def test_etf_table(self, fake_db):
        sq.fetch_ohlcv("etf", ["MKT"])
        assert 'FROM "fact_index_ohlc"' in fake_db.last.sql

    def test_all_tickers_has_no_ticker_filter(self, fake_db):
        sq.fetch_ohlcv("stock")
        assert "ticker = ANY" not in fake_db.last.sql

    def test_empty_ticker_list_skips_query(self, fake_db):
        df = sq.fetch_ohlcv("stock", [], columns=("high", "low"))
        assert df.empty
        assert list(df.columns) == ["ticker", "date", "high", "low"]
        assert fake_db.calls == []

    def test_columns_selected(self, fake_db):
        sq.fetch_ohlcv("stock", ["AAA"], columns=("high", "low"))
        assert 'SELECT ticker, date, "high", "low" FROM' in fake_db.last.sql

    def test_unknown_column_rejected(self, fake_db):
        with pytest.raises(ValueError, match="Unknown OHLCV columns"):
            sq.fetch_ohlcv("stock", ["AAA"], columns=("close; DROP TABLE x",))

    def test_date_bounds_are_parameters(self, fake_db):
        sq.fetch_ohlcv("stock", ["AAA"], start=date(2025, 1, 1), end=date(2025, 6, 30))
        assert '"date" <= %(end)s' in fake_db.last.sql
        assert "date >= %(start)s" in fake_db.last.sql
        assert fake_db.last.params["start"] == date(2025, 1, 1)
        assert fake_db.last.params["end"] == date(2025, 6, 30)

    def test_last_bars_counts_rows_per_ticker(self, fake_db):
        """Look-backs are trading bars per ticker, not calendar days from NOW()."""
        sq.fetch_ohlcv("stock", ["AAA"], last_bars=210)
        text = fake_db.last.sql
        assert "ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC)" in text
        assert "rn <= %(last_bars)s" in text
        assert fake_db.last.params["last_bars"] == 210
        # The scan is bounded to recent sessions of the table itself.
        assert 'SELECT DISTINCT date FROM "fact_commons_ohlc"' in text
        assert fake_db.last.params["prefilter"] == sq.LAST_BARS_PREFILTER_FACTOR * 210

    def test_last_bars_respects_end(self, fake_db):
        sq.fetch_ohlcv("etf", ["MKT"], end=date(2025, 3, 1), last_bars=10)
        assert "WHERE date <= %(end)s ORDER BY date DESC LIMIT %(prefilter)s" in fake_db.last.sql

    def test_warmup_reaches_back_n_sessions_before_start(self, fake_db):
        sq.fetch_ohlcv("stock", start=date(2025, 1, 1), warmup_bars=252)
        text = fake_db.last.sql
        assert "date >= COALESCE((SELECT MIN(date) FROM (SELECT DISTINCT date" in text
        assert "WHERE date < %(start)s ORDER BY date DESC LIMIT %(warmup)s" in text
        assert fake_db.last.params["warmup"] == 252


class TestFetchSymbolOhlcv:
    def test_stock_symbol(self, fake_db):
        fake_db.returns(bars("AAA"))
        assert len(sq.fetch_symbol_ohlcv("AAA")) == 3
        assert len(fake_db.calls) == 1

    def test_falls_back_to_etf_table(self, fake_db):
        """ETF underlyings (e.g. the benchmark) only exist in fact_index_ohlc."""
        fake_db.returns(bars().iloc[0:0], bars("MKT"))
        df = sq.fetch_symbol_ohlcv("MKT", start=date(2025, 1, 1))
        assert df["ticker"].unique().tolist() == ["MKT"]
        assert 'FROM "fact_index_ohlc"' in fake_db.last.sql


class TestMetricsQueries:
    def test_fetch_stock_metrics(self, fake_db):
        sq.fetch_stock_metrics("AAA", date(2025, 1, 1), date(2025, 2, 1))
        assert "FROM fact_stock_metrics WHERE ticker = %(ticker)s" in fake_db.last.sql
        assert fake_db.last.params == {
            "ticker": "AAA",
            "start": date(2025, 1, 1),
            "end": date(2025, 2, 1),
        }

    def test_fetch_stock_metrics_without_dates(self, fake_db):
        sq.fetch_stock_metrics("AAA")
        assert "date" not in fake_db.last.params

    def test_etf_metrics_filter_benchmark(self, fake_db, benchmark_mkt):
        """Old query ignored ref_ticker, mixing benchmarks."""
        sq.fetch_etf_metrics("SEMX", date(2025, 1, 1))
        assert "ref_ticker = %(benchmark)s" in fake_db.last.sql
        assert fake_db.last.params["benchmark"] == "MKT"
        assert fake_db.last.params["etf"] == "SEMX"

    def test_rank_history_filters_benchmark(self, fake_db, benchmark_mkt):
        sq.fetch_all_etf_rank_history()
        assert "WHERE ref_ticker = %(benchmark)s ORDER BY" in fake_db.last.sql
        assert fake_db.last.params == {"benchmark": "MKT"}


class TestFetchLatestEtfGrid:
    def test_one_row_per_etf(self, fake_db, benchmark_mkt):
        """dim_sector/dim_industry map many names to one ETF; the old join fanned out."""
        sq.fetch_latest_etf_grid()
        text = fake_db.last.sql
        assert "SELECT etf_ticker, MIN(name) AS name FROM dim_industry GROUP BY etf_ticker" in text
        assert "SELECT etf_ticker, MIN(name) AS name FROM dim_sector GROUP BY etf_ticker" in text
        assert "COALESCE(s.name, i.name, e.etf_name)" in text
        assert "LEFT JOIN dim_industry" not in text

    def test_latest_date_and_rows_use_benchmark(self, fake_db, benchmark_mkt):
        sq.fetch_latest_etf_grid()
        text = fake_db.last.sql
        assert (
            "SELECT MAX(date) AS date FROM fact_etf_metrics WHERE ref_ticker = %(benchmark)s"
            in text
        )
        assert "WHERE f.ref_ticker = %(benchmark)s" in text
        assert fake_db.last.params == {"benchmark": "MKT"}


class TestFetchExchangeMap:
    def test_returns_codes_and_none_for_missing(self, fake_db):
        fake_db.returns(pd.DataFrame({"ticker": ["AAA", "BBB"], "exchange": ["NMS", None]}))
        assert sq.fetch_exchange_map(["AAA", "BBB", "CCC"]) == {
            "AAA": "NMS",
            "BBB": None,
            "CCC": None,
        }
        assert fake_db.last.params == {"tickers": ["AAA", "BBB", "CCC"]}

    def test_empty_list(self, fake_db):
        assert sq.fetch_exchange_map([]) == {}
        assert fake_db.calls == []


def per_etf_rows():
    """AAA is in two ETFs and ranks best in TCHX; BBB in one."""
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "etf_ticker": ["SEMX", "TCHX", "FINX"],
            "relative_strength": [1.1, 1.1, 0.9],
            "rs_rank_index": [0.40, 0.90, 0.50],
            "rs_rank_overall": [0.80, 0.80, 0.30],
            "rs_z_score": [0.5, 1.5, -0.2],
        }
    )


class TestCollapseEtfRows:
    def test_keeps_best_ranked_etf_and_lists_all(self):
        out = sq.collapse_etf_rows(per_etf_rows()).set_index("ticker")
        assert len(out) == 2
        assert out.loc["AAA", "etf_ticker"] == "TCHX"
        assert out.loc["AAA", "rs_z_score"] == 1.5
        assert out.loc["AAA", "etfs"] == "SEMX, TCHX"
        assert out.loc["BBB", "etfs"] == "FINX"

    def test_missing_rank_loses_to_ranked_row(self):
        rows = per_etf_rows()
        rows.loc[1, "rs_rank_index"] = np.nan
        out = sq.collapse_etf_rows(rows).set_index("ticker")
        assert out.loc["AAA", "etf_ticker"] == "SEMX"

    def test_tie_broken_by_etf_ticker(self):
        rows = per_etf_rows()
        rows.loc[0, "rs_rank_index"] = 0.90
        assert sq.collapse_etf_rows(rows).set_index("ticker").loc["AAA", "etf_ticker"] == "SEMX"

    def test_empty(self):
        assert "etfs" in sq.collapse_etf_rows(per_etf_rows().iloc[0:0]).columns


class TestEnrichWithRs:
    def test_one_row_per_symbol(self, fake_db):
        """Old code kept an arbitrary ETF row and ran without ETF ordering."""
        fake_db.returns(per_etf_rows())
        scan = pd.DataFrame({"symbol": ["AAA", "BBB", "ZZZ"], "signal_value": [1.0, 2.0, 3.0]})
        out = sq.enrich_with_rs(scan)
        assert len(out) == 3
        row = out.set_index("symbol").loc["AAA"]
        assert row["etf_ticker"] == "TCHX"
        assert row["rs_rank_index"] == 0.90
        assert row["etfs"] == "SEMX, TCHX"
        assert pd.isna(out.set_index("symbol").loc["ZZZ", "etf_ticker"])
        assert "m.date = (SELECT MAX(date) FROM fact_stock_metrics)" in fake_db.last.sql
        assert fake_db.last.params == {"tickers": ["AAA", "BBB", "ZZZ"]}

    def test_no_metrics_still_returns_scan(self, fake_db):
        fake_db.returns(per_etf_rows().iloc[0:0])
        scan = pd.DataFrame({"symbol": ["AAA"], "signal_value": [1.0]})
        out = sq.enrich_with_rs(scan)
        assert out["symbol"].tolist() == ["AAA"]
        assert "rs_rank_index" in out.columns

    def test_empty_or_symbolless_frame_passes_through(self, fake_db):
        assert sq.enrich_with_rs(pd.DataFrame()).empty
        frame = pd.DataFrame({"ticker": ["AAA"]})
        assert sq.enrich_with_rs(frame) is frame
        assert fake_db.calls == []


def volume_rows(tickers=("AAA", "BBB")):
    return pd.DataFrame(
        {
            "ticker": list(tickers),
            "yearly_avg_volume": [1000.0] * len(tickers),
            "yearly_std_volume": [100.0] * len(tickers),
            "avg_30d_volume": [1200.0] * len(tickers),
            "current_volume": [1300.0] * len(tickers),
        }
    )


@pytest.fixture
def screener_db(fake_db):
    fake_db.default = dispatch(fact_stock_metrics=per_etf_rows(), fact_commons_ohlc=volume_rows())
    return fake_db


class TestSearchStocksByMetrics:
    def test_one_row_per_ticker_with_etfs(self, screener_db):
        """Old screener returned a ticker once per ETF it belongs to."""
        df = sq.search_stocks_by_metrics()
        assert df["ticker"].tolist() == ["AAA", "BBB"]
        assert df.set_index("ticker").loc["AAA", "etfs"] == "SEMX, TCHX"
        assert df.set_index("ticker").loc["AAA", "etf_ticker"] == "TCHX"
        assert df["volume_z_score"].tolist() == [3.0, 3.0]

    def test_rank_filter_applies_to_best_etf_row(self, screener_db):
        df = sq.search_stocks_by_metrics(rs_rank_min=0.8)
        assert df["ticker"].tolist() == ["AAA"]
        assert sq.search_stocks_by_metrics(rs_rank_max=0.5)["ticker"].tolist() == ["BBB"]

    def test_other_filters(self, screener_db):
        assert sq.search_stocks_by_metrics(rs_rank_overall_min=0.5)["ticker"].tolist() == ["AAA"]
        assert sq.search_stocks_by_metrics(rs_z_score_max=0.0)["ticker"].tolist() == ["BBB"]
        assert sq.search_stocks_by_metrics(avg_volume_min=1500).empty
        assert len(sq.search_stocks_by_metrics(volume_std_dev_min=2.5)) == 2
        assert sq.search_stocks_by_metrics(volume_std_dev_min=3.5).empty

    def test_etf_filter_keeps_that_etfs_row(self, screener_db):
        df = sq.search_stocks_by_metrics(etf_ticker="SEMX")
        assert df["ticker"].tolist() == ["AAA"]
        assert df["etf_ticker"].iloc[0] == "SEMX"
        assert df["rs_rank_index"].iloc[0] == 0.40
        assert df["etfs"].iloc[0] == "SEMX, TCHX"

    def test_restrict_tickers_bound_as_array(self, screener_db):
        sq.search_stocks_by_metrics(restrict_tickers=["AAA"])
        metrics_call = screener_db.calls[0]
        assert "m.ticker = ANY(%(tickers)s)" in metrics_call.sql
        assert metrics_call.params == {"tickers": ["AAA"]}

    def test_empty_restriction_returns_nothing(self, screener_db):
        assert sq.search_stocks_by_metrics(restrict_tickers=[]).empty
        assert screener_db.calls == []

    def test_no_metrics(self, fake_db):
        assert sq.search_stocks_by_metrics().empty

    def test_volume_stats_anchor_on_bars_not_calendar(self, screener_db):
        sq.search_stocks_by_metrics()
        volume_call = screener_db.calls[1]
        text = volume_call.sql
        assert "NOW()" not in text
        assert "rn <= %(stats_bars)s" in text
        assert "FILTER (WHERE rn <= %(recent_bars)s)" in text
        assert "HAVING MAX(date) = (SELECT MAX(date) FROM fact_commons_ohlc)" in text
        assert volume_call.params["stats_bars"] == sq.VOLUME_STATS_BARS
        assert volume_call.params["tickers"] == ["AAA", "BBB"]

    def test_zero_std_gives_zero_z(self, fake_db):
        vol = volume_rows().assign(yearly_std_volume=0.0)
        fake_db.default = dispatch(fact_stock_metrics=per_etf_rows(), fact_commons_ohlc=vol)
        df = sq.search_stocks_by_metrics()
        assert df["volume_z_score"].tolist() == [0.0, 0.0]
        assert sq.search_stocks_by_metrics(volume_std_dev_min=-1.0).empty

    def test_weinstein_stage_fetches_lookback_bars(self, screener_db, monkeypatch):
        stages = {"AAA": "Stage 2", "BBB": "Stage 4"}
        monkeypatch.setattr(sq, "classify_bulk_stages", lambda df: stages)
        screener_db.default = dispatch(
            fact_stock_metrics=per_etf_rows(),
            last_bars=bars(),
            fact_commons_ohlc=volume_rows(),
        )
        df = sq.search_stocks_by_metrics(weinstein_stage="Stage 2")
        assert df["ticker"].tolist() == ["AAA"]
        assert df["stage"].tolist() == ["Stage 2"]
        assert screener_db.last.params["last_bars"] == WEINSTEIN_LOOKBACK_BARS


ROOT = Path(__file__).parents[2]
QUERY_SOURCES = [
    *sorted((ROOT / "src" / "stock_dashboard" / "queries").glob("*.py")),
    ROOT / "research" / "common.py",
]
WALL_CLOCK = re.compile(
    r"\bnow\s*\(|current_date|current_timestamp|localtimestamp|date\.today|datetime\.now",
    re.IGNORECASE,
)


@pytest.mark.parametrize("source", QUERY_SOURCES, ids=lambda p: p.name)
def test_no_query_uses_wall_clock(source):
    """'Latest' is anchored on the data, never NOW() or CURRENT_DATE: no query module
    (nor research/common.py) mentions the wall clock at all."""
    assert WALL_CLOCK.findall(source.read_text()) == []


def test_query_sources_cover_every_query_module():
    names = {p.name for p in QUERY_SOURCES}
    assert {"stocks_queries.py", "option_queries.py", "screener_queries.py", "common.py"} <= names
    assert len(names) >= 8


def test_latest_queries_sent_have_no_wall_clock(fake_db, benchmark_mkt):
    """The SQL actually sent, too (it may be composed at runtime)."""
    fake_db.default = dispatch(fact_stock_metrics=per_etf_rows(), fact_commons_ohlc=volume_rows())
    sq.search_stocks_by_metrics()
    sq.fetch_latest_etf_grid()
    sq.fetch_ohlcv("stock", ["AAA"], last_bars=5)
    sq.fetch_ohlcv("stock", start=date(2025, 1, 1), warmup_bars=5)
    for call in fake_db.calls:
        assert "NOW()" not in call.sql.upper()
        assert "CURRENT_DATE" not in call.sql.upper()


class TestFetchKnownSymbols:
    def test_stocks_etfs_and_the_benchmark(self, fake_db, benchmark_mkt):
        fake_db.returns(pd.DataFrame({"symbol": ["AAA", "TCHX"]}))
        assert sq.fetch_known_symbols(["AAA", "TCHX", "MKT", "ZZZ"]) == {"AAA", "TCHX", "MKT"}
        for table in ("dim_stock", "dim_etf", "dim_sector", "dim_industry"):
            assert table in fake_db.last.sql
        assert fake_db.last.params == {"symbols": ["AAA", "TCHX", "MKT", "ZZZ"]}

    def test_nothing_known(self, fake_db):
        fake_db.returns(pd.DataFrame(columns=["symbol"]))
        assert sq.fetch_known_symbols(["ZZZ"]) == set()

    def test_empty_input_skips_the_query(self, fake_db):
        assert sq.fetch_known_symbols([]) == set()
        assert fake_db.calls == []


class TestFetchTopRankedTicker:
    def test_highest_overall_rank_on_latest_date(self, fake_db):
        fake_db.returns(pd.DataFrame({"ticker": ["AAA"]}))
        assert sq.fetch_top_ranked_ticker() == "AAA"
        assert "ORDER BY rs_rank_overall DESC NULLS LAST, ticker LIMIT 1" in fake_db.last.sql
        assert "MAX(date)" in fake_db.last.sql

    def test_no_metrics(self, fake_db):
        fake_db.returns(pd.DataFrame(columns=["ticker"]))
        assert sq.fetch_top_ranked_ticker() is None
