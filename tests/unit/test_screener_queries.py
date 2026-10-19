"""Tests for queries/screener_queries.py: read_df is faked, SQL text and params are checked."""

import numpy as np
import pandas as pd
import pytest

from stock_dashboard.logic.events import EVENT_REQUIRED_BARS, latest_event_flags_bulk
from stock_dashboard.logic.lookback import screener_lookback_bars
from stock_dashboard.logic.trend_signals import TREND_REQUIRED_BARS
from stock_dashboard.queries import screener_queries as scq


@pytest.fixture
def benchmark_mkt(monkeypatch):
    monkeypatch.setenv("BENCHMARK_TICKER", "MKT")
    scq.get_settings.cache_clear()


class TestFetchEtfRsTickers:
    def test_returns_tickers(self, fake_db):
        fake_db.returns(pd.DataFrame({"ticker": ["AAA", "BBB"]}))
        assert scq.fetch_etf_rs_tickers(0.7) == {"AAA", "BBB"}

    def test_empty(self, fake_db):
        assert scq.fetch_etf_rs_tickers(0.99) == set()

    def test_filters_benchmark(self, fake_db, benchmark_mkt):
        """The old query read every ref_ticker's rows and MAX(date)."""
        scq.fetch_etf_rs_tickers(0.7)
        text = fake_db.last.sql
        assert "WHERE ref_ticker = %(benchmark)s AND rs_rank_index >= %(min_rank)s" in text
        assert "(SELECT MAX(date) FROM fact_etf_metrics WHERE ref_ticker = %(benchmark)s)" in text
        assert fake_db.last.params == {"benchmark": "MKT", "min_rank": 0.7}


class TestFetchOptionVolumeSpikeSymbols:
    def test_returns_symbols(self, fake_db):
        fake_db.returns(pd.DataFrame({"symbol": ["AAA", "TSLA"]}))
        assert scq.fetch_option_volume_spike_symbols(2.0) == {"AAA", "TSLA"}
        assert fake_db.last.params == {"min_multiple": 2.0}

    def test_average_row_from_same_date(self, fake_db):
        """The old join took MAX(date) of the average table independently."""
        scq.fetch_option_volume_spike_symbols(2.0)
        text = fake_db.last.sql
        assert "a.symbol = u.symbol AND a.date = u.collection_date" in text
        assert "MAX(date)" not in text


class TestFetchWeeklyOptionsSymbols:
    def test_returns_symbols(self, fake_db):
        fake_db.returns(pd.DataFrame({"symbol": ["AAA", "SPY"]}))
        assert scq.fetch_weekly_options_symbols() == {"AAA", "SPY"}
        assert fake_db.last.params == {
            "horizon": scq.WEEKLY_HORIZON_DAYS,
            "min_expirations": scq.WEEKLY_MIN_EXPIRATIONS,
        }

    def test_empty(self, fake_db):
        assert scq.fetch_weekly_options_symbols() == set()


class TestFetchTrackedTickers:
    def test_returns_list(self, fake_db):
        fake_db.returns(pd.DataFrame({"ticker": ["AAA", "BBB"]}))
        assert scq.fetch_tracked_tickers() == ["AAA", "BBB"]

    def test_empty(self, fake_db):
        assert scq.fetch_tracked_tickers() == []


class TestFetchOptionMetricsForTickers:
    def test_same_date_join_and_bound_tickers(self, fake_db):
        frame = pd.DataFrame({"symbol": ["AAA"], "option_volume": [10], "option_volume_avg": [5]})
        fake_db.returns(frame)
        assert len(scq.fetch_option_metrics_for_tickers(["AAA", "BBB"])) == 1
        text = fake_db.last.sql
        assert "a.symbol = u.symbol AND a.date = u.collection_date" in text
        assert "MAX(date)" not in text
        assert "u.symbol = ANY(%(tickers)s)" in text
        assert fake_db.last.params == {"tickers": ["AAA", "BBB"]}

    def test_empty_tickers(self, fake_db):
        assert scq.fetch_option_metrics_for_tickers([]).empty
        assert fake_db.calls == []


def trend_ohlcv(n=250, ticker="TEST"):
    """A steady uptrend followed by a high-volume selloff (a washout)."""
    rng = np.random.RandomState(42)
    close = 100.0 * np.cumprod(
        1 + np.concatenate([rng.normal(0.0005, 0.008, n - 50), rng.normal(-0.008, 0.015, 50)])
    )
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.bdate_range("2025-01-01", periods=n),
            "open": close * 1.001,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.concatenate(
                [rng.randint(800_000, 1_200_000, n - 50), rng.randint(1_500_000, 3_000_000, 50)]
            ),
        }
    )


class TestFetchSignalScanSymbols:
    def test_option_scans_share_one_history_and_intersect(self, monkeypatch):
        fetched = []
        monkeypatch.setattr(
            scq.signal_queries, "fetch_option_signal_history", lambda: fetched.append(1) or "H"
        )

        def scan(symbols):
            def run(*, history, enrich):
                assert history == "H"
                assert enrich is False
                return pd.DataFrame({"symbol": symbols})

            return run

        monkeypatch.setitem(scq.OPTION_SCANS, "vanna_extreme", scan(["AAA", "BBB", "CCC"]))
        monkeypatch.setitem(scq.OPTION_SCANS, "volume_spike", scan(["BBB", "CCC", "DDD"]))
        result = scq.fetch_signal_scan_symbols(["vanna_extreme", "volume_spike", "bogus"])
        assert result == {"BBB", "CCC"}
        assert fetched == [1]

    def test_empty_scan_short_circuits(self, monkeypatch):
        monkeypatch.setattr(scq.signal_queries, "fetch_option_signal_history", lambda: None)
        monkeypatch.setitem(scq.OPTION_SCANS, "vanna_extreme", lambda **kw: pd.DataFrame())
        assert scq.fetch_signal_scan_symbols(["vanna_extreme", "washout_bull"]) == set()

    def test_no_names(self):
        assert scq.fetch_signal_scan_symbols([]) == set()

    def test_trend_scan_fetches_required_bars(self, fake_db, monkeypatch):
        monkeypatch.setattr(scq, "fetch_tracked_tickers", lambda: ["TEST"])
        fake_db.returns(trend_ohlcv())
        assert scq.fetch_signal_scan_symbols(["washout_bull"]) == {"TEST"}
        assert fake_db.last.params["last_bars"] == TREND_REQUIRED_BARS
        assert fake_db.last.params["tickers"] == ["TEST"]

    def test_trend_scan_without_data(self, fake_db, monkeypatch):
        monkeypatch.setattr(scq, "fetch_tracked_tickers", lambda: ["TEST"])
        assert scq.fetch_signal_scan_symbols(["washout_bull"]) == set()

    def test_trend_scans_use_the_signal_scans_page_defaults(self, monkeypatch):
        """The pre-filter once used function defaults (bear 5% above the 50MA, not 6%)."""
        indicators = pd.DataFrame(
            {
                "symbol": ["BEAR5", "BEAR7", "COIL"],
                "rsi": [70.0, 70.0, 50.0],
                "mom_20": [10.0, 10.0, 0.0],
                "ratio_50ma": [1.055, 1.07, 1.0],
                "ratio_200ma": [1.3, 1.3, 0.87],  # COIL is 13% below: within the page's 15%
                "bb_width": [15.0, 15.0, 5.0],
                "adr_pct": [3.0, 3.0, 1.0],
                "vol_ratio": [1.0, 1.0, 1.0],
            }
        )
        monkeypatch.setattr(scq, "fetch_tracked_tickers", lambda: ["X"])
        monkeypatch.setattr(scq, "fetch_trend_indicators", lambda tickers: indicators)
        assert scq.fetch_signal_scan_symbols(["washout_bear"]) == {"BEAR7"}
        assert scq.fetch_signal_scan_symbols(["coiled_spring"]) == {"COIL"}

    def test_trend_scan_with_no_match(self, fake_db, monkeypatch):
        monkeypatch.setattr(scq, "fetch_tracked_tickers", lambda: ["TEST"])
        fake_db.returns(trend_ohlcv())
        assert scq.fetch_signal_scan_symbols(["coiled_spring"]) == set()


def test_fetch_trend_indicators_etf_table(fake_db):
    fake_db.returns(trend_ohlcv(ticker="TCHX"))
    out = scq.fetch_trend_indicators(["TCHX"], kind="etf")
    assert out["symbol"].tolist() == ["TCHX"]
    assert '"fact_index_ohlc"' in fake_db.last.sql


class TestScreenerLookback:
    """The screener's TA fetch must cover every indicator it feeds."""

    def test_covers_event_flags(self):
        """The old 300-calendar-day fetch gave ~204 bars, below the 210 the events need."""
        assert screener_lookback_bars() >= EVENT_REQUIRED_BARS

    def test_event_flags_defined_with_fetched_bars(self):
        bars = screener_lookback_bars()
        ohlcv = trend_ohlcv(n=bars)
        assert "TEST" in latest_event_flags_bulk(ohlcv, ["TEST"])
        too_short = ohlcv.iloc[-(EVENT_REQUIRED_BARS - 1) :]
        assert latest_event_flags_bulk(too_short, ["TEST"]) == {}

    def test_grows_with_periods_and_timeframe(self):
        assert screener_lookback_bars(ratio_to_ma_period=300) >= 300
        assert screener_lookback_bars(timeframe="W") >= 5 * 200
        assert screener_lookback_bars(timeframe="M") >= 21 * 200
