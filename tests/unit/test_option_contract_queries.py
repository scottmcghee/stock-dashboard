"""Tests for queries/option_contract_queries.py: read_df is faked."""

import pandas as pd

from stock_dashboard.queries import option_contract_queries as ocq

OHLC_COLUMNS = ["collection_date", "open", "high", "low", "close", "oi", "day_volume"]


def ohlc(**overrides):
    base = {
        "collection_date": pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
        "open": [1.20, 1.25, 1.30],
        "high": [1.35, 1.40, 1.45],
        "low": [1.15, 1.20, 1.25],
        "close": [1.30, 1.35, 1.42],
        "oi": [500, 520, 530],
        "day_volume": [100, 150, 80],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_get_tickers_with_option_data(fake_db):
    fake_db.returns(pd.DataFrame({"symbol": ["AAA", "SPY"]}))
    assert ocq.get_tickers_with_option_data() == ["AAA", "SPY"]
    assert fake_db.last.database == "options"


class TestGetExpirations:
    def test_returns_timestamps(self, fake_db):
        fake_db.returns(
            pd.DataFrame({"expiration_date": pd.to_datetime(["2026-05-16", "2026-06-20"])})
        )
        result = ocq.get_expirations("AAA")
        assert result == [pd.Timestamp("2026-05-16"), pd.Timestamp("2026-06-20")]
        assert fake_db.last.params == {"symbol": "AAA"}

    def test_empty(self, fake_db):
        fake_db.returns(pd.DataFrame({"expiration_date": []}))
        assert ocq.get_expirations("NONE") == []


class TestFetchOiByStrike:
    def test_latest_snapshot_in_one_query(self, fake_db):
        snapshot = pd.DataFrame(
            {
                "strike": [100.0, 100.0],
                "option_type": ["call", "put"],
                "oi": [5, 3],
                "iv": [0.3, 0.31],
            }
        )
        fake_db.returns(snapshot)
        result = ocq.fetch_oi_by_strike("AAA", "2026-05-16")
        assert len(result) == 2
        assert len(fake_db.calls) == 1
        assert (
            "collection_date = ( SELECT MAX(collection_date) FROM option_chain_data"
            in fake_db.last.sql
        )
        assert fake_db.last.sql.startswith("SELECT collection_date, strike")
        assert fake_db.last.params == {"symbol": "AAA", "expiration": "2026-05-16"}


class TestFetchContractOhlcHistory:
    def test_returns_frame_and_binds_contract(self, fake_db):
        fake_db.returns(ohlc())
        result = ocq.fetch_contract_ohlc_history("AAA", "2026-05-16", 100.0, "call")
        assert len(result) == 3
        assert fake_db.last.params == {
            "symbol": "AAA",
            "expiration": "2026-05-16",
            "strike": 100.0,
            "option_type": "call",
        }
        assert "price AS close" in fake_db.last.sql

    def test_drops_rows_without_usable_close(self, fake_db):
        fake_db.returns(ohlc(close=[1.30, None, 0.0]))
        result = ocq.fetch_contract_ohlc_history("AAA", "2026-05-16", 100.0, "call")
        assert result["close"].tolist() == [1.30]

    def test_fills_missing_ohl_with_close(self, fake_db):
        fake_db.returns(ohlc(open=[0.0, 1.0, 1.0], high=[None, 1.0, 1.0], low=[0.0, 1.0, 1.0]))
        row = ocq.fetch_contract_ohlc_history("AAA", "2026-05-16", 100.0, "call").iloc[0]
        assert (row["open"], row["high"], row["low"]) == (1.30, 1.30, 1.30)

    def test_empty(self, fake_db):
        fake_db.returns(pd.DataFrame(columns=OHLC_COLUMNS))
        assert ocq.fetch_contract_ohlc_history("AAA", "2026-05-16", 100.0, "call").empty


def test_fetch_oi_weights_over_time(fake_db):
    weights = pd.DataFrame(
        {
            "collection_date": pd.to_datetime(["2026-04-01", "2026-04-02"]),
            "strike": [100.0, 100.0],
            "option_type": ["call", "call"],
            "weight": [500.0, 520.0],
        }
    )
    fake_db.returns(weights)
    result = ocq.fetch_oi_weights_over_time("AAA", "2026-05-16")
    assert len(result) == 2
    assert "COALESCE(oi, 0)::float AS weight" in fake_db.last.sql
    assert fake_db.last.params == {"symbol": "AAA", "expiration": "2026-05-16"}


class TestDefaults:
    def test_most_open_interest_symbol(self, fake_db):
        fake_db.returns(pd.DataFrame({"symbol": ["AAA"]}))
        assert ocq.fetch_most_open_interest_symbol() == "AAA"
        assert "ORDER BY SUM(COALESCE(oi, 0)) DESC" in fake_db.last.sql
        fake_db.returns(pd.DataFrame(columns=["symbol"]))
        assert ocq.fetch_most_open_interest_symbol() is None

    def test_latest_collection_date(self, fake_db):
        fake_db.returns(pd.DataFrame({"latest": [pd.Timestamp("2026-10-16")]}))
        assert ocq.get_latest_collection_date("AAA") == pd.Timestamp("2026-10-16")
        assert fake_db.last.params == {"symbol": "AAA"}
        fake_db.returns(pd.DataFrame({"latest": [None]}))
        assert ocq.get_latest_collection_date("ZZZ") is None
