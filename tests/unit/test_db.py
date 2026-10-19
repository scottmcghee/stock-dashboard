"""Tests for db.py: pooling, read_df and option_data_available (no real database)."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import psycopg2
import psycopg2.extensions
import pytest

from stock_dashboard import db

DATE_OID = psycopg2.extensions.DATE.values[0]  # noqa: PD011
TEXT_OID = 25
FLOAT_OID = 701


@pytest.fixture
def mock_conn(monkeypatch):
    """A pooled connection whose cursor returns whatever the test sets on it."""
    conn = MagicMock()
    conn.closed = 0
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.description = [("symbol", TEXT_OID), ("value", FLOAT_OID)]
    cursor.fetchall.return_value = [("AAA", 1.5), ("BBB", None)]
    pool = MagicMock()
    pool.getconn.return_value = conn
    monkeypatch.setattr(db, "_get_pool", lambda database: pool)
    monkeypatch.setattr(psycopg2.extensions, "register_type", MagicMock())
    conn.pool = pool
    conn.cur = cursor
    return conn


class TestConnection:
    def test_lends_and_returns_connection(self, mock_conn):
        with db.connection(db.Database.STOCKS) as conn:
            assert conn is mock_conn
        mock_conn.pool.putconn.assert_called_once_with(mock_conn, close=False)

    def test_returns_connection_on_exception(self, mock_conn):
        with pytest.raises(ValueError, match="boom"), db.connection(db.Database.OPTIONS):
            raise ValueError("boom")
        mock_conn.pool.putconn.assert_called_once()

    def test_discards_closed_connection(self, mock_conn):
        with db.connection(db.Database.STOCKS):
            mock_conn.closed = 2
        mock_conn.pool.putconn.assert_called_once_with(mock_conn, close=True)

    def test_registers_numeric_as_float(self, mock_conn):
        with db.connection(db.Database.STOCKS):
            pass
        psycopg2.extensions.register_type.assert_called_once_with(db._NUMERIC_AS_FLOAT, mock_conn)

    def test_numeric_caster_returns_float(self):
        assert db._NUMERIC_AS_FLOAT("12.50", None) == 12.5
        assert db._NUMERIC_AS_FLOAT(None, None) is None


class TestReadDf:
    def test_builds_frame_from_cursor(self, mock_conn):
        df = db.read_df(db.Database.STOCKS, "SELECT symbol, value FROM t WHERE x = %s", (1,))
        assert list(df.columns) == ["symbol", "value"]
        assert df["symbol"].tolist() == ["AAA", "BBB"]
        assert df["value"].iloc[0] == 1.5
        assert pd.isna(df["value"].iloc[1])
        mock_conn.cur.execute.assert_called_once_with(
            "SELECT symbol, value FROM t WHERE x = %s", (1,)
        )

    def test_ends_transaction(self, mock_conn):
        db.read_df(db.Database.STOCKS, "SELECT 1")
        mock_conn.rollback.assert_called_once()

    def test_ends_transaction_when_query_fails(self, mock_conn):
        mock_conn.cur.execute.side_effect = psycopg2.ProgrammingError("bad sql")
        with pytest.raises(psycopg2.ProgrammingError):
            db.read_df(db.Database.STOCKS, "SELEC 1")
        mock_conn.rollback.assert_called_once()
        mock_conn.pool.putconn.assert_called_once()

    def test_skips_rollback_on_dead_connection(self, mock_conn):
        def die(*args):
            mock_conn.closed = 2
            raise psycopg2.OperationalError("server closed the connection")

        mock_conn.cur.execute.side_effect = die
        with pytest.raises(psycopg2.OperationalError):
            db.read_df(db.Database.STOCKS, "SELECT 1")
        mock_conn.rollback.assert_not_called()
        mock_conn.pool.putconn.assert_called_once_with(mock_conn, close=True)

    def test_converts_date_columns(self, mock_conn):
        mock_conn.cur.description = [("date", DATE_OID)]
        mock_conn.cur.fetchall.return_value = [(date(2025, 1, 2),), (date(2025, 1, 3),)]
        df = db.read_df(db.Database.STOCKS, "SELECT date FROM t")
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_all_null_numeric_column_is_float(self, mock_conn):
        mock_conn.cur.fetchall.return_value = [("AAA", None), ("BBB", None)]
        df = db.read_df(db.Database.OPTIONS, "SELECT symbol, value FROM t")
        assert df["value"].dtype == float
        assert df["value"].isna().all()

    def test_empty_result_keeps_columns(self, mock_conn):
        mock_conn.cur.fetchall.return_value = []
        df = db.read_df(db.Database.STOCKS, "SELECT symbol, value FROM t")
        assert df.empty
        assert list(df.columns) == ["symbol", "value"]


class TestOptionDataAvailable:
    def test_true_when_all_tables_exist(self, mock_conn):
        mock_conn.cur.description = [("count", 20)]
        mock_conn.cur.fetchall.return_value = [(3,)]
        assert db.option_data_available() is True
        _, params = mock_conn.cur.execute.call_args.args
        assert sorted(params[0]) == sorted(db.OPTION_CONTRACT_TABLES)

    def test_false_when_tables_missing(self, mock_conn):
        mock_conn.cur.description = [("count", 20)]
        mock_conn.cur.fetchall.return_value = [(2,)]
        assert db.option_data_available() is False

    def test_false_when_database_unreachable(self, monkeypatch):
        def unreachable(database):
            raise psycopg2.OperationalError('database "option_data" does not exist')

        monkeypatch.setattr(db, "_get_pool", unreachable)
        assert db.option_data_available() is False


class TestPools:
    def test_pool_per_database_uses_settings(self, monkeypatch):
        monkeypatch.setenv("OPTION_DB_NAME", "opts")
        db.get_settings.cache_clear()
        created = []

        def fake_pool(minconn, maxconn, **kwargs):
            pool = MagicMock(closed=False)
            created.append(kwargs)
            return pool

        monkeypatch.setattr(db, "ThreadedConnectionPool", fake_pool)
        monkeypatch.setattr(db, "_pools", {})
        stocks = db._get_pool(db.Database.STOCKS)
        assert db._get_pool(db.Database.STOCKS) is stocks
        db._get_pool(db.Database.OPTIONS)
        assert [kw["dbname"] for kw in created] == ["stocks", "opts"]

    def test_close_all_pools(self, monkeypatch):
        open_pool, closed_pool = MagicMock(closed=False), MagicMock(closed=True)
        pools = {db.Database.STOCKS: open_pool, db.Database.OPTIONS: closed_pool}
        monkeypatch.setattr(db, "_pools", pools)
        db.close_all_pools()
        open_pool.closeall.assert_called_once()
        closed_pool.closeall.assert_not_called()
        assert pools == {}
