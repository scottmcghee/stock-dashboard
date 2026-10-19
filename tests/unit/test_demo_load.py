"""Tests for stock_dashboard.demo.load with a fake psycopg2 connection."""

import io
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from psycopg2 import sql

from stock_dashboard.demo.contract import TABLES
from stock_dashboard.demo.load import option_data_ddl, read_closes, write_option_data
from stock_dashboard.demo.options import OptionDemoConfig, generate_option_data


def _fake_connection() -> tuple[MagicMock, MagicMock]:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    conn.__enter__.return_value = conn
    return conn, cur


def _closes() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range(end="2026-10-16", periods=60)
    return pd.concat(
        pd.DataFrame(
            {
                "date": dates,
                "ticker": t,
                "close": 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 60))),
            }
        )
        for t in ("AAA", "BBB")
    )


def test_ddl_is_packaged_and_idempotent() -> None:
    ddl = option_data_ddl()
    creates = [line for line in ddl.splitlines() if line.startswith("CREATE")]
    assert creates
    assert all("IF NOT EXISTS" in line for line in creates)
    for table in TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {table} (" in ddl


def test_write_option_data_replaces_symbols_and_copies_every_table() -> None:
    frames = generate_option_data(_closes(), OptionDemoConfig(sessions=40))
    conn, cur = _fake_connection()
    write_option_data(conn, frames)

    executed = [c.args for c in cur.execute.call_args_list]
    assert executed[0] == (option_data_ddl(),)
    deletes = executed[1:]
    assert len(deletes) == len(TABLES)
    for statement, params in deletes:
        assert isinstance(statement, sql.Composed)
        assert params == (["AAA", "BBB"],)

    copies = cur.copy_expert.call_args_list
    assert len(copies) == len(TABLES)
    for call, (table, df) in zip(copies, frames._asdict().items(), strict=True):
        buffer = call.args[1]
        assert isinstance(buffer, io.BytesIO)
        parsed = pd.read_csv(buffer, header=None, names=list(TABLES[table]))
        assert len(parsed) == len(df)
        date_column = next(c for c, t in TABLES[table].items() if t == "date")
        assert parsed[date_column].iloc[0] == df[date_column].iloc[0].strftime("%Y-%m-%d")
        assert parsed.isna().sum().sum() == df.isna().sum().sum()
    conn.__exit__.assert_called_once()


def test_read_closes_ends_its_transaction() -> None:
    conn, cur = _fake_connection()
    cur.fetchall.return_value = [("2026-10-16", "AAA", 12.5)]
    frame = read_closes(conn)
    assert frame.to_dict("records") == [{"date": "2026-10-16", "ticker": "AAA", "close": 12.5}]
    conn.rollback.assert_called_once()
