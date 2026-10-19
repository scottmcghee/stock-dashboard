"""Read demo inputs from the stocks database and write the option-data tables.

Both functions take an open psycopg2 connection, so callers choose how to
connect (the demo seeding script, or a throwaway container in tests).
"""

import io
import logging
from importlib.resources import files

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
from psycopg2 import sql
from psycopg2.extensions import connection as Connection

from stock_dashboard.demo.contract import DATE, TABLES
from stock_dashboard.demo.options import OptionFrames

logger = logging.getLogger(__name__)

SCHEMA_PACKAGE = "stock_dashboard.schema"
SCHEMA_FILE = "option_data.sql"

_CLOSES_QUERY = """
SELECT date, ticker, close FROM fact_commons_ohlc
UNION ALL
SELECT date, ticker, close FROM fact_index_ohlc
ORDER BY ticker, date
"""


def option_data_ddl() -> str:
    """The contract DDL shipped in the stock_dashboard.schema package."""
    return files(SCHEMA_PACKAGE).joinpath(SCHEMA_FILE).read_text(encoding="utf-8")


def read_closes(conn: Connection) -> pd.DataFrame:
    """Daily closes (date, ticker, close) for every stock and index in the stocks DB."""
    with conn.cursor() as cur:
        cur.execute(_CLOSES_QUERY)
        rows = cur.fetchall()
    conn.rollback()  # read-only; don't leave the session idle in a transaction
    frame = pd.DataFrame(rows, columns=["date", "ticker", "close"])
    frame["close"] = frame["close"].astype(float)
    return frame


def write_option_data(conn: Connection, frames: OptionFrames) -> None:
    """Create the contract tables if needed and replace the generated symbols' rows.

    Idempotent: for each table, rows of the symbols present in `frames` are
    deleted and the new rows are bulk-loaded with COPY; other symbols' rows are
    left alone. Runs in one transaction, committed on success.
    """
    with conn, conn.cursor() as cur:
        cur.execute(option_data_ddl())
        for table, df in frames._asdict().items():
            columns = list(TABLES[table])
            symbols = sorted(df["symbol"].unique().tolist())
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE symbol = ANY(%s)").format(sql.Identifier(table)),
                (symbols,),
            )
            buffer = _to_csv(df, TABLES[table])
            copy = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, NULL '')").format(
                sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, columns))
            )
            cur.copy_expert(copy, buffer)
            logger.info("Loaded %d rows into %s", len(df), table)


def _to_csv(df: pd.DataFrame, columns: dict[str, str]) -> io.BytesIO:
    """CSV for COPY: dates as YYYY-MM-DD, NULL/NaN as empty. Arrow is ~10x faster than pandas."""
    table = pa.Table.from_pandas(df[list(columns)], preserve_index=False)
    for i, (name, sql_type) in enumerate(columns.items()):
        if sql_type == DATE:
            table = table.set_column(i, name, table[name].cast(pa.date32()))
    buffer = io.BytesIO()
    pa_csv.write_csv(table, buffer, pa_csv.WriteOptions(include_header=False))
    buffer.seek(0)
    return buffer
