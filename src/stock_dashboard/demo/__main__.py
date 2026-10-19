"""``stock-dashboard-seed-demo``: fill the option database from the demo market.

Run after rs-pipeline's demo has filled the stocks database. Creates the option
database if it doesn't exist, then generates option data from the stored
closes and replaces the generated symbols' rows.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import psycopg2
from psycopg2 import sql

from stock_dashboard.config import Settings, get_settings
from stock_dashboard.demo.load import read_closes, write_option_data
from stock_dashboard.demo.options import generate_option_data

logger = logging.getLogger(__name__)

# Server-wide database every PostgreSQL installation has; used to create the option database.
MAINTENANCE_DB = "postgres"


def _connect(settings: Settings, dbname: str) -> psycopg2.extensions.connection:
    kwargs: dict[str, Any] = settings.db_kwargs(dbname)
    conn: psycopg2.extensions.connection = psycopg2.connect(**kwargs)
    return conn


def ensure_database(settings: Settings, dbname: str) -> None:
    """Create *dbname* on the configured server unless it exists."""
    conn = _connect(settings, MAINTENANCE_DB)
    conn.autocommit = True  # CREATE DATABASE can't run in a transaction
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone() is None:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
                logger.info("created database %s", dbname)
    finally:
        conn.close()


def seed_demo(settings: Settings) -> None:
    """Generate option data for every ticker in the stocks database and store it."""
    with _connect(settings, settings.stocks_db) as stocks:
        closes = read_closes(stocks)
    stocks.close()
    if closes.empty:
        raise SystemExit("the stocks database has no prices; run rs-pipeline's demo first")

    frames = generate_option_data(closes)
    ensure_database(settings, settings.option_db)
    options = _connect(settings, settings.option_db)
    try:
        write_option_data(options, frames)
    finally:
        options.close()
    logger.info(
        "stored option data: %d chain rows, %d underlyings, latest %s",
        len(frames.option_chain_data),
        frames.underlying_data["symbol"].nunique(),
        frames.underlying_data["collection_date"].max().date(),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        seed_demo(get_settings())
    except psycopg2.Error as exc:
        logger.error("seeding failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
