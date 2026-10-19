"""The app's database error boundary: a message instead of a traceback."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg2
import psycopg2.errors
import streamlit as st

from stock_dashboard.config import ENV_EXAMPLE_URL, README_URL, get_settings


@contextmanager
def database_errors() -> Iterator[None]:
    """Show what failed when a page cannot reach its database or a table is missing.

    A missing table almost always means the stocks database was not filled by
    rs-pipeline (the option pages check their own tables before querying).
    """
    settings = get_settings()
    try:
        yield
    except psycopg2.errors.UndefinedTable as exc:
        detail = exc.diag.message_primary or str(exc).splitlines()[0]
        st.error(
            f"A table the dashboard reads is missing ({detail}). The stocks database "
            f"({settings.stocks_db}) is filled by rs-pipeline: run rs-pipeline first, or point "
            f"STOCKS_DB_NAME at a database it has filled (see the [README]({README_URL}))."
        )
    except psycopg2.OperationalError as exc:
        st.error(
            f"Cannot reach the database on {settings.db_host}:{settings.db_port}. "
            f"Check the POSTGRES_* settings (see [.env.example]({ENV_EXAMPLE_URL}))."
        )
        st.exception(exc)
