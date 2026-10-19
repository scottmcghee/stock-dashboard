"""Graceful degradation when the option database is missing or empty.

Pages built on option data call :func:`require_option_data`, which explains the
situation and stops the page. Pages that use option data only for some
controls call :func:`note_missing_option_data`, which says what is hidden and
returns whether option data is available, and hide those controls.
"""

from __future__ import annotations

import logging

import psycopg2
import streamlit as st

from stock_dashboard import db
from stock_dashboard.config import CACHE_TTL_SHORT, DATA_CONTRACT_URL, get_settings
from stock_dashboard.queries.option_queries import has_underlying_rows

logger = logging.getLogger(__name__)

OPTION_DATA_MISSING = (
    "Option data is not available: the option database `{dbname}` is unreachable, "
    "lacks the contract tables ({tables}), or holds no rows yet. "
    "See the [data contract]({contract_url}). For generated demo data, run "
    "`stock-dashboard-seed-demo` against a database without real option data: it replaces "
    "the option rows of every symbol it generates."
)


@st.cache_data(ttl=CACHE_TTL_SHORT, show_spinner=False)
def option_data_ready() -> bool:
    """True when the option database has the contract tables and some data in them."""
    if not db.option_data_available():
        return False
    try:
        return has_underlying_rows()
    except psycopg2.Error as exc:
        logger.info("Option database unreadable: %s", exc)
        return False


def missing_option_data_message() -> str:
    """The explanation shown when option data is unavailable."""
    return OPTION_DATA_MISSING.format(
        dbname=get_settings().option_db,
        tables=", ".join(db.OPTION_CONTRACT_TABLES),
        contract_url=DATA_CONTRACT_URL,
    )


def require_option_data() -> None:
    """Stop the page with an explanation unless option data is available."""
    if not option_data_ready():
        st.info(missing_option_data_message())
        st.stop()


def note_missing_option_data(hidden: str) -> bool:
    """For pages that only partly need option data: True when it is available,
    otherwise show the explanation, naming what is ``hidden``, and return False."""
    if option_data_ready():
        return True
    st.info(f"{missing_option_data_message()} Hidden on this page: {hidden}.")
    return False
