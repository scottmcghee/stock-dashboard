"""Tests for db.py connection pooling."""

from unittest.mock import MagicMock, patch

import pytest


def test_stocks_connection_returns_and_releases():
    """stocks_connection() yields a conn then puts it back."""
    mock_conn = MagicMock()
    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    mock_pool.closed = False

    with patch("db._stocks_pool", mock_pool):
        from db import stocks_connection
        with stocks_connection() as conn:
            assert conn is mock_conn
        mock_pool.putconn.assert_called_once_with(mock_conn)


def test_option_connection_returns_and_releases():
    mock_conn = MagicMock()
    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    mock_pool.closed = False

    with patch("db._option_pool", mock_pool):
        from db import option_connection
        with option_connection() as conn:
            assert conn is mock_conn
        mock_pool.putconn.assert_called_once_with(mock_conn)


def test_stocks_connection_releases_on_exception():
    """Connection is returned to pool even when body raises."""
    mock_conn = MagicMock()
    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn
    mock_pool.closed = False

    with patch("db._stocks_pool", mock_pool):
        from db import stocks_connection
        with pytest.raises(ValueError):
            with stocks_connection() as conn:
                raise ValueError("boom")
        mock_pool.putconn.assert_called_once_with(mock_conn)


def test_close_all_pools():
    mock_stocks = MagicMock()
    mock_stocks.closed = False
    mock_option = MagicMock()
    mock_option.closed = False

    with patch("db._stocks_pool", mock_stocks), \
         patch("db._option_pool", mock_option):
        from db import close_all_pools
        close_all_pools()
        mock_stocks.closeall.assert_called_once()
        mock_option.closeall.assert_called_once()


def test_close_all_pools_skips_already_closed():
    mock_stocks = MagicMock()
    mock_stocks.closed = True
    mock_option = None  # never initialized

    with patch("db._stocks_pool", mock_stocks), \
         patch("db._option_pool", mock_option):
        from db import close_all_pools
        close_all_pools()  # should not raise
        mock_stocks.closeall.assert_not_called()
