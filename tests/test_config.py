"""Tests for config.py."""

import os
from unittest.mock import patch


def test_stocks_db_config_defaults():
    """Default config uses expected values."""
    # Re-import to pick up env vars from conftest
    import importlib
    import config
    importlib.reload(config)

    assert config.STOCKS_DB_CONFIG["dbname"] == "stocks"
    assert config.STOCKS_DB_CONFIG["user"] == "postgres"
    assert config.STOCKS_DB_CONFIG["host"] == "localhost"
    assert config.STOCKS_DB_CONFIG["port"] == "5432"


def test_option_db_config_defaults():
    import importlib
    import config
    importlib.reload(config)

    assert config.OPTION_DB_CONFIG["dbname"] == "option_data"


def test_config_reads_env_vars(monkeypatch):
    """Config picks up env var overrides."""
    monkeypatch.setenv("POSTGRES_USER", "custom_user")
    monkeypatch.setenv("POSTGRES_HOST", "remote.host")
    monkeypatch.setenv("STOCKS_DB_NAME", "my_stocks")

    import importlib
    import config
    importlib.reload(config)

    assert config.STOCKS_DB_CONFIG["user"] == "custom_user"
    assert config.STOCKS_DB_CONFIG["host"] == "remote.host"
    assert config.STOCKS_DB_CONFIG["dbname"] == "my_stocks"


def test_cache_ttl_constants():
    from config import CACHE_TTL_SHORT, CACHE_TTL_DEFAULT, CACHE_TTL_LONG

    assert CACHE_TTL_SHORT < CACHE_TTL_DEFAULT < CACHE_TTL_LONG
    assert CACHE_TTL_SHORT > 0
