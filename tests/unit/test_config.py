"""Tests for config.py."""

from pathlib import Path

import pytest

from stock_dashboard.config import (
    CACHE_TTL_DEFAULT,
    CACHE_TTL_LONG,
    CACHE_TTL_SHORT,
    Settings,
    get_settings,
)


def test_defaults(monkeypatch):
    monkeypatch.delenv("CACHE_DIR")
    monkeypatch.delenv("PRESET_DIR")
    s = Settings()
    assert (s.db_host, s.db_port, s.db_user, s.db_password) == ("localhost", 5432, "postgres", "")
    assert (s.stocks_db, s.option_db, s.benchmark) == ("stocks", "option_data", "SPY")
    assert s.preset_dir == Path("~/.config/stock_dashboard/presets").expanduser()
    assert s.cache_dir == Path("~/.cache/stock_dashboard").expanduser()


def test_reads_env_vars(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("POSTGRES_USER", "reader")
    monkeypatch.setenv("STOCKS_DB_NAME", "demo_stocks")
    monkeypatch.setenv("OPTION_DB_NAME", "demo_options")
    monkeypatch.setenv("BENCHMARK_TICKER", " mkt ")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "c"))
    s = Settings()
    assert (s.db_host, s.db_port, s.db_user) == ("db.internal", 6543, "reader")
    assert (s.stocks_db, s.option_db) == ("demo_stocks", "demo_options")
    assert s.benchmark == "MKT"
    assert s.cache_dir == tmp_path / "c"


def test_empty_benchmark_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("BENCHMARK_TICKER", "")
    assert Settings().benchmark == "SPY"
    monkeypatch.setenv("BENCHMARK_TICKER", "  ")
    assert Settings().benchmark == "SPY"


@pytest.mark.parametrize(
    ("name", "attr", "default"),
    [
        ("POSTGRES_HOST", "db_host", "localhost"),
        ("POSTGRES_PORT", "db_port", 5432),
        ("POSTGRES_USER", "db_user", "postgres"),
        ("STOCKS_DB_NAME", "stocks_db", "stocks"),
        ("OPTION_DB_NAME", "option_db", "option_data"),
    ],
)
def test_empty_setting_falls_back_to_default(monkeypatch, name, attr, default):
    monkeypatch.setenv(name, "")
    assert getattr(Settings(), attr) == default


def test_password_hidden_from_repr(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "hunter2")
    s = Settings()
    assert s.db_password == "hunter2"
    assert "hunter2" not in repr(s)


def test_db_kwargs(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    assert Settings().db_kwargs("stocks") == {
        "dbname": "stocks",
        "user": "postgres",
        "password": "pw",
        "host": "localhost",
        "port": 5432,
    }


def test_settings_are_frozen():
    with pytest.raises(AttributeError):
        Settings().benchmark = "QQQ"  # type: ignore[misc]


def test_get_settings_is_cached_until_cleared(monkeypatch):
    first = get_settings()
    monkeypatch.setenv("BENCHMARK_TICKER", "QQQ")
    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings().benchmark == "QQQ"


def test_cache_ttl_constants():
    assert 0 < CACHE_TTL_SHORT < CACHE_TTL_DEFAULT < CACHE_TTL_LONG
