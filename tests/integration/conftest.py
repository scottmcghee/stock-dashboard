"""A throwaway PostgreSQL server seeded with the demo data, shared by a test session.

The stocks database is filled by rs-pipeline's own demo (its fictional market,
run through every pipeline step), and the option database by this project's
generator, the same way `docker compose up` does. The tests only read (a few
create extra databases on the server), so one seeded server serves them all.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from stock_dashboard import db
from stock_dashboard.config import Settings, get_settings

# The demo's image, pinned by digest, so the tests follow its Dependabot bumps.
_COMPOSE = Path(__file__).parents[2] / "docker-compose.yml"
_image = re.search(r"image: (postgres:\S+)", _COMPOSE.read_text())
assert _image, f"no postgres image in {_COMPOSE}"
POSTGRES_IMAGE = _image[1]

BENCHMARK = "MKT"  # the synthetic market's benchmark
HISTORY_START = "2022-01-03"


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
    except Exception:
        return False
    return True


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "integration" in item.path.parts:
            item.add_marker(pytest.mark.integration)


def _reset_caches() -> None:
    import streamlit as st

    from stock_dashboard.queries import option_queries

    get_settings.cache_clear()
    option_queries._allowed_columns.cache_clear()
    db.close_all_pools()
    st.cache_data.clear()


@pytest.fixture(scope="session")
def seeded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    """Settings pointing at a server holding the full demo data."""
    testcontainers = pytest.importorskip("testcontainers.postgres")
    if not _docker_available():
        if os.getenv("CI"):
            pytest.fail("Docker is required for the integration tests in CI")
        pytest.skip("Docker is not available; skipping integration tests")

    from rs_pipeline import cli as rs_pipeline

    from stock_dashboard.demo.__main__ import seed_demo

    container = testcontainers.PostgresContainer(POSTGRES_IMAGE, driver=None, dbname="stocks")
    container.start()
    host, port = container.get_container_host_ip(), str(container.get_exposed_port(5432))
    env = {
        # rs-pipeline's settings
        "DB_HOST": host,
        "DB_PORT": port,
        "DB_USER": container.username,
        "DB_PASSWORD": container.password,
        "DB_NAME": "stocks",
        "RS_SOURCE": "synthetic",
        "RS_REF_TICKER": BENCHMARK,
        "RS_BENCHMARK_TICKERS": BENCHMARK,
        "RS_HISTORY_START": HISTORY_START,
        # this project's settings
        "POSTGRES_HOST": host,
        "POSTGRES_PORT": port,
        "POSTGRES_USER": container.username,
        "POSTGRES_PASSWORD": container.password,
        "BENCHMARK_TICKER": BENCHMARK,
        "CACHE_DIR": str(tmp_path_factory.mktemp("cache")),
        "PRESET_DIR": str(tmp_path_factory.mktemp("presets")),
    }
    with pytest.MonkeyPatch.context() as mp:
        for name, value in env.items():
            mp.setenv(name, value)
        try:
            assert rs_pipeline.main(["demo", "--log-level", "WARNING"]) == 0
            _reset_caches()
            seed_demo(get_settings())
            yield get_settings()
        finally:
            _reset_caches()
            try:
                container.stop()
            except Exception:  # a slow Docker daemon; testcontainers' reaper cleans up
                logging.getLogger(__name__).warning("could not stop the test container")


@pytest.fixture
def settings(seeded: Settings) -> Iterator[Settings]:
    """The seeded settings, with every cache cleared between tests."""
    _reset_caches()
    yield get_settings()
    _reset_caches()
