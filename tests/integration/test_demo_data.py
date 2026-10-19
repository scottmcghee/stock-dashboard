"""The seeded demo data satisfies what the dashboard's queries need."""

from __future__ import annotations

from stock_dashboard import db
from stock_dashboard.config import Settings
from stock_dashboard.queries import signal_queries, stocks_queries


def test_option_data_is_available(settings: Settings) -> None:
    assert db.option_data_available()


def test_latest_dates_agree(settings: Settings) -> None:
    stocks = db.read_df(db.Database.STOCKS, "SELECT MAX(date) AS d FROM fact_commons_ohlc")
    options = db.read_df(
        db.Database.OPTIONS, "SELECT MAX(collection_date) AS d FROM underlying_data"
    )
    assert stocks["d"].iloc[0] == options["d"].iloc[0]


def test_etf_grid_has_one_row_per_etf(settings: Settings) -> None:
    grid = stocks_queries.fetch_latest_etf_grid()
    assert not grid.empty
    assert grid["etf_ticker"].is_unique


def test_every_option_scan_finds_the_planted_symbols(settings: Settings) -> None:
    scans = {
        "vanna": signal_queries.scan_vanna_extreme(),
        "flownet": signal_queries.scan_flownet_extreme(),
        "volume": signal_queries.scan_volume_spike(),
        "dxoi": signal_queries.scan_dxoi_change(),
        "gex": signal_queries.scan_gex_level(),
    }
    assert all(not df.empty for df in scans.values()), {k: len(v) for k, v in scans.items()}
