"""Tests for logic/export.py."""

from logic.export import _map_exchange, build_tradingview_watchlist


class TestMapExchange:
    def test_known_nasdaq(self):
        assert _map_exchange("NMS") == "NASDAQ"
        assert _map_exchange("NGM") == "NASDAQ"

    def test_known_nyse(self):
        assert _map_exchange("NYQ") == "NYSE"

    def test_known_amex(self):
        assert _map_exchange("ASE") == "AMEX"
        assert _map_exchange("PCX") == "AMEX"

    def test_none_returns_default(self):
        assert _map_exchange(None) == "NYSE"

    def test_empty_string_returns_default(self):
        assert _map_exchange("") == "NYSE"

    def test_unknown_passed_through(self):
        assert _map_exchange("XLON") == "XLON"


class TestBuildTradingviewWatchlist:
    def test_basic(self):
        exchange_map = {"AAPL": "NMS", "JPM": "NYQ"}
        result = build_tradingview_watchlist(["AAPL", "JPM"], exchange_map)
        assert result == "NASDAQ:AAPL\nNYSE:JPM"

    def test_missing_exchange_uses_default(self):
        result = build_tradingview_watchlist(["XYZ"], {})
        assert result == "NYSE:XYZ"

    def test_none_exchange_uses_default(self):
        result = build_tradingview_watchlist(["XYZ"], {"XYZ": None})
        assert result == "NYSE:XYZ"

    def test_empty_list(self):
        result = build_tradingview_watchlist([], {})
        assert result == ""

    def test_preserves_order(self):
        exchange_map = {"B": "NMS", "A": "NYQ", "C": "ASE"}
        result = build_tradingview_watchlist(["B", "A", "C"], exchange_map)
        lines = result.split("\n")
        assert lines == ["NASDAQ:B", "NYSE:A", "AMEX:C"]

    def test_deduplicates_tickers(self):
        exchange_map = {"AAPL": "NMS"}
        result = build_tradingview_watchlist(["AAPL", "AAPL", "AAPL"], exchange_map)
        assert result == "NASDAQ:AAPL"
