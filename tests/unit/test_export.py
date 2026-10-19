"""Tests for logic/export.py."""

import pytest

from stock_dashboard.logic.export import build_tradingview_watchlist, tradingview_exchange


class TestTradingviewExchange:
    @pytest.mark.parametrize("code", ["NMS", "NGM", "NCM", "NAS"])
    def test_nasdaq_codes(self, code):
        assert tradingview_exchange(code) == "NASDAQ"

    @pytest.mark.parametrize("code", ["NYQ", "NYS"])
    def test_nyse_codes(self, code):
        assert tradingview_exchange(code) == "NYSE"

    @pytest.mark.parametrize("code", ["ASE", "PCX"])
    def test_amex_codes(self, code):
        assert tradingview_exchange(code) == "AMEX"

    @pytest.mark.parametrize("code", ["PNK", "OQB", "OQX"])
    def test_otc_codes(self, code):
        assert tradingview_exchange(code) == "OTC"

    def test_code_is_normalized(self):
        assert tradingview_exchange(" nms ") == "NASDAQ"

    @pytest.mark.parametrize("code", [None, ""])
    def test_missing_code_has_no_prefix(self, code):
        """The old mapping sent a missing code to NYSE."""
        assert tradingview_exchange(code) is None

    @pytest.mark.parametrize("code", ["BTS", "YHD", "XLON"])
    def test_unknown_code_has_no_prefix(self, code):
        """The old mapping guessed AMEX for BTS and passed other codes through as prefixes."""
        assert tradingview_exchange(code) is None


class TestBuildTradingviewWatchlist:
    def test_basic(self):
        exchange_map = {"AAPL": "NMS", "JPM": "NYQ"}
        result = build_tradingview_watchlist(["AAPL", "JPM"], exchange_map)
        assert result == "NASDAQ:AAPL\nNYSE:JPM"

    def test_missing_exchange_writes_bare_ticker(self):
        assert build_tradingview_watchlist(["XYZ"], {}) == "XYZ"

    def test_none_exchange_writes_bare_ticker(self):
        assert build_tradingview_watchlist(["XYZ"], {"XYZ": None}) == "XYZ"

    def test_unknown_exchange_writes_bare_ticker(self):
        assert build_tradingview_watchlist(["XYZ"], {"XYZ": "YHD"}) == "XYZ"

    def test_empty_list(self):
        assert build_tradingview_watchlist([], {}) == ""

    def test_preserves_order(self):
        exchange_map = {"B": "NMS", "A": "NYQ", "C": "ASE"}
        result = build_tradingview_watchlist(["B", "A", "C"], exchange_map)
        assert result.split("\n") == ["NASDAQ:B", "NYSE:A", "AMEX:C"]

    def test_deduplicates_tickers(self):
        result = build_tradingview_watchlist(["AAPL", "AAPL", "AAPL"], {"AAPL": "NMS"})
        assert result == "NASDAQ:AAPL"
