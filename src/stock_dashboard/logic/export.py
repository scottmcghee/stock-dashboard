"""TradingView watchlist export helpers.

``dim_stock.exchange`` holds the market-data vendor's exchange codes. Only
codes with a known TradingView prefix are mapped; any other ticker is written
bare, which TradingView resolves itself, rather than under a guessed exchange.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

VENDOR_EXCHANGE_TO_TRADINGVIEW: dict[str, str] = {
    "NMS": "NASDAQ",  # Nasdaq Global Select
    "NGM": "NASDAQ",  # Nasdaq Global Market
    "NCM": "NASDAQ",  # Nasdaq Capital Market
    "NAS": "NASDAQ",
    "NYQ": "NYSE",
    "NYS": "NYSE",
    "ASE": "AMEX",  # NYSE American
    "PCX": "AMEX",  # NYSE Arca, listed under AMEX on TradingView
    "PNK": "OTC",
    "OQB": "OTC",
    "OQX": "OTC",
}


def tradingview_exchange(code: str | None) -> str | None:
    """TradingView exchange prefix for a vendor exchange code, or None when unknown."""
    if not code:
        return None
    return VENDOR_EXCHANGE_TO_TRADINGVIEW.get(code.strip().upper())


def build_tradingview_watchlist(
    tickers: Iterable[str],
    exchange_map: Mapping[str, str | None],
) -> str:
    """TradingView watchlist text, one ``EXCHANGE:TICKER`` (or bare ``TICKER``) per line.

    Tickers keep their first-seen order; duplicates are dropped.
    """
    lines = []
    for ticker in dict.fromkeys(tickers):
        exchange = tradingview_exchange(exchange_map.get(ticker))
        lines.append(f"{exchange}:{ticker}" if exchange else ticker)
    return "\n".join(lines)
