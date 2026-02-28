"""TradingView watchlist export helpers."""

YFINANCE_TO_TV: dict[str, str] = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NAS": "NASDAQ",
    "NYQ": "NYSE", "NYS": "NYSE",
    "ASE": "AMEX", "PCX": "AMEX", "BTS": "AMEX",
}
DEFAULT_EXCHANGE = "NYSE"


def _map_exchange(raw: str | None) -> str:
    """Map yfinance exchange code to TradingView exchange name."""
    if not raw:
        return DEFAULT_EXCHANGE
    return YFINANCE_TO_TV.get(raw, raw)


def build_tradingview_watchlist(
    tickers: list[str],
    exchange_map: dict[str, str | None],
) -> str:
    """Build TradingView watchlist text: EXCHANGE:TICKER per line."""
    lines = []
    for ticker in tickers:
        exchange = _map_exchange(exchange_map.get(ticker))
        lines.append(f"{exchange}:{ticker}")
    return "\n".join(lines)
