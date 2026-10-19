"""Streamlit entry point: page configuration and navigation.

Run with ``stock-dashboard`` (or ``streamlit run src/stock_dashboard/app.py``).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from stock_dashboard.ui.errors import database_errors
from stock_dashboard.ui.option_guard import missing_option_data_message, option_data_ready

# Not named "pages": Streamlit would treat that folder as its older page-discovery
# mode before st.navigation takes over.
PAGES_DIR = Path(__file__).with_name("views")

# (file in views/, title, icon, one-line purpose)
PAGES: list[tuple[str, str, str, str]] = [
    ("1_single_stock.py", "Single Stock", ":material/candlestick_chart:",
     "price, volume and relative strength of one stock, with an optional preset overlay"),
    ("2_stock_screener.py", "Stock Screener", ":material/filter_alt:",
     "filter stocks by RS, volume, technicals, events, patterns and option metrics"),
    ("3_etf_rs.py", "ETF Relative Strength", ":material/leaderboard:",
     "rank ETFs by relative strength against the benchmark, with the change in rank"),
    ("4_option_data.py", "Option Data", ":material/table_chart:",
     "one underlying's option metrics against their 30-session averages"),
    ("5_option_screener.py", "Option Screener", ":material/manage_search:",
     "underlyings whose latest value of an option metric is at least N times its average"),
    ("6_signal_scans.py", "Signal Scans", ":material/radar:",
     "option-positioning and trend-readiness scans, plus a composite"),
    ("7_technical_zscore.py", "Technical Z-Scores", ":material/stacked_line_chart:",
     "RSI, momentum and close relative to a moving average, z-scored across a universe"),
    ("8_backtester.py", "Backtester", ":material/science:",
     "replay filter combinations over history and test their forward returns"),
    ("9_52w_highs_lows.py", "52-Week Highs & Lows", ":material/trending_up:",
     "market breadth from new 52-week highs and lows"),
    ("10_option_contract.py", "Option Contract", ":material/request_quote:",
     "open interest by strike, max pain and one contract's price history"),
]  # fmt: skip


def home() -> None:
    st.title("Stock Analysis Dashboard")
    st.markdown("\n".join(f"- **{title}**: {purpose}" for _, title, _, purpose in PAGES))
    if not option_data_ready():
        st.info(
            f"{missing_option_data_message()} Pages that need option data say so; "
            "the others work without it."
        )


st.set_page_config(
    page_title="Stock Analysis Dashboard", page_icon=":material/monitoring:", layout="wide"
)
navigation = st.navigation(
    [
        st.Page(home, title="Home", icon=":material/home:", default=True),
        *(st.Page(PAGES_DIR / file, title=title, icon=icon) for file, title, icon, _ in PAGES),
    ]
)
with database_errors():  # UI boundary: say what failed instead of a traceback
    navigation.run()
