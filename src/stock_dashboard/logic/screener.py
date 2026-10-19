"""The stock screener's pure steps: pre-filter intersection, TA filter inputs, result table.

Rank inputs in the UI are percentages (0-100); the database stores fractions
(0-1). :func:`percent_to_fraction` converts at the boundary, and the result
table shows ranks as percentages again.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import pandas as pd

from stock_dashboard.logic.lookback import screener_lookback_bars
from stock_dashboard.logic.option_signals import gex_bucket_label
from stock_dashboard.logic.patterns import PatternMatch
from stock_dashboard.logic.technical import (
    DEFAULT_GAP_LOOKBACK,
    DEFAULT_MOMENTUM_PERIOD,
    DEFAULT_RATIO_MA_PERIOD,
)

PERCENT = 100.0
DEFAULT_GAP_MIN_PCT = 1.0

EVENT_LABELS: dict[str, str] = {
    "evt_rsi_cross_up_30": "RSI cross ↑30",
    "evt_rsi_cross_down_70": "RSI cross ↓70",
    "evt_macd_bullish_cross": "MACD bullish cross",
    "evt_macd_bearish_cross": "MACD bearish cross",
    "evt_breakout_20d_high_vol": "20-day breakout on volume",
    "evt_regain_200ma": "Reclaimed 200-day MA",
    "evt_lose_200ma": "Lost 200-day MA",
}


# Rounding keeps 70% == 0.7 exactly (70 / 100 alone can carry float noise).
FRACTION_DECIMALS = 6


def percent_to_fraction(value: float | None) -> float | None:
    """A UI percentage (0-100) as the fraction stored in the database."""
    return None if value is None else round(float(value) / PERCENT, FRACTION_DECIMALS)


def intersect_prefilters(sets: Sequence[set[str]]) -> list[str] | None:
    """Tickers passing every pre-filter, sorted; None when no pre-filter ran."""
    if not sets:
        return None
    return sorted(set.intersection(*sets))


@dataclass(frozen=True)
class TaFilters:
    """Technical post-filters as entered in the screener (None = not used).

    ``dist_low_126_min_pct``/``_max_pct`` are percentages; everything else is in
    the units :func:`stock_dashboard.logic.technical.apply_ta_filters` takes.
    """

    rsi_min: float | None = None
    rsi_max: float | None = None
    macd_crossover: str | None = None
    ma_touch_period: int | None = None
    adr_min: float | None = None
    gap_type: str | None = None
    momentum_min: float | None = None
    momentum_max: float | None = None
    ratio_to_ma_min: float | None = None
    ratio_to_ma_max: float | None = None
    dist_low_126_min_pct: float | None = None
    dist_low_126_max_pct: float | None = None
    # Settings that only qualify an active filter:
    gap_min_pct: float = DEFAULT_GAP_MIN_PCT
    gap_lookback: int = DEFAULT_GAP_LOOKBACK
    momentum_period: int = DEFAULT_MOMENTUM_PERIOD
    ratio_to_ma_period: int = DEFAULT_RATIO_MA_PERIOD
    timeframe: str = "D"

    _SETTINGS = frozenset(
        {"gap_min_pct", "gap_lookback", "momentum_period", "ratio_to_ma_period", "timeframe"}
    )

    @property
    def active(self) -> bool:
        """True when at least one filter (not just a setting) is set."""
        return any(
            getattr(self, f.name) is not None for f in fields(self) if f.name not in self._SETTINGS
        )

    def apply_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``apply_ta_filters`` (besides the frame and tickers)."""
        return {
            "rsi_min": self.rsi_min,
            "rsi_max": self.rsi_max,
            "macd_crossover": self.macd_crossover,
            "ma_touch_period": self.ma_touch_period,
            "adr_min": self.adr_min,
            "gap_type": self.gap_type,
            "gap_min_pct": self.gap_min_pct,
            "gap_lookback": self.gap_lookback,
            "momentum_min": self.momentum_min,
            "momentum_max": self.momentum_max,
            "momentum_period": self.momentum_period,
            "ratio_to_ma_min": self.ratio_to_ma_min,
            "ratio_to_ma_max": self.ratio_to_ma_max,
            "ratio_to_ma_period": self.ratio_to_ma_period,
            "dist_recent_low_126_min": percent_to_fraction(self.dist_low_126_min_pct),
            "dist_recent_low_126_max": percent_to_fraction(self.dist_low_126_max_pct),
            "timeframe": self.timeframe,
        }

    def lookback_bars(self) -> int:
        """Daily bars per ticker for these filters plus the event flags and pattern scans."""
        return screener_lookback_bars(
            self.momentum_period, self.ratio_to_ma_period, self.timeframe, self.ma_touch_period
        )


def tickers_with_events(
    event_flags: Mapping[str, Mapping[str, bool]], events: Iterable[str]
) -> set[str]:
    """Tickers where at least one of ``events`` fired."""
    wanted = list(events)
    return {t for t, flags in event_flags.items() if any(flags.get(e, False) for e in wanted)}


OPTION_COLUMNS = {
    "vol_multiple": "Opt Vol (x avg)",
    "put_call_ratio": "P/C Ratio",
    "dxoi": "dxoi",
}


def option_signal_table(
    option_metrics: pd.DataFrame, scans: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """One row per symbol: display-named option metrics, the GEX bucket and scan labels.

    ``option_metrics`` comes from ``fetch_option_metrics_for_tickers``; ``scans``
    maps a column name (e.g. "Vanna") to a scan result with symbol and signal_label.
    """
    symbols = list(option_metrics["symbol"]) if not option_metrics.empty else []
    for scan in scans.values():
        if not scan.empty:
            symbols.extend(scan["symbol"])
    table = pd.DataFrame({"symbol": list(dict.fromkeys(symbols))})
    if not option_metrics.empty:
        metrics = option_metrics.drop_duplicates("symbol").set_index("symbol")
        for source, label in OPTION_COLUMNS.items():
            if source in metrics.columns:
                table[label] = table["symbol"].map(metrics[source])
        if "gxoi" in metrics.columns:
            table["GEX"] = [gex_bucket_label(v) for v in table["symbol"].map(metrics["gxoi"])]
    for name, scan in scans.items():
        if scan.empty:
            continue
        labels = scan.drop_duplicates("symbol").set_index("symbol")["signal_label"]
        table[name] = table["symbol"].map(labels)
    return table


RANK_COLUMNS = ("RS Rank in ETF %", "RS Rank Overall %")

_BASE_COLUMNS = {
    "ticker": "Ticker",
    "etf_ticker": "ETF",
    "etfs": "All ETFs",
    "rs_rank_index": RANK_COLUMNS[0],
    "rs_rank_overall": RANK_COLUMNS[1],
    "rs_z_score": "RS Z-Score",
    "volume_z_score": "Volume Z-Score",
    "avg_30d_volume": "Avg 30d Volume",
    "stage": "Weinstein Stage",
}

WHOLE_NUMBER_FORMAT = "%,.0f"  # volumes and exposures: thousands separators, no decimals

# Display formats (printf style) for the numeric result columns.
RESULT_FORMATS: dict[str, str] = {
    RANK_COLUMNS[0]: "%.1f",
    RANK_COLUMNS[1]: "%.1f",
    "RS Z-Score": "%.2f",
    "Volume Z-Score": "%.2f",
    "Avg 30d Volume": WHOLE_NUMBER_FORMAT,
    "RSI": "%.1f",
    "ADR %": "%.1f",
    "Momentum %": "%.1f",
    "Ratio to MA": "%.3f",
    "Dist 126d Low %": "%.1f",
    "Opt Vol (x avg)": "%.1f",
    "P/C Ratio": "%.2f",
    "dxoi": WHOLE_NUMBER_FORMAT,
}


def _gap_text(gap: Mapping[str, Any] | None) -> str | None:
    if not gap:
        return None
    return f"{str(gap.get('type', '')).title()} {gap.get('gap_pct', '')}%"


def _ta_columns(tickers: list[str], ta_info: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    def values(key: str) -> list[Any]:
        return [ta_info.get(t, {}).get(key) for t in tickers]

    dist = [None if v is None else v * PERCENT for v in values("dist_recent_low_126")]
    candidates: dict[str, list[Any]] = {
        "RSI": values("rsi"),
        "MACD Cross": values("macd_cross"),
        "MA Touch": values("ma_touch"),
        "ADR %": values("adr"),
        "Gap": [_gap_text(g) for g in values("gap")],
        "Momentum %": values("momentum"),
        "Ratio to MA": values("ratio_to_ma"),
        "Dist 126d Low %": dist,
    }
    return pd.DataFrame(
        {name: col for name, col in candidates.items() if any(v is not None for v in col)}
    )


def build_results_table(
    results: pd.DataFrame,
    *,
    ta_info: Mapping[str, Mapping[str, Any]] | None = None,
    event_flags: Mapping[str, Mapping[str, bool]] | None = None,
    events: Sequence[str] = (),
    option_table: pd.DataFrame | None = None,
    pattern_info: Mapping[str, list[PatternMatch]] | None = None,
) -> pd.DataFrame:
    """The screener's result grid: one row per ticker, numeric columns kept numeric.

    Ranks are shown as percentages. Optional column groups (TA values, fired
    events, option metrics and labels, patterns) appear only when they hold data.
    """
    base = results[[c for c in _BASE_COLUMNS if c in results.columns]].copy()
    for col in ("rs_rank_index", "rs_rank_overall"):
        if col in base.columns:
            base[col] = base[col] * PERCENT
    table = base.rename(columns=_BASE_COLUMNS).reset_index(drop=True)
    tickers = table["Ticker"].tolist()

    if ta_info:
        table = pd.concat([table, _ta_columns(tickers, ta_info)], axis=1)

    if event_flags and events:
        table["TA Events"] = [
            ", ".join(EVENT_LABELS.get(e, e) for e in events if event_flags.get(t, {}).get(e))
            or None
            for t in tickers
        ]

    if option_table is not None and not option_table.empty:
        options = option_table.drop_duplicates("symbol").set_index("symbol")
        for column in options.columns:
            values = table["Ticker"].map(options[column])
            if values.notna().any():
                table[column] = values

    if pattern_info:
        table["Patterns"] = [
            ", ".join(p["pattern"].replace("_", " ").title() for p in pattern_info.get(t, []))
            or None
            for t in tickers
        ]
    return table
