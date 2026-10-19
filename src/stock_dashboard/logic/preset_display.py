"""Pure-function renderers that describe a preset: its filters and its backtest record.

Used by the screener when a backtester or research-exported preset is loaded.
The stats are the in-sample backtest's, reported as they are; see
docs/DESIGN.md#limitations for what they do and do not show.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from stock_dashboard.config import LIMITATIONS_URL
from stock_dashboard.logic.backtest_report import format_p_value

# Human-readable templates for the backtester filter names.
_FILTER_LABELS: dict[str, str] = {
    "rsi_min": "RSI ≥ {v}",
    "rsi_max": "RSI ≤ {v}",
    "rs_rank_overall_min": "RS rank ≥ {v:.0%}",
    "rs_rank_overall_max": "RS rank ≤ {v:.0%}",
    "rs_z_score_min": "RS z-score ≥ {v:.2f}",
    "rs_z_score_max": "RS z-score ≤ {v:.2f}",
    "momentum_20_min": "20d momentum ≥ {v:.1f}%",
    "momentum_20_max": "20d momentum ≤ {v:.1f}%",
    "momentum_63_min": "63d momentum ≥ {v:.1f}%",
    "momentum_63_max": "63d momentum ≤ {v:.1f}%",
    "momentum_126_min": "126d momentum ≥ {v:.1f}%",
    "momentum_126_max": "126d momentum ≤ {v:.1f}%",
    "macd_histogram_min": "MACD histogram ≥ {v:.3f}",
    "macd_histogram_max": "MACD histogram ≤ {v:.3f}",
    "ratio_to_50ma_min": "Price ≥ {v:.2f}x 50MA",
    "ratio_to_50ma_max": "Price ≤ {v:.2f}x 50MA",
    "ratio_to_200ma_min": "Price ≥ {v:.2f}x 200MA",
    "ratio_to_200ma_max": "Price ≤ {v:.2f}x 200MA",
    "volume_z_min": "Volume z-score ≥ {v:.1f}",
    "volume_z_max": "Volume z-score ≤ {v:.1f}",
    "adr_14_min": "ADR% ≥ {v:.2f}",
    "adr_14_max": "ADR% ≤ {v:.2f}",
    "weinstein_stage": "Weinstein {v}",
    "rs_impulse_10_min": "10d RS impulse ≥ {v:+.2f}",
    "rs_impulse_10_max": "10d RS impulse ≤ {v:+.2f}",
    "rs_impulse_20_min": "20d RS impulse ≥ {v:+.2f}",
    "rs_impulse_20_max": "20d RS impulse ≤ {v:+.2f}",
    "dist_recent_low_126_min": "Dist from 126d low ≥ {v:.1%}",
    "dist_recent_low_126_max": "Dist from 126d low ≤ {v:.1%}",
    "dist_recent_high_126_min": "Dist from 126d high ≥ {v:.1%}",
    "dist_recent_high_126_max": "Dist from 126d high ≤ {v:.1%}",
    "dist_recent_low_63_min": "Dist from 63d low ≥ {v:.1%}",
    "dist_recent_low_63_max": "Dist from 63d low ≤ {v:.1%}",
    "dist_recent_high_63_min": "Dist from 63d high ≥ {v:.1%}",
    "dist_recent_high_63_max": "Dist from 63d high ≤ {v:.1%}",
}

# Pattern filter → readable
_PATTERN_LABELS = {
    "has_bull_flag": "Bull flag",
    "has_bear_flag": "Bear flag",
    "has_high_tight_flag": "High tight flag",
    "has_cup_and_handle": "Cup & handle",
    "has_v_bottom": "V-bottom",
    "has_evt_rsi_cross_up_30": "Event: RSI crossed up through 30 (last 3d)",
    "has_evt_rsi_cross_down_70": "Event: RSI crossed down through 70 (last 3d)",
    "has_evt_macd_bullish_cross": "Event: MACD bullish cross (last 3d)",
    "has_evt_macd_bearish_cross": "Event: MACD bearish cross (last 3d)",
    "has_evt_breakout_20d_high_vol": "Event: 20-day breakout on volume (last 3d)",
    "has_evt_regain_200ma": "Event: reclaimed 200MA (last 3d)",
    "has_evt_lose_200ma": "Event: lost 200MA (last 3d)",
}


def describe_filter(name: str, value: object) -> str:
    """One-line human label for a single filter. Falls back to name=value."""
    if name in _FILTER_LABELS:
        try:
            return _FILTER_LABELS[name].format(v=value)
        except (ValueError, TypeError):
            return f"{name} = {value}"
    if name in _PATTERN_LABELS:
        return _PATTERN_LABELS[name]
    return f"{name} = {value}"


# The horizon quoted in the one-line summary: fixed, so the summary never picks
# the best-looking horizon. The others are in the details table.
HEADLINE_HORIZONS = ("T+20", "T+10", "T+5", "T+63")


def _headline_horizon(stats: dict[str, dict[str, Any]]) -> str | None:
    """T+20, or the first of :data:`HEADLINE_HORIZONS` the stats have."""
    return next((h for h in HEADLINE_HORIZONS if h in stats), None)


def render_preset_summary(preset: dict[str, Any]) -> str:
    """Markdown describing the preset's filters, regime and in-sample backtest stats.

    Handles both backtester-native presets and ta_signal_analysis-discovered presets
    (which have an extra `_discovery` block).
    """
    if not preset:
        return ""

    lines: list[str] = []
    name = preset.get("_name") or preset.get("_stem") or "Preset"
    lines.append(f"**{name}**")

    # Filter description
    raw_filters = preset.get("_raw_filters") or []
    if raw_filters:
        descs = [describe_filter(f["name"], f["value"]) for f in raw_filters]
        lines.append("*Filters:* " + ", ".join(descs))
    if preset.get("_regime"):
        lines.append(f"*Regime:* {preset['_regime'].title()}")

    # In-sample backtest stats: small table
    stats = preset.get("_stats") or {}
    if stats:
        lines.append("")
        lines.append("| Horizon | Mean return | Win rate | Mean/SD | p (BH) |")
        lines.append("|---|---|---|---|---|")
        for horizon in ("T+5", "T+10", "T+20", "T+63"):
            if horizon not in stats:
                continue
            hs = stats[horizon]
            mr = hs.get("mean_return", 0)
            wr = hs.get("win_rate", 0)
            sh = hs.get("sharpe", 0)
            p = hs.get("p_value_adj", 1.0)
            lines.append(f"| {horizon} | {mr:+.2f}% | {wr:.1%} | {sh:+.2f} | {format_p_value(p)} |")
        lines.append("")
        lines.append(
            "In-sample, per signal (mean/SD is not annualized). Overlapping signals make "
            "the p-values optimistic: use them to rank strategies, not as error rates."
        )

    # Discovery metadata (only for ta_signal_analysis presets)
    disc = preset.get("_discovery")
    if disc:
        lines.append("")
        n = disc.get("n_discovery", 0)
        hit = disc.get("cond_hit_rate_disc", 0)
        lift = disc.get("lift_vs_overall_bps_disc", 0)
        p_disc = disc.get("p_fdr_disc", 1.0)
        lines.append(
            f"*Discovery (in-sample pair scan):* {n:,} signal-days; {hit:.1%} of them had a "
            f"forward return in the top decile (10% by chance); mean {lift:+.0f} bps above "
            f"all rows; BH p {format_p_value(p_disc)}."
        )

    return "\n".join(lines)


def interpret_preset(preset: dict[str, Any]) -> str:
    """One neutral line summarizing a preset's in-sample backtest at a fixed horizon."""
    stats = preset.get("_stats") or {}
    horizon = _headline_horizon(stats)
    if horizon is None:
        return "This preset has no backtest stats."

    hs = stats[horizon]
    try:
        mr = float(hs.get("mean_return", 0))
        wr = float(hs.get("win_rate", 0))
        sh = float(hs.get("sharpe", 0))
        p = float(hs.get("p_value_adj", 1.0))
    except (TypeError, ValueError):
        return "This preset's backtest stats are incomplete."
    return (
        f"In-sample backtest, {horizon}: mean {mr:+.2f}%, win rate {wr:.0%}, "
        f"mean/SD {sh:+.2f}, BH p {format_p_value(p)}. Overlapping signals make p-values "
        "optimistic; "
        f"see [the limitations]({LIMITATIONS_URL})."
    )


def screener_regime_note(regime: str | None, regimes: pd.Series | None, benchmark: str) -> str:
    """What the screener does with a preset's regime, which it does not apply.

    The screener filters the latest session only, so the note compares the
    preset's regime with the benchmark's regime on its latest classified date.
    Empty when the preset has no regime.
    """
    if not regime:
        return ""
    labeled = None if regimes is None else regimes.dropna().sort_index()
    if labeled is None or labeled.empty:
        return (
            f"The preset was backtested in the {regime} regime; the screener does not apply "
            f"regimes, and the benchmark {benchmark} has no classified sessions to compare with."
        )
    latest, current = pd.Timestamp(labeled.index[-1]), str(labeled.iloc[-1])
    on_date = f"{benchmark} is in the {current} regime on {latest:%Y-%m-%d}"
    if current == regime:
        return (
            f"The preset was backtested in the {regime} regime; {on_date}, so the latest "
            "session matches it. The screener filters that session only."
        )
    return (
        f"The preset was backtested in the {regime} regime, but {on_date}: the backtested "
        "condition does not hold for the latest session, which is the only one the screener "
        "filters."
    )


# Preset filters whose screener input measures something slightly different
# (docs/DESIGN.md, invariant 5).
SCREENER_DIFFERENCES = {
    "volume_z": (
        "Volume Z-Score: the backtest used the 20-session volume z-score; the screener's "
        "compares the latest volume with the last 252 sessions."
    ),
    "rs_z_score": (
        "RS Z-Score: the backtest averaged the stock's ETF rows (or used its ETF's rows); the "
        "screener filters the row of the ETF where the stock ranks highest."
    ),
}


def screener_differences(raw_filters: list[dict[str, Any]]) -> list[str]:
    """Notes for the preset filters the screener evaluates differently from the backtest."""
    names = {str(f["name"]) for f in raw_filters}
    return [
        note
        for prefix, note in SCREENER_DIFFERENCES.items()
        if any(n.startswith(f"{prefix}_") for n in names)
    ]
