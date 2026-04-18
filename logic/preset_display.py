"""Pure-function renderers for describing a preset's purpose and expected behavior.

Called from screener (and potentially other pages) to show the user "what am I
looking at / expecting" when they load a backtester or TA-discovery preset.
"""

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
    "ratio_to_50ma_min": "Price ≥ {v:.2f}× 50MA",
    "ratio_to_50ma_max": "Price ≤ {v:.2f}× 50MA",
    "ratio_to_200ma_min": "Price ≥ {v:.2f}× 200MA",
    "ratio_to_200ma_max": "Price ≤ {v:.2f}× 200MA",
    "volume_z_min": "Volume z-score ≥ {v:.1f}",
    "volume_z_max": "Volume z-score ≤ {v:.1f}",
    "adr_14_min": "ADR% ≥ {v:.2f}",
    "adr_14_max": "ADR% ≤ {v:.2f}",
    "weinstein_stage": "Weinstein {v}",
    "rs_impulse_10_min": "10d RS impulse ≥ {v:+.2f}",
    "rs_impulse_10_max": "10d RS impulse ≤ {v:+.2f}",
    "rs_impulse_20_min": "20d RS impulse ≥ {v:+.2f}",
    "rs_impulse_20_max": "20d RS impulse ≤ {v:+.2f}",
}

# Pattern filter → readable
_PATTERN_LABELS = {
    "has_bull_flag": "Bull flag",
    "has_bear_flag": "Bear flag",
    "has_high_tight_flag": "High tight flag",
    "has_cup_and_handle": "Cup & handle",
    "has_v_bottom": "V-bottom",
}


def describe_filter(name: str, value) -> str:
    """One-line human label for a single filter. Falls back to name=value."""
    if name in _FILTER_LABELS:
        try:
            return _FILTER_LABELS[name].format(v=value)
        except Exception:
            return f"{name} = {value}"
    if name in _PATTERN_LABELS:
        return _PATTERN_LABELS[name]
    return f"{name} = {value}"


def _best_horizon(stats: dict) -> str | None:
    """Pick the horizon with the highest |mean_return| for headline framing."""
    if not stats:
        return None
    try:
        return max(stats.keys(), key=lambda k: abs(float(stats[k].get("mean_return", 0))))
    except (ValueError, TypeError):
        return None


def render_preset_summary(preset: dict) -> str:
    """Return a markdown blob describing what this preset scans for + expected stats.

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
        lines.append("*Scanning for:* " + ", ".join(descs))
    if preset.get("_regime"):
        lines.append(f"*Regime:* {preset['_regime'].title()}")

    # Expected stats — small table
    stats = preset.get("_stats") or {}
    if stats:
        lines.append("")
        lines.append("| Horizon | Mean return | Win rate | Sharpe | p (BH) |")
        lines.append("|---|---|---|---|---|")
        for horizon in ("T+5", "T+10", "T+20", "T+63"):
            if horizon not in stats:
                continue
            hs = stats[horizon]
            mr = hs.get("mean_return", 0)
            wr = hs.get("win_rate", 0)
            sh = hs.get("sharpe", 0)
            p = hs.get("p_value_adj", 1.0)
            sig_mark = " ✓" if p < 0.05 else ""
            lines.append(f"| {horizon} | {mr:+.2f}% | {wr:.1%} | {sh:+.2f} | {p:.3f}{sig_mark} |")

    # Discovery metadata (only for ta_signal_analysis presets)
    disc = preset.get("_discovery")
    if disc:
        lines.append("")
        n = disc.get("n_discovery", 0)
        hit = disc.get("cond_hit_rate_disc", 0)
        lift = disc.get("lift_vs_overall_bps_disc", 0)
        p_disc = disc.get("p_fdr_disc", 1.0)
        lines.append(
            f"*Discovery:* n={n:,} signal-days, top-decile hit rate "
            f"**{hit:.1%}** (vs ~10% baseline), cross-sectional lift "
            f"**{lift:+.0f} bps**, FDR p={p_disc:.3f}."
        )

    return "\n".join(lines)


def interpret_preset(preset: dict) -> str:
    """One-sentence "what does this tell me" interpretation for a preset.

    Looks at the headline horizon and whether mean_return is +/−, and
    whether the preset is a TA-discovered one (less narrative grounding).
    """
    stats = preset.get("_stats") or {}
    if not stats:
        return "No backtested stats on this preset yet."

    best = _best_horizon(stats)
    if best is None:
        return "Expected behavior is unclear — stats incomplete."

    hs = stats[best]
    mr = float(hs.get("mean_return", 0))
    wr = float(hs.get("win_rate", 0))
    sh = float(hs.get("sharpe", 0))
    p = float(hs.get("p_value_adj", 1.0))

    direction = "outperformance" if mr > 0 else "underperformance"
    strength = "strong" if abs(sh) >= 0.3 else ("moderate" if abs(sh) >= 0.15 else "weak")
    sig = " (statistically significant)" if p < 0.05 else " (not significant at FDR 0.05)"

    return (
        f"Best horizon {best}: {strength} {direction} averaging {mr:+.2f}%, "
        f"win rate {wr:.0%}, Sharpe {sh:+.2f}{sig}. "
        "Use as a screen — stats reflect average behavior of the cohort, not every individual signal."
    )
