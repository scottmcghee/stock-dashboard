"""Shared preset loading and parsing — used by screener, single stock, and backtester."""

import json
from pathlib import Path

PRESET_DIR = Path("~/.config/stock_dashboard/presets").expanduser()

# Maps backtester filter names → (screener widget key, converter)
PRESET_FILTER_MAP: dict[str, tuple] = {
    "rsi_min": ("rsi_min", int),
    "rsi_max": ("rsi_max", int),
    "rs_rank_overall_min": ("rs_rank_overall_min", lambda v: v * 100.0),
    "rs_rank_overall_max": ("rs_rank_overall_max", lambda v: v * 100.0),
    "rs_z_score_min": ("rs_z_score_min", float),
    "rs_z_score_max": ("rs_z_score_max", float),
    "volume_z_min": ("volume_std_dev_min", float),
    "adr_14_min": ("adr_min", int),
    "weinstein_stage": ("weinstein_stage", str),
    # Momentum filters — period extracted separately
    "momentum_20_min": ("momentum_min", float),
    "momentum_20_max": ("momentum_max", float),
    "momentum_63_min": ("momentum_min", float),
    "momentum_63_max": ("momentum_max", float),
    "momentum_126_min": ("momentum_min", float),
    "momentum_126_max": ("momentum_max", float),
    # Ratio-to-MA filters — period extracted separately
    "ratio_to_50ma_min": ("ratio_to_ma_min", float),
    "ratio_to_50ma_max": ("ratio_to_ma_max", float),
    "ratio_to_200ma_min": ("ratio_to_ma_min", float),
    "ratio_to_200ma_max": ("ratio_to_ma_max", float),
}

PATTERN_FILTER_NAMES = {
    "has_bull_flag": "bull_flag",
    "has_bear_flag": "bear_flag",
    "has_high_tight_flag": "high_tight_flag",
    "has_cup_and_handle": "cup_and_handle",
    "has_v_bottom": "v_bottom",
}

# Extract period from momentum filter name: "momentum_63_min" → 63
MOMENTUM_PERIOD_MAP = {
    "momentum_20_min": 20, "momentum_20_max": 20,
    "momentum_63_min": 63, "momentum_63_max": 63,
    "momentum_126_min": 126, "momentum_126_max": 126,
}

# Extract period from ratio-to-MA filter name: "ratio_to_50ma_min" → 50
RATIO_MA_PERIOD_MAP = {
    "ratio_to_50ma_min": 50, "ratio_to_50ma_max": 50,
    "ratio_to_200ma_min": 200, "ratio_to_200ma_max": 200,
}


def list_presets() -> list[str]:
    """Return sorted file stems of saved preset JSONs."""
    if not PRESET_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def parse_preset(stem: str) -> dict:
    """Read a preset JSON and map backtester filter names to screener widget defaults.

    Returns dict with screener widget keys plus metadata:
        _stem, _name, _regime, _stats, _raw_filters, _unmapped, pattern_scans,
        momentum_period, ratio_to_ma_period, and mapped filter values.
    """
    path = PRESET_DIR / f"{stem}.json"
    with open(path) as f:
        data = json.load(f)

    raw_filters = data.get("filters", [])
    result: dict = {
        "_stem": stem,
        "_name": data.get("name", stem),
        "_regime": data.get("regime"),
        "_stats": data.get("stats", {}),
        "_raw_filters": raw_filters,
    }
    unmapped: list[str] = []
    patterns: list[str] = []

    for filt in raw_filters:
        fname, fvalue = filt["name"], filt["value"]
        if fname in PRESET_FILTER_MAP:
            screener_key, converter = PRESET_FILTER_MAP[fname]
            result[screener_key] = converter(fvalue)
            # Extract period for momentum/ratio-to-MA
            if fname in MOMENTUM_PERIOD_MAP:
                result["momentum_period"] = MOMENTUM_PERIOD_MAP[fname]
            if fname in RATIO_MA_PERIOD_MAP:
                result["ratio_to_ma_period"] = RATIO_MA_PERIOD_MAP[fname]
        elif fname in PATTERN_FILTER_NAMES:
            patterns.append(PATTERN_FILTER_NAMES[fname])
        else:
            unmapped.append(f"{fname}={fvalue}")

    if patterns:
        result["pattern_scans"] = patterns
    if unmapped:
        result["_unmapped"] = unmapped
    return result
