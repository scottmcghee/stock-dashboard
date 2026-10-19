"""Shared preset loading and parsing: used by screener, single stock, and backtester."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from stock_dashboard.config import get_settings

logger = logging.getLogger(__name__)

PERCENT = 100.0
# Decimals kept when converting between fractions and percentages, so that
# 0.7 -> 70.0 -> 0.7 round-trips exactly instead of becoming 0.7000000000000001.
CONVERSION_DECIMALS = 6

# Errors that make a preset file unreadable (JSONDecodeError is a ValueError).
PRESET_ERRORS = (OSError, ValueError, KeyError, TypeError)


def _percent(v: float) -> float:
    return round(float(v) * PERCENT, CONVERSION_DECIMALS)


# Maps backtester filter names -> (screener widget key, converter)
PRESET_FILTER_MAP: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "rsi_min": ("rsi_min", float),
    "rsi_max": ("rsi_max", float),
    "rs_rank_overall_min": ("rs_rank_overall_min", _percent),
    "rs_rank_overall_max": ("rs_rank_overall_max", _percent),
    "rs_z_score_min": ("rs_z_score_min", float),
    "rs_z_score_max": ("rs_z_score_max", float),
    "volume_z_min": ("volume_std_dev_min", float),
    "adr_14_min": ("adr_min", float),
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
    # S/R proximity (backtester stores as decimal; screener widget expects %)
    "dist_recent_low_126_min": ("dist_recent_low_126_min", _percent),
    "dist_recent_low_126_max": ("dist_recent_low_126_max", _percent),
}

# Bounds of the screener widgets the mapped filters fill, in the preset's own units;
# a value outside them is rejected rather than crash the widget.
PRESET_VALUE_RANGES: dict[str, tuple[float, float]] = {
    "rsi_min": (0.0, 100.0),
    "rsi_max": (0.0, 100.0),
    "rs_rank_overall_min": (0.0, 1.0),
    "rs_rank_overall_max": (0.0, 1.0),
    "adr_14_min": (0.0, 100.0),
}
FRACTION_FILTERS = frozenset({"rs_rank_overall_min", "rs_rank_overall_max"})


def check_preset_value(name: str, value: Any) -> None:
    """Raise ValueError when a mapped numeric filter's value is not finite (JSON allows
    ``Infinity`` and ``NaN``) or is outside its screener widget's range."""
    mapping = PRESET_FILTER_MAP.get(name)
    if mapping is None or mapping[1] is str:
        return
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}={value} is not a finite number")
    if name not in PRESET_VALUE_RANGES:
        return
    low, high = PRESET_VALUE_RANGES[name]
    if not low <= number <= high:
        hint = " (ranks are fractions: 0.8 means 80%)" if name in FRACTION_FILTERS else ""
        raise ValueError(f"{name}={value} is outside {low:g} to {high:g}{hint}")


PATTERN_FILTER_NAMES = {
    "has_bull_flag": "bull_flag",
    "has_bear_flag": "bear_flag",
    "has_high_tight_flag": "high_tight_flag",
    "has_cup_and_handle": "cup_and_handle",
    "has_v_bottom": "v_bottom",
}

# Extract period from momentum filter name: "momentum_63_min" → 63
MOMENTUM_PERIOD_MAP = {
    "momentum_20_min": 20,
    "momentum_20_max": 20,
    "momentum_63_min": 63,
    "momentum_63_max": 63,
    "momentum_126_min": 126,
    "momentum_126_max": 126,
}

# Extract period from ratio-to-MA filter name: "ratio_to_50ma_min" → 50
RATIO_MA_PERIOD_MAP = {
    "ratio_to_50ma_min": 50,
    "ratio_to_50ma_max": 50,
    "ratio_to_200ma_min": 200,
    "ratio_to_200ma_max": 200,
}


def preset_dir() -> Path:
    """Directory holding preset JSON files (``PRESET_DIR``)."""
    return get_settings().preset_dir


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
MAX_FILENAME_STEM = 80


def build_preset(
    name: str,
    filters: Mapping[str, object],
    *,
    description: str = "",
    regimes: Sequence[str] = (),
    stats: Mapping[str, object] | None = None,
    etf: str | None = None,
) -> dict[str, object]:
    """A preset document in the JSON format :func:`parse_preset` reads.

    ``filters`` maps backtester filter names to values (``{"rsi_min": 30}``) and is stored
    as the ``[{"name": ..., "value": ...}]`` list. A preset carries at most one regime
    (``regime`` is ``null`` for "any"), matching ``backtest.Strategy.regime``. ``etf``
    records the ETF a backtest was restricted to, whose RS metrics its ``rs_z_score``
    filters were evaluated on; it is stored only when set.
    """
    if not name.strip():
        raise ValueError("preset name must not be empty")
    if len(regimes) > 1:
        raise ValueError(f"a preset holds at most one regime, got {list(regimes)}")
    preset: dict[str, object] = {
        "name": name,
        "filters": [{"name": k, "value": v} for k, v in filters.items()],
        "regime": regimes[0] if regimes else None,
        "stats": dict(stats or {}),
    }
    if etf:
        preset["etf"] = etf
    if description:
        preset["description"] = description
    return preset


def preset_filename(name: str) -> str:
    """``<sanitized name>.json``: only ``[A-Za-z0-9._-]``, no leading dots, bounded length."""
    stem = _UNSAFE_FILENAME_CHARS.sub("_", name.strip()).strip("._")[:MAX_FILENAME_STEM]
    if not stem:
        raise ValueError(f"preset name {name!r} has no usable filename characters")
    return f"{stem}.json"


def save_preset(preset: Mapping[str, object], directory: Path | None = None) -> Path:
    """Write ``preset`` as ``<sanitized name>.json`` under ``directory`` (default
    :func:`preset_dir`), creating the directory; returns the written path.

    The name is reduced to safe filename characters, so it cannot leave the
    directory; a file there that resolves elsewhere (a symlink) is refused.
    """
    name = preset.get("name")
    if not isinstance(name, str):
        raise ValueError("preset needs a string 'name'")
    target_dir = (directory if directory is not None else preset_dir()).resolve()
    path = (target_dir / preset_filename(name)).resolve()
    if path.parent != target_dir:  # e.g. an existing symlink pointing elsewhere
        raise ValueError(f"refusing to write preset outside {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preset, indent=2) + "\n")
    return path


# Example presets shipped with the package; a user preset with the same stem wins.
SAMPLE_PRESET_DIR = Path(__file__).resolve().parents[1] / "presets"


def _preset_dirs() -> tuple[Path, ...]:
    """Directories searched for presets, highest priority first."""
    return (preset_dir(), SAMPLE_PRESET_DIR)


def _preset_files() -> dict[str, Path]:
    """Preset stem -> file, user presets shadowing samples of the same stem."""
    files: dict[str, Path] = {}
    for directory in reversed(_preset_dirs()):
        if directory.is_dir():
            files.update({p.stem: p for p in directory.glob("*.json")})
    return files


def _read_preset(path: Path) -> dict[str, Any]:
    """The JSON document at ``path``; raises one of :data:`PRESET_ERRORS` if it is not a preset."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{path.name}: a preset is a JSON object")
    filters = data.get("filters", [])
    if not isinstance(filters, list):
        raise TypeError(f"{path.name}: 'filters' must be a list")
    if not all(isinstance(f, dict) and {"name", "value"} <= f.keys() for f in filters):
        raise TypeError(f"{path.name}: each filter needs a 'name' and a 'value'")
    return data


def list_presets() -> list[str]:
    """Sorted stems of every readable preset: the user's ``preset_dir()`` plus the samples.

    A file that cannot be read as a preset is skipped with a logged warning.
    """
    stems = []
    for stem, path in _preset_files().items():
        try:
            _read_preset(path)
        except PRESET_ERRORS as exc:
            logger.warning("Skipping unreadable preset %s: %s", path, exc)
            continue
        stems.append(stem)
    return sorted(stems)


def preset_label(stem: str) -> str:
    """Display label for a preset stem."""
    return stem.replace("_", " ")


def preset_path(stem: str) -> Path:
    """The file a listed preset stem refers to; KeyError for anything else."""
    try:
        return _preset_files()[stem]
    except KeyError:
        raise KeyError(f"no preset named {stem!r}") from None


def parse_preset(stem: str) -> dict[str, Any]:
    """Read a preset JSON and map backtester filter names to screener widget defaults.

    Returns dict with screener widget keys plus metadata:
        _stem, _name, _description, _regime, _etf, _stats, _raw_filters, _unmapped,
        pattern_scans, momentum_period, ratio_to_ma_period, and mapped filter values.

    The screener has one momentum and one ratio-to-MA period: the first period a
    preset uses is mapped, and filters on another period go to ``_unmapped``
    (filters the screener does not apply) instead of overwriting it. Raises one of
    :data:`PRESET_ERRORS` for an unreadable file, and ValueError for a numeric
    value that is not finite or is outside its widget's range
    (:data:`PRESET_VALUE_RANGES`).
    """
    data = _read_preset(preset_path(stem))

    raw_filters = data.get("filters", [])
    result: dict[str, Any] = {
        "_stem": stem,
        "_name": data.get("name", stem),
        "_description": data.get("description", ""),
        "_regime": data.get("regime"),
        "_etf": data.get("etf"),
        "_stats": data.get("stats", {}),
        "_raw_filters": raw_filters,
        "_discovery": data.get("discovery"),
    }
    unmapped: list[str] = []
    patterns: list[str] = []
    periods = (
        ("momentum_period", MOMENTUM_PERIOD_MAP),
        ("ratio_to_ma_period", RATIO_MA_PERIOD_MAP),
    )

    for filt in raw_filters:
        fname, fvalue = filt["name"], filt["value"]
        if fname in PRESET_FILTER_MAP:
            period_slots = [(key, table[fname]) for key, table in periods if fname in table]
            if any(result.get(key, period) != period for key, period in period_slots):
                unmapped.append(f"{fname}={fvalue}")  # a second period: one slot only
                continue
            check_preset_value(fname, fvalue)
            screener_key, converter = PRESET_FILTER_MAP[fname]
            result[screener_key] = converter(fvalue)
            for key, period in period_slots:
                result[key] = period
        elif fname in PATTERN_FILTER_NAMES:
            patterns.append(PATTERN_FILTER_NAMES[fname])
        else:
            unmapped.append(f"{fname}={fvalue}")

    if patterns:
        result["pattern_scans"] = patterns
    if unmapped:
        result["_unmapped"] = unmapped
    return result
