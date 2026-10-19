"""Each research script runs end to end on the seeded demo data and writes its outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research import (
    option_signal_analysis,
    signal_analysis,
    ta_signal_analysis,
    trend_precursor_analysis,
)
from stock_dashboard.config import Settings
from stock_dashboard.logic.backtest import METRIC_COLUMNS, RS_IMPULSE_COLUMNS
from stock_dashboard.logic.preset_signal import unsupported_filters


def _rows(path: Path) -> int:
    assert path.is_file(), f"missing {path.name}"
    return len(pd.read_csv(path))


def test_signal_analysis(settings: Settings, tmp_path: Path) -> None:
    assert signal_analysis.main(["--horizons", "1,5", "--output-dir", str(tmp_path)]) == 0
    bins = pd.read_csv(tmp_path / signal_analysis.OUTPUT_FILE)
    assert set(bins["signal"]) == {s.name for s in signal_analysis.SIGNALS}
    assert {"mean_fwd_5_bps", "p_fdr_5"} <= set(bins.columns)


def test_signal_analysis_with_unknown_symbols_writes_an_empty_report(
    settings: Settings, tmp_path: Path
) -> None:
    assert signal_analysis.main(["--symbols", "NOSUCH", "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / signal_analysis.OUTPUT_FILE).is_file()


def test_ta_signal_analysis(settings: Settings, tmp_path: Path) -> None:
    presets = tmp_path / "presets"
    argv = [
        "--universe-size", "15",
        "--start", "2024-01-01",
        "--target", "resid_fwd_10",
        "--min-n", "200",
        "--min-group-n", "200",
        "--min-time-half-n", "100",
        "--min-event-n", "20",
        "--min-pair-n", "30",
        "--min-hit-rate", "0",
        "--min-regimes-replicate", "0",
        "--min-time-halves-replicate", "0",
        "--export-presets",
        "--preset-dir", str(presets),
        "--output-dir", str(tmp_path),
    ]  # fmt: skip
    assert ta_signal_analysis.main(argv) == 0
    stem = "universe15_resid_fwd_10"
    assert _rows(tmp_path / f"ta_univariate_{stem}.csv") > 0
    for name in ("per_regime", "per_sector", "per_time", "events_overall", "pairs"):
        assert (tmp_path / f"ta_{name}_{stem}.csv").is_file()
    metric_columns = {*METRIC_COLUMNS, *RS_IMPULSE_COLUMNS}
    for path in presets.glob("*.json"):
        assert unsupported_filters(json.loads(path.read_text())["filters"], metric_columns) == []


def test_ta_signal_analysis_with_no_data_in_range(settings: Settings, tmp_path: Path) -> None:
    argv = ["--start", "2099-01-01", "--output-dir", str(tmp_path)]
    assert ta_signal_analysis.main(argv) == 0


def test_option_signal_analysis(settings: Settings, tmp_path: Path) -> None:
    argv = [
        "--universe-size", "2",
        "--sessions", "60",
        "--min-ticker-n", "200",
        "--min-group-n", "200",
        "--output-dir", str(tmp_path),
    ]  # fmt: skip
    assert option_signal_analysis.main(argv) == 0
    univariate = list(tmp_path.glob("univariate_*_h5_residual_ret.csv"))
    assert len(univariate) == 1
    assert _rows(univariate[0]) > 0
    for name in ("structural", "per_ticker", "replication", "pairwise"):
        assert len(list(tmp_path.glob(f"{name}_*.csv"))) == 1


def test_option_signal_analysis_before_any_chain_data(settings: Settings, tmp_path: Path) -> None:
    argv = ["--start", "2022-01-03", "--end", "2022-06-30", "--output-dir", str(tmp_path)]
    assert option_signal_analysis.main(argv) == 0


def test_trend_precursor_analysis(settings: Settings, tmp_path: Path) -> None:
    argv = ["--focus", "TCHX", "--baseline-samples", "500", "--output-dir", str(tmp_path)]
    assert trend_precursor_analysis.main(argv) == 0
    assert _rows(tmp_path / "trend_trends.csv") > 0
    assert _rows(tmp_path / "trend_profiles.csv") > 0
    assert _rows(tmp_path / "trend_baseline.csv") == 500
    comparison = pd.read_csv(tmp_path / "trend_comparison.csv")
    assert set(comparison["indicator"]) == set(trend_precursor_analysis.INDICATORS)
    assert _rows(tmp_path / "trend_regime_split.csv") == 4


def test_trend_precursor_analysis_without_trends(settings: Settings, tmp_path: Path) -> None:
    argv = ["--min-move", "500", "--output-dir", str(tmp_path)]
    assert trend_precursor_analysis.main(argv) == 0
    assert (tmp_path / "trend_comparison.csv").is_file()
