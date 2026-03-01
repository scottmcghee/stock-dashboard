"""Tests for logic/preset.py — preset loading and parsing."""

import json
from pathlib import Path

import pytest

from logic.preset import (
    MOMENTUM_PERIOD_MAP,
    PATTERN_FILTER_NAMES,
    PRESET_FILTER_MAP,
    RATIO_MA_PERIOD_MAP,
    list_presets,
    parse_preset,
)


class TestListPresets:
    def test_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("logic.preset.PRESET_DIR", tmp_path)
        assert list_presets() == []

    def test_nonexistent_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("logic.preset.PRESET_DIR", tmp_path / "nope")
        assert list_presets() == []

    def test_populated_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("logic.preset.PRESET_DIR", tmp_path)
        (tmp_path / "beta_preset.json").write_text("{}")
        (tmp_path / "alpha_preset.json").write_text("{}")
        (tmp_path / "not_json.txt").write_text("ignore")
        result = list_presets()
        assert result == ["alpha_preset", "beta_preset"]


class TestParsePreset:
    @pytest.fixture
    def preset_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("logic.preset.PRESET_DIR", tmp_path)
        return tmp_path

    def test_basic_mapping(self, preset_dir):
        data = {
            "name": "Test Preset",
            "filters": [
                {"name": "rsi_min", "value": 30},
                {"name": "rsi_max", "value": 70},
            ],
            "regime": "bull",
            "stats": {"T+20": {"mean_return": 2.5}},
        }
        (preset_dir / "test.json").write_text(json.dumps(data))
        result = parse_preset("test")
        assert result["_stem"] == "test"
        assert result["_name"] == "Test Preset"
        assert result["_regime"] == "bull"
        assert result["rsi_min"] == 30
        assert result["rsi_max"] == 70
        assert "_unmapped" not in result

    def test_momentum_period_extraction(self, preset_dir):
        data = {
            "name": "Momentum",
            "filters": [{"name": "momentum_63_min", "value": 20.0}],
        }
        (preset_dir / "mom.json").write_text(json.dumps(data))
        result = parse_preset("mom")
        assert result["momentum_min"] == 20.0
        assert result["momentum_period"] == 63

    def test_momentum_126_period(self, preset_dir):
        data = {
            "name": "Momentum 126",
            "filters": [{"name": "momentum_126_min", "value": 10.0}],
        }
        (preset_dir / "mom126.json").write_text(json.dumps(data))
        result = parse_preset("mom126")
        assert result["momentum_period"] == 126

    def test_ratio_to_ma_period_extraction(self, preset_dir):
        data = {
            "name": "Ratio",
            "filters": [{"name": "ratio_to_200ma_max", "value": 0.9}],
        }
        (preset_dir / "ratio.json").write_text(json.dumps(data))
        result = parse_preset("ratio")
        assert result["ratio_to_ma_max"] == 0.9
        assert result["ratio_to_ma_period"] == 200

    def test_ratio_to_50ma_period(self, preset_dir):
        data = {
            "name": "Ratio 50",
            "filters": [{"name": "ratio_to_50ma_min", "value": 1.05}],
        }
        (preset_dir / "ratio50.json").write_text(json.dumps(data))
        result = parse_preset("ratio50")
        assert result["ratio_to_ma_period"] == 50

    def test_raw_filters_preserved(self, preset_dir):
        filters = [
            {"name": "rsi_min", "value": 30},
            {"name": "momentum_63_min", "value": 20.0},
        ]
        data = {"name": "Raw", "filters": filters}
        (preset_dir / "raw.json").write_text(json.dumps(data))
        result = parse_preset("raw")
        assert result["_raw_filters"] == filters

    def test_pattern_filters(self, preset_dir):
        data = {
            "name": "Patterns",
            "filters": [
                {"name": "has_bull_flag", "value": True},
                {"name": "has_cup_and_handle", "value": True},
            ],
        }
        (preset_dir / "pat.json").write_text(json.dumps(data))
        result = parse_preset("pat")
        assert set(result["pattern_scans"]) == {"bull_flag", "cup_and_handle"}

    def test_unmapped_filters(self, preset_dir):
        data = {
            "name": "Unknown",
            "filters": [
                {"name": "rsi_min", "value": 30},
                {"name": "totally_unknown", "value": 42},
            ],
        }
        (preset_dir / "unk.json").write_text(json.dumps(data))
        result = parse_preset("unk")
        assert result["rsi_min"] == 30
        assert result["_unmapped"] == ["totally_unknown=42"]

    def test_rs_rank_overall_scaling(self, preset_dir):
        data = {
            "name": "RS",
            "filters": [{"name": "rs_rank_overall_min", "value": 0.8}],
        }
        (preset_dir / "rs.json").write_text(json.dumps(data))
        result = parse_preset("rs")
        assert result["rs_rank_overall_min"] == pytest.approx(80.0)

    def test_empty_filters(self, preset_dir):
        data = {"name": "Empty", "filters": []}
        (preset_dir / "empty.json").write_text(json.dumps(data))
        result = parse_preset("empty")
        assert "_unmapped" not in result
        assert "pattern_scans" not in result
        assert result["_raw_filters"] == []
