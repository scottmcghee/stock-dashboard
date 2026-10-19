"""Tests for logic/preset.py — preset loading and parsing."""

import json

import pytest

from stock_dashboard.config import get_settings
from stock_dashboard.logic import preset as preset_module
from stock_dashboard.logic.backtest import FILTER_TO_COLUMN, generate_strategies
from stock_dashboard.logic.preset import (
    PRESET_ERRORS,
    SAMPLE_PRESET_DIR,
    build_preset,
    list_presets,
    parse_preset,
    preset_dir,
    preset_filename,
    preset_label,
    preset_path,
    save_preset,
)
from stock_dashboard.logic.screener import percent_to_fraction

SAMPLE_STEMS = sorted(p.stem for p in SAMPLE_PRESET_DIR.glob("*.json"))


def _use_preset_dir(monkeypatch, path):
    monkeypatch.setenv("PRESET_DIR", str(path))
    get_settings.cache_clear()


@pytest.fixture
def no_samples(monkeypatch, tmp_path):
    monkeypatch.setattr(preset_module, "SAMPLE_PRESET_DIR", tmp_path / "no_samples")


class TestListPresets:
    def test_empty_dir(self, tmp_path, monkeypatch, no_samples):
        _use_preset_dir(monkeypatch, tmp_path)
        assert list_presets() == []

    def test_nonexistent_dir(self, tmp_path, monkeypatch, no_samples):
        _use_preset_dir(monkeypatch, tmp_path / "nope")
        assert list_presets() == []

    def test_populated_dir(self, tmp_path, monkeypatch, no_samples):
        _use_preset_dir(monkeypatch, tmp_path)
        (tmp_path / "beta_preset.json").write_text("{}")
        (tmp_path / "alpha_preset.json").write_text("{}")
        (tmp_path / "not_json.txt").write_text("ignore")
        result = list_presets()
        assert result == ["alpha_preset", "beta_preset"]

    def test_lists_bundled_samples_with_user_presets(self, tmp_path, monkeypatch):
        _use_preset_dir(monkeypatch, tmp_path)
        (tmp_path / "mine.json").write_text("{}")
        assert list_presets() == sorted(["mine", *SAMPLE_STEMS])

    def test_user_preset_shadows_sample_of_same_name(self, tmp_path, monkeypatch):
        _use_preset_dir(monkeypatch, tmp_path)
        stem = SAMPLE_STEMS[0]
        (tmp_path / f"{stem}.json").write_text(json.dumps({"name": "mine", "filters": []}))
        assert list_presets().count(stem) == 1
        assert preset_path(stem) == tmp_path / f"{stem}.json"
        assert parse_preset(stem)["_name"] == "mine"


class TestPresetPath:
    def test_unknown_stem_raises(self):
        with pytest.raises(KeyError):
            preset_path("does_not_exist")

    def test_traversal_is_not_a_preset(self, tmp_path, monkeypatch):
        _use_preset_dir(monkeypatch, tmp_path / "presets")
        (tmp_path / "outside.json").write_text("{}")
        with pytest.raises(KeyError):
            preset_path("../outside")


def test_preset_label():
    assert preset_label("rsi_oversold_bull") == "rsi oversold bull"


class TestSamplePresets:
    def test_ships_two_to_four_samples(self):
        assert 2 <= len(SAMPLE_STEMS) <= 4

    @pytest.mark.parametrize("stem", SAMPLE_STEMS)
    def test_sample_is_well_formed(self, stem):
        data = json.loads((SAMPLE_PRESET_DIR / f"{stem}.json").read_text())
        assert data["name"]
        assert data["description"]
        assert data["filters"]
        assert all(f["name"] in FILTER_TO_COLUMN for f in data["filters"])
        # Samples describe a method; they carry no performance numbers.
        assert data["stats"] == {}

    @pytest.mark.parametrize("stem", SAMPLE_STEMS)
    def test_sample_maps_onto_screener_widgets(self, stem):
        parsed = parse_preset(stem)
        assert "_unmapped" not in parsed
        assert parsed["_description"]


class TestBuildPreset:
    def test_format_matches_what_parse_reads(self, tmp_path, monkeypatch):
        _use_preset_dir(monkeypatch, tmp_path)
        preset = build_preset(
            "My preset",
            {"rsi_min": 60, "rs_rank_overall_min": 0.8},
            description="why",
            regimes=["bull"],
            stats={"T+20": {"mean_return": 1.0}},
        )
        assert preset == {
            "name": "My preset",
            "filters": [
                {"name": "rsi_min", "value": 60},
                {"name": "rs_rank_overall_min", "value": 0.8},
            ],
            "regime": "bull",
            "stats": {"T+20": {"mean_return": 1.0}},
            "description": "why",
        }
        path = save_preset(preset)
        parsed = parse_preset(path.stem)
        assert parsed["_regime"] == "bull"
        assert parsed["rsi_min"] == 60
        assert parsed["rs_rank_overall_min"] == pytest.approx(80.0)
        assert parsed["_description"] == "why"

    def test_defaults(self):
        preset = build_preset("p", {})
        assert preset == {"name": "p", "filters": [], "regime": None, "stats": {}}

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="empty"):
            build_preset("  ", {})

    def test_rejects_several_regimes(self):
        with pytest.raises(ValueError, match="one regime"):
            build_preset("p", {}, regimes=["bull", "bear"])

    def test_every_generated_strategy_name_is_a_valid_filename(self):
        for strategy in generate_strategies(max_filters=1):
            assert preset_filename(strategy.name).endswith(".json")


class TestSavePreset:
    def test_writes_json_under_preset_dir_by_default(self, tmp_path, monkeypatch):
        _use_preset_dir(monkeypatch, tmp_path / "new" / "dir")
        path = save_preset(build_preset("rsi_min=60 & regime=bull", {"rsi_min": 60}))
        assert path.parent == (tmp_path / "new" / "dir").resolve()
        assert path.name == "rsi_min_60_regime_bull.json"
        assert json.loads(path.read_text())["filters"] == [{"name": "rsi_min", "value": 60}]

    @pytest.mark.parametrize("name", ["../../etc/passwd", "/abs/path", "..", "a/../../b"])
    def test_refuses_path_traversal(self, tmp_path, name):
        target = tmp_path / "presets"
        try:
            path = save_preset(build_preset(name, {}), target)
        except ValueError:
            return
        assert path.parent == target.resolve()

    def test_refuses_to_write_through_a_symlink(self, tmp_path):
        target = tmp_path / "presets"
        target.mkdir()
        (target / "evil.json").symlink_to(tmp_path / "outside.json")
        with pytest.raises(ValueError, match="outside"):
            save_preset(build_preset("evil", {}), target)
        assert not (tmp_path / "outside.json").exists()

    def test_unusable_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="filename"):
            save_preset({"name": "///"}, tmp_path)

    def test_needs_a_string_name(self, tmp_path):
        with pytest.raises(ValueError, match="name"):
            save_preset({"filters": []}, tmp_path)

    def test_long_names_are_truncated(self, tmp_path):
        path = save_preset(build_preset("x" * 500, {}), tmp_path)
        assert len(path.stem) == preset_module.MAX_FILENAME_STEM


def test_preset_dir_comes_from_settings(tmp_path, monkeypatch):
    _use_preset_dir(monkeypatch, tmp_path)
    assert preset_dir() == tmp_path


class TestParsePreset:
    @pytest.fixture
    def preset_dir(self, tmp_path, monkeypatch):
        _use_preset_dir(monkeypatch, tmp_path)
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

    def test_second_momentum_period_is_not_applied(self, preset_dir):
        """One momentum slot: the first period is mapped, the other reported, never overwritten."""
        data = {
            "name": "Two periods",
            "filters": [
                {"name": "momentum_20_min", "value": 8.5},
                {"name": "momentum_126_max", "value": -3},
                {"name": "ratio_to_50ma_min", "value": 1.02},
                {"name": "ratio_to_200ma_max", "value": 0.97},
                {"name": "ratio_to_50ma_max", "value": 1.2},
            ],
        }
        (preset_dir / "two.json").write_text(json.dumps(data))
        result = parse_preset("two")
        assert (result["momentum_period"], result["momentum_min"]) == (20, 8.5)
        assert "momentum_max" not in result
        assert result["ratio_to_ma_period"] == 50
        assert (result["ratio_to_ma_min"], result["ratio_to_ma_max"]) == (1.02, 1.2)
        assert result["_unmapped"] == ["momentum_126_max=-3", "ratio_to_200ma_max=0.97"]

    def test_fractional_rsi_and_adr_are_kept(self, preset_dir):
        data = {
            "name": "Fractions",
            "filters": [
                {"name": "rsi_min", "value": 63.7},
                {"name": "adr_14_min", "value": 4.6},
            ],
        }
        (preset_dir / "frac.json").write_text(json.dumps(data))
        result = parse_preset("frac")
        assert (result["rsi_min"], result["adr_min"]) == (63.7, 4.6)

    @pytest.mark.parametrize("fraction", [0.7, 0.29, 0.57, 0.85, 0.1])
    def test_rank_round_trips_exactly(self, preset_dir, fraction):
        """Fraction -> screener % -> fraction is exact, so a rank of exactly 0.7 still passes."""
        data = {"name": "RS", "filters": [{"name": "rs_rank_overall_min", "value": fraction}]}
        (preset_dir / "rs.json").write_text(json.dumps(data))
        assert percent_to_fraction(parse_preset("rs")["rs_rank_overall_min"]) == fraction

    def test_etf_round_trips(self, preset_dir):
        save_preset(build_preset("Within SEMX", {"rs_z_score_min": 1.0}, etf="SEMX"), preset_dir)
        assert parse_preset("Within_SEMX")["_etf"] == "SEMX"
        save_preset(build_preset("Everything", {"rs_z_score_min": 1.0}), preset_dir)
        assert parse_preset("Everything")["_etf"] is None

    @pytest.mark.parametrize(
        "content",
        ['{"name": "cut', "[1, 2]", '{"filters": {"rsi_min": 3}}', '{"filters": [{"name": "x"}]}'],
    )
    def test_unreadable_file_raises_a_preset_error(self, preset_dir, content):
        (preset_dir / "bad.json").write_text(content)
        with pytest.raises(PRESET_ERRORS):
            parse_preset("bad")

    @pytest.mark.parametrize(
        ("name", "value", "message"),
        [
            ("rsi_min", 150, "rsi_min=150 is outside 0 to 100"),
            ("rs_rank_overall_min", 80, r"ranks are fractions: 0\.8 means 80%"),
            ("rs_rank_overall_max", -0.1, "outside 0 to 1"),
            ("adr_14_min", 101, "outside 0 to 100"),
        ],
    )
    def test_value_outside_its_widget_range_raises(self, preset_dir, name, value, message):
        """rsi_min 150 once crashed the screener's number_input on every rerun."""
        data = {"name": "Out", "filters": [{"name": name, "value": value}]}
        (preset_dir / "out.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match=message):
            parse_preset("out")

    @pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
    @pytest.mark.parametrize("name", ["rs_z_score_max", "momentum_63_min", "rsi_min"])
    def test_non_finite_value_raises(self, preset_dir, name, literal):
        """json.loads accepts Infinity and NaN; inf crashed the unbounded RS z-score widget."""
        filters = f'[{{"name": "{name}", "value": {literal}}}]'
        (preset_dir / "inf.json").write_text(f'{{"name": "Inf", "filters": {filters}}}')
        with pytest.raises(ValueError, match="not a finite number"):
            parse_preset("inf")

    def test_range_bounds_are_accepted(self, preset_dir):
        filters = [
            {"name": "rsi_min", "value": 0},
            {"name": "rsi_max", "value": 100},
            {"name": "rs_rank_overall_max", "value": 1.0},
        ]
        (preset_dir / "edge.json").write_text(json.dumps({"name": "Edge", "filters": filters}))
        assert parse_preset("edge")["rs_rank_overall_max"] == 100.0


class TestUnreadablePresets:
    def test_list_skips_unreadable_files_with_a_warning(self, tmp_path, monkeypatch, caplog):
        _use_preset_dir(monkeypatch, tmp_path)
        (tmp_path / "good.json").write_text(json.dumps({"name": "Good", "filters": []}))
        (tmp_path / "truncated.json").write_text('{"name": "Trunc')
        with caplog.at_level("WARNING", logger="stock_dashboard.logic.preset"):
            stems = list_presets()
        assert "good" in stems
        assert "truncated" not in stems
        assert "truncated.json" in caplog.text
