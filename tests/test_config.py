from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from posture_guard.config import Config, apply_settings, load_config, save_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "posture_break_guard.config.json"


class ShippedConfigTests(unittest.TestCase):
    def test_shipped_config_loads_into_dataclass(self) -> None:
        self.assertTrue(CONFIG_PATH.exists())
        config = load_config(CONFIG_PATH)
        self.assertIsInstance(config, Config)
        self.assertEqual(config.posture_min_bad_metrics, 2)

    def test_falls_back_to_defaults_when_file_is_missing(self) -> None:
        config = load_config(REPO_ROOT / "missing.config.json")
        self.assertEqual(config.posture_min_bad_metrics, 2)


class ConfigValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "config.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, payload: object) -> None:
        if isinstance(payload, str):
            self.path.write_text(payload, encoding="utf-8")
        else:
            self.path.write_text(json.dumps(payload), encoding="utf-8")

    def test_valid_override_is_applied(self) -> None:
        self._write({"break_interval_minutes": 30})
        self.assertEqual(load_config(self.path).break_interval_minutes, 30)

    def test_unknown_key_raises_with_key_name(self) -> None:
        self._write({"not_a_real_key": 1})
        with self.assertRaises(ValueError) as ctx:
            load_config(self.path)
        self.assertIn("not_a_real_key", str(ctx.exception))

    def test_min_visibility_above_one_raises(self) -> None:
        self._write({"min_visibility": 1.5})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_min_visibility_below_zero_raises(self) -> None:
        self._write({"min_visibility": -0.1})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_metric_min_confidence_above_one_raises(self) -> None:
        self._write({"metric_min_confidence": 1.5})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_metric_min_confidence_below_zero_raises(self) -> None:
        self._write({"metric_min_confidence": -0.1})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_metric_min_confidence_default(self) -> None:
        self.assertEqual(Config().metric_min_confidence, 0.60)

    def test_calibration_min_metric_coverage_above_one_raises(self) -> None:
        self._write({"calibration_min_metric_coverage": 1.5})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_calibration_min_metric_coverage_below_zero_raises(self) -> None:
        self._write({"calibration_min_metric_coverage": -0.1})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_zero_smoothing_min_observations_raises(self) -> None:
        self._write({"smoothing_min_observations": 0})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_config_defaults_for_new_fields(self) -> None:
        config = Config()
        self.assertEqual(config.calibration_min_metric_coverage, 0.75)
        self.assertEqual(config.smoothing_min_observations, 6)
        self.assertEqual(config.debug_snapshot_path, "history/debug_snapshots.jsonl")
        self.assertTrue(config.posture_v2_observe_only)
        self.assertEqual(config.view_switch_stability_seconds, 0.75)

    def test_invalid_observe_only_type_raises(self) -> None:
        self._write({"posture_v2_observe_only": "yes"})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_default_weights_are_present(self) -> None:
        self.assertEqual(Config().default_weights["head_forward_ratio"], 1.5)

    def test_negative_default_weight_raises(self) -> None:
        self._write({"default_weights": {"head_yaw": -1.0}})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_negative_duration_raises(self) -> None:
        self._write({"break_interval_minutes": -5})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_zero_calibration_frames_raises(self) -> None:
        self._write({"calibration_frames": 0})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_empty_model_path_raises(self) -> None:
        self._write({"model_path": "  "})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_non_numeric_duration_raises(self) -> None:
        self._write({"target_fps": "fast"})
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_invalid_json_raises_value_error(self) -> None:
        self._write("{ not valid json")
        with self.assertRaises(ValueError):
            load_config(self.path)

    def test_non_object_json_raises(self) -> None:
        self._write([1, 2, 3])
        with self.assertRaises(ValueError):
            load_config(self.path)


class SaveConfigTests(unittest.TestCase):
    def test_round_trip_preserves_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(path, Config(break_interval_minutes=33, camera_index=2))
            loaded = load_config(path)
            self.assertEqual(loaded.break_interval_minutes, 33)
            self.assertEqual(loaded.camera_index, 2)

    def test_save_rejects_invalid_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_config(Path(tmp) / "config.json", Config(min_visibility=2.0))


class ApplySettingsTests(unittest.TestCase):
    def test_apply_updates_config_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_config(path, Config())
            config = load_config(path)
            apply_settings(config, {"min_visibility": 0.7, "break_interval_minutes": 30}, path)
            self.assertEqual(config.min_visibility, 0.7)
            self.assertEqual(load_config(path).break_interval_minutes, 30)

    def test_apply_invalid_leaves_config_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = Config()
            with self.assertRaises(ValueError):
                apply_settings(config, {"min_visibility": 2.0}, path)
            self.assertEqual(config.min_visibility, 0.55)


if __name__ == "__main__":
    unittest.main()
