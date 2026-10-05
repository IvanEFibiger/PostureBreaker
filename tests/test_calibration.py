from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from posture_guard.calibration import (
    Calibrator,
    build_thresholds,
    calibration_quality,
    load_calibration_set,
    orientation_summary,
    save_calibration_set,
)
from posture_guard.config import Config
from posture_guard.models import CalibrationProfile, CalibrationSet, DetectionMetrics, ViewProfile, ViewState


class BuildThresholdsTests(unittest.TestCase):
    def test_directional_when_bad_mean_present(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"ear_shoulder_dx": 0.10},
            good_std={"ear_shoulder_dx": 0.01},
            bad_mean={"ear_shoulder_dx": 0.06},
        )
        threshold = build_thresholds(profile, Config())["ear_shoulder_dx"]
        self.assertEqual(threshold.mode, "directional")
        self.assertEqual(threshold.direction, -1)
        self.assertAlmostEqual(threshold.threshold, 0.10 + (0.06 - 0.10) * 0.55, places=9)

    def test_direction_positive_when_bad_above_good(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"chin_drop": 0.10},
            good_std={"chin_drop": 0.01},
            bad_mean={"chin_drop": 0.20},
        )
        threshold = build_thresholds(profile, Config())["chin_drop"]
        self.assertEqual(threshold.direction, 1)
        self.assertAlmostEqual(threshold.threshold, 0.10 + (0.20 - 0.10) * 0.55, places=9)

    def test_absolute_when_no_bad_mean(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"torso_lean_dx": 0.12},
            good_std={"torso_lean_dx": 0.01},
        )
        threshold = build_thresholds(profile, Config())["torso_lean_dx"]
        self.assertEqual(threshold.mode, "absolute")
        self.assertIsNone(threshold.direction)
        self.assertAlmostEqual(threshold.threshold, 0.12, places=9)

    def test_default_margin_when_std_is_small(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"ear_shoulder_dx": 0.10},
            good_std={"ear_shoulder_dx": 0.01},
            bad_mean={"ear_shoulder_dx": 0.06},
        )
        threshold = build_thresholds(profile, Config())["ear_shoulder_dx"]
        self.assertAlmostEqual(threshold.margin, 0.035, places=9)

    def test_margin_grows_with_std(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"ear_shoulder_dx": 0.10},
            good_std={"ear_shoulder_dx": 0.04},
            bad_mean={"ear_shoulder_dx": 0.30},
        )
        threshold = build_thresholds(profile, Config())["ear_shoulder_dx"]
        self.assertAlmostEqual(threshold.margin, 0.04 * 2.5, places=9)

    def test_indistinguishable_metric_is_disabled(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"ear_shoulder_dx": 0.10},
            good_std={"ear_shoulder_dx": 0.01},
            bad_mean={"ear_shoulder_dx": 0.11},
        )
        thresholds = build_thresholds(profile, Config())
        self.assertEqual(thresholds["ear_shoulder_dx"].mode, "disabled")

    def test_does_not_raise_for_good_only_calibration(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"ear_shoulder_dx": 0.10},
            good_std={"ear_shoulder_dx": 0.01},
        )
        thresholds = build_thresholds(profile, Config())
        self.assertEqual(thresholds["ear_shoulder_dx"].mode, "absolute")


def _metrics(values: dict[str, float], side: str = "right") -> DetectionMetrics:
    return DetectionMetrics(side=side, values=dict(values), points={})


class CalibrationQualityTests(unittest.TestCase):
    def test_strong_directional_metric_scores_100(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"a": 0.10},
            good_std={"a": 0.01},
            bad_mean={"a": 0.40},
        )
        thresholds = build_thresholds(profile, Config())
        self.assertEqual(calibration_quality(profile, thresholds), 100.0)

    def test_disabled_metric_lowers_quality(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"a": 0.10, "b": 0.10},
            good_std={"a": 0.01, "b": 0.01},
            bad_mean={"a": 0.40, "b": 0.10},
        )
        thresholds = build_thresholds(profile, Config())
        self.assertEqual(calibration_quality(profile, thresholds), 50.0)

    def test_good_only_profile_uses_absolute_confidence(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"a": 0.10},
            good_std={"a": 0.01},
        )
        thresholds = build_thresholds(profile, Config())
        self.assertEqual(calibration_quality(profile, thresholds), 40.0)

    def test_empty_thresholds_score_zero(self) -> None:
        profile = CalibrationProfile(side="right", good_mean={}, good_std={})
        self.assertEqual(calibration_quality(profile, {}), 0.0)


class BuildProfileTests(unittest.TestCase):
    def test_bad_calibration_with_too_few_discriminative_metrics_is_rejected(self) -> None:
        config = Config(calibration_frames=2)
        calibrator = Calibrator(2)
        calibrator.start("good")
        calibrator.add(_metrics({"a": 0.10, "b": 0.10}))
        calibrator.add(_metrics({"a": 0.10, "b": 0.10}))
        good = calibrator.build_profile(None, config)

        calibrator.start("bad", expected_side="right")
        calibrator.add(_metrics({"a": 0.10, "b": 0.10}))
        calibrator.add(_metrics({"a": 0.10, "b": 0.10}))
        with self.assertRaises(ValueError):
            calibrator.build_profile(good, config)

    def test_bad_calibration_with_enough_discriminative_metrics_is_accepted(self) -> None:
        config = Config(calibration_frames=2)
        calibrator = Calibrator(2)
        calibrator.start("good")
        calibrator.add(_metrics({"a": 0.10, "b": 0.10, "c": 0.10}))
        calibrator.add(_metrics({"a": 0.10, "b": 0.10, "c": 0.10}))
        good = calibrator.build_profile(None, config)

        calibrator.start("bad", expected_side="right")
        calibrator.add(_metrics({"a": 0.40, "b": 0.50, "c": 0.10}))
        calibrator.add(_metrics({"a": 0.40, "b": 0.50, "c": 0.10}))
        profile = calibrator.build_profile(good, config)

        modes = {name: threshold.mode for name, threshold in profile.thresholds.items()}
        self.assertEqual(modes["a"], "directional")
        self.assertEqual(modes["b"], "directional")
        self.assertEqual(modes["c"], "disabled")
        self.assertGreater(profile.quality_score, 0.0)

    def test_partial_samples_are_averaged_over_present_frames(self) -> None:
        calibrator = Calibrator(4)
        calibrator.start("good")
        calibrator.add(_metrics({"a": 0.10, "b": 0.20}))
        calibrator.add(_metrics({"a": 0.30}))
        calibrator.add(_metrics({"a": 0.30, "b": 0.40}))
        calibrator.add(_metrics({"a": 0.30, "b": 0.60}))
        profile = calibrator.build_profile(None, Config(calibration_frames=4))
        self.assertAlmostEqual(profile.good_mean["a"], 0.25)
        self.assertAlmostEqual(profile.good_mean["b"], 0.40)

    def test_metric_with_coverage_above_threshold_is_kept(self) -> None:
        calibrator = Calibrator(4)
        calibrator.start("good")
        for index in range(4):
            calibrator.add(_metrics({"a": 0.10, "b": 0.50} if index < 3 else {"a": 0.10}))
        profile = calibrator.build_profile(None, Config(calibration_frames=4))
        self.assertIn("b", profile.good_mean)

    def test_metric_with_low_coverage_is_excluded(self) -> None:
        calibrator = Calibrator(4)
        calibrator.start("good")
        calibrator.add(_metrics({"a": 0.10, "b": 0.50, "c": 0.10}))
        calibrator.add(_metrics({"a": 0.10, "c": 0.10}))
        calibrator.add(_metrics({"a": 0.10, "c": 0.10}))
        calibrator.add(_metrics({"a": 0.10, "c": 0.10}))
        profile = calibrator.build_profile(None, Config(calibration_frames=4))
        self.assertIn("a", profile.good_mean)
        self.assertNotIn("b", profile.good_mean)

    def test_calibrator_records_orientation_summary(self) -> None:
        calibrator = Calibrator(2)
        calibrator.start("good")
        calibrator.add(
            DetectionMetrics(side="right", values={"a": 0.1, "b": 0.1}, view=ViewState(head_yaw=0.2, torso_yaw=0.05))
        )
        calibrator.add(
            DetectionMetrics(side="right", values={"a": 0.1, "b": 0.1}, view=ViewState(head_yaw=0.4, torso_yaw=0.05))
        )
        calibrator.build_profile(None, Config(calibration_frames=2))
        head_mean, torso_mean, _, _ = calibrator.last_orientation
        self.assertAlmostEqual(head_mean, 0.3)
        self.assertAlmostEqual(torso_mean, 0.05)


class OrientationSummaryTests(unittest.TestCase):
    def test_ignores_none_and_missing_views(self) -> None:
        head_mean, torso_mean, head_std, torso_std = orientation_summary(
            [ViewState(head_yaw=0.2, torso_yaw=0.1), ViewState(head_yaw=0.4), None]
        )
        self.assertAlmostEqual(head_mean, 0.3)
        self.assertAlmostEqual(torso_mean, 0.1)
        self.assertGreater(head_std, 0.0)
        self.assertEqual(torso_std, 0.0)

    def test_all_missing_is_none(self) -> None:
        self.assertEqual(orientation_summary([None, ViewState()]), (None, None, 0.0, 0.0))


class CalibrationSetTests(unittest.TestCase):
    def test_v1_profile_migrates_to_set(self) -> None:
        v1 = CalibrationProfile(side="right", good_mean={"a": 0.1}, good_std={"a": 0.01}).to_json()
        calibration_set = CalibrationSet.from_json(v1)
        self.assertEqual(calibration_set.schema_version, 2)
        self.assertEqual(len(calibration_set.profiles), 1)
        self.assertEqual(calibration_set.active_profile().id, "principal")
        self.assertEqual(calibration_set.active_profile().calibration.good_mean, {"a": 0.1})

    def test_set_round_trip_preserves_orientation_and_posture(self) -> None:
        view = ViewProfile(
            id="monitor_1",
            name="Monitor 1",
            head_yaw_mean=0.38,
            torso_yaw_mean=0.04,
            head_yaw_std=0.03,
            torso_yaw_std=0.02,
            calibration=CalibrationProfile(side="right", good_mean={"a": 0.1}, good_std={"a": 0.01}),
        )
        calibration_set = CalibrationSet(profiles=[view], active_profile_id="monitor_1")
        restored = CalibrationSet.from_json(calibration_set.to_json())
        self.assertEqual(restored.active_profile().name, "Monitor 1")
        self.assertAlmostEqual(restored.active_profile().head_yaw_mean, 0.38)
        self.assertEqual(restored.active_profile().calibration.side, "right")

    def test_missing_active_id_falls_back_to_first_profile(self) -> None:
        profile = ViewProfile(id="a", name="A")
        calibration_set = CalibrationSet(profiles=[profile], active_profile_id="missing")
        self.assertIs(calibration_set.active_profile(), profile)

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            original = CalibrationSet(
                profiles=[ViewProfile(id="monitor_1", name="Monitor 1")],
                active_profile_id="monitor_1",
            )
            save_calibration_set(path, original)
            loaded = load_calibration_set(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.active_profile().name, "Monitor 1")

    def test_load_missing_file_is_none(self) -> None:
        self.assertIsNone(load_calibration_set(Path("nope.json")))


if __name__ == "__main__":
    unittest.main()
