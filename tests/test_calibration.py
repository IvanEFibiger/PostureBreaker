from __future__ import annotations

import unittest

from posture_guard.calibration import build_thresholds
from posture_guard.config import Config
from posture_guard.models import CalibrationProfile


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

    def test_raises_when_good_and_bad_are_indistinguishable(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"ear_shoulder_dx": 0.10},
            good_std={"ear_shoulder_dx": 0.01},
            bad_mean={"ear_shoulder_dx": 0.11},
        )
        with self.assertRaises(ValueError) as ctx:
            build_thresholds(profile, Config())
        self.assertIn("ear_shoulder_dx", str(ctx.exception))

    def test_does_not_raise_for_good_only_calibration(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={"ear_shoulder_dx": 0.10},
            good_std={"ear_shoulder_dx": 0.01},
        )
        thresholds = build_thresholds(profile, Config())
        self.assertEqual(thresholds["ear_shoulder_dx"].mode, "absolute")


if __name__ == "__main__":
    unittest.main()
