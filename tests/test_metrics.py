from __future__ import annotations

import unittest

from posture_guard.config import Config
from posture_guard.issues import PostureIssue
from posture_guard.metrics import (
    METRIC_SPECS,
    CalibrationScope,
    DeviationMode,
    MetricSpec,
    build_metric_baselines,
    build_observations,
    normalize_observation,
    resolve_baseline,
    spec_for,
)
from posture_guard.models import (
    ForwardState,
    MetricBaseline,
    MetricObservation,
    ShoulderState,
    ViewState,
)


def baseline(center: float = 3.0, spread: float = 1.0, confidence: float = 0.9) -> MetricBaseline:
    return MetricBaseline(center=center, spread=spread, coverage=1.0, confidence=confidence, sample_count=5)


VIEW_SPEC = MetricSpec(PostureIssue.HEAD_FORWARD, CalibrationScope.VIEW, DeviationMode.HIGHER_IS_WORSE)
TWO_SIDED_SPEC = MetricSpec(PostureIssue.SHOULDER_ASYMMETRY, CalibrationScope.VIEW, DeviationMode.TWO_SIDED)
LOWER_SPEC = MetricSpec(PostureIssue.NECK_FLEXION, CalibrationScope.VIEW, DeviationMode.LOWER_IS_WORSE)


class NormalizeObservationTests(unittest.TestCase):
    def test_no_baseline_is_zero(self) -> None:
        self.assertEqual(normalize_observation(MetricObservation(10.0, 1.0), None, VIEW_SPEC, 0.5), 0.0)

    def test_low_confidence_is_zero(self) -> None:
        spec = MetricSpec(PostureIssue.HEAD_FORWARD, CalibrationScope.VIEW, DeviationMode.HIGHER_IS_WORSE, min_confidence=0.8)
        self.assertEqual(normalize_observation(MetricObservation(10.0, 0.4), baseline(), spec, 0.5), 0.0)

    def test_higher_is_worse(self) -> None:
        self.assertAlmostEqual(normalize_observation(MetricObservation(10.0, 1.0), baseline(), VIEW_SPEC, 0.5), 6.0)

    def test_higher_is_worse_below_margin_is_zero(self) -> None:
        self.assertEqual(normalize_observation(MetricObservation(4.0, 1.0), baseline(), VIEW_SPEC, 0.5), 0.0)

    def test_two_sided_uses_absolute_deviation(self) -> None:
        self.assertAlmostEqual(normalize_observation(MetricObservation(-3.0, 1.0), baseline(), TWO_SIDED_SPEC, 0.5), 4.0)

    def test_lower_is_worse_flips_sign(self) -> None:
        self.assertAlmostEqual(normalize_observation(MetricObservation(-5.0, 1.0), baseline(), LOWER_SPEC, 0.5), 8.0)

    def test_noise_floor_raises_allowed_margin(self) -> None:
        # spread 5 -> noise margin 20, so a deviation of 10 is still fine.
        self.assertEqual(
            normalize_observation(MetricObservation(13.0, 1.0), baseline(spread=5.0), VIEW_SPEC, 0.5),
            0.0,
        )


class BuildMetricBaselinesTests(unittest.TestCase):
    def test_robust_statistics(self) -> None:
        frames = [{"m": MetricObservation(float(value), 0.9)} for value in (1.0, 2.0, 3.0, 4.0, 5.0)]
        baselines = build_metric_baselines(frames, Config())
        self.assertAlmostEqual(baselines["m"].center, 3.0)
        self.assertAlmostEqual(baselines["m"].spread, 1.0)
        self.assertAlmostEqual(baselines["m"].mean, 3.0)
        self.assertEqual(baselines["m"].sample_count, 5)

    def test_low_coverage_is_excluded(self) -> None:
        frames = [{"m": MetricObservation(1.0, 0.9)}] + [{} for _ in range(3)]
        baselines = build_metric_baselines(frames, Config())
        self.assertEqual(baselines, {})

    def test_low_confidence_is_excluded(self) -> None:
        frames = [{"m": MetricObservation(1.0, 0.2)} for _ in range(4)]
        baselines = build_metric_baselines(frames, Config())
        self.assertEqual(baselines, {})

    def test_empty_frames(self) -> None:
        self.assertEqual(build_metric_baselines([], Config()), {})


class ResolveBaselineTests(unittest.TestCase):
    def test_view_scope_uses_view_baselines(self) -> None:
        view = {"head_roll": baseline()}
        spec = spec_for("head_roll")
        self.assertIs(resolve_baseline("head_roll", spec, view, {}), view["head_roll"])

    def test_global_scope_uses_global_baselines(self) -> None:
        global_baselines = {"neck_roll_delta": baseline()}
        spec = spec_for("neck_roll_delta")
        self.assertIs(resolve_baseline("neck_roll_delta", spec, {}, global_baselines), global_baselines["neck_roll_delta"])


class MetricSpecTests(unittest.TestCase):
    def test_view_context_signals_have_no_issue(self) -> None:
        self.assertIsNone(spec_for("head_yaw").issue)
        self.assertEqual(spec_for("head_yaw").calibration_scope, CalibrationScope.NONE)

    def test_neck_roll_delta_is_global(self) -> None:
        self.assertEqual(spec_for("neck_roll_delta").calibration_scope, CalibrationScope.GLOBAL)

    def test_every_spec_name_is_known(self) -> None:
        for name in METRIC_SPECS:
            self.assertIsNotNone(spec_for(name))


class BuildObservationsTests(unittest.TestCase):
    def test_collects_all_states(self) -> None:
        observations = build_observations(
            ViewState(head_yaw=0.3, head_pitch=0.1, torso_yaw=None, confidence=0.9),
            ShoulderState(roll=2.0, elevation=0.3, confidence=0.8),
            ForwardState(head_forward_ratio=0.2, torso_forward_angle=6.0, confidence=0.7),
        )
        self.assertAlmostEqual(observations["head_yaw"].value, 0.3)
        self.assertAlmostEqual(observations["head_yaw"].confidence, 0.9)
        self.assertAlmostEqual(observations["shoulder_roll"].value, 2.0)
        self.assertAlmostEqual(observations["shoulder_elevation"].value, 0.3)
        self.assertAlmostEqual(observations["torso_forward_angle"].value, 6.0)
        self.assertNotIn("torso_yaw", observations)

    def test_empty_states(self) -> None:
        self.assertEqual(build_observations(None, None, None), {})


if __name__ == "__main__":
    unittest.main()
