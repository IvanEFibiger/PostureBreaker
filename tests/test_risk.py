from __future__ import annotations

import unittest

from posture_guard.config import Config
from posture_guard.issues import PostureIssue
from posture_guard.models import CalibrationProfile, MetricBaseline, MetricObservation, MetricThreshold
from posture_guard.risk import evaluate_issues, evaluate_v2_issues, metric_severity


def make_profile() -> CalibrationProfile:
    return CalibrationProfile(
        side="right",
        good_mean={},
        good_std={},
        thresholds={
            "head_yaw": MetricThreshold(0.2, "directional", 1, 0.1, weight=1.2),
            "shoulder_roll": MetricThreshold(5.0, "absolute", None, 1.0, weight=0.7),
        },
    )


class MetricSeverityTests(unittest.TestCase):
    def test_directional_below_threshold_is_zero(self) -> None:
        self.assertEqual(metric_severity(0.1, MetricThreshold(0.2, "directional", 1, 0.1)), 0.0)

    def test_directional_grows_with_distance(self) -> None:
        self.assertAlmostEqual(metric_severity(0.5, MetricThreshold(0.2, "directional", 1, 0.1)), 3.0)

    def test_absolute_within_margin_is_zero(self) -> None:
        self.assertEqual(metric_severity(5.5, MetricThreshold(5.0, "absolute", None, 1.0)), 0.0)

    def test_absolute_beyond_margin_is_normalized(self) -> None:
        self.assertAlmostEqual(metric_severity(7.0, MetricThreshold(5.0, "absolute", None, 1.0)), 1.0)


class EvaluateIssuesTests(unittest.TestCase):
    def test_no_profile_is_empty(self) -> None:
        evaluation = evaluate_issues({"head_yaw": 0.5}, {"head_yaw": 1.0}, None)
        self.assertEqual(evaluation.risk_score, 0.0)
        self.assertIsNone(evaluation.dominant_issue)

    def test_weighted_risk_and_issue_mapping(self) -> None:
        evaluation = evaluate_issues({"head_yaw": 0.5}, {"head_yaw": 1.0}, make_profile())
        self.assertAlmostEqual(evaluation.risk_score, 3.0 * 1.2)
        self.assertEqual(evaluation.dominant_issue, PostureIssue.NECK_ROTATION)
        self.assertAlmostEqual(evaluation.dominant_severity, 3.6)

    def test_low_confidence_is_ignored(self) -> None:
        evaluation = evaluate_issues({"head_yaw": 0.5}, {"head_yaw": 0.3}, make_profile(), min_confidence=0.6)
        self.assertEqual(evaluation.risk_score, 0.0)

    def test_confident_metric_counts(self) -> None:
        evaluation = evaluate_issues({"head_yaw": 0.5}, {"head_yaw": 0.6}, make_profile(), min_confidence=0.6)
        self.assertGreater(evaluation.risk_score, 0.0)

    def test_disabled_threshold_is_skipped(self) -> None:
        profile = make_profile()
        profile.thresholds["head_yaw"] = MetricThreshold(0.2, "disabled", None, 0.1)
        evaluation = evaluate_issues({"head_yaw": 0.9}, {"head_yaw": 1.0}, profile)
        self.assertEqual(evaluation.risk_score, 0.0)

    def test_metric_without_issue_mapping_is_skipped(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={},
            good_std={},
            thresholds={"mystery": MetricThreshold(0.2, "directional", 1, 0.1)},
        )
        evaluation = evaluate_issues({"mystery": 0.9}, {"mystery": 1.0}, profile)
        self.assertEqual(evaluation.risk_score, 0.0)

    def test_dominant_issue_is_the_highest_weighted(self) -> None:
        profile = make_profile()
        profile.thresholds["head_yaw"] = MetricThreshold(0.2, "directional", 1, 0.1, weight=0.1)
        evaluation = evaluate_issues(
            {"head_yaw": 0.5, "shoulder_roll": 10.0},
            {"head_yaw": 1.0, "shoulder_roll": 1.0},
            profile,
        )
        self.assertEqual(evaluation.dominant_issue, PostureIssue.SHOULDER_ASYMMETRY)
        self.assertGreater(evaluation.risk_score, 0.0)

    def test_v1_correlated_metrics_do_not_inflate_risk(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={},
            good_std={},
            thresholds={
                "chin_drop": MetricThreshold(0.1, "directional", 1, 0.1, weight=1.0),
                "nose_shoulder_dx": MetricThreshold(0.1, "directional", 1, 0.1, weight=1.0),
            },
        )
        evaluation = evaluate_issues(
            {"chin_drop": 0.3, "nose_shoulder_dx": 0.3},
            {"chin_drop": 1.0, "nose_shoulder_dx": 1.0},
            profile,
        )
        self.assertAlmostEqual(evaluation.risk_score, 2.0)  # max, not 4.0


def baseline(center: float = 0.0, spread: float = 0.0) -> MetricBaseline:
    return MetricBaseline(center=center, spread=spread, coverage=1.0, confidence=1.0, sample_count=10)


class EvaluateV2IssuesTests(unittest.TestCase):
    def test_normalizes_deviation_against_baseline(self) -> None:
        observations = {"head_forward_ratio": MetricObservation(0.20, 1.0)}
        evaluation = evaluate_v2_issues(observations, {"head_forward_ratio": baseline()}, {}, Config())
        # deviation 0.20, margin 0.08, scale 0.08 -> severity 1.5; weight 1.5 -> risk 2.25
        self.assertAlmostEqual(evaluation.risk_score, 2.25)
        self.assertEqual(evaluation.dominant_issue, PostureIssue.HEAD_FORWARD)

    def test_within_baseline_is_not_an_issue(self) -> None:
        observations = {"head_forward_ratio": MetricObservation(0.02, 1.0)}
        evaluation = evaluate_v2_issues(observations, {"head_forward_ratio": baseline()}, {}, Config())
        self.assertEqual(evaluation.risk_score, 0.0)

    def test_missing_baseline_is_skipped(self) -> None:
        observations = {"head_forward_ratio": MetricObservation(0.20, 1.0)}
        self.assertEqual(evaluate_v2_issues(observations, {}, {}, Config()).risk_score, 0.0)

    def test_view_context_metric_has_no_issue(self) -> None:
        observations = {"head_yaw": MetricObservation(0.90, 1.0)}
        evaluation = evaluate_v2_issues(observations, {"head_yaw": baseline()}, {}, Config())
        self.assertEqual(evaluation.risk_score, 0.0)

    def test_neck_roll_delta_is_observe_only(self) -> None:
        observations = {"neck_roll_delta": MetricObservation(20.0, 1.0)}
        evaluation = evaluate_v2_issues(observations, {}, {"neck_roll_delta": baseline()}, Config())
        self.assertEqual(evaluation.risk_score, 0.0)

    def test_noise_floor_suppresses_tiny_deviations(self) -> None:
        observations = {"head_forward_ratio": MetricObservation(0.10, 1.0)}
        noisy = {"head_forward_ratio": baseline(spread=0.05)}
        self.assertEqual(evaluate_v2_issues(observations, noisy, {}, Config()).risk_score, 0.0)

    def test_config_min_confidence_governs_v2_runtime(self) -> None:
        observations = {"head_forward_ratio": MetricObservation(0.5, 0.65)}
        baselines = {"head_forward_ratio": baseline()}
        self.assertEqual(evaluate_v2_issues(observations, baselines, {}, Config(metric_min_confidence=0.7)).risk_score, 0.0)
        self.assertGreater(evaluate_v2_issues(observations, baselines, {}, Config(metric_min_confidence=0.6)).risk_score, 0.0)

    def test_shoulder_sides_collapse_to_one_issue(self) -> None:
        sides = {
            "left_shoulder_elevation": MetricObservation(-0.5, 1.0),
            "right_shoulder_elevation": MetricObservation(-0.5, 1.0),
        }
        baselines = {name: baseline() for name in sides}
        evaluation = evaluate_v2_issues(sides, baselines, {}, Config())
        single = evaluate_v2_issues(
            {"left_shoulder_elevation": MetricObservation(-0.5, 1.0)},
            {"left_shoulder_elevation": baseline()},
            {},
            Config(),
        )
        self.assertAlmostEqual(evaluation.risk_score, single.risk_score)
        self.assertAlmostEqual(evaluation.dominant_severity, single.dominant_severity)

    def test_aggregate_shoulder_elevation_contributes_no_risk(self) -> None:
        observations = {"shoulder_elevation": MetricObservation(-0.5, 1.0)}
        evaluation = evaluate_v2_issues(observations, {"shoulder_elevation": baseline()}, {}, Config())
        self.assertEqual(evaluation.risk_score, 0.0)


if __name__ == "__main__":
    unittest.main()
