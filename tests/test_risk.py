from __future__ import annotations

import unittest

from posture_guard.issues import PostureIssue
from posture_guard.models import CalibrationProfile, MetricThreshold
from posture_guard.risk import evaluate_issues, metric_severity


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


if __name__ == "__main__":
    unittest.main()
