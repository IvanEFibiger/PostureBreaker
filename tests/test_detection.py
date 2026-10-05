from __future__ import annotations

import unittest

from posture_guard.detection import classify_posture, dominant_issue, issue_details
from posture_guard.models import CalibrationProfile, DetectionMetrics, MetricThreshold


def make_profile(side: str = "right") -> CalibrationProfile:
    return CalibrationProfile(
        side=side,
        good_mean={},
        good_std={},
        thresholds={
            "ear_shoulder_dx": MetricThreshold(0.10, "directional", -1, 0.05),
            "chin_drop": MetricThreshold(0.20, "directional", 1, 0.05),
            "torso_lean_dx": MetricThreshold(0.30, "absolute", None, 0.05),
        },
    )


def make_metrics(side: str = "right", **values: float) -> DetectionMetrics:
    return DetectionMetrics(side=side, values=dict(values), points={})


class ClassifyPostureTests(unittest.TestCase):
    def test_no_profile_is_not_bad(self) -> None:
        result = classify_posture(None, make_metrics(ear_shoulder_dx=0.0))
        self.assertEqual(result, (False, {}, {}))

    def test_no_metrics_is_not_bad(self) -> None:
        self.assertEqual(classify_posture(make_profile(), None), (False, {}, {}))

    def test_side_mismatch_is_not_bad(self) -> None:
        result = classify_posture(make_profile("right"), make_metrics(side="left", ear_shoulder_dx=0.0))
        self.assertEqual(result, (False, {}, {}))

    def test_directional_metric_bad_below_threshold_for_negative_direction(self) -> None:
        _, bad_by_metric, severity = classify_posture(
            make_profile(), make_metrics(ear_shoulder_dx=0.05)
        )
        self.assertTrue(bad_by_metric["ear_shoulder_dx"])
        self.assertGreater(severity["ear_shoulder_dx"], 0.0)

    def test_directional_metric_good_above_threshold_for_negative_direction(self) -> None:
        _, bad_by_metric, severity = classify_posture(
            make_profile(), make_metrics(ear_shoulder_dx=0.15)
        )
        self.assertFalse(bad_by_metric["ear_shoulder_dx"])
        self.assertEqual(severity["ear_shoulder_dx"], 0.0)

    def test_directional_metric_bad_above_threshold_for_positive_direction(self) -> None:
        _, bad_by_metric, _ = classify_posture(make_profile(), make_metrics(chin_drop=0.30))
        self.assertTrue(bad_by_metric["chin_drop"])

    def test_absolute_metric_bad_beyond_margin(self) -> None:
        _, bad_by_metric, _ = classify_posture(make_profile(), make_metrics(torso_lean_dx=0.40))
        self.assertTrue(bad_by_metric["torso_lean_dx"])

    def test_absolute_metric_good_within_margin(self) -> None:
        _, bad_by_metric, _ = classify_posture(make_profile(), make_metrics(torso_lean_dx=0.28))
        self.assertFalse(bad_by_metric["torso_lean_dx"])

    def test_single_bad_metric_is_not_enough_to_flag_posture(self) -> None:
        is_bad, bad_by_metric, _ = classify_posture(
            make_profile(), make_metrics(ear_shoulder_dx=0.05)
        )
        self.assertFalse(is_bad)
        self.assertEqual(sum(1 for is_metric_bad in bad_by_metric.values() if is_metric_bad), 1)

    def test_two_bad_metrics_flag_posture(self) -> None:
        is_bad, bad_by_metric, _ = classify_posture(
            make_profile(), make_metrics(ear_shoulder_dx=0.05, chin_drop=0.30)
        )
        self.assertTrue(is_bad)
        self.assertEqual(sum(1 for is_metric_bad in bad_by_metric.values() if is_metric_bad), 2)

    def test_unknown_metric_is_ignored(self) -> None:
        is_bad, bad_by_metric, severity = classify_posture(
            make_profile(), make_metrics(unknown_metric=999.0)
        )
        self.assertFalse(is_bad)
        self.assertEqual(bad_by_metric, {})
        self.assertEqual(severity, {})

    def test_severity_is_never_negative(self) -> None:
        _, _, severity = classify_posture(
            make_profile(), make_metrics(ear_shoulder_dx=0.05, chin_drop=0.30, torso_lean_dx=0.40)
        )
        self.assertTrue(all(value >= 0.0 for value in severity.values()))


class DominantIssueTests(unittest.TestCase):
    def test_picks_highest_severity_bad_metric(self) -> None:
        name, label, guidance, severity = dominant_issue(
            {"a": True, "b": True, "c": False},
            {"a": 0.2, "b": 0.9, "c": 5.0},
        )
        self.assertEqual(name, "b")
        self.assertEqual(severity, 0.9)
        self.assertTrue(label)
        self.assertTrue(guidance)

    def test_returns_empty_when_no_metric_is_bad(self) -> None:
        self.assertEqual(dominant_issue({"a": False}, {"a": 1.0}), (None, "", "", 0.0))


class IssueDetailsTests(unittest.TestCase):
    def test_none_metric_returns_empty(self) -> None:
        self.assertEqual(issue_details(None), ("", ""))

    def test_known_metric_returns_copy(self) -> None:
        label, guidance = issue_details("ear_shoulder_dx")
        self.assertEqual(label, "Cabeza adelantada")
        self.assertTrue(guidance)

    def test_unknown_metric_falls_back(self) -> None:
        label, guidance = issue_details("nope")
        self.assertEqual(label, "Postura inestable")
        self.assertTrue(guidance)


if __name__ == "__main__":
    unittest.main()
