from __future__ import annotations

import unittest

from posture_guard.config import Config
from posture_guard.detection import (
    RollingMetrics,
    classify_posture,
    dominant_issue,
    extract_metrics,
    issue_details,
)
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


def make_metrics_with_confidence(
    values: dict[str, float],
    confidence: dict[str, float],
    side: str = "right",
) -> DetectionMetrics:
    return DetectionMetrics(side=side, values=dict(values), confidence=dict(confidence), points={})


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
            make_profile(),
            make_metrics(ear_shoulder_dx=0.05, chin_drop=0.20, torso_lean_dx=0.30),
        )
        self.assertFalse(is_bad)
        self.assertEqual(sum(1 for is_metric_bad in bad_by_metric.values() if is_metric_bad), 1)

    def test_two_bad_metrics_flag_posture(self) -> None:
        is_bad, bad_by_metric, _ = classify_posture(
            make_profile(), make_metrics(ear_shoulder_dx=0.05, chin_drop=0.30)
        )
        self.assertTrue(is_bad)
        self.assertEqual(sum(1 for is_metric_bad in bad_by_metric.values() if is_metric_bad), 2)

    def test_single_bad_metric_flags_posture_when_min_is_one(self) -> None:
        is_bad, _, _ = classify_posture(
            make_profile(), make_metrics(ear_shoulder_dx=0.05), min_bad_metrics=1
        )
        self.assertTrue(is_bad)

    def test_min_bad_metrics_is_clamped_to_at_least_one(self) -> None:
        is_bad, _, _ = classify_posture(
            make_profile(), make_metrics(ear_shoulder_dx=0.05), min_bad_metrics=0
        )
        self.assertTrue(is_bad)

    def test_unknown_metric_is_ignored(self) -> None:
        is_bad, bad_by_metric, severity = classify_posture(
            make_profile(), make_metrics(unknown_metric=999.0)
        )
        self.assertFalse(is_bad)
        self.assertEqual(bad_by_metric, {})
        self.assertEqual(severity, {})

    def test_disabled_metric_is_skipped(self) -> None:
        profile = make_profile()
        profile.thresholds["ear_shoulder_dx"] = MetricThreshold(0.10, "disabled", None, 0.05)
        is_bad, bad_by_metric, _ = classify_posture(
            profile, make_metrics(ear_shoulder_dx=0.0, chin_drop=0.20)
        )
        self.assertNotIn("ear_shoulder_dx", bad_by_metric)
        self.assertFalse(is_bad)

    def test_all_metrics_disabled_is_never_bad(self) -> None:
        profile = make_profile()
        for name in list(profile.thresholds):
            profile.thresholds[name] = MetricThreshold(0.0, "disabled", None, 0.05)
        is_bad, bad_by_metric, severity = classify_posture(
            profile, make_metrics(ear_shoulder_dx=0.0, chin_drop=0.9, torso_lean_dx=0.9)
        )
        self.assertFalse(is_bad)
        self.assertEqual(bad_by_metric, {})
        self.assertEqual(severity, {})

    def test_min_bad_metrics_is_clamped_to_enabled_count(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={},
            good_std={},
            thresholds={"ear_shoulder_dx": MetricThreshold(0.10, "directional", -1, 0.05)},
        )
        is_bad, _, _ = classify_posture(
            profile, make_metrics(ear_shoulder_dx=0.05), min_bad_metrics=2
        )
        self.assertTrue(is_bad)

    def test_two_enabled_metrics_need_only_one_bad(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={},
            good_std={},
            thresholds={
                "a": MetricThreshold(0.10, "directional", 1, 0.05),
                "b": MetricThreshold(0.10, "directional", 1, 0.05),
            },
        )
        is_bad, _, _ = classify_posture(profile, make_metrics(a=0.20, b=0.05))
        self.assertTrue(is_bad)

    def test_severity_is_never_negative(self) -> None:
        _, _, severity = classify_posture(
            make_profile(), make_metrics(ear_shoulder_dx=0.05, chin_drop=0.30, torso_lean_dx=0.40)
        )
        self.assertTrue(all(value >= 0.0 for value in severity.values()))


class ClassifyPostureAvailabilityTests(unittest.TestCase):
    def test_absent_metric_is_not_evaluated(self) -> None:
        profile = make_profile()
        _, bad_by_metric, severity = classify_posture(profile, make_metrics(ear_shoulder_dx=0.05))
        self.assertNotIn("chin_drop", bad_by_metric)
        self.assertNotIn("torso_lean_dx", severity)

    def test_all_metrics_absent_is_not_bad(self) -> None:
        self.assertEqual(classify_posture(make_profile(), make_metrics()), (False, {}, {}))

    def test_low_confidence_metric_is_excluded(self) -> None:
        metrics = make_metrics_with_confidence(
            {"ear_shoulder_dx": 0.05, "chin_drop": 0.20},
            {"ear_shoulder_dx": 0.20, "chin_drop": 0.90},
        )
        is_bad, bad_by_metric, _ = classify_posture(
            make_profile(), metrics, min_confidence=0.60
        )
        self.assertNotIn("ear_shoulder_dx", bad_by_metric)
        self.assertFalse(is_bad)

    def test_confidence_at_threshold_is_accepted(self) -> None:
        metrics = make_metrics_with_confidence(
            {"ear_shoulder_dx": 0.05, "chin_drop": 0.20},
            {"ear_shoulder_dx": 0.60, "chin_drop": 0.90},
        )
        _, bad_by_metric, _ = classify_posture(make_profile(), metrics, min_confidence=0.60)
        self.assertIn("ear_shoulder_dx", bad_by_metric)
        self.assertTrue(bad_by_metric["ear_shoulder_dx"])

    def test_only_available_metrics_drive_classification(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={},
            good_std={},
            thresholds={
                "a": MetricThreshold(0.10, "directional", 1, 0.05),
                "b": MetricThreshold(0.10, "directional", 1, 0.05),
                "c": MetricThreshold(0.10, "directional", 1, 0.05),
                "d": MetricThreshold(0.10, "directional", 1, 0.05),
            },
        )
        metrics = make_metrics(a=0.20, b=0.05)
        is_bad, bad_by_metric, _ = classify_posture(profile, metrics, min_bad_metrics=2)
        self.assertEqual(sorted(bad_by_metric), ["a", "b"])
        self.assertTrue(is_bad)

    def test_all_low_confidence_is_not_bad(self) -> None:
        metrics = make_metrics_with_confidence(
            {"ear_shoulder_dx": 0.0, "chin_drop": 0.90},
            {"ear_shoulder_dx": 0.10, "chin_drop": 0.10},
        )
        self.assertEqual(
            classify_posture(make_profile(), metrics, min_confidence=0.60),
            (False, {}, {}),
        )


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


class _Landmark:
    def __init__(self, x: float = 0.5, y: float = 0.5, visibility: float = 0.9) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


class _Result:
    def __init__(self, landmarks: list[_Landmark]) -> None:
        self.pose_landmarks = [landmarks] if landmarks else []


def make_landmarks(visibility: float = 0.9) -> list[_Landmark]:
    return [_Landmark(visibility=visibility) for _ in range(33)]


class ExtractMetricsTests(unittest.TestCase):
    def test_returns_none_without_pose(self) -> None:
        self.assertIsNone(extract_metrics(_Result([]), Config(), preferred_side="right"))

    def test_returns_metrics_when_all_landmarks_visible(self) -> None:
        metrics = extract_metrics(_Result(make_landmarks()), Config(), preferred_side="right")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.side, "right")
        self.assertIn("torso_lean_dx", metrics.values)

    def test_hip_not_visible_drops_torso_metric_but_keeps_pose(self) -> None:
        landmarks = make_landmarks()
        landmarks[24].visibility = 0.1
        metrics = extract_metrics(_Result(landmarks), Config(), preferred_side="right")
        self.assertIsNotNone(metrics)
        self.assertNotIn("torso_lean_dx", metrics.values)
        self.assertIn("ear_shoulder_dx", metrics.values)

    def test_hip_visible_includes_torso_metric(self) -> None:
        metrics = extract_metrics(_Result(make_landmarks()), Config(), preferred_side="right")
        self.assertIn("torso_lean_dx", metrics.values)

    def test_confidence_is_reported_for_every_metric(self) -> None:
        metrics = extract_metrics(_Result(make_landmarks()), Config(), preferred_side="right")
        self.assertEqual(set(metrics.confidence), set(metrics.values))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in metrics.confidence.values()))

    def test_confidence_reflects_landmark_visibility(self) -> None:
        landmarks = make_landmarks(visibility=0.9)
        landmarks[0].visibility = 0.65  # nose
        metrics = extract_metrics(_Result(landmarks), Config(), preferred_side="right")
        self.assertAlmostEqual(metrics.confidence["chin_drop"], 0.65)
        self.assertAlmostEqual(metrics.confidence["nose_shoulder_dx"], 0.65)
        self.assertAlmostEqual(metrics.confidence["ear_shoulder_dx"], 0.9)

    def test_respects_preferred_side(self) -> None:
        metrics = extract_metrics(_Result(make_landmarks()), Config(), preferred_side="left")
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.side, "left")


class RollingMetricsTests(unittest.TestCase):
    def test_mean_tolerates_metric_missing_in_some_samples(self) -> None:
        smoother = RollingMetrics(window_size=4)
        smoother.append(DetectionMetrics(side="right", values={"a": 0.1}, points={}))
        smoother.append(DetectionMetrics(side="right", values={"a": 0.3, "b": 0.5}, points={}))
        mean = smoother.mean()
        self.assertIsNotNone(mean)
        self.assertAlmostEqual(mean.values["a"], 0.2)
        self.assertAlmostEqual(mean.values["b"], 0.5)

    def test_mean_averages_confidence_over_present_samples(self) -> None:
        smoother = RollingMetrics(window_size=4)
        smoother.append(DetectionMetrics(side="right", values={"a": 0.1}, confidence={"a": 0.8}, points={}))
        smoother.append(DetectionMetrics(side="right", values={"a": 0.3}, confidence={"a": 0.4}, points={}))
        mean = smoother.mean()
        self.assertIsNotNone(mean)
        self.assertAlmostEqual(mean.confidence["a"], 0.6)

    def test_mean_is_weighted_by_confidence(self) -> None:
        smoother = RollingMetrics(window_size=4)
        smoother.append(DetectionMetrics(side="right", values={"a": 1.0}, confidence={"a": 1.0}, points={}))
        smoother.append(DetectionMetrics(side="right", values={"a": 0.0}, confidence={"a": 0.0}, points={}))
        mean = smoother.mean()
        self.assertIsNotNone(mean)
        self.assertAlmostEqual(mean.values["a"], 1.0)

    def test_metric_below_min_observations_is_not_published(self) -> None:
        smoother = RollingMetrics(window_size=4, min_observations=2)
        smoother.append(DetectionMetrics(side="right", values={"a": 0.1, "b": 0.5}, points={}))
        self.assertNotIn("a", smoother.mean().values)
        smoother.append(DetectionMetrics(side="right", values={"a": 0.3}, points={}))
        mean = smoother.mean()
        self.assertIn("a", mean.values)
        self.assertAlmostEqual(mean.values["a"], 0.2)
        self.assertNotIn("b", mean.values)


if __name__ == "__main__":
    unittest.main()
