from __future__ import annotations

import unittest

from posture_guard.issues import (
    ISSUE_POLICIES,
    METRIC_TO_ISSUE,
    PostureIssue,
    issue_details,
    issue_for_metric,
    issue_label_for,
    policy_for,
)


class IssueMappingTests(unittest.TestCase):
    def test_v2_metrics_map_to_issues(self) -> None:
        self.assertEqual(issue_for_metric("head_forward_ratio"), PostureIssue.HEAD_FORWARD)
        self.assertEqual(issue_for_metric("head_yaw"), PostureIssue.NECK_ROTATION)
        self.assertEqual(issue_for_metric("head_pitch"), PostureIssue.NECK_FLEXION)
        self.assertEqual(issue_for_metric("neck_roll_delta"), PostureIssue.HEAD_TILT)
        self.assertEqual(issue_for_metric("shoulder_roll"), PostureIssue.SHOULDER_ASYMMETRY)
        self.assertEqual(issue_for_metric("shoulder_elevation"), PostureIssue.SHOULDER_ELEVATION)
        self.assertEqual(issue_for_metric("torso_forward_angle"), PostureIssue.TORSO_FORWARD)

    def test_legacy_aliases_map_to_issues(self) -> None:
        self.assertEqual(issue_for_metric("ear_shoulder_dx"), PostureIssue.HEAD_FORWARD)
        self.assertEqual(issue_for_metric("chin_drop"), PostureIssue.NECK_FLEXION)
        self.assertEqual(issue_for_metric("torso_lean_dx"), PostureIssue.TORSO_FORWARD)

    def test_unknown_metric_is_none(self) -> None:
        self.assertIsNone(issue_for_metric("nope"))
        self.assertIsNone(issue_for_metric(None))

    def test_every_mapped_issue_has_details(self) -> None:
        for issue in set(METRIC_TO_ISSUE.values()):
            label, guidance = issue_details(issue)
            self.assertTrue(label)
            self.assertTrue(guidance)

    def test_issue_details_none_and_unknown(self) -> None:
        self.assertEqual(issue_details(None), ("", ""))

    def test_issue_label_for_value(self) -> None:
        self.assertEqual(issue_label_for("neck_rotation"), "Cuello girado")
        self.assertEqual(issue_label_for("nope"), "")
        self.assertEqual(issue_label_for(None), "")


class IssuePolicyTests(unittest.TestCase):
    def test_every_issue_has_a_policy(self) -> None:
        for issue in PostureIssue:
            self.assertIn(issue, ISSUE_POLICIES)

    def test_neck_rotation_is_load_based_and_slower(self) -> None:
        policy = policy_for(PostureIssue.NECK_ROTATION)
        self.assertTrue(policy.load_based)
        self.assertGreater(policy.threshold_seconds, policy_for(PostureIssue.HEAD_FORWARD).threshold_seconds)

    def test_unknown_issue_gets_default_policy(self) -> None:
        self.assertEqual(policy_for(PostureIssue.HEAD_FORWARD).threshold_seconds, 20.0)


if __name__ == "__main__":
    unittest.main()
