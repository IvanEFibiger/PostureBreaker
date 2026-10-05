from __future__ import annotations

import unittest

from posture_guard.forward import (
    aggregate_forward,
    estimate_forward_state,
    estimate_head_forward_ratio,
    estimate_torso_forward_angle,
)
from posture_guard.geometry import Point2D
from posture_guard.landmarks import BodyLandmarks
from posture_guard.models import ForwardState


def p(x: float, y: float, visibility: float = 0.9) -> Point2D:
    return Point2D(x=x, y=y, visibility=visibility)


def body(points: dict[str, Point2D]) -> BodyLandmarks:
    return BodyLandmarks(image=dict(points), world={})


def right_side(visibility: float = 0.9) -> dict[str, Point2D]:
    return {
        "right_ear": p(0.45, 0.3, visibility),
        "right_shoulder": p(0.4, 0.5, visibility),
        "right_hip": p(0.4, 0.8, visibility),
    }


class HeadForwardRatioTests(unittest.TestCase):
    def test_lateral_view_publishes_ratio(self) -> None:
        value, confidence = estimate_head_forward_ratio(body(right_side()), "right", "right")
        self.assertAlmostEqual(value, 0.05 / 0.3)
        self.assertAlmostEqual(confidence, 0.9)

    def test_frontal_view_is_unavailable(self) -> None:
        self.assertEqual(estimate_head_forward_ratio(body(right_side()), "right", "center"), (None, 0.0))

    def test_unknown_view_is_unavailable(self) -> None:
        self.assertEqual(estimate_head_forward_ratio(body(right_side()), "right", "unknown"), (None, 0.0))

    def test_missing_hip_is_none(self) -> None:
        image = right_side()
        del image["right_hip"]
        self.assertEqual(estimate_head_forward_ratio(body(image), "right", "right"), (None, 0.0))

    def test_degenerate_torso_is_none(self) -> None:
        image = right_side()
        image["right_hip"] = p(0.4, 0.5)
        self.assertEqual(estimate_head_forward_ratio(body(image), "right", "right"), (None, 0.0))

    def test_low_visibility_is_none(self) -> None:
        image = right_side(visibility=0.2)
        self.assertEqual(
            estimate_head_forward_ratio(body(image), "right", "right", min_visibility=0.5),
            (None, 0.0),
        )


class TorsoForwardAngleTests(unittest.TestCase):
    def test_vertical_torso_is_zero(self) -> None:
        value, confidence = estimate_torso_forward_angle(body(right_side()), "right")
        self.assertAlmostEqual(value, 0.0)
        self.assertAlmostEqual(confidence, 0.9)

    def test_shoulder_right_of_hip_is_positive(self) -> None:
        image = right_side()
        image["right_shoulder"] = p(0.5, 0.5)
        value, _ = estimate_torso_forward_angle(body(image), "right")
        self.assertGreater(value, 0.0)

    def test_shoulder_left_of_hip_is_negative(self) -> None:
        image = right_side()
        image["right_shoulder"] = p(0.3, 0.5)
        value, _ = estimate_torso_forward_angle(body(image), "right")
        self.assertLess(value, 0.0)

    def test_missing_shoulder_is_none(self) -> None:
        image = right_side()
        del image["right_shoulder"]
        self.assertEqual(estimate_torso_forward_angle(body(image), "right"), (None, 0.0))

    def test_degenerate_axis_is_none(self) -> None:
        image = {"right_shoulder": p(0.4, 0.8), "right_hip": p(0.4, 0.8)}
        self.assertEqual(estimate_torso_forward_angle(body(image), "right"), (None, 0.0))

    def test_low_visibility_is_none(self) -> None:
        image = right_side(visibility=0.2)
        self.assertEqual(
            estimate_torso_forward_angle(body(image), "right", min_visibility=0.5),
            (None, 0.0),
        )


class EstimateForwardStateTests(unittest.TestCase):
    def test_lateral_view_publishes_both(self) -> None:
        state = estimate_forward_state(body(right_side()), "right", "right")
        self.assertAlmostEqual(state.head_forward_ratio, 0.05 / 0.3)
        self.assertAlmostEqual(state.torso_forward_angle, 0.0)
        self.assertAlmostEqual(state.confidence, 0.9)

    def test_frontal_view_publishes_only_torso(self) -> None:
        state = estimate_forward_state(body(right_side()), "right", "center")
        self.assertIsNone(state.head_forward_ratio)
        self.assertAlmostEqual(state.torso_forward_angle, 0.0)
        self.assertAlmostEqual(state.confidence, 0.9)

    def test_missing_landmarks_leave_both_none(self) -> None:
        state = estimate_forward_state(body({}), "right", "right")
        self.assertIsNone(state.head_forward_ratio)
        self.assertIsNone(state.torso_forward_angle)
        self.assertEqual(state.confidence, 0.0)


class AggregateForwardTests(unittest.TestCase):
    def test_returns_none_without_states(self) -> None:
        self.assertIsNone(aggregate_forward([None, None]))

    def test_averages_fields_and_preserves_none(self) -> None:
        first = ForwardState(head_forward_ratio=0.2, torso_forward_angle=4.0, confidence=1.0)
        second = ForwardState(head_forward_ratio=0.4, confidence=1.0)
        merged = aggregate_forward([first, second])
        self.assertAlmostEqual(merged.head_forward_ratio, 0.3)
        self.assertAlmostEqual(merged.torso_forward_angle, 4.0)
        self.assertEqual(merged.confidence, 1.0)

    def test_none_never_becomes_zero(self) -> None:
        merged = aggregate_forward([ForwardState(head_forward_ratio=None, confidence=0.9)])
        self.assertIsNone(merged.head_forward_ratio)


if __name__ == "__main__":
    unittest.main()
