from __future__ import annotations

import unittest

from posture_guard.geometry import Point2D
from posture_guard.landmarks import BodyLandmarks
from posture_guard.models import ViewState
from posture_guard.orientation import (
    HEAD_YAW_CENTER_THRESHOLD,
    aggregate_view,
    classify_orientation,
    estimate_head_pitch,
    estimate_head_roll,
    estimate_head_yaw,
    estimate_view_state,
)


def p(x: float, y: float, visibility: float = 0.9) -> Point2D:
    return Point2D(x=x, y=y, visibility=visibility)


def body(points: dict[str, Point2D]) -> BodyLandmarks:
    return BodyLandmarks(image=dict(points), world={})


def face(nose_x: float = 0.5, left_eye_y: float = 0.5, right_eye_y: float = 0.5) -> dict[str, Point2D]:
    return {
        "nose": p(nose_x, 0.55),
        "left_eye": p(0.45, left_eye_y),
        "right_eye": p(0.55, right_eye_y),
        "left_ear": p(0.30, 0.5),
        "right_ear": p(0.70, 0.5),
    }


class HeadYawTests(unittest.TestCase):
    def test_frontal_face_is_near_zero(self) -> None:
        value, confidence = estimate_head_yaw(body(face()))
        self.assertAlmostEqual(value, 0.0)
        self.assertAlmostEqual(confidence, 0.9)

    def test_eye_signal_signs_with_nose_displacement(self) -> None:
        value, _ = estimate_head_yaw(body(face(nose_x=0.55)))
        self.assertGreater(value, 0.0)
        value, _ = estimate_head_yaw(body(face(nose_x=0.45)))
        self.assertLess(value, 0.0)

    def test_fuses_eye_and_ear_signals(self) -> None:
        value, confidence = estimate_head_yaw(body(face(nose_x=0.55)))
        self.assertAlmostEqual(value, 0.375)
        self.assertAlmostEqual(confidence, 0.9)

    def test_falls_back_to_eye_signal_when_ear_missing(self) -> None:
        landmarks = face(nose_x=0.55)
        del landmarks["right_ear"]
        value, _ = estimate_head_yaw(body(landmarks))
        self.assertAlmostEqual(value, 0.5)

    def test_falls_back_to_ear_signal_when_eyes_missing(self) -> None:
        landmarks = face(nose_x=0.55)
        del landmarks["left_eye"]
        del landmarks["right_eye"]
        value, _ = estimate_head_yaw(body(landmarks))
        self.assertAlmostEqual(value, 0.25)

    def test_low_visibility_lowers_confidence(self) -> None:
        landmarks = face()
        del landmarks["left_ear"]
        del landmarks["right_ear"]
        landmarks["left_eye"] = p(0.45, 0.5, visibility=0.2)
        _, confidence = estimate_head_yaw(body(landmarks))
        self.assertAlmostEqual(confidence, 0.2)

    def test_insufficient_information_returns_none(self) -> None:
        self.assertEqual(estimate_head_yaw(body({"nose": p(0.5, 0.5)})), (None, 0.0))

    def test_degenerate_eye_span_is_ignored_but_ears_still_work(self) -> None:
        landmarks = face(nose_x=0.55)
        landmarks["left_eye"] = p(0.5, 0.5)
        landmarks["right_eye"] = p(0.5, 0.5)
        value, _ = estimate_head_yaw(body(landmarks))
        self.assertAlmostEqual(value, 0.25)


class HeadRollTests(unittest.TestCase):
    def test_level_eyes_are_zero(self) -> None:
        value, confidence = estimate_head_roll(body(face()))
        self.assertAlmostEqual(value, 0.0)
        self.assertAlmostEqual(confidence, 0.9)

    def test_right_eye_lower_is_positive(self) -> None:
        value, _ = estimate_head_roll(body(face(right_eye_y=0.6)))
        self.assertAlmostEqual(value, 45.0)

    def test_left_eye_lower_is_negative(self) -> None:
        value, _ = estimate_head_roll(body(face(left_eye_y=0.6)))
        self.assertAlmostEqual(value, -45.0)

    def test_tiny_eye_distance_returns_none(self) -> None:
        landmarks = {"left_eye": p(0.5, 0.5), "right_eye": p(0.5, 0.5)}
        self.assertEqual(estimate_head_roll(body(landmarks)), (None, 0.0))


class HeadPitchTests(unittest.TestCase):
    def test_nose_below_eye_line_is_positive(self) -> None:
        value, _ = estimate_head_pitch(body(face()))
        self.assertAlmostEqual(value, 0.5)

    def test_nose_above_eye_line_is_negative(self) -> None:
        landmarks = face()
        landmarks["nose"] = p(0.5, 0.45)
        value, _ = estimate_head_pitch(body(landmarks))
        self.assertAlmostEqual(value, -0.5)

    def test_missing_eye_returns_none(self) -> None:
        landmarks = face()
        del landmarks["right_eye"]
        self.assertEqual(estimate_head_pitch(body(landmarks)), (None, 0.0))


class ClassifyOrientationTests(unittest.TestCase):
    def test_center_left_right_and_unknown(self) -> None:
        self.assertEqual(classify_orientation(0.0), "center")
        self.assertEqual(classify_orientation(HEAD_YAW_CENTER_THRESHOLD + 0.01), "right")
        self.assertEqual(classify_orientation(-HEAD_YAW_CENTER_THRESHOLD - 0.01), "left")
        self.assertEqual(classify_orientation(None), "unknown")


class EstimateViewStateTests(unittest.TestCase):
    def test_builds_head_view_with_orientation(self) -> None:
        state = estimate_view_state(body(face(nose_x=0.55)))
        self.assertIsNotNone(state)
        self.assertGreater(state.head_yaw, HEAD_YAW_CENTER_THRESHOLD)
        self.assertEqual(state.orientation, "right")
        self.assertGreater(state.confidence, 0.0)
        self.assertIsNone(state.torso_yaw)


class AggregateViewTests(unittest.TestCase):
    def test_returns_none_without_views(self) -> None:
        self.assertIsNone(aggregate_view([None, None]))

    def test_averages_present_fields_by_confidence(self) -> None:
        first = ViewState(head_yaw=0.2, head_pitch=0.5, confidence=1.0)
        second = ViewState(head_yaw=0.4, confidence=1.0)
        merged = aggregate_view([first, second])
        self.assertAlmostEqual(merged.head_yaw, 0.3)
        self.assertAlmostEqual(merged.head_pitch, 0.5)
        self.assertEqual(merged.orientation, "right")
        self.assertAlmostEqual(merged.confidence, 1.0)

    def test_ignores_none_fields(self) -> None:
        state = ViewState(head_yaw=None, confidence=0.8)
        merged = aggregate_view([state])
        self.assertIsNone(merged.head_yaw)
        self.assertEqual(merged.orientation, "unknown")


if __name__ == "__main__":
    unittest.main()
