from __future__ import annotations

import unittest

from posture_guard.geometry import Point2D, Point3D
from posture_guard.landmarks import BodyLandmarks
from posture_guard.models import ViewState
from posture_guard.orientation import (
    HEAD_YAW_CENTER_THRESHOLD,
    aggregate_view,
    classify_orientation,
    estimate_head_pitch,
    estimate_head_roll,
    estimate_head_yaw,
    estimate_neck_roll_delta,
    estimate_torso_lateral_lean,
    estimate_torso_yaw,
    estimate_view_state,
)


def p(x: float, y: float, visibility: float = 0.9) -> Point2D:
    return Point2D(x=x, y=y, visibility=visibility)


def p3(x: float, y: float, z: float, visibility: float = 0.9) -> Point3D:
    return Point3D(x=x, y=y, z=z, visibility=visibility)


def body(points: dict[str, Point2D], world: dict[str, Point3D] | None = None) -> BodyLandmarks:
    return BodyLandmarks(image=dict(points), world=dict(world) if world else {})


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

    def test_builds_torso_signals_when_landmarks_present(self) -> None:
        image = face()
        image.update(
            {
                "left_shoulder": p(0.4, 0.6),
                "right_shoulder": p(0.6, 0.6),
                "left_hip": p(0.4, 0.9),
                "right_hip": p(0.6, 0.9),
            }
        )
        world = {"left_shoulder": p3(0.4, 0.0, 0.0), "right_shoulder": p3(0.6, 0.0, 0.0)}
        state = estimate_view_state(body(image, world=world))
        self.assertAlmostEqual(state.torso_yaw, 0.0)
        self.assertAlmostEqual(state.torso_lateral_lean, 0.0)
        self.assertAlmostEqual(state.neck_roll_delta, state.head_roll)

    def test_min_visibility_drops_torso_signals(self) -> None:
        image = face()
        image.update(
            {
                "left_shoulder": p(0.4, 0.6, visibility=0.2),
                "right_shoulder": p(0.6, 0.6),
                "left_hip": p(0.4, 0.9),
                "right_hip": p(0.6, 0.9),
            }
        )
        state = estimate_view_state(body(image), min_visibility=0.5)
        self.assertIsNone(state.torso_lateral_lean)
        self.assertIsNone(state.neck_roll_delta)

    def test_per_metric_confidence_and_neck_roll_minimum(self) -> None:
        image = face()
        image.update(
            {
                "left_shoulder": p(0.4, 0.6),
                "right_shoulder": p(0.6, 0.6),
                "left_hip": p(0.4, 0.9, visibility=0.4),
                "right_hip": p(0.6, 0.9, visibility=0.4),
            }
        )
        state = estimate_view_state(body(image))
        self.assertAlmostEqual(state.confidences["head_roll"], 0.9)
        self.assertAlmostEqual(state.confidences["neck_roll_delta"], 0.4)


class TorsoYawTests(unittest.TestCase):
    @staticmethod
    def shoulders(left_z: float = 0.0, right_z: float = 0.0, visibility: float = 0.9) -> dict[str, Point3D]:
        return {
            "left_shoulder": p3(0.4, 0.5, left_z, visibility),
            "right_shoulder": p3(0.6, 0.5, right_z, visibility),
        }

    def test_aligned_shoulders_are_zero(self) -> None:
        value, confidence = estimate_torso_yaw(body({}, world=self.shoulders()))
        self.assertAlmostEqual(value, 0.0)
        self.assertAlmostEqual(confidence, 0.9)

    def test_right_shoulder_deeper_is_positive(self) -> None:
        value, _ = estimate_torso_yaw(body({}, world=self.shoulders(right_z=0.2)))
        self.assertAlmostEqual(value, 45.0)

    def test_left_shoulder_deeper_is_negative(self) -> None:
        value, _ = estimate_torso_yaw(body({}, world=self.shoulders(left_z=0.2)))
        self.assertAlmostEqual(value, -45.0)

    def test_missing_world_landmarks_is_none(self) -> None:
        self.assertEqual(estimate_torso_yaw(body({})), (None, 0.0))

    def test_one_shoulder_missing_is_none(self) -> None:
        world = self.shoulders()
        del world["right_shoulder"]
        self.assertEqual(estimate_torso_yaw(body({}, world=world)), (None, 0.0))

    def test_low_visibility_is_none(self) -> None:
        world = self.shoulders(visibility=0.2)
        self.assertEqual(estimate_torso_yaw(body({}, world=world), min_visibility=0.5), (None, 0.0))

    def test_degenerate_shoulder_line_is_none(self) -> None:
        world = {"left_shoulder": p3(0.5, 0.5, 0.0), "right_shoulder": p3(0.5, 0.5, 0.0)}
        self.assertEqual(estimate_torso_yaw(body({}, world=world)), (None, 0.0))


class TorsoLateralLeanTests(unittest.TestCase):
    @staticmethod
    def torso(shoulder_mid_x: float = 0.5, hip_mid_x: float = 0.5) -> dict[str, Point2D]:
        return {
            "left_shoulder": p(shoulder_mid_x - 0.1, 0.4),
            "right_shoulder": p(shoulder_mid_x + 0.1, 0.4),
            "left_hip": p(hip_mid_x - 0.1, 0.8),
            "right_hip": p(hip_mid_x + 0.1, 0.8),
        }

    def test_vertical_torso_is_zero(self) -> None:
        value, confidence = estimate_torso_lateral_lean(body(self.torso()))
        self.assertAlmostEqual(value, 0.0)
        self.assertAlmostEqual(confidence, 0.9)

    def test_shoulders_right_of_hips_is_positive(self) -> None:
        value, _ = estimate_torso_lateral_lean(body(self.torso(shoulder_mid_x=0.6, hip_mid_x=0.5)))
        self.assertGreater(value, 0.0)

    def test_shoulders_left_of_hips_is_negative(self) -> None:
        value, _ = estimate_torso_lateral_lean(body(self.torso(shoulder_mid_x=0.4, hip_mid_x=0.5)))
        self.assertLess(value, 0.0)

    def test_one_hip_missing_is_none(self) -> None:
        image = self.torso()
        del image["left_hip"]
        self.assertEqual(estimate_torso_lateral_lean(body(image)), (None, 0.0))

    def test_one_shoulder_missing_is_none(self) -> None:
        image = self.torso()
        del image["right_shoulder"]
        self.assertEqual(estimate_torso_lateral_lean(body(image)), (None, 0.0))

    def test_degenerate_axis_is_none(self) -> None:
        image = {
            "left_shoulder": p(0.4, 0.5),
            "right_shoulder": p(0.6, 0.5),
            "left_hip": p(0.4, 0.5),
            "right_hip": p(0.6, 0.5),
        }
        self.assertEqual(estimate_torso_lateral_lean(body(image)), (None, 0.0))


class NeckRollDeltaTests(unittest.TestCase):
    def test_subtracts_torso_from_head(self) -> None:
        value, _ = estimate_neck_roll_delta(10.0, 0.9, 9.0, 0.9)
        self.assertAlmostEqual(value, 1.0)

    def test_head_only_tilt_keeps_angle(self) -> None:
        value, _ = estimate_neck_roll_delta(10.0, 0.9, 0.0, 0.9)
        self.assertAlmostEqual(value, 10.0)

    def test_confidence_is_the_minimum_of_components(self) -> None:
        _, confidence = estimate_neck_roll_delta(10.0, 0.8, 9.0, 0.4)
        self.assertAlmostEqual(confidence, 0.4)

    def test_missing_head_roll_is_none(self) -> None:
        self.assertEqual(estimate_neck_roll_delta(None, 0.9, 9.0, 0.9), (None, 0.0))

    def test_missing_torso_lean_is_none(self) -> None:
        self.assertEqual(estimate_neck_roll_delta(10.0, 0.9, None, 0.9), (None, 0.0))


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

    def test_smooths_torso_signals_and_preserves_none(self) -> None:
        first = ViewState(torso_yaw=None, torso_lateral_lean=5.0, neck_roll_delta=1.0, confidence=1.0)
        second = ViewState(torso_yaw=None, torso_lateral_lean=7.0, neck_roll_delta=3.0, confidence=1.0)
        merged = aggregate_view([first, second])
        self.assertIsNone(merged.torso_yaw)
        self.assertAlmostEqual(merged.torso_lateral_lean, 6.0)
        self.assertAlmostEqual(merged.neck_roll_delta, 2.0)

    def test_none_never_becomes_zero(self) -> None:
        state = ViewState(torso_lateral_lean=None, neck_roll_delta=None, confidence=0.9)
        merged = aggregate_view([state])
        self.assertIsNone(merged.torso_lateral_lean)
        self.assertIsNone(merged.neck_roll_delta)

    def test_weighted_by_per_field_confidence(self) -> None:
        first = ViewState(head_roll=1.0, confidence=0.1, confidences={"head_roll": 1.0})
        second = ViewState(head_roll=2.0, confidence=1.0, confidences={"head_roll": 0.0})
        merged = aggregate_view([first, second])
        self.assertAlmostEqual(merged.head_roll, 1.0)


if __name__ == "__main__":
    unittest.main()
