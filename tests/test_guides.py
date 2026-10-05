from __future__ import annotations

import unittest

from posture_guard.guides import (
    DEBUG_LANDMARK_NAMES,
    FACE_COLOR,
    GUIDE_SEGMENTS,
    OPTIONAL_COLOR,
    SHOULDER_COLOR,
    landmark_color,
    planned_segments,
    visible_landmarks,
)
from posture_guard.models import DebugLandmark


def mark(x: float = 0.5, y: float = 0.5, confidence: float = 0.9) -> DebugLandmark:
    return DebugLandmark(x=x, y=y, confidence=confidence)


def full_set(**overrides: float) -> dict[str, DebugLandmark]:
    landmarks = {name: mark() for name in DEBUG_LANDMARK_NAMES}
    for name, confidence in overrides.items():
        landmarks[name] = mark(confidence=confidence)
    return landmarks


def segment_pairs(segments: list[tuple[str, str, tuple[int, int, int], int]]) -> set[tuple[str, str]]:
    return {(a, b) for a, b, _color, _thickness in segments}


class VisibleLandmarksTests(unittest.TestCase):
    def test_drops_landmarks_below_the_gate(self) -> None:
        landmarks = {"left_eye": mark(confidence=0.4), "nose": mark(confidence=0.9)}
        visible = visible_landmarks(landmarks, min_visibility=0.5)
        self.assertIn("nose", visible)
        self.assertNotIn("left_eye", visible)

    def test_keeps_landmarks_at_the_threshold(self) -> None:
        visible = visible_landmarks({"nose": mark(confidence=0.5)}, min_visibility=0.5)
        self.assertIn("nose", visible)

    def test_low_visibility_hip_is_dropped(self) -> None:
        visible = visible_landmarks(full_set(left_hip=0.2), min_visibility=0.55)
        self.assertNotIn("left_hip", visible)


class PlannedSegmentsTests(unittest.TestCase):
    def test_full_set_plans_every_segment(self) -> None:
        planned = planned_segments(set(full_set()))
        self.assertEqual(segment_pairs(planned), segment_pairs(list(GUIDE_SEGMENTS)))

    def test_missing_eye_drops_the_eye_line(self) -> None:
        names = set(full_set()) - {"right_eye"}
        self.assertNotIn(("left_eye", "right_eye"), segment_pairs(planned_segments(names)))

    def test_missing_shoulder_drops_its_lines(self) -> None:
        names = set(full_set()) - {"left_shoulder"}
        pairs = segment_pairs(planned_segments(names))
        self.assertNotIn(("left_shoulder", "right_shoulder"), pairs)
        self.assertNotIn(("left_shoulder", "left_hip"), pairs)
        self.assertNotIn(("left_shoulder", "left_elbow"), pairs)
        self.assertNotIn(("left_ear", "left_shoulder"), pairs)

    def test_hip_line_needs_a_visible_hip(self) -> None:
        visible = visible_landmarks(full_set(left_hip=0.2), min_visibility=0.55)
        pairs = segment_pairs(planned_segments(set(visible)))
        self.assertNotIn(("left_shoulder", "left_hip"), pairs)
        self.assertIn(("right_shoulder", "right_hip"), pairs)

    def test_hip_line_drawn_when_hips_are_visible(self) -> None:
        visible = visible_landmarks(full_set(), min_visibility=0.55)
        pairs = segment_pairs(planned_segments(set(visible)))
        self.assertIn(("left_shoulder", "left_hip"), pairs)
        self.assertIn(("right_shoulder", "right_hip"), pairs)


class OutOfFrameTests(unittest.TestCase):
    def test_hip_outside_horizontal_frame_is_dropped(self) -> None:
        landmarks = full_set()
        landmarks["left_hip"] = DebugLandmark(1.2, 0.5, 0.9)
        self.assertNotIn("left_hip", visible_landmarks(landmarks, min_visibility=0.55))

    def test_hip_below_the_frame_is_dropped(self) -> None:
        landmarks = full_set()
        landmarks["left_hip"] = DebugLandmark(0.5, 1.4, 0.9)
        self.assertNotIn("left_hip", visible_landmarks(landmarks, min_visibility=0.55))

    def test_reliable_in_frame_hip_is_kept(self) -> None:
        self.assertIn("left_hip", visible_landmarks(full_set(), min_visibility=0.55))

    def test_no_hip_line_when_hip_is_out_of_frame(self) -> None:
        landmarks = full_set()
        landmarks["left_hip"] = DebugLandmark(0.5, 1.4, 0.9)
        visible = visible_landmarks(landmarks, min_visibility=0.55)
        pairs = segment_pairs(planned_segments(set(visible)))
        self.assertNotIn(("left_shoulder", "left_hip"), pairs)
        self.assertIn(("right_shoulder", "right_hip"), pairs)

    def test_no_hip_point_when_confidence_and_frame_both_fail(self) -> None:
        landmarks = full_set()
        landmarks["right_hip"] = DebugLandmark(2.0, 2.0, 0.1)
        self.assertNotIn("right_hip", visible_landmarks(landmarks, min_visibility=0.55))


class LandmarkColorTests(unittest.TestCase):
    def test_face_shoulders_and_optional_have_distinct_colors(self) -> None:
        self.assertEqual(landmark_color("nose"), FACE_COLOR)
        self.assertEqual(landmark_color("left_shoulder"), SHOULDER_COLOR)
        self.assertEqual(landmark_color("left_hip"), OPTIONAL_COLOR)
        self.assertEqual(len({FACE_COLOR, SHOULDER_COLOR, OPTIONAL_COLOR}), 3)

    def test_debug_landmark_names_are_unique(self) -> None:
        self.assertEqual(len(DEBUG_LANDMARK_NAMES), len(set(DEBUG_LANDMARK_NAMES)))


if __name__ == "__main__":
    unittest.main()
