from __future__ import annotations

import unittest

from posture_guard.landmarks import LANDMARK_INDICES, extract_body_landmarks


class _Landmark:
    def __init__(self, x: float = 0.5, y: float = 0.5, z: float = 0.0, visibility: float = 0.9) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class _Result:
    def __init__(self, image: list[_Landmark], world: list[_Landmark] | None = None) -> None:
        self.pose_landmarks = [image] if image else []
        if world is not None:
            self.pose_world_landmarks = [world]


def make_row(count: int = 33) -> list[_Landmark]:
    return [_Landmark(x=index / 100.0, visibility=0.9) for index in range(count)]


class ExtractBodyLandmarksTests(unittest.TestCase):
    def test_returns_none_without_pose(self) -> None:
        self.assertIsNone(extract_body_landmarks(_Result([])))

    def test_extracts_named_image_landmarks(self) -> None:
        body = extract_body_landmarks(_Result(make_row()))
        self.assertIsNotNone(body)
        self.assertEqual(set(body.image), set(LANDMARK_INDICES.values()))
        self.assertAlmostEqual(body.image["nose"].x, 0.0)
        self.assertAlmostEqual(body.image["left_hip"].visibility, 0.9)

    def test_missing_world_landmarks_yields_empty_world(self) -> None:
        body = extract_body_landmarks(_Result(make_row()))
        self.assertEqual(body.world, {})

    def test_extracts_world_landmarks_when_present(self) -> None:
        body = extract_body_landmarks(_Result(make_row(), world=make_row()))
        self.assertEqual(set(body.world), set(LANDMARK_INDICES.values()))
        self.assertAlmostEqual(body.world["right_hip"].z, 0.0)

    def test_partial_row_only_exposes_present_indices(self) -> None:
        body = extract_body_landmarks(_Result(make_row(count=9)))
        self.assertIn("right_ear", body.image)
        self.assertNotIn("left_shoulder", body.image)


if __name__ == "__main__":
    unittest.main()
