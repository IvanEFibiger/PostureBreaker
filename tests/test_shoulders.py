from __future__ import annotations

import unittest

from posture_guard.geometry import Point2D
from posture_guard.landmarks import BodyLandmarks
from posture_guard.models import ShoulderState
from posture_guard.shoulders import (
    aggregate_shoulders,
    estimate_shoulder_roll,
    estimate_shoulder_state,
)


def p(x: float, y: float, visibility: float = 0.9) -> Point2D:
    return Point2D(x=x, y=y, visibility=visibility)


def body(points: dict[str, Point2D]) -> BodyLandmarks:
    return BodyLandmarks(image=dict(points), world={})


def both_sides(visibility: float = 0.9) -> dict[str, Point2D]:
    return {
        "left_ear": p(0.4, 0.3, visibility),
        "left_shoulder": p(0.4, 0.5, visibility),
        "left_hip": p(0.4, 0.8, visibility),
        "right_ear": p(0.6, 0.3, visibility),
        "right_shoulder": p(0.6, 0.5, visibility),
        "right_hip": p(0.6, 0.8, visibility),
    }


class ShoulderRollTests(unittest.TestCase):
    def test_level_shoulders_are_zero(self) -> None:
        value, confidence = estimate_shoulder_roll(body(both_sides()))
        self.assertAlmostEqual(value, 0.0)
        self.assertAlmostEqual(confidence, 0.9)

    def test_line_near_180_folds_to_zero(self) -> None:
        image = {"left_shoulder": p(0.6, 0.5), "right_shoulder": p(0.4, 0.5)}
        value, _ = estimate_shoulder_roll(body(image))
        self.assertAlmostEqual(value, 0.0)

    def test_right_shoulder_lower_is_positive(self) -> None:
        image = both_sides()
        image["right_shoulder"] = p(0.6, 0.6)
        value, _ = estimate_shoulder_roll(body(image))
        self.assertGreater(value, 0.0)

    def test_left_shoulder_lower_is_negative(self) -> None:
        image = both_sides()
        image["left_shoulder"] = p(0.4, 0.6)
        value, _ = estimate_shoulder_roll(body(image))
        self.assertLess(value, 0.0)

    def test_degenerate_shoulder_line_is_none(self) -> None:
        image = {"left_shoulder": p(0.5, 0.5), "right_shoulder": p(0.5, 0.5)}
        self.assertEqual(estimate_shoulder_roll(body(image)), (None, 0.0))

    def test_missing_shoulder_is_none(self) -> None:
        image = both_sides()
        del image["left_shoulder"]
        self.assertEqual(estimate_shoulder_roll(body(image)), (None, 0.0))

    def test_low_visibility_is_none(self) -> None:
        image = both_sides(visibility=0.2)
        self.assertEqual(estimate_shoulder_roll(body(image), min_visibility=0.5), (None, 0.0))


class ShoulderElevationTests(unittest.TestCase):
    def test_both_sides_publish_ratio(self) -> None:
        state = estimate_shoulder_state(body(both_sides()))
        self.assertAlmostEqual(state.left_elevation, 1.0)
        self.assertAlmostEqual(state.right_elevation, 1.0)
        self.assertAlmostEqual(state.elevation, 1.0)
        self.assertAlmostEqual(state.confidence, 0.9)

    def test_aggregate_picks_the_most_reliable_side(self) -> None:
        image = both_sides()
        image["right_ear"] = p(0.6, 0.2)  # right ratio 0.3 / 0.2 = 1.5
        for name in ("right_ear", "right_shoulder", "right_hip"):
            image[name] = Point2D(image[name].x, image[name].y, visibility=0.4)
        state = estimate_shoulder_state(body(image))
        self.assertAlmostEqual(state.left_elevation, 1.0)
        self.assertAlmostEqual(state.right_elevation, 1.5)
        self.assertAlmostEqual(state.elevation, 1.0)

    def test_missing_shoulder_drops_both_sides(self) -> None:
        image = both_sides()
        del image["left_shoulder"]
        state = estimate_shoulder_state(body(image))
        self.assertIsNone(state.left_elevation)
        self.assertIsNone(state.right_elevation)

    def test_all_sides_missing_leaves_elevation_none(self) -> None:
        image = both_sides()
        for name in ("left_ear", "left_shoulder", "left_hip", "right_ear", "right_shoulder", "right_hip"):
            del image[name]
        state = estimate_shoulder_state(body(image))
        self.assertIsNone(state.elevation)
        self.assertEqual(state.confidence, 0.0)

    def test_degenerate_shoulder_width_is_none(self) -> None:
        image = both_sides()
        image["left_shoulder"] = p(0.4, 0.5)
        image["right_shoulder"] = p(0.4, 0.5)
        state = estimate_shoulder_state(body(image))
        self.assertIsNone(state.left_elevation)
        self.assertIsNone(state.right_elevation)

    def test_low_visibility_drops_sides(self) -> None:
        image = both_sides(visibility=0.2)
        state = estimate_shoulder_state(body(image), min_visibility=0.5)
        self.assertIsNone(state.left_elevation)
        self.assertIsNone(state.right_elevation)
        self.assertIsNone(state.roll)


class AggregateShouldersTests(unittest.TestCase):
    def test_returns_none_without_states(self) -> None:
        self.assertIsNone(aggregate_shoulders([None, None]))

    def test_averages_fields_and_preserves_none(self) -> None:
        first = ShoulderState(roll=2.0, left_elevation=0.3, elevation=0.3, confidence=1.0)
        second = ShoulderState(roll=4.0, left_elevation=0.5, elevation=0.5, confidence=1.0)
        merged = aggregate_shoulders([first, second])
        self.assertAlmostEqual(merged.roll, 3.0)
        self.assertAlmostEqual(merged.left_elevation, 0.4)
        self.assertIsNone(merged.right_elevation)
        self.assertAlmostEqual(merged.confidence, 1.0)

    def test_none_never_becomes_zero(self) -> None:
        merged = aggregate_shoulders([ShoulderState(roll=None, elevation=None, confidence=0.9)])
        self.assertIsNone(merged.roll)
        self.assertIsNone(merged.elevation)


if __name__ == "__main__":
    unittest.main()
