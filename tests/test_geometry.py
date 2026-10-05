from __future__ import annotations

import unittest

from posture_guard.geometry import (
    Point2D,
    Point3D,
    angle_degrees,
    distance_2d,
    distance_3d,
    midpoint,
    normalize_line_angle,
    visibility_confidence,
    weighted_average,
)


class DistanceTests(unittest.TestCase):
    def test_distance_2d_uses_pythagoras(self) -> None:
        self.assertAlmostEqual(distance_2d(Point2D(0.0, 0.0), Point2D(3.0, 4.0)), 5.0)

    def test_distance_2d_same_point_is_zero(self) -> None:
        self.assertEqual(distance_2d(Point2D(0.2, 0.4), Point2D(0.2, 0.4)), 0.0)

    def test_distance_3d_uses_three_axes(self) -> None:
        self.assertAlmostEqual(distance_3d(Point3D(0.0, 0.0, 0.0), Point3D(0.0, 3.0, 4.0)), 5.0)


class AngleTests(unittest.TestCase):
    def test_angle_right_is_zero(self) -> None:
        self.assertAlmostEqual(angle_degrees(1.0, 0.0), 0.0)

    def test_angle_up_is_ninety(self) -> None:
        self.assertAlmostEqual(angle_degrees(0.0, 1.0), 90.0)

    def test_angle_left_is_one_eighty(self) -> None:
        self.assertAlmostEqual(abs(angle_degrees(-1.0, 0.0)), 180.0)

    def test_angle_down_is_minus_ninety(self) -> None:
        self.assertAlmostEqual(angle_degrees(0.0, -1.0), -90.0)


class NormalizeLineAngleTests(unittest.TestCase):
    def test_folds_into_half_range(self) -> None:
        self.assertAlmostEqual(normalize_line_angle(178.0), -2.0)
        self.assertAlmostEqual(normalize_line_angle(-177.0), 3.0)
        self.assertAlmostEqual(normalize_line_angle(90.0), 90.0)
        self.assertAlmostEqual(normalize_line_angle(0.0), 0.0)
        self.assertAlmostEqual(normalize_line_angle(45.0), 45.0)
        self.assertAlmostEqual(normalize_line_angle(-45.0), -45.0)


class MidpointTests(unittest.TestCase):
    def test_midpoint_is_centered(self) -> None:
        center = midpoint(Point2D(0.0, 0.0), Point2D(4.0, 2.0))
        self.assertAlmostEqual(center.x, 2.0)
        self.assertAlmostEqual(center.y, 1.0)

    def test_midpoint_takes_lowest_visibility(self) -> None:
        center = midpoint(Point2D(0.0, 0.0, 0.9), Point2D(2.0, 2.0, 0.4))
        self.assertAlmostEqual(center.visibility, 0.4)


class VisibilityConfidenceTests(unittest.TestCase):
    def test_no_points_is_zero(self) -> None:
        self.assertEqual(visibility_confidence(), 0.0)

    def test_returns_lowest_visibility(self) -> None:
        self.assertAlmostEqual(
            visibility_confidence(Point2D(0.0, 0.0, 0.9), Point2D(1.0, 1.0, 0.3)),
            0.3,
        )

    def test_objects_without_visibility_default_to_one(self) -> None:
        class Bare:
            pass

        self.assertEqual(visibility_confidence(Bare()), 1.0)


class WeightedAverageTests(unittest.TestCase):
    def test_weights_shift_the_mean(self) -> None:
        self.assertAlmostEqual(weighted_average([(1.0, 1.0), (3.0, 3.0)]), 2.5)

    def test_equal_weights_are_a_plain_mean(self) -> None:
        self.assertAlmostEqual(weighted_average([(1.0, 1.0), (3.0, 1.0)]), 2.0)

    def test_empty_is_none(self) -> None:
        self.assertIsNone(weighted_average([]))

    def test_zero_weight_is_none(self) -> None:
        self.assertIsNone(weighted_average([(1.0, 0.0), (3.0, 0.0)]))


if __name__ == "__main__":
    unittest.main()
