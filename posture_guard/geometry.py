from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

EPSILON = 1e-6


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    visibility: float = 1.0


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float
    visibility: float = 1.0


def distance_2d(a: Point2D, b: Point2D) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def distance_3d(a: Point3D, b: Point3D) -> float:
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)


def angle_degrees(dx: float, dy: float) -> float:
    """Angle of the vector (dx, dy) from the +x axis, in degrees."""
    return math.degrees(math.atan2(dy, dx))


def angle_from_vertical(dx: float, dy: float) -> float:
    """Angle of the vector (dx, dy) from the +y (vertical) axis, in degrees."""
    return math.degrees(math.atan2(dx, dy))


def midpoint(a: Point2D, b: Point2D) -> Point2D:
    return Point2D(
        x=(a.x + b.x) / 2.0,
        y=(a.y + b.y) / 2.0,
        visibility=min(a.visibility, b.visibility),
    )


def visibility_confidence(*points: object) -> float:
    """Lowest visibility among the given points, in [0, 1]."""
    visibilities = [float(getattr(point, "visibility", 1.0)) for point in points]
    if not visibilities:
        return 0.0
    return min(visibilities)


def weighted_average(pairs: Iterable[tuple[float, float]]) -> float | None:
    """Confidence-weighted mean of ``(value, weight)`` pairs.

    Returns ``None`` when there is no positive weight to divide by.
    """
    items = list(pairs)
    total_weight = sum(weight for _, weight in items)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in items) / total_weight
