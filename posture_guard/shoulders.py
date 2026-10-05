from __future__ import annotations

from collections.abc import Iterable

from .geometry import EPSILON, angle_degrees, distance_2d, visibility_confidence, weighted_average
from .landmarks import BodyLandmarks
from .models import ShoulderState

SHOULDER_FIELDS = ("roll", "left_elevation", "right_elevation", "elevation")


def estimate_shoulder_roll(body: BodyLandmarks, min_visibility: float = 0.0) -> tuple[float | None, float]:
    """Tilt of the 2D shoulder line in degrees; positive when the right shoulder is lower."""
    image = body.image
    if any(name not in image for name in ("left_shoulder", "right_shoulder")):
        return None, 0.0
    left_shoulder = image["left_shoulder"]
    right_shoulder = image["right_shoulder"]

    confidence = visibility_confidence(left_shoulder, right_shoulder)
    if confidence < max(min_visibility, EPSILON):
        return None, 0.0
    if distance_2d(left_shoulder, right_shoulder) < EPSILON:
        return None, 0.0
    return angle_degrees(right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y), confidence


def _side_elevation(body: BodyLandmarks, side: str, min_visibility: float) -> tuple[float | None, float]:
    image = body.image
    names = (f"{side}_ear", f"{side}_shoulder", f"{side}_hip")
    if any(name not in image for name in names):
        return None, 0.0
    ear, shoulder, hip = (image[name] for name in names)

    confidence = visibility_confidence(ear, shoulder, hip)
    if confidence < max(min_visibility, EPSILON):
        return None, 0.0

    torso_length = distance_2d(shoulder, hip)
    if torso_length < EPSILON:
        return None, 0.0
    return distance_2d(ear, shoulder) / torso_length, confidence


def estimate_shoulder_state(body: BodyLandmarks, min_visibility: float = 0.0) -> ShoulderState:
    """Observe-only shoulder signals: line tilt plus ear-to-shoulder elevation ratios.

    The elevation direction is intentionally not hardcoded; calibration learns it.
    The aggregate ``elevation`` currently publishes the most reliable side.
    """
    left_elevation, left_confidence = _side_elevation(body, "left", min_visibility)
    right_elevation, right_confidence = _side_elevation(body, "right", min_visibility)
    roll, roll_confidence = estimate_shoulder_roll(body, min_visibility)

    candidates = [
        (value, confidence)
        for value, confidence in ((left_elevation, left_confidence), (right_elevation, right_confidence))
        if value is not None
    ]
    elevation, _ = max(candidates, key=lambda item: item[1]) if candidates else (None, 0.0)

    confidences = [
        confidence
        for value, confidence in (
            (roll, roll_confidence),
            (left_elevation, left_confidence),
            (right_elevation, right_confidence),
        )
        if value is not None
    ]

    return ShoulderState(
        roll=roll,
        left_elevation=left_elevation,
        right_elevation=right_elevation,
        elevation=elevation,
        confidence=sum(confidences) / len(confidences) if confidences else 0.0,
    )


def aggregate_shoulders(states: Iterable[ShoulderState | None]) -> ShoulderState | None:
    """Confidence-weighted smoothing of several shoulder states, e.g. over a window."""
    present = [state for state in states if state is not None]
    if not present:
        return None

    values: dict[str, float | None] = {}
    for field_name in SHOULDER_FIELDS:
        pairs = [(getattr(state, field_name), state.confidence) for state in present if getattr(state, field_name) is not None]
        values[field_name] = weighted_average(pairs) if pairs else None

    confidence = sum(state.confidence for state in present) / len(present)
    return ShoulderState(**values, confidence=confidence)
