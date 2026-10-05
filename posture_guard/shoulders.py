from __future__ import annotations

from collections.abc import Iterable

from .geometry import (
    EPSILON,
    angle_degrees,
    distance_2d,
    normalize_line_angle,
    visibility_confidence,
    weighted_average,
)
from .landmarks import BodyLandmarks
from .metrics import field_confidence, mean_confidences
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
    value = normalize_line_angle(angle_degrees(right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y))
    return value, confidence


def _shoulder_width(image: dict[str, object]) -> float | None:
    if "left_shoulder" not in image or "right_shoulder" not in image:
        return None
    width = distance_2d(image["left_shoulder"], image["right_shoulder"])  # type: ignore[arg-type]
    return width if width >= EPSILON else None


def _side_elevation(body: BodyLandmarks, side: str, min_visibility: float) -> tuple[float | None, float]:
    """Ear-to-shoulder distance normalized by shoulder width (no hip needed)."""
    image = body.image
    names = (f"{side}_ear", f"{side}_shoulder")
    if any(name not in image for name in names):
        return None, 0.0
    width = _shoulder_width(image)
    if width is None:
        return None, 0.0

    ear, shoulder = (image[name] for name in names)
    left_shoulder = image["left_shoulder"]
    right_shoulder = image["right_shoulder"]
    confidence = visibility_confidence(ear, shoulder, left_shoulder, right_shoulder)
    if confidence < max(min_visibility, EPSILON):
        return None, 0.0
    return distance_2d(ear, shoulder) / width, confidence


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
    elevation, elevation_confidence = max(candidates, key=lambda item: item[1]) if candidates else (None, 0.0)

    confidences: dict[str, float] = {}
    if roll is not None:
        confidences["roll"] = roll_confidence
    if left_elevation is not None:
        confidences["left_elevation"] = left_confidence
    if right_elevation is not None:
        confidences["right_elevation"] = right_confidence
    if elevation is not None:
        confidences["elevation"] = elevation_confidence

    present = [
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
        confidence=sum(present) / len(present) if present else 0.0,
        confidences=confidences,
    )


def aggregate_shoulders(states: Iterable[ShoulderState | None]) -> ShoulderState | None:
    """Confidence-weighted smoothing of several shoulder states, e.g. over a window."""
    present = [state for state in states if state is not None]
    if not present:
        return None

    values: dict[str, float | None] = {}
    for field_name in SHOULDER_FIELDS:
        pairs = [
            (getattr(state, field_name), field_confidence(state, field_name))
            for state in present
            if getattr(state, field_name) is not None
        ]
        values[field_name] = weighted_average(pairs) if pairs else None

    confidence = sum(state.confidence for state in present) / len(present)
    return ShoulderState(**values, confidence=confidence, confidences=mean_confidences(present, SHOULDER_FIELDS))
