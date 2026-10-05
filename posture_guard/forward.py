from __future__ import annotations

import math
from collections.abc import Iterable

from .geometry import EPSILON, angle_from_vertical, distance_2d, visibility_confidence, weighted_average
from .landmarks import BodyLandmarks
from .models import ForwardState

FORWARD_FIELDS = ("head_forward_ratio", "torso_forward_angle")

# Head-forward is only meaningful when the camera sees the head from the side.
# A frontal (center) view hides the ear-shoulder offset, so the metric is absent.
LATERAL_ORIENTATIONS = ("left", "right")


def estimate_head_forward_ratio(
    body: BodyLandmarks,
    side: str,
    orientation: str,
    min_visibility: float = 0.0,
) -> tuple[float | None, float]:
    """Signed ear-shoulder horizontal offset normalized by torso length.

    Only published for lateral views; calibration learns the sign per side.
    """
    if orientation not in LATERAL_ORIENTATIONS:
        return None, 0.0

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
    return (ear.x - shoulder.x) / torso_length, confidence


def estimate_torso_forward_angle(
    body: BodyLandmarks,
    side: str,
    min_visibility: float = 0.0,
) -> tuple[float | None, float]:
    """Shoulder-hip axis angle from vertical, in degrees; sign is learned by calibration."""
    image = body.image
    names = (f"{side}_shoulder", f"{side}_hip")
    if any(name not in image for name in names):
        return None, 0.0
    shoulder, hip = (image[name] for name in names)

    confidence = visibility_confidence(shoulder, hip)
    if confidence < max(min_visibility, EPSILON):
        return None, 0.0

    dx = shoulder.x - hip.x
    dy = hip.y - shoulder.y
    if math.hypot(dx, dy) < EPSILON:
        return None, 0.0
    return angle_from_vertical(dx, dy), confidence


def estimate_forward_state(
    body: BodyLandmarks,
    side: str,
    orientation: str,
    min_visibility: float = 0.0,
) -> ForwardState:
    head_forward, head_confidence = estimate_head_forward_ratio(body, side, orientation, min_visibility)
    torso_forward, torso_confidence = estimate_torso_forward_angle(body, side, min_visibility)

    confidences = [
        confidence
        for value, confidence in ((head_forward, head_confidence), (torso_forward, torso_confidence))
        if value is not None
    ]
    return ForwardState(
        head_forward_ratio=head_forward,
        torso_forward_angle=torso_forward,
        confidence=sum(confidences) / len(confidences) if confidences else 0.0,
    )


def aggregate_forward(states: Iterable[ForwardState | None]) -> ForwardState | None:
    """Confidence-weighted smoothing of several forward states, e.g. over a window."""
    present = [state for state in states if state is not None]
    if not present:
        return None

    values: dict[str, float | None] = {}
    for field_name in FORWARD_FIELDS:
        pairs = [(getattr(state, field_name), state.confidence) for state in present if getattr(state, field_name) is not None]
        values[field_name] = weighted_average(pairs) if pairs else None

    confidence = sum(state.confidence for state in present) / len(present)
    return ForwardState(**values, confidence=confidence)
