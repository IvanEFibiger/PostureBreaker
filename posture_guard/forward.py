from __future__ import annotations

import math
from collections.abc import Iterable

from .geometry import EPSILON, angle_from_vertical, distance_2d, visibility_confidence, weighted_average
from .landmarks import BodyLandmarks
from .metrics import field_confidence, mean_confidences
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
    """Canonical signed ear-shoulder offset normalized by shoulder width.

    Hip-free so it works with a desk webcam that only sees head and shoulders.
    The sign is normalized by side so ``HIGHER_IS_WORSE`` stays stable; only
    published for lateral views. The exact inversion still needs camera validation.
    """
    if orientation not in LATERAL_ORIENTATIONS:
        return None, 0.0

    image = body.image
    names = (f"{side}_ear", f"{side}_shoulder")
    if any(name not in image for name in names):
        return None, 0.0
    if "left_shoulder" not in image or "right_shoulder" not in image:
        return None, 0.0
    ear, shoulder = (image[name] for name in names)
    left_shoulder = image["left_shoulder"]
    right_shoulder = image["right_shoulder"]

    width = distance_2d(left_shoulder, right_shoulder)
    if width < EPSILON:
        return None, 0.0
    confidence = visibility_confidence(ear, shoulder, left_shoulder, right_shoulder)
    if confidence < max(min_visibility, EPSILON):
        return None, 0.0

    side_sign = 1.0 if side == "right" else -1.0
    return ((ear.x - shoulder.x) / width) * side_sign, confidence


def estimate_torso_forward_angle(
    body: BodyLandmarks,
    side: str,
    min_visibility: float = 0.0,
) -> tuple[float | None, float]:
    """Canonical shoulder-hip axis angle from vertical, in degrees.

    The sign is normalized by side; the exact inversion still needs camera validation.
    """
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
    side_sign = 1.0 if side == "right" else -1.0
    return angle_from_vertical(dx, dy) * side_sign, confidence


def estimate_forward_state(
    body: BodyLandmarks,
    side: str,
    orientation: str,
    min_visibility: float = 0.0,
) -> ForwardState:
    head_forward, head_confidence = estimate_head_forward_ratio(body, side, orientation, min_visibility)
    torso_forward, torso_confidence = estimate_torso_forward_angle(body, side, min_visibility)

    confidences: dict[str, float] = {}
    if head_forward is not None:
        confidences["head_forward_ratio"] = head_confidence
    if torso_forward is not None:
        confidences["torso_forward_angle"] = torso_confidence

    present = list(confidences.values())
    return ForwardState(
        head_forward_ratio=head_forward,
        torso_forward_angle=torso_forward,
        confidence=sum(present) / len(present) if present else 0.0,
        confidences=confidences,
    )


def aggregate_forward(states: Iterable[ForwardState | None]) -> ForwardState | None:
    """Confidence-weighted smoothing of several forward states, e.g. over a window."""
    present = [state for state in states if state is not None]
    if not present:
        return None

    values: dict[str, float | None] = {}
    for field_name in FORWARD_FIELDS:
        pairs = [
            (getattr(state, field_name), field_confidence(state, field_name))
            for state in present
            if getattr(state, field_name) is not None
        ]
        values[field_name] = weighted_average(pairs) if pairs else None

    confidence = sum(state.confidence for state in present) / len(present)
    return ForwardState(**values, confidence=confidence, confidences=mean_confidences(present, FORWARD_FIELDS))
