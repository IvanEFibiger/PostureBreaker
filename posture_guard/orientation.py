from __future__ import annotations

import math
from collections.abc import Iterable

from .geometry import (
    EPSILON,
    Point2D,
    angle_degrees,
    angle_from_vertical,
    distance_2d,
    midpoint,
    normalize_line_angle,
    visibility_confidence,
    weighted_average,
)
from .landmarks import BodyLandmarks
from .metrics import field_confidence, mean_confidences
from .models import ViewState

# Provisional center band for the categorical orientation, in the same
# (dimensionless) units as ``head_yaw``. Real thresholds should come from the
# controlled dataset before the category drives any decision.
HEAD_YAW_CENTER_THRESHOLD = 0.20

HEAD_FIELDS = ("head_yaw", "head_pitch", "head_roll")
TORSO_FIELDS = ("torso_yaw", "torso_lateral_lean")
DERIVED_FIELDS = ("neck_roll_delta",)
VIEW_FIELDS = HEAD_FIELDS + TORSO_FIELDS + DERIVED_FIELDS


def _eye_yaw_signal(body: BodyLandmarks) -> tuple[float, float] | None:
    image = body.image
    if any(name not in image for name in ("nose", "left_eye", "right_eye")):
        return None
    nose = image["nose"]
    left_eye = image["left_eye"]
    right_eye = image["right_eye"]

    eye_mid_x = (left_eye.x + right_eye.x) / 2.0
    eye_span = abs(right_eye.x - left_eye.x)
    if eye_span < EPSILON:
        return None
    value = (nose.x - eye_mid_x) / max(eye_span, EPSILON)
    return value, visibility_confidence(nose, left_eye, right_eye)


def _ear_yaw_signal(body: BodyLandmarks) -> tuple[float, float] | None:
    image = body.image
    if any(name not in image for name in ("nose", "left_ear", "right_ear")):
        return None
    nose = image["nose"]
    left_ear = image["left_ear"]
    right_ear = image["right_ear"]

    left_distance = abs(nose.x - left_ear.x)
    right_distance = abs(right_ear.x - nose.x)
    span = left_distance + right_distance
    if span < EPSILON:
        return None
    value = (left_distance - right_distance) / max(span, EPSILON)
    return value, visibility_confidence(nose, left_ear, right_ear)


def estimate_head_yaw(body: BodyLandmarks) -> tuple[float | None, float]:
    """Signed, calibrated head yaw from eye and ear signals.

    Positive means the nose is displaced towards +x. The value is a geometric
    signal, not anatomical degrees; calibration provides the personal baseline.
    """
    signals = [signal for signal in (_eye_yaw_signal(body), _ear_yaw_signal(body)) if signal]
    if not signals:
        return None, 0.0
    value = weighted_average(signals)
    if value is None:
        return None, 0.0
    confidence = sum(weight for _, weight in signals) / len(signals)
    return value, confidence


def _line_roll(left: Point2D, right: Point2D) -> float:
    return normalize_line_angle(angle_degrees(right.x - left.x, right.y - left.y))


def estimate_head_roll(body: BodyLandmarks) -> tuple[float | None, float]:
    """Head roll in degrees, fusing the eye line and the mouth line.

    Two roughly parallel face lines make the estimate sturdier than eyes alone;
    positive when the right side is lower.
    """
    image = body.image
    signals: list[tuple[float, float]] = []
    if "left_eye" in image and "right_eye" in image:
        left_eye = image["left_eye"]
        right_eye = image["right_eye"]
        if distance_2d(left_eye, right_eye) >= EPSILON:
            signals.append((_line_roll(left_eye, right_eye), visibility_confidence(left_eye, right_eye)))
    if "left_mouth" in image and "right_mouth" in image:
        left_mouth = image["left_mouth"]
        right_mouth = image["right_mouth"]
        if distance_2d(left_mouth, right_mouth) >= EPSILON:
            signals.append((_line_roll(left_mouth, right_mouth), visibility_confidence(left_mouth, right_mouth)))

    if not signals:
        return None, 0.0
    value = weighted_average(signals)
    if value is None:
        return None, 0.0
    return normalize_line_angle(value), sum(weight for _, weight in signals) / len(signals)


def estimate_head_pitch(body: BodyLandmarks) -> tuple[float | None, float]:
    """Nose drop below the eye line, scaled by the face height when available.

    Using the mouth line as the scale makes it distance-invariant; falls back to
    the eye span when the mouth is not visible. Positive when the nose drops.
    """
    image = body.image
    if any(name not in image for name in ("nose", "left_eye", "right_eye")):
        return None, 0.0
    nose = image["nose"]
    left_eye = image["left_eye"]
    right_eye = image["right_eye"]

    eye_mid_y = (left_eye.y + right_eye.y) / 2.0
    scale = 0.0
    scale_points: list[Point2D] = []
    if "left_mouth" in image and "right_mouth" in image:
        mouth_mid_y = (image["left_mouth"].y + image["right_mouth"].y) / 2.0
        scale = mouth_mid_y - eye_mid_y
        scale_points = [image["left_mouth"], image["right_mouth"]]
    if scale < EPSILON:
        scale = distance_2d(left_eye, right_eye)
        scale_points = []
    if scale < EPSILON:
        return None, 0.0

    confidence = visibility_confidence(nose, left_eye, right_eye, *scale_points)
    return (nose.y - eye_mid_y) / scale, confidence


def classify_orientation(head_yaw: float | None) -> str:
    if head_yaw is None:
        return "unknown"
    if head_yaw > HEAD_YAW_CENTER_THRESHOLD:
        return "right"
    if head_yaw < -HEAD_YAW_CENTER_THRESHOLD:
        return "left"
    return "center"


def estimate_torso_yaw(body: BodyLandmarks, min_visibility: float = 0.0) -> tuple[float | None, float]:
    """Torso yaw in degrees from the 3D shoulder line (world landmarks only).

    Positive means the right shoulder is deeper (larger z). There is deliberately
    no 2D fallback: an absent metric is better than a fabricated one.
    """
    world = body.world
    if any(name not in world for name in ("left_shoulder", "right_shoulder")):
        return None, 0.0
    left_shoulder = world["left_shoulder"]
    right_shoulder = world["right_shoulder"]

    confidence = visibility_confidence(left_shoulder, right_shoulder)
    if confidence < max(min_visibility, EPSILON):
        return None, 0.0

    dx = right_shoulder.x - left_shoulder.x
    dz = right_shoulder.z - left_shoulder.z
    if math.hypot(dx, dz) < EPSILON:
        return None, 0.0
    return angle_degrees(dx, dz), confidence


def estimate_torso_lateral_lean(body: BodyLandmarks, min_visibility: float = 0.0) -> tuple[float | None, float]:
    """Lateral lean of the shoulder-hip axis, in degrees from vertical.

    Positive means the shoulders are shifted towards +x relative to the hips.
    """
    image = body.image
    needed = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(name not in image for name in needed):
        return None, 0.0
    left_shoulder = image["left_shoulder"]
    right_shoulder = image["right_shoulder"]
    left_hip = image["left_hip"]
    right_hip = image["right_hip"]

    confidence = visibility_confidence(left_shoulder, right_shoulder, left_hip, right_hip)
    if confidence < max(min_visibility, EPSILON):
        return None, 0.0

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    dx = shoulder_mid.x - hip_mid.x
    dy = hip_mid.y - shoulder_mid.y
    if math.hypot(dx, dy) < EPSILON:
        return None, 0.0
    return angle_from_vertical(dx, dy), confidence


def estimate_view_state(body: BodyLandmarks, min_visibility: float = 0.0) -> ViewState:
    head_yaw, yaw_confidence = estimate_head_yaw(body)
    head_pitch, pitch_confidence = estimate_head_pitch(body)
    head_roll, roll_confidence = estimate_head_roll(body)
    torso_yaw, torso_yaw_confidence = estimate_torso_yaw(body, min_visibility)
    torso_lean, torso_lean_confidence = estimate_torso_lateral_lean(body, min_visibility)

    # neck_roll_delta is derived cross-state (head vs shoulders) in
    # metrics.attach_neck_roll_delta; it is not a torso-only signal anymore.
    signals = (
        ("head_yaw", head_yaw, yaw_confidence),
        ("head_pitch", head_pitch, pitch_confidence),
        ("head_roll", head_roll, roll_confidence),
        ("torso_yaw", torso_yaw, torso_yaw_confidence),
        ("torso_lateral_lean", torso_lean, torso_lean_confidence),
    )
    confidences = {name: confidence for name, value, confidence in signals if value is not None}
    present = list(confidences.values())

    return ViewState(
        head_yaw=head_yaw,
        head_pitch=head_pitch,
        head_roll=head_roll,
        torso_yaw=torso_yaw,
        torso_lateral_lean=torso_lean,
        neck_roll_delta=None,
        orientation=classify_orientation(head_yaw),
        confidence=sum(present) / len(present) if present else 0.0,
        confidences=confidences,
    )


def aggregate_view(views: Iterable[ViewState | None]) -> ViewState | None:
    """Confidence-weighted smoothing of several view states, e.g. over a window."""
    states = [view for view in views if view is not None]
    if not states:
        return None

    values: dict[str, float | None] = {}
    for field_name in VIEW_FIELDS:
        pairs = [
            (getattr(state, field_name), field_confidence(state, field_name))
            for state in states
            if getattr(state, field_name) is not None
        ]
        values[field_name] = weighted_average(pairs) if pairs else None

    confidence = sum(state.confidence for state in states) / len(states)
    return ViewState(
        **values,
        orientation=classify_orientation(values["head_yaw"]),
        confidence=confidence,
        confidences=mean_confidences(states, VIEW_FIELDS),
    )
