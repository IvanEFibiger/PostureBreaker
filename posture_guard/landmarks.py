from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .geometry import Point2D, Point3D

# MediaPipe Pose landmark indices mapped to the names used across the app.
LANDMARK_INDICES: dict[int, str] = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "left_mouth",
    10: "right_mouth",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}


@dataclass
class BodyLandmarks:
    image: dict[str, Point2D]
    world: dict[str, Point3D]


def _row(result: Any, attribute: str) -> list[Any]:
    rows = getattr(result, attribute, None)
    if not rows:
        return []
    return rows[0]


def extract_body_landmarks(result: Any) -> BodyLandmarks | None:
    """Uniform 2D + 3D landmark view over a MediaPipe Pose result.

    Returns ``None`` when there is no pose. World landmarks are optional: they
    simply come back empty when the model does not provide them.
    """
    image_row = _row(result, "pose_landmarks")
    if not image_row:
        return None
    world_row = _row(result, "pose_world_landmarks")

    image = {
        name: Point2D(lm.x, lm.y, float(getattr(lm, "visibility", 1.0)))
        for index, name in LANDMARK_INDICES.items()
        if index < len(image_row)
        for lm in [image_row[index]]
    }
    world = {
        name: Point3D(lm.x, lm.y, lm.z, float(getattr(lm, "visibility", 1.0)))
        for index, name in LANDMARK_INDICES.items()
        if index < len(world_row)
        for lm in [world_row[index]]
    }
    return BodyLandmarks(image=image, world=world)
