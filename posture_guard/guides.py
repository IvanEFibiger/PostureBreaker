from __future__ import annotations

from .models import DebugLandmark

# 2D landmarks worth drawing for the desk battery. The renderer decides which
# ones to show based on their confidence; the detector only collects them.
DEBUG_LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_mouth",
    "right_mouth",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
)

# Visual groups: one consistent color per group so the desktop framing is easy
# to read at a glance.
FACE_LANDMARKS = frozenset(
    {"nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_mouth", "right_mouth"}
)
SHOULDER_LANDMARKS = frozenset({"left_shoulder", "right_shoulder"})
OPTIONAL_LANDMARKS = frozenset({"left_elbow", "right_elbow", "left_hip", "right_hip"})

# Colors are BGR (OpenCV order).
FACE_COLOR = (255, 0, 255)
SHOULDER_COLOR = (0, 255, 0)
OPTIONAL_COLOR = (255, 200, 0)
EAR_SHOULDER_COLOR = (0, 255, 255)

# (a, b, color, thickness); drawn back-to-front so shoulders and face lines win.
GUIDE_SEGMENTS: tuple[tuple[str, str, tuple[int, int, int], int], ...] = (
    ("left_shoulder", "left_hip", OPTIONAL_COLOR, 2),
    ("right_shoulder", "right_hip", OPTIONAL_COLOR, 2),
    ("left_shoulder", "left_elbow", OPTIONAL_COLOR, 1),
    ("right_shoulder", "right_elbow", OPTIONAL_COLOR, 1),
    ("left_ear", "left_shoulder", EAR_SHOULDER_COLOR, 1),
    ("right_ear", "right_shoulder", EAR_SHOULDER_COLOR, 1),
    ("left_eye", "right_eye", FACE_COLOR, 1),
    ("left_mouth", "right_mouth", FACE_COLOR, 1),
    ("left_shoulder", "right_shoulder", SHOULDER_COLOR, 2),
)


# Landmarks whose anatomical side is worth labelling on the overlay so a wrong
# mirroring is visible immediately.
LATERAL_LANDMARKS = frozenset({"left_shoulder", "right_shoulder", "left_ear", "right_ear"})


def laterality_label(name: str) -> str:
    """Anatomical L/R label for a landmark name (from the model, not the screen)."""
    if name.startswith("left"):
        return "L"
    if name.startswith("right"):
        return "R"
    return ""


def display_x(raw_x: float, mirror: bool) -> float:
    """Map a model x (raw frame) to the display x when the UI is mirrored.

    MediaPipe works on the raw frame; the on-screen preview is flipped for the
    user, so only rendering flips x. y is untouched.
    """
    return 1.0 - raw_x if mirror else raw_x


def display_point(raw_x: float, raw_y: float, mirror: bool) -> tuple[float, float]:
    """Display coordinates for a raw model point; only x may flip."""
    return display_x(raw_x, mirror), raw_y


def landmark_color(name: str) -> tuple[int, int, int]:
    if name in FACE_LANDMARKS:
        return FACE_COLOR
    if name in SHOULDER_LANDMARKS:
        return SHOULDER_COLOR
    return OPTIONAL_COLOR


def _is_inside_frame(landmark: DebugLandmark) -> bool:
    return 0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0


def visible_landmarks(
    landmarks: dict[str, DebugLandmark],
    min_visibility: float = 0.0,
) -> dict[str, DebugLandmark]:
    """Keep only the landmarks reliable enough to draw.

    A landmark must clear the confidence gate AND sit inside the frame. Hips
    below the desk come back with low confidence (or off-frame) and are dropped
    here, so a shoulder->hip line is never drawn from an inferred position.
    """
    return {
        name: landmark
        for name, landmark in landmarks.items()
        if landmark.confidence >= min_visibility and _is_inside_frame(landmark)
    }


def planned_segments(
    visible_names: set[str] | frozenset[str],
) -> list[tuple[str, str, tuple[int, int, int], int]]:
    """Connections to draw given the set of visible landmark names.

    A segment is only planned when BOTH endpoints are visible, so a missing eye
    or shoulder never triggers its line.
    """
    return [(a, b, color, thickness) for a, b, color, thickness in GUIDE_SEGMENTS if a in visible_names and b in visible_names]
