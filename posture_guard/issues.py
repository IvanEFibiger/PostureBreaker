from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PostureIssue(StrEnum):
    HEAD_FORWARD = "head_forward"
    NECK_ROTATION = "neck_rotation"
    NECK_FLEXION = "neck_flexion"
    HEAD_TILT = "head_tilt"
    SHOULDER_ELEVATION = "shoulder_elevation"
    SHOULDER_ASYMMETRY = "shoulder_asymmetry"
    TORSO_FORWARD = "torso_forward"


# Technical metric -> user-visible ergonomic problem. Legacy V1 metric names are
# kept as aliases so both generations map to the same issue.
METRIC_TO_ISSUE: dict[str, PostureIssue] = {
    "head_forward_ratio": PostureIssue.HEAD_FORWARD,
    "ear_shoulder_dx": PostureIssue.HEAD_FORWARD,
    "head_yaw": PostureIssue.NECK_ROTATION,
    "torso_yaw": PostureIssue.NECK_ROTATION,
    "neck_yaw_delta": PostureIssue.NECK_ROTATION,
    "head_pitch": PostureIssue.NECK_FLEXION,
    "chin_drop": PostureIssue.NECK_FLEXION,
    "nose_shoulder_dx": PostureIssue.NECK_FLEXION,
    "head_roll": PostureIssue.HEAD_TILT,
    "neck_roll_delta": PostureIssue.HEAD_TILT,
    "shoulder_roll": PostureIssue.SHOULDER_ASYMMETRY,
    "shoulder_elevation": PostureIssue.SHOULDER_ELEVATION,
    "left_shoulder_elevation": PostureIssue.SHOULDER_ELEVATION,
    "right_shoulder_elevation": PostureIssue.SHOULDER_ELEVATION,
    "torso_forward_angle": PostureIssue.TORSO_FORWARD,
    "torso_lean_dx": PostureIssue.TORSO_FORWARD,
}

ISSUE_DETAILS: dict[PostureIssue, tuple[str, str]] = {
    PostureIssue.HEAD_FORWARD: (
        "Cabeza adelantada",
        "Lleva la cabeza atras y alineala con los hombros.",
    ),
    PostureIssue.NECK_ROTATION: (
        "Cuello girado",
        "Gira el torso hacia el monitor en lugar del cuello.",
    ),
    PostureIssue.NECK_FLEXION: (
        "Mirada muy baja",
        "Subi la mirada y destraba el menton.",
    ),
    PostureIssue.HEAD_TILT: (
        "Cabeza inclinada",
        "Nivela la cabeza sobre los hombros.",
    ),
    PostureIssue.SHOULDER_ELEVATION: (
        "Hombros elevados",
        "Baja los hombros y relaja el trapecio.",
    ),
    PostureIssue.SHOULDER_ASYMMETRY: (
        "Hombros desnivelados",
        "Apoya ambos hombros parejo.",
    ),
    PostureIssue.TORSO_FORWARD: (
        "Torso inclinado",
        "Volve el torso al eje y apoya la espalda.",
    ),
}


@dataclass(frozen=True)
class IssuePolicy:
    threshold_seconds: float
    cooldown_seconds: float = 300.0
    weight: float = 1.0
    load_based: bool = False
    load_threshold: float = 0.0


# Initial policies (plan §69). Final values must come from real validation.
ISSUE_POLICIES: dict[PostureIssue, IssuePolicy] = {
    PostureIssue.HEAD_FORWARD: IssuePolicy(threshold_seconds=20.0, weight=1.5),
    PostureIssue.NECK_ROTATION: IssuePolicy(
        threshold_seconds=120.0, weight=1.2, load_based=True, load_threshold=10.0
    ),
    PostureIssue.NECK_FLEXION: IssuePolicy(threshold_seconds=30.0, weight=1.0),
    PostureIssue.HEAD_TILT: IssuePolicy(threshold_seconds=30.0, weight=0.8),
    PostureIssue.SHOULDER_ELEVATION: IssuePolicy(threshold_seconds=30.0, weight=1.1),
    PostureIssue.SHOULDER_ASYMMETRY: IssuePolicy(threshold_seconds=30.0, weight=0.7),
    PostureIssue.TORSO_FORWARD: IssuePolicy(threshold_seconds=20.0, weight=1.2),
}


def issue_for_metric(metric_name: str | None) -> PostureIssue | None:
    if not metric_name:
        return None
    return METRIC_TO_ISSUE.get(metric_name)


def issue_details(issue: PostureIssue | None) -> tuple[str, str]:
    if issue is None:
        return "", ""
    payload = ISSUE_DETAILS.get(issue)
    if not payload:
        return "Postura inestable", "Reacomoda cabeza, hombros y espalda."
    return payload


def issue_label_for(value: str | None) -> str:
    """Human label for an issue value string, or "" when it is unknown."""
    if not value:
        return ""
    try:
        issue = PostureIssue(value)
    except ValueError:
        return ""
    label, _ = issue_details(issue)
    return label


def policy_for(issue: PostureIssue) -> IssuePolicy:
    return ISSUE_POLICIES.get(issue, IssuePolicy(threshold_seconds=30.0))
