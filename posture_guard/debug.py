from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from .models import DetectionMetrics, ForwardState, ShoulderState, ViewState

STATE_OK = "ok"
STATE_LOW = "low"
STATE_MISSING = "missing"

# cv2's Hershey fonts cannot render unicode, so keep the symbols ASCII.
_STATE_SYMBOLS = {STATE_OK: "ok", STATE_LOW: "?", STATE_MISSING: "--"}

# Scenario labels only: the camera is fixed and which monitor you look at is
# INFERRED from the recorded orientation (head_yaw), never labeled by hand.
# Number keys 1-9,0 jump to the first ten; "n"/"p" cycle through the rest.
SNAPSHOT_LABELS: tuple[str, ...] = (
    "good",
    "front_good",
    "head_forward",
    "torso_forward",
    "head_down",
    "head_tilt_left",
    "head_tilt_right",
    "torso_lean_left",
    "torso_lean_right",
    "shoulders_up",
    "shoulder_left_up",
    "shoulder_right_up",
    "head_turn_only",
    "head_torso_turn",
)


def label_for_digit(char: str) -> str | None:
    """Map a digit key to a scenario label; "1".."9" then "0"."""
    if len(char) != 1 or char not in "1234567890":
        return None
    index = (int(char) - 1) % 10
    if index >= len(SNAPSHOT_LABELS):
        return None
    return SNAPSHOT_LABELS[index]


def next_snapshot_label(current: str | None, step: int) -> str:
    """Cycle through the scenario labels, tolerating an unknown current value."""
    if current in SNAPSHOT_LABELS:
        index = (SNAPSHOT_LABELS.index(current) + step) % len(SNAPSHOT_LABELS)
    else:
        index = 0
    return SNAPSHOT_LABELS[index]


def metric_state(value: float | None, confidence: float, min_confidence: float = 0.0) -> str:
    if value is None:
        return STATE_MISSING
    if confidence < min_confidence:
        return STATE_LOW
    return STATE_OK


def _format_line(name: str, value: float | None, confidence: float, min_confidence: float) -> str:
    symbol = _STATE_SYMBOLS[metric_state(value, confidence, min_confidence)]
    if value is None:
        return f"  {name:<9} {symbol}"
    return f"  {name:<9} {value:+7.2f} {symbol}"


def format_debug_lines(metrics: DetectionMetrics | None, min_confidence: float = 0.0) -> list[str]:
    """Observation-only overlay block for the camera window."""
    if metrics is None:
        return ["DEBUG: sin pose"]

    view = metrics.view
    shoulders = metrics.shoulders
    forward = metrics.forward
    view_confidence = view.confidence if view else 0.0
    shoulder_confidence = shoulders.confidence if shoulders else 0.0
    forward_confidence = forward.confidence if forward else 0.0

    def view_value(name: str) -> float | None:
        return getattr(view, name) if view else None

    def shoulder_value(name: str) -> float | None:
        return getattr(shoulders, name) if shoulders else None

    def forward_value(name: str) -> float | None:
        return getattr(forward, name) if forward else None

    orientation = view.orientation if view else "unknown"
    lines = [f"VIEW {orientation} conf {view_confidence:.0%}"]
    lines.append("HEAD")
    lines.append(_format_line("yaw", view_value("head_yaw"), view_confidence, min_confidence))
    lines.append(_format_line("pitch", view_value("head_pitch"), view_confidence, min_confidence))
    lines.append(_format_line("roll", view_value("head_roll"), view_confidence, min_confidence))
    lines.append("TORSO")
    lines.append(_format_line("yaw", view_value("torso_yaw"), view_confidence, min_confidence))
    lines.append(_format_line("lean", view_value("torso_lateral_lean"), view_confidence, min_confidence))
    lines.append("RELATIVE")
    lines.append(_format_line("neck roll", view_value("neck_roll_delta"), view_confidence, min_confidence))
    lines.append("SHOULDERS")
    lines.append(_format_line("roll", shoulder_value("roll"), shoulder_confidence, min_confidence))
    lines.append(_format_line("L elev", shoulder_value("left_elevation"), shoulder_confidence, min_confidence))
    lines.append(_format_line("R elev", shoulder_value("right_elevation"), shoulder_confidence, min_confidence))
    lines.append(_format_line("elev", shoulder_value("elevation"), shoulder_confidence, min_confidence))
    lines.append("FORWARD")
    lines.append(_format_line("head", forward_value("head_forward_ratio"), forward_confidence, min_confidence))
    lines.append(_format_line("torso", forward_value("torso_forward_angle"), forward_confidence, min_confidence))
    return lines


def _view_payload(view: ViewState | None) -> dict[str, Any] | None:
    if view is None:
        return None
    return {
        "orientation": view.orientation,
        "confidence": view.confidence,
        "head_yaw": view.head_yaw,
        "head_pitch": view.head_pitch,
        "head_roll": view.head_roll,
        "torso_yaw": view.torso_yaw,
        "torso_lateral_lean": view.torso_lateral_lean,
        "neck_roll_delta": view.neck_roll_delta,
    }


def _shoulders_payload(shoulders: ShoulderState | None) -> dict[str, Any] | None:
    if shoulders is None:
        return None
    return {
        "roll": shoulders.roll,
        "left_elevation": shoulders.left_elevation,
        "right_elevation": shoulders.right_elevation,
        "elevation": shoulders.elevation,
        "confidence": shoulders.confidence,
    }


def _forward_payload(forward: ForwardState | None) -> dict[str, Any] | None:
    if forward is None:
        return None
    return {
        "head_forward_ratio": forward.head_forward_ratio,
        "torso_forward_angle": forward.torso_forward_angle,
        "confidence": forward.confidence,
    }


def _observations_payload(metrics: DetectionMetrics | None) -> dict[str, Any]:
    if metrics is None:
        return {}
    return {
        name: {"value": observation.value, "confidence": observation.confidence}
        for name, observation in metrics.observations.items()
    }


def build_snapshot(
    metrics: DetectionMetrics | None,
    label: str,
    timestamp: dt.datetime | None = None,
    active_view_id: str | None = None,
    active_view_name: str | None = None,
) -> dict[str, Any]:
    """Privacy-friendly metric snapshot: no image, only geometry and label.

    The monitor/view is not part of the label: it is recorded automatically from
    the detected orientation and (when calibrated) the active view profile.
    """
    moment = timestamp or dt.datetime.now()
    return {
        "timestamp": moment.isoformat(timespec="seconds"),
        "label": label,
        "side": metrics.side if metrics else None,
        "active_view": {"id": active_view_id, "name": active_view_name},
        "view": _view_payload(metrics.view if metrics else None),
        "shoulders": _shoulders_payload(metrics.shoulders if metrics else None),
        "forward": _forward_payload(metrics.forward if metrics else None),
        "observations": _observations_payload(metrics),
        "metrics": dict(metrics.values) if metrics else {},
        "confidence": dict(metrics.confidence) if metrics else {},
    }


def append_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
