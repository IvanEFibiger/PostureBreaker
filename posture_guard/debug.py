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

# Number-key labels for the camera window, grouped by validation scenario.
SNAPSHOT_LABELS: dict[str, str] = {
    "1": "monitor_1_good",
    "2": "monitor_2_good",
    "3": "frontal",
    "4": "head_forward",
    "5": "looking_down",
    "6": "neck_rotated",
    "7": "neck_and_torso_rotated",
    "8": "shoulders_raised",
    "9": "head_tilted",
    "0": "torso_tilted",
}


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
) -> dict[str, Any]:
    """Privacy-friendly metric snapshot: no image, only geometry and label."""
    moment = timestamp or dt.datetime.now()
    return {
        "timestamp": moment.isoformat(timespec="seconds"),
        "label": label,
        "side": metrics.side if metrics else None,
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
