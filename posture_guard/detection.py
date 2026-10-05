from __future__ import annotations

import statistics
from collections import deque
from typing import Any

from .config import Config
from .models import CalibrationProfile, DetectionMetrics

# MediaPipe Pose landmark indices.
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

ERROR_DETAILS = {
    "ear_shoulder_dx": {
        "label": "Cabeza adelantada",
        "guidance": "Lleva la cabeza atras y alineala con los hombros.",
    },
    "nose_shoulder_dx": {
        "label": "Cuello empujado",
        "guidance": "Aleja la cara de la pantalla y afloja la nuca.",
    },
    "chin_drop": {
        "label": "Mirada muy baja",
        "guidance": "Subi la mirada y destraba el menton.",
    },
    "torso_lean_dx": {
        "label": "Torso inclinado",
        "guidance": "Volve el torso al eje y apoya la espalda.",
    },
}


def issue_details(metric_name: str | None) -> tuple[str, str]:
    if not metric_name:
        return "", ""
    payload = ERROR_DETAILS.get(metric_name)
    if not payload:
        return "Postura inestable", "Reacomoda cabeza, hombros y espalda."
    return str(payload["label"]), str(payload["guidance"])


def dominant_issue(
    bad_by_metric: dict[str, bool],
    severity: dict[str, float],
) -> tuple[str | None, str, str, float]:
    candidates = [name for name, is_bad in bad_by_metric.items() if is_bad]
    if not candidates:
        return None, "", "", 0.0
    metric_name = max(candidates, key=lambda name: severity.get(name, 0.0))
    label, guidance = issue_details(metric_name)
    return metric_name, label, guidance, severity.get(metric_name, 0.0)


def choose_side(landmarks: list[Any], min_visibility: float) -> str | None:
    sides = {
        "left": (LEFT_EAR, LEFT_SHOULDER),
        "right": (RIGHT_EAR, RIGHT_SHOULDER),
    }
    scored: list[tuple[str, float]] = []
    for side, (ear_idx, shoulder_idx) in sides.items():
        ear = landmarks[ear_idx]
        shoulder = landmarks[shoulder_idx]
        score = (ear.visibility + shoulder.visibility) / 2.0
        if ear.visibility >= min_visibility and shoulder.visibility >= min_visibility:
            scored.append((side, score))

    if not scored:
        return None
    return max(scored, key=lambda item: item[1])[0]


def _visibility_confidence(*landmarks: Any) -> float:
    """Lowest landmark visibility involved in a metric, in [0, 1]."""
    return min(float(lm.visibility) for lm in landmarks)


def extract_metrics(result: Any, config: Config, preferred_side: str | None = None) -> DetectionMetrics | None:
    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]
    side = preferred_side or choose_side(landmarks, config.min_visibility)
    if not side:
        return None
    if side not in {"left", "right"}:
        return None

    if side == "left":
        ear_idx, shoulder_idx, hip_idx = LEFT_EAR, LEFT_SHOULDER, LEFT_HIP
    else:
        ear_idx, shoulder_idx, hip_idx = RIGHT_EAR, RIGHT_SHOULDER, RIGHT_HIP

    ear = landmarks[ear_idx]
    shoulder = landmarks[shoulder_idx]
    hip = landmarks[hip_idx]
    nose = landmarks[NOSE]

    required = [ear, shoulder, nose]
    if any(lm.visibility < config.min_visibility for lm in required):
        return None

    values = {
        "ear_shoulder_dx": ear.x - shoulder.x,
        "nose_shoulder_dx": nose.x - shoulder.x,
        "chin_drop": nose.y - ear.y,
    }
    confidence = {
        "ear_shoulder_dx": _visibility_confidence(ear, shoulder),
        "nose_shoulder_dx": _visibility_confidence(nose, shoulder),
        "chin_drop": _visibility_confidence(nose, ear),
    }
    if hip.visibility >= config.min_visibility:
        values["torso_lean_dx"] = shoulder.x - hip.x
        confidence["torso_lean_dx"] = _visibility_confidence(shoulder, hip)

    points = {
        "ear": (ear.x, ear.y),
        "shoulder": (shoulder.x, shoulder.y),
        "hip": (hip.x, hip.y),
        "nose": (nose.x, nose.y),
    }

    return DetectionMetrics(side=side, values=values, confidence=confidence, points=points)


class RollingMetrics:
    def __init__(self, window_size: int, min_observations: int = 1) -> None:
        self.window_size = window_size
        self.min_observations = min_observations
        self.items: deque[DetectionMetrics] = deque(maxlen=window_size)

    def append(self, metrics: DetectionMetrics) -> None:
        self.items.append(metrics)

    def clear(self) -> None:
        self.items.clear()

    def mean(self) -> DetectionMetrics | None:
        if not self.items:
            return None

        side_counts: dict[str, int] = {}
        for item in self.items:
            side_counts[item.side] = side_counts.get(item.side, 0) + 1
        side = max(side_counts, key=side_counts.get)

        filtered = [item for item in self.items if item.side == side]
        metric_names = sorted(set().union(*(item.values.keys() for item in filtered)))
        point_names = sorted(set().union(*(item.points.keys() for item in filtered)))

        values: dict[str, float] = {}
        confidence: dict[str, float] = {}
        for name in metric_names:
            observations = [
                (item.values[name], item.confidence.get(name, 1.0))
                for item in filtered
                if name in item.values
            ]
            if len(observations) < self.min_observations:
                continue
            total_weight = sum(weight for _, weight in observations)
            if total_weight > 0:
                values[name] = sum(value * weight for value, weight in observations) / total_weight
                confidence[name] = total_weight / len(observations)
            else:
                values[name] = statistics.fmean(value for value, _ in observations)
                confidence[name] = 0.0

        points = {
            name: (
                statistics.fmean(item.points[name][0] for item in filtered if name in item.points),
                statistics.fmean(item.points[name][1] for item in filtered if name in item.points),
            )
            for name in point_names
        }
        return DetectionMetrics(side=side, values=values, confidence=confidence, points=points)


def classify_posture(
    profile: CalibrationProfile | None,
    metrics: DetectionMetrics | None,
    min_bad_metrics: int = 2,
    min_confidence: float = 0.0,
) -> tuple[bool, dict[str, bool], dict[str, float]]:
    if not profile or not metrics or metrics.side != profile.side:
        return False, {}, {}

    bad_by_metric: dict[str, bool] = {}
    severity: dict[str, float] = {}

    # A metric only counts when it is enabled in the profile, present in this
    # frame and reliable enough. Missing confidence is treated as fully trusted
    # so legacy metrics keep their previous behaviour.
    available_enabled = [
        name
        for name in metrics.values
        if name in profile.thresholds
        and profile.thresholds[name].mode != "disabled"
        and metrics.confidence.get(name, 1.0) >= min_confidence
    ]

    for metric_name in available_enabled:
        value = metrics.values[metric_name]
        threshold = profile.thresholds[metric_name]

        if threshold.mode == "directional":
            signed_distance = (value - threshold.threshold) * (threshold.direction or 1)
            bad = signed_distance > 0
            normalized = max(0.0, signed_distance / max(threshold.margin, 1e-6))
        else:
            distance = abs(value - threshold.threshold)
            bad = distance > threshold.margin
            normalized = max(0.0, (distance - threshold.margin) / max(threshold.margin, 1e-6))

        bad_by_metric[metric_name] = bad
        severity[metric_name] = normalized

    available_count = len(available_enabled)
    bad_metrics = sum(1 for value in bad_by_metric.values() if value)
    # Never require every available metric to be bad: leave room for one to be off.
    cap = available_count - 1 if available_count > 1 else 1
    effective_min = max(1, min(min_bad_metrics, cap))
    is_bad = available_count > 0 and bad_metrics >= effective_min
    return is_bad, bad_by_metric, severity
