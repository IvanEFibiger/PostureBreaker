from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricThreshold:
    threshold: float
    mode: str  # "directional", "absolute" or "disabled"
    direction: int | None
    margin: float


@dataclass
class CalibrationProfile:
    side: str
    good_mean: dict[str, float]
    good_std: dict[str, float]
    bad_mean: dict[str, float] | None = None
    thresholds: dict[str, MetricThreshold] = field(default_factory=dict)
    quality_score: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "good_mean": self.good_mean,
            "good_std": self.good_std,
            "bad_mean": self.bad_mean,
            "quality_score": self.quality_score,
            "thresholds": {
                name: {
                    "threshold": value.threshold,
                    "mode": value.mode,
                    "direction": value.direction,
                    "margin": value.margin,
                }
                for name, value in self.thresholds.items()
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CalibrationProfile:
        thresholds = {
            name: MetricThreshold(**payload)
            for name, payload in data["thresholds"].items()
        }
        return cls(
            side=data["side"],
            good_mean=data["good_mean"],
            good_std=data["good_std"],
            bad_mean=data.get("bad_mean"),
            thresholds=thresholds,
            quality_score=float(data.get("quality_score", 0.0)),
        )


@dataclass
class ViewState:
    head_yaw: float | None = None
    head_pitch: float | None = None
    head_roll: float | None = None

    torso_yaw: float | None = None
    torso_lateral_lean: float | None = None
    neck_roll_delta: float | None = None

    orientation: str = "unknown"
    confidence: float = 0.0


@dataclass
class ShoulderState:
    roll: float | None = None
    left_elevation: float | None = None
    right_elevation: float | None = None
    elevation: float | None = None

    confidence: float = 0.0


@dataclass
class DetectionMetrics:
    side: str
    values: dict[str, float]
    confidence: dict[str, float] = field(default_factory=dict)
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    view: ViewState | None = None
    shoulders: ShoulderState | None = None


@dataclass
class AlertState:
    bad_posture_streak: float = 0.0
    last_posture_alert_at: float = 0.0
    posture_alert_count: int = 0
    break_alert_count: int = 0
    completed_breaks: int = 0
    work_since_break: float = 0.0
    break_due: bool = False
    away_during_break: float = 0.0
    last_break_alert_at: float = 0.0
    posture_active: bool = False
