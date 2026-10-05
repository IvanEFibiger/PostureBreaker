from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricThreshold:
    threshold: float
    mode: str  # "directional" or "absolute"
    direction: int | None
    margin: float


@dataclass
class CalibrationProfile:
    side: str
    good_mean: dict[str, float]
    good_std: dict[str, float]
    bad_mean: dict[str, float] | None = None
    thresholds: dict[str, MetricThreshold] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "good_mean": self.good_mean,
            "good_std": self.good_std,
            "bad_mean": self.bad_mean,
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
        )


@dataclass
class DetectionMetrics:
    side: str
    values: dict[str, float]
    points: dict[str, tuple[float, float]]


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
