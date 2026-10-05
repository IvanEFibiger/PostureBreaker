from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricThreshold:
    threshold: float
    mode: str  # "directional", "absolute" or "disabled"
    direction: int | None
    margin: float
    weight: float = 1.0


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
                    "weight": value.weight,
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
class ViewProfile:
    id: str
    name: str
    head_yaw_mean: float | None = None
    torso_yaw_mean: float | None = None
    head_yaw_std: float = 0.0
    torso_yaw_std: float = 0.0
    calibration: CalibrationProfile | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "orientation": {
                "head_yaw_mean": self.head_yaw_mean,
                "torso_yaw_mean": self.torso_yaw_mean,
                "head_yaw_std": self.head_yaw_std,
                "torso_yaw_std": self.torso_yaw_std,
            },
            "posture": self.calibration.to_json() if self.calibration else None,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ViewProfile:
        orientation = data.get("orientation") or {}
        posture = data.get("posture")
        return cls(
            id=str(data.get("id", "principal")),
            name=str(data.get("name", "Principal")),
            head_yaw_mean=orientation.get("head_yaw_mean"),
            torso_yaw_mean=orientation.get("torso_yaw_mean"),
            head_yaw_std=float(orientation.get("head_yaw_std", 0.0)),
            torso_yaw_std=float(orientation.get("torso_yaw_std", 0.0)),
            calibration=CalibrationProfile.from_json(posture) if posture else None,
        )


@dataclass
class CalibrationSet:
    schema_version: int = 2
    profiles: list[ViewProfile] = field(default_factory=list)
    active_profile_id: str | None = None

    def active_profile(self) -> ViewProfile | None:
        for profile in self.profiles:
            if profile.id == self.active_profile_id:
                return profile
        return self.profiles[0] if self.profiles else None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profiles": [profile.to_json() for profile in self.profiles],
            "active_profile_id": self.active_profile_id,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CalibrationSet:
        # A file without ``schema_version`` is the V1 single-profile format.
        if "schema_version" not in data:
            view = ViewProfile(id="principal", name="Principal", calibration=CalibrationProfile.from_json(data))
            return cls(schema_version=2, profiles=[view], active_profile_id="principal")
        return cls(
            schema_version=int(data.get("schema_version", 2)),
            profiles=[ViewProfile.from_json(payload) for payload in data.get("profiles", [])],
            active_profile_id=data.get("active_profile_id"),
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
class ForwardState:
    head_forward_ratio: float | None = None
    torso_forward_angle: float | None = None

    confidence: float = 0.0


@dataclass
class DetectionMetrics:
    side: str
    values: dict[str, float]
    confidence: dict[str, float] = field(default_factory=dict)
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    view: ViewState | None = None
    shoulders: ShoulderState | None = None
    forward: ForwardState | None = None


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
