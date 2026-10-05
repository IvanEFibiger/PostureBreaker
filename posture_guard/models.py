from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Coordinate/geometry convention version. Bumped whenever the landmark pipeline
# changes what raw coordinates mean (e.g. the mirrored -> raw MediaPipe fix).
# Calibrations stamped with an older value must be recalibrated, never migrated.
GEOMETRY_VERSION = 2


@dataclass
class MetricThreshold:
    threshold: float
    mode: str  # "directional", "absolute" or "disabled"
    direction: int | None
    margin: float
    weight: float = 1.0


@dataclass(frozen=True)
class MetricObservation:
    value: float
    confidence: float


@dataclass(frozen=True)
class DebugLandmark:
    """A 2D landmark kept with its confidence for the debug overlay.

    Confidence is preserved so the renderer (not the detector) decides whether
    the point is reliable enough to draw.
    """

    x: float
    y: float
    confidence: float


@dataclass
class MetricBaseline:
    center: float
    spread: float
    coverage: float
    confidence: float
    sample_count: int
    mean: float = 0.0
    std: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "center": self.center,
            "spread": self.spread,
            "coverage": self.coverage,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MetricBaseline:
        return cls(
            center=float(data["center"]),
            spread=float(data["spread"]),
            coverage=float(data.get("coverage", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            sample_count=int(data.get("sample_count", 0)),
            mean=float(data.get("mean", 0.0)),
            std=float(data.get("std", 0.0)),
        )


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
    metric_baselines: dict[str, MetricBaseline] = field(default_factory=dict)

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
            "metric_baselines": {
                name: baseline.to_json() for name, baseline in self.metric_baselines.items()
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ViewProfile:
        orientation = data.get("orientation") or {}
        posture = data.get("posture")
        baselines = data.get("metric_baselines") or {}
        return cls(
            id=str(data.get("id", "principal")),
            name=str(data.get("name", "Principal")),
            head_yaw_mean=orientation.get("head_yaw_mean"),
            torso_yaw_mean=orientation.get("torso_yaw_mean"),
            head_yaw_std=float(orientation.get("head_yaw_std", 0.0)),
            torso_yaw_std=float(orientation.get("torso_yaw_std", 0.0)),
            calibration=CalibrationProfile.from_json(posture) if posture else None,
            metric_baselines={
                name: MetricBaseline.from_json(payload) for name, payload in baselines.items()
            },
        )


@dataclass
class CalibrationSet:
    schema_version: int = 3
    geometry_version: int = GEOMETRY_VERSION
    profiles: list[ViewProfile] = field(default_factory=list)
    active_profile_id: str | None = None
    global_baselines: dict[str, MetricBaseline] = field(default_factory=dict)

    def active_profile(self) -> ViewProfile | None:
        for profile in self.profiles:
            if profile.id == self.active_profile_id:
                return profile
        return self.profiles[0] if self.profiles else None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "geometry_version": self.geometry_version,
            "profiles": [profile.to_json() for profile in self.profiles],
            "active_profile_id": self.active_profile_id,
            "global_baselines": {
                name: baseline.to_json() for name, baseline in self.global_baselines.items()
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CalibrationSet:
        # A missing ``geometry_version`` means the old (pre-mirroring) pipeline;
        # default to 0 so it is detected as requiring recalibration.
        geometry_version = int(data.get("geometry_version", 0))
        # A file without ``schema_version`` is the V1 single-profile format.
        if "schema_version" not in data:
            view = ViewProfile(id="principal", name="Principal", calibration=CalibrationProfile.from_json(data))
            return cls(
                schema_version=3,
                geometry_version=geometry_version,
                profiles=[view],
                active_profile_id="principal",
            )
        global_payload = data.get("global_baselines") or {}
        return cls(
            schema_version=3,
            geometry_version=geometry_version,
            profiles=[ViewProfile.from_json(payload) for payload in data.get("profiles", [])],
            active_profile_id=data.get("active_profile_id"),
            global_baselines={
                name: MetricBaseline.from_json(payload) for name, payload in global_payload.items()
            },
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
    confidences: dict[str, float] = field(default_factory=dict)


@dataclass
class ShoulderState:
    roll: float | None = None
    left_elevation: float | None = None
    right_elevation: float | None = None
    elevation: float | None = None

    confidence: float = 0.0
    confidences: dict[str, float] = field(default_factory=dict)


@dataclass
class ForwardState:
    head_forward_ratio: float | None = None
    head_depth_ratio: float | None = None
    torso_forward_angle: float | None = None

    confidence: float = 0.0
    confidences: dict[str, float] = field(default_factory=dict)


@dataclass
class DetectionMetrics:
    side: str
    values: dict[str, float]
    confidence: dict[str, float] = field(default_factory=dict)
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    view: ViewState | None = None
    shoulders: ShoulderState | None = None
    forward: ForwardState | None = None
    observations: dict[str, MetricObservation] = field(default_factory=dict)
    # Both-sides 2D landmarks for the V2 debug overlay (legacy ``points`` only
    # carries the selected side). Debug-only: never feeds metrics or risk.
    debug_landmarks: dict[str, DebugLandmark] = field(default_factory=dict)


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
