from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import StrEnum

from .config import Config
from .issues import PostureIssue
from .models import ForwardState, MetricBaseline, MetricObservation, ShoulderState, ViewState

EPSILON = 1e-6

# Minimum margin multiplier over the calibrated noise (MAD) so we never alert
# just because MediaPipe happens to be very stable on a given signal.
NOISE_MULTIPLIER = 4.0

# Relative relations should not be "forgiven" by changing monitor (plan §"GLOBAL").
V2_METRIC_NAMES = (
    "head_yaw",
    "head_pitch",
    "head_roll",
    "torso_yaw",
    "torso_lateral_lean",
    "neck_roll_delta",
    "shoulder_roll",
    "left_shoulder_elevation",
    "right_shoulder_elevation",
    "shoulder_elevation",
    "head_forward_ratio",
    "torso_forward_angle",
)


class CalibrationScope(StrEnum):
    VIEW = "view"
    GLOBAL = "global"
    NONE = "none"


class DeviationMode(StrEnum):
    HIGHER_IS_WORSE = "higher_is_worse"
    LOWER_IS_WORSE = "lower_is_worse"
    TWO_SIDED = "two_sided"


@dataclass(frozen=True)
class MetricSpec:
    issue: PostureIssue | None
    calibration_scope: CalibrationScope
    deviation_mode: DeviationMode
    default_weight: float = 1.0
    min_confidence: float = 0.6


# Semantic knowledge lives here, not in the calibration: calibration only learns
# what is normal for this user/view and how noisy the signal is.
METRIC_SPECS: dict[str, MetricSpec] = {
    # View-context signals: used to recognize the view, never turned into issues.
    "head_yaw": MetricSpec(None, CalibrationScope.NONE, DeviationMode.TWO_SIDED),
    "torso_yaw": MetricSpec(None, CalibrationScope.NONE, DeviationMode.TWO_SIDED),
    # View posture baselines.
    "head_pitch": MetricSpec(PostureIssue.NECK_FLEXION, CalibrationScope.VIEW, DeviationMode.HIGHER_IS_WORSE, 1.0),
    "head_roll": MetricSpec(PostureIssue.HEAD_TILT, CalibrationScope.VIEW, DeviationMode.TWO_SIDED, 0.8),
    "torso_lateral_lean": MetricSpec(None, CalibrationScope.VIEW, DeviationMode.TWO_SIDED),
    "shoulder_roll": MetricSpec(PostureIssue.SHOULDER_ASYMMETRY, CalibrationScope.VIEW, DeviationMode.TWO_SIDED, 0.7),
    "shoulder_elevation": MetricSpec(
        PostureIssue.SHOULDER_ELEVATION, CalibrationScope.VIEW, DeviationMode.HIGHER_IS_WORSE, 1.1
    ),
    "left_shoulder_elevation": MetricSpec(
        PostureIssue.SHOULDER_ELEVATION, CalibrationScope.VIEW, DeviationMode.HIGHER_IS_WORSE, 1.1
    ),
    "right_shoulder_elevation": MetricSpec(
        PostureIssue.SHOULDER_ELEVATION, CalibrationScope.VIEW, DeviationMode.HIGHER_IS_WORSE, 1.1
    ),
    "head_forward_ratio": MetricSpec(PostureIssue.HEAD_FORWARD, CalibrationScope.VIEW, DeviationMode.HIGHER_IS_WORSE, 1.5),
    "torso_forward_angle": MetricSpec(
        PostureIssue.TORSO_FORWARD, CalibrationScope.VIEW, DeviationMode.HIGHER_IS_WORSE, 1.2
    ),
    # Body-relative relations: calibrated globally so switching monitor cannot hide them.
    "neck_roll_delta": MetricSpec(PostureIssue.HEAD_TILT, CalibrationScope.GLOBAL, DeviationMode.TWO_SIDED, 0.8),
}


def spec_for(metric_name: str) -> MetricSpec | None:
    return METRIC_SPECS.get(metric_name)


def build_observations(
    view: ViewState | None,
    shoulders: ShoulderState | None,
    forward: ForwardState | None,
) -> dict[str, MetricObservation]:
    """Flat V2 signal collection, kept separate from the legacy ``values``."""
    observations: dict[str, MetricObservation] = {}
    if view is not None:
        for name in ("head_yaw", "head_pitch", "head_roll", "torso_yaw", "torso_lateral_lean", "neck_roll_delta"):
            value = getattr(view, name)
            if value is not None:
                observations[name] = MetricObservation(value, view.confidence)
    if shoulders is not None:
        for name, value in (
            ("shoulder_roll", shoulders.roll),
            ("left_shoulder_elevation", shoulders.left_elevation),
            ("right_shoulder_elevation", shoulders.right_elevation),
            ("shoulder_elevation", shoulders.elevation),
        ):
            if value is not None:
                observations[name] = MetricObservation(value, shoulders.confidence)
    if forward is not None:
        for name, value in (
            ("head_forward_ratio", forward.head_forward_ratio),
            ("torso_forward_angle", forward.torso_forward_angle),
        ):
            if value is not None:
                observations[name] = MetricObservation(value, forward.confidence)
    return observations


def _baseline_from_observations(
    observations: list[MetricObservation],
    coverage: float,
    confidence: float,
) -> MetricBaseline:
    values = [observation.value for observation in observations]
    center = statistics.median(values)
    spread = statistics.median([abs(value - center) for value in values])
    return MetricBaseline(
        center=center,
        spread=spread,
        coverage=coverage,
        confidence=confidence,
        sample_count=len(values),
        mean=statistics.fmean(values),
        std=statistics.pstdev(values) if len(values) > 1 else 0.0,
    )


def build_metric_baselines(
    observation_frames: list[dict[str, MetricObservation]],
    config: Config,
) -> dict[str, MetricBaseline]:
    """Per-metric robust baselines from a good calibration window.

    Signals that appear too rarely or too unreliably are simply not calibrated.
    """
    total = len(observation_frames)
    if total == 0:
        return {}

    names = sorted(set().union(*(frame.keys() for frame in observation_frames)))
    baselines: dict[str, MetricBaseline] = {}
    for name in names:
        present = [frame[name] for frame in observation_frames if name in frame]
        coverage = len(present) / total
        if coverage < config.calibration_min_metric_coverage:
            continue
        confidence = statistics.fmean(observation.confidence for observation in present)
        if confidence < config.metric_min_confidence:
            continue
        baselines[name] = _baseline_from_observations(present, coverage, confidence)
    return baselines


def resolve_baseline(
    metric_name: str,
    spec: MetricSpec,
    view_baselines: dict[str, MetricBaseline],
    global_baselines: dict[str, MetricBaseline],
) -> MetricBaseline | None:
    if spec.calibration_scope is CalibrationScope.GLOBAL:
        return global_baselines.get(metric_name)
    return view_baselines.get(metric_name)


def normalize_observation(
    observation: MetricObservation,
    baseline: MetricBaseline | None,
    spec: MetricSpec,
    configured_margin: float,
) -> float:
    """Severity of a signal relative to its calibrated baseline, or 0.0 if fine."""
    if baseline is None or observation.confidence < spec.min_confidence:
        return 0.0

    deviation = observation.value - baseline.center
    if spec.deviation_mode is DeviationMode.TWO_SIDED:
        deviation = abs(deviation)
    elif spec.deviation_mode is DeviationMode.LOWER_IS_WORSE:
        deviation = -deviation
    if deviation <= 0:
        return 0.0

    allowed_margin = max(configured_margin, baseline.spread * NOISE_MULTIPLIER, EPSILON)
    severity_scale = max(configured_margin, EPSILON)
    return max(0.0, (deviation - allowed_margin) / severity_scale)
