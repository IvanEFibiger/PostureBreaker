from __future__ import annotations

import json
import statistics
from pathlib import Path

from .config import Config
from .models import CalibrationProfile, DetectionMetrics, MetricThreshold


def build_thresholds(profile: CalibrationProfile, config: Config) -> dict[str, MetricThreshold]:
    thresholds: dict[str, MetricThreshold] = {}

    for metric_name, good_value in profile.good_mean.items():
        margin = max(
            config.default_margins.get(metric_name, 0.03),
            profile.good_std.get(metric_name, 0.0) * 2.5,
        )

        if profile.bad_mean and metric_name in profile.bad_mean:
            bad_value = profile.bad_mean[metric_name]
            if abs(bad_value - good_value) <= margin:
                thresholds[metric_name] = MetricThreshold(
                    threshold=good_value,
                    mode="disabled",
                    direction=None,
                    margin=margin,
                )
                continue
            direction = 1 if bad_value >= good_value else -1
            threshold = good_value + (bad_value - good_value) * 0.55
            thresholds[metric_name] = MetricThreshold(
                threshold=threshold,
                mode="directional",
                direction=direction,
                margin=margin,
            )
        else:
            thresholds[metric_name] = MetricThreshold(
                threshold=good_value,
                mode="absolute",
                direction=None,
                margin=margin,
            )

    return thresholds


# Confidence assigned to a metric that only has a good baseline (absolute mode).
_ABSOLUTE_METRIC_QUALITY = 0.4


def calibration_quality(
    profile: CalibrationProfile,
    thresholds: dict[str, MetricThreshold],
) -> float:
    """Percentage (0-100) of how much the profile discriminates good vs bad posture."""
    if not thresholds:
        return 0.0

    scores: list[float] = []
    for metric_name, threshold in thresholds.items():
        if threshold.mode == "disabled":
            scores.append(0.0)
        elif threshold.mode == "directional" and profile.bad_mean and metric_name in profile.bad_mean:
            gap = abs(profile.bad_mean[metric_name] - profile.good_mean[metric_name])
            if threshold.margin > 0:
                scores.append(min(1.0, gap / (2.0 * threshold.margin)))
            else:
                scores.append(1.0)
        else:
            scores.append(_ABSOLUTE_METRIC_QUALITY)

    return round(100.0 * sum(scores) / len(scores), 1)


class Calibrator:
    def __init__(self, target_frames: int) -> None:
        self.target_frames = target_frames
        self.mode: str | None = None
        self.samples: list[tuple[str, dict[str, float]]] = []
        self.expected_side: str | None = None

    def start(self, mode: str, expected_side: str | None = None) -> None:
        self.mode = mode
        self.samples = []
        self.expected_side = expected_side

    def cancel(self) -> None:
        self.mode = None
        self.samples = []
        self.expected_side = None

    def is_running(self) -> bool:
        return self.mode is not None

    def add(self, metrics: DetectionMetrics) -> bool:
        if not self.mode:
            return False
        if self.expected_side and metrics.side != self.expected_side:
            return False
        self.samples.append((metrics.side, metrics.values.copy()))
        return len(self.samples) >= self.target_frames

    def build_profile(
        self,
        existing: CalibrationProfile | None,
        config: Config,
    ) -> CalibrationProfile:
        if not self.samples:
            raise ValueError("No hay muestras de calibración.")

        if self.expected_side:
            dominant_side = self.expected_side
            side_samples = [values for side, values in self.samples if side == dominant_side]
            if not side_samples:
                raise ValueError(
                    f"No pude juntar suficientes muestras del lado {dominant_side}. "
                    "Mantené visible ese mismo perfil/lado."
                )
        else:
            side_counts: dict[str, int] = {}
            for side, _ in self.samples:
                side_counts[side] = side_counts.get(side, 0) + 1
            dominant_side = max(side_counts, key=side_counts.get)
            side_samples = [values for side, values in self.samples if side == dominant_side]
        metric_names = sorted(set().union(*(sample.keys() for sample in side_samples)))

        mean_values = {
            name: statistics.fmean(sample[name] for sample in side_samples if name in sample)
            for name in metric_names
        }
        std_values: dict[str, float] = {}
        for name in metric_names:
            present = [sample[name] for sample in side_samples if name in sample]
            std_values[name] = statistics.pstdev(present) if len(present) > 1 else 0.0

        if self.mode == "good":
            profile = CalibrationProfile(
                side=dominant_side,
                good_mean=mean_values,
                good_std=std_values,
                bad_mean=existing.bad_mean if existing and existing.side == dominant_side else None,
            )
        elif self.mode == "bad":
            if not existing or existing.side != dominant_side:
                raise ValueError(
                    "Primero calibrá una postura buena con la misma cámara/ángulo."
                )
            profile = CalibrationProfile(
                side=existing.side,
                good_mean=existing.good_mean,
                good_std=existing.good_std,
                bad_mean=mean_values,
            )
        else:
            raise ValueError("Modo de calibración inválido.")

        profile.thresholds = build_thresholds(profile, config)
        enabled = sum(1 for threshold in profile.thresholds.values() if threshold.mode != "disabled")
        if enabled < config.calibration_min_enabled_metrics:
            self.cancel()
            raise ValueError(
                f"La calibración solo tiene {enabled} métrica(s) discriminante(s) y se "
                f"necesitan al menos {config.calibration_min_enabled_metrics}. "
                "Repetí la calibración con una postura mala más marcada."
            )
        profile.quality_score = calibration_quality(profile, profile.thresholds)
        self.cancel()
        return profile


def save_calibration(path: Path, profile: CalibrationProfile) -> None:
    path.write_text(json.dumps(profile.to_json(), indent=2), encoding="utf-8")


def load_calibration(path: Path) -> CalibrationProfile | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationProfile.from_json(data)
