from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .config import Config
from .models import CalibrationProfile, DetectionMetrics, MetricThreshold


def build_thresholds(profile: CalibrationProfile, config: Config) -> dict[str, MetricThreshold]:
    thresholds: dict[str, MetricThreshold] = {}
    indistinguishable: list[str] = []

    for metric_name, good_value in profile.good_mean.items():
        margin = max(
            config.default_margins.get(metric_name, 0.03),
            profile.good_std.get(metric_name, 0.0) * 2.5,
        )

        if profile.bad_mean and metric_name in profile.bad_mean:
            bad_value = profile.bad_mean[metric_name]
            if abs(bad_value - good_value) <= margin:
                indistinguishable.append(metric_name)
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

    if indistinguishable:
        raise ValueError(
            "La postura mala no se distingue de la buena en: "
            + ", ".join(indistinguishable)
            + ". Repetí la calibración con una postura mala más marcada."
        )

    return thresholds


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
        metric_names = list(side_samples[0].keys())

        mean_values = {
            name: statistics.fmean(sample[name] for sample in side_samples)
            for name in metric_names
        }
        std_values = {
            name: (
                statistics.pstdev(sample[name] for sample in side_samples)
                if len(side_samples) > 1
                else 0.0
            )
            for name in metric_names
        }

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
        self.cancel()
        return profile


def save_calibration(path: Path, profile: CalibrationProfile) -> None:
    path.write_text(json.dumps(profile.to_json(), indent=2), encoding="utf-8")


def load_calibration(path: Path) -> CalibrationProfile | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationProfile.from_json(data)
