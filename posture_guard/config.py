from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    model_path: str = "pose_landmarker.task"
    database_path: str = "history/posture_guard.db"
    camera_index: int = 0
    min_visibility: float = 0.55
    sustained_bad_posture_seconds: float = 20.0
    posture_min_bad_metrics: int = 2
    posture_alert_cooldown_seconds: float = 300.0
    break_interval_minutes: float = 45.0
    break_required_seconds: float = 90.0
    break_repeat_alert_seconds: float = 30.0
    away_reset_seconds: float = 75.0
    calibration_frames: int = 90
    target_fps: float = 15.0
    max_frame_gap_seconds: float = 30.0
    analytics_sample_seconds: float = 10.0
    trend_refresh_seconds: float = 15.0
    focus_posture_multiplier: float = 2.0
    focus_cooldown_multiplier: float = 2.0
    focus_break_repeat_seconds: float = 180.0
    headless: bool = False
    history_dir: str = "history"
    smoothing_window: int = 12
    default_margins: dict[str, float] = field(
        default_factory=lambda: {
            "ear_shoulder_dx": 0.035,
            "nose_shoulder_dx": 0.045,
            "chin_drop": 0.030,
            "torso_lean_dx": 0.040,
        }
    )


_POSITIVE_FIELDS = (
    "sustained_bad_posture_seconds",
    "break_interval_minutes",
    "break_required_seconds",
    "break_repeat_alert_seconds",
    "away_reset_seconds",
    "target_fps",
    "max_frame_gap_seconds",
    "analytics_sample_seconds",
    "trend_refresh_seconds",
    "focus_posture_multiplier",
    "focus_cooldown_multiplier",
    "focus_break_repeat_seconds",
)
_NON_NEGATIVE_FIELDS = ("posture_alert_cooldown_seconds",)
_AT_LEAST_ONE_FIELDS = ("calibration_frames", "smoothing_window", "posture_min_bad_metrics")
_NON_EMPTY_TEXT_FIELDS = ("model_path", "database_path", "history_dir")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate(config: Config) -> None:
    if not isinstance(config.camera_index, int) or isinstance(config.camera_index, bool) or config.camera_index < 0:
        raise ValueError(f"'camera_index' debe ser un entero >= 0 (actual: {config.camera_index!r}).")

    if not _is_number(config.min_visibility) or not 0.0 <= float(config.min_visibility) <= 1.0:
        raise ValueError(f"'min_visibility' debe estar entre 0 y 1 (actual: {config.min_visibility!r}).")

    for name in _POSITIVE_FIELDS:
        value = getattr(config, name)
        if not _is_number(value) or value <= 0:
            raise ValueError(f"'{name}' debe ser un número mayor que 0 (actual: {value!r}).")

    for name in _NON_NEGATIVE_FIELDS:
        value = getattr(config, name)
        if not _is_number(value) or value < 0:
            raise ValueError(f"'{name}' debe ser un número mayor o igual a 0 (actual: {value!r}).")

    for name in _AT_LEAST_ONE_FIELDS:
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"'{name}' debe ser un entero mayor o igual a 1 (actual: {value!r}).")

    for name in _NON_EMPTY_TEXT_FIELDS:
        value = getattr(config, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'{name}' debe ser un texto no vacío (actual: {value!r}).")

    if not isinstance(config.headless, bool):
        raise ValueError(f"'headless' debe ser true o false (actual: {config.headless!r}).")

    if not isinstance(config.default_margins, dict):
        raise ValueError("'default_margins' debe ser un objeto con márgenes numéricos.")
    for name, value in config.default_margins.items():
        if not _is_number(value) or value < 0:
            raise ValueError(f"'default_margins.{name}' debe ser un número mayor o igual a 0 (actual: {value!r}).")


def load_config(path: Path) -> Config:
    if not path.exists():
        return Config()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"No pude leer la config {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"La config {path} debe ser un objeto JSON.")

    known = {spec.name for spec in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            "Claves desconocidas en la config: " + ", ".join(unknown) + ". Revisá el archivo."
        )

    config = Config(**data)
    _validate(config)
    return config
