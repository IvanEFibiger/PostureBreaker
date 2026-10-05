from __future__ import annotations

import json
from dataclasses import dataclass, field
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


def load_config(path: Path) -> Config:
    if not path.exists():
        return Config()

    data = json.loads(path.read_text(encoding="utf-8"))
    return Config(**data)
