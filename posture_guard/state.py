from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SharedState:
    """Thread-safe shared state between camera worker and dashboard."""

    lock: threading.Lock = field(default_factory=threading.Lock)

    camera_frame: object = None
    camera_visible: bool = False
    overlay_message: str = ""
    overlay_bg: str = "#b00020"
    overlay_mode: str = "banner"
    overlay_alpha: float = 0.92
    posture_score: float = 100.0
    is_bad_posture: bool = False
    status: str = "starting"
    error_code: str = ""
    error_message: str = ""
    work_seconds_today: float = 0.0
    bad_streak_seconds: float = 0.0
    bad_posture_seconds_today: float = 0.0
    posture_alerts: int = 0
    suppressed_posture_alerts: int = 0
    break_alerts: int = 0
    breaks_completed: int = 0
    break_due: bool = False
    calibrated: bool = False
    calibrating_mode: str = ""
    calibration_progress: float = 0.0
    has_pose: bool = False
    streak_days: int = 0
    summary_today: str = ""
    current_issue: str = ""
    current_guidance: str = ""
    focus_mode: bool = False
    focus_hint: str = ""
    focus_seconds_today: float = 0.0
    break_progress: float = 0.0
    break_countdown_seconds: float = 0.0
    break_title: str = ""
    break_instruction: str = ""
    break_rule_text: str = ""
    hourly_trend: list[dict[str, Any]] = field(default_factory=list)
    weekly_trend: list[dict[str, Any]] = field(default_factory=list)
    top_errors: list[dict[str, Any]] = field(default_factory=list)
    issue_times_today: list[dict[str, Any]] = field(default_factory=list)
    view_times_today: list[dict[str, Any]] = field(default_factory=list)
    available_cameras: list[int] = field(default_factory=list)
    camera_index: int = 0
    calibration_quality: float = 0.0
    calibration_summary: str = ""
    autostart_enabled: bool = False
    settings: dict[str, float] = field(default_factory=dict)

    validation_active: bool = False
    validation_phase: str = ""
    validation_scenario: str = ""
    validation_title: str = ""
    validation_instruction: str = ""
    validation_progress: float = 0.0
    validation_remaining_seconds: float = 0.0
    validation_scenario_index: int = 0
    validation_scenario_count: int = 0
    validation_sample_count: int = 0
    validation_run_id: str = ""
    validation_view_name: str = ""
    validation_message: str = ""

    cmd_calibrate_good: bool = False
    cmd_calibrate_bad: bool = False
    cmd_calibrate_new_view: bool = False
    cmd_clear_calibration: bool = False
    cmd_toggle_camera: bool = False
    cmd_toggle_focus: bool = False
    cmd_snooze: bool = False
    cmd_clear_history: bool = False
    cmd_export_history: bool = False
    cmd_set_camera: bool = False
    cmd_toggle_autostart: bool = False
    cmd_diagnostics: bool = False
    cmd_snapshot: bool = False
    cmd_apply_settings: bool = False
    cmd_start_validation: bool = False
    cmd_cancel_validation: bool = False
    cmd_hide_window: bool = False
    cmd_show_window: bool = False
    cmd_quit: bool = False
    pending_camera_index: int = -1
    pending_snapshot_label: str = "good"
    pending_settings: dict[str, float] = field(default_factory=dict)

    def update(self, **kwargs: object) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {k: v for k, v in self.__dict__.items() if k != "lock"}

    def consume_command(self, name: str) -> bool:
        with self.lock:
            value = getattr(self, name, False)
            if value:
                setattr(self, name, False)
            return value

    def consume_value(self, name: str, default: object) -> Any:
        with self.lock:
            value = getattr(self, name, default)
            setattr(self, name, default)
            return value
