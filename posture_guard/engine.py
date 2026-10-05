from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .alerts import BreakManager
from .config import Config
from .detection import classify_posture, dominant_issue
from .models import AlertState, CalibrationProfile, DetectionMetrics
from .timing import is_session_gap


class Event(StrEnum):
    POSTURE_BAD_STARTED = "posture_bad_started"
    POSTURE_RECOVERED = "posture_recovered"
    POSTURE_ALERT = "posture_alert"
    POSTURE_ALERT_SUPPRESSED = "posture_alert_suppressed"
    BREAK_ALERT = "break_alert"
    BREAK_ALERT_REPEAT = "break_alert_repeat"
    BREAK_COMPLETED = "break_completed"
    WORK_RESET_ABSENCE = "work_reset_absence"
    FOCUS_STARTED = "focus_started"
    FOCUS_ENDED = "focus_ended"
    SESSION_GAP = "session_gap"
    VIEW_CHANGED = "view_changed"


@dataclass
class EngineResult:
    has_pose: bool
    profile_ready: bool
    posture_bad: bool
    posture_active: bool
    bad_streak_seconds: float
    posture_threshold_seconds: float
    issue_type: str | None
    issue_label: str
    issue_guidance: str
    issue_severity: float
    break_due: bool
    break_progress: float
    break_countdown_seconds: float
    completed_breaks: int
    focus_mode: bool
    work_seconds_delta: float
    posture_seconds_delta: float
    count_posture_time: bool
    session_gap_seconds: float = 0.0
    snoozed: bool = False
    events: list[Event] = field(default_factory=list)


class PostureEngine:
    """Temporal and business logic for posture/break/focus, independent of
    camera, MediaPipe, storage and UI."""

    def __init__(self, config: Config, profile: CalibrationProfile | None = None) -> None:
        self.config = config
        self.profile = profile
        self.state = AlertState()
        self.break_manager = BreakManager(config, self.state)
        self.focus_mode = False
        self.snooze_until = 0.0

    def set_profile(self, profile: CalibrationProfile | None) -> None:
        self.profile = profile

    def clear_profile(self) -> None:
        self.profile = None
        self.state.posture_active = False
        self.state.bad_posture_streak = 0.0

    def reset_posture_state(self) -> None:
        self.state.posture_active = False
        self.state.bad_posture_streak = 0.0

    def toggle_focus(self) -> Event:
        self.focus_mode = not self.focus_mode
        return Event.FOCUS_STARTED if self.focus_mode else Event.FOCUS_ENDED

    def snooze(self, now: float, minutes: float) -> None:
        self.snooze_until = now + minutes * 60.0

    def is_snoozed(self, now: float) -> bool:
        return now < self.snooze_until

    def _posture_threshold(self) -> float:
        multiplier = self.config.focus_posture_multiplier if self.focus_mode else 1.0
        return self.config.sustained_bad_posture_seconds * multiplier

    def _posture_cooldown(self) -> float:
        multiplier = self.config.focus_cooldown_multiplier if self.focus_mode else 1.0
        return self.config.posture_alert_cooldown_seconds * multiplier

    def _break_repeat_interval(self) -> float:
        if self.focus_mode:
            return self.config.focus_break_repeat_seconds
        return self.config.break_repeat_alert_seconds

    def update(
        self,
        metrics: DetectionMetrics | None,
        has_pose: bool,
        dt: float,
        now: float,
    ) -> EngineResult:
        events: list[Event] = []
        session_gap_seconds = 0.0

        if is_session_gap(dt, self.config.max_frame_gap_seconds):
            session_gap_seconds = dt
            self.state.bad_posture_streak = 0.0
            self.state.posture_active = False
            self.break_manager.reset_continuity()
            events.append(Event.SESSION_GAP)
            dt = 0.0

        posture_bad, bad_by_metric, severity = classify_posture(
            self.profile,
            metrics,
            self.config.posture_min_bad_metrics,
            self.config.metric_min_confidence,
        )
        issue_type, issue_label, issue_guidance, issue_severity = dominant_issue(bad_by_metric, severity)

        was_active = self.state.posture_active
        threshold = self._posture_threshold()
        cooldown = self._posture_cooldown()
        snoozed = self.is_snoozed(now)

        if posture_bad:
            self.state.bad_posture_streak += dt
            self.state.posture_active = True
            if not was_active:
                events.append(Event.POSTURE_BAD_STARTED)
            if (
                self.state.bad_posture_streak >= threshold
                and now - self.state.last_posture_alert_at >= cooldown
                and not snoozed
            ):
                self.state.last_posture_alert_at = now
                self.state.posture_alert_count += 1
                events.append(
                    Event.POSTURE_ALERT_SUPPRESSED if self.focus_mode else Event.POSTURE_ALERT
                )
        else:
            if was_active:
                events.append(Event.POSTURE_RECOVERED)
            self.state.bad_posture_streak = 0.0
            self.state.posture_active = False

        for break_event in self.break_manager.update(has_pose, dt, now, self._break_repeat_interval()):
            event = Event(break_event)
            if snoozed and event in (Event.BREAK_ALERT, Event.BREAK_ALERT_REPEAT):
                continue
            events.append(event)

        if self.state.break_due:
            break_progress = min(
                1.0,
                self.state.away_during_break / max(self.config.break_required_seconds, 1.0),
            )
            countdown_seconds = 0.0
        else:
            interval_seconds = self.config.break_interval_minutes * 60.0
            worked = min(interval_seconds, self.state.work_since_break)
            break_progress = min(1.0, worked / max(interval_seconds, 1.0))
            countdown_seconds = max(0.0, interval_seconds - self.state.work_since_break)

        count_posture_time = has_pose and self.profile is not None

        return EngineResult(
            has_pose=has_pose,
            profile_ready=self.profile is not None,
            posture_bad=posture_bad,
            posture_active=self.state.posture_active,
            bad_streak_seconds=self.state.bad_posture_streak,
            posture_threshold_seconds=threshold,
            issue_type=issue_type,
            issue_label=issue_label,
            issue_guidance=issue_guidance,
            issue_severity=issue_severity,
            break_due=self.state.break_due,
            break_progress=break_progress,
            break_countdown_seconds=countdown_seconds,
            completed_breaks=self.state.completed_breaks,
            focus_mode=self.focus_mode,
            work_seconds_delta=dt if has_pose else 0.0,
            posture_seconds_delta=dt if count_posture_time else 0.0,
            count_posture_time=count_posture_time,
            session_gap_seconds=session_gap_seconds,
            snoozed=snoozed,
            events=events,
        )
