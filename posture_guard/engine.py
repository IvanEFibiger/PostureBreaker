from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .alerts import BreakManager
from .config import Config
from .detection import classify_posture, dominant_issue
from .issues import issue_details, policy_for
from .models import AlertState, CalibrationProfile, DetectionMetrics, MetricBaseline
from .risk import IssueEvaluation, evaluate_issues, evaluate_v2_issues
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
    ISSUE_STARTED = "issue_started"
    ISSUE_RECOVERED = "issue_recovered"
    ISSUE_ALERT = "issue_alert"
    POSTURE_LOAD_ALERT = "posture_load_alert"


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
    risk_score: float = 0.0
    dominant_issue: str | None = None
    dominant_issue_label: str = ""
    dominant_issue_guidance: str = ""
    dominant_issue_severity: float = 0.0
    dominant_issue_streak_seconds: float = 0.0
    dominant_issue_load: float = 0.0


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
        self.issue_streaks: dict[str, float] = {}
        self.issue_loads: dict[str, float] = {}
        self._issue_alert_at: dict[str, float] = {}
        self.view_baselines: dict[str, MetricBaseline] = {}
        self.global_baselines: dict[str, MetricBaseline] = {}

    def set_profile(self, profile: CalibrationProfile | None) -> None:
        self.profile = profile

    def set_baselines(
        self,
        view_baselines: dict[str, MetricBaseline],
        global_baselines: dict[str, MetricBaseline],
    ) -> None:
        self.view_baselines = dict(view_baselines)
        self.global_baselines = dict(global_baselines)

    def clear_profile(self) -> None:
        self.profile = None
        self.state.posture_active = False
        self.state.bad_posture_streak = 0.0

    def reset_posture_state(self) -> None:
        self.state.posture_active = False
        self.state.bad_posture_streak = 0.0
        self.issue_streaks.clear()
        self.issue_loads.clear()

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
            self.issue_streaks.clear()
            self.issue_loads.clear()
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

        if metrics and (self.view_baselines or self.global_baselines):
            evaluation = evaluate_v2_issues(
                metrics.observations,
                self.view_baselines,
                self.global_baselines,
                self.config,
            )
        elif self.profile and metrics:
            evaluation = evaluate_issues(
                metrics.values, metrics.confidence, self.profile, self.config.metric_min_confidence
            )
        else:
            evaluation = IssueEvaluation()
        v2_label, v2_guidance = issue_details(evaluation.dominant_issue)

        observed: set[str] = set()
        for issue, weighted in evaluation.severity_by_issue.items():
            key = issue.value
            observed.add(key)
            was_issue_active = key in self.issue_streaks
            self.issue_streaks[key] = self.issue_streaks.get(key, 0.0) + dt
            policy = policy_for(issue)
            if policy.load_based:
                self.issue_loads[key] = self.issue_loads.get(key, 0.0) + weighted * dt
            if self.config.posture_v2_observe_only:
                continue
            if not was_issue_active:
                events.append(Event.ISSUE_STARTED)
            if policy.load_based:
                should_alert = policy.load_threshold > 0 and self.issue_loads.get(key, 0.0) >= policy.load_threshold
            else:
                should_alert = self.issue_streaks[key] >= policy.threshold_seconds
            last_alert = self._issue_alert_at.get(key, 0.0)
            if should_alert and now - last_alert >= policy.cooldown_seconds:
                self._issue_alert_at[key] = now
                events.append(Event.POSTURE_LOAD_ALERT if policy.load_based else Event.ISSUE_ALERT)

        for key in [name for name in self.issue_streaks if name not in observed]:
            del self.issue_streaks[key]
            self.issue_loads.pop(key, None)
            if not self.config.posture_v2_observe_only:
                events.append(Event.ISSUE_RECOVERED)

        dominant_key = evaluation.dominant_issue.value if evaluation.dominant_issue else None
        dominant_streak = self.issue_streaks.get(dominant_key, 0.0) if dominant_key else 0.0
        dominant_load = self.issue_loads.get(dominant_key, 0.0) if dominant_key else 0.0

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
            risk_score=evaluation.risk_score,
            dominant_issue=evaluation.dominant_issue.value if evaluation.dominant_issue else None,
            dominant_issue_label=v2_label,
            dominant_issue_guidance=v2_guidance,
            dominant_issue_severity=evaluation.dominant_severity,
            dominant_issue_streak_seconds=dominant_streak,
            dominant_issue_load=dominant_load,
        )
