from __future__ import annotations

from .config import Config
from .models import AlertState


class BreakManager:
    # Seconds the user must be back on camera before we consider the break
    # interrupted. Brief appearances won't reset away progress.
    RETURN_GRACE_SECONDS = 5.0

    def __init__(self, config: Config, state: AlertState) -> None:
        self.config = config
        self.state = state
        self._back_streak = 0.0
        self._away_streak = 0.0
        self._work_reset_done = False

    def reset_continuity(self) -> None:
        """Drop continuity streaks after a session gap without losing work time."""
        self._back_streak = 0.0
        self._away_streak = 0.0
        self._work_reset_done = False

    def update(
        self,
        has_pose: bool,
        dt: float,
        now_ts: float,
        repeat_interval_seconds: float | None = None,
    ) -> list[str]:
        events: list[str] = []
        repeat_interval = (
            self.config.break_repeat_alert_seconds
            if repeat_interval_seconds is None
            else repeat_interval_seconds
        )

        if self.state.break_due:
            if has_pose:
                self._back_streak += dt
                self._away_streak = 0.0
                self._work_reset_done = False
                if self._back_streak >= self.RETURN_GRACE_SECONDS:
                    self.state.away_during_break = 0.0
                    if now_ts - self.state.last_break_alert_at >= repeat_interval:
                        self.state.last_break_alert_at = now_ts
                        events.append("break_alert_repeat")
            else:
                self._back_streak = 0.0
                self._away_streak += dt
                self.state.away_during_break += dt
                if self.state.away_during_break >= self.config.break_required_seconds:
                    self.state.break_due = False
                    self.state.away_during_break = 0.0
                    self.state.work_since_break = 0.0
                    self.state.completed_breaks += 1
                    self._back_streak = 0.0
                    self._away_streak = 0.0
                    self._work_reset_done = False
                    events.append("break_completed")
            return events

        if has_pose:
            self._away_streak = 0.0
            self._work_reset_done = False
            self.state.work_since_break += dt
            if self.state.work_since_break >= self.config.break_interval_minutes * 60.0:
                self.state.break_due = True
                self.state.away_during_break = 0.0
                self.state.last_break_alert_at = now_ts
                self.state.break_alert_count += 1
                self._back_streak = 0.0
                events.append("break_alert")
        else:
            self._back_streak = 0.0
            self._away_streak += dt
            if (
                not self._work_reset_done
                and self._away_streak >= self.config.away_reset_seconds
                and self.state.work_since_break > 0.0
            ):
                self.state.work_since_break = 0.0
                self._work_reset_done = True
                events.append("work_reset_absence")

        return events

