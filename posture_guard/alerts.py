from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any

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

    def update(self, has_pose: bool, dt: float, now_ts: float) -> list[str]:
        events: list[str] = []

        if self.state.break_due:
            if has_pose:
                self._back_streak += dt
                self._away_streak = 0.0
                self._work_reset_done = False
                if self._back_streak >= self.RETURN_GRACE_SECONDS:
                    self.state.away_during_break = 0.0
                    if now_ts - self.state.last_break_alert_at >= self.config.break_repeat_alert_seconds:
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


class Notifier:
    def __init__(self) -> None:
        self.is_windows = sys.platform.startswith("win")
        if self.is_windows:
            import winsound  # type: ignore

            self._winsound = winsound
        else:
            self._winsound = None

    def beep(self, kind: str) -> None:
        try:
            if self._winsound:
                if kind == "posture":
                    self._winsound.Beep(950, 220)
                elif kind == "break":
                    self._winsound.Beep(700, 500)
                elif kind == "ok":
                    self._winsound.Beep(1100, 180)
                else:
                    self._winsound.Beep(850, 180)
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass


class SessionHistory:
    def __init__(self, history_dir: Path) -> None:
        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._today = datetime.date.today()
        self._data = self._load(self._today)

    def _path_for(self, day: datetime.date) -> Path:
        return self.history_dir / f"{day.isoformat()}.json"

    def _load(self, day: datetime.date) -> dict[str, Any]:
        path = self._path_for(day)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {
            "date": day.isoformat(),
            "total_work_seconds": 0.0,
            "posture_alerts": 0,
            "break_alerts": 0,
            "breaks_completed": 0,
            "events": [],
        }

    def _save(self) -> None:
        path = self._path_for(self._today)
        path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _roll_day(self) -> None:
        today = datetime.date.today()
        if today != self._today:
            self._save()
            self._today = today
            self._data = self._load(today)

    def log_event(self, event_type: str, detail: str = "") -> None:
        self._roll_day()
        entry: dict[str, str] = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "type": event_type,
        }
        if detail:
            entry["detail"] = detail
        self._data["events"].append(entry)

        if event_type == "posture_alert":
            self._data["posture_alerts"] += 1
        elif event_type in {"break_alert", "break_alert_repeat"}:
            self._data["break_alerts"] += 1
        elif event_type == "break_completed":
            self._data["breaks_completed"] += 1

        self._save()

    def add_work_time(self, seconds: float) -> None:
        self._roll_day()
        self._data["total_work_seconds"] += seconds

    def flush(self) -> None:
        self._save()

    def summary_today(self) -> str:
        d = self._data
        work_min = d["total_work_seconds"] / 60.0
        return (
            f"Hoy: {work_min:.0f} min trabajo | "
            f"{d['posture_alerts']} alertas postura | "
            f"{d['break_alerts']} alertas pausa | "
            f"{d['breaks_completed']} pausas hechas"
        )
