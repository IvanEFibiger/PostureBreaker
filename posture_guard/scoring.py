from __future__ import annotations

import datetime
import json
from pathlib import Path


class PostureScorer:
    """Tracks good/bad posture time and computes a daily score percentage."""

    def __init__(self, history_dir: Path) -> None:
        self.history_dir = history_dir
        self._good_seconds = 0.0
        self._bad_seconds = 0.0
        self._today = datetime.date.today()
        self._load_today()

    def _streak_path(self) -> Path:
        return self.history_dir / "streak.json"

    def _load_today(self) -> None:
        path = self.history_dir / f"{self._today.isoformat()}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._good_seconds = data.get("good_posture_seconds", 0.0)
            self._bad_seconds = data.get("bad_posture_seconds", 0.0)

    def _roll_day(self) -> None:
        today = datetime.date.today()
        if today != self._today:
            self._today = today
            self._good_seconds = 0.0
            self._bad_seconds = 0.0
            self._load_today()

    def add(self, dt: float, is_bad: bool) -> None:
        self._roll_day()
        if is_bad:
            self._bad_seconds += dt
        else:
            self._good_seconds += dt

    @property
    def score(self) -> float:
        total = self._good_seconds + self._bad_seconds
        if total < 1.0:
            return 100.0
        return round(self._good_seconds / total * 100.0, 1)

    @property
    def good_seconds(self) -> float:
        return self._good_seconds

    @property
    def bad_seconds(self) -> float:
        return self._bad_seconds

    def save(self, history_data: dict) -> None:
        """Merge score fields into the existing history data and write."""
        history_data["good_posture_seconds"] = self._good_seconds
        history_data["bad_posture_seconds"] = self._bad_seconds
        history_data["posture_score"] = self.score

    def load_streak(self) -> int:
        """Count consecutive past days with score >= 70%."""
        streak = 0
        day = datetime.date.today() - datetime.timedelta(days=1)
        while True:
            path = self.history_dir / f"{day.isoformat()}.json"
            if not path.exists():
                break
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("posture_score", 0) >= 70.0:
                streak += 1
                day -= datetime.timedelta(days=1)
            else:
                break
        # Include today if score is good so far.
        if self.score >= 70.0 and (self._good_seconds + self._bad_seconds) > 60:
            streak += 1
        return streak
