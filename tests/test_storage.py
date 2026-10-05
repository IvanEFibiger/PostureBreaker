from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from posture_guard.storage import AnalyticsStore


class AnalyticsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "history" / "test.db"
        self.store = AnalyticsStore(self.db_path)

    def tearDown(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass
        self._tmp.cleanup()

    def test_score_defaults_to_100_without_data(self) -> None:
        self.assertEqual(self.store.score, 100.0)

    def test_score_reflects_good_and_bad_time(self) -> None:
        self.store.add_posture_time(30.0, is_bad=True)
        self.store.add_posture_time(70.0, is_bad=False)
        self.assertEqual(self.store.score, 70.0)

    def test_work_time_accumulates_and_tracks_focus(self) -> None:
        self.store.add_work_time(10.0, focus_mode=False)
        self.store.add_work_time(5.0, focus_mode=True)
        snapshot = self.store.today_snapshot()
        self.assertEqual(snapshot["total_work_seconds"], 15.0)
        self.assertEqual(snapshot["focus_seconds"], 5.0)

    def test_flush_and_reopen_preserves_score(self) -> None:
        self.store.add_posture_time(60.0, is_bad=False)
        self.store.add_posture_time(40.0, is_bad=True)
        self.store.flush()
        self.store.close()
        self.store = AnalyticsStore(self.db_path)
        self.assertEqual(self.store.score, 60.0)

    def test_roll_day_resets_daily_totals(self) -> None:
        yesterday = dt.date.today() - dt.timedelta(days=1)
        self.store._today = yesterday
        self.store._daily = self.store._empty_daily(yesterday)
        self.store._daily["good_posture_seconds"] = 999.0
        self.store.add_posture_time(5.0, is_bad=False)
        self.assertEqual(self.store._today, dt.date.today())
        self.assertEqual(self.store._daily["good_posture_seconds"], 5.0)

    def test_load_streak_counts_consecutive_good_days(self) -> None:
        today = dt.date.today()
        self._insert_day(today - dt.timedelta(days=1), good=80.0, bad=20.0)
        self._insert_day(today - dt.timedelta(days=2), good=90.0, bad=10.0)
        self.assertEqual(self.store.load_streak(), 2)

    def test_load_streak_stops_at_first_bad_day(self) -> None:
        today = dt.date.today()
        self._insert_day(today - dt.timedelta(days=1), good=80.0, bad=20.0)
        self._insert_day(today - dt.timedelta(days=2), good=50.0, bad=50.0)
        self.assertEqual(self.store.load_streak(), 1)

    def test_load_streak_includes_today_when_good(self) -> None:
        today = dt.date.today()
        self._insert_day(today - dt.timedelta(days=1), good=80.0, bad=20.0)
        self.store.add_posture_time(80.0, is_bad=False)
        self.store.add_posture_time(20.0, is_bad=True)
        self.assertEqual(self.store.load_streak(), 2)

    def test_load_streak_ignores_today_with_too_little_data(self) -> None:
        self.store.add_posture_time(30.0, is_bad=False)
        self.assertEqual(self.store.load_streak(), 0)

    def _insert_day(self, day: dt.date, good: float, bad: float) -> None:
        self.store.conn.execute(
            """
            INSERT OR REPLACE INTO daily_stats(day, good_posture_seconds, bad_posture_seconds)
            VALUES (?, ?, ?)
            """,
            (day.isoformat(), good, bad),
        )
        self.store.conn.commit()


if __name__ == "__main__":
    unittest.main()
