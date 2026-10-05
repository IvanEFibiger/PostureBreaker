from __future__ import annotations

import unittest

from posture_guard.alerts import BreakManager
from posture_guard.config import Config
from posture_guard.models import AlertState


def make_manager(**overrides: float) -> tuple[BreakManager, AlertState]:
    defaults: dict[str, float] = {
        "break_interval_minutes": 1.0,
        "break_required_seconds": 10.0,
        "break_repeat_alert_seconds": 30.0,
        "away_reset_seconds": 5.0,
    }
    defaults.update(overrides)
    state = AlertState()
    return BreakManager(Config(**defaults), state), state


class BreakManagerTests(unittest.TestCase):
    def test_break_alert_fires_after_interval_of_work(self) -> None:
        manager, state = make_manager()
        events = manager.update(has_pose=True, dt=60.0, now_ts=100.0)
        self.assertEqual(events, ["break_alert"])
        self.assertTrue(state.break_due)
        self.assertEqual(state.completed_breaks, 0)

    def test_no_break_alert_before_interval(self) -> None:
        manager, state = make_manager()
        events = manager.update(has_pose=True, dt=30.0, now_ts=100.0)
        self.assertEqual(events, [])
        self.assertFalse(state.break_due)

    def test_break_completed_after_required_away_time(self) -> None:
        manager, state = make_manager()
        manager.update(has_pose=True, dt=60.0, now_ts=100.0)
        events = manager.update(has_pose=False, dt=10.0, now_ts=110.0)
        self.assertEqual(events, ["break_completed"])
        self.assertFalse(state.break_due)
        self.assertEqual(state.completed_breaks, 1)
        self.assertEqual(state.work_since_break, 0.0)

    def test_repeat_alert_when_user_returns_during_break(self) -> None:
        manager, state = make_manager()
        manager.update(has_pose=True, dt=60.0, now_ts=100.0)
        events = manager.update(has_pose=True, dt=5.0, now_ts=200.0)
        self.assertEqual(events, ["break_alert_repeat"])
        self.assertEqual(state.away_during_break, 0.0)

    def test_no_repeat_before_cooldown(self) -> None:
        manager, state = make_manager()
        manager.update(has_pose=True, dt=60.0, now_ts=100.0)
        events = manager.update(has_pose=True, dt=5.0, now_ts=110.0)
        self.assertEqual(events, [])

    def test_repeat_interval_override_is_respected(self) -> None:
        manager, _ = make_manager()
        manager.update(has_pose=True, dt=60.0, now_ts=100.0)
        events = manager.update(has_pose=True, dt=5.0, now_ts=290.0, repeat_interval_seconds=180.0)
        self.assertEqual(events, ["break_alert_repeat"])

    def test_repeat_interval_override_blocks_early_repeat(self) -> None:
        manager, _ = make_manager()
        manager.update(has_pose=True, dt=60.0, now_ts=100.0)
        events = manager.update(has_pose=True, dt=5.0, now_ts=200.0, repeat_interval_seconds=180.0)
        self.assertEqual(events, [])

    def test_brief_visibility_does_not_complete_break(self) -> None:
        manager, state = make_manager()
        manager.update(has_pose=True, dt=60.0, now_ts=100.0)
        manager.update(has_pose=True, dt=1.0, now_ts=101.0)
        events = manager.update(has_pose=False, dt=10.0, now_ts=111.0)
        self.assertEqual(events, ["break_completed"])

    def test_work_reset_after_absence(self) -> None:
        manager, state = make_manager()
        state.work_since_break = 20.0
        events = manager.update(has_pose=False, dt=5.0, now_ts=100.0)
        self.assertEqual(events, ["work_reset_absence"])
        self.assertEqual(state.work_since_break, 0.0)

    def test_absence_reset_happens_only_once(self) -> None:
        manager, state = make_manager()
        state.work_since_break = 20.0
        manager.update(has_pose=False, dt=5.0, now_ts=100.0)
        events = manager.update(has_pose=False, dt=5.0, now_ts=110.0)
        self.assertEqual(events, [])

    def test_no_reset_when_nothing_worked(self) -> None:
        manager, _ = make_manager()
        events = manager.update(has_pose=False, dt=5.0, now_ts=100.0)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
