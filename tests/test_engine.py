from __future__ import annotations

import unittest

from posture_guard.config import Config
from posture_guard.engine import Event, PostureEngine
from posture_guard.models import (
    CalibrationProfile,
    DetectionMetrics,
    MetricBaseline,
    MetricObservation,
    MetricThreshold,
)


def make_config(**overrides: float) -> Config:
    defaults: dict[str, float] = {
        "sustained_bad_posture_seconds": 1.0,
        "posture_alert_cooldown_seconds": 0.0,
        "break_interval_minutes": 1.0,
        "break_required_seconds": 10.0,
        "break_repeat_alert_seconds": 30.0,
        "away_reset_seconds": 5.0,
        "focus_break_repeat_seconds": 180.0,
        "max_frame_gap_seconds": 120.0,
    }
    defaults.update(overrides)
    return Config(**defaults)


def make_profile() -> CalibrationProfile:
    return CalibrationProfile(
        side="right",
        good_mean={},
        good_std={},
        thresholds={"ear_shoulder_dx": MetricThreshold(0.10, "directional", -1, 0.05)},
    )


def bad_metrics() -> DetectionMetrics:
    return DetectionMetrics(side="right", values={"ear_shoulder_dx": 0.05}, points={})


def good_metrics() -> DetectionMetrics:
    return DetectionMetrics(side="right", values={"ear_shoulder_dx": 0.20}, points={})


class TimeAccountingTests(unittest.TestCase):
    def test_work_and_posture_time_counted_with_pose_and_profile(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        result = engine.update(bad_metrics(), True, 0.5, 1.0)
        self.assertEqual(result.work_seconds_delta, 0.5)
        self.assertTrue(result.count_posture_time)
        self.assertEqual(result.posture_seconds_delta, 0.5)

    def test_no_work_time_without_pose(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        result = engine.update(None, False, 0.5, 1.0)
        self.assertEqual(result.work_seconds_delta, 0.0)
        self.assertFalse(result.count_posture_time)

    def test_no_posture_time_without_profile(self) -> None:
        engine = PostureEngine(make_config(), None)
        result = engine.update(bad_metrics(), True, 0.5, 1.0)
        self.assertEqual(result.work_seconds_delta, 0.5)
        self.assertFalse(result.count_posture_time)
        self.assertEqual(result.posture_seconds_delta, 0.0)


class PostureAlertTests(unittest.TestCase):
    def test_posture_alert_after_sustained_bad_posture(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        result = engine.update(bad_metrics(), True, 1.0, 10.0)
        self.assertIn(Event.POSTURE_BAD_STARTED, result.events)
        self.assertIn(Event.POSTURE_ALERT, result.events)

    def test_no_alert_within_cooldown(self) -> None:
        engine = PostureEngine(make_config(posture_alert_cooldown_seconds=100.0), make_profile())
        result = engine.update(bad_metrics(), True, 1.0, 10.0)
        self.assertNotIn(Event.POSTURE_ALERT, result.events)

    def test_alert_fires_after_cooldown(self) -> None:
        engine = PostureEngine(make_config(posture_alert_cooldown_seconds=100.0), make_profile())
        engine.update(bad_metrics(), True, 1.0, 10.0)
        result = engine.update(bad_metrics(), True, 1.0, 1000.0)
        self.assertIn(Event.POSTURE_ALERT, result.events)

    def test_recovery_emits_event(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        engine.update(bad_metrics(), True, 1.0, 10.0)
        result = engine.update(good_metrics(), True, 1.0, 11.0)
        self.assertIn(Event.POSTURE_RECOVERED, result.events)
        self.assertEqual(result.bad_streak_seconds, 0.0)

    def test_focus_mode_suppresses_posture_alert(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        engine.toggle_focus()
        result = engine.update(bad_metrics(), True, 3.0, 10.0)
        self.assertIn(Event.POSTURE_ALERT_SUPPRESSED, result.events)
        self.assertNotIn(Event.POSTURE_ALERT, result.events)


class BreakFlowTests(unittest.TestCase):
    def test_break_alert_after_work_interval(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        result = engine.update(good_metrics(), True, 60.0, 100.0)
        self.assertIn(Event.BREAK_ALERT, result.events)
        self.assertTrue(result.break_due)

    def test_break_completed_after_away_time(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        engine.update(good_metrics(), True, 60.0, 100.0)
        result = engine.update(None, False, 10.0, 110.0)
        self.assertIn(Event.BREAK_COMPLETED, result.events)
        self.assertFalse(result.break_due)
        self.assertEqual(result.completed_breaks, 1)

    def test_focus_changes_break_repeat_interval(self) -> None:
        engine = PostureEngine(make_config(break_repeat_alert_seconds=30.0, focus_break_repeat_seconds=180.0), make_profile())
        engine.update(good_metrics(), True, 60.0, 100.0)
        engine.toggle_focus()
        # 100s after the break alert: normal would repeat, focus should not.
        result = engine.update(good_metrics(), True, 5.0, 200.0)
        self.assertNotIn(Event.BREAK_ALERT_REPEAT, result.events)


class SessionGapTests(unittest.TestCase):
    def test_session_gap_drops_time_and_resets_streak(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        engine.update(bad_metrics(), True, 1.0, 10.0)
        result = engine.update(None, False, 2400.0, 2410.0)
        self.assertIn(Event.SESSION_GAP, result.events)
        self.assertEqual(result.session_gap_seconds, 2400.0)
        self.assertEqual(result.work_seconds_delta, 0.0)
        self.assertEqual(result.posture_seconds_delta, 0.0)
        self.assertEqual(result.bad_streak_seconds, 0.0)


class FocusToggleTests(unittest.TestCase):
    def test_toggle_focus_returns_transition_events(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        self.assertIs(engine.toggle_focus(), Event.FOCUS_STARTED)
        self.assertIs(engine.toggle_focus(), Event.FOCUS_ENDED)
        self.assertFalse(engine.focus_mode)


class SnoozeTests(unittest.TestCase):
    def test_snooze_suppresses_posture_alert(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        engine.snooze(now=0.0, minutes=10.0)
        result = engine.update(bad_metrics(), True, 1.0, 5.0)
        self.assertNotIn(Event.POSTURE_ALERT, result.events)
        self.assertTrue(result.snoozed)

    def test_alert_returns_after_snooze_expires(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        engine.snooze(now=0.0, minutes=10.0)
        engine.update(bad_metrics(), True, 1.0, 5.0)
        result = engine.update(bad_metrics(), True, 1.0, 700.0)
        self.assertIn(Event.POSTURE_ALERT, result.events)
        self.assertFalse(result.snoozed)

    def test_snooze_suppresses_break_alert(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        engine.snooze(now=0.0, minutes=60.0)
        result = engine.update(good_metrics(), True, 60.0, 100.0)
        self.assertNotIn(Event.BREAK_ALERT, result.events)

    def test_v2_risk_is_reported_in_parallel(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        result = engine.update(bad_metrics(), True, 0.5, 1.0)
        self.assertGreater(result.risk_score, 0.0)
        self.assertEqual(result.dominant_issue, "head_forward")
        self.assertTrue(result.dominant_issue_label)

    def test_v2_risk_is_empty_without_bad_metric(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        result = engine.update(good_metrics(), True, 0.5, 1.0)
        self.assertEqual(result.risk_score, 0.0)
        self.assertIsNone(result.dominant_issue)

    def test_v2_issue_events_suppressed_while_observing(self) -> None:
        engine = PostureEngine(make_config(), make_profile())
        result = engine.update(bad_metrics(), True, 25.0, 1000.0)
        self.assertNotIn(Event.ISSUE_STARTED, result.events)
        self.assertNotIn(Event.ISSUE_ALERT, result.events)
        self.assertGreater(result.dominant_issue_streak_seconds, 0.0)

    def test_v2_issue_alert_when_activated(self) -> None:
        engine = PostureEngine(make_config(posture_v2_observe_only=False), make_profile())
        result = engine.update(bad_metrics(), True, 25.0, 1000.0)
        self.assertIn(Event.ISSUE_STARTED, result.events)
        self.assertIn(Event.ISSUE_ALERT, result.events)

    def test_v2_load_alert_for_load_based_issue(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={},
            good_std={},
            thresholds={"head_yaw": MetricThreshold(0.2, "directional", 1, 0.1, weight=1.2)},
        )
        engine = PostureEngine(make_config(posture_v2_observe_only=False, max_frame_gap_seconds=1000.0), profile)
        metrics = DetectionMetrics(side="right", values={"head_yaw": 0.5}, points={})
        result = engine.update(metrics, True, 130.0, 1000.0)
        self.assertIn(Event.POSTURE_LOAD_ALERT, result.events)
        self.assertGreater(result.dominant_issue_load, 0.0)

    def test_load_based_needs_enough_accumulated_load(self) -> None:
        profile = CalibrationProfile(
            side="right",
            good_mean={},
            good_std={},
            thresholds={"head_yaw": MetricThreshold(0.2, "directional", 1, 0.1, weight=1.2)},
        )
        engine = PostureEngine(make_config(posture_v2_observe_only=False, max_frame_gap_seconds=1000.0), profile)
        mild = DetectionMetrics(side="right", values={"head_yaw": 0.3}, points={})
        result = engine.update(mild, True, 5.0, 1000.0)
        self.assertNotIn(Event.POSTURE_LOAD_ALERT, result.events)
        strong = DetectionMetrics(side="right", values={"head_yaw": 4.1}, points={})
        result = engine.update(strong, True, 5.0, 1005.0)
        self.assertIn(Event.POSTURE_LOAD_ALERT, result.events)

    def test_v2_issue_recovery_event(self) -> None:
        engine = PostureEngine(make_config(posture_v2_observe_only=False), make_profile())
        engine.update(bad_metrics(), True, 1.0, 1000.0)
        result = engine.update(good_metrics(), True, 1.0, 1001.0)
        self.assertIn(Event.ISSUE_RECOVERED, result.events)

    def test_v2_risk_uses_calibrated_baselines(self) -> None:
        engine = PostureEngine(make_config(), None)
        engine.set_baselines(
            {"head_forward_ratio": MetricBaseline(0.0, 0.0, 1.0, 1.0, 10)},
            {},
        )
        metrics = DetectionMetrics(
            side="right",
            values={},
            observations={"head_forward_ratio": MetricObservation(0.2, 1.0)},
        )
        result = engine.update(metrics, True, 0.5, 1.0)
        self.assertGreater(result.risk_score, 0.0)
        self.assertEqual(result.dominant_issue, "head_forward")


if __name__ == "__main__":
    unittest.main()
