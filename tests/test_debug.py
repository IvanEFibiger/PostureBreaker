from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from posture_guard.debug import (
    STATE_LOW,
    STATE_MISSING,
    STATE_OK,
    append_snapshot,
    build_snapshot,
    format_debug_lines,
    metric_state,
)
from posture_guard.models import DetectionMetrics, ForwardState, ShoulderState, ViewState


def make_metrics() -> DetectionMetrics:
    return DetectionMetrics(
        side="right",
        values={"ear_shoulder_dx": 0.05},
        confidence={"ear_shoulder_dx": 0.9},
        view=ViewState(
            head_yaw=0.31,
            head_pitch=-0.03,
            head_roll=0.01,
            torso_yaw=0.07,
            torso_lateral_lean=0.02,
            neck_roll_delta=0.01,
            orientation="right",
            confidence=0.92,
        ),
        shoulders=ShoulderState(roll=1.8, left_elevation=0.32, right_elevation=0.30, elevation=0.32, confidence=0.88),
        forward=ForwardState(head_forward_ratio=0.21, torso_forward_angle=6.5, confidence=0.85),
    )


class MetricStateTests(unittest.TestCase):
    def test_missing_value(self) -> None:
        self.assertEqual(metric_state(None, 0.9, 0.6), STATE_MISSING)

    def test_low_confidence(self) -> None:
        self.assertEqual(metric_state(1.0, 0.2, 0.6), STATE_LOW)

    def test_confident_value(self) -> None:
        self.assertEqual(metric_state(1.0, 0.9, 0.6), STATE_OK)


class FormatDebugLinesTests(unittest.TestCase):
    def test_no_metrics(self) -> None:
        self.assertEqual(format_debug_lines(None), ["DEBUG: sin pose"])

    def test_includes_all_sections(self) -> None:
        lines = format_debug_lines(make_metrics(), min_confidence=0.6)
        text = "\n".join(lines)
        for section in ("VIEW", "HEAD", "TORSO", "RELATIVE", "SHOULDERS", "FORWARD"):
            self.assertIn(section, text)
        self.assertIn("right", lines[0])
        self.assertIn("92%", lines[0])

    def test_missing_and_low_signals_are_marked(self) -> None:
        metrics = make_metrics()
        metrics.view.torso_yaw = None
        metrics.view.confidence = 0.4
        lines = format_debug_lines(metrics, min_confidence=0.6)
        text = "\n".join(lines)
        self.assertIn("yaw", text)
        self.assertIn("--", text)  # missing
        self.assertIn("?", text)  # low confidence


class BuildSnapshotTests(unittest.TestCase):
    def test_includes_label_view_and_shoulders(self) -> None:
        moment = dt.datetime(2026, 10, 5, 10, 0, 0)
        snapshot = build_snapshot(make_metrics(), "monitor_1_good", moment)
        self.assertEqual(snapshot["label"], "monitor_1_good")
        self.assertEqual(snapshot["timestamp"], "2026-10-05T10:00:00")
        self.assertEqual(snapshot["side"], "right")
        self.assertAlmostEqual(snapshot["view"]["torso_yaw"], 0.07)
        self.assertAlmostEqual(snapshot["shoulders"]["elevation"], 0.32)
        self.assertAlmostEqual(snapshot["forward"]["torso_forward_angle"], 6.5)
        self.assertAlmostEqual(snapshot["metrics"]["ear_shoulder_dx"], 0.05)

    def test_none_metrics_is_safe(self) -> None:
        snapshot = build_snapshot(None, "manual")
        self.assertIsNone(snapshot["side"])
        self.assertIsNone(snapshot["view"])
        self.assertEqual(snapshot["metrics"], {})


class AppendSnapshotTests(unittest.TestCase):
    def test_writes_one_json_line_per_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "snapshots.jsonl"
            append_snapshot(path, build_snapshot(make_metrics(), "a"))
            append_snapshot(path, build_snapshot(make_metrics(), "b"))
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["label"], "a")
            self.assertEqual(json.loads(lines[1])["label"], "b")


if __name__ == "__main__":
    unittest.main()
