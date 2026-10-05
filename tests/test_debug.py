from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from posture_guard.debug import (
    FULL_BODY_SNAPSHOT_LABELS,
    SNAPSHOT_LABELS,
    STATE_LOW,
    STATE_MISSING,
    STATE_OK,
    V1_HEADER,
    V2_ACTIVE_HEADER,
    V2_OBSERVE_HEADER,
    append_snapshot,
    build_snapshot,
    format_debug_lines,
    format_v1_status_lines,
    format_v2_status_lines,
    label_for_digit,
    metric_state,
    next_snapshot_label,
)
from posture_guard.models import DetectionMetrics, ForwardState, MetricObservation, ShoulderState, ViewState


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
        forward=ForwardState(
            head_forward_ratio=0.21, head_depth_ratio=0.42, torso_forward_angle=6.5, confidence=0.85
        ),
        observations={"head_yaw": MetricObservation(0.31, 0.92)},
    )


class SnapshotLabelTests(unittest.TestCase):
    def test_labels_are_unique(self) -> None:
        self.assertEqual(len(SNAPSHOT_LABELS), len(set(SNAPSHOT_LABELS)))

    def test_labels_are_scenarios_not_monitors(self) -> None:
        self.assertIn("good", SNAPSHOT_LABELS)
        self.assertIn("head_forward", SNAPSHOT_LABELS)
        self.assertIn("head_torso_turn", SNAPSHOT_LABELS)
        self.assertFalse(any("monitor" in label for label in SNAPSHOT_LABELS))

    def test_desk_protocol_excludes_hip_scenarios(self) -> None:
        for label in ("torso_forward", "torso_lean_left", "torso_lean_right"):
            self.assertNotIn(label, SNAPSHOT_LABELS)
            self.assertIn(label, FULL_BODY_SNAPSHOT_LABELS)

    def test_digit_keys_map_to_first_ten(self) -> None:
        self.assertEqual(label_for_digit("1"), SNAPSHOT_LABELS[0])
        self.assertEqual(label_for_digit("9"), SNAPSHOT_LABELS[8])
        self.assertEqual(label_for_digit("0"), SNAPSHOT_LABELS[9])

    def test_non_digit_has_no_label(self) -> None:
        self.assertIsNone(label_for_digit("s"))
        self.assertIsNone(label_for_digit(""))

    def test_cycling_wraps(self) -> None:
        self.assertEqual(next_snapshot_label(SNAPSHOT_LABELS[0], -1), SNAPSHOT_LABELS[-1])
        self.assertEqual(next_snapshot_label(SNAPSHOT_LABELS[-1], 1), SNAPSHOT_LABELS[0])
        self.assertEqual(next_snapshot_label("unknown", 1), SNAPSHOT_LABELS[0])


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
        self.assertIn("depth", text)

    def test_missing_and_low_signals_are_marked(self) -> None:
        metrics = make_metrics()
        metrics.view.torso_yaw = None
        metrics.view.confidence = 0.4
        lines = format_debug_lines(metrics, min_confidence=0.6)
        text = "\n".join(lines)
        self.assertIn("yaw", text)
        self.assertIn("--", text)  # missing
        self.assertIn("?", text)  # low confidence


class StatusHeaderTests(unittest.TestCase):
    def test_v1_block_is_explicit(self) -> None:
        lines = format_v1_status_lines("Cuello empujado", posture_bad=True)
        text = "\n".join(lines)
        self.assertIn(V1_HEADER, text)
        self.assertIn("V1 Issue: Cuello empujado", text)
        self.assertIn("yes", text)

    def test_v2_observe_only_header(self) -> None:
        lines = format_v2_status_lines(True, risk_score=1.42, dominant_issue="head_forward", candidate_label="Cabeza adelantada")
        text = "\n".join(lines)
        self.assertIn(V2_OBSERVE_HEADER, text)
        self.assertIn("V2 Candidate: Cabeza adelantada", text)
        self.assertIn("1.42", text)
        self.assertNotIn(V2_ACTIVE_HEADER, text)

    def test_v2_active_header_when_activated(self) -> None:
        lines = format_v2_status_lines(False, risk_score=0.0, dominant_issue=None)
        text = "\n".join(lines)
        self.assertIn(V2_ACTIVE_HEADER, text)
        self.assertNotIn(V2_OBSERVE_HEADER, text)
        self.assertIn("V2 Candidate: -", text)


class PerMetricConfidenceTests(unittest.TestCase):
    def test_state_follows_each_field_confidence_not_the_average(self) -> None:
        metrics = make_metrics()
        metrics.view.confidence = 0.90
        metrics.view.confidences = {"head_roll": 0.95, "torso_yaw": 0.30}
        lines = format_debug_lines(metrics, min_confidence=0.6)

        head_roll_line = next(line for line in lines if "+0.01" in line)
        torso_yaw_line = next(line for line in lines if "+0.07" in line)
        self.assertIn(STATE_OK, head_roll_line)  # "ok"
        self.assertIn("?", torso_yaw_line)  # low-confidence symbol

    def test_torso_yaw_shows_the_folded_display_value(self) -> None:
        metrics = make_metrics()
        metrics.view.torso_yaw = 141.5
        lines = format_debug_lines(metrics, min_confidence=0.6)
        self.assertTrue(any("-38.50" in line for line in lines))


class BuildSnapshotTests(unittest.TestCase):
    def test_includes_label_view_and_shoulders(self) -> None:
        moment = dt.datetime(2026, 10, 5, 10, 0, 0)
        snapshot = build_snapshot(make_metrics(), "monitor_1_good", moment)
        self.assertEqual(snapshot["label"], "monitor_1_good")
        self.assertEqual(snapshot["timestamp"], "2026-10-05T10:00:00")
        self.assertEqual(snapshot["side"], "right")
        self.assertAlmostEqual(snapshot["view"]["torso_yaw"], 0.07)
        self.assertAlmostEqual(snapshot["shoulders"]["elevation"], 0.32)
        self.assertAlmostEqual(snapshot["forward"]["head_depth_ratio"], 0.42)
        self.assertAlmostEqual(snapshot["forward"]["torso_forward_angle"], 6.5)
        self.assertAlmostEqual(snapshot["observations"]["head_yaw"]["value"], 0.31)
        self.assertAlmostEqual(snapshot["metrics"]["ear_shoulder_dx"], 0.05)

    def test_run_id_is_optional_and_recorded(self) -> None:
        self.assertIsNone(build_snapshot(make_metrics(), "good")["run_id"])
        tagged = build_snapshot(make_metrics(), "good", run_id="good_B")
        self.assertEqual(tagged["run_id"], "good_B")

    def test_records_active_view_automatically(self) -> None:
        snapshot = build_snapshot(make_metrics(), "good", active_view_id="view_1", active_view_name="Monitor 1")
        self.assertEqual(snapshot["active_view"], {"id": "view_1", "name": "Monitor 1"})
        self.assertEqual(snapshot["view"]["orientation"], "right")

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
