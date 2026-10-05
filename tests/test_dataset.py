from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from posture_guard.dataset import (
    compare_to_baseline,
    flatten_snapshot,
    format_comparison,
    format_summary,
    load_snapshots,
    separability,
    summarize_snapshots,
)


def snapshot(label: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "label": label,
        "metrics": {},
        "confidence": {},
        "view": None,
        "shoulders": None,
        "forward": None,
    }
    payload.update(overrides)
    return payload


class LoadSnapshotsTests(unittest.TestCase):
    def test_missing_file_is_empty(self) -> None:
        self.assertEqual(load_snapshots(Path("does_not_exist.jsonl")), [])

    def test_skips_blank_and_invalid_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snaps.jsonl"
            path.write_text('{"label": "a"}\n\nnot json\n{"label": "b"}\n', encoding="utf-8")
            rows = load_snapshots(path)
            self.assertEqual([row["label"] for row in rows], ["a", "b"])


class FlattenSnapshotTests(unittest.TestCase):
    def test_merges_all_side_structures(self) -> None:
        payload = snapshot(
            "x",
            metrics={"ear_shoulder_dx": 0.05},
            confidence={"ear_shoulder_dx": 0.8},
            view={"head_yaw": 0.3, "torso_yaw": None, "confidence": 0.9},
            shoulders={"roll": 2.0, "elevation": 0.33, "confidence": 0.7},
            forward={"torso_forward_angle": 6.0, "confidence": 0.6},
        )
        signals = flatten_snapshot(payload)
        self.assertAlmostEqual(signals["ear_shoulder_dx"][0], 0.05)
        self.assertAlmostEqual(signals["head_yaw"][0], 0.3)
        self.assertAlmostEqual(signals["shoulder_roll"][0], 2.0)
        self.assertAlmostEqual(signals["shoulder_elevation"][0], 0.33)
        self.assertAlmostEqual(signals["torso_forward_angle"][0], 6.0)
        self.assertNotIn("torso_yaw", signals)

    def test_none_side_structures_are_ignored(self) -> None:
        self.assertEqual(flatten_snapshot(snapshot("x")), {})

    def test_reads_observations_payload(self) -> None:
        payload = snapshot("x", observations={"head_yaw": {"value": 0.3, "confidence": 0.8}})
        signals = flatten_snapshot(payload)
        self.assertAlmostEqual(signals["head_yaw"][0], 0.3)
        self.assertAlmostEqual(signals["head_yaw"][1], 0.8)


class SummarizeSnapshotsTests(unittest.TestCase):
    def test_computes_stats_per_label(self) -> None:
        rows = [
            snapshot("good", metrics={"neck_yaw_delta": 4.0}, confidence={"neck_yaw_delta": 0.8}),
            snapshot("good", metrics={"neck_yaw_delta": 6.0}, confidence={"neck_yaw_delta": 0.6}),
            snapshot("bad", metrics={"neck_yaw_delta": 30.0}, confidence={"neck_yaw_delta": 0.9}),
        ]
        summary = summarize_snapshots(rows)
        self.assertAlmostEqual(summary["good"]["neck_yaw_delta"].mean, 5.0)
        self.assertAlmostEqual(summary["good"]["neck_yaw_delta"].confidence, 0.7)
        self.assertEqual(summary["good"]["neck_yaw_delta"].count, 2)
        self.assertAlmostEqual(summary["bad"]["neck_yaw_delta"].mean, 30.0)

    def test_coverage_counts_missing_samples(self) -> None:
        rows = [
            snapshot("good", metrics={"a": 1.0}),
            snapshot("good", metrics={}),
        ]
        summary = summarize_snapshots(rows)
        self.assertAlmostEqual(summary["good"]["a"].coverage, 0.5)

    def test_separability_is_higher_for_separated_labels(self) -> None:
        separated = summarize_snapshots(
            [
                snapshot("good", metrics={"m": 1.0}),
                snapshot("bad", metrics={"m": 10.0}),
            ]
        )
        overlapping = summarize_snapshots(
            [
                snapshot("good", metrics={"m": 5.0}),
                snapshot("bad", metrics={"m": 5.0}),
            ]
        )
        self.assertGreater(separability(separated, "m"), separability(overlapping, "m"))
        self.assertGreater(separability(separated, "m"), 0.0)

    def test_format_summary_mentions_labels(self) -> None:
        summary = summarize_snapshots([snapshot("good", metrics={"m": 1.0})])
        self.assertIn("good", format_summary(summary))

    def test_median_and_mad_are_robust(self) -> None:
        rows = [snapshot("good", metrics={"m": float(value)}) for value in (1.0, 2.0, 3.0, 4.0, 5.0)]
        stats = summarize_snapshots(rows)["good"]["m"]
        self.assertAlmostEqual(stats.median, 3.0)
        self.assertAlmostEqual(stats.mad, 1.0)


class BaselineComparisonTests(unittest.TestCase):
    def test_compares_scenarios_against_baseline(self) -> None:
        rows = [
            snapshot("monitor_1_good", metrics={"neck_yaw_delta": 4.0}),
            snapshot("neck_rotated", metrics={"neck_yaw_delta": 30.0}),
        ]
        summary = summarize_snapshots(rows)
        comparison = compare_to_baseline(summary, "monitor_1_good")
        self.assertIn("neck_rotated", comparison)
        row = comparison["neck_rotated"]["neck_yaw_delta"]
        self.assertAlmostEqual(row["delta"], 26.0)
        self.assertGreater(row["effect"], 0.0)

    def test_missing_baseline_label_is_empty(self) -> None:
        summary = summarize_snapshots([snapshot("a", metrics={"m": 1.0})])
        self.assertEqual(compare_to_baseline(summary, "nope"), {})

    def test_format_comparison_mentions_labels(self) -> None:
        summary = summarize_snapshots(
            [
                snapshot("base", metrics={"m": 1.0}),
                snapshot("scenario", metrics={"m": 2.0}),
            ]
        )
        comparison = compare_to_baseline(summary, "base")
        self.assertIn("scenario", format_comparison(comparison))

    def test_baseline_label_is_excluded(self) -> None:
        summary = summarize_snapshots(
            [
                snapshot("base", metrics={"m": 1.0}),
                snapshot("scenario", metrics={"m": 2.0}),
            ]
        )
        comparison = compare_to_baseline(summary, "base")
        self.assertNotIn("base", comparison)


class SnapshotRoundTripTests(unittest.TestCase):
    def test_json_line_is_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.jsonl"
            path.write_text(json.dumps(snapshot("a", metrics={"m": 1.0})) + "\n", encoding="utf-8")
            self.assertEqual(load_snapshots(path)[0]["label"], "a")


if __name__ == "__main__":
    unittest.main()
