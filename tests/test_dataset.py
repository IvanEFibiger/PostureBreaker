from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from posture_guard.dataset import (
    compare_to_baseline,
    compare_to_view_baselines,
    flatten_snapshot,
    format_comparison,
    format_response_matrix,
    format_summary,
    format_view_comparison,
    load_snapshots,
    metric_response_matrix,
    separability,
    snapshot_view,
    split_label,
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
            forward={"head_depth_ratio": 0.5, "torso_forward_angle": 6.0, "confidence": 0.6},
        )
        signals = flatten_snapshot(payload)
        self.assertAlmostEqual(signals["ear_shoulder_dx"][0], 0.05)
        self.assertAlmostEqual(signals["head_yaw"][0], 0.3)
        self.assertAlmostEqual(signals["shoulder_roll"][0], 2.0)
        self.assertAlmostEqual(signals["shoulder_elevation"][0], 0.33)
        self.assertAlmostEqual(signals["head_depth_ratio"][0], 0.5)
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


class SummarizeByViewTests(unittest.TestCase):
    def test_group_by_view_separates_monitors(self) -> None:
        rows = [
            snapshot("good", view={"orientation": "left"}),
            snapshot("good", view={"orientation": "right"}),
        ]
        summary = summarize_snapshots(rows, group_by_view=True)
        self.assertIn("good @ left", summary)
        self.assertIn("good @ right", summary)

    def test_active_view_name_takes_precedence(self) -> None:
        rows = [snapshot("good", active_view={"id": "v1", "name": "Monitor 1"})]
        summary = summarize_snapshots(rows, group_by_view=True)
        self.assertIn("good @ Monitor 1", summary)

    def test_snapshot_view_falls_back_to_unknown(self) -> None:
        self.assertEqual(snapshot_view(snapshot("good")), "unknown")


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


def view_snapshot(label: str, view_name: str, value: float) -> dict[str, object]:
    return snapshot(
        label,
        metrics={"m": value},
        active_view={"id": view_name, "name": view_name},
    )


class ViewBaselineComparisonTests(unittest.TestCase):
    def test_split_label(self) -> None:
        self.assertEqual(split_label("head_forward @ Monitor 1"), ("head_forward", "Monitor 1"))
        self.assertEqual(split_label("good"), ("good", ""))

    def test_each_scenario_uses_its_own_view_baseline(self) -> None:
        rows = [
            view_snapshot("good", "Monitor 1", 1.0),
            view_snapshot("head_forward", "Monitor 1", 3.0),
            view_snapshot("good", "Monitor 2", 2.0),
            view_snapshot("head_forward", "Monitor 2", 5.0),
        ]
        summary = summarize_snapshots(rows, group_by_view=True)
        comparison = compare_to_view_baselines(summary)

        monitor1 = comparison["head_forward @ Monitor 1"]
        self.assertEqual(monitor1.baseline_label, "good @ Monitor 1")
        self.assertAlmostEqual(monitor1.metrics["m"]["delta"], 2.0)

        monitor2 = comparison["head_forward @ Monitor 2"]
        self.assertEqual(monitor2.baseline_label, "good @ Monitor 2")
        self.assertAlmostEqual(monitor2.metrics["m"]["delta"], 3.0)

    def test_missing_view_baseline_is_not_crossed(self) -> None:
        rows = [
            view_snapshot("head_forward", "Monitor 1", 3.0),
            view_snapshot("good", "Monitor 2", 2.0),
        ]
        summary = summarize_snapshots(rows, group_by_view=True)
        self.assertEqual(compare_to_view_baselines(summary), {})

    def test_baseline_scenario_is_excluded(self) -> None:
        rows = [
            view_snapshot("good", "Monitor 1", 1.0),
            view_snapshot("head_forward", "Monitor 1", 3.0),
        ]
        summary = summarize_snapshots(rows, group_by_view=True)
        comparison = compare_to_view_baselines(summary)
        self.assertNotIn("good @ Monitor 1", comparison)

    def test_custom_baseline_scenario(self) -> None:
        rows = [
            view_snapshot("front_good", "Monitor 2", 1.0),
            view_snapshot("head_forward", "Monitor 2", 2.0),
        ]
        summary = summarize_snapshots(rows, group_by_view=True)
        comparison = compare_to_view_baselines(summary, baseline_scenario="front_good")
        self.assertIn("head_forward @ Monitor 2", comparison)
        self.assertEqual(comparison["head_forward @ Monitor 2"].baseline_label, "front_good @ Monitor 2")

    def test_format_view_comparison_names_both_labels(self) -> None:
        rows = [
            view_snapshot("good", "Monitor 1", 1.0),
            view_snapshot("head_forward", "Monitor 1", 3.0),
        ]
        summary = summarize_snapshots(rows, group_by_view=True)
        text = format_view_comparison(compare_to_view_baselines(summary))
        self.assertIn("head_forward @ Monitor 1 vs good @ Monitor 1", text)

    def test_response_matrix_groups_effects_by_metric(self) -> None:
        rows = [
            view_snapshot("good", "Monitor 1", 1.0),
            view_snapshot("head_forward", "Monitor 1", 3.0),
        ]
        summary = summarize_snapshots(rows, group_by_view=True)
        matrix = metric_response_matrix(compare_to_view_baselines(summary))
        self.assertIn("head_forward @ Monitor 1", matrix["m"])
        self.assertGreater(matrix["m"]["head_forward @ Monitor 1"], 0.0)
        self.assertIn("effect", format_response_matrix(matrix))


class SnapshotRoundTripTests(unittest.TestCase):
    def test_json_line_is_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.jsonl"
            path.write_text(json.dumps(snapshot("a", metrics={"m": 1.0})) + "\n", encoding="utf-8")
            self.assertEqual(load_snapshots(path)[0]["label"], "a")


if __name__ == "__main__":
    unittest.main()
