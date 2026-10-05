from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from posture_guard.debug import SNAPSHOT_LABELS
from posture_guard.models import DetectionMetrics, MetricObservation, ViewState
from posture_guard.validation import (
    EFFECT_DISCLAIMER,
    SCENARIO_GUIDES,
    ValidationPhase,
    ValidationRunner,
    ValidationSession,
    build_desk_scenarios,
    build_run_comparison,
    build_validation_report,
    build_view_difference,
    make_run_id,
    validation_overlay_lines,
)


def make_runner(scenarios, *, interval=0.1, sink=None, run_id="20261005-181530", reset=0.0):
    samples: list[dict] = [] if sink is None else sink
    runner = ValidationRunner(
        scenarios,
        sample_interval_seconds=interval,
        run_id=run_id,
        test_view_id="view_1",
        test_view_name="Monitor 2",
        reset_seconds=reset,
        sample_sink=samples.append,
    )
    return runner, samples


def drive(runner, dt_step=0.1, *, active_id="view_other", active_name="Monitor 1", metrics=None):
    guard = 0
    while runner.is_active and guard < 100_000:
        runner.update(dt_step, metrics, active_view_id=active_id, active_view_name=active_name)
        guard += 1


class ScenarioTests(unittest.TestCase):
    def test_scenarios_follow_snapshot_labels(self) -> None:
        scenarios = build_desk_scenarios(4.0, 20.0)
        self.assertEqual([s.id for s in scenarios], list(SNAPSHOT_LABELS))
        for scenario in scenarios:
            title, instruction = SCENARIO_GUIDES[scenario.id]
            self.assertEqual(scenario.title, title)
            self.assertEqual(scenario.instruction, instruction)
            self.assertEqual(scenario.prepare_seconds, 4.0)
            self.assertEqual(scenario.capture_seconds, 20.0)

    def test_turn_scenarios_are_independent(self) -> None:
        ids = [s.id for s in build_desk_scenarios(4.0, 20.0)]
        for name in (
            "head_turn_left",
            "head_turn_right",
            "head_torso_turn_left",
            "head_torso_turn_right",
        ):
            self.assertIn(name, ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_full_body_scenarios_are_excluded(self) -> None:
        ids = {s.id for s in build_desk_scenarios(4.0, 20.0)}
        self.assertEqual(ids & {"torso_forward", "torso_lean_left", "torso_lean_right"}, set())

    def test_run_id_format(self) -> None:
        self.assertEqual(make_run_id(dt.datetime(2026, 10, 5, 18, 15, 30)), "20261005-181530")


class StateMachineTests(unittest.TestCase):
    def test_completes_all_scenarios(self) -> None:
        scenarios = build_desk_scenarios(0.1, 0.5)
        runner, samples = make_runner(scenarios)
        runner.start()
        drive(runner)
        self.assertIs(runner.phase, ValidationPhase.COMPLETE)
        self.assertFalse(runner.is_active)
        self.assertEqual(runner.scenario_index, runner.scenario_count - 1)
        self.assertEqual({s["scenario"] for s in samples}, set(SNAPSHOT_LABELS))

    def test_no_samples_during_prepare(self) -> None:
        scenarios = build_desk_scenarios(1.0, 0.5)
        runner, samples = make_runner(scenarios)
        runner.start()
        runner.update(0.2, None)
        self.assertIs(runner.phase, ValidationPhase.PREPARE)
        self.assertEqual(runner.sample_count, 0)
        self.assertEqual(samples, [])

    def test_capture_samples_at_interval(self) -> None:
        scenarios = build_desk_scenarios(0.0, 0.5)[:1]
        runner, samples = make_runner(scenarios, interval=0.1)
        runner.start()
        drive(runner)
        self.assertEqual(runner.sample_count, 5)
        self.assertEqual(len(samples), 5)

    def test_metrics_none_still_generates_samples(self) -> None:
        scenarios = build_desk_scenarios(0.0, 0.2)[:1]
        runner, samples = make_runner(scenarios, interval=0.1)
        runner.start()
        drive(runner, metrics=None)
        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample["observations"], {})
            self.assertIsNone(sample["side"])

    def test_cancel_during_prepare(self) -> None:
        scenarios = build_desk_scenarios(1.0, 0.5)
        runner, _ = make_runner(scenarios)
        runner.start()
        runner.update(0.1, None)
        runner.cancel()
        self.assertIs(runner.phase, ValidationPhase.CANCELLED)
        runner.update(0.5, None)
        self.assertIs(runner.phase, ValidationPhase.CANCELLED)

    def test_cancel_during_capture(self) -> None:
        scenarios = build_desk_scenarios(0.0, 1.0)
        runner, _ = make_runner(scenarios)
        runner.start()
        runner.update(0.1, None)
        self.assertIs(runner.phase, ValidationPhase.CAPTURE)
        runner.cancel()
        self.assertIs(runner.phase, ValidationPhase.CANCELLED)


class ResetPhaseTests(unittest.TestCase):
    def test_reset_precedes_prepare_and_samples_nothing(self) -> None:
        scenarios = build_desk_scenarios(0.2, 0.3)[:1]
        runner, samples = make_runner(scenarios, reset=0.5)
        runner.start()
        self.assertIs(runner.phase, ValidationPhase.RESET)
        runner.update(0.2, None)
        self.assertIs(runner.phase, ValidationPhase.RESET)
        self.assertEqual(runner.sample_count, 0)
        runner.update(0.4, None)  # 0.6 >= 0.5 -> PREPARE
        self.assertIs(runner.phase, ValidationPhase.PREPARE)
        drive(runner)
        self.assertIs(runner.phase, ValidationPhase.COMPLETE)
        self.assertTrue(samples)

    def test_overlay_lines_have_reset_phase(self) -> None:
        scenarios = build_desk_scenarios(0.2, 0.3)[:1]
        runner, _ = make_runner(scenarios, reset=1.0)
        runner.start()
        text = "\n".join(validation_overlay_lines(runner))
        self.assertIn("POSTURA NORMAL", text)


class MetadataTests(unittest.TestCase):
    def test_run_id_and_test_view_are_constant(self) -> None:
        scenarios = build_desk_scenarios(0.0, 0.3)
        runner, samples = make_runner(scenarios, interval=0.1, run_id="run-A")
        runner.start()
        drive(runner, active_id="switched", active_name="Monitor 1")
        self.assertTrue(samples)
        self.assertEqual({s["run_id"] for s in samples}, {"run-A"})
        self.assertEqual({s["test_view"]["name"] for s in samples}, {"Monitor 2"})
        self.assertEqual({s["active_view"]["name"] for s in samples}, {"Monitor 1"})

    def test_active_view_changed_is_recorded(self) -> None:
        scenarios = build_desk_scenarios(0.0, 0.3)[:1]
        runner, samples = make_runner(scenarios, interval=0.1)
        runner.start()
        runner.update(0.1, None, active_view_id="A", active_view_name="A")  # PREPARE -> CAPTURE
        runner.update(0.1, None, active_view_id="A", active_view_name="A")  # first sample (A)
        guard = 0
        while runner.is_active and guard < 1000:
            runner.update(0.1, None, active_view_id="B", active_view_name="B")
            guard += 1
        self.assertTrue(any(s["active_view_changed"] for s in samples))
        self.assertTrue(any(not s["active_view_changed"] for s in samples))

    def test_sample_metadata_fields(self) -> None:
        scenarios = build_desk_scenarios(0.0, 0.2)[:1]
        runner, samples = make_runner(scenarios, interval=0.1)
        runner.start()
        drive(runner)
        for index, sample in enumerate(samples):
            self.assertEqual(sample["scenario"], SNAPSHOT_LABELS[0])
            self.assertEqual(sample["sample_index"], index)
            self.assertIn("scenario_elapsed_seconds", sample)
            self.assertIn("test_view", sample)
            self.assertEqual(sample["label"], SNAPSHOT_LABELS[0])


class StatusTests(unittest.TestCase):
    def test_progress_and_remaining(self) -> None:
        scenarios = build_desk_scenarios(0.0, 1.0)[:1]
        runner, _ = make_runner(scenarios)
        runner.start()
        runner.update(0.1, None)  # PREPARE -> CAPTURE (prepare_seconds == 0)
        runner.update(0.25, None)
        self.assertAlmostEqual(runner.remaining_seconds, 0.75, places=3)
        self.assertGreater(runner.progress, 0.0)
        self.assertLess(runner.progress, 1.0)
        drive(runner)
        self.assertEqual(runner.progress, 1.0)

    def test_overlay_lines_per_phase(self) -> None:
        scenarios = build_desk_scenarios(1.0, 1.0)
        runner, _ = make_runner(scenarios)
        runner.start()
        prepare_lines = "\n".join(validation_overlay_lines(runner))
        self.assertIn("TEST V2", prepare_lines)
        self.assertIn("Postura normal".upper(), prepare_lines)
        runner.update(1.0, None)  # PREPARE -> CAPTURE
        capture_lines = "\n".join(validation_overlay_lines(runner))
        self.assertIn("CAPTURANDO", capture_lines)


def view_snapshot(
    scenario: str,
    view_name: str,
    *,
    torso_yaw: float = 0.0,
    depth: float = 0.10,
    left_elev: float = 1.0,
    right_elev: float = 1.0,
    active_changed: bool = False,
) -> dict:
    return {
        "scenario": scenario,
        "label": scenario,
        "run_id": "run",
        "test_view": {"id": view_name, "name": view_name},
        "active_view": {"id": view_name, "name": view_name},
        "active_view_changed": active_changed,
        "metrics": {"chin_drop": 0.03},
        "confidence": {"chin_drop": 1.0},
        "view": {"orientation": "center", "confidence": 0.9, "torso_yaw": torso_yaw},
        "shoulders": None,
        "forward": None,
        "observations": {
            "head_depth_ratio": {"value": depth, "confidence": 0.9},
            "left_shoulder_elevation": {"value": left_elev, "confidence": 0.9},
            "right_shoulder_elevation": {"value": right_elev, "confidence": 0.9},
            "torso_yaw": {"value": torso_yaw, "confidence": 0.9},
        },
    }


class ReportTests(unittest.TestCase):
    def _snapshots(self) -> list[dict]:
        return [
            view_snapshot("good", "Monitor 2", torso_yaw=158.0, depth=0.12),
            view_snapshot("head_forward", "Monitor 2", torso_yaw=158.0, depth=0.13),
            view_snapshot("shoulder_left_up", "Monitor 2", left_elev=0.95, right_elev=0.80, active_changed=True),
            view_snapshot("shoulder_right_up", "Monitor 2", left_elev=0.80, right_elev=0.95),
        ]

    def test_v2_only_by_default_excludes_legacy(self) -> None:
        report = build_validation_report("r", self._snapshots(), test_view_name="Monitor 2")
        self.assertIn("metricas: V2 only", report)
        self.assertIn("head_depth_ratio", report)
        self.assertNotIn("chin_drop", report)

    def test_include_legacy_opt_in(self) -> None:
        report = build_validation_report("r", self._snapshots(), include_legacy=True)
        self.assertIn("metricas: V2 + legacy", report)
        self.assertIn("chin_drop", report)

    def test_torso_yaw_is_displayed_folded(self) -> None:
        report = build_validation_report("r", self._snapshots(), test_view_name="Monitor 2")
        self.assertIn("-22", report)  # 158 -> -22
        self.assertNotIn("+158", report)

    def test_coverage_shows_count_over_total(self) -> None:
        report = build_validation_report("r", self._snapshots())
        self.assertIn("(1/1)", report)

    def test_effect_disclaimer_present(self) -> None:
        report = build_validation_report("r", self._snapshots())
        self.assertIn(EFFECT_DISCLAIMER, report)

    def test_lateralization_section_present(self) -> None:
        report = build_validation_report("r", self._snapshots())
        self.assertIn("Lateralization check", report)
        self.assertIn("shoulder_left_up", report)

    def test_active_view_changes_reported(self) -> None:
        report = build_validation_report("r", self._snapshots())
        self.assertIn("active view changes during run: 1", report)

    def test_run_comparison_reports_reproducibility(self) -> None:
        run_a = [view_snapshot("good", "Monitor 2", depth=0.10), view_snapshot("head_down", "Monitor 2", depth=0.12)]
        run_b = [view_snapshot("good", "Monitor 2", depth=0.10), view_snapshot("head_down", "Monitor 2", depth=0.15)]
        text = build_run_comparison("A", run_a, "B", run_b)
        self.assertIn("Reproducibilidad de baseline", text)
        self.assertIn("Reproducibilidad de respuesta", text)
        self.assertIn("head_depth_ratio", text)

    def test_view_difference_lists_views(self) -> None:
        snaps = [view_snapshot("good", "Monitor 1", depth=0.20), view_snapshot("good", "Monitor 2", depth=0.10)]
        text = build_view_difference(snaps)
        self.assertIn("good @ Monitor 1", text)
        self.assertIn("good @ Monitor 2", text)


class SessionTests(unittest.TestCase):
    def test_run_writes_jsonl_and_report(self) -> None:
        scenarios = build_desk_scenarios(0.0, 0.3)[:2]  # good + head_forward
        values = {0: 0.3, 1: 0.6}

        def metrics_for(index: int) -> DetectionMetrics:
            return DetectionMetrics(
                side="right",
                values={},
                view=ViewState(orientation="center", confidence=0.9),
                observations={"head_depth_ratio": MetricObservation(values[index], 0.9)},
            )

        with tempfile.TemporaryDirectory() as tmp:
            session = ValidationSession.start(
                scenarios=scenarios,
                sample_interval_seconds=0.1,
                run_id="20261005-181530",
                test_view_id="view_1",
                test_view_name="Monitor 2",
                directory=Path(tmp),
            )
            guard = 0
            while session.runner.is_active and guard < 10_000:
                session.update(0.1, metrics_for(session.runner.scenario_index))
                guard += 1
            self.assertTrue(session.completed)
            session.close()

            jsonl = Path(tmp) / "20261005-181530.jsonl"
            lines = [line for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), session.runner.sample_count)
            first = json.loads(lines[0])
            self.assertEqual(first["run_id"], "20261005-181530")
            self.assertFalse(first["cancelled"])

            report_path = session.write_report()
            self.assertTrue(report_path.exists())
            self.assertEqual(report_path.name, "20261005-181530-report.txt")
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("run_id: 20261005-181530", report)
            self.assertIn("head_depth_ratio", report)

    def test_build_report_without_samples(self) -> None:
        report = build_validation_report("run-x", [], test_view_name="Monitor 2")
        self.assertIn("run-x", report)
        self.assertIn("No hay muestras", report)


if __name__ == "__main__":
    unittest.main()
