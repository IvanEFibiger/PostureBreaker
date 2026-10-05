from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .dataset import (
    compare_to_view_baselines,
    format_response_matrix,
    format_summary,
    format_view_comparison,
    load_snapshots,
    metric_response_matrix,
    separability,
    summarize_snapshots,
)
from .debug import SNAPSHOT_LABELS, build_snapshot

# One guided run stays observe-only: it produces a dataset, never a verdict.
DEFAULT_BASELINE_SCENARIO = "good"


@dataclass(frozen=True)
class ValidationScenario:
    id: str
    title: str
    instruction: str
    prepare_seconds: float
    capture_seconds: float


class ValidationPhase(StrEnum):
    IDLE = "idle"
    PREPARE = "prepare"
    CAPTURE = "capture"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


# Human guidance per scenario id. Titles/instructions are the only text; the
# scenario order and ids come from the single source of truth SNAPSHOT_LABELS.
SCENARIO_GUIDES: dict[str, tuple[str, str]] = {
    "good": ("Postura normal", "Sentate como trabajas normalmente y mira tu monitor."),
    "head_forward": ("Cabeza adelantada", "Adelanta la cabeza hacia la pantalla sin mover los hombros."),
    "head_down": ("Mirada hacia abajo", "Baja la mirada sin inclinar deliberadamente el torso."),
    "head_tilt_left": ("Cabeza inclinada a la izquierda", "Inclina la cabeza hacia la izquierda sin subir los hombros."),
    "head_tilt_right": ("Cabeza inclinada a la derecha", "Inclina la cabeza hacia la derecha sin subir los hombros."),
    "shoulders_up": ("Ambos hombros elevados", "Eleva ambos hombros manteniendo la cabeza lo mas estable posible."),
    "shoulder_left_up": ("Hombro izquierdo elevado", "Eleva solo el hombro izquierdo."),
    "shoulder_right_up": ("Hombro derecho elevado", "Eleva solo el hombro derecho."),
    "head_turn_only": ("Giro solo de cabeza", "Gira la cabeza hacia un lado manteniendo los hombros quietos."),
    "head_torso_turn": ("Giro de cabeza y hombros", "Gira cabeza y hombros juntos hacia el mismo lado."),
}


def build_desk_scenarios(prepare_seconds: float, capture_seconds: float) -> list[ValidationScenario]:
    """Desk-core protocol scenarios in SNAPSHOT_LABELS order."""
    scenarios: list[ValidationScenario] = []
    for scenario_id in SNAPSHOT_LABELS:
        title, instruction = SCENARIO_GUIDES.get(scenario_id, (scenario_id, ""))
        scenarios.append(
            ValidationScenario(
                id=scenario_id,
                title=title,
                instruction=instruction,
                prepare_seconds=prepare_seconds,
                capture_seconds=capture_seconds,
            )
        )
    return scenarios


def make_run_id(moment: dt.datetime | None = None) -> str:
    return (moment or dt.datetime.now()).strftime("%Y%m%d-%H%M%S")


class ValidationRunner:
    """Guided capture state machine, independent of camera/UI/storage.

    The caller feeds each processed frame via ``update`` and provides a
    ``sample_sink`` that persists one snapshot dict per scheduled interval. The
    same ``run_id`` and ``test_view`` stay fixed for the whole run; only the
    observed ``active_view`` may drift (recorded per sample for selector
    robustness).
    """

    def __init__(
        self,
        scenarios: Sequence[ValidationScenario],
        *,
        sample_interval_seconds: float,
        run_id: str,
        test_view_id: str | None,
        test_view_name: str,
        sample_sink: Callable[[dict[str, Any]], None],
    ) -> None:
        if not scenarios:
            raise ValueError("La validacion necesita al menos un escenario.")
        self.scenarios = list(scenarios)
        self.sample_interval_seconds = max(sample_interval_seconds, 1e-6)
        self.run_id = run_id
        self.test_view_id = test_view_id
        self.test_view_name = test_view_name
        self._sample_sink = sample_sink

        self.phase = ValidationPhase.IDLE
        self.scenario_index = 0
        self.sample_count = 0
        self._phase_elapsed = 0.0
        self._scenario_sample_index = 0
        self._next_sample_at = 0.0

    # -- status for the UI -------------------------------------------------
    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    @property
    def scenario(self) -> ValidationScenario:
        return self.scenarios[min(self.scenario_index, self.scenario_count - 1)]

    @property
    def is_active(self) -> bool:
        return self.phase in (ValidationPhase.PREPARE, ValidationPhase.CAPTURE)

    @property
    def remaining_seconds(self) -> float:
        if self.phase is ValidationPhase.PREPARE:
            return max(0.0, self.scenario.prepare_seconds - self._phase_elapsed)
        if self.phase is ValidationPhase.CAPTURE:
            return max(0.0, self.scenario.capture_seconds - self._phase_elapsed)
        return 0.0

    @property
    def progress(self) -> float:
        if self.phase is ValidationPhase.COMPLETE:
            return 1.0
        if self.phase is ValidationPhase.CANCELLED or self.scenario_count == 0:
            return 0.0
        scenario = self.scenario
        total = scenario.prepare_seconds + scenario.capture_seconds
        if total <= 0:
            fraction = 1.0
        elif self.phase is ValidationPhase.PREPARE:
            fraction = self._phase_elapsed / total
        elif self.phase is ValidationPhase.CAPTURE:
            fraction = (scenario.prepare_seconds + self._phase_elapsed) / total
        else:
            fraction = 0.0
        return min(1.0, (self.scenario_index + fraction) / self.scenario_count)

    def start(self) -> None:
        self.phase = ValidationPhase.PREPARE
        self.scenario_index = 0
        self._begin_phase()

    def cancel(self) -> None:
        if self.is_active:
            self.phase = ValidationPhase.CANCELLED

    def update(
        self,
        dt: float,
        metrics: Any,
        *,
        active_view_id: str | None = None,
        active_view_name: str | None = None,
        timestamp: dt.datetime | None = None,
    ) -> None:
        """Advance the state machine using ``dt`` (never ``sleep``)."""
        dt = max(0.0, dt)
        if self.phase is ValidationPhase.PREPARE:
            self._phase_elapsed += dt
            if self._phase_elapsed >= self.scenario.prepare_seconds:
                self._begin_capture()
        elif self.phase is ValidationPhase.CAPTURE:
            self._phase_elapsed += dt
            self._emit_due_samples(metrics, active_view_id, active_view_name, timestamp)
            if self._phase_elapsed >= self.scenario.capture_seconds:
                self._finish_scenario()

    # -- internals ---------------------------------------------------------
    def _begin_phase(self) -> None:
        self._phase_elapsed = 0.0

    def _begin_capture(self) -> None:
        self.phase = ValidationPhase.CAPTURE
        self._begin_phase()
        self._scenario_sample_index = 0
        self._next_sample_at = 0.0

    def _emit_due_samples(
        self,
        metrics: Any,
        active_view_id: str | None,
        active_view_name: str | None,
        timestamp: dt.datetime | None,
    ) -> None:
        capture_seconds = self.scenario.capture_seconds
        while self._next_sample_at <= self._phase_elapsed and self._next_sample_at < capture_seconds:
            self._sample_sink(
                self._build_sample(metrics, active_view_id, active_view_name, timestamp, self._next_sample_at)
            )
            self.sample_count += 1
            self._scenario_sample_index += 1
            self._next_sample_at += self.sample_interval_seconds

    def _build_sample(
        self,
        metrics: Any,
        active_view_id: str | None,
        active_view_name: str | None,
        timestamp: dt.datetime | None,
        scenario_elapsed: float,
    ) -> dict[str, Any]:
        sample = build_snapshot(
            metrics,
            self.scenario.id,
            timestamp=timestamp,
            active_view_id=active_view_id,
            active_view_name=active_view_name,
            run_id=self.run_id,
        )
        sample["scenario"] = self.scenario.id
        sample["sample_index"] = self._scenario_sample_index
        sample["scenario_elapsed_seconds"] = round(scenario_elapsed, 3)
        # test_view is the run identity; active_view may drift during turns.
        sample["test_view"] = {"id": self.test_view_id, "name": self.test_view_name}
        sample["cancelled"] = False
        return sample

    def _finish_scenario(self) -> None:
        self.scenario_index += 1
        if self.scenario_index >= self.scenario_count:
            self.phase = ValidationPhase.COMPLETE
            self.scenario_index = self.scenario_count - 1
        else:
            self.phase = ValidationPhase.PREPARE
            self._begin_phase()


def validation_overlay_lines(runner: ValidationRunner) -> list[str]:
    """Big on-camera instruction for the current phase (empty when idle)."""
    scenario = runner.scenario
    header = f"TEST V2 · {runner.scenario_index + 1}/{runner.scenario_count}"
    if runner.phase is ValidationPhase.PREPARE:
        seconds = max(1, int(round(runner.remaining_seconds + 0.5)))
        return [header, "", scenario.title.upper(), "", scenario.instruction, f"Comenzamos en {seconds}..."]
    if runner.phase is ValidationPhase.CAPTURE:
        seconds = max(0, int(round(runner.remaining_seconds + 0.5)))
        return ["CAPTURANDO", "", "Mantene la posicion", "", f"{seconds} s", f"{runner.sample_count} muestras"]
    return []


class ValidationSession:
    """Owns one run: runner + incremental JSONL file + report file.

    No SQLite, camera, MediaPipe or UI dependency, so it is testable on its own
    and never contaminates the productive analytics store.
    """

    def __init__(self, runner: ValidationRunner, handle: Any, path: Path) -> None:
        self.runner = runner
        self._handle = handle
        self.path = path

    @classmethod
    def start(
        cls,
        *,
        scenarios: Sequence[ValidationScenario],
        sample_interval_seconds: float,
        run_id: str,
        test_view_id: str | None,
        test_view_name: str,
        directory: Path,
        open_fn: Callable[..., Any] = open,
    ) -> ValidationSession:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{run_id}.jsonl"
        handle = open_fn(path, "a", encoding="utf-8")

        def sink(sample: dict[str, Any]) -> None:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            handle.flush()

        runner = ValidationRunner(
            scenarios,
            sample_interval_seconds=sample_interval_seconds,
            run_id=run_id,
            test_view_id=test_view_id,
            test_view_name=test_view_name,
            sample_sink=sink,
        )
        runner.start()
        return cls(runner, handle, path)

    def update(
        self,
        dt: float,
        metrics: Any,
        *,
        active_view_id: str | None = None,
        active_view_name: str | None = None,
        timestamp: dt.datetime | None = None,
    ) -> None:
        self.runner.update(
            dt,
            metrics,
            active_view_id=active_view_id,
            active_view_name=active_view_name,
            timestamp=timestamp,
        )

    @property
    def completed(self) -> bool:
        return self.runner.phase is ValidationPhase.COMPLETE

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            pass

    def report_path(self) -> Path:
        return self.path.with_name(f"{self.path.stem}-report.txt")

    def build_report(self) -> str:
        snapshots = load_snapshots(self.path)
        return build_validation_report(
            self.runner.run_id,
            snapshots,
            test_view_name=self.runner.test_view_name,
        )

    def write_report(self) -> Path:
        path = self.report_path()
        path.write_text(self.build_report(), encoding="utf-8")
        return path


# -- report ----------------------------------------------------------------


def _as_view_rows(snapshots: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rewrap validation snapshots so the analyzer groups by ``test_view``.

    The scenario label stays; the run's ``test_view`` replaces the changing
    ``active_view`` as the grouping identity.
    """
    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        row = dict(snap)
        row["label"] = str(snap.get("scenario") or snap.get("label") or "unlabeled")
        view = snap.get("test_view") or snap.get("active_view") or {}
        row["active_view"] = {
            "id": view.get("id"),
            "name": view.get("name") or "unknown",
        }
        rows.append(row)
    return rows


def build_validation_report(
    run_id: str,
    snapshots: Sequence[Mapping[str, Any]],
    *,
    test_view_name: str = "",
    baseline_scenario: str = DEFAULT_BASELINE_SCENARIO,
) -> str:
    """Data-only report: no PASS/FAIL, no automatic verdict.

    Reuses the snapshot analyzer; the run's test view is the baseline identity.
    """
    lines = [
        "Reporte de validacion V2",
        f"run_id: {run_id}",
        f"vista de test: {test_view_name or '-'}",
        f"escenario baseline: {baseline_scenario}",
        f"muestras: {len(snapshots)}",
        "",
    ]
    if not snapshots:
        lines.append("No hay muestras en esta corrida.")
        return "\n".join(lines)

    summary = summarize_snapshots(_as_view_rows(snapshots), group_by_view=True)
    lines.append("Resumen por escenario @ vista (test_view):")
    lines.append(format_summary(summary))

    metric_names = sorted({name for stats in summary.values() for name in stats})
    if metric_names:
        lines.append("")
        lines.append("Separabilidad por metrica (mayor = discrimina mejor):")
        for name in metric_names:
            lines.append(f"  {name:<24} {separability(summary, name):.2f}")

    comparison = compare_to_view_baselines(summary, baseline_scenario)
    if comparison:
        lines.append("")
        lines.append(f"Comparacion por vista contra '{baseline_scenario}' de la misma vista:")
        lines.append(format_view_comparison(comparison))
        lines.append("")
        lines.append("Matriz de respuesta por escenario (efecto por metrica; ojo cross-talk):")
        lines.append(format_response_matrix(metric_response_matrix(comparison)))
    else:
        lines.append("")
        lines.append(f"Sin baseline '{baseline_scenario}' para la vista de test.")

    return "\n".join(lines)
