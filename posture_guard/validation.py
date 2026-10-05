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
from .orientation import torso_yaw_display

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
    RESET = "reset"
    PREPARE = "prepare"
    CAPTURE = "capture"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


# Human guidance per scenario id. Titles/instructions are the only text; the
# scenario order and ids come from the single source of truth SNAPSHOT_LABELS.
SCENARIO_GUIDES: dict[str, tuple[str, str]] = {
    "good": ("Postura normal", "Sentate como trabajas normalmente y mira tu monitor."),
    "head_forward": (
        "Cabeza adelantada",
        "Adelanta claramente la cabeza hacia la pantalla, manteniendo hombros y torso lo mas quietos posible. "
        "No mires hacia abajo ni encores la espalda. Un cambio claro, no el maximo.",
    ),
    "head_down": ("Mirada hacia abajo", "Baja la mirada sin inclinar deliberadamente el torso."),
    "head_tilt_left": (
        "Cabeza inclinada a la izquierda",
        "Inclina claramente la cabeza hacia tu izquierda, manteniendo hombros y mirada lo mas estables posible.",
    ),
    "head_tilt_right": (
        "Cabeza inclinada a la derecha",
        "Inclina claramente la cabeza hacia tu derecha, manteniendo hombros y mirada lo mas estables posible.",
    ),
    "shoulders_up": ("Ambos hombros elevados", "Eleva ambos hombros manteniendo la cabeza lo mas estable posible."),
    "shoulder_left_up": ("Hombro izquierdo elevado", "Eleva solo tu hombro izquierdo."),
    "shoulder_right_up": ("Hombro derecho elevado", "Eleva solo tu hombro derecho."),
    "head_turn_left": (
        "Giro de cabeza a la izquierda",
        "Gira la cabeza hacia tu izquierda manteniendo los hombros quietos.",
    ),
    "head_turn_right": (
        "Giro de cabeza a la derecha",
        "Gira la cabeza hacia tu derecha manteniendo los hombros quietos.",
    ),
    "head_torso_turn_left": (
        "Giro de cabeza y hombros a la izquierda",
        "Gira cabeza y hombros juntos hacia tu izquierda.",
    ),
    "head_torso_turn_right": (
        "Giro de cabeza y hombros a la derecha",
        "Gira cabeza y hombros juntos hacia tu derecha.",
    ),
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
        reset_seconds: float = 0.0,
    ) -> None:
        if not scenarios:
            raise ValueError("La validacion necesita al menos un escenario.")
        self.scenarios = list(scenarios)
        self.sample_interval_seconds = max(sample_interval_seconds, 1e-6)
        self.reset_seconds = max(reset_seconds, 0.0)
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
        self._last_active_view_id: str | None = None

    # -- status for the UI -------------------------------------------------
    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    @property
    def scenario(self) -> ValidationScenario:
        return self.scenarios[min(self.scenario_index, self.scenario_count - 1)]

    @property
    def is_active(self) -> bool:
        return self.phase in (ValidationPhase.RESET, ValidationPhase.PREPARE, ValidationPhase.CAPTURE)

    @property
    def remaining_seconds(self) -> float:
        if self.phase is ValidationPhase.RESET:
            return max(0.0, self.reset_seconds - self._phase_elapsed)
        if self.phase is ValidationPhase.PREPARE:
            return max(0.0, self.scenario.prepare_seconds - self._phase_elapsed)
        if self.phase is ValidationPhase.CAPTURE:
            return max(0.0, self.scenario.capture_seconds - self._phase_elapsed)
        return 0.0

    def _scenario_total(self) -> float:
        return self.reset_seconds + self.scenario.prepare_seconds + self.scenario.capture_seconds

    @property
    def progress(self) -> float:
        if self.phase is ValidationPhase.COMPLETE:
            return 1.0
        if self.phase is ValidationPhase.CANCELLED or self.scenario_count == 0:
            return 0.0
        scenario = self.scenario
        total = self._scenario_total()
        if total <= 0:
            fraction = 1.0
        elif self.phase is ValidationPhase.RESET:
            fraction = self._phase_elapsed / total
        elif self.phase is ValidationPhase.PREPARE:
            fraction = (self.reset_seconds + self._phase_elapsed) / total
        elif self.phase is ValidationPhase.CAPTURE:
            fraction = (self.reset_seconds + scenario.prepare_seconds + self._phase_elapsed) / total
        else:
            fraction = 0.0
        return min(1.0, (self.scenario_index + fraction) / self.scenario_count)

    def start(self) -> None:
        self.scenario_index = 0
        self._last_active_view_id = None
        self._begin_scenario()

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
        if self.phase is ValidationPhase.RESET:
            self._phase_elapsed += dt
            if self._phase_elapsed >= self.reset_seconds:
                self.phase = ValidationPhase.PREPARE
                self._begin_phase()
        elif self.phase is ValidationPhase.PREPARE:
            self._phase_elapsed += dt
            if self._phase_elapsed >= self.scenario.prepare_seconds:
                self._begin_capture()
        elif self.phase is ValidationPhase.CAPTURE:
            self._phase_elapsed += dt
            self._emit_due_samples(metrics, active_view_id, active_view_name, timestamp)
            if self._phase_elapsed >= self.scenario.capture_seconds:
                self._finish_scenario()

    # -- internals ---------------------------------------------------------
    def _begin_scenario(self) -> None:
        self._phase_elapsed = 0.0
        self.phase = ValidationPhase.RESET if self.reset_seconds > 0 else ValidationPhase.PREPARE

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
        sample["active_view_changed"] = (
            self._last_active_view_id is not None and active_view_id != self._last_active_view_id
        )
        self._last_active_view_id = active_view_id
        sample["cancelled"] = False
        return sample

    def _finish_scenario(self) -> None:
        self.scenario_index += 1
        if self.scenario_index >= self.scenario_count:
            self.phase = ValidationPhase.COMPLETE
            self.scenario_index = self.scenario_count - 1
        else:
            self._begin_scenario()


def validation_overlay_lines(runner: ValidationRunner) -> list[str]:
    """Big on-camera instruction for the current phase (empty when idle)."""
    scenario = runner.scenario
    header = f"TEST V2 · {runner.scenario_index + 1}/{runner.scenario_count}"
    if runner.phase is ValidationPhase.RESET:
        seconds = max(1, int(round(runner.remaining_seconds + 0.5)))
        return [header, "", "VOLVE A TU POSTURA NORMAL", "", f"Preparando siguiente prueba... {seconds}"]
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
        reset_seconds: float = 0.0,
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
            reset_seconds=reset_seconds,
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


EFFECT_DISCLAIMER = (
    "Effect is a within-metric separability indicator. "
    "Do not compare effect magnitude directly across different metrics."
)


def _fold_torso_yaw(row: dict[str, Any]) -> None:
    """Report torso_yaw in its readable (folded) form; JSONL keeps the raw value."""
    view = row.get("view")
    if isinstance(view, dict) and view.get("torso_yaw") is not None:
        view["torso_yaw"] = torso_yaw_display(view["torso_yaw"])
    observations = row.get("observations")
    if isinstance(observations, dict):
        payload = observations.get("torso_yaw")
        if isinstance(payload, dict) and payload.get("value") is not None:
            payload["value"] = torso_yaw_display(payload["value"])


def _as_view_rows(
    snapshots: Iterable[Mapping[str, Any]],
    *,
    include_legacy: bool = False,
) -> list[dict[str, Any]]:
    """Rewrap validation snapshots so the analyzer groups by ``test_view``.

    The scenario label stays; the run's ``test_view`` replaces the changing
    ``active_view`` as the grouping identity. Legacy V1 metrics are dropped
    unless ``include_legacy`` is set, and torso_yaw is shown folded.
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
        if not include_legacy:
            row["metrics"] = {}
            row["confidence"] = {}
        _fold_torso_yaw(row)
        rows.append(row)
    return rows


def _active_view_change_lines(snapshots: Sequence[Mapping[str, Any]]) -> list[str]:
    total = sum(1 for snap in snapshots if snap.get("active_view_changed"))
    lines = [f"active view changes during run: {total}"]
    per_scenario: dict[str, int] = {}
    for snap in snapshots:
        if snap.get("active_view_changed"):
            scenario = str(snap.get("scenario") or "?")
            per_scenario[scenario] = per_scenario.get(scenario, 0) + 1
    for scenario in sorted(per_scenario):
        lines.append(f"  {scenario:<24} {per_scenario[scenario]}")
    return lines


def _lateralization_lines(comparison: Mapping[str, Any]) -> list[str]:
    """LEFT == LEFT / RIGHT == RIGHT sanity: which side moves for each shoulder test."""
    lines: list[str] = []
    tests = ("shoulder_left_up", "shoulder_right_up", "shoulders_up")
    for label in sorted(comparison):
        scenario = str(label).split(" @ ", 1)[0]
        if scenario not in tests:
            continue
        entry = comparison[label]
        metrics = entry.metrics
        left = metrics.get("left_shoulder_elevation")
        right = metrics.get("right_shoulder_elevation")
        lines.append(f"  [{label}]")
        if left is not None:
            lines.append(f"    left_shoulder_elevation   delta {left['delta']:+.3f}")
        if right is not None:
            lines.append(f"    right_shoulder_elevation  delta {right['delta']:+.3f}")
    return lines


def build_validation_report(
    run_id: str,
    snapshots: Sequence[Mapping[str, Any]],
    *,
    test_view_name: str = "",
    baseline_scenario: str = DEFAULT_BASELINE_SCENARIO,
    include_legacy: bool = False,
) -> str:
    """Data-only report: no PASS/FAIL, no automatic verdict.

    Reuses the snapshot analyzer; the run's test view is the baseline identity.
    Standard output is V2-only; pass ``include_legacy`` for the V1 diagnostics.
    """
    lines = [
        "Reporte de validacion V2",
        f"run_id: {run_id}",
        f"vista de test: {test_view_name or '-'}",
        f"escenario baseline: {baseline_scenario}",
        f"muestras: {len(snapshots)}",
        f"metricas: {'V2 + legacy' if include_legacy else 'V2 only'}",
        "",
    ]
    if not snapshots:
        lines.append("No hay muestras en esta corrida.")
        return "\n".join(lines)

    summary = summarize_snapshots(_as_view_rows(snapshots, include_legacy=include_legacy), group_by_view=True)
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
        lines.append(f"({EFFECT_DISCLAIMER})")
        lines.append("")
        lines.append("Matriz de respuesta por escenario (efecto por metrica; ojo cross-talk):")
        lines.append(format_response_matrix(metric_response_matrix(comparison)))
        lines.append("")
        lines.append("Lateralization check (verifica LEFT == LEFT, RIGHT == RIGHT):")
        lateral = _lateralization_lines(comparison)
        lines.extend(lateral or ["  (sin escenarios de hombros en esta corrida)"])
    else:
        lines.append("")
        lines.append(f"Sin baseline '{baseline_scenario}' para la vista de test.")

    lines.append("")
    lines.extend(_active_view_change_lines(snapshots))
    return "\n".join(lines)


# -- cross-run comparison --------------------------------------------------


def _run_summary(snapshots: Sequence[Mapping[str, Any]], include_legacy: bool):
    return summarize_snapshots(_as_view_rows(snapshots, include_legacy=include_legacy), group_by_view=True)


def build_run_comparison(
    run_a_id: str,
    snapshots_a: Sequence[Mapping[str, Any]],
    run_b_id: str,
    snapshots_b: Sequence[Mapping[str, Any]],
    *,
    baseline_scenario: str = DEFAULT_BASELINE_SCENARIO,
    include_legacy: bool = False,
) -> str:
    """Compare two runs: baseline reproducibility and response reproducibility.

    Response (delta scenario-baseline) matters more than absolute values.
    """
    summary_a = _run_summary(snapshots_a, include_legacy)
    summary_b = _run_summary(snapshots_b, include_legacy)
    lines = [
        f"Comparacion de runs: {run_a_id} vs {run_b_id}",
        f"baseline: {baseline_scenario}",
        "",
        "Reproducibilidad de baseline (mediana A vs B):",
    ]
    for label in sorted(summary_a):
        if label.split(" @ ", 1)[0] != baseline_scenario or label not in summary_b:
            continue
        lines.append(f"  [{label}]")
        for name in sorted(set(summary_a[label]) & set(summary_b[label])):
            a = summary_a[label][name]
            b = summary_b[label][name]
            lines.append(
                f"    {name:<24} A {a.median:+.3f}  B {b.median:+.3f}  d {b.median - a.median:+.3f}  "
                f"covA {a.coverage:.0%}({a.count}/{a.total})  covB {b.coverage:.0%}({b.count}/{b.total})"
            )

    comparison_a = compare_to_view_baselines(summary_a, baseline_scenario)
    comparison_b = compare_to_view_baselines(summary_b, baseline_scenario)
    lines.append("")
    lines.append("Reproducibilidad de respuesta (delta escenario - baseline):")
    for label in sorted(comparison_a):
        if label not in comparison_b:
            continue
        rows_a = comparison_a[label].metrics
        rows_b = comparison_b[label].metrics
        lines.append(f"  [{label}]")
        for name in sorted(set(rows_a) & set(rows_b)):
            da = rows_a[name]["delta"]
            db = rows_b[name]["delta"]
            lines.append(f"    {name:<24} deltaA {da:+.3f}  deltaB {db:+.3f}  d {db - da:+.3f}")
    return "\n".join(lines)


def build_view_difference(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    scenario: str = DEFAULT_BASELINE_SCENARIO,
    include_legacy: bool = False,
) -> str:
    """Per-metric medians for one scenario across every view in a run."""
    summary = _run_summary(snapshots, include_legacy)
    labels = [label for label in summary if label.split(" @ ", 1)[0] == scenario]
    lines = [f"Diferencia entre vistas para '{scenario}' (mediana por vista):"]
    if not labels:
        lines.append("  (sin datos)")
        return "\n".join(lines)
    for label in sorted(labels):
        lines.append(f"  [{label}]")
        for name in sorted(summary[label]):
            stats = summary[label][name]
            lines.append(
                f"    {name:<24} median {stats.median:+.3f}  "
                f"cov {stats.coverage:.0%} ({stats.count}/{stats.total})  conf {stats.confidence:.0%}"
            )
    return "\n".join(lines)
