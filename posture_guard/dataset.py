from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Snapshot side-structures -> canonical metric names used by the analyzer.
VIEW_METRICS = (
    "head_yaw",
    "head_pitch",
    "head_roll",
    "torso_yaw",
    "torso_lateral_lean",
    "neck_roll_delta",
)
SHOULDER_METRICS = {
    "roll": "shoulder_roll",
    "left_elevation": "left_shoulder_elevation",
    "right_elevation": "right_shoulder_elevation",
    "elevation": "shoulder_elevation",
}
FORWARD_METRICS = ("head_forward_ratio", "head_depth_ratio", "torso_forward_angle")

_SEPARABILITY_FLOOR = 1e-6


@dataclass(frozen=True)
class MetricStats:
    count: int
    mean: float
    std: float
    median: float
    mad: float
    minimum: float
    maximum: float
    coverage: float
    confidence: float


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def flatten_snapshot(snapshot: dict[str, Any]) -> dict[str, tuple[float, float]]:
    signals: dict[str, tuple[float, float]] = {}

    legacy_values = snapshot.get("metrics") or {}
    legacy_confidence = snapshot.get("confidence") or {}
    for name, value in legacy_values.items():
        if value is not None:
            signals[name] = (float(value), float(legacy_confidence.get(name, 1.0)))

    view = snapshot.get("view") or {}
    if view:
        view_confidence = float(view.get("confidence") or 0.0)
        for name in VIEW_METRICS:
            value = view.get(name)
            if value is not None:
                signals[name] = (float(value), view_confidence)

    shoulders = snapshot.get("shoulders") or {}
    if shoulders:
        shoulder_confidence = float(shoulders.get("confidence") or 0.0)
        for source, canonical in SHOULDER_METRICS.items():
            value = shoulders.get(source)
            if value is not None:
                signals[canonical] = (float(value), shoulder_confidence)

    forward = snapshot.get("forward") or {}
    if forward:
        forward_confidence = float(forward.get("confidence") or 0.0)
        for name in FORWARD_METRICS:
            value = forward.get(name)
            if value is not None:
                signals[name] = (float(value), forward_confidence)

    observations = snapshot.get("observations") or {}
    for name, payload in observations.items():
        if isinstance(payload, dict) and payload.get("value") is not None:
            signals[name] = (float(payload["value"]), float(payload.get("confidence", 1.0)))

    return signals


def snapshot_view(snapshot: dict[str, Any]) -> str:
    """Detected view bucket for a snapshot: active profile name or orientation.

    The monitor a snapshot belongs to is NEVER a manual label; it is inferred
    from the recorded orientation / active view.
    """
    active_view = snapshot.get("active_view") or {}
    if active_view.get("name"):
        return str(active_view["name"])
    view = snapshot.get("view") or {}
    return str(view.get("orientation") or "unknown")


def summarize_snapshots(
    snapshots: list[dict[str, Any]],
    group_by_view: bool = False,
) -> dict[str, dict[str, MetricStats]]:
    by_label: dict[str, list[dict[str, tuple[float, float]]]] = {}
    for snapshot in snapshots:
        label = str(snapshot.get("label") or "unlabeled")
        if group_by_view:
            label = f"{label} @ {snapshot_view(snapshot)}"
        by_label.setdefault(label, []).append(flatten_snapshot(snapshot))

    summary: dict[str, dict[str, MetricStats]] = {}
    for label, rows in by_label.items():
        metric_names = sorted(set().union(*(row.keys() for row in rows)))
        stats: dict[str, MetricStats] = {}
        for name in metric_names:
            pairs = [row[name] for row in rows if name in row]
            values = [value for value, _ in pairs]
            confidences = [confidence for _, confidence in pairs]
            median = statistics.median(values)
            stats[name] = MetricStats(
                count=len(values),
                mean=statistics.fmean(values),
                std=statistics.pstdev(values) if len(values) > 1 else 0.0,
                median=median,
                mad=statistics.median([abs(value - median) for value in values]),
                minimum=min(values),
                maximum=max(values),
                coverage=len(values) / len(rows),
                confidence=statistics.fmean(confidences) if confidences else 0.0,
            )
        summary[label] = stats
    return summary


def separability(summary: dict[str, dict[str, MetricStats]], metric_name: str) -> float:
    """Label-to-label spread of a metric's mean, normalized by its pooled spread."""
    labels = [stats[metric_name] for stats in summary.values() if metric_name in stats]
    if len(labels) < 2:
        return 0.0
    means = [stats.mean for stats in labels]
    pooled_std = statistics.fmean(stats.std for stats in labels)
    spread = max(means) - min(means)
    return spread / max(pooled_std, _SEPARABILITY_FLOOR)


def format_summary(summary: dict[str, dict[str, MetricStats]]) -> str:
    lines: list[str] = []
    for label in sorted(summary):
        lines.append(f"[{label}]")
        for name, stats in sorted(summary[label].items()):
            lines.append(
                f"  {name:<24} median {stats.median:+.3f}  mad {stats.mad:.3f}  "
                f"mean {stats.mean:+.3f}  std {stats.std:.3f}  "
                f"min {stats.minimum:+.3f}  max {stats.maximum:+.3f}  "
                f"cov {stats.coverage:.0%}  conf {stats.confidence:.0%}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def compare_to_baseline(
    summary: dict[str, dict[str, MetricStats]],
    baseline_label: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """For each scenario label, how far each metric moved from the baseline view."""
    baseline = summary.get(baseline_label)
    if not baseline:
        return {}

    comparison: dict[str, dict[str, dict[str, float]]] = {}
    for label, stats in summary.items():
        if label == baseline_label:
            continue
        rows: dict[str, dict[str, float]] = {}
        for name, metric in stats.items():
            base = baseline.get(name)
            if base is None:
                continue
            delta = metric.median - base.median
            rows[name] = {
                "delta": delta,
                "effect": delta / max(base.mad, _SEPARABILITY_FLOOR),
                "coverage": metric.coverage,
                "confidence": metric.confidence,
            }
        comparison[label] = rows
    return comparison


def format_comparison(comparison: dict[str, dict[str, dict[str, float]]]) -> str:
    lines: list[str] = []
    for label in sorted(comparison):
        lines.append(f"[{label} vs baseline]")
        for name, row in sorted(comparison[label].items()):
            lines.append(
                f"  {name:<24} delta {row['delta']:+.3f}  effect {row['effect']:+.2f}  "
                f"cov {row['coverage']:.0%}  conf {row['confidence']:.0%}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


@dataclass(frozen=True)
class ViewBaselineComparison:
    """A scenario compared against the ``good`` baseline of its OWN view."""

    baseline_label: str
    metrics: dict[str, dict[str, float]]


def split_label(label: str) -> tuple[str, str]:
    """Split a composite ``"scenario @ view"`` label; view is "" when absent."""
    scenario, separator, view = label.partition(" @ ")
    if not separator:
        return label, ""
    return scenario, view


def compare_to_view_baselines(
    summary: dict[str, dict[str, MetricStats]],
    baseline_scenario: str = "good",
) -> dict[str, ViewBaselineComparison]:
    """Compare each scenario against the baseline scenario of its SAME view.

    Looking at a different monitor is a different camera-relative orientation, so
    crossing views (``head_forward @ Monitor 1`` vs ``good @ Monitor 2``) is
    conceptually wrong. When a view has no baseline the scenario is simply
    omitted: another view's baseline is never used as a fallback.
    """
    comparison: dict[str, ViewBaselineComparison] = {}
    for label, stats in summary.items():
        scenario, view = split_label(label)
        if scenario == baseline_scenario:
            continue
        baseline_label = f"{baseline_scenario} @ {view}" if view else baseline_scenario
        baseline = summary.get(baseline_label)
        if baseline is None:
            continue

        rows: dict[str, dict[str, float]] = {}
        for name, metric in stats.items():
            base = baseline.get(name)
            if base is None:
                continue
            delta = metric.median - base.median
            rows[name] = {
                "delta": delta,
                "effect": delta / max(base.mad, _SEPARABILITY_FLOOR),
                "coverage": metric.coverage,
                "confidence": metric.confidence,
            }
        comparison[label] = ViewBaselineComparison(baseline_label, rows)
    return comparison


def format_view_comparison(comparison: dict[str, ViewBaselineComparison]) -> str:
    lines: list[str] = []
    for label in sorted(comparison):
        entry = comparison[label]
        lines.append(f"[{label} vs {entry.baseline_label}]")
        for name, row in sorted(entry.metrics.items()):
            lines.append(
                f"  {name:<24} delta {row['delta']:+.3f}  effect {row['effect']:+.2f}  "
                f"cov {row['coverage']:.0%}  conf {row['confidence']:.0%}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def metric_response_matrix(
    comparison: dict[str, ViewBaselineComparison],
) -> dict[str, dict[str, float]]:
    """Effect of every scenario on every metric, per metric (cross-talk view).

    A metric that reacts to many unrelated scenarios is not specific enough,
    even if its raw separability is high.
    """
    matrix: dict[str, dict[str, float]] = {}
    for label, entry in comparison.items():
        for name, row in entry.metrics.items():
            matrix.setdefault(name, {})[label] = row["effect"]
    return matrix


def format_response_matrix(matrix: dict[str, dict[str, float]]) -> str:
    lines: list[str] = []
    for name in sorted(matrix):
        lines.append(f"{name}:")
        for label in sorted(matrix[name]):
            lines.append(f"  {label:<28} {matrix[name][label]:+.2f} effect")
        lines.append("")
    return "\n".join(lines).rstrip()
