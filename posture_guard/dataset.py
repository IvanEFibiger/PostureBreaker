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
FORWARD_METRICS = ("head_forward_ratio", "torso_forward_angle")

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


def summarize_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, dict[str, MetricStats]]:
    by_label: dict[str, list[dict[str, tuple[float, float]]]] = {}
    for snapshot in snapshots:
        label = str(snapshot.get("label") or "unlabeled")
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
