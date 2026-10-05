from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .issues import PostureIssue, issue_for_metric
from .metrics import min_confidence_for, normalize_observation, resolve_baseline, spec_for
from .models import CalibrationProfile, MetricBaseline, MetricObservation, MetricThreshold


@dataclass
class IssueEvaluation:
    severity_by_issue: dict[PostureIssue, float] = field(default_factory=dict)
    risk_score: float = 0.0
    dominant_issue: PostureIssue | None = None
    dominant_severity: float = 0.0


def metric_severity(value: float, threshold: MetricThreshold) -> float:
    """Normalized severity of a metric against its threshold; 0.0 means fine."""
    if threshold.mode == "directional":
        signed_distance = (value - threshold.threshold) * (threshold.direction or 1)
        if signed_distance <= 0:
            return 0.0
        return max(0.0, signed_distance / max(threshold.margin, 1e-6))

    distance = abs(value - threshold.threshold)
    if distance <= threshold.margin:
        return 0.0
    return max(0.0, (distance - threshold.margin) / max(threshold.margin, 1e-6))


def evaluate_issues(
    values: dict[str, float],
    confidence: dict[str, float],
    profile: CalibrationProfile | None,
    min_confidence: float = 0.0,
) -> IssueEvaluation:
    """Weighted per-issue risk from the metrics that are present and reliable."""
    evaluation = IssueEvaluation()
    if profile is None:
        return evaluation

    for name, value in values.items():
        threshold = profile.thresholds.get(name)
        if threshold is None or threshold.mode == "disabled":
            continue
        metric_confidence = confidence.get(name, 1.0)
        if metric_confidence < min_confidence:
            continue

        severity = metric_severity(value, threshold)
        if severity <= 0:
            continue
        issue = issue_for_metric(name)
        if issue is None:
            continue

        weighted = severity * max(threshold.weight, 0.0) * metric_confidence
        evaluation.severity_by_issue[issue] = max(evaluation.severity_by_issue.get(issue, 0.0), weighted)

    # Correlated metrics of the same issue must not inflate the score: risk is
    # the sum of one severity per issue, not per metric.
    evaluation.risk_score = sum(evaluation.severity_by_issue.values())
    if evaluation.severity_by_issue:
        evaluation.dominant_issue = max(evaluation.severity_by_issue, key=evaluation.severity_by_issue.get)
        evaluation.dominant_severity = evaluation.severity_by_issue[evaluation.dominant_issue]
    return evaluation


def evaluate_v2_issues(
    observations: dict[str, MetricObservation],
    view_baselines: dict[str, MetricBaseline],
    global_baselines: dict[str, MetricBaseline],
    config: Config,
) -> IssueEvaluation:
    """Issue risk from observations normalized against their calibrated baselines.

    Calibration only provides ``center``/``spread``; the semantic direction and
    issue mapping come from ``MetricSpec``. Signals with ``risk_enabled=False``
    are skipped here even when they have an issue: they stay observed and
    calibrated, but do not move ``risk_score``/``dominant_issue``/streaks.
    """
    evaluation = IssueEvaluation()
    for name, observation in observations.items():
        spec = spec_for(name)
        if spec is None or spec.issue is None or not spec.risk_enabled:
            continue
        baseline = resolve_baseline(name, spec, view_baselines, global_baselines)
        margin = config.default_margins.get(name, 0.03)
        gate = min_confidence_for(spec, config)
        severity = normalize_observation(observation, baseline, spec, margin, gate)
        if severity <= 0:
            continue

        weight = config.default_weights.get(name, spec.default_weight)
        weighted = severity * max(weight, 0.0) * observation.confidence
        evaluation.severity_by_issue[spec.issue] = max(
            evaluation.severity_by_issue.get(spec.issue, 0.0), weighted
        )

    evaluation.risk_score = sum(evaluation.severity_by_issue.values())
    if evaluation.severity_by_issue:
        evaluation.dominant_issue = max(evaluation.severity_by_issue, key=evaluation.severity_by_issue.get)
        evaluation.dominant_severity = evaluation.severity_by_issue[evaluation.dominant_issue]
    return evaluation
