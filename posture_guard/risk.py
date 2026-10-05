from __future__ import annotations

from dataclasses import dataclass, field

from .issues import PostureIssue, issue_for_metric
from .models import CalibrationProfile, MetricThreshold


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
        evaluation.risk_score += weighted
        evaluation.severity_by_issue[issue] = max(evaluation.severity_by_issue.get(issue, 0.0), weighted)

    if evaluation.severity_by_issue:
        evaluation.dominant_issue = max(evaluation.severity_by_issue, key=evaluation.severity_by_issue.get)
        evaluation.dominant_severity = evaluation.severity_by_issue[evaluation.dominant_issue]
    return evaluation
