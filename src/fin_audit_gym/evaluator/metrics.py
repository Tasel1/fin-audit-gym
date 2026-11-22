"""Evaluation metrics computation with clean-book handling."""

from pydantic import BaseModel, ConfigDict, Field

from fin_audit_gym.core.schemas import Finding
from fin_audit_gym.evaluator.matcher import BipartiteMatcher


class AuditMetrics(BaseModel):
    """Auditing accuracy metrics model."""

    model_config = ConfigDict(extra="forbid")

    tp: int = Field(ge=0, description="True Positive count")
    fp: int = Field(ge=0, description="False Positive count")
    fn: int = Field(ge=0, description="False Negative count")
    precision: float = Field(ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(ge=0.0, le=1.0, description="Recall score")
    f1: float = Field(ge=0.0, le=1.0, description="F1 harmonic mean score")


def calculate_metrics(
    predictions: list[Finding],
    ground_truth: list[Finding],
    matcher: BipartiteMatcher | None = None,
) -> AuditMetrics:
    """Calculate TP, FP, FN, Precision, Recall, and F1 with clean-book and 1-to-1 matching rules."""
    active_matcher = matcher or BipartiteMatcher()

    # Clean books rule (Section 05: if truth empty and findings exist, F1=0, P=0, R=0)
    if len(ground_truth) == 0:
        if len(predictions) == 0:
            return AuditMetrics(tp=0, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)
        return AuditMetrics(tp=0, fp=len(predictions), fn=0, precision=0.0, recall=0.0, f1=0.0)

    # Dirty books: 1-to-1 Kuhn-Munkres bipartite matching
    matched_pairs, _, _ = active_matcher.match(predictions, ground_truth)

    tp = len(matched_pairs)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return AuditMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )
