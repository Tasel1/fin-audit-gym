"""Tests for Bipartite Kuhn-Munkres Matcher, Metric Computation, and Anti-Gaming Reward Engine."""

import pytest

from fin_audit_gym.core import FinAuditEnv
from fin_audit_gym.core.schemas import Finding
from fin_audit_gym.evaluator import (
    AuditMetrics,
    BipartiteMatcher,
    RewardEngine,
    calculate_metrics,
)


def test_perfect_auditor():
    gt = [
        Finding(
            finding_id="F-1",
            error_type="duplicate_invoice",
            entity_id="INV-001",
            description="Duplicate invoice",
        ),
        Finding(
            finding_id="F-2",
            error_type="unallocated_payment",
            entity_id="PAY-001",
            description="Unallocated payment",
        ),
        Finding(
            finding_id="F-3",
            error_type="fx_mismatch",
            entity_id="INV-FX-01",
            description="FX mismatch",
        ),
    ]

    preds = [
        Finding(
            finding_id="P-1",
            error_type="duplicate_invoice",
            entity_id="INV-001",
            description="Agent found duplicate invoice",
        ),
        Finding(
            finding_id="P-2",
            error_type="unallocated_payment",
            entity_id="PAY-001",
            description="Agent found unallocated payment",
        ),
        Finding(
            finding_id="P-3",
            error_type="fx_mismatch",
            entity_id="INV-FX-01",
            description="Agent found FX mismatch",
        ),
    ]

    metrics = calculate_metrics(preds, gt)
    assert metrics.tp == 3
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0

    engine = RewardEngine(step_cost=0.002)
    reward = engine.compute_terminal_reward(metrics, steps_taken=10)
    assert reward == pytest.approx(0.98, abs=1e-4)


def test_spam_attack_bipartite_defense():
    # Only 1 true anomaly exists in ground truth
    gt = [
        Finding(
            finding_id="F-1",
            error_type="duplicate_invoice",
            entity_id="INV-001",
            description="Single true duplicate invoice",
        )
    ]

    # Agent spams 100 identical accusations hoping to inflate TP
    preds = [
        Finding(
            finding_id=f"SPAM-{i}",
            error_type="duplicate_invoice",
            entity_id="INV-001",
            description="Spam finding",
        )
        for i in range(100)
    ]

    matcher = BipartiteMatcher()
    matched_pairs, unmatched_p, unmatched_gt = matcher.match(preds, gt)

    # Kuhn-Munkres strictly enforces 1-to-1 matching: exactly 1 match!
    assert len(matched_pairs) == 1
    assert len(unmatched_p) == 99
    assert len(unmatched_gt) == 0

    metrics = calculate_metrics(preds, gt, matcher=matcher)
    assert metrics.tp == 1
    assert metrics.fp == 99
    assert metrics.fn == 0
    assert metrics.precision == 0.01
    assert metrics.recall == 1.0
    assert metrics.f1 < 0.02

    engine = RewardEngine(step_cost=0.002)
    reward = engine.compute_terminal_reward(metrics, steps_taken=10)
    # Reward collapses close to -1.0
    assert reward <= -0.95


def test_clean_book_true_negative():
    # Clean books: 0 true anomalies
    gt = []
    # Honest auditor reports 0 anomalies
    preds = []

    metrics = calculate_metrics(preds, gt)
    assert metrics.tp == 0
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0

    engine = RewardEngine(step_cost=0.002)
    reward = engine.compute_terminal_reward(metrics, steps_taken=5)
    # Maximum reward minus step cost
    assert reward == pytest.approx(0.99, abs=1e-4)


def test_clean_book_false_alarm():
    # Clean books
    gt = []
    # Auditor hallucinates 2 findings
    preds = [
        Finding(
            finding_id="HALLUCINATED-1",
            error_type="duplicate_invoice",
            entity_id="INV-CLEAN-1",
            description="Fake anomaly",
        ),
        Finding(
            finding_id="HALLUCINATED-2",
            error_type="unallocated_payment",
            entity_id="PAY-CLEAN-2",
            description="Fake anomaly 2",
        ),
    ]

    metrics = calculate_metrics(preds, gt)
    assert metrics.tp == 0
    assert metrics.fp == 2
    assert metrics.fn == 0
    assert metrics.precision == 0.0
    assert metrics.f1 == 0.0

    engine = RewardEngine(step_cost=0.002)
    reward = engine.compute_terminal_reward(metrics, steps_taken=5)
    assert reward == -1.0


def test_lazy_auditor_dirty_book():
    # Dirty books with 3 anomalies
    gt = [
        Finding(
            finding_id=f"F-{i}",
            error_type="duplicate_invoice",
            entity_id=f"INV-{i}",
            description="Real anomaly",
        )
        for i in range(3)
    ]
    # Lazy auditor submits nothing
    preds = []

    metrics = calculate_metrics(preds, gt)
    assert metrics.tp == 0
    assert metrics.fp == 0
    assert metrics.fn == 3
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0

    engine = RewardEngine(step_cost=0.002)
    reward = engine.compute_terminal_reward(metrics, steps_taken=1)
    assert reward == -1.0


def test_env_submission_integration():
    env = FinAuditEnv()
    _, _ = env.reset(seed=42)

    # Empty submission
    action = {
        "type": "submit_findings",
        "findings": [],
        "explanation": "No findings submitted",
    }
    _, reward, term, trunc, info_sub = env.step(action)

    assert term is True
    assert trunc is False
    assert "metrics" in info_sub
    metrics = AuditMetrics.model_validate(info_sub["metrics"])
    assert isinstance(metrics.f1, float)
    assert reward <= 0.0  # Dirty book with 0 findings penalized
