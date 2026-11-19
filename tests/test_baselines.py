"""Tests for Baseline Policies and Episode Execution Runner."""

from fin_audit_gym.baselines import (
    OraclePolicy,
    RandomExplorerPolicy,
    ReconcilerPolicy,
    SpamAttackerPolicy,
    run_policy_episode,
)
from fin_audit_gym.core import FinAuditEnv


def test_random_explorer_policy():
    env = FinAuditEnv()
    policy = RandomExplorerPolicy(explore_steps=5, seed=42)
    result = run_policy_episode(env, policy, seed=42)

    assert result["terminated"] is True
    assert result["steps"] <= 10
    assert result["reward"] <= 0.0
    assert len(result["trajectory"]) > 0


def test_reconciler_policy():
    env = FinAuditEnv()
    policy = ReconcilerPolicy()
    result = run_policy_episode(env, policy, seed=42)

    assert result["terminated"] is True
    assert result["steps"] < 20
    assert result["metrics"]["f1"] > 0.0
    assert result["metrics"]["tp"] >= 1
    assert result["reward"] > -1.0


def test_spam_attacker_policy():
    env = FinAuditEnv()
    policy = SpamAttackerPolicy(num_spam=50)
    result = run_policy_episode(env, policy, seed=42)

    assert result["terminated"] is True
    assert result["steps"] == 1
    assert result["metrics"]["fp"] >= 40
    assert result["reward"] == -1.0


def test_oracle_policy():
    env = FinAuditEnv()
    policy = OraclePolicy()
    result = run_policy_episode(env, policy, seed=42)

    assert result["terminated"] is True
    assert result["metrics"]["f1"] == 1.0
    assert result["metrics"]["precision"] == 1.0
    assert result["metrics"]["recall"] == 1.0
    assert result["reward"] >= 0.95
