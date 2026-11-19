"""Tests for Gymnasium-Compatible Environment and Zero-Leak Sandbox."""

import json
from pathlib import Path

import pytest

from fin_audit_gym.core import (
    EpisodeDoneError,
    FinAuditEnv,
    InvalidActionError,
)
from fin_audit_gym.generator import (
    AnomalyConfig,
    AnomalyInjector,
    GeneratorConfig,
    SyntheticLedgerGenerator,
)


def test_env_reset():
    env = FinAuditEnv()
    obs, info = env.reset(seed=42)

    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    # Check JSON serializability
    json.dumps(obs)
    json.dumps(info)

    # Invariants
    assert "company_name" in obs["manifest"]
    assert "fiscal_year" in obs["manifest"]
    assert "available_actions" in obs
    assert "injected_anomalies_count" not in obs["manifest"]  # Zero leakage!

    assert info["steps_taken"] == 0
    assert info["max_steps"] == 100
    assert "episode_id" in info


def test_query_actions_pagination():
    env = FinAuditEnv()
    env.reset(seed=42)

    # 1. Query Ledger
    action_ledger = {"type": "query_ledger", "page": 1, "page_size": 10}
    obs, reward, term, trunc, info = env.step(action_ledger)
    assert reward == 0.0  # Zero intermediate reward shaping
    assert term is False
    assert trunc is False
    assert info["steps_taken"] == 1
    assert obs["page"] == 1
    assert len(obs["items"]) <= 10
    assert obs["total_pages"] >= 1

    # 2. Query Invoices
    action_inv = {"type": "query_invoices", "status": "unpaid", "page": 1, "page_size": 5}
    obs_inv, _, _, _, _ = env.step(action_inv)
    assert obs_inv["page"] == 1
    assert isinstance(obs_inv["items"], list)

    # 3. Query Payments
    action_pmt = {"type": "query_payments", "page": 1, "page_size": 5}
    obs_pmt, _, _, _, _ = env.step(action_pmt)
    assert obs_pmt["page"] == 1
    assert isinstance(obs_pmt["items"], list)

    # 4. Query FX
    action_fx = {"type": "query_fx", "quote_currency": "EUR", "page": 1, "page_size": 5}
    obs_fx, _, _, _, _ = env.step(action_fx)
    assert obs_fx["page"] == 1
    assert isinstance(obs_fx["items"], list)


def test_calculate_action():
    env = FinAuditEnv()
    env.reset(seed=42)

    # Valid arithmetic
    action_calc = {"type": "calculate", "expression": "100 + 25 * 4"}
    obs, reward, _, _, _ = env.step(action_calc)
    assert obs["result"] == "200"
    assert reward == 0.0

    # Malicious expression handled safely without crashing
    action_attack = {"type": "calculate", "expression": "__import__('os').system('ls')"}
    obs_att, _, _, _, _ = env.step(action_attack)
    assert "error" in obs_att
    assert obs_att["result"] is None


def test_submit_findings_terminates():
    env = FinAuditEnv()
    env.reset(seed=42)

    submission_action = {
        "type": "submit_findings",
        "findings": [
            {
                "finding_id": "F-001",
                "error_type": "duplicate_invoice",
                "entity_id": "INV-2026-0001-DUP",
                "description": "Duplicate invoice found",
                "evidence": {"duplicate": True},
            }
        ],
        "explanation": "Audit report complete.",
    }
    obs, _, term, trunc, _ = env.step(submission_action)
    assert term is True
    assert trunc is False
    assert "submission_received" in obs

    # Subsequent steps raise EpisodeDoneError
    with pytest.raises(EpisodeDoneError):
        env.step({"type": "calculate", "expression": "1 + 1"})


def test_max_steps_truncation():
    env = FinAuditEnv(max_steps=3)
    env.reset(seed=42)

    _, _, term, trunc, _ = env.step({"type": "calculate", "expression": "1 + 1"})
    assert not term and not trunc

    _, _, term, trunc, _ = env.step({"type": "calculate", "expression": "2 + 2"})
    assert not term and not trunc

    _, _, _, trunc, info = env.step({"type": "calculate", "expression": "3 + 3"})
    assert trunc is True
    assert info["steps_taken"] == 3


def test_invalid_action():
    env = FinAuditEnv()
    env.reset(seed=42)

    with pytest.raises(InvalidActionError):
        env.step({"type": "unknown_action"})

    with pytest.raises(InvalidActionError):
        env.step("not a dict")  # type: ignore


def test_security_zero_leak(tmp_path: Path):
    # Generate a world and export to directory with truth.json
    gen_config = GeneratorConfig(seed=99, num_vendors=6, num_transactions=25)
    world = SyntheticLedgerGenerator(gen_config).generate()
    anomaly_config = AnomalyConfig(seed=99)
    mutated_world, truth_findings = AnomalyInjector(anomaly_config).inject(world)
    mutated_world.export_to_directory(tmp_path, truth_findings=truth_findings)

    env = FinAuditEnv(data_dir=tmp_path)
    obs, info = env.reset()

    # Secret finding IDs from ground truth
    secret_finding_ids = [f.finding_id for f in truth_findings]
    assert len(secret_finding_ids) > 0

    def assert_no_leak(data: dict):
        serialized = json.dumps(data)
        for fid in secret_finding_ids:
            assert fid not in serialized

    assert_no_leak(obs)
    assert_no_leak(info)

    # Perform query steps and verify zero leakage in queries
    obs_ledger, _, _, _, _ = env.step({"type": "query_ledger", "page": 1, "page_size": 50})
    assert_no_leak(obs_ledger)

    obs_invoices, _, _, _, _ = env.step({"type": "query_invoices", "page": 1, "page_size": 50})
    assert_no_leak(obs_invoices)
