"""Tests for Section 03 & 04 Canonical Specifications.

Verifies:
1. Canonical action interface: list_records, get_record, get_document, calculator, submit.
2. Stable cursor pagination with max 50 limit.
3. policy.json presence, schema, and initial observation budget/policy fields.
4. Canonical findings structure: type, affected_ids, currency, evidence_ids.
5. Ledger, invoice, and payment canonical headers and fields.
"""

from fin_audit_gym.core import FinAuditEnv


def test_initial_observation_section_04():
    """Verify reset() observation contains task, manifest, policy, and remaining_budget (Section 33)."""
    env = FinAuditEnv()
    obs, _info = env.reset(seed=42)

    assert "task" in obs
    assert "manifest" in obs
    assert "policy" in obs
    assert "remaining_budget" in obs
    assert obs["remaining_budget"] == 100
    assert "available_actions" in obs

    # Policy fields
    assert "fx_date_policy" in obs["policy"]
    assert "period_start" in obs["policy"]
    assert "period_end" in obs["policy"]
    assert "timezone" in obs["policy"]
    assert "rounding_rule" in obs["policy"]


def test_list_records_action():
    """Verify list_records table pagination with stable cursor and max 50 limit (Section 36)."""
    env = FinAuditEnv()
    env.reset(seed=42)

    # 1. List ledger records
    action = {"action": "list_records", "table": "ledger", "cursor": "0", "limit": 15}
    obs, reward, _term, _trunc, _info = env.step(action)
    assert reward == 0.0
    assert obs["action"] == "list_records"
    assert obs["table"] == "ledger"
    assert len(obs["items"]) == 15
    assert obs["has_more"] is True
    assert obs["next_cursor"] == "15"

    # Next page using cursor
    action2 = {
        "action": "list_records",
        "table": "ledger",
        "cursor": obs["next_cursor"],
        "limit": 15,
    }
    obs2, _, _, _, _ = env.step(action2)
    assert obs2["cursor"] == "15"
    assert len(obs2["items"]) == 15

    # 2. Clamped limit test: requesting 100 is clamped to 50
    action_clamp = {"action": "list_records", "table": "invoices", "limit": 100}
    obs_clamp, _, _, _, _ = env.step(action_clamp)
    assert obs_clamp["limit"] == 50
    assert len(obs_clamp["items"]) <= 50


def test_get_record_action():
    """Verify get_record retrieves specific items by primary table ID (Section 36)."""
    env = FinAuditEnv()
    env.reset(seed=42)

    # Retrieve first invoice ID
    list_obs, _, _, _, _ = env.step({"action": "list_records", "table": "invoices", "limit": 1})
    inv_id = list_obs["items"][0]["invoice_id"]

    # Fetch record
    get_obs, _, _, _, _ = env.step({"action": "get_record", "table": "invoices", "id": inv_id})
    assert "item" in get_obs
    assert get_obs["item"]["invoice_id"] == inv_id

    # Non-existent record returns bounded error
    err_obs, _, _, _, _ = env.step(
        {"action": "get_record", "table": "invoices", "id": "NON-EXISTENT"}
    )
    assert "error" in err_obs


def test_get_document_action():
    """Verify get_document retrieves invoices or payments by doc_id (Section 36)."""
    env = FinAuditEnv()
    env.reset(seed=42)

    # List payments to get doc_id
    list_obs, _, _, _, _ = env.step({"action": "list_records", "table": "payments", "limit": 1})
    doc_id = list_obs["items"][0]["doc_id"]

    # Fetch document
    doc_obs, _, _, _, _ = env.step({"action": "get_document", "doc_id": doc_id})
    assert "document" in doc_obs
    assert doc_obs["document"]["document_type"] == "payment"
    assert doc_obs["document"]["data"]["doc_id"] == doc_id


def test_calculator_alias():
    """Verify calculator(expression) executes safe arithmetic (Section 37)."""
    env = FinAuditEnv()
    env.reset(seed=42)

    calc_obs, _, _, _, _ = env.step({"action": "calculator", "expression": "1000 * 1.085 + 250"})
    assert float(calc_obs["result"]) == 1335.0


def test_canonical_submit_format():
    """Verify submit accepts canonical findings.json structure with Section 05 fields."""
    env = FinAuditEnv()
    env.reset(seed=42)

    findings_payload = {
        "episode_id": "EP-2026-1001",
        "findings": [
            {
                "type": "duplicate_invoice",
                "affected_ids": ["INV-2026-0012-DUP", "INV-2026-0012"],
                "claimed_amount_minor": 109828,
                "currency": "USD",
                "evidence_ids": ["INV-2026-0012-DUP", "INV-2026-0012"],
                "explanation": "Duplicate billing detected for identical line items.",
            }
        ],
    }

    obs, _reward, term, _trunc, info = env.step({"action": "submit", "findings": findings_payload})
    assert term is True
    assert obs["submission_received"] is True
    assert "metrics" in info
