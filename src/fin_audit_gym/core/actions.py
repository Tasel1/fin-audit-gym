"""Action definitions, constants, and validation for FinAuditEnv."""

from enum import Enum
from typing import Any

from fin_audit_gym.core.exceptions import InvalidActionError


class ActionType(str, Enum):
    """Enumeration of all valid action types in FinAuditEnv (Section 04 + legacy)."""

    # Section 04 Canonical Actions
    LIST_RECORDS = "list_records"
    GET_RECORD = "get_record"
    GET_DOCUMENT = "get_document"
    CALCULATOR = "calculator"
    SUBMIT = "submit"

    # Compatibility & Filtered Query Actions
    QUERY_LEDGER = "query_ledger"
    QUERY_INVOICES = "query_invoices"
    QUERY_PAYMENTS = "query_payments"
    QUERY_FX = "query_fx"
    CALCULATE = "calculate"
    SUBMIT_FINDINGS = "submit_findings"


VALID_ACTION_TYPES = {item.value for item in ActionType}

# Normalized action routing map
ACTION_ALIASES: dict[str, str] = {
    "list_records": ActionType.LIST_RECORDS.value,
    "get_record": ActionType.GET_RECORD.value,
    "get_document": ActionType.GET_DOCUMENT.value,
    "calculator": ActionType.CALCULATOR.value,
    "calculate": ActionType.CALCULATOR.value,
    "submit": ActionType.SUBMIT.value,
    "submit_findings": ActionType.SUBMIT.value,
    "query_ledger": ActionType.QUERY_LEDGER.value,
    "query_invoices": ActionType.QUERY_INVOICES.value,
    "query_payments": ActionType.QUERY_PAYMENTS.value,
    "query_fx": ActionType.QUERY_FX.value,
}

AVAILABLE_ACTION_SCHEMAS: dict[str, Any] = {
    ActionType.LIST_RECORDS.value: {
        "description": "Paginate through financial records with stable cursors (Section 04)",
        "parameters": {
            "table": "str ('ledger', 'invoices', 'payments', 'fx')",
            "cursor": "Optional[str] (opaque or offset cursor)",
            "limit": "int (1-50, default: 20)",
        },
    },
    ActionType.GET_RECORD.value: {
        "description": "Fetch a specific record by primary table ID (Section 04)",
        "parameters": {
            "table": "str ('ledger', 'invoices', 'payments', 'fx')",
            "id": "str (record ID)",
        },
    },
    ActionType.GET_DOCUMENT.value: {
        "description": "Fetch a full document (invoice or payment) by doc_id (Section 04)",
        "parameters": {
            "doc_id": "str (e.g., 'DOC-INV-2026-0001')",
        },
    },
    ActionType.CALCULATOR.value: {
        "description": "Execute safe arithmetic expression in sandbox calculator (Section 04)",
        "parameters": {
            "expression": "str (e.g., '100 + 25 * 4')",
        },
    },
    ActionType.SUBMIT.value: {
        "description": "Submit final detected audit anomalies report to conclude episode (Section 04 & 05)",
        "parameters": {
            "findings": "List of findings [{type, affected_ids, claimed_amount_minor, currency, evidence_ids, explanation}]",
            "explanation": "Optional[str]",
        },
    },
    ActionType.QUERY_LEDGER.value: {
        "description": "Query general ledger rows with pagination and filters",
        "parameters": {
            "account_id": "Optional[str]",
            "date_from": "Optional[str YYYY-MM-DD]",
            "date_to": "Optional[str YYYY-MM-DD]",
            "reference_id": "Optional[str]",
            "page": "int (default: 1)",
            "page_size": "int (1-50, default: 20)",
        },
    },
    ActionType.QUERY_INVOICES.value: {
        "description": "Query company invoices with pagination and filters",
        "parameters": {
            "vendor_id": "Optional[str]",
            "status": "Optional[str] ('unpaid', 'paid')",
            "page": "int (default: 1)",
            "page_size": "int (1-50, default: 20)",
        },
    },
    ActionType.QUERY_PAYMENTS.value: {
        "description": "Query bank disbursement payments with pagination and filters",
        "parameters": {
            "vendor_id": "Optional[str]",
            "invoice_id": "Optional[str]",
            "page": "int (default: 1)",
            "page_size": "int (1-50, default: 20)",
        },
    },
    ActionType.QUERY_FX.value: {
        "description": "Query foreign exchange rates across dates",
        "parameters": {
            "date": "Optional[str YYYY-MM-DD]",
            "quote_currency": "Optional[str] (EUR, GBP, JPY)",
            "page": "int (default: 1)",
            "page_size": "int (1-50, default: 20)",
        },
    },
}


def validate_action(action: Any) -> dict[str, Any]:
    """Validate action input structure and return standardized action dict.

    Accepts action type via 'action' or 'type' key. Supports both Section 04 and legacy names.

    Raises:
        InvalidActionError: If action is not a dict or has invalid type.
    """
    if not isinstance(action, dict):
        raise InvalidActionError(f"Action must be a dictionary, got {type(action).__name__}")

    raw_type = action.get("action") or action.get("type")
    if not raw_type or str(raw_type) not in ACTION_ALIASES:
        raise InvalidActionError(
            f"Invalid action type '{raw_type}'. Valid types: {sorted(VALID_ACTION_TYPES)}"
        )

    canonical_type = ACTION_ALIASES[str(raw_type)]
    normalized = dict(action)
    normalized["type"] = canonical_type
    normalized["action"] = canonical_type
    return normalized
