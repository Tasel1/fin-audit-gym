"""Action definitions, constants, and validation for FinAuditEnv."""

from enum import Enum
from typing import Any

from fin_audit_gym.core.exceptions import InvalidActionError


class ActionType(str, Enum):
    """Enumeration of all valid action types in FinAuditEnv."""

    QUERY_LEDGER = "query_ledger"
    QUERY_INVOICES = "query_invoices"
    QUERY_PAYMENTS = "query_payments"
    QUERY_FX = "query_fx"
    CALCULATE = "calculate"
    SUBMIT_FINDINGS = "submit_findings"


VALID_ACTION_TYPES = {item.value for item in ActionType}

AVAILABLE_ACTION_SCHEMAS: dict[str, Any] = {
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
    ActionType.CALCULATE.value: {
        "description": "Execute safe arithmetic expression in sandbox calculator",
        "parameters": {
            "expression": "str (e.g., '100 + 25 * 4')",
        },
    },
    ActionType.SUBMIT_FINDINGS.value: {
        "description": "Submit final detected audit anomalies report to conclude episode",
        "parameters": {
            "findings": "List of Finding objects [finding_id, error_type, entity_id, description, evidence]",
            "explanation": "Optional[str]",
        },
    },
}


def validate_action(action: Any) -> dict[str, Any]:
    """Validate action input structure and return standardized action dict.

    Accepts action type via either 'type' or 'action' key for maximum interoperability.

    Raises:
        InvalidActionError: If action is not a dict or has invalid type.
    """
    if not isinstance(action, dict):
        raise InvalidActionError(f"Action must be a dictionary, got {type(action).__name__}")

    action_type = action.get("type") or action.get("action")
    if not action_type or action_type not in VALID_ACTION_TYPES:
        raise InvalidActionError(
            f"Invalid action type '{action_type}'. Valid types: {sorted(VALID_ACTION_TYPES)}"
        )

    normalized = dict(action)
    normalized["type"] = action_type
    return normalized
