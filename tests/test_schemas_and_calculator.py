"""Tests for Core Schemas, Currency Math, and Safe AST Calculator."""

import pytest
from pydantic import ValidationError

from fin_audit_gym.core import (
    EpisodeInfo,
    Finding,
    FXRate,
    Invoice,
    InvoiceLineItem,
    LedgerRow,
    Manifest,
    Observation,
    Payment,
    Submission,
    safe_calculate,
)
from fin_audit_gym.utils import convert_currency

# ==============================================================================
# 1. Pydantic Schemas Tests
# ==============================================================================


def test_ledger_row_valid():
    row_debit = LedgerRow(
        account_id="1010",
        date="2024-01-15",
        description="Cash sale",
        debit_minor=50000,
        credit_minor=0,
        entry_id="ENTRY-001",
    )
    assert row_debit.debit_minor == 50000
    assert row_debit.credit_minor == 0
    assert row_debit.reference_id is None

    row_credit = LedgerRow(
        account_id="2010",
        date="2024-01-15",
        description="Accounts payable",
        debit_minor=0,
        credit_minor=25000,
        reference_id="INV-100",
        entry_id="ENTRY-002",
    )
    assert row_credit.credit_minor == 25000
    assert row_credit.reference_id == "INV-100"


def test_ledger_row_zero_or_negative_validation():
    # Both debit and credit cannot be zero
    with pytest.raises(ValidationError):
        LedgerRow(
            account_id="1010",
            date="2024-01-15",
            description="Zero row",
            debit_minor=0,
            credit_minor=0,
            entry_id="ENTRY-003",
        )

    # Negative values are rejected
    with pytest.raises(ValidationError):
        LedgerRow(
            account_id="1010",
            date="2024-01-15",
            description="Negative debit",
            debit_minor=-100,
            credit_minor=0,
            entry_id="ENTRY-004",
        )

    with pytest.raises(ValidationError):
        LedgerRow(
            account_id="1010",
            date="2024-01-15",
            description="Negative credit",
            debit_minor=0,
            credit_minor=-50,
            entry_id="ENTRY-005",
        )


def test_invoice_line_item_and_invoice():
    item1 = InvoiceLineItem(
        item_id="ITEM-1",
        description="Server Hosting",
        quantity=2,
        unit_price_minor=15000,
        total_minor=30000,
    )
    assert item1.quantity == 2

    # quantity must be > 0
    with pytest.raises(ValidationError):
        InvoiceLineItem(
            item_id="ITEM-2",
            description="Invalid Item",
            quantity=0,
            unit_price_minor=1000,
            total_minor=0,
        )

    invoice = Invoice(
        invoice_id="INV-2024-001",
        vendor_id="VEND-10",
        invoice_date="2024-02-01",
        due_date="2024-02-28",
        currency="USD",
        line_items=[item1],
        total_minor=30000,
    )
    assert invoice.status == "unpaid"
    assert len(invoice.line_items) == 1
    assert invoice.total_minor == 30000


def test_payment_schema():
    payment = Payment(
        payment_id="PAY-99",
        invoice_id="INV-2024-001",
        vendor_id="VEND-10",
        payment_date="2024-02-15",
        currency="USD",
        amount_minor=30000,
        bank_reference="TXN-REF-8841",
    )
    assert payment.amount_minor == 30000
    assert payment.invoice_id == "INV-2024-001"

    # amount_minor cannot be negative
    with pytest.raises(ValidationError):
        Payment(
            payment_id="PAY-100",
            vendor_id="VEND-10",
            payment_date="2024-02-15",
            currency="USD",
            amount_minor=-1,
            bank_reference="TXN-REF-8842",
        )


def test_additional_schemas():
    fx = FXRate(
        date="2024-01-01",
        base_currency="EUR",
        quote_currency="USD",
        rate="1.0950",
    )
    assert fx.rate == "1.0950"

    manifest = Manifest(
        episode_id="EP-001",
        company_name="Acme Corp",
        fiscal_year=2024,
        currency="USD",
        initial_cash_minor=100000000,
        total_transactions=450,
        injected_anomalies_count=3,
    )
    assert manifest.fiscal_year == 2024

    finding = Finding(
        finding_id="F-1",
        error_type="duplicate_invoice",
        entity_id="INV-2024-001",
        description="Duplicate invoice entry detected",
        evidence={"matched_invoice_id": "INV-2024-002"},
    )
    submission = Submission(findings=[finding], explanation="Found duplicate invoice")
    assert len(submission.findings) == 1

    obs = Observation(
        page=1,
        total_pages=5,
        items=[{"entry_id": "ENTRY-001", "debit_minor": 50000}],
        metadata={"total_records": 50},
    )
    assert obs.page == 1

    info = EpisodeInfo(
        steps_taken=4,
        max_steps=100,
        current_page=1,
        total_queries=12,
    )
    assert info.steps_taken == 4


# ==============================================================================
# 2. Safe AST Calculator Tests
# ==============================================================================


def test_safe_calculate_valid_expressions():
    assert safe_calculate("100 + 25 * 4") == "200"
    assert safe_calculate("(50 - 20) * 2") == "60"
    assert safe_calculate("-10 + 5") == "-5"
    assert safe_calculate("+15 + 5") == "20"
    assert safe_calculate("10 % 3") == "1"

    # Division with Decimal precision
    res_div = safe_calculate("1000 / 3")
    assert res_div.startswith("333.3333333333333")


def test_safe_calculate_division_by_zero():
    with pytest.raises(ValueError, match="Division by zero"):
        safe_calculate("100 / 0")

    with pytest.raises(ValueError, match="Division by zero"):
        safe_calculate("100 % 0")


def test_safe_calculate_security_attacks():
    malicious_inputs = [
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "exec('import sys')",
        "math.sqrt(4)",
        "eval('1+1')",
        "x + 1",
        "[x for x in [1, 2]]",
        "lambda x: x",
        "import os",
        "print('hello')",
        "(lambda: 1)()",
        "1; import sys",
    ]
    for expr in malicious_inputs:
        with pytest.raises(ValueError):
            safe_calculate(expr)


def test_safe_calculate_invalid_syntax():
    with pytest.raises(ValueError):
        safe_calculate("100 + * 5")

    with pytest.raises(ValueError):
        safe_calculate("")


# ==============================================================================
# 3. Currency Conversion Tests
# ==============================================================================


def test_convert_currency_valid():
    assert convert_currency(10000, "1.25") == 12500
    assert convert_currency(0, "1.50") == 0

    # Round half up checks
    # 105 * 1.005 = 105.525 -> 106
    assert convert_currency(105, "1.005") == 106
    # 105 * 1.004 = 105.42 -> 105
    assert convert_currency(105, "1.004") == 105


def test_convert_currency_invalid_inputs():
    with pytest.raises(ValueError, match="negative"):
        convert_currency(-100, "1.25")

    with pytest.raises(ValueError):
        convert_currency(100, "invalid_rate")

    with pytest.raises(ValueError):
        convert_currency(100, "0")

    with pytest.raises(ValueError):
        convert_currency(100, "-1.25")
