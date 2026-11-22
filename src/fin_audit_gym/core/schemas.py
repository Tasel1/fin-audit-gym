"""Core Pydantic v2 schemas for the Fin-Audit-Gym environment."""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Policy(BaseModel):
    """Corporate accounting policy specification (Section 03)."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default="1.0.0", description="Policy schema version")
    episode_id: str = Field(description="Episode identifier")
    fx_date_policy: str = Field(default="transaction_date", description="Selection of FX rate date")
    period_start: str = Field(description="Reporting period start timestamp (ISO-8601)")
    period_end: str = Field(description="Reporting period end timestamp (ISO-8601)")
    timezone: str = Field(default="UTC", description="Accounting reporting timezone")
    rounding_rule: str = Field(default="ROUND_HALF_EVEN", description="Monetary rounding rule")
    amount_tolerance_minor: int = Field(
        default=100, description="Acceptable amount tolerance in minor units"
    )


class LedgerRow(BaseModel):
    """Represents a single row in the double-entry general ledger."""

    model_config = ConfigDict(extra="ignore")

    # Canonical Section 03 fields
    row_id: str = Field(default="", description="Unique row identifier")
    timestamp: str = Field(default="", description="Transaction timestamp or date ISO-8601")
    account: str = Field(default="", description="Account identifier")
    debit_minor: int = Field(ge=0, default=0, description="Debit amount in integer minor units")
    credit_minor: int = Field(ge=0, default=0, description="Credit amount in integer minor units")
    currency: str = Field(default="USD", description="Currency code")
    entity_id: str | None = Field(default=None, description="Entity/vendor ID associated with row")
    invoice_id: str | None = Field(default=None, description="Linked invoice ID")
    payment_id: str | None = Field(default=None, description="Linked payment ID")

    # Legacy & compatibility aliases
    entry_id: str = Field(default="", description="Unique entry ID")
    account_id: str = Field(default="", description="Account identifier")
    date: str = Field(default="", description="Transaction date YYYY-MM-DD")
    description: str = Field(default="", description="Line description")
    reference_id: str | None = Field(
        default=None, description="External reference ID (invoice/payment)"
    )

    @model_validator(mode="after")
    def sync_and_validate(self) -> Self:
        # Cross-populate identifiers
        if not self.row_id and self.entry_id:
            self.row_id = self.entry_id
        elif not self.entry_id and self.row_id:
            self.entry_id = self.row_id

        if not self.account and self.account_id:
            self.account = self.account_id
        elif not self.account_id and self.account:
            self.account_id = self.account

        if not self.timestamp and self.date:
            self.timestamp = self.date
        elif not self.date and self.timestamp:
            self.date = self.timestamp.split("T")[0] if "T" in self.timestamp else self.timestamp

        if not self.reference_id and (self.invoice_id or self.payment_id):
            self.reference_id = self.invoice_id or self.payment_id
        elif self.reference_id and not self.invoice_id and not self.payment_id:
            if "INV" in self.reference_id:
                self.invoice_id = self.reference_id
            elif "PAY" in self.reference_id or "WIRE" in self.reference_id:
                self.payment_id = self.reference_id

        if self.debit_minor <= 0 and self.credit_minor <= 0:
            raise ValueError("Either debit_minor or credit_minor must be greater than 0")
        return self


class InvoiceLineItem(BaseModel):
    """Line item within an invoice."""

    model_config = ConfigDict(extra="ignore")

    item_id: str = Field(description="Unique item ID")
    description: str = Field(description="Item description")
    quantity: int = Field(gt=0, description="Quantity of items")
    unit_price_minor: int = Field(ge=0, description="Unit price in minor currency units")
    total_minor: int = Field(ge=0, description="Total price in minor currency units")


class Invoice(BaseModel):
    """Invoice document schema."""

    model_config = ConfigDict(extra="ignore")

    # Canonical Section 03 fields
    doc_id: str = Field(default="", description="Unique document ID (e.g. DOC-INV-...)")
    invoice_id: str = Field(description="Unique invoice ID")
    entity_id: str = Field(default="", description="Entity/Vendor identifier")
    issued_at: str = Field(default="", description="Issue timestamp ISO-8601")
    due_at: str = Field(default="", description="Payment due timestamp ISO-8601")
    currency: str = Field(description="Currency code (e.g., USD)")
    total_minor: int = Field(ge=0, description="Total invoice amount in minor units")
    line_items: list[InvoiceLineItem] = Field(
        default_factory=list, description="List of line items"
    )

    # Legacy & compatibility aliases
    vendor_id: str = Field(default="", description="Vendor ID")
    invoice_date: str = Field(default="", description="Invoice issue date YYYY-MM-DD")
    due_date: str = Field(default="", description="Payment due date YYYY-MM-DD")
    status: str = Field(default="unpaid", description="Status of invoice (unpaid, paid, partial)")

    @model_validator(mode="after")
    def sync_invoice_fields(self) -> Self:
        if not self.doc_id:
            self.doc_id = f"DOC-{self.invoice_id}"
        if not self.entity_id and self.vendor_id:
            self.entity_id = self.vendor_id
        elif not self.vendor_id and self.entity_id:
            self.vendor_id = self.entity_id

        if not self.issued_at and self.invoice_date:
            self.issued_at = self.invoice_date
        elif not self.invoice_date and self.issued_at:
            self.invoice_date = (
                self.issued_at.split("T")[0] if "T" in self.issued_at else self.issued_at
            )

        if not self.due_at and self.due_date:
            self.due_at = self.due_date
        elif not self.due_date and self.due_at:
            self.due_date = self.due_at.split("T")[0] if "T" in self.due_at else self.due_at

        return self


class Payment(BaseModel):
    """Cash disbursement or receipt payment."""

    model_config = ConfigDict(extra="ignore")

    # Canonical Section 03 fields
    doc_id: str = Field(default="", description="Unique document ID (e.g. DOC-PAY-...)")
    payment_id: str = Field(description="Unique payment ID")
    timestamp: str = Field(default="", description="Payment timestamp ISO-8601")
    currency: str = Field(description="Currency code")
    amount_minor: int = Field(ge=0, description="Payment amount in minor units")
    invoice_refs: list[str] = Field(default_factory=list, description="Linked invoice IDs")

    # Legacy & compatibility aliases
    invoice_id: str | None = Field(default=None, description="Linked invoice ID if allocated")
    vendor_id: str = Field(default="", description="Vendor/Payee ID")
    entity_id: str = Field(default="", description="Entity ID")
    payment_date: str = Field(default="", description="Payment date YYYY-MM-DD")
    bank_reference: str = Field(default="", description="Bank transaction reference code")

    @model_validator(mode="after")
    def sync_payment_fields(self) -> Self:
        if not self.doc_id:
            self.doc_id = f"DOC-{self.payment_id}"
        if not self.timestamp and self.payment_date:
            self.timestamp = self.payment_date
        elif not self.payment_date and self.timestamp:
            self.payment_date = (
                self.timestamp.split("T")[0] if "T" in self.timestamp else self.timestamp
            )

        if not self.invoice_refs and self.invoice_id:
            self.invoice_refs = [self.invoice_id]
        elif not self.invoice_id and self.invoice_refs:
            self.invoice_id = self.invoice_refs[0]

        if not self.entity_id and self.vendor_id:
            self.entity_id = self.vendor_id
        elif not self.vendor_id and self.entity_id:
            self.vendor_id = self.entity_id

        return self


class FXRate(BaseModel):
    """Foreign exchange conversion rate."""

    model_config = ConfigDict(extra="ignore")

    date: str = Field(description="Effective date YYYY-MM-DD")
    source_currency: str = Field(default="USD", description="Source/Base currency code")
    target_currency: str = Field(default="EUR", description="Target/Quote currency code")
    rate: str = Field(description="Exchange rate as decimal string")

    # Legacy & compatibility aliases
    base_currency: str = Field(default="USD", description="Base currency code")
    quote_currency: str = Field(default="EUR", description="Quote currency code")

    @model_validator(mode="after")
    def sync_currencies(self) -> Self:
        if self.source_currency and not self.base_currency:
            self.base_currency = self.source_currency
        elif self.base_currency and not self.source_currency:
            self.source_currency = self.base_currency

        if self.target_currency and not self.quote_currency:
            self.quote_currency = self.target_currency
        elif self.quote_currency and not self.target_currency:
            self.target_currency = self.quote_currency
        return self


class Manifest(BaseModel):
    """Episode metadata and world manifest (Section 03)."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default="1.0.0", description="Schema version")
    episode_id: str = Field(description="Episode identifier")
    company_name: str = Field(description="Company name")
    fiscal_year: int = Field(description="Fiscal year")
    base_currency: str = Field(default="USD", description="Base currency code")
    currency: str = Field(default="USD", description="Base currency code alias")
    period_start: str = Field(default="2026-01-01T00:00:00+00:00", description="Period start")
    period_end: str = Field(default="2026-12-31T23:59:59+00:00", description="Period end")
    timezone: str = Field(default="UTC", description="Reporting timezone")
    total_records: int = Field(default=0, description="Total ledger rows count")
    total_documents: int = Field(default=0, description="Total invoices and payments count")
    total_transactions: int = Field(default=0, description="Total transactions in episode")
    initial_cash_minor: int = Field(description="Starting cash balance in minor units")
    rounding_rule: str = Field(default="ROUND_HALF_EVEN", description="Rounding rule")
    injected_anomalies_count: int = Field(default=0, description="Count of injected anomalies")

    @model_validator(mode="after")
    def sync_manifest(self) -> Self:
        if not self.base_currency and self.currency:
            self.base_currency = self.currency
        elif not self.currency and self.base_currency:
            self.currency = self.base_currency

        if not self.total_records and self.total_transactions:
            self.total_records = self.total_transactions
        elif not self.total_transactions and self.total_records:
            self.total_transactions = self.total_records

        return self


class Finding(BaseModel):
    """Auditor reported anomaly finding and ground truth representation."""

    model_config = ConfigDict(extra="ignore")

    # Canonical Section 05 fields
    issue_id: str | None = Field(default=None, description="Canonical issue identifier")
    type: str | None = Field(default=None, description="Canonical anomaly type")
    affected_ids: list[str] = Field(
        default_factory=list, description="IDs of affected rows/invoices/payments"
    )
    claimed_amount_minor: int | None = Field(
        default=None, description="Claimed anomaly amount in minor units"
    )
    expected_amount_minor: int | None = Field(
        default=None, description="Expected true anomaly amount in minor units"
    )
    currency: str = Field(default="USD", description="Currency of anomaly")
    evidence_ids: list[str] = Field(
        default_factory=list, description="Supporting evidence IDs (max 10)"
    )
    admissible_evidence_sets: list[list[str]] = Field(
        default_factory=list, description="List of admissible evidence ID sets"
    )
    explanation: str | None = Field(
        default=None, description="Short justification (max 1000 characters)"
    )

    # Legacy & compatibility aliases
    finding_id: str = Field(default="", description="Unique finding ID")
    error_type: str = Field(default="", description="Type of anomaly")
    entity_id: str = Field(default="", description="ID of entity where error occurred")
    description: str = Field(default="", description="Detailed description of anomaly")
    evidence: dict[str, Any] = Field(
        default_factory=dict, description="Supporting evidence details"
    )

    @model_validator(mode="after")
    def sync_finding_fields(self) -> Self:
        # Require minimal valid finding structure
        if not self.finding_id and not self.issue_id and not self.type and not self.error_type:
            raise ValueError("Finding must specify at least an identifier or anomaly type")
        if not self.finding_id and self.issue_id:
            self.finding_id = self.issue_id
        elif not self.issue_id and self.finding_id:
            self.issue_id = self.finding_id

        if not self.type and self.error_type:
            self.type = self.error_type
        elif not self.error_type and self.type:
            self.error_type = self.type

        if not self.affected_ids and self.entity_id:
            self.affected_ids = [self.entity_id]
        elif not self.entity_id and self.affected_ids:
            self.entity_id = self.affected_ids[0]

        if not self.explanation and self.description:
            self.explanation = self.description
        elif not self.description and self.explanation:
            self.description = self.explanation

        # Auto-extract evidence_ids from evidence dict if empty
        if not self.evidence_ids and self.evidence:
            extracted: list[str] = []
            for v in self.evidence.values():
                if isinstance(v, str) and (
                    "INV" in v
                    or "PAY" in v
                    or "WIRE" in v
                    or "ENTRY" in v
                    or "DOC" in v
                    or "ROW" in v
                ):
                    extracted.append(v)
                elif isinstance(v, list):
                    for sub in v:
                        if isinstance(sub, str):
                            extracted.append(sub)
            self.evidence_ids = extracted[:10]

        if not self.admissible_evidence_sets and self.evidence_ids:
            self.admissible_evidence_sets = [list(self.evidence_ids)]

        return self


class Submission(BaseModel):
    """Final audit submission sent by the agent."""

    model_config = ConfigDict(extra="ignore")

    episode_id: str | None = Field(default=None, description="Episode ID")
    findings: list[Finding] = Field(
        default_factory=list, description="List of detected audit findings (max 30)"
    )
    explanation: str | None = Field(
        default=None, description="Optional auditor explanation summary"
    )


class Observation(BaseModel):
    """Observation returned by Gym environment."""

    model_config = ConfigDict(extra="ignore")

    task: str | None = Field(default=None, description="Audit task description")
    manifest: dict[str, Any] | None = Field(default=None, description="Manifest metadata")
    policy: dict[str, Any] | None = Field(default=None, description="Accounting policy summary")
    remaining_budget: int | None = Field(default=None, description="Remaining steps budget")
    page: int = Field(default=1, description="Current page index")
    total_pages: int = Field(default=1, description="Total available pages")
    items: list[dict[str, Any]] = Field(default_factory=list, description="Items on current page")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context metadata"
    )


class EpisodeInfo(BaseModel):
    """Episode progress and execution limits info."""

    model_config = ConfigDict(extra="ignore")

    steps_taken: int = Field(description="Number of steps executed so far")
    max_steps: int = Field(description="Maximum steps allowed per episode")
    remaining_steps: int = Field(default=100, description="Steps remaining before truncation")
    current_page: int = Field(default=1, description="Current page index")
    total_queries: int = Field(default=0, description="Total queries executed")
