"""Core Pydantic v2 schemas for the Fin-Audit-Gym environment."""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LedgerRow(BaseModel):
    """Represents a single row in the double-entry general ledger."""

    model_config = ConfigDict(extra="forbid")

    account_id: str = Field(description="Account identifier")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Transaction date YYYY-MM-DD")
    description: str = Field(description="Line description")
    debit_minor: int = Field(ge=0, description="Debit amount in integer minor units")
    credit_minor: int = Field(ge=0, description="Credit amount in integer minor units")
    reference_id: str | None = Field(
        default=None, description="External reference ID (invoice/payment)"
    )
    entry_id: str = Field(description="Unique entry ID")

    @model_validator(mode="after")
    def validate_debit_or_credit(self) -> Self:
        if self.debit_minor <= 0 and self.credit_minor <= 0:
            raise ValueError("Either debit_minor or credit_minor must be greater than 0")
        return self


class InvoiceLineItem(BaseModel):
    """Line item within an invoice."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(description="Unique item ID")
    description: str = Field(description="Item description")
    quantity: int = Field(gt=0, description="Quantity of items")
    unit_price_minor: int = Field(ge=0, description="Unit price in minor currency units")
    total_minor: int = Field(ge=0, description="Total price in minor currency units")


class Invoice(BaseModel):
    """Invoice document schema."""

    model_config = ConfigDict(extra="forbid")

    invoice_id: str = Field(description="Unique invoice ID")
    vendor_id: str = Field(description="Vendor ID")
    invoice_date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$", description="Invoice issue date YYYY-MM-DD"
    )
    due_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Payment due date YYYY-MM-DD")
    currency: str = Field(description="Currency code (e.g., USD)")
    line_items: list[InvoiceLineItem] = Field(
        default_factory=list, description="List of line items"
    )
    total_minor: int = Field(ge=0, description="Total invoice amount in minor units")
    status: str = Field(default="unpaid", description="Status of invoice (unpaid, paid, partial)")


class Payment(BaseModel):
    """Cash disbursement or receipt payment."""

    model_config = ConfigDict(extra="forbid")

    payment_id: str = Field(description="Unique payment ID")
    invoice_id: str | None = Field(default=None, description="Linked invoice ID if allocated")
    vendor_id: str = Field(description="Vendor/Payee ID")
    payment_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Payment date YYYY-MM-DD")
    currency: str = Field(description="Currency code")
    amount_minor: int = Field(ge=0, description="Payment amount in minor units")
    bank_reference: str = Field(description="Bank transaction reference code")


class FXRate(BaseModel):
    """Foreign exchange conversion rate."""

    model_config = ConfigDict(extra="forbid")

    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Effective date YYYY-MM-DD")
    base_currency: str = Field(description="Base currency code")
    quote_currency: str = Field(description="Quote currency code")
    rate: str = Field(description="Exchange rate as decimal string")


class Manifest(BaseModel):
    """Episode metadata and world manifest."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(description="Episode identifier")
    company_name: str = Field(description="Company name")
    fiscal_year: int = Field(description="Fiscal year")
    currency: str = Field(description="Base currency code")
    initial_cash_minor: int = Field(description="Starting cash balance in minor units")
    total_transactions: int = Field(description="Total transactions in episode")
    injected_anomalies_count: int = Field(description="Count of injected anomalies")


class Finding(BaseModel):
    """Auditor reported anomaly finding."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(description="Unique finding ID")
    error_type: str = Field(description="Type of anomaly (duplicate_invoice, fx_mismatch, etc.)")
    entity_id: str = Field(description="ID of entity where error occurred")
    description: str = Field(description="Detailed description of anomaly")
    evidence: dict[str, Any] = Field(
        default_factory=dict, description="Supporting evidence details"
    )


class Submission(BaseModel):
    """Final audit submission sent by the agent."""

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(
        default_factory=list, description="List of detected audit findings"
    )
    explanation: str | None = Field(
        default=None, description="Optional auditor explanation summary"
    )


class Observation(BaseModel):
    """Observation returned by Gym environment."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(description="Current page index")
    total_pages: int = Field(description="Total available pages")
    items: list[dict[str, Any]] = Field(default_factory=list, description="Items on current page")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context metadata"
    )


class EpisodeInfo(BaseModel):
    """Episode progress and execution limits info."""

    model_config = ConfigDict(extra="forbid")

    steps_taken: int = Field(description="Number of steps executed so far")
    max_steps: int = Field(description="Maximum steps allowed per episode")
    current_page: int = Field(description="Current page index")
    total_queries: int = Field(description="Total queries executed")
