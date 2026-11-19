"""Synthetic double-entry ledger generator producing balanced financial worlds."""

import csv
import datetime
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fin_audit_gym.core.schemas import (
    FXRate,
    Invoice,
    InvoiceLineItem,
    LedgerRow,
    Manifest,
    Payment,
)
from fin_audit_gym.generator.provenance import ProvenanceTracker
from fin_audit_gym.utils.currency import convert_currency


class GeneratorConfig(BaseModel):
    """Configuration options for synthetic financial world generation."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, description="Random seed for deterministic generation")
    num_vendors: int = Field(default=10, description="Number of distinct vendors")
    num_transactions: int = Field(default=50, description="Total target transaction events")
    base_currency: str = Field(default="USD", description="Base accounting currency")
    fiscal_year: int = Field(default=2026, description="Fiscal reporting year")
    initial_cash_minor: int = Field(
        default=100_000_000, description="Starting cash balance in minor units (1,000,000 USD)"
    )


class GeneratedWorld(BaseModel):
    """Container representing a fully generated corporate financial universe."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    manifest: Manifest
    ledger: list[LedgerRow]
    invoices: list[Invoice]
    payments: list[Payment]
    fx_rates: list[FXRate]
    provenance: dict[str, Any]

    def verify_double_entry_invariants(self) -> bool:
        """Verify strict mathematical double-entry equality for all entries and ledger."""
        entries: dict[str, list[LedgerRow]] = defaultdict(list)
        total_debits = 0
        total_credits = 0

        for row in self.ledger:
            entries[row.entry_id].append(row)
            total_debits += row.debit_minor
            total_credits += row.credit_minor

        for entry_id, rows in entries.items():
            entry_debit = sum(r.debit_minor for r in rows)
            entry_credit = sum(r.credit_minor for r in rows)
            if entry_debit != entry_credit:
                raise ValueError(
                    f"Double-entry invariant violated: entry '{entry_id}' is unbalanced "
                    f"(debits={entry_debit}, credits={entry_credit})"
                )

        if total_debits != total_credits:
            raise ValueError(
                f"Double-entry invariant violated: total debits ({total_debits}) != "
                f"total credits ({total_credits})"
            )

        return True

    def export_to_directory(self, dir_path: Path | str) -> None:
        """Export world artifacts to target directory as CSV, JSON, and JSONL files."""
        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export ledger.csv
        ledger_file = out_dir / "ledger.csv"
        with ledger_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "entry_id",
                    "account_id",
                    "date",
                    "description",
                    "debit_minor",
                    "credit_minor",
                    "reference_id",
                ]
            )
            for row in self.ledger:
                writer.writerow(
                    [
                        row.entry_id,
                        row.account_id,
                        row.date,
                        row.description,
                        row.debit_minor,
                        row.credit_minor,
                        row.reference_id or "",
                    ]
                )

        # 2. Export invoices.jsonl
        invoices_file = out_dir / "invoices.jsonl"
        with invoices_file.open("w", encoding="utf-8") as f:
            for inv in self.invoices:
                f.write(inv.model_dump_json() + "\n")

        # 3. Export payments.jsonl
        payments_file = out_dir / "payments.jsonl"
        with payments_file.open("w", encoding="utf-8") as f:
            for pmt in self.payments:
                f.write(pmt.model_dump_json() + "\n")

        # 4. Export fx.csv
        fx_file = out_dir / "fx.csv"
        with fx_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "base_currency", "quote_currency", "rate"])
            for fx in self.fx_rates:
                writer.writerow([fx.date, fx.base_currency, fx.quote_currency, fx.rate])

        # 5. Export manifest.json
        manifest_file = out_dir / "manifest.json"
        manifest_file.write_text(self.manifest.model_dump_json(indent=2), encoding="utf-8")

        # 6. Export provenance.json
        provenance_file = out_dir / "provenance.json"
        provenance_file.write_text(json.dumps(self.provenance, indent=2), encoding="utf-8")


class SyntheticLedgerGenerator:
    """Deterministic generator for synthetic double-entry financial universes."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.tracker = ProvenanceTracker()

    def _build_date(self, day_offset: int) -> str:
        """Return ISO date string for given day offset within fiscal year."""
        safe_offset = min(max(day_offset, 0), 364)
        base = datetime.date(self.config.fiscal_year, 1, 1)
        return (base + datetime.timedelta(days=safe_offset)).isoformat()

    def _generate_fx_rates(self) -> tuple[list[FXRate], dict[tuple[str, str], str]]:
        """Generate deterministic FX rates for foreign currencies across fiscal year."""
        fx_rates: list[FXRate] = []
        fx_lookup: dict[tuple[str, str], str] = {}

        # Base exchange rate profiles relative to USD
        profiles = {
            "EUR": (1.0850, 0.0300, 4),
            "GBP": (1.2750, 0.0350, 4),
            "JPY": (0.006700, 0.000300, 6),
        }

        for day in range(365):
            d_str = self._build_date(day)
            for quote, (base_val, swing, decimals) in profiles.items():
                offset = (self.rng.random() * 2.0 - 1.0) * swing
                rate_val = round(base_val + offset, decimals)
                rate_str = f"{rate_val:.{decimals}f}"

                rate_obj = FXRate(
                    date=d_str,
                    base_currency=self.config.base_currency,
                    quote_currency=quote,
                    rate=rate_str,
                )
                fx_rates.append(rate_obj)
                fx_lookup[(d_str, quote)] = rate_str

        return fx_rates, fx_lookup

    def generate(self) -> GeneratedWorld:
        """Generate complete, deterministic, and balanced double-entry financial world."""
        vendors = [f"VEND-{i:03d}" for i in range(1, self.config.num_vendors + 1)]
        customers = [f"CUST-{i:03d}" for i in range(1, 6)]

        fx_rates, fx_lookup = self._generate_fx_rates()

        ledger: list[LedgerRow] = []
        invoices: list[Invoice] = []
        payments: list[Payment] = []

        unpaid_invoices: list[Invoice] = []
        entry_counter = 1
        txn_counter = 1

        company_names = [
            "Apex Horizon Dynamics Inc.",
            "Nexus Frontier Analytics Corp.",
            "Vanguard Quantum Logistics Ltd.",
            "Starlight Enterprise Solutions Inc.",
            "Zenith Global Technologies Corp.",
        ]
        company_name = company_names[self.config.seed % len(company_names)]

        # Target partition of transactions
        num_txns = self.config.num_transactions
        day_stride = max(1, 340 // max(num_txns, 1))

        for idx in range(num_txns):
            curr_day = min(idx * day_stride + self.rng.randint(0, day_stride), 340)
            date_str = self._build_date(curr_day)
            tx_type_roll = self.rng.random()

            # 1. Payment settling an existing unpaid invoice if available
            if unpaid_invoices and tx_type_roll < 0.35:
                inv = unpaid_invoices.pop(0)
                inv.status = "paid"
                pay_id = f"PAY-{self.config.fiscal_year}-{len(payments) + 1:04d}"
                entry_id = f"ENTRY-{entry_counter:05d}"
                entry_counter += 1
                txn_id = f"TXN-{txn_counter:05d}"
                txn_counter += 1

                pay_day = min(curr_day + self.rng.randint(2, 20), 364)
                pay_date = self._build_date(pay_day)

                if inv.currency == self.config.base_currency:
                    # Debit AP (2000), Credit Cash (1000)
                    r1 = LedgerRow(
                        account_id="2000",
                        date=pay_date,
                        description=f"Payment for invoice {inv.invoice_id}",
                        debit_minor=inv.total_minor,
                        credit_minor=0,
                        reference_id=pay_id,
                        entry_id=entry_id,
                    )
                    r2 = LedgerRow(
                        account_id="1000",
                        date=pay_date,
                        description=f"Cash settlement for invoice {inv.invoice_id}",
                        debit_minor=0,
                        credit_minor=inv.total_minor,
                        reference_id=pay_id,
                        entry_id=entry_id,
                    )
                    ledger.extend([r1, r2])
                else:
                    # Foreign currency settlement with FX calculation
                    orig_fx = fx_lookup.get((inv.invoice_date, inv.currency), "1.0850")
                    orig_usd = convert_currency(inv.total_minor, orig_fx)

                    pay_fx = fx_lookup.get((pay_date, inv.currency), orig_fx)
                    pay_usd = convert_currency(inv.total_minor, pay_fx)

                    # AP is debited for original recorded liability
                    r_ap = LedgerRow(
                        account_id="2000",
                        date=pay_date,
                        description=f"AP settlement for FX invoice {inv.invoice_id}",
                        debit_minor=orig_usd,
                        credit_minor=0,
                        reference_id=pay_id,
                        entry_id=entry_id,
                    )
                    # Cash is credited for actual payment amount in USD
                    r_cash = LedgerRow(
                        account_id="1000",
                        date=pay_date,
                        description=f"Cash FX disbursement for invoice {inv.invoice_id}",
                        debit_minor=0,
                        credit_minor=pay_usd,
                        reference_id=pay_id,
                        entry_id=entry_id,
                    )
                    ledger.append(r_ap)

                    if pay_usd > orig_usd:
                        # FX Loss (Debit 7000)
                        diff = pay_usd - orig_usd
                        r_fx = LedgerRow(
                            account_id="7000",
                            date=pay_date,
                            description=f"FX loss on invoice {inv.invoice_id}",
                            debit_minor=diff,
                            credit_minor=0,
                            reference_id=pay_id,
                            entry_id=entry_id,
                        )
                        ledger.append(r_fx)
                    elif pay_usd < orig_usd:
                        # FX Gain (Credit 7000)
                        diff = orig_usd - pay_usd
                        r_fx = LedgerRow(
                            account_id="7000",
                            date=pay_date,
                            description=f"FX gain on invoice {inv.invoice_id}",
                            debit_minor=0,
                            credit_minor=diff,
                            reference_id=pay_id,
                            entry_id=entry_id,
                        )
                        ledger.append(r_fx)

                    ledger.append(r_cash)

                pmt = Payment(
                    payment_id=pay_id,
                    invoice_id=inv.invoice_id,
                    vendor_id=inv.vendor_id,
                    payment_date=pay_date,
                    currency=inv.currency,
                    amount_minor=inv.total_minor,
                    bank_reference=f"WIRE-{self.rng.randint(100000, 999999)}",
                )
                payments.append(pmt)

                self.tracker.record_transaction(
                    transaction_id=txn_id,
                    tx_type="vendor_payment",
                    entity_ids=[pay_id, inv.invoice_id, inv.vendor_id],
                    ledger_entry_ids=[entry_id],
                    timestamp=pay_date,
                    metadata={"amount_minor": inv.total_minor, "currency": inv.currency},
                )

            # 2. Customer sales revenue and AR (Debit AR 1200, Credit Revenue 4000)
            elif tx_type_roll < 0.60:
                cust = self.rng.choice(customers)
                sale_id = f"SALE-{self.config.fiscal_year}-{idx + 1:04d}"
                entry_id = f"ENTRY-{entry_counter:05d}"
                entry_counter += 1
                txn_id = f"TXN-{txn_counter:05d}"
                txn_counter += 1

                amount_minor = self.rng.randint(20_000, 500_000)  # 200 - 5000 USD
                r1 = LedgerRow(
                    account_id="1200",
                    date=date_str,
                    description=f"Accounts receivable for {cust}",
                    debit_minor=amount_minor,
                    credit_minor=0,
                    reference_id=sale_id,
                    entry_id=entry_id,
                )
                r2 = LedgerRow(
                    account_id="4000",
                    date=date_str,
                    description=f"Operating sales revenue from {cust}",
                    debit_minor=0,
                    credit_minor=amount_minor,
                    reference_id=sale_id,
                    entry_id=entry_id,
                )
                ledger.extend([r1, r2])

                # Immediate or subsequent customer cash collection
                entry_id_receipt = f"ENTRY-{entry_counter:05d}"
                entry_counter += 1
                receipt_date = self._build_date(min(curr_day + self.rng.randint(1, 15), 364))
                rc1 = LedgerRow(
                    account_id="1000",
                    date=receipt_date,
                    description=f"Cash receipt from customer {cust}",
                    debit_minor=amount_minor,
                    credit_minor=0,
                    reference_id=sale_id,
                    entry_id=entry_id_receipt,
                )
                rc2 = LedgerRow(
                    account_id="1200",
                    date=receipt_date,
                    description=f"AR settlement from customer {cust}",
                    debit_minor=0,
                    credit_minor=amount_minor,
                    reference_id=sale_id,
                    entry_id=entry_id_receipt,
                )
                ledger.extend([rc1, rc2])

                self.tracker.record_transaction(
                    transaction_id=txn_id,
                    tx_type="customer_sale",
                    entity_ids=[sale_id, cust],
                    ledger_entry_ids=[entry_id, entry_id_receipt],
                    timestamp=date_str,
                    metadata={"amount_minor": amount_minor},
                )

            # 3. Vendor invoice (Domestic or FX) (Debit Expense 5000, Credit AP 2000)
            else:
                vendor = self.rng.choice(vendors)
                inv_id = f"INV-{self.config.fiscal_year}-{len(invoices) + 1:04d}"
                entry_id = f"ENTRY-{entry_counter:05d}"
                entry_counter += 1
                txn_id = f"TXN-{txn_counter:05d}"
                txn_counter += 1

                is_fx = self.rng.random() < 0.25
                inv_currency = (
                    self.rng.choice(["EUR", "GBP"]) if is_fx else self.config.base_currency
                )

                due_day = min(curr_day + 30, 364)
                due_date_str = self._build_date(due_day)

                # Generate 1-3 line items
                num_items = self.rng.randint(1, 3)
                line_items: list[InvoiceLineItem] = []
                inv_total = 0
                for item_idx in range(num_items):
                    qty = self.rng.randint(1, 10)
                    unit_price = self.rng.randint(5_000, 150_000)
                    item_total = qty * unit_price
                    inv_total += item_total
                    line_items.append(
                        InvoiceLineItem(
                            item_id=f"ITEM-{len(invoices) + 1:03d}-{item_idx + 1}",
                            description=f"Operational supply {item_idx + 1}",
                            quantity=qty,
                            unit_price_minor=unit_price,
                            total_minor=item_total,
                        )
                    )

                inv = Invoice(
                    invoice_id=inv_id,
                    vendor_id=vendor,
                    invoice_date=date_str,
                    due_date=due_date_str,
                    currency=inv_currency,
                    line_items=line_items,
                    total_minor=inv_total,
                    status="unpaid",
                )
                invoices.append(inv)
                unpaid_invoices.append(inv)

                # Ledger records in base currency
                if inv_currency == self.config.base_currency:
                    rec_amount = inv_total
                else:
                    fx_val = fx_lookup.get((date_str, inv_currency), "1.0850")
                    rec_amount = convert_currency(inv_total, fx_val)

                r1 = LedgerRow(
                    account_id="5000",
                    date=date_str,
                    description=f"Operating expense for invoice {inv_id}",
                    debit_minor=rec_amount,
                    credit_minor=0,
                    reference_id=inv_id,
                    entry_id=entry_id,
                )
                r2 = LedgerRow(
                    account_id="2000",
                    date=date_str,
                    description=f"Accounts payable for invoice {inv_id}",
                    debit_minor=0,
                    credit_minor=rec_amount,
                    reference_id=inv_id,
                    entry_id=entry_id,
                )
                ledger.extend([r1, r2])

                self.tracker.record_transaction(
                    transaction_id=txn_id,
                    tx_type="vendor_invoice",
                    entity_ids=[inv_id, vendor],
                    ledger_entry_ids=[entry_id],
                    timestamp=date_str,
                    metadata={"total_minor": inv_total, "currency": inv_currency},
                )

        manifest = Manifest(
            episode_id=f"EP-{self.config.fiscal_year}-{self.config.seed:04d}",
            company_name=company_name,
            fiscal_year=self.config.fiscal_year,
            currency=self.config.base_currency,
            initial_cash_minor=self.config.initial_cash_minor,
            total_transactions=len(self.tracker.transactions),
            injected_anomalies_count=0,
        )

        world = GeneratedWorld(
            manifest=manifest,
            ledger=ledger,
            invoices=invoices,
            payments=payments,
            fx_rates=fx_rates,
            provenance=self.tracker.to_dict(),
        )

        # Enforce mathematical double-entry invariants
        world.verify_double_entry_invariants()
        return world
