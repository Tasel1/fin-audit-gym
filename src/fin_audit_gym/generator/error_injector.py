"""Deterministic anomaly injection engine for financial audit scenarios."""

import copy
import random

from pydantic import BaseModel, ConfigDict, Field

from fin_audit_gym.core.schemas import (
    Finding,
    Invoice,
    InvoiceLineItem,
    LedgerRow,
    Payment,
)
from fin_audit_gym.generator.distractors import DistractorGenerator
from fin_audit_gym.generator.engine import GeneratedWorld


class AnomalyConfig(BaseModel):
    """Configuration specifying quantities and types of anomalies to inject."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, description="Random seed for deterministic injection")
    num_duplicate_invoices: int = Field(
        default=1, description="Number of duplicate invoice anomalies"
    )
    num_unallocated_payments: int = Field(default=1, description="Number of unallocated payments")
    num_fx_mismatches: int = Field(
        default=1, description="Number of FX conversion mismatch anomalies"
    )
    num_period_cutoffs: int = Field(default=1, description="Number of period cutoff violations")
    num_composite_round_trips: int = Field(
        default=1, description="Number of circular round-trip transactions"
    )
    num_composite_split_invoices: int = Field(
        default=1, description="Number of structured split invoice anomalies"
    )
    num_distractors: int = Field(
        default=3, description="Number of innocent look-alike distractors to inject"
    )


class AnomalyInjector:
    """Injects calibrated accounting anomalies and innocent distractors into a pristine world."""

    def __init__(self, config: AnomalyConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    def inject(self, world: GeneratedWorld) -> tuple[GeneratedWorld, list[Finding]]:
        """Inject configured anomalies and distractors, returning mutated world and ground truth findings."""
        # Deepcopy world components to avoid mutating source
        ledger = [row.model_copy() for row in world.ledger]
        invoices = [inv.model_copy(deep=True) for inv in world.invoices]
        payments = [pmt.model_copy() for pmt in world.payments]
        fx_rates = [fx.model_copy() for fx in world.fx_rates]
        provenance = copy.deepcopy(world.provenance)

        findings: list[Finding] = []
        finding_counter = 1
        entry_idx = len(ledger) + 100

        # 0. Inject legitimate distractors (NOT in truth findings)
        if self.config.num_distractors > 0:
            distractor_gen = DistractorGenerator(self.rng)
            entry_idx = distractor_gen.inject_recurring_retainer(
                ledger, invoices, payments, world.manifest.fiscal_year, entry_idx
            )
            entry_idx = distractor_gen.inject_partial_milestone_payment(
                ledger, invoices, payments, world.manifest.fiscal_year, entry_idx
            )
            entry_idx = distractor_gen.inject_legitimate_fx_rounding(
                ledger, world.manifest.fiscal_year, entry_idx
            )

        # 1. Duplicate Invoices
        for _ in range(self.config.num_duplicate_invoices):
            if not invoices:
                break
            orig = self.rng.choice(invoices)
            dup_id = f"{orig.invoice_id}-DUP"
            entry_id = f"ENTRY-ANOM-{entry_idx:05d}"
            entry_idx += 1

            dup_invoice = orig.model_copy(update={"invoice_id": dup_id, "status": "unpaid"})
            invoices.append(dup_invoice)

            # Record duplicate expense entry in ledger
            r1 = LedgerRow(
                account_id="5000",
                date=orig.invoice_date,
                description=f"Duplicate invoice entry for {dup_id}",
                debit_minor=orig.total_minor,
                credit_minor=0,
                reference_id=dup_id,
                entry_id=entry_id,
            )
            r2 = LedgerRow(
                account_id="2000",
                date=orig.invoice_date,
                description=f"Accounts payable for duplicate {dup_id}",
                debit_minor=0,
                credit_minor=orig.total_minor,
                reference_id=dup_id,
                entry_id=entry_id,
            )
            ledger.extend([r1, r2])

            findings.append(
                Finding(
                    finding_id=f"FINDING-{finding_counter:03d}",
                    error_type="duplicate_invoice",
                    entity_id=dup_id,
                    description=(
                        f"Duplicate invoice detected: {dup_id} duplicates invoice {orig.invoice_id} "
                        f"with vendor {orig.vendor_id} and amount {orig.total_minor}"
                    ),
                    evidence={
                        "duplicate_invoice_id": dup_id,
                        "original_invoice_id": orig.invoice_id,
                        "vendor_id": orig.vendor_id,
                        "amount_minor": orig.total_minor,
                    },
                )
            )
            finding_counter += 1

        # 2. Unallocated Payments
        for _ in range(self.config.num_unallocated_payments):
            pay_id = f"PAY-UNALLOC-{finding_counter:03d}"
            entry_id = f"ENTRY-ANOM-{entry_idx:05d}"
            entry_idx += 1
            unalloc_amt = self.rng.choice([150_000, 320_000, 480_000])
            vendor_id = f"VEND-GHOST-{self.rng.randint(100, 999)}"
            pay_date = f"{world.manifest.fiscal_year}-06-15"

            pmt = Payment(
                payment_id=pay_id,
                invoice_id=None,
                vendor_id=vendor_id,
                payment_date=pay_date,
                currency="USD",
                amount_minor=unalloc_amt,
                bank_reference=f"WIRE-UNALLOC-{self.rng.randint(100000, 999999)}",
            )
            payments.append(pmt)

            # Ledger row: Debit 2000 (unlinked AP/suspense), Credit 1000 (Cash)
            r1 = LedgerRow(
                account_id="2000",
                date=pay_date,
                description=f"Unallocated vendor disbursement {pay_id}",
                debit_minor=unalloc_amt,
                credit_minor=0,
                reference_id=pay_id,
                entry_id=entry_id,
            )
            r2 = LedgerRow(
                account_id="1000",
                date=pay_date,
                description=f"Cash disbursement for unallocated payment {pay_id}",
                debit_minor=0,
                credit_minor=unalloc_amt,
                reference_id=pay_id,
                entry_id=entry_id,
            )
            ledger.extend([r1, r2])

            findings.append(
                Finding(
                    finding_id=f"FINDING-{finding_counter:03d}",
                    error_type="unallocated_payment",
                    entity_id=pay_id,
                    description=f"Unallocated cash payment {pay_id} without supporting invoice reference",
                    evidence={
                        "payment_id": pay_id,
                        "amount_minor": unalloc_amt,
                        "vendor_id": vendor_id,
                        "bank_reference": pmt.bank_reference,
                    },
                )
            )
            finding_counter += 1

        # 3. FX Conversion Mismatch
        for _ in range(self.config.num_fx_mismatches):
            # Target an invoice with foreign currency or create one
            fx_invs = [inv for inv in invoices if inv.currency in ("EUR", "GBP")]
            if not fx_invs:
                target_inv = invoices[0]
                target_inv.currency = "EUR"
            else:
                target_inv = fx_invs[0]

            expected_amount = int(target_inv.total_minor * 1.085)
            # Inject 25% discrepancy in recorded amount
            recorded_amount = int(expected_amount * 1.25)
            entry_id = f"ENTRY-ANOM-{entry_idx:05d}"
            entry_idx += 1

            r1 = LedgerRow(
                account_id="5000",
                date=target_inv.invoice_date,
                description=f"Operating expense for FX invoice {target_inv.invoice_id}",
                debit_minor=recorded_amount,
                credit_minor=0,
                reference_id=target_inv.invoice_id,
                entry_id=entry_id,
            )
            r2 = LedgerRow(
                account_id="2000",
                date=target_inv.invoice_date,
                description=f"Accounts payable for FX invoice {target_inv.invoice_id}",
                debit_minor=0,
                credit_minor=recorded_amount,
                reference_id=target_inv.invoice_id,
                entry_id=entry_id,
            )
            ledger.extend([r1, r2])

            findings.append(
                Finding(
                    finding_id=f"FINDING-{finding_counter:03d}",
                    error_type="fx_mismatch",
                    entity_id=target_inv.invoice_id,
                    description=(
                        f"Incorrect foreign exchange conversion applied to invoice {target_inv.invoice_id}: "
                        f"recorded {recorded_amount} minor units vs expected {expected_amount}"
                    ),
                    evidence={
                        "invoice_id": target_inv.invoice_id,
                        "currency": target_inv.currency,
                        "recorded_amount_minor": recorded_amount,
                        "expected_amount_minor": expected_amount,
                    },
                )
            )
            finding_counter += 1

        # 4. Period Cutoff Violation
        for _ in range(self.config.num_period_cutoffs):
            cutoff_inv_id = f"INV-{world.manifest.fiscal_year}-CUTOFF"
            entry_id = f"ENTRY-ANOM-{entry_idx:05d}"
            entry_idx += 1
            next_year = world.manifest.fiscal_year + 1
            improper_date = f"{next_year}-01-05"  # Belongs to next fiscal year!

            cutoff_inv = Invoice(
                invoice_id=cutoff_inv_id,
                vendor_id="VEND-CUTOFF-01",
                invoice_date=improper_date,
                due_date=f"{next_year}-02-05",
                currency="USD",
                line_items=[
                    InvoiceLineItem(
                        item_id=f"ITEM-{cutoff_inv_id}-1",
                        description="Next year consulting project advance",
                        quantity=1,
                        unit_price_minor=200_000,
                        total_minor=200_000,
                    )
                ],
                total_minor=200_000,
                status="unpaid",
            )
            invoices.append(cutoff_inv)

            # Improperly recorded in current fiscal year general ledger
            r1 = LedgerRow(
                account_id="5000",
                date=f"{world.manifest.fiscal_year}-12-31",
                description=f"Prematurely recognized expense for {cutoff_inv_id}",
                debit_minor=200_000,
                credit_minor=0,
                reference_id=cutoff_inv_id,
                entry_id=entry_id,
            )
            r2 = LedgerRow(
                account_id="2000",
                date=f"{world.manifest.fiscal_year}-12-31",
                description=f"AP recognized before service date for {cutoff_inv_id}",
                debit_minor=0,
                credit_minor=200_000,
                reference_id=cutoff_inv_id,
                entry_id=entry_id,
            )
            ledger.extend([r1, r2])

            findings.append(
                Finding(
                    finding_id=f"FINDING-{finding_counter:03d}",
                    error_type="period_cutoff",
                    entity_id=cutoff_inv_id,
                    description=(
                        f"Period cutoff violation: invoice {cutoff_inv_id} dated {improper_date} "
                        f"improperly booked into fiscal year {world.manifest.fiscal_year}"
                    ),
                    evidence={
                        "invoice_id": cutoff_inv_id,
                        "invoice_date": improper_date,
                        "booked_date": f"{world.manifest.fiscal_year}-12-31",
                        "fiscal_year": world.manifest.fiscal_year,
                    },
                )
            )
            finding_counter += 1

        # 5. Composite Round-Trip Transactions
        for _ in range(self.config.num_composite_round_trips):
            rt_id = f"RT-{world.manifest.fiscal_year}-{finding_counter:03d}"
            vendor_a = "VEND-SHELL-A"
            vendor_b = "VEND-SHELL-B"
            rt_amount = 500_000  # 5,000 USD
            rt_date = f"{world.manifest.fiscal_year}-08-20"

            entry_out = f"ENTRY-ANOM-{entry_idx:05d}"
            entry_idx += 1
            entry_in = f"ENTRY-ANOM-{entry_idx:05d}"
            entry_idx += 1

            # Outbound sham payment to Vendor A
            ledger.append(
                LedgerRow(
                    account_id="5000",
                    date=rt_date,
                    description=f"Advisory fee to {vendor_a} for strategic evaluation",
                    debit_minor=rt_amount,
                    credit_minor=0,
                    reference_id=rt_id,
                    entry_id=entry_out,
                )
            )
            ledger.append(
                LedgerRow(
                    account_id="1000",
                    date=rt_date,
                    description=f"Cash payment to {vendor_a}",
                    debit_minor=0,
                    credit_minor=rt_amount,
                    reference_id=rt_id,
                    entry_id=entry_out,
                )
            )

            # Inbound return payment from Vendor B on same day
            ledger.append(
                LedgerRow(
                    account_id="1000",
                    date=rt_date,
                    description=f"Cash receipt from {vendor_b} for market research rebate",
                    debit_minor=rt_amount,
                    credit_minor=0,
                    reference_id=rt_id,
                    entry_id=entry_in,
                )
            )
            ledger.append(
                LedgerRow(
                    account_id="4000",
                    date=rt_date,
                    description=f"Other income from {vendor_b}",
                    debit_minor=0,
                    credit_minor=rt_amount,
                    reference_id=rt_id,
                    entry_id=entry_in,
                )
            )

            findings.append(
                Finding(
                    finding_id=f"FINDING-{finding_counter:03d}",
                    error_type="composite_round_trip",
                    entity_id=rt_id,
                    description=(
                        f"Round-trip transaction without economic substance detected: "
                        f"outflow to {vendor_a} matched by inflow from {vendor_b} for {rt_amount} minor units"
                    ),
                    evidence={
                        "round_trip_id": rt_id,
                        "vendor_a": vendor_a,
                        "vendor_b": vendor_b,
                        "amount_minor": rt_amount,
                        "date": rt_date,
                    },
                )
            )
            finding_counter += 1

        # 6. Composite Structured Split Invoices
        for _ in range(self.config.num_composite_split_invoices):
            split_id = f"SPLIT-{world.manifest.fiscal_year}-{finding_counter:03d}"
            split_vendor = "VEND-STRUCT-01"
            split_date = f"{world.manifest.fiscal_year}-09-10"

            sub_amounts = [850_000, 850_000, 800_000]  # Total: 2,500,000 (Threshold $25,000)
            sub_inv_ids: list[str] = []

            for sub_i, amt in enumerate(sub_amounts, start=1):
                sub_inv_id = f"INV-SPLIT-{finding_counter:02d}-{sub_i}"
                sub_inv_ids.append(sub_inv_id)
                entry_id = f"ENTRY-ANOM-{entry_idx:05d}"
                entry_idx += 1

                inv = Invoice(
                    invoice_id=sub_inv_id,
                    vendor_id=split_vendor,
                    invoice_date=split_date,
                    due_date=f"{world.manifest.fiscal_year}-10-10",
                    currency="USD",
                    line_items=[
                        InvoiceLineItem(
                            item_id=f"ITEM-{sub_inv_id}-1",
                            description=f"Specialized infrastructure services slice {sub_i}",
                            quantity=1,
                            unit_price_minor=amt,
                            total_minor=amt,
                        )
                    ],
                    total_minor=amt,
                    status="unpaid",
                )
                invoices.append(inv)

                ledger.append(
                    LedgerRow(
                        account_id="5000",
                        date=split_date,
                        description=f"Infrastructure slice {sub_i} for {sub_inv_id}",
                        debit_minor=amt,
                        credit_minor=0,
                        reference_id=sub_inv_id,
                        entry_id=entry_id,
                    )
                )
                ledger.append(
                    LedgerRow(
                        account_id="2000",
                        date=split_date,
                        description=f"AP slice {sub_i} for {sub_inv_id}",
                        debit_minor=0,
                        credit_minor=amt,
                        reference_id=sub_inv_id,
                        entry_id=entry_id,
                    )
                )

            findings.append(
                Finding(
                    finding_id=f"FINDING-{finding_counter:03d}",
                    error_type="composite_split_invoice",
                    entity_id=split_id,
                    description=(
                        f"Structured split invoices detected: 3 invoices ({', '.join(sub_inv_ids)}) "
                        f"issued to vendor {split_vendor} below the $25,000 approval limit totaling 2500000 minor units"
                    ),
                    evidence={
                        "split_id": split_id,
                        "vendor_id": split_vendor,
                        "sub_invoice_ids": sub_inv_ids,
                        "total_amount_minor": 2_500_000,
                        "individual_amounts_minor": sub_amounts,
                    },
                )
            )
            finding_counter += 1

        mutated_manifest = world.manifest.model_copy(
            update={
                "injected_anomalies_count": len(findings),
                "total_transactions": len(ledger) // 2,
            }
        )

        mutated_world = GeneratedWorld(
            manifest=mutated_manifest,
            ledger=ledger,
            invoices=invoices,
            payments=payments,
            fx_rates=fx_rates,
            provenance=provenance,
        )

        return mutated_world, findings
