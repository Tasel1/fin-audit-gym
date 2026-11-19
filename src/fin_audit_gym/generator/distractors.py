"""Distractor generator injecting innocent but suspicious-looking financial events."""

import datetime
import random

from fin_audit_gym.core.schemas import (
    Invoice,
    InvoiceLineItem,
    LedgerRow,
    Payment,
)


class DistractorGenerator:
    """Injects legitimate, benign accounting patterns that trigger naive heuristics.

    These are intentionally innocent 'look-alikes':
    1. Recurring retainer payments (identical amount, same vendor, exactly 30 days apart).
    2. Legitimate split/partial payments (single invoice paid across multiple milestones).
    3. Legitimate 1-cent FX currency rounding differences balanced in account 7000.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def _format_date(self, fiscal_year: int, day_offset: int) -> str:
        safe_day = min(max(day_offset, 0), 364)
        base = datetime.date(fiscal_year, 1, 1)
        return (base + datetime.timedelta(days=safe_day)).isoformat()

    def inject_recurring_retainer(
        self,
        ledger: list[LedgerRow],
        invoices: list[Invoice],
        payments: list[Payment],
        fiscal_year: int,
        entry_idx: int,
    ) -> int:
        """Inject 2-3 identical monthly retainers with legitimate separate invoices and payments."""
        vendor_id = f"VEND-RETAINER-{self.rng.randint(100, 999)}"
        retainer_amount = self.rng.choice([50_000, 100_000, 250_000])  # 500, 1000, 2500 USD
        start_day = self.rng.randint(20, 120)

        for month in range(3):
            inv_day = start_day + month * 30
            inv_date = self._format_date(fiscal_year, inv_day)
            pay_date = self._format_date(fiscal_year, inv_day + 5)
            inv_id = f"INV-{fiscal_year}-RET-{entry_idx:04d}"
            pay_id = f"PAY-{fiscal_year}-RET-{entry_idx:04d}"
            entry_inv = f"ENTRY-DIST-{entry_idx:05d}"
            entry_idx += 1
            entry_pay = f"ENTRY-DIST-{entry_idx:05d}"
            entry_idx += 1

            # Invoice
            inv = Invoice(
                invoice_id=inv_id,
                vendor_id=vendor_id,
                invoice_date=inv_date,
                due_date=self._format_date(fiscal_year, inv_day + 30),
                currency="USD",
                line_items=[
                    InvoiceLineItem(
                        item_id=f"ITEM-{inv_id}-1",
                        description=f"Monthly legal and advisory retainer month {month + 1}",
                        quantity=1,
                        unit_price_minor=retainer_amount,
                        total_minor=retainer_amount,
                    )
                ],
                total_minor=retainer_amount,
                status="paid",
            )
            invoices.append(inv)

            # Invoice ledger rows (Debit Expense 5000, Credit AP 2000)
            ledger.append(
                LedgerRow(
                    account_id="5000",
                    date=inv_date,
                    description=f"Legitimate monthly advisory retainer {month + 1}",
                    debit_minor=retainer_amount,
                    credit_minor=0,
                    reference_id=inv_id,
                    entry_id=entry_inv,
                )
            )
            ledger.append(
                LedgerRow(
                    account_id="2000",
                    date=inv_date,
                    description=f"Accounts payable for retainer {inv_id}",
                    debit_minor=0,
                    credit_minor=retainer_amount,
                    reference_id=inv_id,
                    entry_id=entry_inv,
                )
            )

            # Payment
            pmt = Payment(
                payment_id=pay_id,
                invoice_id=inv_id,
                vendor_id=vendor_id,
                payment_date=pay_date,
                currency="USD",
                amount_minor=retainer_amount,
                bank_reference=f"ACH-RET-{self.rng.randint(100000, 999999)}",
            )
            payments.append(pmt)

            # Payment ledger rows (Debit AP 2000, Credit Cash 1000)
            ledger.append(
                LedgerRow(
                    account_id="2000",
                    date=pay_date,
                    description=f"Payment settlement for retainer {inv_id}",
                    debit_minor=retainer_amount,
                    credit_minor=0,
                    reference_id=pay_id,
                    entry_id=entry_pay,
                )
            )
            ledger.append(
                LedgerRow(
                    account_id="1000",
                    date=pay_date,
                    description=f"Cash disbursement for retainer {inv_id}",
                    debit_minor=0,
                    credit_minor=retainer_amount,
                    reference_id=pay_id,
                    entry_id=entry_pay,
                )
            )

        return entry_idx

    def inject_partial_milestone_payment(
        self,
        ledger: list[LedgerRow],
        invoices: list[Invoice],
        payments: list[Payment],
        fiscal_year: int,
        entry_idx: int,
    ) -> int:
        """Inject legitimate split installment payment settling a single large milestone invoice."""
        vendor_id = f"VEND-CONTRACT-{self.rng.randint(100, 999)}"
        inv_day = self.rng.randint(30, 200)
        inv_date = self._format_date(fiscal_year, inv_day)
        inv_id = f"INV-{fiscal_year}-PART-{entry_idx:04d}"
        entry_inv = f"ENTRY-DIST-{entry_idx:05d}"
        entry_idx += 1

        total_amount = 300_000  # 3,000 USD
        part1 = 180_000  # Milestone 1 (60%)
        part2 = 120_000  # Milestone 2 (40%)

        inv = Invoice(
            invoice_id=inv_id,
            vendor_id=vendor_id,
            invoice_date=inv_date,
            due_date=self._format_date(fiscal_year, inv_day + 60),
            currency="USD",
            line_items=[
                InvoiceLineItem(
                    item_id=f"ITEM-{inv_id}-1",
                    description="Custom software implementation milestone deliverables",
                    quantity=1,
                    unit_price_minor=total_amount,
                    total_minor=total_amount,
                )
            ],
            total_minor=total_amount,
            status="paid",
        )
        invoices.append(inv)

        # Invoice ledger rows
        ledger.append(
            LedgerRow(
                account_id="5000",
                date=inv_date,
                description=f"Software milestone deliverables {inv_id}",
                debit_minor=total_amount,
                credit_minor=0,
                reference_id=inv_id,
                entry_id=entry_inv,
            )
        )
        ledger.append(
            LedgerRow(
                account_id="2000",
                date=inv_date,
                description=f"Accounts payable for milestone {inv_id}",
                debit_minor=0,
                credit_minor=total_amount,
                reference_id=inv_id,
                entry_id=entry_inv,
            )
        )

        # Installment 1
        pay1_date = self._format_date(fiscal_year, inv_day + 10)
        pay1_id = f"PAY-{fiscal_year}-P1-{entry_idx:04d}"
        entry_p1 = f"ENTRY-DIST-{entry_idx:05d}"
        entry_idx += 1
        payments.append(
            Payment(
                payment_id=pay1_id,
                invoice_id=inv_id,
                vendor_id=vendor_id,
                payment_date=pay1_date,
                currency="USD",
                amount_minor=part1,
                bank_reference=f"WIRE-P1-{self.rng.randint(100000, 999999)}",
            )
        )
        ledger.append(
            LedgerRow(
                account_id="2000",
                date=pay1_date,
                description=f"Installment 1 (60%) for {inv_id}",
                debit_minor=part1,
                credit_minor=0,
                reference_id=pay1_id,
                entry_id=entry_p1,
            )
        )
        ledger.append(
            LedgerRow(
                account_id="1000",
                date=pay1_date,
                description=f"Cash payment installment 1 for {inv_id}",
                debit_minor=0,
                credit_minor=part1,
                reference_id=pay1_id,
                entry_id=entry_p1,
            )
        )

        # Installment 2
        pay2_date = self._format_date(fiscal_year, inv_day + 40)
        pay2_id = f"PAY-{fiscal_year}-P2-{entry_idx:04d}"
        entry_p2 = f"ENTRY-DIST-{entry_idx:05d}"
        entry_idx += 1
        payments.append(
            Payment(
                payment_id=pay2_id,
                invoice_id=inv_id,
                vendor_id=vendor_id,
                payment_date=pay2_date,
                currency="USD",
                amount_minor=part2,
                bank_reference=f"WIRE-P2-{self.rng.randint(100000, 999999)}",
            )
        )
        ledger.append(
            LedgerRow(
                account_id="2000",
                date=pay2_date,
                description=f"Installment 2 (40% final) for {inv_id}",
                debit_minor=part2,
                credit_minor=0,
                reference_id=pay2_id,
                entry_id=entry_p2,
            )
        )
        ledger.append(
            LedgerRow(
                account_id="1000",
                date=pay2_date,
                description=f"Cash payment installment 2 for {inv_id}",
                debit_minor=0,
                credit_minor=part2,
                reference_id=pay2_id,
                entry_id=entry_p2,
            )
        )

        return entry_idx

    def inject_legitimate_fx_rounding(
        self,
        ledger: list[LedgerRow],
        fiscal_year: int,
        entry_idx: int,
    ) -> int:
        """Inject legitimate 1-cent FX roundoff balanced in account 7000."""
        date_str = self._format_date(fiscal_year, self.rng.randint(50, 300))
        entry_id = f"ENTRY-DIST-{entry_idx:05d}"
        entry_idx += 1

        # 1-cent rounding adjustment between foreign intercompany reconciliation
        ledger.append(
            LedgerRow(
                account_id="7000",
                date=date_str,
                description="Legitimate rounding reconciliation adjustment in foreign currency settlement",
                debit_minor=1,
                credit_minor=0,
                reference_id=None,
                entry_id=entry_id,
            )
        )
        ledger.append(
            LedgerRow(
                account_id="1000",
                date=date_str,
                description="Cash balancing adjustment for exchange fraction",
                debit_minor=0,
                credit_minor=1,
                reference_id=None,
                entry_id=entry_id,
            )
        )
        return entry_idx
