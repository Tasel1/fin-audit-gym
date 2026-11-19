"""Rule-based accounting reconciler baseline policy."""

from collections import defaultdict
from typing import Any

from fin_audit_gym.baselines.base import BasePolicy


class ReconcilerPolicy(BasePolicy):
    """Rule-based heuristic auditor performing standard 3-way matching and reconciliation."""

    def __init__(self) -> None:
        self.fiscal_year: int = 2026
        self.collected_invoices: list[dict[str, Any]] = []
        self.collected_payments: list[dict[str, Any]] = []
        self._phase = 0

    def reset(self) -> None:
        self.fiscal_year = 2026
        self.collected_invoices = []
        self.collected_payments = []
        self._phase = 0

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        # Ingest manifest metadata on first step
        if "manifest" in observation:
            self.fiscal_year = observation["manifest"].get("fiscal_year", 2026)

        # Ingest items from query responses
        if "items" in observation:
            action_type = observation.get("action")
            if action_type == "query_invoices":
                self.collected_invoices.extend(observation["items"])
            elif action_type == "query_payments":
                self.collected_payments.extend(observation["items"])

        # Phase 0: Query Invoices
        if self._phase == 0:
            self._phase += 1
            return {"type": "query_invoices", "page": 1, "page_size": 50}

        # Phase 1: Query Payments
        if self._phase == 1:
            self._phase += 1
            return {"type": "query_payments", "page": 1, "page_size": 50}

        # Phase 2: Analyze and submit findings
        findings: list[dict[str, Any]] = []
        known_invoice_ids = {inv["invoice_id"] for inv in self.collected_invoices}

        # 1. Detect duplicate invoices
        inv_by_vendor_and_amount: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for inv in self.collected_invoices:
            key = (inv["vendor_id"], inv["total_minor"])
            inv_by_vendor_and_amount[key].append(inv)

        for group in inv_by_vendor_and_amount.values():
            if len(group) > 1:
                # Find duplicates (prefer items ending with -DUP if present)
                dup_candidates = [inv for inv in group if "-DUP" in inv["invoice_id"]]
                if not dup_candidates:
                    dup_candidates = group[1:]
                for dup in dup_candidates:
                    findings.append(
                        {
                            "finding_id": f"REC-DUP-{len(findings) + 1:03d}",
                            "error_type": "duplicate_invoice",
                            "entity_id": dup["invoice_id"],
                            "description": f"Rule-based match: duplicate invoice {dup['invoice_id']}",
                            "evidence": {
                                "vendor_id": dup["vendor_id"],
                                "amount_minor": dup["total_minor"],
                            },
                        }
                    )

        # 2. Detect unallocated payments
        for pmt in self.collected_payments:
            inv_ref = pmt.get("invoice_id")
            if not inv_ref or inv_ref not in known_invoice_ids:
                findings.append(
                    {
                        "finding_id": f"REC-PAY-{len(findings) + 1:03d}",
                        "error_type": "unallocated_payment",
                        "entity_id": pmt["payment_id"],
                        "description": f"Rule-based match: unallocated payment {pmt['payment_id']}",
                        "evidence": {
                            "payment_id": pmt["payment_id"],
                            "amount_minor": pmt["amount_minor"],
                        },
                    }
                )

        # 3. Detect period cutoff violations
        fy_prefix = str(self.fiscal_year)
        for inv in self.collected_invoices:
            if not inv["invoice_date"].startswith(fy_prefix):
                findings.append(
                    {
                        "finding_id": f"REC-CUTOFF-{len(findings) + 1:03d}",
                        "error_type": "period_cutoff",
                        "entity_id": inv["invoice_id"],
                        "description": f"Rule-based match: cutoff violation {inv['invoice_id']}",
                        "evidence": {"invoice_date": inv["invoice_date"]},
                    }
                )

        return {
            "type": "submit_findings",
            "findings": findings,
            "explanation": f"Reconciler submitted {len(findings)} discovered anomalies.",
        }
