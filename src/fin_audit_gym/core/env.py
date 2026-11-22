"""Gymnasium-style reinforcement learning environment for financial audit simulation."""

import csv
import json
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fin_audit_gym.core.actions import AVAILABLE_ACTION_SCHEMAS, ActionType, validate_action
from fin_audit_gym.core.calculator import safe_calculate
from fin_audit_gym.core.exceptions import EpisodeDoneError
from fin_audit_gym.core.schemas import (
    Finding,
    FXRate,
    Invoice,
    LedgerRow,
    Manifest,
    Payment,
    Policy,
    Submission,
)
from fin_audit_gym.generator.engine import GeneratedWorld


def _parse_pagination(action: dict[str, Any]) -> tuple[int, int]:
    """Safely extract and sanitize page and page_size from action dictionary."""
    try:
        raw_page = action.get("page", 1)
        page = max(1, int(raw_page))
    except (ValueError, TypeError, OverflowError):
        page = 1

    try:
        raw_size = action.get("page_size", 20)
        page_size = min(max(1, int(raw_size)), 50)
    except (ValueError, TypeError, OverflowError):
        page_size = 20

    return page, page_size


class FinAuditEnv:
    """Gymnasium-compatible Financial Audit RL Environment.

    Provides discrete investigative action queries over double-entry accounting records,
    safe calculator sandbox, and terminal submission evaluation with zero benchmark leakage.
    Supports both Section 04 canonical actions (list_records, get_record, get_document, calculator, submit)
    and legacy query actions (query_ledger, query_invoices, query_payments, query_fx, calculate, submit_findings).
    """

    def __init__(self, data_dir: str | Path | None = None, max_steps: int = 100) -> None:
        self.data_dir = Path(data_dir) if data_dir else None
        self.max_steps = max_steps
        self._reward_engine: Any = None

        self._steps_taken = 0
        self._total_queries = 0
        self._terminated = False
        self._truncated = False

        self._world: GeneratedWorld | None = None
        self._policy: Policy | None = None
        self._ground_truth_findings: list[Finding] = []

    @property
    def reward_engine(self) -> Any:
        if self._reward_engine is None:
            from fin_audit_gym.evaluator.reward import RewardEngine

            self._reward_engine = RewardEngine()
        return self._reward_engine

    def _load_from_data_dir(self, dir_path: Path) -> GeneratedWorld:
        """Load financial universe and ground truth from existing exported directory."""
        manifest_file = dir_path / "manifest.json"
        manifest = Manifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))

        policy_file = dir_path / "policy.json"
        if policy_file.is_file():
            self._policy = Policy.model_validate_json(policy_file.read_text(encoding="utf-8"))
        else:
            self._policy = Policy(
                episode_id=manifest.episode_id,
                period_start=manifest.period_start,
                period_end=manifest.period_end,
                timezone=manifest.timezone,
            )

        ledger: list[LedgerRow] = []
        ledger_file = dir_path / "ledger.csv"
        with ledger_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry_id = row.get("row_id") or row.get("entry_id") or ""
                account_id = row.get("account") or row.get("account_id") or ""
                date_str = row.get("timestamp") or row.get("date") or ""
                currency = row.get("currency") or "USD"
                ref_id = row.get("invoice_id") or row.get("payment_id") or row.get("reference_id")
                entity_id = row.get("entity_id")
                ledger.append(
                    LedgerRow(
                        row_id=entry_id,
                        entry_id=entry_id,
                        account=account_id,
                        account_id=account_id,
                        timestamp=date_str,
                        date=date_str.split("T")[0] if "T" in date_str else date_str,
                        description=row.get("description", ""),
                        debit_minor=int(row["debit_minor"]),
                        credit_minor=int(row["credit_minor"]),
                        currency=currency,
                        entity_id=entity_id,
                        invoice_id=row.get("invoice_id"),
                        payment_id=row.get("payment_id"),
                        reference_id=ref_id if ref_id else None,
                    )
                )

        invoices: list[Invoice] = []
        invoices_file = dir_path / "invoices.jsonl"
        with invoices_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    invoices.append(Invoice.model_validate_json(line))

        payments: list[Payment] = []
        payments_file = dir_path / "payments.jsonl"
        with payments_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    payments.append(Payment.model_validate_json(line))

        fx_rates: list[FXRate] = []
        fx_file = dir_path / "fx.csv"
        with fx_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src_curr = row.get("source_currency") or row.get("base_currency") or "USD"
                tgt_curr = row.get("target_currency") or row.get("quote_currency") or "EUR"
                fx_rates.append(
                    FXRate(
                        date=row["date"],
                        source_currency=src_curr,
                        target_currency=tgt_curr,
                        base_currency=src_curr,
                        quote_currency=tgt_curr,
                        rate=row["rate"],
                    )
                )

        provenance: dict[str, Any] = {}
        prov_file = dir_path / "provenance.json"
        if prov_file.is_file():
            provenance = json.loads(prov_file.read_text(encoding="utf-8"))

        truth_file = dir_path / "truth.json"
        if truth_file.is_file():
            raw_truth = json.loads(truth_file.read_text(encoding="utf-8"))
            self._ground_truth_findings = [Finding.model_validate(item) for item in raw_truth]
        else:
            self._ground_truth_findings = []

        return GeneratedWorld(
            manifest=manifest,
            ledger=ledger,
            invoices=invoices,
            payments=payments,
            fx_rates=fx_rates,
            provenance=provenance,
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state, generating or reloading the episode universe."""
        self._steps_taken = 0
        self._total_queries = 0
        self._terminated = False
        self._truncated = False

        if self.data_dir and self.data_dir.is_dir():
            self._world = self._load_from_data_dir(self.data_dir)
        else:
            from fin_audit_gym.generator import (
                AnomalyConfig,
                AnomalyInjector,
                GeneratorConfig,
                SyntheticLedgerGenerator,
            )

            effective_seed = seed if seed is not None else 42
            gen_cfg = GeneratorConfig(seed=effective_seed)
            clean_world = SyntheticLedgerGenerator(gen_cfg).generate()

            anom_cfg = AnomalyConfig(seed=effective_seed)
            self._world, self._ground_truth_findings = AnomalyInjector(anom_cfg).inject(clean_world)
            self._policy = Policy(
                episode_id=self._world.manifest.episode_id,
                period_start=self._world.manifest.period_start,
                period_end=self._world.manifest.period_end,
                timezone=self._world.manifest.timezone,
            )

        manifest = self._world.manifest

        # ZERO DATA LEAKAGE: Mask injected_anomalies_count from agent observation
        obs: dict[str, Any] = {
            "task": (
                "Conduct an independent financial audit of the general ledger and documents "
                "for the reporting period. Discover and report all accounting discrepancies, "
                "cutoff violations, unallocated cash disbursements, and duplicate invoices."
            ),
            "manifest": {
                "schema_version": manifest.schema_version,
                "episode_id": manifest.episode_id,
                "company_name": manifest.company_name,
                "fiscal_year": manifest.fiscal_year,
                "base_currency": manifest.base_currency,
                "currency": manifest.currency,
                "period_start": manifest.period_start,
                "period_end": manifest.period_end,
                "timezone": manifest.timezone,
                "initial_cash_minor": manifest.initial_cash_minor,
                "total_records": manifest.total_records,
                "total_transactions": manifest.total_transactions,
            },
            "policy": {
                "fx_date_policy": self._policy.fx_date_policy
                if self._policy
                else "transaction_date",
                "period_start": manifest.period_start,
                "period_end": manifest.period_end,
                "timezone": manifest.timezone,
                "rounding_rule": manifest.rounding_rule,
                "amount_tolerance_minor": 100,
            },
            "remaining_budget": self.max_steps,
            "available_actions": AVAILABLE_ACTION_SCHEMAS,
        }

        info = {
            "steps_taken": 0,
            "max_steps": self.max_steps,
            "remaining_steps": self.max_steps,
            "episode_id": manifest.episode_id,
            "current_page": 1,
            "total_queries": 0,
        }

        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one investigative step in the financial audit environment.

        Returns:
            observation (dict): Result of query, calculation, or submission.
            reward (float): Step reward (0.0 for exploration, terminal evaluated reward on submit).
            terminated (bool): True if agent called submit / submit_findings.
            truncated (bool): True if steps_taken >= max_steps.
            info (dict): Diagnostic info dictionary.
        """
        if self._terminated or self._truncated:
            raise EpisodeDoneError(
                "Episode has concluded. Call reset() before calling step() again."
            )

        validated_action = validate_action(action)
        self._steps_taken += 1
        if self._steps_taken >= self.max_steps:
            self._truncated = True

        action_type = validated_action["type"]
        obs: dict[str, Any] = {}
        reward = 0.0

        if self._world is None:
            self.reset()
            assert self._world is not None

        # =====================================================================
        # 1. CANONICAL SECTION 04 ACTIONS
        # =====================================================================

        # list_records(table, cursor, limit)
        if action_type == ActionType.LIST_RECORDS.value:
            self._total_queries += 1
            table = str(validated_action.get("table", "ledger")).lower()
            raw_limit = validated_action.get("limit", 20)
            try:
                limit = min(max(1, int(raw_limit)), 50)  # Maximum 50 rows per Section 36
            except (ValueError, TypeError):
                limit = 20

            cursor_val = validated_action.get("cursor", "0")
            try:
                offset = max(0, int(cursor_val) if cursor_val is not None else 0)
            except (ValueError, TypeError):
                offset = 0

            source_items: list[dict[str, Any]] = []
            if table == "ledger":
                source_items = [r.model_dump() for r in self._world.ledger]
            elif table == "invoices":
                source_items = [inv.model_dump() for inv in self._world.invoices]
            elif table == "payments":
                source_items = [p.model_dump() for p in self._world.payments]
            elif table == "fx":
                source_items = [fx.model_dump() for fx in self._world.fx_rates]
            else:
                obs = {
                    "action": action_type,
                    "error": f"Unknown table '{table}'. Valid tables: ['ledger', 'invoices', 'payments', 'fx']",
                }

            if "error" not in obs:
                slice_items = source_items[offset : offset + limit]
                has_more = (offset + limit) < len(source_items)
                next_cursor = str(offset + limit) if has_more else None
                obs = {
                    "action": action_type,
                    "table": table,
                    "items": slice_items,
                    "cursor": str(offset),
                    "next_cursor": next_cursor,
                    "has_more": has_more,
                    "limit": limit,
                    "total_count": len(source_items),
                }

        # get_record(table, id)
        elif action_type == ActionType.GET_RECORD.value:
            self._total_queries += 1
            table = str(validated_action.get("table", "ledger")).lower()
            rec_id = str(validated_action.get("id") or validated_action.get("record_id") or "")
            found_rec = None

            if table == "ledger":
                for r in self._world.ledger:
                    if r.row_id == rec_id or r.entry_id == rec_id:
                        found_rec = r.model_dump()
                        break
            elif table == "invoices":
                for inv in self._world.invoices:
                    if inv.invoice_id == rec_id or inv.doc_id == rec_id:
                        found_rec = inv.model_dump()
                        break
            elif table == "payments":
                for pmt in self._world.payments:
                    if pmt.payment_id == rec_id or pmt.doc_id == rec_id:
                        found_rec = pmt.model_dump()
                        break
            elif table == "fx":
                for fx in self._world.fx_rates:
                    if fx.date == rec_id:
                        found_rec = fx.model_dump()
                        break
            else:
                obs = {
                    "action": action_type,
                    "error": f"Unknown table '{table}'. Valid tables: ['ledger', 'invoices', 'payments', 'fx']",
                }

            if found_rec is not None:
                obs = {"action": action_type, "table": table, "id": rec_id, "item": found_rec}
            elif "error" not in obs:
                obs = {
                    "action": action_type,
                    "table": table,
                    "id": rec_id,
                    "error": f"Record '{rec_id}' not found in table '{table}'",
                }

        # get_document(doc_id)
        elif action_type == ActionType.GET_DOCUMENT.value:
            self._total_queries += 1
            doc_id = str(validated_action.get("doc_id") or validated_action.get("id") or "")
            found_doc = None

            for inv in self._world.invoices:
                if inv.doc_id == doc_id or inv.invoice_id == doc_id:
                    found_doc = {"document_type": "invoice", "data": inv.model_dump()}
                    break

            if found_doc is None:
                for pmt in self._world.payments:
                    if pmt.doc_id == doc_id or pmt.payment_id == doc_id:
                        found_doc = {"document_type": "payment", "data": pmt.model_dump()}
                        break

            if found_doc is not None:
                obs = {"action": action_type, "doc_id": doc_id, "document": found_doc}
            else:
                obs = {
                    "action": action_type,
                    "doc_id": doc_id,
                    "error": f"Document '{doc_id}' not found in episode",
                }

        # calculator(expression)
        elif action_type in (ActionType.CALCULATOR.value, ActionType.CALCULATE.value):
            expression = str(validated_action.get("expression", ""))
            try:
                calc_res = safe_calculate(expression)
                obs = {
                    "action": action_type,
                    "expression": expression,
                    "result": calc_res,
                }
            except ValueError as exc:
                obs = {
                    "action": action_type,
                    "expression": expression,
                    "result": None,
                    "error": str(exc),
                }

        # =====================================================================
        # 2. LEGACY / COMPATIBILITY QUERY ACTIONS
        # =====================================================================

        # query_ledger
        elif action_type == ActionType.QUERY_LEDGER.value:
            self._total_queries += 1
            account_id = validated_action.get("account_id") or validated_action.get("account")
            date_from = validated_action.get("date_from")
            date_to = validated_action.get("date_to")
            ref_id = validated_action.get("reference_id")

            filtered = self._world.ledger
            if account_id:
                filtered = [
                    r for r in filtered if r.account_id == account_id or r.account == account_id
                ]
            if date_from is not None:
                str_date_from = str(date_from)
                filtered = [
                    r for r in filtered if r.date >= str_date_from or r.timestamp >= str_date_from
                ]
            if date_to is not None:
                str_date_to = str(date_to)
                filtered = [
                    r for r in filtered if r.date <= str_date_to or r.timestamp <= str_date_to
                ]
            if ref_id:
                filtered = [
                    r
                    for r in filtered
                    if r.reference_id == ref_id or r.invoice_id == ref_id or r.payment_id == ref_id
                ]

            page, page_size = _parse_pagination(validated_action)
            start_i = (page - 1) * page_size
            page_rows = filtered[start_i : start_i + page_size]
            total_pages = max(1, math.ceil(len(filtered) / page_size))

            obs = {
                "action": action_type,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total_count": len(filtered),
                "items": [r.model_dump() for r in page_rows],
            }

        # query_invoices
        elif action_type == ActionType.QUERY_INVOICES.value:
            self._total_queries += 1
            vendor_id = validated_action.get("vendor_id") or validated_action.get("entity_id")
            status = validated_action.get("status")

            filtered_inv = self._world.invoices
            if vendor_id:
                filtered_inv = [
                    inv
                    for inv in filtered_inv
                    if inv.vendor_id == vendor_id or inv.entity_id == vendor_id
                ]
            if status:
                filtered_inv = [inv for inv in filtered_inv if inv.status == status]

            page, page_size = _parse_pagination(validated_action)
            start_i = (page - 1) * page_size
            page_items = filtered_inv[start_i : start_i + page_size]
            total_pages = max(1, math.ceil(len(filtered_inv) / page_size))

            obs = {
                "action": action_type,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total_count": len(filtered_inv),
                "items": [inv.model_dump() for inv in page_items],
            }

        # query_payments
        elif action_type == ActionType.QUERY_PAYMENTS.value:
            self._total_queries += 1
            vendor_id = validated_action.get("vendor_id") or validated_action.get("entity_id")
            inv_id = validated_action.get("invoice_id")

            filtered_pmt = self._world.payments
            if vendor_id:
                filtered_pmt = [
                    p for p in filtered_pmt if p.vendor_id == vendor_id or p.entity_id == vendor_id
                ]
            if inv_id:
                filtered_pmt = [
                    p for p in filtered_pmt if p.invoice_id == inv_id or inv_id in p.invoice_refs
                ]

            page, page_size = _parse_pagination(validated_action)
            start_i = (page - 1) * page_size
            page_items = filtered_pmt[start_i : start_i + page_size]
            total_pages = max(1, math.ceil(len(filtered_pmt) / page_size))

            obs = {
                "action": action_type,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total_count": len(filtered_pmt),
                "items": [p.model_dump() for p in page_items],
            }

        # query_fx
        elif action_type == ActionType.QUERY_FX.value:
            self._total_queries += 1
            date_query = validated_action.get("date")
            quote_query = validated_action.get("quote_currency") or validated_action.get(
                "target_currency"
            )

            filtered_fx = self._world.fx_rates
            if date_query:
                filtered_fx = [fx for fx in filtered_fx if fx.date == date_query]
            if quote_query:
                filtered_fx = [
                    fx
                    for fx in filtered_fx
                    if fx.quote_currency == quote_query or fx.target_currency == quote_query
                ]

            page, page_size = _parse_pagination(validated_action)
            start_i = (page - 1) * page_size
            page_items = filtered_fx[start_i : start_i + page_size]
            total_pages = max(1, math.ceil(len(filtered_fx) / page_size))

            obs = {
                "action": action_type,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total_count": len(filtered_fx),
                "items": [fx.model_dump() for fx in page_items],
            }

        # =====================================================================
        # 3. SUBMIT FINDINGS (submit / submit_findings)
        # =====================================================================
        elif action_type in (ActionType.SUBMIT.value, ActionType.SUBMIT_FINDINGS.value):
            from fin_audit_gym.evaluator.metrics import AuditMetrics, calculate_metrics

            self._terminated = True
            raw_input = validated_action.get("findings")
            explanation = validated_action.get("explanation")

            # Handle case where findings payload is {episode_id: ..., findings: [...]}
            if isinstance(raw_input, dict) and "findings" in raw_input:
                raw_findings = raw_input.get("findings", [])
                if explanation is None:
                    explanation = raw_input.get("explanation")
            else:
                raw_findings = raw_input if raw_input is not None else []

            # Enforce Section 05 Hackathon limits: max 30 findings, max 1000 chars explanation
            if not isinstance(raw_findings, list) or len(raw_findings) > 30:
                reward = -1.0
                fp_count = len(raw_findings) if isinstance(raw_findings, list) else 0
                obs = {
                    "action": action_type,
                    "submission_received": False,
                    "error": (
                        "Invalid findings structure: list exceeds maximum limit of 30 findings, "
                        f"got {fp_count if isinstance(raw_findings, list) else type(raw_findings).__name__}"
                    ),
                }
                submission_metrics = AuditMetrics(
                    tp=0,
                    fp=fp_count,
                    fn=len(self._ground_truth_findings),
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                )
            elif explanation is not None and len(str(explanation)) > 1000:
                reward = -1.0
                obs = {
                    "action": action_type,
                    "submission_received": False,
                    "error": (
                        f"Explanation length ({len(str(explanation))}) exceeds limit of 1000 characters"
                    ),
                }
                submission_metrics = AuditMetrics(
                    tp=0,
                    fp=len(raw_findings),
                    fn=len(self._ground_truth_findings),
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                )
            else:
                try:
                    parsed_findings: list[Finding] = []
                    for f in raw_findings:
                        if not isinstance(f, (dict, Finding)):
                            raise TypeError(f"Each finding must be a dict, got {type(f).__name__}")
                        pf = Finding.model_validate(f) if isinstance(f, dict) else f
                        # Enforce max 10 evidence items per finding
                        evidence_count = (
                            len(pf.evidence_ids) if pf.evidence_ids else len(pf.evidence)
                        )
                        if evidence_count > 10:
                            raise ValueError(
                                f"Finding '{pf.finding_id or pf.issue_id}' exceeds limit of 10 evidence fields"
                            )
                        if pf.explanation and len(str(pf.explanation)) > 1000:
                            raise ValueError(
                                f"Finding explanation exceeds 1000 chars: {len(str(pf.explanation))}"
                            )
                        parsed_findings.append(pf)

                    submission = Submission(findings=parsed_findings, explanation=explanation)
                    submission_metrics = calculate_metrics(
                        submission.findings, self._ground_truth_findings
                    )
                    reward = self.reward_engine.compute_terminal_reward(
                        submission_metrics, self._steps_taken
                    )
                    obs = {
                        "action": action_type,
                        "submission_received": True,
                        "findings_count": len(submission.findings),
                    }
                except (ValidationError, ValueError, TypeError) as exc:
                    reward = -1.0
                    obs = {
                        "action": action_type,
                        "submission_received": False,
                        "error": f"Malformed findings structure: {exc}",
                    }
                    submission_metrics = calculate_metrics([], self._ground_truth_findings)

        info = {
            "steps_taken": self._steps_taken,
            "max_steps": self.max_steps,
            "remaining_steps": max(0, self.max_steps - self._steps_taken),
            "episode_id": self._world.manifest.episode_id,
            "total_queries": self._total_queries,
        }
        if action_type in (ActionType.SUBMIT.value, ActionType.SUBMIT_FINDINGS.value):
            info["metrics"] = submission_metrics.model_dump()
        elif self._truncated:
            from fin_audit_gym.evaluator.metrics import calculate_metrics

            reward = -1.0
            info["metrics"] = calculate_metrics([], self._ground_truth_findings).model_dump()

        return obs, reward, self._terminated, self._truncated, info
