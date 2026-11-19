"""Tests for Double-Entry Synthetic Generator and Provenance Engine."""

import json
from pathlib import Path

import pytest

from fin_audit_gym.core.schemas import FXRate, Invoice, Manifest, Payment
from fin_audit_gym.generator import (
    GeneratedWorld,
    GeneratorConfig,
    ProvenanceTracker,
    SyntheticLedgerGenerator,
)


def test_provenance_tracker():
    tracker = ProvenanceTracker()
    tracker.record_transaction(
        transaction_id="TXN-001",
        tx_type="vendor_invoice",
        entity_ids=["INV-001", "VEND-001"],
        ledger_entry_ids=["ENTRY-001"],
        timestamp="2026-01-10",
        metadata={"amount_minor": 50000},
    )

    lineage = tracker.get_lineage("INV-001")
    assert lineage is not None
    assert len(lineage) == 1
    assert lineage[0]["transaction_id"] == "TXN-001"
    assert lineage[0]["tx_type"] == "vendor_invoice"

    graph_dict = tracker.to_dict()
    assert "TXN-001" in graph_dict["transactions"]
    assert "INV-001" in graph_dict["entity_map"]


def test_generator_deterministic_reproducibility():
    config1 = GeneratorConfig(seed=42, num_vendors=5, num_transactions=20)
    config2 = GeneratorConfig(seed=42, num_vendors=5, num_transactions=20)

    world1 = SyntheticLedgerGenerator(config1).generate()
    world2 = SyntheticLedgerGenerator(config2).generate()

    assert world1.manifest.episode_id == world2.manifest.episode_id
    assert len(world1.ledger) == len(world2.ledger)
    assert len(world1.invoices) == len(world2.invoices)
    assert len(world1.payments) == len(world2.payments)

    for r1, r2 in zip(world1.ledger, world2.ledger, strict=True):
        assert r1.entry_id == r2.entry_id
        assert r1.account_id == r2.account_id
        assert r1.debit_minor == r2.debit_minor
        assert r1.credit_minor == r2.credit_minor


def test_generator_different_seeds():
    config1 = GeneratorConfig(seed=42, num_vendors=5, num_transactions=20)
    config2 = GeneratorConfig(seed=999, num_vendors=5, num_transactions=20)

    world1 = SyntheticLedgerGenerator(config1).generate()
    world2 = SyntheticLedgerGenerator(config2).generate()

    assert world1.manifest.episode_id != world2.manifest.episode_id
    # Amounts or transaction counts or dates should differ
    ledger_debits1 = [r.debit_minor for r in world1.ledger]
    ledger_debits2 = [r.debit_minor for r in world2.ledger]
    assert ledger_debits1 != ledger_debits2


def test_double_entry_invariants():
    config = GeneratorConfig(seed=123, num_vendors=8, num_transactions=30)
    world = SyntheticLedgerGenerator(config).generate()

    # Pristine world satisfies invariants
    assert world.verify_double_entry_invariants() is True

    # Tampering with an entry must cause verification to raise ValueError
    tampered_row = world.ledger[0].model_copy(
        update={"debit_minor": world.ledger[0].debit_minor + 100}
    )
    tampered_world = GeneratedWorld(
        manifest=world.manifest,
        ledger=[tampered_row, *world.ledger[1:]],
        invoices=world.invoices,
        payments=world.payments,
        fx_rates=world.fx_rates,
        provenance=world.provenance,
    )
    with pytest.raises(ValueError, match="Double-entry invariant violated"):
        tampered_world.verify_double_entry_invariants()


def test_export_to_directory(tmp_path: Path):
    config = GeneratorConfig(seed=77, num_vendors=6, num_transactions=25)
    world = SyntheticLedgerGenerator(config).generate()

    world.export_to_directory(tmp_path)

    # Check files exist
    assert (tmp_path / "ledger.csv").is_file()
    assert (tmp_path / "invoices.jsonl").is_file()
    assert (tmp_path / "payments.jsonl").is_file()
    assert (tmp_path / "fx.csv").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "provenance.json").is_file()

    # Validate manifest.json
    manifest_data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest = Manifest.model_validate(manifest_data)
    assert manifest.fiscal_year == config.fiscal_year

    # Validate invoices.jsonl
    with (tmp_path / "invoices.jsonl").open(encoding="utf-8") as f:
        invoices = [Invoice.model_validate_json(line) for line in f if line.strip()]
    assert len(invoices) == len(world.invoices)

    # Validate payments.jsonl
    with (tmp_path / "payments.jsonl").open(encoding="utf-8") as f:
        payments = [Payment.model_validate_json(line) for line in f if line.strip()]
    assert len(payments) == len(world.payments)

    # Validate fx.csv
    fx_lines = (tmp_path / "fx.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(fx_lines) > 1  # Header + records
    first_row = fx_lines[1].split(",")
    first_fx = FXRate(
        date=first_row[0],
        base_currency=first_row[1],
        quote_currency=first_row[2],
        rate=first_row[3],
    )
    assert first_fx.base_currency == config.base_currency
