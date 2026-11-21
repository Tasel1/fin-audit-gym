"""Tests for Anomaly Injection Engine and Distractor Generator."""

import json
from pathlib import Path

from fin_audit_gym.core.schemas import Finding
from fin_audit_gym.generator import (
    AnomalyConfig,
    AnomalyInjector,
    GeneratorConfig,
    SyntheticLedgerGenerator,
)


def test_anomaly_injection_counts_and_types():
    gen_config = GeneratorConfig(seed=42, num_vendors=8, num_transactions=40)
    world = SyntheticLedgerGenerator(gen_config).generate()

    anomaly_config = AnomalyConfig(
        seed=42,
        num_duplicate_invoices=1,
        num_unallocated_payments=1,
        num_fx_mismatches=1,
        num_period_cutoffs=1,
        num_composite_round_trips=1,
        num_composite_split_invoices=1,
        num_distractors=3,
    )
    injector = AnomalyInjector(anomaly_config)
    mutated_world, findings = injector.inject(world)

    assert len(findings) == 6
    assert mutated_world.manifest.injected_anomalies_count == 6

    error_types = {f.error_type for f in findings}
    expected_types = {
        "duplicate_invoice",
        "unallocated_payment",
        "fx_mismatch",
        "period_cutoff",
        "composite_round_trip",
        "composite_split_invoice",
    }
    assert error_types == expected_types

    # Ensure every finding has valid evidence
    for f in findings:
        assert isinstance(f, Finding)
        assert f.entity_id
        assert len(f.description) > 0
        assert isinstance(f.evidence, dict)
        assert len(f.evidence) > 0


def test_distractors_not_in_truth():
    gen_config = GeneratorConfig(seed=101, num_vendors=8, num_transactions=40)
    world = SyntheticLedgerGenerator(gen_config).generate()

    anomaly_config = AnomalyConfig(
        seed=101,
        num_duplicate_invoices=1,
        num_unallocated_payments=0,
        num_fx_mismatches=0,
        num_period_cutoffs=0,
        num_composite_round_trips=0,
        num_composite_split_invoices=0,
        num_distractors=3,
    )
    injector = AnomalyInjector(anomaly_config)
    mutated_world, findings = injector.inject(world)

    # Truth should only contain the 1 duplicate invoice, zero distractors
    assert len(findings) == 1
    assert mutated_world.manifest.injected_anomalies_count == 1
    assert len(mutated_world.ledger) > len(world.ledger)
    assert findings[0].error_type == "duplicate_invoice"

    # Distractor entities must not appear as findings in truth
    finding_entities = {f.entity_id for f in findings}
    assert len(finding_entities) == 1


def test_anomaly_injection_deterministic():
    gen_config = GeneratorConfig(seed=777, num_vendors=8, num_transactions=35)
    world1 = SyntheticLedgerGenerator(gen_config).generate()
    world2 = SyntheticLedgerGenerator(gen_config).generate()

    anomaly_config = AnomalyConfig(seed=777)
    w1_mut, f1 = AnomalyInjector(anomaly_config).inject(world1)
    w2_mut, f2 = AnomalyInjector(anomaly_config).inject(world2)

    assert len(f1) == len(f2)
    for finding1, finding2 in zip(f1, f2, strict=True):
        assert finding1.finding_id == finding2.finding_id
        assert finding1.error_type == finding2.error_type
        assert finding1.entity_id == finding2.entity_id
        assert finding1.evidence == finding2.evidence

    assert len(w1_mut.ledger) == len(w2_mut.ledger)
    assert len(w1_mut.invoices) == len(w2_mut.invoices)


def test_export_with_truth(tmp_path: Path):
    gen_config = GeneratorConfig(seed=55, num_vendors=6, num_transactions=30)
    world = SyntheticLedgerGenerator(gen_config).generate()

    anomaly_config = AnomalyConfig(seed=55)
    mutated_world, findings = AnomalyInjector(anomaly_config).inject(world)

    mutated_world.export_to_directory(tmp_path, truth_findings=findings)

    truth_file = tmp_path / "truth.json"
    assert truth_file.is_file()

    truth_raw = json.loads(truth_file.read_text(encoding="utf-8"))
    assert isinstance(truth_raw, list)
    assert len(truth_raw) == len(findings)

    loaded_findings = [Finding.model_validate(item) for item in truth_raw]
    assert len(loaded_findings) == len(findings)
    assert loaded_findings[0].finding_id == findings[0].finding_id


def test_multiple_period_cutoffs_and_fx_mismatches_uniqueness():
    """Verify that multiple period cutoff violations and fx mismatches produce distinct IDs and target multiple items."""
    gen_config = GeneratorConfig(seed=42, num_vendors=8, num_transactions=40)
    world = SyntheticLedgerGenerator(gen_config).generate()

    anomaly_config = AnomalyConfig(
        seed=42,
        num_duplicate_invoices=0,
        num_unallocated_payments=0,
        num_fx_mismatches=2,
        num_period_cutoffs=2,
        num_composite_round_trips=0,
        num_composite_split_invoices=0,
        num_distractors=0,
    )
    _, findings = AnomalyInjector(anomaly_config).inject(world)

    cutoff_findings = [f for f in findings if f.error_type == "period_cutoff"]
    assert len(cutoff_findings) == 2
    entity_ids = {f.entity_id for f in cutoff_findings}
    assert len(entity_ids) == 2
    assert f"INV-{world.manifest.fiscal_year}-CUTOFF-01" in entity_ids
    assert f"INV-{world.manifest.fiscal_year}-CUTOFF-02" in entity_ids

    fx_findings = [f for f in findings if f.error_type == "fx_mismatch"]
    assert len(fx_findings) == 2
