#!/usr/bin/env python3
"""Generate synthetic financial audit episodes for train and dev splits."""

import argparse
from pathlib import Path

# Preload core package to ensure clean module dependency resolution
import fin_audit_gym.core  # noqa: F401
from fin_audit_gym.generator import (
    AnomalyConfig,
    AnomalyInjector,
    GeneratorConfig,
    SyntheticLedgerGenerator,
)


def generate_dataset(
    output_dir: Path | str,
    num_train: int = 10,
    num_dev: int = 10,
    seed_offset: int = 1000,
    num_transactions: int = 85,
) -> None:
    """Generate train and dev episodic universes with balanced double-entry books (Section 20)."""
    base_path = Path(output_dir)

    splits = [("train", num_train, 0), ("dev", num_dev, 1000)]

    for split_name, count, split_seed_offset in splits:
        if count <= 0:
            continue
        split_dir = base_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, count + 1):
            seed = seed_offset + split_seed_offset + i
            ep_dir = split_dir / f"ep_{i:03d}"

            gen_config = GeneratorConfig(
                seed=seed, num_vendors=8, num_transactions=num_transactions
            )
            world = SyntheticLedgerGenerator(gen_config).generate()

            # Calibrate episode difficulty: Clean cases (Section 16), atomic errors, and interacting scenarios (Section 15 & 67)
            if i in (1, 2):
                anomaly_config = AnomalyConfig(
                    seed=seed,
                    num_duplicate_invoices=0,
                    num_unallocated_payments=0,
                    num_fx_mismatches=0,
                    num_period_cutoffs=0,
                    num_composite_round_trips=0,
                    num_composite_split_invoices=0,
                    num_distractors=3,
                )
            elif i in (3, 4, 5):
                anomaly_config = AnomalyConfig(
                    seed=seed,
                    num_duplicate_invoices=1,
                    num_unallocated_payments=1,
                    num_fx_mismatches=1,
                    num_period_cutoffs=1,
                    num_composite_round_trips=0,
                    num_composite_split_invoices=0,
                    num_distractors=2,
                )
            else:
                anomaly_config = AnomalyConfig(
                    seed=seed,
                    num_duplicate_invoices=1,
                    num_unallocated_payments=1,
                    num_fx_mismatches=1,
                    num_period_cutoffs=1,
                    num_composite_round_trips=1,
                    num_composite_split_invoices=1,
                    num_distractors=3,
                )

            mutated_world, truth = AnomalyInjector(anomaly_config).inject(world)

            mutated_world.export_to_directory(ep_dir, truth_findings=truth)
            print(f"Generated {split_name} episode {i:03d} (seed {seed}) at {ep_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic financial audit episodes.")
    parser.add_argument("--output-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--num-train", type=int, default=10, help="Number of train episodes")
    parser.add_argument("--num-dev", type=int, default=10, help="Number of dev episodes")
    parser.add_argument("--seed-offset", type=int, default=1000, help="Base seed offset")
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_dev=args.num_dev,
        seed_offset=args.seed_offset,
    )


if __name__ == "__main__":
    main()
