# Fin-Audit-Gym

> Reproducible Gymnasium-compatible RL environment and benchmark simulator for training and evaluating AI financial audit models.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

`Fin-Audit-Gym` is an offline, reproducible benchmark environment designed for AI financial auditing agents. The simulator models realistic corporate financial bookkeeping, transaction records, documents, exchange rates, and intentional accounting anomalies.

### Key Capabilities

1. **Synthetic Bookkeeping Generator**:
   - Double-entry balance enforcement (Debit = Credit).
   - Generates `ledger.csv`, `invoices.jsonl`, `payments.jsonl`, `fx.csv`, `policy.json`, and `manifest.json`.
   - 4 independent error types: Duplicate invoices, unallocated/partial payments, FX conversion mismatches, and reporting period cut-off errors.
   - 2 multi-error interaction scenarios (e.g. cross-currency partial payment on period boundary).
   - Realistic distractors: split payments, refunds, and recurring identical amounts.

2. **Gymnasium-Compatible Agent Environment**:
   - Discrete step budget (max 100 steps per episode).
   - Constrained action interface: `list_records`, `get_record`, `get_document`, `calculator`, `submit`.
   - Sandboxed arithmetic calculator (safe AST evaluation, zero code execution).
   - Zero-leak architecture: ground-truth files (`truth.json`, `provenance.json`) remain strictly outside the agent's observation space.

3. **Bijective Matcher & Anti-Gaming Reward Engine**:
   - 1-to-1 canonical finding-to-truth matching with tolerance bounds and evidence verification.
   - Robust to adversarial strategies (spam attacks, duplicates, clean book silence).
   - Standard terminal reward function:
     $$\text{Reward} = \text{clip}(2 \cdot F_1 - 1 - 0.002 \cdot \text{inspection\_steps}, -1, 1)$$

4. **Reference Baseline Policies**:
   - `RandomExplorer`: Random action execution with fixed seed.
   - `Reconciler`: Rule-based baseline for financial discrepancy matching.
   - `SpamAttacker`: Adversarial policy attempting to claim all records.
   - `Oracle`: Perfect-information baseline measuring the theoretical ceiling of the environment.

---

## Repository Structure

```
fin-audit-gym/
├── src/
│   └── fin_audit_gym/
│       ├── core/          # Gymnasium environment, action space, safe calculator, schemas
│       ├── generator/     # Double-entry synthetic data generator and error injectors
│       ├── evaluator/     # Bijective matcher, F1 calculation, and terminal reward engine
│       ├── baselines/     # 4 reference policies (Random, Reconciler, Spam, Oracle)
│       └── utils/         # Currency arithmetic, FX handling, and deterministic serialization
├── data/
│   ├── train/             # 10 reference training episodes
│   └── dev/               # 10 reference evaluation episodes
├── trajectories/          # Serialized policy execution logs (trajectories.jsonl)
├── tests/                 # Unit, integration, isolation, and anti-gaming tests
├── scripts/               # Episode generation and batch evaluation scripts
├── Dockerfile             # Offline, zero-GPU container setup
├── Makefile               # Automated testing and verification commands
└── pyproject.toml         # Package definition and pinned dependencies
```

---

## Quick Start

### Local Setup

```bash
# Clone the repository
git clone https://github.com/Tasel1/fin-audit-gym.git
cd fin-audit-gym

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package with development dependencies
make install
```

### Running Tests

```bash
# Run test suite
make test

# Run linter
make lint
```

### Docker Verification

```bash
# Build offline container
make docker-build

# Execute tests inside container
make docker-test
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
