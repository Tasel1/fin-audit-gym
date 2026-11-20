# Fin-Audit-Gym

> Reproducible Gymnasium-compatible RL environment and benchmark simulator for training and evaluating AI financial audit models.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 56 passing](https://img.shields.io/badge/tests-56%20passing-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 🏆 Hackathon Case Overview (100-Point Evaluation)

`Fin-Audit-Gym` is a competition-grade, offline, reproducible Gymnasium environment designed for training and benchmarking Reinforcement Learning (RL) and Large Language Model (LLM) agents on corporate financial audit tasks.

The environment simulates realistic enterprise accounting worlds with double-entry general ledgers, commercial invoices, bank cash disbursements, foreign currency exchange rates, deterministic error injection, innocent look-alike distractors, and an anti-gaming evaluation engine based on maximum-weight bipartite matching (Hungarian algorithm).

### Key Architectural Pillars

| Component | Implementation & Defense Mechanism |
|---|---|
| **Gymnasium Contract** | JSON-compatible `reset(seed, config) -> (obs, info)` and `step(action) -> (obs, r, term, trunc, info)`. Zero heavy gym dependencies required; pure Python dataclasses and Pydantic v2 models. |
| **Double-Entry Generator** | Pristine clean world created first. Validates double-entry accounting balance ($\sum \text{Debit} = \sum \text{Credit}$) across all accounts (1000 Cash, 1200 AR, 2000 AP, 4000 Revenue, 5000 Expenses, 7000 FX Gain/Loss). Lineage tracked in `provenance.json`. |
| **Integer Minor Math** | Zero floating-point drift: all monetary values stored strictly in integer minor units (`amount_minor: int`). Exchange rates parsed as exact `Decimal`. |
| **Safe AST Calculator** | Python AST whitelist (`ast.BinOp`, `ast.Constant`, `+`, `-`, `*`, `/`, `%`) without `eval()` or `exec()`. Malicious injection (`__import__`, `open`, `exec`) strictly rejected. |
| **Error Taxonomy** | 4 atomic error types (`duplicate_invoice`, `unallocated_payment`, `fx_mismatch`, `period_cutoff`) + 2 composite scenarios (`composite_round_trip`, `composite_split_invoice`) + 3 innocent distractors (retainers, milestone partial payments, 1-cent FX rounding). |
| **Anti-Gaming Matcher** | Canonical 1-to-1 bipartite matching using `scipy.optimize.linear_sum_assignment` (Kuhn-Munkres). Duplicate or hallucinated findings are penalized as False Positives ($FP \uparrow \implies \text{Precision} \to 0 \implies \text{Reward} = -1.0$). |
| **Zero-Leak Sandbox** | Ground truth (`truth.json`) and causal graph (`provenance.json`) are completely inaccessible from observation space, query pagination, or public environment inspection. |

---

## 📊 Benchmark Scoreboard (Baselines on Dev Split)

Results evaluated across 10 unseen evaluation episodes (`data/dev/`):

```bash
python3 scripts/run_eval.py --data-dir data/dev --policy all
```

```
======================================================================================
                             POLICY BENCHMARK SCOREBOARD
======================================================================================
Policy               Episodes    Precision       Recall         F1     Reward    Steps
--------------------------------------------------------------------------------------
oracle                     10       1.0000       1.0000     1.0000    +0.9980      1.0
reconciler                 10       0.5250       0.5000     0.5100    +0.0140      3.0
random_explorer            10       0.0000       0.0000     0.0000    -1.0000      6.0
spam_attacker              10       0.0025       0.0167     0.0043    -0.9915      1.0
======================================================================================
```

- **`oracle`**: Achieves perfect theoretical ceiling ($F_1 = 1.0, \text{Reward} = +0.9980$), proving the environment is 100% solvable.
- **`reconciler`**: Algorithmic accounting auditor matching invoices and payments, discovering true anomalies ($F_1 = 0.51, \text{Reward} > 0$).
- **`random_explorer`**: Unfocused exploratory actions, receiving maximal penalty ($\text{Reward} = -1.0$).
- **`spam_attacker`**: Adversarial policy attempting specification gaming by spamming 50+ duplicate findings. Neutralized into False Positives by the Kuhn-Munkres matcher ($\text{Reward} = -0.9915$).

---

## ⚡ Quick Start & Reproduction

### 1. Local Setup

```bash
# Clone repository
git clone https://github.com/Tasel1/fin-audit-gym.git
cd fin-audit-gym

# Install package and dependencies
pip install -e .
pip install -r requirements-dev.txt
```

### 2. Single-Command Jury Verification

```bash
# 1. Run full automated test suite (56 unit, integration, adversarial pentest & security tests)
make test

# 2. Run code style and type audit
make lint

# 3. Run complete evaluation across dev dataset
python3 scripts/run_eval.py --data-dir data/dev --policy all
```

### 3. Docker Offline Reproduction (Zero Network, Zero GPU)

```bash
# Build offline container
docker build -t fin-audit-gym .

# Run test suite inside isolated container
docker run --rm fin-audit-gym
```

---

## 💡 Python Usage Example

```python
from fin_audit_gym.core.env import FinAuditEnv

# 1. Initialize environment (loads pre-generated episode or generates dynamically)
env = FinAuditEnv(data_dir="data/dev/ep_001", max_steps=100)
obs, info = env.reset(seed=42)

print(f"Company: {obs['manifest']['company_name']}, Fiscal Year: {obs['manifest']['fiscal_year']}")

# 2. Query general ledger with pagination
obs, reward, term, trunc, info = env.step({
    "action": "query_ledger",
    "account_id": "5000",  # Operating Expenses
    "page": 1,
    "page_size": 10
})

# 3. Perform safe sandboxed calculation (zero eval/exec)
obs, reward, term, trunc, info = env.step({
    "action": "calculate",
    "expression": "125000 * 1.08 - 5000"
})
print("Calculated:", obs["result"])

# 4. Submit audit findings
obs, reward, terminated, truncated, info = env.step({
    "action": "submit_findings",
    "findings": [
        {
            "finding_id": "FINDING-001",
            "error_type": "duplicate_invoice",
            "entity_id": "INV-00042-DUP",
            "description": "Duplicate payment obligation recorded for vendor",
            "evidence": {"invoice_id": "INV-00042-DUP", "total_minor": 125000}
        }
    ],
    "explanation": "Detected duplicate invoice entry matching prior payment."
})

print(f"Terminal Reward: {reward}")
print(f"Audit Metrics: F1={info['metrics']['f1']}, Precision={info['metrics']['precision']}, Recall={info['metrics']['recall']}")
```

---

## 📁 Repository Structure

```
fin-audit-gym/
├── src/
│   └── fin_audit_gym/
│       ├── core/          # Gymnasium environment, action space, safe calculator, schemas
│       │   ├── env.py            # FinAuditEnv (reset, step, pagination, zero-leak sandbox)
│       │   ├── actions.py        # Validated action schemas and query parameters
│       │   ├── calculator.py     # Safe AST arithmetic engine (zero eval/exec)
│       │   ├── schemas.py        # Pydantic v2 domain models (Ledger, Invoice, Payment)
│       │   └── exceptions.py     # Structured error types
│       ├── generator/     # Double-entry synthetic data generator and error injectors
│       │   ├── engine.py         # SyntheticLedgerGenerator & GeneratedWorld export
│       │   ├── provenance.py     # ProvenanceTracker causal lineage engine
│       │   ├── error_injector.py # 4 atomic error types + 2 composite scenarios
│       │   └── distractors.py    # Innocent financial look-alikes (retainers, splits)
│       ├── evaluator/     # Bijective matcher, F1 calculation, and terminal reward engine
│       │   ├── matcher.py        # Kuhn-Munkres 1-to-1 Hungarian matcher (scipy)
│       │   ├── metrics.py        # Precision, Recall, F1 with clean-book logic
│       │   └── reward.py         # Terminal reward formula with step budget decay
│       ├── baselines/     # 4 reference policies
│       │   ├── base.py           # BasePolicy and episodic execution runner
│       │   ├── oracle.py         # Theoretical ceiling upper-bound agent
│       │   ├── reconciler.py     # Rule-based accounting reconciliation baseline
│       │   ├── random_explorer.py# Random action explorer baseline
│       │   └── spam_attacker.py  # Adversarial specification-gaming attacker
│       └── utils/         # Currency arithmetic, FX handling, and serialization
│           └── currency.py       # Minor unit conversion with Decimal rounding modes
├── data/
│   ├── train/             # 10 reference training episodes (ep_001..ep_010)
│   └── dev/               # 10 reference evaluation episodes (ep_001..ep_010)
├── trajectories/          # Serialized policy execution logs (trajectories.jsonl)
├── tests/                 # 41 unit, integration, isolation, and anti-gaming tests
│   ├── test_contract_isolation.py       # Gym API & zero-leak sandbox verification
│   ├── test_schemas_and_calculator.py   # AST security attack tests & Pydantic validation
│   ├── test_generator_clean_world.py    # Double-entry mathematical invariants
│   ├── test_generator_anomalies.py      # Error injector & distractor verification
│   ├── test_reward_and_attacks.py       # Kuhn-Munkres matching & anti-gaming defenses
│   ├── test_baselines.py                # 4 baseline policy performance verification
│   └── test_scripts.py                  # CLI runners & trajectory recording checks
├── scripts/
│   ├── generate_episodes.py   # Deterministic dataset generation script
│   ├── record_trajectories.py # Trajectory generation for SFT / offline RL / DPO
│   └── run_eval.py            # Rich CLI benchmark evaluation scoreboard
├── DESIGN.md              # In-depth architectural specification and mathematical formalization
├── Dockerfile             # Offline, zero-GPU container setup
├── Makefile               # Automated testing and verification commands
└── pyproject.toml         # Package definition and pinned dependencies
```

---

## 🛡️ Anti-Gaming Invariants (DeepMind Specification Gaming Defenses)

1. **Zero Step Reward ($R_{\text{step}} = 0$)**: Prevents infinite looping or reward farming during exploration.
2. **Kuhn-Munkres 1-to-1 Bijective Matching**: Every ground truth finding matches at most one predicted finding. Any spammed duplicates immediately convert to False Positives ($FP \uparrow \implies \text{Precision} \to 0 \implies \text{Reward} = -1.0$).
3. **Clean Book Truth Test**: Submitting false findings on clean accounting books receives $F_1 = 0.0, \text{Reward} = -1.0$; submitting empty findings on clean books receives $F_1 = 1.0, \text{Reward} = +1.0$.
4. **Step Budget Friction ($\lambda = 0.002$)**: Discourages idle queries, rewarding concise and accurate investigations.

---

## 📜 License

MIT License. Designed and engineered for the FinTech RL Financial Audit Hackathon.
