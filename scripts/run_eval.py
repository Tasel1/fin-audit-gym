#!/usr/bin/env python3
"""Run multi-episode policy evaluation benchmark and print formatted scoreboard."""

import argparse
from pathlib import Path

from fin_audit_gym.baselines import (
    BasePolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    ReconcilerPolicy,
    SpamAttackerPolicy,
    run_policy_episode,
)
from fin_audit_gym.core.env import FinAuditEnv


def get_policy(name: str) -> BasePolicy:
    """Instantiate policy by name."""
    name_lower = name.strip().lower()
    if name_lower == "oracle":
        return OraclePolicy()
    if name_lower == "reconciler":
        return ReconcilerPolicy()
    if name_lower == "random_explorer":
        return RandomExplorerPolicy(explore_steps=5)
    if name_lower == "spam_attacker":
        return SpamAttackerPolicy(num_spam=40)
    raise ValueError(f"Unknown policy name: {name}")


def evaluate_policies(
    data_dir: Path | str,
    policy_choice: str = "all",
) -> None:
    """Evaluate requested policies across all episodes in data_dir."""
    base_dir = Path(data_dir)
    ep_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    if not ep_dirs and (base_dir / "manifest.json").is_file():
        ep_dirs = [base_dir]

    if not ep_dirs:
        raise FileNotFoundError(f"No episode directories found in {base_dir}")

    all_policies = ["oracle", "reconciler", "random_explorer", "spam_attacker"]
    if policy_choice.lower() == "all":
        target_policies = all_policies
    else:
        target_policies = [policy_choice.lower()]

    results: list[dict] = []

    for pol_name in target_policies:
        rewards: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        steps: list[int] = []

        for ep_dir in ep_dirs:
            env = FinAuditEnv(data_dir=ep_dir)
            policy = get_policy(pol_name)
            res = run_policy_episode(env, policy)

            rewards.append(res["reward"])
            steps.append(res["steps"])
            metrics = res.get("metrics", {})
            precisions.append(metrics.get("precision", 0.0))
            recalls.append(metrics.get("recall", 0.0))
            f1s.append(metrics.get("f1", 0.0))

        n_ep = len(ep_dirs)
        results.append(
            {
                "policy": pol_name,
                "episodes": n_ep,
                "precision": sum(precisions) / n_ep,
                "recall": sum(recalls) / n_ep,
                "f1": sum(f1s) / n_ep,
                "reward": sum(rewards) / n_ep,
                "steps": sum(steps) / n_ep,
            }
        )

    # Print ASCII scoreboard
    print("=" * 86)
    print("                             POLICY BENCHMARK SCOREBOARD")
    print("=" * 86)
    header = f"{'Policy':<20} {'Episodes':>8} {'Precision':>12} {'Recall':>12} {'F1':>10} {'Reward':>10} {'Steps':>8}"
    print(header)
    print("-" * 86)
    for r in results:
        line = (
            f"{r['policy']:<20} {r['episodes']:>8} {r['precision']:>12.4f} "
            f"{r['recall']:>12.4f} {r['f1']:>10.4f} {r['reward']:>10.4f} {r['steps']:>8.1f}"
        )
        print(line)
    print("=" * 86)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline policies on audit dataset.")
    parser.add_argument("--data-dir", type=str, default="data/dev", help="Dataset directory")
    parser.add_argument(
        "--policy",
        type=str,
        default="all",
        choices=["all", "oracle", "reconciler", "random_explorer", "spam_attacker"],
        help="Policy to evaluate",
    )
    args = parser.parse_args()

    evaluate_policies(data_dir=args.data_dir, policy_choice=args.policy)


if __name__ == "__main__":
    main()
