#!/usr/bin/env python3
"""Record multi-turn agent interaction trajectories for offline RL and SFT training."""

import argparse
import json
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
        return SpamAttackerPolicy(num_spam=30)
    raise ValueError(f"Unknown policy name: {name}")


def record_trajectories(
    data_dir: Path | str,
    output_file: Path | str,
    policy_names: list[str],
) -> None:
    """Run specified policies across all episodes in data_dir and save trajectory traces."""
    base_dir = Path(data_dir)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover episode directories
    if not base_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Data directory is not a directory: {base_dir}")

    if (base_dir / "manifest.json").is_file():
        ep_dirs = [base_dir]
    else:
        ep_dirs = sorted(
            [d for d in base_dir.iterdir() if d.is_dir() and (d / "manifest.json").is_file()]
        )

    if not ep_dirs:
        raise FileNotFoundError(
            f"No valid episode directories (with manifest.json) found in {base_dir}"
        )

    records: list[dict] = []

    for ep_dir in ep_dirs:
        for pol_name in policy_names:
            env = FinAuditEnv(data_dir=ep_dir)
            policy = get_policy(pol_name)
            result = run_policy_episode(env, policy)

            record = {
                "policy": pol_name,
                "episode_id": ep_dir.name,
                "reward": result["reward"],
                "steps": result["steps"],
                "terminated": result["terminated"],
                "truncated": result["truncated"],
                "metrics": result["metrics"],
                "trajectory": result["trajectory"],
            }
            records.append(record)
            print(f"Recorded trajectory: {pol_name} on {ep_dir.name} -> Reward: {result['reward']}")

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Successfully wrote {len(records)} trajectories to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record agent trajectories on episodes.")
    parser.add_argument("--data-dir", type=str, default="data/train", help="Episode directory")
    parser.add_argument(
        "--output-file",
        type=str,
        default="trajectories/trajectories.jsonl",
        help="Target JSONL file",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="oracle,reconciler,random_explorer",
        help="Comma-separated policy list",
    )
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    record_trajectories(
        data_dir=args.data_dir,
        output_file=args.output_file,
        policy_names=policies,
    )


if __name__ == "__main__":
    main()
