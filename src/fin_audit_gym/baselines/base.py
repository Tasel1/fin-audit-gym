"""Base policy abstractions and episodic execution runner."""

from abc import ABC, abstractmethod
from typing import Any

from fin_audit_gym.core.env import FinAuditEnv


class BasePolicy(ABC):
    """Abstract base class for all FinAuditGym agent policies."""

    def reset(self) -> None:
        """Reset internal agent memory/state between episodes."""

    @abstractmethod
    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        """Produce the next action dictionary given current observation and info."""


def run_policy_episode(
    env: FinAuditEnv,
    policy: BasePolicy,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute a complete episodic interaction loop between policy and environment."""
    policy.reset()
    obs, info = env.reset(seed=seed)

    # Allow policies with privileged benchmark access (like Oracle) to bind env state
    if hasattr(policy, "bind_env"):
        policy.bind_env(env)

    trajectory: list[dict[str, Any]] = []
    final_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = policy.act(obs, info)
        next_obs, reward, term, trunc, next_info = env.step(action)

        trajectory.append(
            {
                "action": action,
                "observation": next_obs,
                "reward": reward,
                "terminated": term,
                "truncated": trunc,
                "info": next_info,
            }
        )

        obs = next_obs
        info = next_info
        final_reward = reward
        terminated = term
        truncated = trunc

    return {
        "reward": final_reward,
        "steps": info.get("steps_taken", len(trajectory)),
        "terminated": terminated,
        "truncated": truncated,
        "metrics": info.get("metrics", {}),
        "trajectory": trajectory,
    }
