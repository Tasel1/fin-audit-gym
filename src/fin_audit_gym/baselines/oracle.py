"""Oracle baseline policy achieving theoretical maximum reward."""

from typing import Any

from fin_audit_gym.baselines.base import BasePolicy
from fin_audit_gym.core.env import FinAuditEnv
from fin_audit_gym.core.schemas import Finding


class OraclePolicy(BasePolicy):
    """Optimal benchmark policy that submits exact ground truth findings."""

    def __init__(self, truth_findings: list[Finding] | None = None) -> None:
        self.truth_findings: list[dict[str, Any]] = (
            [f.model_dump() for f in truth_findings] if truth_findings else []
        )

    def bind_env(self, env: FinAuditEnv) -> None:
        """Bind privileged access to ground truth findings from environment."""
        if hasattr(env, "_ground_truth_findings"):
            self.truth_findings = [f.model_dump() for f in env._ground_truth_findings]

    def reset(self) -> None:
        self.truth_findings = []

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "submit_findings",
            "findings": self.truth_findings,
            "explanation": "Oracle policy theoretical maximum audit report.",
        }
