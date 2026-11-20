"""Adversarial spam attacker policy designed to test anti-gaming defenses."""

from typing import Any

from fin_audit_gym.baselines.base import BasePolicy


class SpamAttackerPolicy(BasePolicy):
    """Submits dozens of duplicated accusations to trigger Kuhn-Munkres false positive explosion."""

    def __init__(self, num_spam: int = 30) -> None:
        self.num_spam = num_spam

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        spam_findings = [
            {
                "finding_id": f"SPAM-{i:03d}",
                "error_type": "duplicate_invoice",
                "entity_id": "INV-2026-0001-DUP",
                "description": f"Spammed finding accusation {i}",
                "evidence": {"spam_index": i},
            }
            for i in range(self.num_spam)
        ]

        return {
            "type": "submit_findings",
            "findings": spam_findings,
            "explanation": "Adversarial spam attack submission.",
        }
