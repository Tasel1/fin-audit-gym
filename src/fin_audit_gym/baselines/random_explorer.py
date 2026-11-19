"""Random exploration baseline policy."""

import random
from typing import Any

from fin_audit_gym.baselines.base import BasePolicy


class RandomExplorerPolicy(BasePolicy):
    """Explores random query actions before submitting an empty/heuristic audit report."""

    def __init__(self, explore_steps: int = 5, seed: int = 42) -> None:
        self.explore_steps = explore_steps
        self.rng = random.Random(seed)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        self._step += 1
        if self._step > self.explore_steps:
            # Terminate episode by submitting report
            return {
                "type": "submit_findings",
                "findings": [],
                "explanation": "Random explorer completed exploratory queries.",
            }

        action_choices = [
            {"type": "query_ledger", "page": self.rng.randint(1, 3), "page_size": 10},
            {"type": "query_invoices", "page": self.rng.randint(1, 2), "page_size": 10},
            {"type": "query_payments", "page": 1, "page_size": 10},
            {"type": "query_fx", "quote_currency": "EUR", "page": 1, "page_size": 10},
            {
                "type": "calculate",
                "expression": f"{self.rng.randint(10, 100)} + {self.rng.randint(5, 50)}",
            },
        ]
        return self.rng.choice(action_choices)
