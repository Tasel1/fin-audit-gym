"""Terminal reward calculation engine preventing specification gaming."""

from fin_audit_gym.evaluator.metrics import AuditMetrics


class RewardEngine:
    """Calculates terminal episode rewards based on F1 accuracy and investigative efficiency."""

    def __init__(
        self,
        step_cost: float = 0.002,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
    ) -> None:
        self.step_cost = step_cost
        self.min_reward = min_reward
        self.max_reward = max_reward

    def compute_terminal_reward(self, metrics: AuditMetrics, steps_taken: int) -> float:
        """Compute bounded terminal reward based on DeepMind audit gym specification.

        Formula:
            R = clip(2.0 * F1 - 1.0 - (step_cost * steps_taken), min_reward, max_reward)
        """
        raw_reward = 2.0 * metrics.f1 - 1.0 - (self.step_cost * steps_taken)
        clipped_reward = max(self.min_reward, min(self.max_reward, raw_reward))
        return round(clipped_reward, 4)
