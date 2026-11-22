"""Bipartite 1-to-1 matching engine using the Kuhn-Munkres (Hungarian) algorithm."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from fin_audit_gym.core.schemas import Finding


class BipartiteMatcher:
    """Computes optimal 1-to-1 matching between predicted findings and ground truth anomalies."""

    @staticmethod
    def _is_match(pred: Finding, gt: Finding) -> bool:
        if pred.error_type != gt.error_type:
            return False
        if pred.entity_id == gt.entity_id:
            return True
        return bool(
            isinstance(gt.evidence, dict)
            and pred.entity_id in gt.evidence.get("sub_invoice_ids", [])
        )

    def match(
        self,
        predictions: list[Finding],
        ground_truth: list[Finding],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match predictions to ground truth via minimum weight bipartite assignment.

        Returns:
            matched_pairs: List of (pred_idx, gt_idx) tuples.
            unmatched_preds: List of prediction indices that did not match ground truth.
            unmatched_gts: List of ground truth indices that were missed.
        """
        n_preds = len(predictions)
        n_gts = len(ground_truth)

        if n_preds == 0 or n_gts == 0:
            return [], list(range(n_preds)), list(range(n_gts))

        # Cost matrix: -1 for valid matches, 100 for non-matches
        cost_matrix = np.full((n_preds, n_gts), 100.0, dtype=float)

        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                if self._is_match(pred, gt):
                    cost_matrix[i, j] = -1.0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: list[tuple[int, int]] = []
        for r, c in zip(row_ind, col_ind, strict=True):
            p = predictions[r]
            g = ground_truth[c]
            if self._is_match(p, g):
                matched_pairs.append((r, c))

        matched_p_set = {r for r, _ in matched_pairs}
        matched_gt_set = {c for _, c in matched_pairs}

        unmatched_preds = [i for i in range(n_preds) if i not in matched_p_set]
        unmatched_gts = [j for j in range(n_gts) if j not in matched_gt_set]

        return matched_pairs, unmatched_preds, unmatched_gts
