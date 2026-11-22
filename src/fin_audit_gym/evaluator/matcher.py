"""Bipartite 1-to-1 matching engine using the Kuhn-Munkres (Hungarian) algorithm (Section 05)."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from fin_audit_gym.core.schemas import Finding


class BipartiteMatcher:
    """Computes optimal 1-to-1 matching between predicted findings and ground truth anomalies."""

    @staticmethod
    def _is_match(pred: Finding, gt: Finding) -> bool:
        """Evaluate if a predicted finding matches a ground truth issue (Section 05 criteria).

        True Positive criteria:
        1. Correct anomaly type (type / error_type).
        2. Canonical affected_ids overlap or matching entity_id.
        3. Currency agreement.
        4. Claimed amount within tolerance (if provided).
        5. Supporting evidence coverage (if ground truth specifies admissible sets).
        """
        # 1. Type match
        p_type = getattr(pred, "type", None) or getattr(pred, "error_type", None)
        g_type = getattr(gt, "type", None) or getattr(gt, "error_type", None)
        if p_type != g_type:
            return False

        # 2. Currency match (if both specified)
        p_curr = getattr(pred, "currency", None)
        g_curr = getattr(gt, "currency", None)
        if p_curr and g_curr and p_curr.upper() != g_curr.upper():
            return False

        # 3. Affected IDs match
        p_affected = set(getattr(pred, "affected_ids", []) or [])
        if not p_affected and getattr(pred, "entity_id", None):
            p_affected.add(pred.entity_id)

        g_affected = set(getattr(gt, "affected_ids", []) or [])
        if not g_affected and getattr(gt, "entity_id", None):
            g_affected.add(gt.entity_id)

        id_match = False
        if p_affected and g_affected:
            if bool(p_affected.intersection(g_affected)):
                id_match = True
        elif getattr(pred, "entity_id", None) == getattr(gt, "entity_id", None):
            id_match = True

        if not id_match and isinstance(gt.evidence, dict):
            sub_ids = gt.evidence.get("sub_invoice_ids")
            if isinstance(sub_ids, (list, tuple, set)) and (
                pred.entity_id in sub_ids or bool(p_affected.intersection(set(sub_ids)))
            ):
                id_match = True

        if not id_match:
            return False

        # 4. Amount tolerance check (Section 05: within explicit tolerance)
        claimed = getattr(pred, "claimed_amount_minor", None)
        expected = getattr(gt, "expected_amount_minor", None)
        if claimed is not None and expected is not None and expected > 0:
            diff = abs(claimed - expected)
            tolerance = max(100, int(0.05 * expected))  # 5% tolerance or 100 minor units
            if diff > tolerance:
                return False

        # 5. Admissible Evidence Sets check
        admissible_sets = getattr(gt, "admissible_evidence_sets", None)
        if admissible_sets and isinstance(admissible_sets, list) and len(admissible_sets) > 0:
            pred_ev = set(getattr(pred, "evidence_ids", []) or [])
            if not pred_ev and isinstance(getattr(pred, "evidence", None), dict):
                for v in pred.evidence.values():
                    if isinstance(v, str):
                        pred_ev.add(v)
                    elif isinstance(v, (list, tuple, set)):
                        for item in v:
                            if isinstance(item, str):
                                pred_ev.add(item)
            if pred_ev:
                # Check if matches any admissible evidence subset
                has_ev_overlap = any(
                    bool(set(adm_set).intersection(pred_ev))
                    for adm_set in admissible_sets
                    if isinstance(adm_set, (list, tuple, set))
                )
                if not has_ev_overlap:
                    return False

        return True

    def match(
        self,
        predictions: list[Finding],
        ground_truth: list[Finding],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match predictions to ground truth via minimum weight bipartite assignment.

        Returns:
            matched_pairs: List of (pred_idx, gt_idx) tuples.
            unmatched_preds: List of prediction indices that did not match ground truth (FP).
            unmatched_gts: List of ground truth indices that were missed (FN).
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
