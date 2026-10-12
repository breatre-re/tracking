"""Phase 4 CTDA objective (focal loss) and association decoding.

Paper source: Section 3.3.3 Eq. (19) focal loss with ``alpha_t = 0.75`` and
focusing parameter ``gamma_t = 2``; Section 3.4 Eq. (20) thresholding with
``lambda_a`` followed by Kuhn-Munkres one-to-one assignment (Algorithm 1).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Masked binary focal loss (Eq. 19).

    ``logits``/``targets`` are ``(B, P, P)``; ``row_mask``/``col_mask`` are
    ``(B, P)`` boolean masks. Padded cells contribute nothing.
    """

    valid = (row_mask[:, :, None] & col_mask[:, None, :]).to(logits.dtype)
    # Numerically stable BCE-with-logits, then focal re-weighting.
    bce = torch.clamp(logits, min=0) - logits * targets + torch.log1p(torch.exp(-torch.abs(logits)))
    probability = torch.exp(-bce)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = alpha_t * (1.0 - probability).pow(gamma) * bce
    denominator = valid.sum().clamp(min=1.0)
    return (focal * valid).sum() / denominator


def decode_association(
    probabilities: np.ndarray,
    row_mask: np.ndarray,
    col_mask: np.ndarray,
    threshold: float = 0.1,
    one_to_one: bool = True,
) -> np.ndarray:
    """Threshold + optional Kuhn-Munkres decoding (Eq. 20 / Algorithm 1)."""

    rows = int(row_mask.sum())
    cols = int(col_mask.sum())
    association = np.zeros_like(probabilities, dtype=bool)
    if rows == 0 or cols == 0:
        return association
    valid = probabilities[:rows, :cols]
    if one_to_one:
        cost = -valid
        try:
            from scipy.optimize import linear_sum_assignment

            row_indices, column_indices = linear_sum_assignment(cost)
        except Exception:  # pragma: no cover - scipy fallback
            pairs = sorted(
                ((valid[i, j], i, j) for i in range(rows) for j in range(cols)), reverse=True
            )
            used_rows, used_cols = set(), set()
            row_indices, column_indices = [], []
            for _, i, j in pairs:
                if i in used_rows or j in used_cols:
                    continue
                row_indices.append(i)
                column_indices.append(j)
                used_rows.add(i)
                used_cols.add(j)
        for i, j in zip(row_indices, column_indices):
            if valid[i, j] >= threshold:
                association[i, j] = True
    else:
        association[:rows, :cols] = valid >= threshold
    return association


def association_metrics(
    probabilities: np.ndarray,
    target: np.ndarray,
    row_mask: np.ndarray,
    col_mask: np.ndarray,
    threshold: float = 0.1,
    one_to_one: bool = True,
) -> dict:
    """Association-only metrics for one example (accuracy, precision, recall, F1)."""

    rows = int(row_mask.sum())
    cols = int(col_mask.sum())
    predicted = decode_association(probabilities, row_mask, col_mask, threshold, one_to_one)
    truth = (target[:rows, :cols] > 0.5)
    prediction = predicted[:rows, :cols]
    true_positive = int(np.logical_and(prediction, truth).sum())
    false_positive = int(np.logical_and(prediction, ~truth).sum())
    false_negative = int(np.logical_and(~prediction, truth).sum())
    true_negative = int(np.logical_and(~prediction, ~truth).sum())
    total = true_positive + false_positive + false_negative + true_negative
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "cells": float(total),
        "true_positives": float(true_positive),
        "false_positives": float(false_positive),
        "false_negatives": float(false_negative),
        "true_negatives": float(true_negative),
        "accuracy": float((true_positive + true_negative) / total) if total else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "rows": float(rows),
        "cols": float(cols),
        "unmatched_rows": float(rows - int(truth.any(axis=1).sum())) if rows else 0.0,
        "unmatched_cols": float(cols - int(truth.any(axis=0).sum())) if cols else 0.0,
    }


def pairs_from_association(association: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a boolean association matrix into ``(row, column)`` pairs."""

    rows, cols = np.nonzero(association)
    return [(int(i), int(j)) for i, j in zip(rows, cols)]
