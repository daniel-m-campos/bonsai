"""The single implementation of every benchmark metric.

Primary metric per task (the protocol's rule): r2 for regression, AUC for
binary, accuracy for multiclass, NDCG@10 for ranking. rmse/mae are recorded
as secondaries and never headline a claim.
"""

from __future__ import annotations

import numpy as np


def r2(y_true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rmse(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - pred) ** 2)))


def mae(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - pred)))


def acc(y_true: np.ndarray, pred_labels: np.ndarray) -> float:
    return float(np.mean(y_true == pred_labels))


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via sklearn when available; exact rank-statistic fallback
    (tie-aware) so the module works without the [bench] extra."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except ImportError:
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        sorted_scores = scores[order]
        i = 0
        r = 1.0
        while i < len(scores):
            j = i
            while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            ranks[order[i:j + 1]] = (r + r + (j - i)) / 2.0
            r += j - i + 1
            i = j + 1
        pos = y_true > 0
        n_pos = int(pos.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0)
                     / (n_pos * n_neg))
