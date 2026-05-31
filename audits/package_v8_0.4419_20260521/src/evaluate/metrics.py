"""
Evaluation metrics for table tennis prediction.
Score = 0.4 * macro_f1(actionId) + 0.4 * macro_f1(pointId) + 0.2 * auc_roc(serverGetPoint)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        result = roc_auc_score(y_true, y_prob)
        return float(result) if not np.isnan(result) else 0.5
    except ValueError:
        return 0.5


def optimize_multiclass_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Find per-class scaling factors that maximize macro F1.

    Instead of argmax(proba), we compute argmax(proba * scale) where
    scale[c] is tuned per class.  Scaling up a class makes it predicted
    more often (higher recall, lower precision) — the right trade-off
    when the class is rare and recall is the bottleneck for its F1.

    Returns:
        scales: array of shape (n_classes,) — multiply proba by this
                before argmax to get the optimized predictions.
    """
    scales = np.ones(n_classes)
    best_f1 = macro_f1(y_true, proba.argmax(axis=1))

    # Greedy coordinate ascent: for each class, try a few scale values
    # and keep whichever improves macro F1.  Two passes for convergence.
    candidates = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0]
    for _pass in range(3):
        for c in range(n_classes):
            best_s = scales[c]
            for s in candidates:
                scales[c] = s
                preds = (proba * scales).argmax(axis=1)
                f = macro_f1(y_true, preds)
                if f > best_f1:
                    best_f1 = f
                    best_s = s
            scales[c] = best_s

    return scales


def evaluate_predictions(
    action_true: np.ndarray,
    action_pred: np.ndarray,
    point_true: np.ndarray,
    point_pred: np.ndarray,
    server_true: np.ndarray,
    server_prob: np.ndarray,
) -> dict:
    s1 = macro_f1(action_true, action_pred)
    s2 = macro_f1(point_true, point_pred)
    s3 = auc_roc(server_true, server_prob)
    total = 0.4 * s1 + 0.4 * s2 + 0.2 * s3

    return {
        "actionId_macro_f1": s1,
        "pointId_macro_f1": s2,
        "serverGetPoint_auc": s3,
        "total_score": total,
    }
