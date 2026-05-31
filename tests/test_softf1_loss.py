"""Unit tests for src/train_v11_mulminet_softf1.softf1_loss.

Addresses Codex fix P2-5:
- "uniform -> 1 - 1/K" only holds under balanced-label assumptions; use a
  tiny BALANCED fixture for that test.
- Compare soft-F1 to sklearn macro-F1 ONLY in the one-hot prediction limit.

Plus:
- Masking correctness: if a class has no positive support in the batch, it
  must not contribute to the loss.
- Gradient flows through softmax+softf1 to the input logits.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from train_v11_mulminet_softf1 import softf1_loss  # noqa: E402


# ----- Test fixtures -----------------------------------------------------

K = 5  # 5 eval classes for compactness
EVAL = list(range(K))


def _balanced_targets(n_per_class: int = 4) -> torch.Tensor:
    """Build a balanced label vector: n_per_class samples per class, in order."""
    rows = []
    for c in range(K):
        rows.extend([c] * n_per_class)
    return torch.tensor(rows, dtype=torch.long)


def _onehot_logits(targets: torch.Tensor, scale: float = 50.0) -> torch.Tensor:
    """Logits that, after softmax, are essentially one-hot at the target class."""
    n = targets.shape[0]
    logits = torch.zeros(n, K)
    logits[torch.arange(n), targets] = scale
    return logits


# ----- Tests -------------------------------------------------------------


def test_loss_zero_on_perfect_onehot():
    """Perfectly correct one-hot predictions -> macro F1 -> 1 -> loss -> 0."""
    targets = _balanced_targets(4)
    logits = _onehot_logits(targets)
    loss, _ = softf1_loss(logits, targets, EVAL)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_loss_uniform_balanced_balanced_classes():
    """Codex P2-5: uniform softmax over BALANCED labels.

    With balanced labels (n_per_class samples × K classes = N total):
      Each class has support N/K.
      Each row gives 1/K probability to every class.
      TP_c = (N/K) * (1/K) = N/K^2
      FP_c = (N - N/K) * (1/K) = N(K-1)/K^2
      FN_c = (N/K) * (1 - 1/K) = N(K-1)/K^2
      F1_c = 2 * N/K^2 / (2 * N/K^2 + 2 * N(K-1)/K^2)
           = 2 / (2 + 2(K-1)) = 2 / 2K = 1/K
      Macro F1 = 1/K, Loss = 1 - 1/K.
    """
    n_per_class = 4
    n = K * n_per_class
    targets = _balanced_targets(n_per_class)
    logits = torch.zeros(n, K)  # softmax -> 1/K everywhere
    loss, _ = softf1_loss(logits, targets, EVAL)
    expected = 1.0 - (1.0 / K)
    assert loss.item() == pytest.approx(expected, abs=1e-4), (
        f"uniform pred on balanced labels: loss={loss.item():.4f}, expected={expected:.4f}")


def test_loss_approximates_sklearn_macrof1_onehot_limit():
    """In the one-hot prediction limit, soft-F1 -> sklearn macro F1 exactly.

    Build a non-trivial confusion: every other sample misclassified by +1.
    """
    n_per_class = 6
    targets = _balanced_targets(n_per_class).tolist()
    # Predict y -> y for first half, y+1 (mod K) for second half
    predictions = list(targets)
    for i in range(len(predictions)):
        if i % 2 == 0:
            predictions[i] = (predictions[i] + 1) % K
    predictions_t = torch.tensor(predictions, dtype=torch.long)
    targets_t = torch.tensor(targets, dtype=torch.long)

    logits = _onehot_logits(predictions_t, scale=100.0)
    loss, _ = softf1_loss(logits, targets_t, EVAL)
    soft_f1 = 1.0 - loss.item()
    sklearn_f1 = f1_score(targets, predictions, labels=EVAL, average="macro", zero_division=0)
    assert soft_f1 == pytest.approx(sklearn_f1, abs=1e-3), (
        f"soft_f1={soft_f1:.4f}  sklearn_macro_f1={sklearn_f1:.4f}")


def test_masking_absent_classes():
    """A class with zero positive support in the batch MUST not contribute.

    Build a batch where only classes 0 and 1 have samples; predictions are
    perfect on those. Mean over only the active classes should give F1=1.0,
    so loss=0.
    """
    targets = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    logits = _onehot_logits(targets, scale=50.0)
    loss, diag = softf1_loss(logits, targets, EVAL, return_support=True)
    assert loss.item() == pytest.approx(0.0, abs=1e-3), (
        f"loss with only classes 0,1 active should be 0 (perfect on active), got {loss.item():.4f}")
    assert diag["n_classes_active"] == 2
    assert diag["n_classes_eval"] == K
    assert diag["support_per_class"][0] == 3
    assert diag["support_per_class"][1] == 3
    assert diag["support_per_class"][2] == 0


def test_masking_unequal_class_support():
    """If class 0 has 8 samples and class 1 has 2, mask still correct, F1 still
    computed per class (mean over 2 active classes if predictions perfect)."""
    targets = torch.tensor([0]*8 + [1]*2, dtype=torch.long)
    logits = _onehot_logits(targets, scale=50.0)
    loss, diag = softf1_loss(logits, targets, EVAL, return_support=True)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)
    assert diag["n_classes_active"] == 2
    assert diag["support_per_class"][0] == 8
    assert diag["support_per_class"][1] == 2


def test_gradient_flows():
    """Autograd check: gradient w.r.t. logits is non-zero and finite."""
    targets = _balanced_targets(4)
    logits = torch.randn(K * 4, K, requires_grad=True)
    loss, _ = softf1_loss(logits, targets, EVAL)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # Gradient magnitude should be non-trivial (not all zeros)
    assert logits.grad.abs().sum() > 0


def test_eval_classes_subset():
    """If eval_classes is a strict subset (e.g., [0, 1] of K=5), only those
    classes contribute. Validate by comparing to sklearn run on the same subset."""
    n_per_class = 5
    targets = _balanced_targets(n_per_class)
    # Misclassify some to give non-trivial F1
    preds = targets.clone()
    preds[0] = 1; preds[1] = 2  # cls 0 wrong, cls 1 wrong
    logits = _onehot_logits(preds, scale=100.0)
    eval_subset = [0, 1]
    loss, diag = softf1_loss(logits, targets, eval_subset, return_support=True)
    soft_f1 = 1.0 - loss.item()
    sklearn_f1 = f1_score(targets.numpy(), preds.numpy(),
                          labels=eval_subset, average="macro", zero_division=0)
    assert soft_f1 == pytest.approx(sklearn_f1, abs=1e-3)
    assert diag["n_classes_eval"] == 2
    assert diag["n_classes_active"] == 2


def test_empty_eval_batch_returns_zero_with_grad():
    """If NO eval class is present in the batch (pathological), loss is 0
    with a valid gradient connection (returned as probs.sum() * 0)."""
    # Use eval_classes outside of label range
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    logits = torch.randn(4, K, requires_grad=True)
    loss, _ = softf1_loss(logits, targets, [3, 4])  # classes 3,4 absent
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# ----- Banned-name audit ------------------------------------------------

def test_no_banned_function_names():
    """Sanity: the soft-F1 module should not accidentally reuse banned names."""
    import train_v11_mulminet_softf1
    members = set(dir(train_v11_mulminet_softf1))
    banned = {"final_shot_id", "rally_winner_flag", "terminal_action"}
    overlap = members & banned
    assert not overlap, f"Banned names present: {overlap}"
