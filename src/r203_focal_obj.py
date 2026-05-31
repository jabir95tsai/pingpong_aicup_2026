"""R-203: Focal CE multiclass objective + Cui et al. Class-Balanced weights.

Implements:
  - Focal cross-entropy multiclass objective for LightGBM custom-obj training
    (Lin et al. 2017, "Focal Loss for Dense Object Detection")
    L_focal(p_y) = -alpha_y * (1 - p_y)^gamma * log(p_y)
  - Cui et al. 2019 Class-Balanced weight:
    alpha_c = (1 - beta) / (1 - beta^n_c)
    where n_c is the per-class effective sample count.
  - Optional additional boost for push family (action ids 5,6,13) and Loop (1)
    per R-203 spec.

LightGBM custom objective contract (multiclass):
  - preds:  shape (n_samples * num_class,) flattened, ORDER = column-major:
            [class0_row0..class0_rowN-1, class1_row0..class1_rowN-1, ...]
  - return: (grad, hess) each shape (n_samples * num_class,) same layout.
  - LightGBM provides raw logits z_{i,k}; we softmax internally.

Gradient derivation (k = class index):
  p_{i,k} = softmax(z_i)_k
  L_i     = -alpha_{y_i} * (1 - p_{y_i})^gamma * log(p_{y_i})

  Let f_i = alpha_{y_i} * (1 - p_{y_i})^{gamma-1} *
            (gamma * p_{y_i} * log(p_{y_i}) - (1 - p_{y_i}))

  dL_i/dz_{i,k} = f_i * (delta_{k,y_i} - p_{i,k})

Hessian (diagonal approximation, used by all major focal-loss GBM impls
because exact second derivative explodes near boundary; this is the same
convention as facebookresearch/Detectron2):
  d2L_i/dz_{i,k}^2 ~ |f_i| * p_{i,k} * (1 - p_{i,k})

NUMERICAL SAFETY:
  - p_y clipped to [1e-7, 1-1e-7] before log
  - exp() with row-max subtraction (standard log-sum-exp trick)
  - Hessian floored at 1e-9 (LightGBM requires hess > 0)

This module is self-contained (no Project imports) so it can be unit-tested
in isolation.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Cui et al. 2019 Class-Balanced weights
# ---------------------------------------------------------------------------

def cui_cb_weights(class_counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Compute Cui et al. 2019 Class-Balanced weights.

    alpha_c = (1 - beta) / (1 - beta^n_c)

    Then normalized so that mean weight across present classes = 1.0
    (keeps overall loss magnitude comparable to uniform weighting).

    Args:
        class_counts: shape (K,), integer count per class. Zeros allowed
            (class missing from training data) — those get weight 0.
        beta: hyperparameter in [0,1). 0.999 is the Cui paper default and is
            appropriate for class sizes O(10^3-10^4).

    Returns:
        shape (K,) float array of weights, mean(weights[counts>0]) == 1.0.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    eff_num = 1.0 - np.power(beta, counts)
    # Avoid division by zero for absent classes
    weights = np.where(counts > 0, (1.0 - beta) / np.maximum(eff_num, 1e-12), 0.0)
    # Normalize to mean=1 over present classes
    present = counts > 0
    if present.any():
        weights = weights * (present.sum() / weights[present].sum())
    return weights.astype(np.float32)


def apply_focal_boost(
    cb_weights: np.ndarray,
    boost_classes: list[int],
    boost_factor: float = 1.5,
) -> np.ndarray:
    """Multiplicatively boost selected class weights (R-203 spec: push+Loop).

    Args:
        cb_weights: shape (K,) weights from cui_cb_weights.
        boost_classes: class indices to boost (R-203 default: [1, 5, 6, 13]).
        boost_factor: multiplier (default 1.5).

    Returns:
        New weights array (does not modify input).
    """
    out = cb_weights.copy()
    for c in boost_classes:
        if 0 <= c < len(out):
            out[c] *= boost_factor
    return out


# ---------------------------------------------------------------------------
# Focal multiclass objective for LightGBM
# ---------------------------------------------------------------------------

def make_focal_multiclass_obj(
    num_class: int,
    class_weights: np.ndarray,
    gamma: float = 2.0,
    clip_eps: float = 1e-7,
):
    """Factory returning a LightGBM-compatible custom objective callable.

    Args:
        num_class: K. The trained model must be created with num_class=K.
        class_weights: shape (K,) alpha_c values from cui_cb_weights (possibly
            post-boost). Per-sample alpha is alpha[y_i].
        gamma: focal exponent (Lin 2017 default = 2.0).
        clip_eps: numerical clip on p_y before log.

    Returns:
        Callable obj(preds, dataset) -> (grad, hess) suitable to pass as
        LightGBM `fobj` (LGBM 3.x) or as `objective=` (LGBM 4.x callable form).
    """
    K = int(num_class)
    alpha = np.asarray(class_weights, dtype=np.float64).copy()
    assert alpha.shape == (K,), f"class_weights shape {alpha.shape} != ({K},)"
    g = float(gamma)
    eps = float(clip_eps)

    def _obj(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        # LightGBM passes preds flattened. Multi-class layout = column-major.
        y = dataset.get_label().astype(np.int64)
        n = y.shape[0]
        # Reshape (n*K,) -> (K, n) -> transpose -> (n, K)
        z = preds.reshape(K, n).T.astype(np.float64, copy=False)
        # Softmax (row-wise) with numerical stability
        z_max = z.max(axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        p = exp_z / exp_z.sum(axis=1, keepdims=True)            # (n, K)
        # Per-sample p_y
        rows = np.arange(n)
        p_y = p[rows, y]
        p_y = np.clip(p_y, eps, 1.0 - eps)
        a_y = alpha[y]                                          # (n,)
        # Focal scalar factor f_i
        # f = alpha_y * (1-p_y)^(gamma-1) * (gamma * p_y * log(p_y) - (1-p_y))
        one_minus_py = 1.0 - p_y
        f = a_y * np.power(one_minus_py, g - 1.0) * (
            g * p_y * np.log(p_y) - one_minus_py
        )                                                       # (n,)
        # delta_{k,y} - p_{i,k}
        delta_minus_p = -p.copy()
        delta_minus_p[rows, y] += 1.0                           # (n, K)
        # Gradient: shape (n, K)
        grad = f[:, None] * delta_minus_p
        # Hessian: diagonal approximation
        # h = |f| * p * (1 - p) (per-element)
        hess = np.abs(f)[:, None] * p * (1.0 - p)
        # Floor hessian for LightGBM stability
        hess = np.maximum(hess, 1e-9)
        # Re-flatten to (n*K,) column-major to match LightGBM layout
        grad_flat = grad.T.reshape(-1).astype(np.float64)
        hess_flat = hess.T.reshape(-1).astype(np.float64)
        return grad_flat, hess_flat

    return _obj


# ---------------------------------------------------------------------------
# Multi-class logloss eval (for monitoring; LightGBM custom-obj disables
# built-in metric so we provide one).
# ---------------------------------------------------------------------------

def make_focal_multiclass_eval(num_class: int, name: str = "focal_mlogloss"):
    """Factory returning a LightGBM-compatible custom eval callable.

    Reports mean negative-log-likelihood (NOT focal) for comparability.
    """
    K = int(num_class)

    def _eval(preds: np.ndarray, dataset) -> Tuple[str, float, bool]:
        y = dataset.get_label().astype(np.int64)
        n = y.shape[0]
        z = preds.reshape(K, n).T.astype(np.float64, copy=False)
        z_max = z.max(axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        p = exp_z / exp_z.sum(axis=1, keepdims=True)
        p_y = np.clip(p[np.arange(n), y], 1e-9, 1.0)
        loss = float(-np.log(p_y).mean())
        return name, loss, False  # lower-is-better

    return _eval


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_cb_weights():
    # Roughly v14 action distribution (15 classes; some rare like 8, 14)
    counts = np.array(
        [3000, 8000, 4000, 200, 1500, 5000, 6000, 800, 50, 100, 7000, 1200, 2500, 4500, 60],
        dtype=np.int64,
    )
    w = cui_cb_weights(counts, beta=0.999)
    assert w.shape == (15,)
    # Mean over present classes = 1.0
    assert abs(w[counts > 0].mean() - 1.0) < 1e-5, f"mean={w[counts>0].mean()}"
    # Rare classes should have higher weight than common classes
    assert w[8] > w[1] and w[14] > w[10], "rare class weight must exceed common"
    print(f"  test_cb_weights: PASS  weights={np.round(w, 3).tolist()}")


def _test_focal_boost():
    w0 = np.ones(15, dtype=np.float32)
    w1 = apply_focal_boost(w0, [1, 5, 6, 13], boost_factor=1.5)
    assert w1[1] == 1.5 and w1[5] == 1.5 and w1[6] == 1.5 and w1[13] == 1.5
    assert w1[0] == 1.0 and w1[2] == 1.0
    print(f"  test_focal_boost: PASS")


def _numerical_gradient_check():
    """Verify focal-obj gradient analytically against finite-difference."""
    np.random.seed(42)
    K = 4
    n = 5
    alpha = np.array([1.0, 1.5, 0.8, 1.2])
    gamma = 2.0
    y = np.array([0, 1, 2, 3, 1])
    z = np.random.randn(n, K) * 0.5

    # Analytical gradient via the obj factory
    obj = make_focal_multiclass_obj(num_class=K, class_weights=alpha, gamma=gamma)

    class _MockDS:
        def get_label(self):
            return y.astype(np.float64)

    # LGBM layout: (K, n) flat
    z_flat = z.T.reshape(-1)
    grad_flat, _ = obj(z_flat, _MockDS())
    grad_analytical = grad_flat.reshape(K, n).T  # (n, K)

    # Finite-difference gradient
    def L_focal(z_):
        z_2d = z_.reshape(n, K)
        z_max = z_2d.max(axis=1, keepdims=True)
        e = np.exp(z_2d - z_max)
        p = e / e.sum(axis=1, keepdims=True)
        p_y = np.clip(p[np.arange(n), y], 1e-7, 1 - 1e-7)
        loss = -alpha[y] * np.power(1 - p_y, gamma) * np.log(p_y)
        return loss.sum()

    eps = 1e-5
    grad_numerical = np.zeros_like(z)
    z_flat2 = z.reshape(-1).copy()
    for i in range(n * K):
        zp = z_flat2.copy(); zp[i] += eps
        zm = z_flat2.copy(); zm[i] -= eps
        grad_numerical.flat[i] = (L_focal(zp) - L_focal(zm)) / (2 * eps)

    max_err = np.abs(grad_analytical - grad_numerical).max()
    rel_err = max_err / (np.abs(grad_numerical).max() + 1e-9)
    assert rel_err < 1e-4, f"gradient mismatch: max_err={max_err}, rel={rel_err}"
    print(
        f"  test_numerical_gradient: PASS  "
        f"max_abs_err={max_err:.2e}  rel_err={rel_err:.2e}"
    )


def _test_lgb_end_to_end():
    """Smoke: train a tiny LightGBM model with focal obj and verify it learns."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  test_lgb_end_to_end: SKIP (lightgbm not installed)")
        return

    np.random.seed(0)
    K = 5
    n = 600
    # Synthetic class-imbalanced data
    class_probs = np.array([0.35, 0.30, 0.20, 0.10, 0.05])
    y = np.random.choice(K, size=n, p=class_probs)
    # Features: class-correlated with noise
    X = np.zeros((n, 6))
    for i in range(n):
        X[i, 0] = y[i] + np.random.randn() * 0.5  # informative
        X[i, 1] = (y[i] % 2) + np.random.randn() * 0.8
        X[i, 2:] = np.random.randn(4) * 0.3  # noise

    counts = np.bincount(y, minlength=K)
    w = cui_cb_weights(counts, beta=0.999)
    obj = make_focal_multiclass_obj(num_class=K, class_weights=w, gamma=2.0)
    eval_fn = make_focal_multiclass_eval(num_class=K)

    ds_tr = lgb.Dataset(X[:500], label=y[:500])
    ds_va = lgb.Dataset(X[500:], label=y[500:], reference=ds_tr)
    params = dict(
        num_class=K, num_leaves=15, learning_rate=0.1,
        objective=obj, verbose=-1, metric="None",
    )
    model = lgb.train(
        params, ds_tr, num_boost_round=30,
        valid_sets=[ds_va], feval=eval_fn,
        callbacks=[lgb.log_evaluation(0)],
    )
    raw = model.predict(X[500:], raw_score=True)  # (n_va, K)
    e = np.exp(raw - raw.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    acc = (pred == y[500:]).mean()
    print(f"  test_lgb_end_to_end: PASS  val_acc={acc:.3f} (chance={1/K:.3f})")
    assert acc > 0.30, f"focal-LGB failed to learn on synthetic task: acc={acc}"


if __name__ == "__main__":
    print("=== R-203 focal_obj self-tests ===")
    _test_cb_weights()
    _test_focal_boost()
    _numerical_gradient_check()
    _test_lgb_end_to_end()
    print("All tests passed.")
