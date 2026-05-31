"""R-210 — POINT prior-shift correction, OOF smoke (mechanism validation only).

Hypothesis: the R-034 point blend over-predicts FH-short (~4.4x) and
under-predicts mid zones because its predicted class marginal is mismatched
to the true class frequencies. A single-parameter prior-shift correction

    P'(c|x) ∝ P(c|x) * ( pi_true(c) / pi_pred(c) ) ** beta

re-balances the decision toward the true prior. beta=0 -> raw argmax;
beta=1 -> full label-shift correction. We sweep beta on OOF and report
macro-F1_p + per-class F1 + a collapse check (no previously-healthy class
may drop hard). This is theoretically grounded (label/prior shift) and uses
ONE free parameter, so it is far less overfit-prone than 10 free thresholds.

Transfer hypothesis: pi_true(train) ~= pi_true(test_new). If the test class
balance matches train (same sport/rules), the correction transfers. If LB
rejects, the test point-class distribution differs from train.

NO CSV is written here — this only validates the OOF lift.
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_oldtest_blend import (  # noqa: E402
    load_components, evaluate_subset_none, POINT_EVAL, N_POINT,
)

R034_SUBSET = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]
POINT_NAMES = {
    0: "miss/net", 1: "FH-short", 2: "mid-short", 3: "BH-short",
    4: "FH-half", 5: "mid-half", 6: "BH-half", 7: "FH-long", 8: "mid-long", 9: "BH-long",
}


def per_class_f1(y_true, y_pred, labels, n_total):
    cm = np.bincount(y_true.astype(np.int64) * n_total + y_pred.astype(np.int64),
                     minlength=n_total * n_total).reshape(n_total, n_total)
    diag = np.diag(cm); col = cm.sum(0); row = cm.sum(1)
    f1 = {}
    for c in labels:
        tp = diag[c]; fp = col[c] - tp; fn = row[c] - tp
        den = 2 * tp + fp + fn
        f1[c] = 0.0 if den <= 0 else (2 * tp) / den
    return f1, float(np.mean([f1[c] for c in labels]))


def main():
    comp, y_a, y_p, y_s, mask, _ = load_components(R034_SUBSET)
    res = evaluate_subset_none(R034_SUBSET, comp, y_a, y_p, y_s, optimize=True, n_samples=400)
    w_p = res["w_p"]
    pt_stack = np.stack([comp[t]["oof_pt"] for t in R034_SUBSET], axis=0)
    blend_p = (w_p[:, None, None] * pt_stack).sum(axis=0)            # (N, 10) probs
    blend_p = blend_p / blend_p.sum(axis=1, keepdims=True).clip(1e-9)

    # priors
    pi_pred = blend_p.mean(axis=0)                                   # predicted marginal
    pi_true = np.bincount(y_p, minlength=N_POINT).astype(np.float64)
    pi_true = pi_true / pi_true.sum()

    base_f1c, base_macro = per_class_f1(y_p, blend_p.argmax(1), POINT_EVAL, N_POINT)
    print("=" * 70)
    print(" R-210 POINT prior-shift correction — OOF sweep")
    print("=" * 70)
    print(f" raw argmax F1_p = {base_macro:.4f}")
    print(f" {'beta':>5} {'F1_p':>7} {'dF1':>8}   worst-class-drop (healthy classes)")

    ratio = np.where(pi_pred > 1e-9, pi_true / np.maximum(pi_pred, 1e-9), 1.0)
    best = (0.0, base_macro, base_f1c)
    for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]:
        adj = blend_p * (ratio[None, :] ** beta)
        pred = adj.argmax(1)
        f1c, macro = per_class_f1(y_p, pred, POINT_EVAL, N_POINT)
        # collapse check: among classes healthy at baseline (F1>=0.30), worst drop
        drops = [base_f1c[c] - f1c[c] for c in POINT_EVAL if base_f1c[c] >= 0.30]
        worst_drop = max(drops) if drops else 0.0
        flag = "  <== best" if macro > best[1] else ""
        print(f" {beta:>5.1f} {macro:>7.4f} {macro-base_macro:>+8.4f}   worst_drop={worst_drop:+.4f}{flag}")
        if macro > best[1]:
            best = (beta, macro, f1c)

    beta, macro, f1c = best
    print("\n best beta =", beta, " F1_p =", f"{macro:.4f}", f"(dF1_p {macro-base_macro:+.4f}, dOV {0.4*(macro-base_macro):+.4f})")
    print(f"\n per-class F1 at best beta vs raw:")
    print(f" {'cls':>3} {'name':<10} {'raw':>6} {'corr':>6} {'d':>7}")
    for c in POINT_EVAL:
        print(f" {c:>3} {POINT_NAMES[c]:<10} {base_f1c[c]:>6.3f} {f1c[c]:>6.3f} {f1c[c]-base_f1c[c]:>+7.3f}")


if __name__ == "__main__":
    main()
