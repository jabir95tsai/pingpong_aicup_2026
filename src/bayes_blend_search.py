"""Bayesian (gradient-based) weight search over a blend subset.

Dirichlet random search produces ~+0.0019 OOF on R-052 in 300 samples.
Bayesian / scipy.optimize.minimize with COBYLA may find tighter optima.

Each task (action, point, srv) optimized independently.
"""
import json
import os
import sys
from typing import List

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from analyze_oldtest_blend import (  # noqa: E402
    load_components, fast_macro_f1, pad_act19, N_ACTION, N_POINT, ACTION_EVAL, POINT_EVAL,
)


def search_best_weights(comp, subset: List[str], y_a, y_p, y_s, dirichlet_samples=500,
                        bayes_restarts: int = 30, seed: int = 20260522):
    """Two-phase weight search:
    1. Dirichlet random to find good initial point (500 samples)
    2. Bayesian/scipy COBYLA refinement (30 restarts from top-k Dirichlet)
    """
    rng = np.random.default_rng(seed)
    n = len(subset)

    act_stack = np.stack([comp[t]["oof_act"] for t in subset], axis=0)
    pt_stack = np.stack([comp[t]["oof_pt"] for t in subset], axis=0)
    srv_stack = np.stack([comp[t]["oof_srv"] for t in subset], axis=0)

    def eval_action(w):
        b = (w[:, None, None] * act_stack).sum(axis=0)
        return fast_macro_f1(y_a, b.argmax(axis=1), ACTION_EVAL, N_ACTION)

    def eval_point(w):
        b = (w[:, None, None] * pt_stack).sum(axis=0)
        return fast_macro_f1(y_p, b.argmax(axis=1), POINT_EVAL, N_POINT)

    def eval_srv(w):
        b = (w[:, None] * srv_stack).sum(axis=0)
        return roc_auc_score(y_s, b)

    # Phase 1: Dirichlet random search to find good starting points
    best_a = (-1.0, np.full(n, 1.0 / n))
    best_p = (-1.0, np.full(n, 1.0 / n))
    best_s = (-1.0, np.full(n, 1.0 / n))

    top_a, top_p, top_s = [], [], []
    for _ in range(dirichlet_samples):
        w = rng.dirichlet(np.ones(n))
        fa = eval_action(w)
        fp = eval_point(w)
        fs = eval_srv(w)
        if fa > best_a[0]:
            best_a = (fa, w.copy())
        if fp > best_p[0]:
            best_p = (fp, w.copy())
        if fs > best_s[0]:
            best_s = (fs, w.copy())
        top_a.append((fa, w.copy()))
        top_p.append((fp, w.copy()))
        top_s.append((fs, w.copy()))

    # Phase 2: Bayesian/COBYLA refinement
    # Negative score + simplex constraint (w >= 0, sum(w)=1)
    cons = [{"type": "eq", "fun": lambda w: 1.0 - w.sum()}]
    bounds = [(0.0, 1.0)] * n

    def refine(eval_fn, init_w):
        try:
            res = minimize(lambda w: -eval_fn(w),
                           init_w,
                           method="COBYLA",
                           constraints=cons,
                           options={"maxiter": 200, "rhobeg": 0.05},
                           )
            if res.success:
                w = np.clip(res.x, 0, None)
                w = w / max(w.sum(), 1e-9)
                return eval_fn(w), w
        except Exception:
            pass
        return -1.0, init_w

    # Refine from top-bayes_restarts seeds (sorted by score)
    top_a.sort(key=lambda x: -x[0])
    top_p.sort(key=lambda x: -x[0])
    top_s.sort(key=lambda x: -x[0])
    for s, w in top_a[:bayes_restarts]:
        score, ww = refine(eval_action, w)
        if score > best_a[0]:
            best_a = (score, ww)
    for s, w in top_p[:bayes_restarts]:
        score, ww = refine(eval_point, w)
        if score > best_p[0]:
            best_p = (score, ww)
    for s, w in top_s[:bayes_restarts]:
        score, ww = refine(eval_srv, w)
        if score > best_s[0]:
            best_s = (score, ww)

    ov = 0.4 * best_a[0] + 0.4 * best_p[0] + 0.2 * best_s[0]
    return {
        "F1_a": best_a[0], "F1_p": best_p[0], "AUC": best_s[0], "OV": ov,
        "w_a": best_a[1], "w_p": best_p[1], "w_s": best_s[1],
    }


def main() -> None:
    R052 = [
        "v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3",
        "meta_stack_v2_logistic", "v11_mulminet_aug_avg3",
    ]
    R034 = R052[:5]

    print("Loading components ...")
    comp, y_a, y_p, y_s, _, test_uid = load_components(R052)
    print(f"Loaded {len(comp)}/{len(R052)}")

    print("\n=== R-034 PAIR baseline (5-comp, Dirichlet 500 + Bayes refine) ===")
    base = search_best_weights(comp, R034, y_a, y_p, y_s,
                                dirichlet_samples=500, bayes_restarts=30)
    print(f"  OV={base['OV']:.4f}  F1_a={base['F1_a']:.4f}  F1_p={base['F1_p']:.4f}  AUC={base['AUC']:.4f}")

    print("\n=== R-052 7-comp baseline (Dirichlet 500 + Bayes refine) ===")
    r052 = search_best_weights(comp, R052, y_a, y_p, y_s,
                                dirichlet_samples=500, bayes_restarts=30)
    print(f"  OV={r052['OV']:.4f}  F1_a={r052['F1_a']:.4f}  F1_p={r052['F1_p']:.4f}  AUC={r052['AUC']:.4f}")
    print(f"  dOV vs R-034: {r052['OV'] - base['OV']:+.4f}")

    # Compare to previous Dirichlet-only finding (R-052 = +0.0041 with n=300)
    print(f"\n  Previous Dirichlet-only OV (R-052, n=300): 0.3836")
    print(f"  Bayes-refined OV (R-052): {r052['OV']:.4f}")
    print(f"  Bayes lift over Dirichlet-only: {r052['OV'] - 0.3836:+.4f}")

    out = {
        "R034_baseline": {"OV": base["OV"], "F1_a": base["F1_a"],
                           "F1_p": base["F1_p"], "AUC": base["AUC"]},
        "R052_bayes": {"OV": r052["OV"], "F1_a": r052["F1_a"],
                        "F1_p": r052["F1_p"], "AUC": r052["AUC"],
                        "dOV_vs_R034": r052["OV"] - base["OV"]},
        "subset": R052,
        "w_a": list(map(float, r052["w_a"])),
        "w_p": list(map(float, r052["w_p"])),
        "w_s": list(map(float, r052["w_s"])),
    }
    out_path = "submissions/bayes_r052_search.json"
    os.makedirs("submissions", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
