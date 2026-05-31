"""Bayes weight search over R-034 PAIR (5 SAFE LB-validated components only).

Per user 2026-05-24 ("run all" option #2):
This is the SAFE companion to bayes_blend_search.py (R-052 7-comp), which
LB-failed catastrophically (R-055 = −0.0141) because it included v11_mulminet
(B-impure toxic) + meta_stack_v2_logistic (B-meta toxic). Bayes amplified
those toxic components.

R-034 PAIR has 5 LB-VALIDATED components ONLY:
  v11_aug_oldtest, v11plus, v13_oldtest, v14_seed2_v15feat_a, v16_avg3

All 5 transferred at ratio 1.01+ to LB-best 0.3838 (R-034) / 0.3866 (R-042
with rule_override). Bayes search on these can ONLY redistribute weight
among proven-safe components. No toxic-class risk.

Output:
- submissions/bayes_r034_safe_search.json (weights + OOF metrics)
- submissions/submission_R068_bayes_r034_safe.csv (base NONE blend)
- submissions/submission_R068r_bayes_r034_safe_PLUS_RULE.csv (+ rule_override)

USAGE:
    python -u src/bayes_blend_search_r034_safe.py
"""
import json
import os
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from analyze_oldtest_blend import (  # noqa: E402
    load_components, fast_macro_f1, pad_act19,
    N_ACTION, N_POINT, ACTION_EVAL, POINT_EVAL,
    build_none_test, write_submission,
)
from bayes_blend_search import search_best_weights  # noqa: E402
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa: E402

R034_SAFE = [
    "v11_aug_oldtest",   # B-pure (R-027 lineage, validated)
    "v11plus",            # original transformer (validated in R-027/R-034)
    "v13_oldtest",        # B-pure GBM
    "v14_seed2_v15feat_a",  # B-feature LB-WIN (R-034 +0.0028 vs R-027)
    "v16_avg3",           # multi-seed avg, LB-validated
]

# Anchors from R-042 (current LB-best)
R042_LB = 0.3866
RATIO_CONS = 1.0035   # R-027 conservative
RATIO_OPT = 1.0142    # R-042 observed
RULE_LIFT_LB = 0.0028  # R-042 observed rule_override lift


def run_rule_override(in_csv: str, out_csv: str) -> str:
    train_csv = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_csv = os.path.join(PROJECT_ROOT, "data", "test_new.csv")
    script = os.path.join(PROJECT_ROOT, "src", "apply_rule_override.py")
    cmd = ["python", "-u", script,
           "--input", in_csv, "--train", train_csv,
           "--test", test_csv, "--output", out_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rule_override failed: {r.stderr}")
    return r.stdout


def main() -> None:
    print("=" * 78)
    print(" R-068 Bayes weight search — R-034 PAIR (5 SAFE LB-validated comps only)")
    print(" SAFETY: zero toxic-class risk; all 5 components LB-tested OK")
    print("=" * 78)

    comp, y_a, y_p, y_s, _, test_uid = load_components(R034_SAFE)
    print(f"\n Loaded {len(comp)}/{len(R034_SAFE)} components.")

    # ─── Dirichlet baseline for comparison ──────────────────────────────────
    print("\n--- Dirichlet baseline (n=500 samples, current production) ---")
    from analyze_oldtest_blend import evaluate_subset_none
    base = evaluate_subset_none(R034_SAFE, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=500, seed=20260524)
    print(f"  OV={base['OV']:.4f}  F1_a={base['F1_a']:.4f}  "
          f"F1_p={base['F1_p']:.4f}  AUC={base['AUC']:.4f}")
    base_ov = base["OV"]

    # ─── Bayes (Dirichlet 500 + COBYLA 30 restarts) ──────────────────────────
    print("\n--- Bayes search (Dirichlet 500 + COBYLA 30 restarts) ---")
    bayes = search_best_weights(comp, R034_SAFE, y_a, y_p, y_s,
                                 dirichlet_samples=500, bayes_restarts=30,
                                 seed=20260524)
    print(f"  OV={bayes['OV']:.4f}  F1_a={bayes['F1_a']:.4f}  "
          f"F1_p={bayes['F1_p']:.4f}  AUC={bayes['AUC']:.4f}")
    print(f"  Bayes lift over Dirichlet: {bayes['OV'] - base_ov:+.4f}")

    pred_lb_lo = bayes["OV"] * RATIO_CONS + RULE_LIFT_LB
    pred_lb_hi = bayes["OV"] * RATIO_OPT + RULE_LIFT_LB
    delta_lb_lo = pred_lb_lo - R042_LB
    delta_lb_hi = pred_lb_hi - R042_LB
    print(f"\n Predicted LB+rule: {pred_lb_lo:.4f} - {pred_lb_hi:.4f}")
    print(f"  vs R-042 {R042_LB}: {delta_lb_lo:+.4f} to {delta_lb_hi:+.4f}")

    # ─── Build candidate CSV ─────────────────────────────────────────────────
    pred_a, pred_p, blend_s = build_none_test(
        R034_SAFE, comp,
        w_a=bayes["w_a"], w_p=bayes["w_p"], w_s=bayes["w_s"],
    )
    fname_base = "submission_R068_bayes_r034_safe.csv"
    out_base = write_submission(test_uid, pred_a, pred_p, blend_s, fname_base)

    fname_rule = "submission_R068r_bayes_r034_safe_PLUS_RULE.csv"
    out_rule = os.path.join(SUBMISSION_DIR, fname_rule)
    rule_log = run_rule_override(out_base, out_rule)
    print(f"\n rule_override: {len([l for l in rule_log.splitlines() if 'rally=' in l])} row changes")

    # ─── Save manifest ──────────────────────────────────────────────────────
    out_json = {
        "rid": "R-068",
        "ts": "2026-05-24",
        "subset": R034_SAFE,
        "method": "Bayes (Dirichlet 500 + COBYLA 30 restarts)",
        "dirichlet_baseline": {
            "OV": float(base["OV"]),
            "F1_a": float(base["F1_a"]),
            "F1_p": float(base["F1_p"]),
            "AUC": float(base["AUC"]),
        },
        "bayes_refined": {
            "OV": float(bayes["OV"]),
            "F1_a": float(bayes["F1_a"]),
            "F1_p": float(bayes["F1_p"]),
            "AUC": float(bayes["AUC"]),
            "lift_vs_dirichlet": float(bayes["OV"] - base_ov),
        },
        "weights": {
            "w_a": list(map(float, bayes["w_a"])),
            "w_p": list(map(float, bayes["w_p"])),
            "w_s": list(map(float, bayes["w_s"])),
        },
        "predicted_LB_plus_rule_lo": float(pred_lb_lo),
        "predicted_LB_plus_rule_hi": float(pred_lb_hi),
        "vs_R042_lo": float(delta_lb_lo),
        "vs_R042_hi": float(delta_lb_hi),
        "submission_base": fname_base,
        "submission_plus_rule": fname_rule,
        "safety_note": "All 5 components LB-validated; zero toxic-class risk.",
    }
    out_path = os.path.join(SUBMISSION_DIR, "bayes_r034_safe_search.json")
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n Saved: {out_path}")

    print("\n=== R-068 DECISION GATE ===")
    if bayes["OV"] - base_ov >= 0.0010:
        print(f"  Bayes lift +{bayes['OV'] - base_ov:.4f} >= +0.0010 → CANDIDATE viable")
        print(f"  Pred LB+rule midpoint: {(pred_lb_lo + pred_lb_hi)/2:.4f}")
        print(f"  vs R-042 0.3866: {((pred_lb_lo + pred_lb_hi)/2) - R042_LB:+.4f}")
    else:
        print(f"  Bayes lift +{bayes['OV'] - base_ov:.4f} < +0.0010 → marginal; consider parking")


if __name__ == "__main__":
    main()
