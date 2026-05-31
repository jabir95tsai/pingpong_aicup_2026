"""Build R-059r: v15feat_a_oldtest_avg3 SWAP into R-034 + rule_override.

This is the same B-feature SWAP pattern as R-034 (the LB-WIN), but using the
3-seed avg of v15feat_a_oldtest (the FEATURE SET that won R-034 originally).
Hypothesis: combining the winning feature set with seed-averaging gives a
cleaner B-feature swap than the original R-034.

USAGE:
    python -u src/build_r059_candidate.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa: E402
from analyze_oldtest_blend import (  # noqa: E402
    load_components, evaluate_subset_none, build_none_test, write_submission,
)


R034 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]
SLOT = "v14_seed2_v15feat_a"
SWAP_IN = "v14_seed2_v15feat_a_oldtest_avg3"
RID = "R059"
R042_LB = 0.3866
RATIO_CONS = 1.0035
RATIO_OPT = 1.0142
RULE_LIFT_LB = 0.0028


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
    all_tags = list(set(R034 + [SWAP_IN]))
    print(f"Loading {len(all_tags)} components ...")
    comp, y_a, y_p, y_s, _, test_uid = load_components(all_tags)
    if SWAP_IN not in comp:
        raise RuntimeError(f"FATAL: {SWAP_IN} OOF not found in oof_predictions/")

    # R-034 baseline
    print("\n=== R-034 PAIR baseline (n=300 Dirichlet) ===")
    base = evaluate_subset_none(R034, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=300, seed=20260523)
    print(f"  OV={base['OV']:.4f}  F1_a={base['F1_a']:.4f}  "
          f"F1_p={base['F1_p']:.4f}  AUC={base['AUC']:.4f}")
    base_ov = base["OV"]

    # SWAP
    new_subset = [SWAP_IN if t == SLOT else t for t in R034]
    print(f"\n=== {RID}: {SLOT} -> {SWAP_IN} ===")
    m = evaluate_subset_none(new_subset, comp, y_a, y_p, y_s,
                              optimize=True, n_samples=300, seed=20260523)
    d = m["OV"] - base_ov
    pred_lb_lo = m["OV"] * RATIO_CONS + RULE_LIFT_LB
    pred_lb_hi = m["OV"] * RATIO_OPT + RULE_LIFT_LB
    print(f"  OV={m['OV']:.4f}  dOV={d:+.4f}  F1_a={m['F1_a']:.4f}  "
          f"F1_p={m['F1_p']:.4f}  AUC={m['AUC']:.4f}")
    print(f"  pred_LB + rule: {pred_lb_lo:.4f} - {pred_lb_hi:.4f}   "
          f"(R-042 best={R042_LB})")

    # Build base CSV
    pred_a, pred_p, blend_s = build_none_test(
        new_subset, comp, w_a=m["w_a"], w_p=m["w_p"], w_s=m["w_s"]
    )
    fname_base = f"submission_{RID}_v15feat_a_oldtest_avg3_swap.csv"
    out_base = write_submission(test_uid, pred_a, pred_p, blend_s, fname_base)

    # Apply rule_override
    fname_rule = f"submission_{RID}r_v15feat_a_oldtest_avg3_PLUS_RULE.csv"
    out_rule = os.path.join(SUBMISSION_DIR, fname_rule)
    rule_log = run_rule_override(out_base, out_rule)
    last = rule_log.strip().splitlines()[-1] if rule_log else "OK"
    print(f"  rule_override: {last}")

    # Save summary
    summary = {
        "rid": RID, "swap_in": SWAP_IN, "slot": SLOT,
        "OV": float(m["OV"]), "dOV_vs_R034": float(d),
        "F1_a": float(m["F1_a"]), "F1_p": float(m["F1_p"]),
        "AUC": float(m["AUC"]),
        "pred_LB_plus_rule_lo": float(pred_lb_lo),
        "pred_LB_plus_rule_hi": float(pred_lb_hi),
        "submission_base": fname_base,
        "submission_plus_rule": fname_rule,
        "R034_baseline_OV": float(base_ov),
    }
    out_json = os.path.join(SUBMISSION_DIR, "r059_candidate.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_rule}")


if __name__ == "__main__":
    main()
