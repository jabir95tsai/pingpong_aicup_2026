"""Build R-062r: v14_seed2_v16match_v2 SWAP into R-034 + rule_override.

R-032 v2 Kaggle kernel completed 2026-05-23 with v16match_v2 features (cross-rally
LORO match-pair aggregates). Standalone metrics:
  FINAL OV (base): 0.3683
  FINAL OV (opt):  0.3747  (vs v15feat_a ~0.3690 = +0.0057)

This is B-feature class (same arch, new features) — same family as R-034's
LB-WIN. Codex-approved scope, capped K=22 LORO features.

USAGE:
    python -u src/build_r062_v16match_candidate.py
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa: E402
from analyze_oldtest_blend import (  # noqa: E402
    load_components, evaluate_subset_none, build_none_test, write_submission,
)


R034 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]
SLOT = "v14_seed2_v15feat_a"
SWAP_IN = "v14_seed2_v16match_v2"
RID = "R062"
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
        raise RuntimeError(f"FATAL: {SWAP_IN} OOF not found")

    # Baseline
    print("\n=== R-034 PAIR baseline (n=300 Dirichlet) ===")
    base = evaluate_subset_none(R034, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=300, seed=20260523)
    print(f"  OV={base['OV']:.4f}  F1a={base['F1_a']:.4f}  F1p={base['F1_p']:.4f}  AUC={base['AUC']:.4f}")
    base_ov = base["OV"]

    # SWAP
    new_subset = [SWAP_IN if t == SLOT else t for t in R034]
    print(f"\n=== {RID}: {SLOT} -> {SWAP_IN} ===")
    print(f"     v16match_v2 LORO cross-rally features (R-032 v2, Codex-approved)")
    m = evaluate_subset_none(new_subset, comp, y_a, y_p, y_s,
                              optimize=True, n_samples=300, seed=20260523)
    d = m["OV"] - base_ov
    pred_lb_lo = m["OV"] * RATIO_CONS + RULE_LIFT_LB
    pred_lb_hi = m["OV"] * RATIO_OPT + RULE_LIFT_LB
    marker = " *** STRONG ***" if d >= 0.002 else (" *POSITIVE*" if d > 0 else "")
    print(f"  OV={m['OV']:.4f}  dOV={d:+.4f}  F1a={m['F1_a']:.4f}  F1p={m['F1_p']:.4f}  AUC={m['AUC']:.4f}{marker}")
    print(f"  pred_LB+rule: {pred_lb_lo:.4f} - {pred_lb_hi:.4f}   (R-042 best={R042_LB})")

    # Build base + rule CSVs
    pred_a, pred_p, blend_s = build_none_test(
        new_subset, comp, w_a=m["w_a"], w_p=m["w_p"], w_s=m["w_s"]
    )
    fname_base = f"submission_{RID}_v16match_v2_swap.csv"
    out_base = write_submission(test_uid, pred_a, pred_p, blend_s, fname_base)

    fname_rule = f"submission_{RID}r_v16match_v2_PLUS_RULE.csv"
    out_rule = os.path.join(SUBMISSION_DIR, fname_rule)
    rule_log = run_rule_override(out_base, out_rule)
    last = rule_log.strip().splitlines()[-1] if rule_log else "OK"
    n_changes = sum(1 for line in rule_log.splitlines() if "rally=" in line)
    print(f"  rule_override: {n_changes} row changes; last: {last}")

    summary = {
        "rid": RID, "swap_in": SWAP_IN, "slot": SLOT,
        "OV": float(m["OV"]), "dOV_vs_R034": float(d),
        "F1_a": float(m["F1_a"]), "F1_p": float(m["F1_p"]), "AUC": float(m["AUC"]),
        "pred_LB_plus_rule_lo": float(pred_lb_lo),
        "pred_LB_plus_rule_hi": float(pred_lb_hi),
        "submission_plus_rule": fname_rule,
        "R034_baseline_OV": float(base_ov),
        "rule_override_changes": n_changes,
    }
    out_json = os.path.join(SUBMISSION_DIR, "r062_candidate.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_rule}")


if __name__ == "__main__":
    main()
