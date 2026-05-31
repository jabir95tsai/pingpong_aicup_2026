"""Rebuild R-037r with the CORRECT v14_recvprofile (NO _oldtest) swap.

The original overnight chain 1 used `v14_recvprofile_oldtest` which was NOT
the audit-tested variant. The 2026-05-21 parked audit's +0.0007 OOF lift was
for `v14_recvprofile` (no oldtest suffix). Rebuilding R-037r with the
correct variant.

USAGE:
    python -u src/rebuild_recvprofile_candidate.py
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
R042_LB = 0.3866
RATIO_CONS = 1.0035
RATIO_OPT = 1.0142
RULE_LIFT_LB = 0.0028

# Audit-confirmed B-feature swaps (non-oldtest variants per parked_audit_summary.csv)
CANDIDATES = [
    ("R060", "v14_recvprofile",
     "v14_recvprofile NO-oldtest (audit dOV +0.0007 — corrected variant)"),
    ("R061", "v14_recvhand",
     "v14_recvhand NO-oldtest (audit-listed B-feature, retesting)"),
]


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
    swap_ins = [c[1] for c in CANDIDATES]
    all_tags = list(set(R034 + swap_ins))
    print(f"Loading {len(all_tags)} components ...")
    comp, y_a, y_p, y_s, _, test_uid = load_components(all_tags)

    print("\n=== R-034 PAIR baseline (n=300 Dirichlet) ===")
    base = evaluate_subset_none(R034, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=300, seed=20260523)
    print(f"  OV={base['OV']:.4f}")
    base_ov = base["OV"]

    rows = []
    for rid, swap_in, desc in CANDIDATES:
        if swap_in not in comp:
            print(f"\n[{rid}] SKIP - {swap_in} not loaded")
            continue
        new_subset = [swap_in if t == SLOT else t for t in R034]
        print(f"\n=== {rid}: {SLOT} -> {swap_in} ===")
        print(f"     {desc}")
        m = evaluate_subset_none(new_subset, comp, y_a, y_p, y_s,
                                  optimize=True, n_samples=300, seed=20260523)
        d = m["OV"] - base_ov
        pred_lb_lo = m["OV"] * RATIO_CONS + RULE_LIFT_LB
        pred_lb_hi = m["OV"] * RATIO_OPT + RULE_LIFT_LB
        print(f"  OV={m['OV']:.4f}  dOV={d:+.4f}  F1_a={m['F1_a']:.4f}  F1_p={m['F1_p']:.4f}  AUC={m['AUC']:.4f}")
        print(f"  pred_LB+rule: {pred_lb_lo:.4f} - {pred_lb_hi:.4f}   (R-042 best={R042_LB})")

        pred_a, pred_p, blend_s = build_none_test(
            new_subset, comp, w_a=m["w_a"], w_p=m["w_p"], w_s=m["w_s"]
        )
        fname_base = f"submission_{rid}_{swap_in}_swap.csv"
        out_base = write_submission(test_uid, pred_a, pred_p, blend_s, fname_base)

        fname_rule = f"submission_{rid}r_{swap_in}_PLUS_RULE.csv"
        out_rule = os.path.join(SUBMISSION_DIR, fname_rule)
        rule_log = run_rule_override(out_base, out_rule)
        last = rule_log.strip().splitlines()[-1] if rule_log else "OK"
        print(f"  rule_override: {last}")

        rows.append({
            "rid": rid, "swap_in": swap_in, "desc": desc,
            "OV": float(m["OV"]), "dOV_vs_R034": float(d),
            "pred_LB_plus_rule_lo": float(pred_lb_lo),
            "pred_LB_plus_rule_hi": float(pred_lb_hi),
            "submission_plus_rule": fname_rule,
        })

    out_json = os.path.join(SUBMISSION_DIR, "rebuilt_recvprofile_candidates.json")
    with open(out_json, "w") as f:
        json.dump({"candidates": rows, "R034_baseline_OV": float(base_ov)}, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
