"""Build 3 B-feature SWAP candidates for tomorrow's LB upload.

All candidates follow the proven R-034 LB-WIN pattern: SWAP one R-034 slot with a
B-feature class component, then apply rule_override post-process.

Candidates:
  R-058r: v14_seed2_v15feat_a -> v14_seed2_v15feat_c_oldtest_avg3
          (fresh 3-seed avg of v15feat_c — score-pressure features)
  R-035r: v14_seed2_v15feat_a -> v14_recvhand_oldtest
          (receiver-hand B-feature, was Stage 1 +0.0004 OOF in audit)
  R-037r: v14_seed2_v15feat_a -> v14_recvprofile_oldtest
          (receiver-profile B-feature, was Stage 1 +0.0007 OOF in audit)

All 3 use Dirichlet weight search (NOT Bayes — Bayes is banned per R-055 lesson)
and apply_rule_override post-process.

USAGE:
    python -u src/build_overnight_candidates.py
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
SLOT = "v14_seed2_v15feat_a"  # swap-out slot for all candidates
R034_LB = 0.3838  # anchor
R042_LB = 0.3866  # current best (R-034 + rule_override)
RATIO_CONS = 1.0035
RATIO_OPT = 1.0142  # R-042 actual
RULE_LIFT_LB = 0.0028  # observed R-042 lift

CANDIDATES = [
    ("R058", "v14_seed2_v15feat_c_oldtest_avg3",
     "v15feat_c 3-seed avg (B-feature, freshest)"),
    ("R035", "v14_recvhand_oldtest",
     "v14_recvhand (B-feature, audit +0.0004 OOF)"),
    ("R037", "v14_recvprofile_oldtest",
     "v14_recvprofile (B-feature, audit +0.0007 OOF)"),
]


def run_rule_override(in_csv: str, out_csv: str) -> str:
    train_csv = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_csv = os.path.join(PROJECT_ROOT, "data", "test_new.csv")
    script = os.path.join(PROJECT_ROOT, "src", "apply_rule_override.py")
    cmd = ["python", "-u", script,
           "--input", in_csv,
           "--train", train_csv,
           "--test", test_csv,
           "--output", out_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rule_override failed: {r.stderr}")
    return r.stdout


def main() -> None:
    # Need R-034 components + all 3 candidate swap-ins
    swap_ins = [c[1] for c in CANDIDATES]
    all_tags = list(set(R034 + swap_ins))
    print(f"Loading {len(all_tags)} components ...")
    comp, y_a, y_p, y_s, _, test_uid = load_components(all_tags)
    available = {c: c in comp for c in swap_ins}
    print(f"  Available swap-ins: {available}")

    # R-034 baseline (apples-to-apples for delta computation)
    print("\n=== R-034 PAIR baseline (n=300 Dirichlet) ===")
    base = evaluate_subset_none(R034, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=300, seed=20260523)
    print(f"  OV={base['OV']:.4f}  F1_a={base['F1_a']:.4f}  "
          f"F1_p={base['F1_p']:.4f}  AUC={base['AUC']:.4f}")
    base_ov = base["OV"]

    rows = []
    for rid, swap_in, desc in CANDIDATES:
        if not available.get(swap_in, False):
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
        print(f"  OV={m['OV']:.4f}  dOV={d:+.4f}  "
              f"F1_a={m['F1_a']:.4f}  F1_p={m['F1_p']:.4f}  AUC={m['AUC']:.4f}")
        print(f"  pred_LB+rule: {pred_lb_lo:.4f} - {pred_lb_hi:.4f}   "
              f"(current best R-042={R042_LB})")

        # Build base submission CSV
        pred_a, pred_p, blend_s = build_none_test(
            new_subset, comp, w_a=m["w_a"], w_p=m["w_p"], w_s=m["w_s"]
        )
        fname_base = f"submission_{rid}_{swap_in[:30]}_swap.csv"
        out_base = write_submission(test_uid, pred_a, pred_p, blend_s, fname_base)

        # Apply rule_override
        fname_rule = f"submission_{rid}r_{swap_in[:30]}_PLUS_RULE.csv"
        out_rule = os.path.join(SUBMISSION_DIR, fname_rule)
        rule_log = run_rule_override(out_base, out_rule)
        print(f"  rule_override: {rule_log.strip().splitlines()[-1] if rule_log else 'OK'}")

        rows.append({
            "rid": rid, "swap_in": swap_in, "desc": desc,
            "OV": float(m["OV"]), "dOV_vs_R034": float(d),
            "F1_a": float(m["F1_a"]), "F1_p": float(m["F1_p"]),
            "AUC": float(m["AUC"]),
            "pred_LB_plus_rule_lo": float(pred_lb_lo),
            "pred_LB_plus_rule_hi": float(pred_lb_hi),
            "submission_base": fname_base,
            "submission_plus_rule": fname_rule,
            "w_a": list(map(float, m["w_a"])),
            "w_p": list(map(float, m["w_p"])),
            "w_s": list(map(float, m["w_s"])),
        })

    # Sort by OV and print summary
    rows.sort(key=lambda r: -r["OV"])
    print("\n" + "=" * 78)
    print(" OVERNIGHT CANDIDATES — ranked by OOF")
    print("=" * 78)
    print(f" Current LB best: R-042 = {R042_LB} (R-034 + rule_override)")
    print()
    for i, r in enumerate(rows, 1):
        print(f"  #{i}  {r['rid']}r  swap={r['swap_in']}")
        print(f"       OOF={r['OV']:.4f}  dOV={r['dOV_vs_R034']:+.4f}  "
              f"pred LB+rule: {r['pred_LB_plus_rule_lo']:.4f}-{r['pred_LB_plus_rule_hi']:.4f}")
        print(f"       file: {r['submission_plus_rule']}")
        print()

    # Save summary JSON
    out_json = os.path.join(SUBMISSION_DIR, "overnight_candidates.json")
    with open(out_json, "w") as f:
        json.dump({
            "R034_baseline": {"OV": float(base_ov),
                              "F1_a": float(base["F1_a"]),
                              "F1_p": float(base["F1_p"]),
                              "AUC": float(base["AUC"])},
            "R042_LB": R042_LB,
            "candidates": rows,
        }, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
