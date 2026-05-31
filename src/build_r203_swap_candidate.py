"""R-203 — LB candidate via COMPONENT SWAP inside the R-034 PAIR blend.

CRITICAL DESIGN NOTE (why swap, not additive):
  R-094 v2 mixed a new action component ADDITIVELY into R-067cr at low weight
  ((1-a)*base + a*new) and LB-FAILED — that pattern is logged as the toxic class
  `B-impure-additive-low-weight`. R-203 must NOT repeat it. Instead this builder
  REPLACES one action component inside the proven R-034 PAIR set and RE-DERIVES
  the Dirichlet weights — the LB-safe "single-SWAP" pattern (R-069 family).

  R-203's full kernel changes only the v14 ACTION LGB objective (focal CE + Cui
  CB). So only the action half of the blend is touched; point + SGP are taken
  verbatim from the R-067cr LB-best base.

GATING:
  Requires oof_predictions/{SWAP_IN}_oof_act.npy + _test_act.npy, produced by the
  full-5-fold R-203 Kaggle kernel. If absent, prints WAITING and exits 0 (so this
  can be staged before the kernel completes).

USAGE:
    python -u src/build_r203_swap_candidate.py \
        --swap-out v14_seed2_v15feat_a --swap-in v14_r203_full
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR
from analyze_oldtest_blend import load_components, evaluate_subset_none

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
R067CR_BASE = os.path.join(SUBMISSION_DIR,
                           "submission_R067cr_alpha030_v22_blend_PLUS_RULE.csv")

R034_COMPONENTS = ["v11_aug_oldtest", "v11plus", "v13_oldtest",
                   "v14_seed2_v15feat_a", "v16_avg3"]

N_ACTION_FULL = 19


def pad19(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] >= N_ACTION_FULL:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], N_ACTION_FULL), dtype=np.float32)
    out[:, :arr.shape[1]] = arr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-out", default="v14_seed2_v15feat_a",
                    help="Component tag to remove from R-034 PAIR")
    ap.add_argument("--swap-in", default="v14_r203_full",
                    help="Component tag (R-203 focal v14) to insert")
    ap.add_argument("--seed", type=int, default=20260524)
    args = ap.parse_args()

    # Gate: require swap-in OOF to exist
    need = [os.path.join(OOF_DIR, f"{args.swap_in}_oof_act.npy"),
            os.path.join(OOF_DIR, f"{args.swap_in}_test_act.npy")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print("=" * 80)
        print(" R-203 swap candidate — WAITING for full-5-fold R-203 OOF")
        print("=" * 80)
        for p in missing:
            print(f"   missing: {p}")
        print("\n Run the full-5-fold kernel (aicup-r203-focal-full5fold), pull its")
        print(" oof_predictions/v14_r203_full_*.npy into this repo, then re-run.")
        sys.exit(0)

    assert args.swap_out in R034_COMPONENTS, \
        f"{args.swap_out} not in R-034 PAIR set {R034_COMPONENTS}"
    new_components = [args.swap_in if c == args.swap_out else c
                      for c in R034_COMPONENTS]

    print("=" * 80)
    print(f" R-203 COMPONENT SWAP: {args.swap_out} -> {args.swap_in}")
    print("=" * 80)

    # Step 1: baseline R-034 PAIR (for OOF comparison)
    print("\n Step 1: baseline R-034 PAIR OOF")
    comp_oof_b, y_a, y_p, y_s, mask, test_uid = load_components(R034_COMPONENTS)
    base = evaluate_subset_none(R034_COMPONENTS, comp_oof_b, y_a, y_p, y_s,
                                optimize=True, n_samples=300, seed=args.seed)
    print(f"   baseline:  F1_a={base['F1_a']:.4f}  F1_p={base['F1_p']:.4f}")

    # Step 2: swapped PAIR
    print("\n Step 2: swapped R-034 PAIR OOF (re-derived Dirichlet weights)")
    comp_oof_s, y_a2, y_p2, y_s2, mask2, test_uid2 = load_components(new_components)
    swp = evaluate_subset_none(new_components, comp_oof_s, y_a2, y_p2, y_s2,
                               optimize=True, n_samples=300, seed=args.seed)
    print(f"   swapped:   F1_a={swp['F1_a']:.4f}  F1_p={swp['F1_p']:.4f}")

    d_f1a = swp["F1_a"] - base["F1_a"]
    d_f1p = swp["F1_p"] - base["F1_p"]
    d_ov  = 0.4 * d_f1a + 0.4 * d_f1p
    print(f"\n   OOF delta: F1_a {d_f1a:+.4f}  F1_p {d_f1p:+.4f}  (~OV {d_ov:+.4f})")

    # Step 3: reconstruct swapped test ACTION probs (action half only)
    print("\n Step 3: reconstruct swapped test action argmax")
    comp_test = {t: pad19(np.load(os.path.join(OOF_DIR, f"{t}_test_act.npy")))
                 for t in new_components}
    test_stack = np.stack([comp_test[t] for t in new_components], axis=0)
    swapped_test_act = (swp["w_a"][:, None, None] * test_stack).sum(axis=0)
    new_action = swapped_test_act.argmax(axis=1)

    # Step 4: build CSV — action from swapped blend, point + SGP from R-067cr
    r067 = pd.read_csv(R067CR_BASE)
    orig = pd.read_csv(R067CR_BASE)
    n_act_diff = int((new_action != orig["actionId"].to_numpy()).sum())
    r067["actionId"] = new_action
    assert (r067["pointId"].to_numpy() == orig["pointId"].to_numpy()).all()
    assert (r067["serverGetPoint"].to_numpy() == orig["serverGetPoint"].to_numpy()).all()

    raw_csv = os.path.join(SUBMISSION_DIR,
                           f"submission_R203_swap_{args.swap_in}.csv")
    rule_csv = raw_csv.replace(".csv", "_PLUS_RULE.csv")
    r067.to_csv(raw_csv, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\n Step 4: action diffs vs R-067cr base = {n_act_diff}")
    print(f"   saved raw: {raw_csv}")

    # Step 5: rule_override Layer A
    print("\n Step 5: apply rule_override Layer A")
    cmd = [sys.executable, "-u", os.path.join("src", "apply_rule_override.py"),
           "--input", raw_csv, "--train", os.path.join("data", "train.csv"),
           "--test", os.path.join("data", "test_new.csv"),
           "--output", rule_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   WARN rule_override failed: {r.stderr[-500:]}")
    else:
        print("\n".join(f"   {ln}" for ln in r.stdout.strip().split("\n")[-4:]))

    # Step 6: manifest + v0.4 report
    manifest = {
        "rid": "R-203",
        "mechanism": "component-swap (NOT additive): replace v14 action component "
                     "with focal-CE+Cui-CB v14 inside R-034 PAIR, re-derive Dirichlet",
        "swap_out": args.swap_out,
        "swap_in": args.swap_in,
        "oof_delta_f1a": round(float(d_f1a), 5),
        "oof_delta_f1p": round(float(d_f1p), 5),
        "oof_delta_ov_est": round(float(d_ov), 5),
        "n_action_diffs_vs_base": n_act_diff,
        "raw_csv": raw_csv,
        "output_csv": rule_csv,
        "feature_set_caveat":
            "If swap-in uses a different feature_set than swap-out, the OOF delta "
            "conflates loss-change with feature-change. For a clean focal-only test, "
            "swap-in should match swap-out's feature set. Default v14_r203_full uses "
            "feature_set=v9.",
        "v04_candidate_report": {
            "theoretical_generalization_reason":
                "Focal CE + Cui CB reshape the v14 action decision boundary toward "
                "minority/hard classes (macro-F1 aligned) without changing data or "
                "features. Replacing the standard-CE v14 action component with the "
                "focal one inside the proven R-034 PAIR tests whether that reshaped "
                "component improves the blend. Component-swap (re-optimized weights) "
                "avoids the additive-low-weight pattern that made R-094 v2 toxic.",
            "why_transfers_to_test_new":
                "B-feature class (same arch/data, new objective) ~0.9 LB transfer "
                "empirically (R-034). No test-specific or player-specific signal.",
            "smoke_sanity_pass": "set from fold-1 smoke verdict (push F1 >= +0.005 "
                                 "AND est OV >= +0.003)",
            "lb_probe_worthy": "iff OOF swap delta OV >= +0.003 AND smoke passed",
            "lb_confirm_hypothesis":
                "LB ΔOV >= +0.003 => focal-trained component improves the blend; "
                "open the B-feature focal-objective track (extend to XGB, point).",
            "lb_reject_hypothesis":
                "LB ΔOV <= -0.003 => focal component does not transfer as a blend "
                "member; close GBM-focal route, log calibration entry (2nd failed "
                "loss-level imbalance attack after R-094 v2).",
        },
    }
    mpath = os.path.join(SUBMISSION_DIR, f"r203_swap_{args.swap_in}_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved manifest: {mpath}")

    print("\n" + "=" * 80)
    if d_ov >= 0.003:
        print(" ARTIFACT_READY_FOR_JABIR_UPLOAD_REVIEW")
        print(f" File: {rule_csv}")
        print(f" OOF swap ΔOV est {d_ov:+.4f} (>= +0.003 threshold)")
    else:
        print(" NOT ARTIFACT-READY — OOF swap delta below +0.003 LB-probe threshold")
        print(f" OOF swap ΔOV est {d_ov:+.4f}")
        print(" Per lb_reject path: do not upload; log calibration entry.")
    print("=" * 80)


if __name__ == "__main__":
    main()
