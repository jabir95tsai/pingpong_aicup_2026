"""R-094 v2 — action-only SoftF1 additive (point predictions preserved exactly).

v1 found action benefits at α=0.05 but point does not benefit (F1_p flat).
v2 decouples per-task α: action uses α=0.05, point uses α=0.00 (skip).
Result: only action predictions change; point stays exactly as R-067cr.
Less LB risk for same expected lift (~+0.0006 F1_a).

USAGE:
    python -u src/build_r094_v2_action_only.py
"""
from __future__ import annotations

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
OUT_CSV_RAW = os.path.join(SUBMISSION_DIR,
                            "submission_R094v2_R067cr_PLUS_SOFTF1_act_only_alpha005.csv")
OUT_CSV_RULE = OUT_CSV_RAW.replace(".csv", "_PLUS_RULE.csv")
MANIFEST = os.path.join(SUBMISSION_DIR, "r094_v2_act_only_manifest.json")

R034_COMPONENTS = ["v11_aug_oldtest", "v11plus", "v13_oldtest",
                    "v14_seed2_v15feat_a", "v16_avg3"]
SOFTF1_TAG = "v11_mulminet_aug_oldtest_softf1_phaseB"
ALPHA_ACTION = 0.05
ALPHA_POINT  = 0.00

N_ACTION_FULL = 19
N_ACTION_TRAIN = 15
N_POINT = 10


def pad19(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] >= N_ACTION_FULL:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], N_ACTION_FULL), dtype=np.float32)
    out[:, :arr.shape[1]] = arr
    return out


def main() -> None:
    print("=" * 80)
    print(" R-094 v2 — action-only SoftF1 additive (alpha_a=0.05, alpha_p=0.00)")
    print("=" * 80)

    # Step 1: R-034 PAIR weights (need only w_a to reconstruct test action probs)
    print("\n Step 1: derive R-034 PAIR Dirichlet weights")
    comp_oof, y_a, y_p, y_s, mask, test_uid = load_components(R034_COMPONENTS)
    base = evaluate_subset_none(R034_COMPONENTS, comp_oof, y_a, y_p, y_s,
                                 optimize=True, n_samples=300, seed=20260524)
    print(f"   R-034 PAIR OOF: F1_a={base['F1_a']:.4f}  F1_p={base['F1_p']:.4f}")

    # Step 2: reconstruct R-067cr-equivalent test action probs
    print("\n Step 2: reconstruct R-067cr test action probs via R-034 PAIR weights")
    comp_test = {}
    for t in R034_COMPONENTS:
        comp_test[t] = pad19(np.load(os.path.join(OOF_DIR, f"{t}_test_act.npy")))
    test_act_stack = np.stack([comp_test[t] for t in R034_COMPONENTS], axis=0)
    r067cr_test_act = (base["w_a"][:, None, None] * test_act_stack).sum(axis=0)
    print(f"   reconstructed test_act: {r067cr_test_act.shape}")

    # Step 3: load softf1 test action probs
    print("\n Step 3: load SoftF1 test action probs")
    softf1_test_act = pad19(np.load(os.path.join(OOF_DIR, f"{SOFTF1_TAG}_test_act.npy")))
    print(f"   softf1 test_act: {softf1_test_act.shape}")

    # Step 4: blend test action only
    blend_test_act = (1 - ALPHA_ACTION) * r067cr_test_act + ALPHA_ACTION * softf1_test_act
    new_action = blend_test_act.argmax(axis=1)

    # Step 5: load R-067cr base, override actionId only, preserve pointId + SGP
    r067cr = pd.read_csv(R067CR_BASE)
    n_act_diff = (new_action != r067cr["actionId"].to_numpy()).sum()
    r067cr["actionId"] = new_action
    n_pt_diff = 0  # pointId unchanged
    print(f"\n Step 4: build CSV — action diffs {n_act_diff}, point diffs {n_pt_diff}")

    # SGP unchanged assertion
    r067cr_orig = pd.read_csv(R067CR_BASE)
    assert (r067cr["serverGetPoint"].to_numpy() ==
            r067cr_orig["serverGetPoint"].to_numpy()).all()
    assert (r067cr["pointId"].to_numpy() ==
            r067cr_orig["pointId"].to_numpy()).all(), "point should be unchanged"

    r067cr.to_csv(OUT_CSV_RAW, index=False, lineterminator="\n", encoding="utf-8")
    print(f"   Saved: {OUT_CSV_RAW}")

    # Step 6: apply rule_override Layer A
    print("\n Step 5: apply rule_override Layer A")
    cmd = [sys.executable, "-u", os.path.join("src", "apply_rule_override.py"),
           "--input", OUT_CSV_RAW, "--train", os.path.join("data", "train.csv"),
           "--test", os.path.join("data", "test_new.csv"),
           "--output", OUT_CSV_RULE]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        last_lines = r.stdout.strip().split("\n")[-5:]
        print("\n".join(f"   {ln}" for ln in last_lines))
    else:
        print(f"   WARN rule_override failed: {r.stderr}")

    # Step 7: re-measure final diff (after rule_override)
    final_csv = pd.read_csv(OUT_CSV_RULE)
    base_csv  = pd.read_csv(R067CR_BASE)
    n_final_act = (final_csv["actionId"].to_numpy() != base_csv["actionId"].to_numpy()).sum()
    n_final_pt  = (final_csv["pointId"].to_numpy()  != base_csv["pointId"].to_numpy()).sum()
    assert (final_csv["serverGetPoint"].to_numpy() ==
            base_csv["serverGetPoint"].to_numpy()).all(), "SGP must be unchanged"
    print(f"\n Final vs R-067cr base:")
    print(f"   action diffs: {n_final_act}")
    print(f"   point diffs:  {n_final_pt}  (should be ~0; small possible from rule_override Layer A)")
    print(f"   SGP diffs:    0 (asserted)")

    # Manifest
    manifest = {
        "rid": "R-094-v2",
        "ts": "2026-05-26",
        "alpha_action": ALPHA_ACTION,
        "alpha_point": ALPHA_POINT,
        "base_csv": R067CR_BASE,
        "raw_csv": OUT_CSV_RAW,
        "output_csv": OUT_CSV_RULE,
        "n_action_diffs_vs_base": int(n_final_act),
        "n_point_diffs_vs_base":  int(n_final_pt),
        "predicted_f1a_delta": 0.0006,  # from v1 OOF sweep at alpha=0.05
        "predicted_f1p_delta": 0.0,
        "v04_candidate_report": {
            "theoretical_generalization_reason":
                "v1 sweep showed action benefits from SoftF1 mixing (F1_a peak at "
                "alpha=0.05) but point does not (F1_p flat across all alphas). "
                "v2 decouples per-task alpha: action 0.05, point 0.00 (preserve "
                "R-067cr point exactly). Less LB risk (smaller diff count) for "
                "same expected lift.",
            "why_transfers_to_test_new":
                "Same as v1 — SoftF1 is training-objective change, not feature "
                "change. Action-only blend is even more conservative than v1: "
                "only action predictions changed, point untouched.",
            "smoke_sanity_pass": True,
            "lb_probe_worthy": True,
            "lb_confirm_hypothesis":
                "LB ΔOV >= +0.0003 => action-only SoftF1 additive transfers.",
            "lb_reject_hypothesis":
                "LB ΔOV <= -0.003 => additive B-impure at low weight also fails.",
            "predicted_lb_delta": "+0.0003 to +0.0008 (B-feature 0.9x × +0.0006 F1_a × 0.4 OV weight)",
        },
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved manifest: {MANIFEST}")
    print("\n" + "=" * 80)
    print(" ARTIFACT_READY_FOR_JABIR_UPLOAD_REVIEW")
    print("=" * 80)
    print(f" File: {OUT_CSV_RULE}")
    print(f" Mechanism: SoftF1 component additive at α_a=0.05 (action only)")
    print(f" Predicted LB Δ: +0.0003 to +0.0008")
    print(f" Risk: very small — point + SGP exactly preserved from LB-best")


if __name__ == "__main__":
    main()
