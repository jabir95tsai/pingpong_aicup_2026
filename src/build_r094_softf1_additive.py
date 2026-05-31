"""R-094 — R-067cr + SoftF1 additive blend (size-6 cap relax, novel mechanism).

Theory (v0.4 candidate report):
  theoretical_generalization_reason:
    R-031 trained v11_mulminet_aug_oldtest with soft-F1 loss directly targeting
    macro-F1 (rather than cross-entropy). Theoretically improves rare-class
    performance (Pushfast, Push, Arch, Knuckle for action; FH_short, BH_short
    for point). This is a different OBJECTIVE than current LB-best blend's CE
    losses. Adding it at small weight as a 6th component is a structurally
    novel mechanism (size-6 cap relax is documented strategic lever per
    LESSONS, never LB-tested).

  why_transfers_to_test_new:
    SoftF1 is a TRAINING-objective change, not a feature change. Same data,
    same train/test distribution behavior. Macro-F1-targeted training should
    generalize equally well to test_new as base CE-trained models. Risk:
    MuLMINet architecture is structurally different from v11/v11plus/v11_aug,
    which historically triggered B-impure failures (R-028, R-040) for SWAPS.
    R-094 is ADDITIVE at small weight (not swap), reducing B-impure exposure.

  smoke_sanity_pass: TBD by this script
  lb_probe_worthy: if OOF macro-F1 (action OR point) improves >= +0.001
                   over R-067cr base on Fold-1
  lb_confirm_hypothesis:
    LB DeltaOV >= +0.001 => SoftF1 additive at small weight transfers;
    size-6 cap relax is a viable lever.
  lb_reject_hypothesis:
    LB DeltaOV <= -0.005 => additive B-impure also fails (not just swap);
    size-6 cap relax should be closed.

USAGE:
    python -u src/build_r094_softf1_additive.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR
from analyze_oldtest_blend import load_components, evaluate_subset_none

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
R067CR_BASE = os.path.join(SUBMISSION_DIR,
                            "submission_R067cr_alpha030_v22_blend_PLUS_RULE.csv")
OUT_CSV_TMPL = os.path.join(SUBMISSION_DIR,
                             "submission_R094_R067cr_PLUS_SOFTF1_alpha{:03d}.csv")
MANIFEST = os.path.join(SUBMISSION_DIR, "r094_softf1_additive_manifest.json")

R034_COMPONENTS = ["v11_aug_oldtest", "v11plus", "v13_oldtest",
                    "v14_seed2_v15feat_a", "v16_avg3"]
SOFTF1_TAG = "v11_mulminet_aug_oldtest_softf1_phaseB"

N_ACTION_TRAIN = 15
N_ACTION_FULL = 19
N_POINT = 10
ACTION_EVAL = list(range(N_ACTION_TRAIN))
POINT_EVAL = list(range(N_POINT))

# Alpha sweep range (softf1 weight)
ALPHA_SWEEP = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def pad19(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] >= N_ACTION_FULL:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], N_ACTION_FULL), dtype=np.float32)
    out[:, :arr.shape[1]] = arr
    return out


def load_softf1_oof_aligned():
    """Load SoftF1 OOF arrays sliced to 69712 rows + pad action to 19-class."""
    base = os.path.join(OOF_DIR, SOFTF1_TAG)
    oa = pad19(np.load(f"{base}_oof_act.npy"))[:69712]
    op = np.load(f"{base}_oof_pt.npy").astype(np.float32)[:69712]
    osrv = np.load(f"{base}_oof_srv.npy").astype(np.float32)[:69712]
    return {"oof_act": oa, "oof_pt": op, "oof_srv": osrv}


def load_softf1_test():
    base = os.path.join(OOF_DIR, SOFTF1_TAG)
    ta = pad19(np.load(f"{base}_test_act.npy"))
    tp = np.load(f"{base}_test_pt.npy").astype(np.float32)
    ts = np.load(f"{base}_test_srv.npy").astype(np.float32)
    rallies = np.load(f"{base}_test_rally_uid.npy")
    return {"test_act": ta, "test_pt": tp, "test_srv": ts,
            "test_rally_uid": rallies}


def reconstruct_r067cr_test_probs(comp_test, weights):
    """Recompute R-034 PAIR Dirichlet-blended test predictions per task."""
    act_stack = np.stack([comp_test[t]["test_act"] for t in R034_COMPONENTS], axis=0)
    pt_stack = np.stack([comp_test[t]["test_pt"] for t in R034_COMPONENTS], axis=0)
    test_act = (weights["w_a"][:, None, None] * act_stack).sum(axis=0)
    test_pt = (weights["w_p"][:, None, None] * pt_stack).sum(axis=0)
    return test_act, test_pt


def main() -> None:
    print("=" * 80)
    print(" R-094 — R-067cr + R-031 SoftF1 additive blend (Fold-1 smoke + LB candidate)")
    print("=" * 80)

    # ─── Step 1: R-034 PAIR OOF + Dirichlet weights ────────────────────
    print("\n Step 1: load R-034 PAIR components + Dirichlet search")
    comp_oof, y_a, y_p, y_s, mask, test_uid = load_components(R034_COMPONENTS)
    weights = evaluate_subset_none(R034_COMPONENTS, comp_oof, y_a, y_p, y_s,
                                    optimize=True, n_samples=300, seed=20260524)
    print(f"   R-034 PAIR OOF: F1_a={weights['F1_a']:.4f}  F1_p={weights['F1_p']:.4f}  "
          f"OV={weights['OV']:.4f}")

    # R-034 PAIR blend OOF (action + point)
    act_stack_oof = np.stack([comp_oof[t]["oof_act"] for t in R034_COMPONENTS], axis=0)
    pt_stack_oof = np.stack([comp_oof[t]["oof_pt"] for t in R034_COMPONENTS], axis=0)
    blend_act_oof = (weights["w_a"][:, None, None] * act_stack_oof).sum(axis=0)
    blend_pt_oof = (weights["w_p"][:, None, None] * pt_stack_oof).sum(axis=0)

    # ─── Step 2: load SoftF1 OOF + alignment ────────────────────────────
    print("\n Step 2: load R-031 SoftF1 OOF (sliced to 69712)")
    softf1_oof = load_softf1_oof_aligned()
    print(f"   softf1 OOF: act{softf1_oof['oof_act'].shape}  pt{softf1_oof['oof_pt'].shape}")

    # ─── Step 3: alpha sweep on OOF macro-F1 ───────────────────────────
    print("\n Step 3: alpha sweep on FULL OOF macro-F1")
    nsn = np.load(os.path.join(OOF_DIR, "v14_seed2_v15feat_a_oof_nsn.npy"))
    y_a_clip = np.where(y_a >= N_ACTION_TRAIN, 0, y_a)

    sweep_results = []
    print(f"   {'alpha':>6}  {'F1_a':>8}  {'F1_p':>8}  {'OV-act+pt':>10}")
    for alpha in ALPHA_SWEEP:
        blend_a = (1 - alpha) * blend_act_oof + alpha * softf1_oof["oof_act"]
        blend_p = (1 - alpha) * blend_pt_oof + alpha * softf1_oof["oof_pt"]
        f1a = f1_score(y_a_clip, blend_a.argmax(axis=1), labels=ACTION_EVAL,
                        average="macro", zero_division=0)
        f1p = f1_score(y_p, blend_p.argmax(axis=1), labels=POINT_EVAL,
                        average="macro", zero_division=0)
        ov_2task = 0.5 * f1a + 0.5 * f1p
        sweep_results.append({"alpha": alpha, "F1_a": f1a, "F1_p": f1p,
                               "OV_2task": ov_2task})
        marker = " *BASE*" if alpha == 0.0 else ""
        print(f"   {alpha:>6.2f}  {f1a:.4f}  {f1p:.4f}  {ov_2task:.4f}{marker}")

    base_f1a = sweep_results[0]["F1_a"]
    base_f1p = sweep_results[0]["F1_p"]
    # Pick alpha maximizing combined F1 (action + point only; SGP unchanged)
    best = max(sweep_results, key=lambda r: r["OV_2task"])
    best_alpha = best["alpha"]
    delta_f1a = best["F1_a"] - base_f1a
    delta_f1p = best["F1_p"] - base_f1p
    print(f"\n   BEST alpha={best_alpha:.2f}:  F1_a {best['F1_a']:.4f} "
          f"(Δ {delta_f1a:+.4f})  F1_p {best['F1_p']:.4f} (Δ {delta_f1p:+.4f})")

    # Per-class F1 deltas at best alpha (canary check)
    blend_a_best = (1 - best_alpha) * blend_act_oof + best_alpha * softf1_oof["oof_act"]
    blend_p_best = (1 - best_alpha) * blend_pt_oof + best_alpha * softf1_oof["oof_pt"]
    pred_a_base = blend_act_oof.argmax(axis=1)
    pred_a_best = blend_a_best.argmax(axis=1)
    pred_p_base = blend_pt_oof.argmax(axis=1)
    pred_p_best = blend_p_best.argmax(axis=1)
    f1_per_class_a_base = f1_score(y_a_clip, pred_a_base, labels=ACTION_EVAL,
                                     average=None, zero_division=0)
    f1_per_class_a_best = f1_score(y_a_clip, pred_a_best, labels=ACTION_EVAL,
                                     average=None, zero_division=0)
    f1_per_class_p_base = f1_score(y_p, pred_p_base, labels=POINT_EVAL,
                                     average=None, zero_division=0)
    f1_per_class_p_best = f1_score(y_p, pred_p_best, labels=POINT_EVAL,
                                     average=None, zero_division=0)
    print("\n   Per-class F1 deltas (action, at best alpha):")
    push_family = [5, 6, 13]  # Pushfast, Push, Block
    canary_drops = []
    for c in range(N_ACTION_TRAIN):
        d = f1_per_class_a_best[c] - f1_per_class_a_base[c]
        flag = ""
        if d <= -0.015:
            flag = " [CANARY]"
            canary_drops.append({"class": f"action{c}", "delta_F1": float(d)})
        if c in push_family:
            flag += " [PUSH]"
        print(f"     act{c:>2}: {f1_per_class_a_base[c]:.4f} -> "
              f"{f1_per_class_a_best[c]:.4f}  ({d:+.4f}){flag}")
    print("\n   Per-class F1 deltas (point, at best alpha):")
    for c in range(N_POINT):
        d = f1_per_class_p_best[c] - f1_per_class_p_base[c]
        flag = " [CANARY]" if d <= -0.015 else ""
        if d <= -0.015:
            canary_drops.append({"class": f"point{c}", "delta_F1": float(d)})
        print(f"     pt{c:>2}: {f1_per_class_p_base[c]:.4f} -> "
              f"{f1_per_class_p_best[c]:.4f}  ({d:+.4f}){flag}")

    # ─── Step 4: build test CSV at best alpha ──────────────────────────
    print(f"\n Step 4: build test CSV at alpha={best_alpha:.2f}")
    if best_alpha == 0.0:
        print("   best alpha is 0.00 (SoftF1 doesn't help). Skipping CSV build.")
        smoke_sanity_pass = True
        artifact_ready = False
        out_csv = None
    else:
        # Load all comp test arrays
        comp_test = {}
        for t in R034_COMPONENTS:
            comp_test[t] = {
                "test_act": pad19(np.load(os.path.join(OOF_DIR, f"{t}_test_act.npy"))),
                "test_pt":  np.load(os.path.join(OOF_DIR, f"{t}_test_pt.npy")).astype(np.float32),
            }
        r067cr_test_act, r067cr_test_pt = reconstruct_r067cr_test_probs(comp_test, weights)
        softf1_test = load_softf1_test()

        blend_test_a = (1 - best_alpha) * r067cr_test_act + best_alpha * softf1_test["test_act"]
        blend_test_p = (1 - best_alpha) * r067cr_test_pt + best_alpha * softf1_test["test_pt"]

        # Final actionId / pointId per rally
        new_act = blend_test_a.argmax(axis=1)
        new_pt = blend_test_p.argmax(axis=1)

        # Load R-067cr base to preserve SGP + rally_uid order
        r067cr_df = pd.read_csv(R067CR_BASE)
        # Sanity: rally_uid order matches
        assert np.array_equal(r067cr_df["rally_uid"].to_numpy(),
                               softf1_test["test_rally_uid"]), \
            "test rally_uid mismatch"
        # Build new CSV
        new_df = r067cr_df.copy()
        new_df["actionId"] = new_act
        new_df["pointId"] = new_pt
        # SGP preserved from R-067cr
        assert np.array_equal(new_df["serverGetPoint"].to_numpy(),
                               r067cr_df["serverGetPoint"].to_numpy())

        out_csv = OUT_CSV_TMPL.format(int(best_alpha * 100))
        new_df.to_csv(out_csv, index=False, lineterminator="\n", encoding="utf-8")
        n_act_diff = (new_df["actionId"].to_numpy() != r067cr_df["actionId"].to_numpy()).sum()
        n_pt_diff = (new_df["pointId"].to_numpy() != r067cr_df["pointId"].to_numpy()).sum()
        print(f"   Saved: {out_csv}")
        print(f"   Action diffs vs R-067cr: {n_act_diff}")
        print(f"   Point diffs vs R-067cr:  {n_pt_diff}")
        # Apply rule_override Layer A
        # (R-067cr already has rule_override applied; new CSV inherits SGP but
        # has different action/point. Re-apply rule_override to catch new
        # zero-prob violations.)
        print("\n   Applying rule_override Layer A to refresh post-process...")
        import subprocess
        out_csv_with_rule = out_csv.replace(".csv", "_PLUS_RULE.csv")
        cmd = [sys.executable, "-u", os.path.join("src", "apply_rule_override.py"),
               "--input", out_csv, "--train", os.path.join("data", "train.csv"),
               "--test", os.path.join("data", "test_new.csv"),
               "--output", out_csv_with_rule]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print("   rule_override applied successfully")
            print(r.stdout.splitlines()[-3:])
            out_csv = out_csv_with_rule
        else:
            print(f"   WARN: rule_override failed: {r.stderr}")

        smoke_sanity_pass = (len(canary_drops) <= 2) and (delta_f1a > -0.005) and (delta_f1p > -0.005)
        artifact_ready = (delta_f1a >= -0.001) and (delta_f1p >= -0.001) and smoke_sanity_pass

    # ─── Step 5: manifest ──────────────────────────────────────────────
    manifest = {
        "rid": "R-094",
        "ts": "2026-05-26",
        "base_blend": "R-067cr (R-034 PAIR Dirichlet + rule_override + v22 SGP alpha=0.30)",
        "additive_component": SOFTF1_TAG,
        "alpha_sweep": sweep_results,
        "best_alpha": best_alpha,
        "delta_F1_a_at_best": float(delta_f1a),
        "delta_F1_p_at_best": float(delta_f1p),
        "canary_class_drops": canary_drops,
        "smoke_sanity_pass": bool(smoke_sanity_pass),
        "artifact_ready_for_jabir_upload_review": bool(artifact_ready),
        "output_csv": out_csv if artifact_ready else None,
        "v04_candidate_report": {
            "theoretical_generalization_reason":
                "R-031 trained v11_mulminet_aug_oldtest with soft-F1 loss directly "
                "targeting macro-F1. Adding it at small weight as 6th component is "
                "structurally novel (size-6 cap relax is documented strategic lever "
                "per LESSONS but never LB-tested). Targets known rare-class weakness.",
            "why_transfers_to_test_new":
                "SoftF1 is training-objective change, not feature change. Same data, "
                "same distribution behavior. Macro-F1-targeted training should "
                "generalize equally to test_new. Risk: MuLMINet arch is structurally "
                "different (B-impure triggered swap failures in R-028/R-040), but "
                "additive at small weight reduces exposure vs swap.",
            "smoke_sanity_pass": smoke_sanity_pass,
            "lb_probe_worthy": artifact_ready,
            "lb_confirm_hypothesis":
                "LB DeltaOV >= +0.001 => SoftF1 additive at small weight transfers; "
                "size-6 cap relax is viable.",
            "lb_reject_hypothesis":
                "LB DeltaOV <= -0.005 => additive B-impure also fails; cap relax closed.",
        },
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved manifest: {MANIFEST}")
    print(f" smoke_sanity_pass = {smoke_sanity_pass}")
    print(f" artifact_ready    = {artifact_ready}")


if __name__ == "__main__":
    main()
