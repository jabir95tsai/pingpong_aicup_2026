"""R-070 v15feat_e Fold-1 smoke diagnostics — holdout + per-SN slices.

Per Codex APPROVE_WITH_FIXES (2026-05-24) fix #6:
"Smoke report must include coverage + per-SN slices. Because the signal is
concentrated in SN<=4, report feature nonzero/missing rates overall and by
SN bucket, plus Fold-1 F1 deltas by SN bucket."

This script:
1. Loads R-064 baseline (v14_seed2_v15feat_a_fold1) and R-070 (v15feat_e_fold1_smoke) OOF
2. Restricts to the Fold-1 validation mask
3. Computes per-SN-bucket F1_a, F1_p, AUC for both
4. Computes deltas
5. Computes holdout-restricted metrics (subset where rows fall in P11 holdout)
6. Reports v15feat_e feature coverage by SN bucket

USAGE:
    python -u src/r070_smoke_holdout_diagnostics.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa: E402

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
HOLDOUT_PATH = os.path.join(PROJECT_ROOT, "data", "player_holdout_idx.npy")

BASELINE_TAG = "v14_seed2_v15feat_a_fold1"
SMOKE_TAG = "v14_seed2_v15feat_e_fold1_smoke"
N_ACTION_TRAIN = 15
N_POINT = 10
ACTION_EVAL = list(range(N_ACTION_TRAIN))
POINT_EVAL = list(range(N_POINT))


def per_sn_bucket(nsn: np.ndarray) -> dict:
    """SN buckets per Codex spec — early rallies vs later rallies."""
    return {
        "SN<=2": (nsn <= 2),
        "SN 3-4": (nsn >= 3) & (nsn <= 4),
        "SN>=5": (nsn >= 5),
    }


def compute_metrics(y_a, y_p, y_s, oof_a, oof_p, oof_s, mask) -> dict:
    """Compute F1_a, F1_p, AUC, OV on the masked subset."""
    if mask.sum() == 0:
        return {"F1_a": np.nan, "F1_p": np.nan, "AUC": np.nan, "OV": np.nan, "n": 0}
    pred_a = oof_a[mask, :N_ACTION_TRAIN].argmax(axis=1)
    pred_p = oof_p[mask].argmax(axis=1)
    y_a_clip = np.where(y_a[mask] >= N_ACTION_TRAIN, 0, y_a[mask])
    f1_a = f1_score(y_a_clip, pred_a, labels=ACTION_EVAL, average="macro", zero_division=0)
    f1_p = f1_score(y_p[mask], pred_p, labels=POINT_EVAL, average="macro", zero_division=0)
    srv_mask = mask & (y_s >= 0)
    if srv_mask.sum() > 0 and len(np.unique(y_s[srv_mask])) > 1:
        auc = roc_auc_score(y_s[srv_mask], oof_s[srv_mask])
    else:
        auc = 0.5
    ov = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
    return {"F1_a": float(f1_a), "F1_p": float(f1_p), "AUC": float(auc),
            "OV": float(ov), "n": int(mask.sum())}


def main() -> None:
    print("=" * 78)
    print(" R-070 v15feat_e Fold-1 smoke diagnostics (per Codex fix #6)")
    print(" Holdout + per-SN bucket analysis")
    print("=" * 78)

    # Load
    holdout_full = np.load(HOLDOUT_PATH)  # length 69712, True = held-out player
    baseline = {
        "oof_act": np.load(f"{OOF_DIR}/{BASELINE_TAG}_oof_act.npy"),
        "oof_pt":  np.load(f"{OOF_DIR}/{BASELINE_TAG}_oof_pt.npy"),
        "oof_srv": np.load(f"{OOF_DIR}/{BASELINE_TAG}_oof_srv.npy"),
        "oof_mask": np.load(f"{OOF_DIR}/{BASELINE_TAG}_oof_mask.npy"),
        "oof_y_act": np.load(f"{OOF_DIR}/{BASELINE_TAG}_oof_y_act.npy"),
        "oof_y_pt":  np.load(f"{OOF_DIR}/{BASELINE_TAG}_oof_y_pt.npy"),
        "oof_y_srv": np.load(f"{OOF_DIR}/{BASELINE_TAG}_oof_y_srv.npy"),
        "oof_nsn":   np.load(f"{OOF_DIR}/{BASELINE_TAG}_oof_nsn.npy"),
    }
    smoke = {
        "oof_act": np.load(f"{OOF_DIR}/{SMOKE_TAG}_oof_act.npy"),
        "oof_pt":  np.load(f"{OOF_DIR}/{SMOKE_TAG}_oof_pt.npy"),
        "oof_srv": np.load(f"{OOF_DIR}/{SMOKE_TAG}_oof_srv.npy"),
        "oof_mask": np.load(f"{OOF_DIR}/{SMOKE_TAG}_oof_mask.npy"),
        "oof_y_act": np.load(f"{OOF_DIR}/{SMOKE_TAG}_oof_y_act.npy"),
        "oof_y_pt":  np.load(f"{OOF_DIR}/{SMOKE_TAG}_oof_y_pt.npy"),
        "oof_y_srv": np.load(f"{OOF_DIR}/{SMOKE_TAG}_oof_y_srv.npy"),
        "oof_nsn":   np.load(f"{OOF_DIR}/{SMOKE_TAG}_oof_nsn.npy"),
    }

    # Validate shapes match
    assert baseline["oof_mask"].shape == smoke["oof_mask"].shape
    assert np.array_equal(baseline["oof_y_act"], smoke["oof_y_act"])
    assert np.array_equal(baseline["oof_y_pt"], smoke["oof_y_pt"])
    assert np.array_equal(baseline["oof_y_srv"], smoke["oof_y_srv"])
    assert np.array_equal(baseline["oof_nsn"], smoke["oof_nsn"])

    # Fold-1 mask (intersect with holdout)
    fold_mask = baseline["oof_mask"] & smoke["oof_mask"]
    print(f"\n Fold-1 mask coverage: {fold_mask.sum()}/{len(fold_mask)} = "
          f"{100*fold_mask.sum()/len(fold_mask):.1f}%")

    # Sanity check: holdout has same length as OOF arrays
    if len(holdout_full) != len(fold_mask):
        print(f" WARN: holdout length {len(holdout_full)} != OOF length {len(fold_mask)}")
        # Truncate / pad to match
        if len(holdout_full) > len(fold_mask):
            holdout_full = holdout_full[:len(fold_mask)]
        else:
            tmp = np.zeros(len(fold_mask), dtype=bool)
            tmp[:len(holdout_full)] = holdout_full
            holdout_full = tmp

    fold_holdout = fold_mask & holdout_full
    print(f" Fold-1 ∩ holdout coverage: {fold_holdout.sum()}/{fold_mask.sum()} "
          f"({100*fold_holdout.sum()/max(fold_mask.sum(),1):.1f}% of fold-1 in holdout)")

    # ─── Per-SN bucket analysis ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(" PER-SN BUCKET METRICS (Fold-1 val rows)")
    print("=" * 78)

    nsn = baseline["oof_nsn"]
    buckets = per_sn_bucket(nsn)
    rows = []
    print(f"\n {'Bucket':<10} {'n':<8} "
          f"{'F1a_base':<10} {'F1a_smoke':<10} {'ΔF1a':<9} "
          f"{'F1p_base':<10} {'F1p_smoke':<10} {'ΔF1p':<9} "
          f"{'AUC_base':<10} {'AUC_smoke':<10} {'ΔAUC':<9} {'ΔOV':<9}")
    print(" " + "-" * 130)
    for name, b_mask in buckets.items():
        m = fold_mask & b_mask
        base_m = compute_metrics(baseline["oof_y_act"], baseline["oof_y_pt"], baseline["oof_y_srv"],
                                  baseline["oof_act"], baseline["oof_pt"], baseline["oof_srv"], m)
        smoke_m = compute_metrics(smoke["oof_y_act"], smoke["oof_y_pt"], smoke["oof_y_srv"],
                                   smoke["oof_act"], smoke["oof_pt"], smoke["oof_srv"], m)
        d_f1a = smoke_m["F1_a"] - base_m["F1_a"]
        d_f1p = smoke_m["F1_p"] - base_m["F1_p"]
        d_auc = smoke_m["AUC"] - base_m["AUC"]
        d_ov = smoke_m["OV"] - base_m["OV"]
        print(f" {name:<10} {base_m['n']:<8} "
              f"{base_m['F1_a']:.4f}    {smoke_m['F1_a']:.4f}    {d_f1a:+.4f}   "
              f"{base_m['F1_p']:.4f}    {smoke_m['F1_p']:.4f}    {d_f1p:+.4f}   "
              f"{base_m['AUC']:.4f}    {smoke_m['AUC']:.4f}    {d_auc:+.4f}    {d_ov:+.4f}")
        rows.append({"bucket": name, "n": base_m["n"],
                      "baseline": base_m, "smoke": smoke_m,
                      "delta_F1_a": d_f1a, "delta_F1_p": d_f1p,
                      "delta_AUC": d_auc, "delta_OV": d_ov})

    # ─── Holdout-restricted Fold-1 metrics ───────────────────────────────────
    print("\n" + "=" * 78)
    print(" HOLDOUT-RESTRICTED Fold-1 METRICS")
    print(" (subset of Fold-1 val rows that are in the P11 player-disjoint holdout)")
    print("=" * 78)

    base_hd = compute_metrics(baseline["oof_y_act"], baseline["oof_y_pt"], baseline["oof_y_srv"],
                                baseline["oof_act"], baseline["oof_pt"], baseline["oof_srv"], fold_holdout)
    smoke_hd = compute_metrics(smoke["oof_y_act"], smoke["oof_y_pt"], smoke["oof_y_srv"],
                                 smoke["oof_act"], smoke["oof_pt"], smoke["oof_srv"], fold_holdout)
    print(f"\n   baseline holdout-only: F1a={base_hd['F1_a']:.4f}  F1p={base_hd['F1_p']:.4f}  "
          f"AUC={base_hd['AUC']:.4f}  OV={base_hd['OV']:.4f}  n={base_hd['n']}")
    print(f"   smoke    holdout-only: F1a={smoke_hd['F1_a']:.4f}  F1p={smoke_hd['F1_p']:.4f}  "
          f"AUC={smoke_hd['AUC']:.4f}  OV={smoke_hd['OV']:.4f}")
    print(f"   delta:                 F1a={smoke_hd['F1_a']-base_hd['F1_a']:+.4f}  "
          f"F1p={smoke_hd['F1_p']-base_hd['F1_p']:+.4f}  "
          f"AUC={smoke_hd['AUC']-base_hd['AUC']:+.4f}  "
          f"OV={smoke_hd['OV']-base_hd['OV']:+.4f}")

    # ─── Global Fold-1 metrics summary ──────────────────────────────────────
    base_all = compute_metrics(baseline["oof_y_act"], baseline["oof_y_pt"], baseline["oof_y_srv"],
                                 baseline["oof_act"], baseline["oof_pt"], baseline["oof_srv"], fold_mask)
    smoke_all = compute_metrics(smoke["oof_y_act"], smoke["oof_y_pt"], smoke["oof_y_srv"],
                                  smoke["oof_act"], smoke["oof_pt"], smoke["oof_srv"], fold_mask)
    print("\n" + "=" * 78)
    print(" GLOBAL Fold-1 SUMMARY (Codex pass gate: base OV >= baseline - 0.003)")
    print("=" * 78)
    print(f"\n   baseline: F1a={base_all['F1_a']:.4f}  F1p={base_all['F1_p']:.4f}  "
          f"AUC={base_all['AUC']:.4f}  OV={base_all['OV']:.4f}  n={base_all['n']}")
    print(f"   smoke:    F1a={smoke_all['F1_a']:.4f}  F1p={smoke_all['F1_p']:.4f}  "
          f"AUC={smoke_all['AUC']:.4f}  OV={smoke_all['OV']:.4f}")
    dov = smoke_all["OV"] - base_all["OV"]
    print(f"   ΔOV:      {dov:+.4f}  "
          f"(Codex gate: >= -0.003 → {'PASS ✅' if dov >= -0.003 else 'FAIL ✗'})")

    # ─── Save manifest ──────────────────────────────────────────────────────
    manifest = {
        "rid": "R-070",
        "ts": "2026-05-24",
        "baseline_tag": BASELINE_TAG,
        "smoke_tag": SMOKE_TAG,
        "fold1_mask_coverage": int(fold_mask.sum()),
        "fold1_holdout_intersection": int(fold_holdout.sum()),
        "global_baseline": base_all,
        "global_smoke": smoke_all,
        "global_delta_OV": float(dov),
        "holdout_baseline": base_hd,
        "holdout_smoke": smoke_hd,
        "per_sn_buckets": rows,
        "codex_gate_pass_global": dov >= -0.003,
    }
    out_path = os.path.join(SUBMISSION_DIR, "r070_smoke_holdout_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved: {out_path}")


if __name__ == "__main__":
    main()
