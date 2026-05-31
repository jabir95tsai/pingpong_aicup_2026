"""R-070 v15feat_e_nomismatch Fold-1 ablation smoke diagnostics.

Per Codex 2026-05-24 verdict on the 7-feature smoke:
    "Implement v15feat_e_nomismatch by dropping stroke_position_mismatch_proxy
     and mismatch_AND_far_gap. Run tests + Fold-1 smoke only. Report global,
     holdout, and SN bucket deltas vs exact v14_seed2_v15feat_a_fold1
     baseline. No Group C extension and no full 5-fold until Codex reviews
     the ablation artifact."

Extends r070_smoke_holdout_diagnostics.py with per-class F1 canary drops so
the candidate_goal v0.2 scoring can pick up canary regressions.

USAGE:
    python -u src/r070_nomismatch_smoke_diagnostics.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa: E402

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
HOLDOUT_PATH = os.path.join(PROJECT_ROOT, "data", "player_holdout_idx.npy")

BASELINE_TAG = "v14_seed2_v15feat_a_fold1"
SMOKE_TAG    = "v14_seed2_v15feat_e_nomismatch_fold1_smoke"

N_ACTION_TRAIN = 15
N_POINT = 10
ACTION_EVAL = list(range(N_ACTION_TRAIN))
POINT_EVAL  = list(range(N_POINT))

# Canary threshold matches src/candidate_goal.py CANARY_CLASS_DROP_THRESHOLD
CANARY_F1_DROP_THRESHOLD = -0.015


def per_sn_buckets(nsn: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "SN<=2":  (nsn <= 2),
        "SN 3-4": (nsn >= 3) & (nsn <= 4),
        "SN>=5":  (nsn >= 5),
    }


def compute_metrics(y_a, y_p, y_s, oof_a, oof_p, oof_s, mask) -> dict:
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


def per_class_f1(y_true, oof_logits, n_classes, mask, name_prefix: str) -> List[dict]:
    """Per-class F1 on the masked subset for each of n_classes."""
    if mask.sum() == 0:
        return []
    pred = oof_logits[mask, :n_classes].argmax(axis=1)
    y = y_true[mask]
    if name_prefix == "action":
        # Clip serve actions to 0 (matches global eval)
        y = np.where(y >= n_classes, 0, y)
    f1_per_class = f1_score(y, pred,
                            labels=list(range(n_classes)),
                            average=None,
                            zero_division=0)
    rows = []
    for cls_id, f1 in enumerate(f1_per_class):
        n_in_cls = int((y == cls_id).sum())
        rows.append({"class": f"{name_prefix}{cls_id}", "n": n_in_cls,
                     "F1": float(f1)})
    return rows


def canary_drops(per_cls_baseline: List[dict], per_cls_smoke: List[dict]
                 ) -> List[dict]:
    """Find classes with F1 drop <= CANARY_F1_DROP_THRESHOLD."""
    out = []
    by_cls_base = {row["class"]: row for row in per_cls_baseline}
    for row_s in per_cls_smoke:
        cls = row_s["class"]
        row_b = by_cls_base.get(cls)
        if row_b is None:
            continue
        delta = row_s["F1"] - row_b["F1"]
        if delta <= CANARY_F1_DROP_THRESHOLD:
            out.append({"class": cls, "n": row_b["n"],
                        "baseline_F1": row_b["F1"], "smoke_F1": row_s["F1"],
                        "delta_F1": float(delta)})
    return out


def load(tag: str) -> dict:
    return {
        "oof_act":  np.load(f"{OOF_DIR}/{tag}_oof_act.npy"),
        "oof_pt":   np.load(f"{OOF_DIR}/{tag}_oof_pt.npy"),
        "oof_srv":  np.load(f"{OOF_DIR}/{tag}_oof_srv.npy"),
        "oof_mask": np.load(f"{OOF_DIR}/{tag}_oof_mask.npy"),
        "oof_y_act": np.load(f"{OOF_DIR}/{tag}_oof_y_act.npy"),
        "oof_y_pt":  np.load(f"{OOF_DIR}/{tag}_oof_y_pt.npy"),
        "oof_y_srv": np.load(f"{OOF_DIR}/{tag}_oof_y_srv.npy"),
        "oof_nsn":   np.load(f"{OOF_DIR}/{tag}_oof_nsn.npy"),
    }


def main() -> None:
    print("=" * 80)
    print(" R-070 v15feat_e_nomismatch (5-feature) ablation smoke diagnostics")
    print(f" Baseline: {BASELINE_TAG}")
    print(f" Smoke:    {SMOKE_TAG}")
    print("=" * 80)

    baseline = load(BASELINE_TAG)
    smoke    = load(SMOKE_TAG)
    holdout_full = np.load(HOLDOUT_PATH)

    assert baseline["oof_mask"].shape == smoke["oof_mask"].shape
    assert np.array_equal(baseline["oof_y_act"], smoke["oof_y_act"])
    assert np.array_equal(baseline["oof_y_pt"],  smoke["oof_y_pt"])
    assert np.array_equal(baseline["oof_y_srv"], smoke["oof_y_srv"])
    assert np.array_equal(baseline["oof_nsn"],   smoke["oof_nsn"])

    fold_mask = baseline["oof_mask"] & smoke["oof_mask"]
    print(f"\n Fold-1 val coverage: {fold_mask.sum()}/{len(fold_mask)} = "
          f"{100*fold_mask.sum()/len(fold_mask):.1f}%")

    if len(holdout_full) != len(fold_mask):
        if len(holdout_full) > len(fold_mask):
            holdout_full = holdout_full[:len(fold_mask)]
        else:
            tmp = np.zeros(len(fold_mask), dtype=bool)
            tmp[:len(holdout_full)] = holdout_full
            holdout_full = tmp

    fold_holdout = fold_mask & holdout_full
    print(f" Fold-1 ∩ P11 holdout: {fold_holdout.sum()}/{fold_mask.sum()} "
          f"({100*fold_holdout.sum()/max(fold_mask.sum(),1):.1f}%)")

    # ─── Global Fold-1 ──────────────────────────────────────────────────
    base_all  = compute_metrics(baseline["oof_y_act"], baseline["oof_y_pt"], baseline["oof_y_srv"],
                                 baseline["oof_act"],  baseline["oof_pt"],  baseline["oof_srv"],
                                 fold_mask)
    smoke_all = compute_metrics(smoke["oof_y_act"], smoke["oof_y_pt"], smoke["oof_y_srv"],
                                 smoke["oof_act"],  smoke["oof_pt"],  smoke["oof_srv"],
                                 fold_mask)
    global_delta_OV = smoke_all["OV"] - base_all["OV"]

    print("\n" + "=" * 80)
    print(" GLOBAL Fold-1 (Codex gate: base ΔOV >= -0.003 to pass)")
    print("=" * 80)
    print(f"   baseline: F1a={base_all['F1_a']:.4f}  F1p={base_all['F1_p']:.4f}  "
          f"AUC={base_all['AUC']:.4f}  OV={base_all['OV']:.4f}  n={base_all['n']}")
    print(f"   smoke:    F1a={smoke_all['F1_a']:.4f}  F1p={smoke_all['F1_p']:.4f}  "
          f"AUC={smoke_all['AUC']:.4f}  OV={smoke_all['OV']:.4f}")
    print(f"   ΔOV:      {global_delta_OV:+.4f}  "
          f"(Codex gate: {'PASS' if global_delta_OV >= -0.003 else 'FAIL'})")
    print(f"   ΔF1_a:    {smoke_all['F1_a']-base_all['F1_a']:+.4f}")
    print(f"   ΔF1_p:    {smoke_all['F1_p']-base_all['F1_p']:+.4f}")
    print(f"   ΔAUC:     {smoke_all['AUC']-base_all['AUC']:+.4f}")

    # ─── Holdout-restricted ─────────────────────────────────────────────
    base_hd  = compute_metrics(baseline["oof_y_act"], baseline["oof_y_pt"], baseline["oof_y_srv"],
                                baseline["oof_act"],  baseline["oof_pt"],  baseline["oof_srv"],
                                fold_holdout)
    smoke_hd = compute_metrics(smoke["oof_y_act"], smoke["oof_y_pt"], smoke["oof_y_srv"],
                                smoke["oof_act"],  smoke["oof_pt"],  smoke["oof_srv"],
                                fold_holdout)
    holdout_delta_OV = smoke_hd["OV"] - base_hd["OV"]
    print("\n" + "=" * 80)
    print(" HOLDOUT-RESTRICTED Fold-1 (advisory only, not a hard gate)")
    print("=" * 80)
    print(f"   baseline holdout: F1a={base_hd['F1_a']:.4f}  F1p={base_hd['F1_p']:.4f}  "
          f"AUC={base_hd['AUC']:.4f}  OV={base_hd['OV']:.4f}  n={base_hd['n']}")
    print(f"   smoke    holdout: F1a={smoke_hd['F1_a']:.4f}  F1p={smoke_hd['F1_p']:.4f}  "
          f"AUC={smoke_hd['AUC']:.4f}  OV={smoke_hd['OV']:.4f}")
    print(f"   ΔOV (holdout):    {holdout_delta_OV:+.4f}")

    # ─── Per-SN bucket ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" PER-SN BUCKET (Fold-1 val rows, slice penalty if any ΔOV <= -0.005)")
    print("=" * 80)
    nsn = baseline["oof_nsn"]
    buckets = per_sn_buckets(nsn)
    sn_rows: List[dict] = []
    print(f"\n {'Bucket':<10} {'n':<6} {'baseOV':<9} {'smokeOV':<9} "
          f"{'ΔOV':<10} {'ΔF1a':<10} {'ΔF1p':<10} {'ΔAUC':<10}")
    print(" " + "-" * 74)
    for name, b_mask in buckets.items():
        m = fold_mask & b_mask
        b = compute_metrics(baseline["oof_y_act"], baseline["oof_y_pt"], baseline["oof_y_srv"],
                            baseline["oof_act"], baseline["oof_pt"], baseline["oof_srv"], m)
        s = compute_metrics(smoke["oof_y_act"], smoke["oof_y_pt"], smoke["oof_y_srv"],
                            smoke["oof_act"], smoke["oof_pt"], smoke["oof_srv"], m)
        d_f1a = s["F1_a"] - b["F1_a"]
        d_f1p = s["F1_p"] - b["F1_p"]
        d_auc = s["AUC"] - b["AUC"]
        d_ov  = s["OV"]   - b["OV"]
        print(f" {name:<10} {b['n']:<6} {b['OV']:.4f}    {s['OV']:.4f}    "
              f"{d_ov:+.4f}    {d_f1a:+.4f}    {d_f1p:+.4f}    {d_auc:+.4f}")
        sn_rows.append({"bucket": name, "n": b["n"],
                        "baseline": b, "smoke": s,
                        "delta_F1_a": d_f1a, "delta_F1_p": d_f1p,
                        "delta_AUC": d_auc, "delta_OV": d_ov})

    sn_regressions = [r for r in sn_rows if r["delta_OV"] <= -0.005]

    # ─── Per-class F1 canary scan ──────────────────────────────────────
    print("\n" + "=" * 80)
    print(f" CANARY CLASS DROPS (threshold ΔF1 <= {CANARY_F1_DROP_THRESHOLD})")
    print("=" * 80)
    cls_base_a = per_class_f1(baseline["oof_y_act"], baseline["oof_act"],
                              N_ACTION_TRAIN, fold_mask, "action")
    cls_smk_a  = per_class_f1(smoke["oof_y_act"], smoke["oof_act"],
                              N_ACTION_TRAIN, fold_mask, "action")
    cls_base_p = per_class_f1(baseline["oof_y_pt"], baseline["oof_pt"],
                              N_POINT, fold_mask, "point")
    cls_smk_p  = per_class_f1(smoke["oof_y_pt"], smoke["oof_pt"],
                              N_POINT, fold_mask, "point")
    canary_action = canary_drops(cls_base_a, cls_smk_a)
    canary_point  = canary_drops(cls_base_p, cls_smk_p)
    canary_all    = canary_action + canary_point

    if not canary_all:
        print("   OK: No canary class drops (no per-class delta-F1 <= -0.015)")
    else:
        print(f"   {len(canary_all)} canary class drop(s):")
        for c in canary_all:
            print(f"     {c['class']:<10} n={c['n']:<5} "
                  f"base={c['baseline_F1']:.4f}  smoke={c['smoke_F1']:.4f}  "
                  f"ΔF1={c['delta_F1']:+.4f}")

    # Full per-class table for reference
    print("\n   Action per-class F1 deltas (full table):")
    for b, s in zip(cls_base_a, cls_smk_a):
        d = s["F1"] - b["F1"]
        flag = " [CANARY]" if d <= CANARY_F1_DROP_THRESHOLD else ""
        print(f"     {b['class']:<10} n={b['n']:<5} base={b['F1']:.4f}  "
              f"smoke={s['F1']:.4f}  Δ={d:+.4f}{flag}")
    print("\n   Point per-class F1 deltas (full table):")
    for b, s in zip(cls_base_p, cls_smk_p):
        d = s["F1"] - b["F1"]
        flag = " [CANARY]" if d <= CANARY_F1_DROP_THRESHOLD else ""
        print(f"     {b['class']:<10} n={b['n']:<5} base={b['F1']:.4f}  "
              f"smoke={s['F1']:.4f}  Δ={d:+.4f}{flag}")

    # ─── Save manifest ─────────────────────────────────────────────────
    manifest = {
        "rid": "R-070-nomismatch",
        "ts": "2026-05-25",
        "baseline_tag": BASELINE_TAG,
        "smoke_tag": SMOKE_TAG,
        "fold1_mask_coverage": int(fold_mask.sum()),
        "fold1_holdout_intersection": int(fold_holdout.sum()),
        "global_baseline": base_all,
        "global_smoke": smoke_all,
        "global_delta_OV": float(global_delta_OV),
        "global_delta_F1_a": float(smoke_all["F1_a"] - base_all["F1_a"]),
        "global_delta_F1_p": float(smoke_all["F1_p"] - base_all["F1_p"]),
        "global_delta_AUC":  float(smoke_all["AUC"]  - base_all["AUC"]),
        "holdout_baseline": base_hd,
        "holdout_smoke":    smoke_hd,
        "holdout_delta_OV": float(holdout_delta_OV),
        "per_sn_buckets":     sn_rows,
        "sn_bucket_regressions": [
            {"bucket": r["bucket"], "delta_OV": r["delta_OV"]} for r in sn_regressions
        ],
        "per_class_action_baseline": cls_base_a,
        "per_class_action_smoke":    cls_smk_a,
        "per_class_point_baseline":  cls_base_p,
        "per_class_point_smoke":     cls_smk_p,
        "canary_class_drops":        canary_all,
        "codex_gate_pass_global":    bool(global_delta_OV >= -0.003),
    }
    out_path = os.path.join(SUBMISSION_DIR,
                            "r070_nomismatch_smoke_holdout_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved: {out_path}")
    print("\n SUMMARY")
    print(f"   global ΔOV:        {global_delta_OV:+.4f}")
    print(f"   holdout ΔOV:       {holdout_delta_OV:+.4f}")
    print(f"   SN regressions:    {len(sn_regressions)} bucket(s)")
    print(f"   canary class drops:{len(canary_all)} class(es)")
    print(f"   codex_global_gate: {'PASS' if global_delta_OV >= -0.003 else 'FAIL'}")


if __name__ == "__main__":
    main()
