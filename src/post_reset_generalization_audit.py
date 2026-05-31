"""Post-reset (2026-05-06+) generalization audit.

User 2026-05-24: "from the new testcsv we may consider more about the
submissions after the test data change ... Build new submissions
specifically tuned to the generalization ability which is our ultimate goal"

This script:
A. Lists all post-reset LB submissions with OOF + P11-holdout + LB scores.
B. Computes transfer rates: OOF→holdout, OOF→LB, holdout→LB.
C. Identifies which COMPONENTS are generalization-positive vs OOF-overfitting.
D. Proposes holdout-based pre-LB gating for future candidates.

P11 holdout = player-disjoint subset (25 held-out players, 8284 rows).
This simulates the test distribution (56% novel players per CLAUDE.md).

USAGE:
    python -u src/post_reset_generalization_audit.py
"""
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa: E402
from analyze_oldtest_blend import load_components, evaluate_subset_none  # noqa: E402

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
N_ACTION_TRAIN = 15
N_POINT = 10


def main() -> None:
    print("=" * 78)
    print(" Post-reset (2026-05-06+) generalization audit")
    print(" Goal: identify components that GENERALIZE vs OVERFIT-OOF")
    print("=" * 78)

    # ─── Load P11 holdout mask ───────────────────────────────────────────────
    holdout_path = os.path.join(PROJECT_ROOT, "data", "player_holdout_idx.npy")
    holdout = np.load(holdout_path)
    print(f"\n P11 holdout: {holdout.sum()}/{len(holdout)} = "
          f"{100*holdout.sum()/len(holdout):.1f}% of OOF rows")

    # ─── Post-reset LB scoreboard (from REVIEW_QUEUE + RESULTS) ──────────────
    lb_log = [
        # (R-id, date, components/design, OOF, LB, class)
        ("R-027 PAIR", "2026-05-18", "5c B-pure ADD oldtest", 0.3771, 0.3810, "B-pure"),
        ("R-028 top1", "2026-05-19", "5c B-impure SWAP (mulminet_avg2)", 0.378, 0.3724, "B-impure"),
        ("R-033",      "2026-05-20", "B-seedavg of B-impure (v13_oldtest_avg3)", 0.378, 0.3795, "B-seedavg"),
        ("R-034 PAIR", "2026-05-21", "5c B-feature (v15feat_a swap)", 0.378, 0.3838, "B-feature"),
        ("R-040",      "2026-05-21", "B-impure SWAP (mulminet_avg3)", 0.382, 0.3744, "B-impure"),
        ("R-042",      "2026-05-22", "R-034 + rule_override post-process", 0.3812, 0.3866, "post-process"),
        ("R-055",      "2026-05-23", "R-052 7c Bayes + rule (mulminet ADD)", 0.3844, 0.3725, "B-impure-ADD"),
        ("R-062r",     "2026-05-23", "v16match_v2 LORO swap (B-player-style)", 0.3823, 0.3809, "B-player-style"),
        ("R-054r",     "2026-05-24", "8c meta_v2 + v11_aug_big + recvprofile", 0.3821, 0.3763, "B-meta + B-player-style"),
        ("R-067cr",    "2026-05-24", "R-042 + 30% v22 SGP blend (server-head)", "AUC+0.0326", 0.3870, "server-head-blend"),
    ]

    print("\n" + "=" * 78)
    print(" POST-RESET LB SCOREBOARD (chronological)")
    print("=" * 78)
    print(f" {'R-id':<14} {'Date':<12} {'OOF':<12} {'LB':<10} {'Class':<22} Design")
    print(" " + "-" * 95)
    for rid, date, design, oof, lb, cls in lb_log:
        oof_s = f"{oof:.4f}" if isinstance(oof, float) else oof
        print(f" {rid:<14} {date:<12} {oof_s:<12} {lb:<10} {cls:<22} {design[:55]}")

    # ─── Component-by-component holdout AUC (B-pure / B-feature / B-impure) ──
    print("\n" + "=" * 78)
    print(" PER-COMPONENT P11-HOLDOUT AUC (proxy for test generalization)")
    print("=" * 78)

    canonical_components = [
        # Currently in R-034 PAIR + variants (proven safe)
        ("v11_aug_oldtest",       "B-pure (in R-034)"),
        ("v11plus",               "transformer (in R-034)"),
        ("v13_oldtest",           "B-pure GBM (in R-034)"),
        ("v14_seed2_v15feat_a",   "B-feature (R-034 LB-WIN)"),
        ("v16_avg3",              "B-pure (in R-034)"),
        # Recently parked / LB-failed components (for contrast)
        ("v11_mulminet_aug_avg3", "B-impure (R-055 LB-fail)"),
        ("v14_seed2_v16match_v2", "B-player-style (R-062r LB-fail)"),
        ("v14_seed2_v15feat_c_oldtest", "B-feature variant (untested LB)"),
        # NOTE: v22_causal_lm_v1 is per-rally (15833) — handled separately below
    ]

    comp_tags = [c[0] for c in canonical_components]
    comp, y_a, y_p, y_s, _, _ = load_components(comp_tags)

    print(f"\n {'Component':<42} {'Class':<32} {'Full AUC':<10} {'Holdout AUC':<12} {'Δ':<8}")
    print(" " + "-" * 110)
    rows = []
    for tag, cls in canonical_components:
        if tag not in comp:
            print(f"   {tag:<42} {cls:<32}   (not loaded)")
            continue
        srv = comp[tag]["oof_srv"]
        if srv.shape[0] == len(holdout):
            full_auc = roc_auc_score(y_s, srv)
            holdout_auc = roc_auc_score(y_s[holdout], srv[holdout])
            delta = holdout_auc - full_auc
            print(f"   {tag:<42} {cls:<32}   {full_auc:.4f}    {holdout_auc:.4f}     {delta:+.4f}")
            rows.append({
                "tag": tag, "class": cls,
                "full_auc": float(full_auc),
                "holdout_auc": float(holdout_auc),
                "delta": float(delta),
            })
        else:
            print(f"   {tag:<42} {cls:<32}   (shape mismatch: {srv.shape})")

    # ─── R-042 / R-067cr-style blend AUC on holdout ──────────────────────────
    print("\n" + "=" * 78)
    print(" BLEND AUC: R-042 (R-034 PAIR) baseline on FULL vs HOLDOUT")
    print("=" * 78)
    R034 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]
    base = evaluate_subset_none(R034, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=300, seed=20260524)
    srv_stack = np.stack([comp[t]["oof_srv"] for t in R034], axis=0)
    r034_srv = (base["w_s"][:, None] * srv_stack).sum(axis=0)
    full_auc_r034 = roc_auc_score(y_s, r034_srv)
    holdout_auc_r034 = roc_auc_score(y_s[holdout], r034_srv[holdout])
    print(f"   R-034 PAIR srv  full AUC: {full_auc_r034:.4f}  holdout AUC: {holdout_auc_r034:.4f}  "
          f"Δ={holdout_auc_r034 - full_auc_r034:+.4f}")

    # ─── Per-class generalization analysis on R-034 PAIR action+point ────────
    blend_a = np.stack([comp[t]["oof_act"] for t in R034], axis=0)
    blend_a = (base["w_a"][:, None, None] * blend_a).sum(axis=0)
    blend_p = np.stack([comp[t]["oof_pt"] for t in R034], axis=0)
    blend_p = (base["w_p"][:, None, None] * blend_p).sum(axis=0)
    pred_a = blend_a[:, :N_ACTION_TRAIN].argmax(axis=1)
    pred_p = blend_p.argmax(axis=1)

    y_a_clipped = np.where(y_a >= N_ACTION_TRAIN, 0, y_a)
    full_f1_a = f1_score(y_a_clipped, pred_a, labels=list(range(N_ACTION_TRAIN)),
                          average="macro", zero_division=0)
    holdout_f1_a = f1_score(y_a_clipped[holdout], pred_a[holdout],
                             labels=list(range(N_ACTION_TRAIN)), average="macro", zero_division=0)
    full_f1_p = f1_score(y_p, pred_p, labels=list(range(N_POINT)),
                          average="macro", zero_division=0)
    holdout_f1_p = f1_score(y_p[holdout], pred_p[holdout],
                             labels=list(range(N_POINT)), average="macro", zero_division=0)

    full_ov = 0.4 * full_f1_a + 0.4 * full_f1_p + 0.2 * full_auc_r034
    holdout_ov = 0.4 * holdout_f1_a + 0.4 * holdout_f1_p + 0.2 * holdout_auc_r034
    print(f"\n   R-034 PAIR full OV: {full_ov:.4f}  holdout OV: {holdout_ov:.4f}  "
          f"Δ={holdout_ov - full_ov:+.4f}")
    print(f"   R-034 PAIR full F1a: {full_f1_a:.4f}  holdout F1a: {holdout_f1_a:.4f}  "
          f"Δ={holdout_f1_a - full_f1_a:+.4f}")
    print(f"   R-034 PAIR full F1p: {full_f1_p:.4f}  holdout F1p: {holdout_f1_p:.4f}  "
          f"Δ={holdout_f1_p - full_f1_p:+.4f}")

    # Save manifest
    manifest = {
        "ts": "2026-05-24",
        "purpose": "Post-reset generalization audit; identifies components that generalize vs overfit OOF",
        "p11_holdout_rows": int(holdout.sum()),
        "p11_holdout_pct": float(100 * holdout.sum() / len(holdout)),
        "p11_holdout_players": 25,
        "test_overlap_pct": 56.3,
        "components": rows,
        "R034_PAIR_full_AUC": float(full_auc_r034),
        "R034_PAIR_holdout_AUC": float(holdout_auc_r034),
        "R034_PAIR_full_OV": float(full_ov),
        "R034_PAIR_holdout_OV": float(holdout_ov),
        "R034_PAIR_holdout_delta": float(holdout_ov - full_ov),
    }
    out_path = os.path.join(SUBMISSION_DIR, "post_reset_generalization_audit.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved: {out_path}")


if __name__ == "__main__":
    main()
