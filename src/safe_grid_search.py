"""Comprehensive safe-component grid search (R-069 prep).

Per user 2026-05-24 ("run all" option #3):
Exhaustive scan over EXCLUSIVELY-PROVEN-SAFE components and combinations.

Banned classes (per LESSONS_CHECKLIST 2026-05-24):
- B-impure: v11_mulminet family (all variants)
- B-meta: meta_stack v1 / meta_stack_v2_logistic
- B-player-style: v16match_v2 family, v14_recvprofile, v14_recvhand (per R-062r/R-054r)
- B-seedavg-of-toxic: v13_oldtest_avg3, v11_mulminet_aug_avg2/3
- Smoke / dry / parked: v18, v17_momentum, v14_pseudo_v1, v15_pp, v15_player_only

Safe pool:
- B-pure: v11_aug_oldtest, v13_oldtest, v16_avg3, v16_testhist_aug_oldtest, *_avg2/avg3 of safe families
- B-feature: v14_seed2_v15feat_a, v15feat_b, v15feat_c (B-feature class, untoxic)
- Originals: v11plus, v11_aug, v14_seed2

Search modes:
A. Single-swap into R-034 PAIR (5 components)
B. Single-ADD to R-034 PAIR (6 components) — RISK: ADDs were toxic on R-055
   ↳ skip ADDs in this run; only SWAPS

Output:
- submissions/r069_safe_grid_search.json (full ranking)
- submissions/r069_safe_grid_summary.txt (top-10 readable)

USAGE:
    python -u src/safe_grid_search.py
"""
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa: E402
from analyze_oldtest_blend import (  # noqa: E402
    load_components, evaluate_subset_none,
)

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")

R034_PAIR = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]
R042_LB = 0.3866
RATIO_CONS = 1.0035
RATIO_OPT = 1.0142
RULE_LIFT_LB = 0.0028

# Banned components (LB-confirmed toxic class, per LESSONS_CHECKLIST 2026-05-24)
BANNED = {
    # B-impure
    "v11_mulminet", "v11_mulminet_aug", "v11_mulminet_aug_oldtest",
    "v11_mulminet_aug_avg2", "v11_mulminet_aug_avg3",
    "v11_mulminet_aug_oldtest_avg2", "v11_mulminet_aug_oldtest_avg3",
    "v11_mulminet_aug_lam01", "v11_mulminet_aug_s12345", "v11_mulminet_aug_s31337",
    "v11_mulminet_aug_oldtest_seed7", "v11_mulminet_aug_oldtest_seed31337",
    "v11_mulminet_aug_oldtest_seed51966", "v11_mulminet_aug_oldtest_softf1_phaseB",
    "v11_mulminet_dry", "v11_mulminet_oldtest", "v11_mulminet_pretrained_aug",
    "v11_mulminet_pretrained_aug_smoke", "v11_mulminet_smoke",
    "v11_mulminet_uncertainty_aug",
    # B-meta
    "meta_stack", "meta_stack_v2_logistic",
    # B-player-style (LB-confirmed via R-062r / R-054r)
    "v14_seed2_v16match_v2", "v14_seed2_v16match_v2_smoke", "v14_seed2_v16match_v2_smoke_capped",
    "v14_recvprofile", "v14_recvprofile_oldtest",
    "v14_recvhand", "v14_recvhand_oldtest",
    # B-seedavg of toxic
    "v13_oldtest_avg3",   # R-033 −0.0015 LB
    # Hard-parked per RESULTS / LESSONS
    "v18", "v17_momentum", "v17_momentum_smoke_all", "v17_momentum_smoke_core",
    "v14_pseudo_v1", "v15_pp", "v15_player_only", "v15_hist_only",
    "v14_5f_nocb", "v11_aug_big", "v11_big", "v11plus_aug",
    # Smoke / dry / gate1 (handled by name-substring filter below)
    # SGP-prefix specialists (R-030 PARKED + row-order mismatch with canonical OOF)
    "sgp_prefix_v3_full", "sgp_prefix_v3_full_oldtest",
    # Misc row-order mismatches (sn2_expert / v12cb fail UID check; exclude proactively)
    "sn2_expert", "v12cb",
}

# Slots within R-034 PAIR
SLOTS = R034_PAIR


def discover_safe_components() -> List[str]:
    """All OOF tags excluding banned, smoke/dry, and R-034 PAIR members."""
    safe = []
    for fn in sorted(os.listdir(OOF_DIR)):
        if not fn.endswith("_oof_act.npy"):
            continue
        tag = fn[:-len("_oof_act.npy")]
        if tag in BANNED:
            continue
        if any(s in tag for s in ("_smoke", "_dry", "_gate1")):
            continue
        if tag in R034_PAIR:
            continue
        safe.append(tag)
    return safe


def main() -> None:
    print("=" * 78)
    print(" R-069 Safe-component grid search — single-SWAP into R-034 PAIR")
    print(" Banned classes filtered: B-impure, B-meta, B-player-style, hard-parked")
    print("=" * 78)

    safe_pool = discover_safe_components()
    print(f"\n Safe pool size: {len(safe_pool)}")
    print(f" R-034 PAIR slots: {SLOTS}")

    needed = list(set(R034_PAIR + safe_pool))
    comp, y_a, y_p, y_s, _, test_uid = load_components(needed)

    # Filter to actually loaded
    safe_loaded = [t for t in safe_pool if t in comp]
    print(f" Loaded: {len(safe_loaded)} safe pool components")

    # Baseline
    print("\n--- R-034 PAIR baseline (n=300 Dirichlet) ---")
    base = evaluate_subset_none(R034_PAIR, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=300, seed=20260524)
    print(f"  OV={base['OV']:.4f}")
    base_ov = base["OV"]

    # All swaps
    rows = []
    n_trials = len(safe_loaded) * len(SLOTS)
    print(f"\n--- Running {n_trials} swap trials ---")
    for cand in safe_loaded:
        for slot in SLOTS:
            new_subset = [cand if t == slot else t for t in R034_PAIR]
            try:
                m = evaluate_subset_none(new_subset, comp, y_a, y_p, y_s,
                                          optimize=True, n_samples=200, seed=20260524)
            except Exception as e:
                continue
            d = m["OV"] - base_ov
            pred_lb_lo = m["OV"] * RATIO_CONS + RULE_LIFT_LB
            pred_lb_hi = m["OV"] * RATIO_OPT + RULE_LIFT_LB
            rows.append({
                "swap_in": cand, "slot": slot,
                "OV": float(m["OV"]), "dOV": float(d),
                "F1_a": float(m["F1_a"]), "F1_p": float(m["F1_p"]), "AUC": float(m["AUC"]),
                "pred_LB_lo": float(pred_lb_lo), "pred_LB_hi": float(pred_lb_hi),
                "vs_R042_lo": float(pred_lb_lo - R042_LB),
                "vs_R042_hi": float(pred_lb_hi - R042_LB),
            })
            if d > 0:
                print(f"  + {cand:<40} → {slot:<24}  dOV={d:+.4f}  pred LB+rule={pred_lb_lo:.4f}-{pred_lb_hi:.4f}")

    rows.sort(key=lambda r: -r["OV"])
    print(f"\n=== TOP 10 (sorted by OV desc) ===")
    print(f"  baseline R-034 OV={base_ov:.4f}  (pred LB+rule {base_ov * RATIO_CONS + RULE_LIFT_LB:.4f}-{base_ov * RATIO_OPT + RULE_LIFT_LB:.4f}, R-042=0.3866)")
    print()
    for i, r in enumerate(rows[:10], 1):
        sign = "+" if r["dOV"] >= 0 else " "
        print(f"  #{i:>2}  {sign}{r['dOV']:+.4f}  {r['swap_in']:<42} → {r['slot']:<24}  OV={r['OV']:.4f}  pred_LB+rule={r['pred_LB_lo']:.4f}-{r['pred_LB_hi']:.4f}")

    out_path = os.path.join(SUBMISSION_DIR, "r069_safe_grid_search.json")
    with open(out_path, "w") as f:
        json.dump({
            "rid": "R-069",
            "ts": "2026-05-24",
            "baseline_R034_OV": float(base_ov),
            "R042_LB": R042_LB,
            "safe_pool": safe_loaded,
            "results": rows,
        }, f, indent=2)
    print(f"\n Saved: {out_path}")


if __name__ == "__main__":
    main()
