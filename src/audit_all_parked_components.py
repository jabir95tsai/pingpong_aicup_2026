"""Audit ALL never-LB-tested components via blend-swap against R-034 PAIR (LB 0.3838).

Per user directive (2026-05-21):
  "dont make the same mistake. and never make a conclusion without really
   submission score. and list out all the things that has never submit and
   check but just parked"

After R-034's +0.0028 LB win revealed that the old gate over-rejected
viable blend components, we now exhaustively scan EVERY parked OOF for
blend-swap potential. No verdicts are issued in this script — only OOF
deltas and predicted LB ranges. The user decides what to upload.

LB-best baseline: R-034 PAIR (0.3838279, 2026-05-21)
  Components: v11_aug_oldtest, v11plus, v13_oldtest, v14_seed2_v15feat_a, v16_avg3
  Transfer ratio (CLASS B-feature): 1.0121 (R-034 actual)
  Conservative ratio (CLASS B-pure): 1.0035 (R-027 PAIR origin)

Each parked component is tried as a swap into EACH of the 5 R-034 slots.
The best-OOF position is reported per component, plus a global ranking.

USAGE:
    python -u src/audit_all_parked_components.py
    python -u src/audit_all_parked_components.py --write-top-k 3

Outputs:
    submissions/parked_audit_full_ranking.csv  — all swap attempts ranked
    submissions/parked_audit_summary.csv       — best slot per component
"""
import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa: E402
from analyze_oldtest_blend import (  # noqa: E402
    load_components,
    evaluate_subset_none,
    build_none_test,
    write_submission,
)

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")

# ------------------------------------------------------------------
# R-034 PAIR baseline (current LB best, 0.3838279)
# ------------------------------------------------------------------
R034_LB = 0.3838279
R034_RATIO_CONSERVATIVE = 1.0035   # R-027 PAIR original (CLASS B-pure)
R034_RATIO_OPTIMISTIC = 1.0121     # R-034 actual (CLASS B-feature)

R034_SUBSET = [
    "v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3",
]

# ------------------------------------------------------------------
# LB-tested components (do NOT re-audit — already settled by LB)
# ------------------------------------------------------------------
LB_TESTED = {
    # Currently in R-034 PAIR
    "v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3",
    # Previously in R-027 PAIR (replaced by R-034 swap)
    "v14_seed2",
    # zoo lineage LB-tested
    "v11", "v11_aug", "v13", "v14_seed0", "v14_seed1",
    "v16_testhist_aug", "v16",   # v16 is alias for v16_testhist_aug single-seed
    "v17_momentum",
    # LB-FAILED components (per RESULTS.md)
    "v14_pseudo_v1",       # zoo_v12 elig1, LB 0.3626 — banned
    "v15_pp",              # LB 0.3507 — banned
    "v15_player_only",     # LB 0.3555 — banned (non-transfer player profile)
    "v14_5f_nocb",         # LB 0.3599 — superseded
    # R-028/R-033 LB-FAILED swaps (parts tested)
    "v11_mulminet_aug_oldtest_avg2",   # R-028 top1: -0.0086
    "v11_mulminet_aug_oldtest_avg3",   # R-033 CLASSBimpure: -0.0015
    "v13_oldtest_avg3",                # R-033 CLASSBpure: -0.0015
}


def discover_all_components() -> List[str]:
    """All unique component tags present in oof_predictions/."""
    tags = set()
    for fn in os.listdir(OOF_DIR):
        if fn.endswith("_oof_act.npy"):
            tags.add(fn[: -len("_oof_act.npy")])
    return sorted(tags)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Dirichlet samples per task (default 200; use 100 for fast mode)")
    parser.add_argument("--write-top-k", type=int, default=0,
                        help="Write top-K submission CSVs (default 0 = no writes)")
    parser.add_argument("--include-smoke", action="store_true",
                        help="Include _smoke/_dry/_gate1 dev runs (default: skip)")
    args = parser.parse_args()

    print("=" * 78)
    print(" PARKED-COMPONENT AUDIT — R-034 PAIR baseline (LB 0.3838279)")
    print("=" * 78)
    print(f" Baseline: {R034_SUBSET}")
    print(f" LB ratio (conservative B-pure): {R034_RATIO_CONSERVATIVE}")
    print(f" LB ratio (optimistic  B-feat ): {R034_RATIO_OPTIMISTIC}")
    print()

    all_tags = discover_all_components()
    print(f" Total OOF components discovered: {len(all_tags)}")

    # Filter
    parked = []
    for t in all_tags:
        if t in LB_TESTED:
            continue
        if not args.include_smoke and ("_smoke" in t or "_dry" in t or "_gate1" in t):
            continue
        parked.append(t)

    print(f" LB-tested (excluded from audit):   {len(LB_TESTED)}")
    print(f" Parked candidates to audit:         {len(parked)}")
    print()

    # Show parked components
    print(" === PARKED COMPONENTS (never LB-tested) ===")
    for i, t in enumerate(parked, 1):
        print(f"  {i:>3}. {t}")
    print()

    # Load all needed OOF arrays
    print(" Loading OOF arrays ...")
    needed = list(set(R034_SUBSET + parked))
    comp, y_a, y_p, y_s, mask, test_uid = load_components(needed)
    print(f"   Loaded: {len(comp)}/{len(needed)} tags")
    print()

    # Baseline
    print(" --- BASELINE: R-034 PAIR ---")
    base = evaluate_subset_none(R034_SUBSET, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=args.n_samples)
    print(f"   OV={base['OV']:.4f}  F1_a={base['F1_a']:.4f}  "
          f"F1_p={base['F1_p']:.4f}  AUC={base['AUC']:.4f}")
    print(f"   LB anchor: {R034_LB:.7f}   OOF→LB observed ratio: {R034_LB / base['OV']:.4f}")
    print()

    # Run all swaps
    rows: List[Dict] = []
    rows.append({
        "label": "BASELINE_R034_PAIR",
        "cand": "", "slot": "",
        "OV": base["OV"], "F1_a": base["F1_a"], "F1_p": base["F1_p"], "AUC": base["AUC"],
        "delta_OV": 0.0,
        "pred_LB_lo": base["OV"] * R034_RATIO_CONSERVATIVE,
        "pred_LB_hi": base["OV"] * R034_RATIO_OPTIMISTIC,
    })

    print(" --- Running parked-component swaps ---")
    missing_in_comp = []
    for cand in parked:
        if cand not in comp:
            missing_in_comp.append(cand)
            continue
        for slot in R034_SUBSET:
            new_subset = [cand if t == slot else t for t in R034_SUBSET]
            try:
                m = evaluate_subset_none(new_subset, comp, y_a, y_p, y_s,
                                          optimize=True, n_samples=args.n_samples)
            except Exception as e:
                print(f"   [error] {cand} into {slot}: {e}")
                continue
            delta = m["OV"] - base["OV"]
            rows.append({
                "label": f"SWAP_{slot}_TO_{cand}",
                "cand": cand, "slot": slot,
                "OV": m["OV"], "F1_a": m["F1_a"], "F1_p": m["F1_p"], "AUC": m["AUC"],
                "delta_OV": delta,
                "pred_LB_lo": m["OV"] * R034_RATIO_CONSERVATIVE,
                "pred_LB_hi": m["OV"] * R034_RATIO_OPTIMISTIC,
                "w_a": m["w_a"], "w_p": m["w_p"], "w_s": m["w_s"],
                "subset": new_subset,
            })
            marker = ""
            if delta >= 0.002:
                marker = " *** STRONG ***"
            elif delta >= 0.0:
                marker = " (tied/positive)"
            elif delta >= -0.002:
                marker = " (near-tied)"
            print(f"   {slot:<22} <- {cand:<40}  OV {m['OV']:.4f}  dOV {delta:+.4f}{marker}")
        print()

    if missing_in_comp:
        print(f" Missing OOF (skipped): {len(missing_in_comp)}")
        for t in missing_in_comp:
            print(f"   - {t}")

    # ------------------------------------------------------------------
    # Rankings
    # ------------------------------------------------------------------
    rows_sorted = sorted(rows, key=lambda r: -r["OV"])

    print()
    print("=" * 78)
    print(" GLOBAL RANKING — All swap attempts, sorted by OOF")
    print("=" * 78)
    print(f" {'#':>3}  {'label':<58}  {'OV':>7}  {'dOV':>8}  {'pred_LB(lo-hi)':>20}")
    print(" " + "-" * 100)
    for i, r in enumerate(rows_sorted[:40], 1):
        marker = ""
        if r["delta_OV"] >= 0.002:
            marker = "  ***"
        elif r["delta_OV"] >= 0.0:
            marker = "  +"
        elif r["delta_OV"] >= -0.001:
            marker = "  ~"
        print(f" {i:>3}  {r['label']:<58}  {r['OV']:.4f}  {r['delta_OV']:+.4f}  "
              f"{r['pred_LB_lo']:.4f}-{r['pred_LB_hi']:.4f}{marker}")

    # Best slot per component
    print()
    print("=" * 78)
    print(" BEST SLOT PER COMPONENT")
    print("=" * 78)
    best_per: Dict[str, Dict] = {}
    for r in rows[1:]:
        cand = r["cand"]
        if cand and (cand not in best_per or r["OV"] > best_per[cand]["OV"]):
            best_per[cand] = r

    best_sorted = sorted(best_per.values(), key=lambda r: -r["OV"])
    print(f" {'#':>3}  {'component':<42}  {'best slot':<22}  {'OV':>7}  {'dOV':>8}  {'pred_LB(lo-hi)':>20}")
    print(" " + "-" * 110)
    for i, r in enumerate(best_sorted, 1):
        marker = ""
        if r["delta_OV"] >= 0.002:
            marker = "  ***"
        elif r["delta_OV"] >= 0.0:
            marker = "  +"
        elif r["delta_OV"] >= -0.001:
            marker = "  ~"
        elif r["delta_OV"] >= -0.005:
            marker = "  ."
        print(f" {i:>3}  {r['cand']:<42}  {r['slot']:<22}  {r['OV']:.4f}  "
              f"{r['delta_OV']:+.4f}  {r['pred_LB_lo']:.4f}-{r['pred_LB_hi']:.4f}{marker}")

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    full_rows = [{k: v for k, v in r.items() if k not in {"w_a", "w_p", "w_s", "subset"}}
                 for r in rows_sorted]
    full_path = os.path.join(SUBMISSION_DIR, "parked_audit_full_ranking.csv")
    pd.DataFrame(full_rows).to_csv(full_path, index=False)
    print()
    print(f" Full ranking saved -> {full_path}")

    summary_rows = []
    for r in best_sorted:
        summary_rows.append({
            "component": r["cand"],
            "best_slot": r["slot"],
            "OV": r["OV"],
            "delta_OV": r["delta_OV"],
            "pred_LB_lo": r["pred_LB_lo"],
            "pred_LB_hi": r["pred_LB_hi"],
            "F1_a": r["F1_a"],
            "F1_p": r["F1_p"],
            "AUC": r["AUC"],
        })
    summary_path = os.path.join(SUBMISSION_DIR, "parked_audit_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f" Best-slot summary saved -> {summary_path}")

    # ------------------------------------------------------------------
    # Two-stage gate framework classification
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(" CANDIDATES BY GATE STATUS (two-stage framework, 2026-05-21)")
    print("=" * 78)

    # Stage 1: standalone-positive (dOV >= 0)
    s1 = [r for r in best_sorted if r["delta_OV"] >= 0.0]
    # Stage 2: blend-eligible (-0.002 <= dOV < 0)
    s2 = [r for r in best_sorted if -0.002 <= r["delta_OV"] < 0.0]
    # Stage 3: marginal (-0.005 <= dOV < -0.002)
    s3 = [r for r in best_sorted if -0.005 <= r["delta_OV"] < -0.002]
    # Park-hard (dOV < -0.005)
    park = [r for r in best_sorted if r["delta_OV"] < -0.005]

    print(f"\n [STAGE 1] STRONG / TIED (dOV >= 0): {len(s1)} candidate(s)")
    print("   → ELIGIBLE for direct LB upload (existing standalone fast-track)")
    for r in s1:
        print(f"     {r['cand']:<42}  slot={r['slot']:<22}  dOV={r['delta_OV']:+.4f}")

    print(f"\n [STAGE 2] NEAR-TIED (-0.002 <= dOV < 0): {len(s2)} candidate(s)")
    print("   → ELIGIBLE for blend-swap diagnostic upload (new gate, post-R-034)")
    for r in s2:
        print(f"     {r['cand']:<42}  slot={r['slot']:<22}  dOV={r['delta_OV']:+.4f}")

    print(f"\n [STAGE 3] MARGINAL (-0.005 <= dOV < -0.002): {len(s3)} candidate(s)")
    print("   → DIAGNOSTIC ONLY — hold unless new-signal-class evidence")
    for r in s3:
        print(f"     {r['cand']:<42}  slot={r['slot']:<22}  dOV={r['delta_OV']:+.4f}")

    print(f"\n [PARKED] dOV < -0.005: {len(park)} candidate(s) — PARK (still no LB evidence; user may override)")
    for r in park:
        print(f"     {r['cand']:<42}  slot={r['slot']:<22}  dOV={r['delta_OV']:+.4f}")

    # ------------------------------------------------------------------
    # Optional: write top-K submission CSVs
    # ------------------------------------------------------------------
    if args.write_top_k > 0:
        print()
        print(f" Writing top-{args.write_top_k} submissions ...")
        eligible_for_write = s1 + s2
        for i, r in enumerate(eligible_for_write[: args.write_top_k], 1):
            # Need full subset
            full_row = next(rr for rr in rows[1:]
                            if rr["cand"] == r["cand"] and rr["slot"] == r["slot"])
            pa, pp, ps = build_none_test(
                full_row["subset"], comp,
                w_a=full_row["w_a"], w_p=full_row["w_p"], w_s=full_row["w_s"],
            )
            fname = f"submission_R036_top{i}_{r['slot']}_TO_{r['cand']}.csv"
            write_submission(test_uid, pa, pp, ps, fname)


if __name__ == "__main__":
    main()
