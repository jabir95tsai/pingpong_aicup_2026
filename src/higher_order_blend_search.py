"""Higher-order blend search (9-comp, 10-comp) building on R-052.

R-052 7-comp baseline: +0.0041 OOF over R-034 (LB 0.3838 baseline).
Prior 10-comp finding: +0.0026 (lower than 7-comp).
Hypothesis: specific 9-comp / 10-comp combos in the unexplored space may
exceed +0.0041 by adding diversifying but not redundant components.

Strategy:
  Phase A (forward greedy): pick best 8th, then best 9th given 8th, then best 10th.
                            Fast — len(pool) * 3 trials.
  Phase B (exhaustive C(N,2)): all pairs added to R-052 -> 9-comp.
  Phase C (exhaustive C(N,3)): all triples added to R-052 -> 10-comp.

Candidate pool = parked components with positive blend signal in the
previous parked audit, EXCLUDING:
  - LB-confirmed failures (per audit_all_parked_components.py LB_TESTED)
  - Components already in R-052
  - Smoke/dry/gate1 development runs

Each blend uses Dirichlet random search (n_samples per task, default 150).

USAGE:
    python -u src/higher_order_blend_search.py
    python -u src/higher_order_blend_search.py --n-samples 300 --skip-c
"""
import argparse
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from analyze_oldtest_blend import (  # noqa: E402
    load_components, evaluate_subset_none,
)


# R-052 baseline (7-comp, +0.0041 OOF over R-034)
R052 = [
    "v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3",
    "meta_stack_v2_logistic", "v11_mulminet_aug_avg3",
]

# Candidate pool — parked components NOT in R-052 and NOT LB-failed.
# Curated for diversity: different arch / aug / seed-avg / feature recipes.
POOL = [
    # B-feature variants (different feature sets, not yet LB-tested)
    "v14_seed2_v15feat_b_oldtest",   # v15feat_b + oldtest (parked ~-0.0007)
    "v14_seed2_v15feat_c_oldtest",   # v15feat_c = +pressure features (R-047)
    "v14_seed2_v15feat_a_oldtest",   # variant of in-blend v14_seed2_v15feat_a + oldtest
    # B-pure / B-seedavg
    "v11_aug_oldtest_avg3",          # B-seedavg of in-blend slot
    "v11plus_oldtest_avg2",          # avg2 of in-blend v11plus + oldtest
    "v16_testhist_aug_oldtest_avg3", # B-pure GBM aug
    "v16_testhist_aug_oldtest_avg5", # 5-seed avg
    "v13_oldtest_avg2",              # avg2 GBM
    # B-feature add-ons
    "v14_recvprofile_oldtest",       # R-026 recvprofile B-feature
    "v14_recvhand_oldtest",          # R-024 recvhand B-feature
    "v14_seed2_oldtest",             # canonical v14 + oldtest only
    # Diversity slots
    "v11_mulminet_aug_oldtest",      # B-impure (different arch); single seed
]

# Known LB-failed / parked-hard components (from RESULTS.md / audit) — exclude.
LB_FAILED = {
    "v14_pseudo_v1", "v15_pp", "v15_player_only", "v14_5f_nocb",
    "v11_mulminet_aug_oldtest_avg2",   # R-028 swap fail
    "v11_mulminet_aug_oldtest_avg3",   # R-033 swap fail
    "v13_oldtest_avg3",                # R-033 swap fail (CLASS B-pure variant)
    "v18",                              # RESULTS.md: "Do NOT blend v18 into any zoo"
    "v17_momentum_smoke_all",          # smoke-fold artifact, parked-hard
}


def evaluate(subset, comp, y_a, y_p, y_s, n_samples):
    return evaluate_subset_none(
        subset, comp, y_a, y_p, y_s,
        optimize=True, n_samples=n_samples,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=150,
                    help="Dirichlet samples per task per blend (default 150)")
    ap.add_argument("--skip-c", action="store_true",
                    help="Skip Phase C (exhaustive C(N,3) — slowest)")
    ap.add_argument("--skip-b", action="store_true",
                    help="Skip Phase B (exhaustive C(N,2))")
    ap.add_argument("--pool-limit", type=int, default=0,
                    help="If >0, truncate pool to first N after presence-filter")
    args = ap.parse_args()

    print("=" * 78)
    print(" HIGHER-ORDER BLEND SEARCH — building on R-052 7-comp baseline")
    print("=" * 78)
    print(f" R-052 subset: {R052}")
    print(f" Candidate pool (raw): {len(POOL)}")
    print()

    # Filter pool: drop LB-failed and dedup with R-052
    pool_filtered = [t for t in POOL if t not in R052 and t not in LB_FAILED]
    print(f" Pool after exclusions (dedup + LB-failed): {len(pool_filtered)}")

    # Load components — only keep those whose OOF actually exists
    needed = list(set(R052 + pool_filtered))
    print(f" Loading components ({len(needed)}) ...")
    comp, y_a, y_p, y_s, _, test_uid = load_components(needed)
    pool = [t for t in pool_filtered if t in comp]
    missing = [t for t in pool_filtered if t not in comp]
    print(f"   Loaded: {len(comp)}/{len(needed)}")
    if missing:
        print(f"   Missing (skipped): {missing}")
    if args.pool_limit > 0:
        pool = pool[:args.pool_limit]
        print(f"   Truncated pool to {len(pool)} (--pool-limit)")
    print(f" Final pool ({len(pool)}): {pool}")
    print()

    # R-052 baseline at the same n_samples (apples-to-apples)
    print(" --- R-052 7-comp baseline ---")
    t0 = time.time()
    base = evaluate(R052, comp, y_a, y_p, y_s, args.n_samples)
    print(f"   OV={base['OV']:.4f}  F1_a={base['F1_a']:.4f}  F1_p={base['F1_p']:.4f}  "
          f"AUC={base['AUC']:.4f}  [{time.time()-t0:.1f}s]")
    base_ov = base["OV"]
    print()

    results: Dict[str, List[Dict]] = {"phase_a_greedy": [], "phase_b_pairs": [], "phase_c_triples": []}

    # ------------------------------------------------------------------
    # PHASE A — forward greedy: pick best 8th, then 9th, then 10th
    # ------------------------------------------------------------------
    print("=" * 78)
    print(" PHASE A — Forward greedy add (8th, 9th, 10th)")
    print("=" * 78)
    cur_subset = list(R052)
    cur_ov = base_ov
    avail = list(pool)
    for step in (8, 9, 10):
        print(f"\n  Step: picking {step}-th component (avail={len(avail)}) ...")
        best = None
        for cand in avail:
            t1 = time.time()
            m = evaluate(cur_subset + [cand], comp, y_a, y_p, y_s, args.n_samples)
            dt = time.time() - t1
            d = m["OV"] - cur_ov
            marker = "*" if d > 0 else " "
            print(f"   {marker} {cand:<40} OV={m['OV']:.4f}  dOV={d:+.4f}  [{dt:.1f}s]")
            row = {
                "step": step, "add": cand, "OV": m["OV"], "dOV_vs_prev": d,
                "dOV_vs_r052": m["OV"] - base_ov,
            }
            results["phase_a_greedy"].append(row)
            if best is None or m["OV"] > best["OV"]:
                best = {**row, "subset": cur_subset + [cand], "m": m}
        if best is None:
            print("    (no candidates left)")
            break
        cur_subset = best["subset"]
        cur_ov = best["OV"]
        avail.remove(best["add"])
        print(f"\n  PICKED step-{step}: add={best['add']} -> OV={cur_ov:.4f}  "
              f"dOV vs R-052={cur_ov - base_ov:+.4f}")

    # ------------------------------------------------------------------
    # PHASE B — exhaustive C(pool, 2) -> 9-comp
    # ------------------------------------------------------------------
    if not args.skip_b:
        print()
        print("=" * 78)
        n_pairs = len(pool) * (len(pool) - 1) // 2
        print(f" PHASE B — Exhaustive 9-comp (R-052 + 2 from pool) — {n_pairs} pairs")
        print("=" * 78)
        for i, (a, b) in enumerate(itertools.combinations(pool, 2), 1):
            t1 = time.time()
            m = evaluate(R052 + [a, b], comp, y_a, y_p, y_s, args.n_samples)
            dt = time.time() - t1
            d = m["OV"] - base_ov
            marker = "*" if d > 0 else " "
            print(f"   [{i:>3}/{n_pairs}]{marker} +{a:<32} +{b:<32}  OV={m['OV']:.4f}  dOV={d:+.4f}  [{dt:.1f}s]")
            results["phase_b_pairs"].append({
                "add1": a, "add2": b, "OV": m["OV"], "dOV_vs_r052": d,
                "F1_a": m["F1_a"], "F1_p": m["F1_p"], "AUC": m["AUC"],
            })

    # ------------------------------------------------------------------
    # PHASE C — exhaustive C(pool, 3) -> 10-comp
    # ------------------------------------------------------------------
    if not args.skip_c:
        print()
        print("=" * 78)
        n_trip = len(pool) * (len(pool) - 1) * (len(pool) - 2) // 6
        print(f" PHASE C — Exhaustive 10-comp (R-052 + 3 from pool) — {n_trip} triples")
        print("=" * 78)
        for i, (a, b, c) in enumerate(itertools.combinations(pool, 3), 1):
            t1 = time.time()
            m = evaluate(R052 + [a, b, c], comp, y_a, y_p, y_s, args.n_samples)
            dt = time.time() - t1
            d = m["OV"] - base_ov
            marker = "*" if d > 0 else " "
            if d > 0 or i % 10 == 0 or i == n_trip:
                print(f"   [{i:>3}/{n_trip}]{marker} +{a[:24]:<24} +{b[:24]:<24} +{c[:24]:<24}  "
                      f"OV={m['OV']:.4f}  dOV={d:+.4f}  [{dt:.1f}s]")
            results["phase_c_triples"].append({
                "add1": a, "add2": b, "add3": c, "OV": m["OV"], "dOV_vs_r052": d,
                "F1_a": m["F1_a"], "F1_p": m["F1_p"], "AUC": m["AUC"],
            })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(" SUMMARY — top results per phase")
    print("=" * 78)

    if results["phase_a_greedy"]:
        top_a = sorted(results["phase_a_greedy"], key=lambda r: -r["dOV_vs_r052"])[:5]
        print("\n PHASE A top 5 (greedy steps):")
        for r in top_a:
            print(f"   step={r['step']}  add={r['add']:<40}  dOV vs R-052={r['dOV_vs_r052']:+.4f}")

    if results["phase_b_pairs"]:
        top_b = sorted(results["phase_b_pairs"], key=lambda r: -r["dOV_vs_r052"])[:10]
        print("\n PHASE B top 10 (9-comp = R-052 + 2):")
        for r in top_b:
            print(f"   +{r['add1']:<32} +{r['add2']:<32}  dOV={r['dOV_vs_r052']:+.4f}")

    if results["phase_c_triples"]:
        top_c = sorted(results["phase_c_triples"], key=lambda r: -r["dOV_vs_r052"])[:10]
        print("\n PHASE C top 10 (10-comp = R-052 + 3):")
        for r in top_c:
            print(f"   +{r['add1']:<24} +{r['add2']:<24} +{r['add3']:<24}  dOV={r['dOV_vs_r052']:+.4f}")

    # Save full JSON for downstream submission building
    out_path = "submissions/higher_order_blend_search.json"
    os.makedirs("submissions", exist_ok=True)
    summary = {
        "R052_baseline": {"OV": float(base_ov), "F1_a": float(base["F1_a"]),
                          "F1_p": float(base["F1_p"]), "AUC": float(base["AUC"])},
        "n_samples_per_task": args.n_samples,
        "pool": pool,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n Saved: {out_path}")


if __name__ == "__main__":
    main()
