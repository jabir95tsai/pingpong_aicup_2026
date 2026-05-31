"""Lean Phase 3 analyzer (2026-05-20, post-moderation).

Single purpose: test whether any new oldtest component (raw or averaged)
beats R-027 PAIR (current LB-best 0.3810401) via single-swap into the
R-027 subset.

Baseline blend = R-027 PAIR:
    (v11_aug_oldtest, v11plus, v13_oldtest, v14_seed2, v16_avg3)

For each candidate replacement (variance-positive families only — per the
2026-05-20 LESSONS finding that v11_aug/v11plus seed averaging is a no-op),
swap into the slot and measure the OOF lift.

USAGE:
    python -u src/analyze_phase3.py
    python -u src/analyze_phase3.py --write-top1

NO SUBMISSION written by default. Pass --write-top1 to materialise the
single highest-OOF candidate (only after human review).
"""
import argparse
import itertools
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

# Current LB best (CLASS B-pure transfer ratio 1.0035)
R027_LB = 0.3810401
R027_RATIO = 1.0035
R027_SUBSET = [
    "v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2", "v16_avg3",
]

# Variance-positive families only (per 2026-05-20 LESSONS).
# v11_aug, v11plus seed averaging is a no-op — excluded.
PHASE3_CANDIDATES = {
    # Slot in R-027 PAIR  →  candidate replacement
    "v13_oldtest": [
        "v13_oldtest_avg2", "v13_oldtest_avg3",
        # also test seed swaps for diagnostic
        "v13_oldtest_seed51966",   # best single seed (OV 0.3700)
        "v13_oldtest_seed9",       # second-best single (OV 0.3695)
    ],
    "v14_seed2": [
        "v14_oldtest_avg2", "v14_oldtest_avg3",
        # also: v14_seed0_oldtest (0.3680), v14_seed1_oldtest (0.3684)
    ],
    "v16_avg3": [
        "v16_testhist_aug_oldtest_avg3",  # original 3-seed avg
        "v16_testhist_aug_oldtest_avg5",  # new 5-seed avg
    ],
    "v11plus": [
        # v11_mulminet variants only (since v11plus seeds are zero-variance)
        "v11_mulminet_aug_oldtest_avg3",  # NEW Phase 3 derived
        "v11_mulminet_aug_oldtest_avg2",  # Phase 2 derived
    ],
    # v11_aug_oldtest: skip — all 4 seeds are 0.3253 exactly (zero variance)
}

# Pair candidates (both slots variance-positive)
PHASE3_PAIRS = [
    ("v13_oldtest", "v14_seed2"),
    ("v13_oldtest", "v16_avg3"),
    ("v14_seed2", "v16_avg3"),
    ("v13_oldtest", "v11plus"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-top1", action="store_true",
        help="If set, materialise the top candidate (delta_OV > +0.002 vs R-027) "
             "as a NONE-blend submission CSV. NO upload — just file generation.",
    )
    parser.add_argument("--n-samples", type=int, default=300)
    args = parser.parse_args()

    print("=" * 70)
    print(" Phase 3 analyzer — R-027 PAIR baseline only (lean)")
    print("=" * 70)
    print(f" R-027 baseline LB: {R027_LB:.7f}  (transfer ratio {R027_RATIO})")
    print(f" R-027 subset: {R027_SUBSET}")
    print()

    # Collect tags: baseline + all candidates
    all_tags = list(set(R027_SUBSET))
    for slot, cands in PHASE3_CANDIDATES.items():
        all_tags.extend(cands)
    all_tags = list(set(all_tags))

    comp, y_a, y_p, y_s, _, test_uid = load_components(all_tags)

    available = {}
    for slot, cands in PHASE3_CANDIDATES.items():
        for c in cands:
            if c in comp:
                available.setdefault(slot, []).append(c)

    missing = []
    for slot, cands in PHASE3_CANDIDATES.items():
        for c in cands:
            if c not in comp:
                missing.append((slot, c))

    print(f" Candidates available: {sum(len(v) for v in available.values())}"
          f" / {sum(len(v) for v in PHASE3_CANDIDATES.values())}")
    for slot, cs in available.items():
        for c in cs:
            print(f"   {slot:<18} -> {c}")
    if missing:
        print(f" Missing ({len(missing)}):")
        for slot, c in missing:
            print(f"   {slot:<18} -> {c}  [no OOF found]")
    print()

    # Baseline
    print(" --- R-027 PAIR baseline ---")
    base = evaluate_subset_none(R027_SUBSET, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=args.n_samples)
    print(f"   OV={base['OV']:.4f}  F1_a={base['F1_a']:.4f}  "
          f"F1_p={base['F1_p']:.4f}  AUC={base['AUC']:.4f}")
    print()

    rows: List[Dict] = []
    rows.append({"label": "BASELINE R-027 PAIR", "subset": R027_SUBSET,
                 "kind": "baseline", **base})

    # Single swaps
    print(" --- Single swaps ---")
    for slot, cands in available.items():
        for c in cands:
            new_subset = [c if t == slot else t for t in R027_SUBSET]
            m = evaluate_subset_none(new_subset, comp, y_a, y_p, y_s,
                                      optimize=True, n_samples=args.n_samples)
            delta = m["OV"] - base["OV"]
            rows.append({
                "label": f"SWAP {slot}->{c}",
                "subset": new_subset, "kind": "single",
                **m, "delta_OV": delta,
                "pred_LB": m["OV"] * R027_RATIO,
            })

    # Pair swaps (variance-positive pairs only)
    print(" --- Pair swaps ---")
    for slot_a, slot_b in PHASE3_PAIRS:
        cands_a = available.get(slot_a, [])
        cands_b = available.get(slot_b, [])
        if not cands_a or not cands_b:
            continue
        for ca, cb in itertools.product(cands_a, cands_b):
            new_subset = []
            for t in R027_SUBSET:
                if t == slot_a:
                    new_subset.append(ca)
                elif t == slot_b:
                    new_subset.append(cb)
                else:
                    new_subset.append(t)
            m = evaluate_subset_none(new_subset, comp, y_a, y_p, y_s,
                                      optimize=True, n_samples=args.n_samples)
            delta = m["OV"] - base["OV"]
            rows.append({
                "label": f"PAIR {slot_a}+{slot_b}={ca[:25]}|{cb[:25]}",
                "subset": new_subset, "kind": "pair",
                **m, "delta_OV": delta,
                "pred_LB": m["OV"] * R027_RATIO,
            })

    # Ensure baseline has delta_OV/pred_LB for sort
    rows[0]["delta_OV"] = 0.0
    rows[0]["pred_LB"] = rows[0]["OV"] * R027_RATIO

    # Rank
    rows_sorted = sorted(rows, key=lambda r: -r["OV"])

    print()
    print(" === Ranking by OOF OV (Dirichlet(300) per-task NONE blend) ===")
    print(f" {'#':>2}  {'label':<58}  {'OV':>6}  {'dOV(R027)':>10}  {'pred_LB':>8}")
    print(" " + "-" * 95)
    for i, r in enumerate(rows_sorted, start=1):
        marker = ""
        if r["delta_OV"] >= 0.002:
            marker = " *** STRONG ***"
        elif r["delta_OV"] >= 0.001:
            marker = " (weak)"
        elif r["delta_OV"] < 0:
            marker = ""
        print(f" {i:>2}  {r['label']:<58}  {r['OV']:.4f}  "
              f"{r['delta_OV']:+.4f}     {r['pred_LB']:.4f}{marker}")
    print()

    # Save ranking
    rank_rows = []
    for r in rows_sorted:
        rank_rows.append({
            "label": r["label"],
            "kind": r["kind"],
            "OV": r["OV"],
            "delta_OV": r["delta_OV"],
            "pred_LB": r["pred_LB"],
            "subset": ",".join(r["subset"]),
        })
    rank_path = os.path.join(SUBMISSION_DIR, "phase3_ranking.csv")
    pd.DataFrame(rank_rows).to_csv(rank_path, index=False)
    print(f" Ranking saved → {rank_path}")
    print()

    # Verdict
    strong = [r for r in rows_sorted if r["delta_OV"] >= 0.002 and r["kind"] != "baseline"]
    weak = [r for r in rows_sorted if 0.001 <= r["delta_OV"] < 0.002 and r["kind"] != "baseline"]

    print(" === Verdict ===")
    if strong:
        print(f" STRONG candidates (dOV ≥ +0.002): {len(strong)}")
        top = strong[0]
        print(f"   Top: {top['label']}")
        print(f"   OV {top['OV']:.4f}, dOV {top['delta_OV']:+.4f}, pred_LB {top['pred_LB']:.4f}")
        print(f"   Subset: {top['subset']}")
        if args.write_top1:
            pa, pp, ps = build_none_test(
                top["subset"], comp,
                w_a=top.get("w_a"), w_p=top.get("w_p"), w_s=top.get("w_s"),
            )
            fname = f"submission_R030_top1_NONE_{top['label'][:60].replace(' ', '_').replace('>', 'TO').replace('|', '_')}.csv"
            write_submission(test_uid, pa, pp, ps, fname)
            print(f"   Submission CSV written: {fname}")
        else:
            print("   --write-top1 NOT set — CSV not generated. Pass --write-top1 to materialise.")
    elif weak:
        print(f" WEAK candidates only (+0.001 ≤ dOV < +0.002): {len(weak)}")
        print(f"   Top: {weak[0]['label']}  dOV {weak[0]['delta_OV']:+.4f}")
        print("   Below LB-candidate threshold. Diagnostic only.")
    else:
        print(" NO candidate beat R-027 PAIR by ≥ +0.001 OOF.")
        print(" Phase 3 produces no new LB candidate via swap analysis.")

    print()
    print(" Note: predicted_LB uses R-027 empirical ratio 1.0035 (CLASS B-pure).")
    print(" Architecture-change swaps (CLASS B-impure) revert to ~0.97 ratio per R-028 top1.")
    print(" Verify swap is LIKE-FOR-LIKE oldtest (same arch + only data change) before upload.")


if __name__ == "__main__":
    main()
