"""Bayes-refine top 9-comp / 10-comp candidates from higher_order_blend_search.

Higher-order search found +0.0013 OV lifts at n_samples=120 for several 9/10-comp
blends. Compare apples-to-apples vs R-055 Bayes 7-comp (n=500 + COBYLA = 0.3844)
by Bayes-refining the same top candidates.

If any 9/10-comp Bayes-refined > 0.3844, it becomes a new submission candidate.

USAGE:
    python -u src/bayes_refine_top_higher_order.py
"""
import json
import os
import sys
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from analyze_oldtest_blend import load_components  # noqa: E402
from bayes_blend_search import search_best_weights  # noqa: E402


R052 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a",
        "v16_avg3", "meta_stack_v2_logistic", "v11_mulminet_aug_avg3"]

# Top candidates from higher_order Phase B/C summary (all +0.0013 at n=120)
CANDS = [
    # name, additions to R-052
    ("9c-A: +avg5 +mulminet_oldtest",
     ["v16_testhist_aug_oldtest_avg5", "v11_mulminet_aug_oldtest"]),
    ("9c-B: +avg3 +mulminet_oldtest",
     ["v16_testhist_aug_oldtest_avg3", "v11_mulminet_aug_oldtest"]),
    ("9c-C: +avg3 +avg5",
     ["v16_testhist_aug_oldtest_avg3", "v16_testhist_aug_oldtest_avg5"]),
    ("9c-D: +v15feat_c +avg5",
     ["v14_seed2_v15feat_c_oldtest", "v16_testhist_aug_oldtest_avg5"]),
    ("10c-A: +v15feat_c +v11_aug_avg3 +mulminet_oldtest",
     ["v14_seed2_v15feat_c_oldtest", "v11_aug_oldtest_avg3", "v11_mulminet_aug_oldtest"]),
    ("10c-B: +v15feat_b +avg3 +mulminet_oldtest",
     ["v14_seed2_v15feat_b_oldtest", "v16_testhist_aug_oldtest_avg3", "v11_mulminet_aug_oldtest"]),
]


def main() -> None:
    all_tags = list(set(R052 + sum([a for _, a in CANDS], [])))
    print(f"Loading {len(all_tags)} components ...")
    comp, y_a, y_p, y_s, _, _ = load_components(all_tags)
    print(f"  Loaded: {len(comp)}/{len(all_tags)}")

    # R-052 reference (Bayes-refined)
    print("\n=== R-052 7c Bayes-refined (reference) ===")
    r052 = search_best_weights(comp, R052, y_a, y_p, y_s,
                                dirichlet_samples=500, bayes_restarts=30,
                                seed=20260522)
    print(f"  OV={r052['OV']:.4f}  F1a={r052['F1_a']:.4f}  F1p={r052['F1_p']:.4f}  AUC={r052['AUC']:.4f}")
    ref_ov = r052["OV"]

    results = [{"name": "R-052 Bayes ref", "subset": R052, **{k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in r052.items() if k != "w_a" and k != "w_p" and k != "w_s"}}]

    # Candidates
    for name, adds in CANDS:
        subset = R052 + adds
        print(f"\n=== {name}  ({len(subset)}-comp) ===")
        r = search_best_weights(comp, subset, y_a, y_p, y_s,
                                 dirichlet_samples=500, bayes_restarts=30,
                                 seed=20260522)
        d = r["OV"] - ref_ov
        marker = "  *NEW LEADER*" if r["OV"] > ref_ov else ""
        print(f"  OV={r['OV']:.4f}  F1a={r['F1_a']:.4f}  F1p={r['F1_p']:.4f}  AUC={r['AUC']:.4f}  dOV vs R-052 Bayes={d:+.4f}{marker}")
        results.append({
            "name": name, "subset": subset,
            "OV": float(r["OV"]), "F1_a": float(r["F1_a"]),
            "F1_p": float(r["F1_p"]), "AUC": float(r["AUC"]),
            "dOV_vs_R052_bayes": float(d),
            "w_a": list(map(float, r["w_a"])),
            "w_p": list(map(float, r["w_p"])),
            "w_s": list(map(float, r["w_s"])),
        })

    print("\n" + "=" * 70)
    print(" FINAL RANKING (Bayes-refined, n=500 + COBYLA)")
    print("=" * 70)
    rsorted = sorted(results, key=lambda r: -r["OV"])
    for i, r in enumerate(rsorted, 1):
        d = r["OV"] - ref_ov
        print(f"  #{i:>2}  OV={r['OV']:.4f}  dOV={d:+.4f}  {r['name']}")

    out_path = "submissions/bayes_higher_order_refined.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
