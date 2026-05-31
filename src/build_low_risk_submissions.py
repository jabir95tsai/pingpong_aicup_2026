"""Build LB-ready submission CSVs for the LOW-RISK STAGE 1 candidates.

LOW-RISK = B-feature (R-034 LB-WIN class) or B-meta (NEW SIGNAL CLASS, never LB-tested).

These are the candidates where the OOF→LB transfer ratio is most likely to
hold up (≥1.0035, possibly 1.0121 like R-034). HIGH-RISK CLASS B-impure
swaps are NOT built here — the user can opt in to those separately.

Outputs (R-036+ series):
    submission_R036_meta_stack_TO_v14_seed2_v15feat_a.csv
    submission_R037_v14_recvprofile_TO_v14_seed2_v15feat_a.csv
    submission_R038_meta_stack_v2_logistic_TO_v13_oldtest.csv
    submission_R039_v14_recvhand_TO_v14_seed2_v15feat_a.csv  (same as R-035 confirm)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_oldtest_blend import (
    load_components,
    evaluate_subset_none,
    build_none_test,
    write_submission,
)

R034_SUBSET = [
    "v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3",
]

# (R-id, slot, candidate, comment)
CANDIDATES = [
    ("R036", "v14_seed2_v15feat_a", "meta_stack",
     "STAGE 1 OOF +0.0012, NEW SIGNAL CLASS (stacking ensemble, never LB-tested)"),
    ("R037", "v14_seed2_v15feat_a", "v14_recvprofile",
     "STAGE 1 OOF +0.0007, B-feature class (R-034 LB-WIN pattern)"),
    ("R038", "v13_oldtest", "meta_stack_v2_logistic",
     "STAGE 1 OOF +0.0006, NEW SIGNAL CLASS (stacking v2, never LB-tested)"),
    ("R039", "v14_seed2_v15feat_a", "v14_recvhand",
     "STAGE 1 OOF +0.0004, B-feature class (also R-035 if user prefers shorter naming)"),
]


def main() -> None:
    os.environ["ALLOW_UID_MISMATCH"] = "1"
    all_tags = list(set(R034_SUBSET + [c[2] for c in CANDIDATES]))
    comp, y_a, y_p, y_s, _, test_uid = load_components(all_tags)

    base = evaluate_subset_none(R034_SUBSET, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=200)
    print(f"R-034 baseline OV (n=200): {base['OV']:.4f}")
    print()

    summary = []
    for rid, slot, cand, comment in CANDIDATES:
        if cand not in comp:
            print(f"[{rid}] SKIP — {cand} not loaded")
            continue
        new_subset = [cand if t == slot else t for t in R034_SUBSET]
        m = evaluate_subset_none(new_subset, comp, y_a, y_p, y_s,
                                  optimize=True, n_samples=200)
        delta = m["OV"] - base["OV"]
        print(f"[{rid}] swap {slot} -> {cand}")
        print(f"       OV {m['OV']:.4f}  dOV {delta:+.4f}  (n=200 final)")
        print(f"       Comment: {comment}")

        pa, pp, ps = build_none_test(
            new_subset, comp,
            w_a=m["w_a"], w_p=m["w_p"], w_s=m["w_s"],
        )
        fname = f"submission_{rid}_{slot}_TO_{cand}.csv"
        write_submission(test_uid, pa, pp, ps, fname)
        summary.append((rid, fname, m["OV"], delta, comment))
        print()

    print("=" * 70)
    print(" SUMMARY OF LB-READY CSVs (LOW-RISK candidates)")
    print("=" * 70)
    for rid, fname, ov, delta, comment in summary:
        print(f"  {rid}: OV {ov:.4f} (dOV {delta:+.4f})")
        print(f"        file: {fname}")
        print(f"        why : {comment}")


if __name__ == "__main__":
    main()
