"""Build HIGH-RISK CLASS B-impure diagnostic submissions.

These are the strongest OOF candidates from the parked audit. They are CLASS
B-impure (architecture-change swaps) where R-028 historically LB-FAILED at
ratio ~0.97. We build the CSVs in case the user wants to test whether the
bigger OOF lift (+0.0039) can overcome the 3% ratio drop.

Math example (v11_mulminet_aug_avg3 → v11_aug_oldtest, OV ≈ 0.3819 at n=80):
  At ratio 0.97  → pred LB ≈ 0.3704 (REGRESS −0.0134 from 0.3838 LB best)
  At ratio 1.0035 → pred LB ≈ 0.3832 (still REGRESS −0.0006)
  At ratio 1.0121 → pred LB ≈ 0.3865 (NEW BEST +0.0027)

Outputs (R-040+ series for HIGH-RISK):
    submission_R040_v11_mulminet_aug_avg3_TO_v11_aug_oldtest.csv
    submission_R041_v11_mulminet_aug_avg2_TO_v11_aug_oldtest.csv
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

# HIGH-RISK candidates — biggest OOF lifts, but B-impure transfer hazard
CANDIDATES = [
    ("R040", "v11_aug_oldtest", "v11_mulminet_aug_avg3",
     "BIGGEST OOF lift +0.0039 STAGE 1; B-impure HIGH RISK (R-028 ratio 0.97 → pred LB regress)"),
    ("R041", "v11_aug_oldtest", "v11_mulminet_aug_avg2",
     "OOF +0.0027 STAGE 1; B-impure HIGH RISK"),
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
        fname = f"submission_{rid}_{cand}_TO_{slot}.csv"
        write_submission(test_uid, pa, pp, ps, fname)
        summary.append((rid, fname, m["OV"], delta, comment))
        print()

    print("=" * 70)
    print(" HIGH-RISK SUBMISSION CSVs BUILT — user decides whether to upload")
    print("=" * 70)
    for rid, fname, ov, delta, comment in summary:
        for ratio_name, ratio in [("0.97  B-impure", 0.97), ("1.0035 B-pure", 1.0035), ("1.0121 B-feat", 1.0121)]:
            print(f"  {rid} @ ratio {ratio_name}: pred LB {ov * ratio:.4f}")
        print(f"        file: {fname}")
        print(f"        why : {comment}")
        print()


if __name__ == "__main__":
    main()
