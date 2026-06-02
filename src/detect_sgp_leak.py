"""serverGetPoint LEAK detector — guard the FINAL submission against the
apply_server_leak pattern (overwriting SGP with old test.csv rally-level truth).

The leak overwrites serverGetPoint for rallies overlapping data/test.csv with
the integer truth (0/1). A CLEAN, generalizing submission outputs continuous
probabilities (metric is AUC), so it will NOT have a high fraction of exact
0/1 values that match the old-test truth.

Signals (on rallies overlapping the old-test truth source):
  - frac_exact_int : fraction of submission SGP values that are exactly 0.0/1.0
  - frac_match_truth: among those exact-int rows, fraction equal to truth
LEAK if frac_exact_int > 0.50 AND frac_match_truth > 0.95.

Exit code 0 = CLEAN, 2 = LEAK DETECTED (so it can gate a pipeline).

USAGE:
    python src/detect_sgp_leak.py submissions/<file>.csv
    python src/detect_sgp_leak.py <file>.csv --truth data/test.csv
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd

EXACT_INT_THRESH = 0.50
MATCH_TRUTH_THRESH = 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--truth", default="data/test.csv",
                    help="old test.csv with serverGetPoint truth (default data/test.csv)")
    args = ap.parse_args()

    sub = pd.read_csv(args.submission)
    if "serverGetPoint" not in sub.columns or "rally_uid" not in sub.columns:
        print(f"[skip] {args.submission}: missing rally_uid/serverGetPoint")
        sys.exit(0)

    if not os.path.exists(args.truth):
        print(f"[warn] truth source {args.truth} not found — cannot check leak")
        sys.exit(0)
    truth = pd.read_csv(args.truth)
    if "serverGetPoint" not in truth.columns:
        print(f"[warn] {args.truth} has no serverGetPoint — no leak source")
        sys.exit(0)

    srv_true = truth.groupby("rally_uid")["serverGetPoint"].first().astype(int)
    overlap = sub[sub["rally_uid"].isin(srv_true.index)].copy()
    n_overlap = len(overlap)
    if n_overlap == 0:
        print("[clean] no rally overlap with truth source — leak not applicable")
        sys.exit(0)

    sgp = overlap["serverGetPoint"].to_numpy(dtype=float)
    is_exact_int = np.isin(sgp, [0.0, 1.0])
    frac_exact_int = is_exact_int.mean()

    overlap["_truth"] = overlap["rally_uid"].map(srv_true).astype(int)
    ei = overlap[is_exact_int]
    frac_match_truth = (ei["serverGetPoint"].astype(int) == ei["_truth"]).mean() if len(ei) else 0.0

    print(f"file: {args.submission}")
    print(f"  overlap rallies vs {args.truth}: {n_overlap}")
    print(f"  frac SGP exactly 0/1 : {frac_exact_int:.3f}  (clean submissions ~0; metric is AUC → probabilities)")
    print(f"  of those, frac == truth: {frac_match_truth:.3f}")

    leak = (frac_exact_int > EXACT_INT_THRESH) and (frac_match_truth > MATCH_TRUTH_THRESH)
    if leak:
        print("  >>> LEAK DETECTED — serverGetPoint matches old-test truth as hard labels.")
        print("      DO NOT upload. Use the pre-apply_server_leak (probabilistic) version.")
        sys.exit(2)
    print("  >>> CLEAN — serverGetPoint is model-predicted (no old-test truth overwrite).")
    sys.exit(0)


if __name__ == "__main__":
    main()
