"""Build supervised action/point training pairs from test.csv known history.

For each test rally with visible shots 1 .. n-1 (where shot n is the contest
prediction target), the feature builder in is_train=True mode generates
(n-1) - 1 = n-2 training pairs (predicting each non-first shot from its
preceding history).  Summing over all 1,236 rallies:

    pairs = total_test_shots - n_rallies = 3589 - 1236 = 2353

This script only pre-processes the raw test rows and saves them as a
parquet that train_v16_testhist_aug.py will consume.  All actual feature
building happens inside the training script per fold, using fold_stats
derived exclusively from real training rows.

Output: data/test_history_pairs.parquet
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TEST_PATH

EXPECTED_TEST_ROWS  = 3589
EXPECTED_RALLIES    = 1236
EXPECTED_AUG_PAIRS  = EXPECTED_TEST_ROWS - EXPECTED_RALLIES   # 2353


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default=TEST_PATH,
                        help="Path to test.csv")
    parser.add_argument("--out", type=str, default=None,
                        help="Output parquet path (default: data/test_history_pairs.parquet)")
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        data_dir = os.path.dirname(os.path.abspath(args.test))
        out_path = os.path.join(data_dir, "test_history_pairs.parquet")

    print(f"Loading test data from: {args.test}")
    raw_test = pd.read_csv(args.test)

    n_rows    = len(raw_test)
    n_rallies = raw_test["rally_uid"].nunique()
    print(f"  Test rows    : {n_rows}   (expected {EXPECTED_TEST_ROWS})")
    print(f"  Test rallies : {n_rallies}  (expected {EXPECTED_RALLIES})")

    expected_pairs = n_rows - n_rallies
    print(f"  Expected aug pairs (rows - rallies): {expected_pairs}"
          f"  (canonical {EXPECTED_AUG_PAIRS})")

    # ── SGP guard: overwrite any existing SGP with dummy -1 ─────────────────
    # test.csv may contain real serverGetPoint labels (0/1) as row-level context.
    # Per hard rules these must NEVER be used as truth or features in any model.
    # We unconditionally overwrite with -1 as a dummy placeholder for the feature
    # builder, and log how many real values were discarded.
    if "serverGetPoint" in raw_test.columns:
        n_real = (raw_test["serverGetPoint"].notnull() &
                  (raw_test["serverGetPoint"] != -1)).sum()
        if n_real > 0:
            print(f"  SGP guard: discarding {n_real} real SGP values from test.csv "
                  f"(overwriting with -1 per hard rules)")
        else:
            print(f"  SGP guard: column present, all null/-1 (clean)")
    raw_test["serverGetPoint"] = -1      # dummy placeholder for feature builder

    assert (raw_test["serverGetPoint"] == -1).all(), \
        "SGP guard failed: not all aug rows have serverGetPoint == -1"

    raw_test["is_aug"] = 1               # flag for exclusion from server model

    print("  NO_TRUE_TEST_SGP_USED = True")
    print(f"  SGP column set to -1 for all {len(raw_test)} rows")

    # ── Per-SN distribution (informational) ──────────────────────────────────
    print("\n  Per-strikeNumber distribution of test rows:")
    sn_counts = raw_test["strikeNumber"].value_counts().sort_index()
    for sn, cnt in sn_counts.items():
        print(f"    SN={sn}: {cnt}")

    # ── Save ──────────────────────────────────────────────────────────────────
    raw_test.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"  Rows: {len(raw_test)}")
    print("\nDone. Verify aug feature count == 2353 inside train_v16_testhist_aug.py.")


if __name__ == "__main__":
    main()
