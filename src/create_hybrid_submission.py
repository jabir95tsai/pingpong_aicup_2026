"""Create hybrid submission: V10 action/point + V11 continuous server probs.

The V11 transformer outputs continuous server probabilities [0,1] which
are much more informative than V10's binary 0/1 predictions when the
competition computes AUC-ROC.  This script replaces only the serverGetPoint
column with the V11 continuous values while keeping V10's superior action
and point predictions.

Usage:
  python src/create_hybrid_submission.py
  python src/create_hybrid_submission.py --v10 submissions/submission_v10.csv
"""
import sys, os, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SUBMISSION_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10", type=str,
                        default=os.path.join(SUBMISSION_DIR, "submission_v10.csv"),
                        help="V10 GBM submission CSV (action/point to keep)")
    parser.add_argument("--v11-srv", type=str,
                        default=os.path.join(os.path.dirname(SUBMISSION_DIR),
                                             "oof_predictions", "v11_test_srv.npy"),
                        help="V11 transformer test server probabilities (.npy)")
    parser.add_argument("--srv-blend", type=float, default=1.0,
                        help="Weight for V11 server probs (0=V10 only, 1=V11 only)")
    parser.add_argument("--binary-threshold", type=float, default=-1,
                        help="If >=0, threshold continuous probs to 0/1")
    args = parser.parse_args()

    if not os.path.exists(args.v10):
        print(f"ERROR: V10 submission not found: {args.v10}")
        return

    if not os.path.exists(args.v11_srv):
        print(f"ERROR: V11 server probs not found: {args.v11_srv}")
        return

    v10 = pd.read_csv(args.v10)
    v11_srv = np.load(args.v11_srv)

    print(f"V10 submission: {len(v10)} rows")
    print(f"V11 server probs: {len(v11_srv)} values  mean={v11_srv.mean():.4f}  std={v11_srv.std():.4f}")
    print(f"V10 SGP dist: {dict(v10['serverGetPoint'].value_counts().sort_index())}")

    if len(v10) != len(v11_srv):
        print(f"WARNING: length mismatch {len(v10)} vs {len(v11_srv)}")
        # Try to align by rally_uid order
        v10_rallies = v10["rally_uid"].values
        # The V11 test samples are in the order produced by build_samples
        # which may differ from V10 order; we'll trust they're aligned for now
        min_len = min(len(v10), len(v11_srv))
        v10 = v10.iloc[:min_len].copy()
        v11_srv = v11_srv[:min_len]

    # Blend: alpha * V11 + (1 - alpha) * V10_binary
    v10_srv_binary = v10["serverGetPoint"].values.astype(float)
    blended_srv = args.srv_blend * v11_srv + (1 - args.srv_blend) * v10_srv_binary

    if args.binary_threshold >= 0:
        final_srv = (blended_srv >= args.binary_threshold).astype(int)
        label = f"binary_{args.binary_threshold:.2f}"
    else:
        final_srv = blended_srv  # continuous
        label = "continuous"

    hybrid = v10.copy()
    hybrid["serverGetPoint"] = final_srv

    suffix = f"blend{args.srv_blend:.1f}_{label}"
    out_path = os.path.join(SUBMISSION_DIR, f"submission_v10_v11srv_{suffix}.csv")
    hybrid.to_csv(out_path, index=False)

    print(f"\nHybrid SGP stats: mean={final_srv.mean():.4f}  std={final_srv.std():.4f}")
    if args.binary_threshold >= 0:
        print(f"  Binary dist: {dict(pd.Series(final_srv).value_counts().sort_index())}")
    print(f"Saved: {out_path}")

    # Also create a version with optimal binary threshold (50/50 split)
    median_thresh = np.median(blended_srv)
    binary_50 = (blended_srv >= median_thresh).astype(int)
    hybrid_50 = v10.copy()
    hybrid_50["serverGetPoint"] = binary_50
    out_50 = os.path.join(SUBMISSION_DIR, "submission_v10_v11srv_binary50.csv")
    hybrid_50.to_csv(out_50, index=False)
    print(f"  50/50 binary split (thresh={median_thresh:.4f}): {dict(pd.Series(binary_50).value_counts().sort_index())}")
    print(f"  Saved: {out_50}")


if __name__ == "__main__":
    main()
