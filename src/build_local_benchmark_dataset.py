"""Build a self-contained local benchmark dataset from train.csv.

Outputs:
- train.csv: train rows excluding holdout matches
- test.csv: prefix rows for holdout rallies
- truth.csv: one-row-per-rally next-shot labels for holdout rallies
- sample_submission.csv: submission header for convenience

This lets any existing training script run in "competition mode" by pointing
``PINGPONG_DATA_DIR`` to the generated dataset directory.
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import PROJECT_ROOT, TRAIN_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(PROJECT_ROOT, "artifacts", "local_benchmark"),
        help="Directory to save benchmark train/test/truth files.",
    )
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_PATH)
    rally_first = df.groupby("rally_uid", sort=False).first().reset_index()
    groups = rally_first["match"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.holdout_frac, random_state=args.seed)
    train_idx, holdout_idx = next(splitter.split(rally_first, groups=groups))

    train_rallies = set(rally_first.iloc[train_idx]["rally_uid"].tolist())
    holdout_rallies = set(rally_first.iloc[holdout_idx]["rally_uid"].tolist())

    train_df = df[df["rally_uid"].isin(train_rallies)].copy()
    holdout_df = df[df["rally_uid"].isin(holdout_rallies)].copy()
    holdout_df = holdout_df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)

    prefix_parts = []
    truth_rows = []
    dropped_short = 0

    for rally_uid, group in holdout_df.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber")
        if len(group) < 2:
            dropped_short += 1
            continue

        prefix_parts.append(group.iloc[:-1].copy())
        target = group.iloc[-1]
        truth_rows.append(
            {
                "rally_uid": int(rally_uid),
                "actionId": int(target["actionId"]),
                "pointId": int(target["pointId"]),
                "serverGetPoint": int(target["serverGetPoint"]),
            }
        )

    test_df = pd.concat(prefix_parts, axis=0).reset_index(drop=True)
    truth_df = pd.DataFrame(truth_rows).sort_values("rally_uid").reset_index(drop=True)
    sample_submission_df = truth_df[["rally_uid"]].copy()
    sample_submission_df["actionId"] = 0
    sample_submission_df["pointId"] = 0
    sample_submission_df["serverGetPoint"] = 0

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.csv")
    test_path = os.path.join(args.out_dir, "test.csv")
    truth_path = os.path.join(args.out_dir, "truth.csv")
    sample_path = os.path.join(args.out_dir, "sample_submission.csv")
    meta_path = os.path.join(args.out_dir, "meta.txt")

    train_df.to_csv(train_path, index=False, lineterminator="\n", encoding="utf-8")
    test_df.to_csv(test_path, index=False, lineterminator="\n", encoding="utf-8")
    truth_df.to_csv(truth_path, index=False, lineterminator="\n", encoding="utf-8")
    sample_submission_df.to_csv(sample_path, index=False, lineterminator="\n", encoding="utf-8")

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"source_train={TRAIN_PATH}\n")
        f.write(f"holdout_frac={args.holdout_frac}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"train_rallies={len(train_rallies)}\n")
        f.write(f"holdout_rallies={len(truth_df)}\n")
        f.write(f"train_matches={train_df['match'].nunique()}\n")
        f.write(f"holdout_matches={holdout_df['match'].nunique()}\n")
        f.write(f"train_rows={len(train_df)}\n")
        f.write(f"test_prefix_rows={len(test_df)}\n")
        f.write(f"dropped_short_rallies={dropped_short}\n")

    print(f"Saved benchmark train : {train_path} ({train_df.shape})")
    print(f"Saved benchmark test  : {test_path} ({test_df.shape})")
    print(f"Saved benchmark truth : {truth_path} ({truth_df.shape})")
    print(f"Saved sample sub      : {sample_path} ({sample_submission_df.shape})")
    print(f"Saved meta            : {meta_path}")


if __name__ == "__main__":
    main()
