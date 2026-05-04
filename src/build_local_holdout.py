"""Build a local next-shot holdout set from train.csv.

For each selected holdout rally:
- prefix rows = all strikes except the last one
- truth row   = the last strike, converted to submission format

This mimics the competition's one-prediction-per-rally setup much better than
scoring directly against raw ``test.csv`` prefix rows.
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
        default=os.path.join(PROJECT_ROOT, "artifacts", "local_holdout"),
        help="Directory to save prefix/truth CSV files.",
    )
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_PATH)
    rally_first = df.groupby("rally_uid", sort=False).first().reset_index()
    groups = rally_first["match"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.holdout_frac, random_state=args.seed)
    _, holdout_idx = next(splitter.split(rally_first, groups=groups))
    holdout_rallies = set(rally_first.iloc[holdout_idx]["rally_uid"].tolist())

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

    prefix_df = pd.concat(prefix_parts, axis=0).reset_index(drop=True)
    truth_df = pd.DataFrame(truth_rows).sort_values("rally_uid").reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    prefix_path = os.path.join(args.out_dir, "prefix.csv")
    truth_path = os.path.join(args.out_dir, "truth.csv")
    meta_path = os.path.join(args.out_dir, "meta.txt")

    prefix_df.to_csv(prefix_path, index=False, lineterminator="\n", encoding="utf-8")
    truth_df.to_csv(truth_path, index=False, lineterminator="\n", encoding="utf-8")

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"train_path={TRAIN_PATH}\n")
        f.write(f"holdout_frac={args.holdout_frac}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"holdout_rallies={len(truth_df)}\n")
        f.write(f"prefix_rows={len(prefix_df)}\n")
        f.write(f"dropped_short_rallies={dropped_short}\n")
        f.write(f"holdout_matches={holdout_df['match'].nunique()}\n")

    print(f"Saved prefix: {prefix_path} ({prefix_df.shape})")
    print(f"Saved truth : {truth_path} ({truth_df.shape})")
    print(f"Saved meta  : {meta_path}")
    print(f"Holdout rallies: {len(truth_df)} | Holdout matches: {holdout_df['match'].nunique()}")


if __name__ == "__main__":
    main()
