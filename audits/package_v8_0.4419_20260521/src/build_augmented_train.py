"""
Build an augmented training set by appending an older fully-labelled test
file to ``train.csv``.

Why: when a competition releases a new test set without target columns, the
older test file (which still has all three target columns populated) is a
free source of labelled rally data. Concatenating it onto ``train.csv``
expands the training pool by ~8% with rows that match the same schema and
distribution.

The older test's ``rally_uid`` values are shifted by a configurable offset
so they remain disjoint from the original train rallies.

Usage:
    python -m src.build_augmented_train \\
        --train data/raw/train.csv \\
        --extra data/raw/test.csv \\
        --output data/raw/train_augmented.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_RALLY_UID_OFFSET = 20_000


def build_augmented_train(
    train: pd.DataFrame,
    extra: pd.DataFrame,
    rally_uid_offset: int = DEFAULT_RALLY_UID_OFFSET,
) -> pd.DataFrame:
    """Concatenate ``extra`` rows onto ``train`` after offsetting rally_uid.

    ``extra`` must contain every column in ``train`` (extras can have more,
    they are dropped) and have the three target columns populated.
    """
    targets = ["actionId", "pointId", "serverGetPoint"]
    missing = [t for t in targets if t not in extra.columns or extra[t].isna().any()]
    if missing:
        raise ValueError(
            f"extra dataframe missing/has NA in target columns: {missing}. "
            "Cannot use as augmentation source."
        )

    if not set(train.columns).issubset(set(extra.columns)):
        diff = set(train.columns) - set(extra.columns)
        raise ValueError(f"extra is missing columns required by train: {diff}")

    extra_aligned = extra[list(train.columns)].copy()
    extra_aligned["rally_uid"] = extra_aligned["rally_uid"] + rally_uid_offset

    overlap = set(train["rally_uid"]) & set(extra_aligned["rally_uid"])
    if overlap:
        raise ValueError(
            f"rally_uid offset {rally_uid_offset} did not prevent overlap "
            f"({len(overlap)} colliding ids). Pick a larger offset."
        )

    return pd.concat([train, extra_aligned], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--train", required=True, help="Base train.csv")
    p.add_argument("--extra", required=True, help="Older test.csv with full labels")
    p.add_argument("--output", required=True, help="Output augmented train CSV")
    p.add_argument("--rally-uid-offset", type=int, default=DEFAULT_RALLY_UID_OFFSET,
                   help="Offset added to extra's rally_uid (default 20000).")
    args = p.parse_args()

    train = pd.read_csv(args.train)
    extra = pd.read_csv(args.extra)
    out = build_augmented_train(train, extra, rally_uid_offset=args.rally_uid_offset)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"  train rallies: {train['rally_uid'].nunique()}")
    print(f"  extra rallies: {extra['rally_uid'].nunique()}")
    print(f"  combined:      {out['rally_uid'].nunique()}")
    print(f"  combined rows: {len(out)} ({len(out) - len(train):+d} vs train)")


if __name__ == "__main__":
    main()
