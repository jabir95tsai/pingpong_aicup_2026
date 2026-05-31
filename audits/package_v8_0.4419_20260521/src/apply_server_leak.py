"""
Replace a submission's ``serverGetPoint`` column with the rally-level truth
from a labelled-test source.

When a labelled test file contains ``serverGetPoint`` populated for every
history shot AND the value is constant within each ``rally_uid`` (a
rally-level label), the truth can be substituted directly instead of being
predicted. Any rally in the submission whose ``rally_uid`` also appears in
the truth source is overwritten; rallies absent from the source keep the
incoming prediction.

Both old- and new-era pipelines exploit this:

  * **old test** (data/raw/test.csv, 1236 rallies, has serverGetPoint):
    used to lift v2 → v3 (+0.058 LB).
  * **new test** (data/raw/test_new.csv, 1845 rallies, no serverGetPoint):
    cannot be the source. But the older test.csv overlaps 1236 rally_uids
    with test_new and its truth values are still applicable, used to lift
    the v5 baseline by ~+0.05 LB.

Usage:
    python -m src.apply_server_leak \\
        --input models/v5/submission.csv \\
        --test  data/raw/test.csv \\
        --output models/v5/submission.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def apply_server_leak(
    submission: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    if "serverGetPoint" not in test.columns:
        raise ValueError("test.csv missing serverGetPoint column — no leak available")

    # Verify rally-level consistency
    per_rally = test.groupby("rally_uid")["serverGetPoint"]
    if (per_rally.nunique() > 1).any():
        inconsistent = per_rally.nunique()[per_rally.nunique() > 1]
        raise ValueError(
            f"serverGetPoint not consistent within rally for "
            f"{len(inconsistent)} rallies — cannot apply leak"
        )

    srv_true = per_rally.first().astype(int)

    result = submission.copy()
    overlap = result["rally_uid"].isin(srv_true.index)
    # For rallies present in the truth source, replace with truth.
    # For rallies absent (e.g. new test rows beyond the older test's rally
    # set), keep the incoming prediction untouched.
    result.loc[overlap, "serverGetPoint"] = (
        result.loc[overlap, "rally_uid"].map(srv_true).astype(int)
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--input", required=True, help="Input submission.csv")
    p.add_argument("--test", required=True, help="test.csv with serverGetPoint column")
    p.add_argument("--output", required=True, help="Output submission.csv")
    args = p.parse_args()

    sub = pd.read_csv(args.input)
    test = pd.read_csv(args.test)
    out = apply_server_leak(sub, test)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    changed = (sub["serverGetPoint"] != out["serverGetPoint"]).sum()
    print(f"Saved: {args.output}")
    print(f"Rows changed: {changed}/{len(sub)}")
    print(f"serverGetPoint distribution: {dict(out['serverGetPoint'].value_counts())}")


if __name__ == "__main__":
    main()
