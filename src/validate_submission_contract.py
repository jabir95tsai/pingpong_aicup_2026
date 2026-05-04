"""Validate submission CSVs and cross-file rally alignment.

Use this to sanity-check PR artifacts before trusting offline or leaderboard
results. It verifies:
  - required submission columns exist
  - no duplicate rally_uid rows
  - actionId / pointId are integer-like and inside valid class ranges
  - serverGetPoint stays inside [0, 1]
  - multiple submission files share the same rally_uid order
  - optional reference coverage against a prefix test.csv or submission-like CSV
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]


def fail(message):
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def infer_expected_rallies(path):
    df = pd.read_csv(path)
    if "rally_uid" not in df.columns:
        fail(f"Reference file {path} does not contain rally_uid.")

    if "strikeNumber" in df.columns:
        # Prefix-style test.csv: preserve the visible-rally order from the last
        # observed row of each rally.
        last_rows = (
            df.reset_index()
            .sort_values(["rally_uid", "strikeNumber", "index"])
            .groupby("rally_uid", sort=False)
            .tail(1)
            .sort_values("index")
        )
        return last_rows["rally_uid"].tolist(), "prefix"

    if df["rally_uid"].duplicated().any():
        dup_count = int(df["rally_uid"].duplicated().sum())
        fail(
            f"Reference file {path} has duplicate rally_uid values ({dup_count} duplicates) "
            "but no strikeNumber column, so expected submission order cannot be inferred safely."
        )

    return df["rally_uid"].tolist(), "submission"


def is_integer_like(series):
    values = series.dropna().to_numpy()
    if values.size == 0:
        return True
    return np.allclose(values, np.round(values))


def validate_value_column(df, col, low, high, require_integer=True):
    if df[col].isnull().any():
        fail(f"{col} contains null values.")
    if require_integer and not is_integer_like(df[col]):
        fail(f"{col} must be integer-like, but found non-integer values.")
    values = df[col].to_numpy()
    if values.min() < low or values.max() > high:
        fail(f"{col} must be within [{low}, {high}], found range [{values.min()}, {values.max()}].")


def validate_submission(path, require_binary_server):
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        fail(f"{path} is missing required columns: {missing}")

    if df["rally_uid"].duplicated().any():
        dup_count = int(df["rally_uid"].duplicated().sum())
        fail(f"{path} has duplicate rally_uid values ({dup_count} duplicates).")

    validate_value_column(df, "actionId", 0, 18, require_integer=True)
    validate_value_column(df, "pointId", 0, 9, require_integer=True)
    validate_value_column(df, "serverGetPoint", 0.0, 1.0, require_integer=require_binary_server)

    return df[REQUIRED_COLUMNS].copy()


def compare_rally_order(reference_name, reference_rallies, candidate_name, candidate_rallies):
    if len(reference_rallies) != len(candidate_rallies):
        fail(
            f"{candidate_name} row count {len(candidate_rallies)} does not match "
            f"{reference_name} row count {len(reference_rallies)}."
        )

    ref_set = set(reference_rallies)
    cand_set = set(candidate_rallies)
    if ref_set != cand_set:
        missing = list(ref_set - cand_set)[:5]
        extra = list(cand_set - ref_set)[:5]
        fail(
            f"{candidate_name} rally_uid set does not match {reference_name}. "
            f"Missing sample={missing} Extra sample={extra}"
        )

    mismatches = [idx for idx, (a, b) in enumerate(zip(reference_rallies, candidate_rallies)) if a != b]
    if mismatches:
        first = mismatches[0]
        fail(
            f"{candidate_name} rally_uid order differs from {reference_name} at row {first}: "
            f"{reference_rallies[first]} != {candidate_rallies[first]}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Submission CSV path(s) to validate.")
    parser.add_argument(
        "--reference-path",
        help="Optional submission-like CSV or prefix test.csv used to verify expected rally coverage/order.",
    )
    parser.add_argument(
        "--allow-continuous-server",
        action="store_true",
        help="Allow serverGetPoint to be continuous in [0,1] instead of forcing binary 0/1.",
    )
    args = parser.parse_args()

    validated = []
    for path in args.paths:
        if not os.path.exists(path):
            fail(f"File not found: {path}")
        df = validate_submission(path, require_binary_server=not args.allow_continuous_server)
        validated.append((path, df))
        server_values = df["serverGetPoint"].to_numpy()
        print(
            f"[OK] {os.path.basename(path)} rows={len(df)} "
            f"action_range=[{int(df['actionId'].min())},{int(df['actionId'].max())}] "
            f"point_range=[{int(df['pointId'].min())},{int(df['pointId'].max())}] "
            f"server_range=[{server_values.min():.6f},{server_values.max():.6f}]"
        )

    if args.reference_path:
        expected_rallies, ref_kind = infer_expected_rallies(args.reference_path)
        for path, df in validated:
            compare_rally_order(
                f"{args.reference_path} ({ref_kind})",
                expected_rallies,
                os.path.basename(path),
                df["rally_uid"].tolist(),
            )
        print(f"[OK] reference match against {args.reference_path}")

    base_name, base_df = validated[0]
    base_rallies = base_df["rally_uid"].tolist()
    for path, df in validated[1:]:
        compare_rally_order(
            os.path.basename(base_name),
            base_rallies,
            os.path.basename(path),
            df["rally_uid"].tolist(),
        )
        print(f"[OK] rally order match: {os.path.basename(path)} == {os.path.basename(base_name)}")

    print("[PASS] submission contract validation completed.")


if __name__ == "__main__":
    main()
