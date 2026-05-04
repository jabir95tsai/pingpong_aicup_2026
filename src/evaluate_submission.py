"""Evaluate submission CSV files against a truth file in submission format.

Important: the competition ``test.csv`` is prefix data with one or more rows per
``rally_uid``. Even if it contains action/point/server columns, those are the
observed prefix rows, not the hidden next-shot targets, so it must NOT be used
directly as submission ground truth.
"""
import argparse
import os
from glob import glob

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from config import SUBMISSION_DIR


def macro_f1(y_true, y_pred, n_classes):
    return f1_score(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        average="macro",
        zero_division=0,
    )


def evaluate_submission(submission_path, truth_df):
    pred_df = pd.read_csv(submission_path)
    required_cols = ["rally_uid", "actionId", "pointId", "serverGetPoint"]

    missing = [col for col in required_cols if col not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing columns in {submission_path}: {missing}")

    merged = truth_df.merge(
        pred_df[required_cols],
        on="rally_uid",
        how="left",
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )

    if merged[["actionId_pred", "pointId_pred", "serverGetPoint_pred"]].isnull().any().any():
        raise ValueError(f"{submission_path} does not cover all rally_uid values in local test.csv")

    f1_action = macro_f1(merged["actionId_true"], merged["actionId_pred"], 19)
    f1_point = macro_f1(merged["pointId_true"], merged["pointId_pred"], 10)
    auc_server = roc_auc_score(merged["serverGetPoint_true"], merged["serverGetPoint_pred"])
    ov = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc_server

    return {
        "submission": os.path.basename(submission_path),
        "rows": len(merged),
        "f1_action": f1_action,
        "f1_point": f1_point,
        "auc_server": auc_server,
        "ov": ov,
    }


def load_truth_df(truth_path):
    truth_df = pd.read_csv(truth_path)
    required_cols = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    missing = [col for col in required_cols if col not in truth_df.columns]
    if missing:
        raise ValueError(f"Missing columns in truth file {truth_path}: {missing}")

    if truth_df["rally_uid"].duplicated().any():
        dup_count = int(truth_df["rally_uid"].duplicated().sum())
        raise ValueError(
            f"Truth file {truth_path} has duplicate rally_uid values ({dup_count} duplicates). "
            "That usually means you passed raw competition test/prefix rows instead of a one-row-per-rally "
            "next-shot truth file."
        )

    return truth_df[required_cols].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--truth-path",
        required=True,
        help="Path to a one-row-per-rally ground-truth CSV in submission format.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Submission CSV file(s) or glob patterns. Defaults to submissions/*.csv",
    )
    args = parser.parse_args()
    truth_df = load_truth_df(args.truth_path)

    if args.paths:
        expanded_paths = []
        for path in args.paths:
            matches = glob(path)
            if matches:
                expanded_paths.extend(matches)
            elif os.path.exists(path):
                expanded_paths.append(path)
            else:
                raise FileNotFoundError(f"No files matched: {path}")
    else:
        expanded_paths = sorted(glob(os.path.join(SUBMISSION_DIR, "*.csv")))

    if not expanded_paths:
        raise FileNotFoundError("No submission CSV files found to evaluate.")

    results = []
    for path in sorted(set(expanded_paths)):
        results.append(evaluate_submission(path, truth_df))

    result_df = pd.DataFrame(results).sort_values("ov", ascending=False)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
