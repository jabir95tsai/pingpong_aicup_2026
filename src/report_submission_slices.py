"""Detailed slice report for a submission against local benchmark truth.

Reports:
- overall OV / task metrics
- per next-strike-number slice metrics
- per-class F1 for actionId / pointId
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def macro_f1(y_true, y_pred, n_classes):
    return f1_score(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        average="macro",
        zero_division=0,
    )


def next_sn_group(sn):
    if sn == 1:
        return "SN=1"
    if sn == 2:
        return "SN=2"
    if sn <= 4:
        return "SN=3-4"
    if sn <= 8:
        return "SN=5-8"
    if sn <= 12:
        return "SN=9-12"
    return "SN=13+"


def evaluate_block(df):
    f1_action = macro_f1(df["actionId_true"], df["actionId_pred"], 19)
    f1_point = macro_f1(df["pointId_true"], df["pointId_pred"], 10)
    if df["serverGetPoint_true"].nunique() < 2:
        auc_server = 0.5
    else:
        auc_server = roc_auc_score(df["serverGetPoint_true"], df["serverGetPoint_pred"])
        if np.isnan(auc_server):
            auc_server = 0.5
    ov = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc_server
    return {
        "rows": len(df),
        "f1_action": f1_action,
        "f1_point": f1_point,
        "auc_server": auc_server,
        "ov": ov,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-path", required=True, help="Benchmark prefix test.csv path")
    parser.add_argument("--truth-path", required=True, help="Benchmark truth.csv path")
    parser.add_argument("submission_path", help="Submission CSV path")
    args = parser.parse_args()

    prefix_df = pd.read_csv(args.prefix_path)
    truth_df = pd.read_csv(args.truth_path)
    pred_df = pd.read_csv(args.submission_path)

    prefix_last = (
        prefix_df.sort_values(["rally_uid", "strikeNumber"])
        .groupby("rally_uid", sort=False)
        .tail(1)[["rally_uid", "strikeNumber"]]
        .copy()
    )
    prefix_last["next_strikeNumber"] = prefix_last["strikeNumber"] + 1
    prefix_last["sn_group"] = prefix_last["next_strikeNumber"].apply(next_sn_group)

    merged = truth_df.merge(pred_df, on="rally_uid", suffixes=("_true", "_pred"), validate="one_to_one")
    merged = merged.merge(prefix_last[["rally_uid", "next_strikeNumber", "sn_group"]], on="rally_uid", validate="one_to_one")

    overall = evaluate_block(merged)
    print("[overall]")
    print(pd.DataFrame([overall]).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    rows = []
    order = ["SN=1", "SN=2", "SN=3-4", "SN=5-8", "SN=9-12", "SN=13+"]
    for group in order:
        sub = merged[merged["sn_group"] == group]
        if len(sub) == 0:
            continue
        metrics = evaluate_block(sub)
        metrics["sn_group"] = group
        rows.append(metrics)

    print("\n[by sn_group]")
    if rows:
        report_df = pd.DataFrame(rows)[["sn_group", "rows", "f1_action", "f1_point", "auc_server", "ov"]]
        print(report_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    action_f1 = f1_score(
        merged["actionId_true"],
        merged["actionId_pred"],
        labels=list(range(19)),
        average=None,
        zero_division=0,
    )
    point_f1 = f1_score(
        merged["pointId_true"],
        merged["pointId_pred"],
        labels=list(range(10)),
        average=None,
        zero_division=0,
    )

    print("\n[action per-class f1]")
    for idx, val in enumerate(action_f1):
        print(f"{idx:2d} {val:.6f}")

    print("\n[point per-class f1]")
    for idx, val in enumerate(point_f1):
        print(f"{idx:2d} {val:.6f}")


if __name__ == "__main__":
    main()
