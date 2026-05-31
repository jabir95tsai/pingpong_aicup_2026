"""Pre-LB-upload sanity check on submission CSVs.

Catches potential issues before burning an LB slot:
  - NaN/Inf values
  - Out-of-range actionId / pointId / serverGetPoint
  - Wrong rally_uid order vs test_new.csv first-appearance
  - Wrong row count
  - File encoding / line-ending issues
  - Duplicate rally_uid rows

USAGE:
    python -u src/pre_lb_sanity_check.py [path1.csv path2.csv ...]
    # No args = check all 3 ARTIFACT_READY candidates
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT

DEFAULT_CSVS = [
    "submissions/submission_R094v2_R067cr_PLUS_SOFTF1_act_only_alpha005_PLUS_RULE.csv",
    "submissions/submission_R094_R067cr_PLUS_SOFTF1_alpha005_PLUS_RULE.csv",
    "submissions/submission_R081v2_R067cr_PLUS_CORRECTOR.csv",
    "submissions/submission_R067cr_alpha030_v22_blend_PLUS_RULE.csv",   # baseline ref
]


def check_one(path: str) -> dict:
    print(f"\n  Checking: {path}")
    report = {"path": path, "passes": [], "fails": [], "warnings": []}

    if not os.path.exists(path):
        report["fails"].append(f"MISSING FILE")
        return report

    # File-level checks
    raw = open(path, "rb").read()
    # BOM check
    if raw.startswith(b"\xef\xbb\xbf"):
        report["fails"].append("UTF-8 BOM present (rule violation)")
    else:
        report["passes"].append("No BOM ok")
    # Line ending check — must be LF
    crlf_count = raw.count(b"\r\n")
    if crlf_count > 0:
        report["fails"].append(f"CRLF line endings detected ({crlf_count})")
    else:
        report["passes"].append("LF line endings ok")

    # Parse with pandas
    try:
        df = pd.read_csv(path)
    except Exception as e:
        report["fails"].append(f"CSV parse error: {e}")
        return report

    # Column check
    expected_cols = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    if list(df.columns) != expected_cols:
        report["fails"].append(f"Column mismatch: got {list(df.columns)} expected {expected_cols}")
        return report
    report["passes"].append("Column order ok")

    # Row count vs test_new
    test_new = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test_new.csv"))
    unique_uids = test_new["rally_uid"].drop_duplicates().to_numpy()
    if len(df) != len(unique_uids):
        report["fails"].append(f"Row count {len(df)} != unique rally count {len(unique_uids)}")
    else:
        report["passes"].append(f"Row count = {len(df)} matches unique rally count")

    # rally_uid order
    if not np.array_equal(df["rally_uid"].to_numpy(), unique_uids):
        report["fails"].append("rally_uid order does NOT match test_new.csv first-appearance order")
    else:
        report["passes"].append("rally_uid order matches test_new.csv")

    # Duplicate rally_uid
    if df["rally_uid"].duplicated().any():
        report["fails"].append(f"Duplicate rally_uid: {df['rally_uid'].duplicated().sum()}")
    else:
        report["passes"].append("No duplicate rally_uid")

    # actionId range [0, 18] (full 19-class)
    a = df["actionId"].to_numpy()
    if np.isnan(a).any():
        report["fails"].append("actionId contains NaN")
    elif (a < 0).any() or (a > 18).any():
        report["fails"].append(f"actionId out of range: min={a.min()} max={a.max()}")
    else:
        report["passes"].append(f"actionId range [{a.min()}, {a.max()}] in [0, 18]")

    # pointId range [0, 9]
    p = df["pointId"].to_numpy()
    if np.isnan(p).any():
        report["fails"].append("pointId contains NaN")
    elif (p < 0).any() or (p > 9).any():
        report["fails"].append(f"pointId out of range: min={p.min()} max={p.max()}")
    else:
        report["passes"].append(f"pointId range [{p.min()}, {p.max()}] in [0, 9]")

    # serverGetPoint range [0, 1] (float prob)
    s = df["serverGetPoint"].to_numpy()
    if np.isnan(s).any() or np.isinf(s).any():
        n_bad = int(np.isnan(s).sum() + np.isinf(s).sum())
        report["fails"].append(f"serverGetPoint contains {n_bad} NaN/Inf")
    elif (s < 0).any() or (s > 1).any():
        report["fails"].append(f"serverGetPoint out of range: min={s.min():.4f} max={s.max():.4f}")
    else:
        report["passes"].append(f"serverGetPoint range [{s.min():.4f}, {s.max():.4f}] in [0, 1]")

    # Action distribution sanity (shouldn't be all one class)
    from collections import Counter
    cnt_a = Counter(a.tolist())
    most_common_a, most_common_a_count = cnt_a.most_common(1)[0]
    if most_common_a_count / len(df) > 0.5:
        report["warnings"].append(f"Most-common action {most_common_a} = {most_common_a_count}/{len(df)} "
                                    f"= {100*most_common_a_count/len(df):.1f}% (suspicious)")
    else:
        report["passes"].append(f"Action distribution: most-common act{most_common_a} = "
                                  f"{100*most_common_a_count/len(df):.1f}%")

    cnt_p = Counter(p.tolist())
    most_common_p, most_common_p_count = cnt_p.most_common(1)[0]
    if most_common_p_count / len(df) > 0.5:
        report["warnings"].append(f"Most-common point {most_common_p} = "
                                    f"{100*most_common_p_count/len(df):.1f}% (suspicious)")

    # SGP mean sanity (should be near 0.5 for balanced)
    sgp_mean = s.mean()
    if sgp_mean < 0.2 or sgp_mean > 0.8:
        report["warnings"].append(f"SGP mean = {sgp_mean:.4f} (outside [0.2, 0.8] — unusual)")
    else:
        report["passes"].append(f"SGP mean = {sgp_mean:.4f}")

    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", help="CSV paths to check (default: 3 ARTIFACT_READY)")
    args = p.parse_args()
    paths = args.paths if args.paths else DEFAULT_CSVS

    print("=" * 80)
    print(" Pre-LB-upload sanity check")
    print("=" * 80)

    reports = []
    for path in paths:
        r = check_one(path)
        reports.append(r)

    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    for r in reports:
        name = os.path.basename(r["path"])
        n_pass = len(r["passes"])
        n_fail = len(r["fails"])
        n_warn = len(r["warnings"])
        status = "PASS" if n_fail == 0 else "FAIL"
        print(f"\n  {status} {name}")
        print(f"    passes: {n_pass}")
        if n_warn > 0:
            print(f"    warnings: {n_warn}")
            for w in r["warnings"]:
                print(f"      WARN: {w}")
        if n_fail > 0:
            print(f"    failures: {n_fail}")
            for f in r["fails"]:
                print(f"      FAIL: {f}")

    any_fail = any(len(r["fails"]) > 0 for r in reports)
    print()
    if any_fail:
        print(" Some CSVs FAILED sanity check. DO NOT UPLOAD failing files to LB.")
        sys.exit(1)
    else:
        print(" All CSVs PASS sanity check. Safe for LB upload (subject to your judgment).")


if __name__ == "__main__":
    main()
