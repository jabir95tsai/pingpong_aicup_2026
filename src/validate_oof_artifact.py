"""Validate a single OOF + test artifact set for blender intake eligibility.

Used by the deadline orchestrator after each successful training job, per the
user's 2026-05-18 directive: "After each completed component, validate
OOF/test artifact shape, test_rally_uid alignment, and whether it is
eligible for blender intake."

Checks performed:
  1. EXISTS  — all required `.npy` files present in oof_predictions/
  2. SHAPE   — OOF rows in {69712, 72065} (standard / oldtest+2353);
                test rows = 1845; act columns in {15, 19}; pt columns = 10
  3. FINITE  — no NaN or Inf anywhere in probabilities
  4. UID     — test_rally_uid byte-equal to reference v11_aug
  5. ALIGN   — for oldtest variants, first N_REF y_act/y_pt/y_srv rows
                byte-equal to reference (slice [:N_REF] alignment, per
                LESSONS rule and `analyze_oldtest_blend.load_components`)
  6. MASK    — `oof_mask.sum() == N_REF` (where present)
  7. PROBS   — action / point probabilities sum to ~1 per row (softmax sanity)
  8. SGP RANGE — server probabilities in [0, 1]

Exit codes:
  0 — ELIGIBLE for blender intake
  1 — FAIL  (one or more hard checks violated)
  2 — WARN  (passes hard checks but has anomalies worth noting)

Usage:
    python -u src/validate_oof_artifact.py --tag v13_oldtest_seed31337
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
REF_TAG = "v11_aug"

# Standard OOF length (canonical visible-prefix sample count from train.csv)
N_REF_STANDARD = 69712
# Oldtest OOF length (standard + 2353 prediction-eligible old test rows)
N_REF_OLDTEST = 72065

REQUIRED_FILES = [
    "oof_act",
    "oof_pt",
    "oof_srv",
    "test_act",
    "test_pt",
    "test_srv",
    "test_rally_uid",
]
OPTIONAL_FILES = [
    "oof_y_act",
    "oof_y_pt",
    "oof_y_srv",
    "oof_mask",
    "oof_nsn",
    "oof_pt_bin",
]


class ValidationReport:
    def __init__(self, tag: str):
        self.tag = tag
        self.checks: List[dict] = []
        self.fail_count = 0
        self.warn_count = 0

    def add(self, name: str, status: str, detail: str = "") -> None:
        """status in {'PASS', 'WARN', 'FAIL'}."""
        self.checks.append({"name": name, "status": status, "detail": detail})
        if status == "FAIL":
            self.fail_count += 1
        elif status == "WARN":
            self.warn_count += 1

    def verdict(self) -> str:
        if self.fail_count > 0:
            return "INELIGIBLE"
        if self.warn_count > 0:
            return "ELIGIBLE_WITH_WARNINGS"
        return "ELIGIBLE"

    def exit_code(self) -> int:
        if self.fail_count > 0:
            return 1
        if self.warn_count > 0:
            return 2
        return 0

    def render(self) -> str:
        lines = [
            f"=== Validation report for tag={self.tag} ===",
        ]
        for c in self.checks:
            marker = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[c["status"]]
            lines.append(f"  {marker} {c['name']}{(': ' + c['detail']) if c['detail'] else ''}")
        lines.append(f"  VERDICT: {self.verdict()}  (failures={self.fail_count}, warnings={self.warn_count})")
        return "\n".join(lines)


def _path(tag: str, suffix: str) -> str:
    return os.path.join(OOF_DIR, f"{tag}_{suffix}.npy")


def _safe_load(path: str):
    try:
        return np.load(path)
    except Exception as exc:  # noqa: BLE001
        return exc


def validate(tag: str) -> ValidationReport:
    report = ValidationReport(tag)

    # ---- 1. EXISTS ----
    missing = [s for s in REQUIRED_FILES if not os.path.exists(_path(tag, s))]
    if missing:
        report.add("EXISTS_REQUIRED", "FAIL",
                   f"missing required suffixes: {missing}")
        return report  # bail early; subsequent checks would crash
    report.add("EXISTS_REQUIRED", "PASS",
               f"{len(REQUIRED_FILES)} required files present")

    optional_present = [s for s in OPTIONAL_FILES
                        if os.path.exists(_path(tag, s))]
    report.add("EXISTS_OPTIONAL", "PASS",
               f"optional files present: {optional_present}")

    # ---- 2. SHAPE (OOF) ----
    oof_act = np.load(_path(tag, "oof_act"))
    oof_pt = np.load(_path(tag, "oof_pt"))
    oof_srv = np.load(_path(tag, "oof_srv"))

    n_oof = oof_act.shape[0]
    if n_oof == N_REF_STANDARD:
        oof_kind = "standard"
    elif n_oof == N_REF_OLDTEST:
        oof_kind = "oldtest (+2353)"
    else:
        report.add("SHAPE_OOF", "FAIL",
                   f"oof_act has {n_oof} rows; expected "
                   f"{N_REF_STANDARD} or {N_REF_OLDTEST}")
        return report
    report.add("SHAPE_OOF_ROWS", "PASS",
               f"oof rows = {n_oof} ({oof_kind})")

    if oof_pt.shape[0] != n_oof:
        report.add("SHAPE_OOF_PT", "FAIL",
                   f"oof_pt rows = {oof_pt.shape[0]} != oof_act rows = {n_oof}")
    if oof_srv.shape[0] != n_oof:
        report.add("SHAPE_OOF_SRV", "FAIL",
                   f"oof_srv rows = {oof_srv.shape[0]} != oof_act rows = {n_oof}")
    n_act_cols = oof_act.shape[1] if oof_act.ndim == 2 else 1
    n_pt_cols = oof_pt.shape[1] if oof_pt.ndim == 2 else 1
    if n_act_cols not in (15, 19):
        report.add("SHAPE_OOF_ACT_COLS", "FAIL",
                   f"oof_act has {n_act_cols} cols; expected 15 or 19")
    else:
        report.add("SHAPE_OOF_ACT_COLS", "PASS",
                   f"oof_act cols = {n_act_cols}")
    if n_pt_cols != 10:
        report.add("SHAPE_OOF_PT_COLS", "FAIL",
                   f"oof_pt has {n_pt_cols} cols; expected 10")
    else:
        report.add("SHAPE_OOF_PT_COLS", "PASS",
                   f"oof_pt cols = {n_pt_cols}")

    # ---- 3. SHAPE (test) ----
    test_act = np.load(_path(tag, "test_act"))
    test_pt = np.load(_path(tag, "test_pt"))
    test_srv = np.load(_path(tag, "test_srv"))
    test_uid = np.load(_path(tag, "test_rally_uid"))

    if test_act.shape[0] != 1845:
        report.add("SHAPE_TEST_ROWS", "FAIL",
                   f"test_act has {test_act.shape[0]} rows; expected 1845")
        return report
    for nm, arr in (("test_pt", test_pt), ("test_srv", test_srv),
                     ("test_rally_uid", test_uid)):
        if arr.shape[0] != 1845:
            report.add(f"SHAPE_{nm.upper()}_ROWS", "FAIL",
                       f"{nm} has {arr.shape[0]} rows; expected 1845")
    report.add("SHAPE_TEST", "PASS", "test arrays = 1845 rows each")

    # ---- 4. FINITE ----
    for nm, arr in (("oof_act", oof_act), ("oof_pt", oof_pt),
                     ("oof_srv", oof_srv), ("test_act", test_act),
                     ("test_pt", test_pt), ("test_srv", test_srv)):
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            report.add(f"FINITE_{nm.upper()}", "FAIL",
                       f"{n_bad} non-finite values in {nm}")
        else:
            report.add(f"FINITE_{nm.upper()}", "PASS",
                       f"all values finite in {nm}")

    # ---- 5. UID ALIGNMENT vs reference ----
    ref_uid_path = _path(REF_TAG, "test_rally_uid")
    if not os.path.exists(ref_uid_path):
        report.add("UID_ALIGNMENT", "WARN",
                   f"reference {REF_TAG}_test_rally_uid not found; skipping check")
    else:
        ref_uid = np.load(ref_uid_path)
        if not np.array_equal(test_uid, ref_uid):
            report.add("UID_ALIGNMENT", "FAIL",
                       f"test_rally_uid differs from reference {REF_TAG}")
        else:
            report.add("UID_ALIGNMENT", "PASS",
                       f"test_rally_uid byte-equal to {REF_TAG} reference")

    # ---- 6. Y-LABEL ALIGNMENT (for oldtest variants, slice [:N_REF] alignment) ----
    ref_y = {}
    for suf in ("oof_y_act", "oof_y_pt", "oof_y_srv"):
        rp = _path(REF_TAG, suf)
        if os.path.exists(rp):
            ref_y[suf] = np.load(rp)
    if ref_y:
        n_ref = len(ref_y["oof_y_act"]) if "oof_y_act" in ref_y else N_REF_STANDARD
        for suf, ref_arr in ref_y.items():
            tag_path = _path(tag, suf)
            if not os.path.exists(tag_path):
                continue
            tag_arr = np.load(tag_path)
            # Slice if oldtest (72065 -> 69712 first N_REF)
            if tag_arr.shape[0] != ref_arr.shape[0]:
                if tag_arr.shape[0] < ref_arr.shape[0]:
                    report.add(f"Y_ALIGN_{suf.upper()}", "FAIL",
                               f"{tag}_{suf} has fewer rows ({tag_arr.shape[0]}) "
                               f"than reference ({ref_arr.shape[0]})")
                    continue
                head = tag_arr[: ref_arr.shape[0]]
            else:
                head = tag_arr
            if np.array_equal(head, ref_arr):
                report.add(f"Y_ALIGN_{suf.upper()}", "PASS",
                           f"head[:{ref_arr.shape[0]}] matches {REF_TAG}")
            else:
                report.add(f"Y_ALIGN_{suf.upper()}", "FAIL",
                           f"head[:{ref_arr.shape[0]}] differs from {REF_TAG} — "
                           f"row order assumption broken")

    # ---- 7. MASK ----
    mask_path = _path(tag, "oof_mask")
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        ref_y_a_path = _path(REF_TAG, "oof_y_act")
        expected_n = (np.load(ref_y_a_path).shape[0]
                      if os.path.exists(ref_y_a_path)
                      else N_REF_STANDARD)
        if int(mask.sum()) != expected_n:
            # For oldtest variants the mask might cover the longer set
            if int(mask.sum()) == oof_act.shape[0]:
                report.add("OOF_MASK_SUM", "PASS",
                           f"mask.sum()={int(mask.sum())} matches oof rows "
                           f"({oof_act.shape[0]})")
            else:
                report.add("OOF_MASK_SUM", "WARN",
                           f"mask.sum()={int(mask.sum())} != reference "
                           f"({expected_n}) nor oof rows ({oof_act.shape[0]})")
        else:
            report.add("OOF_MASK_SUM", "PASS",
                       f"mask.sum()={expected_n} matches reference")

    # ---- 8. PROB SUM-TO-ONE (action / point) ----
    sums_act = oof_act.sum(axis=1)
    if not np.allclose(sums_act, 1.0, atol=1e-3):
        bad = int((np.abs(sums_act - 1.0) > 1e-3).sum())
        report.add("OOF_ACT_PROB_SUM", "WARN",
                   f"{bad}/{len(sums_act)} OOF action rows have sum != 1±0.001 "
                   f"(min={sums_act.min():.4f}, max={sums_act.max():.4f})")
    else:
        report.add("OOF_ACT_PROB_SUM", "PASS", "all rows sum to 1±0.001")

    sums_pt = oof_pt.sum(axis=1)
    if not np.allclose(sums_pt, 1.0, atol=1e-3):
        bad = int((np.abs(sums_pt - 1.0) > 1e-3).sum())
        report.add("OOF_PT_PROB_SUM", "WARN",
                   f"{bad}/{len(sums_pt)} OOF point rows have sum != 1±0.001 "
                   f"(min={sums_pt.min():.4f}, max={sums_pt.max():.4f})")
    else:
        report.add("OOF_PT_PROB_SUM", "PASS", "all rows sum to 1±0.001")

    # ---- 9. SGP RANGE ----
    sgp = oof_srv if oof_srv.ndim == 1 else oof_srv[:, 0]
    if (sgp < 0.0).any() or (sgp > 1.0).any():
        n_bad = int(((sgp < 0.0) | (sgp > 1.0)).sum())
        report.add("OOF_SRV_RANGE", "FAIL",
                   f"{n_bad} OOF SGP values outside [0, 1] "
                   f"(min={sgp.min():.4f}, max={sgp.max():.4f})")
    else:
        report.add("OOF_SRV_RANGE", "PASS",
                   f"OOF SGP in [0, 1] (min={sgp.min():.4f}, max={sgp.max():.4f})")

    test_sgp = test_srv if test_srv.ndim == 1 else test_srv[:, 0]
    if (test_sgp < 0.0).any() or (test_sgp > 1.0).any():
        n_bad = int(((test_sgp < 0.0) | (test_sgp > 1.0)).sum())
        report.add("TEST_SRV_RANGE", "FAIL",
                   f"{n_bad} test SGP values outside [0, 1]")
    else:
        report.add("TEST_SRV_RANGE", "PASS",
                   f"test SGP in [0, 1] (min={test_sgp.min():.4f}, max={test_sgp.max():.4f})")

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True,
                        help="OOF tag to validate (e.g. v13_oldtest_seed31337)")
    parser.add_argument("--log-file", default=None,
                        help="If set, also append a one-line summary to this file.")
    args = parser.parse_args()

    report = validate(args.tag)
    print(report.render())

    if args.log_file:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (f"[{ts}] {args.tag} verdict={report.verdict()} "
                f"failures={report.fail_count} warnings={report.warn_count}")
        with open(args.log_file, "a") as f:
            f.write(line + "\n")

    sys.exit(report.exit_code())


if __name__ == "__main__":
    main()
