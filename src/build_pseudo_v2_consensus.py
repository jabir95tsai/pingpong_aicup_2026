"""R-065 Stage-0 dry-run generator (no training).

Builds Consensus Pseudo V2 candidate parquet + manifest from 5 teacher OOF/test
outputs. Per Codex BLOCK→APPROVE_WITH_FIXES (2026-05-23): output only; no
trainer launch.

Consensus rules (per Codex sanity-checked thresholds):

  Action:
    - >=4 of 5 teachers agree on top-1 class
    - mean(top-1 prob) across agreeing teachers >= 0.60
    - mean(top-1) - mean(top-2) >= 0.10
    - top-1 class NOT in {15..18} (skip serves)

  Point:
    - >=4 of 5 teachers agree on top-1 class
    - mean(top-1 prob) >= 0.50
    - mean(top-1) - mean(top-2) >= 0.08
    - top-1 class != 0  (skip cls0)

  Server:
    - NEVER pseudo-labelled (V1 weakness)
    - serverGetPoint = -1 sentinel

Per-task masking: a row may pass action consensus but not point consensus
(or vice versa). Parquet schema includes `kept_action`, `kept_point` booleans.

USAGE:
    python -u src/build_pseudo_v2_consensus.py

Outputs:
    data/pseudo_v2.parquet                                   — candidate rows
    data/pseudo_v2.parquet.manifest.json                     — manifest
    submissions/r065_consensus_pool_summary.json             — class distros
"""
import hashlib
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT  # noqa: E402

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 5 candidate teachers (same as audit script)
TEACHERS = [
    "v14_seed2_v15feat_a",
    "v11_aug_oldtest",
    "v16_testhist_aug_oldtest",
    "v13_oldtest",
    "v14_seed2_v16match_v2",
]

# Consensus thresholds (subject to Codex-driven tuning in R-065b)
CONSENSUS_THRESHOLDS = {
    "min_agree_count": 4,           # of 5 teachers
    "action_top1_min": 0.60,
    "action_sep_min": 0.10,         # mean(top1) - mean(top2)
    "action_skip_classes": [15, 16, 17, 18],   # serves
    "point_top1_min": 0.50,
    "point_sep_min": 0.08,
    "point_skip_classes": [0],
}

# Codex-required default (R-065b training will sweep further)
DEFAULT_PSEUDO_WEIGHT = 0.1

# Action class slice for top-1 fairness — slice GBM to first 15 classes so
# transformer (15-class) and GBM (19-class) can vote on the same domain.
N_ACT_FOR_CONSENSUS = 15
N_PT = 10


def load_test(tag: str) -> Dict[str, np.ndarray]:
    out = {}
    for suffix in ["test_act", "test_pt", "test_srv", "test_rally_uid"]:
        path = os.path.join(OOF_DIR, f"{tag}_{suffix}.npy")
        out[suffix] = np.load(path)
    return out


def slice_to_15(act: np.ndarray) -> np.ndarray:
    if act.shape[1] < N_ACT_FOR_CONSENSUS:
        return act.astype(np.float32)
    sliced = act[:, :N_ACT_FOR_CONSENSUS].astype(np.float64)
    s = sliced.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return (sliced / s).astype(np.float32)


def hash_uids(uids: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(uids).tobytes())
    return h.hexdigest()[:16]


def compute_per_row_consensus(probs_by_teacher: Dict[str, np.ndarray],
                              n_classes: int,
                              top1_min: float,
                              sep_min: float,
                              skip_classes: List[int],
                              min_agree: int) -> Dict[str, np.ndarray]:
    """For each row, determine consensus pseudo-label or no-consensus sentinel.

    Returns dict with arrays of length n_rows:
        consensus_class:   int (-1 if no consensus)
        agree_count:       int (# of teachers agreeing on top-1)
        mean_top1_p:       float
        mean_sep:          float (mean top-1 - mean top-2)
        passed:            bool
    """
    n_rows = next(iter(probs_by_teacher.values())).shape[0]
    n_teachers = len(probs_by_teacher)

    # Each row → list of (top1_class, top1_p, top2_p) per teacher
    top1_classes = np.zeros((n_teachers, n_rows), dtype=np.int32)
    top1_probs = np.zeros((n_teachers, n_rows), dtype=np.float32)
    top2_probs = np.zeros((n_teachers, n_rows), dtype=np.float32)

    for ti, (_tag, probs) in enumerate(probs_by_teacher.items()):
        sorted_idx = np.argsort(-probs, axis=1)  # descending
        top1_classes[ti] = sorted_idx[:, 0]
        for r in range(n_rows):
            top1_probs[ti, r] = float(probs[r, sorted_idx[r, 0]])
            top2_probs[ti, r] = float(probs[r, sorted_idx[r, 1]])

    consensus_class = np.full(n_rows, -1, dtype=np.int32)
    agree_count = np.zeros(n_rows, dtype=np.int32)
    mean_top1_p = np.zeros(n_rows, dtype=np.float32)
    mean_sep = np.zeros(n_rows, dtype=np.float32)
    passed = np.zeros(n_rows, dtype=bool)
    skip_set = set(skip_classes)

    for r in range(n_rows):
        # Majority vote
        votes = top1_classes[:, r]
        unique, counts = np.unique(votes, return_counts=True)
        best_idx = int(np.argmax(counts))
        cand_class = int(unique[best_idx])
        cand_count = int(counts[best_idx])
        agree_count[r] = cand_count
        if cand_count < min_agree:
            continue
        if cand_class in skip_set:
            continue
        # Mean top1 / top2 over AGREEING teachers
        mask = (top1_classes[:, r] == cand_class)
        mean_top1 = float(top1_probs[mask, r].mean())
        mean_top2 = float(top2_probs[mask, r].mean())
        sep = mean_top1 - mean_top2
        mean_top1_p[r] = mean_top1
        mean_sep[r] = sep
        if mean_top1 < top1_min:
            continue
        if sep < sep_min:
            continue
        consensus_class[r] = cand_class
        passed[r] = True

    return {
        "consensus_class": consensus_class,
        "agree_count": agree_count,
        "mean_top1_p": mean_top1_p,
        "mean_sep": mean_sep,
        "passed": passed,
    }


def main() -> None:
    print("=" * 78)
    print(" R-065 Stage-0 — Consensus Pseudo V2 candidate generator (DRY RUN)")
    print(" Codex BLOCK→APPROVE_WITH_FIXES 2026-05-23 — no training")
    print("=" * 78)

    teacher_data: Dict[str, Dict[str, np.ndarray]] = {}
    for tag in TEACHERS:
        teacher_data[tag] = load_test(tag)
        print(f"  loaded {tag}: test_act shape={teacher_data[tag]['test_act'].shape}")

    # Reference test UID
    ref_uid = teacher_data[TEACHERS[0]]["test_rally_uid"]
    n_rows = len(ref_uid)
    uid_hash = hash_uids(ref_uid)
    print(f"\n  test rallies: {n_rows}  uid_hash: {uid_hash}")

    # UID alignment check (must be identical across teachers)
    for tag in TEACHERS:
        assert np.array_equal(teacher_data[tag]["test_rally_uid"], ref_uid), (
            f"test_rally_uid mismatch for {tag}"
        )
    print("  test_rally_uid alignment: OK")

    # ─── Action consensus ────────────────────────────────────────────────────
    action_probs = {t: slice_to_15(d["test_act"]) for t, d in teacher_data.items()}
    print(f"\n--- Action consensus (n_classes={N_ACT_FOR_CONSENSUS}, "
          f"min_agree={CONSENSUS_THRESHOLDS['min_agree_count']}/5, "
          f"top1>={CONSENSUS_THRESHOLDS['action_top1_min']}, "
          f"sep>={CONSENSUS_THRESHOLDS['action_sep_min']}, "
          f"skip serves) ---")
    act_consensus = compute_per_row_consensus(
        action_probs,
        n_classes=N_ACT_FOR_CONSENSUS,
        top1_min=CONSENSUS_THRESHOLDS["action_top1_min"],
        sep_min=CONSENSUS_THRESHOLDS["action_sep_min"],
        skip_classes=CONSENSUS_THRESHOLDS["action_skip_classes"],
        min_agree=CONSENSUS_THRESHOLDS["min_agree_count"],
    )

    # ─── Point consensus ─────────────────────────────────────────────────────
    point_probs = {t: d["test_pt"].astype(np.float32) for t, d in teacher_data.items()}
    print(f"\n--- Point consensus (n_classes={N_PT}, "
          f"min_agree={CONSENSUS_THRESHOLDS['min_agree_count']}/5, "
          f"top1>={CONSENSUS_THRESHOLDS['point_top1_min']}, "
          f"sep>={CONSENSUS_THRESHOLDS['point_sep_min']}, "
          f"skip cls0) ---")
    pt_consensus = compute_per_row_consensus(
        point_probs,
        n_classes=N_PT,
        top1_min=CONSENSUS_THRESHOLDS["point_top1_min"],
        sep_min=CONSENSUS_THRESHOLDS["point_sep_min"],
        skip_classes=CONSENSUS_THRESHOLDS["point_skip_classes"],
        min_agree=CONSENSUS_THRESHOLDS["min_agree_count"],
    )

    # ─── Build candidate parquet ─────────────────────────────────────────────
    rows = []
    n_action_kept = int(act_consensus["passed"].sum())
    n_point_kept = int(pt_consensus["passed"].sum())
    n_any_kept = int((act_consensus["passed"] | pt_consensus["passed"]).sum())
    n_both = int((act_consensus["passed"] & pt_consensus["passed"]).sum())

    print(f"\n=== Consensus pool counts ===")
    print(f"  Action kept:   {n_action_kept} / {n_rows}  ({100*n_action_kept/n_rows:.1f}%)")
    print(f"  Point kept:    {n_point_kept} / {n_rows}  ({100*n_point_kept/n_rows:.1f}%)")
    print(f"  Any task kept: {n_any_kept} / {n_rows}  ({100*n_any_kept/n_rows:.1f}%)")
    print(f"  Both tasks:    {n_both} / {n_rows}  ({100*n_both/n_rows:.1f}%)")

    # Codex pool floor: >=100 total task-labels, >=50 per task
    pool_floor_total = (n_action_kept + n_point_kept) >= 100
    pool_floor_each = (n_action_kept >= 50) and (n_point_kept >= 50)
    print(f"\n  Codex pool floor (>=100 total task-labels): "
          f"{n_action_kept + n_point_kept} → {'PASS' if pool_floor_total else 'FAIL'}")
    print(f"  Codex pool floor (>=50 per task):           "
          f"action={n_action_kept}, point={n_point_kept} → "
          f"{'PASS' if pool_floor_each else 'FAIL'}")

    for r in range(n_rows):
        kept_action = bool(act_consensus["passed"][r])
        kept_point = bool(pt_consensus["passed"][r])
        if not (kept_action or kept_point):
            continue
        rows.append({
            "rally_uid": int(ref_uid[r]),
            "pseudo_actionId": int(act_consensus["consensus_class"][r]) if kept_action else -1,
            "pseudo_pointId":  int(pt_consensus["consensus_class"][r])  if kept_point  else -1,
            "act_top1_p":      float(act_consensus["mean_top1_p"][r]) if kept_action else 0.0,
            "act_sep":         float(act_consensus["mean_sep"][r])    if kept_action else 0.0,
            "pt_top1_p":       float(pt_consensus["mean_top1_p"][r])  if kept_point  else 0.0,
            "pt_sep":          float(pt_consensus["mean_sep"][r])     if kept_point  else 0.0,
            "act_agree_count": int(act_consensus["agree_count"][r]),
            "pt_agree_count":  int(pt_consensus["agree_count"][r]),
            "kept_action":     kept_action,
            "kept_point":      kept_point,
            "kept":            kept_action or kept_point,
            "serverGetPoint":  -1,   # sentinel — V2 NEVER pseudo-labels SGP
        })

    # ─── Class distributions on kept rows ───────────────────────────────────
    print("\n=== Class distributions on consensus pool ===")
    if n_action_kept > 0:
        kept_act_classes = [r["pseudo_actionId"] for r in rows if r["kept_action"]]
        act_dist = pd.Series(kept_act_classes).value_counts().sort_index()
        print(f"  Action class distribution (n={n_action_kept}):")
        for c, n in act_dist.items():
            print(f"    cls{int(c):>2}: {n:>3}")
    else:
        print("  Action class distribution: (empty)")

    if n_point_kept > 0:
        kept_pt_classes = [r["pseudo_pointId"] for r in rows if r["kept_point"]]
        pt_dist = pd.Series(kept_pt_classes).value_counts().sort_index()
        print(f"  Point class distribution (n={n_point_kept}):")
        for c, n in pt_dist.items():
            print(f"    cls{int(c):>2}: {n:>3}")
    else:
        print("  Point class distribution: (empty)")

    # ─── Dry-run guard checks ───────────────────────────────────────────────
    print("\n=== Dry-run guard checks (V1 lessons preserved) ===")
    parquet_df = pd.DataFrame(rows)
    if len(parquet_df) > 0:
        sgp_unique = parquet_df["serverGetPoint"].unique().tolist()
        assert sgp_unique == [-1], f"GUARD FAIL: SGP sentinel violated: {sgp_unique}"
        print(f"  serverGetPoint sentinel = -1 only: PASS ({len(parquet_df)} rows)")
        assert (parquet_df["kept_action"] | parquet_df["kept_point"]).all(), \
            "GUARD FAIL: kept=False rows in pool"
        print(f"  All kept rows pass at least one task: PASS")
    else:
        print("  (empty pool — guards trivially pass)")

    # ─── Write outputs ──────────────────────────────────────────────────────
    out_parquet = os.path.join(DATA_DIR, "pseudo_v2.parquet")
    parquet_df.to_parquet(out_parquet, index=False)
    print(f"\n  Wrote: {out_parquet}  ({len(parquet_df)} rows)")

    manifest = {
        "stage": "R-065 Stage-0 candidate parquet",
        "ts": "2026-05-23",
        "teachers": TEACHERS,
        "teacher_count": len(TEACHERS),
        "thresholds": CONSENSUS_THRESHOLDS,
        "test_uid_count": int(n_rows),
        "test_uid_sha256_16": uid_hash,
        "n_action_kept": n_action_kept,
        "n_point_kept": n_point_kept,
        "n_any_kept": n_any_kept,
        "n_both_kept": n_both,
        "pool_floor_total_pass": pool_floor_total,
        "pool_floor_each_pass": pool_floor_each,
        "default_pseudo_weight": DEFAULT_PSEUDO_WEIGHT,
        "n_action_classes_consensus": N_ACT_FOR_CONSENSUS,
        "schema": [
            "rally_uid (int)", "pseudo_actionId (int, -1=skipped)",
            "pseudo_pointId (int, -1=skipped)",
            "act_top1_p (float)", "act_sep (float)",
            "pt_top1_p (float)", "pt_sep (float)",
            "act_agree_count (int)", "pt_agree_count (int)",
            "kept_action (bool)", "kept_point (bool)", "kept (bool)",
            "serverGetPoint (int, always -1 sentinel)",
        ],
    }
    out_manifest = os.path.join(DATA_DIR, "pseudo_v2.parquet.manifest.json")
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote: {out_manifest}")

    summary = {
        **manifest,
        "action_class_distribution": {
            int(c): int(n) for c, n in (
                pd.Series([r["pseudo_actionId"] for r in rows if r["kept_action"]])
                .value_counts().sort_index().items() if n_action_kept > 0 else []
            )
        },
        "point_class_distribution": {
            int(c): int(n) for c, n in (
                pd.Series([r["pseudo_pointId"] for r in rows if r["kept_point"]])
                .value_counts().sort_index().items() if n_point_kept > 0 else []
            )
        },
    }
    out_summary = os.path.join(PROJECT_ROOT, "submissions", "r065_consensus_pool_summary.json")
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote: {out_summary}")

    print("\n=== STAGE 0 DONE — next: open R-065b with these results ===")
    print(f"  Decision: train v14_pseudo_v2 only after Codex reviews R-065b.")


if __name__ == "__main__":
    main()
