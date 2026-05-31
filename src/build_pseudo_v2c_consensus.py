"""R-065c consensus generator (NO TRAINING).

Cluster-aware consensus pseudo-label generator. Per Codex BLOCK on R-065b:
- GBM cluster collapsed to ONE vote (mean of 4 GBM probabilities).
- No duplicate transformer votes (v11_aug_oldtest_avg3 ≡ v11_aug_oldtest excluded;
  v11plus_oldtest_avg2 ≡ v11plus_oldtest also excluded).
- Deterministic class cap by confidence ranking (no random subsampling).
- Versioned outputs: `pseudo_v2c.parquet`, manifest, summary JSON.

Recommended teacher set (from `src/audit_teacher_pool_v2c.py`):
  gbm_cluster, v11_uncertainty_aug, v11, v11_aug, v11_aug_oldtest, v11plus_oldtest

USAGE:
    python -u src/build_pseudo_v2c_consensus.py
    python -u src/build_pseudo_v2c_consensus.py --action-top1-min 0.50

Outputs:
    data/pseudo_v2c.parquet
    data/pseudo_v2c.parquet.manifest.json
    submissions/r065c_consensus_pool_summary.json
"""
import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT  # noqa: E402

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# GBM cluster — collapsed to ONE teacher (mean of probabilities).
GBM_CLUSTER = [
    "v14_seed2_v15feat_a",
    "v13_oldtest",
    "v16_testhist_aug_oldtest",
    "v14_seed2_v16match_v2",
]
GBM_CLUSTER_TAG = "gbm_cluster"

# Distinct transformers per R-065c audit (all prob corr < 0.85 with gbm_cluster
# and with each other under greedy selection).
TRANSFORMER_TEACHERS = [
    "v11_uncertainty_aug",
    "v11",
    "v11_aug",
    "v11_aug_oldtest",
    "v11plus_oldtest",
]

ALL_TEACHERS = [GBM_CLUSTER_TAG] + TRANSFORMER_TEACHERS  # 6 votes total

N_ACT_FOR_CONSENSUS = 15
N_PT = 10

# Default thresholds — 4-of-6 majority + moderate confidence
DEFAULT_THRESHOLDS = {
    "min_agree_count": 4,            # of 6 (66.7%)
    "action_top1_min": 0.55,
    "action_sep_min":  0.08,
    "action_skip_classes": [15, 16, 17, 18],   # serves
    "point_top1_min":  0.40,
    "point_sep_min":   0.05,
    "point_skip_classes": [0],
    "class_cap_pct":   0.30,         # deterministic cap: max 30% of pool per class
    "pseudo_weight":   0.1,          # Codex fix #5 default
}


def load_test(tag: str) -> Dict[str, np.ndarray]:
    out = {}
    for suffix in ["test_act", "test_pt", "test_srv", "test_rally_uid"]:
        path = os.path.join(OOF_DIR, f"{tag}_{suffix}.npy")
        out[suffix] = np.load(path)
    return out


def collapse_gbm_test(cluster_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    tags = list(cluster_data.keys())
    out = {}
    for suffix in ["test_act", "test_pt", "test_srv"]:
        stack = np.stack([cluster_data[t][suffix].astype(np.float64) for t in tags], axis=0)
        mean = stack.mean(axis=0)
        if suffix in ("test_act", "test_pt"):
            s = mean.sum(axis=1, keepdims=True)
            s = np.where(s == 0, 1.0, s)
            mean = mean / s
        out[suffix] = mean.astype(np.float32)
    out["test_rally_uid"] = cluster_data[tags[0]]["test_rally_uid"]
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


def compute_consensus(
    probs_by_teacher: Dict[str, np.ndarray],
    top1_min: float,
    sep_min: float,
    skip_classes: List[int],
    min_agree: int,
) -> Dict[str, np.ndarray]:
    """Per-row consensus over N teachers. Returns dict of length-n_rows arrays."""
    teachers = list(probs_by_teacher.keys())
    n_teachers = len(teachers)
    n_rows = next(iter(probs_by_teacher.values())).shape[0]

    top1_classes = np.zeros((n_teachers, n_rows), dtype=np.int32)
    top1_probs = np.zeros((n_teachers, n_rows), dtype=np.float32)
    top2_probs = np.zeros((n_teachers, n_rows), dtype=np.float32)

    for ti, t in enumerate(teachers):
        sorted_idx = np.argsort(-probs_by_teacher[t], axis=1)
        top1_classes[ti] = sorted_idx[:, 0]
        for r in range(n_rows):
            top1_probs[ti, r] = float(probs_by_teacher[t][r, sorted_idx[r, 0]])
            top2_probs[ti, r] = float(probs_by_teacher[t][r, sorted_idx[r, 1]])

    consensus_class = np.full(n_rows, -1, dtype=np.int32)
    agree_count = np.zeros(n_rows, dtype=np.int32)
    mean_top1_p = np.zeros(n_rows, dtype=np.float32)
    mean_sep = np.zeros(n_rows, dtype=np.float32)
    passed = np.zeros(n_rows, dtype=bool)
    skip_set = set(skip_classes)

    for r in range(n_rows):
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


def deterministic_class_cap(
    rows: List[Dict],
    class_field: str,
    rank_fields: List[str],
    cap_pct: float,
) -> Tuple[List[Dict], List[Dict]]:
    """Cap any class to <= cap_pct of total pool size. Selection is deterministic:
    sort kept-class rows by `rank_fields` (descending; rally_uid ascending),
    keep top-K.

    Returns (kept_rows, dropped_rows).
    """
    if not rows:
        return [], []
    total = len(rows)
    cap = int(np.floor(cap_pct * total))
    by_class: Dict[int, List[Dict]] = {}
    for r in rows:
        by_class.setdefault(int(r[class_field]), []).append(r)

    kept: List[Dict] = []
    dropped: List[Dict] = []
    for cls, group in by_class.items():
        if len(group) <= cap:
            kept.extend(group)
            continue
        # Sort: rank_fields desc, then rally_uid asc (last is ASCENDING)
        def sort_key(r: Dict) -> Tuple:
            keys = []
            for f in rank_fields[:-1]:
                keys.append(-r[f])  # negative for descending
            keys.append(r[rank_fields[-1]])  # rally_uid ascending
            return tuple(keys)

        group_sorted = sorted(group, key=sort_key)
        kept.extend(group_sorted[:cap])
        dropped.extend(group_sorted[cap:])
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-top1-min", type=float, default=DEFAULT_THRESHOLDS["action_top1_min"])
    ap.add_argument("--action-sep-min",  type=float, default=DEFAULT_THRESHOLDS["action_sep_min"])
    ap.add_argument("--point-top1-min",  type=float, default=DEFAULT_THRESHOLDS["point_top1_min"])
    ap.add_argument("--point-sep-min",   type=float, default=DEFAULT_THRESHOLDS["point_sep_min"])
    ap.add_argument("--min-agree",       type=int,   default=DEFAULT_THRESHOLDS["min_agree_count"])
    ap.add_argument("--class-cap-pct",   type=float, default=DEFAULT_THRESHOLDS["class_cap_pct"])
    ap.add_argument("--pseudo-weight",   type=float, default=DEFAULT_THRESHOLDS["pseudo_weight"])
    args = ap.parse_args()

    thresholds = {
        "min_agree_count": args.min_agree,
        "action_top1_min": args.action_top1_min,
        "action_sep_min":  args.action_sep_min,
        "action_skip_classes": DEFAULT_THRESHOLDS["action_skip_classes"],
        "point_top1_min":  args.point_top1_min,
        "point_sep_min":   args.point_sep_min,
        "point_skip_classes": DEFAULT_THRESHOLDS["point_skip_classes"],
        "class_cap_pct":   args.class_cap_pct,
        "pseudo_weight":   args.pseudo_weight,
    }

    print("=" * 78)
    print(" R-065c Consensus Pseudo V2c — generator (DRY RUN)")
    print("=" * 78)
    print(f" Teachers: {ALL_TEACHERS}")
    print(f" Thresholds: {thresholds}")

    # Load GBM cluster + collapse
    cluster_data: Dict[str, Dict[str, np.ndarray]] = {}
    for tag in GBM_CLUSTER:
        cluster_data[tag] = load_test(tag)
    gbm_collapsed = collapse_gbm_test(cluster_data)

    # Load transformer teachers
    teacher_data: Dict[str, Dict[str, np.ndarray]] = {GBM_CLUSTER_TAG: gbm_collapsed}
    for tag in TRANSFORMER_TEACHERS:
        teacher_data[tag] = load_test(tag)

    # UID alignment check (all teachers must agree)
    ref_uid = teacher_data[GBM_CLUSTER_TAG]["test_rally_uid"]
    for tag in TRANSFORMER_TEACHERS:
        assert np.array_equal(teacher_data[tag]["test_rally_uid"], ref_uid), \
            f"test_rally_uid mismatch for {tag}"
    n_rows = len(ref_uid)
    uid_hash = hash_uids(ref_uid)
    print(f"\n test rallies: {n_rows}  uid_hash: {uid_hash}")

    # Action consensus
    print(f"\n--- Action consensus ({len(ALL_TEACHERS)} teachers, min_agree={thresholds['min_agree_count']}) ---")
    action_probs = {t: slice_to_15(d["test_act"]) for t, d in teacher_data.items()}
    act_consensus = compute_consensus(
        action_probs,
        top1_min=thresholds["action_top1_min"],
        sep_min=thresholds["action_sep_min"],
        skip_classes=thresholds["action_skip_classes"],
        min_agree=thresholds["min_agree_count"],
    )
    n_action_kept_raw = int(act_consensus["passed"].sum())
    print(f"  Action kept (pre-cap): {n_action_kept_raw} / {n_rows} ({100*n_action_kept_raw/n_rows:.1f}%)")

    # Point consensus
    print(f"\n--- Point consensus ({len(ALL_TEACHERS)} teachers, min_agree={thresholds['min_agree_count']}) ---")
    point_probs = {t: d["test_pt"].astype(np.float32) for t, d in teacher_data.items()}
    pt_consensus = compute_consensus(
        point_probs,
        top1_min=thresholds["point_top1_min"],
        sep_min=thresholds["point_sep_min"],
        skip_classes=thresholds["point_skip_classes"],
        min_agree=thresholds["min_agree_count"],
    )
    n_point_kept_raw = int(pt_consensus["passed"].sum())
    print(f"  Point kept (pre-cap): {n_point_kept_raw} / {n_rows} ({100*n_point_kept_raw/n_rows:.1f}%)")

    # ─── Build action subset + deterministic cap ─────────────────────────────
    action_rows = []
    for r in range(n_rows):
        if not act_consensus["passed"][r]:
            continue
        action_rows.append({
            "rally_uid":       int(ref_uid[r]),
            "pseudo_actionId": int(act_consensus["consensus_class"][r]),
            "act_top1_p":      float(act_consensus["mean_top1_p"][r]),
            "act_sep":         float(act_consensus["mean_sep"][r]),
            "act_agree_count": int(act_consensus["agree_count"][r]),
        })
    action_kept, action_dropped = deterministic_class_cap(
        action_rows,
        class_field="pseudo_actionId",
        rank_fields=["act_top1_p", "act_sep", "act_agree_count", "rally_uid"],
        cap_pct=thresholds["class_cap_pct"],
    )
    print(f"\n  Action AFTER cap ({thresholds['class_cap_pct']*100:.0f}%): "
          f"kept={len(action_kept)}, dropped={len(action_dropped)}")

    # Point subset + deterministic cap
    point_rows = []
    for r in range(n_rows):
        if not pt_consensus["passed"][r]:
            continue
        point_rows.append({
            "rally_uid":      int(ref_uid[r]),
            "pseudo_pointId": int(pt_consensus["consensus_class"][r]),
            "pt_top1_p":      float(pt_consensus["mean_top1_p"][r]),
            "pt_sep":         float(pt_consensus["mean_sep"][r]),
            "pt_agree_count": int(pt_consensus["agree_count"][r]),
        })
    point_kept, point_dropped = deterministic_class_cap(
        point_rows,
        class_field="pseudo_pointId",
        rank_fields=["pt_top1_p", "pt_sep", "pt_agree_count", "rally_uid"],
        cap_pct=thresholds["class_cap_pct"],
    )
    print(f"  Point AFTER cap ({thresholds['class_cap_pct']*100:.0f}%): "
          f"kept={len(point_kept)}, dropped={len(point_dropped)}")

    n_action_kept = len(action_kept)
    n_point_kept = len(point_kept)

    # Codex pool floor
    pool_floor_each = (n_action_kept >= 50) and (n_point_kept >= 50)
    pool_floor_total = (n_action_kept + n_point_kept) >= 100
    print(f"\n  Codex pool floor (>=50 per task): "
          f"action={n_action_kept}, point={n_point_kept} → "
          f"{'PASS' if pool_floor_each else 'FAIL'}")
    print(f"  Codex pool floor (>=100 total): "
          f"{n_action_kept + n_point_kept} → "
          f"{'PASS' if pool_floor_total else 'FAIL'}")

    # ─── Class distributions ────────────────────────────────────────────────
    print("\n=== Class distributions on consensus pool (after cap) ===")
    if n_action_kept > 0:
        act_dist = pd.Series([r["pseudo_actionId"] for r in action_kept]).value_counts().sort_index()
        print(f"  Action class distribution (n={n_action_kept}):")
        for c, n in act_dist.items():
            print(f"    cls{int(c):>2}: {n:>3}")
    if n_point_kept > 0:
        pt_dist = pd.Series([r["pseudo_pointId"] for r in point_kept]).value_counts().sort_index()
        print(f"  Point class distribution (n={n_point_kept}):")
        for c, n in pt_dist.items():
            print(f"    cls{int(c):>2}: {n:>3}")

    # ─── Per-task parquet (merge action + point by rally_uid) ───────────────
    action_by_uid = {r["rally_uid"]: r for r in action_kept}
    point_by_uid = {r["rally_uid"]: r for r in point_kept}
    all_uids = sorted(set(list(action_by_uid.keys()) + list(point_by_uid.keys())))
    rows = []
    for uid in all_uids:
        a = action_by_uid.get(uid)
        p = point_by_uid.get(uid)
        rows.append({
            "rally_uid":       uid,
            "pseudo_actionId": a["pseudo_actionId"] if a else -1,
            "pseudo_pointId":  p["pseudo_pointId"] if p else -1,
            "act_top1_p":      a["act_top1_p"] if a else 0.0,
            "act_sep":         a["act_sep"] if a else 0.0,
            "act_agree_count": a["act_agree_count"] if a else 0,
            "pt_top1_p":       p["pt_top1_p"] if p else 0.0,
            "pt_sep":          p["pt_sep"] if p else 0.0,
            "pt_agree_count":  p["pt_agree_count"] if p else 0,
            "kept_action":     a is not None,
            "kept_point":      p is not None,
            "kept":            (a is not None) or (p is not None),
            "serverGetPoint":  -1,
        })
    parquet_df = pd.DataFrame(rows)

    # ─── Dry-run guards ─────────────────────────────────────────────────────
    print("\n=== Guard checks ===")
    if len(parquet_df) > 0:
        assert (parquet_df["serverGetPoint"] == -1).all(), "SGP sentinel violated"
        assert (parquet_df["kept_action"] | parquet_df["kept_point"]).all(), "kept=False row leaked into pool"
        # Determinism check: re-running cap should give same kept rows
        action_kept_2, _ = deterministic_class_cap(
            action_rows,
            class_field="pseudo_actionId",
            rank_fields=["act_top1_p", "act_sep", "act_agree_count", "rally_uid"],
            cap_pct=thresholds["class_cap_pct"],
        )
        assert [r["rally_uid"] for r in action_kept_2] == [r["rally_uid"] for r in action_kept], \
            "Cap is non-deterministic"
        print(f"  serverGetPoint sentinel = -1 only: PASS ({len(parquet_df)} rows)")
        print(f"  All kept rows pass at least one task: PASS")
        print(f"  Cap deterministic on re-run: PASS")
    else:
        print("  (empty pool — guards trivially pass)")

    # ─── Write outputs ──────────────────────────────────────────────────────
    out_parquet = os.path.join(DATA_DIR, "pseudo_v2c.parquet")
    parquet_df.to_parquet(out_parquet, index=False)
    print(f"\n  Wrote: {out_parquet}  ({len(parquet_df)} rows)")

    # Reproducibility hash of kept row IDs
    action_uids_hash = hashlib.sha256(
        json.dumps(sorted([r["rally_uid"] for r in action_kept])).encode()
    ).hexdigest()[:16]
    point_uids_hash = hashlib.sha256(
        json.dumps(sorted([r["rally_uid"] for r in point_kept])).encode()
    ).hexdigest()[:16]

    manifest = {
        "stage": "R-065c candidate parquet",
        "ts": "2026-05-23",
        "teachers": ALL_TEACHERS,
        "teacher_count": len(ALL_TEACHERS),
        "gbm_cluster_members": GBM_CLUSTER,
        "transformer_teachers": TRANSFORMER_TEACHERS,
        "thresholds": thresholds,
        "test_uid_count": int(n_rows),
        "test_uid_sha256_16": uid_hash,
        "n_action_kept_raw": n_action_kept_raw,
        "n_action_kept_after_cap": n_action_kept,
        "n_action_dropped_by_cap": len(action_dropped),
        "n_point_kept_raw": n_point_kept_raw,
        "n_point_kept_after_cap": n_point_kept,
        "n_point_dropped_by_cap": len(point_dropped),
        "pool_floor_total_pass": pool_floor_total,
        "pool_floor_each_pass": pool_floor_each,
        "action_kept_uids_sha256_16": action_uids_hash,
        "point_kept_uids_sha256_16": point_uids_hash,
        "deterministic_cap_rank_fields": ["top1_p desc", "sep desc", "agree_count desc", "rally_uid asc"],
        "default_pseudo_weight": thresholds["pseudo_weight"],
        "schema": [
            "rally_uid (int)", "pseudo_actionId (int, -1 if not kept_action)",
            "pseudo_pointId (int, -1 if not kept_point)",
            "act_top1_p (float)", "act_sep (float)", "act_agree_count (int)",
            "pt_top1_p (float)", "pt_sep (float)", "pt_agree_count (int)",
            "kept_action (bool)", "kept_point (bool)", "kept (bool)",
            "serverGetPoint (int, always -1 sentinel)",
        ],
    }
    out_manifest = os.path.join(DATA_DIR, "pseudo_v2c.parquet.manifest.json")
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote: {out_manifest}")

    summary = {
        **manifest,
        "action_class_distribution": {
            int(c): int(n) for c, n in
            (pd.Series([r["pseudo_actionId"] for r in action_kept])
             .value_counts().sort_index().items() if n_action_kept > 0 else [])
        },
        "point_class_distribution": {
            int(c): int(n) for c, n in
            (pd.Series([r["pseudo_pointId"] for r in point_kept])
             .value_counts().sort_index().items() if n_point_kept > 0 else [])
        },
        "stop_gate_verdict": (
            "viable_train_pseudo" if pool_floor_each
            else "action_only_fallback" if (n_action_kept >= 50 and n_point_kept < 50)
            else "abandon_pseudo_labelling"
        ),
    }
    out_summary = os.path.join(PROJECT_ROOT, "submissions", "r065c_consensus_pool_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote: {out_summary}")

    print("\n=== R-065c STAGE-0 DONE ===")
    print(f"  Verdict: {summary['stop_gate_verdict']}")
    print(f"  Action pool: {n_action_kept} (floor>=50: {'PASS' if n_action_kept >= 50 else 'FAIL'})")
    print(f"  Point pool:  {n_point_kept} (floor>=50: {'PASS' if n_point_kept >= 50 else 'FAIL'})")


if __name__ == "__main__":
    main()
