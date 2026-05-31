"""build_pseudo_v1 — generate pseudo-label parquet + immutable teacher manifest.

R-009 V1a-capped per Codex APPROVE_WITH_FIXES (2026-05-10):

- Filter: act_top1_p > --act-thr  AND  pt_top1_p > --point-thr
  AND  (pseudo_pointId != 0  if  --drop-point-cls0).
- Greedy per-class cap: sort kept rows by `act_top1_p * pt_top1_p` desc;
  keep rows while enforcing `--per-action-cap` per pseudo_actionId AND
  `--per-point-cap` per pseudo_pointId.
- Sanity row-cap (`--row-cap`) acts as a hard upper bound after per-class
  caps — defaults to 1500 (Codex V1a expected ~274 rows).
- Writes:
    * `<out>` (parquet) — all 1845 rows with `kept` flag + pseudo labels
      + confidence columns + serverGetPoint sentinel −1.
    * `<out>.manifest.json` (immutable teacher manifest) — teacher source
      submission filename, sorted component list, calibration `NONE`,
      exact per-task weights, test_rally_uid sha256, kept_count.

This script is T0 analysis: no training, no model artifacts, no submission.
"""
import argparse
import hashlib
import json
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# zoo_v10 elig2 source: v11_aug + v11plus + v13 + v14_seed2 + v16_avg3 (NONE).
SOURCE_COMPONENTS_SORTED = sorted(
    ["v11_aug", "v11plus", "v13", "v14_seed2", "v16_avg3"])
TEACHER_SUBMISSION = (
    "submission_zoo_v10_elig2_none_v11_aug_v11plus_v13_v14s2_v16_avg3.csv")
TEACHER_CALIBRATION = "NONE"
TEACHER_RANK_IN_ZOO_V2_RANKING = 218  # at the time of R-004

N_ACTION = 19
N_ACTION_EVAL = 15  # train action labels are 0..14
N_POINT = 10


def pad19(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] >= N_ACTION:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], N_ACTION), dtype=np.float32)
    out[:, :arr.shape[1]] = arr
    return out


def load_zoo_weights() -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read per-task blend weights for the zoo_v10 elig2 entry."""
    rank_csv = os.path.join(SUBMISSION_DIR, "zoo_v2_ranking.csv")
    df = pd.read_csv(rank_csv)
    target = "+".join(SOURCE_COMPONENTS_SORTED)
    rows = df[(df["subset"] == target) & (df["calibration"] == "NONE")]
    if len(rows) == 0:
        raise RuntimeError(f"zoo entry not found for subset {target}")
    row = rows.sort_values("rank").iloc[0]
    w_a = np.array([float(x) for x in row["w_a"].split(",")], dtype=np.float64)
    w_p = np.array([float(x) for x in row["w_p"].split(",")], dtype=np.float64)
    w_s = np.array([float(x) for x in row["w_s"].split(",")], dtype=np.float64)
    w_a /= w_a.sum(); w_p /= w_p.sum(); w_s /= w_s.sum()
    meta = {
        "rank_in_ranking_csv": int(row["rank"]),
        "oof_ov": float(row["oof_ov"]),
    }
    return w_a, w_p, w_s, meta


def greedy_per_class_cap(act_p: np.ndarray, pt_p: np.ndarray,
                         pseudo_act: np.ndarray, pseudo_pt: np.ndarray,
                         filter_mask: np.ndarray,
                         per_action_cap: int, per_point_cap: int,
                         row_cap: int) -> np.ndarray:
    """Greedy capped subset: sort by act_p * pt_p desc, keep rows
    that stay under both per-action and per-point caps.

    Returns boolean array of length n with kept-after-cap rows."""
    n = len(act_p)
    keep = np.zeros(n, dtype=bool)
    if not filter_mask.any():
        return keep

    eligible_idx = np.where(filter_mask)[0]
    combined_conf = act_p * pt_p
    order = eligible_idx[np.argsort(-combined_conf[eligible_idx])]

    act_count: dict = {}
    pt_count: dict = {}
    for idx in order:
        a = int(pseudo_act[idx])
        p = int(pseudo_pt[idx])
        if act_count.get(a, 0) >= per_action_cap:
            continue
        if pt_count.get(p, 0) >= per_point_cap:
            continue
        keep[idx] = True
        act_count[a] = act_count.get(a, 0) + 1
        pt_count[p] = pt_count.get(p, 0) + 1
        if int(keep.sum()) >= row_cap:
            break
    return keep


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-thr", type=float, default=0.40,
                    help="Action top-1 probability threshold (default 0.40 per Codex V1a).")
    ap.add_argument("--point-thr", type=float, default=0.25,
                    help="Point top-1 probability threshold (default 0.25 per Codex V1a).")
    ap.add_argument("--drop-point-cls0", action="store_true", default=True,
                    help="Drop rows where pseudo_pointId == 0 (default: True).")
    ap.add_argument("--keep-point-cls0", dest="drop_point_cls0",
                    action="store_false", help="Override: keep cls0 rows.")
    ap.add_argument("--per-action-cap", type=int, default=120,
                    help="Max kept rows per pseudo_actionId class (Codex V1a: 120).")
    ap.add_argument("--per-point-cap", type=int, default=120,
                    help="Max kept rows per pseudo_pointId class (Codex V1a: 120).")
    ap.add_argument("--row-cap", type=int, default=1500,
                    help="Hard upper bound on kept rows (default 1500).")
    ap.add_argument("--out", type=str,
                    default=os.path.join(DATA_DIR, "pseudo_v1.parquet"))
    ap.add_argument("--manifest", type=str, default=None,
                    help="Manifest JSON path (default: <out>.manifest.json).")
    ap.add_argument("--sane-min", type=int, default=200,
                    help="Abort if kept_count < this (V1a sane range 200-350).")
    ap.add_argument("--sane-max", type=int, default=350,
                    help="Abort if kept_count > this.")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"=== build_pseudo_v1 (parameterised, Codex V1a-capped) ===")
    print(f"Filter: act_top1_p > {args.act_thr}, pt_top1_p > {args.point_thr}, "
          f"drop_point_cls0={args.drop_point_cls0}")
    print(f"Per-class caps: per_action={args.per_action_cap}  per_point={args.per_point_cap}")
    print(f"Row cap: {args.row_cap}")
    print(f"Sane kept range: [{args.sane_min}, {args.sane_max}]")
    print(f"Output: {args.out}")
    manifest_path = args.manifest or (args.out + ".manifest.json")
    print(f"Manifest: {manifest_path}")
    print(f"Source components (sorted): {SOURCE_COMPONENTS_SORTED}")

    w_a, w_p, w_s, zoo_meta = load_zoo_weights()
    print(f"\nLoaded teacher (zoo_v10 elig2):")
    print(f"  rank in zoo_v2_ranking.csv: {zoo_meta['rank_in_ranking_csv']}")
    print(f"  oof_ov: {zoo_meta['oof_ov']:.4f}")
    print(f"  w_a = {[f'{w:.3f}' for w in w_a]}")
    print(f"  w_p = {[f'{w:.3f}' for w in w_p]}")
    print(f"  w_s = {[f'{w:.3f}' for w in w_s]} (NOT used; SGP sentinel)")

    print("\nLoading test artifacts...")
    test_acts, test_pts = [], []
    test_uid = None
    for tag in SOURCE_COMPONENTS_SORTED:
        ta = pad19(np.load(f"{OOF_DIR}/{tag}_test_act.npy"))
        tp = np.load(f"{OOF_DIR}/{tag}_test_pt.npy").astype(np.float32, copy=False)
        tu = np.load(f"{OOF_DIR}/{tag}_test_rally_uid.npy")
        if test_uid is None:
            test_uid = tu
        else:
            assert np.array_equal(tu, test_uid), f"test_uid mismatch for {tag}"
        test_acts.append(ta); test_pts.append(tp)
    n_test = len(test_uid)
    print(f"  n_test rallies: {n_test}")

    # Blend
    blend_a = np.zeros((n_test, N_ACTION), dtype=np.float32)
    for w, ta in zip(w_a, test_acts):
        blend_a += w * ta
    blend_p = np.zeros((n_test, N_POINT), dtype=np.float32)
    for w, tp in zip(w_p, test_pts):
        blend_p += w * tp

    pseudo_act = blend_a[:, :N_ACTION_EVAL].argmax(axis=1)
    pseudo_pt = blend_p.argmax(axis=1)
    act_top1_p = blend_a[np.arange(n_test), pseudo_act]
    pt_top1_p = blend_p[np.arange(n_test), pseudo_pt]

    print(f"\nUnfiltered stats:")
    print(f"  act_top1_p: mean={act_top1_p.mean():.3f}  median={np.median(act_top1_p):.3f}")
    print(f"  pt_top1_p:  mean={pt_top1_p.mean():.3f}  median={np.median(pt_top1_p):.3f}")

    # Filter cascade
    mask_act = act_top1_p > args.act_thr
    mask_pt = pt_top1_p > args.point_thr
    mask_no_cls0 = (pseudo_pt != 0) if args.drop_point_cls0 else np.ones(n_test, dtype=bool)
    filter_mask = mask_act & mask_pt & mask_no_cls0
    print(f"\nFilter cascade:")
    print(f"  act>{args.act_thr}: {int(mask_act.sum())}")
    print(f"  pt>{args.point_thr}: {int(mask_pt.sum())}")
    print(f"  pt!=cls0:          {int(mask_no_cls0.sum())}")
    print(f"  combined:          {int(filter_mask.sum())}")

    # Greedy per-class cap
    keep = greedy_per_class_cap(
        act_top1_p, pt_top1_p, pseudo_act, pseudo_pt, filter_mask,
        per_action_cap=args.per_action_cap,
        per_point_cap=args.per_point_cap,
        row_cap=args.row_cap)
    n_kept = int(keep.sum())
    print(f"\nAfter greedy per-class cap: {n_kept}")

    # Sanity gate
    if not (args.sane_min <= n_kept <= args.sane_max):
        print(f"\n*** ABORT: kept_count {n_kept} outside sane range "
              f"[{args.sane_min}, {args.sane_max}]. Refusing to write parquet. ***")
        sys.exit(2)

    # Build dataframe
    raw_test_path = os.path.join(DATA_DIR, "test_new.csv")
    raw_test = pd.read_csv(raw_test_path)
    last_sn_per_rally = raw_test.groupby("rally_uid", sort=False)["strikeNumber"].max()
    next_sn = np.array([int(last_sn_per_rally[r]) + 1 for r in test_uid])

    df = pd.DataFrame({
        "rally_uid": test_uid,
        "next_strikeNumber": next_sn,
        "pseudo_actionId": pseudo_act.astype(np.int32),
        "pseudo_pointId": pseudo_pt.astype(np.int32),
        "act_top1_p": act_top1_p.astype(np.float32),
        "pt_top1_p": pt_top1_p.astype(np.float32),
        "kept": keep,
        "is_pseudo": 1,
        "serverGetPoint": -1,  # sentinel — must be excluded from server model entirely
    })
    df.to_parquet(args.out, index=False)
    print(f"\nSaved parquet: {args.out}")

    # Per-class kept distribution
    kept_df = df[df["kept"]]
    print(f"\nKept-row class distributions (n={len(kept_df)}):")
    print(f"  Action:")
    act_counts = kept_df["pseudo_actionId"].value_counts().sort_index()
    for c in range(N_ACTION_EVAL):
        n_c = int(act_counts.get(c, 0))
        if n_c > 0:
            print(f"    cls{c:2d}: {n_c:4d}  ({100.0*n_c/n_kept:.1f}%)")
    print(f"  Point:")
    pt_counts = kept_df["pseudo_pointId"].value_counts().sort_index()
    for c in range(N_POINT):
        n_c = int(pt_counts.get(c, 0))
        if n_c > 0:
            print(f"    cls{c}: {n_c:4d}  ({100.0*n_c/n_kept:.1f}%)")
    print(f"  Confidence stats (kept rows):")
    print(f"    act_top1_p: min={kept_df['act_top1_p'].min():.3f}  median={kept_df['act_top1_p'].median():.3f}  max={kept_df['act_top1_p'].max():.3f}")
    print(f"    pt_top1_p:  min={kept_df['pt_top1_p'].min():.3f}  median={kept_df['pt_top1_p'].median():.3f}  max={kept_df['pt_top1_p'].max():.3f}")

    # Immutable teacher manifest
    test_uid_str = ",".join(str(r) for r in test_uid)
    test_uid_sha256 = hashlib.sha256(test_uid_str.encode("utf-8")).hexdigest()
    manifest = {
        "format_version": 1,
        "purpose": "R-009 V1a pseudo-label teacher manifest (Codex APPROVE_WITH_FIXES 2026-05-10)",
        "teacher_submission": TEACHER_SUBMISSION,
        "teacher_calibration": TEACHER_CALIBRATION,
        "teacher_rank_in_zoo_v2_ranking_csv_at_time_of_R004": TEACHER_RANK_IN_ZOO_V2_RANKING,
        "teacher_oof_ov_at_build_time": zoo_meta["oof_ov"],
        "teacher_rank_in_csv_at_build_time": zoo_meta["rank_in_ranking_csv"],
        "components_sorted": SOURCE_COMPONENTS_SORTED,
        "weights": {
            "w_a_per_component_in_sorted_order": w_a.tolist(),
            "w_p_per_component_in_sorted_order": w_p.tolist(),
            "w_s_per_component_in_sorted_order_NOT_USED": w_s.tolist(),
        },
        "sgp_policy": "server weights ignored / SGP not used / pseudo rows must be EXCLUDED from server model entirely",
        "n_test_rallies": int(n_test),
        "test_rally_uid_sha256": test_uid_sha256,
        "filter": {
            "act_thr": args.act_thr,
            "point_thr": args.point_thr,
            "drop_point_cls0": bool(args.drop_point_cls0),
        },
        "per_class_caps": {
            "per_action_cap": args.per_action_cap,
            "per_point_cap": args.per_point_cap,
            "row_cap": args.row_cap,
        },
        "kept_count": int(n_kept),
        "kept_action_distribution": {
            int(c): int(act_counts.get(c, 0)) for c in range(N_ACTION_EVAL)
        },
        "kept_point_distribution": {
            int(c): int(pt_counts.get(c, 0)) for c in range(N_POINT)
        },
        "kept_confidence_stats": {
            "act_top1_p_min": float(kept_df["act_top1_p"].min()),
            "act_top1_p_median": float(kept_df["act_top1_p"].median()),
            "act_top1_p_max": float(kept_df["act_top1_p"].max()),
            "pt_top1_p_min": float(kept_df["pt_top1_p"].min()),
            "pt_top1_p_median": float(kept_df["pt_top1_p"].median()),
            "pt_top1_p_max": float(kept_df["pt_top1_p"].max()),
        },
        "trainer_contract": {
            "action_loss": "include pseudo rows with sample_weight = pseudo_weight",
            "point_loss": "include pseudo rows with sample_weight = pseudo_weight",
            "server_loss": "EXCLUDE pseudo rows entirely (do NOT just mask BCE)",
            "flip_aug": "do NOT flip-augment pseudo rows; real-row aug unchanged",
            "oof_artifacts": "OOF arrays (oof_act, oof_pt, oof_srv, oof_mask, oof_y_*, oof_nsn) MUST remain length 69712 (real train rows only)",
            "test_predictions": "produced over the same test_rally_uid order (length 1845)",
        },
        "generated_by": "src/build_pseudo_v1.py",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved immutable teacher manifest: {manifest_path}")
    print(f"  test_rally_uid_sha256: {test_uid_sha256}")
    print(f"  kept_count: {n_kept}")


if __name__ == "__main__":
    main()
