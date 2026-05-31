"""R-072 OOF validation — leave-fold-out estimate of rule_override v2 lift.

Strategy:
  1. Load an existing OOF prediction array (use v14_seed2; it's the cleanest
     B-feature baseline in the R-067cr blend).
  2. For each fold:
     - Build R-072 override tables (Layers A/B/C/D) from train rallies that are
       NOT in this fold's validation set (= leave-fold-out rule construction).
     - For each held-out rally, take the OOF prediction's argmax actionId /
       pointId, apply R-072 overrides using its context, and compare against
       the true next-shot label.
  3. Aggregate F1 delta across folds (overridden vs not).

This is a PROXY for the on-test lift (since the actual test predictions come
from a multi-component blend, not v14_seed2 alone), but it's the cleanest
mechanism check available without rebuilding the full blend OOF.

USAGE:
    python -u src/validate_r072_oof.py
"""
from __future__ import annotations

import os
import sys
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, TRAIN_PATH

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
N_ACTION_TRAIN = 15
N_POINT = 10
N_FOLDS = 5
RANDOM_SEED = 42

MIN_CONTEXT_SAMPLES_A = 30
MIN_CONTEXT_SAMPLES_B = 20
MIN_CONTEXT_SAMPLES_C = 25
MIN_CONTEXT_SAMPLES_D = 25

LAYERS = [
    ("A", ["prev_actionId", "last_actionId", "last_pointId"], MIN_CONTEXT_SAMPLES_A),
    ("B", ["prev_prev_actionId", "prev_actionId", "last_actionId", "last_pointId"], MIN_CONTEXT_SAMPLES_B),
    ("C", ["last_handId", "prev_actionId", "last_actionId", "last_pointId"], MIN_CONTEXT_SAMPLES_C),
    ("D", ["last_positionId", "prev_actionId", "last_actionId", "last_pointId"], MIN_CONTEXT_SAMPLES_D),
]

BASELINE_OOF_TAG = "v14_seed2"


def build_override_table(train: pd.DataFrame, context_cols: List[str],
                          target_col: str, min_samples: int) -> Dict[tuple, tuple]:
    df = train.sort_values(["rally_uid", "strikeNumber"]).copy()
    df["prev_actionId"] = df.groupby("rally_uid")["actionId"].shift(1)
    df["prev_prev_actionId"] = df.groupby("rally_uid")["actionId"].shift(2)
    df["last_actionId"] = df["actionId"]
    df["last_pointId"] = df["pointId"]
    df["last_handId"] = df["handId"]
    df["last_positionId"] = df["positionId"]
    df[f"next_{target_col}"] = df.groupby("rally_uid")[target_col].shift(-1)
    df = df.dropna(subset=context_cols + [f"next_{target_col}"])
    for c in context_cols + [f"next_{target_col}"]:
        df[c] = df[c].astype(int)
    table = {}
    for key, vals in df.groupby(context_cols)[f"next_{target_col}"]:
        if len(vals) < min_samples:
            continue
        probs = vals.value_counts(normalize=True).to_dict()
        mode = int(vals.value_counts().idxmax())
        key_tuple = tuple(int(k) for k in (key if isinstance(key, tuple) else (key,)))
        table[key_tuple] = (probs, mode, len(vals))
    return table


def main() -> None:
    print("=" * 80)
    print(" R-072 OOF validation via leave-fold-out rule construction")
    print(f" Baseline OOF: {BASELINE_OOF_TAG}")
    print("=" * 80)

    raw_train = pd.read_csv(TRAIN_PATH)
    print(f" train.csv: {len(raw_train)} rows, {raw_train['rally_uid'].nunique()} rallies")

    # Load OOF arrays for baseline
    oof_act = np.load(f"{OOF_DIR}/{BASELINE_OOF_TAG}_oof_act.npy")
    oof_pt  = np.load(f"{OOF_DIR}/{BASELINE_OOF_TAG}_oof_pt.npy")
    oof_mask = np.load(f"{OOF_DIR}/{BASELINE_OOF_TAG}_oof_mask.npy")
    y_act = np.load(f"{OOF_DIR}/{BASELINE_OOF_TAG}_oof_y_act.npy")
    y_pt  = np.load(f"{OOF_DIR}/{BASELINE_OOF_TAG}_oof_y_pt.npy")
    print(f" OOF arrays: act {oof_act.shape}, pt {oof_pt.shape}, mask {oof_mask.sum()}/{len(oof_mask)} valid")

    # Build per-row context from train (each row is a shot; we need shots that
    # have a NEXT shot, i.e. position 1..N-1 within each rally)
    df = raw_train.sort_values(["rally_uid", "strikeNumber"]).copy()
    df["row_idx"] = np.arange(len(df))
    df["prev_actionId"] = df.groupby("rally_uid")["actionId"].shift(1)
    df["prev_prev_actionId"] = df.groupby("rally_uid")["actionId"].shift(2)
    df["last_actionId"] = df["actionId"]
    df["last_pointId"] = df["pointId"]
    df["last_handId"] = df["handId"]
    df["last_positionId"] = df["positionId"]
    df["next_actionId"] = df.groupby("rally_uid")["actionId"].shift(-1)
    df["next_pointId"]  = df.groupby("rally_uid")["pointId"].shift(-1)

    # Predictions of NEXT shot: we need to align OOF[row_idx_of_position_t]
    # with the predicted distribution for shot t+1. v14_seed2 OOF is one
    # prediction per training row in the train_v14 sample layout. We need to
    # know the row-to-OOF index mapping. v14_seed2 builds samples in
    # ascending (rally_uid, strikeNumber) order, same as train_v14.py's
    # `prepare_data` flow (verified by reading train_v14.py). So OOF index
    # matches row index in sort_values.

    n_rows = len(df)
    if len(oof_act) != n_rows:
        # OOF arrays might be on a per-sample subset (after aug filtering).
        # Take the simplest approach: align by `oof_mask`, which indexes into
        # the same sorted train. If still mismatched, abort with clear msg.
        print(f" WARNING: OOF length {len(oof_act)} != train length {n_rows}")
        if len(oof_act) > n_rows:
            print(" OOF longer than train; truncating (likely aug rows at end).")
            oof_act = oof_act[:n_rows]
            oof_pt = oof_pt[:n_rows]
            oof_mask = oof_mask[:n_rows]
            y_act = y_act[:n_rows]
            y_pt = y_pt[:n_rows]
        else:
            print(f" CANNOT align — OOF shorter than train. Skipping validation.")
            return

    # Fold split by match (same as production training)
    gkf = GroupKFold(n_splits=N_FOLDS)
    rallies = df.drop_duplicates("rally_uid")[["rally_uid", "match"]].reset_index(drop=True)
    fold_of_rally = {}
    for fold, (_, val_idx) in enumerate(gkf.split(rallies["rally_uid"], groups=rallies["match"])):
        for ridx in val_idx:
            fold_of_rally[int(rallies["rally_uid"].iloc[ridx])] = fold
    df["fold"] = df["rally_uid"].map(fold_of_rally)
    print(f" Built {N_FOLDS}-fold split over {rallies['match'].nunique()} matches")

    # We will simulate per-row override decision using OOF predictions
    # Mask: row must have (a) oof_mask=True, (b) next-shot exists, (c) fold assigned
    has_next_action = df["next_actionId"].notna() & df["prev_actionId"].notna()
    valid_row = (oof_mask) & (has_next_action.to_numpy()) & (df["fold"].notna().to_numpy())
    print(f" Valid rows for validation: {int(valid_row.sum())} / {n_rows}")

    # Predictions from OOF
    pred_a_orig = oof_act[:, :N_ACTION_TRAIN].argmax(axis=1)
    pred_p_orig = oof_pt[:, :N_POINT].argmax(axis=1)
    pred_a_v2 = pred_a_orig.copy()
    pred_p_v2 = pred_p_orig.copy()
    pred_a_v2.setflags(write=True)
    pred_p_v2.setflags(write=True)

    touched_a = np.zeros(n_rows, dtype=bool)
    touched_p = np.zeros(n_rows, dtype=bool)
    per_fold_layer_changes = defaultdict(lambda: defaultdict(int))

    # For each fold, build rules from the OTHER 4 folds, apply to this fold's rows
    for held_fold in range(N_FOLDS):
        train_idx_mask = (df["fold"].to_numpy() != held_fold) & df["fold"].notna().to_numpy()
        train_sub = df.loc[train_idx_mask, raw_train.columns.tolist() +
                            ["prev_actionId", "prev_prev_actionId", "last_handId",
                             "last_positionId"]].copy()
        # `raw_train` shape may be needed for build_override_table; pass minimal columns
        train_sub_min = train_sub[["rally_uid", "strikeNumber", "actionId", "pointId",
                                    "handId", "positionId"]]

        # Build tables (per layer × per target)
        tables = {}
        for layer_id, cols, min_n in LAYERS:
            tables[layer_id] = {
                "actionId": build_override_table(train_sub_min, cols, "actionId", min_n),
                "pointId":  build_override_table(train_sub_min, cols, "pointId",  min_n),
                "context_cols": cols,
            }

        # Apply to held fold rows
        fold_mask = (df["fold"].to_numpy() == held_fold) & valid_row
        held_indices = np.where(fold_mask)[0]
        for layer_id, cols, _ in LAYERS:
            act_tbl = tables[layer_id]["actionId"]
            pt_tbl  = tables[layer_id]["pointId"]
            for idx in held_indices:
                row = df.iloc[idx]
                key_vals = []
                ok = True
                for c in cols:
                    val = row[c]
                    if pd.isna(val) or val == -1:
                        ok = False; break
                    key_vals.append(int(val))
                if not ok:
                    continue
                key = tuple(key_vals)
                # Action
                if not touched_a[idx] and key in act_tbl:
                    probs, mode, _ = act_tbl[key]
                    v = int(pred_a_v2[idx])
                    if probs.get(v, 0.0) == 0.0 and mode != v:
                        pred_a_v2[idx] = mode
                        touched_a[idx] = True
                        per_fold_layer_changes[held_fold][f"{layer_id}_action"] += 1
                # Point
                if not touched_p[idx] and key in pt_tbl:
                    probs, mode, _ = pt_tbl[key]
                    v = int(pred_p_v2[idx])
                    if probs.get(v, 0.0) == 0.0 and mode != v:
                        pred_p_v2[idx] = mode
                        touched_p[idx] = True
                        per_fold_layer_changes[held_fold][f"{layer_id}_point"] += 1
        print(f"   Fold {held_fold}: action overrides {touched_a[fold_mask].sum()}, "
              f"point overrides {touched_p[fold_mask].sum()}")

    # Scoring — compare F1 on valid rows
    eval_mask = valid_row
    y_a_clip = np.where(y_act[eval_mask] >= N_ACTION_TRAIN, 0, y_act[eval_mask])
    f1_a_orig = f1_score(y_a_clip, pred_a_orig[eval_mask],
                          labels=list(range(N_ACTION_TRAIN)),
                          average="macro", zero_division=0)
    f1_a_v2   = f1_score(y_a_clip, pred_a_v2[eval_mask],
                          labels=list(range(N_ACTION_TRAIN)),
                          average="macro", zero_division=0)
    f1_p_orig = f1_score(y_pt[eval_mask], pred_p_orig[eval_mask],
                          labels=list(range(N_POINT)),
                          average="macro", zero_division=0)
    f1_p_v2   = f1_score(y_pt[eval_mask], pred_p_v2[eval_mask],
                          labels=list(range(N_POINT)),
                          average="macro", zero_division=0)

    d_f1a = f1_a_v2 - f1_a_orig
    d_f1p = f1_p_v2 - f1_p_orig
    d_ov  = 0.4 * d_f1a + 0.4 * d_f1p   # AUC unchanged (server head not touched)

    print("\n" + "=" * 80)
    print(" OOF VALIDATION RESULTS")
    print("=" * 80)
    print(f"  Total action overrides: {int(touched_a.sum())} / {int(eval_mask.sum())}")
    print(f"  Total point overrides:  {int(touched_p.sum())} / {int(eval_mask.sum())}")
    print()
    print(f"  Baseline ({BASELINE_OOF_TAG}):  F1_a={f1_a_orig:.4f}  F1_p={f1_p_orig:.4f}")
    print(f"  + R-072 v2 override:           F1_a={f1_a_v2:.4f}    F1_p={f1_p_v2:.4f}")
    print(f"  Delta:                          ΔF1_a={d_f1a:+.4f}  ΔF1_p={d_f1p:+.4f}")
    print(f"  Implied OOF ΔOV (0.4*F1a + 0.4*F1p, AUC unchanged): {d_ov:+.4f}")
    print()
    print(" PER-FOLD CHANGES")
    for fold in range(N_FOLDS):
        cnts = per_fold_layer_changes[fold]
        print(f"  Fold {fold}: " + ", ".join(f"{k}={v}" for k, v in sorted(cnts.items())))

    out = {
        "baseline_tag": BASELINE_OOF_TAG,
        "total_action_overrides": int(touched_a.sum()),
        "total_point_overrides":  int(touched_p.sum()),
        "f1_a_orig": float(f1_a_orig), "f1_a_v2": float(f1_a_v2),
        "f1_p_orig": float(f1_p_orig), "f1_p_v2": float(f1_p_v2),
        "delta_F1_a": float(d_f1a), "delta_F1_p": float(d_f1p),
        "implied_oof_delta_OV": float(d_ov),
        "per_fold_layer_changes": {str(f): dict(v) for f, v in per_fold_layer_changes.items()},
    }
    out_path = os.path.join(os.path.dirname(OOF_DIR), "submissions",
                             "r072_oof_validation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n Saved: {out_path}")


if __name__ == "__main__":
    main()
