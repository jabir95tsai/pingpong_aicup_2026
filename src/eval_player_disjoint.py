"""Evaluate any tag's OOF on the player-disjoint holdout slice (P11).

Reads `data/player_holdout_idx.npy` (built by build_player_disjoint_holdout.py)
and any tag's OOF artifacts (`oof_predictions/{tag}_oof_*.npy`). Computes:
  - full_OV  : OOF OV across all 69,712 rows (matches the standard reported number)
  - holdout_OV: OOF OV restricted to holdout rows (player-disjoint slice)
  - gap      : full_OV − holdout_OV  (negative gap → model overfits to known players)

Used as ADVISORY signal initially (Codex 2026-05-05). Hard gate only after a
leave-one-out / rank-consistency check (zoo_v2 > zoo_v3 > V15) holds.

CLI:
  python src/eval_player_disjoint.py --tags v14_seed0 v16_testhist_aug v11 v11_aug
  python src/eval_player_disjoint.py --tags v14_seed0 --out submissions/p11_eval.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
ACTION_EVAL = list(range(15))
POINT_EVAL  = list(range(10))


def pad_act19(arr):
    if arr.shape[1] >= 19:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], 19), dtype=np.float32)
    out[:, :arr.shape[1]] = arr
    return out


def macro_f1(y, probs, labels):
    return f1_score(y, probs.argmax(axis=1), labels=labels,
                    average="macro", zero_division=0)


def safe_auc(y, probs):
    if len(np.unique(y)) < 2:
        return 0.5
    return roc_auc_score(y, probs)


def eval_tag(tag: str, holdout_mask: np.ndarray) -> dict:
    """Return {full_OV, holdout_OV, full_F1a/p/AUC, holdout_F1a/p/AUC, n_full, n_holdout}.

    Reference y arrays loaded from v14_seed0 if the tag's own y arrays are absent
    (e.g., v11). All component OOFs share the same row order over the 69712 mask.
    """
    act_path = os.path.join(OOF_DIR, f"{tag}_oof_act.npy")
    pt_path  = os.path.join(OOF_DIR, f"{tag}_oof_pt.npy")
    srv_path = os.path.join(OOF_DIR, f"{tag}_oof_srv.npy")

    if not all(os.path.exists(p) for p in (act_path, pt_path, srv_path)):
        return {"tag": tag, "error": "missing OOF artifacts"}

    act = pad_act19(np.load(act_path))
    pt  = np.load(pt_path).astype(np.float32, copy=False)
    srv = np.load(srv_path).astype(np.float32, copy=False)

    # Reference y arrays (use v14_seed0 if tag-specific not present, e.g., v11)
    def _load_y(name, fallback="v14_seed0"):
        p = os.path.join(OOF_DIR, f"{tag}_oof_{name}.npy")
        if os.path.exists(p):
            return np.load(p)
        return np.load(os.path.join(OOF_DIR, f"{fallback}_oof_{name}.npy"))

    y_a = _load_y("y_act")
    y_p = _load_y("y_pt")
    y_s = _load_y("y_srv")

    # Tag's own mask (some tags may have missing rows; we restrict to mask AND holdout)
    mask_path = os.path.join(OOF_DIR, f"{tag}_oof_mask.npy")
    if os.path.exists(mask_path):
        tag_mask = np.load(mask_path).astype(bool)
    else:
        tag_mask = np.ones(len(y_a), dtype=bool)

    if len(holdout_mask) != len(tag_mask):
        return {"tag": tag,
                "error": f"holdout len {len(holdout_mask)} != tag mask len {len(tag_mask)}"}

    full_idx    = tag_mask
    holdout_idx = tag_mask & holdout_mask

    f1_a_full = macro_f1(y_a[full_idx], act[full_idx], ACTION_EVAL)
    f1_p_full = macro_f1(y_p[full_idx], pt[full_idx],  POINT_EVAL)
    auc_full  = safe_auc(y_s[full_idx], srv[full_idx])
    full_OV   = 0.4 * f1_a_full + 0.4 * f1_p_full + 0.2 * auc_full

    f1_a_h = macro_f1(y_a[holdout_idx], act[holdout_idx], ACTION_EVAL)
    f1_p_h = macro_f1(y_p[holdout_idx], pt[holdout_idx],  POINT_EVAL)
    auc_h  = safe_auc(y_s[holdout_idx], srv[holdout_idx])
    holdout_OV = 0.4 * f1_a_h + 0.4 * f1_p_h + 0.2 * auc_h

    return {
        "tag":         tag,
        "n_full":      int(full_idx.sum()),
        "n_holdout":   int(holdout_idx.sum()),
        "full_F1a":    float(f1_a_full),
        "full_F1p":    float(f1_p_full),
        "full_AUC":    float(auc_full),
        "full_OV":     float(full_OV),
        "holdout_F1a": float(f1_a_h),
        "holdout_F1p": float(f1_p_h),
        "holdout_AUC": float(auc_h),
        "holdout_OV":  float(holdout_OV),
        "gap":         float(full_OV - holdout_OV),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="OOF tags to evaluate (e.g. v14_seed0 v16_testhist_aug v11 v11_aug).")
    ap.add_argument("--holdout-mask",
                    default=os.path.join(PROJECT_ROOT, "data", "player_holdout_idx.npy"))
    ap.add_argument("--out", default=None, help="Optional CSV path for results.")
    args = ap.parse_args()

    if not os.path.exists(args.holdout_mask):
        print(f"ERROR: holdout mask not found at {args.holdout_mask}")
        print("Run: python src/build_player_disjoint_holdout.py")
        sys.exit(1)

    holdout_mask = np.load(args.holdout_mask).astype(bool)
    print(f"Loaded holdout mask: {len(holdout_mask)} rows  "
          f"({int(holdout_mask.sum())} held out)")

    rows = []
    for tag in args.tags:
        r = eval_tag(tag, holdout_mask)
        rows.append(r)
        if "error" in r:
            print(f"  {tag}: ERROR — {r['error']}")
            continue
        print(f"  {tag:<22} full_OV={r['full_OV']:.4f}  holdout_OV={r['holdout_OV']:.4f}  "
              f"gap={r['gap']:+.4f}  n_holdout={r['n_holdout']}")

    if args.out:
        df = pd.DataFrame(rows)
        df.to_csv(args.out, index=False)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
