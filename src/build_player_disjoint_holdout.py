"""Build a player-disjoint holdout mask over the 69,712 OOF rows (P11).

Goal: produce a boolean mask `data/player_holdout_idx.npy` of length 69,712 such
that mask[i]==True means OOF row i belongs to a rally whose primary player is in
the held-out player set. Used by `eval_player_disjoint.py` to compute a
"player-disjoint OOF OV" alongside standard match-OOF — proxy for LB transfer
since LB has only 63.5% player overlap with train.

Codex sign-off (2026-05-05): advisory signal initially. First gate is
leave-one-out / rank-consistency (zoo_v2 > zoo_v3 > V15 on holdout).

Row-order convention:
  V14/V16/V12 OOF arrays follow the row order produced by build_features_v9
  on train_df with is_train=True. Internally this iterates train_df rallies
  via groupby("rally_uid", sort=False) and emits one row per (rally, target
  strikeNumber) for target ∈ [2..n_shots]. We replicate that order here
  WITHOUT calling features_v9 (to avoid the 5–10 min joint-priors build).

Output:
  data/player_holdout_idx.npy  - boolean mask, length 69712
  data/player_holdout_meta.txt - human-readable summary

CLI:
  python src/build_player_disjoint_holdout.py [--seed 42] [--frac 0.15]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, PROJECT_ROOT
from data_cleaning import clean_data

EXPECTED_OOF_ROWS = 69712


def build_row_to_rally(train_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the OOF row order: one row per (rally_uid, target_shot) for
    target_shot in [2..n_shots] of each rally, iterated in groupby('rally_uid',
    sort=False) order.

    Returns a DataFrame with columns: rally_uid, target_sn, primary_player.
    """
    rows = []
    for rally_uid, grp in train_df.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber")
        n = len(grp)
        if n < 2:
            continue  # build_features_v9 also skips these
        primary = int(grp["gamePlayerId"].iloc[0])
        opponent = int(grp["gamePlayerOtherId"].iloc[0])
        # One OOF row per target_sn ∈ [2..n] (predicting shot k from k-1 shots history)
        for target_sn in range(2, n + 1):
            rows.append({
                "rally_uid": rally_uid,
                "target_sn": target_sn,
                "primary_player": primary,
                "opponent_player": opponent,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for player sampling.")
    ap.add_argument("--frac", type=float, default=0.15,
                    help="Fraction of distinct primary players to hold out.")
    ap.add_argument("--out-mask",
                    default=os.path.join(PROJECT_ROOT, "data", "player_holdout_idx.npy"))
    ap.add_argument("--out-meta",
                    default=os.path.join(PROJECT_ROOT, "data", "player_holdout_meta.txt"))
    args = ap.parse_args()

    print("=" * 70)
    print("P11 player-disjoint holdout builder")
    print(f"  seed={args.seed}  frac={args.frac}")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(TRAIN_PATH)  # placeholder; clean_data needs both
    # We don't actually need test for the holdout, so reuse train_df from clean_data.
    train_df, _, player_map = clean_data(raw_train, raw_test)
    print(f"  train_df rows: {len(train_df)}  unique players: {len(player_map)}")

    print("\n--- Reproducing OOF row order ---")
    row_df = build_row_to_rally(train_df)
    n_rows = len(row_df)
    print(f"  Generated {n_rows} OOF row mappings")
    if n_rows != EXPECTED_OOF_ROWS:
        raise AssertionError(
            f"GUARD FAIL: expected {EXPECTED_OOF_ROWS} OOF rows, got {n_rows}. "
            "Row-order replication does not match V14/V16/V12 build_features_v9.")

    # Sanity: cross-check against an existing OOF y array (matches by length only).
    y_a_path = os.path.join(PROJECT_ROOT, "oof_predictions", "v14_seed0_oof_y_act.npy")
    if os.path.exists(y_a_path):
        y_a = np.load(y_a_path)
        if len(y_a) != n_rows:
            raise AssertionError(
                f"GUARD FAIL: row count {n_rows} != v14_seed0 y_act length {len(y_a)}.")

    # Distinct primary players
    primary_players = row_df["primary_player"].values
    distinct_players = np.unique(primary_players)
    n_distinct = len(distinct_players)
    print(f"\n  Distinct primary players in OOF rows: {n_distinct}")

    # Sample holdout players
    rng = np.random.default_rng(args.seed)
    n_holdout = max(1, int(round(args.frac * n_distinct)))
    holdout_players = set(rng.choice(distinct_players, size=n_holdout, replace=False))
    print(f"  Held-out players: {n_holdout} / {n_distinct} ({100*n_holdout/n_distinct:.1f}%)")

    # Mask: True iff rally's primary player is in holdout set
    mask = np.array([p in holdout_players for p in primary_players], dtype=bool)
    n_holdout_rows = int(mask.sum())
    print(f"  Holdout rows: {n_holdout_rows} / {n_rows} ({100*n_holdout_rows/n_rows:.1f}%)")

    # Class diversity sanity (use v14_seed0's y_pt if available)
    y_pt_path = os.path.join(PROJECT_ROOT, "oof_predictions", "v14_seed0_oof_y_pt.npy")
    if os.path.exists(y_pt_path):
        y_pt = np.load(y_pt_path)
        print(f"\n  Holdout pointId distribution (counts):")
        for c in range(10):
            n_c = int((y_pt[mask] == c).sum())
            print(f"    cls {c}: {n_c}")

    # Save
    os.makedirs(os.path.dirname(args.out_mask), exist_ok=True)
    np.save(args.out_mask, mask)
    print(f"\n  Saved mask: {args.out_mask}  shape={mask.shape}  sum={n_holdout_rows}")

    with open(args.out_meta, "w", encoding="utf-8") as f:
        f.write(f"P11 player-disjoint holdout meta\n")
        f.write(f"seed={args.seed}  frac={args.frac}\n")
        f.write(f"distinct_players={n_distinct}  held_out={n_holdout}\n")
        f.write(f"holdout_rows={n_holdout_rows}/{n_rows}\n")
        f.write(f"holdout_player_ids={sorted(holdout_players)}\n")
    print(f"  Saved meta: {args.out_meta}")


if __name__ == "__main__":
    main()
