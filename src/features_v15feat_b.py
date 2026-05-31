"""Feature engineering V15feat_b: extends V15feat (Batch A) with 33
empirical transition prior features (R-029b Batch B, clean-room).

V15feat (Batch A) baseline + 33 new features per row, derived from
training-fold-only empirical conditional distributions:

  Action priors (19 features):
    trans_action_prior_{0..18}  — P(next_action | last_action, next_is_serve_side)
      where:
        - last_action = actionId of the most recent visible history shot
        - next_is_serve_side = 1 if predicted shot is on serve side (odd
          strikeNumber), 0 if receive side (even strikeNumber)
        - Probability is estimated from training-fold transition counts
        - Falls back to marginal P(next_action) when context unseen

  Point priors (10 features):
    trans_point_prior_{0..9}    — P(next_point | last_action, last_point)
      where:
        - last_point = pointId of the most recent visible history shot
        - Falls back to marginal P(next_point) when context unseen

  Summary statistics (4 features):
    trans_action_entropy        — Shannon entropy of action prior
    trans_point_entropy         — Shannon entropy of point prior
    trans_action_top1           — max prior probability for action
    trans_point_top1            — max prior probability for point

R-029b clean-room: implemented from the conceptual specification of empirical
conditional priors. The teammate package in `audits/teammate_table_tennis_2026-05-18/`
was not consulted while writing this code. Function names, data structures,
and edge-case handling are independently chosen.

Fold-safe by construction: `compute_global_stats_v15feat_b` is called per
training fold with that fold's training rows only. Tables built from val
or test rows would constitute leakage — guarded by the train_v14 call
pattern (`compute_global_stats(train_df_fold)` per fold).
"""
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v15feat import (  # noqa: E402
    build_features_v15feat,
    compute_global_stats_v15feat,
    get_feature_names_v15feat,
)

_N_ACT_FULL = 19   # actionId domain 0..18
_N_PT = 10         # pointId domain 0..9


def _build_action_prior_table(
    train_df: pd.DataFrame,
) -> Tuple[dict, np.ndarray]:
    """Empirical P(next_action | last_action, next_is_serve_side) from train.

    Returns:
        (table, marginal) where
          - table: dict mapping (last_action, next_is_serve_side) → ndarray(19,)
          - marginal: ndarray(19,) — global P(next_action) for fallback
    """
    # Sort and compute next_actionId per rally (shift -1)
    df = train_df.sort_values(["rally_uid", "strikeNumber"]).copy()
    df["_next_action"] = df.groupby("rally_uid")["actionId"].shift(-1)
    df["_next_strike"] = df.groupby("rally_uid")["strikeNumber"].shift(-1)

    valid = df.dropna(subset=["_next_action", "_next_strike"])
    last_action = valid["actionId"].astype(int).values
    next_action = valid["_next_action"].astype(int).values
    next_sn = valid["_next_strike"].astype(int).values
    next_is_serve = (next_sn % 2 == 1).astype(int)

    # Global marginal first
    marginal_counts = np.bincount(
        np.clip(next_action, 0, _N_ACT_FULL - 1),
        minlength=_N_ACT_FULL,
    ).astype(np.float64)
    total = marginal_counts.sum()
    marginal = (marginal_counts / total).astype(np.float32) if total > 0 else \
               np.full(_N_ACT_FULL, 1.0 / _N_ACT_FULL, dtype=np.float32)

    # Per-(last_action, next_is_serve) table
    table: dict = {}
    composite_key = last_action.astype(np.int64) * 4 + next_is_serve.astype(np.int64)
    next_action_clip = np.clip(next_action, 0, _N_ACT_FULL - 1)

    unique_keys, inverse = np.unique(composite_key, return_inverse=True)
    for idx, key in enumerate(unique_keys):
        mask = inverse == idx
        counts = np.bincount(
            next_action_clip[mask], minlength=_N_ACT_FULL,
        ).astype(np.float64)
        n = counts.sum()
        if n <= 0:
            continue
        la = int(key // 4)
        sv = int(key % 4)
        table[(la, sv)] = (counts / n).astype(np.float32)
    return table, marginal


def _build_point_prior_table(
    train_df: pd.DataFrame,
) -> Tuple[dict, np.ndarray]:
    """Empirical P(next_point | last_action, last_point) from train.

    Returns:
        (table, marginal) where
          - table: dict mapping (last_action, last_point) → ndarray(10,)
          - marginal: ndarray(10,) — global P(next_point) for fallback
    """
    df = train_df.sort_values(["rally_uid", "strikeNumber"]).copy()
    df["_next_point"] = df.groupby("rally_uid")["pointId"].shift(-1)

    valid = df.dropna(subset=["_next_point"])
    last_action = valid["actionId"].astype(int).values
    last_point = valid["pointId"].astype(int).values
    next_point = valid["_next_point"].astype(int).values

    marginal_counts = np.bincount(
        np.clip(next_point, 0, _N_PT - 1),
        minlength=_N_PT,
    ).astype(np.float64)
    total = marginal_counts.sum()
    marginal = (marginal_counts / total).astype(np.float32) if total > 0 else \
               np.full(_N_PT, 1.0 / _N_PT, dtype=np.float32)

    table: dict = {}
    last_action_clip = np.clip(last_action, 0, _N_ACT_FULL - 1)
    last_point_clip = np.clip(last_point, 0, _N_PT - 1)
    composite_key = last_action_clip.astype(np.int64) * 16 + last_point_clip.astype(np.int64)
    next_point_clip = np.clip(next_point, 0, _N_PT - 1)

    unique_keys, inverse = np.unique(composite_key, return_inverse=True)
    for idx, key in enumerate(unique_keys):
        mask = inverse == idx
        counts = np.bincount(
            next_point_clip[mask], minlength=_N_PT,
        ).astype(np.float64)
        n = counts.sum()
        if n <= 0:
            continue
        la = int(key // 16)
        lp = int(key % 16)
        table[(la, lp)] = (counts / n).astype(np.float32)
    return table, marginal


def compute_global_stats_v15feat_b(train_df: pd.DataFrame) -> dict:
    """Extend V15feat global stats with R-029b transition prior tables.

    Called once per training fold by `train_v14.py`. The transition tables
    are built from this fold's training rows only — guaranteeing
    fold-safe priors (val and test rows are never observed at table
    construction time).
    """
    stats = compute_global_stats_v15feat(train_df)

    action_table, action_marginal = _build_action_prior_table(train_df)
    point_table, point_marginal = _build_point_prior_table(train_df)

    stats["v15feat_b_action_table"] = action_table
    stats["v15feat_b_action_marginal"] = action_marginal
    stats["v15feat_b_point_table"] = point_table
    stats["v15feat_b_point_marginal"] = point_marginal
    return stats


def get_feature_names_v15feat_b(feat_df: pd.DataFrame) -> list:
    """V15feat_b adds 33 columns on top of V15feat (Batch A)."""
    return get_feature_names_v15feat(feat_df)


def _shannon_entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy (base e) per row of a (N, K) probability matrix."""
    safe = np.where(probs > 0, probs, 1.0)  # log(1) = 0 contributes nothing
    return -np.sum(probs * np.log(safe), axis=1).astype(np.float32)


def build_features_v15feat_b(
    df: pd.DataFrame,
    is_train: bool,
    global_stats_v9: dict,
    raw_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build V15feat (Batch A) features + 33 R-029b transition prior features.

    Args:
        df: shot-level dataframe (post-`clean_data`)
        is_train: whether this is the training set
        global_stats_v9: dict returned by `compute_global_stats_v15feat_b`
            (kwarg named `global_stats_v9` to match v9-family wrapper convention)
        raw_df: raw shot-level dataframe (defaults to df)
    """
    global_stats_v15feat_b = global_stats_v9  # local alias for readability
    # Start from V15feat (Batch A)
    feat_df = build_features_v15feat(
        df, is_train=is_train,
        global_stats_v9=global_stats_v15feat_b,
        raw_df=raw_df,
    )

    # Look up tables built per fold
    action_table = global_stats_v15feat_b["v15feat_b_action_table"]
    action_marginal = global_stats_v15feat_b["v15feat_b_action_marginal"]
    point_table = global_stats_v15feat_b["v15feat_b_point_table"]
    point_marginal = global_stats_v15feat_b["v15feat_b_point_marginal"]

    # Pull last shot's actionId / pointId via raw_df lookup (mirror v9.py pattern).
    # V9/V7 columns expose `last_action_category` but not raw `last_actionId`,
    # so we recompute from raw_df: for each feat_df row, the LAST visible
    # shot's strike number = next_strikeNumber - 1.
    next_sn_arr = feat_df["next_strikeNumber"].values.astype(np.int32)
    next_is_serve = (next_sn_arr % 2 == 1).astype(np.int32)

    shot_lookup = raw_df[["rally_uid", "strikeNumber", "actionId", "pointId"]].copy()
    shot_lookup["strikeNumber"] = shot_lookup["strikeNumber"].astype(int)
    merge_left = pd.DataFrame({
        "rally_uid": feat_df["rally_uid"].values,
        "strikeNumber": next_sn_arr - 1,  # the most recent visible shot
    })
    merged = merge_left.merge(shot_lookup, on=["rally_uid", "strikeNumber"], how="left")
    # Empty history (next_sn_arr=1) → sentinel -1; clipped to marginal in lookup loop
    last_action_arr = merged["actionId"].fillna(-1).astype(int).values.astype(np.int32)
    last_point_arr = merged["pointId"].fillna(-1).astype(int).values.astype(np.int32)

    n_rows = len(feat_df)
    out_action = np.empty((n_rows, _N_ACT_FULL), dtype=np.float32)
    out_point = np.empty((n_rows, _N_PT), dtype=np.float32)

    # Vectorized lookup via per-row dict access. With ~70k rows this is
    # still fast (<5s) because the dict has at most 19*4 = 76 action keys
    # and 19*10 = 190 point keys.
    for i in range(n_rows):
        la = int(last_action_arr[i])
        lp = int(last_point_arr[i])
        srv = int(next_is_serve[i])
        # Clip out-of-domain to safe values
        if la < 0 or la >= _N_ACT_FULL:
            out_action[i] = action_marginal
        else:
            out_action[i] = action_table.get((la, srv), action_marginal)

        if la < 0 or la >= _N_ACT_FULL or lp < 0 or lp >= _N_PT:
            out_point[i] = point_marginal
        else:
            out_point[i] = point_table.get((la, lp), point_marginal)

    # Write 29 prior columns
    for c in range(_N_ACT_FULL):
        feat_df[f"trans_action_prior_{c}"] = out_action[:, c]
    for c in range(_N_PT):
        feat_df[f"trans_point_prior_{c}"] = out_point[:, c]

    # Summary stats (4 columns)
    feat_df["trans_action_entropy"] = _shannon_entropy_from_probs(out_action)
    feat_df["trans_point_entropy"] = _shannon_entropy_from_probs(out_point)
    feat_df["trans_action_top1"] = out_action.max(axis=1).astype(np.float32)
    feat_df["trans_point_top1"] = out_point.max(axis=1).astype(np.float32)

    return feat_df


# Convenience: column names added by V15feat_b on top of V15feat
V15FEAT_B_ADDED_COLUMNS = (
    [f"trans_action_prior_{c}" for c in range(_N_ACT_FULL)]
    + [f"trans_point_prior_{c}" for c in range(_N_PT)]
    + [
        "trans_action_entropy",
        "trans_point_entropy",
        "trans_action_top1",
        "trans_point_top1",
    ]
)

assert len(V15FEAT_B_ADDED_COLUMNS) == 33, (
    f"V15feat_b column count drift: {len(V15FEAT_B_ADDED_COLUMNS)} != 33"
)
