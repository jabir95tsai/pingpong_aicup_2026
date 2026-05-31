"""Feature engineering V15feat: extends V9 with 36 prefix aggregate features
(R-029a Batch A, clean-room implementation).

V9 baseline + 36 new features per row, all derived from the visible history
prefix of the rally (shots with strikeNumber < next_strikeNumber):

  Per-class frequencies (29 features):
    hist_action_freq_{0..18}  — bincount(actions) / n_prefix
    hist_point_freq_{0..9}    — bincount(points)  / n_prefix

  Distribution shape (4 features):
    hist_action_entropy       — Shannon entropy (base-e) of action distribution
    hist_point_entropy        — Shannon entropy of point distribution
    hist_action_dominance     — max(action_counts) / n_prefix
    hist_point_dominance      — max(point_counts)  / n_prefix

  Tail streaks (3 features):
    streak_action_tail        — count of consecutive identical actionId at end
    streak_point_tail         — count of consecutive identical pointId at end
    consecutive_same_player   — count of consecutive same gamePlayerId at end

Empty-history default: all 36 features = 0.0 (or 0 for streaks).

R-029a clean-room: this module is implemented from the conceptual feature
specification only. The teammate package in
`audits/teammate_table_tennis_2026-05-18/` was not consulted while writing
this code; variable names, function structure, and edge-case handling are
independently chosen.

Fold-safe by construction: each row's features come from the visible
prefix of its own rally only. No cross-rally information, no global
statistics from training data are required. `compute_global_stats_v15feat`
is a pure pass-through to V9.
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v9 import (  # noqa: E402
    build_features_v9,
    compute_global_stats_v9,
    get_feature_names_v9,
)

# Class cardinalities used for bincount sizing
_N_ACT_FULL = 19   # actionId range: 0..18 (includes serve actions 15-18)
_N_PT = 10         # pointId range: 0..9


def compute_global_stats_v15feat(train_df: pd.DataFrame) -> dict:
    """Build the global statistics dict for V15feat.

    V15feat adds zero new global tables — all its features are pure prefix
    aggregates from the per-rally visible history. The returned dict is
    structurally identical to V9's so downstream code can use either
    interchangeably.
    """
    return compute_global_stats_v9(train_df)


def get_feature_names_v15feat(feat_df: pd.DataFrame) -> list:
    """Return feature column names. V15feat adds 36 columns to V9.

    Implemented by reusing V9's name helper — the V15feat columns are
    already present in `feat_df` by the time this is called.
    """
    return get_feature_names_v9(feat_df)




def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy (base e) of a count vector. Zero counts ignored.

    Returns 0.0 if the vector sums to zero (degenerate empty distribution).
    """
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    nz = counts[counts > 0]
    p = nz.astype(np.float64) / total
    return float(-np.sum(p * np.log(p)))


def _tail_run_length(seq: np.ndarray) -> int:
    """Length of the run of identical values at the END of `seq`.

    Empty sequence returns 0. A single-element sequence returns 1.
    Example: [1, 2, 2, 3, 3, 3] → 3.
    """
    L = len(seq)
    if L == 0:
        return 0
    target = seq[-1]
    run = 1
    for j in range(L - 2, -1, -1):
        if seq[j] == target:
            run += 1
        else:
            break
    return run


def build_features_v15feat(
    df: pd.DataFrame,
    is_train: bool,
    global_stats_v9: dict,
    raw_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build V9 features + 36 V15feat prefix-aggregate features.

    Args:
        df: shot-level dataframe (post-`clean_data`).
        is_train: whether this is the training set.
        global_stats_v9: dict returned by `compute_global_stats_v15feat`
            (named `global_stats_v9` to match the v9-family kwarg convention
            used by `train_v14.py`'s build_features_v6 wrapper).
        raw_df: raw shot-level dataframe carrying full rally histories
            (defaults to `df`). Used to look up the prefix history for each
            prediction target.

    Returns:
        feat_df with V9 columns + 36 new columns prefixed `hist_*`, `streak_*`,
        `consecutive_same_player`. Empty histories yield all-zero features.
    """
    # Start from V9. V15feat does not depend on V9 outputs internally — it
    # simply lives alongside V9 columns in the same feature matrix.
    feat_df = build_features_v9(
        df, is_train=is_train,
        global_stats_v9=global_stats_v9,
        raw_df=raw_df,
    )

    if raw_df is None:
        raw_df = df

    # Cache per-rally arrays for prefix lookup. The arrays are kept sorted
    # by strikeNumber so prefix-by-strike comparison is a simple
    # `strike < next_sn` mask.
    rally_cache: dict[int, dict] = {}
    raw_sorted = raw_df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    for rid, grp in raw_sorted.groupby("rally_uid", sort=False):
        rally_cache[int(rid)] = {
            "act":    grp["actionId"].values.astype(np.int32),
            "pt":     grp["pointId"].values.astype(np.int32),
            "player": grp["gamePlayerId"].values.astype(np.int64),
            "strike": grp["strikeNumber"].values.astype(np.int32),
        }

    n_rows = len(feat_df)
    rid_arr = feat_df["rally_uid"].astype(np.int64).values
    nsn_arr = feat_df["next_strikeNumber"].astype(np.int32).values

    # Pre-allocate output buffers
    out_act_freq = np.zeros((n_rows, _N_ACT_FULL), dtype=np.float32)
    out_pt_freq = np.zeros((n_rows, _N_PT), dtype=np.float32)
    out_act_entropy = np.zeros(n_rows, dtype=np.float32)
    out_pt_entropy = np.zeros(n_rows, dtype=np.float32)
    out_act_dominance = np.zeros(n_rows, dtype=np.float32)
    out_pt_dominance = np.zeros(n_rows, dtype=np.float32)
    out_streak_act = np.zeros(n_rows, dtype=np.int32)
    out_streak_pt = np.zeros(n_rows, dtype=np.int32)
    out_player_streak = np.zeros(n_rows, dtype=np.int32)

    for i in range(n_rows):
        rid = int(rid_arr[i])
        next_sn = int(nsn_arr[i])
        cache = rally_cache.get(rid)
        if cache is None:
            # No rally rows present — leave zeros (matches empty-history default)
            continue
        prefix_mask = cache["strike"] < next_sn
        n_prefix = int(prefix_mask.sum())
        if n_prefix == 0:
            continue

        prefix_act = cache["act"][prefix_mask]
        prefix_pt = cache["pt"][prefix_mask]
        prefix_player = cache["player"][prefix_mask]

        # Per-class frequencies (clip to valid range so out-of-range IDs
        # don't blow up bincount; pointId=0 and actionId in 0..18 are the
        # supported domains).
        act_clipped = np.clip(prefix_act, 0, _N_ACT_FULL - 1)
        pt_clipped = np.clip(prefix_pt, 0, _N_PT - 1)
        act_counts = np.bincount(act_clipped, minlength=_N_ACT_FULL)
        pt_counts = np.bincount(pt_clipped, minlength=_N_PT)

        inv_n = 1.0 / float(n_prefix)
        out_act_freq[i] = act_counts.astype(np.float32) * np.float32(inv_n)
        out_pt_freq[i] = pt_counts.astype(np.float32) * np.float32(inv_n)

        out_act_entropy[i] = _shannon_entropy_from_counts(act_counts)
        out_pt_entropy[i] = _shannon_entropy_from_counts(pt_counts)
        out_act_dominance[i] = float(act_counts.max()) * inv_n
        out_pt_dominance[i] = float(pt_counts.max()) * inv_n

        out_streak_act[i] = _tail_run_length(prefix_act)
        out_streak_pt[i] = _tail_run_length(prefix_pt)
        out_player_streak[i] = _tail_run_length(prefix_player)

    # Materialise the 36 new columns. Frequencies first, then summary stats,
    # then streaks — keeps the column order self-explanatory.
    for c in range(_N_ACT_FULL):
        feat_df[f"hist_action_freq_{c}"] = out_act_freq[:, c]
    for c in range(_N_PT):
        feat_df[f"hist_point_freq_{c}"] = out_pt_freq[:, c]
    feat_df["hist_action_entropy"] = out_act_entropy
    feat_df["hist_point_entropy"] = out_pt_entropy
    feat_df["hist_action_dominance"] = out_act_dominance
    feat_df["hist_point_dominance"] = out_pt_dominance
    feat_df["streak_action_tail"] = out_streak_act
    feat_df["streak_point_tail"] = out_streak_pt
    feat_df["consecutive_same_player"] = out_player_streak

    return feat_df


# Convenience: list of column names V15feat adds (used by tests + audits)
V15FEAT_ADDED_COLUMNS = (
    [f"hist_action_freq_{c}" for c in range(_N_ACT_FULL)]
    + [f"hist_point_freq_{c}" for c in range(_N_PT)]
    + [
        "hist_action_entropy",
        "hist_point_entropy",
        "hist_action_dominance",
        "hist_point_dominance",
        "streak_action_tail",
        "streak_point_tail",
        "consecutive_same_player",
    ]
)

assert len(V15FEAT_ADDED_COLUMNS) == 36, (
    f"V15feat column count drift: {len(V15FEAT_ADDED_COLUMNS)} != 36"
)
