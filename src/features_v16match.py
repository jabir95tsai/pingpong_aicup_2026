"""R-032 — Within-match cross-rally context features (LORO).

Status: AWAITING_CODEX REVIEW (2026-05-21). DO NOT TRAIN UNTIL APPROVED.

Builds 40 LORO (leave-one-rally-out) features per training/test sample that
capture player style + match tactical signature from the OTHER rallies in
the same match. Attacks the de-identified-player structural problem by
using cross-rally observations that standard within-rally features ignore.

Feature families (40 total):
  Family A (33): match-level action/point distribution, entropy, dominance
  Family B (5):  target-player-specific hand/strength/action stats in match
  Family C (2):  match-other-count, match-other-avg-rally-length

LEAK SAFETY:
  - Target rally R's own data is EXCLUDED from R's features (LORO).
  - Only PREFIX shots of other rallies used (not their target shots).
  - Train/test matches are disjoint -> GroupKFold by match -> no cross-fold leak.
  - All shots from "other rallies" come from before-prediction visibility,
    matching test-time conditions exactly.

GATING (per REVIEW_QUEUE R-032):
  - Smoke Fold 1 OV >= v14_seed2 Fold-1 OV + 0.003
  - Sample-size mismatch audit (E) must show train can simulate test conditions
  - Counts-only diagnostic (F) must NOT lift OV by itself

PERFORMANCE:
  - Naive O(N^2) per match. Train has ~17000 rallies * ~80 other = ~1.36M
    aggregations per pass. ~1 min per fold. Acceptable.

USAGE (from train_v14):
  --feature-set v16match

INTERFACE matches features_v15feat.py / features_v9.py for plug-in compat:
  compute_global_stats_v16match(train_df) -> stats_dict
  build_features_v16match(df, is_train, global_stats_v9, raw_df) -> feature_df
  get_feature_names_v16match(feat_df) -> list[str]
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from features_v15feat import (  # noqa: E402
    build_features_v15feat,
    compute_global_stats_v15feat,
    get_feature_names_v15feat,
)


# ---- Constants ----------------------------------------------------------

N_ACTION_RAW = 19   # action classes 0..18 (full range, not eval-15)
N_POINT = 10        # point classes 0..9
N_HAND = 3          # 0=none, 1=forehand, 2=backhand

# Maximum number of OTHER rallies to aggregate over. Train has ~80
# others/match; test has ~22. Clipping train to ~22 reduces match-size
# distribution shift between train and test. Codex Q1 in REVIEW_QUEUE.
DEFAULT_MAX_OTHER_RALLIES = 22

# Minimum number of OTHER rallies before we trust the features.
# Below this, fall back to zeros (model conditions on match_other_count
# feature anyway). Codex Q3 in REVIEW_QUEUE.
DEFAULT_MIN_OTHER_RALLIES = 3


# ---- Banned-name audit (mirrors v15feat regimen) -----------------------

V16MATCH_ADDED_COLUMNS = (
    # Family A: 19 action freqs + 10 point freqs + 2 entropy + 2 dominance = 33
    [f"match_other_action_freq_{c}" for c in range(N_ACTION_RAW)]
    + [f"match_other_point_freq_{c}" for c in range(N_POINT)]
    + [
        "match_other_action_entropy", "match_other_point_entropy",
        "match_other_action_dominance", "match_other_point_dominance",
    ]
    # Family B: 3 hand freqs + 1 strength mean + 1 action entropy = 5
    + [f"target_player_hand_freq_in_match_{h}" for h in range(N_HAND)]
    + [
        "target_player_strength_mean_in_match",
        "target_player_action_entropy_in_match",
    ]
    # Family C: 2 (NOT including match_avg_rally_length - banned)
    + [
        "match_other_count_log1p",
        "match_other_avg_rally_length",
    ]
)
assert len(V16MATCH_ADDED_COLUMNS) == 40, (
    f"Expected 40 added columns, got {len(V16MATCH_ADDED_COLUMNS)}")
assert len(set(V16MATCH_ADDED_COLUMNS)) == 40, "Duplicate column names"


def audit_no_banned_names_v16match(cols: list) -> None:
    """Ensure no banned feature names slip through.

    Banned per REVIEW_QUEUE R-032 §3:
    - Anything aggregating the target rally R itself
    - Anything using TARGET shots of other rallies
    - match_other_avg_serverGetPoint (train/test asymmetry)
    - match_avg_rally_length (rally-end-info leak)
    """
    banned = (
        "match_other_avg_serverGetPoint",
        "match_avg_rally_length",
        "match_other_final_action",
        "match_other_final_point",
        "match_other_terminal_action",
    )
    hits = [c for c in cols if c in banned]
    if hits:
        raise ValueError(f"Banned R-032 feature names present: {hits}")


# ---- Shannon entropy helper (vectorized) -------------------------------

def _shannon_entropy_freq(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector. Returns 0 for all-zero."""
    total = float(probs.sum())
    if total <= 0:
        return 0.0
    p = probs / total
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


# ---- LORO aggregation core ---------------------------------------------

def _aggregate_match_prefix(
    match_df: pd.DataFrame,
    max_other_rallies: int = DEFAULT_MAX_OTHER_RALLIES,
    rng: np.random.Generator = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """For each rally in match_df, compute aggregates from the OTHER rallies'
    PREFIX shots.

    Returns: {rally_uid: {action_counts, point_counts, hand_counts_by_player,
                          strength_sum_by_player, action_counts_by_player,
                          n_other_rallies, total_other_shots, avg_rally_len}}

    LORO logic: features for rally R are computed from match_df's rallies
    EXCLUDING R. To stay close to test conditions, only PREFIX shots of
    those other rallies are used. We approximate prefix as all shots except
    the last one of each rally (representing the visible history of any
    test-time rally aggregation step).
    """
    if rng is None:
        rng = np.random.default_rng(20260520)

    rally_groups = list(match_df.groupby("rally_uid"))
    # Pre-compute each rally's prefix-shots (all but last) and full-stats
    rally_prefix = {}
    for rally_uid, gdf in rally_groups:
        g = gdf.sort_values("strikeNumber").reset_index(drop=True)
        if len(g) <= 1:
            rally_prefix[int(rally_uid)] = pd.DataFrame()
        else:
            rally_prefix[int(rally_uid)] = g.iloc[:-1]

    # Per-match action/point counts (across all prefixes of all rallies)
    # This lets us subtract rally R's contribution for fast LORO.
    all_action = np.zeros(N_ACTION_RAW, dtype=np.int64)
    all_point = np.zeros(N_POINT, dtype=np.int64)
    rally_action_counts = {}  # rally_uid -> ndarray(N_ACTION_RAW)
    rally_point_counts = {}
    rally_n_shots = {}        # number of prefix shots in this rally
    for rally_uid, p in rally_prefix.items():
        if len(p) == 0:
            rally_action_counts[rally_uid] = np.zeros(N_ACTION_RAW, dtype=np.int64)
            rally_point_counts[rally_uid] = np.zeros(N_POINT, dtype=np.int64)
            rally_n_shots[rally_uid] = 0
        else:
            ac = np.bincount(p["actionId"].astype(int).clip(0, N_ACTION_RAW - 1).values,
                             minlength=N_ACTION_RAW)
            pc = np.bincount(p["pointId"].astype(int).clip(0, N_POINT - 1).values,
                             minlength=N_POINT)
            rally_action_counts[rally_uid] = ac
            rally_point_counts[rally_uid] = pc
            rally_n_shots[rally_uid] = len(p)
            all_action += ac
            all_point += pc

    # LORO subtraction
    out: Dict[int, Dict[str, np.ndarray]] = {}
    all_rally_uids = list(rally_prefix.keys())
    for rally_uid in all_rally_uids:
        other_action = all_action - rally_action_counts[rally_uid]
        other_point = all_point - rally_point_counts[rally_uid]
        n_other_rallies = len(all_rally_uids) - 1
        other_n_shots = sum(
            rally_n_shots[uid] for uid in all_rally_uids if uid != rally_uid
        )

        # If too many other rallies, subsample to match test conditions.
        # Subsampling done by re-aggregating from a random subset.
        if max_other_rallies > 0 and n_other_rallies > max_other_rallies:
            other_uids = [uid for uid in all_rally_uids if uid != rally_uid]
            chosen = rng.choice(other_uids, size=max_other_rallies, replace=False)
            other_action = np.zeros(N_ACTION_RAW, dtype=np.int64)
            other_point = np.zeros(N_POINT, dtype=np.int64)
            other_n_shots = 0
            for uid in chosen:
                other_action += rally_action_counts[uid]
                other_point += rally_point_counts[uid]
                other_n_shots += rally_n_shots[uid]
            n_other_rallies = max_other_rallies

        avg_rally_len = (other_n_shots / max(n_other_rallies, 1)
                         if n_other_rallies > 0 else 0.0)

        out[int(rally_uid)] = {
            "action_counts": other_action,
            "point_counts": other_point,
            "n_other_rallies": n_other_rallies,
            "total_other_shots": other_n_shots,
            "avg_rally_len": float(avg_rally_len),
        }
    return out


def _aggregate_player_in_match(
    match_df: pd.DataFrame,
    target_player_id: int,
    target_rally_uid: int,
) -> Dict[str, float]:
    """For a specific player in this match, aggregate hand/strength/action
    stats from their shots in OTHER rallies (excluding target_rally_uid)."""
    other = match_df[match_df["rally_uid"] != target_rally_uid]
    if len(other) == 0:
        return {
            "hand_counts": np.zeros(N_HAND, dtype=np.int64),
            "strength_mean": 0.0,
            "action_entropy": 0.0,
        }
    # First filter to PREFIX shots of each rally (drop the RALLY's last shot,
    # not the player's last shot) — matching test-time visibility.
    other = other.copy()
    other["_rally_last_sn"] = other.groupby("rally_uid")["strikeNumber"].transform("max")
    prefix_other = other[other["strikeNumber"] < other["_rally_last_sn"]]
    # Then filter prefix shots to those owned by target_player_id
    prefix_shots = prefix_other[prefix_other["gamePlayerId"] == target_player_id]
    if len(prefix_shots) == 0:
        return {
            "hand_counts": np.zeros(N_HAND, dtype=np.int64),
            "strength_mean": 0.0,
            "action_entropy": 0.0,
        }
    hand_counts = np.bincount(
        prefix_shots["handId"].astype(int).clip(0, N_HAND - 1).values,
        minlength=N_HAND,
    )
    strength_vals = prefix_shots["strengthId"].astype(float).values
    strength_mean = float(strength_vals.mean()) if len(strength_vals) > 0 else 0.0
    action_counts = np.bincount(
        prefix_shots["actionId"].astype(int).clip(0, N_ACTION_RAW - 1).values,
        minlength=N_ACTION_RAW,
    )
    return {
        "hand_counts": hand_counts,
        "strength_mean": strength_mean,
        "action_entropy": _shannon_entropy_freq(action_counts.astype(np.float64)),
    }


# ---- Public interface (mirrors features_v15feat) -----------------------

def compute_global_stats_v16match(train_df: pd.DataFrame) -> dict:
    """Compute global stats for v16match. Delegates to v15feat's stats
    (we extend, don't override). v16match doesn't need extra global stats —
    LORO is per-fold per-match, computed at feature build time.
    """
    return compute_global_stats_v15feat(train_df)


def _build_v16match_added_columns(
    feat_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    max_other_rallies: int = DEFAULT_MAX_OTHER_RALLIES,
    min_other_rallies: int = DEFAULT_MIN_OTHER_RALLIES,
    seed: int = 20260520,
) -> pd.DataFrame:
    """Add the 40 v16match LORO features to feat_df in place.

    feat_df: per-sample feature matrix (one row per (rally_uid, target_sn))
    raw_df: the raw shot-level data the features are derived from (train_df
            or test_df with strikeNumber, match, actionId, etc.)

    Each row in feat_df has rally_uid; we look up that rally's match, then
    aggregate from OTHER rallies in the same match.
    """
    rng = np.random.default_rng(seed)

    # Map rally_uid -> match_id (use first row of each rally in raw_df)
    rally_to_match = (
        raw_df.drop_duplicates("rally_uid")
        .set_index("rally_uid")["match"].astype(int).to_dict()
    )
    # Map rally_uid -> target gamePlayerId
    rally_to_player = (
        raw_df.drop_duplicates("rally_uid")
        .set_index("rally_uid")["gamePlayerId"].astype(int).to_dict()
    )

    # For each match, compute LORO aggregates once
    print(f"  [v16match] Aggregating LORO features over {raw_df['match'].nunique()} matches "
          f"(max_other_rallies={max_other_rallies}, min={min_other_rallies}) ...")
    import time
    t0 = time.time()
    match_aggs: Dict[int, Dict[int, Dict]] = {}
    for match_id, mgdf in raw_df.groupby("match"):
        match_aggs[int(match_id)] = _aggregate_match_prefix(
            mgdf, max_other_rallies=max_other_rallies, rng=rng,
        )
    print(f"  [v16match] match aggregates built in {time.time()-t0:.1f}s")

    # Build the 40 new columns per row in feat_df
    n = len(feat_df)
    cols = {c: np.zeros(n, dtype=np.float32) for c in V16MATCH_ADDED_COLUMNS}

    feat_rally_uids = feat_df["rally_uid"].astype(int).values
    for i in range(n):
        rally_uid = int(feat_rally_uids[i])
        match_id = rally_to_match.get(rally_uid, -1)
        target_player = rally_to_player.get(rally_uid, -1)
        if match_id == -1 or match_id not in match_aggs:
            continue
        agg = match_aggs[match_id].get(rally_uid)
        if agg is None:
            continue
        n_other = int(agg["n_other_rallies"])
        cols["match_other_count_log1p"][i] = float(np.log1p(n_other))
        cols["match_other_avg_rally_length"][i] = float(agg["avg_rally_len"])
        if n_other < min_other_rallies:
            continue  # min-count guard: keep Family C, zero out A/B
        # Family A: action freq
        ac_total = float(agg["action_counts"].sum())
        if ac_total > 0:
            ac_freq = agg["action_counts"].astype(np.float64) / ac_total
            for c in range(N_ACTION_RAW):
                cols[f"match_other_action_freq_{c}"][i] = ac_freq[c]
            cols["match_other_action_entropy"][i] = _shannon_entropy_freq(
                agg["action_counts"].astype(np.float64))
            cols["match_other_action_dominance"][i] = float(ac_freq.max())
        # Family A: point freq
        pc_total = float(agg["point_counts"].sum())
        if pc_total > 0:
            pc_freq = agg["point_counts"].astype(np.float64) / pc_total
            for c in range(N_POINT):
                cols[f"match_other_point_freq_{c}"][i] = pc_freq[c]
            cols["match_other_point_entropy"][i] = _shannon_entropy_freq(
                agg["point_counts"].astype(np.float64))
            cols["match_other_point_dominance"][i] = float(pc_freq.max())

        # Family B: player-specific stats (for target rally's gamePlayerId)
        # Need raw_df subset for this match; lookup is per-row but cached
        # per (match, player) pair would be faster. v1: just compute.
        match_df = raw_df[raw_df["match"] == match_id]
        pl = _aggregate_player_in_match(match_df, target_player, rally_uid)
        ph_total = int(pl["hand_counts"].sum())
        if ph_total > 0:
            for h in range(N_HAND):
                cols[f"target_player_hand_freq_in_match_{h}"][i] = (
                    pl["hand_counts"][h] / ph_total)
            cols["target_player_strength_mean_in_match"][i] = pl["strength_mean"]
            cols["target_player_action_entropy_in_match"][i] = pl["action_entropy"]

    print(f"  [v16match] feature columns assembled ({time.time()-t0:.1f}s total)")
    # Concat all 40 new columns at once (avoids DataFrame fragmentation)
    added = pd.DataFrame(cols, index=feat_df.index)
    return pd.concat([feat_df, added], axis=1)


def build_features_v16match(
    df: pd.DataFrame,
    is_train: bool,
    global_stats_v9: dict = None,
    raw_df: pd.DataFrame = None,
    max_other_rallies: int = DEFAULT_MAX_OTHER_RALLIES,
    min_other_rallies: int = DEFAULT_MIN_OTHER_RALLIES,
) -> pd.DataFrame:
    """Build v16match feature matrix = v15feat + 40 LORO match-context features.

    Args:
        df: cleaned dataframe (after clean_data())
        is_train: True for train rows, False for test
        global_stats_v9: stats dict from compute_global_stats_v16match
        raw_df: the raw shot-level frame used to LORO-aggregate. Same as df
                in standard pipeline.
        max_other_rallies: cap on how many other rallies to aggregate per
                          match (mitigates train/test size mismatch). Default
                          22 to match test conditions.
        min_other_rallies: if fewer than this many other rallies, zero out
                          Family A/B (keep Family C count features only).
    Returns:
        Feature DataFrame with v15feat columns + 40 v16match columns.
    """
    # First build the v15feat backbone (inherits v15feat -> v9)
    if global_stats_v9 is None:
        global_stats_v9 = compute_global_stats_v15feat(df if raw_df is None else raw_df)
    feat = build_features_v15feat(df, is_train=is_train,
                                   global_stats_v9=global_stats_v9,
                                   raw_df=raw_df if raw_df is not None else df)
    # Then add v16match cross-rally features
    feat = _build_v16match_added_columns(
        feat,
        raw_df=raw_df if raw_df is not None else df,
        max_other_rallies=max_other_rallies,
        min_other_rallies=min_other_rallies,
    )
    audit_no_banned_names_v16match(feat.columns.tolist())
    return feat


def get_feature_names_v16match(feat_df: pd.DataFrame) -> list:
    """Return feature column names for v16match (delegates to v15feat then
    appends the 40 LORO columns).
    """
    base = get_feature_names_v15feat(feat_df)
    extra = [c for c in V16MATCH_ADDED_COLUMNS if c in feat_df.columns]
    return base + extra
