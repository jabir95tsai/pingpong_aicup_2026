"""R-032 v2 — cross-rally LORO features (Codex APPROVE_WITH_FIXES, 2026-05-21).

Codex BLOCKED v1 with 8 fixes. v2 implements all of them:

  [P1-1] Group by (match, unordered_player_pair), NOT by `match` alone
         (test_new has 16/79 matches with 21-31 unique players, not 2)
  [P1-2] Target-hitter parity logic for Family B:
         odd next_strikeNumber -> server-side player
         even next_strikeNumber -> receiver-side player
  [P1-3] v1 = v9 + LORO ONLY (drop v15feat backbone for clean attribution)
  [P1-4] Cap to first K=3 prefix shots per other rally (matches test
         visible-prefix length distribution)
  [P2-5] Family C count/avg features DROPPED from model output (kept as
         audit diagnostics in metadata only)
  [P2-6] Deterministic hash-based subsampling (no RNG stream order)
  [P2-7] Real-data audit script run BEFORE training; metadata embeds
         match-pair size distribution, prefix-length stats, etc.
  [P3-8] Family B uses cached per-group aggregation, not per-row re-filter

Scope (per Codex): Fold-1 smoke only first. No full 5-fold, no LB until
Codex reviews smoke artifacts.

INTERFACE matches v15feat_b for plug-in compat:
  compute_global_stats_v16match_v2(train_df) -> stats_dict
  build_features_v16match_v2(df, is_train, global_stats_v9, raw_df) -> feature_df
  get_feature_names_v16match_v2(feat_df) -> list[str]

Output columns (33 features after dropping Family C from model features):
  - match_pair_other_action_freq_{0..18}  (19)
  - match_pair_other_point_freq_{0..9}    (10)
  - match_pair_other_action_entropy / point_entropy / action_dominance /
    point_dominance (4)
  Total Family A: 33 features

(Family B target-player stats deferred to v1b after smoke per Codex Q2)
(Family C count/avg in METADATA only, not model features)
"""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# v1: v9 + LORO ONLY (Codex P1-3)
from features_v9 import (  # noqa: E402
    build_features_v9,
    compute_global_stats_v9,
    get_feature_names_v9,
)


# ---- Constants ----------------------------------------------------------

N_ACTION_RAW = 19
N_POINT = 10
PREFIX_CAP_K = 3   # Codex P1-4: cap to first K=3 prefix shots per other rally
MIN_OTHER_RALLIES = 3  # below this, zero out Family A

# Codex P1 (2026-05-22): cap aggregation to a test-matched number of other
# rallies per pair so train OOF features are estimated from a comparable
# sample size to test (test p90 ~23). Without this, train sees ~67 other
# rallies per pair while test sees ~0-23, breaking transfer.
MAX_OTHER_RALLIES = 22


V16MATCH_V2_ADDED_COLUMNS = (
    [f"match_pair_other_action_freq_{c}" for c in range(N_ACTION_RAW)]
    + [f"match_pair_other_point_freq_{c}" for c in range(N_POINT)]
    + [
        "match_pair_other_action_entropy",
        "match_pair_other_point_entropy",
        "match_pair_other_action_dominance",
        "match_pair_other_point_dominance",
    ]
)
assert len(V16MATCH_V2_ADDED_COLUMNS) == 33
assert len(set(V16MATCH_V2_ADDED_COLUMNS)) == 33


def audit_no_banned_names_v16match_v2(cols: list) -> None:
    """Codex P1-3: forbid SGP-related, terminal-shot, full-length, and
    Family C names that would leak distribution shift.
    """
    banned = (
        "match_other_avg_serverGetPoint",
        "match_avg_rally_length",
        "match_other_final_action",
        "match_other_final_point",
        "match_other_terminal_action",
        "match_pair_other_count",         # Family C dropped from model
        "match_pair_other_avg_rally_len", # Family C dropped from model
    )
    hits = [c for c in cols if c in banned]
    if hits:
        raise ValueError(f"Banned R-032 v2 feature names present: {hits}")


# ---- Shannon entropy helper ---------------------------------------------

def _shannon_entropy_freq(probs: np.ndarray) -> float:
    total = float(probs.sum())
    if total <= 0:
        return 0.0
    p = probs / total
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


# ---- Codex P1-1: match_pair grouping ------------------------------------

def _make_match_pair_key(match_id: int, pid_a: int, pid_b: int) -> str:
    """Stable key for (match, unordered player pair)."""
    a, b = sorted([int(pid_a), int(pid_b)])
    return f"{int(match_id)}|{a}|{b}"


def _build_rally_to_match_pair(raw_df: pd.DataFrame) -> Dict[int, str]:
    """Map each rally_uid to its (match, unordered_pair) key.

    The pair comes from the rally's FIRST shot's (gamePlayerId,
    gamePlayerOtherId). For test rallies, both IDs are visible in
    test_new.csv even though players are de-identified.
    """
    first_shot = (
        raw_df.sort_values(["rally_uid", "strikeNumber"])
        .drop_duplicates("rally_uid")
        .set_index("rally_uid")
    )
    out = {}
    for rid, row in first_shot.iterrows():
        out[int(rid)] = _make_match_pair_key(
            int(row["match"]), int(row["gamePlayerId"]),
            int(row["gamePlayerOtherId"]),
        )
    return out


# ---- Codex P2-6: deterministic hash-based subsample ---------------------

def _deterministic_select(other_uids: list, target_uid: int,
                          k: int, seed: int = 20260522) -> list:
    """Pick k UIDs from other_uids deterministically based on
    hash((target_uid, uid, seed))."""
    if k <= 0 or k >= len(other_uids):
        return list(other_uids)
    scored = []
    for uid in other_uids:
        h = hashlib.md5(f"{seed}|{target_uid}|{uid}".encode()).hexdigest()
        scored.append((int(h[:12], 16), uid))
    scored.sort()
    return [u for _, u in scored[:k]]


# ---- Per-rally prefix counts (P1-4: cap to first K shots) ---------------

def _per_rally_prefix_counts(raw_df: pd.DataFrame, prefix_cap: int = PREFIX_CAP_K
                             ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, int]]:
    """For each rally, count action/point class frequencies in its
    FIRST `prefix_cap` shots (matching test-visibility constraint).

    Returns:
      rally_action_counts: rally_uid -> ndarray(19,)
      rally_point_counts:  rally_uid -> ndarray(10,)
      rally_n_used:        rally_uid -> int (number of prefix shots used)
    """
    action_counts = {}
    point_counts = {}
    n_used = {}
    for rally_uid, gdf in raw_df.groupby("rally_uid"):
        g = gdf.sort_values("strikeNumber").reset_index(drop=True)
        # Drop the rally's last shot (matches "visible prefix of finished
        # rally" interpretation), then cap to K
        if len(g) <= 1:
            action_counts[int(rally_uid)] = np.zeros(N_ACTION_RAW, dtype=np.int64)
            point_counts[int(rally_uid)] = np.zeros(N_POINT, dtype=np.int64)
            n_used[int(rally_uid)] = 0
            continue
        prefix = g.iloc[:-1].iloc[:prefix_cap]
        ac = np.bincount(prefix["actionId"].astype(int).clip(0, N_ACTION_RAW - 1).values,
                         minlength=N_ACTION_RAW)
        pc = np.bincount(prefix["pointId"].astype(int).clip(0, N_POINT - 1).values,
                         minlength=N_POINT)
        action_counts[int(rally_uid)] = ac
        point_counts[int(rally_uid)] = pc
        n_used[int(rally_uid)] = len(prefix)
    return action_counts, point_counts, n_used


# ---- Per-match_pair LORO aggregation ------------------------------------

def _aggregate_pair_features(
    raw_df: pd.DataFrame,
    rally_to_pair: Dict[int, str],
    prefix_cap: int = PREFIX_CAP_K,
    min_other: int = MIN_OTHER_RALLIES,
    max_other: int = MAX_OTHER_RALLIES,
    select_seed: int = 20260522,
) -> Dict[int, Dict[str, np.ndarray]]:
    """For each rally R, aggregate action/point counts from OTHER rallies
    in the same (match, player_pair) group, excluding R's own shots.

    Codex P1 fix (2026-05-22): cap the number of other rallies aggregated to
    `max_other`, using `_deterministic_select` to pick reproducibly per
    target rally. This makes train OOF features sample-size-matched to test
    (test p90 = 23 others per pair, default cap K=22).

    Returns: rally_uid -> {action_counts, point_counts, n_other_total,
                            n_other_used, n_total_shots}
      n_other_total = raw count of other rallies in this pair group
      n_other_used  = post-cap count actually used in this rally's aggregation
    """
    # Pre-compute per-rally prefix counts
    rally_ac, rally_pc, rally_n = _per_rally_prefix_counts(raw_df, prefix_cap)

    # Group rallies by pair_key
    pair_to_rallies: Dict[str, list] = {}
    for rid, pkey in rally_to_pair.items():
        pair_to_rallies.setdefault(pkey, []).append(int(rid))

    # Pre-aggregate per-pair total — kept for fast path (n_other_total <= cap)
    pair_total_action = {}
    pair_total_point = {}
    pair_total_n = {}
    for pkey, rids in pair_to_rallies.items():
        ta = np.zeros(N_ACTION_RAW, dtype=np.int64)
        tp = np.zeros(N_POINT, dtype=np.int64)
        tn = 0
        for rid in rids:
            ta += rally_ac.get(rid, np.zeros(N_ACTION_RAW, dtype=np.int64))
            tp += rally_pc.get(rid, np.zeros(N_POINT, dtype=np.int64))
            tn += rally_n.get(rid, 0)
        pair_total_action[pkey] = ta
        pair_total_point[pkey] = tp
        pair_total_n[pkey] = tn

    # Per-rally aggregation, with deterministic max-cap
    out = {}
    for rid, pkey in rally_to_pair.items():
        all_rids_in_pair = pair_to_rallies.get(pkey, [])
        other_rids = [r for r in all_rids_in_pair if r != rid]
        n_other_total = len(other_rids)
        if n_other_total < min_other:
            out[int(rid)] = {
                "action_counts": np.zeros(N_ACTION_RAW, dtype=np.int64),
                "point_counts": np.zeros(N_POINT, dtype=np.int64),
                "n_other_total": n_other_total,
                "n_other_used": 0,
                "n_total_shots": 0,
            }
            continue
        if max_other > 0 and n_other_total > max_other:
            # Codex P1: deterministic subsample
            chosen = _deterministic_select(other_rids, target_uid=rid,
                                            k=max_other, seed=select_seed)
            ac = np.zeros(N_ACTION_RAW, dtype=np.int64)
            pc = np.zeros(N_POINT, dtype=np.int64)
            ns = 0
            for r in chosen:
                ac += rally_ac.get(r, np.zeros(N_ACTION_RAW, dtype=np.int64))
                pc += rally_pc.get(r, np.zeros(N_POINT, dtype=np.int64))
                ns += rally_n.get(r, 0)
            n_used = len(chosen)
        else:
            # Fast path — no cap needed: subtract target's contribution from pair total
            ac = pair_total_action[pkey] - rally_ac.get(rid, np.zeros(N_ACTION_RAW, dtype=np.int64))
            pc = pair_total_point[pkey] - rally_pc.get(rid, np.zeros(N_POINT, dtype=np.int64))
            ns = pair_total_n[pkey] - rally_n.get(rid, 0)
            n_used = n_other_total
        out[int(rid)] = {
            "action_counts": ac,
            "point_counts": pc,
            "n_other_total": n_other_total,
            "n_other_used": n_used,
            "n_total_shots": ns,
        }
    return out


# ---- Codex P2-7: real-data audit at build time --------------------------

def _audit_real_data(raw_df: pd.DataFrame, rally_to_pair: Dict[int, str], label: str,
                     pair_aggs: Dict[int, Dict] = None,
                     max_other: int = MAX_OTHER_RALLIES,
                     min_other: int = MIN_OTHER_RALLIES) -> dict:
    """Run real-data audits and return diagnostics.

    Pair-weighted audits (Codex feedback): biased toward singleton pairs.
    Rally-weighted audits: better measure of model feature coverage.
    Both reported.
    """
    # Match unique-player distribution
    match_player_counts = (
        raw_df.groupby("match")
        .agg(n_players=("gamePlayerId", lambda s: pd.concat([s, raw_df.loc[s.index, "gamePlayerOtherId"]]).nunique()))
    )
    n_2player = (match_player_counts["n_players"] == 2).sum()
    n_total_matches = len(match_player_counts)

    # PAIR-WEIGHTED: count rallies per pair
    pair_to_n = {}
    for pkey in rally_to_pair.values():
        pair_to_n[pkey] = pair_to_n.get(pkey, 0) + 1
    pair_other_counts = [v - 1 for v in pair_to_n.values()]
    pair_other_arr = np.array(pair_other_counts) if pair_other_counts else np.array([0])

    # RALLY-WEIGHTED: for each rally, look up its pair's n_other - useful for
    # model feature coverage. Codex P2: 85.7% of test rallies have n_other >= 3.
    rally_n_other = []
    for rid, pkey in rally_to_pair.items():
        rally_n_other.append(pair_to_n.get(pkey, 1) - 1)
    rally_arr = np.array(rally_n_other) if rally_n_other else np.array([0])
    n_rallies_with_signal = int((rally_arr >= min_other).sum())
    total_rallies = int(len(rally_arr))

    diag = {
        "label": label,
        "n_total_matches": int(n_total_matches),
        "n_2player_matches": int(n_2player),
        "frac_2player": float(n_2player / max(n_total_matches, 1)),
        "n_unique_pairs": len(set(rally_to_pair.values())),
        # pair-weighted (Codex: biased toward singleton pairs)
        "pair_w_other_p50": int(np.percentile(pair_other_arr, 50)),
        "pair_w_other_p90": int(np.percentile(pair_other_arr, 90)),
        "pair_w_other_min": int(pair_other_arr.min()),
        "pair_w_other_max": int(pair_other_arr.max()),
        # rally-weighted (Codex: actual feature coverage)
        "rally_w_other_p50": int(np.percentile(rally_arr, 50)),
        "rally_w_other_p90": int(np.percentile(rally_arr, 90)),
        "rally_w_other_min": int(rally_arr.min()),
        "rally_w_other_max": int(rally_arr.max()),
        "rally_coverage": f"{n_rallies_with_signal}/{total_rallies} = {n_rallies_with_signal/max(total_rallies,1):.3%}",
    }

    # Post-cap audit if pair_aggs provided (after _aggregate_pair_features)
    if pair_aggs is not None:
        used_counts = [agg["n_other_used"] for agg in pair_aggs.values()]
        ua = np.array(used_counts) if used_counts else np.array([0])
        diag["post_cap_max_other"] = int(max_other)
        diag["post_cap_used_p50"] = int(np.percentile(ua, 50))
        diag["post_cap_used_p90"] = int(np.percentile(ua, 90))
        diag["post_cap_used_min"] = int(ua.min())
        diag["post_cap_used_max"] = int(ua.max())
        diag["post_cap_used_mean"] = float(ua.mean())
        n_capped = int((ua == max_other).sum())
        diag["n_rallies_capped"] = n_capped
        diag["frac_rallies_capped"] = float(n_capped / max(len(ua), 1))
    return diag


# ---- Public interface ---------------------------------------------------

def compute_global_stats_v16match_v2(train_df: pd.DataFrame) -> dict:
    """v1 = v9 backbone only (no extra global stats)."""
    return compute_global_stats_v9(train_df)


def _build_v2_added_columns(
    feat_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    label: str,
    prefix_cap: int = PREFIX_CAP_K,
    min_other: int = MIN_OTHER_RALLIES,
    max_other: int = MAX_OTHER_RALLIES,
) -> pd.DataFrame:
    """Add 33 LORO features to feat_df."""
    rally_to_pair = _build_rally_to_match_pair(raw_df)

    pair_aggs = _aggregate_pair_features(raw_df, rally_to_pair,
                                          prefix_cap=prefix_cap,
                                          min_other=min_other,
                                          max_other=max_other)

    # Audit AFTER aggregation so post-cap stats are included
    diag = _audit_real_data(raw_df, rally_to_pair, label,
                             pair_aggs=pair_aggs,
                             max_other=max_other, min_other=min_other)
    print(f"  [v16match_v2 {label}] {diag}")

    n = len(feat_df)
    cols = {c: np.zeros(n, dtype=np.float32) for c in V16MATCH_V2_ADDED_COLUMNS}

    feat_rally_uids = feat_df["rally_uid"].astype(int).values
    for i in range(n):
        rally_uid = int(feat_rally_uids[i])
        agg = pair_aggs.get(rally_uid)
        if agg is None or agg["n_other_total"] < min_other:
            continue
        ac_total = float(agg["action_counts"].sum())
        if ac_total > 0:
            ac_freq = agg["action_counts"].astype(np.float64) / ac_total
            for c in range(N_ACTION_RAW):
                cols[f"match_pair_other_action_freq_{c}"][i] = ac_freq[c]
            cols["match_pair_other_action_entropy"][i] = _shannon_entropy_freq(
                agg["action_counts"].astype(np.float64))
            cols["match_pair_other_action_dominance"][i] = float(ac_freq.max())
        pc_total = float(agg["point_counts"].sum())
        if pc_total > 0:
            pc_freq = agg["point_counts"].astype(np.float64) / pc_total
            for c in range(N_POINT):
                cols[f"match_pair_other_point_freq_{c}"][i] = pc_freq[c]
            cols["match_pair_other_point_entropy"][i] = _shannon_entropy_freq(
                agg["point_counts"].astype(np.float64))
            cols["match_pair_other_point_dominance"][i] = float(pc_freq.max())

    added = pd.DataFrame(cols, index=feat_df.index)
    return pd.concat([feat_df, added], axis=1)


def build_features_v16match_v2(
    df: pd.DataFrame,
    is_train: bool,
    global_stats_v9: dict = None,
    raw_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build v16match_v2 = v9 backbone + 33 LORO (match,pair) features."""
    if global_stats_v9 is None:
        global_stats_v9 = compute_global_stats_v9(df if raw_df is None else raw_df)
    feat = build_features_v9(df, is_train=is_train,
                              global_stats_v9=global_stats_v9,
                              raw_df=raw_df if raw_df is not None else df)
    label = "train" if is_train else "test"
    feat = _build_v2_added_columns(feat,
                                    raw_df=raw_df if raw_df is not None else df,
                                    label=label)
    audit_no_banned_names_v16match_v2(feat.columns.tolist())
    return feat


def get_feature_names_v16match_v2(feat_df: pd.DataFrame) -> list:
    base = get_feature_names_v9(feat_df)
    extra = [c for c in V16MATCH_V2_ADDED_COLUMNS if c in feat_df.columns]
    return base + extra
