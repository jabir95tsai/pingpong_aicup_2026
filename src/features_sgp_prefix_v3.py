"""Feature engineering for R-030 sgp_prefix_v3 — Prefix-only SGP head.

Codex APPROVED scope (2026-05-20): `core` feature profile only for v1 smoke.
Full 46-feature distribution Family E is deferred to v1b ablation.

Design:
- Build features from VISIBLE PREFIX shots 1..n-1 of each rally.
- For training: one sample per (rally, target_shot_n) where n ≥ 2.
  Label = rally-level `serverGetPoint`.
- For test_new: one sample per rally, using all visible prefix shots.
- Output one continuous probability per rally.

Hard rules (locked):
- NEVER use target shot n's data as feature
- NEVER use any shot with strikeNumber ≥ n
- NEVER use rally_uid as a feature (memorization risk)
- NEVER use rally-level outcome (serverGetPoint of OTHER rallies) as feature
- Feature names cannot contain: full_length, final_shot, terminal, winner,
  n_shots_total, n_shots_remaining, rally_winner, point_winner

Feature families (all prefix-safe):

Family A — Last-2 lag shot features (14 features)
  lag1_{actionId, pointId, handId, strengthId, spinId, positionId, strikeId}
  lag2_{...same...}

Family C — Score-state features (11 features)
  scoreSelf, scoreOther, score_diff, score_total
  is_deuce, match_point_self, match_point_other
  points_to_win_self, points_to_win_other
  numberGame, sex

Family D — Serve/receive pattern features (5 features)
  is_target_serve_side (1 if next strikeNumber odd)
  prefix_serve_side_count, prefix_receive_side_count
  consecutive_same_side_at_tail
  last_action_category (0=other, 1=attack, 2=control, 3=defense, 4=serve)

Family E core — Reduced distributions (34 features)
  action_category_freq_{attack, control, defense, serve} × 4
  top8_action_freq_{0,1,2,5,6,10,11,13} × 8  (8 most common AICUP actions)
  top5_point_freq_{0,5,7,8,9} × 5            (5 most common AICUP points)
  hand_freq_{0,1,2} × 3                       (forehand/backhand/none)
  strength_freq_{0,1,2,3} × 4                 (none/strong/medium/weak)
  spin_freq_{0,1,2,3,4,5} × 6                 (none/topspin/.../sidedown)
  action_entropy, action_dominance (2)
  point_entropy, point_dominance (2)

Family F — One prefix-length feature (1 feature)
  prefix_length_log = log1p(next_strikeNumber - 1)
  (NOT next_strikeNumber directly to avoid triplication)

TOTAL: 65 features in core profile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# AICUP action category mapping (per CLAUDE.md schema)
_ATTACK_ACTIONS = {1, 2, 3, 4, 5, 6, 7}   # Loop, Cloop, Smash, Flip, Pushfast, Push, Flick
_CONTROL_ACTIONS = {8, 9, 10, 11}          # Arch, Knuckle, Chop_r, ShortStop
_DEFENSE_ACTIONS = {12, 13, 14}            # Chop, Block, Lob
_SERVE_ACTIONS = {15, 16, 17, 18}          # Traditional, Hook, Reverse-spin, Squat

# Fixed top-k class IDs per Codex (NOT computed from train; declared in plan)
_TOP8_ACTION_IDS = (0, 1, 2, 5, 6, 10, 11, 13)
_TOP5_POINT_IDS = (0, 5, 7, 8, 9)

# Cardinalities
_N_ACT_FULL = 19
_N_PT = 10
_N_HAND = 3       # 0=none, 1=forehand, 2=backhand
_N_STRENGTH = 4   # 0=none, 1=strong, 2=medium, 3=weak
_N_SPIN = 6       # 0=none, 1=topspin, 2=backspin, 3=no-spin, 4=side-top, 5=side-down


# ---- Banned feature name regex ----
_BANNED_NAME_PATTERN = (
    r"full_length|final_shot|terminal|winner|"
    r"n_shots_total|n_shots_remaining|rally_winner|point_winner"
)


def _categorize_action(action_id: int) -> int:
    """0=other/empty, 1=attack, 2=control, 3=defense, 4=serve."""
    if action_id in _ATTACK_ACTIONS:
        return 1
    if action_id in _CONTROL_ACTIONS:
        return 2
    if action_id in _DEFENSE_ACTIONS:
        return 3
    if action_id in _SERVE_ACTIONS:
        return 4
    return 0


def _shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy (base e). Zero counts contribute 0. Returns 0 if all zero."""
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    nz = counts[counts > 0]
    p = nz.astype(np.float64) / total
    return float(-np.sum(p * np.log(p)))


def _tail_run_length(arr: np.ndarray) -> int:
    """Length of run of identical values at end of array."""
    L = len(arr)
    if L == 0:
        return 0
    target = arr[-1]
    n = 1
    for i in range(L - 2, -1, -1):
        if arr[i] == target:
            n += 1
        else:
            break
    return n


def _build_one_row(
    rally_uid: int,
    target_strike: int,
    prefix_act: np.ndarray,
    prefix_pt: np.ndarray,
    prefix_hand: np.ndarray,
    prefix_strength: np.ndarray,
    prefix_spin: np.ndarray,
    prefix_pos: np.ndarray,
    prefix_strike_id: np.ndarray,
    prefix_strike_num: np.ndarray,
    prefix_player: np.ndarray,
    server_id: int,
    context_score_self: int,
    context_score_other: int,
    context_num_game: int,
    context_sex: int,
) -> dict:
    """Build a single feature row from the prefix of one rally.

    target_strike = strikeNumber of the SHOT WE'RE PREDICTING (target shot n).
    Prefix arrays contain only shots with strikeNumber < target_strike.
    """
    n_prefix = len(prefix_act)
    row: dict = {"rally_uid": rally_uid, "next_strikeNumber": target_strike}

    # ---------- Family A — Last-2 lag features ----------
    # lag1 = most recent visible shot (prefix[-1]) if exists
    # lag2 = 2nd most recent (prefix[-2]) if exists
    def _lag(arr, k):
        if n_prefix >= k:
            return int(arr[n_prefix - k])
        return -1
    row["lag1_actionId"]    = _lag(prefix_act, 1)
    row["lag1_pointId"]     = _lag(prefix_pt, 1)
    row["lag1_handId"]      = _lag(prefix_hand, 1)
    row["lag1_strengthId"]  = _lag(prefix_strength, 1)
    row["lag1_spinId"]      = _lag(prefix_spin, 1)
    row["lag1_positionId"]  = _lag(prefix_pos, 1)
    row["lag1_strikeId"]    = _lag(prefix_strike_id, 1)
    row["lag2_actionId"]    = _lag(prefix_act, 2)
    row["lag2_pointId"]     = _lag(prefix_pt, 2)
    row["lag2_handId"]      = _lag(prefix_hand, 2)
    row["lag2_strengthId"]  = _lag(prefix_strength, 2)
    row["lag2_spinId"]      = _lag(prefix_spin, 2)
    row["lag2_positionId"]  = _lag(prefix_pos, 2)
    row["lag2_strikeId"]    = _lag(prefix_strike_id, 2)

    # ---------- Family C — Score-state ----------
    row["scoreSelf"]    = context_score_self
    row["scoreOther"]   = context_score_other
    row["score_diff"]   = context_score_self - context_score_other
    row["score_total"]  = context_score_self + context_score_other
    row["is_deuce"]     = int(context_score_self >= 10 and context_score_other >= 10)
    row["match_point_self"]  = int(context_score_self >= 10 and (context_score_self - context_score_other) >= 0)
    row["match_point_other"] = int(context_score_other >= 10 and (context_score_other - context_score_self) >= 0)
    row["points_to_win_self"]  = max(0, 11 - context_score_self)
    row["points_to_win_other"] = max(0, 11 - context_score_other)
    row["numberGame"]   = context_num_game
    row["sex"]          = context_sex

    # ---------- Family D — Serve/receive pattern ----------
    row["is_target_serve_side"] = int(target_strike % 2 == 1)
    if n_prefix > 0:
        # Side of each prefix shot: 1 if strikeNumber odd (serve side), 0 if even
        prefix_sides = (prefix_strike_num % 2 == 1).astype(np.int32)
        row["prefix_serve_side_count"]   = int(prefix_sides.sum())
        row["prefix_receive_side_count"] = int((1 - prefix_sides).sum())
        row["consecutive_same_side_at_tail"] = _tail_run_length(prefix_sides)
        last_action = int(prefix_act[-1])
        row["last_action_category"] = _categorize_action(last_action)
    else:
        row["prefix_serve_side_count"] = 0
        row["prefix_receive_side_count"] = 0
        row["consecutive_same_side_at_tail"] = 0
        row["last_action_category"] = 0

    # ---------- Family E core — Reduced distributions ----------
    if n_prefix > 0:
        n_inv = 1.0 / float(n_prefix)
        # Action category counts
        cat_counts = np.zeros(5, dtype=np.float64)
        for a in prefix_act:
            cat_counts[_categorize_action(int(a))] += 1
        # action_category_freq for {attack, control, defense, serve} only (index 1..4)
        row["action_cat_attack_freq"]   = float(cat_counts[1]) * n_inv
        row["action_cat_control_freq"]  = float(cat_counts[2]) * n_inv
        row["action_cat_defense_freq"]  = float(cat_counts[3]) * n_inv
        row["action_cat_serve_freq"]    = float(cat_counts[4]) * n_inv

        # Top-8 action freqs (fixed class IDs)
        act_clipped = np.clip(prefix_act, 0, _N_ACT_FULL - 1)
        act_counts = np.bincount(act_clipped, minlength=_N_ACT_FULL)
        for c in _TOP8_ACTION_IDS:
            row[f"top8_action_freq_{c}"] = float(act_counts[c]) * n_inv

        # Top-5 point freqs (fixed class IDs)
        pt_clipped = np.clip(prefix_pt, 0, _N_PT - 1)
        pt_counts = np.bincount(pt_clipped, minlength=_N_PT)
        for c in _TOP5_POINT_IDS:
            row[f"top5_point_freq_{c}"] = float(pt_counts[c]) * n_inv

        # Hand / strength / spin distributions
        hand_clipped = np.clip(prefix_hand, 0, _N_HAND - 1)
        hand_counts = np.bincount(hand_clipped, minlength=_N_HAND)
        for c in range(_N_HAND):
            row[f"hand_freq_{c}"] = float(hand_counts[c]) * n_inv
        strength_clipped = np.clip(prefix_strength, 0, _N_STRENGTH - 1)
        strength_counts = np.bincount(strength_clipped, minlength=_N_STRENGTH)
        for c in range(_N_STRENGTH):
            row[f"strength_freq_{c}"] = float(strength_counts[c]) * n_inv
        spin_clipped = np.clip(prefix_spin, 0, _N_SPIN - 1)
        spin_counts = np.bincount(spin_clipped, minlength=_N_SPIN)
        for c in range(_N_SPIN):
            row[f"spin_freq_{c}"] = float(spin_counts[c]) * n_inv

        # Entropy + dominance
        row["action_entropy"]   = _shannon_entropy(act_counts)
        row["action_dominance"] = float(act_counts.max()) * n_inv
        row["point_entropy"]    = _shannon_entropy(pt_counts)
        row["point_dominance"]  = float(pt_counts.max()) * n_inv
    else:
        # Empty history → all zeros
        for c in ("attack", "control", "defense", "serve"):
            row[f"action_cat_{c}_freq"] = 0.0
        for c in _TOP8_ACTION_IDS:
            row[f"top8_action_freq_{c}"] = 0.0
        for c in _TOP5_POINT_IDS:
            row[f"top5_point_freq_{c}"] = 0.0
        for c in range(_N_HAND):
            row[f"hand_freq_{c}"] = 0.0
        for c in range(_N_STRENGTH):
            row[f"strength_freq_{c}"] = 0.0
        for c in range(_N_SPIN):
            row[f"spin_freq_{c}"] = 0.0
        row["action_entropy"]   = 0.0
        row["action_dominance"] = 0.0
        row["point_entropy"]    = 0.0
        row["point_dominance"]  = 0.0

    # ---------- Family F — One prefix-length feature ----------
    # log1p(prefix_length) — single constrained representation per Codex
    row["prefix_length_log"] = float(np.log1p(n_prefix))

    return row


def build_features_sgp_v3(raw_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """Build prefix-only SGP features for either train or test.

    For training (is_train=True):
        For each rally R with L shots (L ≥ 2), generate L-1 samples:
        target_strike = 2, 3, ..., L. Features use shots {1, ..., target-1}.
        Each sample carries the rally-level SGP truth.

    For test_new (is_train=False):
        For each rally R with L_visible shots, generate ONE sample:
        target_strike = L_visible + 1. Features use all L_visible visible shots.

    Returns:
        DataFrame with columns: rally_uid, next_strikeNumber, all 65 features,
        plus (if is_train) serverGetPoint.
    """
    df_sorted = raw_df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)

    rows = []
    for rally_uid, group in df_sorted.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber").reset_index(drop=True)
        L = len(group)
        if L < 2:
            # Need at least 1 shot to predict from
            if not is_train:
                # For test rallies with L_visible=1, we still emit one row
                pass  # handled below
            else:
                continue

        # Player IDs to determine server (the player at strikeNumber=1)
        server_player_id = int(group.iloc[0]["gamePlayerId"])

        # Pre-extract raw arrays
        all_act    = group["actionId"].astype(int).values
        all_pt     = group["pointId"].astype(int).values
        all_hand   = group["handId"].astype(int).values
        all_strength = group["strengthId"].astype(int).values
        all_spin   = group["spinId"].astype(int).values
        all_pos    = group["positionId"].astype(int).values
        all_strike_id  = group["strikeId"].astype(int).values
        all_strike_num = group["strikeNumber"].astype(int).values
        all_player = group["gamePlayerId"].astype(int).values

        # Context columns (from rally start; same across all shots in rally)
        ctx_self  = int(group.iloc[0]["scoreSelf"])
        ctx_other = int(group.iloc[0]["scoreOther"])
        ctx_game  = int(group.iloc[0]["numberGame"])
        ctx_sex   = int(group.iloc[0]["sex"])

        # SGP label (rally-level)
        if is_train and "serverGetPoint" in group.columns:
            sgp_label = int(group.iloc[0]["serverGetPoint"])
        else:
            sgp_label = None

        if is_train:
            # Generate one sample per target strike 2..L
            for i in range(1, L):
                target_strike = int(all_strike_num[i])
                # Prefix = shots with strikeNumber < target_strike
                # Strictly less than — never include the target shot itself
                prefix_mask = all_strike_num < target_strike
                # AUDIT INVARIANT: max(prefix_strikeNumber) < target_strike
                if prefix_mask.any():
                    assert all_strike_num[prefix_mask].max() < target_strike, \
                        f"Prefix containment violated: max={all_strike_num[prefix_mask].max()}, target={target_strike}"
                row = _build_one_row(
                    rally_uid=int(rally_uid),
                    target_strike=target_strike,
                    prefix_act=all_act[prefix_mask],
                    prefix_pt=all_pt[prefix_mask],
                    prefix_hand=all_hand[prefix_mask],
                    prefix_strength=all_strength[prefix_mask],
                    prefix_spin=all_spin[prefix_mask],
                    prefix_pos=all_pos[prefix_mask],
                    prefix_strike_id=all_strike_id[prefix_mask],
                    prefix_strike_num=all_strike_num[prefix_mask],
                    prefix_player=all_player[prefix_mask],
                    server_id=server_player_id,
                    context_score_self=ctx_self,
                    context_score_other=ctx_other,
                    context_num_game=ctx_game,
                    context_sex=ctx_sex,
                )
                if sgp_label is not None:
                    row["serverGetPoint"] = sgp_label
                rows.append(row)
        else:
            # TEST: one sample per rally, using ALL visible shots as prefix.
            target_strike = int(all_strike_num[-1]) + 1
            row = _build_one_row(
                rally_uid=int(rally_uid),
                target_strike=target_strike,
                prefix_act=all_act,
                prefix_pt=all_pt,
                prefix_hand=all_hand,
                prefix_strength=all_strength,
                prefix_spin=all_spin,
                prefix_pos=all_pos,
                prefix_strike_id=all_strike_id,
                prefix_strike_num=all_strike_num,
                prefix_player=all_player,
                server_id=server_player_id,
                context_score_self=ctx_self,
                context_score_other=ctx_other,
                context_num_game=ctx_game,
                context_sex=ctx_sex,
            )
            rows.append(row)

    return pd.DataFrame(rows)


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return the model feature column names (exclude rally_uid + target)."""
    drop = {"rally_uid", "next_strikeNumber", "serverGetPoint"}
    return [c for c in df.columns if c not in drop]


# Convenience list (for audit/grep)
_ALL_FEATURE_COLS = (
    # Family A — last-2 lags (14)
    [f"lag{k}_{c}" for k in (1, 2)
     for c in ("actionId", "pointId", "handId", "strengthId",
               "spinId", "positionId", "strikeId")]
    # Family C — score-state (11)
    + ["scoreSelf", "scoreOther", "score_diff", "score_total",
       "is_deuce", "match_point_self", "match_point_other",
       "points_to_win_self", "points_to_win_other",
       "numberGame", "sex"]
    # Family D — serve/receive pattern (5)
    + ["is_target_serve_side", "prefix_serve_side_count",
       "prefix_receive_side_count", "consecutive_same_side_at_tail",
       "last_action_category"]
    # Family E core — distributions (34)
    + [f"action_cat_{c}_freq" for c in ("attack", "control", "defense", "serve")]
    + [f"top8_action_freq_{c}" for c in _TOP8_ACTION_IDS]
    + [f"top5_point_freq_{c}" for c in _TOP5_POINT_IDS]
    + [f"hand_freq_{c}" for c in range(_N_HAND)]
    + [f"strength_freq_{c}" for c in range(_N_STRENGTH)]
    + [f"spin_freq_{c}" for c in range(_N_SPIN)]
    + ["action_entropy", "action_dominance",
       "point_entropy", "point_dominance"]
    # Family F — one length feature (1)
    + ["prefix_length_log"]
)

assert len(_ALL_FEATURE_COLS) == 65, \
    f"Feature count drift: {len(_ALL_FEATURE_COLS)} != 65"

# Public alias
SGP_V3_CORE_COLUMNS = _ALL_FEATURE_COLS


def audit_no_banned_names(feature_cols: list) -> None:
    """Raise if any feature name contains banned substrings (per LESSONS)."""
    import re
    pattern = re.compile(_BANNED_NAME_PATTERN, re.IGNORECASE)
    bad = [c for c in feature_cols if pattern.search(c)]
    if bad:
        raise ValueError(f"Banned feature names detected: {bad}")
