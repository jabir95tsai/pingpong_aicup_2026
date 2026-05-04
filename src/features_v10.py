"""Feature engineering V10: extends V9 with player profiles, history class
frequencies, and streak features.

V9 baseline: 1170 features.
V10 adds:

  Group 7 — Player profile (fold-safe, NO SGP-derived winrate):
    Per-player action distribution (15 features): pp_act_freq_0..14
    Per-player point  distribution (10 features): pp_pt_freq_0..9
    Opponent mirror                (25 features): opp_act_freq_0..14, opp_pt_freq_0..9
    player_n_rallies, opp_n_rallies               (2 features)
    Total: 52 features
    Fallback for unknown / rare players (n_rallies < MIN_RALLIES): sex-level marginal.
    EXCLUDED: player_win_rate / opp_win_rate / win_rate_diff (SGP-derived, see policy note).

  Group 8 — History class frequencies (rally-internal, fold-safe):
    hist_action_freq_0..14  (15): actionId frequencies in {sn < target_sn, same rally}
    hist_point_freq_0..9   (10): pointId  frequencies in same slice
    hist_action_entropy     (1): Shannon entropy of action distribution
    hist_point_entropy      (1): Shannon entropy of point  distribution
    hist_action_dominance   (1): max action frequency (top-1 class weight)
    hist_point_dominance    (1): max point  frequency
    hist_len                (1): total shots in history / max_possible (normalised)
    Total: 30 features
    SN=1 (no history): all values = 0.

  Group 9 — Streak features (rally-internal):
    streak_action    (1): consecutive identical actionId before target
    streak_point     (1): consecutive identical pointId before target
    streak_len       (1): combined rally streak length (max of the two)
    Total: 3 features

Grand total: 1170 + 52 + 30 + 3 = 1255 features.

Fold-safe contract:
  - compute_global_stats_v10(train_fold_raw_df) must be called INSIDE the CV fold loop
    with only the training fold's raw data.  Test inference uses full-train profile.
  - build_features_v10 uses whatever profile is stored in global_stats["v10_profiles"].
  - History freq / streak: computed only from same-rally shots with sn < target sn.
    The target shot's own actionId/pointId is NEVER included.
"""
import numpy as np
import pandas as pd
import sys, os
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v9 import (
    build_features_v9, compute_global_stats_v9, get_feature_names_v9,
)

N_ACT = 15   # actionId classes tracked (0-14; 15-18 are serve-only, rare in non-SN1 rows)
N_PT  = 10   # pointId  classes 0-9
MIN_RALLIES = 5   # minimum rally count to use a player's own profile


# ─────────────────────────────────────────────────────────────────────────────
# Group 7 helpers: player profiles
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sex_marginals(train_df: pd.DataFrame) -> dict:
    """Fallback marginal distributions per sex (1=male, 2=female)."""
    marginals = {}
    for sex_val in [1, 2]:
        sub = train_df[train_df["sex"] == sex_val]
        n   = len(sub)
        if n == 0:
            marginals[sex_val] = {
                "act": np.full(N_ACT, 1.0 / N_ACT, dtype=np.float32),
                "pt":  np.full(N_PT,  1.0 / N_PT,  dtype=np.float32),
                "n_rallies": 0,
            }
            continue
        act_cnts = np.zeros(N_ACT, dtype=np.float64)
        for a in sub["actionId"].values:
            if 0 <= int(a) < N_ACT:
                act_cnts[int(a)] += 1
        act_s = act_cnts.sum()
        act_f = (act_cnts / act_s).astype(np.float32) if act_s > 0 else \
                np.full(N_ACT, 1.0 / N_ACT, dtype=np.float32)

        pt_cnts = np.zeros(N_PT, dtype=np.float64)
        for p in sub["pointId"].values:
            if 0 <= int(p) < N_PT:
                pt_cnts[int(p)] += 1
        pt_s = pt_cnts.sum()
        pt_f = (pt_cnts / pt_s).astype(np.float32) if pt_s > 0 else \
               np.full(N_PT, 1.0 / N_PT, dtype=np.float32)

        marginals[sex_val] = {"act": act_f, "pt": pt_f,
                               "n_rallies": sub["rally_uid"].nunique()}
    return marginals


def compute_player_profiles(train_df: pd.DataFrame) -> dict:
    """Compute per-player action/point frequency profiles from train_df.

    NO serverGetPoint-derived features.  Returns:
      {
        "by_player": {pid: {"act": [N_ACT], "pt": [N_PT], "n_rallies": int}},
        "sex_marginals": {sex_val: {"act": ..., "pt": ..., "n_rallies": int}},
      }
    """
    sex_marginals = _compute_sex_marginals(train_df)
    by_player = {}
    for pid, grp in train_df.groupby("gamePlayerId", sort=False):
        n_rallies = grp["rally_uid"].nunique()
        act_cnts = np.zeros(N_ACT, dtype=np.float64)
        for a in grp["actionId"].values:
            if 0 <= int(a) < N_ACT:
                act_cnts[int(a)] += 1
        act_s = act_cnts.sum()
        act_f = (act_cnts / act_s).astype(np.float32) if act_s > 0 else \
                np.full(N_ACT, 1.0 / N_ACT, dtype=np.float32)

        pt_cnts = np.zeros(N_PT, dtype=np.float64)
        for p in grp["pointId"].values:
            if 0 <= int(p) < N_PT:
                pt_cnts[int(p)] += 1
        pt_s = pt_cnts.sum()
        pt_f = (pt_cnts / pt_s).astype(np.float32) if pt_s > 0 else \
               np.full(N_PT, 1.0 / N_PT, dtype=np.float32)

        by_player[int(pid)] = {"act": act_f, "pt": pt_f, "n_rallies": n_rallies}
    return {"by_player": by_player, "sex_marginals": sex_marginals}


def _get_profile(pid, sex, profiles):
    """Return (act_freq[N_ACT], pt_freq[N_PT], n_rallies) for player pid.
    Falls back to sex-level marginal if unseen or sparse."""
    by_player = profiles["by_player"]
    fallback  = profiles["sex_marginals"].get(
        int(sex), profiles["sex_marginals"].get(1))
    p = by_player.get(int(pid))
    if p is None or p["n_rallies"] < MIN_RALLIES:
        return fallback["act"], fallback["pt"], 0
    return p["act"], p["pt"], p["n_rallies"]


# ─────────────────────────────────────────────────────────────────────────────
# Group 8 helpers: history class frequencies
# ─────────────────────────────────────────────────────────────────────────────

def _safe_entropy(freq):
    """Shannon entropy (nats) of a probability vector. Returns 0 for empty."""
    freq = np.asarray(freq, dtype=np.float64)
    freq = freq[freq > 0]
    if len(freq) == 0:
        return 0.0
    return float(scipy_entropy(freq))


def _build_hist_features(raw_df: pd.DataFrame,
                          target_rally_uid: np.ndarray,
                          target_sn: np.ndarray) -> np.ndarray:
    """For each target row, compute history-class-frequency features using only
    shots in the same rally with strikeNumber < target strikeNumber.

    Returns array of shape (n_targets, 30):
      cols 0-14  : hist_action_freq_0..14
      cols 15-24 : hist_point_freq_0..9
      col  25    : hist_action_entropy
      col  26    : hist_point_entropy
      col  27    : hist_action_dominance
      col  28    : hist_point_dominance
      col  29    : hist_len (count / 20 capped, normalised)
    """
    n = len(target_rally_uid)
    out = np.zeros((n, 30), dtype=np.float32)

    # Build a dict: rally_uid -> sorted list of (sn, actionId, pointId)
    rally_shots: dict = {}
    for row in raw_df[["rally_uid", "strikeNumber", "actionId", "pointId"]].itertuples(index=False):
        uid = row.rally_uid
        if uid not in rally_shots:
            rally_shots[uid] = []
        rally_shots[uid].append((int(row.strikeNumber), int(row.actionId), int(row.pointId)))
    # sort each rally by sn
    for uid in rally_shots:
        rally_shots[uid].sort()

    for i in range(n):
        uid = target_rally_uid[i]
        sn  = int(target_sn[i])
        shots = rally_shots.get(uid, [])

        act_cnts = np.zeros(N_ACT, dtype=np.float64)
        pt_cnts  = np.zeros(N_PT,  dtype=np.float64)
        hist_len = 0
        for (shot_sn, act, pt) in shots:
            if shot_sn >= sn:
                break            # sorted, so we can break early
            if 0 <= act < N_ACT:
                act_cnts[act] += 1
            if 0 <= pt < N_PT:
                pt_cnts[pt] += 1
            hist_len += 1

        if hist_len == 0:
            # SN=1 or no history: all zeros (already initialised)
            continue

        act_freq = (act_cnts / hist_len).astype(np.float32)
        pt_freq  = (pt_cnts  / hist_len).astype(np.float32)

        out[i, 0:N_ACT]        = act_freq
        out[i, N_ACT:N_ACT+N_PT] = pt_freq
        out[i, 25] = float(_safe_entropy(act_freq))
        out[i, 26] = float(_safe_entropy(pt_freq))
        out[i, 27] = float(act_freq.max())
        out[i, 28] = float(pt_freq.max())
        out[i, 29] = min(hist_len / 20.0, 1.0)   # normalised, capped at 20

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Group 9 helpers: streak features
# ─────────────────────────────────────────────────────────────────────────────

def _build_streak_features(raw_df: pd.DataFrame,
                            target_rally_uid: np.ndarray,
                            target_sn: np.ndarray) -> np.ndarray:
    """Compute streak_action, streak_point, streak_len for each target row.

    streak_action: length of the longest consecutive run of identical actionId
                   immediately before the target shot (0 if no history).
    streak_point:  same for pointId.
    streak_len:    max(streak_action, streak_point).

    Returns array of shape (n_targets, 3).
    """
    n = len(target_rally_uid)
    out = np.zeros((n, 3), dtype=np.float32)

    # Build rally_shots dict: rally_uid -> sorted [(sn, actionId, pointId)]
    rally_shots: dict = {}
    for row in raw_df[["rally_uid", "strikeNumber", "actionId", "pointId"]].itertuples(index=False):
        uid = row.rally_uid
        if uid not in rally_shots:
            rally_shots[uid] = []
        rally_shots[uid].append((int(row.strikeNumber), int(row.actionId), int(row.pointId)))
    for uid in rally_shots:
        rally_shots[uid].sort()

    for i in range(n):
        uid = target_rally_uid[i]
        sn  = int(target_sn[i])
        shots = rally_shots.get(uid, [])

        history = [(act, pt) for (shot_sn, act, pt) in shots if shot_sn < sn]
        if not history:
            continue

        # streak_action: scan backwards
        last_act = history[-1][0]
        streak_a = 0
        for (act, _) in reversed(history):
            if act == last_act:
                streak_a += 1
            else:
                break

        # streak_point: scan backwards
        last_pt = history[-1][1]
        streak_p = 0
        for (_, pt) in reversed(history):
            if pt == last_pt:
                streak_p += 1
            else:
                break

        out[i, 0] = float(streak_a)
        out[i, 1] = float(streak_p)
        out[i, 2] = float(max(streak_a, streak_p))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_stats_v10(train_df: pd.DataFrame) -> dict:
    """Build V9 stats + V10 player profiles.

    IMPORTANT: call this INSIDE the fold loop with train_fold_raw_df only.
    At test time, call with the full training dataframe.
    """
    stats = compute_global_stats_v9(train_df)
    stats["v10_profiles"] = compute_player_profiles(train_df)
    return stats


def get_feature_names_v10(feat_df: pd.DataFrame) -> list:
    """Delegate to V9 → V7 → V6 exclusion filter (removes y_*, metadata, SGP proxies)."""
    return get_feature_names_v9(feat_df)


def build_features_v10(df: pd.DataFrame, is_train: bool,
                        global_stats_v10: dict,
                        raw_df: pd.DataFrame = None,
                        include_player_profile: bool = True,
                        include_hist_freq: bool = True,
                        include_streak: bool = True) -> pd.DataFrame:
    """Build V9 features + (optional) player profiles, history class freq, streaks.

    Ablation flags allow disabling individual feature groups for V15 ablation.
    """
    feat_df = build_features_v9(df, is_train=is_train,
                                  global_stats_v9=global_stats_v10,
                                  raw_df=raw_df)
    if raw_df is None:
        raw_df = df

    n = len(feat_df)

    # ── Group 7: player profiles ─────────────────────────────────────────────
    if include_player_profile:
        profiles = global_stats_v10["v10_profiles"]
        pp_act  = np.zeros((n, N_ACT), dtype=np.float32)
        pp_pt   = np.zeros((n, N_PT),  dtype=np.float32)
        opp_act = np.zeros((n, N_ACT), dtype=np.float32)
        opp_pt  = np.zeros((n, N_PT),  dtype=np.float32)
        pp_n    = np.zeros(n, dtype=np.float32)
        opp_n   = np.zeros(n, dtype=np.float32)

        pid_arr  = feat_df["gamePlayerId"].values  if "gamePlayerId"      in feat_df.columns else np.zeros(n)
        opid_arr = feat_df["gamePlayerOtherId"].values if "gamePlayerOtherId" in feat_df.columns else np.zeros(n)
        sex_arr  = feat_df["sex"].values if "sex" in feat_df.columns else np.ones(n, int)

        for i in range(n):
            a_f, p_f, nr = _get_profile(pid_arr[i], sex_arr[i], profiles)
            pp_act[i] = a_f; pp_pt[i] = p_f; pp_n[i] = nr

            a_f2, p_f2, nr2 = _get_profile(opid_arr[i], sex_arr[i], profiles)
            opp_act[i] = a_f2; opp_pt[i] = p_f2; opp_n[i] = nr2

        for c in range(N_ACT):
            feat_df[f"pp_act_freq_{c}"]  = pp_act[:, c]
            feat_df[f"opp_act_freq_{c}"] = opp_act[:, c]
        for c in range(N_PT):
            feat_df[f"pp_pt_freq_{c}"]  = pp_pt[:, c]
            feat_df[f"opp_pt_freq_{c}"] = opp_pt[:, c]
        feat_df["pp_n_rallies"]  = pp_n
        feat_df["opp_n_rallies"] = opp_n

    # ── Group 8: history class frequencies ──────────────────────────────────
    if include_hist_freq:
        target_uid = feat_df["rally_uid"].values
        target_sn  = feat_df["next_strikeNumber"].values.astype(int)

        hist_feat = _build_hist_features(raw_df, target_uid, target_sn)

        for c in range(N_ACT):
            feat_df[f"hist_act_freq_{c}"] = hist_feat[:, c]
        for c in range(N_PT):
            feat_df[f"hist_pt_freq_{c}"]  = hist_feat[:, N_ACT + c]
        feat_df["hist_act_entropy"]    = hist_feat[:, 25]
        feat_df["hist_pt_entropy"]     = hist_feat[:, 26]
        feat_df["hist_act_dominance"]  = hist_feat[:, 27]
        feat_df["hist_pt_dominance"]   = hist_feat[:, 28]
        feat_df["hist_len_norm"]       = hist_feat[:, 29]

    # ── Group 9: streak features ─────────────────────────────────────────────
    if include_streak:
        target_uid = feat_df["rally_uid"].values
        target_sn  = feat_df["next_strikeNumber"].values.astype(int)
        streak_feat = _build_streak_features(raw_df, target_uid, target_sn)
        feat_df["streak_action"] = streak_feat[:, 0]
        feat_df["streak_point"]  = streak_feat[:, 1]
        feat_df["streak_len"]    = streak_feat[:, 2]

    return feat_df
