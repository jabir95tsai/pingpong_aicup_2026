"""Feature engineering V5: extends V4 with EDA-driven enhanced features.

Adds on top of V4's ~870 features:
- Explicit prev1/prev2 transition pair categoricals
- Phase bucket features (serve/receive/early/mid/late)
- Enhanced score context (pressure, bucket, close)
- Terminal-like signal features (from observable columns only)
- Conservative player features (frequency encoding, rare bucket)
- Markov-style conditional features

All features are safe for train and test (no target leakage).
"""
import numpy as np
import pandas as pd

from features_v4 import (
    build_features_v4, compute_global_stats_v4, get_feature_names_v4,
    add_v4_features,
)
from config import (
    N_ACTION_CLASSES, N_POINT_CLASSES,
    ACTION_ATTACK, ACTION_CONTROL, ACTION_DEFENSE, ACTION_SERVE,
)


# ======================= V5 GLOBAL STATS ====================================

def compute_global_stats_v5(train_df):
    """Compute V4 stats + V5-specific stats for enhanced features.

    Additional stats:
    - Player frequency counts (for frequency encoding)
    - Phase-conditioned transition matrices
    - Second-order Markov tables (bigram -> next action)
    """
    stats = compute_global_stats_v4(train_df)

    # --- Player frequency encoding ---
    player_counts = train_df["gamePlayerId"].value_counts().to_dict()
    other_counts = train_df["gamePlayerOtherId"].value_counts().to_dict()
    # Merge both views
    all_player_counts = {}
    for pid, cnt in player_counts.items():
        all_player_counts[int(pid)] = all_player_counts.get(int(pid), 0) + cnt
    for pid, cnt in other_counts.items():
        all_player_counts[int(pid)] = all_player_counts.get(int(pid), 0) + cnt
    stats["player_freq"] = all_player_counts
    total_player_rows = sum(all_player_counts.values())
    stats["total_player_rows"] = total_player_rows

    # Rare player threshold: players with < 50 appearances
    rare_threshold = 50
    stats["rare_player_set"] = {pid for pid, cnt in all_player_counts.items() if cnt < rare_threshold}
    stats["rare_threshold"] = rare_threshold

    # --- Second-order Markov: (prev2_action, prev1_action) -> next_action distribution ---
    bigram_to_next = {}  # (a2, a1) -> counts array
    rallies = train_df.groupby("rally_uid", sort=False)
    for _, group in rallies:
        group = group.sort_values("strikeNumber")
        actions = group["actionId"].values.astype(int)
        for i in range(2, len(actions)):
            key = (int(actions[i-2]), int(actions[i-1]))
            if key not in bigram_to_next:
                bigram_to_next[key] = np.zeros(N_ACTION_CLASSES, dtype=np.float64)
            a = int(actions[i])
            if a < N_ACTION_CLASSES:
                bigram_to_next[key][a] += 1

    # Normalize
    for key in bigram_to_next:
        total = bigram_to_next[key].sum()
        if total > 0:
            bigram_to_next[key] /= total
    stats["bigram_to_next_action"] = bigram_to_next

    # --- Phase-conditioned action distribution ---
    # Phase: 1=serve(sn=1), 2=receive(sn=2), 3=early(sn=3-4), 4=mid(sn=5-8), 5=late(sn>=9)
    phase_action_dist = {}
    for _, group in rallies:
        group = group.sort_values("strikeNumber")
        sns = group["strikeNumber"].values.astype(int)
        actions = group["actionId"].values.astype(int)
        for i in range(len(actions)):
            sn = sns[i]
            if sn == 1:
                phase = 1
            elif sn == 2:
                phase = 2
            elif sn <= 4:
                phase = 3
            elif sn <= 8:
                phase = 4
            else:
                phase = 5
            if phase not in phase_action_dist:
                phase_action_dist[phase] = np.zeros(N_ACTION_CLASSES, dtype=np.float64)
            if actions[i] < N_ACTION_CLASSES:
                phase_action_dist[phase][actions[i]] += 1

    for phase in phase_action_dist:
        total = phase_action_dist[phase].sum()
        if total > 0:
            phase_action_dist[phase] /= total
    stats["phase_action_dist"] = phase_action_dist

    # --- Phase-conditioned point distribution ---
    phase_point_dist = {}
    for _, group in rallies:
        group = group.sort_values("strikeNumber")
        sns = group["strikeNumber"].values.astype(int)
        points = group["pointId"].values.astype(int)
        for i in range(len(points)):
            sn = sns[i]
            if sn == 1:
                phase = 1
            elif sn == 2:
                phase = 2
            elif sn <= 4:
                phase = 3
            elif sn <= 8:
                phase = 4
            else:
                phase = 5
            if phase not in phase_point_dist:
                phase_point_dist[phase] = np.zeros(N_POINT_CLASSES, dtype=np.float64)
            if points[i] < N_POINT_CLASSES:
                phase_point_dist[phase][points[i]] += 1

    for phase in phase_point_dist:
        total = phase_point_dist[phase].sum()
        if total > 0:
            phase_point_dist[phase] /= total
    stats["phase_point_dist"] = phase_point_dist

    return stats


# ======================= V5 FEATURE ADDITIONS ===============================

def add_v5_features(feat_df, global_stats_v5):
    """Add V5 enhanced features on top of V4 features. Vectorised where possible."""
    df = feat_df.copy()

    player_freq = global_stats_v5.get("player_freq", {})
    total_player_rows = global_stats_v5.get("total_player_rows", 1)
    rare_set = global_stats_v5.get("rare_player_set", set())
    bigram_to_next = global_stats_v5.get("bigram_to_next_action", {})
    phase_action_dist = global_stats_v5.get("phase_action_dist", {})
    phase_point_dist = global_stats_v5.get("phase_point_dist", {})

    nsn = df["next_strikeNumber"].values.astype(int)

    # ---------------------------------------------------------------
    # 1. Phase bucket features
    # ---------------------------------------------------------------
    # Phase bucket for NEXT strike: 0=serve(1), 1=receive(2), 2=early(3-4), 3=mid(5-8), 4=late(9-12), 5=very_late(13+)
    phase_bucket = np.where(nsn == 1, 0,
                   np.where(nsn == 2, 1,
                   np.where(nsn <= 4, 2,
                   np.where(nsn <= 8, 3,
                   np.where(nsn <= 12, 4, 5)))))
    df["v5_phase_bucket"] = phase_bucket

    # Binary phase indicators
    df["v5_is_serve_phase"] = (nsn == 1).astype(int)
    df["v5_is_receive_phase"] = (nsn == 2).astype(int)
    df["v5_is_early_phase"] = ((nsn >= 3) & (nsn <= 4)).astype(int)
    df["v5_is_mid_phase"] = ((nsn >= 5) & (nsn <= 8)).astype(int)
    df["v5_is_late_phase"] = (nsn >= 9).astype(int)

    # Phase x sex interaction
    df["v5_phase_x_sex"] = phase_bucket * 2 + df["sex"].values

    # ---------------------------------------------------------------
    # 2. Enhanced score context features
    # ---------------------------------------------------------------
    score_self = df["scoreSelf"].values.astype(int)
    score_other = df["scoreOther"].values.astype(int)
    score_diff = score_self - score_other
    score_sum = score_self + score_other

    df["v5_scoreDiff"] = score_diff
    df["v5_scoreSum"] = score_sum
    df["v5_isCloseScore"] = (np.abs(score_diff) <= 2).astype(int)
    df["v5_isPressureScore"] = ((score_self >= 9) | (score_other >= 9)).astype(int)
    df["v5_isCriticalPoint"] = ((score_self >= 10) & (score_other >= 10)).astype(int)

    # Score bucket: bin the sum into meaningful ranges
    df["v5_scoreBucket"] = np.where(score_sum <= 4, 0,
                           np.where(score_sum <= 10, 1,
                           np.where(score_sum <= 16, 2,
                           np.where(score_sum <= 20, 3, 4))))

    # Score diff bucket
    df["v5_scoreDiffBucket"] = np.clip(score_diff, -5, 5) + 5  # 0..10

    # Score pressure interactions
    df["v5_pressure_x_phase"] = df["v5_isPressureScore"] * phase_bucket
    df["v5_close_x_phase"] = df["v5_isCloseScore"] * phase_bucket
    df["v5_scoreDiff_x_phase"] = score_diff * phase_bucket

    # ---------------------------------------------------------------
    # 3. Explicit prev1/prev2 transition pair features
    # ---------------------------------------------------------------
    # These are categorical features encoding the transition context
    lag1_a = df["lag1_actionId"].values.astype(int) if "lag1_actionId" in df.columns else np.full(len(df), -1)
    lag2_a = df["lag2_actionId"].values.astype(int) if "lag2_actionId" in df.columns else np.full(len(df), -1)
    lag1_p = df["lag1_pointId"].values.astype(int) if "lag1_pointId" in df.columns else np.full(len(df), -1)
    lag2_p = df["lag2_pointId"].values.astype(int) if "lag2_pointId" in df.columns else np.full(len(df), -1)
    lag1_h = df["lag1_handId"].values.astype(int) if "lag1_handId" in df.columns else np.full(len(df), -1)
    lag1_spin = df["lag1_spinId"].values.astype(int) if "lag1_spinId" in df.columns else np.full(len(df), -1)
    lag1_str = df["lag1_strengthId"].values.astype(int) if "lag1_strengthId" in df.columns else np.full(len(df), -1)
    lag1_pos = df["lag1_positionId"].values.astype(int) if "lag1_positionId" in df.columns else np.full(len(df), -1)
    lag2_h = df["lag2_handId"].values.astype(int) if "lag2_handId" in df.columns else np.full(len(df), -1)
    lag2_spin = df["lag2_spinId"].values.astype(int) if "lag2_spinId" in df.columns else np.full(len(df), -1)
    lag2_str = df["lag2_strengthId"].values.astype(int) if "lag2_strengthId" in df.columns else np.full(len(df), -1)
    lag2_pos = df["lag2_positionId"].values.astype(int) if "lag2_positionId" in df.columns else np.full(len(df), -1)

    # prev2 -> prev1 action pair (second-order Markov key)
    valid_bigram = (lag2_a >= 0) & (lag1_a >= 0)
    df["v5_prev2_prev1_action_pair"] = np.where(valid_bigram,
                                                  lag2_a * N_ACTION_CLASSES + lag1_a, -1)

    # prev2 -> prev1 point pair
    valid_pt_bigram = (lag2_p >= 0) & (lag1_p >= 0)
    df["v5_prev2_prev1_point_pair"] = np.where(valid_pt_bigram,
                                                lag2_p * N_POINT_CLASSES + lag1_p, -1)

    # prev1 action x prev1 point
    valid_ap = (lag1_a >= 0) & (lag1_p >= 0)
    df["v5_prev1_action_point"] = np.where(valid_ap,
                                            lag1_a * N_POINT_CLASSES + lag1_p, -1)

    # prev1 action x prev1 spin
    valid_as = (lag1_a >= 0) & (lag1_spin >= 0)
    df["v5_prev1_action_spin"] = np.where(valid_as,
                                           lag1_a * 6 + lag1_spin, -1)

    # prev1 point x prev1 position
    valid_pp = (lag1_p >= 0) & (lag1_pos >= 0)
    df["v5_prev1_point_position"] = np.where(valid_pp,
                                              lag1_p * 4 + lag1_pos, -1)

    # prev1 hand x prev1 position
    valid_hp = (lag1_h >= 0) & (lag1_pos >= 0)
    df["v5_prev1_hand_position"] = np.where(valid_hp,
                                             lag1_h * 4 + lag1_pos, -1)

    # prev1 strength x prev1 spin
    valid_ss = (lag1_str >= 0) & (lag1_spin >= 0)
    df["v5_prev1_strength_spin"] = np.where(valid_ss,
                                             lag1_str * 6 + lag1_spin, -1)

    # Phase x prev1 action (what action category given phase)
    df["v5_phase_x_prev1_action"] = np.where(lag1_a >= 0,
                                              phase_bucket * N_ACTION_CLASSES + lag1_a, -1)

    # Phase x prev1 point
    df["v5_phase_x_prev1_point"] = np.where(lag1_p >= 0,
                                             phase_bucket * N_POINT_CLASSES + lag1_p, -1)

    # ---------------------------------------------------------------
    # 4. Second-order Markov features (bigram -> next action prediction)
    # ---------------------------------------------------------------
    # Top predicted action from (prev2, prev1) bigram
    bigram_keys = list(zip(lag2_a, lag1_a))
    markov2_top1 = np.full(len(df), -1, dtype=int)
    markov2_top1_prob = np.zeros(len(df), dtype=np.float32)
    markov2_entropy = np.zeros(len(df), dtype=np.float32)

    for i, (a2, a1) in enumerate(bigram_keys):
        if a2 >= 0 and a1 >= 0:
            key = (int(a2), int(a1))
            if key in bigram_to_next:
                dist = bigram_to_next[key]
                markov2_top1[i] = int(np.argmax(dist))
                markov2_top1_prob[i] = float(np.max(dist))
                nonzero = dist[dist > 0]
                if len(nonzero) > 0:
                    markov2_entropy[i] = float(-np.sum(nonzero * np.log(nonzero + 1e-10)))

    df["v5_markov2_top1"] = markov2_top1
    df["v5_markov2_top1_prob"] = markov2_top1_prob
    df["v5_markov2_entropy"] = markov2_entropy

    # ---------------------------------------------------------------
    # 5. Phase-conditioned distribution features
    # ---------------------------------------------------------------
    # What's the expected action/point distribution for the next strike's phase?
    next_phase = np.where(nsn == 1, 1,
                 np.where(nsn == 2, 2,
                 np.where(nsn <= 4, 3,
                 np.where(nsn <= 8, 4, 5))))

    phase_top1_action = np.full(len(df), -1, dtype=int)
    phase_top1_action_prob = np.zeros(len(df), dtype=np.float32)
    phase_top1_point = np.full(len(df), -1, dtype=int)
    phase_top1_point_prob = np.zeros(len(df), dtype=np.float32)

    for i in range(len(df)):
        p = int(next_phase[i])
        if p in phase_action_dist:
            dist = phase_action_dist[p]
            phase_top1_action[i] = int(np.argmax(dist))
            phase_top1_action_prob[i] = float(np.max(dist))
        if p in phase_point_dist:
            dist = phase_point_dist[p]
            phase_top1_point[i] = int(np.argmax(dist))
            phase_top1_point_prob[i] = float(np.max(dist))

    df["v5_phase_top1_action"] = phase_top1_action
    df["v5_phase_top1_action_prob"] = phase_top1_action_prob
    df["v5_phase_top1_point"] = phase_top1_point
    df["v5_phase_top1_point_prob"] = phase_top1_point_prob

    # ---------------------------------------------------------------
    # 6. Terminal-like features (from observable columns only)
    # ---------------------------------------------------------------
    # These detect patterns that correlate with end-of-rally
    # IMPORTANT: only use current-row observable features, no target leakage
    # lag1 values of 0 for hand/strength/spin suggest terminal state of PREVIOUS strike

    df["v5_lag1_zero_hand"] = (lag1_h == 0).astype(int)
    df["v5_lag1_zero_strength"] = (lag1_str == 0).astype(int)
    df["v5_lag1_zero_spin"] = (lag1_spin == 0).astype(int)
    df["v5_lag1_zero_point"] = (lag1_p == 0).astype(int)

    # Count of zero signals in lag1
    df["v5_lag1_terminal_score"] = (df["v5_lag1_zero_hand"] +
                                     df["v5_lag1_zero_strength"] +
                                     df["v5_lag1_zero_spin"] +
                                     df["v5_lag1_zero_point"])

    # Same for lag2
    df["v5_lag2_zero_hand"] = (lag2_h == 0).astype(int)
    df["v5_lag2_zero_strength"] = (lag2_str == 0).astype(int)
    df["v5_lag2_zero_spin"] = (lag2_spin == 0).astype(int)
    df["v5_lag2_zero_point"] = (lag2_p == 0).astype(int)
    df["v5_lag2_terminal_score"] = (df["v5_lag2_zero_hand"] +
                                     df["v5_lag2_zero_strength"] +
                                     df["v5_lag2_zero_spin"] +
                                     df["v5_lag2_zero_point"])

    # Rally length relative to average (long rally = more likely to end)
    rally_len = df["rally_length"].values.astype(float)
    df["v5_rally_len_relative"] = rally_len / 5.65  # train avg
    df["v5_rally_len_over_avg"] = (rally_len > 5.65).astype(int)

    # ---------------------------------------------------------------
    # 7. Conservative player features
    # ---------------------------------------------------------------
    # Frequency encoding (how common is this player in training data)
    hitter_ids = df["next_hitter_id"].values.astype(int)
    receiver_ids = df["next_receiver_id"].values.astype(int)

    hitter_freq = np.array([player_freq.get(int(pid), 0) for pid in hitter_ids], dtype=np.float32)
    receiver_freq = np.array([player_freq.get(int(pid), 0) for pid in receiver_ids], dtype=np.float32)

    # Log frequency (more stable)
    df["v5_hitter_log_freq"] = np.log1p(hitter_freq)
    df["v5_receiver_log_freq"] = np.log1p(receiver_freq)

    # Normalized frequency
    df["v5_hitter_freq_norm"] = hitter_freq / max(total_player_rows, 1)
    df["v5_receiver_freq_norm"] = receiver_freq / max(total_player_rows, 1)

    # Is rare player
    df["v5_hitter_is_rare"] = np.array([1 if int(pid) in rare_set else 0 for pid in hitter_ids])
    df["v5_receiver_is_rare"] = np.array([1 if int(pid) in rare_set else 0 for pid in receiver_ids])

    # Frequency bucket: 0=very_rare(<20), 1=rare(<100), 2=medium(<500), 3=common(>=500)
    df["v5_hitter_freq_bucket"] = np.where(hitter_freq < 20, 0,
                                  np.where(hitter_freq < 100, 1,
                                  np.where(hitter_freq < 500, 2, 3)))
    df["v5_receiver_freq_bucket"] = np.where(receiver_freq < 20, 0,
                                   np.where(receiver_freq < 100, 1,
                                   np.where(receiver_freq < 500, 2, 3)))

    # Freq difference (experience asymmetry)
    df["v5_freq_diff"] = df["v5_hitter_log_freq"] - df["v5_receiver_log_freq"]

    # ---------------------------------------------------------------
    # 8. Cross-feature interactions (EDA-driven)
    # ---------------------------------------------------------------
    # Phase x score pressure (critical moments differ by phase)
    df["v5_phase_x_pressure"] = phase_bucket * 2 + df["v5_isPressureScore"]

    # Phase x terminal score
    df["v5_phase_x_terminal"] = phase_bucket * 5 + np.clip(df["v5_lag1_terminal_score"].values, 0, 4)

    # Score diff x rally length (longer rallies under pressure)
    df["v5_scoreDiff_x_rallyLen"] = score_diff * rally_len.astype(int)

    # Hitter rare x phase (rare players may differ by phase)
    df["v5_rare_hitter_x_phase"] = df["v5_hitter_is_rare"] * phase_bucket

    # prev1 action category x phase
    attack_set = np.array(list(ACTION_ATTACK))
    control_set = np.array(list(ACTION_CONTROL))
    defense_set = np.array(list(ACTION_DEFENSE))
    serve_set = np.array(list(ACTION_SERVE))

    lag1_cat = np.where(np.isin(lag1_a, attack_set), 0,
               np.where(np.isin(lag1_a, control_set), 1,
               np.where(np.isin(lag1_a, defense_set), 2,
               np.where(np.isin(lag1_a, serve_set), 3, 4))))
    df["v5_lag1_cat_x_phase"] = lag1_cat * 6 + phase_bucket

    # Score bucket x hitter freq bucket
    df["v5_scoreBucket_x_hitterFreq"] = df["v5_scoreBucket"] * 4 + df["v5_hitter_freq_bucket"]

    return df


# ======================= MAIN API ===========================================

def build_features_v5(df, is_train=True, global_stats_v5=None):
    """Build V5 feature matrix: V4 features + V5 enhanced features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data (train or test).
    is_train : bool
        Whether this is training data.
    global_stats_v5 : dict
        Output of compute_global_stats_v5().

    Returns
    -------
    pd.DataFrame
        Feature matrix with V4 + V5 features.
    """
    # Build V4 features (global_stats_v5 is a superset of V4 stats)
    feat_df = build_features_v4(df, is_train=is_train, global_stats_v4=global_stats_v5)

    # Add V5 features on top
    if global_stats_v5 is not None:
        feat_df = add_v5_features(feat_df, global_stats_v5)

    return feat_df


def get_feature_names_v5(feat_df):
    """Return all feature column names (excludes rally_uid, targets, and raw player IDs)."""
    exclude = {"rally_uid", "y_actionId", "y_pointId", "y_serverGetPoint",
               "gamePlayerId", "gamePlayerOtherId", "next_hitter_id", "next_receiver_id"}
    return [c for c in feat_df.columns if c not in exclude]
