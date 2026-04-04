"""Feature engineering V4: extends V3 with ~70 additional features.

Adds target-encoding features, rally-level serverGetPoint predictors,
and 3-way interaction features on top of V3's 802 features.
All new features are computed vectorised on the full DataFrame.
"""
import numpy as np
import pandas as pd
from collections import defaultdict

from features_v3 import build_features_v3, compute_global_stats, get_feature_names_v3
from config import (
    N_ACTION_CLASSES, N_POINT_CLASSES,
    ACTION_ATTACK, ACTION_CONTROL, ACTION_DEFENSE, ACTION_SERVE,
)


# ======================= GLOBAL STATS V4 ====================================

def compute_global_stats_v4(train_df):
    """Compute V3 stats + additional target-encoding tables and rally-level stats.

    Returns a dict with all V3 stats plus V4-specific tables.
    """
    # Start with V3 stats
    stats = compute_global_stats(train_df)

    # ----- Target encoding tables -----
    # For categorical targets (actionId, pointId): use coarse category encoding
    #   action -> attack_rate (proportion of attack actions in group)
    #   point -> long_rate (proportion of long points 7/8/9 in group)
    # This avoids treating nominal class IDs as ordinal numbers.

    # Coarse category definitions
    ATTACK_SET = {1, 2, 3, 4, 5, 6, 7}
    LONG_SET = {7, 8, 9}

    global_attack_rate = train_df["actionId"].isin(ATTACK_SET).mean()
    global_long_rate = train_df["pointId"].isin(LONG_SET).mean()
    global_sgp_mean = train_df.groupby("rally_uid")["serverGetPoint"].first().mean()

    SMOOTH_K = 20  # smoothing factor for target encoding

    # 1. gamePlayerId -> attack_rate, long_rate (coarse TE)
    player_action_te = {}
    player_point_te = {}
    for pid, grp in train_df.groupby("gamePlayerId"):
        n = len(grp)
        smooth = n / (n + SMOOTH_K)
        player_action_te[int(pid)] = smooth * grp["actionId"].isin(ATTACK_SET).mean() + (1 - smooth) * global_attack_rate
        player_point_te[int(pid)] = smooth * grp["pointId"].isin(LONG_SET).mean() + (1 - smooth) * global_long_rate

    stats["te_player_action"] = player_action_te
    stats["te_player_point"] = player_point_te

    # 2. positionId x handId cross -> attack_rate, long_rate
    pos_hand_action_te = {}
    pos_hand_point_te = {}
    for (pos, hand), grp in train_df.groupby(["positionId", "handId"]):
        n = len(grp)
        smooth = n / (n + SMOOTH_K)
        key = (int(pos), int(hand))
        pos_hand_action_te[key] = smooth * grp["actionId"].isin(ATTACK_SET).mean() + (1 - smooth) * global_attack_rate
        pos_hand_point_te[key] = smooth * grp["pointId"].isin(LONG_SET).mean() + (1 - smooth) * global_long_rate
    stats["te_pos_hand_action"] = pos_hand_action_te
    stats["te_pos_hand_point"] = pos_hand_point_te

    # 3. strikeNumber -> attack_rate, long_rate
    sn_action_te = {}
    sn_point_te = {}
    for sn, grp in train_df.groupby("strikeNumber"):
        n = len(grp)
        smooth = n / (n + SMOOTH_K)
        sn_action_te[int(sn)] = smooth * grp["actionId"].isin(ATTACK_SET).mean() + (1 - smooth) * global_attack_rate
        sn_point_te[int(sn)] = smooth * grp["pointId"].isin(LONG_SET).mean() + (1 - smooth) * global_long_rate
    stats["te_sn_action"] = sn_action_te
    stats["te_sn_point"] = sn_point_te

    # 4. numberGame -> mean serverGetPoint
    game_sgp_te = {}
    rally_first = train_df.groupby("rally_uid").first().reset_index()
    for ng, grp in rally_first.groupby("numberGame"):
        n = len(grp)
        smooth = n / (n + SMOOTH_K)
        game_sgp_te[int(ng)] = smooth * grp["serverGetPoint"].mean() + (1 - smooth) * global_sgp_mean
    stats["te_game_sgp"] = game_sgp_te

    # 5. score_diff bins -> mean serverGetPoint
    rally_first["score_diff"] = rally_first["scoreSelf"] - rally_first["scoreOther"]
    bins = [-999, -5, -3, -1, 0, 1, 3, 5, 999]
    labels = list(range(len(bins) - 1))
    rally_first["sd_bin"] = pd.cut(rally_first["score_diff"], bins=bins, labels=labels).astype(int)
    sd_bin_sgp_te = {}
    for b, grp in rally_first.groupby("sd_bin"):
        n = len(grp)
        smooth = n / (n + SMOOTH_K)
        sd_bin_sgp_te[int(b)] = smooth * grp["serverGetPoint"].mean() + (1 - smooth) * global_sgp_mean
    stats["te_sd_bin_sgp"] = sd_bin_sgp_te
    stats["sd_bins"] = bins  # store bin edges for test-time use

    # ----- Rally-level aggregation stats for serverGetPoint -----

    # Per-player server win rate at different score situations
    # Bin score_total into [0-5], [6-10], [11-15], [16+]
    player_score_sit_wr = defaultdict(lambda: defaultdict(list))
    for _, grp in train_df.groupby("rally_uid"):
        first = grp.iloc[0]
        pid = int(first["gamePlayerId"])
        st = int(first["scoreSelf"]) + int(first["scoreOther"])
        sit = min(st // 5, 3)  # 0,1,2,3
        sgp = int(first["serverGetPoint"])
        player_score_sit_wr[pid][sit].append(sgp)

    player_score_sit_mean = {}
    for pid, sit_dict in player_score_sit_wr.items():
        player_score_sit_mean[pid] = {}
        for sit, vals in sit_dict.items():
            n = len(vals)
            smooth = n / (n + SMOOTH_K)
            player_score_sit_mean[pid][sit] = smooth * np.mean(vals) + (1 - smooth) * global_sgp_mean
    stats["player_score_sit_wr"] = player_score_sit_mean

    # Per-serve-type (serve_actionId) win rate
    serve_type_wr = {}
    for _, grp in train_df.groupby("rally_uid"):
        first = grp.iloc[0]
        sa = int(first["actionId"])
        sgp = int(first["serverGetPoint"])
        if sa not in serve_type_wr:
            serve_type_wr[sa] = []
        serve_type_wr[sa].append(sgp)
    serve_type_wr_mean = {}
    for sa, vals in serve_type_wr.items():
        n = len(vals)
        smooth = n / (n + SMOOTH_K)
        serve_type_wr_mean[sa] = smooth * np.mean(vals) + (1 - smooth) * global_sgp_mean
    stats["serve_type_wr"] = serve_type_wr_mean

    # Rally length -> win rate mapping
    rally_len_wr = defaultdict(list)
    for _, grp in train_df.groupby("rally_uid"):
        rlen = len(grp)
        sgp = int(grp.iloc[0]["serverGetPoint"])
        rally_len_wr[rlen].append(sgp)
    rally_len_wr_mean = {}
    for rlen, vals in rally_len_wr.items():
        n = len(vals)
        smooth = n / (n + SMOOTH_K)
        rally_len_wr_mean[rlen] = smooth * np.mean(vals) + (1 - smooth) * global_sgp_mean
    stats["rally_len_wr"] = rally_len_wr_mean

    # Store global means for fallback
    stats["global_attack_rate"] = global_attack_rate
    stats["global_long_rate"] = global_long_rate
    stats["global_sgp_mean"] = global_sgp_mean

    return stats


# ======================= ADD V4 FEATURES ====================================

def add_v4_features(feat_df, global_stats_v4):
    """Add ~70 new features to the V3 feature DataFrame. Vectorised."""
    df = feat_df.copy()

    # Shorthand accessors
    te_player_action = global_stats_v4["te_player_action"]
    te_player_point = global_stats_v4["te_player_point"]
    te_pos_hand_action = global_stats_v4["te_pos_hand_action"]
    te_pos_hand_point = global_stats_v4["te_pos_hand_point"]
    te_sn_action = global_stats_v4["te_sn_action"]
    te_sn_point = global_stats_v4["te_sn_point"]
    te_game_sgp = global_stats_v4["te_game_sgp"]
    te_sd_bin_sgp = global_stats_v4["te_sd_bin_sgp"]
    sd_bins = global_stats_v4["sd_bins"]
    player_score_sit_wr = global_stats_v4["player_score_sit_wr"]
    serve_type_wr = global_stats_v4["serve_type_wr"]
    rally_len_wr = global_stats_v4["rally_len_wr"]
    global_attack_rate = global_stats_v4["global_attack_rate"]
    global_long_rate = global_stats_v4["global_long_rate"]
    global_sgp_mean = global_stats_v4["global_sgp_mean"]

    # ---------------------------------------------------------------
    # 1. Target encoding features
    # ---------------------------------------------------------------

    # Player -> action/point target encoding (hitter and receiver)
    df["te_hitter_action"] = df["next_hitter_id"].map(te_player_action).fillna(global_attack_rate)
    df["te_hitter_point"] = df["next_hitter_id"].map(te_player_point).fillna(global_long_rate)
    df["te_receiver_action"] = df["next_receiver_id"].map(te_player_action).fillna(global_attack_rate)
    df["te_receiver_point"] = df["next_receiver_id"].map(te_player_point).fillna(global_long_rate)

    # positionId x handId cross target encoding
    # Use lag1 position and lag1 hand (the last observed values)
    if "lag1_positionId" in df.columns and "lag1_handId" in df.columns:
        pos_hand_keys = list(zip(df["lag1_positionId"].values, df["lag1_handId"].values))
        df["te_pos_hand_action"] = [te_pos_hand_action.get((int(p), int(h)), global_attack_rate)
                                     if p >= 0 and h >= 0 else global_attack_rate
                                     for p, h in pos_hand_keys]
        df["te_pos_hand_point"] = [te_pos_hand_point.get((int(p), int(h)), global_long_rate)
                                    if p >= 0 and h >= 0 else global_long_rate
                                    for p, h in pos_hand_keys]
    else:
        df["te_pos_hand_action"] = global_attack_rate
        df["te_pos_hand_point"] = global_long_rate

    # strikeNumber -> action/point target encoding
    df["te_sn_action"] = df["next_strikeNumber"].map(te_sn_action).fillna(global_attack_rate)
    df["te_sn_point"] = df["next_strikeNumber"].map(te_sn_point).fillna(global_long_rate)

    # numberGame -> serverGetPoint target encoding
    df["te_game_sgp"] = df["numberGame"].map(te_game_sgp).fillna(global_sgp_mean)

    # score_diff bin -> serverGetPoint target encoding
    score_diff = df["score_diff"].values
    sd_bin = np.digitize(score_diff, bins=sd_bins[1:-1])  # returns 0..len(bins)-2
    sd_bin = np.clip(sd_bin, 0, len(sd_bins) - 2)
    df["te_sd_bin_sgp"] = [te_sd_bin_sgp.get(int(b), global_sgp_mean) for b in sd_bin]

    # Difference features from target encoding
    df["te_hitter_vs_receiver_action"] = df["te_hitter_action"] - df["te_receiver_action"]
    df["te_hitter_vs_receiver_point"] = df["te_hitter_point"] - df["te_receiver_point"]

    # ---------------------------------------------------------------
    # 2. Rally-level serverGetPoint prediction features
    # ---------------------------------------------------------------

    # Per-player server win rate at score situation
    score_total = df["scoreSelf"].values + df["scoreOther"].values
    score_sit = np.minimum(score_total // 5, 3).astype(int)
    player_ids = df["gamePlayerId"].values.astype(int)

    df["player_score_sit_wr"] = [
        player_score_sit_wr.get(int(pid), {}).get(int(sit), global_sgp_mean)
        for pid, sit in zip(player_ids, score_sit)
    ]

    # Serve type win rate
    df["serve_type_wr"] = df["serve_actionId"].map(serve_type_wr).fillna(global_sgp_mean)

    # Rally length -> win rate
    df["rally_len_wr"] = df["rally_length"].map(rally_len_wr).fillna(global_sgp_mean)

    # Combined server prediction features
    df["sgp_pred_avg"] = (df["te_game_sgp"] + df["te_sd_bin_sgp"] +
                          df["player_score_sit_wr"] + df["serve_type_wr"] +
                          df["rally_len_wr"]) / 5.0

    df["sgp_pred_hitter_wr_x_sit"] = df["hitter_win_rate"] * df["player_score_sit_wr"]
    df["sgp_pred_matchup_x_serve"] = df["matchup_winrate_a"] * df["serve_type_wr"]

    # Score situation indicator
    df["score_situation"] = score_sit

    # ---------------------------------------------------------------
    # 3. Additional interaction features that V3 missed
    # ---------------------------------------------------------------

    # 3a. lag1_action x lag1_hand x lag1_position (3-way)
    lag1_a = df["lag1_actionId"].values if "lag1_actionId" in df.columns else np.full(len(df), -1)
    lag1_h = df["lag1_handId"].values if "lag1_handId" in df.columns else np.full(len(df), -1)
    lag1_p = df["lag1_positionId"].values if "lag1_positionId" in df.columns else np.full(len(df), -1)

    valid_3way = (lag1_a >= 0) & (lag1_h >= 0) & (lag1_p >= 0)
    # Encode as a single int: action * 12 + hand * 4 + position
    inter_3way = np.where(valid_3way,
                          lag1_a.astype(int) * 12 + lag1_h.astype(int) * 4 + lag1_p.astype(int),
                          -1)
    df["inter3_lag1_act_hand_pos"] = inter_3way

    # 3b. serve_action x serve_point x serve_spin (3-way serve combo)
    sa = df["serve_actionId"].values.astype(int)
    sp = df["serve_pointId"].values.astype(int)
    ss = df["serve_spinId"].values.astype(int)
    # Encode: action * 60 + point * 6 + spin
    df["inter3_serve_act_pt_spin"] = sa * 60 + sp * 6 + ss

    # 3c. hitter_attack_rate x receiver_defense_rate
    df["inter_hitter_atk_x_recv_def"] = df["hitter_attack_rate"] * df["receiver_defense_rate"]

    # 3d. score_pressure = is_game_point x abs(score_diff)
    df["score_pressure"] = df["is_game_point"] * df["abs_score_diff"]

    # ---------------------------------------------------------------
    # 4. More interaction & derived features
    # ---------------------------------------------------------------

    # Lag1 action category (attack/control/defense/serve)
    attack_set = np.array(list(ACTION_ATTACK))
    control_set = np.array(list(ACTION_CONTROL))
    defense_set = np.array(list(ACTION_DEFENSE))
    serve_set = np.array(list(ACTION_SERVE))

    lag1_a_int = lag1_a.astype(int)
    df["lag1_is_attack"] = np.isin(lag1_a_int, attack_set).astype(int)
    df["lag1_is_control"] = np.isin(lag1_a_int, control_set).astype(int)
    df["lag1_is_defense"] = np.isin(lag1_a_int, defense_set).astype(int)
    df["lag1_is_serve"] = np.isin(lag1_a_int, serve_set).astype(int)

    # Lag2 action category
    lag2_a = df["lag2_actionId"].values.astype(int) if "lag2_actionId" in df.columns else np.full(len(df), -1)
    df["lag2_is_attack"] = np.isin(lag2_a.astype(int), attack_set).astype(int)
    df["lag2_is_control"] = np.isin(lag2_a.astype(int), control_set).astype(int)

    # Category transition: lag2_cat -> lag1_cat
    lag1_cat = np.where(np.isin(lag1_a_int, attack_set), 0,
               np.where(np.isin(lag1_a_int, control_set), 1,
               np.where(np.isin(lag1_a_int, defense_set), 2,
               np.where(np.isin(lag1_a_int, serve_set), 3, 4))))
    lag2_cat = np.where(np.isin(lag2_a.astype(int), attack_set), 0,
               np.where(np.isin(lag2_a.astype(int), control_set), 1,
               np.where(np.isin(lag2_a.astype(int), defense_set), 2,
               np.where(np.isin(lag2_a.astype(int), serve_set), 3, 4))))
    df["cat_transition_lag2_lag1"] = lag2_cat * 5 + lag1_cat

    # Hitter vs receiver rate differences
    df["hitter_minus_receiver_atk"] = df["hitter_attack_rate"] - df["receiver_attack_rate"]
    df["hitter_minus_receiver_def"] = df["hitter_defense_rate"] - df["receiver_defense_rate"]

    # Win rate difference
    df["wr_diff_hitter_receiver"] = df["hitter_win_rate"] - df["receiver_win_rate"]

    # Rally momentum: attack ratio in rally * score_diff
    df["inter_atk_ratio_x_score_diff"] = df["action_attack_ratio"] * df["score_diff"]

    # Serve combo condensed: serve_action * 10 + serve_point (simpler than 3-way)
    df["serve_combo_act_pt"] = sa * N_POINT_CLASSES + sp

    # Lag1 zone
    lag1_pt = df["lag1_pointId"].values.astype(int) if "lag1_pointId" in df.columns else np.full(len(df), -1)
    lag1_zone = np.where(np.isin(lag1_pt, [1, 2, 3]), 1,
                np.where(np.isin(lag1_pt, [4, 5, 6]), 2,
                np.where(np.isin(lag1_pt, [7, 8, 9]), 3, 0)))

    # Lag1 zone x lag1 action category
    df["inter_lag1_zone_x_cat"] = lag1_zone * 5 + lag1_cat

    # Serve zone
    serve_zone = np.where(np.isin(sp, [1, 2, 3]), 1,
                 np.where(np.isin(sp, [4, 5, 6]), 2,
                 np.where(np.isin(sp, [7, 8, 9]), 3, 0)))
    df["serve_zone"] = serve_zone

    # Score context interactions
    df["inter_score_sit_x_game"] = score_sit * df["numberGame"].values
    df["inter_score_sit_x_rally_len"] = score_sit * df["rally_length"].values

    # Game point pressure interactions
    df["inter_game_pt_x_hitter_wr"] = df["is_game_point"] * df["hitter_win_rate"]
    df["inter_game_pt_x_rally_len"] = df["is_game_point"] * df["rally_length"]
    df["inter_deuce_x_hitter_wr"] = df["is_deuce"] * df["hitter_win_rate"]

    # Serve effectiveness proxy
    df["serve_spin_x_strength"] = ss * df["serve_strengthId"].values

    # Rally length buckets
    rl = df["rally_length"].values
    df["rally_len_bucket"] = np.where(rl <= 2, 0,
                             np.where(rl <= 5, 1,
                             np.where(rl <= 10, 2, 3)))

    # Rally length bucket x next_is_server
    df["inter_rl_bucket_x_server"] = df["rally_len_bucket"] * df["next_is_server"]

    # Parity of next_strikeNumber (already in V3 as next_sn_parity but let's add
    # interaction with rally_phase)
    df["inter_sn_parity_x_phase"] = df["next_sn_parity"] * df["rally_phase"]

    # Difference between hitter and receiver avg strength
    df["str_diff_hitter_receiver"] = df["hitter_avg_strength"] - df["receiver_avg_strength"]

    # TE ratio features
    df["te_action_ratio_h_r"] = df["te_hitter_action"] / (df["te_receiver_action"] + 1e-6)
    df["te_point_ratio_h_r"] = df["te_hitter_point"] / (df["te_receiver_point"] + 1e-6)

    # Combined TE signal
    df["te_combined_action"] = (df["te_hitter_action"] + df["te_sn_action"] + df["te_pos_hand_action"]) / 3.0
    df["te_combined_point"] = (df["te_hitter_point"] + df["te_sn_point"] + df["te_pos_hand_point"]) / 3.0

    # ---------------------------------------------------------------
    # 5. Additional derived features to expand coverage
    # ---------------------------------------------------------------

    # Lag1 strength x lag1 zone interaction
    lag1_str = df["lag1_strengthId"].values.astype(int) if "lag1_strengthId" in df.columns else np.full(len(df), -1)
    lag1_spin = df["lag1_spinId"].values.astype(int) if "lag1_spinId" in df.columns else np.full(len(df), -1)
    df["inter_lag1_str_x_zone"] = np.where(lag1_str >= 0, lag1_str * 4 + lag1_zone, -1)

    # Lag1 spin x lag1 zone
    df["inter_lag1_spin_x_zone"] = np.where(lag1_spin >= 0, lag1_spin * 4 + lag1_zone, -1)

    # Serve spin x serve zone
    df["inter_serve_spin_x_zone"] = ss * 4 + serve_zone

    # Rally length x sex
    df["inter_rally_len_x_sex"] = df["rally_length"] * df["sex"]

    # Number game x next_is_server
    df["inter_game_x_server"] = df["numberGame"] * df["next_is_server"]

    # Score diff x next_is_server
    df["inter_sd_x_server"] = df["score_diff"] * df["next_is_server"]

    # Hitter attack rate x score pressure
    df["inter_atk_rate_x_pressure"] = df["hitter_attack_rate"] * df["score_pressure"]

    # Receiver defense rate x rally length
    df["inter_recv_def_x_rlen"] = df["receiver_defense_rate"] * df["rally_length"]

    # Win rate product (hitter x receiver)
    df["wr_product_h_r"] = df["hitter_win_rate"] * df["receiver_win_rate"]

    # Attack-defense balance in rally context
    df["atk_def_balance"] = df["action_attack_ratio"] - df["action_defense_ratio"]

    # Lag2 zone (if available)
    lag2_pt = df["lag2_pointId"].values.astype(int) if "lag2_pointId" in df.columns else np.full(len(df), -1)
    lag2_zone = np.where(np.isin(lag2_pt, [1, 2, 3]), 1,
                np.where(np.isin(lag2_pt, [4, 5, 6]), 2,
                np.where(np.isin(lag2_pt, [7, 8, 9]), 3, 0)))
    # Zone transition: lag2 -> lag1
    df["zone_trans_lag2_lag1"] = np.where(lag2_pt >= 0, lag2_zone * 4 + lag1_zone, -1)

    # Serve zone x hitter win rate
    df["inter_serve_zone_x_hwr"] = serve_zone * 10 + (df["hitter_win_rate"] * 10).astype(int).clip(0, 9)

    # TE deviation from global mean (how much does this player differ)
    df["te_hitter_action_dev"] = df["te_hitter_action"] - global_attack_rate
    df["te_hitter_point_dev"] = df["te_hitter_point"] - global_long_rate
    df["te_receiver_action_dev"] = df["te_receiver_action"] - global_attack_rate
    df["te_receiver_point_dev"] = df["te_receiver_point"] - global_long_rate

    # SGP prediction variance (how much do the different predictors disagree)
    sgp_cols = np.column_stack([
        df["te_game_sgp"].values,
        df["te_sd_bin_sgp"].values,
        df["player_score_sit_wr"].values,
        df["serve_type_wr"].values,
        df["rally_len_wr"].values,
    ])
    df["sgp_pred_std"] = np.std(sgp_cols, axis=1)

    # Score momentum: are scores converging or diverging?
    df["score_converging"] = (df["abs_score_diff"] <= 1).astype(int)
    df["score_diverging"] = (df["abs_score_diff"] >= 4).astype(int)

    # Next strike number bucket (1=serve, 2=return, 3-4=early rally, 5+=late rally)
    nsn = df["next_strikeNumber"].values
    df["sn_bucket"] = np.where(nsn == 1, 0,
                     np.where(nsn == 2, 1,
                     np.where(nsn <= 4, 2, 3)))

    # sn_bucket x sex
    df["inter_sn_bucket_x_sex"] = df["sn_bucket"] * df["sex"]

    # Rally length x score_diff interaction (nonlinear)
    df["inter_rlen_x_abs_sd"] = df["rally_length"] * df["abs_score_diff"]

    return df


# ======================= MAIN API ===========================================

def build_features_v4(df, is_train=True, global_stats_v4=None):
    """Build V4 feature matrix: V3 features + ~70 new features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data (train or test).
    is_train : bool
        Whether this is training data.
    global_stats_v4 : dict
        Output of compute_global_stats_v4(). Contains both V3 and V4 stats.

    Returns
    -------
    pd.DataFrame
        Feature matrix with V3 + V4 features.
    """
    # Build V3 features (global_stats_v4 is a superset of V3 stats)
    feat_df = build_features_v3(df, is_train=is_train, global_stats=global_stats_v4)

    # Add V4 features on top
    if global_stats_v4 is not None:
        feat_df = add_v4_features(feat_df, global_stats_v4)

    return feat_df


def get_feature_names_v4(feat_df):
    """Return all feature column names (excludes rally_uid, targets, and raw player IDs)."""
    exclude = {"rally_uid", "y_actionId", "y_pointId", "y_serverGetPoint",
               "gamePlayerId", "gamePlayerOtherId", "next_hitter_id", "next_receiver_id"}
    return [c for c in feat_df.columns if c not in exclude]
