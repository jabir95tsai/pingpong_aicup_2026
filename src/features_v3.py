"""Feature engineering V3: 800+ features built on top of V2 features.

Adds one-hot lag encoding, full transition probability vectors, cumulative
counts, per-class player distributions, interaction features, zone features,
momentum features, conditional features, score context, bigram/trigram
frequency features, and expanded serve-return patterns.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from config import (
    LAG_STEPS, LAG_COLS, CATEGORICAL_STRIKE_COLS,
    ACTION_ATTACK, ACTION_CONTROL, ACTION_DEFENSE, ACTION_SERVE,
    N_ACTION_CLASSES, N_POINT_CLASSES,
)

# ---------------------------------------------------------------------------
# Category sizes for one-hot encoding
# ---------------------------------------------------------------------------
_CAT_SIZES = {
    "actionId": N_ACTION_CLASSES,     # 19
    "pointId": N_POINT_CLASSES,       # 10
    "handId": 3,                      # 0,1,2
    "strengthId": 4,                  # 0,1,2,3
    "spinId": 6,                      # 0-5
    "positionId": 4,                  # 0-3
    "strikeId": 5,                    # 0-4 after remap
}

N_ZONES = 4  # 0=none, 1=short, 2=mid, 3=long

# Pre-compute one-hot feature name strings to avoid repeated f-string creation
_OH_LAGS = [1, 2, 3, 5]
_OH_NAMES_CACHE = {}
for _k in _OH_LAGS:
    for _col, _nc in _CAT_SIZES.items():
        _OH_NAMES_CACHE[(_k, _col)] = [f"oh_lag{_k}_{_col}_{_c}" for _c in range(_nc)]

# Pre-compute other repeated feature name lists
_TRANS_ACT_ACT_NAMES = [f"trans_act_act_{a}" for a in range(N_ACTION_CLASSES)]
_TRANS_PT_PT_NAMES = [f"trans_pt_pt_{p}" for p in range(N_POINT_CLASSES)]
_TRANS_POS_ACT_NAMES = [f"trans_pos_act_{a}" for a in range(N_ACTION_CLASSES)]
_TRANS_ACT_PT_NAMES = [f"trans_act_pt_{p}" for p in range(N_POINT_CLASSES)]
_TRANS_PT_ACT_NAMES = [f"trans_pt_act_{a}" for a in range(N_ACTION_CLASSES)]
_HITTER_ACT_NAMES = [f"hitter_act_prob_{a}" for a in range(N_ACTION_CLASSES)]
_HITTER_PT_NAMES = [f"hitter_pt_prob_{p}" for p in range(N_POINT_CLASSES)]
_RECEIVER_ACT_NAMES = [f"receiver_act_prob_{a}" for a in range(N_ACTION_CLASSES)]
_RECEIVER_PT_NAMES = [f"receiver_pt_prob_{p}" for p in range(N_POINT_CLASSES)]
_CTX_ACTION_COUNT_NAMES = [f"ctx_action_count_{a}" for a in range(N_ACTION_CLASSES)]
_CTX_ACTION_FRAC_NAMES = [f"ctx_action_frac_{a}" for a in range(N_ACTION_CLASSES)]
_CTX_POINT_COUNT_NAMES = [f"ctx_point_count_{p}" for p in range(N_POINT_CLASSES)]
_CTX_POINT_FRAC_NAMES = [f"ctx_point_frac_{p}" for p in range(N_POINT_CLASSES)]
_SR_AA_NAMES = [f"sr_aa_{a}" for a in range(N_ACTION_CLASSES)]
_SR_AP_NAMES = [f"sr_ap_{p}" for p in range(N_POINT_CLASSES)]
_SR_PA_NAMES = [f"sr_pa_{a}" for a in range(N_ACTION_CLASSES)]
_SR_SPIN_ACT_NAMES = [f"sr_spin_act_{a}" for a in range(N_ACTION_CLASSES)]
_COND_SERVE_ACT_NAMES = [f"cond_serve_act_{a}" for a in range(N_ACTION_CLASSES)]
_COND_SERVE_PT_NAMES = [f"cond_serve_pt_{p}" for p in range(N_POINT_CLASSES)]
_SERVE_ACT_IS_NAMES = [f"serve_act_is_{a}" for a in range(N_ACTION_CLASSES)]
_ZONE_TRANS_NAMES = [f"zone_trans_{z}" for z in range(N_ZONES)]
_CTX_ZONE_DETAIL_NAMES = [f"ctx_zone_detail_{p}" for p in range(N_POINT_CLASSES)]


def _zone(p):
    if p in (1, 2, 3): return 1  # short
    if p in (4, 5, 6): return 2  # mid
    if p in (7, 8, 9): return 3  # long
    return 0


# ============================= GLOBAL STATS ================================

def compute_global_stats(train_df):
    """Compute global statistics from training data for feature engineering.

    Extends V2 global stats with bigram/trigram frequency tables, zone
    transition matrices, serve-return detailed patterns, and per-player
    full distributions.
    """
    stats = {}

    rallies = train_df.groupby("rally_uid", sort=False)

    # ---------- Transition matrices ----------
    action_trans = np.zeros((N_ACTION_CLASSES, N_ACTION_CLASSES), dtype=np.float64)
    point_trans = np.zeros((N_POINT_CLASSES, N_POINT_CLASSES), dtype=np.float64)
    pos_action = np.zeros((4, N_ACTION_CLASSES), dtype=np.float64)
    action_point = np.zeros((N_ACTION_CLASSES, N_POINT_CLASSES), dtype=np.float64)
    serve_return_action = np.zeros((N_ACTION_CLASSES, N_ACTION_CLASSES), dtype=np.float64)
    zone_trans = np.zeros((N_ZONES, N_ZONES), dtype=np.float64)
    point_action_trans = np.zeros((N_POINT_CLASSES, N_ACTION_CLASSES), dtype=np.float64)

    # Serve-return expanded: serve_action -> return_point, serve_point -> return_action
    serve_action_to_return_point = np.zeros((N_ACTION_CLASSES, N_POINT_CLASSES), dtype=np.float64)
    serve_point_to_return_action = np.zeros((N_POINT_CLASSES, N_ACTION_CLASSES), dtype=np.float64)
    serve_spin_to_return_action = np.zeros((6, N_ACTION_CLASSES), dtype=np.float64)
    serve_action_to_return_action = np.zeros((N_ACTION_CLASSES, N_ACTION_CLASSES), dtype=np.float64)

    # Bigram / trigram counters
    action_bigram_counts = defaultdict(int)
    action_trigram_counts = defaultdict(int)
    point_bigram_counts = defaultdict(int)
    total_bigrams = 0
    total_trigrams = 0

    # Rally length -> action distribution
    rally_len_action = defaultdict(list)

    for _, group in rallies:
        group = group.sort_values("strikeNumber")
        actions = group["actionId"].values.astype(int)
        points = group["pointId"].values.astype(int)
        positions = group["positionId"].values.astype(int)
        spins = group["spinId"].values.astype(int)

        for i in range(len(actions)):
            a, p, pos = actions[i], points[i], positions[i]
            if pos < 4 and a < N_ACTION_CLASSES:
                pos_action[pos, a] += 1
            if a < N_ACTION_CLASSES and p < N_POINT_CLASSES:
                action_point[a, p] += 1

            if i > 0:
                prev_a, prev_p = int(actions[i-1]), int(points[i-1])
                if prev_a < N_ACTION_CLASSES and a < N_ACTION_CLASSES:
                    action_trans[prev_a, a] += 1
                if prev_p < N_POINT_CLASSES and p < N_POINT_CLASSES:
                    point_trans[prev_p, p] += 1
                if prev_p < N_POINT_CLASSES and a < N_ACTION_CLASSES:
                    point_action_trans[prev_p, a] += 1

                z_prev = _zone(prev_p)
                z_cur = _zone(p)
                zone_trans[z_prev, z_cur] += 1

                # Bigrams
                bg_a = prev_a * 100 + a
                action_bigram_counts[bg_a] += 1
                bg_p = prev_p * 100 + p
                point_bigram_counts[bg_p] += 1
                total_bigrams += 1

                if i == 1:  # serve -> return
                    serve_return_action[prev_a, a] += 1
                    serve_action_to_return_action[prev_a, a] += 1
                    prev_spin = int(spins[i-1])
                    if prev_spin < 6:
                        serve_spin_to_return_action[prev_spin, a] += 1
                    if prev_a < N_ACTION_CLASSES and p < N_POINT_CLASSES:
                        serve_action_to_return_point[prev_a, p] += 1
                    if prev_p < N_POINT_CLASSES and a < N_ACTION_CLASSES:
                        serve_point_to_return_action[prev_p, a] += 1

                if i >= 2:
                    tri_a = int(actions[i-2]) * 10000 + prev_a * 100 + a
                    action_trigram_counts[tri_a] += 1
                    total_trigrams += 1

                rally_len_action[i].append(a)

    # Normalize transition matrices
    for mat in [action_trans, point_trans, pos_action, action_point,
                serve_return_action, zone_trans, point_action_trans,
                serve_action_to_return_point, serve_point_to_return_action,
                serve_spin_to_return_action, serve_action_to_return_action]:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat /= row_sums

    stats["action_trans"] = action_trans
    stats["point_trans"] = point_trans
    stats["pos_action"] = pos_action
    stats["action_point"] = action_point
    stats["serve_return"] = serve_return_action
    stats["zone_trans"] = zone_trans
    stats["point_action_trans"] = point_action_trans
    stats["serve_action_to_return_point"] = serve_action_to_return_point
    stats["serve_point_to_return_action"] = serve_point_to_return_action
    stats["serve_spin_to_return_action"] = serve_spin_to_return_action
    stats["serve_action_to_return_action"] = serve_action_to_return_action

    # Bigram / trigram frequency maps (store raw counts and totals)
    stats["action_bigram_counts"] = dict(action_bigram_counts)
    stats["point_bigram_counts"] = dict(point_bigram_counts)
    stats["action_trigram_counts"] = dict(action_trigram_counts)
    stats["total_bigrams"] = max(total_bigrams, 1)
    stats["total_trigrams"] = max(total_trigrams, 1)

    # Global distributions
    stats["global_action_dist"] = np.bincount(
        train_df["actionId"].values, minlength=N_ACTION_CLASSES).astype(float)
    stats["global_action_dist"] /= max(stats["global_action_dist"].sum(), 1)
    stats["global_point_dist"] = np.bincount(
        train_df["pointId"].values, minlength=N_POINT_CLASSES).astype(float)
    stats["global_point_dist"] /= max(stats["global_point_dist"].sum(), 1)

    # Per-player detailed stats
    player_stats = {}
    for pid, grp in train_df.groupby("gamePlayerId"):
        pid = int(pid)
        actions = grp["actionId"].values
        points = grp["pointId"].values
        n = max(len(actions), 1)

        action_dist = np.bincount(actions, minlength=N_ACTION_CLASSES).astype(float)
        action_dist /= max(action_dist.sum(), 1)
        point_dist = np.bincount(points, minlength=N_POINT_CLASSES).astype(float)
        point_dist /= max(point_dist.sum(), 1)

        rally_results = grp.groupby("rally_uid")["serverGetPoint"].first()
        win_rate = float(rally_results.mean()) if len(rally_results) > 0 else 0.5

        pos_dists = {}
        for pos in range(4):
            mask = grp["positionId"] == pos
            if mask.sum() > 0:
                d = np.bincount(grp.loc[mask, "actionId"].values, minlength=N_ACTION_CLASSES).astype(float)
                d /= d.sum()
                pos_dists[pos] = d

        serve_mask = grp["strikeId"] == 0  # 0=serve only (after remap)
        serve_actions = grp.loc[serve_mask, "actionId"].values if serve_mask.sum() > 0 else np.array([])

        serve_action_dist = (np.bincount(serve_actions.astype(int), minlength=N_ACTION_CLASSES).astype(float)
                             / max(len(serve_actions), 1)) if len(serve_actions) > 0 else np.zeros(N_ACTION_CLASSES)

        serve_points = grp.loc[serve_mask, "pointId"].values if serve_mask.sum() > 0 else np.array([])
        serve_point_dist = (np.bincount(serve_points.astype(int), minlength=N_POINT_CLASSES).astype(float)
                            / max(len(serve_points), 1)) if len(serve_points) > 0 else np.zeros(N_POINT_CLASSES)

        player_stats[pid] = {
            "action_dist": action_dist,
            "point_dist": point_dist,
            "top_action": int(np.argmax(action_dist)),
            "top_point": int(np.argmax(point_dist)),
            "attack_rate": sum(1 for a in actions if a in ACTION_ATTACK) / n,
            "control_rate": sum(1 for a in actions if a in ACTION_CONTROL) / n,
            "defense_rate": sum(1 for a in actions if a in ACTION_DEFENSE) / n,
            "win_rate": win_rate,
            "n_rallies": len(rally_results),
            "avg_strength": float(grp["strengthId"].mean()),
            "avg_spin": float(grp["spinId"].mean()),
            "pos_dists": pos_dists,
            "serve_action_dist": serve_action_dist,
            "serve_point_dist": serve_point_dist,
        }

    stats["player_stats"] = player_stats

    # Matchup stats
    matchup_stats = defaultdict(lambda: {"wins_a": 0, "total": 0})
    for _, grp in rallies:
        grp = grp.sort_values("strikeNumber")
        first = grp.iloc[0]
        pa, pb = int(first["gamePlayerId"]), int(first["gamePlayerOtherId"])
        key = (min(pa, pb), max(pa, pb))
        matchup_stats[key]["total"] += 1
        if pa == key[0]:
            matchup_stats[key]["wins_a"] += int(first["serverGetPoint"])
        else:
            matchup_stats[key]["wins_a"] += 1 - int(first["serverGetPoint"])
    stats["matchup_stats"] = dict(matchup_stats)

    # Score diff -> win rate
    score_diff_wins = defaultdict(list)
    for _, grp in rallies:
        first = grp.iloc[0]
        diff = int(first["scoreSelf"]) - int(first["scoreOther"])
        score_diff_wins[diff].append(int(first["serverGetPoint"]))
    stats["score_diff_win_rate"] = {k: np.mean(v) for k, v in score_diff_wins.items()}

    # Strike number -> action distribution
    stats["strike_num_action_dist"] = {}
    for sn, acts in rally_len_action.items():
        d = np.bincount(acts, minlength=N_ACTION_CLASSES).astype(float)
        d /= max(d.sum(), 1)
        stats["strike_num_action_dist"][sn] = d

    return stats


# ============================= SAMPLE BUILDER ==============================

def _build_one_sample(rally_uid, context, target_row, is_train, global_stats):
    """Build feature vector with 800+ features."""
    feat = {"rally_uid": rally_uid}

    # --- Targets ---
    if is_train and target_row is not None:
        feat["y_actionId"] = int(target_row["actionId"])
        feat["y_pointId"] = int(target_row["pointId"])
        feat["y_serverGetPoint"] = int(target_row["serverGetPoint"])

    # Pre-extract all columns as numpy arrays for fast access
    _ctx_actionId = context["actionId"].values.astype(int)
    _ctx_pointId = context["pointId"].values.astype(int)
    _ctx_handId = context["handId"].values.astype(int)
    _ctx_strengthId = context["strengthId"].values.astype(int)
    _ctx_spinId = context["spinId"].values.astype(int)
    _ctx_positionId = context["positionId"].values.astype(int)
    _ctx_strikeId = context["strikeId"].values.astype(int)
    _ctx_sex = context["sex"].values.astype(int)
    _ctx_numberGame = context["numberGame"].values.astype(int)
    _ctx_scoreSelf = context["scoreSelf"].values.astype(int)
    _ctx_scoreOther = context["scoreOther"].values.astype(int)
    _ctx_strikeNumber = context["strikeNumber"].values.astype(int)
    _ctx_gamePlayerId = context["gamePlayerId"].values.astype(int)
    _ctx_gamePlayerOtherId = context["gamePlayerOtherId"].values.astype(int)

    # Column arrays dict for one-hot by name
    _ctx_cols = {
        "actionId": _ctx_actionId, "pointId": _ctx_pointId,
        "handId": _ctx_handId, "strengthId": _ctx_strengthId,
        "spinId": _ctx_spinId, "positionId": _ctx_positionId,
        "strikeId": _ctx_strikeId,
    }
    ctx_len = len(_ctx_actionId)
    n_ctx = max(ctx_len, 1)

    # ===================================================================
    # A. Basic features (same as V2)
    # ===================================================================
    feat["sex"] = int(_ctx_sex[-1])
    feat["numberGame"] = int(_ctx_numberGame[-1])
    feat["scoreSelf"] = int(_ctx_scoreSelf[-1])
    feat["scoreOther"] = int(_ctx_scoreOther[-1])
    feat["score_diff"] = feat["scoreSelf"] - feat["scoreOther"]
    feat["score_total"] = feat["scoreSelf"] + feat["scoreOther"]
    feat["rally_length"] = ctx_len
    feat["next_strikeNumber"] = int(_ctx_strikeNumber[-1]) + 1
    feat["next_is_server"] = 1 if feat["next_strikeNumber"] % 2 == 1 else 0

    if feat["next_strikeNumber"] == 1:
        feat["next_strikeId"] = 1
    elif feat["next_strikeNumber"] == 2:
        feat["next_strikeId"] = 2
    else:
        feat["next_strikeId"] = 4

    # Score pressure
    feat["is_game_point"] = 1 if (feat["scoreSelf"] >= 10 or feat["scoreOther"] >= 10) else 0
    feat["is_deuce"] = 1 if (feat["scoreSelf"] >= 10 and feat["scoreOther"] >= 10 and
                              abs(feat["score_diff"]) <= 1) else 0
    feat["score_max"] = max(feat["scoreSelf"], feat["scoreOther"])
    feat["score_min"] = min(feat["scoreSelf"], feat["scoreOther"])

    # ===================================================================
    # I. Score context (NEW)
    # ===================================================================
    feat["is_leading"] = 1 if feat["score_diff"] > 0 else 0
    feat["is_trailing"] = 1 if feat["score_diff"] < 0 else 0
    feat["is_tied"] = 1 if feat["score_diff"] == 0 else 0
    feat["score_product"] = feat["scoreSelf"] * feat["scoreOther"]
    feat["score_ratio"] = feat["scoreSelf"] / (feat["scoreSelf"] + feat["scoreOther"] + 1)
    feat["game_progress"] = feat["score_total"] / 21.0
    feat["abs_score_diff"] = abs(feat["score_diff"])

    # ===================================================================
    # B. Lag features (raw values, same as V2)
    # ===================================================================
    for k in LAG_STEPS:
        for col in LAG_COLS:
            if ctx_len >= k:
                feat[f"lag{k}_{col}"] = int(_ctx_cols[col][-k])
            else:
                feat[f"lag{k}_{col}"] = -1

    # ===================================================================
    # A-NEW. One-hot encoding of last K strikes (K=1,2,3,5)
    # ===================================================================
    for k in _OH_LAGS:
        for col, n_classes in _CAT_SIZES.items():
            names_list = _OH_NAMES_CACHE[(k, col)]
            if ctx_len >= k:
                val = int(_ctx_cols[col][-k])
                for c in range(n_classes):
                    feat[names_list[c]] = 1 if val == c else 0
            else:
                for c in range(n_classes):
                    feat[names_list[c]] = 0

    # ===================================================================
    # C. Rally-level statistics (V2 base)
    # ===================================================================
    actions = _ctx_actionId
    points = _ctx_pointId
    hands = _ctx_handId
    strengths = _ctx_strengthId
    spins = _ctx_spinId
    positions_arr = _ctx_positionId

    for col in CATEGORICAL_STRIKE_COLS:
        vals = _ctx_cols[col]
        if len(vals) > 0:
            counts = np.bincount(vals)
            feat[f"{col}_mode"] = int(np.argmax(counts))
        else:
            feat[f"{col}_mode"] = -1
        feat[f"{col}_nunique"] = len(set(vals.tolist()))

    for col_name, col_arr in [("strengthId", strengths), ("spinId", spins)]:
        vals = col_arr.astype(float)
        feat[f"{col_name}_mean"] = float(np.mean(vals)) if len(vals) > 0 else 0
        feat[f"{col_name}_std"] = float(np.std(vals)) if len(vals) > 0 else 0

    feat["action_attack_ratio"] = np.isin(actions, list(ACTION_ATTACK)).sum() / n_ctx
    feat["action_control_ratio"] = np.isin(actions, list(ACTION_CONTROL)).sum() / n_ctx
    feat["action_defense_ratio"] = np.isin(actions, list(ACTION_DEFENSE)).sum() / n_ctx
    feat["action_serve_ratio"] = np.isin(actions, list(ACTION_SERVE)).sum() / n_ctx

    if len(hands) > 1:
        feat["hand_alternation"] = np.sum(hands[1:] != hands[:-1]) / (len(hands) - 1)
    else:
        feat["hand_alternation"] = 0

    feat["point_short_ratio"] = np.isin(points, [1, 2, 3]).sum() / n_ctx
    feat["point_mid_ratio"] = np.isin(points, [4, 5, 6]).sum() / n_ctx
    feat["point_long_ratio"] = np.isin(points, [7, 8, 9]).sum() / n_ctx
    feat["point_zero_ratio"] = (points == 0).sum() / n_ctx

    # ===================================================================
    # C-NEW. Cumulative action/point counts and fractions
    # ===================================================================
    action_counts = np.bincount(actions, minlength=N_ACTION_CLASSES)
    point_counts = np.bincount(points, minlength=N_POINT_CLASSES)
    for a in range(N_ACTION_CLASSES):
        feat[_CTX_ACTION_COUNT_NAMES[a]] = int(action_counts[a])
        feat[_CTX_ACTION_FRAC_NAMES[a]] = float(action_counts[a]) / n_ctx
    for p in range(N_POINT_CLASSES):
        feat[_CTX_POINT_COUNT_NAMES[p]] = int(point_counts[p])
        feat[_CTX_POINT_FRAC_NAMES[p]] = float(point_counts[p]) / n_ctx

    # ===================================================================
    # D. Temporal / transition features (V2 base)
    # ===================================================================
    if ctx_len >= 2:
        feat["action_bigram"] = int(actions[-2]) * 100 + int(actions[-1])
        feat["point_bigram"] = int(points[-2]) * 100 + int(points[-1])
        feat["action_changed"] = int(actions[-1] != actions[-2])
        feat["point_changed"] = int(points[-1] != points[-2])
        feat["hand_changed"] = int(hands[-1] != hands[-2])
        feat["point_zone_trend"] = _zone(int(points[-1])) - _zone(int(points[-2]))
    else:
        feat["action_bigram"] = -1
        feat["point_bigram"] = -1
        feat["action_changed"] = 0
        feat["point_changed"] = 0
        feat["hand_changed"] = 0
        feat["point_zone_trend"] = 0

    if ctx_len >= 3:
        feat["action_trigram"] = (int(actions[-3]) * 10000 +
                                   int(actions[-2]) * 100 +
                                   int(actions[-1]))
        feat["point_trigram"] = (int(points[-3]) * 10000 +
                                  int(points[-2]) * 100 +
                                  int(points[-1]))
    else:
        feat["action_trigram"] = -1
        feat["point_trigram"] = -1

    # ===================================================================
    # E. Player features (V2 base + IDs)
    # ===================================================================
    feat["gamePlayerId"] = int(_ctx_gamePlayerId[-1])
    feat["gamePlayerOtherId"] = int(_ctx_gamePlayerOtherId[-1])

    if feat["next_strikeNumber"] % 2 == 1:
        feat["next_hitter_id"] = int(_ctx_gamePlayerId[0])
        feat["next_receiver_id"] = int(_ctx_gamePlayerOtherId[0])
    else:
        feat["next_hitter_id"] = int(_ctx_gamePlayerOtherId[0])
        feat["next_receiver_id"] = int(_ctx_gamePlayerId[0])

    # ===================================================================
    # F. Serve-specific features (V2 base)
    # ===================================================================
    feat["serve_actionId"] = int(_ctx_actionId[0])
    feat["serve_spinId"] = int(_ctx_spinId[0])
    feat["serve_pointId"] = int(_ctx_pointId[0])
    feat["serve_strengthId"] = int(_ctx_strengthId[0])
    feat["serve_positionId"] = int(_ctx_positionId[0])

    # ===================================================================
    # GLOBAL-STATS-DEPENDENT FEATURES
    # ===================================================================
    if global_stats is not None:
        player_stats = global_stats.get("player_stats", {})
        last_action = int(actions[-1])
        last_point = int(points[-1])
        last_position = int(positions_arr[-1])

        # ---------------------------------------------------------------
        # G. Transition probability features (V2 base: top-K)
        # ---------------------------------------------------------------
        action_trans = global_stats["action_trans"]
        if last_action < N_ACTION_CLASSES:
            trans_probs = action_trans[last_action]
            top_actions = np.argsort(trans_probs)[::-1]
            feat["trans_top1_action"] = int(top_actions[0])
            feat["trans_top1_action_prob"] = float(trans_probs[top_actions[0]])
            feat["trans_top2_action"] = int(top_actions[1])
            feat["trans_top2_action_prob"] = float(trans_probs[top_actions[1]])
            feat["trans_top3_action"] = int(top_actions[2])
            feat["trans_entropy_action"] = float(-np.sum(trans_probs[trans_probs > 0] * np.log(trans_probs[trans_probs > 0] + 1e-10)))
        else:
            feat["trans_top1_action"] = -1
            feat["trans_top1_action_prob"] = 0
            feat["trans_top2_action"] = -1
            feat["trans_top2_action_prob"] = 0
            feat["trans_top3_action"] = -1
            feat["trans_entropy_action"] = 0

        point_trans = global_stats["point_trans"]
        if last_point < N_POINT_CLASSES:
            pt_probs = point_trans[last_point]
            top_pts = np.argsort(pt_probs)[::-1]
            feat["trans_top1_point"] = int(top_pts[0])
            feat["trans_top1_point_prob"] = float(pt_probs[top_pts[0]])
            feat["trans_top2_point"] = int(top_pts[1])
            feat["trans_entropy_point"] = float(-np.sum(pt_probs[pt_probs > 0] * np.log(pt_probs[pt_probs > 0] + 1e-10)))
        else:
            feat["trans_top1_point"] = -1
            feat["trans_top1_point_prob"] = 0
            feat["trans_top2_point"] = -1
            feat["trans_entropy_point"] = 0

        pos_action_mat = global_stats["pos_action"]
        if last_position < 4:
            pa_probs = pos_action_mat[last_position]
            feat["pos_top1_action"] = int(np.argmax(pa_probs))
            feat["pos_top1_action_prob"] = float(np.max(pa_probs))
        else:
            feat["pos_top1_action"] = -1
            feat["pos_top1_action_prob"] = 0

        action_point_mat = global_stats["action_point"]
        if last_action < N_ACTION_CLASSES:
            ap_probs = action_point_mat[last_action]
            feat["act_top1_point"] = int(np.argmax(ap_probs))
            feat["act_top1_point_prob"] = float(np.max(ap_probs))
        else:
            feat["act_top1_point"] = -1
            feat["act_top1_point_prob"] = 0

        # Serve -> return (V2)
        if feat["next_strikeNumber"] == 2:
            sr = global_stats["serve_return"]
            sa = int(_ctx_actionId[0])
            if sa < N_ACTION_CLASSES:
                sr_probs = sr[sa]
                feat["serve_return_top1"] = int(np.argmax(sr_probs))
                feat["serve_return_top1_prob"] = float(np.max(sr_probs))
            else:
                feat["serve_return_top1"] = -1
                feat["serve_return_top1_prob"] = 0
        else:
            feat["serve_return_top1"] = -1
            feat["serve_return_top1_prob"] = 0

        # ---------------------------------------------------------------
        # B-NEW. Full transition probability vectors
        # ---------------------------------------------------------------
        # P(next_action=a | last_action) for all a
        if last_action < N_ACTION_CLASSES:
            probs = action_trans[last_action]
            for a in range(N_ACTION_CLASSES):
                feat[_TRANS_ACT_ACT_NAMES[a]] = float(probs[a])
        else:
            for a in range(N_ACTION_CLASSES):
                feat[_TRANS_ACT_ACT_NAMES[a]] = 0.0

        # P(next_point=p | last_point) for all p
        if last_point < N_POINT_CLASSES:
            probs = point_trans[last_point]
            for p in range(N_POINT_CLASSES):
                feat[_TRANS_PT_PT_NAMES[p]] = float(probs[p])
        else:
            for p in range(N_POINT_CLASSES):
                feat[_TRANS_PT_PT_NAMES[p]] = 0.0

        # P(next_action=a | last_position) for all a
        if last_position < 4:
            probs = pos_action_mat[last_position]
            for a in range(N_ACTION_CLASSES):
                feat[_TRANS_POS_ACT_NAMES[a]] = float(probs[a])
        else:
            for a in range(N_ACTION_CLASSES):
                feat[_TRANS_POS_ACT_NAMES[a]] = 0.0

        # P(next_point=p | last_action) for all p
        if last_action < N_ACTION_CLASSES:
            probs = action_point_mat[last_action]
            for p in range(N_POINT_CLASSES):
                feat[_TRANS_ACT_PT_NAMES[p]] = float(probs[p])
        else:
            for p in range(N_POINT_CLASSES):
                feat[_TRANS_ACT_PT_NAMES[p]] = 0.0

        # P(next_action=a | last_point) for all a
        point_action_mat = global_stats["point_action_trans"]
        if last_point < N_POINT_CLASSES:
            probs = point_action_mat[last_point]
            for a in range(N_ACTION_CLASSES):
                feat[_TRANS_PT_ACT_NAMES[a]] = float(probs[a])
        else:
            for a in range(N_ACTION_CLASSES):
                feat[_TRANS_PT_ACT_NAMES[a]] = 0.0

        # ---------------------------------------------------------------
        # D-NEW. Per-class player features (full distributions)
        # ---------------------------------------------------------------
        hitter = feat["next_hitter_id"]
        receiver = feat["next_receiver_id"]

        _default_action_dist = np.zeros(N_ACTION_CLASSES)
        _default_point_dist = np.zeros(N_POINT_CLASSES)

        # Hitter full distributions
        if hitter in player_stats:
            ps = player_stats[hitter]
            h_action_dist = ps["action_dist"]
            h_point_dist = ps["point_dist"]
        else:
            h_action_dist = _default_action_dist
            h_point_dist = _default_point_dist
        for a in range(N_ACTION_CLASSES):
            feat[_HITTER_ACT_NAMES[a]] = float(h_action_dist[a])
        for p in range(N_POINT_CLASSES):
            feat[_HITTER_PT_NAMES[p]] = float(h_point_dist[p])

        # Receiver full distributions
        if receiver in player_stats:
            ps = player_stats[receiver]
            r_action_dist = ps["action_dist"]
            r_point_dist = ps["point_dist"]
        else:
            r_action_dist = _default_action_dist
            r_point_dist = _default_point_dist
        for a in range(N_ACTION_CLASSES):
            feat[_RECEIVER_ACT_NAMES[a]] = float(r_action_dist[a])
        for p in range(N_POINT_CLASSES):
            feat[_RECEIVER_PT_NAMES[p]] = float(r_point_dist[p])

        # ---------------------------------------------------------------
        # H. Player V2 summary features
        # ---------------------------------------------------------------
        if hitter in player_stats:
            ps = player_stats[hitter]
            feat["hitter_top_action"] = ps["top_action"]
            feat["hitter_top_point"] = ps["top_point"]
            feat["hitter_attack_rate"] = ps["attack_rate"]
            feat["hitter_control_rate"] = ps["control_rate"]
            feat["hitter_defense_rate"] = ps["defense_rate"]
            feat["hitter_win_rate"] = ps["win_rate"]
            feat["hitter_n_rallies"] = ps["n_rallies"]
            feat["hitter_avg_strength"] = ps["avg_strength"]
            feat["hitter_avg_spin"] = ps["avg_spin"]

            ad = ps["action_dist"]
            top3 = np.argsort(ad)[::-1][:3]
            feat["hitter_top1_act"] = int(top3[0])
            feat["hitter_top1_act_prob"] = float(ad[top3[0]])
            feat["hitter_top2_act"] = int(top3[1])
            feat["hitter_top2_act_prob"] = float(ad[top3[1]])
            feat["hitter_top3_act"] = int(top3[2])

            ad_pos = ad[ad > 0]
            feat["hitter_action_entropy"] = float(-np.sum(ad_pos * np.log(ad_pos + 1e-10)))

            pd_dist = ps["point_dist"]
            top2_pt = np.argsort(pd_dist)[::-1][:2]
            feat["hitter_top1_pt"] = int(top2_pt[0])
            feat["hitter_top1_pt_prob"] = float(pd_dist[top2_pt[0]])
            feat["hitter_top2_pt"] = int(top2_pt[1])

            if last_position in ps.get("pos_dists", {}):
                ppd = ps["pos_dists"][last_position]
                feat["hitter_pos_top_act"] = int(np.argmax(ppd))
                feat["hitter_pos_top_act_prob"] = float(np.max(ppd))
            else:
                feat["hitter_pos_top_act"] = -1
                feat["hitter_pos_top_act_prob"] = 0

            if feat["next_strikeNumber"] == 1:
                sd = ps["serve_action_dist"]
                feat["hitter_serve_top1"] = int(np.argmax(sd))
                feat["hitter_serve_top1_prob"] = float(np.max(sd))
            else:
                feat["hitter_serve_top1"] = -1
                feat["hitter_serve_top1_prob"] = 0
        else:
            for k in ["hitter_top_action", "hitter_top_point", "hitter_top1_act",
                       "hitter_top2_act", "hitter_top3_act", "hitter_top1_pt",
                       "hitter_top2_pt", "hitter_pos_top_act", "hitter_serve_top1"]:
                feat[k] = -1
            for k in ["hitter_attack_rate", "hitter_control_rate", "hitter_defense_rate",
                       "hitter_top1_act_prob", "hitter_top2_act_prob", "hitter_top1_pt_prob",
                       "hitter_pos_top_act_prob", "hitter_serve_top1_prob",
                       "hitter_avg_strength", "hitter_avg_spin", "hitter_action_entropy"]:
                feat[k] = 0.0
            feat["hitter_win_rate"] = 0.5
            feat["hitter_n_rallies"] = 0

        # Receiver summary
        if receiver in player_stats:
            ps = player_stats[receiver]
            feat["receiver_attack_rate"] = ps["attack_rate"]
            feat["receiver_defense_rate"] = ps["defense_rate"]
            feat["receiver_win_rate"] = ps["win_rate"]
            feat["receiver_avg_strength"] = ps["avg_strength"]
        else:
            feat["receiver_attack_rate"] = 0.0
            feat["receiver_defense_rate"] = 0.0
            feat["receiver_win_rate"] = 0.5
            feat["receiver_avg_strength"] = 0.0

        # ---------------------------------------------------------------
        # Matchup features (V2)
        # ---------------------------------------------------------------
        pa, pb = feat["gamePlayerId"], feat["gamePlayerOtherId"]
        key = (min(pa, pb), max(pa, pb))
        matchup = global_stats.get("matchup_stats", {}).get(key)
        if matchup and matchup["total"] > 0:
            feat["matchup_games"] = matchup["total"]
            feat["matchup_winrate_a"] = matchup["wins_a"] / matchup["total"]
        else:
            feat["matchup_games"] = 0
            feat["matchup_winrate_a"] = 0.5

        # Score diff historical win rate (V2)
        sdwr = global_stats.get("score_diff_win_rate", {})
        feat["score_diff_hist_winrate"] = sdwr.get(feat["score_diff"], 0.5)

        # Strike-number conditioned action distribution (V2)
        sn_dist = global_stats.get("strike_num_action_dist", {})
        nsn = feat["next_strikeNumber"]
        if nsn in sn_dist:
            d = sn_dist[nsn]
            top2 = np.argsort(d)[::-1][:2]
            feat["sn_top1_action"] = int(top2[0])
            feat["sn_top1_action_prob"] = float(d[top2[0]])
            feat["sn_top2_action"] = int(top2[1])
            feat["sn_top2_action_prob"] = float(d[top2[1]])
        else:
            feat["sn_top1_action"] = -1
            feat["sn_top1_action_prob"] = 0
            feat["sn_top2_action"] = -1
            feat["sn_top2_action_prob"] = 0

        # ---------------------------------------------------------------
        # F-NEW. Zone-based features
        # ---------------------------------------------------------------
        zone_trans_mat = global_stats["zone_trans"]
        last_zone = _zone(last_point)
        # Zone transition probs: P(next_zone | last_zone)
        for z in range(N_ZONES):
            feat[_ZONE_TRANS_NAMES[z]] = float(zone_trans_mat[last_zone, z])

        # Zone sequence: last 3 zones
        for k in [1, 2, 3]:
            if ctx_len >= k:
                feat[f"zone_lag{k}"] = _zone(int(_ctx_pointId[-k]))
            else:
                feat[f"zone_lag{k}"] = -1

        # Cross-table zone counts (9 point zones + zone=0)
        for pid_val in range(N_POINT_CLASSES):
            feat[_CTX_ZONE_DETAIL_NAMES[pid_val]] = int(point_counts[pid_val])

        # Zone x hand interaction for last strike
        last_hand = int(hands[-1]) if len(hands) > 0 else 0
        feat["zone_x_hand_last"] = last_zone * 3 + last_hand

        # ---------------------------------------------------------------
        # J-NEW. Bigram/trigram frequency features
        # ---------------------------------------------------------------
        bg_counts_a = global_stats["action_bigram_counts"]
        bg_counts_p = global_stats["point_bigram_counts"]
        tri_counts_a = global_stats["action_trigram_counts"]
        total_bg = global_stats["total_bigrams"]
        total_tri = global_stats["total_trigrams"]

        if ctx_len >= 2:
            bg_key_a = int(actions[-2]) * 100 + int(actions[-1])
            bg_key_p = int(points[-2]) * 100 + int(points[-1])
            feat["bigram_action_freq"] = bg_counts_a.get(bg_key_a, 0) / total_bg
            feat["bigram_point_freq"] = bg_counts_p.get(bg_key_p, 0) / total_bg
            feat["bigram_action_log_freq"] = float(np.log1p(bg_counts_a.get(bg_key_a, 0)))
            feat["bigram_point_log_freq"] = float(np.log1p(bg_counts_p.get(bg_key_p, 0)))
        else:
            feat["bigram_action_freq"] = 0
            feat["bigram_point_freq"] = 0
            feat["bigram_action_log_freq"] = 0
            feat["bigram_point_log_freq"] = 0

        if ctx_len >= 3:
            tri_key_a = int(actions[-3]) * 10000 + int(actions[-2]) * 100 + int(actions[-1])
            feat["trigram_action_freq"] = tri_counts_a.get(tri_key_a, 0) / total_tri
            feat["trigram_action_log_freq"] = float(np.log1p(tri_counts_a.get(tri_key_a, 0)))
        else:
            feat["trigram_action_freq"] = 0
            feat["trigram_action_log_freq"] = 0

        # ---------------------------------------------------------------
        # K-NEW. Serve-return expanded patterns
        # ---------------------------------------------------------------
        sa = int(_ctx_actionId[0])
        sp = int(_ctx_pointId[0])
        s_spin = int(_ctx_spinId[0])

        # Full serve_action -> return_action vector (19 probs)
        sr_aa = global_stats["serve_action_to_return_action"]
        if sa < N_ACTION_CLASSES:
            row = sr_aa[sa]
            for a in range(N_ACTION_CLASSES):
                feat[_SR_AA_NAMES[a]] = float(row[a])
        else:
            for a in range(N_ACTION_CLASSES):
                feat[_SR_AA_NAMES[a]] = 0.0

        # Serve_action -> return_point (10 probs)
        sr_ap = global_stats["serve_action_to_return_point"]
        if sa < N_ACTION_CLASSES:
            row = sr_ap[sa]
            for p in range(N_POINT_CLASSES):
                feat[_SR_AP_NAMES[p]] = float(row[p])
        else:
            for p in range(N_POINT_CLASSES):
                feat[_SR_AP_NAMES[p]] = 0.0

        # Serve_point -> return_action (19 probs)
        sr_pa = global_stats["serve_point_to_return_action"]
        if sp < N_POINT_CLASSES:
            row = sr_pa[sp]
            for a in range(N_ACTION_CLASSES):
                feat[_SR_PA_NAMES[a]] = float(row[a])
        else:
            for a in range(N_ACTION_CLASSES):
                feat[_SR_PA_NAMES[a]] = 0.0

        # Serve_spin -> return_action (19 probs)
        sr_sa = global_stats["serve_spin_to_return_action"]
        if s_spin < 6:
            row = sr_sa[s_spin]
            for a in range(N_ACTION_CLASSES):
                feat[_SR_SPIN_ACT_NAMES[a]] = float(row[a])
        else:
            for a in range(N_ACTION_CLASSES):
                feat[_SR_SPIN_ACT_NAMES[a]] = 0.0

        # ---------------------------------------------------------------
        # H-NEW. Conditional features (serve / return / late rally)
        # ---------------------------------------------------------------
        if feat["next_strikeNumber"] == 1:
            # Serve-specific: hitter's serve distributions
            if hitter in player_stats:
                sd_a = player_stats[hitter]["serve_action_dist"]
                sd_p = player_stats[hitter].get("serve_point_dist", np.zeros(N_POINT_CLASSES))
                for a in range(N_ACTION_CLASSES):
                    feat[_COND_SERVE_ACT_NAMES[a]] = float(sd_a[a])
                for p in range(N_POINT_CLASSES):
                    feat[_COND_SERVE_PT_NAMES[p]] = float(sd_p[p])
            else:
                for a in range(N_ACTION_CLASSES):
                    feat[_COND_SERVE_ACT_NAMES[a]] = 0.0
                for p in range(N_POINT_CLASSES):
                    feat[_COND_SERVE_PT_NAMES[p]] = 0.0
            feat["cond_is_serve"] = 1
            feat["cond_is_return"] = 0
            feat["cond_is_late"] = 0
        elif feat["next_strikeNumber"] == 2:
            for a in range(N_ACTION_CLASSES):
                feat[_COND_SERVE_ACT_NAMES[a]] = 0.0
            for p in range(N_POINT_CLASSES):
                feat[_COND_SERVE_PT_NAMES[p]] = 0.0
            feat["cond_is_serve"] = 0
            feat["cond_is_return"] = 1
            feat["cond_is_late"] = 0
        else:
            for a in range(N_ACTION_CLASSES):
                feat[_COND_SERVE_ACT_NAMES[a]] = 0.0
            for p in range(N_POINT_CLASSES):
                feat[_COND_SERVE_PT_NAMES[p]] = 0.0
            feat["cond_is_serve"] = 0
            feat["cond_is_return"] = 0
            feat["cond_is_late"] = 1 if feat["next_strikeNumber"] >= 4 else 0

    else:
        # No global stats -- fill zeros for all global-stat-dependent features
        # (This branch is mainly for safety; in practice global_stats is always provided)
        _fill_zero_global_feats(feat)

    # ===================================================================
    # L. Rally phase features (V2 base + extensions)
    # ===================================================================
    rl = feat["rally_length"]
    feat["rally_phase"] = 0 if rl <= 2 else (1 if rl <= 5 else 2)

    # Consecutive same action count
    if len(actions) >= 2:
        count = 1
        for i in range(len(actions)-2, -1, -1):
            if actions[i] == actions[-1]:
                count += 1
            else:
                break
        feat["consecutive_same_action"] = count
    else:
        feat["consecutive_same_action"] = 1

    last_act = int(actions[-1]) if len(actions) > 0 else 0
    feat["last_action_category"] = (0 if last_act in ACTION_ATTACK else
                                     1 if last_act in ACTION_CONTROL else
                                     2 if last_act in ACTION_DEFENSE else
                                     3 if last_act in ACTION_SERVE else 4)

    feat["forehand_ratio"] = (hands == 1).sum() / n_ctx
    feat["backhand_ratio"] = (hands == 2).sum() / n_ctx

    # Strength trend
    if len(strengths) >= 3:
        feat["strength_trend"] = float(strengths[-1]) - float(strengths[-3])
    elif len(strengths) >= 2:
        feat["strength_trend"] = float(strengths[-1]) - float(strengths[-2])
    else:
        feat["strength_trend"] = 0

    # Position changes
    if len(positions_arr) >= 2:
        feat["position_changes"] = np.sum(positions_arr[1:] != positions_arr[:-1]) / (len(positions_arr) - 1)
    else:
        feat["position_changes"] = 0

    # Last 2 strikes combos (V2)
    if ctx_len >= 2:
        feat["last2_action_combo"] = int(actions[-2]) * N_ACTION_CLASSES + int(actions[-1])
        feat["last2_point_combo"] = int(points[-2]) * N_POINT_CLASSES + int(points[-1])
        feat["last2_hand_combo"] = int(hands[-2]) * 3 + int(hands[-1])
        feat["last2_zone_combo"] = _zone(int(points[-2])) * 4 + _zone(int(points[-1]))
    else:
        feat["last2_action_combo"] = -1
        feat["last2_point_combo"] = -1
        feat["last2_hand_combo"] = -1
        feat["last2_zone_combo"] = -1

    # Action/point diversity (V2)
    feat["action_diversity"] = len(set(actions.tolist())) / n_ctx if len(actions) > 0 else 0
    feat["point_diversity"] = len(set(points.tolist())) / n_ctx if len(points) > 0 else 0

    # ===================================================================
    # E-NEW. Interaction features
    # ===================================================================
    feat["inter_last_action_x_point"] = int(actions[-1]) * N_POINT_CLASSES + int(points[-1]) if len(actions) > 0 else -1
    feat["inter_last_hand_x_pos"] = int(hands[-1]) * 4 + int(positions_arr[-1]) if len(hands) > 0 else -1
    feat["inter_last_str_x_spin"] = int(strengths[-1]) * 6 + int(spins[-1]) if len(strengths) > 0 else -1
    feat["inter_score_diff_x_rally_len"] = feat["score_diff"] * feat["rally_length"]
    feat["inter_sex_x_next_sn"] = feat["sex"] * feat["next_strikeNumber"]
    feat["inter_game_x_score_diff"] = feat["numberGame"] * feat["score_diff"]
    feat["inter_serve_act_x_pt"] = feat["serve_actionId"] * N_POINT_CLASSES + feat["serve_pointId"]

    # Lag1 and Lag2 interaction features
    for k in [1, 2]:
        if ctx_len >= k:
            a_k = int(_ctx_actionId[-k])
            h_k = int(_ctx_handId[-k])
            pos_k = int(_ctx_positionId[-k])
            pt_k = int(_ctx_pointId[-k])
            feat[f"inter_lag{k}_act_x_hand"] = a_k * 3 + h_k
            feat[f"inter_lag{k}_act_x_pos"] = a_k * 4 + pos_k
            feat[f"inter_lag{k}_pt_x_pos"] = pt_k * 4 + pos_k
        else:
            feat[f"inter_lag{k}_act_x_hand"] = -1
            feat[f"inter_lag{k}_act_x_pos"] = -1
            feat[f"inter_lag{k}_pt_x_pos"] = -1

    # ===================================================================
    # G-NEW. Rally momentum features
    # ===================================================================
    # Strength trend: diff last vs first, last vs mean, slope
    if len(strengths) > 0:
        feat["strength_last_vs_first"] = float(strengths[-1]) - float(strengths[0])
        feat["strength_last_vs_mean"] = float(strengths[-1]) - float(np.mean(strengths))
        if len(strengths) >= 2:
            x = np.arange(len(strengths), dtype=float)
            y = strengths.astype(float)
            feat["strength_slope"] = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else 0.0
        else:
            feat["strength_slope"] = 0.0
    else:
        feat["strength_last_vs_first"] = 0
        feat["strength_last_vs_mean"] = 0
        feat["strength_slope"] = 0.0

    # Spin variety
    feat["spin_variety"] = len(set(spins.tolist())) if len(spins) > 0 else 0

    # Action variety in last 3 vs first 3
    if len(actions) >= 6:
        feat["action_variety_last3"] = len(set(actions[-3:].tolist()))
        feat["action_variety_first3"] = len(set(actions[:3].tolist()))
        feat["action_variety_diff"] = feat["action_variety_last3"] - feat["action_variety_first3"]
    elif len(actions) >= 3:
        feat["action_variety_last3"] = len(set(actions[-3:].tolist()))
        feat["action_variety_first3"] = len(set(actions[:min(3, len(actions))].tolist()))
        feat["action_variety_diff"] = feat["action_variety_last3"] - feat["action_variety_first3"]
    else:
        feat["action_variety_last3"] = len(set(actions.tolist())) if len(actions) > 0 else 0
        feat["action_variety_first3"] = feat["action_variety_last3"]
        feat["action_variety_diff"] = 0

    # Aggression score
    attack_c = np.isin(actions, list(ACTION_ATTACK)).sum()
    control_c = np.isin(actions, list(ACTION_CONTROL)).sum()
    defense_c = np.isin(actions, list(ACTION_DEFENSE)).sum()
    feat["aggression_score"] = int(attack_c) * 3 + int(control_c) * 1 - int(defense_c) * 2
    feat["aggression_score_norm"] = feat["aggression_score"] / max(n_ctx, 1)

    # Rally rhythm: position change rate (already computed as position_changes)
    # Add: hand change rate
    if len(hands) >= 2:
        feat["hand_change_rate"] = float(np.sum(hands[1:] != hands[:-1])) / (len(hands) - 1)
    else:
        feat["hand_change_rate"] = 0.0

    # Spin trend
    if len(spins) >= 2:
        feat["spin_changed_last"] = int(spins[-1] != spins[-2])
    else:
        feat["spin_changed_last"] = 0

    # ===================================================================
    # Extra: rally_length interactions and log
    # ===================================================================
    feat["log_rally_length"] = float(np.log1p(feat["rally_length"]))
    feat["rally_length_sq"] = feat["rally_length"] ** 2
    feat["next_sn_sq"] = feat["next_strikeNumber"] ** 2
    feat["next_sn_log"] = float(np.log1p(feat["next_strikeNumber"]))

    # ===================================================================
    # EXTRA: More interaction features to reach 800+ total
    # ===================================================================

    # Parity-based features
    feat["next_sn_parity"] = feat["next_strikeNumber"] % 2
    feat["rally_length_parity"] = feat["rally_length"] % 2

    # Score interactions
    feat["inter_score_max_x_game"] = feat["score_max"] * feat["numberGame"]
    feat["inter_score_diff_x_game_pt"] = feat["score_diff"] * feat["is_game_point"]
    feat["inter_score_diff_sq"] = feat["score_diff"] ** 2
    feat["inter_sex_x_game"] = feat["sex"] * feat["numberGame"]
    feat["inter_sex_x_score_diff"] = feat["sex"] * feat["score_diff"]
    feat["inter_game_progress_x_rally_len"] = feat["game_progress"] * feat["rally_length"]
    feat["inter_game_x_is_deuce"] = feat["numberGame"] * feat["is_deuce"]

    # More lag interaction combos (lag1)
    if ctx_len >= 1:
        a1 = int(actions[-1])
        s1 = int(spins[-1])
        str1 = int(strengths[-1])
        feat["inter_lag1_act_x_spin"] = a1 * 6 + s1
        feat["inter_lag1_act_x_str"] = a1 * 4 + str1
        feat["inter_lag1_spin_x_str"] = s1 * 4 + str1
        feat["inter_lag1_pt_x_hand"] = int(points[-1]) * 3 + int(hands[-1])
        feat["inter_lag1_spin_x_pos"] = s1 * 4 + int(positions_arr[-1])
    else:
        feat["inter_lag1_act_x_spin"] = -1
        feat["inter_lag1_act_x_str"] = -1
        feat["inter_lag1_spin_x_str"] = -1
        feat["inter_lag1_pt_x_hand"] = -1
        feat["inter_lag1_spin_x_pos"] = -1

    # Lag2 more combos
    if ctx_len >= 2:
        a2 = int(actions[-2])
        s2 = int(spins[-2])
        feat["inter_lag2_act_x_spin"] = a2 * 6 + s2
        feat["inter_lag2_spin_x_str"] = s2 * 4 + int(strengths[-2])
    else:
        feat["inter_lag2_act_x_spin"] = -1
        feat["inter_lag2_spin_x_str"] = -1

    # Rolling window stats (last 3 vs last 5)
    for window, suffix in [(3, "w3"), (5, "w5")]:
        if len(actions) >= window:
            w_acts = actions[-window:]
            w_pts = points[-window:]
            w_str = strengths[-window:]
            w_spin = spins[-window:]
            feat[f"attack_ratio_{suffix}"] = float(np.isin(w_acts, list(ACTION_ATTACK)).sum()) / window
            feat[f"control_ratio_{suffix}"] = float(np.isin(w_acts, list(ACTION_CONTROL)).sum()) / window
            feat[f"defense_ratio_{suffix}"] = float(np.isin(w_acts, list(ACTION_DEFENSE)).sum()) / window
            feat[f"short_ratio_{suffix}"] = float(np.isin(w_pts, [1, 2, 3]).sum()) / window
            feat[f"long_ratio_{suffix}"] = float(np.isin(w_pts, [7, 8, 9]).sum()) / window
            feat[f"strength_mean_{suffix}"] = float(np.mean(w_str))
            feat[f"spin_mean_{suffix}"] = float(np.mean(w_spin))
            feat[f"action_nunique_{suffix}"] = len(set(w_acts.tolist()))
            feat[f"point_nunique_{suffix}"] = len(set(w_pts.tolist()))
        else:
            feat[f"attack_ratio_{suffix}"] = feat["action_attack_ratio"]
            feat[f"control_ratio_{suffix}"] = feat["action_control_ratio"]
            feat[f"defense_ratio_{suffix}"] = feat["action_defense_ratio"]
            feat[f"short_ratio_{suffix}"] = feat["point_short_ratio"]
            feat[f"long_ratio_{suffix}"] = feat["point_long_ratio"]
            feat[f"strength_mean_{suffix}"] = feat["strengthId_mean"]
            feat[f"spin_mean_{suffix}"] = feat["spinId_mean"]
            feat[f"action_nunique_{suffix}"] = feat["actionId_nunique"]
            feat[f"point_nunique_{suffix}"] = feat["pointId_nunique"]

    # Cumulative hand counts
    hand_counts = np.bincount(hands, minlength=3)
    for h in range(3):
        feat[f"ctx_hand_count_{h}"] = int(hand_counts[h])
        feat[f"ctx_hand_frac_{h}"] = float(hand_counts[h]) / n_ctx

    # Cumulative strength counts
    str_counts = np.bincount(strengths, minlength=4)
    for s in range(4):
        feat[f"ctx_str_count_{s}"] = int(str_counts[s])
        feat[f"ctx_str_frac_{s}"] = float(str_counts[s]) / n_ctx

    # Cumulative spin counts
    spin_counts = np.bincount(spins, minlength=6)
    for s in range(6):
        feat[f"ctx_spin_count_{s}"] = int(spin_counts[s])
        feat[f"ctx_spin_frac_{s}"] = float(spin_counts[s]) / n_ctx

    # Cumulative position counts
    pos_counts = np.bincount(positions_arr, minlength=4)
    for p in range(4):
        feat[f"ctx_pos_count_{p}"] = int(pos_counts[p])
        feat[f"ctx_pos_frac_{p}"] = float(pos_counts[p]) / n_ctx

    # Zone-based cumulative
    zones_arr = np.array([_zone(int(p)) for p in points])
    zone_counts = np.bincount(zones_arr, minlength=N_ZONES)
    for z in range(N_ZONES):
        feat[f"ctx_zone_count_{z}"] = int(zone_counts[z])
        feat[f"ctx_zone_frac_{z}"] = float(zone_counts[z]) / n_ctx

    # Serve type one-hot (useful for all strikes in the rally)
    for a in range(N_ACTION_CLASSES):
        feat[_SERVE_ACT_IS_NAMES[a]] = 1 if feat["serve_actionId"] == a else 0

    # Consecutive same point count
    if len(points) >= 2:
        count_p = 1
        for i in range(len(points)-2, -1, -1):
            if points[i] == points[-1]:
                count_p += 1
            else:
                break
        feat["consecutive_same_point"] = count_p
    else:
        feat["consecutive_same_point"] = 1

    # Consecutive same hand
    if len(hands) >= 2:
        count_h = 1
        for i in range(len(hands)-2, -1, -1):
            if hands[i] == hands[-1]:
                count_h += 1
            else:
                break
        feat["consecutive_same_hand"] = count_h
    else:
        feat["consecutive_same_hand"] = 1

    # Entropy of actions/points in context
    if len(actions) > 0:
        a_probs = action_counts[action_counts > 0].astype(float) / n_ctx
        feat["ctx_action_entropy"] = float(-np.sum(a_probs * np.log(a_probs + 1e-10)))
        p_probs = point_counts[point_counts > 0].astype(float) / n_ctx
        feat["ctx_point_entropy"] = float(-np.sum(p_probs * np.log(p_probs + 1e-10)))
    else:
        feat["ctx_action_entropy"] = 0.0
        feat["ctx_point_entropy"] = 0.0

    return feat


def _fill_zero_global_feats(feat):
    """Fill all global-stats-dependent features with zeros when stats unavailable."""
    # Full transition vectors
    for prefix in ["trans_act_act", "trans_pos_act", "trans_pt_act"]:
        for a in range(N_ACTION_CLASSES):
            feat[f"{prefix}_{a}"] = 0.0
    for prefix in ["trans_pt_pt", "trans_act_pt"]:
        for p in range(N_POINT_CLASSES):
            feat[f"{prefix}_{p}"] = 0.0

    # Player distributions
    for prefix in ["hitter_act_prob", "receiver_act_prob"]:
        for a in range(N_ACTION_CLASSES):
            feat[f"{prefix}_{a}"] = 0.0
    for prefix in ["hitter_pt_prob", "receiver_pt_prob"]:
        for p in range(N_POINT_CLASSES):
            feat[f"{prefix}_{p}"] = 0.0

    # Top-K transition (V2)
    for k in ["trans_top1_action", "trans_top2_action", "trans_top3_action",
              "trans_top1_point", "trans_top2_point",
              "pos_top1_action", "act_top1_point",
              "serve_return_top1", "sn_top1_action", "sn_top2_action"]:
        feat[k] = -1
    for k in ["trans_top1_action_prob", "trans_top2_action_prob",
              "trans_entropy_action", "trans_top1_point_prob",
              "trans_entropy_point", "pos_top1_action_prob",
              "act_top1_point_prob", "serve_return_top1_prob",
              "sn_top1_action_prob", "sn_top2_action_prob",
              "score_diff_hist_winrate"]:
        feat[k] = 0.0
    feat["matchup_games"] = 0
    feat["matchup_winrate_a"] = 0.5

    # Player summary (V2)
    for k in ["hitter_top_action", "hitter_top_point", "hitter_top1_act",
               "hitter_top2_act", "hitter_top3_act", "hitter_top1_pt",
               "hitter_top2_pt", "hitter_pos_top_act", "hitter_serve_top1"]:
        feat[k] = -1
    for k in ["hitter_attack_rate", "hitter_control_rate", "hitter_defense_rate",
               "hitter_top1_act_prob", "hitter_top2_act_prob", "hitter_top1_pt_prob",
               "hitter_pos_top_act_prob", "hitter_serve_top1_prob",
               "hitter_avg_strength", "hitter_avg_spin", "hitter_action_entropy"]:
        feat[k] = 0.0
    feat["hitter_win_rate"] = 0.5
    feat["hitter_n_rallies"] = 0
    feat["receiver_attack_rate"] = 0.0
    feat["receiver_defense_rate"] = 0.0
    feat["receiver_win_rate"] = 0.5
    feat["receiver_avg_strength"] = 0.0

    # Zone features
    for z in range(N_ZONES):
        feat[f"zone_trans_{z}"] = 0.0
    for k in [1, 2, 3]:
        feat[f"zone_lag{k}"] = -1
    for pid_val in range(N_POINT_CLASSES):
        feat[f"ctx_zone_detail_{pid_val}"] = 0
    feat["zone_x_hand_last"] = 0

    # Bigram/trigram freq
    for k in ["bigram_action_freq", "bigram_point_freq",
              "bigram_action_log_freq", "bigram_point_log_freq",
              "trigram_action_freq", "trigram_action_log_freq"]:
        feat[k] = 0

    # Serve-return expanded
    for prefix in ["sr_aa", "sr_pa", "sr_spin_act"]:
        for a in range(N_ACTION_CLASSES):
            feat[f"{prefix}_{a}"] = 0.0
    for p in range(N_POINT_CLASSES):
        feat[f"sr_ap_{p}"] = 0.0

    # Conditional features
    for a in range(N_ACTION_CLASSES):
        feat[f"cond_serve_act_{a}"] = 0.0
    for p in range(N_POINT_CLASSES):
        feat[f"cond_serve_pt_{p}"] = 0.0
    feat["cond_is_serve"] = 0
    feat["cond_is_return"] = 0
    feat["cond_is_late"] = 0


# ============================= BUILD FEATURES ==============================

def build_features_v3(df, is_train=True, global_stats=None):
    """Build feature matrix with 800+ features."""
    rallies = df.groupby("rally_uid", sort=False)
    records = []

    for rally_uid, group in rallies:
        group = group.sort_values("strikeNumber")

        if is_train:
            for target_idx in range(1, len(group)):
                context = group.iloc[:target_idx]
                target_row = group.iloc[target_idx]
                feat = _build_one_sample(rally_uid, context, target_row,
                                         is_train, global_stats)
                records.append(feat)
        else:
            context = group
            feat = _build_one_sample(rally_uid, context, None,
                                     is_train, global_stats)
            records.append(feat)

    return pd.DataFrame(records)


def get_feature_names_v3(feat_df):
    """Return feature column names (excludes rally_uid, targets, and raw player IDs)."""
    exclude = {"rally_uid", "y_actionId", "y_pointId", "y_serverGetPoint",
               "gamePlayerId", "gamePlayerOtherId", "next_hitter_id", "next_receiver_id"}
    return [c for c in feat_df.columns if c not in exclude]
