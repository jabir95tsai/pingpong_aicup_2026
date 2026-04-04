"""Enhanced feature engineering V2: transition matrices, player distributions,
rally phase features, score pressure, and target encoding."""
import numpy as np
import pandas as pd
from collections import defaultdict
from config import (
    LAG_STEPS, LAG_COLS, CATEGORICAL_STRIKE_COLS,
    ACTION_ATTACK, ACTION_CONTROL, ACTION_DEFENSE, ACTION_SERVE,
    N_ACTION_CLASSES, N_POINT_CLASSES,
)


def _zone(p):
    if p in {1, 2, 3}: return 1  # short
    if p in {4, 5, 6}: return 2  # mid
    if p in {7, 8, 9}: return 3  # long
    return 0


def compute_global_stats(train_df):
    """Compute global statistics from training data for feature engineering."""
    stats = {}

    # 1. Global transition matrices: P(action_next | action_prev)
    rallies = train_df.groupby("rally_uid", sort=False)
    action_trans = np.zeros((N_ACTION_CLASSES, N_ACTION_CLASSES), dtype=np.float64)
    point_trans = np.zeros((N_POINT_CLASSES, N_POINT_CLASSES), dtype=np.float64)
    # Also: P(action | position)
    pos_action = np.zeros((4, N_ACTION_CLASSES), dtype=np.float64)
    # P(point | action)
    action_point = np.zeros((N_ACTION_CLASSES, N_POINT_CLASSES), dtype=np.float64)
    # Serve -> return transition
    serve_return = np.zeros((N_ACTION_CLASSES, N_ACTION_CLASSES), dtype=np.float64)

    for _, group in rallies:
        group = group.sort_values("strikeNumber")
        actions = group["actionId"].values
        points = group["pointId"].values
        positions = group["positionId"].values

        for i in range(len(actions)):
            a = int(actions[i])
            p = int(points[i])
            pos = int(positions[i])
            if pos < 4 and a < N_ACTION_CLASSES:
                pos_action[pos, a] += 1
            if a < N_ACTION_CLASSES and p < N_POINT_CLASSES:
                action_point[a, p] += 1

            if i > 0:
                prev_a = int(actions[i-1])
                prev_p = int(points[i-1])
                if prev_a < N_ACTION_CLASSES and a < N_ACTION_CLASSES:
                    action_trans[prev_a, a] += 1
                if prev_p < N_POINT_CLASSES and p < N_POINT_CLASSES:
                    point_trans[prev_p, p] += 1
                if i == 1:  # serve -> return
                    serve_return[prev_a, a] += 1

    # Normalize to probabilities
    for mat in [action_trans, point_trans, pos_action, action_point, serve_return]:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat /= row_sums

    stats["action_trans"] = action_trans
    stats["point_trans"] = point_trans
    stats["pos_action"] = pos_action
    stats["action_point"] = action_point
    stats["serve_return"] = serve_return

    # 2. Global action/point distributions
    stats["global_action_dist"] = np.bincount(
        train_df["actionId"].values, minlength=N_ACTION_CLASSES).astype(float)
    stats["global_action_dist"] /= max(stats["global_action_dist"].sum(), 1)
    stats["global_point_dist"] = np.bincount(
        train_df["pointId"].values, minlength=N_POINT_CLASSES).astype(float)
    stats["global_point_dist"] /= max(stats["global_point_dist"].sum(), 1)

    # 3. Per-player detailed stats
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

        # Per-position action distribution
        pos_dists = {}
        for pos in range(4):
            mask = grp["positionId"] == pos
            if mask.sum() > 0:
                d = np.bincount(grp.loc[mask, "actionId"].values, minlength=N_ACTION_CLASSES).astype(float)
                d /= d.sum()
                pos_dists[pos] = d

        # Serve-specific stats
        serve_mask = grp["strikeId"] == 0  # after remap, 0 = serve only
        serve_actions = grp.loc[serve_mask, "actionId"].values if serve_mask.sum() > 0 else np.array([])

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
            "serve_action_dist": np.bincount(serve_actions.astype(int), minlength=N_ACTION_CLASSES).astype(float) / max(len(serve_actions), 1) if len(serve_actions) > 0 else np.zeros(N_ACTION_CLASSES),
        }

    stats["player_stats"] = player_stats

    # 4. Per-matchup stats (player pair)
    matchup_stats = defaultdict(lambda: {"wins_a": 0, "total": 0, "actions_a": [], "actions_b": []})
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

    # 5. Score situation stats: P(serverGetPoint | score_diff)
    score_diff_wins = defaultdict(list)
    for _, grp in rallies:
        first = grp.iloc[0]
        diff = int(first["scoreSelf"]) - int(first["scoreOther"])
        score_diff_wins[diff].append(int(first["serverGetPoint"]))
    stats["score_diff_win_rate"] = {k: np.mean(v) for k, v in score_diff_wins.items()}

    # 6. Rally length -> action distribution
    rally_len_action = defaultdict(list)
    for _, grp in rallies:
        grp = grp.sort_values("strikeNumber")
        for i in range(1, len(grp)):
            rally_len_action[i].append(int(grp.iloc[i]["actionId"]))
    stats["strike_num_action_dist"] = {}
    for sn, acts in rally_len_action.items():
        d = np.bincount(acts, minlength=N_ACTION_CLASSES).astype(float)
        d /= max(d.sum(), 1)
        stats["strike_num_action_dist"][sn] = d

    return stats


def _build_one_sample(rally_uid, context, target_row, is_train, global_stats):
    """Build enhanced feature vector."""
    feat = {"rally_uid": rally_uid}

    # --- Targets ---
    if is_train and target_row is not None:
        feat["y_actionId"] = int(target_row["actionId"])
        feat["y_pointId"] = int(target_row["pointId"])
        feat["y_serverGetPoint"] = int(target_row["serverGetPoint"])

    # --- A. Basic features ---
    last_ctx = context.iloc[-1]
    feat["sex"] = int(last_ctx["sex"])
    feat["numberGame"] = int(last_ctx["numberGame"])
    feat["scoreSelf"] = int(last_ctx["scoreSelf"])
    feat["scoreOther"] = int(last_ctx["scoreOther"])
    feat["score_diff"] = feat["scoreSelf"] - feat["scoreOther"]
    feat["score_total"] = feat["scoreSelf"] + feat["scoreOther"]
    feat["rally_length"] = len(context)
    feat["next_strikeNumber"] = int(last_ctx["strikeNumber"]) + 1
    feat["next_is_server"] = 1 if feat["next_strikeNumber"] % 2 == 1 else 0

    if feat["next_strikeNumber"] == 1:
        feat["next_strikeId"] = 1
    elif feat["next_strikeNumber"] == 2:
        feat["next_strikeId"] = 2
    else:
        feat["next_strikeId"] = 4

    # Score pressure features
    feat["is_game_point"] = 1 if (feat["scoreSelf"] >= 10 or feat["scoreOther"] >= 10) else 0
    feat["is_deuce"] = 1 if (feat["scoreSelf"] >= 10 and feat["scoreOther"] >= 10 and
                              abs(feat["score_diff"]) <= 1) else 0
    feat["score_max"] = max(feat["scoreSelf"], feat["scoreOther"])
    feat["score_min"] = min(feat["scoreSelf"], feat["scoreOther"])

    # --- B. Lag features (last K strikes) ---
    for k in LAG_STEPS:
        for col in LAG_COLS:
            if len(context) >= k:
                feat[f"lag{k}_{col}"] = int(context.iloc[-k][col])
            else:
                feat[f"lag{k}_{col}"] = -1

    # --- C. Rally-level statistics ---
    for col in CATEGORICAL_STRIKE_COLS:
        vals = context[col].values
        if len(vals) > 0:
            counts = np.bincount(vals)
            feat[f"{col}_mode"] = int(np.argmax(counts))
        else:
            feat[f"{col}_mode"] = -1
        feat[f"{col}_nunique"] = len(set(vals.tolist()))

    for col in ["strengthId", "spinId"]:
        vals = context[col].values.astype(float)
        feat[f"{col}_mean"] = float(np.mean(vals)) if len(vals) > 0 else 0
        feat[f"{col}_std"] = float(np.std(vals)) if len(vals) > 0 else 0

    actions = context["actionId"].values
    points = context["pointId"].values
    n_ctx = max(len(actions), 1)

    feat["action_attack_ratio"] = sum(1 for a in actions if a in ACTION_ATTACK) / n_ctx
    feat["action_control_ratio"] = sum(1 for a in actions if a in ACTION_CONTROL) / n_ctx
    feat["action_defense_ratio"] = sum(1 for a in actions if a in ACTION_DEFENSE) / n_ctx
    feat["action_serve_ratio"] = sum(1 for a in actions if a in ACTION_SERVE) / n_ctx

    hands = context["handId"].values
    if len(hands) > 1:
        feat["hand_alternation"] = sum(1 for i in range(1, len(hands))
                                       if hands[i] != hands[i-1]) / (len(hands) - 1)
    else:
        feat["hand_alternation"] = 0

    feat["point_short_ratio"] = sum(1 for p in points if p in {1, 2, 3}) / n_ctx
    feat["point_mid_ratio"] = sum(1 for p in points if p in {4, 5, 6}) / n_ctx
    feat["point_long_ratio"] = sum(1 for p in points if p in {7, 8, 9}) / n_ctx
    feat["point_zero_ratio"] = sum(1 for p in points if p == 0) / n_ctx

    # --- D. Temporal / transition features ---
    if len(context) >= 2:
        feat["action_bigram"] = int(context.iloc[-2]["actionId"]) * 100 + int(context.iloc[-1]["actionId"])
        feat["point_bigram"] = int(context.iloc[-2]["pointId"]) * 100 + int(context.iloc[-1]["pointId"])
        feat["action_changed"] = int(context.iloc[-1]["actionId"] != context.iloc[-2]["actionId"])
        feat["point_changed"] = int(context.iloc[-1]["pointId"] != context.iloc[-2]["pointId"])
        feat["hand_changed"] = int(context.iloc[-1]["handId"] != context.iloc[-2]["handId"])
        feat["point_zone_trend"] = _zone(int(context.iloc[-1]["pointId"])) - _zone(int(context.iloc[-2]["pointId"]))
    else:
        feat["action_bigram"] = -1
        feat["point_bigram"] = -1
        feat["action_changed"] = 0
        feat["point_changed"] = 0
        feat["hand_changed"] = 0
        feat["point_zone_trend"] = 0

    # Trigram
    if len(context) >= 3:
        feat["action_trigram"] = (int(context.iloc[-3]["actionId"]) * 10000 +
                                   int(context.iloc[-2]["actionId"]) * 100 +
                                   int(context.iloc[-1]["actionId"]))
        feat["point_trigram"] = (int(context.iloc[-3]["pointId"]) * 10000 +
                                  int(context.iloc[-2]["pointId"]) * 100 +
                                  int(context.iloc[-1]["pointId"]))
    else:
        feat["action_trigram"] = -1
        feat["point_trigram"] = -1

    # --- E. Player features ---
    feat["gamePlayerId"] = int(last_ctx["gamePlayerId"])
    feat["gamePlayerOtherId"] = int(last_ctx["gamePlayerOtherId"])

    if feat["next_strikeNumber"] % 2 == 1:
        feat["next_hitter_id"] = int(context.iloc[0]["gamePlayerId"])
        feat["next_receiver_id"] = int(context.iloc[0]["gamePlayerOtherId"])
    else:
        feat["next_hitter_id"] = int(context.iloc[0]["gamePlayerOtherId"])
        feat["next_receiver_id"] = int(context.iloc[0]["gamePlayerId"])

    # --- F. Serve-specific features ---
    serve_row = context.iloc[0]
    feat["serve_actionId"] = int(serve_row["actionId"])
    feat["serve_spinId"] = int(serve_row["spinId"])
    feat["serve_pointId"] = int(serve_row["pointId"])
    feat["serve_strengthId"] = int(serve_row["strengthId"])
    feat["serve_positionId"] = int(serve_row["positionId"])

    # === NEW V2 FEATURES ===

    if global_stats is not None:
        player_stats = global_stats.get("player_stats", {})

        # --- G. Transition probability features ---
        last_action = int(context.iloc[-1]["actionId"])
        last_point = int(context.iloc[-1]["pointId"])
        last_position = int(context.iloc[-1]["positionId"])

        # Top-K most likely next actions from transition matrix
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

        # Point transition
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

        # Position -> action
        pos_action = global_stats["pos_action"]
        if last_position < 4:
            pa_probs = pos_action[last_position]
            feat["pos_top1_action"] = int(np.argmax(pa_probs))
            feat["pos_top1_action_prob"] = float(np.max(pa_probs))
        else:
            feat["pos_top1_action"] = -1
            feat["pos_top1_action_prob"] = 0

        # Action -> point
        action_point = global_stats["action_point"]
        if last_action < N_ACTION_CLASSES:
            ap_probs = action_point[last_action]
            feat["act_top1_point"] = int(np.argmax(ap_probs))
            feat["act_top1_point_prob"] = float(np.max(ap_probs))
        else:
            feat["act_top1_point"] = -1
            feat["act_top1_point_prob"] = 0

        # Serve -> return transition
        if feat["next_strikeNumber"] == 2:
            serve_return = global_stats["serve_return"]
            sa = int(serve_row["actionId"])
            if sa < N_ACTION_CLASSES:
                sr_probs = serve_return[sa]
                feat["serve_return_top1"] = int(np.argmax(sr_probs))
                feat["serve_return_top1_prob"] = float(np.max(sr_probs))
            else:
                feat["serve_return_top1"] = -1
                feat["serve_return_top1_prob"] = 0
        else:
            feat["serve_return_top1"] = -1
            feat["serve_return_top1_prob"] = 0

        # --- H. Player distribution features ---
        hitter = feat["next_hitter_id"]
        receiver = feat["next_receiver_id"]

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

            # Top 3 actions for this player
            ad = ps["action_dist"]
            top3 = np.argsort(ad)[::-1][:3]
            feat["hitter_top1_act"] = int(top3[0])
            feat["hitter_top1_act_prob"] = float(ad[top3[0]])
            feat["hitter_top2_act"] = int(top3[1])
            feat["hitter_top2_act_prob"] = float(ad[top3[1]])
            feat["hitter_top3_act"] = int(top3[2])

            # Player action entropy
            ad_pos = ad[ad > 0]
            feat["hitter_action_entropy"] = float(-np.sum(ad_pos * np.log(ad_pos + 1e-10)))

            # Player point distribution top
            pd_dist = ps["point_dist"]
            top2_pt = np.argsort(pd_dist)[::-1][:2]
            feat["hitter_top1_pt"] = int(top2_pt[0])
            feat["hitter_top1_pt_prob"] = float(pd_dist[top2_pt[0]])
            feat["hitter_top2_pt"] = int(top2_pt[1])

            # Position-specific action for this player
            if last_position in ps.get("pos_dists", {}):
                ppd = ps["pos_dists"][last_position]
                feat["hitter_pos_top_act"] = int(np.argmax(ppd))
                feat["hitter_pos_top_act_prob"] = float(np.max(ppd))
            else:
                feat["hitter_pos_top_act"] = -1
                feat["hitter_pos_top_act_prob"] = 0

            # Serve distribution for this player
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

        # Receiver stats (opponent awareness)
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

        # --- I. Matchup features ---
        pa, pb = feat["gamePlayerId"], feat["gamePlayerOtherId"]
        key = (min(pa, pb), max(pa, pb))
        matchup = global_stats.get("matchup_stats", {}).get(key)
        if matchup and matchup["total"] > 0:
            feat["matchup_games"] = matchup["total"]
            feat["matchup_winrate_a"] = matchup["wins_a"] / matchup["total"]
        else:
            feat["matchup_games"] = 0
            feat["matchup_winrate_a"] = 0.5

        # --- J. Score situation features ---
        sdwr = global_stats.get("score_diff_win_rate", {})
        feat["score_diff_hist_winrate"] = sdwr.get(feat["score_diff"], 0.5)

        # --- K. Strike-number conditioned action distribution ---
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

    # --- L. Rally phase features ---
    rl = feat["rally_length"]
    feat["rally_phase"] = 0 if rl <= 2 else (1 if rl <= 5 else 2)  # early/mid/late

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

    # Last action category
    last_act = int(context.iloc[-1]["actionId"])
    feat["last_action_category"] = (0 if last_act in ACTION_ATTACK else
                                     1 if last_act in ACTION_CONTROL else
                                     2 if last_act in ACTION_DEFENSE else
                                     3 if last_act in ACTION_SERVE else 4)

    # Forehand/backhand ratio in rally
    feat["forehand_ratio"] = sum(1 for h in hands if h == 1) / n_ctx
    feat["backhand_ratio"] = sum(1 for h in hands if h == 2) / n_ctx

    # Strength trend (last 3)
    strengths = context["strengthId"].values
    if len(strengths) >= 3:
        feat["strength_trend"] = float(strengths[-1]) - float(strengths[-3])
    elif len(strengths) >= 2:
        feat["strength_trend"] = float(strengths[-1]) - float(strengths[-2])
    else:
        feat["strength_trend"] = 0

    # Position changes
    positions = context["positionId"].values
    if len(positions) >= 2:
        feat["position_changes"] = sum(1 for i in range(1, len(positions))
                                        if positions[i] != positions[i-1]) / (len(positions) - 1)
    else:
        feat["position_changes"] = 0

    # Last 2 strikes interaction features
    if len(context) >= 2:
        feat["last2_action_combo"] = int(context.iloc[-2]["actionId"]) * N_ACTION_CLASSES + int(context.iloc[-1]["actionId"])
        feat["last2_point_combo"] = int(context.iloc[-2]["pointId"]) * N_POINT_CLASSES + int(context.iloc[-1]["pointId"])
        feat["last2_hand_combo"] = int(context.iloc[-2]["handId"]) * 3 + int(context.iloc[-1]["handId"])
        feat["last2_zone_combo"] = _zone(int(context.iloc[-2]["pointId"])) * 4 + _zone(int(context.iloc[-1]["pointId"]))
    else:
        feat["last2_action_combo"] = -1
        feat["last2_point_combo"] = -1
        feat["last2_hand_combo"] = -1
        feat["last2_zone_combo"] = -1

    # Action diversity in rally
    if len(actions) > 0:
        feat["action_diversity"] = len(set(actions)) / n_ctx
        feat["point_diversity"] = len(set(points)) / n_ctx
    else:
        feat["action_diversity"] = 0
        feat["point_diversity"] = 0

    return feat


def build_features_v2(df, is_train=True, global_stats=None):
    """Build enhanced feature matrix."""
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


def get_feature_names_v2(feat_df):
    """Return feature column names."""
    exclude = {"rally_uid", "y_actionId", "y_pointId", "y_serverGetPoint"}
    return [c for c in feat_df.columns if c not in exclude]
