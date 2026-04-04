"""Feature engineering: convert rally sequences into flat feature vectors."""
import numpy as np
import pandas as pd
from config import (
    LAG_STEPS, LAG_COLS, CATEGORICAL_STRIKE_COLS,
    ACTION_ATTACK, ACTION_CONTROL, ACTION_DEFENSE, ACTION_SERVE,
    N_ACTION_CLASSES, N_POINT_CLASSES,
)


def _zone(p):
    if p in {1, 2, 3}: return 1
    if p in {4, 5, 6}: return 2
    if p in {7, 8, 9}: return 3
    return 0


def _build_one_sample(rally_uid, context, target_row, is_train, player_stats):
    """Build a single feature vector from context strikes and optional target."""
    feat = {"rally_uid": rally_uid}

    # --- Targets (train only) ---
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

    # Who is hitting next? (odd/even strikeNumber)
    feat["next_is_server"] = 1 if feat["next_strikeNumber"] % 2 == 1 else 0

    # Next strike type: 1=serve, 2=return, 4=rally
    if feat["next_strikeNumber"] == 1:
        feat["next_strikeId"] = 1
    elif feat["next_strikeNumber"] == 2:
        feat["next_strikeId"] = 2
    else:
        feat["next_strikeId"] = 4

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
        feat[f"{col}_mode"] = int(pd.Series(vals).mode().iloc[0]) if len(vals) > 0 else -1
        feat[f"{col}_nunique"] = len(set(vals))

    # Strength / spin stats
    for col in ["strengthId", "spinId"]:
        vals = context[col].values.astype(float)
        feat[f"{col}_mean"] = float(np.mean(vals)) if len(vals) > 0 else 0
        feat[f"{col}_std"] = float(np.std(vals)) if len(vals) > 0 else 0

    # Action category proportions
    actions = context["actionId"].values
    n_ctx = max(len(actions), 1)
    feat["action_attack_ratio"] = sum(1 for a in actions if a in ACTION_ATTACK) / n_ctx
    feat["action_control_ratio"] = sum(1 for a in actions if a in ACTION_CONTROL) / n_ctx
    feat["action_defense_ratio"] = sum(1 for a in actions if a in ACTION_DEFENSE) / n_ctx
    feat["action_serve_ratio"] = sum(1 for a in actions if a in ACTION_SERVE) / n_ctx

    # Hand alternation frequency
    hands = context["handId"].values
    if len(hands) > 1:
        feat["hand_alternation"] = sum(1 for i in range(1, len(hands))
                                       if hands[i] != hands[i-1]) / (len(hands) - 1)
    else:
        feat["hand_alternation"] = 0

    # Point zone distribution (short/mid/long)
    points = context["pointId"].values
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

    # --- E. Player features ---
    feat["gamePlayerId"] = int(last_ctx["gamePlayerId"])
    feat["gamePlayerOtherId"] = int(last_ctx["gamePlayerOtherId"])

    # Which player hits next
    if feat["next_strikeNumber"] % 2 == 1:
        feat["next_hitter_id"] = int(context.iloc[0]["gamePlayerId"])
    else:
        feat["next_hitter_id"] = int(context.iloc[0]["gamePlayerOtherId"])

    # Player history stats (if available)
    if player_stats is not None:
        hitter = feat["next_hitter_id"]
        if hitter in player_stats:
            ps = player_stats[hitter]
            feat["hitter_top_action"] = ps.get("top_action", -1)
            feat["hitter_top_point"] = ps.get("top_point", -1)
            feat["hitter_attack_rate"] = ps.get("attack_rate", 0.0)
            feat["hitter_win_rate"] = ps.get("win_rate", 0.5)
        else:
            feat["hitter_top_action"] = -1
            feat["hitter_top_point"] = -1
            feat["hitter_attack_rate"] = 0.0
            feat["hitter_win_rate"] = 0.5

    # --- F. Serve-specific features ---
    serve_row = context.iloc[0]
    feat["serve_actionId"] = int(serve_row["actionId"])
    feat["serve_spinId"] = int(serve_row["spinId"])
    feat["serve_pointId"] = int(serve_row["pointId"])
    feat["serve_strengthId"] = int(serve_row["strengthId"])

    return feat


def build_features(df: pd.DataFrame, is_train: bool = True,
                   player_stats: dict = None) -> pd.DataFrame:
    """Build feature matrix from raw data.

    For train: generate samples for ALL intermediate targets (strike 2..n),
               not just the last strike. This gives a realistic distribution.
    For test: all strikes are context, target is unknown.

    Returns a DataFrame with one row per prediction target.
    """
    rallies = df.groupby("rally_uid", sort=False)
    records = []

    for rally_uid, group in rallies:
        group = group.sort_values("strikeNumber")

        if is_train:
            # Generate a sample for each possible target (strike 2 to n)
            for target_idx in range(1, len(group)):
                context = group.iloc[:target_idx]
                target_row = group.iloc[target_idx]
                feat = _build_one_sample(rally_uid, context, target_row,
                                         is_train, player_stats)
                records.append(feat)
        else:
            # Test: predict the next strike after all shown strikes
            context = group
            feat = _build_one_sample(rally_uid, context, None,
                                     is_train, player_stats)
            records.append(feat)

    return pd.DataFrame(records)


def compute_player_stats(train_df: pd.DataFrame) -> dict:
    """Compute per-player statistics from training data."""
    stats = {}
    for pid, grp in train_df.groupby("gamePlayerId"):
        actions = grp["actionId"].values
        points = grp["pointId"].values
        action_counts = pd.Series(actions).value_counts()
        point_counts = pd.Series(points).value_counts()
        rally_results = grp.groupby("rally_uid")["serverGetPoint"].first()

        stats[int(pid)] = {
            "top_action": int(action_counts.index[0]) if len(action_counts) > 0 else -1,
            "top_point": int(point_counts.index[0]) if len(point_counts) > 0 else -1,
            "attack_rate": sum(1 for a in actions if a in ACTION_ATTACK) / max(len(actions), 1),
            "win_rate": float(rally_results.mean()) if len(rally_results) > 0 else 0.5,
        }

    return stats


def get_feature_names(feat_df: pd.DataFrame) -> list:
    """Return feature column names (excluding rally_uid and targets)."""
    exclude = {"rally_uid", "y_actionId", "y_pointId", "y_serverGetPoint"}
    return [c for c in feat_df.columns if c not in exclude]
