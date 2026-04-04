"""Training pipeline with K-Fold CV."""
import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_PATH, MODEL_DIR, N_FOLDS, RANDOM_SEED,
    N_ACTION_CLASSES, N_POINT_CLASSES,
)
from features import build_features, compute_player_stats, get_feature_names
from models import (
    train_lgb_multiclass, train_lgb_binary,
    predict_multiclass, predict_binary,
    apply_action_constraints,
    eval_macro_f1, eval_auc,
)


def main():
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"  {len(train_df)} rows, {train_df.rally_uid.nunique()} rallies")

    print("Computing player stats...")
    player_stats = compute_player_stats(train_df)

    print("Building features...")
    feat_df = build_features(train_df, is_train=True, player_stats=player_stats)
    print(f"  Feature matrix: {feat_df.shape}")

    feature_names = get_feature_names(feat_df)
    X = feat_df[feature_names].values
    y_action = feat_df["y_actionId"].values
    y_point = feat_df["y_pointId"].values
    y_server = feat_df["y_serverGetPoint"].values
    rally_uids = feat_df["rally_uid"].values
    next_sn = feat_df["next_strikeNumber"].values

    # Use match as group for GroupKFold (rallies from same match stay together)
    # But we don't have match in feat_df, so use rally_uid as group
    # Actually, let's get match from original data
    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_df["rally_uid"].map(rally_to_match).values

    print(f"\nStarting {N_FOLDS}-Fold CV...")
    gkf = GroupKFold(n_splits=N_FOLDS)

    action_f1_scores = []
    point_f1_scores = []
    server_auc_scores = []

    action_models = []
    point_models = []
    server_models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*60}")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_act_tr, y_act_val = y_action[train_idx], y_action[val_idx]
        y_pt_tr, y_pt_val = y_point[train_idx], y_point[val_idx]
        y_srv_tr, y_srv_val = y_server[train_idx], y_server[val_idx]
        val_next_sn = next_sn[val_idx]

        # Task 1: actionId
        print("\n[Task 1] Training actionId model...")
        action_model = train_lgb_multiclass(X_tr, y_act_tr, X_val, y_act_val, N_ACTION_CLASSES)
        action_probs = predict_multiclass(action_model, X_val)
        action_probs = apply_action_constraints(action_probs, val_next_sn)
        action_f1 = eval_macro_f1(y_act_val, action_probs, N_ACTION_CLASSES)
        print(f"  actionId Macro F1: {action_f1:.4f}")
        action_f1_scores.append(action_f1)
        action_models.append(action_model)

        # Task 2: pointId
        print("\n[Task 2] Training pointId model...")
        point_model = train_lgb_multiclass(X_tr, y_pt_tr, X_val, y_pt_val, N_POINT_CLASSES)
        point_probs = predict_multiclass(point_model, X_val)
        point_f1 = eval_macro_f1(y_pt_val, point_probs, N_POINT_CLASSES)
        print(f"  pointId Macro F1: {point_f1:.4f}")
        point_f1_scores.append(point_f1)
        point_models.append(point_model)

        # Task 3: serverGetPoint
        print("\n[Task 3] Training serverGetPoint model...")
        server_model = train_lgb_binary(X_tr, y_srv_tr, X_val, y_srv_val)
        server_probs = predict_binary(server_model, X_val)
        server_auc = eval_auc(y_srv_val, server_probs)
        print(f"  serverGetPoint AUC: {server_auc:.4f}")
        server_auc_scores.append(server_auc)
        server_models.append(server_model)

    # Summary
    print(f"\n{'='*60}")
    print("CV Results Summary")
    print(f"{'='*60}")
    mean_action_f1 = np.mean(action_f1_scores)
    mean_point_f1 = np.mean(point_f1_scores)
    mean_server_auc = np.mean(server_auc_scores)
    composite = 0.4 * mean_action_f1 + 0.4 * mean_point_f1 + 0.2 * mean_server_auc

    print(f"  actionId Macro F1:     {mean_action_f1:.4f} (+/- {np.std(action_f1_scores):.4f})")
    print(f"  pointId Macro F1:      {mean_point_f1:.4f} (+/- {np.std(point_f1_scores):.4f})")
    print(f"  serverGetPoint AUC:    {mean_server_auc:.4f} (+/- {np.std(server_auc_scores):.4f})")
    print(f"  Composite Score:       {composite:.4f}")
    print(f"  Baseline:              0.2800")
    print(f"  Above baseline:        {'YES' if composite > 0.28 else 'NO'}")

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "action_models.pkl"), "wb") as f:
        pickle.dump(action_models, f)
    with open(os.path.join(MODEL_DIR, "point_models.pkl"), "wb") as f:
        pickle.dump(point_models, f)
    with open(os.path.join(MODEL_DIR, "server_models.pkl"), "wb") as f:
        pickle.dump(server_models, f)
    with open(os.path.join(MODEL_DIR, "player_stats.pkl"), "wb") as f:
        pickle.dump(player_stats, f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    print(f"\nModels saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
