"""Generate predictions on test set and create submission file."""
import sys
import os
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TEST_PATH, MODEL_DIR, SUBMISSION_DIR,
    N_ACTION_CLASSES, N_POINT_CLASSES,
)
from features import build_features, get_feature_names
from models import (
    predict_multiclass, predict_binary,
    apply_action_constraints,
)


def main():
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"  {len(test_df)} rows, {test_df.rally_uid.nunique()} rallies")

    # Load models and artifacts
    print("Loading models...")
    with open(os.path.join(MODEL_DIR, "action_models.pkl"), "rb") as f:
        action_models = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "point_models.pkl"), "rb") as f:
        point_models = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "server_models.pkl"), "rb") as f:
        server_models = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "player_stats.pkl"), "rb") as f:
        player_stats = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    print("Building features...")
    feat_df = build_features(test_df, is_train=False, player_stats=player_stats)
    X = feat_df[feature_names].values
    next_sn = feat_df["next_strikeNumber"].values
    rally_uids = feat_df["rally_uid"].values

    # Ensemble: average predictions across folds
    print("Predicting...")

    # Task 1: actionId
    action_probs = np.zeros((len(X), N_ACTION_CLASSES))
    for model in action_models:
        action_probs += predict_multiclass(model, X)
    action_probs /= len(action_models)
    action_probs = apply_action_constraints(action_probs, next_sn)
    action_preds = np.argmax(action_probs, axis=1)

    # Task 2: pointId
    point_probs = np.zeros((len(X), N_POINT_CLASSES))
    for model in point_models:
        point_probs += predict_multiclass(model, X)
    point_probs /= len(point_models)
    point_preds = np.argmax(point_probs, axis=1)

    # Task 3: serverGetPoint
    server_probs = np.zeros(len(X))
    for model in server_models:
        server_probs += predict_binary(model, X)
    server_probs /= len(server_models)
    server_preds = (server_probs >= 0.5).astype(int)

    # Create submission
    submission = pd.DataFrame({
        "rally_uid": rally_uids.astype(int),
        "actionId": action_preds.astype(int),
        "pointId": point_preds.astype(int),
        "serverGetPoint": server_preds.astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nSubmission saved to {out_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  actionId distribution:\n{submission.actionId.value_counts().sort_index()}")
    print(f"  pointId distribution:\n{submission.pointId.value_counts().sort_index()}")
    print(f"  serverGetPoint distribution:\n{submission.serverGetPoint.value_counts()}")


if __name__ == "__main__":
    main()
