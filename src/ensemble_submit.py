"""Ensemble: combine Transformer V2 + CatBoost predictions for best submission."""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS
from data_cleaning import clean_data
from features import build_features, compute_player_stats, get_feature_names
from transformer_model import prepare_sequences
from transformer_v2 import PingPongTransformerV2, PingPongDatasetV2

import torch
import torch.nn.functional as F

N_ACTION, N_POINT = 19, 10
SERVE_OK = {0, 15, 16, 17, 18}
SERVE_FORBIDDEN = {15, 16, 17, 18}


def macro_f1(y_true, y_probs, n_classes):
    y_pred = np.argmax(y_probs, axis=1)
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def apply_action_rules(probs, next_sns):
    preds = probs.copy()
    for i in range(len(preds)):
        sn = next_sns[i]
        if sn == 1:
            mask = np.zeros(preds.shape[1])
            for a in SERVE_OK:
                if a < preds.shape[1]: mask[a] = 1.0
            preds[i] *= mask
        elif sn == 2:
            for a in SERVE_FORBIDDEN:
                if a < preds.shape[1]: preds[i, a] = 0.0
        total = preds[i].sum()
        if total > 0: preds[i] /= total
        else: preds[i] = np.ones(preds.shape[1]) / preds.shape[1]
    return preds


def get_catboost_predictions(train_df, test_df, player_stats):
    """Train CatBoost 5-fold and return OOF + test predictions."""
    from catboost import CatBoostClassifier

    feat_train = build_features(train_df, is_train=True, player_stats=player_stats)
    feat_test = build_features(test_df, is_train=False, player_stats=player_stats)
    feature_names = get_feature_names(feat_train)

    X_train = feat_train[feature_names].values
    y_act = feat_train["y_actionId"].values
    y_pt = feat_train["y_pointId"].values
    y_srv = feat_train["y_serverGetPoint"].values
    next_sn = feat_train["next_strikeNumber"].values

    X_test = feat_test[feature_names].values
    test_next_sn = feat_test["next_strikeNumber"].values

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_train["rally_uid"].map(rally_to_match).values

    gkf = GroupKFold(n_splits=N_FOLDS)

    # OOF predictions
    oof_act = np.zeros((len(X_train), N_ACTION))
    oof_pt = np.zeros((len(X_train), N_POINT))
    oof_srv = np.zeros(len(X_train))

    # Test predictions (averaged)
    test_act = np.zeros((len(X_test), N_ACTION))
    test_pt = np.zeros((len(X_test), N_POINT))
    test_srv = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, groups=groups)):
        print(f"  CatBoost Fold {fold+1}...")
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]

        # Action
        m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8,
                               loss_function="MultiClass", classes_count=N_ACTION,
                               auto_class_weights="Balanced", early_stopping_rounds=100,
                               verbose=0, random_seed=42)
        m.fit(X_tr, y_act[tr_idx], eval_set=(X_val, y_act[val_idx]))
        oof_act[val_idx] = m.predict_proba(X_val)
        test_act += m.predict_proba(X_test) / N_FOLDS

        # Point
        m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8,
                               loss_function="MultiClass", classes_count=N_POINT,
                               auto_class_weights="Balanced", early_stopping_rounds=100,
                               verbose=0, random_seed=42)
        m.fit(X_tr, y_pt[tr_idx], eval_set=(X_val, y_pt[val_idx]))
        oof_pt[val_idx] = m.predict_proba(X_val)
        test_pt += m.predict_proba(X_test) / N_FOLDS

        # Server
        m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               early_stopping_rounds=100, verbose=0, random_seed=42)
        m.fit(X_tr, y_srv[tr_idx], eval_set=(X_val, y_srv[val_idx]))
        oof_srv[val_idx] = m.predict_proba(X_val)[:, 1]
        test_srv += m.predict_proba(X_test)[:, 1] / N_FOLDS

    # Evaluate CatBoost alone
    oof_act_ruled = apply_action_rules(oof_act, next_sn)
    f1a = macro_f1(y_act, oof_act_ruled, N_ACTION)
    f1p = macro_f1(y_pt, oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, oof_srv)
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc
    print(f"  CatBoost OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    return {
        "oof_act": oof_act, "oof_pt": oof_pt, "oof_srv": oof_srv,
        "test_act": test_act, "test_pt": test_pt, "test_srv": test_srv,
        "y_act": y_act, "y_pt": y_pt, "y_srv": y_srv,
        "next_sn": next_sn, "test_next_sn": test_next_sn,
    }


def get_transformer_predictions(train_df, test_df, n_players, device):
    """Load trained Transformer V2 models and get OOF + test predictions."""
    train_samples = prepare_sequences(train_df, is_train=True)
    test_samples = prepare_sequences(test_df, is_train=False)

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = np.array([rally_to_match.get(s["rally_uid"], 0) for s in train_samples])

    gkf = GroupKFold(n_splits=N_FOLDS)
    max_seq = 50

    oof_act = np.zeros((len(train_samples), N_ACTION))
    oof_pt = np.zeros((len(train_samples), N_POINT))
    oof_srv = np.zeros(len(train_samples))

    test_act = np.zeros((len(test_samples), N_ACTION))
    test_pt = np.zeros((len(test_samples), N_POINT))
    test_srv = np.zeros(len(test_samples))

    test_ds = PingPongDatasetV2(test_samples, max_seq_len=max_seq)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_samples)), groups=groups)):
        model_path = os.path.join(MODEL_DIR, f"transformer_v2_fold{fold}.pt")
        if not os.path.exists(model_path):
            print(f"  Transformer V2 fold {fold} not found, skipping")
            continue

        print(f"  Loading Transformer V2 fold {fold+1}...")
        model = PingPongTransformerV2(d_model=256, nhead=8, n_layers=3, n_players=n_players)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device).eval()

        # OOF predictions
        val_s = [train_samples[i] for i in val_idx]
        val_ds = PingPongDatasetV2(val_s, max_seq_len=max_seq)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256)

        a_list, p_list, s_list = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                al, pl, sl = model(
                    batch["cat_seq"].to(device), batch["num_seq"].to(device),
                    batch["context"].to(device), batch["player_ids"].to(device),
                    batch["mask"].to(device))
                a_list.append(F.softmax(al.float(), dim=-1).cpu().numpy())
                p_list.append(F.softmax(pl.float(), dim=-1).cpu().numpy())
                s_list.append(torch.sigmoid(sl.float()).cpu().numpy())
        oof_act[val_idx] = np.concatenate(a_list)
        oof_pt[val_idx] = np.concatenate(p_list)
        oof_srv[val_idx] = np.concatenate(s_list)

        # Test predictions
        a_list, p_list, s_list = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                al, pl, sl = model(
                    batch["cat_seq"].to(device), batch["num_seq"].to(device),
                    batch["context"].to(device), batch["player_ids"].to(device),
                    batch["mask"].to(device))
                a_list.append(F.softmax(al.float(), dim=-1).cpu().numpy())
                p_list.append(F.softmax(pl.float(), dim=-1).cpu().numpy())
                s_list.append(torch.sigmoid(sl.float()).cpu().numpy())
        test_act += np.concatenate(a_list) / N_FOLDS
        test_pt += np.concatenate(p_list) / N_FOLDS
        test_srv += np.concatenate(s_list) / N_FOLDS
        model.cpu()

    # Evaluate
    y_act = np.array([s["y_action"] for s in train_samples])
    y_pt = np.array([s["y_point"] for s in train_samples])
    y_srv = np.array([s["y_server"] for s in train_samples])
    next_sn = np.array([len(s["cat_seq"]) + 1 for s in train_samples])
    test_next_sn = np.array([len(s["cat_seq"]) + 1 for s in test_samples])

    oof_act_ruled = apply_action_rules(oof_act, next_sn)
    f1a = macro_f1(y_act, oof_act_ruled, N_ACTION)
    f1p = macro_f1(y_pt, oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, oof_srv)
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc
    print(f"  Transformer V2 OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    return {
        "oof_act": oof_act, "oof_pt": oof_pt, "oof_srv": oof_srv,
        "test_act": test_act, "test_pt": test_pt, "test_srv": test_srv,
        "y_act": y_act, "y_pt": y_pt, "y_srv": y_srv,
        "next_sn": next_sn, "test_next_sn": test_next_sn,
        "test_rally_uids": [s["rally_uid"] for s in test_samples],
    }


def search_blend_weights(cb_results, tf_results):
    """Search optimal blend weight between CatBoost and Transformer."""
    # Note: they may have different number of samples (GBDT uses flat features)
    # For simplicity, evaluate on transformer's OOF which has more samples
    best_ov = -1
    best_w = 0.5

    for w_tf in np.arange(0.0, 1.05, 0.05):
        w_cb = 1 - w_tf
        # We can only blend test predictions since OOF are different indices
        # Just search based on individual scores
        pass

    # Simple: try different weights on test and pick the one that maximizes diversity
    # Since we can't blend OOF easily (different sample sets), just use 0.5
    return 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    n_players = len(player_map)

    # CatBoost predictions
    print("\n--- CatBoost ---")
    player_stats = compute_player_stats(train_df)
    cb = get_catboost_predictions(train_df, test_df, player_stats)

    # Transformer V2 predictions
    print("\n--- Transformer V2 ---")
    tf = get_transformer_predictions(train_df, test_df, n_players, device)

    # Blend test predictions
    print("\n--- Ensemble ---")
    for w_tf in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w_cb = 1 - w_tf
        blend_act = w_tf * tf["test_act"] + w_cb * cb["test_act"]
        blend_pt = w_tf * tf["test_pt"] + w_cb * cb["test_pt"]
        blend_srv = w_tf * tf["test_srv"] + w_cb * cb["test_srv"]
        print(f"  w_tf={w_tf:.1f}: ready for submission")

    # Use 0.5 blend for now
    w_tf = 0.5
    blend_act = w_tf * tf["test_act"] + (1-w_tf) * cb["test_act"]
    blend_pt = w_tf * tf["test_pt"] + (1-w_tf) * cb["test_pt"]
    blend_srv = w_tf * tf["test_srv"] + (1-w_tf) * cb["test_srv"]

    # Apply rules
    test_next_sn = tf["test_next_sn"]
    blend_act = apply_action_rules(blend_act, test_next_sn)

    submission = pd.DataFrame({
        "rally_uid": tf["test_rally_uids"],
        "actionId": np.argmax(blend_act, axis=1).astype(int),
        "pointId": np.argmax(blend_pt, axis=1).astype(int),
        "serverGetPoint": (blend_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_ensemble.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nEnsemble submission saved: {out_path} ({submission.shape})")


if __name__ == "__main__":
    main()
