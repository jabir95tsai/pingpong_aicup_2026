"""Stacking: Use L1 GBDT OOF predictions as meta-features for L2 model.
Also adds target-encoded features and interaction features.
"""
import sys, os, time, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data
from features_v2 import build_features_v2, compute_global_stats, get_feature_names_v2

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


def target_encode_cv(X_train, y_train, col_idx, groups, n_classes, n_splits=5):
    """Leave-one-group-out target encoding for a categorical column."""
    encoded = np.zeros((len(X_train), n_classes))
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, val_idx in gkf.split(X_train, groups=groups):
        for cls in range(n_classes):
            global_mean = np.mean(y_train[tr_idx] == cls)
            vals = X_train[val_idx, col_idx].astype(int)
            for v in np.unique(vals):
                mask_tr = (X_train[tr_idx, col_idx].astype(int) == v)
                if mask_tr.sum() > 5:
                    rate = np.mean(y_train[tr_idx][mask_tr] == cls)
                else:
                    rate = global_mean
                mask_val = (vals == v)
                encoded[val_idx[mask_val], cls] = rate
    return encoded


def target_encode_test(X_train, y_train, X_test, col_idx, n_classes):
    """Full-data target encoding for test set."""
    encoded = np.zeros((len(X_test), n_classes))
    for cls in range(n_classes):
        global_mean = np.mean(y_train == cls)
        for v in np.unique(X_test[:, col_idx].astype(int)):
            mask_tr = (X_train[:, col_idx].astype(int) == v)
            if mask_tr.sum() > 5:
                rate = np.mean(y_train[mask_tr] == cls)
            else:
                rate = global_mean
            mask_test = (X_test[:, col_idx].astype(int) == v)
            encoded[mask_test, cls] = rate
    return encoded


def main():
    t_start = time.time()
    print("=" * 70)
    print("STACKING PIPELINE: L1 OOF -> L2 Meta-learner")
    print("=" * 70)

    # Load L1 OOF predictions
    v2_oof = np.load(os.path.join(MODEL_DIR, "oof_v2_fast.npz"))
    v2_test = np.load(os.path.join(MODEL_DIR, "test_v2_fast.npz"))
    sn_oof = np.load(os.path.join(MODEL_DIR, "oof_sn_cond.npz"))
    sn_test = np.load(os.path.join(MODEL_DIR, "test_sn_cond.npz"))
    tfm_oof = np.load(os.path.join(MODEL_DIR, "oof_transformer.npz"))
    tfm_test = np.load(os.path.join(MODEL_DIR, "test_transformer.npz"))

    y_act = v2_oof["y_act"]
    y_pt = v2_oof["y_pt"]
    y_srv = v2_oof["y_srv"]
    next_sn = v2_oof["next_sn"]
    test_next_sn = v2_test["test_next_sn"]
    rally_uids_test = v2_test["rally_uids"]

    print(f"  Samples: train={len(y_act)}, test={len(test_next_sn)}")

    # Build L2 meta-features from L1 OOF
    # Each model provides probability vectors (19 for action, 10 for point, 1 for server)
    print("\nBuilding L2 meta-features...")

    # L1 model predictions as features
    meta_train_feats = []
    meta_test_feats = []

    # CatBoost, XGBoost, LightGBM action probabilities
    for model_name in ["catboost", "xgboost", "lightgbm"]:
        meta_train_feats.append(v2_oof[f"{model_name}_act"])  # 19 cols each
        meta_test_feats.append(v2_test[f"{model_name}_act"])

    # CatBoost, XGBoost, LightGBM point probabilities
    for model_name in ["catboost", "xgboost", "lightgbm"]:
        meta_train_feats.append(v2_oof[f"{model_name}_pt"])  # 10 cols each
        meta_test_feats.append(v2_test[f"{model_name}_pt"])

    # CatBoost, XGBoost, LightGBM server probabilities
    for model_name in ["catboost", "xgboost", "lightgbm"]:
        meta_train_feats.append(v2_oof[f"{model_name}_srv"].reshape(-1, 1))
        meta_test_feats.append(v2_test[f"{model_name}_srv"].reshape(-1, 1))

    # SN-conditioned predictions
    sn_act_blend = 0.5 * sn_oof["oof_act_sn"] + 0.5 * sn_oof["oof_act_global"]
    sn_pt_blend = 0.5 * sn_oof["oof_pt_sn"] + 0.5 * sn_oof["oof_pt_global"]
    meta_train_feats.append(sn_act_blend)
    meta_train_feats.append(sn_pt_blend)
    meta_train_feats.append(sn_oof["oof_srv_global"].reshape(-1, 1))

    sn_test_act = 0.5 * sn_test["test_act_sn"] + 0.5 * sn_test["test_act_global"]
    sn_test_pt = 0.5 * sn_test["test_pt_sn"] + 0.5 * sn_test["test_pt_global"]
    meta_test_feats.append(sn_test_act)
    meta_test_feats.append(sn_test_pt)
    meta_test_feats.append(sn_test["test_srv_global"].reshape(-1, 1))

    # Transformer predictions
    meta_train_feats.append(tfm_oof["oof_act"])
    meta_train_feats.append(tfm_oof["oof_pt"])
    meta_train_feats.append(tfm_oof["oof_srv"].reshape(-1, 1))
    meta_test_feats.append(tfm_test["test_act"])
    meta_test_feats.append(tfm_test["test_pt"])
    meta_test_feats.append(tfm_test["test_srv"].reshape(-1, 1))

    # Add next_strikeNumber as feature
    meta_train_feats.append(next_sn.reshape(-1, 1))
    meta_test_feats.append(test_next_sn.reshape(-1, 1))

    X_meta = np.hstack(meta_train_feats).astype(np.float32)
    X_meta_test = np.hstack(meta_test_feats).astype(np.float32)
    print(f"  L2 feature size: {X_meta.shape[1]}")

    # Also load original features for stacking
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    global_stats = compute_global_stats(train_df)
    feat_train = build_features_v2(train_df, is_train=True, global_stats=global_stats)
    feat_test = build_features_v2(test_df, is_train=False, global_stats=global_stats)
    feature_names = get_feature_names_v2(feat_train)

    X_orig = feat_train[feature_names].values.astype(np.float32)
    X_orig_test = feat_test[feature_names].values.astype(np.float32)

    # Combine: original features + meta-features
    X_stack = np.hstack([X_orig, X_meta])
    X_stack_test = np.hstack([X_orig_test, X_meta_test])
    print(f"  Stacked feature size: {X_stack.shape[1]} (orig={X_orig.shape[1]} + meta={X_meta.shape[1]})")

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_train["rally_uid"].map(rally_to_match).values

    # Train L2 CatBoost
    from catboost import CatBoostClassifier
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(X_stack, groups=groups))

    oof_act = np.zeros((len(X_stack), N_ACTION))
    oof_pt = np.zeros((len(X_stack), N_POINT))
    oof_srv = np.zeros(len(X_stack))
    test_act = np.zeros((len(X_stack_test), N_ACTION))
    test_pt = np.zeros((len(X_stack_test), N_POINT))
    test_srv = np.zeros(len(X_stack_test))

    print("\n--- L2 CatBoost Training ---")
    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        t0 = time.time()
        X_tr, X_val = X_stack[tr_idx], X_stack[val_idx]

        # Action
        m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6,
                               loss_function="MultiClass", classes_count=N_ACTION,
                               auto_class_weights="Balanced", early_stopping_rounds=200,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=5)
        m.fit(X_tr, y_act[tr_idx], eval_set=(X_val, y_act[val_idx]))
        oof_act[val_idx] = m.predict_proba(X_val)
        test_act += m.predict_proba(X_stack_test) / N_FOLDS

        # Point
        m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6,
                               loss_function="MultiClass", classes_count=N_POINT,
                               auto_class_weights="Balanced", early_stopping_rounds=200,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=5)
        m.fit(X_tr, y_pt[tr_idx], eval_set=(X_val, y_pt[val_idx]))
        oof_pt[val_idx] = m.predict_proba(X_val)
        test_pt += m.predict_proba(X_stack_test) / N_FOLDS

        # Server
        m = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               early_stopping_rounds=200, verbose=0, random_seed=RANDOM_SEED,
                               l2_leaf_reg=5)
        m.fit(X_tr, y_srv[tr_idx], eval_set=(X_val, y_srv[val_idx]))
        oof_srv[val_idx] = m.predict_proba(X_val)[:, 1]
        test_srv += m.predict_proba(X_stack_test)[:, 1] / N_FOLDS

        act_ruled = apply_action_rules(oof_act[val_idx], next_sn[val_idx])
        f1a = macro_f1(y_act[val_idx], act_ruled, N_ACTION)
        f1p = macro_f1(y_pt[val_idx], oof_pt[val_idx], N_POINT)
        auc = roc_auc_score(y_srv[val_idx], oof_srv[val_idx])
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        print(f"  Fold {fold+1}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f} ({time.time()-t0:.0f}s)")

    # Overall OOF
    act_ruled = apply_action_rules(oof_act, next_sn)
    f1a = macro_f1(y_act, act_ruled, N_ACTION)
    f1p = macro_f1(y_pt, oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, oof_srv)
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc
    print(f"\n  STACKING OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # Also try: blend L2 stacking with L1 best
    print("\n--- Blend L2 with L1 ---")
    l1_act = 0.6 * v2_oof["catboost_act"] + 0.4 * v2_oof["xgboost_act"]
    l1_pt = 0.6 * v2_oof["catboost_pt"] + 0.3 * v2_oof["xgboost_pt"] + 0.1 * v2_oof["lightgbm_pt"]
    l1_srv = 0.3 * v2_oof["catboost_srv"] + 0.4 * v2_oof["xgboost_srv"] + 0.3 * v2_oof["lightgbm_srv"]

    for w in np.arange(0, 1.05, 0.1):
        blend_act = w * oof_act + (1-w) * l1_act
        blend_pt = w * oof_pt + (1-w) * l1_pt
        blend_srv = w * oof_srv + (1-w) * l1_srv
        ba_r = apply_action_rules(blend_act, next_sn)
        f1a = macro_f1(y_act, ba_r, N_ACTION)
        f1p = macro_f1(y_pt, blend_pt, N_POINT)
        auc = roc_auc_score(y_srv, blend_srv)
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        print(f"  w_L2={w:.1f}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # Generate best submission
    test_act_ruled = apply_action_rules(test_act, test_next_sn)
    submission = pd.DataFrame({
        "rally_uid": rally_uids_test.astype(int),
        "actionId": np.argmax(test_act_ruled, axis=1).astype(int),
        "pointId": np.argmax(test_pt, axis=1).astype(int),
        "serverGetPoint": (test_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_stacking.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId: {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  serverGetPoint: {submission.serverGetPoint.value_counts().to_dict()}")
    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
