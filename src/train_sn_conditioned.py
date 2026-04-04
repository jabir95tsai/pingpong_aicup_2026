"""StrikeNumber-conditioned models: separate CatBoost for each SN range.

Key insight: the action/point distribution varies dramatically by strikeNumber.
SN=1: serves only (action {0,15,16,17,18})
SN=2: returns (dominated by action 10,1,4)
SN=3: third ball (dominated by 1,10,6)
SN>=4: rally (dominated by 1,13,2)

Training separate models for each SN range captures these distinct patterns.
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

# SN ranges for separate models
SN_BINS = {
    "sn1": lambda sn: sn == 1,        # serve
    "sn2": lambda sn: sn == 2,        # return
    "sn3": lambda sn: sn == 3,        # third ball
    "sn4_5": lambda sn: sn in (4, 5), # early rally
    "sn6p": lambda sn: sn >= 6,       # late rally
}


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


def main():
    t_start = time.time()
    print("=" * 70)
    print("SN-CONDITIONED ENSEMBLE")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)

    print("\nComputing global statistics...")
    global_stats = compute_global_stats(train_df)

    print("Building V2 features...")
    t0 = time.time()
    feat_train = build_features_v2(train_df, is_train=True, global_stats=global_stats)
    feat_test = build_features_v2(test_df, is_train=False, global_stats=global_stats)
    feature_names = get_feature_names_v2(feat_train)
    print(f"  Done in {time.time()-t0:.1f}s, Train: {feat_train.shape}, Features: {len(feature_names)}")

    X = feat_train[feature_names].values.astype(np.float32)
    y_act = feat_train["y_actionId"].values
    y_pt = feat_train["y_pointId"].values
    y_srv = feat_train["y_serverGetPoint"].values
    next_sn = feat_train["next_strikeNumber"].values
    X_test = feat_test[feature_names].values.astype(np.float32)
    test_next_sn = feat_test["next_strikeNumber"].values

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_train["rally_uid"].map(rally_to_match).values

    from catboost import CatBoostClassifier

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(X, groups=groups))

    # Strategy A: Global model (baseline)
    print("\n--- Strategy A: Global CatBoost ---")
    oof_act_global = np.zeros((len(X), N_ACTION))
    oof_pt_global = np.zeros((len(X), N_POINT))
    oof_srv_global = np.zeros(len(X))
    test_act_global = np.zeros((len(X_test), N_ACTION))
    test_pt_global = np.zeros((len(X_test), N_POINT))
    test_srv_global = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        t0 = time.time()
        X_tr, X_val = X[tr_idx], X[val_idx]

        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_ACTION,
                               auto_class_weights="Balanced", early_stopping_rounds=200,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
        m.fit(X_tr, y_act[tr_idx], eval_set=(X_val, y_act[val_idx]))
        oof_act_global[val_idx] = m.predict_proba(X_val)
        test_act_global += m.predict_proba(X_test) / N_FOLDS

        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_POINT,
                               auto_class_weights="Balanced", early_stopping_rounds=200,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
        m.fit(X_tr, y_pt[tr_idx], eval_set=(X_val, y_pt[val_idx]))
        oof_pt_global[val_idx] = m.predict_proba(X_val)
        test_pt_global += m.predict_proba(X_test) / N_FOLDS

        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               early_stopping_rounds=200, verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
        m.fit(X_tr, y_srv[tr_idx], eval_set=(X_val, y_srv[val_idx]))
        oof_srv_global[val_idx] = m.predict_proba(X_val)[:, 1]
        test_srv_global += m.predict_proba(X_test)[:, 1] / N_FOLDS

        print(f"  Fold {fold+1} done ({time.time()-t0:.0f}s)")

    act_ruled = apply_action_rules(oof_act_global, next_sn)
    f1a_global = macro_f1(y_act, act_ruled, N_ACTION)
    f1p_global = macro_f1(y_pt, oof_pt_global, N_POINT)
    auc_global = roc_auc_score(y_srv, oof_srv_global)
    ov_global = 0.4*f1a_global + 0.4*f1p_global + 0.2*auc_global
    print(f"\n  Global OOF: F1a={f1a_global:.4f} F1p={f1p_global:.4f} AUC={auc_global:.4f} OV={ov_global:.4f}")

    # Strategy B: SN-conditioned models
    print("\n--- Strategy B: SN-Conditioned CatBoost ---")
    oof_act_sn = np.zeros((len(X), N_ACTION))
    oof_pt_sn = np.zeros((len(X), N_POINT))
    test_act_sn = np.zeros((len(X_test), N_ACTION))
    test_pt_sn = np.zeros((len(X_test), N_POINT))

    for sn_name, sn_fn in SN_BINS.items():
        train_mask = np.array([sn_fn(sn) for sn in next_sn])
        test_mask = np.array([sn_fn(sn) for sn in test_next_sn])
        n_train = train_mask.sum()
        n_test = test_mask.sum()
        print(f"\n  {sn_name}: train={n_train}, test={n_test}")

        if n_train < 50 or n_test == 0:
            # Too few samples, use global prediction
            oof_act_sn[train_mask] = oof_act_global[train_mask]
            oof_pt_sn[train_mask] = oof_pt_global[train_mask]
            test_act_sn[test_mask] = test_act_global[test_mask]
            test_pt_sn[test_mask] = test_pt_global[test_mask]
            print(f"    Using global (too few samples)")
            continue

        X_sn = X[train_mask]
        y_act_sn = y_act[train_mask]
        y_pt_sn = y_pt[train_mask]
        groups_sn = groups[train_mask]
        next_sn_sn = next_sn[train_mask]

        # Map indices
        sn_indices = np.where(train_mask)[0]
        test_sn_indices = np.where(test_mask)[0]

        gkf_sn = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups_sn))))

        oof_act_part = np.zeros((len(X_sn), N_ACTION))
        oof_pt_part = np.zeros((len(X_sn), N_POINT))
        test_act_part = np.zeros((n_test, N_ACTION))
        test_pt_part = np.zeros((n_test, N_POINT))

        n_folds_actual = 0
        for fold, (tr_idx, val_idx) in enumerate(gkf_sn.split(X_sn, groups=groups_sn)):
            t0 = time.time()
            X_tr, X_val = X_sn[tr_idx], X_sn[val_idx]

            # Action
            m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=7,
                                   loss_function="MultiClass", classes_count=N_ACTION,
                                   auto_class_weights="Balanced", early_stopping_rounds=200,
                                   verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
            m.fit(X_tr, y_act_sn[tr_idx], eval_set=(X_val, y_act_sn[val_idx]))
            oof_act_part[val_idx] = m.predict_proba(X_val)
            test_act_part += m.predict_proba(X_test[test_mask])
            n_folds_actual += 1

            # Point
            m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=7,
                                   loss_function="MultiClass", classes_count=N_POINT,
                                   auto_class_weights="Balanced", early_stopping_rounds=200,
                                   verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
            m.fit(X_tr, y_pt_sn[tr_idx], eval_set=(X_val, y_pt_sn[val_idx]))
            oof_pt_part[val_idx] = m.predict_proba(X_val)
            test_pt_part += m.predict_proba(X_test[test_mask])

        test_act_part /= n_folds_actual
        test_pt_part /= n_folds_actual

        # Store back
        oof_act_sn[train_mask] = oof_act_part
        oof_pt_sn[train_mask] = oof_pt_part
        test_act_sn[test_mask] = test_act_part
        test_pt_sn[test_mask] = test_pt_part

        act_ruled_part = apply_action_rules(oof_act_part, next_sn_sn)
        f1a = macro_f1(y_act_sn, act_ruled_part, N_ACTION)
        f1p = macro_f1(y_pt_sn, oof_pt_part, N_POINT)
        print(f"    F1a={f1a:.4f} F1p={f1p:.4f}")

    act_ruled_sn = apply_action_rules(oof_act_sn, next_sn)
    f1a_sn = macro_f1(y_act, act_ruled_sn, N_ACTION)
    f1p_sn = macro_f1(y_pt, oof_pt_sn, N_POINT)
    ov_sn = 0.4*f1a_sn + 0.4*f1p_sn + 0.2*auc_global  # reuse global server
    print(f"\n  SN-Cond OOF: F1a={f1a_sn:.4f} F1p={f1p_sn:.4f} OV={ov_sn:.4f}")

    # Strategy C: Blend global + SN-conditioned
    print("\n--- Strategy C: Blend Global + SN-Cond ---")
    best_ov = -1
    best_w = 0.5
    for w in np.arange(0, 1.05, 0.1):
        blend_act = w * oof_act_sn + (1-w) * oof_act_global
        blend_pt = w * oof_pt_sn + (1-w) * oof_pt_global
        blend_act_r = apply_action_rules(blend_act, next_sn)
        f1a = macro_f1(y_act, blend_act_r, N_ACTION)
        f1p = macro_f1(y_pt, blend_pt, N_POINT)
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc_global
        if ov > best_ov:
            best_ov = ov
            best_w = w
        print(f"  w_sn={w:.1f}: F1a={f1a:.4f} F1p={f1p:.4f} OV={ov:.4f}")

    print(f"\n  Best blend: w_sn={best_w:.1f}, OV={best_ov:.4f}")

    # Generate submission with best blend
    blend_test_act = best_w * test_act_sn + (1-best_w) * test_act_global
    blend_test_pt = best_w * test_pt_sn + (1-best_w) * test_pt_global
    blend_test_act = apply_action_rules(blend_test_act, test_next_sn)

    submission = pd.DataFrame({
        "rally_uid": feat_test["rally_uid"].values.astype(int),
        "actionId": np.argmax(blend_test_act, axis=1).astype(int),
        "pointId": np.argmax(blend_test_pt, axis=1).astype(int),
        "serverGetPoint": (test_srv_global >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_sn_cond.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nSaved: {out_path} ({submission.shape})")
    print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId: {submission.pointId.value_counts().sort_index().to_dict()}")

    # Save predictions for mega-blend
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(os.path.join(MODEL_DIR, "oof_sn_cond.npz"),
             oof_act_global=oof_act_global, oof_pt_global=oof_pt_global, oof_srv_global=oof_srv_global,
             oof_act_sn=oof_act_sn, oof_pt_sn=oof_pt_sn,
             y_act=y_act, y_pt=y_pt, y_srv=y_srv, next_sn=next_sn)
    np.savez(os.path.join(MODEL_DIR, "test_sn_cond.npz"),
             test_act_global=test_act_global, test_pt_global=test_pt_global, test_srv_global=test_srv_global,
             test_act_sn=test_act_sn, test_pt_sn=test_pt_sn,
             test_next_sn=test_next_sn,
             rally_uids=feat_test["rally_uid"].values.astype(int))

    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
