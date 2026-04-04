"""V3 Champion Pipeline: 802 base features + feature combinations +
multi-model ensemble (CatBoost + XGBoost + LightGBM) with feature selection.

Strategy:
1. Build V3 features (802 base)
2. Generate top-K feature combinations (multiply, add, ratio) → ~5000+ total
3. Feature selection via XGBoost gain (top 800)
4. Train CatBoost + XGBoost + LightGBM with selected features
5. Optimal blend + post-processing rules
"""
import sys, os, time, warnings, gc
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data
from features_v3 import build_features_v3, compute_global_stats, get_feature_names_v3

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


def generate_combinations_fast(X, feature_names, top_k=30, valid_mask=None):
    """Generate pairwise feature combinations for top-K features by variance.
    Uses top_k=30 for speed (30*29/2 = 435 pairs × 3 ops = 1305 + 30 sq = 1335).
    If valid_mask is provided, apply it instead of computing from data.
    """
    print(f"  Generating combinations from top {top_k} features by variance...")
    t0 = time.time()

    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[::-1][:top_k]

    new_features = []
    new_names = []

    # Squared
    for i in top_indices:
        new_features.append(X[:, i] ** 2)
        new_names.append(f"{feature_names[i]}_sq")

    # Pairwise multiply, add, ratio
    for i, j in combinations(top_indices, 2):
        new_features.append(X[:, i] * X[:, j])
        new_names.append(f"{feature_names[i]}_x_{feature_names[j]}")
        new_features.append(X[:, i] + X[:, j])
        new_names.append(f"{feature_names[i]}_p_{feature_names[j]}")
        safe_d = np.where(np.abs(X[:, j]) > 0.001, X[:, j], 1.0)
        new_features.append(X[:, i] / safe_d)
        new_names.append(f"{feature_names[i]}_d_{feature_names[j]}")

    new_X = np.column_stack(new_features).astype(np.float32)
    new_X = np.nan_to_num(new_X, nan=0, posinf=0, neginf=0)

    if valid_mask is None:
        # Compute valid mask from training data
        valid_mask = (np.std(new_X, axis=0) > 1e-10) & \
                     (~np.any(np.isnan(new_X), axis=0)) & \
                     (~np.any(np.isinf(new_X), axis=0))

    new_X = new_X[:, valid_mask]
    new_names = [n for n, v in zip(new_names, valid_mask) if v]

    print(f"  Generated {new_X.shape[1]} combination features in {time.time()-t0:.1f}s")
    return new_X, new_names, valid_mask


def feature_selection_gain(X, y, n_classes, top_k=800, task="multi"):
    """Select top features by XGBoost gain."""
    import xgboost as xgb
    print(f"  XGBoost gain selection (top {top_k} from {X.shape[1]})...")

    dtrain = xgb.DMatrix(X, label=y)
    if task == "multi":
        params = {"objective": "multi:softprob", "num_class": n_classes,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8,
                  "colsample_bytree": 0.5, "seed": RANDOM_SEED, "verbosity": 0}
    else:
        params = {"objective": "binary:logistic", "eval_metric": "auc",
                  "tree_method": "hist", "learning_rate": 0.1, "max_depth": 6,
                  "subsample": 0.8, "colsample_bytree": 0.5, "seed": RANDOM_SEED, "verbosity": 0}

    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=0)
    importance = model.get_score(importance_type='gain')

    feat_gains = {}
    for fname, gain in importance.items():
        idx = int(fname.replace('f', ''))
        feat_gains[idx] = gain

    sorted_feats = sorted(feat_gains.items(), key=lambda x: x[1], reverse=True)
    selected = [idx for idx, _ in sorted_feats[:top_k]]
    print(f"    Selected {len(selected)} features")
    return selected


def compute_sample_weights(next_sn_train, y_act):
    """Weight samples by test strikeNumber distribution and class frequency."""
    # Test SN distribution (from prior analysis)
    test_sn_dist = {2: 0.35, 3: 0.21, 4: 0.14, 5: 0.10, 6: 0.07,
                    7: 0.05, 8: 0.03, 9: 0.02, 10: 0.01}
    train_sn_counts = np.bincount(next_sn_train.astype(int), minlength=30)
    train_sn_counts = np.maximum(train_sn_counts, 1)

    weights = np.ones(len(next_sn_train))
    for sn, target_frac in test_sn_dist.items():
        if sn < len(train_sn_counts):
            train_frac = train_sn_counts[sn] / len(next_sn_train)
            if train_frac > 0:
                ratio = target_frac / train_frac
                weights[next_sn_train == sn] *= min(ratio, 3.0)

    # Also upweight rare action classes
    act_counts = np.bincount(y_act.astype(int), minlength=N_ACTION)
    median_count = np.median(act_counts[act_counts > 0])
    for cls in range(N_ACTION):
        if act_counts[cls] > 0 and act_counts[cls] < median_count:
            ratio = min(median_count / act_counts[cls], 2.0)
            weights[y_act == cls] *= np.sqrt(ratio)

    return weights / weights.mean()


def main():
    t_start = time.time()
    print("=" * 70)
    print("V3 CHAMPION PIPELINE: 802 features + combinations + ensemble")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    global_stats = compute_global_stats(train_df)

    print("\nBuilding V3 features (802 base)...")
    t0 = time.time()
    feat_train = build_features_v3(train_df, is_train=True, global_stats=global_stats)
    feat_test = build_features_v3(test_df, is_train=False, global_stats=global_stats)
    feature_names = get_feature_names_v3(feat_train)
    print(f"  Done in {time.time()-t0:.1f}s: {len(feature_names)} features, {len(feat_train)} samples")

    X = feat_train[feature_names].values.astype(np.float32)
    X_test = feat_test[feature_names].values.astype(np.float32)
    y_act = feat_train["y_actionId"].values
    y_pt = feat_train["y_pointId"].values
    y_srv = feat_train["y_serverGetPoint"].values
    next_sn = feat_train["next_strikeNumber"].values
    test_next_sn = feat_test["next_strikeNumber"].values

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_train["rally_uid"].map(rally_to_match).values

    # Clean NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    # ========================================
    # STEP 1: Feature Combinations
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 1: Feature Combinations")
    print(f"{'='*60}")

    combo_X, combo_names, valid_mask = generate_combinations_fast(X, feature_names, top_k=30)
    combo_test_X, _, _ = generate_combinations_fast(X_test, feature_names, top_k=30, valid_mask=valid_mask)

    X_all = np.hstack([X, combo_X]).astype(np.float32)
    X_test_all = np.hstack([X_test, combo_test_X]).astype(np.float32)
    all_names = feature_names + combo_names

    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)
    X_test_all = np.nan_to_num(X_test_all, nan=0, posinf=0, neginf=0)

    print(f"\n  Total features: {X_all.shape[1]} (base={X.shape[1]} + combo={combo_X.shape[1]})")
    del combo_X, combo_test_X
    gc.collect()

    # ========================================
    # STEP 2: Feature Selection
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 2: Feature Selection (XGBoost Gain)")
    print(f"{'='*60}")

    print("\n--- Action ---")
    sel_act = feature_selection_gain(X_all, y_act, N_ACTION, top_k=600, task="multi")
    print("\n--- Point ---")
    sel_pt = feature_selection_gain(X_all, y_pt, N_POINT, top_k=600, task="multi")
    print("\n--- Server ---")
    sel_srv = feature_selection_gain(X_all, y_srv, 2, top_k=400, task="binary")

    # Union for shared features (for models that train on all)
    all_selected = sorted(set(sel_act) | set(sel_pt) | set(sel_srv))
    print(f"\n  Union of all selected: {len(all_selected)} features")

    X_sel = X_all[:, all_selected].astype(np.float32)
    X_test_sel = X_test_all[:, all_selected].astype(np.float32)

    # Also keep task-specific subsets
    X_act = X_all[:, sel_act].astype(np.float32)
    X_test_act = X_test_all[:, sel_act].astype(np.float32)
    X_pt = X_all[:, sel_pt].astype(np.float32)
    X_test_pt = X_test_all[:, sel_pt].astype(np.float32)
    X_srv = X_all[:, sel_srv].astype(np.float32)
    X_test_srv = X_test_all[:, sel_srv].astype(np.float32)

    del X_all, X_test_all
    gc.collect()

    # ========================================
    # STEP 3: Multi-Model Ensemble Training
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 3: Multi-Model Ensemble Training")
    print(f"{'='*60}")

    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(X_sel, groups=groups))

    sample_weights = compute_sample_weights(next_sn, y_act)

    # --- CatBoost ---
    print("\n--- CatBoost ---")
    cb_oof_act = np.zeros((len(X_sel), N_ACTION))
    cb_oof_pt = np.zeros((len(X_sel), N_POINT))
    cb_oof_srv = np.zeros(len(X_sel))
    cb_test_act = np.zeros((len(X_test_sel), N_ACTION))
    cb_test_pt = np.zeros((len(X_test_sel), N_POINT))
    cb_test_srv = np.zeros(len(X_test_sel))

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        t0 = time.time()

        # Action (task-specific features)
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_ACTION,
                               auto_class_weights="Balanced", early_stopping_rounds=200,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                               bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m.fit(X_act[tr_idx], y_act[tr_idx], eval_set=(X_act[val_idx], y_act[val_idx]),
              sample_weight=sample_weights[tr_idx])
        cb_oof_act[val_idx] = m.predict_proba(X_act[val_idx])
        cb_test_act += m.predict_proba(X_test_act) / N_FOLDS

        # Point
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_POINT,
                               auto_class_weights="Balanced", early_stopping_rounds=200,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                               bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m.fit(X_pt[tr_idx], y_pt[tr_idx], eval_set=(X_pt[val_idx], y_pt[val_idx]))
        cb_oof_pt[val_idx] = m.predict_proba(X_pt[val_idx])
        cb_test_pt += m.predict_proba(X_test_pt) / N_FOLDS

        # Server
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               early_stopping_rounds=200, verbose=0,
                               random_seed=RANDOM_SEED, l2_leaf_reg=3)
        m.fit(X_srv[tr_idx], y_srv[tr_idx], eval_set=(X_srv[val_idx], y_srv[val_idx]))
        cb_oof_srv[val_idx] = m.predict_proba(X_srv[val_idx])[:, 1]
        cb_test_srv += m.predict_proba(X_test_srv)[:, 1] / N_FOLDS

        act_r = apply_action_rules(cb_oof_act[val_idx], next_sn[val_idx])
        f1a = macro_f1(y_act[val_idx], act_r, N_ACTION)
        f1p = macro_f1(y_pt[val_idx], cb_oof_pt[val_idx], N_POINT)
        auc = roc_auc_score(y_srv[val_idx], cb_oof_srv[val_idx])
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        print(f"  CB Fold {fold+1}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f} ({time.time()-t0:.0f}s)")

    act_r = apply_action_rules(cb_oof_act, next_sn)
    f1a = macro_f1(y_act, act_r, N_ACTION)
    f1p = macro_f1(y_pt, cb_oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, cb_oof_srv)
    print(f"  CB OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={0.4*f1a+0.4*f1p+0.2*auc:.4f}")

    # --- XGBoost ---
    print("\n--- XGBoost ---")
    xg_oof_act = np.zeros((len(X_sel), N_ACTION))
    xg_oof_pt = np.zeros((len(X_sel), N_POINT))
    xg_oof_srv = np.zeros(len(X_sel))
    xg_test_act = np.zeros((len(X_test_sel), N_ACTION))
    xg_test_pt = np.zeros((len(X_test_sel), N_POINT))
    xg_test_srv = np.zeros(len(X_test_sel))

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        t0 = time.time()

        # Action
        dtrain = xgb.DMatrix(X_act[tr_idx], label=y_act[tr_idx], weight=sample_weights[tr_idx])
        dval = xgb.DMatrix(X_act[val_idx], label=y_act[val_idx])
        params = {"objective": "multi:softprob", "num_class": N_ACTION,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                  "subsample": 0.8, "colsample_bytree": 0.7,
                  "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        m = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        xg_oof_act[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        xg_test_act += m.predict(xgb.DMatrix(X_test_act), iteration_range=(0, m.best_iteration+1)) / N_FOLDS

        # Point
        dtrain = xgb.DMatrix(X_pt[tr_idx], label=y_pt[tr_idx])
        dval = xgb.DMatrix(X_pt[val_idx], label=y_pt[val_idx])
        params["num_class"] = N_POINT
        m = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        xg_oof_pt[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        xg_test_pt += m.predict(xgb.DMatrix(X_test_pt), iteration_range=(0, m.best_iteration+1)) / N_FOLDS

        # Server
        dtrain = xgb.DMatrix(X_srv[tr_idx], label=y_srv[tr_idx])
        dval = xgb.DMatrix(X_srv[val_idx], label=y_srv[val_idx])
        params_bin = {"objective": "binary:logistic", "eval_metric": "auc",
                      "tree_method": "hist", "learning_rate": 0.03, "max_depth": 8,
                      "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
                      "lambda": 1, "seed": RANDOM_SEED, "verbosity": 0}
        m = xgb.train(params_bin, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        xg_oof_srv[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        xg_test_srv += m.predict(xgb.DMatrix(X_test_srv), iteration_range=(0, m.best_iteration+1)) / N_FOLDS

        act_r = apply_action_rules(xg_oof_act[val_idx], next_sn[val_idx])
        f1a = macro_f1(y_act[val_idx], act_r, N_ACTION)
        f1p = macro_f1(y_pt[val_idx], xg_oof_pt[val_idx], N_POINT)
        auc = roc_auc_score(y_srv[val_idx], xg_oof_srv[val_idx])
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        print(f"  XGB Fold {fold+1}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f} ({time.time()-t0:.0f}s)")

    act_r = apply_action_rules(xg_oof_act, next_sn)
    f1a = macro_f1(y_act, act_r, N_ACTION)
    f1p = macro_f1(y_pt, xg_oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, xg_oof_srv)
    print(f"  XGB OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={0.4*f1a+0.4*f1p+0.2*auc:.4f}")

    # --- LightGBM ---
    print("\n--- LightGBM ---")
    lg_oof_act = np.zeros((len(X_sel), N_ACTION))
    lg_oof_pt = np.zeros((len(X_sel), N_POINT))
    lg_oof_srv = np.zeros(len(X_sel))
    lg_test_act = np.zeros((len(X_test_sel), N_ACTION))
    lg_test_pt = np.zeros((len(X_test_sel), N_POINT))
    lg_test_srv = np.zeros(len(X_test_sel))

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        t0 = time.time()

        # Action
        dtrain = lgb.Dataset(X_act[tr_idx], label=y_act[tr_idx], weight=sample_weights[tr_idx])
        dval = lgb.Dataset(X_act[val_idx], label=y_act[val_idx], reference=dtrain)
        params = {"objective": "multiclass", "num_class": N_ACTION,
                  "metric": "multi_logloss", "learning_rate": 0.03,
                  "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                  "subsample": 0.8, "colsample_bytree": 0.7, "is_unbalance": True,
                  "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        lg_oof_act[val_idx] = m.predict(X_act[val_idx])
        lg_test_act += m.predict(X_test_act) / N_FOLDS

        # Point
        dtrain = lgb.Dataset(X_pt[tr_idx], label=y_pt[tr_idx])
        dval = lgb.Dataset(X_pt[val_idx], label=y_pt[val_idx], reference=dtrain)
        params["num_class"] = N_POINT
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        lg_oof_pt[val_idx] = m.predict(X_pt[val_idx])
        lg_test_pt += m.predict(X_test_pt) / N_FOLDS

        # Server
        dtrain = lgb.Dataset(X_srv[tr_idx], label=y_srv[tr_idx])
        dval = lgb.Dataset(X_srv[val_idx], label=y_srv[val_idx], reference=dtrain)
        params_bin = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
                      "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                      "subsample": 0.8, "colsample_bytree": 0.8, "is_unbalance": True,
                      "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        m = lgb.train(params_bin, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        lg_oof_srv[val_idx] = m.predict(X_srv[val_idx])
        lg_test_srv += m.predict(X_test_srv) / N_FOLDS

        act_r = apply_action_rules(lg_oof_act[val_idx], next_sn[val_idx])
        f1a = macro_f1(y_act[val_idx], act_r, N_ACTION)
        f1p = macro_f1(y_pt[val_idx], lg_oof_pt[val_idx], N_POINT)
        auc = roc_auc_score(y_srv[val_idx], lg_oof_srv[val_idx])
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        print(f"  LGB Fold {fold+1}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f} ({time.time()-t0:.0f}s)")

    act_r = apply_action_rules(lg_oof_act, next_sn)
    f1a = macro_f1(y_act, act_r, N_ACTION)
    f1p = macro_f1(y_pt, lg_oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, lg_oof_srv)
    print(f"  LGB OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={0.4*f1a+0.4*f1p+0.2*auc:.4f}")

    # ========================================
    # STEP 4: Optimal Blend Search
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 4: Optimal Blend Search")
    print(f"{'='*60}")

    best_ov = -1
    best_blend = None
    best_params = None

    # Grid search over blend weights
    for w_cb in np.arange(0.2, 0.8, 0.1):
        for w_xg in np.arange(0.1, 0.8 - w_cb + 0.05, 0.1):
            w_lg = round(1.0 - w_cb - w_xg, 1)
            if w_lg < 0 or w_lg > 0.5:
                continue

            blend_act = w_cb * cb_oof_act + w_xg * xg_oof_act + w_lg * lg_oof_act
            blend_pt = w_cb * cb_oof_pt + w_xg * xg_oof_pt + w_lg * lg_oof_pt
            blend_srv = w_cb * cb_oof_srv + w_xg * xg_oof_srv + w_lg * lg_oof_srv

            ba_r = apply_action_rules(blend_act, next_sn)
            f1a = macro_f1(y_act, ba_r, N_ACTION)
            f1p = macro_f1(y_pt, blend_pt, N_POINT)
            auc = roc_auc_score(y_srv, blend_srv)
            ov = 0.4*f1a + 0.4*f1p + 0.2*auc

            if ov > best_ov:
                best_ov = ov
                best_params = (w_cb, w_xg, w_lg)
                best_blend = (blend_act.copy(), blend_pt.copy(), blend_srv.copy())

    w_cb, w_xg, w_lg = best_params
    print(f"\n  Best V3 blend: CB={w_cb:.1f} XGB={w_xg:.1f} LGB={w_lg:.1f}")
    ba_r = apply_action_rules(best_blend[0], next_sn)
    f1a = macro_f1(y_act, ba_r, N_ACTION)
    f1p = macro_f1(y_pt, best_blend[1], N_POINT)
    auc = roc_auc_score(y_srv, best_blend[2])
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc
    print(f"  V3 OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # ========================================
    # STEP 5: Blend with V2 predictions
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 5: Blend V3 with V2 Ensemble")
    print(f"{'='*60}")

    v2_oof_file = os.path.join(MODEL_DIR, "oof_v2_fast.npz")
    if os.path.exists(v2_oof_file):
        v2 = np.load(v2_oof_file)
        v2_act = 0.6 * v2["catboost_act"] + 0.4 * v2["xgboost_act"]
        v2_pt = 0.6 * v2["catboost_pt"] + 0.3 * v2["xgboost_pt"] + 0.1 * v2["lightgbm_pt"]
        v2_srv = 0.3 * v2["catboost_srv"] + 0.4 * v2["xgboost_srv"] + 0.3 * v2["lightgbm_srv"]

        best_mega_ov = -1
        best_mega_w = 0
        for w in np.arange(0, 1.05, 0.05):
            ma = w * best_blend[0] + (1-w) * v2_act
            mp = w * best_blend[1] + (1-w) * v2_pt
            ms = w * best_blend[2] + (1-w) * v2_srv
            mar = apply_action_rules(ma, next_sn)
            f1a = macro_f1(y_act, mar, N_ACTION)
            f1p = macro_f1(y_pt, mp, N_POINT)
            auc = roc_auc_score(y_srv, ms)
            ov = 0.4*f1a + 0.4*f1p + 0.2*auc
            if ov > best_mega_ov:
                best_mega_ov = ov
                best_mega_w = w
            if w in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                print(f"  w_v3={w:.2f}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

        print(f"\n  Best mega blend: w_v3={best_mega_w:.2f}, OV={best_mega_ov:.4f}")

        # Generate submission with mega blend
        v2t = np.load(os.path.join(MODEL_DIR, "test_v2_fast.npz"))
        v2_test_act = 0.6 * v2t["catboost_act"] + 0.4 * v2t["xgboost_act"]
        v2_test_pt = 0.6 * v2t["catboost_pt"] + 0.3 * v2t["xgboost_pt"] + 0.1 * v2t["lightgbm_pt"]
        v2_test_srv = 0.3 * v2t["catboost_srv"] + 0.4 * v2t["xgboost_srv"] + 0.3 * v2t["lightgbm_srv"]

        v3_test_act = w_cb * cb_test_act + w_xg * xg_test_act + w_lg * lg_test_act
        v3_test_pt = w_cb * cb_test_pt + w_xg * xg_test_pt + w_lg * lg_test_pt
        v3_test_srv = w_cb * cb_test_srv + w_xg * xg_test_srv + w_lg * lg_test_srv

        final_act = best_mega_w * v3_test_act + (1-best_mega_w) * v2_test_act
        final_pt = best_mega_w * v3_test_pt + (1-best_mega_w) * v2_test_pt
        final_srv = best_mega_w * v3_test_srv + (1-best_mega_w) * v2_test_srv
    else:
        print("  No V2 OOF found, using V3 only")
        final_act = w_cb * cb_test_act + w_xg * xg_test_act + w_lg * lg_test_act
        final_pt = w_cb * cb_test_pt + w_xg * xg_test_pt + w_lg * lg_test_pt
        final_srv = w_cb * cb_test_srv + w_xg * xg_test_srv + w_lg * lg_test_srv

    # Apply rules
    final_act = apply_action_rules(final_act, test_next_sn)

    submission = pd.DataFrame({
        "rally_uid": feat_test["rally_uid"].values.astype(int),
        "actionId": np.argmax(final_act, axis=1).astype(int),
        "pointId": np.argmax(final_pt, axis=1).astype(int),
        "serverGetPoint": (final_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_v3_champion.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId: {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  serverGetPoint: {submission.serverGetPoint.value_counts().to_dict()}")

    # Save OOF
    np.savez(os.path.join(MODEL_DIR, "oof_v3_champion.npz"),
             catboost_act=cb_oof_act, xgboost_act=xg_oof_act, lightgbm_act=lg_oof_act,
             catboost_pt=cb_oof_pt, xgboost_pt=xg_oof_pt, lightgbm_pt=lg_oof_pt,
             catboost_srv=cb_oof_srv, xgboost_srv=xg_oof_srv, lightgbm_srv=lg_oof_srv,
             y_act=y_act, y_pt=y_pt, y_srv=y_srv, next_sn=next_sn)
    np.savez(os.path.join(MODEL_DIR, "test_v3_champion.npz"),
             catboost_act=cb_test_act, xgboost_act=xg_test_act, lightgbm_act=lg_test_act,
             catboost_pt=cb_test_pt, xgboost_pt=xg_test_pt, lightgbm_pt=lg_test_pt,
             catboost_srv=cb_test_srv, xgboost_srv=xg_test_srv, lightgbm_srv=lg_test_srv,
             test_next_sn=test_next_sn,
             rally_uids=feat_test["rally_uid"].values.astype(int))

    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
