"""V4 Fast: Same as V4 Ultimate but skip SMOTE + stacking for speed.
Uses saved feature selection from V4 if available, otherwise re-selects.
Focuses on getting OOF saved and blended submission.
"""
import sys, os, time, warnings, gc
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data

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


def generate_combinations(X, feature_names, top_k=80, top_indices=None, valid_mask=None):
    """Generate combinations from top features.

    IMPORTANT: top_indices and valid_mask must be computed from TRAINING data
    and reused for test data to ensure feature alignment.
    """
    from itertools import combinations as combs
    print(f"  Generating combinations from top {top_k} features...")
    t0 = time.time()

    if top_indices is None:
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[::-1][:top_k]

    new_features = []
    new_names = []

    for i in top_indices:
        new_features.append(X[:, i] ** 2)
        new_names.append(f"{feature_names[i]}_sq")

    for i, j in combs(top_indices, 2):
        new_features.append(X[:, i] * X[:, j])
        new_names.append(f"{feature_names[i]}_x_{feature_names[j]}")
        new_features.append(X[:, i] + X[:, j])
        new_names.append(f"{feature_names[i]}_p_{feature_names[j]}")
        safe_d = np.where(np.abs(X[:, j]) > 0.001, X[:, j], 1.0)
        new_features.append(X[:, i] / safe_d)
        new_names.append(f"{feature_names[i]}_d_{feature_names[j]}")
        new_features.append(np.abs(X[:, i] - X[:, j]))
        new_names.append(f"{feature_names[i]}_abs_{feature_names[j]}")

    new_X = np.column_stack(new_features).astype(np.float32)
    new_X = np.nan_to_num(new_X, nan=0, posinf=0, neginf=0)

    if valid_mask is None:
        valid_mask = (np.std(new_X, axis=0) > 1e-10) & \
                     (~np.any(np.isnan(new_X), axis=0)) & \
                     (~np.any(np.isinf(new_X), axis=0))

    new_X = new_X[:, valid_mask]
    new_names = [n for n, v in zip(new_names, valid_mask) if v]

    print(f"  Generated {new_X.shape[1]} features in {time.time()-t0:.1f}s")
    return new_X, new_names, valid_mask, top_indices


def feature_selection_gain(X, y, n_classes, top_k=600, task="multi"):
    """Select top features by XGBoost gain."""
    import xgboost as xgb
    print(f"  XGBoost gain selection (top {top_k} from {X.shape[1]})...")
    t0 = time.time()

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
        if idx < X.shape[1]:
            feat_gains[idx] = gain

    sorted_feats = sorted(feat_gains.items(), key=lambda x: x[1], reverse=True)
    selected = [idx for idx, _ in sorted_feats[:top_k]]
    print(f"    Selected {len(selected)} features in {time.time()-t0:.1f}s")
    return selected


def main():
    t_start = time.time()
    print("=" * 70)
    print("V4 FAST PIPELINE (no SMOTE, no stacking)")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)

    # Build V4 features
    print("\n--- Building V4 features ---")
    t0 = time.time()
    try:
        from features_v4 import build_features_v4, compute_global_stats_v4, get_feature_names_v4
        global_stats = compute_global_stats_v4(train_df)
        feat_train = build_features_v4(train_df, is_train=True, global_stats_v4=global_stats)
        feat_test = build_features_v4(test_df, is_train=False, global_stats_v4=global_stats)
        feature_names = get_feature_names_v4(feat_train)
    except Exception as e:
        print(f"  V4 failed ({e}), falling back to V3")
        from features_v3 import build_features_v3, compute_global_stats, get_feature_names_v3
        global_stats = compute_global_stats(train_df)
        feat_train = build_features_v3(train_df, is_train=True, global_stats=global_stats)
        feat_test = build_features_v3(test_df, is_train=False, global_stats=global_stats)
        feature_names = get_feature_names_v3(feat_train)

    print(f"  {len(feature_names)} features, {len(feat_train)} samples in {time.time()-t0:.1f}s")

    X = feat_train[feature_names].values.astype(np.float32)
    X_test = feat_test[feature_names].values.astype(np.float32)
    y_act = feat_train["y_actionId"].values
    y_pt = feat_train["y_pointId"].values
    y_srv = feat_train["y_serverGetPoint"].values
    next_sn = feat_train["next_strikeNumber"].values
    test_next_sn = feat_test["next_strikeNumber"].values

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_train["rally_uid"].map(rally_to_match).values

    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    # Feature combinations (top 80 by variance)
    print("\n--- Feature Combinations ---")
    combo_X, combo_names, valid_mask, train_top_indices = generate_combinations(X, feature_names, top_k=80)
    combo_test_X, _, _, _ = generate_combinations(X_test, feature_names, top_k=80,
                                                    top_indices=train_top_indices, valid_mask=valid_mask)

    X_all = np.hstack([X, combo_X]).astype(np.float32)
    X_test_all = np.hstack([X_test, combo_test_X]).astype(np.float32)
    all_names = feature_names + combo_names
    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)
    X_test_all = np.nan_to_num(X_test_all, nan=0, posinf=0, neginf=0)
    print(f"  Total: {X_all.shape[1]} features")
    del combo_X, combo_test_X; gc.collect()

    # Feature selection (single stage - gain only, for speed)
    print("\n--- Feature Selection ---")
    sel_act = feature_selection_gain(X_all, y_act, N_ACTION, top_k=600, task="multi")
    sel_pt = feature_selection_gain(X_all, y_pt, N_POINT, top_k=600, task="multi")
    sel_srv = feature_selection_gain(X_all, y_srv, 2, top_k=400, task="binary")

    X_act = X_all[:, sel_act]; X_test_act = X_test_all[:, sel_act]
    X_pt = X_all[:, sel_pt]; X_test_pt = X_test_all[:, sel_pt]
    X_srv = X_all[:, sel_srv]; X_test_srv = X_test_all[:, sel_srv]
    del X_all, X_test_all; gc.collect()

    print(f"  Act: {X_act.shape[1]}, Pt: {X_pt.shape[1]}, Srv: {X_srv.shape[1]}")

    # Multi-model ensemble
    print("\n--- Training CB + XGB + LGB ---")
    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(X_act, groups=groups))

    results = {}
    for model_name in ["CB", "XGB", "LGB"]:
        oof_act = np.zeros((len(X_act), N_ACTION))
        oof_pt = np.zeros((len(X_pt), N_POINT))
        oof_srv = np.zeros(len(X_srv))
        t_act = np.zeros((len(X_test_act), N_ACTION))
        t_pt = np.zeros((len(X_test_pt), N_POINT))
        t_srv = np.zeros(len(X_test_srv))

        print(f"\n  --- {model_name} ---")
        for fold, (tr_idx, val_idx) in enumerate(fold_splits):
            t0 = time.time()

            if model_name == "CB":
                # Action
                m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                                       loss_function="MultiClass", classes_count=N_ACTION,
                                       auto_class_weights="Balanced", early_stopping_rounds=200,
                                       verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                                       bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
                m.fit(X_act[tr_idx], y_act[tr_idx], eval_set=(X_act[val_idx], y_act[val_idx]))
                oof_act[val_idx] = m.predict_proba(X_act[val_idx])
                t_act += m.predict_proba(X_test_act) / N_FOLDS

                # Point
                m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                                       loss_function="MultiClass", classes_count=N_POINT,
                                       auto_class_weights="Balanced", early_stopping_rounds=200,
                                       verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                                       bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
                m.fit(X_pt[tr_idx], y_pt[tr_idx], eval_set=(X_pt[val_idx], y_pt[val_idx]))
                oof_pt[val_idx] = m.predict_proba(X_pt[val_idx])
                t_pt += m.predict_proba(X_test_pt) / N_FOLDS

                # Server
                m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                                       loss_function="Logloss", auto_class_weights="Balanced",
                                       early_stopping_rounds=200, verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
                m.fit(X_srv[tr_idx], y_srv[tr_idx], eval_set=(X_srv[val_idx], y_srv[val_idx]))
                oof_srv[val_idx] = m.predict_proba(X_srv[val_idx])[:, 1]
                t_srv += m.predict_proba(X_test_srv)[:, 1] / N_FOLDS

            elif model_name == "XGB":
                dtrain = xgb.DMatrix(X_act[tr_idx], label=y_act[tr_idx])
                dval = xgb.DMatrix(X_act[val_idx], label=y_act[val_idx])
                params = {"objective": "multi:softprob", "num_class": N_ACTION,
                          "eval_metric": "mlogloss", "tree_method": "hist",
                          "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                          "subsample": 0.8, "colsample_bytree": 0.7,
                          "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
                m = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                              early_stopping_rounds=200, verbose_eval=0)
                oof_act[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
                t_act += m.predict(xgb.DMatrix(X_test_act), iteration_range=(0, m.best_iteration+1)) / N_FOLDS

                dtrain = xgb.DMatrix(X_pt[tr_idx], label=y_pt[tr_idx])
                dval = xgb.DMatrix(X_pt[val_idx], label=y_pt[val_idx])
                params["num_class"] = N_POINT
                m = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                              early_stopping_rounds=200, verbose_eval=0)
                oof_pt[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
                t_pt += m.predict(xgb.DMatrix(X_test_pt), iteration_range=(0, m.best_iteration+1)) / N_FOLDS

                dtrain = xgb.DMatrix(X_srv[tr_idx], label=y_srv[tr_idx])
                dval = xgb.DMatrix(X_srv[val_idx], label=y_srv[val_idx])
                params_bin = {"objective": "binary:logistic", "eval_metric": "auc",
                              "tree_method": "hist", "learning_rate": 0.03, "max_depth": 8,
                              "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
                              "lambda": 1, "seed": RANDOM_SEED, "verbosity": 0}
                m = xgb.train(params_bin, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                              early_stopping_rounds=200, verbose_eval=0)
                oof_srv[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
                t_srv += m.predict(xgb.DMatrix(X_test_srv), iteration_range=(0, m.best_iteration+1)) / N_FOLDS

            else:  # LGB
                dtrain = lgb.Dataset(X_act[tr_idx], label=y_act[tr_idx])
                dval = lgb.Dataset(X_act[val_idx], label=y_act[val_idx], reference=dtrain)
                params = {"objective": "multiclass", "num_class": N_ACTION,
                          "metric": "multi_logloss", "learning_rate": 0.03,
                          "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                          "subsample": 0.8, "colsample_bytree": 0.7, "is_unbalance": True,
                          "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
                m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
                oof_act[val_idx] = m.predict(X_act[val_idx])
                t_act += m.predict(X_test_act) / N_FOLDS

                dtrain = lgb.Dataset(X_pt[tr_idx], label=y_pt[tr_idx])
                dval = lgb.Dataset(X_pt[val_idx], label=y_pt[val_idx], reference=dtrain)
                params["num_class"] = N_POINT
                m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
                oof_pt[val_idx] = m.predict(X_pt[val_idx])
                t_pt += m.predict(X_test_pt) / N_FOLDS

                dtrain = lgb.Dataset(X_srv[tr_idx], label=y_srv[tr_idx])
                dval = lgb.Dataset(X_srv[val_idx], label=y_srv[val_idx], reference=dtrain)
                params_bin = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
                              "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                              "subsample": 0.8, "colsample_bytree": 0.8, "is_unbalance": True,
                              "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
                m = lgb.train(params_bin, dtrain, num_boost_round=3000, valid_sets=[dval],
                              callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
                oof_srv[val_idx] = m.predict(X_srv[val_idx])
                t_srv += m.predict(X_test_srv) / N_FOLDS

            ar = apply_action_rules(oof_act[val_idx], next_sn[val_idx])
            f1a = macro_f1(y_act[val_idx], ar, N_ACTION)
            f1p = macro_f1(y_pt[val_idx], oof_pt[val_idx], N_POINT)
            auc = roc_auc_score(y_srv[val_idx], oof_srv[val_idx])
            ov = 0.4*f1a + 0.4*f1p + 0.2*auc
            print(f"    {model_name} Fold {fold+1}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f} ({time.time()-t0:.0f}s)")

        ar = apply_action_rules(oof_act, next_sn)
        f1a = macro_f1(y_act, ar, N_ACTION)
        f1p = macro_f1(y_pt, oof_pt, N_POINT)
        auc = roc_auc_score(y_srv, oof_srv)
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        print(f"    {model_name} OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

        results[model_name] = {
            "oof_act": oof_act, "oof_pt": oof_pt, "oof_srv": oof_srv,
            "test_act": t_act, "test_pt": t_pt, "test_srv": t_srv,
        }

    # Save all OOF
    np.savez(os.path.join(MODEL_DIR, "oof_v4_fast.npz"),
             catboost_act=results["CB"]["oof_act"], xgboost_act=results["XGB"]["oof_act"], lightgbm_act=results["LGB"]["oof_act"],
             catboost_pt=results["CB"]["oof_pt"], xgboost_pt=results["XGB"]["oof_pt"], lightgbm_pt=results["LGB"]["oof_pt"],
             catboost_srv=results["CB"]["oof_srv"], xgboost_srv=results["XGB"]["oof_srv"], lightgbm_srv=results["LGB"]["oof_srv"],
             y_act=y_act, y_pt=y_pt, y_srv=y_srv, next_sn=next_sn)
    np.savez(os.path.join(MODEL_DIR, "test_v4_fast.npz"),
             catboost_act=results["CB"]["test_act"], xgboost_act=results["XGB"]["test_act"], lightgbm_act=results["LGB"]["test_act"],
             catboost_pt=results["CB"]["test_pt"], xgboost_pt=results["XGB"]["test_pt"], lightgbm_pt=results["LGB"]["test_pt"],
             catboost_srv=results["CB"]["test_srv"], xgboost_srv=results["XGB"]["test_srv"], lightgbm_srv=results["LGB"]["test_srv"],
             test_next_sn=test_next_sn,
             rally_uids=feat_test["rally_uid"].values.astype(int))
    print("\n  OOF and test predictions saved!")

    # Optimal blend
    print("\n--- Optimal Blend Search ---")
    best_ov = -1
    best_params = None
    for w_cb in np.arange(0.2, 0.7, 0.1):
        for w_xg in np.arange(0.1, 0.7 - w_cb + 0.05, 0.1):
            w_lg = round(1.0 - w_cb - w_xg, 1)
            if w_lg < 0 or w_lg > 0.5: continue

            ba = w_cb * results["CB"]["oof_act"] + w_xg * results["XGB"]["oof_act"] + w_lg * results["LGB"]["oof_act"]
            bp = w_cb * results["CB"]["oof_pt"] + w_xg * results["XGB"]["oof_pt"] + w_lg * results["LGB"]["oof_pt"]
            bs = w_cb * results["CB"]["oof_srv"] + w_xg * results["XGB"]["oof_srv"] + w_lg * results["LGB"]["oof_srv"]

            bar = apply_action_rules(ba, next_sn)
            f1a = macro_f1(y_act, bar, N_ACTION)
            f1p = macro_f1(y_pt, bp, N_POINT)
            auc = roc_auc_score(y_srv, bs)
            ov = 0.4*f1a + 0.4*f1p + 0.2*auc
            if ov > best_ov:
                best_ov = ov
                best_params = (w_cb, w_xg, w_lg)

    w_cb, w_xg, w_lg = best_params
    print(f"  Best: CB={w_cb:.1f} XGB={w_xg:.1f} LGB={w_lg:.1f} OV={best_ov:.4f}")

    # Blend V4 with V3 and V2
    print("\n--- Blend V4 with V2+V3 ---")
    blend_act = w_cb * results["CB"]["oof_act"] + w_xg * results["XGB"]["oof_act"] + w_lg * results["LGB"]["oof_act"]
    blend_pt = w_cb * results["CB"]["oof_pt"] + w_xg * results["XGB"]["oof_pt"] + w_lg * results["LGB"]["oof_pt"]
    blend_srv = w_cb * results["CB"]["oof_srv"] + w_xg * results["XGB"]["oof_srv"] + w_lg * results["LGB"]["oof_srv"]

    test_blend_act = w_cb * results["CB"]["test_act"] + w_xg * results["XGB"]["test_act"] + w_lg * results["LGB"]["test_act"]
    test_blend_pt = w_cb * results["CB"]["test_pt"] + w_xg * results["XGB"]["test_pt"] + w_lg * results["LGB"]["test_pt"]
    test_blend_srv = w_cb * results["CB"]["test_srv"] + w_xg * results["XGB"]["test_srv"] + w_lg * results["LGB"]["test_srv"]

    # Try blending with older versions
    for ver_name, oof_file, test_file in [
        ("V3", "oof_v3_champion.npz", "test_v3_champion.npz"),
        ("V2", "oof_v2_fast.npz", "test_v2_fast.npz"),
    ]:
        oof_path = os.path.join(MODEL_DIR, oof_file)
        test_path = os.path.join(MODEL_DIR, test_file)
        if not os.path.exists(oof_path): continue

        v = np.load(oof_path)
        # Build the best blend from that version
        if "catboost_act" in v:
            va = 0.5 * v["catboost_act"] + 0.3 * v["xgboost_act"] + 0.2 * v["lightgbm_act"]
            vp = 0.5 * v["catboost_pt"] + 0.3 * v["xgboost_pt"] + 0.2 * v["lightgbm_pt"]
            vs = 0.5 * v["catboost_srv"] + 0.3 * v["xgboost_srv"] + 0.2 * v["lightgbm_srv"]
        else:
            continue

        best_w = 1.0
        best_mega = best_ov
        for w in np.arange(0.5, 1.05, 0.05):
            ma = w * blend_act + (1-w) * va
            mp = w * blend_pt + (1-w) * vp
            ms = w * blend_srv + (1-w) * vs
            mar = apply_action_rules(ma, next_sn)
            f1a = macro_f1(y_act, mar, N_ACTION)
            f1p = macro_f1(y_pt, mp, N_POINT)
            auc = roc_auc_score(y_srv, ms)
            ov = 0.4*f1a + 0.4*f1p + 0.2*auc
            if ov > best_mega:
                best_mega = ov
                best_w = w

        if best_w < 1.0:
            vt = np.load(test_path)
            if "catboost_act" in vt:
                vta = 0.5 * vt["catboost_act"] + 0.3 * vt["xgboost_act"] + 0.2 * vt["lightgbm_act"]
                vtp = 0.5 * vt["catboost_pt"] + 0.3 * vt["xgboost_pt"] + 0.2 * vt["lightgbm_pt"]
                vts = 0.5 * vt["catboost_srv"] + 0.3 * vt["xgboost_srv"] + 0.2 * vt["lightgbm_srv"]
                test_blend_act = best_w * test_blend_act + (1-best_w) * vta
                test_blend_pt = best_w * test_blend_pt + (1-best_w) * vtp
                test_blend_srv = best_w * test_blend_srv + (1-best_w) * vts

        print(f"  +{ver_name}: w_v4={best_w:.2f}, OV={best_mega:.4f}")

    # Class weight calibration
    print("\n--- Class Weight Calibration ---")
    final_act_ruled = apply_action_rules(blend_act, next_sn)

    best_act_f1 = macro_f1(y_act, final_act_ruled, N_ACTION)
    best_act_weights = np.ones(N_ACTION)
    np.random.seed(42)
    for trial in range(500):
        weights = np.ones(N_ACTION)
        n_perturb = np.random.randint(1, 4)
        for _ in range(n_perturb):
            cls = np.random.randint(0, N_ACTION)
            weights[cls] = np.random.uniform(0.5, 5.0)
        weighted = final_act_ruled * weights
        weighted /= weighted.sum(axis=1, keepdims=True)
        f1 = macro_f1(y_act, weighted, N_ACTION)
        if f1 > best_act_f1:
            best_act_f1 = f1
            best_act_weights = weights.copy()

    best_pt_f1 = macro_f1(y_pt, blend_pt, N_POINT)
    best_pt_weights = np.ones(N_POINT)
    for trial in range(500):
        weights = np.ones(N_POINT)
        n_perturb = np.random.randint(1, 3)
        for _ in range(n_perturb):
            cls = np.random.randint(0, N_POINT)
            weights[cls] = np.random.uniform(0.5, 5.0)
        weighted = blend_pt * weights
        weighted /= weighted.sum(axis=1, keepdims=True)
        f1 = macro_f1(y_pt, weighted, N_POINT)
        if f1 > best_pt_f1:
            best_pt_f1 = f1
            best_pt_weights = weights.copy()

    # Final calibrated OOF
    cal_act = final_act_ruled * best_act_weights
    cal_act /= cal_act.sum(axis=1, keepdims=True)
    cal_pt = blend_pt * best_pt_weights
    cal_pt /= cal_pt.sum(axis=1, keepdims=True)

    f1a = macro_f1(y_act, cal_act, N_ACTION)
    f1p = macro_f1(y_pt, cal_pt, N_POINT)
    auc = roc_auc_score(y_srv, blend_srv)
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc
    print(f"  Calibrated OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # Generate submission
    final_act = apply_action_rules(test_blend_act, test_next_sn)
    final_act = final_act * best_act_weights
    final_act /= final_act.sum(axis=1, keepdims=True)
    final_pt = test_blend_pt * best_pt_weights
    final_pt /= final_pt.sum(axis=1, keepdims=True)

    submission = pd.DataFrame({
        "rally_uid": feat_test["rally_uid"].values.astype(int),
        "actionId": np.argmax(final_act, axis=1).astype(int),
        "pointId": np.argmax(final_pt, axis=1).astype(int),
        "serverGetPoint": (test_blend_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out = os.path.join(SUBMISSION_DIR, "submission_v4_fast.csv")
    submission.to_csv(out, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nSaved: {out}")
    print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId: {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  serverGetPoint: {submission.serverGetPoint.value_counts().to_dict()}")
    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
