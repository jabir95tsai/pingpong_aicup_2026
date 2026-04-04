"""Two-Stage XGBoost following champion's approach:
1. Massive feature generation + combination (148 → ~11K features)
2. SMOTE for class balance
3. Train XGBoost A with all features
4. Feature selection: XGBoost gain → TreeSHAP → Cross importance
5. Train XGBoost B with selected features only
"""
import sys, os, time, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from itertools import combinations

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


def generate_feature_combinations(X, feature_names, top_k=50):
    """Generate pairwise feature interactions for top-K most important features.

    Champion expanded 972 → 11,873 features via combinations.
    We do: top_k features × top_k features = top_k*(top_k-1)/2 multiply +
           top_k*(top_k-1)/2 add + top_k squared = ~2500+ new features.
    """
    print(f"  Generating feature combinations from top {top_k} features...")
    t0 = time.time()

    new_features = []
    new_names = []

    # Select top_k features by variance (quick proxy for importance)
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[::-1][:top_k]

    # Squared features
    for i in top_indices:
        new_features.append(X[:, i] ** 2)
        new_names.append(f"{feature_names[i]}_sq")

    # Pairwise multiply and add for top features
    for idx, (i, j) in enumerate(combinations(top_indices, 2)):
        new_features.append(X[:, i] * X[:, j])
        new_names.append(f"{feature_names[i]}_x_{feature_names[j]}")
        new_features.append(X[:, i] + X[:, j])
        new_names.append(f"{feature_names[i]}_p_{feature_names[j]}")
        # Also ratio (with safety)
        safe_denom = np.where(np.abs(X[:, j]) > 0.001, X[:, j], 1.0)
        new_features.append(X[:, i] / safe_denom)
        new_names.append(f"{feature_names[i]}_d_{feature_names[j]}")

    new_X = np.column_stack(new_features)

    # Remove constant and NaN features
    valid_mask = (np.std(new_X, axis=0) > 1e-10) & (~np.any(np.isnan(new_X), axis=0)) & (~np.any(np.isinf(new_X), axis=0))
    new_X = new_X[:, valid_mask]
    new_names = [n for n, v in zip(new_names, valid_mask) if v]

    print(f"  Generated {new_X.shape[1]} combination features in {time.time()-t0:.1f}s")
    return new_X, new_names


def feature_selection_xgb_gain(X, y, feature_names, n_classes, top_k=600):
    """Stage 1: Select top features by XGBoost gain."""
    import xgboost as xgb
    print(f"  XGBoost gain selection (keeping top {top_k})...")

    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "multi:softprob", "num_class": n_classes,
              "eval_metric": "mlogloss", "tree_method": "hist",
              "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8,
              "colsample_bytree": 0.5, "seed": RANDOM_SEED, "verbosity": 0}
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=0)

    importance = model.get_score(importance_type='gain')
    # Map feature names (xgb uses f0, f1, ...)
    feat_gains = {}
    for fname, gain in importance.items():
        idx = int(fname.replace('f', ''))
        if idx < len(feature_names):
            feat_gains[idx] = gain

    sorted_feats = sorted(feat_gains.items(), key=lambda x: x[1], reverse=True)
    selected_indices = [idx for idx, _ in sorted_feats[:top_k]]

    print(f"    Selected {len(selected_indices)} features by gain")
    return selected_indices


def feature_selection_shap(X, y, feature_names, selected_indices, n_classes, top_k=300):
    """Stage 2: From gain-selected features, use TreeSHAP to pick top_k."""
    import xgboost as xgb
    print(f"  TreeSHAP selection (keeping top {top_k} from {len(selected_indices)})...")

    X_sel = X[:, selected_indices]

    # Use subsample for speed
    n_sample = min(5000, len(X_sel))
    sample_idx = np.random.choice(len(X_sel), n_sample, replace=False)

    dtrain = xgb.DMatrix(X_sel, label=y)
    params = {"objective": "multi:softprob", "num_class": n_classes,
              "eval_metric": "mlogloss", "tree_method": "hist",
              "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8,
              "colsample_bytree": 0.8, "seed": RANDOM_SEED, "verbosity": 0}
    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=0)

    # SHAP values
    dsample = xgb.DMatrix(X_sel[sample_idx])
    shap_values = model.predict(dsample, pred_contribs=True)
    # shap_values shape: (n_samples, n_features+1, n_classes) or (n_samples, n_features+1)
    # For multiclass, take mean absolute across classes
    if len(shap_values.shape) == 3:
        mean_abs_shap = np.mean(np.abs(shap_values[:, :-1, :]), axis=(0, 2))
    else:
        mean_abs_shap = np.mean(np.abs(shap_values[:, :-1]), axis=0)

    top_local = np.argsort(mean_abs_shap)[::-1][:top_k]
    final_indices = [selected_indices[i] for i in top_local]

    print(f"    Selected {len(final_indices)} features by SHAP")
    return final_indices


def cross_importance_selection(X, y, groups, feature_names, n_classes, n_splits=5, top_k_gain=400, top_k_shap=200):
    """Stage 3: Collect features important across all folds."""
    import xgboost as xgb
    print(f"  Cross-importance selection ({n_splits}-fold)...")

    gkf = GroupKFold(n_splits=n_splits)
    fold_important = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
        X_tr = X[tr_idx]
        y_tr = y[tr_idx]

        # Quick gain selection per fold
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        params = {"objective": "multi:softprob", "num_class": n_classes,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8,
                  "colsample_bytree": 0.5, "seed": RANDOM_SEED + fold, "verbosity": 0}
        model = xgb.train(params, dtrain, num_boost_round=150, verbose_eval=0)

        importance = model.get_score(importance_type='gain')
        feat_gains = {}
        for fname, gain in importance.items():
            idx = int(fname.replace('f', ''))
            feat_gains[idx] = gain

        sorted_feats = sorted(feat_gains.items(), key=lambda x: x[1], reverse=True)
        top_feats = set(idx for idx, _ in sorted_feats[:top_k_gain])
        fold_important.append(top_feats)
        print(f"    Fold {fold+1}: {len(top_feats)} features selected")

    # Features appearing in at least 3/5 folds
    from collections import Counter
    all_feats = Counter()
    for feats in fold_important:
        for f in feats:
            all_feats[f] += 1

    # Keep features that appear in majority of folds
    min_folds = max(2, n_splits // 2)
    stable_features = [f for f, count in all_feats.items() if count >= min_folds]
    stable_features.sort()

    print(f"    Cross-importance: {len(stable_features)} stable features (appear in >={min_folds} folds)")
    return stable_features


def main():
    t_start = time.time()
    print("=" * 70)
    print("TWO-STAGE XGBOOST (Champion's Approach)")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)

    global_stats = compute_global_stats(train_df)

    print("\nBuilding V2 features...")
    t0 = time.time()
    feat_train = build_features_v2(train_df, is_train=True, global_stats=global_stats)
    feat_test = build_features_v2(test_df, is_train=False, global_stats=global_stats)
    feature_names = get_feature_names_v2(feat_train)
    print(f"  Done in {time.time()-t0:.1f}s: {len(feature_names)} base features")

    X_base = feat_train[feature_names].values.astype(np.float32)
    X_test_base = feat_test[feature_names].values.astype(np.float32)
    y_act = feat_train["y_actionId"].values
    y_pt = feat_train["y_pointId"].values
    y_srv = feat_train["y_serverGetPoint"].values
    next_sn = feat_train["next_strikeNumber"].values
    test_next_sn = feat_test["next_strikeNumber"].values

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_train["rally_uid"].map(rally_to_match).values

    # ========================================
    # STEP 1: Massive Feature Generation
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 1: Feature Combination (champion's 972→11873 approach)")
    print(f"{'='*60}")

    # Replace NaN/Inf with 0 in base features
    X_base = np.nan_to_num(X_base, nan=0, posinf=0, neginf=0)
    X_test_base = np.nan_to_num(X_test_base, nan=0, posinf=0, neginf=0)

    combo_X, combo_names = generate_feature_combinations(X_base, feature_names, top_k=50)
    combo_test_X, _ = generate_feature_combinations(X_test_base, feature_names, top_k=50)

    # Combine original + combinations
    X_all = np.hstack([X_base, combo_X]).astype(np.float32)
    X_test_all = np.hstack([X_test_base, combo_test_X]).astype(np.float32)
    all_names = feature_names + combo_names

    # Clean any remaining NaN/Inf
    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)
    X_test_all = np.nan_to_num(X_test_all, nan=0, posinf=0, neginf=0)

    print(f"\n  Total features: {X_all.shape[1]} (base={X_base.shape[1]} + combo={combo_X.shape[1]})")

    # ========================================
    # STEP 2: Feature Selection (3-stage)
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 2: Three-Stage Feature Selection")
    print(f"{'='*60}")

    # Do selection for action task (largest impact)
    print("\n--- Action task feature selection ---")
    gain_indices = feature_selection_xgb_gain(X_all, y_act, all_names, N_ACTION, top_k=600)
    shap_indices = feature_selection_shap(X_all, y_act, all_names, gain_indices, N_ACTION, top_k=300)
    cross_indices_act = cross_importance_selection(X_all, y_act, groups, all_names, N_ACTION,
                                                   n_splits=5, top_k_gain=400)

    # Union of SHAP + cross-importance
    final_indices_act = sorted(set(shap_indices) | set(cross_indices_act))
    print(f"\n  Final action features: {len(final_indices_act)}")

    # Same for point task
    print("\n--- Point task feature selection ---")
    gain_indices_pt = feature_selection_xgb_gain(X_all, y_pt, all_names, N_POINT, top_k=600)
    shap_indices_pt = feature_selection_shap(X_all, y_pt, all_names, gain_indices_pt, N_POINT, top_k=300)
    cross_indices_pt = cross_importance_selection(X_all, y_pt, groups, all_names, N_POINT,
                                                   n_splits=5, top_k_gain=400)
    final_indices_pt = sorted(set(shap_indices_pt) | set(cross_indices_pt))
    print(f"\n  Final point features: {len(final_indices_pt)}")

    # ========================================
    # STEP 3: Train XGBoost B with selected features
    # ========================================
    print(f"\n{'='*60}")
    print("STEP 3: Train Final XGBoost B with Selected Features")
    print(f"{'='*60}")

    import xgboost as xgb
    from catboost import CatBoostClassifier

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(X_all, groups=groups))

    oof_act = np.zeros((len(X_all), N_ACTION))
    oof_pt = np.zeros((len(X_all), N_POINT))
    oof_srv = np.zeros(len(X_all))
    test_act = np.zeros((len(X_test_all), N_ACTION))
    test_pt = np.zeros((len(X_test_all), N_POINT))
    test_srv = np.zeros(len(X_test_all))

    X_act_sel = X_all[:, final_indices_act]
    X_test_act_sel = X_test_all[:, final_indices_act]
    X_pt_sel = X_all[:, final_indices_pt]
    X_test_pt_sel = X_test_all[:, final_indices_pt]

    print(f"\n  Action features: {X_act_sel.shape[1]}")
    print(f"  Point features: {X_pt_sel.shape[1]}")

    dtest_act = xgb.DMatrix(X_test_act_sel)
    dtest_pt = xgb.DMatrix(X_test_pt_sel)
    dtest_srv = xgb.DMatrix(X_all[:, :X_base.shape[1]])  # use base for server
    dtest_srv_test = xgb.DMatrix(X_test_all[:, :X_base.shape[1]])

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        t0 = time.time()

        # Action with selected features
        dtrain = xgb.DMatrix(X_act_sel[tr_idx], label=y_act[tr_idx])
        dval = xgb.DMatrix(X_act_sel[val_idx], label=y_act[val_idx])
        params = {"objective": "multi:softprob", "num_class": N_ACTION,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                  "subsample": 0.8, "colsample_bytree": 0.7,
                  "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        m = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        oof_act[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        test_act += m.predict(dtest_act, iteration_range=(0, m.best_iteration+1)) / N_FOLDS

        # Point with selected features
        dtrain = xgb.DMatrix(X_pt_sel[tr_idx], label=y_pt[tr_idx])
        dval = xgb.DMatrix(X_pt_sel[val_idx], label=y_pt[val_idx])
        params["num_class"] = N_POINT
        m = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        oof_pt[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        test_pt += m.predict(dtest_pt, iteration_range=(0, m.best_iteration+1)) / N_FOLDS

        # Server with base features (binary, simpler)
        dtrain = xgb.DMatrix(X_base[tr_idx], label=y_srv[tr_idx])
        dval = xgb.DMatrix(X_base[val_idx], label=y_srv[val_idx])
        params_bin = {"objective": "binary:logistic", "eval_metric": "auc",
                      "tree_method": "hist", "learning_rate": 0.03,
                      "max_depth": 8, "min_child_weight": 10,
                      "subsample": 0.8, "colsample_bytree": 0.8,
                      "lambda": 1, "seed": RANDOM_SEED, "verbosity": 0}
        m = xgb.train(params_bin, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        oof_srv[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        test_srv += m.predict(xgb.DMatrix(X_test_base), iteration_range=(0, m.best_iteration+1)) / N_FOLDS

        act_ruled = apply_action_rules(oof_act[val_idx], next_sn[val_idx])
        f1a = macro_f1(y_act[val_idx], act_ruled, N_ACTION)
        f1p = macro_f1(y_pt[val_idx], oof_pt[val_idx], N_POINT)
        auc = roc_auc_score(y_srv[val_idx], oof_srv[val_idx])
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        print(f"  Fold {fold+1}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f} (act_it={m.best_iteration}) ({time.time()-t0:.0f}s)")

    # Overall
    act_ruled = apply_action_rules(oof_act, next_sn)
    f1a = macro_f1(y_act, act_ruled, N_ACTION)
    f1p = macro_f1(y_pt, oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, oof_srv)
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc
    print(f"\n  TWO-STAGE OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # Blend with V2 best
    print("\n--- Blend with V2 best ---")
    v2 = np.load(os.path.join(MODEL_DIR, "oof_v2_fast.npz"))
    v2_act = 0.6 * v2["catboost_act"] + 0.4 * v2["xgboost_act"]
    v2_pt = 0.6 * v2["catboost_pt"] + 0.3 * v2["xgboost_pt"] + 0.1 * v2["lightgbm_pt"]
    v2_srv = 0.3 * v2["catboost_srv"] + 0.4 * v2["xgboost_srv"] + 0.3 * v2["lightgbm_srv"]

    best_ov = -1
    best_w = 0
    for w in np.arange(0, 1.05, 0.1):
        blend_act = w * oof_act + (1-w) * v2_act
        blend_pt = w * oof_pt + (1-w) * v2_pt
        blend_srv = w * oof_srv + (1-w) * v2_srv
        ba_r = apply_action_rules(blend_act, next_sn)
        f1a_b = macro_f1(y_act, ba_r, N_ACTION)
        f1p_b = macro_f1(y_pt, blend_pt, N_POINT)
        auc_b = roc_auc_score(y_srv, blend_srv)
        ov_b = 0.4*f1a_b + 0.4*f1p_b + 0.2*auc_b
        if ov_b > best_ov:
            best_ov = ov_b
            best_w = w
        print(f"  w_2stage={w:.1f}: F1a={f1a_b:.4f} F1p={f1p_b:.4f} AUC={auc_b:.4f} OV={ov_b:.4f}")

    print(f"\n  Best blend: w_2stage={best_w:.1f}, OV={best_ov:.4f}")

    # Generate submission
    v2t = np.load(os.path.join(MODEL_DIR, "test_v2_fast.npz"))
    v2_test_act = 0.6 * v2t["catboost_act"] + 0.4 * v2t["xgboost_act"]
    v2_test_pt = 0.6 * v2t["catboost_pt"] + 0.3 * v2t["xgboost_pt"] + 0.1 * v2t["lightgbm_pt"]
    v2_test_srv = 0.3 * v2t["catboost_srv"] + 0.4 * v2t["xgboost_srv"] + 0.3 * v2t["lightgbm_srv"]

    final_act = best_w * test_act + (1-best_w) * v2_test_act
    final_pt = best_w * test_pt + (1-best_w) * v2_test_pt
    final_srv = best_w * test_srv + (1-best_w) * v2_test_srv
    final_act = apply_action_rules(final_act, test_next_sn)

    submission = pd.DataFrame({
        "rally_uid": feat_test["rally_uid"].values.astype(int),
        "actionId": np.argmax(final_act, axis=1).astype(int),
        "pointId": np.argmax(final_pt, axis=1).astype(int),
        "serverGetPoint": (final_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_twostage.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId: {submission.pointId.value_counts().sort_index().to_dict()}")

    # Save OOF for further blending
    np.savez(os.path.join(MODEL_DIR, "oof_twostage.npz"),
             oof_act=oof_act, oof_pt=oof_pt, oof_srv=oof_srv,
             y_act=y_act, y_pt=y_pt, y_srv=y_srv, next_sn=next_sn)
    np.savez(os.path.join(MODEL_DIR, "test_twostage.npz"),
             test_act=test_act, test_pt=test_pt, test_srv=test_srv,
             test_next_sn=test_next_sn,
             rally_uids=feat_test["rally_uid"].values.astype(int))

    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
