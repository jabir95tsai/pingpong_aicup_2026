"""V7 Enhanced Pipeline: V5 features + fold-safe GroupKFold + 3-model blend + threshold opt.

Key improvements over V5/V6:
1. Uses features_v5 (931 features with EDA-driven enhancements)
2. Fold-safe CV with GroupKFold(by match) — same as V5
3. Integrated threshold optimization (was separate in V6)
4. All 3 models (CB+XGB+LGB) for both CV and final
5. Phase/score/transition/Markov features from EDA findings
"""
import sys, os, time, warnings, gc, argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data

N_ACTION, N_POINT = 19, 10
SERVE_FORBIDDEN = {15, 16, 17, 18}


# ======================= UTILITIES ==========================================

def macro_f1(y_true, y_probs, n_classes):
    y_pred = np.argmax(y_probs, axis=1)
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def apply_action_rules(probs, next_sns):
    preds = probs.copy()
    for i in range(len(preds)):
        if next_sns[i] == 2:
            for a in SERVE_FORBIDDEN:
                if a < preds.shape[1]:
                    preds[i, a] = 0.0
        total = preds[i].sum()
        if total > 0:
            preds[i] /= total
        else:
            preds[i] = np.ones(preds.shape[1]) / preds.shape[1]
    return preds


def feature_selection_gain(X_tr, y_tr, n_classes, top_k=600, task="multi"):
    """Feature selection using XGBoost gain on training data only."""
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
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
        if idx < X_tr.shape[1]:
            feat_gains[idx] = gain
    sorted_feats = sorted(feat_gains.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in sorted_feats[:top_k]]


def optimize_class_weights_greedy(probs, y_true, n_classes):
    """Greedy per-class weight optimization for Macro F1."""
    weights = np.ones(n_classes)
    for _ in range(3):
        improved = False
        for cls in range(n_classes):
            best_w = weights[cls]
            adj = probs * weights[np.newaxis, :]
            adj /= adj.sum(axis=1, keepdims=True)
            best_score = f1_score(y_true, np.argmax(adj, axis=1),
                                  labels=list(range(n_classes)), average="macro", zero_division=0)
            for w in np.arange(0.3, 10.0, 0.2):
                test_w = weights.copy()
                test_w[cls] = w
                adj = probs * test_w[np.newaxis, :]
                adj /= adj.sum(axis=1, keepdims=True)
                score = f1_score(y_true, np.argmax(adj, axis=1),
                                 labels=list(range(n_classes)), average="macro", zero_division=0)
                if score > best_score + 1e-6:
                    best_score = score
                    best_w = w
                    improved = True
            weights[cls] = best_w
        if not improved:
            break
    adj = probs * weights[np.newaxis, :]
    adj /= adj.sum(axis=1, keepdims=True)
    final_f1 = f1_score(y_true, np.argmax(adj, axis=1),
                        labels=list(range(n_classes)), average="macro", zero_division=0)
    return weights, final_f1


def optimize_threshold(probs, y_true, n_classes, next_sn=None, is_action=False):
    """Temperature scaling + class weight optimization."""
    working_probs = probs.copy()
    if is_action and next_sn is not None:
        working_probs = apply_action_rules(working_probs, next_sn)

    # Temperature
    best_t, best_score = 1.0, -1
    for t in np.arange(0.3, 3.0, 0.05):
        scaled = working_probs ** (1.0 / t)
        row_sums = scaled.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        scaled /= row_sums
        score = f1_score(y_true, np.argmax(scaled, axis=1),
                         labels=list(range(n_classes)), average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t

    scaled_probs = working_probs ** (1.0 / best_t)
    row_sums = scaled_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    scaled_probs /= row_sums

    base_f1 = f1_score(y_true, np.argmax(working_probs, axis=1),
                       labels=list(range(n_classes)), average="macro", zero_division=0)
    print(f"    Temp: {best_t:.2f} (F1: {base_f1:.4f} -> {best_score:.4f})")

    weights, f1_opt = optimize_class_weights_greedy(scaled_probs, y_true, n_classes)
    print(f"    Greedy weights F1: {f1_opt:.4f}")
    return best_t, weights, f1_opt


# ======================= MAIN ===============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (1 fold, fewer trees)")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="Number of CV folds")
    args = parser.parse_args()

    is_smoke = args.smoke
    n_folds = 1 if is_smoke else args.folds
    n_boost = 100 if is_smoke else 2000
    es_rounds = 30 if is_smoke else 150
    top_k_act = 100 if is_smoke else 600
    top_k_pt = 100 if is_smoke else 600
    top_k_srv = 80 if is_smoke else 400

    t_start = time.time()
    print("=" * 70)
    print(f"V7 ENHANCED PIPELINE {'(SMOKE TEST)' if is_smoke else ''}")
    print(f"  - V5 features (931 dims with EDA enhancements)")
    print(f"  - GroupKFold(match) {n_folds}-fold CV")
    print(f"  - 3-model blend (CB+XGB+LGB)")
    print(f"  - Integrated threshold optimization")
    print("=" * 70)

    # --- Load & clean data ---
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)

    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    from features_v5 import compute_global_stats_v5, build_features_v5, get_feature_names_v5

    # --- Preflight: build full features for sample indexing ---
    print("\n--- Preflight: feature dimensions ---")
    t0 = time.time()
    gs_full = compute_global_stats_v5(train_df)
    feat_full = build_features_v5(train_df, is_train=True, global_stats_v5=gs_full)
    feature_names_full = get_feature_names_v5(feat_full)
    n_samples = len(feat_full)
    print(f"  {len(feature_names_full)} features, {n_samples} samples ({time.time()-t0:.1f}s)")

    y_act_all = feat_full["y_actionId"].values
    y_pt_all = feat_full["y_pointId"].values
    y_srv_all = feat_full["y_serverGetPoint"].values
    next_sn_all = feat_full["next_strikeNumber"].values
    sample_rally_uids = feat_full["rally_uid"].values

    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    sample_to_match = np.array([rally_to_match.get(r, -1) for r in sample_rally_uids])

    # --- GroupKFold at sample level ---
    sample_gkf = GroupKFold(n_splits=max(n_folds, 2))  # min 2 for GroupKFold
    all_splits = list(sample_gkf.split(np.arange(n_samples), groups=sample_to_match))
    if is_smoke:
        all_splits = all_splits[:1]  # Just 1 fold for smoke test

    # OOF containers
    oof_act = {m: np.zeros((n_samples, N_ACTION)) for m in ["CB", "XGB", "LGB"]}
    oof_pt = {m: np.zeros((n_samples, N_POINT)) for m in ["CB", "XGB", "LGB"]}
    oof_srv = {m: np.zeros(n_samples) for m in ["CB", "XGB", "LGB"]}

    # ========================================
    # CV LOOP
    # ========================================
    for fold, (tr_idx, val_idx) in enumerate(all_splits):
        t_fold = time.time()
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{len(all_splits)} (train={len(tr_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        tr_rallies = set(sample_rally_uids[tr_idx])
        val_rallies = set(sample_rally_uids[val_idx])
        tr_raw = train_df[train_df["rally_uid"].isin(tr_rallies)]
        val_raw = train_df[train_df["rally_uid"].isin(val_rallies)]

        # FOLD-SAFE stats
        print("  Computing fold-safe stats...")
        t0 = time.time()
        fold_stats = compute_global_stats_v5(tr_raw)
        print(f"    Done ({time.time()-t0:.1f}s)")

        # Build features
        print("  Building features...")
        t0 = time.time()
        feat_tr = build_features_v5(tr_raw, is_train=True, global_stats_v5=fold_stats)
        feat_val = build_features_v5(val_raw, is_train=True, global_stats_v5=fold_stats)
        fnames = get_feature_names_v5(feat_tr)
        print(f"    {len(fnames)} features, train={len(feat_tr)}, val={len(feat_val)} ({time.time()-t0:.1f}s)")

        X_tr = np.nan_to_num(feat_tr[fnames].values.astype(np.float32), nan=0, posinf=0, neginf=0)
        X_val = np.nan_to_num(feat_val[fnames].values.astype(np.float32), nan=0, posinf=0, neginf=0)
        ya_tr, ya_val = feat_tr["y_actionId"].values, feat_val["y_actionId"].values
        yp_tr, yp_val = feat_tr["y_pointId"].values, feat_val["y_pointId"].values
        ys_tr, ys_val = feat_tr["y_serverGetPoint"].values, feat_val["y_serverGetPoint"].values
        sn_val = feat_val["next_strikeNumber"].values

        # Feature selection (fold-safe)
        print("  Feature selection...")
        t0 = time.time()
        sel_act = feature_selection_gain(X_tr, ya_tr, N_ACTION, top_k=top_k_act)
        sel_pt = feature_selection_gain(X_tr, yp_tr, N_POINT, top_k=top_k_pt)
        sel_srv = feature_selection_gain(X_tr, ys_tr, 2, top_k=top_k_srv, task="binary")
        print(f"    Act={len(sel_act)}, Pt={len(sel_pt)}, Srv={len(sel_srv)} ({time.time()-t0:.1f}s)")

        Xa_tr, Xa_val = X_tr[:, sel_act], X_val[:, sel_act]
        Xp_tr, Xp_val = X_tr[:, sel_pt], X_val[:, sel_pt]
        Xs_tr, Xs_val = X_tr[:, sel_srv], X_val[:, sel_srv]

        # --- CatBoost ---
        print("  Training CatBoost...")
        t0 = time.time()
        m = CatBoostClassifier(iterations=n_boost, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_ACTION,
                               auto_class_weights="Balanced", early_stopping_rounds=es_rounds,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                               bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m.fit(Xa_tr, ya_tr, eval_set=(Xa_val, ya_val))
        oof_act["CB"][val_idx] = m.predict_proba(Xa_val)

        m = CatBoostClassifier(iterations=n_boost, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_POINT,
                               auto_class_weights="Balanced", early_stopping_rounds=es_rounds,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                               bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m.fit(Xp_tr, yp_tr, eval_set=(Xp_val, yp_val))
        oof_pt["CB"][val_idx] = m.predict_proba(Xp_val)

        m = CatBoostClassifier(iterations=n_boost, learning_rate=0.03, depth=8,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               early_stopping_rounds=es_rounds, verbose=0,
                               random_seed=RANDOM_SEED, l2_leaf_reg=3)
        m.fit(Xs_tr, ys_tr, eval_set=(Xs_val, ys_val))
        oof_srv["CB"][val_idx] = m.predict_proba(Xs_val)[:, 1]
        print(f"    CB done ({time.time()-t0:.0f}s)")

        # --- XGBoost ---
        print("  Training XGBoost...")
        t0 = time.time()
        dtrain = xgb.DMatrix(Xa_tr, label=ya_tr)
        dval = xgb.DMatrix(Xa_val, label=ya_val)
        params_xgb = {"objective": "multi:softprob", "num_class": N_ACTION,
                      "eval_metric": "mlogloss", "tree_method": "hist",
                      "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                      "subsample": 0.8, "colsample_bytree": 0.7,
                      "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        mx = xgb.train(params_xgb, dtrain, num_boost_round=n_boost, evals=[(dval, "val")],
                       early_stopping_rounds=es_rounds, verbose_eval=0)
        oof_act["XGB"][val_idx] = mx.predict(dval, iteration_range=(0, mx.best_iteration+1))

        dtrain = xgb.DMatrix(Xp_tr, label=yp_tr)
        dval = xgb.DMatrix(Xp_val, label=yp_val)
        params_xgb["num_class"] = N_POINT
        mx = xgb.train(params_xgb, dtrain, num_boost_round=n_boost, evals=[(dval, "val")],
                       early_stopping_rounds=es_rounds, verbose_eval=0)
        oof_pt["XGB"][val_idx] = mx.predict(dval, iteration_range=(0, mx.best_iteration+1))

        dtrain = xgb.DMatrix(Xs_tr, label=ys_tr)
        dval = xgb.DMatrix(Xs_val, label=ys_val)
        params_bin = {"objective": "binary:logistic", "eval_metric": "auc",
                      "tree_method": "hist", "learning_rate": 0.03, "max_depth": 8,
                      "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
                      "lambda": 1, "seed": RANDOM_SEED, "verbosity": 0}
        mx = xgb.train(params_bin, dtrain, num_boost_round=n_boost, evals=[(dval, "val")],
                       early_stopping_rounds=es_rounds, verbose_eval=0)
        oof_srv["XGB"][val_idx] = mx.predict(dval, iteration_range=(0, mx.best_iteration+1))
        print(f"    XGB done ({time.time()-t0:.0f}s)")

        # --- LightGBM ---
        print("  Training LightGBM...")
        t0 = time.time()
        dtrain_l = lgb.Dataset(Xa_tr, label=ya_tr)
        dval_l = lgb.Dataset(Xa_val, label=ya_val, reference=dtrain_l)
        params_lgb = {"objective": "multiclass", "num_class": N_ACTION,
                      "metric": "multi_logloss", "learning_rate": 0.03,
                      "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                      "subsample": 0.8, "colsample_bytree": 0.7, "is_unbalance": True,
                      "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        ml = lgb.train(params_lgb, dtrain_l, num_boost_round=n_boost, valid_sets=[dval_l],
                       callbacks=[lgb.early_stopping(es_rounds), lgb.log_evaluation(0)])
        oof_act["LGB"][val_idx] = ml.predict(Xa_val)

        dtrain_l = lgb.Dataset(Xp_tr, label=yp_tr)
        dval_l = lgb.Dataset(Xp_val, label=yp_val, reference=dtrain_l)
        params_lgb["num_class"] = N_POINT
        ml = lgb.train(params_lgb, dtrain_l, num_boost_round=n_boost, valid_sets=[dval_l],
                       callbacks=[lgb.early_stopping(es_rounds), lgb.log_evaluation(0)])
        oof_pt["LGB"][val_idx] = ml.predict(Xp_val)

        dtrain_l = lgb.Dataset(Xs_tr, label=ys_tr)
        dval_l = lgb.Dataset(Xs_val, label=ys_val, reference=dtrain_l)
        params_lgb_bin = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
                          "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                          "subsample": 0.8, "colsample_bytree": 0.8, "is_unbalance": True,
                          "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        ml = lgb.train(params_lgb_bin, dtrain_l, num_boost_round=n_boost, valid_sets=[dval_l],
                       callbacks=[lgb.early_stopping(es_rounds), lgb.log_evaluation(0)])
        oof_srv["LGB"][val_idx] = ml.predict(Xs_val)
        print(f"    LGB done ({time.time()-t0:.0f}s)")

        # Fold summary
        for name in ["CB", "XGB", "LGB"]:
            ar = apply_action_rules(oof_act[name][val_idx], sn_val)
            f1a = macro_f1(ya_val, ar, N_ACTION)
            f1p = macro_f1(yp_val, oof_pt[name][val_idx], N_POINT)
            try:
                auc = roc_auc_score(ys_val, oof_srv[name][val_idx])
            except ValueError:
                auc = 0.5
            ov = 0.4*f1a + 0.4*f1p + 0.2*auc
            print(f"    {name}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

        print(f"  Fold {fold+1}: {(time.time()-t_fold)/60:.1f} min")
        gc.collect()

    # ========================================
    # OVERALL OOF EVALUATION
    # ========================================
    if is_smoke:
        # In smoke mode, only evaluate on the single fold's val set
        val_mask = all_splits[0][1]
        eval_mask = val_mask
    else:
        eval_mask = np.arange(n_samples)

    print(f"\n{'='*60}")
    print(f"OOF RESULTS ({len(eval_mask)} samples)")
    print(f"{'='*60}")

    for name in ["CB", "XGB", "LGB"]:
        ar = apply_action_rules(oof_act[name][eval_mask], next_sn_all[eval_mask])
        f1a = macro_f1(y_act_all[eval_mask], ar, N_ACTION)
        f1p = macro_f1(y_pt_all[eval_mask], oof_pt[name][eval_mask], N_POINT)
        try:
            auc = roc_auc_score(y_srv_all[eval_mask], oof_srv[name][eval_mask])
        except ValueError:
            auc = 0.5
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        print(f"  {name}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # Blend search
    print("\n--- Blend Search (step=0.05) ---")
    best_ov = -1
    best_blend = (0.5, 0.25, 0.25)
    for w_cb in np.arange(0.3, 0.85, 0.05):
        for w_xg in np.arange(0.0, 0.55, 0.05):
            w_lg = round(1.0 - w_cb - w_xg, 2)
            if w_lg < 0 or w_lg > 0.5:
                continue
            ba = w_cb * oof_act["CB"][eval_mask] + w_xg * oof_act["XGB"][eval_mask] + w_lg * oof_act["LGB"][eval_mask]
            bp = w_cb * oof_pt["CB"][eval_mask] + w_xg * oof_pt["XGB"][eval_mask] + w_lg * oof_pt["LGB"][eval_mask]
            bs = w_cb * oof_srv["CB"][eval_mask] + w_xg * oof_srv["XGB"][eval_mask] + w_lg * oof_srv["LGB"][eval_mask]

            bar = apply_action_rules(ba, next_sn_all[eval_mask])
            f1a = macro_f1(y_act_all[eval_mask], bar, N_ACTION)
            f1p = macro_f1(y_pt_all[eval_mask], bp, N_POINT)
            try:
                auc = roc_auc_score(y_srv_all[eval_mask], bs)
            except ValueError:
                auc = 0.5
            ov = 0.4*f1a + 0.4*f1p + 0.2*auc
            if ov > best_ov:
                best_ov = ov
                best_blend = (w_cb, w_xg, w_lg)

    w_cb, w_xg, w_lg = best_blend
    print(f"  Best: CB={w_cb:.2f} XGB={w_xg:.2f} LGB={w_lg:.2f} OV={best_ov:.4f}")

    # Threshold optimization on blended OOF
    print("\n--- Threshold Optimization ---")
    blend_act = w_cb * oof_act["CB"][eval_mask] + w_xg * oof_act["XGB"][eval_mask] + w_lg * oof_act["LGB"][eval_mask]
    blend_pt = w_cb * oof_pt["CB"][eval_mask] + w_xg * oof_pt["XGB"][eval_mask] + w_lg * oof_pt["LGB"][eval_mask]
    blend_srv = w_cb * oof_srv["CB"][eval_mask] + w_xg * oof_srv["XGB"][eval_mask] + w_lg * oof_srv["LGB"][eval_mask]

    print("  actionId:")
    temp_act, weights_act, f1a_opt = optimize_threshold(
        blend_act, y_act_all[eval_mask], N_ACTION,
        next_sn=next_sn_all[eval_mask], is_action=True)

    print("  pointId:")
    temp_pt, weights_pt, f1p_opt = optimize_threshold(
        blend_pt, y_pt_all[eval_mask], N_POINT)

    try:
        auc_base = roc_auc_score(y_srv_all[eval_mask], blend_srv)
    except ValueError:
        auc_base = 0.5
    ov_opt = 0.4 * f1a_opt + 0.4 * f1p_opt + 0.2 * auc_base
    print(f"\n  Optimized OV: {ov_opt:.4f} (blend={best_ov:.4f}, threshold gain={ov_opt-best_ov:.4f})")

    # Save OOF
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(os.path.join(MODEL_DIR, "oof_v7.npz"),
             **{f"{m}_act": oof_act[m] for m in ["CB", "XGB", "LGB"]},
             **{f"{m}_pt": oof_pt[m] for m in ["CB", "XGB", "LGB"]},
             **{f"{m}_srv": oof_srv[m] for m in ["CB", "XGB", "LGB"]},
             y_act=y_act_all, y_pt=y_pt_all, y_srv=y_srv_all, next_sn=next_sn_all)

    # ========================================
    # FINAL SUBMISSION
    # ========================================
    print(f"\n{'='*60}")
    print("FINAL: Full-data train -> test predict")
    print(f"{'='*60}")

    t0 = time.time()
    full_stats = compute_global_stats_v5(train_df)
    feat_train_full = build_features_v5(train_df, is_train=True, global_stats_v5=full_stats)
    feat_test_full = build_features_v5(test_df, is_train=False, global_stats_v5=full_stats)
    fnames = get_feature_names_v5(feat_train_full)
    print(f"  Features: {len(fnames)} ({time.time()-t0:.1f}s)")

    X_full = np.nan_to_num(feat_train_full[fnames].values.astype(np.float32), nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(feat_test_full[fnames].values.astype(np.float32), nan=0, posinf=0, neginf=0)
    ya_full = feat_train_full["y_actionId"].values
    yp_full = feat_train_full["y_pointId"].values
    ys_full = feat_train_full["y_serverGetPoint"].values
    test_sn = feat_test_full["next_strikeNumber"].values

    # Feature selection on full data
    sel_act = feature_selection_gain(X_full, ya_full, N_ACTION, top_k=top_k_act)
    sel_pt = feature_selection_gain(X_full, yp_full, N_POINT, top_k=top_k_pt)
    sel_srv = feature_selection_gain(X_full, ys_full, 2, top_k=top_k_srv, task="binary")
    print(f"  Selected: Act={len(sel_act)}, Pt={len(sel_pt)}, Srv={len(sel_srv)}")

    test_act = np.zeros((len(X_test), N_ACTION))
    test_pt = np.zeros((len(X_test), N_POINT))
    test_srv = np.zeros(len(X_test))

    n_final_boost = 100 if is_smoke else 1500

    for task_name, X_sel_tr, y_sel, X_sel_te, n_cls in [
        ("actionId", X_full[:, sel_act], ya_full, X_test[:, sel_act], N_ACTION),
        ("pointId", X_full[:, sel_pt], yp_full, X_test[:, sel_pt], N_POINT),
    ]:
        print(f"\n  Training {task_name}...")
        t0 = time.time()

        # CatBoost
        m_cb = CatBoostClassifier(iterations=n_final_boost, learning_rate=0.03, depth=8,
                                  loss_function="MultiClass", classes_count=n_cls,
                                  auto_class_weights="Balanced", verbose=0,
                                  random_seed=RANDOM_SEED, l2_leaf_reg=3,
                                  bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m_cb.fit(X_sel_tr, y_sel)
        cb_pred = m_cb.predict_proba(X_sel_te)

        # XGBoost
        dtrain = xgb.DMatrix(X_sel_tr, label=y_sel)
        params = {"objective": "multi:softprob", "num_class": n_cls,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                  "subsample": 0.8, "colsample_bytree": 0.7,
                  "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        mx = xgb.train(params, dtrain, num_boost_round=n_final_boost, verbose_eval=0)
        xgb_pred = mx.predict(xgb.DMatrix(X_sel_te))

        # LightGBM
        dtrain_l = lgb.Dataset(X_sel_tr, label=y_sel)
        params_l = {"objective": "multiclass", "num_class": n_cls,
                    "metric": "multi_logloss", "learning_rate": 0.03,
                    "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                    "subsample": 0.8, "colsample_bytree": 0.7, "is_unbalance": True,
                    "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        ml = lgb.train(params_l, dtrain_l, num_boost_round=n_final_boost)
        lgb_pred = ml.predict(X_sel_te)

        blended = w_cb * cb_pred + w_xg * xgb_pred + w_lg * lgb_pred
        if task_name == "actionId":
            test_act[:] = blended
        else:
            test_pt[:] = blended
        print(f"    Done ({time.time()-t0:.0f}s)")

    # Server
    print(f"\n  Training serverGetPoint...")
    t0 = time.time()
    m_cb = CatBoostClassifier(iterations=n_final_boost, learning_rate=0.03, depth=8,
                              loss_function="Logloss", auto_class_weights="Balanced",
                              verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
    m_cb.fit(X_full[:, sel_srv], ys_full)
    cb_srv = m_cb.predict_proba(X_test[:, sel_srv])[:, 1]

    dtrain = xgb.DMatrix(X_full[:, sel_srv], label=ys_full)
    params_bin = {"objective": "binary:logistic", "eval_metric": "auc",
                  "tree_method": "hist", "learning_rate": 0.03, "max_depth": 8,
                  "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
                  "lambda": 1, "seed": RANDOM_SEED, "verbosity": 0}
    mx = xgb.train(params_bin, dtrain, num_boost_round=n_final_boost, verbose_eval=0)
    xgb_srv = mx.predict(xgb.DMatrix(X_test[:, sel_srv]))

    dtrain_l = lgb.Dataset(X_full[:, sel_srv], label=ys_full)
    params_l_bin = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
                    "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                    "subsample": 0.8, "colsample_bytree": 0.8, "is_unbalance": True,
                    "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
    ml = lgb.train(params_l_bin, dtrain_l, num_boost_round=n_final_boost)
    lgb_srv = ml.predict(X_test[:, sel_srv])

    test_srv[:] = w_cb * cb_srv + w_xg * xgb_srv + w_lg * lgb_srv
    print(f"    Done ({time.time()-t0:.0f}s)")

    # ========================================
    # APPLY THRESHOLDS & GENERATE SUBMISSION
    # ========================================
    print(f"\n{'='*60}")
    print("GENERATING SUBMISSION")
    print(f"{'='*60}")

    # Action rules
    test_act_ruled = apply_action_rules(test_act, test_sn)

    # Temperature scaling
    test_act_scaled = test_act_ruled ** (1.0 / temp_act)
    row_sums = test_act_scaled.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    test_act_scaled /= row_sums

    test_pt_scaled = test_pt ** (1.0 / temp_pt)
    row_sums = test_pt_scaled.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    test_pt_scaled /= row_sums

    # Class weights
    test_act_adj = test_act_scaled * weights_act[np.newaxis, :]
    test_act_adj /= test_act_adj.sum(axis=1, keepdims=True)

    test_pt_adj = test_pt_scaled * weights_pt[np.newaxis, :]
    test_pt_adj /= test_pt_adj.sum(axis=1, keepdims=True)

    pred_act = np.argmax(test_act_adj, axis=1).astype(int)
    pred_pt = np.argmax(test_pt_adj, axis=1).astype(int)
    pred_srv = (test_srv >= 0.5).astype(int)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission = pd.DataFrame({
        "rally_uid": feat_test_full["rally_uid"].values.astype(int),
        "actionId": pred_act,
        "pointId": pred_pt,
        "serverGetPoint": pred_srv,
    })
    suffix = "_smoke" if is_smoke else ""
    out = os.path.join(SUBMISSION_DIR, f"submission_v7{suffix}.csv")
    submission.to_csv(out, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\n  Saved: {out}")
    print(f"  actionId dist: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId dist:  {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  SGP dist:      {submission.serverGetPoint.value_counts().to_dict()}")

    # Also save base submission (no threshold)
    pred_act_base = np.argmax(test_act_ruled, axis=1).astype(int)
    pred_pt_base = np.argmax(test_pt, axis=1).astype(int)
    submission_base = pd.DataFrame({
        "rally_uid": feat_test_full["rally_uid"].values.astype(int),
        "actionId": pred_act_base,
        "pointId": pred_pt_base,
        "serverGetPoint": pred_srv,
    })
    out_base = os.path.join(SUBMISSION_DIR, f"submission_v7_base{suffix}.csv")
    submission_base.to_csv(out_base, index=False, lineterminator="\n", encoding="utf-8")
    print(f"  Saved base: {out_base}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Features: {len(fnames)} (V5 with EDA enhancements)")
    print(f"  Blend: CB={w_cb:.2f} XGB={w_xg:.2f} LGB={w_lg:.2f}")
    print(f"  OOF blend OV: {best_ov:.4f}")
    print(f"  OOF optimized OV: {ov_opt:.4f}")
    print(f"  Threshold gain: +{ov_opt-best_ov:.4f}")
    print(f"  Total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
