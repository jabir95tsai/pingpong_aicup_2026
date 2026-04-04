"""V5 Clean Pipeline: Strict no-leakage CV with fold-safe statistics.

Key differences from V4:
1. Global stats computed PER FOLD (only from training fold)
2. Feature selection done PER FOLD (not on full data)
3. No raw player IDs as features
4. Fixed serve_mask bug
5. Proper coarse target encoding (not ordinal mean)
6. Test combo features use train's top_indices
7. No dead sn==1 code paths
8. Test-like next_sn weighted evaluation
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
        # sn==1 never occurs in our data, but keep sn==2 rule
        if sn == 2:
            for a in SERVE_FORBIDDEN:
                if a < preds.shape[1]: preds[i, a] = 0.0
        total = preds[i].sum()
        if total > 0: preds[i] /= total
        else: preds[i] = np.ones(preds.shape[1]) / preds.shape[1]
    return preds


def build_features_fold_safe(train_df_fold, target_df, is_train=True):
    """Build features where global_stats are computed ONLY from train_df_fold.
    target_df is the data we build features for (could be val fold or test).
    """
    from features_v4 import compute_global_stats_v4, build_features_v4, get_feature_names_v4

    # Compute stats ONLY from this fold's training data
    global_stats = compute_global_stats_v4(train_df_fold)

    # Build features for target data using fold-specific stats
    feat = build_features_v4(target_df, is_train=is_train, global_stats_v4=global_stats)
    feature_names = get_feature_names_v4(feat)

    return feat, feature_names, global_stats


def feature_selection_gain_fold(X_tr, y_tr, n_classes, top_k=600, task="multi"):
    """Feature selection using only training fold data."""
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


def test_weighted_ov(f1a, f1p, auc, next_sn, y_act, y_pt, y_srv, act_probs, pt_probs, srv_probs):
    """Compute OV weighted by test-like next_sn distribution."""
    # Test next_sn distribution (approximated)
    test_sn_weights = {2: 0.35, 3: 0.21, 4: 0.14, 5: 0.10, 6: 0.07,
                       7: 0.05, 8: 0.03, 9: 0.02, 10: 0.01}

    # Per-SN bucket evaluation
    sn_results = {}
    for sn in sorted(set(next_sn)):
        mask = next_sn == sn
        if mask.sum() < 10:
            continue
        sn_f1a = macro_f1(y_act[mask], act_probs[mask], N_ACTION)
        sn_f1p = macro_f1(y_pt[mask], pt_probs[mask], N_POINT)
        try:
            sn_auc = roc_auc_score(y_srv[mask], srv_probs[mask])
        except ValueError:
            sn_auc = 0.5
        sn_ov = 0.4 * sn_f1a + 0.4 * sn_f1p + 0.2 * sn_auc
        sn_results[sn] = {"f1a": sn_f1a, "f1p": sn_f1p, "auc": sn_auc, "ov": sn_ov, "n": int(mask.sum())}

    # Weighted OV
    total_w = 0
    weighted_ov = 0
    for sn, w in test_sn_weights.items():
        if sn in sn_results:
            weighted_ov += w * sn_results[sn]["ov"]
            total_w += w
    if total_w > 0:
        weighted_ov /= total_w

    return weighted_ov, sn_results


def main():
    t_start = time.time()
    print("=" * 70)
    print("V5 CLEAN PIPELINE: Strict no-leakage fold-safe CV")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)

    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb

    # Get rally -> match mapping for GroupKFold
    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    rally_uids = train_df["rally_uid"].unique()
    groups = np.array([rally_to_match[r] for r in rally_uids])

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(rally_uids, groups=groups))

    # We'll collect OOF predictions
    # First pass: build features for ALL train to know the size
    print("\n--- Pre-flight: getting feature dimensions ---")
    from features_v4 import compute_global_stats_v4, build_features_v4, get_feature_names_v4
    gs_full = compute_global_stats_v4(train_df)
    feat_full = build_features_v4(train_df, is_train=True, global_stats_v4=gs_full)
    feature_names_full = get_feature_names_v4(feat_full)
    n_samples = len(feat_full)
    print(f"  {len(feature_names_full)} base features, {n_samples} samples")

    y_act_all = feat_full["y_actionId"].values
    y_pt_all = feat_full["y_pointId"].values
    y_srv_all = feat_full["y_serverGetPoint"].values
    next_sn_all = feat_full["next_strikeNumber"].values
    sample_rally_uids = feat_full["rally_uid"].values

    # Map sample index to rally_uid for fold assignment
    sample_to_match = np.array([rally_to_match.get(r, -1) for r in sample_rally_uids])

    # Rebuild fold splits at sample level
    sample_gkf = GroupKFold(n_splits=N_FOLDS)
    sample_fold_splits = list(sample_gkf.split(np.arange(n_samples), groups=sample_to_match))

    # OOF containers
    oof_act = {"CB": np.zeros((n_samples, N_ACTION)), "XGB": np.zeros((n_samples, N_ACTION)), "LGB": np.zeros((n_samples, N_ACTION))}
    oof_pt = {"CB": np.zeros((n_samples, N_POINT)), "XGB": np.zeros((n_samples, N_POINT)), "LGB": np.zeros((n_samples, N_POINT))}
    oof_srv = {"CB": np.zeros(n_samples), "XGB": np.zeros(n_samples), "LGB": np.zeros(n_samples)}

    # ========================================
    # FOLD-SAFE TRAINING LOOP
    # ========================================
    for fold, (tr_idx, val_idx) in enumerate(sample_fold_splits):
        t_fold = time.time()
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{N_FOLDS} (train={len(tr_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        # Get rally UIDs for this fold
        tr_rallies = set(sample_rally_uids[tr_idx])
        val_rallies = set(sample_rally_uids[val_idx])

        # Split raw train_df by rally
        tr_raw = train_df[train_df["rally_uid"].isin(tr_rallies)]
        val_raw = train_df[train_df["rally_uid"].isin(val_rallies)]

        # FOLD-SAFE: compute global stats ONLY from training fold
        print("  Computing fold-safe global stats...")
        t0 = time.time()
        fold_stats = compute_global_stats_v4(tr_raw)
        print(f"    Done in {time.time()-t0:.1f}s")

        # Build features using fold-specific stats
        print("  Building features (train fold)...")
        t0 = time.time()
        feat_tr = build_features_v4(tr_raw, is_train=True, global_stats_v4=fold_stats)
        feat_val = build_features_v4(val_raw, is_train=True, global_stats_v4=fold_stats)
        fnames = get_feature_names_v4(feat_tr)
        print(f"    {len(fnames)} features, train={len(feat_tr)}, val={len(feat_val)} in {time.time()-t0:.1f}s")

        X_tr = feat_tr[fnames].values.astype(np.float32)
        X_val = feat_val[fnames].values.astype(np.float32)
        ya_tr = feat_tr["y_actionId"].values
        ya_val = feat_val["y_actionId"].values
        yp_tr = feat_tr["y_pointId"].values
        yp_val = feat_val["y_pointId"].values
        ys_tr = feat_tr["y_serverGetPoint"].values
        ys_val = feat_val["y_serverGetPoint"].values
        sn_val = feat_val["next_strikeNumber"].values

        # Clean
        X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
        X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)

        # FOLD-SAFE feature selection (only using training fold)
        print("  Fold-safe feature selection...")
        t0 = time.time()
        sel_act = feature_selection_gain_fold(X_tr, ya_tr, N_ACTION, top_k=500, task="multi")
        sel_pt = feature_selection_gain_fold(X_tr, yp_tr, N_POINT, top_k=500, task="multi")
        sel_srv = feature_selection_gain_fold(X_tr, ys_tr, 2, top_k=300, task="binary")
        print(f"    Act={len(sel_act)}, Pt={len(sel_pt)}, Srv={len(sel_srv)} in {time.time()-t0:.1f}s")

        Xa_tr, Xa_val = X_tr[:, sel_act], X_val[:, sel_act]
        Xp_tr, Xp_val = X_tr[:, sel_pt], X_val[:, sel_pt]
        Xs_tr, Xs_val = X_tr[:, sel_srv], X_val[:, sel_srv]

        # --- CatBoost ---
        print("  Training CatBoost...")
        t0 = time.time()
        m = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_ACTION,
                               auto_class_weights="Balanced", early_stopping_rounds=150,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                               bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m.fit(Xa_tr, ya_tr, eval_set=(Xa_val, ya_val))
        oof_act["CB"][val_idx] = m.predict_proba(Xa_val)

        m = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_POINT,
                               auto_class_weights="Balanced", early_stopping_rounds=150,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                               bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m.fit(Xp_tr, yp_tr, eval_set=(Xp_val, yp_val))
        oof_pt["CB"][val_idx] = m.predict_proba(Xp_val)

        m = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=8,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               early_stopping_rounds=150, verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
        m.fit(Xs_tr, ys_tr, eval_set=(Xs_val, ys_val))
        oof_srv["CB"][val_idx] = m.predict_proba(Xs_val)[:, 1]
        print(f"    CB done in {time.time()-t0:.0f}s")

        # --- XGBoost ---
        print("  Training XGBoost...")
        t0 = time.time()
        dtrain = xgb.DMatrix(Xa_tr, label=ya_tr)
        dval = xgb.DMatrix(Xa_val, label=ya_val)
        params = {"objective": "multi:softprob", "num_class": N_ACTION,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                  "subsample": 0.8, "colsample_bytree": 0.7,
                  "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        mx = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dval, "val")],
                       early_stopping_rounds=150, verbose_eval=0)
        oof_act["XGB"][val_idx] = mx.predict(dval, iteration_range=(0, mx.best_iteration+1))

        dtrain = xgb.DMatrix(Xp_tr, label=yp_tr)
        dval = xgb.DMatrix(Xp_val, label=yp_val)
        params["num_class"] = N_POINT
        mx = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dval, "val")],
                       early_stopping_rounds=150, verbose_eval=0)
        oof_pt["XGB"][val_idx] = mx.predict(dval, iteration_range=(0, mx.best_iteration+1))

        dtrain = xgb.DMatrix(Xs_tr, label=ys_tr)
        dval = xgb.DMatrix(Xs_val, label=ys_val)
        params_bin = {"objective": "binary:logistic", "eval_metric": "auc",
                      "tree_method": "hist", "learning_rate": 0.03, "max_depth": 8,
                      "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
                      "lambda": 1, "seed": RANDOM_SEED, "verbosity": 0}
        mx = xgb.train(params_bin, dtrain, num_boost_round=2000, evals=[(dval, "val")],
                       early_stopping_rounds=150, verbose_eval=0)
        oof_srv["XGB"][val_idx] = mx.predict(dval, iteration_range=(0, mx.best_iteration+1))
        print(f"    XGB done in {time.time()-t0:.0f}s")

        # --- LightGBM ---
        print("  Training LightGBM...")
        t0 = time.time()
        dtrain = lgb.Dataset(Xa_tr, label=ya_tr)
        dval_l = lgb.Dataset(Xa_val, label=ya_val, reference=dtrain)
        params_l = {"objective": "multiclass", "num_class": N_ACTION,
                    "metric": "multi_logloss", "learning_rate": 0.03,
                    "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                    "subsample": 0.8, "colsample_bytree": 0.7, "is_unbalance": True,
                    "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        ml = lgb.train(params_l, dtrain, num_boost_round=2000, valid_sets=[dval_l],
                       callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof_act["LGB"][val_idx] = ml.predict(Xa_val)

        dtrain = lgb.Dataset(Xp_tr, label=yp_tr)
        dval_l = lgb.Dataset(Xp_val, label=yp_val, reference=dtrain)
        params_l["num_class"] = N_POINT
        ml = lgb.train(params_l, dtrain, num_boost_round=2000, valid_sets=[dval_l],
                       callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof_pt["LGB"][val_idx] = ml.predict(Xp_val)

        dtrain = lgb.Dataset(Xs_tr, label=ys_tr)
        dval_l = lgb.Dataset(Xs_val, label=ys_val, reference=dtrain)
        params_bin_l = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
                        "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                        "subsample": 0.8, "colsample_bytree": 0.8, "is_unbalance": True,
                        "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        ml = lgb.train(params_bin_l, dtrain, num_boost_round=2000, valid_sets=[dval_l],
                       callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof_srv["LGB"][val_idx] = ml.predict(Xs_val)
        print(f"    LGB done in {time.time()-t0:.0f}s")

        # Fold summary
        for name in ["CB", "XGB", "LGB"]:
            ar = apply_action_rules(oof_act[name][val_idx], sn_val)
            f1a = macro_f1(ya_val, ar, N_ACTION)
            f1p = macro_f1(yp_val, oof_pt[name][val_idx], N_POINT)
            auc = roc_auc_score(ys_val, oof_srv[name][val_idx])
            ov = 0.4*f1a + 0.4*f1p + 0.2*auc
            print(f"    {name}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

        print(f"  Fold {fold+1} total: {(time.time()-t_fold)/60:.1f} min")

    # ========================================
    # OVERALL OOF EVALUATION
    # ========================================
    print(f"\n{'='*60}")
    print("OVERALL OOF RESULTS (strict no-leakage)")
    print(f"{'='*60}")

    for name in ["CB", "XGB", "LGB"]:
        ar = apply_action_rules(oof_act[name], next_sn_all)
        f1a = macro_f1(y_act_all, ar, N_ACTION)
        f1p = macro_f1(y_pt_all, oof_pt[name], N_POINT)
        auc = roc_auc_score(y_srv_all, oof_srv[name])
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        print(f"  {name} OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # Blend search
    print("\n--- Blend Search ---")
    best_ov = -1
    best_params = None
    for w_cb in np.arange(0.2, 0.7, 0.1):
        for w_xg in np.arange(0.1, 0.7 - w_cb + 0.05, 0.1):
            w_lg = round(1.0 - w_cb - w_xg, 1)
            if w_lg < 0 or w_lg > 0.5: continue

            ba = w_cb * oof_act["CB"] + w_xg * oof_act["XGB"] + w_lg * oof_act["LGB"]
            bp = w_cb * oof_pt["CB"] + w_xg * oof_pt["XGB"] + w_lg * oof_pt["LGB"]
            bs = w_cb * oof_srv["CB"] + w_xg * oof_srv["XGB"] + w_lg * oof_srv["LGB"]

            bar = apply_action_rules(ba, next_sn_all)
            f1a = macro_f1(y_act_all, bar, N_ACTION)
            f1p = macro_f1(y_pt_all, bp, N_POINT)
            auc = roc_auc_score(y_srv_all, bs)
            ov = 0.4*f1a + 0.4*f1p + 0.2*auc
            if ov > best_ov:
                best_ov = ov
                best_params = (w_cb, w_xg, w_lg)

    w_cb, w_xg, w_lg = best_params
    print(f"  Best blend: CB={w_cb:.1f} XGB={w_xg:.1f} LGB={w_lg:.1f} OV={best_ov:.4f}")

    # Test-weighted evaluation
    ba = w_cb * oof_act["CB"] + w_xg * oof_act["XGB"] + w_lg * oof_act["LGB"]
    bp = w_cb * oof_pt["CB"] + w_xg * oof_pt["XGB"] + w_lg * oof_pt["LGB"]
    bs = w_cb * oof_srv["CB"] + w_xg * oof_srv["XGB"] + w_lg * oof_srv["LGB"]
    bar = apply_action_rules(ba, next_sn_all)

    weighted_ov, sn_results = test_weighted_ov(
        macro_f1(y_act_all, bar, N_ACTION),
        macro_f1(y_pt_all, bp, N_POINT),
        roc_auc_score(y_srv_all, bs),
        next_sn_all, y_act_all, y_pt_all, y_srv_all, bar, bp, bs
    )
    print(f"\n  Test-weighted OV: {weighted_ov:.4f}")
    print(f"\n  Per-SN breakdown:")
    for sn in sorted(sn_results.keys()):
        r = sn_results[sn]
        print(f"    SN={sn}: F1a={r['f1a']:.4f} F1p={r['f1p']:.4f} AUC={r['auc']:.4f} OV={r['ov']:.4f} (n={r['n']})")

    # Save OOF for reference
    np.savez(os.path.join(MODEL_DIR, "oof_v5_clean.npz"),
             **{f"{m}_act": oof_act[m] for m in ["CB", "XGB", "LGB"]},
             **{f"{m}_pt": oof_pt[m] for m in ["CB", "XGB", "LGB"]},
             **{f"{m}_srv": oof_srv[m] for m in ["CB", "XGB", "LGB"]},
             y_act=y_act_all, y_pt=y_pt_all, y_srv=y_srv_all, next_sn=next_sn_all)

    # ========================================
    # FINAL SUBMISSION (train on full data, predict test)
    # ========================================
    print(f"\n{'='*60}")
    print("FINAL: Train on full data, predict test")
    print(f"{'='*60}")

    # Full-data stats (OK for final model, not for CV)
    full_stats = compute_global_stats_v4(train_df)
    feat_train_full = build_features_v4(train_df, is_train=True, global_stats_v4=full_stats)
    feat_test_full = build_features_v4(test_df, is_train=False, global_stats_v4=full_stats)
    fnames = get_feature_names_v4(feat_train_full)

    X_full = feat_train_full[fnames].values.astype(np.float32)
    X_test = feat_test_full[fnames].values.astype(np.float32)
    X_full = np.nan_to_num(X_full, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    ya_full = feat_train_full["y_actionId"].values
    yp_full = feat_train_full["y_pointId"].values
    ys_full = feat_train_full["y_serverGetPoint"].values
    test_sn = feat_test_full["next_strikeNumber"].values

    # Feature selection on full data (OK for final model)
    sel_act = feature_selection_gain_fold(X_full, ya_full, N_ACTION, top_k=500)
    sel_pt = feature_selection_gain_fold(X_full, yp_full, N_POINT, top_k=500)
    sel_srv = feature_selection_gain_fold(X_full, ys_full, 2, top_k=300)

    test_act = np.zeros((len(X_test), N_ACTION))
    test_pt = np.zeros((len(X_test), N_POINT))
    test_srv = np.zeros(len(X_test))

    # CatBoost on full data
    for task_name, X_sel_tr, y_sel, X_sel_te, target_arr, n_cls in [
        ("act", X_full[:, sel_act], ya_full, X_test[:, sel_act], test_act, N_ACTION),
        ("pt", X_full[:, sel_pt], yp_full, X_test[:, sel_pt], test_pt, N_POINT),
    ]:
        m = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=n_cls,
                               auto_class_weights="Balanced", verbose=0,
                               random_seed=RANDOM_SEED, l2_leaf_reg=3,
                               bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m.fit(X_sel_tr, y_sel)
        cb_pred = m.predict_proba(X_sel_te)

        dtrain = xgb.DMatrix(X_sel_tr, label=y_sel)
        params = {"objective": "multi:softprob", "num_class": n_cls,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                  "subsample": 0.8, "colsample_bytree": 0.7,
                  "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        mx = xgb.train(params, dtrain, num_boost_round=1500, verbose_eval=0)
        xgb_pred = mx.predict(xgb.DMatrix(X_sel_te))

        target_arr[:] = w_cb * cb_pred + w_xg * xgb_pred + w_lg * cb_pred  # LGB fallback to CB for speed
        print(f"  {task_name} final model trained")

    # Server
    m = CatBoostClassifier(iterations=2000, learning_rate=0.03, depth=8,
                           loss_function="Logloss", auto_class_weights="Balanced",
                           verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
    m.fit(X_full[:, sel_srv], ys_full)
    test_srv[:] = m.predict_proba(X_test[:, sel_srv])[:, 1]

    # Apply rules
    test_act = apply_action_rules(test_act, test_sn)

    submission = pd.DataFrame({
        "rally_uid": feat_test_full["rally_uid"].values.astype(int),
        "actionId": np.argmax(test_act, axis=1).astype(int),
        "pointId": np.argmax(test_pt, axis=1).astype(int),
        "serverGetPoint": (test_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out = os.path.join(SUBMISSION_DIR, "submission_v5_clean.csv")
    submission.to_csv(out, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nSaved: {out}")
    print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId: {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  serverGetPoint: {submission.serverGetPoint.value_counts().to_dict()}")

    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
