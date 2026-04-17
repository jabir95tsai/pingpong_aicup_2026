"""V8 Champion Pipeline: V7 + cross-task stacking + extreme pointId class weights.

Key improvements over V7:
1. Cross-task stacking: actionId OOF probs appended as pointId features
2. Extreme class weights for FH_short(1) and BH_short(3) — currently F1=0
3. serverGetPoint safety: works when test col is NaN/missing (new test.csv)

Cross-task design (no leakage):
  - CV: train action models → get val OOF probs → append to val point features
  - Train point features get zeros (val OOF probs not known at train time)
  - Final: train action on full data → predict test → append to test point features
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

# Extreme class weights for pointId to fix F1=0 on FH_short(1) and BH_short(3)
# class order: 0=miss/net, 1=FH_short, 2=mid_short, 3=BH_short,
#              4=FH_half, 5=mid_half, 6=BH_half, 7=FH_long, 8=mid_long, 9=BH_long
POINT_CLASS_WEIGHTS = [0.8, 8.0, 2.0, 8.0, 2.0, 1.5, 2.0, 1.0, 1.0, 1.0]
POINT_WEIGHT_MAP = {i: w for i, w in enumerate(POINT_CLASS_WEIGHTS)}


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
    working_probs = probs.copy()
    if is_action and next_sn is not None:
        working_probs = apply_action_rules(working_probs, next_sn)

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


def make_point_sample_weights(y):
    """Per-sample weights for pointId based on class."""
    return np.array([POINT_WEIGHT_MAP.get(int(yi), 1.0) for yi in y], dtype=np.float32)


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
    print(f"V8 CHAMPION PIPELINE {'(SMOKE TEST)' if is_smoke else ''}")
    print(f"  - V5 features (931 dims) + cross-task action->point stacking")
    print(f"  - Extreme pointId class weights: FH_short x8, BH_short x8")
    print(f"  - GroupKFold(match) {n_folds}-fold CV")
    print(f"  - 3-model blend (CB+XGB+LGB)")
    print(f"  - Integrated threshold optimization")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)

    # Safety: ensure serverGetPoint in test doesn't leak into features
    # (handles both current test with real values and future test with NaN)
    test_df["serverGetPoint"] = -1

    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    from features_v5 import compute_global_stats_v5, build_features_v5, get_feature_names_v5

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

    sample_gkf = GroupKFold(n_splits=max(n_folds, 2))
    all_splits = list(sample_gkf.split(np.arange(n_samples), groups=sample_to_match))
    if is_smoke:
        all_splits = all_splits[:1]

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

        print("  Computing fold-safe stats...")
        t0 = time.time()
        fold_stats = compute_global_stats_v5(tr_raw)
        print(f"    Done ({time.time()-t0:.1f}s)")

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

        print("  Feature selection...")
        t0 = time.time()
        sel_act = feature_selection_gain(X_tr, ya_tr, N_ACTION, top_k=top_k_act)
        sel_pt = feature_selection_gain(X_tr, yp_tr, N_POINT, top_k=top_k_pt)
        sel_srv = feature_selection_gain(X_tr, ys_tr, 2, top_k=top_k_srv, task="binary")
        print(f"    Act={len(sel_act)}, Pt={len(sel_pt)}, Srv={len(sel_srv)} ({time.time()-t0:.1f}s)")

        Xa_tr, Xa_val = X_tr[:, sel_act], X_val[:, sel_act]
        Xp_tr, Xp_val = X_tr[:, sel_pt], X_val[:, sel_pt]
        Xs_tr, Xs_val = X_tr[:, sel_srv], X_val[:, sel_srv]

        # ---- ACTION MODELS (same as V7) ----
        print("  Training CatBoost (action)...")
        t0 = time.time()
        m = CatBoostClassifier(iterations=n_boost, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_ACTION,
                               auto_class_weights="Balanced", early_stopping_rounds=es_rounds,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                               bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m.fit(Xa_tr, ya_tr, eval_set=(Xa_val, ya_val))
        oof_act["CB"][val_idx] = m.predict_proba(Xa_val)

        print("  Training XGBoost (action)...")
        dtrain = xgb.DMatrix(Xa_tr, label=ya_tr)
        dval = xgb.DMatrix(Xa_val, label=ya_val)
        params_xgb_act = {"objective": "multi:softprob", "num_class": N_ACTION,
                          "eval_metric": "mlogloss", "tree_method": "hist",
                          "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                          "subsample": 0.8, "colsample_bytree": 0.7,
                          "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        mx_act = xgb.train(params_xgb_act, dtrain, num_boost_round=n_boost, evals=[(dval, "val")],
                           early_stopping_rounds=es_rounds, verbose_eval=0)
        oof_act["XGB"][val_idx] = mx_act.predict(dval, iteration_range=(0, mx_act.best_iteration+1))

        print("  Training LightGBM (action)...")
        dtrain_l = lgb.Dataset(Xa_tr, label=ya_tr)
        dval_l = lgb.Dataset(Xa_val, label=ya_val, reference=dtrain_l)
        params_lgb_act = {"objective": "multiclass", "num_class": N_ACTION,
                          "metric": "multi_logloss", "learning_rate": 0.03,
                          "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                          "subsample": 0.8, "colsample_bytree": 0.7, "is_unbalance": True,
                          "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        ml_act = lgb.train(params_lgb_act, dtrain_l, num_boost_round=n_boost, valid_sets=[dval_l],
                           callbacks=[lgb.early_stopping(es_rounds), lgb.log_evaluation(0)])
        oof_act["LGB"][val_idx] = ml_act.predict(Xa_val)
        print(f"    Action models done ({time.time()-t0:.0f}s)")

        # ---- CROSS-TASK: append val action OOF probs to point features ----
        # Val: use actual OOF action probs (3-model average)
        val_act_blend = (oof_act["CB"][val_idx] +
                         oof_act["XGB"][val_idx] +
                         oof_act["LGB"][val_idx]) / 3.0
        # Train: zeros (action probs not available without leakage)
        Xp_tr_aug = np.hstack([Xp_tr, np.zeros((len(Xp_tr), N_ACTION), dtype=np.float32)])
        Xp_val_aug = np.hstack([Xp_val, val_act_blend.astype(np.float32)])

        # ---- POINT MODELS (extreme class weights) ----
        print("  Training CatBoost (point, extreme weights)...")
        t0 = time.time()
        m = CatBoostClassifier(iterations=n_boost, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_POINT,
                               class_weights=POINT_CLASS_WEIGHTS,  # extreme, not auto
                               early_stopping_rounds=es_rounds,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                               bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m.fit(Xp_tr_aug, yp_tr, eval_set=(Xp_val_aug, yp_val))
        oof_pt["CB"][val_idx] = m.predict_proba(Xp_val_aug)

        print("  Training XGBoost (point, sample weights)...")
        sw_tr = make_point_sample_weights(yp_tr)
        dtrain = xgb.DMatrix(Xp_tr_aug, label=yp_tr, weight=sw_tr)
        dval = xgb.DMatrix(Xp_val_aug, label=yp_val)
        params_xgb_pt = {"objective": "multi:softprob", "num_class": N_POINT,
                         "eval_metric": "mlogloss", "tree_method": "hist",
                         "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                         "subsample": 0.8, "colsample_bytree": 0.7,
                         "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        mx_pt = xgb.train(params_xgb_pt, dtrain, num_boost_round=n_boost, evals=[(dval, "val")],
                          early_stopping_rounds=es_rounds, verbose_eval=0)
        oof_pt["XGB"][val_idx] = mx_pt.predict(dval, iteration_range=(0, mx_pt.best_iteration+1))

        print("  Training LightGBM (point, sample weights)...")
        dtrain_l = lgb.Dataset(Xp_tr_aug, label=yp_tr, weight=sw_tr)
        dval_l = lgb.Dataset(Xp_val_aug, label=yp_val, reference=dtrain_l)
        params_lgb_pt = {"objective": "multiclass", "num_class": N_POINT,
                         "metric": "multi_logloss", "learning_rate": 0.03,
                         "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                         "subsample": 0.8, "colsample_bytree": 0.7,
                         "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        ml_pt = lgb.train(params_lgb_pt, dtrain_l, num_boost_round=n_boost, valid_sets=[dval_l],
                          callbacks=[lgb.early_stopping(es_rounds), lgb.log_evaluation(0)])
        oof_pt["LGB"][val_idx] = ml_pt.predict(Xp_val_aug)
        print(f"    Point models done ({time.time()-t0:.0f}s)")

        # ---- SERVER MODELS (same as V7) ----
        print("  Training server models...")
        t0 = time.time()
        m = CatBoostClassifier(iterations=n_boost, learning_rate=0.03, depth=8,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               early_stopping_rounds=es_rounds, verbose=0,
                               random_seed=RANDOM_SEED, l2_leaf_reg=3)
        m.fit(Xs_tr, ys_tr, eval_set=(Xs_val, ys_val))
        oof_srv["CB"][val_idx] = m.predict_proba(Xs_val)[:, 1]

        dtrain = xgb.DMatrix(Xs_tr, label=ys_tr)
        dval = xgb.DMatrix(Xs_val, label=ys_val)
        params_bin = {"objective": "binary:logistic", "eval_metric": "auc",
                      "tree_method": "hist", "learning_rate": 0.03, "max_depth": 8,
                      "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
                      "lambda": 1, "seed": RANDOM_SEED, "verbosity": 0}
        mx = xgb.train(params_bin, dtrain, num_boost_round=n_boost, evals=[(dval, "val")],
                       early_stopping_rounds=es_rounds, verbose_eval=0)
        oof_srv["XGB"][val_idx] = mx.predict(dval, iteration_range=(0, mx.best_iteration+1))

        dtrain_l = lgb.Dataset(Xs_tr, label=ys_tr)
        dval_l = lgb.Dataset(Xs_val, label=ys_val, reference=dtrain_l)
        params_lgb_bin = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
                          "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                          "subsample": 0.8, "colsample_bytree": 0.8, "is_unbalance": True,
                          "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        ml = lgb.train(params_lgb_bin, dtrain_l, num_boost_round=n_boost, valid_sets=[dval_l],
                       callbacks=[lgb.early_stopping(es_rounds), lgb.log_evaluation(0)])
        oof_srv["LGB"][val_idx] = ml.predict(Xs_val)
        print(f"    Server models done ({time.time()-t0:.0f}s)")

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

        # Per-class breakdown for pointId (to verify short zones)
        pt_blend_val = (oof_pt["CB"][val_idx] + oof_pt["XGB"][val_idx] + oof_pt["LGB"][val_idx]) / 3
        pt_pred_val = np.argmax(pt_blend_val, axis=1)
        per_class_f1 = f1_score(yp_val, pt_pred_val, labels=list(range(N_POINT)),
                                average=None, zero_division=0)
        short_classes = {1: "FH_short", 3: "BH_short"}
        for cls_id, cls_name in short_classes.items():
            print(f"      pointId {cls_name}(cls {cls_id}): F1={per_class_f1[cls_id]:.4f} "
                  f"pred_count={(pt_pred_val==cls_id).sum()}")

        print(f"  Fold {fold+1}: {(time.time()-t_fold)/60:.1f} min")
        gc.collect()

    # ========================================
    # OVERALL OOF EVALUATION
    # ========================================
    if is_smoke:
        eval_mask = all_splits[0][1]
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

    # Full pointId per-class breakdown
    pt_blend_all = (oof_pt["CB"][eval_mask] + oof_pt["XGB"][eval_mask] + oof_pt["LGB"][eval_mask]) / 3
    pt_pred_all = np.argmax(pt_blend_all, axis=1)
    per_class_f1_all = f1_score(y_pt_all[eval_mask], pt_pred_all, labels=list(range(N_POINT)),
                                average=None, zero_division=0)
    print("\n  pointId per-class F1 (OOF blend):")
    zone_names = ["miss/net", "FH_short", "mid_short", "BH_short",
                  "FH_half", "mid_half", "BH_half", "FH_long", "mid_long", "BH_long"]
    for i, (name, f1v) in enumerate(zip(zone_names, per_class_f1_all)):
        marker = " ← was F1=0 in V7" if i in [1, 3] else ""
        print(f"    [{i}] {name}: {f1v:.4f}{marker}")

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

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(os.path.join(MODEL_DIR, "oof_v8.npz"),
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

    sel_act = feature_selection_gain(X_full, ya_full, N_ACTION, top_k=top_k_act)
    sel_pt = feature_selection_gain(X_full, yp_full, N_POINT, top_k=top_k_pt)
    sel_srv = feature_selection_gain(X_full, ys_full, 2, top_k=top_k_srv, task="binary")
    print(f"  Selected: Act={len(sel_act)}, Pt={len(sel_pt)}, Srv={len(sel_srv)}")

    test_act = np.zeros((len(X_test), N_ACTION))
    test_pt = np.zeros((len(X_test), N_POINT))
    test_srv = np.zeros(len(X_test))

    n_final_boost = 100 if is_smoke else 1500

    # ---- Train action models first (needed for cross-task) ----
    print(f"\n  Training actionId (full data)...")
    t0 = time.time()
    Xa_full, Xa_test = X_full[:, sel_act], X_test[:, sel_act]

    m_cb_act = CatBoostClassifier(iterations=n_final_boost, learning_rate=0.03, depth=8,
                                  loss_function="MultiClass", classes_count=N_ACTION,
                                  auto_class_weights="Balanced", verbose=0,
                                  random_seed=RANDOM_SEED, l2_leaf_reg=3,
                                  bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
    m_cb_act.fit(Xa_full, ya_full)
    cb_act_pred = m_cb_act.predict_proba(Xa_test)

    dtrain = xgb.DMatrix(Xa_full, label=ya_full)
    params = {"objective": "multi:softprob", "num_class": N_ACTION,
              "eval_metric": "mlogloss", "tree_method": "hist",
              "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
              "subsample": 0.8, "colsample_bytree": 0.7,
              "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
    mx_act_full = xgb.train(params, dtrain, num_boost_round=n_final_boost, verbose_eval=0)
    xgb_act_pred = mx_act_full.predict(xgb.DMatrix(Xa_test))

    dtrain_l = lgb.Dataset(Xa_full, label=ya_full)
    params_l = {"objective": "multiclass", "num_class": N_ACTION,
                "metric": "multi_logloss", "learning_rate": 0.03,
                "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                "subsample": 0.8, "colsample_bytree": 0.7, "is_unbalance": True,
                "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
    ml_act_full = lgb.train(params_l, dtrain_l, num_boost_round=n_final_boost)
    lgb_act_pred = ml_act_full.predict(Xa_test)

    test_act[:] = w_cb * cb_act_pred + w_xg * xgb_act_pred + w_lg * lgb_act_pred
    # Cross-task: blend test action probs for point feature augmentation
    test_act_for_pt = (cb_act_pred + xgb_act_pred + lgb_act_pred) / 3.0
    print(f"    actionId done ({time.time()-t0:.0f}s)")

    # ---- Train point models with cross-task augmentation ----
    print(f"\n  Training pointId (full data, cross-task + extreme weights)...")
    t0 = time.time()
    Xp_full_base = X_full[:, sel_pt]
    Xp_test_base = X_test[:, sel_pt]

    # Full train: action probs not available without leakage → zeros
    Xp_full_aug = np.hstack([Xp_full_base, np.zeros((len(Xp_full_base), N_ACTION), dtype=np.float32)])
    # Test: use actual predicted action probs
    Xp_test_aug = np.hstack([Xp_test_base, test_act_for_pt.astype(np.float32)])

    sw_full = make_point_sample_weights(yp_full)

    m_cb_pt = CatBoostClassifier(iterations=n_final_boost, learning_rate=0.03, depth=8,
                                 loss_function="MultiClass", classes_count=N_POINT,
                                 class_weights=POINT_CLASS_WEIGHTS,
                                 verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                                 bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
    m_cb_pt.fit(Xp_full_aug, yp_full)
    cb_pt_pred = m_cb_pt.predict_proba(Xp_test_aug)

    dtrain = xgb.DMatrix(Xp_full_aug, label=yp_full, weight=sw_full)
    params_pt = {"objective": "multi:softprob", "num_class": N_POINT,
                 "eval_metric": "mlogloss", "tree_method": "hist",
                 "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                 "subsample": 0.8, "colsample_bytree": 0.7,
                 "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
    mx_pt_full = xgb.train(params_pt, dtrain, num_boost_round=n_final_boost, verbose_eval=0)
    xgb_pt_pred = mx_pt_full.predict(xgb.DMatrix(Xp_test_aug))

    dtrain_l = lgb.Dataset(Xp_full_aug, label=yp_full, weight=sw_full)
    params_l_pt = {"objective": "multiclass", "num_class": N_POINT,
                   "metric": "multi_logloss", "learning_rate": 0.03,
                   "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                   "subsample": 0.8, "colsample_bytree": 0.7,
                   "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
    ml_pt_full = lgb.train(params_l_pt, dtrain_l, num_boost_round=n_final_boost)
    lgb_pt_pred = ml_pt_full.predict(Xp_test_aug)

    test_pt[:] = w_cb * cb_pt_pred + w_xg * xgb_pt_pred + w_lg * lgb_pt_pred
    print(f"    pointId done ({time.time()-t0:.0f}s)")

    # ---- Server (same as V7) ----
    print(f"\n  Training serverGetPoint (full data)...")
    t0 = time.time()
    Xs_full, Xs_test = X_full[:, sel_srv], X_test[:, sel_srv]

    m_cb = CatBoostClassifier(iterations=n_final_boost, learning_rate=0.03, depth=8,
                              loss_function="Logloss", auto_class_weights="Balanced",
                              verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
    m_cb.fit(Xs_full, ys_full)
    cb_srv = m_cb.predict_proba(Xs_test)[:, 1]

    dtrain = xgb.DMatrix(Xs_full, label=ys_full)
    params_bin = {"objective": "binary:logistic", "eval_metric": "auc",
                  "tree_method": "hist", "learning_rate": 0.03, "max_depth": 8,
                  "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
                  "lambda": 1, "seed": RANDOM_SEED, "verbosity": 0}
    mx = xgb.train(params_bin, dtrain, num_boost_round=n_final_boost, verbose_eval=0)
    xgb_srv = mx.predict(xgb.DMatrix(Xs_test))

    dtrain_l = lgb.Dataset(Xs_full, label=ys_full)
    params_l_bin = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
                    "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                    "subsample": 0.8, "colsample_bytree": 0.8, "is_unbalance": True,
                    "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
    ml = lgb.train(params_l_bin, dtrain_l, num_boost_round=n_final_boost)
    lgb_srv = ml.predict(Xs_test)

    test_srv[:] = w_cb * cb_srv + w_xg * xgb_srv + w_lg * lgb_srv
    print(f"    serverGetPoint done ({time.time()-t0:.0f}s)")

    # ========================================
    # APPLY THRESHOLDS & GENERATE SUBMISSION
    # ========================================
    print(f"\n{'='*60}")
    print("GENERATING SUBMISSION")
    print(f"{'='*60}")

    test_act_ruled = apply_action_rules(test_act, test_sn)

    test_act_scaled = test_act_ruled ** (1.0 / temp_act)
    row_sums = test_act_scaled.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    test_act_scaled /= row_sums

    test_pt_scaled = test_pt ** (1.0 / temp_pt)
    row_sums = test_pt_scaled.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    test_pt_scaled /= row_sums

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
    out = os.path.join(SUBMISSION_DIR, f"submission_v8_champion{suffix}.csv")
    submission.to_csv(out, index=False, lineterminator="\n", encoding="utf-8")

    print(f"\n  Saved: {out}")
    print(f"  actionId dist: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId dist:  {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  SGP dist:      {submission.serverGetPoint.value_counts().to_dict()}")

    # Base submission (no threshold)
    pred_act_base = np.argmax(test_act_ruled, axis=1).astype(int)
    pred_pt_base = np.argmax(test_pt, axis=1).astype(int)
    submission_base = pd.DataFrame({
        "rally_uid": feat_test_full["rally_uid"].values.astype(int),
        "actionId": pred_act_base,
        "pointId": pred_pt_base,
        "serverGetPoint": pred_srv,
    })
    out_base = os.path.join(SUBMISSION_DIR, f"submission_v8_base{suffix}.csv")
    submission_base.to_csv(out_base, index=False, lineterminator="\n", encoding="utf-8")
    print(f"  Saved base: {out_base}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Features: {len(fnames)} (V5) + {N_ACTION} cross-task action probs = {len(fnames)+N_ACTION} for point")
    print(f"  pointId weights: {dict(zip(zone_names, POINT_CLASS_WEIGHTS))}")
    print(f"  Blend: CB={w_cb:.2f} XGB={w_xg:.2f} LGB={w_lg:.2f}")
    print(f"  OOF blend OV:     {best_ov:.4f}")
    print(f"  OOF optimized OV: {ov_opt:.4f}")
    print(f"  Threshold gain:   +{ov_opt-best_ov:.4f}")
    print(f"  Total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    zone_names = ["miss/net", "FH_short", "mid_short", "BH_short",
                  "FH_half", "mid_half", "BH_half", "FH_long", "mid_long", "BH_long"]
    np.random.seed(RANDOM_SEED)
    main()
