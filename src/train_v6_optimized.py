"""V6 Optimized Pipeline: V5 base + bug fixes + per-class threshold optimization.

Changes from V5:
1. Fixed LGB bug (line 452): now trains actual LightGBM for final submission
2. Trains all 3 models (CB+XGB+LGB) for SGP in final submission
3. Per-class threshold optimization on OOF for Macro F1
4. Joint temperature scaling + class weight optimization
5. Finer blend weight search (0.05 step)
6. Reuses V5 OOF for threshold optimization (no need to retrain CV)
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, RANDOM_SEED

N_ACTION, N_POINT = 19, 10
SERVE_FORBIDDEN = {15, 16, 17, 18}


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


def optimize_class_weights_joint(probs, y_true, n_classes, n_iter=200):
    """Joint optimization of per-class weights to maximize Macro F1.
    Uses Nelder-Mead on log-weights for unconstrained optimization.
    """
    def neg_macro_f1(log_weights):
        w = np.exp(log_weights)
        adj = probs * w[np.newaxis, :]
        adj /= adj.sum(axis=1, keepdims=True)
        preds = np.argmax(adj, axis=1)
        return -f1_score(y_true, preds, labels=list(range(n_classes)),
                         average="macro", zero_division=0)

    # Start from uniform weights
    x0 = np.zeros(n_classes)
    result = minimize(neg_macro_f1, x0, method="Nelder-Mead",
                      options={"maxiter": n_iter * n_classes, "xatol": 0.01, "fatol": 1e-5})
    best_weights = np.exp(result.x)
    best_f1 = -result.fun
    return best_weights, best_f1


def optimize_class_weights_greedy(probs, y_true, n_classes):
    """Greedy per-class weight optimization with multiple passes."""
    weights = np.ones(n_classes)

    for pass_num in range(3):  # Multiple passes for interaction effects
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
                preds = np.argmax(adj, axis=1)
                score = f1_score(y_true, preds, labels=list(range(n_classes)),
                                 average="macro", zero_division=0)
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


def optimize_temperature_then_weights(probs, y_true, n_classes, next_sn=None, is_action=False):
    """Two-stage: temperature scaling then per-class weight optimization."""
    # Stage 1: Temperature scaling
    best_t = 1.0
    best_score = -1

    working_probs = probs.copy()
    if is_action and next_sn is not None:
        working_probs = apply_action_rules(working_probs, next_sn)

    for t in np.arange(0.3, 3.0, 0.05):
        scaled = working_probs ** (1.0 / t)
        row_sums = scaled.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        scaled /= row_sums
        preds = np.argmax(scaled, axis=1)
        score = f1_score(y_true, preds, labels=list(range(n_classes)),
                         average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t

    # Apply best temperature
    scaled_probs = working_probs ** (1.0 / best_t)
    row_sums = scaled_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    scaled_probs /= row_sums

    base_f1 = f1_score(y_true, np.argmax(working_probs, axis=1),
                       labels=list(range(n_classes)), average="macro", zero_division=0)
    print(f"    Temperature: {best_t:.2f} (F1: {base_f1:.4f} -> {best_score:.4f})")

    # Stage 2: Per-class weight optimization on temperature-scaled probs
    weights_greedy, f1_greedy = optimize_class_weights_greedy(scaled_probs, y_true, n_classes)
    print(f"    Greedy weights F1: {f1_greedy:.4f}")

    # Stage 3: Joint optimization (starting from greedy solution)
    weights_joint, f1_joint = optimize_class_weights_joint(scaled_probs, y_true, n_classes)
    print(f"    Joint opt F1: {f1_joint:.4f}")

    # Pick the best
    if f1_joint > f1_greedy:
        print(f"    -> Using joint optimization ({f1_joint:.4f} > {f1_greedy:.4f})")
        return best_t, weights_joint, f1_joint
    else:
        print(f"    -> Using greedy optimization ({f1_greedy:.4f} >= {f1_joint:.4f})")
        return best_t, weights_greedy, f1_greedy


def main():
    t_start = time.time()
    print("=" * 70)
    print("V6 OPTIMIZED PIPELINE")
    print("  - Bug fixes + threshold optimization + proper 3-model blend")
    print("=" * 70)

    # ========================================
    # PHASE 1: Load V5 OOF and optimize thresholds
    # ========================================
    oof_path = os.path.join(MODEL_DIR, "oof_v5_clean.npz")
    if not os.path.exists(oof_path):
        print(f"ERROR: {oof_path} not found. Run train_v5_clean.py first.")
        return
    data = np.load(oof_path, allow_pickle=True)

    y_act = data["y_act"]
    y_pt = data["y_pt"]
    y_srv = data["y_srv"]
    next_sn = data["next_sn"]

    models_oof = {}
    for prefix in ["CB", "XGB", "LGB"]:
        models_oof[prefix] = {
            "act": data[f"{prefix}_act"],
            "pt": data[f"{prefix}_pt"],
            "srv": data[f"{prefix}_srv"],
        }

    print(f"\nLoaded OOF: {len(y_act)} samples")

    # --- Finer blend search ---
    print("\n--- Finer Blend Search (step=0.05) ---")
    best_ov = -1
    best_blend = None
    for w_cb in np.arange(0.3, 0.85, 0.05):
        for w_xg in np.arange(0.0, 0.5, 0.05):
            w_lg = round(1.0 - w_cb - w_xg, 2)
            if w_lg < 0 or w_lg > 0.5:
                continue
            ba = w_cb * models_oof["CB"]["act"] + w_xg * models_oof["XGB"]["act"] + w_lg * models_oof["LGB"]["act"]
            bp = w_cb * models_oof["CB"]["pt"] + w_xg * models_oof["XGB"]["pt"] + w_lg * models_oof["LGB"]["pt"]
            bs = w_cb * models_oof["CB"]["srv"] + w_xg * models_oof["XGB"]["srv"] + w_lg * models_oof["LGB"]["srv"]

            bar = apply_action_rules(ba, next_sn)
            f1a = f1_score(y_act, np.argmax(bar, axis=1), labels=list(range(N_ACTION)),
                           average="macro", zero_division=0)
            f1p = f1_score(y_pt, np.argmax(bp, axis=1), labels=list(range(N_POINT)),
                           average="macro", zero_division=0)
            auc = roc_auc_score(y_srv, bs)
            ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
            if ov > best_ov:
                best_ov = ov
                best_blend = (w_cb, w_xg, w_lg)

    w_cb, w_xg, w_lg = best_blend
    print(f"  Best blend: CB={w_cb:.2f} XGB={w_xg:.2f} LGB={w_lg:.2f} OV={best_ov:.4f}")

    # Create blended OOF
    blend_act = w_cb * models_oof["CB"]["act"] + w_xg * models_oof["XGB"]["act"] + w_lg * models_oof["LGB"]["act"]
    blend_pt = w_cb * models_oof["CB"]["pt"] + w_xg * models_oof["XGB"]["pt"] + w_lg * models_oof["LGB"]["pt"]
    blend_srv = w_cb * models_oof["CB"]["srv"] + w_xg * models_oof["XGB"]["srv"] + w_lg * models_oof["LGB"]["srv"]

    # --- Baseline OV (argmax) ---
    blend_act_ruled = apply_action_rules(blend_act, next_sn)
    f1a_base = f1_score(y_act, np.argmax(blend_act_ruled, axis=1),
                        labels=list(range(N_ACTION)), average="macro", zero_division=0)
    f1p_base = f1_score(y_pt, np.argmax(blend_pt, axis=1),
                        labels=list(range(N_POINT)), average="macro", zero_division=0)
    auc_base = roc_auc_score(y_srv, blend_srv)
    ov_base = 0.4 * f1a_base + 0.4 * f1p_base + 0.2 * auc_base
    print(f"\n  Baseline OV (argmax): {ov_base:.4f}")
    print(f"    F1_action={f1a_base:.4f}  F1_point={f1p_base:.4f}  AUC={auc_base:.4f}")

    # --- Threshold optimization ---
    print("\n--- Threshold Optimization: actionId ---")
    temp_act, weights_act, f1a_opt = optimize_temperature_then_weights(
        blend_act, y_act, N_ACTION, next_sn=next_sn, is_action=True)

    print("\n--- Threshold Optimization: pointId ---")
    temp_pt, weights_pt, f1p_opt = optimize_temperature_then_weights(
        blend_pt, y_pt, N_POINT, next_sn=None, is_action=False)

    # --- Optimized OV ---
    ov_opt = 0.4 * f1a_opt + 0.4 * f1p_opt + 0.2 * auc_base
    print(f"\n{'='*60}")
    print(f"  OV COMPARISON:")
    print(f"    V5 baseline (argmax):     {ov_base:.4f}")
    print(f"    V6 optimized (threshold): {ov_opt:.4f}  (+{ov_opt - ov_base:.4f})")
    print(f"    F1_action: {f1a_base:.4f} -> {f1a_opt:.4f} (+{f1a_opt-f1a_base:.4f})")
    print(f"    F1_point:  {f1p_base:.4f} -> {f1p_opt:.4f} (+{f1p_opt-f1p_base:.4f})")
    print(f"    AUC:       {auc_base:.4f} (unchanged)")
    print(f"{'='*60}")

    # Save threshold params
    threshold_params = {
        "blend_weights": best_blend,
        "temp_act": temp_act,
        "weights_act": weights_act,
        "temp_pt": temp_pt,
        "weights_pt": weights_pt,
    }
    np.savez(os.path.join(MODEL_DIR, "threshold_params_v6.npz"), **{
        k: v for k, v in threshold_params.items()
    })
    print(f"\n  Saved threshold params to {MODEL_DIR}/threshold_params_v6.npz")

    # ========================================
    # PHASE 2: Retrain final models on full data with proper 3-model blend
    # ========================================
    print(f"\n{'='*60}")
    print("FINAL SUBMISSION: Full-data training with 3-model blend")
    print(f"{'='*60}")

    from data_cleaning import clean_data
    from features_v4 import compute_global_stats_v4, build_features_v4, get_feature_names_v4
    from train_v5_clean import feature_selection_gain_fold
    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)

    # Build features on full data
    full_stats = compute_global_stats_v4(train_df)
    feat_train = build_features_v4(train_df, is_train=True, global_stats_v4=full_stats)
    feat_test = build_features_v4(test_df, is_train=False, global_stats_v4=full_stats)
    fnames = get_feature_names_v4(feat_train)

    X_full = feat_train[fnames].values.astype(np.float32)
    X_test = feat_test[fnames].values.astype(np.float32)
    X_full = np.nan_to_num(X_full, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    ya_full = feat_train["y_actionId"].values
    yp_full = feat_train["y_pointId"].values
    ys_full = feat_train["y_serverGetPoint"].values
    test_sn = feat_test["next_strikeNumber"].values

    # Feature selection on full data
    print("\n  Feature selection on full data...")
    sel_act = feature_selection_gain_fold(X_full, ya_full, N_ACTION, top_k=500)
    sel_pt = feature_selection_gain_fold(X_full, yp_full, N_POINT, top_k=500)
    sel_srv = feature_selection_gain_fold(X_full, ys_full, 2, top_k=300, task="binary")
    print(f"    Act={len(sel_act)}, Pt={len(sel_pt)}, Srv={len(sel_srv)}")

    # Train all 3 models for action and point
    test_act = np.zeros((len(X_test), N_ACTION))
    test_pt = np.zeros((len(X_test), N_POINT))
    test_srv = np.zeros(len(X_test))

    for task_name, X_sel_tr, y_sel, X_sel_te, n_cls in [
        ("actionId", X_full[:, sel_act], ya_full, X_test[:, sel_act], N_ACTION),
        ("pointId", X_full[:, sel_pt], yp_full, X_test[:, sel_pt], N_POINT),
    ]:
        print(f"\n  Training {task_name} ({n_cls} classes)...")

        # CatBoost
        t0 = time.time()
        m_cb = CatBoostClassifier(
            iterations=2000, learning_rate=0.03, depth=8,
            loss_function="MultiClass", classes_count=n_cls,
            auto_class_weights="Balanced", verbose=0,
            random_seed=RANDOM_SEED, l2_leaf_reg=3,
            bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m_cb.fit(X_sel_tr, y_sel)
        cb_pred = m_cb.predict_proba(X_sel_te)
        print(f"    CB done ({time.time()-t0:.0f}s)")

        # XGBoost
        t0 = time.time()
        dtrain = xgb.DMatrix(X_sel_tr, label=y_sel)
        params = {"objective": "multi:softprob", "num_class": n_cls,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                  "subsample": 0.8, "colsample_bytree": 0.7,
                  "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        mx = xgb.train(params, dtrain, num_boost_round=1500, verbose_eval=0)
        xgb_pred = mx.predict(xgb.DMatrix(X_sel_te))
        print(f"    XGB done ({time.time()-t0:.0f}s)")

        # LightGBM (FIX: actually train LGB instead of reusing CB)
        t0 = time.time()
        dtrain_lgb = lgb.Dataset(X_sel_tr, label=y_sel)
        params_lgb = {"objective": "multiclass", "num_class": n_cls,
                      "metric": "multi_logloss", "learning_rate": 0.03,
                      "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                      "subsample": 0.8, "colsample_bytree": 0.7, "is_unbalance": True,
                      "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        ml = lgb.train(params_lgb, dtrain_lgb, num_boost_round=1500)
        lgb_pred = ml.predict(X_sel_te)
        print(f"    LGB done ({time.time()-t0:.0f}s)")

        # Proper 3-model blend
        blended = w_cb * cb_pred + w_xg * xgb_pred + w_lg * lgb_pred

        if task_name == "actionId":
            test_act[:] = blended
        else:
            test_pt[:] = blended

    # Server: train all 3 models
    print(f"\n  Training serverGetPoint (binary)...")
    t0 = time.time()
    m_cb = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=8,
        loss_function="Logloss", auto_class_weights="Balanced",
        verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
    m_cb.fit(X_full[:, sel_srv], ys_full)
    cb_srv = m_cb.predict_proba(X_test[:, sel_srv])[:, 1]

    dtrain = xgb.DMatrix(X_full[:, sel_srv], label=ys_full)
    params_bin = {"objective": "binary:logistic", "eval_metric": "auc",
                  "tree_method": "hist", "learning_rate": 0.03, "max_depth": 8,
                  "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
                  "lambda": 1, "seed": RANDOM_SEED, "verbosity": 0}
    mx = xgb.train(params_bin, dtrain, num_boost_round=1500, verbose_eval=0)
    xgb_srv = mx.predict(xgb.DMatrix(X_test[:, sel_srv]))

    dtrain_lgb = lgb.Dataset(X_full[:, sel_srv], label=ys_full)
    params_lgb_bin = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
                      "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                      "subsample": 0.8, "colsample_bytree": 0.8, "is_unbalance": True,
                      "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
    ml = lgb.train(params_lgb_bin, dtrain_lgb, num_boost_round=1500)
    lgb_srv = ml.predict(X_test[:, sel_srv])

    test_srv[:] = w_cb * cb_srv + w_xg * xgb_srv + w_lg * lgb_srv
    print(f"    SGP done ({time.time()-t0:.0f}s)")

    # ========================================
    # PHASE 3: Apply threshold optimization + rules -> submission
    # ========================================
    print(f"\n{'='*60}")
    print("APPLYING THRESHOLD OPTIMIZATION")
    print(f"{'='*60}")

    # Apply action rules first
    test_act = apply_action_rules(test_act, test_sn)

    # Apply temperature scaling
    test_act_scaled = test_act ** (1.0 / temp_act)
    row_sums = test_act_scaled.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    test_act_scaled /= row_sums

    test_pt_scaled = test_pt ** (1.0 / temp_pt)
    row_sums = test_pt_scaled.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    test_pt_scaled /= row_sums

    # Apply class weights
    test_act_adj = test_act_scaled * weights_act[np.newaxis, :]
    test_act_adj /= test_act_adj.sum(axis=1, keepdims=True)

    test_pt_adj = test_pt_scaled * weights_pt[np.newaxis, :]
    test_pt_adj /= test_pt_adj.sum(axis=1, keepdims=True)

    # Generate predictions
    pred_act = np.argmax(test_act_adj, axis=1).astype(int)
    pred_pt = np.argmax(test_pt_adj, axis=1).astype(int)
    pred_srv = (test_srv >= 0.5).astype(int)

    # Also generate V5-style submission (argmax, no threshold) for comparison
    pred_act_v5 = np.argmax(test_act, axis=1).astype(int)
    pred_pt_v5 = np.argmax(test_pt, axis=1).astype(int)

    print(f"\n  Prediction changes from threshold optimization:")
    act_changed = (pred_act != pred_act_v5).sum()
    pt_changed = (pred_pt != pred_pt_v5).sum()
    print(f"    actionId: {act_changed}/{len(pred_act)} predictions changed ({100*act_changed/len(pred_act):.1f}%)")
    print(f"    pointId:  {pt_changed}/{len(pred_pt)} predictions changed ({100*pt_changed/len(pred_pt):.1f}%)")

    # Save submissions
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # V6 optimized submission
    submission_v6 = pd.DataFrame({
        "rally_uid": feat_test["rally_uid"].values.astype(int),
        "actionId": pred_act,
        "pointId": pred_pt,
        "serverGetPoint": pred_srv,
    })
    out_v6 = os.path.join(SUBMISSION_DIR, "submission_v6_optimized.csv")
    submission_v6.to_csv(out_v6, index=False, lineterminator="\n", encoding="utf-8")

    # V6 base submission (bug-fixed blend, no threshold)
    submission_v6_base = pd.DataFrame({
        "rally_uid": feat_test["rally_uid"].values.astype(int),
        "actionId": pred_act_v5,
        "pointId": pred_pt_v5,
        "serverGetPoint": pred_srv,
    })
    out_v6_base = os.path.join(SUBMISSION_DIR, "submission_v6_base.csv")
    submission_v6_base.to_csv(out_v6_base, index=False, lineterminator="\n", encoding="utf-8")

    print(f"\n  Saved: {out_v6}")
    print(f"  Saved: {out_v6_base}")

    print(f"\n  V6 optimized actionId dist: {submission_v6.actionId.value_counts().sort_index().to_dict()}")
    print(f"  V6 optimized pointId dist:  {submission_v6.pointId.value_counts().sort_index().to_dict()}")
    print(f"  V6 optimized SGP dist:      {submission_v6.serverGetPoint.value_counts().to_dict()}")

    print(f"\n  V6 base actionId dist:      {submission_v6_base.actionId.value_counts().sort_index().to_dict()}")
    print(f"  V6 base pointId dist:       {submission_v6_base.pointId.value_counts().sort_index().to_dict()}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Blend weights: CB={w_cb:.2f} XGB={w_xg:.2f} LGB={w_lg:.2f}")
    print(f"  Temperature: actionId={temp_act:.2f}, pointId={temp_pt:.2f}")
    print(f"  Class weights actionId: {np.round(weights_act, 2)}")
    print(f"  Class weights pointId:  {np.round(weights_pt, 2)}")
    print(f"\n  OOF CV scores:")
    print(f"    V5 baseline (argmax): OV={ov_base:.4f}")
    print(f"    V6 optimized:         OV={ov_opt:.4f} (+{ov_opt-ov_base:.4f})")
    print(f"\n  Submissions generated:")
    print(f"    1. {out_v6} (with threshold optimization)")
    print(f"    2. {out_v6_base} (bug-fixed blend, no threshold)")
    print(f"\n  Total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
