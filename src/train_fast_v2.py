"""Fast V2 Ensemble: Enhanced features, no Optuna, sample weighting by SN distribution."""
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


def compute_sample_weights(next_sn):
    """Weight samples to match test distribution (concentrated at low SN)."""
    # Test distribution: SN=2:398, SN=3:297, SN=4:206, SN=5:119, SN=6:82...
    # Upweight low SN, downweight high SN
    test_sn_dist = {2: 398, 3: 297, 4: 206, 5: 119, 6: 82, 7: 49, 8: 27}
    total_test = sum(test_sn_dist.values())

    weights = np.ones(len(next_sn))
    for i, sn in enumerate(next_sn):
        if sn in test_sn_dist:
            weights[i] = test_sn_dist[sn] / total_test * 5  # upweight test-like SNs
        elif sn <= 1:
            weights[i] = 0.5  # serve position (SN=1)
        else:
            weights[i] = 0.3  # rare high SN
    return weights


def main():
    t_start = time.time()
    print("=" * 70)
    print("FAST V2 ENSEMBLE: Enhanced Features + Sample Weighting")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)

    print("\nComputing global statistics...")
    t0 = time.time()
    global_stats = compute_global_stats(train_df)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\nBuilding V2 features...")
    t0 = time.time()
    feat_train = build_features_v2(train_df, is_train=True, global_stats=global_stats)
    feat_test = build_features_v2(test_df, is_train=False, global_stats=global_stats)
    feature_names = get_feature_names_v2(feat_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Train: {feat_train.shape}, Test: {feat_test.shape}, Features: {len(feature_names)}")

    X = feat_train[feature_names].values.astype(np.float32)
    y_act = feat_train["y_actionId"].values
    y_pt = feat_train["y_pointId"].values
    y_srv = feat_train["y_serverGetPoint"].values
    next_sn = feat_train["next_strikeNumber"].values
    X_test = feat_test[feature_names].values.astype(np.float32)
    test_next_sn = feat_test["next_strikeNumber"].values

    # Sample weights based on test SN distribution
    sample_weights = compute_sample_weights(next_sn)
    print(f"\n  Sample weights range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_train["rally_uid"].map(rally_to_match).values

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(X, groups=groups))

    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    oof = {
        "catboost": {"act": np.zeros((len(X), N_ACTION)), "pt": np.zeros((len(X), N_POINT)), "srv": np.zeros(len(X))},
        "xgboost": {"act": np.zeros((len(X), N_ACTION)), "pt": np.zeros((len(X), N_POINT)), "srv": np.zeros(len(X))},
        "lightgbm": {"act": np.zeros((len(X), N_ACTION)), "pt": np.zeros((len(X), N_POINT)), "srv": np.zeros(len(X))},
    }
    test_preds = {
        "catboost": {"act": np.zeros((len(X_test), N_ACTION)), "pt": np.zeros((len(X_test), N_POINT)), "srv": np.zeros(len(X_test))},
        "xgboost": {"act": np.zeros((len(X_test), N_ACTION)), "pt": np.zeros((len(X_test), N_POINT)), "srv": np.zeros(len(X_test))},
        "lightgbm": {"act": np.zeros((len(X_test), N_ACTION)), "pt": np.zeros((len(X_test), N_POINT)), "srv": np.zeros(len(X_test))},
    }

    dtest_xgb = xgb.DMatrix(X_test)

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{N_FOLDS} (train={len(tr_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")
        X_tr, X_val = X[tr_idx], X[val_idx]
        sw_tr = sample_weights[tr_idx]

        # --- CatBoost ---
        t0 = time.time()
        for task_name, y_all, n_cls in [("act", y_act, N_ACTION), ("pt", y_pt, N_POINT), ("srv", y_srv, None)]:
            if task_name != "srv":
                m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                                       loss_function="MultiClass", classes_count=n_cls,
                                       auto_class_weights="Balanced", early_stopping_rounds=200,
                                       verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
                m.fit(X_tr, y_all[tr_idx], eval_set=(X_val, y_all[val_idx]),
                      sample_weight=sw_tr)
                oof["catboost"][task_name][val_idx] = m.predict_proba(X_val)
                test_preds["catboost"][task_name] += m.predict_proba(X_test) / N_FOLDS
            else:
                m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                                       loss_function="Logloss", auto_class_weights="Balanced",
                                       early_stopping_rounds=200, verbose=0, random_seed=RANDOM_SEED,
                                       l2_leaf_reg=3)
                m.fit(X_tr, y_all[tr_idx], eval_set=(X_val, y_all[val_idx]),
                      sample_weight=sw_tr)
                oof["catboost"]["srv"][val_idx] = m.predict_proba(X_val)[:, 1]
                test_preds["catboost"]["srv"] += m.predict_proba(X_test)[:, 1] / N_FOLDS

        act_ruled = apply_action_rules(oof["catboost"]["act"][val_idx], next_sn[val_idx])
        f1a = macro_f1(y_act[val_idx], act_ruled, N_ACTION)
        f1p = macro_f1(y_pt[val_idx], oof["catboost"]["pt"][val_idx], N_POINT)
        auc_s = roc_auc_score(y_srv[val_idx], oof["catboost"]["srv"][val_idx])
        ov_val = 0.4*f1a + 0.4*f1p + 0.2*auc_s
        print(f"  catboost  : F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc_s:.4f} OV={ov_val:.4f} ({time.time()-t0:.0f}s)")

        # --- XGBoost ---
        t0 = time.time()
        dtrain = xgb.DMatrix(X_tr, label=y_act[tr_idx], weight=sw_tr)
        dval = xgb.DMatrix(X_val, label=y_act[val_idx])
        params = {"objective": "multi:softprob", "num_class": N_ACTION,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        m = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        oof["xgboost"]["act"][val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        test_preds["xgboost"]["act"] += m.predict(dtest_xgb, iteration_range=(0, m.best_iteration+1)) / N_FOLDS

        dtrain = xgb.DMatrix(X_tr, label=y_pt[tr_idx], weight=sw_tr)
        dval = xgb.DMatrix(X_val, label=y_pt[val_idx])
        params["num_class"] = N_POINT
        m = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        oof["xgboost"]["pt"][val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        test_preds["xgboost"]["pt"] += m.predict(dtest_xgb, iteration_range=(0, m.best_iteration+1)) / N_FOLDS

        dtrain = xgb.DMatrix(X_tr, label=y_srv[tr_idx], weight=sw_tr)
        dval = xgb.DMatrix(X_val, label=y_srv[val_idx])
        params_bin = {"objective": "binary:logistic", "eval_metric": "auc",
                      "tree_method": "hist", "learning_rate": 0.03,
                      "max_depth": 8, "min_child_weight": 10,
                      "subsample": 0.8, "colsample_bytree": 0.8,
                      "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        m = xgb.train(params_bin, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        oof["xgboost"]["srv"][val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        test_preds["xgboost"]["srv"] += m.predict(dtest_xgb, iteration_range=(0, m.best_iteration+1)) / N_FOLDS

        act_ruled = apply_action_rules(oof["xgboost"]["act"][val_idx], next_sn[val_idx])
        f1a = macro_f1(y_act[val_idx], act_ruled, N_ACTION)
        f1p = macro_f1(y_pt[val_idx], oof["xgboost"]["pt"][val_idx], N_POINT)
        auc_s = roc_auc_score(y_srv[val_idx], oof["xgboost"]["srv"][val_idx])
        ov_val = 0.4*f1a + 0.4*f1p + 0.2*auc_s
        print(f"  xgboost   : F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc_s:.4f} OV={ov_val:.4f} ({time.time()-t0:.0f}s)")

        # --- LightGBM ---
        t0 = time.time()
        for task_name, y_all, n_cls in [("act", y_act, N_ACTION), ("pt", y_pt, N_POINT), ("srv", y_srv, None)]:
            dtrain = lgb.Dataset(X_tr, label=y_all[tr_idx], weight=sw_tr)
            dval_lgb = lgb.Dataset(X_val, label=y_all[val_idx], reference=dtrain)
            if task_name != "srv":
                params = {"objective": "multiclass", "num_class": n_cls,
                          "metric": "multi_logloss", "boosting_type": "gbdt",
                          "learning_rate": 0.03, "num_leaves": 127, "max_depth": -1,
                          "min_child_samples": 10, "feature_fraction": 0.8,
                          "bagging_fraction": 0.8, "bagging_freq": 5,
                          "lambda_l1": 0.05, "lambda_l2": 1,
                          "seed": RANDOM_SEED, "verbose": -1, "is_unbalance": True, "num_threads": 4}
                m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval_lgb],
                              callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
                oof["lightgbm"][task_name][val_idx] = m.predict(X_val, num_iteration=m.best_iteration)
                test_preds["lightgbm"][task_name] += m.predict(X_test, num_iteration=m.best_iteration) / N_FOLDS
            else:
                params = {"objective": "binary", "metric": "auc", "boosting_type": "gbdt",
                          "learning_rate": 0.03, "num_leaves": 127, "max_depth": -1,
                          "min_child_samples": 10, "feature_fraction": 0.8,
                          "bagging_fraction": 0.8, "bagging_freq": 5,
                          "lambda_l1": 0.05, "lambda_l2": 1,
                          "seed": RANDOM_SEED, "verbose": -1, "is_unbalance": True, "num_threads": 4}
                m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval_lgb],
                              callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
                oof["lightgbm"]["srv"][val_idx] = m.predict(X_val, num_iteration=m.best_iteration)
                test_preds["lightgbm"]["srv"] += m.predict(X_test, num_iteration=m.best_iteration) / N_FOLDS

        act_ruled = apply_action_rules(oof["lightgbm"]["act"][val_idx], next_sn[val_idx])
        f1a = macro_f1(y_act[val_idx], act_ruled, N_ACTION)
        f1p = macro_f1(y_pt[val_idx], oof["lightgbm"]["pt"][val_idx], N_POINT)
        auc_s = roc_auc_score(y_srv[val_idx], oof["lightgbm"]["srv"][val_idx])
        ov_val = 0.4*f1a + 0.4*f1p + 0.2*auc_s
        print(f"  lightgbm  : F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc_s:.4f} OV={ov_val:.4f} ({time.time()-t0:.0f}s)")

    # Individual OOF
    print(f"\n{'='*60}")
    print("INDIVIDUAL OOF SCORES (V2 features + sample weights)")
    print(f"{'='*60}")
    for name in oof:
        act_ruled = apply_action_rules(oof[name]["act"], next_sn)
        f1a = macro_f1(y_act, act_ruled, N_ACTION)
        f1p = macro_f1(y_pt, oof[name]["pt"], N_POINT)
        auc_s = roc_auc_score(y_srv, oof[name]["srv"])
        ov_val = 0.4*f1a + 0.4*f1p + 0.2*auc_s
        print(f"  {name:10s}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc_s:.4f} OV={ov_val:.4f}")

    # Blend weight search
    print(f"\n{'='*60}")
    print("BLEND WEIGHT SEARCH")
    print(f"{'='*60}")

    model_names = list(oof.keys())
    best_weights = {}
    weight_grid = np.arange(0, 1.05, 0.1)

    for task, n_cls, y_true in [("act", N_ACTION, y_act), ("pt", N_POINT, y_pt)]:
        best_score = -1
        best_w = None
        for w0 in weight_grid:
            for w1 in weight_grid:
                w2 = 1 - w0 - w1
                if w2 < -0.01 or w2 > 1.01: continue
                w2 = max(0, w2)
                blend = (w0 * oof[model_names[0]][task] +
                         w1 * oof[model_names[1]][task] +
                         w2 * oof[model_names[2]][task])
                if task == "act": blend = apply_action_rules(blend, next_sn)
                score = macro_f1(y_true, blend, n_cls)
                if score > best_score:
                    best_score = score
                    best_w = {model_names[0]: w0, model_names[1]: w1, model_names[2]: w2}
        best_weights[task] = best_w
        print(f"    {task}: F1={best_score:.4f} weights={best_w}")

    best_score = -1
    best_w = None
    for w0 in weight_grid:
        for w1 in weight_grid:
            w2 = 1 - w0 - w1
            if w2 < -0.01 or w2 > 1.01: continue
            w2 = max(0, w2)
            blend = (w0*oof[model_names[0]]["srv"] + w1*oof[model_names[1]]["srv"] +
                     w2*oof[model_names[2]]["srv"])
            score = roc_auc_score(y_srv, blend)
            if score > best_score:
                best_score = score
                best_w = {model_names[0]: w0, model_names[1]: w1, model_names[2]: w2}
    best_weights["srv"] = best_w
    print(f"    srv: AUC={best_score:.4f} weights={best_w}")

    # Final ensemble
    blend_act = sum(best_weights["act"][n] * oof[n]["act"] for n in oof)
    blend_pt = sum(best_weights["pt"][n] * oof[n]["pt"] for n in oof)
    blend_srv = sum(best_weights["srv"][n] * oof[n]["srv"] for n in oof)

    blend_act_ruled = apply_action_rules(blend_act, next_sn)
    f1a = macro_f1(y_act, blend_act_ruled, N_ACTION)
    f1p = macro_f1(y_pt, blend_pt, N_POINT)
    auc_s = roc_auc_score(y_srv, blend_srv)
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc_s
    print(f"\n  V2 ENSEMBLE OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc_s:.4f} OV={ov:.4f}")

    # Submission
    blend_test_act = sum(best_weights["act"].get(n, 0) * test_preds[n]["act"] for n in test_preds)
    blend_test_pt = sum(best_weights["pt"].get(n, 0) * test_preds[n]["pt"] for n in test_preds)
    blend_test_srv = sum(best_weights["srv"].get(n, 0) * test_preds[n]["srv"] for n in test_preds)
    blend_test_act = apply_action_rules(blend_test_act, test_next_sn)

    submission = pd.DataFrame({
        "rally_uid": feat_test["rally_uid"].values.astype(int),
        "actionId": np.argmax(blend_test_act, axis=1).astype(int),
        "pointId": np.argmax(blend_test_pt, axis=1).astype(int),
        "serverGetPoint": (blend_test_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_v2_fast.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nSaved: {out_path} ({submission.shape})")
    print(f"  actionId dist: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId dist: {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  serverGetPoint: {submission.serverGetPoint.value_counts().to_dict()}")

    # Also save OOF/test for later blending
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(os.path.join(MODEL_DIR, "oof_v2_fast.npz"),
             **{f"{n}_{t}": oof[n][t] for n in oof for t in ["act", "pt", "srv"]},
             y_act=y_act, y_pt=y_pt, y_srv=y_srv, next_sn=next_sn)
    np.savez(os.path.join(MODEL_DIR, "test_v2_fast.npz"),
             **{f"{n}_{t}": test_preds[n][t] for n in test_preds for t in ["act", "pt", "srv"]},
             test_next_sn=test_next_sn,
             rally_uids=feat_test["rally_uid"].values.astype(int))

    with open(os.path.join(MODEL_DIR, "global_stats_v2.pkl"), "wb") as f:
        pickle.dump(global_stats, f)

    print(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
