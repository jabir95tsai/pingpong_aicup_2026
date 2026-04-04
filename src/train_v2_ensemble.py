"""V2 Ensemble: Enhanced features + CatBoost/XGBoost/LightGBM with Optuna tuning.
Target: OV > 0.65
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


def train_catboost(X_tr, y_tr, X_val, y_val, task, n_classes=None, params=None):
    from catboost import CatBoostClassifier
    if params is None:
        params = {}
    if task == "multiclass":
        defaults = dict(iterations=3000, learning_rate=0.03, depth=8,
                        loss_function="MultiClass", classes_count=n_classes,
                        auto_class_weights="Balanced", early_stopping_rounds=200,
                        verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                        border_count=254, random_strength=1)
        defaults.update(params)
        m = CatBoostClassifier(**defaults)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        return m, m.predict_proba(X_val)
    else:
        defaults = dict(iterations=3000, learning_rate=0.03, depth=8,
                        loss_function="Logloss", auto_class_weights="Balanced",
                        early_stopping_rounds=200, verbose=0, random_seed=RANDOM_SEED,
                        l2_leaf_reg=3, border_count=254)
        defaults.update(params)
        m = CatBoostClassifier(**defaults)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        return m, m.predict_proba(X_val)[:, 1]


def train_xgb(X_tr, y_tr, X_val, y_val, task, n_classes=None, params=None):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    if task == "multiclass":
        defaults = {"objective": "multi:softprob", "num_class": n_classes,
                    "eval_metric": "mlogloss", "tree_method": "hist",
                    "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0,
                    "max_bin": 256}
        if params: defaults.update(params)
        m = xgb.train(defaults, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        return m, m.predict(dval, iteration_range=(0, m.best_iteration+1))
    else:
        defaults = {"objective": "binary:logistic", "eval_metric": "auc",
                    "tree_method": "hist", "learning_rate": 0.03,
                    "max_depth": 8, "min_child_weight": 10,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        if params: defaults.update(params)
        m = xgb.train(defaults, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        return m, m.predict(dval, iteration_range=(0, m.best_iteration+1))


def train_lgb(X_tr, y_tr, X_val, y_val, task, n_classes=None, params=None):
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    if task == "multiclass":
        defaults = {"objective": "multiclass", "num_class": n_classes,
                    "metric": "multi_logloss", "boosting_type": "gbdt",
                    "learning_rate": 0.03, "num_leaves": 127, "max_depth": -1,
                    "min_child_samples": 10, "feature_fraction": 0.8,
                    "bagging_fraction": 0.8, "bagging_freq": 5,
                    "lambda_l1": 0.05, "lambda_l2": 1,
                    "seed": RANDOM_SEED, "verbose": -1, "is_unbalance": True, "num_threads": 4}
        if params: defaults.update(params)
        m = lgb.train(defaults, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        return m, m.predict(X_val, num_iteration=m.best_iteration)
    else:
        defaults = {"objective": "binary", "metric": "auc", "boosting_type": "gbdt",
                    "learning_rate": 0.03, "num_leaves": 127, "max_depth": -1,
                    "min_child_samples": 10, "feature_fraction": 0.8,
                    "bagging_fraction": 0.8, "bagging_freq": 5,
                    "lambda_l1": 0.05, "lambda_l2": 1,
                    "seed": RANDOM_SEED, "verbose": -1, "is_unbalance": True, "num_threads": 4}
        if params: defaults.update(params)
        m = lgb.train(defaults, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        return m, m.predict(X_val, num_iteration=m.best_iteration)


def optuna_tune_catboost(X, y_act, y_pt, y_srv, next_sn, groups, n_trials=30):
    """Quick Optuna tuning for CatBoost action model (biggest impact)."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not available, using defaults")
        return {}

    gkf = GroupKFold(n_splits=2)  # Use 2 folds for speed
    splits = list(gkf.split(X, groups=groups))

    def objective(trial):
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 6, 10),
            "l2_leaf_reg": trial.suggest_float("l2", 1, 10, log=True),
            "random_strength": trial.suggest_float("rs", 0.5, 5),
            "bagging_temperature": trial.suggest_float("bt", 0, 2),
            "border_count": trial.suggest_categorical("bc", [128, 254]),
            "loss_function": "MultiClass",
            "classes_count": N_ACTION,
            "auto_class_weights": "Balanced",
            "early_stopping_rounds": 100,
            "verbose": 0,
            "random_seed": RANDOM_SEED,
        }

        scores = []
        for tr_idx, val_idx in splits:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(**params)
            m.fit(X[tr_idx], y_act[tr_idx], eval_set=(X[val_idx], y_act[val_idx]))
            probs = m.predict_proba(X[val_idx])
            probs = apply_action_rules(probs, next_sn[val_idx])
            scores.append(macro_f1(y_act[val_idx], probs, N_ACTION))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Optuna best F1_action: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Convert to CatBoost params
    bp = study.best_params
    return {
        "learning_rate": bp["lr"],
        "depth": bp["depth"],
        "l2_leaf_reg": bp["l2"],
        "random_strength": bp["rs"],
        "bagging_temperature": bp["bt"],
        "border_count": bp["bc"],
    }


def search_blend_weights(oof_dict, y_act, y_pt, y_srv, next_sn):
    """Grid search for optimal per-task blend weights."""
    model_names = list(oof_dict.keys())
    best_weights = {}
    weight_grid = np.arange(0, 1.05, 0.1)

    for task, n_cls, y_true in [("act", N_ACTION, y_act), ("pt", N_POINT, y_pt)]:
        best_score = -1
        best_w = None

        if len(model_names) == 3:
            for w0 in weight_grid:
                for w1 in weight_grid:
                    w2 = 1 - w0 - w1
                    if w2 < -0.01 or w2 > 1.01: continue
                    w2 = max(0, w2)
                    blend = (w0 * oof_dict[model_names[0]][task] +
                             w1 * oof_dict[model_names[1]][task] +
                             w2 * oof_dict[model_names[2]][task])
                    if task == "act":
                        blend = apply_action_rules(blend, next_sn)
                    score = macro_f1(y_true, blend, n_cls)
                    if score > best_score:
                        best_score = score
                        best_w = {model_names[0]: w0, model_names[1]: w1, model_names[2]: w2}
        else:
            for w0 in weight_grid:
                w1 = 1 - w0
                blend = w0 * oof_dict[model_names[0]][task] + w1 * oof_dict[model_names[1]][task]
                if task == "act": blend = apply_action_rules(blend, next_sn)
                score = macro_f1(y_true, blend, n_cls)
                if score > best_score:
                    best_score = score
                    best_w = {model_names[0]: w0, model_names[1]: w1}

        best_weights[task] = best_w
        print(f"    {task}: F1={best_score:.4f} weights={best_w}")

    # Server
    best_score = -1
    best_w = None
    if len(model_names) == 3:
        for w0 in weight_grid:
            for w1 in weight_grid:
                w2 = 1 - w0 - w1
                if w2 < -0.01 or w2 > 1.01: continue
                w2 = max(0, w2)
                blend = (w0*oof_dict[model_names[0]]["srv"] +
                         w1*oof_dict[model_names[1]]["srv"] +
                         w2*oof_dict[model_names[2]]["srv"])
                score = roc_auc_score(y_srv, blend)
                if score > best_score:
                    best_score = score
                    best_w = {model_names[0]: w0, model_names[1]: w1, model_names[2]: w2}
    best_weights["srv"] = best_w
    print(f"    srv: AUC={best_score:.4f} weights={best_w}")

    return best_weights


def main():
    t_start = time.time()
    print("=" * 70)
    print("V2 ENSEMBLE: Enhanced Features + Optuna Tuning")
    print("=" * 70)

    # Load and clean
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)

    # Compute global stats
    print("\nComputing global statistics...")
    t0 = time.time()
    global_stats = compute_global_stats(train_df)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Players: {len(global_stats['player_stats'])}")
    print(f"  Transition matrix shape: {global_stats['action_trans'].shape}")

    # Build V2 features
    print("\nBuilding V2 features...")
    t0 = time.time()
    feat_train = build_features_v2(train_df, is_train=True, global_stats=global_stats)
    feat_test = build_features_v2(test_df, is_train=False, global_stats=global_stats)
    feature_names = get_feature_names_v2(feat_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Train: {feat_train.shape}, Test: {feat_test.shape}")
    print(f"  Features: {len(feature_names)}")

    X = feat_train[feature_names].values.astype(np.float32)
    y_act = feat_train["y_actionId"].values
    y_pt = feat_train["y_pointId"].values
    y_srv = feat_train["y_serverGetPoint"].values
    next_sn = feat_train["next_strikeNumber"].values
    X_test = feat_test[feature_names].values.astype(np.float32)
    test_next_sn = feat_test["next_strikeNumber"].values

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_train["rally_uid"].map(rally_to_match).values

    # Optuna tuning for CatBoost action (biggest impact task)
    print("\n--- Optuna Tuning (CatBoost action) ---")
    t0 = time.time()
    cb_best_params = optuna_tune_catboost(X, y_act, y_pt, y_srv, next_sn, groups, n_trials=10)
    print(f"  Tuning took {time.time()-t0:.0f}s")

    # 5-fold training
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(X, groups=groups))

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

    # Store models for test prediction
    import xgboost as xgb

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{N_FOLDS} (train={len(tr_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")
        X_tr, X_val = X[tr_idx], X[val_idx]

        for name in ["catboost", "xgboost", "lightgbm"]:
            t0 = time.time()

            if name == "catboost":
                train_fn = train_catboost
                extra_params = cb_best_params
            elif name == "xgboost":
                train_fn = train_xgb
                extra_params = None
            else:
                train_fn = train_lgb
                extra_params = None

            # Action
            m_act, probs = train_fn(X_tr, y_act[tr_idx], X_val, y_act[val_idx],
                                     "multiclass", N_ACTION, extra_params)
            oof[name]["act"][val_idx] = probs

            # Point
            m_pt, probs = train_fn(X_tr, y_pt[tr_idx], X_val, y_pt[val_idx],
                                    "multiclass", N_POINT, extra_params if name == "catboost" else None)
            oof[name]["pt"][val_idx] = probs

            # Server
            m_srv, probs = train_fn(X_tr, y_srv[tr_idx], X_val, y_srv[val_idx],
                                     "binary", params=extra_params if name == "catboost" else None)
            oof[name]["srv"][val_idx] = probs

            # Test predictions
            if name == "xgboost":
                dtest = xgb.DMatrix(X_test)
                test_preds[name]["act"] += m_act.predict(dtest, iteration_range=(0, m_act.best_iteration+1)) / N_FOLDS
                test_preds[name]["pt"] += m_pt.predict(dtest, iteration_range=(0, m_pt.best_iteration+1)) / N_FOLDS
                test_preds[name]["srv"] += m_srv.predict(dtest, iteration_range=(0, m_srv.best_iteration+1)) / N_FOLDS
            elif name == "lightgbm":
                test_preds[name]["act"] += m_act.predict(X_test, num_iteration=m_act.best_iteration) / N_FOLDS
                test_preds[name]["pt"] += m_pt.predict(X_test, num_iteration=m_pt.best_iteration) / N_FOLDS
                test_preds[name]["srv"] += m_srv.predict(X_test, num_iteration=m_srv.best_iteration) / N_FOLDS
            else:
                test_preds[name]["act"] += m_act.predict_proba(X_test) / N_FOLDS
                test_preds[name]["pt"] += m_pt.predict_proba(X_test) / N_FOLDS
                test_preds[name]["srv"] += m_srv.predict_proba(X_test)[:, 1] / N_FOLDS

            # Fold score
            act_ruled = apply_action_rules(oof[name]["act"][val_idx], next_sn[val_idx])
            f1a = macro_f1(y_act[val_idx], act_ruled, N_ACTION)
            f1p = macro_f1(y_pt[val_idx], oof[name]["pt"][val_idx], N_POINT)
            auc_s = roc_auc_score(y_srv[val_idx], oof[name]["srv"][val_idx])
            ov_val = 0.4*f1a + 0.4*f1p + 0.2*auc_s
            print(f"  {name:10s}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc_s:.4f} OV={ov_val:.4f} ({time.time()-t0:.0f}s)")

    # Individual OOF scores
    print(f"\n{'='*60}")
    print("INDIVIDUAL MODEL OOF SCORES (V2 Features)")
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
    best_weights = search_blend_weights(oof, y_act, y_pt, y_srv, next_sn)

    # Ensemble evaluation
    blend_act = sum(best_weights["act"][n] * oof[n]["act"] for n in oof)
    blend_pt = sum(best_weights["pt"][n] * oof[n]["pt"] for n in oof)
    blend_srv = sum(best_weights["srv"][n] * oof[n]["srv"] for n in oof)

    blend_act_ruled = apply_action_rules(blend_act, next_sn)
    f1a = macro_f1(y_act, blend_act_ruled, N_ACTION)
    f1p = macro_f1(y_pt, blend_pt, N_POINT)
    auc_s = roc_auc_score(y_srv, blend_srv)
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc_s
    print(f"\n  V2 ENSEMBLE OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc_s:.4f} OV={ov:.4f}")

    # Generate submission
    print("\nGenerating submission...")
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
    out_path = os.path.join(SUBMISSION_DIR, "submission_v2_ensemble.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"Saved: {out_path} ({submission.shape})")
    print(f"  actionId dist: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId dist: {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  serverGetPoint dist: {submission.serverGetPoint.value_counts().to_dict()}")

    # Save global stats for later use
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "global_stats_v2.pkl"), "wb") as f:
        pickle.dump(global_stats, f)

    # Save OOF predictions for later blending with Transformer
    np.savez(os.path.join(MODEL_DIR, "oof_v2.npz"),
             blend_act=blend_act, blend_pt=blend_pt, blend_srv=blend_srv,
             y_act=y_act, y_pt=y_pt, y_srv=y_srv, next_sn=next_sn)
    np.savez(os.path.join(MODEL_DIR, "test_preds_v2.npz"),
             blend_act=blend_test_act, blend_pt=blend_test_pt, blend_srv=blend_test_srv,
             test_next_sn=test_next_sn,
             rally_uids=feat_test["rally_uid"].values.astype(int))

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
