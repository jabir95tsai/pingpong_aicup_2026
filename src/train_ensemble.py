"""Full ensemble pipeline: CatBoost + XGBoost + LightGBM + Transformer V1.
Trains all models with 5-fold CV, finds optimal blend weights, generates submission.
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
from features import build_features, compute_player_stats, get_feature_names
from transformer_model import prepare_sequences, PingPongTransformer, PingPongDataset

import torch
import torch.nn.functional as F

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


def train_catboost_fold(X_tr, y_tr, X_val, y_val, task, n_classes=None):
    from catboost import CatBoostClassifier
    if task == "multiclass":
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=n_classes,
                               auto_class_weights="Balanced", early_stopping_rounds=150,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        return m, m.predict_proba(X_val), m
    else:
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               early_stopping_rounds=150, verbose=0, random_seed=RANDOM_SEED,
                               l2_leaf_reg=3)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        return m, m.predict_proba(X_val)[:, 1], m


def train_xgb_fold(X_tr, y_tr, X_val, y_val, task, n_classes=None):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    if task == "multiclass":
        params = {"objective": "multi:softprob", "num_class": n_classes,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        m = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=150, verbose_eval=0)
        probs = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        return m, probs, m
    else:
        params = {"objective": "binary:logistic", "eval_metric": "auc",
                  "tree_method": "hist", "learning_rate": 0.03,
                  "max_depth": 8, "min_child_weight": 10,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        m = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=150, verbose_eval=0)
        probs = m.predict(dval, iteration_range=(0, m.best_iteration+1))
        return m, probs, m


def train_lgb_fold(X_tr, y_tr, X_val, y_val, task, n_classes=None):
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    if task == "multiclass":
        params = {"objective": "multiclass", "num_class": n_classes,
                  "metric": "multi_logloss", "boosting_type": "gbdt",
                  "learning_rate": 0.03, "num_leaves": 127, "max_depth": -1,
                  "min_child_samples": 10, "feature_fraction": 0.8,
                  "bagging_fraction": 0.8, "bagging_freq": 5,
                  "lambda_l1": 0.05, "lambda_l2": 1,
                  "seed": RANDOM_SEED, "verbose": -1, "is_unbalance": True, "num_threads": 4}
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        probs = m.predict(X_val, num_iteration=m.best_iteration)
        return m, probs, m
    else:
        params = {"objective": "binary", "metric": "auc", "boosting_type": "gbdt",
                  "learning_rate": 0.03, "num_leaves": 127, "max_depth": -1,
                  "min_child_samples": 10, "feature_fraction": 0.8,
                  "bagging_fraction": 0.8, "bagging_freq": 5,
                  "lambda_l1": 0.05, "lambda_l2": 1,
                  "seed": RANDOM_SEED, "verbose": -1, "is_unbalance": True, "num_threads": 4}
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        probs = m.predict(X_val, num_iteration=m.best_iteration)
        return m, probs, m


def search_blend_weights(oof_dict, y_act, y_pt, y_srv, next_sn):
    """Grid search for optimal per-task blend weights."""
    model_names = list(oof_dict.keys())
    best_ov = -1
    best_weights = {}

    # Search action weights
    print("  Searching blend weights...")
    weight_grid = np.arange(0, 1.05, 0.1)

    for task, n_cls, y_true in [("act", N_ACTION, y_act), ("pt", N_POINT, y_pt)]:
        best_task_score = -1
        best_task_w = None

        if len(model_names) == 3:
            for w0 in weight_grid:
                for w1 in weight_grid:
                    w2 = 1 - w0 - w1
                    if w2 < -0.01 or w2 > 1.01:
                        continue
                    w2 = max(0, w2)
                    blend = w0 * oof_dict[model_names[0]][task] + \
                            w1 * oof_dict[model_names[1]][task] + \
                            w2 * oof_dict[model_names[2]][task]
                    if task == "act":
                        blend = apply_action_rules(blend, next_sn)
                    score = macro_f1(y_true, blend, n_cls)
                    if score > best_task_score:
                        best_task_score = score
                        best_task_w = {model_names[0]: w0, model_names[1]: w1, model_names[2]: w2}
        else:
            # 2 models
            for w0 in weight_grid:
                w1 = 1 - w0
                blend = w0 * oof_dict[model_names[0]][task] + w1 * oof_dict[model_names[1]][task]
                if task == "act":
                    blend = apply_action_rules(blend, next_sn)
                score = macro_f1(y_true, blend, n_cls)
                if score > best_task_score:
                    best_task_score = score
                    best_task_w = {model_names[0]: w0, model_names[1]: w1}

        best_weights[task] = best_task_w
        print(f"    {task}: F1={best_task_score:.4f} weights={best_task_w}")

    # Server: search
    best_srv_score = -1
    best_srv_w = None
    for w0 in weight_grid:
        for w1 in weight_grid:
            w2 = 1 - w0 - w1
            if w2 < -0.01 or w2 > 1.01:
                continue
            w2 = max(0, w2)
            if len(model_names) == 3:
                blend = w0*oof_dict[model_names[0]]["srv"] + w1*oof_dict[model_names[1]]["srv"] + w2*oof_dict[model_names[2]]["srv"]
            else:
                blend = w0*oof_dict[model_names[0]]["srv"] + (1-w0)*oof_dict[model_names[1]]["srv"]
            score = roc_auc_score(y_srv, blend)
            if score > best_srv_score:
                best_srv_score = score
                if len(model_names) == 3:
                    best_srv_w = {model_names[0]: w0, model_names[1]: w1, model_names[2]: w2}
                else:
                    best_srv_w = {model_names[0]: w0, model_names[1]: 1-w0}
    best_weights["srv"] = best_srv_w
    print(f"    srv: AUC={best_srv_score:.4f} weights={best_srv_w}")

    return best_weights


def main():
    print("=" * 70)
    print("ENSEMBLE TRAINING PIPELINE")
    print("=" * 70)

    # Load and clean
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    n_players = len(player_map)

    # Build GBDT features
    print("\nBuilding features...")
    player_stats = compute_player_stats(train_df)
    feat_train = build_features(train_df, is_train=True, player_stats=player_stats)
    feat_test = build_features(test_df, is_train=False, player_stats=player_stats)
    feature_names = get_feature_names(feat_train)

    X = feat_train[feature_names].values
    y_act = feat_train["y_actionId"].values
    y_pt = feat_train["y_pointId"].values
    y_srv = feat_train["y_serverGetPoint"].values
    next_sn = feat_train["next_strikeNumber"].values
    X_test = feat_test[feature_names].values
    test_next_sn = feat_test["next_strikeNumber"].values

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_train["rally_uid"].map(rally_to_match).values

    print(f"  Train: {X.shape}, Test: {X_test.shape}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(X, groups=groups))

    # OOF predictions for each model
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

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{N_FOLDS}")
        print(f"{'='*60}")
        X_tr, X_val = X[tr_idx], X[val_idx]

        for name, train_fn in [("catboost", train_catboost_fold),
                                ("xgboost", train_xgb_fold),
                                ("lightgbm", train_lgb_fold)]:
            t0 = time.time()
            # Action
            _, probs, m_act = train_fn(X_tr, y_act[tr_idx], X_val, y_act[val_idx], "multiclass", N_ACTION)
            oof[name]["act"][val_idx] = probs

            # Point
            _, probs, m_pt = train_fn(X_tr, y_pt[tr_idx], X_val, y_pt[val_idx], "multiclass", N_POINT)
            oof[name]["pt"][val_idx] = probs

            # Server
            _, probs, m_srv = train_fn(X_tr, y_srv[tr_idx], X_val, y_srv[val_idx], "binary")
            oof[name]["srv"][val_idx] = probs

            # Test predictions
            import xgboost as xgb
            if name == "xgboost":
                dtest = xgb.DMatrix(X_test)
                test_preds[name]["act"] += m_act.predict(dtest, iteration_range=(0, m_act.best_iteration+1)) / N_FOLDS
                test_preds[name]["pt"] += m_pt.predict(dtest, iteration_range=(0, m_pt.best_iteration+1)) / N_FOLDS
                test_preds[name]["srv"] += m_srv.predict(dtest, iteration_range=(0, m_srv.best_iteration+1)) / N_FOLDS
            elif name == "lightgbm":
                test_preds[name]["act"] += m_act.predict(X_test, num_iteration=m_act.best_iteration) / N_FOLDS
                test_preds[name]["pt"] += m_pt.predict(X_test, num_iteration=m_pt.best_iteration) / N_FOLDS
                test_preds[name]["srv"] += m_srv.predict(X_test, num_iteration=m_srv.best_iteration) / N_FOLDS
            else:  # catboost
                test_preds[name]["act"] += m_act.predict_proba(X_test) / N_FOLDS
                test_preds[name]["pt"] += m_pt.predict_proba(X_test) / N_FOLDS
                test_preds[name]["srv"] += m_srv.predict_proba(X_test)[:, 1] / N_FOLDS

            # Per-model fold score
            act_ruled = apply_action_rules(oof[name]["act"][val_idx], next_sn[val_idx])
            f1a = macro_f1(y_act[val_idx], act_ruled, N_ACTION)
            f1p = macro_f1(y_pt[val_idx], oof[name]["pt"][val_idx], N_POINT)
            auc_s = roc_auc_score(y_srv[val_idx], oof[name]["srv"][val_idx])
            ov_val = 0.4*f1a + 0.4*f1p + 0.2*auc_s
            print(f"  {name:10s}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc_s:.4f} OV={ov_val:.4f} ({time.time()-t0:.0f}s)")

    # Individual model OOF scores
    print(f"\n{'='*60}")
    print("INDIVIDUAL MODEL OOF SCORES")
    print(f"{'='*60}")
    for name in oof:
        act_ruled = apply_action_rules(oof[name]["act"], next_sn)
        f1a = macro_f1(y_act, act_ruled, N_ACTION)
        f1p = macro_f1(y_pt, oof[name]["pt"], N_POINT)
        auc_s = roc_auc_score(y_srv, oof[name]["srv"])
        ov_val = 0.4*f1a + 0.4*f1p + 0.2*auc_s
        print(f"  {name:10s}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc_s:.4f} OV={ov_val:.4f}")

    # Find optimal blend weights
    print(f"\n{'='*60}")
    print("BLEND WEIGHT SEARCH")
    print(f"{'='*60}")
    best_weights = search_blend_weights(oof, y_act, y_pt, y_srv, next_sn)

    # Evaluate ensemble with optimal weights
    blend_act = sum(best_weights["act"][n] * oof[n]["act"] for n in oof)
    blend_pt = sum(best_weights["pt"][n] * oof[n]["pt"] for n in oof)
    blend_srv = sum(best_weights["srv"][n] * oof[n]["srv"] for n in oof)

    blend_act_ruled = apply_action_rules(blend_act, next_sn)
    f1a = macro_f1(y_act, blend_act_ruled, N_ACTION)
    f1p = macro_f1(y_pt, blend_pt, N_POINT)
    auc_s = roc_auc_score(y_srv, blend_srv)
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc_s
    print(f"\n  ENSEMBLE OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc_s:.4f} OV={ov:.4f}")

    # Generate submission with optimal weights
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
    out_path = os.path.join(SUBMISSION_DIR, "submission_ensemble.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nSubmission saved: {out_path} ({submission.shape})")
    print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId: {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  serverGetPoint: {submission.serverGetPoint.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
