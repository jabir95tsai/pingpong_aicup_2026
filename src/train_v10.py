"""V10 Champion Pipeline — Two-Pass Action→Point Stacking + Wider Temporal Context

Key improvements over V9:
1. Features V6: extra one-hot lag steps (4,6,8,10) — ~204 additional features
2. TWO-PASS architecture:
   - Pass A: Train action models (LGB+XGB+CB) in each fold
   - Pass B: Append action probability predictions (15-dim) as extra features
             for the point models (binary miss + 10-class)
   - This exploits the strong action→point physical correlation:
     ShortStop→short zone, Loop/Smash→long zone, etc.
3. Action OOF probs also added as extra features for server prediction
4. Same threshold optimisation pipeline (temperature → greedy → scipy)
"""
import sys, os, time, warnings, gc, argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data

N_ACTION       = 19        # full action probability width (0-18)
N_ACTION_TRAIN = 15        # classes 0-14 only appear as next-shot targets
N_POINT        = 10

ACTION_EVAL_LABELS = list(range(15))   # 15-class macro F1 for action
POINT_EVAL_LABELS  = list(range(10))

# ─── Class weight maps ────────────────────────────────────────────────────────
ACTION_CW = {
    0: 1.5, 1: 0.6, 2: 0.9, 3: 1.5, 4: 1.2, 5: 1.0,
    6: 0.8, 7: 1.8, 8: 14.0, 9: 8.0, 10: 0.6, 11: 1.2,
    12: 0.9, 13: 0.7, 14: 10.0,
    15: 0.01, 16: 0.01, 17: 0.01, 18: 0.01,
}
POINT_CW = {
    0: 0.5, 1: 12.0, 2: 2.5, 3: 22.0, 4: 2.0,
    5: 0.9, 6: 1.5, 7: 0.8, 8: 0.7, 9: 0.6,
}
POINT_FLIP = {1: 3, 3: 1, 4: 6, 6: 4, 7: 9, 9: 7}


# ─── Metric helpers ──────────────────────────────────────────────────────────

def action_macro_f1(y_true, probs):
    return f1_score(y_true, np.argmax(probs, axis=1),
                    labels=ACTION_EVAL_LABELS, average="macro", zero_division=0)

def point_macro_f1(y_true, probs):
    return f1_score(y_true, np.argmax(probs, axis=1),
                    labels=POINT_EVAL_LABELS, average="macro", zero_division=0)

def apply_action_rules(probs, next_sns):
    out = probs.copy()
    serve_mask  = (next_sns == 1)
    non_serve   = ~serve_mask
    out[serve_mask, :15] = 0.0
    for c in [15, 16, 17, 18]:
        if c < out.shape[1]:
            out[non_serve, c] = 0.0
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return out / row_sums


# ─── Augmentation ────────────────────────────────────────────────────────────

def build_flip_map(feature_names):
    fn_idx = {n: i for i, n in enumerate(feature_names)}
    pairs  = []
    for k in [1, 2, 3, 4, 5, 6, 8, 10]:   # cover all V6 lags
        for (a, b) in [
            (f"oh_lag{k}_handId_1",     f"oh_lag{k}_handId_2"),
            (f"oh_lag{k}_positionId_1", f"oh_lag{k}_positionId_3"),
            (f"oh_lag{k}_pointId_1",    f"oh_lag{k}_pointId_3"),
            (f"oh_lag{k}_pointId_4",    f"oh_lag{k}_pointId_6"),
            (f"oh_lag{k}_pointId_7",    f"oh_lag{k}_pointId_9"),
        ]:
            if a in fn_idx and b in fn_idx:
                pairs.append((fn_idx[a], fn_idx[b]))
    return pairs


def augment_flip(X, y_act, y_pt, y_srv, flip_pairs):
    if not flip_pairs:
        return X, y_act, y_pt, y_srv
    X_flip = X.copy()
    for (ia, ib) in flip_pairs:
        X_flip[:, ia], X_flip[:, ib] = X[:, ib].copy(), X[:, ia].copy()
    y_pt_flip = np.array([POINT_FLIP.get(int(v), int(v)) for v in y_pt])
    return (np.vstack([X, X_flip]),
            np.concatenate([y_act, y_act]),
            np.concatenate([y_pt, y_pt_flip]),
            np.concatenate([y_srv, y_srv]))


# ─── Threshold optimisation ───────────────────────────────────────────────────

def optimize_thresholds(probs, y_true, eval_labels, init_cw_dict=None, n_classes=10):
    best_t, best_f1 = 1.0, -1.0
    for t in np.arange(0.2, 3.5, 0.1):
        scaled = probs ** (1.0 / t)
        scaled /= scaled.sum(axis=1, keepdims=True)
        s = f1_score(y_true, np.argmax(scaled, axis=1), labels=eval_labels,
                     average="macro", zero_division=0)
        if s > best_f1:
            best_f1 = s; best_t = t
    probs_t = probs ** (1.0 / best_t)
    probs_t /= probs_t.sum(axis=1, keepdims=True)
    print(f"    Temp={best_t:.1f} -> F1={best_f1:.4f}")

    if init_cw_dict is not None:
        w = np.array([init_cw_dict.get(c, 1.0) for c in range(n_classes)])
    else:
        w = np.ones(n_classes)
    cur_f1 = f1_score(y_true, np.argmax(probs_t * w, axis=1), labels=eval_labels,
                      average="macro", zero_division=0)
    for c in range(n_classes):
        best_wc, best_local = w[c], cur_f1
        for wc in np.concatenate([np.arange(0.05, 1.0, 0.1),
                                   np.arange(1.0, 40.0, 1.0)]):
            trial = w.copy(); trial[c] = wc
            f = f1_score(y_true, np.argmax(probs_t * trial, axis=1),
                         labels=eval_labels, average="macro", zero_division=0)
            if f > best_local:
                best_local = f; best_wc = wc
        w[c] = best_wc; cur_f1 = best_local
    print(f"    Greedy -> F1={cur_f1:.4f}")

    def neg_f1(log_w):
        ww = np.exp(np.clip(log_w, -5, 5))
        return -f1_score(y_true, np.argmax(probs_t * ww, axis=1),
                         labels=eval_labels, average="macro", zero_division=0)
    try:
        res = minimize(neg_f1, np.log(np.clip(w, 0.01, 100)),
                       method="Nelder-Mead",
                       options={"maxiter": 8000, "xatol": 1e-4, "fatol": 1e-4})
        w_sp = np.exp(np.clip(res.x, -5, 5))
        f_sp = -res.fun
        if f_sp > cur_f1:
            print(f"    Scipy -> F1={f_sp:.4f} (improved)")
            w = w_sp; cur_f1 = f_sp
        else:
            print(f"    Scipy -> F1={f_sp:.4f} (no improve, keeping greedy)")
    except Exception as e:
        print(f"    Scipy failed: {e}")

    return best_t, w, cur_f1


# ─── Two-stage pointId ────────────────────────────────────────────────────────

def blend_two_stage(probs_10, miss_prob, alpha=0.4):
    out = probs_10.copy()
    out[:, 0] = alpha * miss_prob + (1 - alpha) * out[:, 0]
    out /= out.sum(axis=1, keepdims=True)
    return out


# ─── Helpers ─────────────────────────────────────────────────────────────────

def pad_proba(probs, model_classes, n_classes):
    if probs.shape[1] == n_classes:
        return probs
    out = np.zeros((probs.shape[0], n_classes), dtype=np.float32)
    for col_idx, cls_label in enumerate(model_classes):
        out[:, int(cls_label)] = probs[:, col_idx]
    return out


def extend_action(p):
    """Pad N_ACTION_TRAIN-dim action probs to N_ACTION=19 (cols 15-18 → 0)."""
    out = np.zeros((p.shape[0], N_ACTION), dtype=np.float32)
    out[:, :N_ACTION_TRAIN] = p
    return out


def extract_Xy(feat_df, fnames):
    X   = feat_df[fnames].values.astype(np.float32)
    y_a = feat_df["y_actionId"].values.astype(np.int32)
    y_p = feat_df["y_pointId"].values.astype(np.int32)
    y_s = feat_df["y_serverGetPoint"].values.astype(np.int32)
    nsn = feat_df["next_strikeNumber"].values.astype(np.int32)
    y_a = np.where(y_a >= N_ACTION_TRAIN, 0, y_a)
    return X, y_a, y_p, y_s, nsn


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",    action="store_true")
    parser.add_argument("--folds",    type=int, default=N_FOLDS)
    parser.add_argument("--no-aug",   action="store_true")
    parser.add_argument("--no-stack", action="store_true",
                        help="Disable two-pass action->point stacking")
    args = parser.parse_args()

    is_smoke    = args.smoke
    n_folds     = 1 if is_smoke else args.folds
    n_boost     = 200 if is_smoke else 3000
    es_rounds   = 30  if is_smoke else 200
    use_aug     = not args.no_aug
    use_stack   = not args.no_stack

    t_start = time.time()
    print("=" * 70)
    print(f"V10 CHAMPION PIPELINE {'(SMOKE)' if is_smoke else ''}")
    print(f"  aug={use_aug}  folds={n_folds}  n_boost={n_boost}  stack={use_stack}")
    print(f"  ACTION macro: 15 classes (0-14, excluding serve 15-18)")
    print("=" * 70)

    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    from features_v6 import (compute_global_stats_v6, build_features_v6,
                              get_feature_names_v6)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(TEST_PATH)
    train_df, test_df, _ = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1

    # ── Preflight ────────────────────────────────────────────────────────────
    print("\n--- Preflight ---")
    t0 = time.time()
    gs_full   = compute_global_stats_v6(train_df)
    feat_full = build_features_v6(train_df, is_train=True,
                                   global_stats_v6=gs_full,
                                   raw_df=train_df)
    fnames    = get_feature_names_v6(feat_full)
    n_samples = len(feat_full)
    print(f"  {len(fnames)} features, {n_samples} samples ({time.time()-t0:.1f}s)")

    flip_pairs = build_flip_map(fnames)
    print(f"  Flip pairs: {len(flip_pairs)}")

    X_all, y_a_all, y_p_all, y_s_all, nsn_all = extract_Xy(feat_full, fnames)
    rally_uids_all = feat_full["rally_uid"].values
    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_all = np.array([rally_to_match.get(r, -1) for r in rally_uids_all])

    # OOF containers (original samples only)
    oof_act    = np.zeros((n_samples, N_ACTION))
    oof_pt     = np.zeros((n_samples, N_POINT))
    oof_srv    = np.zeros(n_samples)
    oof_pt_bin = np.zeros(n_samples)

    gkf    = GroupKFold(n_splits=max(n_folds, 2))
    splits = list(gkf.split(np.arange(n_samples), groups=match_all))
    if is_smoke:
        splits = splits[:1]

    # Test feature matrix (ONE row per test rally)
    feat_test  = build_features_v6(test_df, is_train=False,
                                    global_stats_v6=gs_full,
                                    raw_df=test_df)
    X_test     = feat_test[fnames].values.astype(np.float32)
    nsn_test   = feat_test["next_strikeNumber"].values.astype(np.int32)
    rally_test = feat_test["rally_uid"].values

    test_act_acc = np.zeros((len(X_test), N_ACTION))
    test_pt_acc  = np.zeros((len(X_test), N_POINT))
    test_srv_acc = np.zeros(len(X_test))
    test_bin_acc = np.zeros(len(X_test))

    # Accumulator for test action probs (used as stacking feature for point model)
    test_act15_acc = np.zeros((len(X_test), N_ACTION_TRAIN))

    # ── Fold loop ────────────────────────────────────────────────────────────
    for fold, (tr_idx, val_idx) in enumerate(splits):
        t_fold = time.time()
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{len(splits)}")
        print(f"{'='*60}")

        tr_rallies  = set(rally_uids_all[tr_idx])
        val_rallies = set(rally_uids_all[val_idx])
        tr_raw  = train_df[train_df["rally_uid"].isin(tr_rallies)]
        val_raw = train_df[train_df["rally_uid"].isin(val_rallies)]

        fold_stats = compute_global_stats_v6(tr_raw)
        feat_tr  = build_features_v6(tr_raw, is_train=True,
                                      global_stats_v6=fold_stats,
                                      raw_df=tr_raw)
        feat_val = build_features_v6(val_raw, is_train=True,
                                      global_stats_v6=fold_stats,
                                      raw_df=tr_raw)   # lag lookup uses tr for test-like setup

        X_tr, y_a_tr, y_p_tr, y_s_tr, nsn_tr     = extract_Xy(feat_tr,  fnames)
        X_val, y_a_val, y_p_val, y_s_val, nsn_val = extract_Xy(feat_val, fnames)

        # ── Augmentation ─────────────────────────────────────────────────────
        if use_aug:
            X_tr_aug, y_a_aug, y_p_aug, y_s_aug = augment_flip(
                X_tr, y_a_tr, y_p_tr, y_s_tr, flip_pairs)
            print(f"  Augmented: {len(X_tr)} -> {len(X_tr_aug)} samples")
        else:
            X_tr_aug, y_a_aug, y_p_aug, y_s_aug = X_tr, y_a_tr, y_p_tr, y_s_tr

        sw_a = np.array([ACTION_CW.get(int(c), 1.0) for c in y_a_aug], dtype=np.float32)
        sw_p = np.array([POINT_CW.get(int(c),  1.0) for c in y_p_aug], dtype=np.float32)

        # ══════════════════════════════════════════════════════════════════════
        # PASS A — ACTION models (LGB + XGB + CB)
        # ══════════════════════════════════════════════════════════════════════
        lgb_a_p = dict(n_estimators=n_boost, learning_rate=0.04,
                       num_leaves=127, max_depth=9, min_child_samples=8,
                       subsample=0.8, colsample_bytree=0.7,
                       reg_alpha=0.1, reg_lambda=1.0,
                       objective="multiclass", metric="multi_logloss",
                       num_class=N_ACTION_TRAIN, random_state=RANDOM_SEED,
                       n_jobs=-1, verbose=-1)
        lgb_a = lgb.train(lgb_a_p,
            lgb.Dataset(X_tr_aug, label=y_a_aug, weight=sw_a),
            valid_sets=[lgb.Dataset(X_val, label=y_a_val)],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])

        xgb_a = xgb.XGBClassifier(
            n_estimators=n_boost, learning_rate=0.04, max_depth=7,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            objective="multi:softprob", num_class=N_ACTION_TRAIN,
            eval_metric="mlogloss", early_stopping_rounds=es_rounds,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0, tree_method="hist")
        xgb_a.fit(X_tr_aug, y_a_aug, sample_weight=sw_a,
                  eval_set=[(X_val, y_a_val)], verbose=False)

        cb_a = CatBoostClassifier(
            iterations=n_boost, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
            loss_function="MultiClass", classes_count=N_ACTION_TRAIN,
            random_seed=RANDOM_SEED, verbose=False, allow_writing_files=False,
            early_stopping_rounds=es_rounds)
        cb_a.fit(X_tr_aug, y_a_aug, sample_weight=sw_a,
                 eval_set=(X_val, y_a_val), use_best_model=True)

        # Action probabilities (15-dim for pass A output)
        pa_val_lgb = lgb_a.predict(X_val)                                              # (n_val, 15)
        pa_val_xgb = pad_proba(xgb_a.predict_proba(X_val), xgb_a.classes_, N_ACTION_TRAIN)
        pa_val_cb  = pad_proba(cb_a.predict_proba(X_val),  cb_a.classes_,  N_ACTION_TRAIN)
        pa_val_15  = (pa_val_lgb + pa_val_xgb + pa_val_cb) / 3.0  # (n_val, 15)

        # Extend to 19-dim for OOF storage
        pa_val_19 = extend_action(pa_val_15)
        pa_ruled  = apply_action_rules(pa_val_19, nsn_val)
        f1_a_val  = action_macro_f1(y_a_val, pa_ruled)
        print(f"  [Action] F1={f1_a_val:.4f}")

        val_mask = np.isin(rally_uids_all, list(set(rally_uids_all[val_idx])))
        oof_act[val_mask] = pa_ruled

        # Action probs for TRAINING rows (used as stacking features for point model)
        # Note: slight leakage (model predicts on its own training data), but standard
        # practice for within-fold stacking — point model still trained on separate labels.
        if use_stack:
            pa_tr_lgb = lgb_a.predict(X_tr)
            pa_tr_xgb = pad_proba(xgb_a.predict_proba(X_tr), xgb_a.classes_, N_ACTION_TRAIN)
            pa_tr_cb  = pad_proba(cb_a.predict_proba(X_tr),  cb_a.classes_,  N_ACTION_TRAIN)
            pa_tr_15  = (pa_tr_lgb + pa_tr_xgb + pa_tr_cb) / 3.0  # (n_tr, 15)

            # For the augmented training set, the flip doesn't change action class
            pa_tr_aug_15 = np.vstack([pa_tr_15, pa_tr_15])  # duplicate for flipped half

        # Test action probs (accumulate for stacking in all folds)
        pa_test_lgb = lgb_a.predict(X_test)
        pa_test_xgb = pad_proba(xgb_a.predict_proba(X_test), xgb_a.classes_, N_ACTION_TRAIN)
        pa_test_cb  = pad_proba(cb_a.predict_proba(X_test),  cb_a.classes_,  N_ACTION_TRAIN)
        pa_test_15  = (pa_test_lgb + pa_test_xgb + pa_test_cb) / 3.0
        test_act15_acc += pa_test_15 / len(splits)

        # Save action probs for test submission (full 19-dim)
        test_act_acc += extend_action(pa_test_15) / len(splits)

        # ══════════════════════════════════════════════════════════════════════
        # PASS B — POINT models (binary miss + 10-class LGB/XGB/CB)
        #          with action probs as extra stacking features
        # ══════════════════════════════════════════════════════════════════════
        if use_stack:
            # Extend feature matrices with action probs (15 extra cols each)
            X_tr_ext      = np.hstack([X_tr,     pa_tr_15])       # (n_tr,   F+15)
            X_tr_aug_ext  = np.hstack([X_tr_aug, pa_tr_aug_15])   # (n_aug,  F+15)
            X_val_ext     = np.hstack([X_val,    pa_val_15])       # (n_val,  F+15)
        else:
            X_tr_ext      = X_tr
            X_tr_aug_ext  = X_tr_aug
            X_val_ext     = X_val

        y_miss_aug = (y_p_aug == 0).astype(np.int32)
        y_miss_val = (y_p_val == 0).astype(np.int32)

        # ── POINT binary (miss vs non-miss) ───────────────────────────────────
        lgb_pb_p = dict(n_estimators=n_boost, learning_rate=0.04,
                        num_leaves=63, max_depth=7, min_child_samples=10,
                        subsample=0.8, colsample_bytree=0.7,
                        objective="binary", metric="binary_logloss",
                        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        lgb_pb = lgb.train(lgb_pb_p,
            lgb.Dataset(X_tr_aug_ext, label=y_miss_aug.astype(np.float32)),
            valid_sets=[lgb.Dataset(X_val_ext, label=y_miss_val.astype(np.float32))],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        pb_val = lgb_pb.predict(X_val_ext)
        oof_pt_bin[val_mask] = pb_val

        # ── POINT 10-class (LGB + XGB + CB) ──────────────────────────────────
        lgb_p_p = dict(n_estimators=n_boost, learning_rate=0.04,
                       num_leaves=127, max_depth=9, min_child_samples=5,
                       subsample=0.8, colsample_bytree=0.7,
                       reg_alpha=0.1, reg_lambda=1.0,
                       objective="multiclass", metric="multi_logloss",
                       num_class=N_POINT, random_state=RANDOM_SEED,
                       n_jobs=-1, verbose=-1)
        lgb_p = lgb.train(lgb_p_p,
            lgb.Dataset(X_tr_aug_ext, label=y_p_aug, weight=sw_p),
            valid_sets=[lgb.Dataset(X_val_ext, label=y_p_val)],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        pp_lgb = lgb_p.predict(X_val_ext)

        xgb_p = xgb.XGBClassifier(
            n_estimators=n_boost, learning_rate=0.04, max_depth=7,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            objective="multi:softprob", num_class=N_POINT,
            eval_metric="mlogloss", early_stopping_rounds=es_rounds,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0, tree_method="hist")
        xgb_p.fit(X_tr_aug_ext, y_p_aug, sample_weight=sw_p,
                  eval_set=[(X_val_ext, y_p_val)], verbose=False)
        pp_xgb = xgb_p.predict_proba(X_val_ext)

        cb_p = CatBoostClassifier(
            iterations=n_boost, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
            loss_function="MultiClass", classes_count=N_POINT,
            random_seed=RANDOM_SEED, verbose=False, allow_writing_files=False,
            early_stopping_rounds=es_rounds)
        cb_p.fit(X_tr_aug_ext, y_p_aug, sample_weight=sw_p,
                 eval_set=(X_val_ext, y_p_val), use_best_model=True)
        pp_cb = cb_p.predict_proba(X_val_ext)

        pp_xgb  = pad_proba(pp_xgb, xgb_p.classes_, N_POINT)
        pp_cb   = pad_proba(pp_cb,  cb_p.classes_,  N_POINT)
        pp_blend = (pp_lgb + pp_xgb + pp_cb) / 3.0
        pp_2stage = blend_two_stage(pp_blend, pb_val)
        f1_p_val  = point_macro_f1(y_p_val, pp_2stage)
        print(f"  [Point]  F1={f1_p_val:.4f}")
        oof_pt[val_mask] = pp_2stage

        # Test point predictions: use fold-averaged action probs as stacking features
        if use_stack:
            X_test_ext = np.hstack([X_test, test_act15_acc * len(splits) / (fold + 1)])
        else:
            X_test_ext = X_test

        test_pt_acc  += (lgb_p.predict(X_test_ext) +
                         pad_proba(xgb_p.predict_proba(X_test_ext), xgb_p.classes_, N_POINT) +
                         pad_proba(cb_p.predict_proba(X_test_ext),  cb_p.classes_,  N_POINT)
                         ) / 3.0 / len(splits)
        test_bin_acc += lgb_pb.predict(X_test_ext) / len(splits)

        # ── SERVER (LGB + XGB) — optionally with action stacking features ──────
        if use_stack:
            X_tr_srv_ext  = np.hstack([X_tr_aug, pa_tr_aug_15])
            X_val_srv_ext = np.hstack([X_val,    pa_val_15])
        else:
            X_tr_srv_ext  = X_tr_aug
            X_val_srv_ext = X_val

        lgb_s_p = dict(n_estimators=n_boost, learning_rate=0.04,
                       num_leaves=63, max_depth=7, min_child_samples=15,
                       subsample=0.8, colsample_bytree=0.7,
                       objective="binary", metric="auc",
                       random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        lgb_s = lgb.train(lgb_s_p,
            lgb.Dataset(X_tr_srv_ext, label=y_s_aug.astype(np.float32)),
            valid_sets=[lgb.Dataset(X_val_srv_ext, label=y_s_val.astype(np.float32))],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        ps_lgb = lgb_s.predict(X_val_srv_ext)

        xgb_s = xgb.XGBClassifier(
            n_estimators=n_boost, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.7,
            objective="binary:logistic", eval_metric="auc",
            early_stopping_rounds=es_rounds,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0, tree_method="hist")
        xgb_s.fit(X_tr_srv_ext, y_s_aug, eval_set=[(X_val_srv_ext, y_s_val)], verbose=False)
        ps_xgb = xgb_s.predict_proba(X_val_srv_ext)[:, 1]

        ps_blend = (ps_lgb + ps_xgb) / 2.0
        auc_val  = roc_auc_score(y_s_val, ps_blend)
        print(f"  [Server] AUC={auc_val:.4f}")
        oof_srv[val_mask] = ps_blend

        ov_fold = 0.4*f1_a_val + 0.4*f1_p_val + 0.2*auc_val
        print(f"\n  FOLD OV={ov_fold:.4f}  [{time.time()-t_fold:.0f}s]")

        # Test server accumulation
        if use_stack:
            X_test_srv_ext = np.hstack([X_test, test_act15_acc * len(splits) / (fold + 1)])
        else:
            X_test_srv_ext = X_test
        test_srv_acc += (lgb_s.predict(X_test_srv_ext) +
                         xgb_s.predict_proba(X_test_srv_ext)[:, 1]) / 2.0 / len(splits)

        gc.collect()

    # ─── Global OOF evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GLOBAL OOF RESULTS")

    oof_mask = oof_act.sum(axis=1) > 0
    n_oof    = oof_mask.sum()
    print(f"  OOF samples: {n_oof}/{n_samples} ({100*n_oof/n_samples:.0f}%)")

    oof_act_ruled = apply_action_rules(oof_act[oof_mask], nsn_all[oof_mask])
    f1_a_oof  = action_macro_f1(y_a_all[oof_mask], oof_act_ruled)
    f1_p_oof  = point_macro_f1(y_p_all[oof_mask], oof_pt[oof_mask])
    auc_oof   = roc_auc_score(y_s_all[oof_mask], oof_srv[oof_mask])
    ov_oof    = 0.4*f1_a_oof + 0.4*f1_p_oof + 0.2*auc_oof
    print(f"  Base:  action={f1_a_oof:.4f}  point={f1_p_oof:.4f}  AUC={auc_oof:.4f}  OV={ov_oof:.4f}")

    # Per-class breakdown
    print("\n  PointId per-class F1:")
    pp_pred = np.argmax(oof_pt[oof_mask], axis=1)
    pf1s    = f1_score(y_p_all[oof_mask], pp_pred, labels=POINT_EVAL_LABELS,
                       average=None, zero_division=0)
    zone_names = ["miss","FH_short","mid_short","BH_short","FH_half",
                  "mid_half","BH_half","FH_long","mid_long","BH_long"]
    for i, (nm, f) in enumerate(zip(zone_names, pf1s)):
        n_cls = (y_p_all[oof_mask] == i).sum()
        print(f"    {nm:12s}(cls{i}): F1={f:.4f}  n={n_cls}")

    print("\n  ActionId per-class F1:")
    ap_pred = np.argmax(oof_act_ruled, axis=1)
    af1s    = f1_score(y_a_all[oof_mask], ap_pred, labels=ACTION_EVAL_LABELS,
                       average=None, zero_division=0)
    action_names = ["None","Loop","Cloop","Smash","Flip","Pushfast","Push","Flick",
                    "Arch","Knuckle","Chop_r","ShortStop","Chop","Block","Lob"]
    for i, (nm, f) in enumerate(zip(action_names, af1s)):
        n_cls = (y_a_all[oof_mask] == i).sum()
        print(f"    {nm:10s}(cls{i:2d}): F1={f:.4f}  n={n_cls}")

    # ─── Threshold optimisation ───────────────────────────────────────────────
    print("\n  [Optimize] Action thresholds...")
    t_a, w_a, f1_a_opt = optimize_thresholds(
        oof_act_ruled, y_a_all[oof_mask], ACTION_EVAL_LABELS, ACTION_CW, N_ACTION)
    print("\n  [Optimize] Point thresholds...")
    t_p, w_p, f1_p_opt = optimize_thresholds(
        oof_pt[oof_mask], y_p_all[oof_mask], POINT_EVAL_LABELS, POINT_CW, N_POINT)

    ov_opt = 0.4*f1_a_opt + 0.4*f1_p_opt + 0.2*auc_oof
    print(f"\n  Optimized: action={f1_a_opt:.4f}  point={f1_p_opt:.4f}  OV={ov_opt:.4f}")
    print(f"  Gain from threshold opt: {ov_opt - ov_oof:+.4f}")

    # ─── Generate submission ──────────────────────────────────────────────────
    print("\n--- Generating submission ---")

    # Final test point predictions: use full averaged action probs as stacking
    if use_stack:
        X_test_ext_final = np.hstack([X_test, test_act15_acc])
        test_pt_acc_final  = (lgb_p.predict(X_test_ext_final) +
                              pad_proba(xgb_p.predict_proba(X_test_ext_final), xgb_p.classes_, N_POINT) +
                              pad_proba(cb_p.predict_proba(X_test_ext_final), cb_p.classes_, N_POINT)
                              ) / 3.0
        test_bin_acc_final = lgb_pb.predict(X_test_ext_final)
        # Average with fold-accumulated predictions for robustness
        test_pt_acc  = (test_pt_acc + test_pt_acc_final) / 2.0
        test_bin_acc = (test_bin_acc + test_bin_acc_final) / 2.0

    test_act_ruled = apply_action_rules(test_act_acc, nsn_test)
    test_act_t     = test_act_ruled ** (1.0 / t_a)
    test_act_t    /= test_act_t.sum(axis=1, keepdims=True)
    test_act_adj   = test_act_t * w_a[np.newaxis, :]
    pred_act       = np.argmax(test_act_adj, axis=1)

    test_pt_2s = blend_two_stage(test_pt_acc, test_bin_acc)
    test_pt_t  = test_pt_2s ** (1.0 / t_p)
    test_pt_t /= test_pt_t.sum(axis=1, keepdims=True)
    test_pt_adj = test_pt_t * w_p[np.newaxis, :]
    pred_pt     = np.argmax(test_pt_adj, axis=1)

    pred_srv = (test_srv_acc >= 0.5).astype(int)

    sub = pd.DataFrame({
        "rally_uid":      rally_test,
        "actionId":       pred_act,
        "pointId":        pred_pt,
        "serverGetPoint": pred_srv,
    })

    out_path = os.path.join(SUBMISSION_DIR, "submission_v10.csv")
    sub.to_csv(out_path, index=False)
    print(f"  actionId dist:  {dict(pd.Series(pred_act).value_counts().sort_index())}")
    print(f"  pointId dist:   {dict(pd.Series(pred_pt).value_counts().sort_index())}")
    print(f"  SGP dist:       {dict(pd.Series(pred_srv).value_counts().sort_index())}")
    print(f"  Saved: {out_path}")

    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal time: {elapsed:.1f} min")
    print(f"\n{'='*70}")
    print(f"FINAL OV (base):  {ov_oof:.4f}")
    print(f"FINAL OV (opt):   {ov_opt:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
