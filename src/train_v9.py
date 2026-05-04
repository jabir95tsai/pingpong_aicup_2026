"""V9 Champion Pipeline — Target OV > 0.6

Key fixes vs V8:
1. CORRECT macro F1: action uses labels=0..14 (serve classes 15-18 never appear
   as next-shot targets, including them forces macro/19 instead of macro/15 → 26% CV penalty removed)
2. Hand-flip augmentation: FH↔BH mirror doubles rare-class pointId samples
3. Extreme class weights: pointId class3×20, class1×12, action8×12, action14×10
4. Two-stage pointId: binary miss-model + 10-class blended
5. Proper threshold optimization: scipy Nelder-Mead over per-class weights
6. Test is one row per rally (build_features already uses full context as input)
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

N_ACTION = 19       # probability array width (0-18)
N_ACTION_TRAIN = 15 # classes 0-14 only appear as next-shot targets
N_POINT = 10

# For macro F1, only count classes that can appear as next-shot targets
# Classes 15-18 (serves) never appear as targets → include only 0-14
ACTION_EVAL_LABELS = list(range(15))   # 15 classes for action macro F1
POINT_EVAL_LABELS  = list(range(10))   # all 10 for point

# ─── Class weight maps ────────────────────────────────────────────────────────
ACTION_CW = {
    0: 1.5,   # 無/其他
    1: 0.6,   # 拉球 (most common, downweight)
    2: 0.9,   # 反拉
    3: 1.5,   # 殺球
    4: 1.2,   # 擰球
    5: 1.0,   # 快帶
    6: 0.8,   # 推擠
    7: 1.8,   # 挑撥
    8: 14.0,  # 拱球 (372 samples — rarest!)
    9: 8.0,   # 磕球
    10: 0.6,  # 搓球 (very common)
    11: 1.2,  # 擺短
    12: 0.9,  # 削球
    13: 0.7,  # 擋球 (common)
    14: 10.0, # 放高球 (rare)
    15: 0.01, 16: 0.01, 17: 0.01, 18: 0.01,  # serve: essentially impossible as next shot
}

POINT_CW = {
    0: 0.5,   # miss/net (most common ~22%)
    1: 12.0,  # FH_short (582 samples)
    2: 2.5,   # mid_short
    3: 22.0,  # BH_short (203 samples — rarest!)
    4: 2.0,   # FH_half
    5: 0.9,   # mid_half
    6: 1.5,   # BH_half
    7: 0.8,   # FH_long
    8: 0.7,   # mid_long
    9: 0.6,   # BH_long (most common valid)
}

POINT_FLIP = {1: 3, 3: 1, 4: 6, 6: 4, 7: 9, 9: 7}  # left-right mirror
# handId: 1(FH)↔2(BH),  positionId: 1(left)↔3(right)


# ─── Metric helpers ──────────────────────────────────────────────────────────

def action_macro_f1(y_true, probs):
    """Correct 15-class macro F1 for actionId (exclude zero-support serve classes)."""
    preds = np.argmax(probs, axis=1)
    return f1_score(y_true, preds, labels=ACTION_EVAL_LABELS,
                    average="macro", zero_division=0)


def point_macro_f1(y_true, probs):
    preds = np.argmax(probs, axis=1)
    return f1_score(y_true, preds, labels=POINT_EVAL_LABELS,
                    average="macro", zero_division=0)


def apply_action_rules(probs, next_sns):
    """Zero out serve classes for non-serve shots; enforce serve classes for serves."""
    out = probs.copy()
    serve_mask = (next_sns == 1)
    non_serve = ~serve_mask
    # if next shot is serve: zero out non-serve classes
    out[serve_mask, :15] = 0.0
    # if next shot is not serve: zero out serve classes
    for c in [15, 16, 17, 18]:
        if c < out.shape[1]:
            out[non_serve, c] = 0.0
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return out / row_sums


# ─── Augmentation ────────────────────────────────────────────────────────────

def build_flip_map(feature_names):
    """Return list of (col_a, col_b) index pairs to swap for hand-flip augmentation."""
    fn_idx = {n: i for i, n in enumerate(feature_names)}
    pairs = []
    for k in [1, 2, 3, 5]:
        for (a, b) in [
            (f"oh_lag{k}_handId_1",     f"oh_lag{k}_handId_2"),      # FH↔BH
            (f"oh_lag{k}_positionId_1", f"oh_lag{k}_positionId_3"),  # left↔right
            (f"oh_lag{k}_pointId_1",    f"oh_lag{k}_pointId_3"),     # 1↔3
            (f"oh_lag{k}_pointId_4",    f"oh_lag{k}_pointId_6"),     # 4↔6
            (f"oh_lag{k}_pointId_7",    f"oh_lag{k}_pointId_9"),     # 7↔9
        ]:
            if a in fn_idx and b in fn_idx:
                pairs.append((fn_idx[a], fn_idx[b]))
    return pairs


def augment_flip(X, y_act, y_pt, y_srv, flip_pairs):
    """Create left-right mirrored copy of training samples.

    y_pointId is flipped (1↔3, 4↔6, 7↔9).
    y_actionId and y_serverGetPoint are unchanged (hand-agnostic in aggregate).
    """
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


# ─── Threshold optimization ───────────────────────────────────────────────────

def optimize_thresholds(probs, y_true, eval_labels, init_cw_dict=None, n_classes=10):
    """Three-phase optimization: temperature → greedy → scipy → return best weights."""

    # Phase 1: temperature
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
    print(f"    Temp={best_t:.1f} → F1={best_f1:.4f}")

    # Phase 2: greedy per-class weight search
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
    print(f"    Greedy → F1={cur_f1:.4f}")

    # Phase 3: scipy Nelder-Mead
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
    """Blend 10-class probs with binary miss model.

    alpha: weight for binary model (higher = trust binary model more).
    """
    out = probs_10.copy()
    out[:, 0] = alpha * miss_prob + (1 - alpha) * out[:, 0]
    out /= out.sum(axis=1, keepdims=True)
    return out


# ─── Probability padding (XGB/CB may see fewer classes than N_ACTION) ─────────

def pad_proba(probs, model_classes, n_classes):
    """Expand probability matrix to n_classes columns.

    XGBClassifier / CatBoost may output fewer columns if some classes are absent
    from the training fold (e.g. serve classes 15-18 never appear as targets).
    """
    if probs.shape[1] == n_classes:
        return probs
    out = np.zeros((probs.shape[0], n_classes), dtype=np.float32)
    for col_idx, cls_label in enumerate(model_classes):
        out[:, int(cls_label)] = probs[:, col_idx]
    return out


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_Xy(feat_df, fnames):
    X = feat_df[fnames].values.astype(np.float32)
    y_a = feat_df["y_actionId"].values.astype(np.int32)
    y_p = feat_df["y_pointId"].values.astype(np.int32)
    y_s = feat_df["y_serverGetPoint"].values.astype(np.int32)
    nsn = feat_df["next_strikeNumber"].values.astype(np.int32)
    # Clip action labels: serve classes (15-18) should never appear as next-shot
    # targets; 1-2 anomalous rows exist in data → remap to class 0 (other)
    y_a = np.where(y_a >= N_ACTION_TRAIN, 0, y_a)
    return X, y_a, y_p, y_s, nsn


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--no-aug", action="store_true")
    args = parser.parse_args()

    is_smoke  = args.smoke
    n_folds   = 1 if is_smoke else args.folds
    n_boost   = 200 if is_smoke else 3000
    es_rounds = 30 if is_smoke else 200
    use_aug   = not args.no_aug

    t_start = time.time()
    print("=" * 70)
    print(f"V9 CHAMPION PIPELINE {'(SMOKE)' if is_smoke else ''}")
    print(f"  aug={use_aug}  folds={n_folds}  n_boost={n_boost}")
    print(f"  ACTION macro: 15 classes (0-14, excluding serve 15-18)")
    print("=" * 70)

    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    from features_v5 import compute_global_stats_v5, build_features_v5, get_feature_names_v5

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(TEST_PATH)
    train_df, test_df, _ = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1

    # ── Preflight ────────────────────────────────────────────────────────────
    print("\n--- Preflight ---")
    t0 = time.time()
    gs_full   = compute_global_stats_v5(train_df)
    feat_full = build_features_v5(train_df, is_train=True, global_stats_v5=gs_full)
    fnames    = get_feature_names_v5(feat_full)
    n_samples = len(feat_full)
    print(f"  {len(fnames)} features, {n_samples} samples ({time.time()-t0:.1f}s)")

    flip_pairs = build_flip_map(fnames)
    print(f"  Flip pairs: {len(flip_pairs)}")

    X_all, y_a_all, y_p_all, y_s_all, nsn_all = extract_Xy(feat_full, fnames)
    rally_uids_all = feat_full["rally_uid"].values
    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_all = np.array([rally_to_match.get(r, -1) for r in rally_uids_all])

    # OOF containers (original samples only)
    oof_act      = np.zeros((n_samples, N_ACTION))
    oof_pt       = np.zeros((n_samples, N_POINT))
    oof_srv      = np.zeros(n_samples)
    oof_pt_bin   = np.zeros(n_samples)

    gkf = GroupKFold(n_splits=max(n_folds, 2))
    splits = list(gkf.split(np.arange(n_samples), groups=match_all))
    if is_smoke:
        splits = splits[:1]

    # Test feature matrix (ONE row per test rally)
    feat_test  = build_features_v5(test_df, is_train=False, global_stats_v5=gs_full)
    X_test     = feat_test[fnames].values.astype(np.float32)
    nsn_test   = feat_test["next_strikeNumber"].values.astype(np.int32)
    rally_test = feat_test["rally_uid"].values

    test_act_acc = np.zeros((len(X_test), N_ACTION))
    test_pt_acc  = np.zeros((len(X_test), N_POINT))
    test_srv_acc = np.zeros(len(X_test))
    test_bin_acc = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(splits):
        t_fold = time.time()
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{len(splits)}")
        print(f"{'='*60}")

        tr_rallies  = set(rally_uids_all[tr_idx])
        val_rallies = set(rally_uids_all[val_idx])
        tr_raw  = train_df[train_df["rally_uid"].isin(tr_rallies)]
        val_raw = train_df[train_df["rally_uid"].isin(val_rallies)]

        fold_stats = compute_global_stats_v5(tr_raw)
        feat_tr  = build_features_v5(tr_raw,  is_train=True, global_stats_v5=fold_stats)
        feat_val = build_features_v5(val_raw, is_train=True, global_stats_v5=fold_stats)

        X_tr, y_a_tr, y_p_tr, y_s_tr, nsn_tr  = extract_Xy(feat_tr,  fnames)
        X_val, y_a_val, y_p_val, y_s_val, nsn_val = extract_Xy(feat_val, fnames)

        # ── Augmentation ─────────────────────────────────────────────────────
        if use_aug:
            X_tr_aug, y_a_aug, y_p_aug, y_s_aug = augment_flip(
                X_tr, y_a_tr, y_p_tr, y_s_tr, flip_pairs)
            print(f"  Augmented: {len(X_tr)} → {len(X_tr_aug)} samples")
        else:
            X_tr_aug, y_a_aug, y_p_aug, y_s_aug = X_tr, y_a_tr, y_p_tr, y_s_tr

        sw_a = np.array([ACTION_CW.get(int(c), 1.0) for c in y_a_aug], dtype=np.float32)
        sw_p = np.array([POINT_CW.get(int(c), 1.0)  for c in y_p_aug], dtype=np.float32)
        y_miss_aug = (y_p_aug == 0).astype(np.int32)
        y_miss_val = (y_p_val == 0).astype(np.int32)

        # ── ACTION: LGB + XGB + CB ────────────────────────────────────────────
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
        pa_lgb = lgb_a.predict(X_val)

        xgb_a = xgb.XGBClassifier(
            n_estimators=n_boost, learning_rate=0.04, max_depth=7,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            objective="multi:softprob", num_class=N_ACTION_TRAIN,
            eval_metric="mlogloss", early_stopping_rounds=es_rounds,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0, tree_method="hist")
        xgb_a.fit(X_tr_aug, y_a_aug, sample_weight=sw_a,
                  eval_set=[(X_val, y_a_val)], verbose=False)
        pa_xgb = xgb_a.predict_proba(X_val)

        cb_a = CatBoostClassifier(
            iterations=n_boost, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
            loss_function="MultiClass", classes_count=N_ACTION_TRAIN,
            random_seed=RANDOM_SEED, verbose=False, allow_writing_files=False,
            early_stopping_rounds=es_rounds)
        cb_a.fit(X_tr_aug, y_a_aug, sample_weight=sw_a,
                 eval_set=(X_val, y_a_val), use_best_model=True)
        pa_cb = cb_a.predict_proba(X_val)

        pa_xgb = pad_proba(pa_xgb, xgb_a.classes_, N_ACTION_TRAIN)
        pa_cb  = pad_proba(pa_cb,  cb_a.classes_,  N_ACTION_TRAIN)
        # Extend to full N_ACTION=19 (columns 15-18 stay 0)
        def extend_action(p):
            out = np.zeros((p.shape[0], N_ACTION), dtype=np.float32)
            out[:, :N_ACTION_TRAIN] = p
            return out
        pa_lgb = extend_action(pa_lgb)
        pa_xgb = extend_action(pa_xgb)
        pa_cb  = extend_action(pa_cb)
        pa_blend = (pa_lgb + pa_xgb + pa_cb) / 3.0
        pa_ruled = apply_action_rules(pa_blend, nsn_val)
        f1_a_val = action_macro_f1(y_a_val, pa_ruled)
        print(f"  [Action] F1={f1_a_val:.4f}")

        # store OOF
        val_uids = set(rally_uids_all[val_idx])
        val_mask = np.isin(rally_uids_all, list(val_uids))
        oof_act[val_mask] = pa_ruled

        # ── POINT binary ──────────────────────────────────────────────────────
        lgb_pb_p = dict(n_estimators=n_boost, learning_rate=0.04,
                        num_leaves=63, max_depth=7, min_child_samples=10,
                        subsample=0.8, colsample_bytree=0.7,
                        objective="binary", metric="binary_logloss",
                        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        lgb_pb = lgb.train(lgb_pb_p,
            lgb.Dataset(X_tr_aug, label=y_miss_aug.astype(np.float32)),
            valid_sets=[lgb.Dataset(X_val, label=y_miss_val.astype(np.float32))],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        pb_val = lgb_pb.predict(X_val)
        oof_pt_bin[val_mask] = pb_val

        # ── POINT 10-class ────────────────────────────────────────────────────
        lgb_p_p = dict(n_estimators=n_boost, learning_rate=0.04,
                       num_leaves=127, max_depth=9, min_child_samples=5,
                       subsample=0.8, colsample_bytree=0.7,
                       reg_alpha=0.1, reg_lambda=1.0,
                       objective="multiclass", metric="multi_logloss",
                       num_class=N_POINT, random_state=RANDOM_SEED,
                       n_jobs=-1, verbose=-1)
        lgb_p = lgb.train(lgb_p_p,
            lgb.Dataset(X_tr_aug, label=y_p_aug, weight=sw_p),
            valid_sets=[lgb.Dataset(X_val, label=y_p_val)],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        pp_lgb = lgb_p.predict(X_val)

        xgb_p = xgb.XGBClassifier(
            n_estimators=n_boost, learning_rate=0.04, max_depth=7,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            objective="multi:softprob", num_class=N_POINT,
            eval_metric="mlogloss", early_stopping_rounds=es_rounds,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0, tree_method="hist")
        xgb_p.fit(X_tr_aug, y_p_aug, sample_weight=sw_p,
                  eval_set=[(X_val, y_p_val)], verbose=False)
        pp_xgb = xgb_p.predict_proba(X_val)

        cb_p = CatBoostClassifier(
            iterations=n_boost, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
            loss_function="MultiClass", classes_count=N_POINT,
            random_seed=RANDOM_SEED, verbose=False, allow_writing_files=False,
            early_stopping_rounds=es_rounds)
        cb_p.fit(X_tr_aug, y_p_aug, sample_weight=sw_p,
                 eval_set=(X_val, y_p_val), use_best_model=True)
        pp_cb = cb_p.predict_proba(X_val)

        pp_xgb = pad_proba(pp_xgb, xgb_p.classes_, N_POINT)
        pp_cb  = pad_proba(pp_cb,  cb_p.classes_,  N_POINT)
        pp_blend = (pp_lgb + pp_xgb + pp_cb) / 3.0
        pp_2stage = blend_two_stage(pp_blend, pb_val)
        f1_p_val = point_macro_f1(y_p_val, pp_2stage)
        print(f"  [Point]  F1={f1_p_val:.4f}")
        oof_pt[val_mask] = pp_2stage

        # ── SERVER ────────────────────────────────────────────────────────────
        lgb_s_p = dict(n_estimators=n_boost, learning_rate=0.04,
                       num_leaves=63, max_depth=7, min_child_samples=15,
                       subsample=0.8, colsample_bytree=0.7,
                       objective="binary", metric="auc",
                       random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        lgb_s = lgb.train(lgb_s_p,
            lgb.Dataset(X_tr_aug, label=y_s_aug.astype(np.float32)),
            valid_sets=[lgb.Dataset(X_val, label=y_s_val.astype(np.float32))],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        ps_lgb = lgb_s.predict(X_val)

        xgb_s = xgb.XGBClassifier(
            n_estimators=n_boost, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.7,
            objective="binary:logistic", eval_metric="auc",
            early_stopping_rounds=es_rounds,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0, tree_method="hist")
        xgb_s.fit(X_tr_aug, y_s_aug, eval_set=[(X_val, y_s_val)], verbose=False)
        ps_xgb = xgb_s.predict_proba(X_val)[:, 1]

        ps_blend = (ps_lgb + ps_xgb) / 2.0
        auc_val = roc_auc_score(y_s_val, ps_blend)
        print(f"  [Server] AUC={auc_val:.4f}")
        oof_srv[val_mask] = ps_blend

        ov_fold = 0.4*f1_a_val + 0.4*f1_p_val + 0.2*auc_val
        print(f"\n  FOLD OV={ov_fold:.4f}  [{time.time()-t_fold:.0f}s]")

        # accumulate test predictions
        def _act_proba_test(m_lgb, m_xgb, m_cb):
            p_l = extend_action(m_lgb.predict(X_test))
            p_x = extend_action(pad_proba(m_xgb.predict_proba(X_test), m_xgb.classes_, N_ACTION_TRAIN))
            p_c = extend_action(pad_proba(m_cb.predict_proba(X_test),  m_cb.classes_,  N_ACTION_TRAIN))
            return (p_l + p_x + p_c) / 3.0
        test_act_acc += _act_proba_test(lgb_a, xgb_a, cb_a) / len(splits)
        test_pt_acc  += (lgb_p.predict(X_test) +
                         pad_proba(xgb_p.predict_proba(X_test), xgb_p.classes_, N_POINT) +
                         pad_proba(cb_p.predict_proba(X_test),  cb_p.classes_,  N_POINT)) / 3.0 / len(splits)
        test_bin_acc += lgb_pb.predict(X_test) / len(splits)
        test_srv_acc += (lgb_s.predict(X_test) +
                         xgb_s.predict_proba(X_test)[:, 1]) / 2.0 / len(splits)
        gc.collect()

    # ─── Global OOF evaluation (only on samples that were in a val fold) ─────
    print("\n" + "="*70)
    print("GLOBAL OOF RESULTS")

    # Only evaluate on samples that received OOF predictions (val fold samples)
    # Samples with all-zero action probs were never in a val fold (smoke test)
    oof_mask = oof_act.sum(axis=1) > 0
    n_oof = oof_mask.sum()
    print(f"  OOF samples: {n_oof}/{n_samples} ({100*n_oof/n_samples:.0f}%)")

    oof_act_ruled = apply_action_rules(oof_act[oof_mask], nsn_all[oof_mask])
    f1_a_oof = action_macro_f1(y_a_all[oof_mask], oof_act_ruled)
    f1_p_oof = point_macro_f1(y_p_all[oof_mask], oof_pt[oof_mask])
    auc_oof  = roc_auc_score(y_s_all[oof_mask], oof_srv[oof_mask])
    ov_oof   = 0.4*f1_a_oof + 0.4*f1_p_oof + 0.2*auc_oof
    print(f"  Base:  action={f1_a_oof:.4f}  point={f1_p_oof:.4f}  AUC={auc_oof:.4f}  OV={ov_oof:.4f}")

    # Per-class F1 breakdown
    print("\n  PointId per-class F1:")
    pp_pred = np.argmax(oof_pt[oof_mask], axis=1)
    pf1s = f1_score(y_p_all[oof_mask], pp_pred, labels=POINT_EVAL_LABELS,
                    average=None, zero_division=0)
    zone_names = ["miss","FH_short","mid_short","BH_short","FH_half",
                  "mid_half","BH_half","FH_long","mid_long","BH_long"]
    for i, (nm, f) in enumerate(zip(zone_names, pf1s)):
        n_oof_cls = (y_p_all[oof_mask] == i).sum()
        print(f"    {nm:12s}(cls{i}): F1={f:.4f}  n_oof={n_oof_cls}  n_total={(y_p_all==i).sum()}")

    print("\n  ActionId per-class F1:")
    ap_pred = np.argmax(oof_act_ruled, axis=1)
    af1s = f1_score(y_a_all[oof_mask], ap_pred, labels=ACTION_EVAL_LABELS,
                    average=None, zero_division=0)
    action_names = ["None","Loop","Cloop","Smash","Flip","Pushfast","Push","Flick",
                    "Arch","Knuckle","Chop_r","ShortStop","Chop","Block","Lob"]
    for i, (nm, f) in enumerate(zip(action_names, af1s)):
        n_oof_cls = (y_a_all[oof_mask] == i).sum()
        print(f"    {nm:10s}(cls{i:2d}): F1={f:.4f}  n_oof={n_oof_cls}")

    # ─── Threshold optimization ───────────────────────────────────────────────
    print("\n  [Optimize] Action thresholds...")
    t_a, w_a, f1_a_opt = optimize_thresholds(oof_act_ruled, y_a_all[oof_mask],
                                              ACTION_EVAL_LABELS, ACTION_CW, N_ACTION)
    print("\n  [Optimize] Point thresholds...")
    t_p, w_p, f1_p_opt = optimize_thresholds(oof_pt[oof_mask], y_p_all[oof_mask],
                                              POINT_EVAL_LABELS, POINT_CW, N_POINT)

    ov_opt = 0.4*f1_a_opt + 0.4*f1_p_opt + 0.2*auc_oof
    print(f"\n  Optimized: action={f1_a_opt:.4f}  point={f1_p_opt:.4f}  OV={ov_opt:.4f}")
    print(f"  Gain from threshold opt: {ov_opt - ov_oof:+.4f}")

    # ─── Generate submission ──────────────────────────────────────────────────
    print("\n--- Generating submission ---")

    # Apply temperature + weights to test predictions
    test_act_ruled = apply_action_rules(test_act_acc, nsn_test)
    test_act_t = test_act_ruled ** (1.0 / t_a)
    test_act_t /= test_act_t.sum(axis=1, keepdims=True)
    test_act_adj = test_act_t * w_a[np.newaxis, :]
    pred_act = np.argmax(test_act_adj, axis=1)

    test_pt_2s = blend_two_stage(test_pt_acc, test_bin_acc)
    test_pt_t = test_pt_2s ** (1.0 / t_p)
    test_pt_t /= test_pt_t.sum(axis=1, keepdims=True)
    test_pt_adj = test_pt_t * w_p[np.newaxis, :]
    pred_pt = np.argmax(test_pt_adj, axis=1)

    pred_srv = (test_srv_acc >= 0.5).astype(int)

    # Build submission (one row per test rally)
    sub = pd.DataFrame({
        "rally_uid":       rally_test,
        "actionId":        pred_act,
        "pointId":         pred_pt,
        "serverGetPoint":  pred_srv,
    })

    out_path = os.path.join(SUBMISSION_DIR, "submission_v9.csv")
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
