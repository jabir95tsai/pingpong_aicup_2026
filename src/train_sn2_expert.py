"""SN=2 Expert: specialist model for receive shots (next_strikeNumber == 2).

Trains the same V12 two-pass action→point stacking architecture, but ONLY
on rows where next_strikeNumber == 2. SN=2 is currently the weakest slice
(OV ~0.27 vs global ~0.37) and accounts for 21.5% of all data.

Uses features_v8 (V7 receive priors + V8 receive point priors).

Outputs (full-length arrays; non-SN=2 rows are zeros):
  oof_predictions/sn2_expert_oof_act.npy        (N_total, 19)
  oof_predictions/sn2_expert_oof_pt.npy         (N_total, 10)
  oof_predictions/sn2_expert_oof_srv.npy        (N_total,)
  oof_predictions/sn2_expert_oof_mask.npy       (N_total,) bool, True for SN=2 OOF rows
  oof_predictions/sn2_expert_oof_y_*.npy
  oof_predictions/sn2_expert_oof_nsn.npy

Test outputs (only the SN=2 test rows):
  oof_predictions/sn2_expert_test_act.npy       (N_sn2_test, 19)
  oof_predictions/sn2_expert_test_pt.npy        (N_sn2_test, 10)
  oof_predictions/sn2_expert_test_srv.npy       (N_sn2_test,)
  oof_predictions/sn2_expert_test_rally_uid.npy (N_sn2_test,)
"""
import sys, os, time, warnings, gc, argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, SUBMISSION_DIR, RANDOM_SEED
from data_cleaning import clean_data
from train_v12 import (
    N_ACTION, N_ACTION_TRAIN, N_POINT,
    ACTION_EVAL_LABELS, POINT_EVAL_LABELS,
    ACTION_CW, POINT_CW,
    action_macro_f1, point_macro_f1, apply_action_rules,
    optimize_thresholds, blend_two_stage,
    pad_proba, extend_action, extract_Xy,
)


def fit_xgb_compact(X_tr, y_tr, X_val, y_val, sw, n_classes_full,
                    n_boost, es_rounds, max_depth, min_child_weight, seed):
    """Fit XGBClassifier with class remapping (XGB requires contiguous labels).

    Returns (xgb_model, predict_fn) where predict_fn(X) returns full
    n_classes_full-dim probabilities (zero-padded for missing classes).
    """
    import xgboost as xgb
    unique_classes = sorted(np.unique(y_tr).tolist())
    n_unique = len(unique_classes)
    cls_to_compact = {c: i for i, c in enumerate(unique_classes)}
    y_tr_c  = np.array([cls_to_compact[c] for c in y_tr], dtype=np.int32)
    y_val_c = np.array([cls_to_compact.get(c, 0) for c in y_val], dtype=np.int32)

    model = xgb.XGBClassifier(
        n_estimators=n_boost, learning_rate=0.04, max_depth=max_depth,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=min_child_weight,
        objective="multi:softprob", num_class=n_unique,
        eval_metric="mlogloss", early_stopping_rounds=es_rounds,
        random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist")
    model.fit(X_tr, y_tr_c, sample_weight=sw,
              eval_set=[(X_val, y_val_c)], verbose=False)

    def predict_full(X):
        proba_compact = model.predict_proba(X)  # (n, n_unique)
        proba_full = np.zeros((len(X), n_classes_full), dtype=np.float32)
        for i, c in enumerate(unique_classes):
            if c < n_classes_full:
                proba_full[:, c] = proba_compact[:, i]
        return proba_full

    return model, predict_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds",   type=int, default=3)
    parser.add_argument("--n-boost", type=int, default=1500)
    parser.add_argument("--es",      type=int, default=150)
    parser.add_argument("--skip-cb", action="store_true", default=True,
                        help="Skip CatBoost (default for SN=2 expert; small dataset)")
    parser.add_argument("--with-cb", action="store_true",
                        help="Force-enable CatBoost")
    parser.add_argument("--tag",     type=str, default="sn2_expert")
    args = parser.parse_args()

    skip_cb  = args.skip_cb and not args.with_cb
    n_folds  = args.folds
    n_boost  = args.n_boost
    es_rounds = args.es
    out_tag  = args.tag

    t_start = time.time()
    print("=" * 70)
    print(f"SN=2 EXPERT  tag={out_tag}")
    print(f"  folds={n_folds}  n_boost={n_boost}  es={es_rounds}  skip_cb={skip_cb}")
    print("=" * 70)

    import xgboost as xgb
    import lightgbm as lgb
    if not skip_cb:
        from catboost import CatBoostClassifier
    from features_v8 import (compute_global_stats_v8, build_features_v8,
                              get_feature_names_v8)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(TEST_PATH)
    train_df, test_df, _ = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1

    # Preflight: build features on full data (we still need fold-safe stats per fold)
    print("\n--- Preflight ---")
    t0 = time.time()
    gs_full   = compute_global_stats_v8(train_df)
    feat_full = build_features_v8(train_df, is_train=True,
                                   global_stats_v8=gs_full,
                                   raw_df=train_df)
    fnames    = get_feature_names_v8(feat_full)
    n_total   = len(feat_full)
    print(f"  {len(fnames)} features, {n_total} samples ({time.time()-t0:.1f}s)")

    # Identify SN=2 rows
    nsn_full   = feat_full["next_strikeNumber"].values.astype(np.int32)
    sn2_mask_full = (nsn_full == 2)
    n_sn2 = sn2_mask_full.sum()
    print(f"  SN=2 rows: {n_sn2}/{n_total} ({100*n_sn2/n_total:.1f}%)")

    feat_sn2 = feat_full[sn2_mask_full].reset_index(drop=True)

    X_sn2, y_a_sn2, y_p_sn2, y_s_sn2, nsn_sn2 = extract_Xy(feat_sn2, fnames)
    rally_uids_sn2 = feat_sn2["rally_uid"].values
    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_sn2 = np.array([rally_to_match.get(r, -1) for r in rally_uids_sn2])

    # Full-length arrays (filled only at SN=2 indices)
    y_a_all = np.zeros(n_total, dtype=np.int32); y_a_all[sn2_mask_full] = y_a_sn2
    y_p_all = np.zeros(n_total, dtype=np.int32); y_p_all[sn2_mask_full] = y_p_sn2
    y_s_all = np.zeros(n_total, dtype=np.int32); y_s_all[sn2_mask_full] = y_s_sn2
    # Use V12 labels for non-SN=2 rows so blending tools that read sn2_expert can still
    # compute global metrics if desired. Load original labels:
    feat_all_y_a = feat_full["y_actionId"].values.astype(np.int32)
    feat_all_y_p = feat_full["y_pointId"].values.astype(np.int32)
    feat_all_y_s = feat_full["y_serverGetPoint"].values.astype(np.int32)
    feat_all_y_a = np.where(feat_all_y_a >= N_ACTION_TRAIN, 0, feat_all_y_a)
    y_a_all[~sn2_mask_full] = feat_all_y_a[~sn2_mask_full]
    y_p_all[~sn2_mask_full] = feat_all_y_p[~sn2_mask_full]
    y_s_all[~sn2_mask_full] = feat_all_y_s[~sn2_mask_full]

    # OOF containers (full-length, but we only fill SN=2 rows)
    oof_act    = np.zeros((n_total, N_ACTION))
    oof_pt     = np.zeros((n_total, N_POINT))
    oof_srv    = np.zeros(n_total)
    oof_pt_bin = np.zeros(n_total)

    # GroupKFold by match on the SN=2 subset
    gkf    = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(np.arange(n_sn2), groups=match_sn2))

    # Build rally_uid → full_index map for OOF assignment.
    # Each rally has exactly one row with next_strikeNumber == 2 in feat_full.
    sn2_full_indices = np.where(sn2_mask_full)[0]
    sn2_rally_uids   = feat_full["rally_uid"].values[sn2_full_indices].astype(int)
    rally_to_full_idx = dict(zip(sn2_rally_uids, sn2_full_indices))

    # Build full test features and filter to SN=2 test rows
    feat_test = build_features_v8(test_df, is_train=False,
                                   global_stats_v8=gs_full,
                                   raw_df=test_df)
    nsn_test_full   = feat_test["next_strikeNumber"].values.astype(np.int32)
    sn2_mask_test   = (nsn_test_full == 2)
    feat_test_sn2   = feat_test[sn2_mask_test].reset_index(drop=True)
    X_test_sn2      = feat_test_sn2[fnames].values.astype(np.float32)
    rally_test_sn2  = feat_test_sn2["rally_uid"].values
    nsn_test_sn2    = feat_test_sn2["next_strikeNumber"].values.astype(np.int32)
    n_test_sn2      = len(X_test_sn2)
    print(f"  Test SN=2 rows: {n_test_sn2}/{len(feat_test)} "
          f"({100*n_test_sn2/len(feat_test):.1f}%)")

    test_act_acc = np.zeros((n_test_sn2, N_ACTION))
    test_pt_acc  = np.zeros((n_test_sn2, N_POINT))
    test_srv_acc = np.zeros(n_test_sn2)
    test_bin_acc = np.zeros(n_test_sn2)
    test_act15_acc = np.zeros((n_test_sn2, N_ACTION_TRAIN))

    # ── Fold loop ────────────────────────────────────────────────────────────
    for fold, (tr_idx, val_idx) in enumerate(splits):
        t_fold = time.time()
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")

        tr_rallies  = set(rally_uids_sn2[tr_idx])
        val_rallies = set(rally_uids_sn2[val_idx])

        # Rebuild fold-safe stats from training rallies (full rallies, not just SN=2)
        # so that lag features include preceding shots
        tr_raw_full   = train_df[train_df["rally_uid"].isin(tr_rallies)]
        val_raw_full  = train_df[train_df["rally_uid"].isin(val_rallies)]

        fold_stats = compute_global_stats_v8(tr_raw_full)
        feat_tr_full = build_features_v8(tr_raw_full, is_train=True,
                                          global_stats_v8=fold_stats,
                                          raw_df=tr_raw_full)
        feat_val_full = build_features_v8(val_raw_full, is_train=True,
                                           global_stats_v8=fold_stats,
                                           raw_df=val_raw_full)
        # Filter to SN=2 rows
        feat_tr  = feat_tr_full [feat_tr_full ["next_strikeNumber"] == 2].reset_index(drop=True)
        feat_val = feat_val_full[feat_val_full["next_strikeNumber"] == 2].reset_index(drop=True)
        del feat_tr_full, feat_val_full

        X_tr,  y_a_tr,  y_p_tr,  y_s_tr,  nsn_tr  = extract_Xy(feat_tr,  fnames)
        X_val, y_a_val, y_p_val, y_s_val, nsn_val = extract_Xy(feat_val, fnames)
        print(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

        sw_a = np.array([ACTION_CW.get(int(c), 1.0) for c in y_a_tr], dtype=np.float32)
        sw_p = np.array([POINT_CW.get(int(c),  1.0) for c in y_p_tr], dtype=np.float32)

        # ════ ACTION ══════════════════════════════════════════════════════════
        lgb_a_p = dict(n_estimators=n_boost, learning_rate=0.04,
                        num_leaves=127, max_depth=9, min_child_samples=8,
                        subsample=0.8, colsample_bytree=0.7,
                        reg_alpha=0.1, reg_lambda=1.0,
                        objective="multiclass", metric="multi_logloss",
                        num_class=N_ACTION_TRAIN, random_state=RANDOM_SEED,
                        n_jobs=-1, verbose=-1)
        lgb_a = lgb.train(lgb_a_p,
            lgb.Dataset(X_tr, label=y_a_tr, weight=sw_a),
            valid_sets=[lgb.Dataset(X_val, label=y_a_val)],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        xgb_a, predict_xgb_a = fit_xgb_compact(
            X_tr, y_a_tr, X_val, y_a_val, sw_a, N_ACTION_TRAIN,
            n_boost, es_rounds, max_depth=7, min_child_weight=5, seed=RANDOM_SEED)
        cb_a = None
        if not skip_cb:
            cb_a = CatBoostClassifier(
                iterations=n_boost, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
                loss_function="MultiClass", classes_count=N_ACTION_TRAIN,
                random_seed=RANDOM_SEED, verbose=False, allow_writing_files=False,
                early_stopping_rounds=es_rounds)
            cb_a.fit(X_tr, y_a_tr, sample_weight=sw_a,
                     eval_set=(X_val, y_a_val), use_best_model=True)

        pa_val_lgb = lgb_a.predict(X_val)
        pa_val_xgb = predict_xgb_a(X_val)
        if cb_a is not None:
            pa_val_cb = pad_proba(cb_a.predict_proba(X_val), cb_a.classes_, N_ACTION_TRAIN)
            pa_val_15 = (pa_val_lgb + pa_val_xgb + pa_val_cb) / 3.0
        else:
            pa_val_15 = (pa_val_lgb + pa_val_xgb) / 2.0
        pa_val_19 = extend_action(pa_val_15)
        pa_ruled  = apply_action_rules(pa_val_19, nsn_val)
        f1_a_val  = action_macro_f1(y_a_val, pa_ruled)
        print(f"  [Action] F1={f1_a_val:.4f}")

        # Map val rows back to full-length indices via rally_uid.
        # Each rally has exactly ONE row with next_strikeNumber == 2.
        full_val_idx_sorted = np.array([rally_to_full_idx[int(r)]
                                         for r in feat_val["rally_uid"].values])

        oof_act[full_val_idx_sorted] = pa_ruled

        # Test action
        pa_test_lgb = lgb_a.predict(X_test_sn2)
        pa_test_xgb = predict_xgb_a(X_test_sn2)
        if cb_a is not None:
            pa_test_cb = pad_proba(cb_a.predict_proba(X_test_sn2), cb_a.classes_, N_ACTION_TRAIN)
            pa_test_15 = (pa_test_lgb + pa_test_xgb + pa_test_cb) / 3.0
        else:
            pa_test_15 = (pa_test_lgb + pa_test_xgb) / 2.0
        test_act15_acc += pa_test_15 / n_folds
        test_act_acc   += extend_action(pa_test_15) / n_folds

        # Action probs on training data (for stacking)
        pa_tr_lgb = lgb_a.predict(X_tr)
        pa_tr_xgb = predict_xgb_a(X_tr)
        if cb_a is not None:
            pa_tr_cb = pad_proba(cb_a.predict_proba(X_tr), cb_a.classes_, N_ACTION_TRAIN)
            pa_tr_15 = (pa_tr_lgb + pa_tr_xgb + pa_tr_cb) / 3.0
        else:
            pa_tr_15 = (pa_tr_lgb + pa_tr_xgb) / 2.0

        # ════ POINT (with action stacking) ════════════════════════════════════
        X_tr_ext  = np.hstack([X_tr,  pa_tr_15])
        X_val_ext = np.hstack([X_val, pa_val_15])

        y_miss_tr  = (y_p_tr  == 0).astype(np.int32)
        y_miss_val = (y_p_val == 0).astype(np.int32)

        lgb_pb_p = dict(n_estimators=n_boost, learning_rate=0.04,
                        num_leaves=63, max_depth=7, min_child_samples=10,
                        subsample=0.8, colsample_bytree=0.7,
                        objective="binary", metric="binary_logloss",
                        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        lgb_pb = lgb.train(lgb_pb_p,
            lgb.Dataset(X_tr_ext, label=y_miss_tr.astype(np.float32)),
            valid_sets=[lgb.Dataset(X_val_ext, label=y_miss_val.astype(np.float32))],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        pb_val = lgb_pb.predict(X_val_ext)
        oof_pt_bin[full_val_idx_sorted] = pb_val

        lgb_p_p = dict(n_estimators=n_boost, learning_rate=0.04,
                       num_leaves=127, max_depth=9, min_child_samples=5,
                       subsample=0.8, colsample_bytree=0.7,
                       reg_alpha=0.1, reg_lambda=1.0,
                       objective="multiclass", metric="multi_logloss",
                       num_class=N_POINT, random_state=RANDOM_SEED,
                       n_jobs=-1, verbose=-1)
        lgb_p = lgb.train(lgb_p_p,
            lgb.Dataset(X_tr_ext, label=y_p_tr, weight=sw_p),
            valid_sets=[lgb.Dataset(X_val_ext, label=y_p_val)],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        pp_lgb = lgb_p.predict(X_val_ext)

        xgb_p, predict_xgb_p = fit_xgb_compact(
            X_tr_ext, y_p_tr, X_val_ext, y_p_val, sw_p, N_POINT,
            n_boost, es_rounds, max_depth=7, min_child_weight=3, seed=RANDOM_SEED)
        pp_xgb = predict_xgb_p(X_val_ext)
        cb_p = None
        if not skip_cb:
            cb_p = CatBoostClassifier(
                iterations=n_boost, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
                loss_function="MultiClass", classes_count=N_POINT,
                random_seed=RANDOM_SEED, verbose=False, allow_writing_files=False,
                early_stopping_rounds=es_rounds)
            cb_p.fit(X_tr_ext, y_p_tr, sample_weight=sw_p,
                     eval_set=(X_val_ext, y_p_val), use_best_model=True)
            pp_cb = pad_proba(cb_p.predict_proba(X_val_ext), cb_p.classes_, N_POINT)
            pp_blend = (pp_lgb + pp_xgb + pp_cb) / 3.0
        else:
            pp_blend = (pp_lgb + pp_xgb) / 2.0
        pp_2stage = blend_two_stage(pp_blend, pb_val)
        f1_p_val  = point_macro_f1(y_p_val, pp_2stage)
        print(f"  [Point]  F1={f1_p_val:.4f}")
        oof_pt[full_val_idx_sorted] = pp_2stage

        # Test point
        X_test_ext = np.hstack([X_test_sn2, test_act15_acc * n_folds / (fold + 1)])
        if cb_p is not None:
            test_pt_acc += (lgb_p.predict(X_test_ext) +
                             predict_xgb_p(X_test_ext) +
                             pad_proba(cb_p.predict_proba(X_test_ext),  cb_p.classes_,  N_POINT)
                             ) / 3.0 / n_folds
        else:
            test_pt_acc += (lgb_p.predict(X_test_ext) +
                             predict_xgb_p(X_test_ext)
                             ) / 2.0 / n_folds
        test_bin_acc += lgb_pb.predict(X_test_ext) / n_folds

        # ════ SERVER ══════════════════════════════════════════════════════════
        X_tr_srv_ext  = np.hstack([X_tr,  pa_tr_15])
        X_val_srv_ext = np.hstack([X_val, pa_val_15])

        lgb_s_p = dict(n_estimators=n_boost, learning_rate=0.04,
                       num_leaves=63, max_depth=7, min_child_samples=15,
                       subsample=0.8, colsample_bytree=0.7,
                       objective="binary", metric="auc",
                       random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        lgb_s = lgb.train(lgb_s_p,
            lgb.Dataset(X_tr_srv_ext, label=y_s_tr.astype(np.float32)),
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
        xgb_s.fit(X_tr_srv_ext, y_s_tr,
                  eval_set=[(X_val_srv_ext, y_s_val)], verbose=False)
        ps_xgb = xgb_s.predict_proba(X_val_srv_ext)[:, 1]

        ps_blend = (ps_lgb + ps_xgb) / 2.0
        if y_s_val.std() < 1e-9:
            auc_val = 0.5
        else:
            auc_val = roc_auc_score(y_s_val, ps_blend)
        print(f"  [Server] AUC={auc_val:.4f}")
        oof_srv[full_val_idx_sorted] = ps_blend

        X_test_srv_ext = np.hstack([X_test_sn2, test_act15_acc * n_folds / (fold + 1)])
        test_srv_acc += (lgb_s.predict(X_test_srv_ext) +
                         xgb_s.predict_proba(X_test_srv_ext)[:, 1]) / 2.0 / n_folds

        ov_fold = 0.4*f1_a_val + 0.4*f1_p_val + 0.2*auc_val
        print(f"\n  FOLD OV={ov_fold:.4f}  [{time.time()-t_fold:.0f}s]")
        gc.collect()

    # ─── Global SN=2 OOF evaluation ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SN=2 OOF RESULTS")
    oof_sn2_mask = sn2_mask_full & (oof_act.sum(axis=1) > 0)
    n_oof = oof_sn2_mask.sum()
    print(f"  SN=2 OOF samples: {n_oof}/{n_sn2}")

    oof_act_ruled = apply_action_rules(oof_act[oof_sn2_mask], nsn_full[oof_sn2_mask])
    f1_a = action_macro_f1(y_a_all[oof_sn2_mask], oof_act_ruled)
    f1_p = point_macro_f1(y_p_all[oof_sn2_mask], oof_pt[oof_sn2_mask])
    if y_s_all[oof_sn2_mask].std() < 1e-9:
        auc = 0.5
    else:
        auc = roc_auc_score(y_s_all[oof_sn2_mask], oof_srv[oof_sn2_mask])
    ov = 0.4*f1_a + 0.4*f1_p + 0.2*auc
    print(f"  SN=2 OOF: F1_a={f1_a:.4f}  F1_p={f1_p:.4f}  AUC={auc:.4f}  OV={ov:.4f}")

    # Compare against V12+V11 baseline on SN=2
    print(f"  Baseline (V12+V11 blend on SN=2): F1_a~0.249  F1_p~0.161  AUC~0.539  OV~0.271")

    # Per-class breakdown (SN=2 only)
    pp_pred = np.argmax(oof_pt[oof_sn2_mask], axis=1)
    pf1s    = f1_score(y_p_all[oof_sn2_mask], pp_pred, labels=POINT_EVAL_LABELS,
                       average=None, zero_division=0)
    zone_names = ["miss","FH_short","mid_short","BH_short","FH_half",
                   "mid_half","BH_half","FH_long","mid_long","BH_long"]
    print("  SN=2 PointId per-class F1:")
    for i, (nm, f) in enumerate(zip(zone_names, pf1s)):
        n_cls = (y_p_all[oof_sn2_mask] == i).sum()
        print(f"    {nm:12s}(cls{i}): F1={f:.4f}  n={n_cls}")

    # ─── Threshold optimisation (SN=2 only) ───────────────────────────────────
    print("\n  [Optimize] SN=2 Action thresholds...")
    t_a, w_a, f1_a_opt = optimize_thresholds(
        oof_act_ruled, y_a_all[oof_sn2_mask], ACTION_EVAL_LABELS, ACTION_CW, N_ACTION)
    print("\n  [Optimize] SN=2 Point thresholds...")
    t_p, w_p, f1_p_opt = optimize_thresholds(
        oof_pt[oof_sn2_mask], y_p_all[oof_sn2_mask],
        POINT_EVAL_LABELS, POINT_CW, N_POINT)
    ov_opt = 0.4*f1_a_opt + 0.4*f1_p_opt + 0.2*auc
    print(f"  Optimized SN=2 OV={ov_opt:.4f}  (from {ov:.4f}, +{ov_opt-ov:+.4f})")

    # ─── Save artifacts ───────────────────────────────────────────────────────
    oof_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")
    os.makedirs(oof_dir, exist_ok=True)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_act.npy"), oof_act)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_pt.npy"),  oof_pt)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_srv.npy"), oof_srv)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_pt_bin.npy"), oof_pt_bin)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_mask.npy"), oof_sn2_mask)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_y_act.npy"), y_a_all)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_y_pt.npy"),  y_p_all)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_y_srv.npy"), y_s_all)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_nsn.npy"),   nsn_full)

    # Test arrays: full-length, fill SN=2 rows; non-SN=2 rows are zeros.
    # Also save a separate arr with only SN=2 test rows + their rally_uids for hybrid.
    test_act_full = np.zeros((len(feat_test), N_ACTION))
    test_pt_full  = np.zeros((len(feat_test), N_POINT))
    test_srv_full = np.zeros(len(feat_test))
    test_act_full[sn2_mask_test] = test_act_acc
    test_pt_full [sn2_mask_test] = test_pt_acc
    test_srv_full[sn2_mask_test] = test_srv_acc

    np.save(os.path.join(oof_dir, f"{out_tag}_test_act.npy"), test_act_full)
    np.save(os.path.join(oof_dir, f"{out_tag}_test_pt.npy"),  test_pt_full)
    np.save(os.path.join(oof_dir, f"{out_tag}_test_srv.npy"), test_srv_full)
    np.save(os.path.join(oof_dir, f"{out_tag}_test_rally_uid.npy"),
            feat_test["rally_uid"].values)
    np.save(os.path.join(oof_dir, f"{out_tag}_test_sn2_mask.npy"), sn2_mask_test)

    # Save threshold-opt parameters
    np.save(os.path.join(oof_dir, f"{out_tag}_thresh_t_a.npy"), np.array([t_a]))
    np.save(os.path.join(oof_dir, f"{out_tag}_thresh_w_a.npy"), w_a)
    np.save(os.path.join(oof_dir, f"{out_tag}_thresh_t_p.npy"), np.array([t_p]))
    np.save(os.path.join(oof_dir, f"{out_tag}_thresh_w_p.npy"), w_p)

    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal time: {elapsed:.1f} min")
    print(f"\n{'='*70}")
    print(f"SN=2 OV (base):  {ov:.4f}")
    print(f"SN=2 OV (opt):   {ov_opt:.4f}")
    print(f"  vs baseline V12+V11 on SN=2: {ov_opt - 0.271:+.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
