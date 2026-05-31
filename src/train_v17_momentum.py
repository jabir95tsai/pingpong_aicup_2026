"""V17 Pipeline — V16 backbone + momentum/initiative/pressure features (R-015)

Cloned from train_v16_testhist_aug.py 2026-05-11. Preserves V16's supervised
test-history augmentation backbone. Adds:

  --feature-set v9_momentum       (default): import features_v17_momentum.
                                  Other choices: v9, v9_recvhand for ablation.
  --momentum-groups core|all      (default core): selects v17m feature subset.
                                  core = Groups 1+2+3 (26 features)
                                  all  = Groups 1+2+3+4+5 (41 features)
  --max-folds N                   (default 0): if >0, run only first N folds
                                  of the standard 5-fold partition with full
                                  n_boost. R-011 / R-002 pattern.

Per Codex APPROVE_WITH_FIXES (2026-05-11) on R-015:
  - Same-budget Fold-1 smoke (use --max-folds 1 --n-boost 3000 --es 200).
  - core smoke first; all smoke only if core passes.
  - Pressure scalar fixed-constant only (no fold stats).
  - SOURCE_COLS asserted in features_v17_momentum at build time.

Guard 1 — SGP isolation: aug rows carry serverGetPoint=-1 (dummy); assert all
values exactly -1 and never pass aug rows to the server model.

Guard 2 — fold-stats isolation: compute_global_stats_v9 is called on real
tr_raw only; aug features are built using those fold_stats.

OOF shape is identical to V14/V16: n_samples = 69,712 real training samples.
Aug rows are training-only; they never appear in val or OOF.
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
from data_cleaning import clean_data, STRIKE_ID_MAP

# Dynamic because the organisers may replace the test file. For the 2026-05-06
# reset, test_new.csv has 5668 rows / 1845 rallies -> 3823 aug pairs.
EXPECTED_AUG_ROWS  = None
EXPECTED_AUG_PAIRS = None

N_ACTION       = 19
N_ACTION_TRAIN = 15
N_POINT        = 10

ACTION_EVAL_LABELS = list(range(15))
POINT_EVAL_LABELS  = list(range(10))

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


# ─── Left-right flip augmentation ────────────────────────────────────────────

def build_flip_map(feature_names):
    fn_idx = {n: i for i, n in enumerate(feature_names)}
    pairs  = []
    for k in [1, 2, 3, 4, 5, 6, 8, 10]:
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
    parser.add_argument("--aug",     type=str, default=None,
                        help="Path to test_history_pairs.parquet "
                             "(default: <data_dir>/test_history_pairs.parquet)")
    parser.add_argument("--smoke",   action="store_true")
    parser.add_argument("--folds",   type=int, default=N_FOLDS)
    parser.add_argument("--no-aug",  action="store_true",
                        help="Disable left-right flip augmentation (test-history aug always on)")
    parser.add_argument("--no-stack", action="store_true",
                        help="Disable two-pass action->point stacking")
    parser.add_argument("--skip-cb", action="store_true", default=True,
                        help="Skip CatBoost (default: True). Pass --use-cb to override.")
    parser.add_argument("--use-cb",  dest="skip_cb", action="store_false",
                        help="Include CatBoost (NOT recommended)")
    parser.add_argument("--n-boost", type=int, default=-1,
                        help="Override n_boost (default smoke=200, full=3000)")
    parser.add_argument("--es",      type=int, default=-1,
                        help="Override early stopping rounds")
    parser.add_argument("--tag",     type=str, default="v17_momentum",
                        help="Tag for OOF/test output filenames")
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED,
                        help="Random seed for LGB/XGB/CB model init "
                             "(GroupKFold and flip-aug are deterministic).")
    parser.add_argument("--test-path", type=str, default=None,
                        help="Override TEST_PATH (e.g. data/test_new.csv after the "
                             "2026-05-06 LB reset). Default: config TEST_PATH. The aug "
                             "parquet auto-pick logic uses this path's basename to "
                             "select test_history_pairs_new.parquet vs the legacy file.")
    parser.add_argument("--feature-set", type=str, default="v9_momentum",
                        choices=["v9", "v9_recvhand", "v9_momentum"],
                        help="Feature set: 'v9' (no recvhand), 'v9_recvhand' "
                             "(R-001 baseline), or 'v9_momentum' (R-015 default: "
                             "v9 + recvhand + momentum/initiative/pressure features).")
    parser.add_argument("--momentum-groups", type=str, default="core",
                        choices=["core", "all"],
                        help="(R-015) v17m feature subset. core=Groups 1+2+3 "
                             "(26 features), all=Groups 1+2+3+4+5 (41 features). "
                             "Only used when --feature-set v9_momentum.")
    parser.add_argument("--max-folds", type=int, default=0,
                        help="If >0, run only the first N folds of the standard "
                             "5-fold GroupKFold partition with FULL n_boost. "
                             "R-011 / R-002 pattern. Default 0 = run all --folds. "
                             "For Codex same-budget Fold-1 smoke: --max-folds 1.")
    args = parser.parse_args()

    is_smoke  = args.smoke
    n_folds   = 1 if is_smoke else args.folds
    n_boost   = (200 if is_smoke else 3000) if args.n_boost < 0 else args.n_boost
    es_rounds = (30  if is_smoke else 200)  if args.es      < 0 else args.es
    use_flip  = not args.no_aug
    use_stack = not args.no_stack
    skip_cb   = args.skip_cb
    out_tag   = args.tag
    seed      = args.seed
    np.random.seed(seed)

    # Resolve test + aug path (aug auto-picks based on test basename)
    test_path = args.test_path or TEST_PATH
    aug_path = args.aug
    if aug_path is None:
        data_dir = os.path.dirname(os.path.abspath(test_path))
        aug_name = (
            "test_history_pairs_new.parquet"
            if os.path.basename(test_path) == "test_new.csv"
            else "test_history_pairs.parquet"
        )
        aug_path = os.path.join(data_dir, aug_name)

    t_start = time.time()
    print("=" * 70)
    print(f"V16 PIPELINE (V14 + test-history augmentation) "
          f"{'(SMOKE)' if is_smoke else ''}")
    print(f"  flip={use_flip}  folds={n_folds}  n_boost={n_boost}  "
          f"stack={use_stack}  skip_cb={skip_cb}  es={es_rounds}  seed={seed}")
    print(f"  ACTION macro: 15 classes (0-14, excluding serve 15-18)")
    print("=" * 70)

    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb

    if args.feature_set == "v9_momentum":
        # R-015: set MOMENTUM_GROUPS_ACTIVE before module import
        os.environ["MOMENTUM_GROUPS_ACTIVE"] = args.momentum_groups
        from features_v17_momentum import (
            compute_global_stats_v17_momentum as compute_global_stats_v9,
            build_features_v17_momentum as build_features_v9,
            get_feature_names_v17_momentum as get_feature_names_v9,
        )
        print(f"  Feature set: v9_momentum (R-015, momentum_groups={args.momentum_groups})")
    elif args.feature_set == "v9_recvhand":
        from features_v9_recvhand import (
            compute_global_stats_v9_recvhand as compute_global_stats_v9,
            build_features_v9_recvhand as build_features_v9,
            get_feature_names_v9_recvhand as get_feature_names_v9,
        )
        print("  Feature set: v9_recvhand (R-001 baseline)")
    else:
        from features_v9 import (compute_global_stats_v9, build_features_v9,
                                  get_feature_names_v9)
        print("  Feature set: v9 (no recvhand, no momentum)")

    compute_global_stats_v6 = compute_global_stats_v9
    get_feature_names_v6     = get_feature_names_v9
    def build_features_v6(df, is_train, global_stats_v6, raw_df=None):
        return build_features_v9(df, is_train=is_train,
                                  global_stats_v9=global_stats_v6,
                                  raw_df=raw_df)

    # ── Load and clean data ──────────────────────────────────────────────────
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(test_path)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1

    # ── Load aug parquet and apply same cleaning ──────────────────────────────
    print(f"\n--- Loading aug pairs from: {aug_path} ---")
    aug_raw = pd.read_parquet(aug_path)

    # Guard 1: verify SGP is dummy -1 and is_aug flag is set
    assert (aug_raw["serverGetPoint"] == -1).all(), \
        "GUARD FAIL: aug parquet serverGetPoint contains non-(-1) values. " \
        "Real SGP labels must NOT be present."
    assert (aug_raw["is_aug"] == 1).all(), \
        "GUARD FAIL: aug parquet missing is_aug=1 flag."
    expected_aug_rows = EXPECTED_AUG_ROWS or len(aug_raw)
    expected_aug_pairs = EXPECTED_AUG_PAIRS or (
        len(aug_raw) - aug_raw["rally_uid"].nunique()
    )
    assert len(aug_raw) == expected_aug_rows, \
        f"GUARD FAIL: expected {expected_aug_rows} raw aug rows, got {len(aug_raw)}"
    print(f"  NO_TRUE_TEST_SGP_USED = True")
    print(f"  Aug raw rows: {len(aug_raw)}  (expected {expected_aug_rows})")
    print(f"  Aug expected pairs: {expected_aug_pairs}  (rows - rallies)")

    # Apply same cleaning steps as clean_data (Guard 2: player_map comes from
    # clean_data called on real train only → consistent encoding)
    aug_raw = aug_raw.copy()
    aug_raw["strikeId"] = aug_raw["strikeId"].map(STRIKE_ID_MAP).fillna(0).astype(int)
    for col in ["gamePlayerId", "gamePlayerOtherId"]:
        aug_raw[col] = aug_raw[col].map(player_map).fillna(-1).astype(int)
    aug_raw["numberGame"] = aug_raw["numberGame"].clip(upper=7)
    print(f"  Aug cleaning applied (STRIKE_ID_MAP + player_map + numberGame clip)")

    # ── Preflight ─────────────────────────────────────────────────────────────
    print("\n--- Preflight ---")
    t0 = time.time()
    gs_full   = compute_global_stats_v6(train_df)
    feat_full = build_features_v6(train_df, is_train=True,
                                   global_stats_v6=gs_full,
                                   raw_df=train_df)
    fnames    = get_feature_names_v6(feat_full)
    n_samples = len(feat_full)
    print(f"  {len(fnames)} features, {n_samples} real train samples ({time.time()-t0:.1f}s)")

    flip_pairs = build_flip_map(fnames)
    print(f"  Flip pairs: {len(flip_pairs)}")

    X_all, y_a_all, y_p_all, y_s_all, nsn_all = extract_Xy(feat_full, fnames)
    rally_uids_all = feat_full["rally_uid"].values
    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_all = np.array([rally_to_match.get(r, -1) for r in rally_uids_all])

    # OOF containers (real training samples only — aug rows never written here)
    oof_act    = np.zeros((n_samples, N_ACTION))
    oof_pt     = np.zeros((n_samples, N_POINT))
    oof_srv    = np.zeros(n_samples)
    oof_pt_bin = np.zeros(n_samples)

    gkf    = GroupKFold(n_splits=max(n_folds, 2))
    splits = list(gkf.split(np.arange(n_samples), groups=match_all))
    if is_smoke:
        splits = splits[:1]
    elif args.max_folds and args.max_folds > 0:
        # R-015 same-budget Fold-1 smoke / R-011 partial-fold pattern
        splits = splits[:args.max_folds]
        print(f"  --max-folds {args.max_folds} active: running first "
              f"{len(splits)} of {n_folds} folds with full n_boost.")

    # Test features (one row per test rally)
    feat_test  = build_features_v6(test_df, is_train=False,
                                    global_stats_v6=gs_full,
                                    raw_df=test_df)
    X_test     = feat_test[fnames].values.astype(np.float32)
    nsn_test   = feat_test["next_strikeNumber"].values.astype(np.int32)
    rally_test = feat_test["rally_uid"].values

    test_act_acc  = np.zeros((len(X_test), N_ACTION))
    test_pt_acc   = np.zeros((len(X_test), N_POINT))
    test_srv_acc  = np.zeros(len(X_test))
    test_bin_acc  = np.zeros(len(X_test))
    test_act15_acc = np.zeros((len(X_test), N_ACTION_TRAIN))

    # ── Fold loop ─────────────────────────────────────────────────────────────
    for fold, (tr_idx, val_idx) in enumerate(splits):
        t_fold = time.time()
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{len(splits)}")
        print(f"{'='*60}")

        tr_rallies  = set(rally_uids_all[tr_idx])
        val_rallies = set(rally_uids_all[val_idx])
        tr_raw  = train_df[train_df["rally_uid"].isin(tr_rallies)]
        val_raw = train_df[train_df["rally_uid"].isin(val_rallies)]

        # Guard 2: fold_stats from real train only
        fold_stats = compute_global_stats_v6(tr_raw)
        feat_tr  = build_features_v6(tr_raw, is_train=True,
                                      global_stats_v6=fold_stats,
                                      raw_df=tr_raw)
        feat_val = build_features_v6(val_raw, is_train=True,
                                      global_stats_v6=fold_stats,
                                      raw_df=val_raw)

        X_tr, y_a_tr, y_p_tr, y_s_tr, nsn_tr     = extract_Xy(feat_tr,  fnames)
        X_val, y_a_val, y_p_val, y_s_val, nsn_val = extract_Xy(feat_val, fnames)

        # ── Test-history augmentation (Guard 2: uses fold_stats) ──────────────
        feat_aug_fold = build_features_v6(aug_raw, is_train=True,
                                           global_stats_v6=fold_stats,
                                           raw_df=aug_raw)
        assert len(feat_aug_fold) == expected_aug_pairs, \
            f"GUARD FAIL fold {fold+1}: expected {expected_aug_pairs} aug pairs, " \
            f"got {len(feat_aug_fold)}"
        X_aug, y_a_aug_th, y_p_aug_th, y_s_aug_th, _ = extract_Xy(feat_aug_fold, fnames)

        # Guard 1: assert aug rows have no real SGP labels
        assert (y_s_aug_th == -1).all(), \
            f"GUARD FAIL fold {fold+1}: aug y_serverGetPoint contains non-(-1) values"

        n_nan_aug = np.isnan(X_aug).sum() + np.isinf(X_aug).sum()
        print(f"  Aug pairs: {len(X_aug)} (expected {expected_aug_pairs})  "
              f"NaN/inf={n_nan_aug}  [NO_TRUE_TEST_SGP_USED=True]")

        # ── Left-right flip augmentation on real train rows ────────────────────
        if use_flip:
            X_tr_flip, y_a_flip, y_p_flip, y_s_flip = augment_flip(
                X_tr, y_a_tr, y_p_tr, y_s_tr, flip_pairs)
            print(f"  Flip aug: {len(X_tr)} -> {len(X_tr_flip)} real train samples")
        else:
            X_tr_flip, y_a_flip, y_p_flip, y_s_flip = X_tr, y_a_tr, y_p_tr, y_s_tr

        # Combined train for action + point models (flip_real + test_hist)
        X_tr_combined  = np.vstack([X_tr_flip, X_aug])
        y_a_combined   = np.concatenate([y_a_flip, y_a_aug_th])
        y_p_combined   = np.concatenate([y_p_flip, y_p_aug_th])

        sw_a = np.array([ACTION_CW.get(int(c), 1.0) for c in y_a_combined], dtype=np.float32)
        sw_p = np.array([POINT_CW.get(int(c),  1.0) for c in y_p_combined], dtype=np.float32)

        # ══════════════════════════════════════════════════════════════════════
        # PASS A — ACTION models (LGB + XGB [+ CB])
        # ══════════════════════════════════════════════════════════════════════
        lgb_a_p = dict(n_estimators=n_boost, learning_rate=0.04,
                       num_leaves=127, max_depth=9, min_child_samples=8,
                       subsample=0.8, colsample_bytree=0.7,
                       reg_alpha=0.1, reg_lambda=1.0,
                       objective="multiclass", metric="multi_logloss",
                       num_class=N_ACTION_TRAIN, random_state=seed,
                       n_jobs=-1, verbose=-1)
        lgb_a = lgb.train(lgb_a_p,
            lgb.Dataset(X_tr_combined, label=y_a_combined, weight=sw_a),
            valid_sets=[lgb.Dataset(X_val, label=y_a_val)],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])

        xgb_a = xgb.XGBClassifier(
            n_estimators=n_boost, learning_rate=0.04, max_depth=7,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            objective="multi:softprob", num_class=N_ACTION_TRAIN,
            eval_metric="mlogloss", early_stopping_rounds=es_rounds,
            random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist")
        xgb_a.fit(X_tr_combined, y_a_combined, sample_weight=sw_a,
                  eval_set=[(X_val, y_a_val)], verbose=False)

        if skip_cb:
            cb_a = None
        else:
            cb_a = CatBoostClassifier(
                iterations=n_boost, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
                loss_function="MultiClass", classes_count=N_ACTION_TRAIN,
                random_seed=seed, verbose=False, allow_writing_files=False,
                early_stopping_rounds=es_rounds)
            cb_a.fit(X_tr_combined, y_a_combined, sample_weight=sw_a,
                     eval_set=(X_val, y_a_val), use_best_model=True)

        pa_val_lgb = lgb_a.predict(X_val)
        pa_val_xgb = pad_proba(xgb_a.predict_proba(X_val), xgb_a.classes_, N_ACTION_TRAIN)
        if cb_a is not None:
            pa_val_cb = pad_proba(cb_a.predict_proba(X_val), cb_a.classes_, N_ACTION_TRAIN)
            pa_val_15 = (pa_val_lgb + pa_val_xgb + pa_val_cb) / 3.0
        else:
            pa_val_15 = (pa_val_lgb + pa_val_xgb) / 2.0

        pa_val_19 = extend_action(pa_val_15)
        pa_ruled  = apply_action_rules(pa_val_19, nsn_val)
        f1_a_val  = action_macro_f1(y_a_val, pa_ruled)
        print(f"  [Action] F1={f1_a_val:.4f}")

        val_mask = np.isin(rally_uids_all, list(set(rally_uids_all[val_idx])))
        oof_act[val_mask] = pa_ruled

        # Action probs for stacking — real train rows (predict on their own training data)
        if use_stack:
            pa_tr_lgb = lgb_a.predict(X_tr)
            pa_tr_xgb = pad_proba(xgb_a.predict_proba(X_tr), xgb_a.classes_, N_ACTION_TRAIN)
            if cb_a is not None:
                pa_tr_cb = pad_proba(cb_a.predict_proba(X_tr), cb_a.classes_, N_ACTION_TRAIN)
                pa_tr_15 = (pa_tr_lgb + pa_tr_xgb + pa_tr_cb) / 3.0
            else:
                pa_tr_15 = (pa_tr_lgb + pa_tr_xgb) / 2.0
            # Duplicate for flipped half (action class unchanged by flip)
            if use_flip:
                pa_tr_flip_15 = np.vstack([pa_tr_15, pa_tr_15])
            else:
                pa_tr_flip_15 = pa_tr_15

            # Action probs for test-history aug rows (model predicts on training data —
            # standard within-fold stacking; point model still trained on separate labels)
            pa_aug_lgb = lgb_a.predict(X_aug)
            pa_aug_xgb = pad_proba(xgb_a.predict_proba(X_aug), xgb_a.classes_, N_ACTION_TRAIN)
            if cb_a is not None:
                pa_aug_cb = pad_proba(cb_a.predict_proba(X_aug), cb_a.classes_, N_ACTION_TRAIN)
                pa_aug_15 = (pa_aug_lgb + pa_aug_xgb + pa_aug_cb) / 3.0
            else:
                pa_aug_15 = (pa_aug_lgb + pa_aug_xgb) / 2.0

        # Test action probs
        pa_test_lgb = lgb_a.predict(X_test)
        pa_test_xgb = pad_proba(xgb_a.predict_proba(X_test), xgb_a.classes_, N_ACTION_TRAIN)
        if cb_a is not None:
            pa_test_cb = pad_proba(cb_a.predict_proba(X_test), cb_a.classes_, N_ACTION_TRAIN)
            pa_test_15 = (pa_test_lgb + pa_test_xgb + pa_test_cb) / 3.0
        else:
            pa_test_15 = (pa_test_lgb + pa_test_xgb) / 2.0
        test_act15_acc += pa_test_15 / len(splits)
        test_act_acc   += extend_action(pa_test_15) / len(splits)

        # ══════════════════════════════════════════════════════════════════════
        # PASS B — POINT models (binary miss + 10-class)
        #          with action probs as stacking features
        # ══════════════════════════════════════════════════════════════════════
        if use_stack:
            X_tr_ext     = np.hstack([X_tr,      pa_tr_15])
            X_tr_flip_ext = np.hstack([X_tr_flip, pa_tr_flip_15])
            X_aug_ext    = np.hstack([X_aug,      pa_aug_15])
            X_val_ext    = np.hstack([X_val,      pa_val_15])
            # Combined: flip real train + test-history aug
            X_pt_combined_ext = np.vstack([X_tr_flip_ext, X_aug_ext])
        else:
            X_tr_ext          = X_tr
            X_tr_flip_ext     = X_tr_flip
            X_aug_ext         = X_aug
            X_val_ext         = X_val
            X_pt_combined_ext = np.vstack([X_tr_flip, X_aug])

        y_miss_combined = (y_p_combined == 0).astype(np.int32)
        y_miss_val      = (y_p_val == 0).astype(np.int32)

        # ── POINT binary (miss vs non-miss) ──────────────────────────────────
        lgb_pb_p = dict(n_estimators=n_boost, learning_rate=0.04,
                        num_leaves=63, max_depth=7, min_child_samples=10,
                        subsample=0.8, colsample_bytree=0.7,
                        objective="binary", metric="binary_logloss",
                        random_state=seed, n_jobs=-1, verbose=-1)
        lgb_pb = lgb.train(lgb_pb_p,
            lgb.Dataset(X_pt_combined_ext, label=y_miss_combined.astype(np.float32)),
            valid_sets=[lgb.Dataset(X_val_ext, label=y_miss_val.astype(np.float32))],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        pb_val = lgb_pb.predict(X_val_ext)
        oof_pt_bin[val_mask] = pb_val

        # ── POINT 10-class (LGB + XGB [+ CB]) ────────────────────────────────
        lgb_p_p = dict(n_estimators=n_boost, learning_rate=0.04,
                       num_leaves=127, max_depth=9, min_child_samples=5,
                       subsample=0.8, colsample_bytree=0.7,
                       reg_alpha=0.1, reg_lambda=1.0,
                       objective="multiclass", metric="multi_logloss",
                       num_class=N_POINT, random_state=seed,
                       n_jobs=-1, verbose=-1)
        lgb_p = lgb.train(lgb_p_p,
            lgb.Dataset(X_pt_combined_ext, label=y_p_combined, weight=sw_p),
            valid_sets=[lgb.Dataset(X_val_ext, label=y_p_val)],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        pp_lgb = lgb_p.predict(X_val_ext)

        xgb_p = xgb.XGBClassifier(
            n_estimators=n_boost, learning_rate=0.04, max_depth=7,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            objective="multi:softprob", num_class=N_POINT,
            eval_metric="mlogloss", early_stopping_rounds=es_rounds,
            random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist")
        xgb_p.fit(X_pt_combined_ext, y_p_combined, sample_weight=sw_p,
                  eval_set=[(X_val_ext, y_p_val)], verbose=False)
        pp_xgb = xgb_p.predict_proba(X_val_ext)

        if skip_cb:
            cb_p = None
        else:
            cb_p = CatBoostClassifier(
                iterations=n_boost, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
                loss_function="MultiClass", classes_count=N_POINT,
                random_seed=seed, verbose=False, allow_writing_files=False,
                early_stopping_rounds=es_rounds)
            cb_p.fit(X_pt_combined_ext, y_p_combined, sample_weight=sw_p,
                     eval_set=(X_val_ext, y_p_val), use_best_model=True)

        pp_xgb  = pad_proba(pp_xgb, xgb_p.classes_, N_POINT)
        if cb_p is not None:
            pp_cb    = pad_proba(cb_p.predict_proba(X_val_ext), cb_p.classes_, N_POINT)
            pp_blend = (pp_lgb + pp_xgb + pp_cb) / 3.0
        else:
            pp_blend = (pp_lgb + pp_xgb) / 2.0
        pp_2stage = blend_two_stage(pp_blend, pb_val)
        f1_p_val  = point_macro_f1(y_p_val, pp_2stage)
        print(f"  [Point]  F1={f1_p_val:.4f}")
        oof_pt[val_mask] = pp_2stage

        if use_stack:
            X_test_ext = np.hstack([X_test, test_act15_acc * len(splits) / (fold + 1)])
        else:
            X_test_ext = X_test

        if cb_p is not None:
            test_pt_acc += (lgb_p.predict(X_test_ext) +
                             pad_proba(xgb_p.predict_proba(X_test_ext), xgb_p.classes_, N_POINT) +
                             pad_proba(cb_p.predict_proba(X_test_ext),  cb_p.classes_,  N_POINT)
                             ) / 3.0 / len(splits)
        else:
            test_pt_acc += (lgb_p.predict(X_test_ext) +
                             pad_proba(xgb_p.predict_proba(X_test_ext), xgb_p.classes_, N_POINT)
                             ) / 2.0 / len(splits)
        test_bin_acc += lgb_pb.predict(X_test_ext) / len(splits)

        # ══════════════════════════════════════════════════════════════════════
        # SERVER model — Guard 1: real train rows ONLY, no test-history aug
        # ══════════════════════════════════════════════════════════════════════
        if use_stack:
            X_tr_srv_ext  = np.hstack([X_tr_flip, pa_tr_flip_15])
            X_val_srv_ext = np.hstack([X_val,     pa_val_15])
        else:
            X_tr_srv_ext  = X_tr_flip
            X_val_srv_ext = X_val

        lgb_s_p = dict(n_estimators=n_boost, learning_rate=0.04,
                       num_leaves=63, max_depth=7, min_child_samples=15,
                       subsample=0.8, colsample_bytree=0.7,
                       objective="binary", metric="auc",
                       random_state=seed, n_jobs=-1, verbose=-1)
        lgb_s = lgb.train(lgb_s_p,
            lgb.Dataset(X_tr_srv_ext, label=y_s_flip.astype(np.float32)),
            valid_sets=[lgb.Dataset(X_val_srv_ext, label=y_s_val.astype(np.float32))],
            callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                       lgb.log_evaluation(-1)])
        ps_lgb = lgb_s.predict(X_val_srv_ext)

        xgb_s = xgb.XGBClassifier(
            n_estimators=n_boost, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.7,
            objective="binary:logistic", eval_metric="auc",
            early_stopping_rounds=es_rounds,
            random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist")
        xgb_s.fit(X_tr_srv_ext, y_s_flip, eval_set=[(X_val_srv_ext, y_s_val)], verbose=False)
        ps_xgb = xgb_s.predict_proba(X_val_srv_ext)[:, 1]

        ps_blend = (ps_lgb + ps_xgb) / 2.0
        auc_val  = roc_auc_score(y_s_val, ps_blend)
        print(f"  [Server] AUC={auc_val:.4f}")
        oof_srv[val_mask] = ps_blend

        ov_fold = 0.4*f1_a_val + 0.4*f1_p_val + 0.2*auc_val
        print(f"\n  FOLD OV={ov_fold:.4f}  [{time.time()-t_fold:.0f}s]")

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

    # Full-run assertion: all real training rows must be covered in OOF
    # Skip when --max-folds N < N_FOLDS (R-015 same-budget Fold-1 smoke pattern)
    if not is_smoke and n_folds == N_FOLDS and not (args.max_folds and args.max_folds > 0):
        assert n_oof == n_samples, \
            f"GUARD FAIL: OOF mask={n_oof}, expected n_samples={n_samples}"
        print(f"  ASSERT PASS: OOF mask == {n_samples} real train rows")
    elif args.max_folds and args.max_folds > 0:
        print(f"  ASSERT SKIP: --max-folds {args.max_folds} active; "
              f"OOF mask={n_oof} (subset OK, full-coverage assertion deferred to R-016 full run)")

    oof_act_ruled = apply_action_rules(oof_act[oof_mask], nsn_all[oof_mask])
    f1_a_oof  = action_macro_f1(y_a_all[oof_mask], oof_act_ruled)
    f1_p_oof  = point_macro_f1(y_p_all[oof_mask], oof_pt[oof_mask])
    auc_oof   = roc_auc_score(y_s_all[oof_mask], oof_srv[oof_mask])
    ov_oof    = 0.4*f1_a_oof + 0.4*f1_p_oof + 0.2*auc_oof
    print(f"  Base:  action={f1_a_oof:.4f}  point={f1_p_oof:.4f}  AUC={auc_oof:.4f}  OV={ov_oof:.4f}")

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

    if use_stack:
        X_test_ext_final = np.hstack([X_test, test_act15_acc])
        if cb_p is not None:
            test_pt_acc_final = (lgb_p.predict(X_test_ext_final) +
                                  pad_proba(xgb_p.predict_proba(X_test_ext_final), xgb_p.classes_, N_POINT) +
                                  pad_proba(cb_p.predict_proba(X_test_ext_final), cb_p.classes_, N_POINT)
                                  ) / 3.0
        else:
            test_pt_acc_final = (lgb_p.predict(X_test_ext_final) +
                                  pad_proba(xgb_p.predict_proba(X_test_ext_final), xgb_p.classes_, N_POINT)
                                  ) / 2.0
        test_bin_acc_final = lgb_pb.predict(X_test_ext_final)
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

    sub_cont = pd.DataFrame({
        "rally_uid":      rally_test,
        "actionId":       pred_act,
        "pointId":        pred_pt,
        "serverGetPoint": test_srv_acc,
    })
    out_path_cont = os.path.join(SUBMISSION_DIR, f"submission_{out_tag}.csv")
    sub_cont.to_csv(out_path_cont, index=False, lineterminator="\n")

    sub_bin = sub_cont.copy()
    sub_bin["serverGetPoint"] = (test_srv_acc >= 0.5).astype(int)
    sub_bin.to_csv(os.path.join(SUBMISSION_DIR, f"submission_{out_tag}_binary_srv.csv"),
                    index=False, lineterminator="\n")

    oof_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")
    os.makedirs(oof_dir, exist_ok=True)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_act.npy"),   oof_act)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_pt.npy"),    oof_pt)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_srv.npy"),   oof_srv)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_pt_bin.npy"), oof_pt_bin)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_mask.npy"),  oof_mask)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_y_act.npy"), y_a_all)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_y_pt.npy"),  y_p_all)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_y_srv.npy"), y_s_all)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_nsn.npy"),   nsn_all)
    np.save(os.path.join(oof_dir, f"{out_tag}_test_act.npy"),  test_act_acc)
    np.save(os.path.join(oof_dir, f"{out_tag}_test_pt.npy"),   test_pt_acc)
    np.save(os.path.join(oof_dir, f"{out_tag}_test_srv.npy"),  test_srv_acc)
    np.save(os.path.join(oof_dir, f"{out_tag}_test_rally_uid.npy"), rally_test)
    print(f"  Saved {out_tag} OOF + test predictions to {oof_dir}")
    print(f"  actionId dist:  {dict(pd.Series(pred_act).value_counts().sort_index())}")
    print(f"  pointId dist:   {dict(pd.Series(pred_pt).value_counts().sort_index())}")
    print(f"  SGP mean/std:   mean={test_srv_acc.mean():.4f}  std={test_srv_acc.std():.4f}")
    print(f"  Saved: {out_path_cont}")

    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal time: {elapsed:.1f} min")
    print(f"\n{'='*70}")
    print(f"FINAL OV (base):  {ov_oof:.4f}")
    print(f"FINAL OV (opt):   {ov_opt:.4f}")
    print(f"{'='*70}")
    print(f"\n--- V16 Smoke/Run assertions ---")
    print(f"  aug_rows_in_parquet : {expected_aug_rows}")
    print(f"  aug_pairs_per_fold  : {expected_aug_pairs}")
    print(f"  oof_mask_sum        : {n_oof}  (real train rows in OOF)")
    print(f"  NO_TRUE_TEST_SGP_USED = True")
    print(f"  server_aug_rows     : 0  (Guard 1 — no aug rows in server path)")


if __name__ == "__main__":
    main()
