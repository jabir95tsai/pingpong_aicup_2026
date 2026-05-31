"""V14 Pipeline — V12 + features_v9 (joint serve-receive priors))

Identical architecture to V10 (Two-Pass Action→Point Stacking) but uses
features_v9 which adds 25 joint serve-receive prior features on top of V6:
  - P(point_depth | prev_action, phase) — 4 features
  - P(point_side  | prev_action, phase) — 4 features
  - P(is_valid_landing | prev_action, [prev_point], phase) — 2 features
  - P(point_depth | prev_action, prev_point, phase) — 4 features (refined)
  - P(pt=0 | prev_action, [phase]) — 2 terminal priors
  - Trigram (prev2, prev1, phase) → next_action — 4 features
  - Receive prior (SN=2, conditioned on serve_action × sex) — 4 features

V12 also saves fold-aware OOF predictions for downstream slice analysis and
blend with V11 transformer.
"""
import sys, os, time, warnings, gc, argparse, json
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


# NOTE 2026-05-23: A full-prefix expansion helper was considered as part of
# Codex Tier 3 ("train full-prefix sequence expansion") but verified REDUNDANT:
# features_v3.build_features_v3 (inherited by v5/v6/v9/v15feat) already iterates
# `for target_idx in range(1, len(group))` per train rally, producing N-1
# feature rows per rally with N shots. The standard 69712 OOF baseline IS the
# full-prefix-expanded training set. The flag was removed from --argparse below
# to avoid double-expansion.


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
    parser.add_argument("--skip-cb",  action="store_true", default=True,
                        help="Skip CatBoost models (default: True per STRATEGY no-CB policy). "
                             "Pass --use-cb to override.")
    parser.add_argument("--use-cb",   dest="skip_cb", action="store_false",
                        help="Override default: include CatBoost (NOT recommended; LB shows degradation)")
    parser.add_argument("--n-boost",  type=int, default=-1,
                        help="Override n_boost (default smoke=200, full=3000)")
    parser.add_argument("--es",       type=int, default=-1,
                        help="Override early stopping rounds")
    parser.add_argument("--tag",      type=str, default="v14",
                        help="Tag for OOF/test output filenames (default: v12)")
    parser.add_argument("--seed",     type=int, default=RANDOM_SEED,
                        help=f"Global random seed (default: {RANDOM_SEED}). "
                             "Controls numpy, LGB, XGB, and CB random states.")
    parser.add_argument("--test-path", type=str, default=None,
                        help="Override TEST_PATH (e.g. data/test_new.csv after the "
                             "2026-05-06 LB reset). Default: config TEST_PATH.")
    parser.add_argument("--feature-set", type=str, default="v9",
                        choices=["v9", "v9_recvhand", "v9_recvprofile",
                                 "v15feat", "v15feat_b", "v15feat_c", "v15feat_d",
                                 "v15feat_e", "v15feat_e_nomismatch",
                                 "v16match", "v16match_v2"],
                        help="Feature set: 'v9' (default, baseline), "
                             "'v9_recvhand' (adds recv_hand_est, R-001), "
                             "'v9_recvprofile' (R-011: v9 + recv_hand_est + "
                             "4 receiver-mode axes; see features_v9_recvprofile.py), "
                             "'v15feat' (R-029a: v9 + 36 prefix aggregate features — "
                             "per-class freqs, entropy, dominance, streaks; "
                             "see features_v15feat.py), "
                             "'v15feat_b' (R-029b: v15feat + 33 empirical "
                             "transition priors; see features_v15feat_b.py), or "
                             "'v16match' (R-032: v15feat + 40 LORO cross-rally "
                             "match-context features — attacks player de-id; see "
                             "features_v16match.py).")
    parser.add_argument("--recvprofile-axes", type=str,
                        default="action,point,strength,spin",
                        help="(R-011) Comma-separated subset of "
                             "{action,point,strength,spin} for ablation. "
                             "Only used when --feature-set v9_recvprofile.")
    parser.add_argument("--max-folds", type=int, default=0,
                        help="If > 0, run only the first N folds of the standard "
                             "GroupKFold(--folds) partition with full epochs / "
                             "n_boost. Used for stop-gate dry runs (R-002 R-011 "
                             "pattern). Default 0 = run all --folds.")
    parser.add_argument("--include-old-test", type=str, default=None,
                        help="(NEW 2026-05-13) Path to old test.csv. Per AICUP "
                             "organizers' announcement allowing old test as training data.")
    parser.add_argument("--pseudo-parquet", type=str, default=None,
                        help="Path to pseudo-label parquet (R-009 V1). When set, kept rows "
                             "are appended to action/point training sets per --pseudo-mode. "
                             "Pseudo rows are NEVER added to server training, NEVER flip-augmented, "
                             "and NEVER appear in saved OOF arrays.")
    parser.add_argument("--pseudo-mode", type=str, default="action_and_point",
                        choices=["action_and_point", "action_only"],
                        help="Pseudo-row policy. 'action_and_point' (V1a) uses pseudo "
                             "for both action and point losses. 'action_only' (V1b, "
                             "NOT approved by Codex 2026-05-10) uses pseudo for action "
                             "only; point loss skips pseudo rows.")
    parser.add_argument("--pseudo-weight", type=float, default=0.3,
                        help="Flat sample weight for pseudo rows (Codex V1a: 0.3). NOT "
                             "multiplied by ACTION_CW/POINT_CW; treated as absolute weight.")
    # NOTE 2026-05-23: --full-prefix flag was considered but REVERTED. The existing
    # features_v3.build_features_v3 (inherited by all downstream feature modules)
    # already iterates `for target_idx in range(1, len(group))` per train rally,
    # producing N-1 feature rows per rally with N shots. The 69712 OOF baseline
    # IS the full-prefix-expanded training set. Adding a --full-prefix flag
    # would double-expand (already-expanded features × synthetic truncations).
    # ── R-203 (2026-05-29): Focal CE + Cui CB weights for ACTION models ───────
    parser.add_argument("--r203-focal", action="store_true",
                        help="(R-203) Replace LGB action objective with focal CE "
                             "(gamma) using Cui et al. CB class weights. Only "
                             "affects ACTION-task LGB model (XGB stays on CE). "
                             "Removes sample-weight injection of ACTION_CW "
                             "(class weighting now lives inside the focal alpha).")
    parser.add_argument("--r203-gamma", type=float, default=2.0,
                        help="(R-203) Focal exponent gamma (default: 2.0)")
    parser.add_argument("--r203-cb-beta", type=float, default=0.999,
                        help="(R-203) Cui CB beta hyperparameter (default: 0.999)")
    parser.add_argument("--r203-boost-classes", type=str, default="1,5,6,13",
                        help="(R-203) Comma-separated action class ids to apply "
                             "additional focal-boost factor (default: 1,5,6,13 "
                             "= Loop + push-family per spec).")
    parser.add_argument("--r203-boost-factor", type=float, default=1.5,
                        help="(R-203) Multiplicative boost for --r203-boost-classes "
                             "(default 1.5)")
    args = parser.parse_args()

    is_smoke    = args.smoke
    n_folds     = 1 if is_smoke else args.folds
    n_boost     = (200 if is_smoke else 3000) if args.n_boost < 0 else args.n_boost
    es_rounds   = (30  if is_smoke else 200)  if args.es      < 0 else args.es
    use_aug     = not args.no_aug
    use_stack   = not args.no_stack
    skip_cb     = args.skip_cb
    out_tag     = args.tag
    seed        = args.seed

    np.random.seed(seed)

    t_start = time.time()
    print("=" * 70)
    print(f"V14 PIPELINE (V12 + features_v9) {'(SMOKE)' if is_smoke else ''}")
    print(f"  aug={use_aug}  folds={n_folds}  n_boost={n_boost}  stack={use_stack}  "
          f"skip_cb={skip_cb}  es={es_rounds}  seed={seed}")
    print(f"  ACTION macro: 15 classes (0-14, excluding serve 15-18)")
    print("=" * 70)

    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    # R-203 imports (no-op unless --r203-focal)
    from r203_focal_obj import (
        cui_cb_weights, apply_focal_boost,
        make_focal_multiclass_obj, make_focal_multiclass_eval,
    )
    if args.feature_set == "v9_recvhand":
        from features_v9_recvhand import (
            compute_global_stats_v9_recvhand as compute_global_stats_v9,
            build_features_v9_recvhand as build_features_v9,
            get_feature_names_v9_recvhand as get_feature_names_v9,
        )
        print("  Feature set: v9_recvhand (v9 + recv_hand_est)")
    elif args.feature_set == "v9_recvside":
        # R-211: v9 + recv_side_est (receiver's own prior point-SIDE mode).
        # Stronger receiver-axis signal than recvhand's handId-mode proxy
        # (probe side-spread +0.147). Prefix-only, within-rally, hard-rule clean.
        from features_v9_recvside import (
            compute_global_stats_v9_recvside as compute_global_stats_v9,
            build_features_v9_recvside as build_features_v9,
            get_feature_names_v9_recvside as get_feature_names_v9,
        )
        print("  Feature set: v9_recvside (v9 + recv_side_est, R-211)")
    elif args.feature_set == "v9_recvprofile":
        # R-011: set ablation axes via env var BEFORE module import.
        os.environ["RECVPROFILE_AXES"] = args.recvprofile_axes
        from features_v9_recvprofile import (
            compute_global_stats_v9_recvprofile as compute_global_stats_v9,
            build_features_v9_recvprofile as build_features_v9,
            get_feature_names_v9_recvprofile as get_feature_names_v9,
        )
        print(f"  Feature set: v9_recvprofile (v9 + recv_hand_est + "
              f"axes={args.recvprofile_axes})")
    elif args.feature_set == "v15feat":
        # R-029a: v9 + 36 prefix aggregate features (per-class freqs +
        # entropy/dominance/streaks). Clean-room from teammate audit Batch A.
        from features_v15feat import (
            compute_global_stats_v15feat as compute_global_stats_v9,
            build_features_v15feat as build_features_v9,
            get_feature_names_v15feat as get_feature_names_v9,
        )
        print("  Feature set: v15feat (v9 + 36 prefix aggregates: per-class "
              "freqs + entropy/dominance + tail streaks)")
    elif args.feature_set == "v15feat_b":
        # R-029b: v15feat + 33 empirical transition priors. Clean-room
        # from teammate audit Batch B. Per-fold computation = leak-safe.
        from features_v15feat_b import (
            compute_global_stats_v15feat_b as compute_global_stats_v9,
            build_features_v15feat_b as build_features_v9,
            get_feature_names_v15feat_b as get_feature_names_v9,
        )
        print("  Feature set: v15feat_b (v15feat + 33 transition priors: "
              "P(next_action|last_action,is_serve_side) + P(next_point|last_action,last_point))")
    elif args.feature_set == "v15feat_c":
        # R-047: v15feat_b + 8 teammate-v8 score-pressure features
        # (is_serve_side, is_deuce, match_point_*, total_points,
        # points_to_win_*, score_lead_abs). New B-feature subclass.
        from features_v15feat_c import (
            compute_global_stats_v15feat_c as compute_global_stats_v9,
            build_features_v15feat_c as build_features_v9,
            get_feature_names_v15feat_c as get_feature_names_v9,
        )
        print("  Feature set: v15feat_c (v15feat_b + 8 score-pressure features)")
    elif args.feature_set == "v15feat_d":
        # R-064 (Codex APPROVE_WITH_FIXES 2026-05-23): v15feat + 13 spin-aware
        # features (5 smoothed spin priors α=20 + 4 last-spin physics flags +
        # 4 serve_spin_class one-hot). User insight: position×action constrains
        # next-shot spin, which physically constrains receiver's counter.
        from features_v15feat_d import (
            compute_global_stats_v15feat_d as compute_global_stats_v9,
            build_features_v15feat_d as build_features_v9,
            get_feature_names_v15feat_d as get_feature_names_v9,
        )
        print("  Feature set: v15feat_d (v15feat + 13 spin-aware features: "
              "5 smoothed priors P(next_spin|last_act,last_pos) + 4 physics flags "
              "+ 4 serve_spin_class one-hot)")
    elif args.feature_set == "v15feat_e":
        # R-070 (Codex APPROVE_WITH_FIXES 2026-05-24): v15feat + 7 neutral
        # stroke-position movement features (mismatch proxy, 2D pointId
        # side/depth decomposition + missingness flags, lateral gap, optional
        # interaction). User intuition: cross-court reach + far follow-up =
        # harder next shot.
        from features_v15feat_e import (
            compute_global_stats_v15feat_e as compute_global_stats_v9,
            build_features_v15feat_e as build_features_v9,
            get_feature_names_v15feat_e as get_feature_names_v9,
        )
        print("  Feature set: v15feat_e (v15feat + 7 neutral movement/position features)")
    elif args.feature_set == "v15feat_e_nomismatch":
        # R-070 ablation (Codex 2026-05-24): drop mismatch_proxy + interaction.
        # Keep only 5 point side/depth/gap/missingness features. Tests if the
        # SN≤2 regression in the 7-feature smoke came from the mismatch proxy.
        from features_v15feat_e_nomismatch import (
            compute_global_stats_v15feat_e_nomismatch as compute_global_stats_v9,
            build_features_v15feat_e_nomismatch as build_features_v9,
            get_feature_names_v15feat_e_nomismatch as get_feature_names_v9,
        )
        print("  Feature set: v15feat_e_nomismatch (v15feat + 5 point/gap/missingness features)")
    elif args.feature_set == "v16match":
        # R-032 v1 (Codex BLOCKED — use v16match_v2 instead).
        from features_v16match import (
            compute_global_stats_v16match as compute_global_stats_v9,
            build_features_v16match as build_features_v9,
            get_feature_names_v16match as get_feature_names_v9,
        )
        print("  Feature set: v16match v1 (Codex-BLOCKED; prefer v16match_v2)")
    elif args.feature_set == "v16match_v2":
        # R-032 v2 (Codex APPROVE_WITH_FIXES 2026-05-21):
        # v9 + 33 LORO features grouped by (match, unordered_player_pair).
        # Family C dropped from model features; prefix-cap K=3.
        from features_v16match_v2 import (
            compute_global_stats_v16match_v2 as compute_global_stats_v9,
            build_features_v16match_v2 as build_features_v9,
            get_feature_names_v16match_v2 as get_feature_names_v9,
        )
        print("  Feature set: v16match_v2 (v9 + 33 LORO match-pair features)")
    else:
        from features_v9 import (compute_global_stats_v9, build_features_v9,
                                  get_feature_names_v9)
        print("  Feature set: v9 (baseline)")
    # Wrappers so existing call sites that use v6 kwarg names still work
    compute_global_stats_v6 = compute_global_stats_v9
    get_feature_names_v6     = get_feature_names_v9
    def build_features_v6(df, is_train, global_stats_v6, raw_df=None):
        return build_features_v9(df, is_train=is_train,
                                  global_stats_v9=global_stats_v6,
                                  raw_df=raw_df)

    test_path = args.test_path or TEST_PATH
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(test_path)
    if args.include_old_test:
        old_test = pd.read_csv(args.include_old_test)
        n_before = len(raw_train)
        required_cols = list(raw_train.columns)
        missing_cols = [c for c in required_cols if c not in old_test.columns]
        if missing_cols:
            raise ValueError(f"old test missing columns: {missing_cols}")
        raw_train = pd.concat([raw_train, old_test[required_cols]], ignore_index=True)
        print(f"  [include-old-test] Added {len(raw_train) - n_before} rows from {args.include_old_test} "
              f"({old_test['rally_uid'].nunique()} rallies, {old_test['match'].nunique()} matches)")
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
    elif args.max_folds and args.max_folds > 0:
        # R-011 / R-002 pattern: run only the first N folds of the standard
        # 5-fold GroupKFold partition (NOT a different partition). Full epochs.
        splits = splits[:args.max_folds]
        print(f"  --max-folds {args.max_folds} active: running first "
              f"{len(splits)} of {n_folds} folds with full n_boost.")

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

    # ── Pseudo-label loading (R-009 V1a, optional) ──────────────────────────
    # Per Codex APPROVE_WITH_FIXES (2026-05-10):
    #  - kept rows joined to feat_test inference rows by rally_uid
    #  - server training EXCLUDES pseudo entirely
    #  - flip-aug NEVER applied to pseudo rows
    #  - OOF arrays MUST stay length n_samples (real train rows only)
    pseudo_X       = None
    pseudo_y_act   = None
    pseudo_y_pt    = None
    pseudo_act_p   = None
    pseudo_pt_p    = None
    n_pseudo       = 0
    pseudo_mode    = args.pseudo_mode
    pseudo_weight  = args.pseudo_weight
    if args.pseudo_parquet:
        print(f"\n--- Loading pseudo-label parquet ---")
        print(f"  Path: {args.pseudo_parquet}")
        pdf = pd.read_parquet(args.pseudo_parquet)
        pdf_kept = pdf[pdf["kept"]].reset_index(drop=True)
        n_pseudo = len(pdf_kept)
        print(f"  Total parquet rows: {len(pdf)}  Kept rows: {n_pseudo}")
        print(f"  Pseudo mode: {pseudo_mode}  Pseudo weight: {pseudo_weight}")
        # Verify SGP sentinel
        if "serverGetPoint" in pdf_kept.columns:
            srv_vals = pdf_kept["serverGetPoint"].unique()
            assert set(srv_vals.tolist()).issubset({-1}), \
                f"Pseudo parquet has non-sentinel serverGetPoint values: {srv_vals}"
            print(f"  serverGetPoint sentinel verified: all -1 ({len(srv_vals)} unique)")
        # Verify manifest exists
        manifest_path = args.pseudo_parquet + ".manifest.json"
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"  Manifest loaded: teacher={manifest.get('teacher_submission')}")
            print(f"  Manifest test_rally_uid_sha256: {manifest.get('test_rally_uid_sha256')}")
        else:
            print(f"  WARN: manifest not found at {manifest_path}")
        # Build pseudo features by joining kept rows to feat_test
        test_rally_to_idx = {r: i for i, r in enumerate(rally_test)}
        missing = [r for r in pdf_kept["rally_uid"] if r not in test_rally_to_idx]
        assert not missing, f"{len(missing)} pseudo rally_uids not in feat_test"
        pseudo_idx = np.array([test_rally_to_idx[r] for r in pdf_kept["rally_uid"]])
        pseudo_X    = X_test[pseudo_idx].astype(np.float32, copy=True)
        pseudo_y_act = pdf_kept["pseudo_actionId"].values.astype(np.int64)
        pseudo_y_pt  = pdf_kept["pseudo_pointId"].values.astype(np.int64)
        pseudo_act_p = pdf_kept["act_top1_p"].values.astype(np.float32)
        pseudo_pt_p  = pdf_kept["pt_top1_p"].values.astype(np.float32)
        print(f"  Pseudo features built: {pseudo_X.shape}  (joined from feat_test)")
        # Class distribution log
        from collections import Counter
        act_dist = Counter(int(c) for c in pseudo_y_act)
        pt_dist  = Counter(int(c) for c in pseudo_y_pt)
        print(f"  Pseudo action class dist: {dict(sorted(act_dist.items()))}")
        print(f"  Pseudo point  class dist: {dict(sorted(pt_dist.items()))}")
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
                                      raw_df=val_raw)  # lag lookup uses val_raw so extra-lag
                                                       # one-hots see the validation rally history
                                                       # (mirrors how test features are built)

        X_tr, y_a_tr, y_p_tr, y_s_tr, nsn_tr     = extract_Xy(feat_tr,  fnames)
        X_val, y_a_val, y_p_val, y_s_val, nsn_val = extract_Xy(feat_val, fnames)

        # ── Augmentation ─────────────────────────────────────────────────────
        if use_aug:
            X_tr_aug, y_a_aug, y_p_aug, y_s_aug = augment_flip(
                X_tr, y_a_tr, y_p_tr, y_s_tr, flip_pairs)
            print(f"  Augmented: {len(X_tr)} -> {len(X_tr_aug)} samples")
        else:
            X_tr_aug, y_a_aug, y_p_aug, y_s_aug = X_tr, y_a_tr, y_p_tr, y_s_tr

        sw_a_real = np.array([ACTION_CW.get(int(c), 1.0) for c in y_a_aug], dtype=np.float32)
        sw_p_real = np.array([POINT_CW.get(int(c),  1.0) for c in y_p_aug], dtype=np.float32)

        # ── Pseudo-row per-task injection (R-009 V1) ────────────────────────
        # Pseudo rows: appended after X_tr_aug (NO flip-aug), with flat
        # pseudo_weight. Server training EXCLUDES pseudo entirely.
        if n_pseudo > 0:
            sw_a_pseudo = np.full(n_pseudo, pseudo_weight, dtype=np.float32)
            X_tr_act_combined = np.vstack([X_tr_aug, pseudo_X]).astype(np.float32)
            y_a_act_combined  = np.concatenate([y_a_aug, pseudo_y_act])
            sw_a              = np.concatenate([sw_a_real, sw_a_pseudo])
            n_pseudo_act = n_pseudo
            mass_a_real   = float(sw_a_real.sum())
            mass_a_pseudo = float(sw_a_pseudo.sum())

            if pseudo_mode == "action_and_point":
                sw_p_pseudo = np.full(n_pseudo, pseudo_weight, dtype=np.float32)
                X_tr_pt_combined = np.vstack([X_tr_aug, pseudo_X]).astype(np.float32)
                y_p_pt_combined  = np.concatenate([y_p_aug, pseudo_y_pt])
                sw_p             = np.concatenate([sw_p_real, sw_p_pseudo])
                n_pseudo_pt = n_pseudo
                mass_p_real   = float(sw_p_real.sum())
                mass_p_pseudo = float(sw_p_pseudo.sum())
            else:  # action_only — pseudo NOT in point training
                X_tr_pt_combined = X_tr_aug
                y_p_pt_combined  = y_p_aug
                sw_p             = sw_p_real
                n_pseudo_pt = 0
                mass_p_real   = float(sw_p_real.sum())
                mass_p_pseudo = 0.0

            # Server training NEVER includes pseudo (Codex constraint #5).
            n_pseudo_srv = 0
            print(f"  [Pseudo] mode={pseudo_mode}  weight={pseudo_weight}")
            print(f"  [Pseudo] action: real={len(y_a_aug)}  pseudo={n_pseudo_act}  "
                  f"sw_mass real={mass_a_real:.1f} pseudo={mass_a_pseudo:.1f} "
                  f"({100*mass_a_pseudo/(mass_a_real+mass_a_pseudo):.1f}% pseudo)")
            print(f"  [Pseudo] point : real={len(y_p_aug)}  pseudo={n_pseudo_pt}  "
                  f"sw_mass real={mass_p_real:.1f} pseudo={mass_p_pseudo:.1f}")
            print(f"  [Pseudo] server: real={len(y_s_aug)}  pseudo={n_pseudo_srv}  (EXCLUDED per R-009)")
            assert n_pseudo_srv == 0, "INVARIANT VIOLATION: pseudo rows entered server training"
        else:
            X_tr_act_combined = X_tr_aug
            y_a_act_combined  = y_a_aug
            sw_a              = sw_a_real
            X_tr_pt_combined  = X_tr_aug
            y_p_pt_combined   = y_p_aug
            sw_p              = sw_p_real

        # ══════════════════════════════════════════════════════════════════════
        # PASS A — ACTION models (LGB + XGB + CB)
        # ══════════════════════════════════════════════════════════════════════
        if args.r203_focal:
            # R-203: focal CE multiclass with Cui CB class weights.
            # Class weighting moves OUT of sample weights and INTO the focal
            # objective's `alpha[y]`. Sample weights now carry only pseudo-row
            # downweighting (not class weighting), to avoid double-weighting.
            class_counts = np.bincount(y_a_act_combined, minlength=N_ACTION_TRAIN)
            cb_w = cui_cb_weights(class_counts, beta=args.r203_cb_beta)
            boost_cls = [int(c) for c in args.r203_boost_classes.split(",") if c.strip()]
            cb_w = apply_focal_boost(cb_w, boost_cls, boost_factor=args.r203_boost_factor)
            print(f"  [R-203] CB weights (beta={args.r203_cb_beta}, gamma={args.r203_gamma}, "
                  f"boost={boost_cls}×{args.r203_boost_factor}):")
            for c in range(N_ACTION_TRAIN):
                print(f"    cls{c:2d} n={class_counts[c]:5d}  w={cb_w[c]:.3f}"
                      f"{'  [boost]' if c in boost_cls else ''}")
            r203_focal_obj_fn = make_focal_multiclass_obj(
                num_class=N_ACTION_TRAIN, class_weights=cb_w,
                gamma=args.r203_gamma,
            )
            r203_focal_eval_fn = make_focal_multiclass_eval(num_class=N_ACTION_TRAIN)
            # Pseudo-row downweighting: real=1.0, pseudo=pseudo_weight.
            # When no pseudo data is present, sw_a == sw_a_real and every entry
            # was originally `ACTION_CW[y_i]`. Under R-203 we replace those with
            # 1.0 (class weighting is in alpha). Pseudo rows retain their
            # explicit pseudo_weight (which was already non-class-derived).
            if n_pseudo > 0:
                # Last n_pseudo entries of sw_a are pseudo_weight; preserve them.
                r203_sw_a = np.ones_like(sw_a)
                r203_sw_a[-n_pseudo:] = sw_a[-n_pseudo:]
            else:
                r203_sw_a = None  # no per-sample weighting needed
            lgb_a_p = dict(
                num_leaves=127, max_depth=9, min_child_samples=8,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0,
                learning_rate=0.04,
                objective=r203_focal_obj_fn,
                num_class=N_ACTION_TRAIN,
                metric="None",
                random_state=seed, n_jobs=-1, verbose=-1,
            )
            ds_tr = lgb.Dataset(X_tr_act_combined, label=y_a_act_combined,
                                weight=r203_sw_a)
            ds_va = lgb.Dataset(X_val, label=y_a_val, reference=ds_tr)
            lgb_a = lgb.train(
                lgb_a_p, ds_tr,
                num_boost_round=n_boost,
                valid_sets=[ds_va],
                feval=r203_focal_eval_fn,
                callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            # LightGBM with custom-obj multiclass returns raw logits from
            # predict(). Downstream code assumes probabilities. Wrap with a
            # thin adapter that auto-softmaxes (preserving raw_score=True
            # opt-in for the rare caller that wants logits).
            _raw_lgb_a = lgb_a
            class _SoftmaxLGBWrapper:
                def __init__(self, m): self._m = m
                def predict(self, X, *args, **kwargs):
                    want_raw = kwargs.pop('raw_score', False)
                    raw = self._m.predict(X, raw_score=True, *args, **kwargs)
                    if want_raw:
                        return raw
                    e = np.exp(raw - raw.max(axis=1, keepdims=True))
                    return e / e.sum(axis=1, keepdims=True)
                def __getattr__(self, name):
                    return getattr(self._m, name)
            lgb_a = _SoftmaxLGBWrapper(lgb_a)
        else:
            lgb_a_p = dict(n_estimators=n_boost, learning_rate=0.04,
                           num_leaves=127, max_depth=9, min_child_samples=8,
                           subsample=0.8, colsample_bytree=0.7,
                           reg_alpha=0.1, reg_lambda=1.0,
                           objective="multiclass", metric="multi_logloss",
                           num_class=N_ACTION_TRAIN, random_state=seed,
                           n_jobs=-1, verbose=-1)
            lgb_a = lgb.train(lgb_a_p,
                lgb.Dataset(X_tr_act_combined, label=y_a_act_combined, weight=sw_a),
                valid_sets=[lgb.Dataset(X_val, label=y_a_val)],
                callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                           lgb.log_evaluation(-1)])

        xgb_a = xgb.XGBClassifier(
            n_estimators=n_boost, learning_rate=0.04, max_depth=7,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            objective="multi:softprob", num_class=N_ACTION_TRAIN,
            eval_metric="mlogloss", early_stopping_rounds=es_rounds,
            random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist")
        xgb_a.fit(X_tr_act_combined, y_a_act_combined, sample_weight=sw_a,
                  eval_set=[(X_val, y_a_val)], verbose=False)

        if skip_cb:
            cb_a = None
        else:
            cb_a = CatBoostClassifier(
                iterations=n_boost, learning_rate=0.04, depth=7, l2_leaf_reg=3.0,
                loss_function="MultiClass", classes_count=N_ACTION_TRAIN,
                random_seed=seed, verbose=False, allow_writing_files=False,
                early_stopping_rounds=es_rounds)
            cb_a.fit(X_tr_act_combined, y_a_act_combined, sample_weight=sw_a,
                     eval_set=(X_val, y_a_val), use_best_model=True)

        # Action probabilities (15-dim for pass A output)
        pa_val_lgb = lgb_a.predict(X_val)                                              # (n_val, 15)
        pa_val_xgb = pad_proba(xgb_a.predict_proba(X_val), xgb_a.classes_, N_ACTION_TRAIN)
        if cb_a is not None:
            pa_val_cb = pad_proba(cb_a.predict_proba(X_val), cb_a.classes_, N_ACTION_TRAIN)
            pa_val_15 = (pa_val_lgb + pa_val_xgb + pa_val_cb) / 3.0
        else:
            pa_val_15 = (pa_val_lgb + pa_val_xgb) / 2.0

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
            if cb_a is not None:
                pa_tr_cb = pad_proba(cb_a.predict_proba(X_tr), cb_a.classes_, N_ACTION_TRAIN)
                pa_tr_15 = (pa_tr_lgb + pa_tr_xgb + pa_tr_cb) / 3.0
            else:
                pa_tr_15 = (pa_tr_lgb + pa_tr_xgb) / 2.0

            # For the augmented training set, the flip doesn't change action class
            pa_tr_aug_15 = np.vstack([pa_tr_15, pa_tr_15])  # duplicate for flipped half
            if not use_aug:
                pa_tr_aug_15 = pa_tr_15  # no flip done

        # Test action probs (accumulate for stacking in all folds)
        pa_test_lgb = lgb_a.predict(X_test)
        pa_test_xgb = pad_proba(xgb_a.predict_proba(X_test), xgb_a.classes_, N_ACTION_TRAIN)
        if cb_a is not None:
            pa_test_cb = pad_proba(cb_a.predict_proba(X_test), cb_a.classes_, N_ACTION_TRAIN)
            pa_test_15 = (pa_test_lgb + pa_test_xgb + pa_test_cb) / 3.0
        else:
            pa_test_15 = (pa_test_lgb + pa_test_xgb) / 2.0
        test_act15_acc += pa_test_15 / len(splits)

        # Save action probs for test submission (full 19-dim)
        test_act_acc += extend_action(pa_test_15) / len(splits)

        # ══════════════════════════════════════════════════════════════════════
        # PASS B — POINT models (binary miss + 10-class LGB/XGB/CB)
        #          with action probs as extra stacking features
        # ══════════════════════════════════════════════════════════════════════
        # Build action stacking features for training rows.
        # If point_combined includes pseudo, also predict action probs on pseudo
        # to attach the same per-row action stack column.
        if use_stack:
            X_tr_ext      = np.hstack([X_tr,     pa_tr_15])       # (n_tr,   F+15) — real, unflipped
            X_val_ext     = np.hstack([X_val,    pa_val_15])      # (n_val,  F+15)
            X_tr_aug_ext  = np.hstack([X_tr_aug, pa_tr_aug_15])   # (n_aug,  F+15) — real flipped
            if n_pseudo > 0 and pseudo_mode == "action_and_point":
                # Action probs on pseudo rows (predicted by current-fold action model).
                pa_pseudo_lgb = lgb_a.predict(pseudo_X)
                pa_pseudo_xgb = pad_proba(xgb_a.predict_proba(pseudo_X), xgb_a.classes_, N_ACTION_TRAIN)
                if cb_a is not None:
                    pa_pseudo_cb = pad_proba(cb_a.predict_proba(pseudo_X), cb_a.classes_, N_ACTION_TRAIN)
                    pa_pseudo_15 = (pa_pseudo_lgb + pa_pseudo_xgb + pa_pseudo_cb) / 3.0
                else:
                    pa_pseudo_15 = (pa_pseudo_lgb + pa_pseudo_xgb) / 2.0
                pseudo_X_ext = np.hstack([pseudo_X, pa_pseudo_15]).astype(np.float32)
                X_tr_pt_aug_ext = np.vstack([X_tr_aug_ext, pseudo_X_ext])
            else:
                X_tr_pt_aug_ext = X_tr_aug_ext
        else:
            X_tr_ext      = X_tr
            X_val_ext     = X_val
            X_tr_aug_ext  = X_tr_aug
            X_tr_pt_aug_ext = X_tr_pt_combined  # pseudo (no action-stack columns) appended

        y_miss_combined = (y_p_pt_combined == 0).astype(np.int32)
        y_miss_val      = (y_p_val == 0).astype(np.int32)

        # ── POINT binary (miss vs non-miss) ───────────────────────────────────
        lgb_pb_p = dict(n_estimators=n_boost, learning_rate=0.04,
                        num_leaves=63, max_depth=7, min_child_samples=10,
                        subsample=0.8, colsample_bytree=0.7,
                        objective="binary", metric="binary_logloss",
                        random_state=seed, n_jobs=-1, verbose=-1)
        lgb_pb = lgb.train(lgb_pb_p,
            lgb.Dataset(X_tr_pt_aug_ext, label=y_miss_combined.astype(np.float32)),
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
                       num_class=N_POINT, random_state=seed,
                       n_jobs=-1, verbose=-1)
        lgb_p = lgb.train(lgb_p_p,
            lgb.Dataset(X_tr_pt_aug_ext, label=y_p_pt_combined, weight=sw_p),
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
        xgb_p.fit(X_tr_pt_aug_ext, y_p_pt_combined, sample_weight=sw_p,
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
            cb_p.fit(X_tr_pt_aug_ext, y_p_pt_combined, sample_weight=sw_p,
                     eval_set=(X_val_ext, y_p_val), use_best_model=True)

        pp_xgb  = pad_proba(pp_xgb, xgb_p.classes_, N_POINT)
        if cb_p is not None:
            pp_cb = pad_proba(cb_p.predict_proba(X_val_ext), cb_p.classes_, N_POINT)
            pp_blend = (pp_lgb + pp_xgb + pp_cb) / 3.0
        else:
            pp_blend = (pp_lgb + pp_xgb) / 2.0
        pp_2stage = blend_two_stage(pp_blend, pb_val)
        f1_p_val  = point_macro_f1(y_p_val, pp_2stage)
        print(f"  [Point]  F1={f1_p_val:.4f}")
        oof_pt[val_mask] = pp_2stage

        # Test point predictions: use fold-averaged action probs as stacking features
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
                       random_state=seed, n_jobs=-1, verbose=-1)
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
            random_state=seed, n_jobs=-1, verbosity=0, tree_method="hist")
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

    # ─── R-009 invariant checks (Codex requirement #4) ────────────────────────
    if args.pseudo_parquet:
        assert oof_act.shape[0] == n_samples == 69712, \
            f"OOF shape invariant violated: oof_act.shape[0]={oof_act.shape[0]} vs n_samples={n_samples}"
        assert oof_pt.shape[0] == n_samples, \
            f"oof_pt.shape[0]={oof_pt.shape[0]} vs n_samples={n_samples}"
        assert oof_srv.shape[0] == n_samples, \
            f"oof_srv.shape[0]={oof_srv.shape[0]} vs n_samples={n_samples}"
        assert y_a_all.shape[0] == n_samples
        assert y_p_all.shape[0] == n_samples
        assert y_s_all.shape[0] == n_samples
        assert nsn_all.shape[0] == n_samples
        print(f"\n[R-009 invariant] OOF arrays length = {n_samples} (real train rows only) [PASS]")
        print(f"[R-009 invariant] Pseudo rows seen by server training = 0 (excluded entirely) [PASS]")
        print(f"[R-009 invariant] Pseudo rows flip-augmented = 0 (no flip on pseudo) [PASS]")

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

    # Save TWO submissions: continuous (better AUC, recommended) and binary
    sub_cont = pd.DataFrame({
        "rally_uid":      rally_test,
        "actionId":       pred_act,
        "pointId":        pred_pt,
        "serverGetPoint": test_srv_acc,  # continuous in [0, 1] for AUC
    })
    out_path_cont = os.path.join(SUBMISSION_DIR, f"submission_{out_tag}.csv")
    sub_cont.to_csv(out_path_cont, index=False)

    sub_bin = sub_cont.copy()
    sub_bin["serverGetPoint"] = (test_srv_acc >= 0.5).astype(int)
    sub_bin.to_csv(os.path.join(SUBMISSION_DIR, f"submission_{out_tag}_binary_srv.csv"),
                    index=False)

    # Save OOF + test predictions for blend / analysis
    oof_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")
    os.makedirs(oof_dir, exist_ok=True)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_act.npy"), oof_act)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_pt.npy"),  oof_pt)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_srv.npy"), oof_srv)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_pt_bin.npy"), oof_pt_bin)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_mask.npy"), oof_mask)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_y_act.npy"), y_a_all)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_y_pt.npy"),  y_p_all)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_y_srv.npy"), y_s_all)
    np.save(os.path.join(oof_dir, f"{out_tag}_oof_nsn.npy"), nsn_all)
    np.save(os.path.join(oof_dir, f"{out_tag}_test_act.npy"), test_act_acc)
    np.save(os.path.join(oof_dir, f"{out_tag}_test_pt.npy"),  test_pt_acc)
    np.save(os.path.join(oof_dir, f"{out_tag}_test_srv.npy"), test_srv_acc)
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


if __name__ == "__main__":
    main()
