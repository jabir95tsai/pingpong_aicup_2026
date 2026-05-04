"""V12 Hierarchical PointId Head.

Trains 3 sub-models on features_v7 (re-using V12 fold structure):
  Head V (valid):  binary  — P(pointId != 0)
  Head D (depth):  4-class — P(none / short / half / long)  (trained on pt!=0 only)
  Head S (side):   4-class — P(none / FH / mid / BH)        (trained on pt!=0 only)

Joint reconstruction at inference:
  P(pt=0) = P(V=0)
  For k in 1..9:
    depth(k) = depth_bucket[k]  ∈ {1,2,3}
    side(k)  = side_bucket[k]   ∈ {1,2,3}
    P(pt=k) = P(V=1) × P(D=depth(k)) × P(S=side(k)) / Z
  Z = sum over k=1..9 of P(D=depth(k)) × P(S=side(k))   (renormalize on conditional pt!=0)

Final pointId prediction: argmax_k P(pt=k) over 0..9.

Optional blend with flat 10-class V12 OOF:
  P_final = α × P_hier + (1-α) × P_flat
α searched on OOF for max macro-F1.

This script reads V12's OOF/test feature matrices via train_v12.py preflight
and trains the 3 heads using GroupKFold(match) on the same fold structure.
"""
import sys, os, time, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data
from features_v7 import (
    build_features_v7, compute_global_stats_v7, get_feature_names_v7,
)

N_POINT = 10
POINT_EVAL = list(range(N_POINT))

# pointId classes:  0=miss, 1-3=short(FH/mid/BH), 4-6=half, 7-9=long
DEPTH_BUCKET = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3}
SIDE_BUCKET  = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 1, 8: 2, 9: 3}

# Reverse: (depth, side) -> pointId
PT_FROM_DS = {(1, 1): 1, (1, 2): 2, (1, 3): 3,
              (2, 1): 4, (2, 2): 5, (2, 3): 6,
              (3, 1): 7, (3, 2): 8, (3, 3): 9}


def joint_reconstruct(p_valid, p_depth, p_side):
    """Reconstruct full 10-class point prob from 3 hierarchical heads.

    p_valid: (N,)        P(pt!=0)
    p_depth: (N, 4)      [P(d=0), P(d=1=short), P(d=2=half), P(d=3=long)]
    p_side:  (N, 4)      [P(s=0), P(s=1=FH),    P(s=2=mid),  P(s=3=BH)]
    """
    n = len(p_valid)
    out = np.zeros((n, N_POINT), dtype=np.float32)
    out[:, 0] = 1.0 - p_valid
    # Compute conditional P(pt=k | V=1)  k=1..9 from depth/side
    # Renormalize over k=1..9 (exclude depth=0 and side=0)
    for k in range(1, 10):
        d = DEPTH_BUCKET[k]
        s = SIDE_BUCKET[k]
        out[:, k] = p_valid * p_depth[:, d] * p_side[:, s]
    # Renormalize so the conditional pt!=0 sums to p_valid
    s_pos = out[:, 1:].sum(axis=1)
    s_pos = np.where(s_pos < 1e-9, 1.0, s_pos)
    scale = p_valid / s_pos
    out[:, 1:] = out[:, 1:] * scale[:, np.newaxis]
    # Final clamp
    return np.clip(out, 1e-9, 1.0)


def train_hierarchical(X_tr, y_p_tr, X_val, y_p_val, n_boost=2000, es=150):
    """Train 3 LightGBM heads on training fold."""
    import lightgbm as lgb

    # Head V: binary valid
    y_v_tr  = (y_p_tr != 0).astype(int)
    y_v_val = (y_p_val != 0).astype(int)

    valid_p = dict(n_estimators=n_boost, learning_rate=0.04,
                   num_leaves=127, max_depth=9, min_child_samples=8,
                   subsample=0.8, colsample_bytree=0.7,
                   reg_alpha=0.1, reg_lambda=1.0,
                   objective="binary", metric="binary_logloss",
                   random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    m_valid = lgb.train(valid_p,
        lgb.Dataset(X_tr, label=y_v_tr),
        valid_sets=[lgb.Dataset(X_val, label=y_v_val)],
        callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(-1)])

    # Head D: depth 4-class (only trained on pt != 0 samples)
    mask_pos = (y_p_tr != 0)
    y_d_tr = np.array([DEPTH_BUCKET[int(p)] for p in y_p_tr])
    y_s_tr = np.array([SIDE_BUCKET[int(p)] for p in y_p_tr])

    # NOTE: We mask loss by training only on positive samples to avoid noisy
    # 0-class signal. For inference we still need 4-class output from depth/side.
    depth_p = dict(n_estimators=n_boost, learning_rate=0.04,
                   num_leaves=127, max_depth=9, min_child_samples=8,
                   subsample=0.8, colsample_bytree=0.7,
                   reg_alpha=0.1, reg_lambda=1.0,
                   objective="multiclass", metric="multi_logloss",
                   num_class=4, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    # Train on full (including pt=0) so model learns d=0 → "out" too
    m_depth = lgb.train(depth_p,
        lgb.Dataset(X_tr, label=y_d_tr),
        valid_sets=[lgb.Dataset(X_val, label=np.array([DEPTH_BUCKET[int(p)] for p in y_p_val]))],
        callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(-1)])

    side_p = dict(n_estimators=n_boost, learning_rate=0.04,
                  num_leaves=127, max_depth=9, min_child_samples=8,
                  subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=0.1, reg_lambda=1.0,
                  objective="multiclass", metric="multi_logloss",
                  num_class=4, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    m_side = lgb.train(side_p,
        lgb.Dataset(X_tr, label=y_s_tr),
        valid_sets=[lgb.Dataset(X_val, label=np.array([SIDE_BUCKET[int(p)] for p in y_p_val]))],
        callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(-1)])

    return m_valid, m_depth, m_side


def extract_Xy(feat_df, fnames):
    X = feat_df[fnames].values.astype(np.float32)
    y_p = feat_df["y_pointId"].values.astype(np.int32)
    nsn = feat_df["next_strikeNumber"].values.astype(np.int32)
    return X, y_p, nsn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-boost", type=int, default=-1)
    parser.add_argument("--es", type=int, default=-1)
    args = parser.parse_args()

    is_smoke  = args.smoke
    n_folds   = 1 if is_smoke else args.folds
    n_boost   = (200 if is_smoke else 1500) if args.n_boost < 0 else args.n_boost
    es        = (30 if is_smoke else 150) if args.es < 0 else args.es

    t_start = time.time()
    print("=" * 70)
    print(f"V12 HIERARCHICAL POINT HEAD {'(SMOKE)' if is_smoke else ''}")
    print(f"  folds={n_folds}  n_boost={n_boost}")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(TEST_PATH)
    train_df, test_df, _ = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1

    print("\n--- Preflight: build full features for index alignment ---")
    t0 = time.time()
    gs_full = compute_global_stats_v7(train_df)
    feat_full = build_features_v7(train_df, is_train=True,
                                    global_stats_v7=gs_full,
                                    raw_df=train_df)
    fnames = get_feature_names_v7(feat_full)
    n_samples = len(feat_full)
    print(f"  {len(fnames)} features, {n_samples} samples ({time.time()-t0:.1f}s)")

    X_all, y_p_all, nsn_all = extract_Xy(feat_full, fnames)
    rally_uids_all = feat_full["rally_uid"].values
    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_all = np.array([rally_to_match.get(r, -1) for r in rally_uids_all])

    oof_valid = np.zeros(n_samples, dtype=np.float32)
    oof_depth = np.zeros((n_samples, 4), dtype=np.float32)
    oof_side  = np.zeros((n_samples, 4), dtype=np.float32)
    oof_mask  = np.zeros(n_samples, dtype=bool)

    feat_test = build_features_v7(test_df, is_train=False,
                                    global_stats_v7=gs_full,
                                    raw_df=test_df)
    X_test = feat_test[fnames].values.astype(np.float32)
    rally_test = feat_test["rally_uid"].values

    test_valid_acc = np.zeros(len(X_test), dtype=np.float32)
    test_depth_acc = np.zeros((len(X_test), 4), dtype=np.float32)
    test_side_acc  = np.zeros((len(X_test), 4), dtype=np.float32)

    gkf = GroupKFold(n_splits=max(n_folds, 2))
    splits = list(gkf.split(np.arange(n_samples), groups=match_all))
    if is_smoke:
        splits = splits[:1]

    for fold, (tr_idx, val_idx) in enumerate(splits):
        t_fold = time.time()
        print(f"\n=== FOLD {fold+1}/{len(splits)} ===")

        tr_rallies  = set(rally_uids_all[tr_idx])
        val_rallies = set(rally_uids_all[val_idx])
        tr_raw  = train_df[train_df["rally_uid"].isin(tr_rallies)]
        val_raw = train_df[train_df["rally_uid"].isin(val_rallies)]

        fold_stats = compute_global_stats_v7(tr_raw)
        feat_tr  = build_features_v7(tr_raw, is_train=True,
                                       global_stats_v7=fold_stats,
                                       raw_df=tr_raw)
        feat_val = build_features_v7(val_raw, is_train=True,
                                       global_stats_v7=fold_stats,
                                       raw_df=val_raw)
        X_tr, y_p_tr, _ = extract_Xy(feat_tr, fnames)
        X_val, y_p_val, _ = extract_Xy(feat_val, fnames)
        print(f"  train={len(X_tr)}  val={len(X_val)}  features={X_tr.shape[1]}")

        m_valid, m_depth, m_side = train_hierarchical(
            X_tr, y_p_tr, X_val, y_p_val, n_boost=n_boost, es=es)

        # OOF predictions
        p_v = m_valid.predict(X_val)
        p_d = m_depth.predict(X_val)
        p_s = m_side.predict(X_val)
        oof_valid[val_idx] = p_v
        oof_depth[val_idx] = p_d
        oof_side[val_idx]  = p_s
        oof_mask[val_idx]  = True

        # Test predictions accumulator
        test_valid_acc += m_valid.predict(X_test) / len(splits)
        test_depth_acc += m_depth.predict(X_test) / len(splits)
        test_side_acc  += m_side.predict(X_test)  / len(splits)

        # Report fold metrics
        print(f"  fold valid AUC: tr-vs-val (rough)")
        f_valid = f1_score((y_p_val != 0).astype(int), (p_v >= 0.5).astype(int),
                            average="binary", zero_division=0)
        print(f"  fold V binary F1: {f_valid:.4f}")
        # Joint reconstruction OOF
        oof_pt_hier = joint_reconstruct(p_v, p_d, p_s)
        f1_p = f1_score(y_p_val, oof_pt_hier.argmax(axis=1),
                         labels=POINT_EVAL, average="macro", zero_division=0)
        print(f"  fold hierarchical pointId macro F1: {f1_p:.4f}")
        print(f"  fold time: {time.time()-t_fold:.1f}s")

    # ── Joint reconstruction OOF ──────────────────────────────────────────────
    print("\n--- Joint reconstruction OOF ---")
    if oof_mask.sum() == 0:
        print("WARNING: no OOF samples (smoke?)")
        return
    p_hier = joint_reconstruct(oof_valid[oof_mask],
                                 oof_depth[oof_mask],
                                 oof_side[oof_mask])
    y_p_oof = y_p_all[oof_mask]
    f1_p_oof = f1_score(y_p_oof, p_hier.argmax(axis=1),
                          labels=POINT_EVAL, average="macro", zero_division=0)
    print(f"  Hierarchical pointId macro F1: {f1_p_oof:.4f}")

    # Per-class breakdown
    print("\n  Per-class F1:")
    pp = p_hier.argmax(axis=1)
    pf1s = f1_score(y_p_oof, pp, labels=POINT_EVAL, average=None, zero_division=0)
    for c, f in enumerate(pf1s):
        n = (y_p_oof == c).sum()
        print(f"    pt={c}: F1={f:.4f}  n={n}")

    # Try blending with existing V12 OOF if present
    oof_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")
    v12_pt_path = os.path.join(oof_dir, "v12_oof_pt.npy")
    if os.path.exists(v12_pt_path):
        print("\n--- Blend with V12 flat point OOF ---")
        oof_pt_flat = np.load(v12_pt_path)
        if len(oof_pt_flat) == n_samples:
            best_a, best_f1 = 0.0, f1_p_oof
            for a in np.arange(0, 1.05, 0.05):
                blend = a * p_hier + (1 - a) * oof_pt_flat[oof_mask]
                f1 = f1_score(y_p_oof, blend.argmax(axis=1),
                               labels=POINT_EVAL, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_a, best_f1 = a, f1
            print(f"  Best blend α={best_a:.2f}  F1_p={best_f1:.4f}  "
                  f"(vs flat={f1_score(y_p_oof, oof_pt_flat[oof_mask].argmax(axis=1), labels=POINT_EVAL, average='macro', zero_division=0):.4f}, "
                  f"hier={f1_p_oof:.4f})")
        else:
            print(f"  V12 flat OOF length mismatch ({len(oof_pt_flat)} vs {n_samples})")

    # Save outputs
    np.save(os.path.join(oof_dir, "v12_hier_oof_valid.npy"), oof_valid)
    np.save(os.path.join(oof_dir, "v12_hier_oof_depth.npy"), oof_depth)
    np.save(os.path.join(oof_dir, "v12_hier_oof_side.npy"),  oof_side)
    np.save(os.path.join(oof_dir, "v12_hier_oof_mask.npy"),  oof_mask)
    np.save(os.path.join(oof_dir, "v12_hier_test_valid.npy"), test_valid_acc)
    np.save(os.path.join(oof_dir, "v12_hier_test_depth.npy"), test_depth_acc)
    np.save(os.path.join(oof_dir, "v12_hier_test_side.npy"),  test_side_acc)
    np.save(os.path.join(oof_dir, "v12_hier_test_rally_uid.npy"), rally_test)
    print(f"\n  Saved hierarchical OOF + test to {oof_dir}")
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
