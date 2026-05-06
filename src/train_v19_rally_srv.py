"""V19 — rally-pooled serverGetPoint head (P10).

Per STRATEGY/TRAIN_PLAN: serverGetPoint is a rally-CONSTANT label, but every
existing model predicts it per-shot. AUC stuck at 0.61 is suspiciously low for
a label that doesn't vary within rally. This script trains a SEPARATE binary
classifier on RALLY-LEVEL pooled features and broadcasts the per-rally
prediction to all per-shot rows.

Output artifacts (SGP-only — does NOT touch action/point):
  oof_predictions/v19_rally_srv_oof_srv.npy    (length 69712, broadcast per shot)
  oof_predictions/v19_rally_srv_test_srv.npy   (length 1236, broadcast per shot)
  oof_predictions/v19_rally_srv_test_rally_uid.npy
  oof_predictions/v19_rally_srv_oof_mask.npy
  oof_predictions/v19_rally_srv_oof_y_srv.npy

The action/point arrays are NOT saved (this model has nothing to say about
those). When swapped into the zoo, only the SGP channel changes.

CRITICAL: test.csv `serverGetPoint` is overwritten with -1 before any feature
or label use (mirrors V14/V16 / build_test_history_pairs.py policy). Never use
test SGP as a target.

Pooled features per rally (computed from V9 features on the rally's shots):
  - mean / max / min / last of the V9 numerical feature stack
  - rally length (n_shots)
  - score_diff at end of visible history
  - last-shot action one-hot (15 dims)
  - sex, numberGame
  - serve action one-hot (action of strikeNumber=1)

Train binary LGB+XGB (no CB), AUC objective, GroupKFold by match.

CLI:
  python src/train_v19_rally_srv.py --tag v19_rally_srv [--smoke] [--folds 5]
"""
import argparse
import gc
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data

N_ACTION = 19


def build_rally_features(raw_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """Per-rally pooled features. Returns one row per rally_uid.

    LEAK GUARDS (lessons from 2026-05-06 smoke that hit AUC=0.9996):
      (a) `scoreSelf` / `scoreOther` update WITHIN a train rally to reflect
          the outcome — drop ALL score aggregates everywhere.
      (b) For TRAIN rallies (`is_train=True`), drop the LAST shot — that shot
          is the rally decider; including its action/point/hand encodes the
          outcome (e.g. `point_last==0` means the deciding hitter MISSED).
          For TEST rallies (`is_train=False`), keep all n visible shots — the
          actual decider is the un-visible (n+1)-th shot we're predicting.
      (c) Both train and test still use `n_shots` and `sn_max` — these are
          meta features describing the visible context size, not the outcome.

    Resulting train/test symmetry: both sides use "rally lead-up to the
    decider" — train explicitly drops the decider; test never had it.
    """
    feats = []
    for rally_uid, grp in raw_df.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        n_full = len(grp)
        # Drop decider for train; keep all visible shots for test.
        if is_train and n_full > 1:
            grp_use = grp.iloc[:-1].reset_index(drop=True)
        else:
            grp_use = grp
        n = len(grp_use)
        if n == 0:
            continue   # 1-shot train rally with decider dropped → no context

        sn       = grp_use["strikeNumber"].values.astype(np.float32)
        action   = grp_use["actionId"].values.astype(int)
        point_   = grp_use["pointId"].values.astype(int)
        hand     = grp_use["handId"].values.astype(int)
        strength = grp_use["strengthId"].values.astype(int)
        spin     = grp_use["spinId"].values.astype(int)
        position = grp_use["positionId"].values.astype(int)

        feat = {
            "rally_uid":  rally_uid,
            "match":      grp["match"].iloc[0],
            "sex":        int(grp["sex"].iloc[0]),
            "numberGame": int(grp["numberGame"].iloc[0]),
            "n_shots":    n,
            "sn_max":     float(sn[-1]),
        }
        # Aggregates over the (lead-up only) shots — NO scoreSelf / scoreOther.
        for name, arr in [("hand", hand), ("strength", strength), ("spin", spin),
                           ("position", position), ("action", action), ("point", point_)]:
            feat[f"{name}_mode"] = float(np.bincount(np.clip(arr, 0, 19)).argmax())
            feat[f"{name}_mean"] = float(arr.mean())
            feat[f"{name}_max"]  = float(arr.max())
            feat[f"{name}_min"]  = float(arr.min())
            feat[f"{name}_last"] = float(arr[-1])
        # Serve action one-hot (action of shot 1; never leaks)
        serve_act = int(action[0]) if n >= 1 else 0
        for c in range(N_ACTION):
            feat[f"serve_act_{c}"] = 1.0 if serve_act == c else 0.0
        # Last (non-decider) shot action one-hot — for train this is shot N-1,
        # for test this is shot n. Both are pre-decider.
        last_act = int(action[-1])
        for c in range(N_ACTION):
            feat[f"last_act_{c}"] = 1.0 if last_act == c else 0.0
        feats.append(feat)
    return pd.DataFrame(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",     type=str, default="v19_rally_srv")
    ap.add_argument("--smoke",   action="store_true")
    ap.add_argument("--folds",   type=int, default=N_FOLDS)
    ap.add_argument("--n-boost", type=int, default=-1)
    ap.add_argument("--es",      type=int, default=-1)
    ap.add_argument("--seed",    type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    is_smoke = args.smoke
    n_folds  = 1 if is_smoke else args.folds
    n_boost  = (200 if is_smoke else 3000) if args.n_boost < 0 else args.n_boost
    es       = (30  if is_smoke else 200)  if args.es      < 0 else args.es
    seed     = args.seed
    np.random.seed(seed)

    t_start = time.time()
    print("=" * 70)
    print(f"V19 RALLY SERVER HEAD {'(SMOKE)' if is_smoke else ''}  tag={args.tag}")
    print(f"  folds={n_folds}  n_boost={n_boost}  es={es}  seed={seed}")
    print("=" * 70)

    import lightgbm as lgb
    import xgboost as xgb

    # ─── Load and clean ─────────────────────────────────────────────────────
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(TEST_PATH)
    train_df, test_df, _ = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1   # P10 hard guard: never use test SGP

    # ─── Build per-rally features ───────────────────────────────────────────
    print("\n--- Building per-rally features ---")
    t0 = time.time()
    train_rallies = build_rally_features(train_df, is_train=True)
    test_rallies  = build_rally_features(test_df,  is_train=False)
    print(f"  Train rallies: {len(train_rallies)}  Test rallies: {len(test_rallies)}  "
          f"[{time.time()-t0:.1f}s]")

    # Per-rally label: take the rally's serverGetPoint (constant within rally).
    rally_sgp = train_df.groupby("rally_uid")["serverGetPoint"].first()
    train_rallies["y_srv"] = train_rallies["rally_uid"].map(rally_sgp).astype(int)

    # ─── OOF + test setup ────────────────────────────────────────────────────
    # We need per-shot OOF arrays (length 69712) for compatibility with the zoo.
    # Map rally → list of OOF row indices using the V14/V16 row order replication.
    print("\n--- Mapping per-rally predictions to per-shot OOF rows ---")
    rally_to_oof_rows: dict[int, list[int]] = {}
    row_idx = 0
    for rally_uid, grp in train_df.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber")
        n = len(grp)
        if n < 2:
            continue
        rally_to_oof_rows[rally_uid] = list(range(row_idx, row_idx + n - 1))
        row_idx += n - 1
    n_oof_rows = row_idx
    assert n_oof_rows == 69712, f"GUARD FAIL: expected 69712 OOF rows, got {n_oof_rows}"
    print(f"  Mapped {len(rally_to_oof_rows)} train rallies to {n_oof_rows} OOF rows")

    # Test rally → test row mapping (one row per rally; the test "OOF" row is the rally itself)
    test_rally_uids = test_rallies["rally_uid"].values

    # Feature columns
    label_col = "y_srv"
    drop_cols = ("rally_uid", "match", label_col)
    feat_cols = [c for c in train_rallies.columns if c not in drop_cols]
    print(f"  Per-rally features: {len(feat_cols)}")

    # OOF containers
    oof_srv      = np.zeros(n_oof_rows, dtype=np.float32)
    oof_mask     = np.zeros(n_oof_rows, dtype=bool)
    test_srv_acc = np.zeros(len(test_rallies), dtype=np.float32)

    # GroupKFold by match
    matches = train_rallies["match"].values
    gkf = GroupKFold(n_splits=max(n_folds, 2))
    splits = list(gkf.split(np.arange(len(train_rallies)), groups=matches))
    if is_smoke:
        splits = splits[:1]

    Xt = test_rallies[feat_cols].values.astype(np.float32)

    for fold, (tr_idx, val_idx) in enumerate(splits):
        t_fold = time.time()
        print(f"\n{'='*60}\n  FOLD {fold+1}/{len(splits)}\n{'='*60}")

        X_tr = train_rallies.iloc[tr_idx][feat_cols].values.astype(np.float32)
        X_val = train_rallies.iloc[val_idx][feat_cols].values.astype(np.float32)
        y_tr = train_rallies.iloc[tr_idx][label_col].values
        y_val = train_rallies.iloc[val_idx][label_col].values

        # LGB
        lgb_p = dict(n_estimators=n_boost, learning_rate=0.04,
                     num_leaves=63, max_depth=7, min_child_samples=15,
                     subsample=0.8, colsample_bytree=0.7,
                     objective="binary", metric="auc",
                     random_state=seed, n_jobs=-1, verbose=-1)
        lgb_m = lgb.train(lgb_p,
            lgb.Dataset(X_tr, label=y_tr),
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(-1)])
        p_lgb = lgb_m.predict(X_val)

        # XGB
        xgb_m = xgb.XGBClassifier(
            n_estimators=n_boost, learning_rate=0.04, max_depth=7,
            subsample=0.8, colsample_bytree=0.7,
            objective="binary:logistic", eval_metric="auc",
            early_stopping_rounds=es, random_state=seed, n_jobs=-1,
            verbosity=0, tree_method="hist")
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        p_xgb = xgb_m.predict_proba(X_val)[:, 1]

        p_val = (p_lgb + p_xgb) / 2.0
        rally_auc = roc_auc_score(y_val, p_val)
        print(f"  Rally AUC (val): {rally_auc:.4f}")

        # Broadcast per-rally prediction to per-shot OOF rows
        val_uids = train_rallies.iloc[val_idx]["rally_uid"].values
        for uid, prob in zip(val_uids, p_val):
            for r in rally_to_oof_rows.get(uid, []):
                oof_srv[r] = prob
                oof_mask[r] = True

        # Test predictions (per-rally; broadcast at the end)
        p_test_lgb = lgb_m.predict(Xt)
        p_test_xgb = xgb_m.predict_proba(Xt)[:, 1]
        test_srv_acc += (p_test_lgb + p_test_xgb) / 2.0 / len(splits)

        print(f"  fold time: {time.time() - t_fold:.0f}s")

        del lgb_m, xgb_m; gc.collect()

    # Per-shot AUC on OOF
    n_in_mask = int(oof_mask.sum())
    print(f"\n--- OOF stats ---")
    print(f"  per-shot OOF mask: {n_in_mask}/{n_oof_rows} ({100*n_in_mask/n_oof_rows:.0f}%)")
    # Build per-shot y_srv via the same broadcast
    y_srv_per_shot = np.zeros(n_oof_rows, dtype=np.float32)
    for rally_uid, rows in rally_to_oof_rows.items():
        y = int(rally_sgp.get(rally_uid, 0))
        for r in rows:
            y_srv_per_shot[r] = y
    if n_in_mask > 0:
        per_shot_auc = roc_auc_score(y_srv_per_shot[oof_mask], oof_srv[oof_mask])
        print(f"  per-shot AUC (broadcast): {per_shot_auc:.4f}")
        # Compare to V14 baseline AUC ~0.61
        print(f"  V14 baseline per-shot AUC: 0.6101")
        print(f"  Delta: {per_shot_auc - 0.6101:+.4f}")

    # ─── Save artifacts ──────────────────────────────────────────────────────
    oof_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")
    os.makedirs(oof_dir, exist_ok=True)

    np.save(os.path.join(oof_dir, f"{args.tag}_oof_srv.npy"),  oof_srv)
    np.save(os.path.join(oof_dir, f"{args.tag}_oof_mask.npy"), oof_mask)
    np.save(os.path.join(oof_dir, f"{args.tag}_oof_y_srv.npy"), y_srv_per_shot.astype(np.int32))
    np.save(os.path.join(oof_dir, f"{args.tag}_test_srv.npy"), test_srv_acc)
    np.save(os.path.join(oof_dir, f"{args.tag}_test_rally_uid.npy"), test_rally_uids)
    print(f"\n  Saved SGP-only artifacts to {oof_dir} (tag={args.tag})")
    print(f"  NO action/point arrays saved — this model only contributes to SGP channel.")

    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
