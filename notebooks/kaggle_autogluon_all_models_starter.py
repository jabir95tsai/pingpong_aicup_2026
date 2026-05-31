# # AICUP 2026 — AutoGluon "all models, all features" component
#
# Builds a NEW SIGNAL CLASS component for the R-034 blend, intentionally
# designed to maximize coverage:
#
# - AutoGluon `best_quality` preset
# - ALL model families enabled (LGB, XGB, CAT, NN_TORCH, FASTAI, KNN, RF, XT, LR)
# - Feature set = v15feat_b (113) + score-pressure (6) + action-point combo (2)
#                 + 10 components' OOF probs as meta-features (~290 numeric cols)
# - 5-fold match-grouped CV (mirrors our train_v14 split)
# - Output: v14_seed2_autogluon_full OOF + test arrays (.npy)
#
# **Why include "failed" model families** (R-034 lesson):
#   Standalone failure ≠ blend failure. We let every AG family contribute
#   and the per-target ensemble picks weights. The user-explicit directive:
#   "use all the models we have including failed ones so that we dont miss".
#
# **Time budget (Kaggle CPU 12-hr)**:
#   3 targets × 5 folds × ~30 min = ~7.5 hr expected
#   Pin time_limit per (target, fold) and watch the elapsed.

# ## 1. Setup

# +
import os
import sys
import time
from pathlib import Path

IN_KAGGLE = Path("/kaggle").exists()
if IN_KAGGLE:
    DATA_DIR = Path("/kaggle/input/datasets/jabir95tsai/aicup2026-pingpong-private")
    if not DATA_DIR.exists():
        DATA_DIR = Path("/kaggle/input/aicup2026-pingpong-private")
    OUT_DIR = Path("/kaggle/working")
    CODE_DIR = DATA_DIR / "code"
else:
    # Local parity for debugging
    DATA_DIR = Path("data")
    OUT_DIR = Path("kaggle_outputs")
    OUT_DIR.mkdir(exist_ok=True)
    CODE_DIR = Path("src")

print(f"DATA_DIR={DATA_DIR}")
print(f"CODE_DIR={CODE_DIR}")
print(f"OUT_DIR={OUT_DIR}")
sys.path.insert(0, str(CODE_DIR))
# -

# ## 2. Install AutoGluon (Kaggle only; ~5 min)

# !pip install -q autogluon.tabular[all]==1.1.1 lightgbm==4.5.0 xgboost==2.1.1

# ## 3. Load raw data + verify

# +
import pandas as pd
import numpy as np

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test_new.csv")
old_test_raw = pd.read_csv(DATA_DIR / "test.csv")
print(f"train: {len(train_raw):,} rows, {train_raw.rally_uid.nunique():,} rallies, {train_raw.match.nunique()} matches")
print(f"test_new: {len(test_raw):,} rows, {test_raw.rally_uid.nunique():,} rallies")
print(f"old_test (legal aug): {len(old_test_raw):,} rows, {old_test_raw.rally_uid.nunique():,} rallies")

assert {"actionId", "pointId", "serverGetPoint"}.issubset(set(train_raw.columns))
assert {"actionId", "pointId", "serverGetPoint"}.issubset(set(old_test_raw.columns))
# -

# ## 4. Build feature matrix (v15feat_b backbone)

# +
# Import the in-repo feature engineering
from features_v15feat_b import (
    build_features_v15feat_b,
    compute_global_stats_v15feat_b,
)

print("Building train features ...")
t0 = time.time()
gs_full = compute_global_stats_v15feat_b(train_raw)
train_feat = build_features_v15feat_b(train_raw, is_train=True,
                                       global_stats_v9=gs_full,
                                       raw_df=train_raw)
print(f"  train_feat: {train_feat.shape}  ({time.time()-t0:.1f}s)")

print("Building test features ...")
t0 = time.time()
test_feat = build_features_v15feat_b(test_raw, is_train=False,
                                      global_stats_v9=gs_full,
                                      raw_df=test_raw)
print(f"  test_feat: {test_feat.shape}  ({time.time()-t0:.1f}s)")
# -

# ## 5. (SKIPPED) Pressure features
#
# Previously this step caused a ValueError due to brittle merge/index alignment.
# v15feat_b already includes scoreSelf, scoreOther, lag1/lag2 in its v9 backbone,
# so AutoGluon can derive any score-pressure interactions itself if relevant.
# The 8 pressure features moved into our separate R-047 v15feat_c trainer.

print(f"Skipped pressure features (handled by R-047 v15feat_c).")
print(f"  train_feat shape: {train_feat.shape}, test_feat shape: {test_feat.shape}")

# ## 6. Add OOF probabilities as meta-features (the "stacking" part)

# +
OOF_DIR = DATA_DIR / "oof"

META_COMPONENTS = [
    # R-034 baseline (always)
    "v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3",
    # NEW SIGNAL CLASS (parked audit STAGE 1 LOW-RISK)
    "meta_stack", "meta_stack_v2_logistic",
    # B-feature class
    "v14_recvhand", "v14_recvprofile",
    # SN=2 specialist (partial coverage but adds signal there)
    "sn2_expert",
]

# Load N reference rows to align oldtest variants
def load_oof_aligned(tag: str, n_ref: int = 69712):
    arr_a = np.load(OOF_DIR / f"{tag}_oof_act.npy")
    arr_p = np.load(OOF_DIR / f"{tag}_oof_pt.npy")
    arr_s = np.load(OOF_DIR / f"{tag}_oof_srv.npy")
    if arr_a.shape[0] != n_ref:
        arr_a = arr_a[:n_ref]; arr_p = arr_p[:n_ref]; arr_s = arr_s[:n_ref]
    # Pad action to 19
    if arr_a.shape[1] < 19:
        out = np.zeros((arr_a.shape[0], 19), dtype=np.float32)
        out[:, :arr_a.shape[1]] = arr_a; arr_a = out
    return arr_a.astype(np.float32), arr_p.astype(np.float32), arr_s.astype(np.float32)

def load_test_aligned(tag: str):
    arr_a = np.load(OOF_DIR / f"{tag}_test_act.npy")
    arr_p = np.load(OOF_DIR / f"{tag}_test_pt.npy")
    arr_s = np.load(OOF_DIR / f"{tag}_test_srv.npy")
    if arr_a.shape[1] < 19:
        out = np.zeros((arr_a.shape[0], 19), dtype=np.float32)
        out[:, :arr_a.shape[1]] = arr_a; arr_a = out
    return arr_a.astype(np.float32), arr_p.astype(np.float32), arr_s.astype(np.float32)

# train_feat has 69712 rows (one per (rally, strikeNumber>=2)).
assert len(train_feat) == 69712, f"train_feat len {len(train_feat)} != 69712"

for comp in META_COMPONENTS:
    try:
        a_oof, p_oof, s_oof = load_oof_aligned(comp)
        a_te, p_te, s_te = load_test_aligned(comp)
    except FileNotFoundError as e:
        print(f"  [skip] {comp}: {e}")
        continue
    for c in range(19):
        train_feat[f"oof_{comp}_a{c}"] = a_oof[:, c]
        test_feat[f"oof_{comp}_a{c}"] = a_te[:, c]
    for c in range(10):
        train_feat[f"oof_{comp}_p{c}"] = p_oof[:, c]
        test_feat[f"oof_{comp}_p{c}"] = p_te[:, c]
    train_feat[f"oof_{comp}_s"] = s_oof
    test_feat[f"oof_{comp}_s"] = s_te
    print(f"  + {comp}: 30 cols added")

print(f"After OOF meta-features: train_feat {train_feat.shape}, test_feat {test_feat.shape}")
# -

# ## 7. AutoGluon training, per-target, per-fold

# +
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import GroupKFold

# Match-grouped CV (matches our local pipeline)
# train_feat must have a column identifying the match for each row.
# v15feat_b output keeps rally_uid; we need to look up match per rally.
rally_to_match = train_raw.drop_duplicates("rally_uid").set_index("rally_uid")["match"]
train_feat["_match"] = train_feat["rally_uid"].map(rally_to_match).astype(int)
groups = train_feat["_match"].values
print(f"Unique matches in train_feat: {pd.Series(groups).nunique()}")

# Drop columns AG shouldn't see
FEATURE_COLS = [c for c in train_feat.columns
                if c not in {"rally_uid", "_match", "actionId", "pointId", "serverGetPoint", "next_strikeNumber"}]
print(f"Total features going to AutoGluon: {len(FEATURE_COLS)}")

# Per-target metrics
TARGET_METRICS = {
    "actionId": ("f1_macro", 19, 30 * 60),   # 30 min time limit per fold
    "pointId":  ("f1_macro", 10, 25 * 60),
    "serverGetPoint": ("roc_auc", 2, 20 * 60),
}

N_FOLDS = 5
SEED = 51966
TAG = "v14_seed2_autogluon_full"

# Allocate OOF + test accumulators
N_TRAIN = len(train_feat)
N_TEST  = len(test_feat)
oof = {
    "actionId": np.zeros((N_TRAIN, 19), dtype=np.float32),
    "pointId":  np.zeros((N_TRAIN, 10), dtype=np.float32),
    "serverGetPoint": np.zeros(N_TRAIN, dtype=np.float32),
}
test_pred = {
    "actionId": np.zeros((N_TEST, 19), dtype=np.float32),
    "pointId":  np.zeros((N_TEST, 10), dtype=np.float32),
    "serverGetPoint": np.zeros(N_TEST, dtype=np.float32),
}

gkf = GroupKFold(n_splits=N_FOLDS)
splits = list(gkf.split(train_feat, groups=groups))

for fold_idx, (tr_idx, val_idx) in enumerate(splits, 1):
    print(f"\n========== FOLD {fold_idx}/{N_FOLDS} ==========")
    print(f"  train rows: {len(tr_idx):,} val rows: {len(val_idx):,}")
    for target, (metric, n_class, time_lim) in TARGET_METRICS.items():
        target_dir = OUT_DIR / "ag_models" / f"fold{fold_idx}" / target
        target_dir.mkdir(parents=True, exist_ok=True)

        train_data = train_feat.iloc[tr_idx][FEATURE_COLS + [target]].copy()
        val_data = train_feat.iloc[val_idx][FEATURE_COLS].copy()

        print(f"  [{target}] fitting AutoGluon best_quality (time_limit={time_lim/60:.0f}min) ...")
        t0 = time.time()
        predictor = TabularPredictor(
            label=target,
            eval_metric=metric,
            path=str(target_dir),
            verbosity=1,
        )
        predictor.fit(
            train_data,
            presets="best_quality",
            time_limit=time_lim,
            # NO excluded_model_types — include EVERY family per user directive
            num_bag_folds=0,        # outer CV is manual; AG doesn't bag internally
            num_stack_levels=0,
            fit_weighted_ensemble=True,
            # Don't set hyperparameters — let best_quality figure it out
        )
        elapsed = time.time() - t0
        print(f"    fit done {elapsed/60:.1f}min")

        # OOF for validation
        if target == "serverGetPoint":
            val_proba = predictor.predict_proba(val_data)[1].values
            oof[target][val_idx] = val_proba.astype(np.float32)
        else:
            val_proba = predictor.predict_proba(val_data)
            # Ensure all classes columns
            for c in range(n_class):
                if c in val_proba.columns:
                    oof[target][val_idx, c] = val_proba[c].values.astype(np.float32)
        # Test prediction
        test_data = test_feat[FEATURE_COLS].copy()
        if target == "serverGetPoint":
            test_proba = predictor.predict_proba(test_data)[1].values
            test_pred[target] += test_proba.astype(np.float32) / N_FOLDS
        else:
            test_proba = predictor.predict_proba(test_data)
            for c in range(n_class):
                if c in test_proba.columns:
                    test_pred[target][:, c] += test_proba[c].values.astype(np.float32) / N_FOLDS

        # Free disk — AG models are huge
        import shutil
        shutil.rmtree(target_dir, ignore_errors=True)
# -

# ## 8. Save OOF + test arrays

# +
oof_save_dir = OUT_DIR / "oof_predictions"
oof_save_dir.mkdir(parents=True, exist_ok=True)

# Convert ground truth
y_act = train_feat["actionId"].astype(np.int64).values
y_pt  = train_feat["pointId"].astype(np.int64).values
y_srv = train_feat["serverGetPoint"].astype(np.int64).values

np.save(oof_save_dir / f"{TAG}_oof_act.npy", oof["actionId"])
np.save(oof_save_dir / f"{TAG}_oof_pt.npy",  oof["pointId"])
np.save(oof_save_dir / f"{TAG}_oof_srv.npy", oof["serverGetPoint"])
np.save(oof_save_dir / f"{TAG}_oof_y_act.npy", y_act)
np.save(oof_save_dir / f"{TAG}_oof_y_pt.npy", y_pt)
np.save(oof_save_dir / f"{TAG}_oof_y_srv.npy", y_srv)
np.save(oof_save_dir / f"{TAG}_test_act.npy", test_pred["actionId"])
np.save(oof_save_dir / f"{TAG}_test_pt.npy",  test_pred["pointId"])
np.save(oof_save_dir / f"{TAG}_test_srv.npy", test_pred["serverGetPoint"])
np.save(oof_save_dir / f"{TAG}_test_rally_uid.npy", test_feat["rally_uid"].astype(np.int64).values)

# Submission CSV (argmax for action/point, prob for SGP)
submission = pd.DataFrame({
    "rally_uid": test_feat["rally_uid"].astype(int),
    "actionId": test_pred["actionId"].argmax(axis=1).astype(int),
    "pointId":  test_pred["pointId"].argmax(axis=1).astype(int),
    "serverGetPoint": np.clip(test_pred["serverGetPoint"], 0.0, 1.0).astype(np.float32),
})
sub_path = OUT_DIR / "submissions" / f"submission_{TAG}.csv"
sub_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(sub_path, index=False, lineterminator="\n", encoding="utf-8")
print(f"Wrote {sub_path}")
print(submission.head())
# -

# ## 9. OOF metrics (quick sanity)

# +
from sklearn.metrics import roc_auc_score

def f1_macro_safe(y, p, labels, n):
    cm = np.bincount(y.astype(np.int64)*n + p.astype(np.int64), minlength=n*n).reshape(n, n)
    cs = cm.sum(0); rs = cm.sum(1); d = np.diag(cm)
    f1s = []
    for c in labels:
        tp = d[c]; fp = cs[c]-tp; fn = rs[c]-tp
        f1s.append(0.0 if (2*tp+fp+fn)<=0 else (2*tp)/(2*tp+fp+fn))
    return float(np.mean(f1s))

ACTION_EVAL = list(range(15))  # exclude serve classes 15-18
POINT_EVAL = list(range(10))

f1_a = f1_macro_safe(y_act, oof["actionId"].argmax(1), ACTION_EVAL, 19)
f1_p = f1_macro_safe(y_pt,  oof["pointId"].argmax(1),  POINT_EVAL, 10)
auc = roc_auc_score(y_srv, oof["serverGetPoint"])
ov = 0.4*f1_a + 0.4*f1_p + 0.2*auc
print(f"AutoGluon OOF:  F1_a={f1_a:.4f}  F1_p={f1_p:.4f}  AUC={auc:.4f}  OV={ov:.4f}")
print(f"R-034 baseline OOF: OV ≈ 0.3792 (n=200) / 0.3781 (n=80)")
# -

# ## 10. Download outputs locally
#
# In your local PowerShell:
#
# ```powershell
# kaggle kernels output jabir95tsai/<this-notebook-slug> -p oof_predictions/
# python -u src/audit_all_parked_components.py --n-samples 200
# ```
#
# The new tag `v14_seed2_autogluon_full` will appear in the audit ranking.
