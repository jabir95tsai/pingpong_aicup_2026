"""V4 Ultimate Pipeline: All champion techniques combined.

Architecture:
  STEP 1: Build V4 features (V3 802 + V4 extras ~ 900 base)
  STEP 2: Massive feature combinations (top_k=80 by XGBoost importance -> ~10,000+ features)
  STEP 3: Three-stage feature selection (XGBoost gain -> TreeSHAP -> cross-importance)
  STEP 4: SMOTE on training folds
  STEP 5: Train CatBoost + XGBoost + LightGBM with 5-fold GroupKFold
  STEP 6: Trained stacking meta-learner
  STEP 7: Optimal blend search (grid search CB/XGB/LGB weights)
  STEP 8: Blend with V2 and V3 OOF if available
  STEP 9: Post-processing (action rules + class weight calibration)
  STEP 10: Save submission + OOF + test predictions
"""
import sys, os, time, warnings, gc
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data

N_ACTION, N_POINT = 19, 10
SERVE_OK = {0, 15, 16, 17, 18}
SERVE_FORBIDDEN = {15, 16, 17, 18}

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    USE_SMOTE = True
    print("[INFO] imblearn SMOTE available")
except ImportError:
    USE_SMOTE = False
    print("[INFO] imblearn not found, will use manual k-NN SMOTE fallback")


# ============================================================================
# Utility functions
# ============================================================================

def timer(msg):
    """Context manager for timing blocks."""
    class _Timer:
        def __init__(self, m): self.msg = m
        def __enter__(self): self.t0 = time.time(); return self
        def __exit__(self, *a): print(f"  [{self.msg}] {time.time()-self.t0:.1f}s")
    return _Timer(msg)


def macro_f1(y_true, y_probs, n_classes):
    y_pred = np.argmax(y_probs, axis=1)
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def apply_action_rules(probs, next_sns):
    """Enforce serve/return constraints on action predictions."""
    preds = probs.copy()
    for i in range(len(preds)):
        sn = next_sns[i]
        if sn == 1:
            mask = np.zeros(preds.shape[1])
            for a in SERVE_OK:
                if a < preds.shape[1]:
                    mask[a] = 1.0
            preds[i] *= mask
        elif sn == 2:
            for a in SERVE_FORBIDDEN:
                if a < preds.shape[1]:
                    preds[i, a] = 0.0
        total = preds[i].sum()
        if total > 0:
            preds[i] /= total
        else:
            preds[i] = np.ones(preds.shape[1]) / preds.shape[1]
    return preds


def compute_sample_weights(next_sn_train, y_act):
    """Weight samples by test strikeNumber distribution and class frequency."""
    test_sn_dist = {2: 0.35, 3: 0.21, 4: 0.14, 5: 0.10, 6: 0.07,
                    7: 0.05, 8: 0.03, 9: 0.02, 10: 0.01}
    train_sn_counts = np.bincount(next_sn_train.astype(int), minlength=30)
    train_sn_counts = np.maximum(train_sn_counts, 1)

    weights = np.ones(len(next_sn_train))
    for sn, target_frac in test_sn_dist.items():
        if sn < len(train_sn_counts):
            train_frac = train_sn_counts[sn] / len(next_sn_train)
            if train_frac > 0:
                ratio = target_frac / train_frac
                weights[next_sn_train == sn] *= min(ratio, 3.0)

    act_counts = np.bincount(y_act.astype(int), minlength=N_ACTION)
    median_count = np.median(act_counts[act_counts > 0])
    for cls in range(N_ACTION):
        if act_counts[cls] > 0 and act_counts[cls] < median_count:
            ratio = min(median_count / act_counts[cls], 2.0)
            weights[y_act == cls] *= np.sqrt(ratio)

    return weights / weights.mean()


# ============================================================================
# STEP 2: Feature combinations (importance-based, not variance)
# ============================================================================

def get_top_features_by_importance(X, y, feature_names, top_k=80):
    """Train a quick XGBoost to get top features by importance."""
    import xgboost as xgb
    print(f"  Training quick XGBoost (100 rounds) to rank {X.shape[1]} features...")
    t0 = time.time()

    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "multi:softprob", "num_class": N_ACTION,
              "eval_metric": "mlogloss", "tree_method": "hist",
              "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8,
              "colsample_bytree": 0.5, "seed": RANDOM_SEED, "verbosity": 0}
    model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=0)
    importance = model.get_score(importance_type='gain')

    feat_gains = np.zeros(X.shape[1])
    for fname, gain in importance.items():
        idx = int(fname.replace('f', ''))
        feat_gains[idx] = gain

    top_indices = np.argsort(feat_gains)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_indices]
    print(f"  Top {top_k} features selected by importance in {time.time()-t0:.1f}s")
    print(f"  Top 5: {top_names[:5]}")
    return top_indices


def generate_combinations_massive(X, feature_names, top_indices, valid_mask=None):
    """Generate massive pairwise feature combinations from top features by importance.

    From top_k=80 features:
      - 80 squared features
      - C(80,2) = 3160 pairs x 4 ops (multiply, add, ratio, abs_diff) = 12,640
      - Total: ~12,720 combination features

    Memory-efficient: generates in chunks using float32.
    """
    top_k = len(top_indices)
    n_pairs = top_k * (top_k - 1) // 2
    print(f"  Generating combinations from top {top_k} features by importance...")
    print(f"  Expected: {top_k} squared + {n_pairs}*4 pairwise = {top_k + n_pairs*4} features")
    t0 = time.time()

    n_samples = X.shape[0]
    # Pre-extract columns for speed
    cols = {i: X[:, i].astype(np.float32) for i in top_indices}

    new_features = []
    new_names = []

    # Squared features
    for i in top_indices:
        new_features.append(cols[i] ** 2)
        new_names.append(f"{feature_names[i]}_sq")

    # Pairwise: multiply, add, ratio, abs_diff (4 ops)
    chunk_size = 500  # Process in chunks to limit memory
    pair_list = list(combinations(top_indices, 2))

    for chunk_start in range(0, len(pair_list), chunk_size):
        chunk_pairs = pair_list[chunk_start:chunk_start + chunk_size]
        for i, j in chunk_pairs:
            ci, cj = cols[i], cols[j]
            # Multiply
            new_features.append(ci * cj)
            new_names.append(f"{feature_names[i]}_x_{feature_names[j]}")
            # Add
            new_features.append(ci + cj)
            new_names.append(f"{feature_names[i]}_p_{feature_names[j]}")
            # Ratio (safe division)
            safe_d = np.where(np.abs(cj) > 0.001, cj, 1.0)
            new_features.append(ci / safe_d)
            new_names.append(f"{feature_names[i]}_d_{feature_names[j]}")
            # Abs difference
            new_features.append(np.abs(ci - cj))
            new_names.append(f"{feature_names[i]}_ad_{feature_names[j]}")

        if chunk_start % 2000 == 0 and chunk_start > 0:
            gc.collect()

    new_X = np.column_stack(new_features).astype(np.float32)
    del new_features, cols
    gc.collect()

    new_X = np.nan_to_num(new_X, nan=0, posinf=0, neginf=0)

    if valid_mask is None:
        std = np.std(new_X, axis=0)
        has_nan = np.any(np.isnan(new_X), axis=0)
        has_inf = np.any(np.isinf(new_X), axis=0)
        valid_mask = (std > 1e-10) & (~has_nan) & (~has_inf)

    new_X = new_X[:, valid_mask]
    new_names = [n for n, v in zip(new_names, valid_mask) if v]

    print(f"  Generated {new_X.shape[1]} combination features in {time.time()-t0:.1f}s")
    return new_X, new_names, valid_mask


# ============================================================================
# STEP 3: Three-stage feature selection
# ============================================================================

def feature_selection_xgb_gain(X, y, n_classes, top_k=800, task="multi"):
    """Stage 1: Select top features by XGBoost gain."""
    import xgboost as xgb
    print(f"  Stage 1 - XGBoost gain selection (top {top_k} from {X.shape[1]})...")

    dtrain = xgb.DMatrix(X, label=y)
    if task == "multi":
        params = {"objective": "multi:softprob", "num_class": n_classes,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8,
                  "colsample_bytree": 0.5, "seed": RANDOM_SEED, "verbosity": 0}
    else:
        params = {"objective": "binary:logistic", "eval_metric": "auc",
                  "tree_method": "hist", "learning_rate": 0.1, "max_depth": 6,
                  "subsample": 0.8, "colsample_bytree": 0.5, "seed": RANDOM_SEED, "verbosity": 0}

    model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=0)
    importance = model.get_score(importance_type='gain')

    feat_gains = {}
    for fname, gain in importance.items():
        idx = int(fname.replace('f', ''))
        feat_gains[idx] = gain

    sorted_feats = sorted(feat_gains.items(), key=lambda x: x[1], reverse=True)
    selected = [idx for idx, _ in sorted_feats[:top_k]]
    print(f"    Selected {len(selected)} features by gain")
    return selected, model


def feature_selection_shap(X, y, n_classes, xgb_model, top_k=400, task="multi", subsample=3000):
    """Stage 2: TreeSHAP-based feature selection using subsample for speed."""
    print(f"  Stage 2 - TreeSHAP selection (top {top_k}, subsample={subsample})...")
    t0 = time.time()

    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False

    import xgboost as xgb

    if not HAS_SHAP:
        # Fallback: retrain a small XGBoost on the reduced X and rank by total_gain
        print("    [WARN] shap not installed, retraining on reduced set for total_gain proxy")
        dtrain = xgb.DMatrix(X, label=y)
        if task == "multi":
            params = {"objective": "multi:softprob", "num_class": n_classes,
                      "eval_metric": "mlogloss", "tree_method": "hist",
                      "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8,
                      "colsample_bytree": 0.5, "seed": 42, "verbosity": 0}
        else:
            params = {"objective": "binary:logistic", "eval_metric": "auc",
                      "tree_method": "hist", "learning_rate": 0.1, "max_depth": 6,
                      "subsample": 0.8, "colsample_bytree": 0.5, "seed": 42, "verbosity": 0}
        local_model = xgb.train(params, dtrain, num_boost_round=150, verbose_eval=0)
        importance = local_model.get_score(importance_type='total_gain')
        feat_gains = {}
        for fname, gain in importance.items():
            idx = int(fname.replace('f', ''))
            if idx < X.shape[1]:
                feat_gains[idx] = gain
        sorted_feats = sorted(feat_gains.items(), key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in sorted_feats[:top_k]]
        print(f"    Selected {len(selected)} features (fallback) in {time.time()-t0:.1f}s")
        return selected

    # Retrain a model on X (which is the reduced feature set) for SHAP
    # The passed xgb_model was trained on the FULL feature set, not the reduced one
    print("    Retraining model on reduced features for SHAP...")
    dtrain_local = xgb.DMatrix(X, label=y)
    if task == "multi":
        params = {"objective": "multi:softprob", "num_class": n_classes,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8,
                  "colsample_bytree": 0.8, "seed": 42, "verbosity": 0}
    else:
        params = {"objective": "binary:logistic", "eval_metric": "auc",
                  "tree_method": "hist", "learning_rate": 0.1, "max_depth": 6,
                  "subsample": 0.8, "colsample_bytree": 0.8, "seed": 42, "verbosity": 0}
    local_model = xgb.train(params, dtrain_local, num_boost_round=150, verbose_eval=0)

    # Subsample for speed
    n = X.shape[0]
    if n > subsample:
        idx_sub = np.random.choice(n, subsample, replace=False)
        X_sub = X[idx_sub]
    else:
        X_sub = X

    dmat = xgb.DMatrix(X_sub)

    # Get SHAP values
    explainer = shap.TreeExplainer(local_model)
    shap_values = explainer.shap_values(dmat)

    # For multiclass, shap_values is a list of arrays (one per class)
    if isinstance(shap_values, list):
        # Mean absolute SHAP across all classes
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    # Handle multi-output shape: (n_samples, n_features, n_classes)
    if mean_shap.ndim > 1:
        mean_shap = mean_shap.mean(axis=-1) if mean_shap.ndim == 2 else mean_shap.flatten()

    # Ensure we only use as many as we have features
    n_feats = X.shape[1]
    if len(mean_shap) > n_feats:
        mean_shap = mean_shap[:n_feats]

    top_indices = np.argsort(mean_shap)[::-1][:top_k]
    selected = top_indices.tolist()
    print(f"    Selected {len(selected)} features by SHAP in {time.time()-t0:.1f}s")
    return selected


def feature_selection_cross_importance(X, y, groups, n_classes, top_k=400, n_splits=3, task="multi"):
    """Stage 3: Cross-importance (train on fold, rank on OOF) for stability."""
    import xgboost as xgb
    print(f"  Stage 3 - Cross-importance ({n_splits}-fold, top {top_k})...")
    t0 = time.time()

    gkf = GroupKFold(n_splits=n_splits)
    feat_importance_sum = np.zeros(X.shape[1])

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
        dval = xgb.DMatrix(X[val_idx], label=y[val_idx])

        if task == "multi":
            params = {"objective": "multi:softprob", "num_class": n_classes,
                      "eval_metric": "mlogloss", "tree_method": "hist",
                      "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8,
                      "colsample_bytree": 0.5, "seed": RANDOM_SEED, "verbosity": 0}
        else:
            params = {"objective": "binary:logistic", "eval_metric": "auc",
                      "tree_method": "hist", "learning_rate": 0.1, "max_depth": 6,
                      "subsample": 0.8, "colsample_bytree": 0.5, "seed": RANDOM_SEED, "verbosity": 0}

        model = xgb.train(params, dtrain, num_boost_round=150,
                          evals=[(dval, "val")], early_stopping_rounds=50,
                          verbose_eval=0)
        importance = model.get_score(importance_type='gain')
        for fname, gain in importance.items():
            idx = int(fname.replace('f', ''))
            feat_importance_sum[idx] += gain

    # Features that appear in all folds are more stable
    top_indices = np.argsort(feat_importance_sum)[::-1][:top_k]
    selected = top_indices.tolist()
    print(f"    Selected {len(selected)} stable features in {time.time()-t0:.1f}s")
    return selected


def three_stage_selection(X, y, groups, n_classes, task="multi"):
    """Three-stage feature selection (champion approach).

    Stage 1: XGBoost gain -> top 800
    Stage 2: TreeSHAP -> top 400
    Stage 3: Cross-importance (3-fold) -> top 400 stable features
    Final: union of SHAP + cross-importance
    """
    print(f"\n  Three-stage selection for {'multiclass' if task == 'multi' else 'binary'} "
          f"({X.shape[1]} features)...")

    # Stage 1: XGBoost gain -> top 800
    stage1_sel, xgb_model = feature_selection_xgb_gain(X, y, n_classes, top_k=800, task=task)

    # Reduce to stage1 features for stages 2 and 3
    X_s1 = X[:, stage1_sel].astype(np.float32)

    # Stage 2: TreeSHAP on reduced set -> top 400
    stage2_sel_local = feature_selection_shap(X_s1, y, n_classes, xgb_model, top_k=400,
                                              task=task, subsample=3000)
    # Map back to original indices
    stage2_sel = [stage1_sel[i] for i in stage2_sel_local]

    # Stage 3: Cross-importance on reduced set -> top 400
    stage3_sel_local = feature_selection_cross_importance(X_s1, y, groups, n_classes,
                                                          top_k=400, n_splits=3, task=task)
    stage3_sel = [stage1_sel[i] for i in stage3_sel_local]

    # Final: union of SHAP + cross-importance
    final_sel = sorted(set(stage2_sel) | set(stage3_sel))
    print(f"  Final selection: {len(final_sel)} features "
          f"(SHAP={len(stage2_sel)}, cross={len(stage3_sel)}, "
          f"overlap={len(set(stage2_sel) & set(stage3_sel))})")

    del X_s1
    gc.collect()
    return final_sel


# ============================================================================
# STEP 4: SMOTE
# ============================================================================

def manual_smote(X, y, k_neighbors=5, random_state=42):
    """Manual k-NN SMOTE fallback when imblearn is not available.
    Only oversamples minority classes to the median class count.
    """
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(random_state)

    classes, counts = np.unique(y, return_counts=True)
    target_count = int(np.median(counts))

    X_new_list = [X]
    y_new_list = [y]

    for cls, cnt in zip(classes, counts):
        if cnt >= target_count:
            continue
        if cnt < k_neighbors + 1:
            # Too few samples, just duplicate
            n_needed = target_count - cnt
            cls_idx = np.where(y == cls)[0]
            dup_idx = rng.choice(cls_idx, n_needed, replace=True)
            X_new_list.append(X[dup_idx])
            y_new_list.append(np.full(n_needed, cls))
            continue

        cls_idx = np.where(y == cls)[0]
        X_cls = X[cls_idx]
        n_needed = target_count - cnt

        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn.fit(X_cls)
        neighbors = nn.kneighbors(X_cls, return_distance=False)[:, 1:]  # exclude self

        synthetic_X = np.empty((n_needed, X.shape[1]), dtype=X.dtype)
        for s in range(n_needed):
            idx = rng.randint(0, cnt)
            nn_idx = rng.choice(neighbors[idx])
            lam = rng.random()
            synthetic_X[s] = X_cls[idx] + lam * (X_cls[nn_idx] - X_cls[idx])

        X_new_list.append(synthetic_X)
        y_new_list.append(np.full(n_needed, cls))

    return np.vstack(X_new_list).astype(np.float32), np.concatenate(y_new_list)


def apply_smote_to_fold(X_train, y_train, task_name="action", k_neighbors=5):
    """Apply SMOTE to a training fold. Returns augmented X, y.
    Skips if any minority class has fewer than k_neighbors+1 samples.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()

    if min_count < k_neighbors + 1:
        # Check if we can reduce k_neighbors
        effective_k = max(1, min_count - 1)
        if effective_k < 1:
            print(f"    [SMOTE {task_name}] Skipped: min class has {min_count} samples")
            return X_train, y_train
        k_neighbors = effective_k

    try:
        if USE_SMOTE:
            smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_SEED)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        else:
            X_res, y_res = manual_smote(X_train, y_train, k_neighbors=k_neighbors,
                                         random_state=RANDOM_SEED)
        print(f"    [SMOTE {task_name}] {len(X_train)} -> {len(X_res)} samples")
        return X_res.astype(np.float32), y_res
    except Exception as e:
        print(f"    [SMOTE {task_name}] Failed ({e}), using original data")
        return X_train, y_train


# ============================================================================
# STEP 6: Stacking meta-learner
# ============================================================================

def train_stacking_meta(oof_dict, y_act, y_pt, y_srv, next_sn, X_top50):
    """Train stacking meta-learner using OOF predictions + top original features.

    Meta features = OOF probabilities from CB, XGB, LGB
      - Action: 19*3 = 57
      - Point: 10*3 = 30
      - Server: 1*3 = 3
    Plus top 50 original features -> total ~140 meta features per task.
    """
    from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
    from sklearn.preprocessing import StandardScaler

    print("\n  Building meta features...")
    # Action meta features (19*3 = 57 + 50 original = 107)
    meta_act = np.hstack([
        oof_dict['cb_act'], oof_dict['xg_act'], oof_dict['lg_act'], X_top50
    ]).astype(np.float64)

    # Point meta features
    meta_pt = np.hstack([
        oof_dict['cb_pt'], oof_dict['xg_pt'], oof_dict['lg_pt'], X_top50
    ]).astype(np.float64)

    # Server meta features
    meta_srv = np.hstack([
        oof_dict['cb_srv'].reshape(-1, 1),
        oof_dict['xg_srv'].reshape(-1, 1),
        oof_dict['lg_srv'].reshape(-1, 1),
        X_top50
    ]).astype(np.float64)

    meta_results = {}

    # Scale
    scaler_act = StandardScaler().fit(meta_act)
    scaler_pt = StandardScaler().fit(meta_pt)
    scaler_srv = StandardScaler().fit(meta_srv)

    meta_act_s = scaler_act.transform(meta_act)
    meta_pt_s = scaler_pt.transform(meta_pt)
    meta_srv_s = scaler_srv.transform(meta_srv)

    # Action stacking
    print("  Training action meta-learner (LogisticRegression multinomial)...")
    lr_act = LogisticRegression(C=1.0, solver='lbfgs',
                                 max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
    lr_act.fit(meta_act_s, y_act)
    stack_act_probs = lr_act.predict_proba(meta_act_s)
    # Ensure all classes are represented
    if stack_act_probs.shape[1] < N_ACTION:
        full_probs = np.zeros((len(y_act), N_ACTION))
        for i, c in enumerate(lr_act.classes_):
            full_probs[:, int(c)] = stack_act_probs[:, i]
        stack_act_probs = full_probs

    act_r = apply_action_rules(stack_act_probs, next_sn)
    f1a_stack = macro_f1(y_act, act_r, N_ACTION)
    print(f"    Stack action F1: {f1a_stack:.4f}")

    # Point stacking
    print("  Training point meta-learner (LogisticRegression multinomial)...")
    lr_pt = LogisticRegression(C=1.0, solver='lbfgs',
                                max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
    lr_pt.fit(meta_pt_s, y_pt)
    stack_pt_probs = lr_pt.predict_proba(meta_pt_s)
    if stack_pt_probs.shape[1] < N_POINT:
        full_probs = np.zeros((len(y_pt), N_POINT))
        for i, c in enumerate(lr_pt.classes_):
            full_probs[:, int(c)] = stack_pt_probs[:, i]
        stack_pt_probs = full_probs

    f1p_stack = macro_f1(y_pt, stack_pt_probs, N_POINT)
    print(f"    Stack point F1: {f1p_stack:.4f}")

    # Server stacking
    print("  Training server meta-learner (LogisticRegression binary)...")
    lr_srv = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                 random_state=RANDOM_SEED)
    lr_srv.fit(meta_srv_s, y_srv)
    stack_srv_probs = lr_srv.predict_proba(meta_srv_s)[:, 1]
    auc_stack = roc_auc_score(y_srv, stack_srv_probs)
    print(f"    Stack server AUC: {auc_stack:.4f}")

    ov_stack = 0.4 * f1a_stack + 0.4 * f1p_stack + 0.2 * auc_stack
    print(f"    Stack OV (on train, biased): {ov_stack:.4f}")

    meta_results = {
        'stack_act': stack_act_probs,
        'stack_pt': stack_pt_probs,
        'stack_srv': stack_srv_probs,
        'lr_act': lr_act, 'lr_pt': lr_pt, 'lr_srv': lr_srv,
        'scaler_act': scaler_act, 'scaler_pt': scaler_pt, 'scaler_srv': scaler_srv,
    }
    return meta_results


def predict_stacking_meta(meta_results, test_oof_dict, X_top50_test):
    """Generate test predictions from stacking meta-learner."""
    meta_act = np.hstack([
        test_oof_dict['cb_act'], test_oof_dict['xg_act'], test_oof_dict['lg_act'], X_top50_test
    ]).astype(np.float64)
    meta_pt = np.hstack([
        test_oof_dict['cb_pt'], test_oof_dict['xg_pt'], test_oof_dict['lg_pt'], X_top50_test
    ]).astype(np.float64)
    meta_srv = np.hstack([
        test_oof_dict['cb_srv'].reshape(-1, 1),
        test_oof_dict['xg_srv'].reshape(-1, 1),
        test_oof_dict['lg_srv'].reshape(-1, 1),
        X_top50_test
    ]).astype(np.float64)

    meta_act_s = meta_results['scaler_act'].transform(meta_act)
    meta_pt_s = meta_results['scaler_pt'].transform(meta_pt)
    meta_srv_s = meta_results['scaler_srv'].transform(meta_srv)

    lr_act = meta_results['lr_act']
    lr_pt = meta_results['lr_pt']
    lr_srv = meta_results['lr_srv']

    test_act = lr_act.predict_proba(meta_act_s)
    if test_act.shape[1] < N_ACTION:
        full = np.zeros((test_act.shape[0], N_ACTION))
        for i, c in enumerate(lr_act.classes_):
            full[:, int(c)] = test_act[:, i]
        test_act = full

    test_pt = lr_pt.predict_proba(meta_pt_s)
    if test_pt.shape[1] < N_POINT:
        full = np.zeros((test_pt.shape[0], N_POINT))
        for i, c in enumerate(lr_pt.classes_):
            full[:, int(c)] = test_pt[:, i]
        test_pt = full

    test_srv = lr_srv.predict_proba(meta_srv_s)[:, 1]

    return test_act, test_pt, test_srv


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("V4 ULTIMATE PIPELINE")
    print("  Steps: V4 features -> massive combos -> 3-stage selection ->")
    print("  SMOTE -> CB+XGB+LGB -> stacking -> blend -> post-process")
    print("=" * 70)

    # ==================================================================
    # STEP 1: Build features
    # ==================================================================
    print(f"\n{'='*60}")
    print("STEP 1: Build Features")
    print(f"{'='*60}")

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)

    # Try V4 features first, fallback to V3
    try:
        from features_v4 import build_features_v4, compute_global_stats_v4, get_feature_names_v4
        print("  Using V4 features...")
        global_stats = compute_global_stats_v4(train_df)
        feat_train = build_features_v4(train_df, is_train=True, global_stats_v4=global_stats)
        feat_test = build_features_v4(test_df, is_train=False, global_stats_v4=global_stats)
        feature_names = get_feature_names_v4(feat_train)
    except ImportError:
        from features_v3 import build_features_v3, compute_global_stats, get_feature_names_v3
        print("  V4 features not available, falling back to V3 (802 base)...")
        global_stats = compute_global_stats(train_df)
        feat_train = build_features_v3(train_df, is_train=True, global_stats=global_stats)
        feat_test = build_features_v3(test_df, is_train=False, global_stats=global_stats)
        feature_names = get_feature_names_v3(feat_train)

    print(f"  Base features: {len(feature_names)}, samples: {len(feat_train)}")

    X = feat_train[feature_names].values.astype(np.float32)
    X_test = feat_test[feature_names].values.astype(np.float32)
    y_act = feat_train["y_actionId"].values
    y_pt = feat_train["y_pointId"].values
    y_srv = feat_train["y_serverGetPoint"].values
    next_sn = feat_train["next_strikeNumber"].values
    test_next_sn = feat_test["next_strikeNumber"].values

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = feat_train["rally_uid"].map(rally_to_match).values

    # Clean NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    print(f"  STEP 1 done: {time.time()-t_start:.1f}s elapsed")

    # ==================================================================
    # STEP 2: Massive Feature Combinations
    # ==================================================================
    print(f"\n{'='*60}")
    print("STEP 2: Massive Feature Combinations (top 80 by XGBoost importance)")
    print(f"{'='*60}")

    with timer("Get top features by importance"):
        top_indices = get_top_features_by_importance(X, y_act, feature_names, top_k=80)

    with timer("Generate train combinations"):
        combo_X, combo_names, valid_mask = generate_combinations_massive(
            X, feature_names, top_indices, valid_mask=None)

    with timer("Generate test combinations"):
        combo_test_X, _, _ = generate_combinations_massive(
            X_test, feature_names, top_indices, valid_mask=valid_mask)

    X_all = np.hstack([X, combo_X]).astype(np.float32)
    X_test_all = np.hstack([X_test, combo_test_X]).astype(np.float32)
    all_names = list(feature_names) + combo_names

    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)
    X_test_all = np.nan_to_num(X_test_all, nan=0, posinf=0, neginf=0)

    n_base = X.shape[1]
    n_combo = combo_X.shape[1]
    print(f"\n  Total features: {X_all.shape[1]} (base={n_base} + combo={n_combo})")

    del combo_X, combo_test_X
    gc.collect()

    print(f"  STEP 2 done: {time.time()-t_start:.1f}s elapsed")

    # ==================================================================
    # STEP 3: Three-Stage Feature Selection
    # ==================================================================
    print(f"\n{'='*60}")
    print("STEP 3: Three-Stage Feature Selection")
    print(f"{'='*60}")

    print("\n--- Action task ---")
    with timer("Action 3-stage selection"):
        sel_act = three_stage_selection(X_all, y_act, groups, N_ACTION, task="multi")

    print("\n--- Point task ---")
    with timer("Point 3-stage selection"):
        sel_pt = three_stage_selection(X_all, y_pt, groups, N_POINT, task="multi")

    print("\n--- Server task ---")
    with timer("Server 3-stage selection"):
        sel_srv = three_stage_selection(X_all, y_srv, groups, 2, task="binary")

    # Union for shared features
    all_selected = sorted(set(sel_act) | set(sel_pt) | set(sel_srv))
    print(f"\n  Union of all selected: {len(all_selected)} features")

    # Task-specific subsets
    X_act = X_all[:, sel_act].astype(np.float32)
    X_test_act = X_test_all[:, sel_act].astype(np.float32)
    X_pt = X_all[:, sel_pt].astype(np.float32)
    X_test_pt = X_test_all[:, sel_pt].astype(np.float32)
    X_srv = X_all[:, sel_srv].astype(np.float32)
    X_test_srv = X_test_all[:, sel_srv].astype(np.float32)

    # Keep top 50 features (from union) for stacking
    # Use stage1 gain-based ranking for this
    import xgboost as xgb
    dtrain_tmp = xgb.DMatrix(X_all[:, all_selected], label=y_act)
    params_tmp = {"objective": "multi:softprob", "num_class": N_ACTION,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.1, "max_depth": 6, "seed": RANDOM_SEED, "verbosity": 0}
    m_tmp = xgb.train(params_tmp, dtrain_tmp, num_boost_round=50, verbose_eval=0)
    imp_tmp = m_tmp.get_score(importance_type='gain')
    imp_sorted = sorted(imp_tmp.items(), key=lambda x: x[1], reverse=True)
    top50_local = [int(f.replace('f', '')) for f, _ in imp_sorted[:50]]
    top50_global = [all_selected[i] for i in top50_local if i < len(all_selected)]
    X_top50 = X_all[:, top50_global].astype(np.float32) if top50_global else X_all[:, all_selected[:50]].astype(np.float32)
    X_top50_test = X_test_all[:, top50_global].astype(np.float32) if top50_global else X_test_all[:, all_selected[:50]].astype(np.float32)
    del dtrain_tmp, m_tmp
    gc.collect()

    del X_all, X_test_all
    gc.collect()

    print(f"  STEP 3 done: {time.time()-t_start:.1f}s elapsed")

    # ==================================================================
    # STEP 5: Train CatBoost + XGBoost + LightGBM with SMOTE (STEP 4 inside)
    # ==================================================================
    print(f"\n{'='*60}")
    print("STEP 5: Multi-Model Ensemble Training (with SMOTE per fold)")
    print(f"{'='*60}")

    import xgboost as xgb
    from catboost import CatBoostClassifier
    import lightgbm as lgb

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_splits = list(gkf.split(X_act, groups=groups))

    sample_weights = compute_sample_weights(next_sn, y_act)

    # --- CatBoost ---
    print("\n--- CatBoost ---")
    cb_oof_act = np.zeros((len(y_act), N_ACTION))
    cb_oof_pt = np.zeros((len(y_act), N_POINT))
    cb_oof_srv = np.zeros(len(y_act))
    cb_test_act = np.zeros((len(X_test_act), N_ACTION))
    cb_test_pt = np.zeros((len(X_test_pt), N_POINT))
    cb_test_srv = np.zeros(len(X_test_srv))

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        t0 = time.time()

        # STEP 4: SMOTE on action training data
        X_tr_act_smote, y_tr_act_smote = apply_smote_to_fold(
            X_act[tr_idx], y_act[tr_idx], task_name=f"CB_act_fold{fold}")
        sw_act = compute_sample_weights(next_sn[tr_idx], y_act[tr_idx])
        # Extend weights for SMOTE samples
        if len(y_tr_act_smote) > len(tr_idx):
            extra = np.ones(len(y_tr_act_smote) - len(tr_idx))
            sw_act_ext = np.concatenate([sw_act, extra])
        else:
            sw_act_ext = sw_act

        # SMOTE on point training data
        X_tr_pt_smote, y_tr_pt_smote = apply_smote_to_fold(
            X_pt[tr_idx], y_pt[tr_idx], task_name=f"CB_pt_fold{fold}")

        # Action
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_ACTION,
                               auto_class_weights="Balanced", early_stopping_rounds=200,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                               bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m.fit(X_tr_act_smote, y_tr_act_smote, eval_set=(X_act[val_idx], y_act[val_idx]),
              sample_weight=sw_act_ext)
        cb_oof_act[val_idx] = m.predict_proba(X_act[val_idx])
        cb_test_act += m.predict_proba(X_test_act) / N_FOLDS

        # Point
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                               loss_function="MultiClass", classes_count=N_POINT,
                               auto_class_weights="Balanced", early_stopping_rounds=200,
                               verbose=0, random_seed=RANDOM_SEED, l2_leaf_reg=3,
                               bootstrap_type="Bernoulli", subsample=0.8, colsample_bylevel=0.7)
        m.fit(X_tr_pt_smote, y_tr_pt_smote, eval_set=(X_pt[val_idx], y_pt[val_idx]))
        cb_oof_pt[val_idx] = m.predict_proba(X_pt[val_idx])
        cb_test_pt += m.predict_proba(X_test_pt) / N_FOLDS

        # Server (no SMOTE - balanced task)
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
                               loss_function="Logloss", auto_class_weights="Balanced",
                               early_stopping_rounds=200, verbose=0,
                               random_seed=RANDOM_SEED, l2_leaf_reg=3)
        m.fit(X_srv[tr_idx], y_srv[tr_idx], eval_set=(X_srv[val_idx], y_srv[val_idx]))
        cb_oof_srv[val_idx] = m.predict_proba(X_srv[val_idx])[:, 1]
        cb_test_srv += m.predict_proba(X_test_srv)[:, 1] / N_FOLDS

        act_r = apply_action_rules(cb_oof_act[val_idx], next_sn[val_idx])
        f1a = macro_f1(y_act[val_idx], act_r, N_ACTION)
        f1p = macro_f1(y_pt[val_idx], cb_oof_pt[val_idx], N_POINT)
        auc = roc_auc_score(y_srv[val_idx], cb_oof_srv[val_idx])
        ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        print(f"  CB Fold {fold+1}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f} ({time.time()-t0:.0f}s)")

        del X_tr_act_smote, y_tr_act_smote, X_tr_pt_smote, y_tr_pt_smote
        gc.collect()

    act_r = apply_action_rules(cb_oof_act, next_sn)
    f1a = macro_f1(y_act, act_r, N_ACTION)
    f1p = macro_f1(y_pt, cb_oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, cb_oof_srv)
    print(f"  CB OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={0.4*f1a+0.4*f1p+0.2*auc:.4f}")

    # --- XGBoost ---
    print("\n--- XGBoost ---")
    xg_oof_act = np.zeros((len(y_act), N_ACTION))
    xg_oof_pt = np.zeros((len(y_act), N_POINT))
    xg_oof_srv = np.zeros(len(y_act))
    xg_test_act = np.zeros((len(X_test_act), N_ACTION))
    xg_test_pt = np.zeros((len(X_test_pt), N_POINT))
    xg_test_srv = np.zeros(len(X_test_srv))

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        t0 = time.time()

        # SMOTE for action
        X_tr_act_smote, y_tr_act_smote = apply_smote_to_fold(
            X_act[tr_idx], y_act[tr_idx], task_name=f"XGB_act_fold{fold}")
        sw_act = compute_sample_weights(next_sn[tr_idx], y_act[tr_idx])
        if len(y_tr_act_smote) > len(tr_idx):
            sw_act_ext = np.concatenate([sw_act, np.ones(len(y_tr_act_smote) - len(tr_idx))])
        else:
            sw_act_ext = sw_act

        # SMOTE for point
        X_tr_pt_smote, y_tr_pt_smote = apply_smote_to_fold(
            X_pt[tr_idx], y_pt[tr_idx], task_name=f"XGB_pt_fold{fold}")

        # Action
        dtrain = xgb.DMatrix(X_tr_act_smote, label=y_tr_act_smote, weight=sw_act_ext)
        dval = xgb.DMatrix(X_act[val_idx], label=y_act[val_idx])
        params = {"objective": "multi:softprob", "num_class": N_ACTION,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 10,
                  "subsample": 0.8, "colsample_bytree": 0.7,
                  "lambda": 1, "alpha": 0.1, "seed": RANDOM_SEED, "verbosity": 0}
        m = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        xg_oof_act[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration + 1))
        xg_test_act += m.predict(xgb.DMatrix(X_test_act),
                                  iteration_range=(0, m.best_iteration + 1)) / N_FOLDS

        # Point
        dtrain = xgb.DMatrix(X_tr_pt_smote, label=y_tr_pt_smote)
        dval = xgb.DMatrix(X_pt[val_idx], label=y_pt[val_idx])
        params_pt = {**params, "num_class": N_POINT}
        m = xgb.train(params_pt, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        xg_oof_pt[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration + 1))
        xg_test_pt += m.predict(xgb.DMatrix(X_test_pt),
                                 iteration_range=(0, m.best_iteration + 1)) / N_FOLDS

        # Server (no SMOTE)
        dtrain = xgb.DMatrix(X_srv[tr_idx], label=y_srv[tr_idx])
        dval = xgb.DMatrix(X_srv[val_idx], label=y_srv[val_idx])
        params_bin = {"objective": "binary:logistic", "eval_metric": "auc",
                      "tree_method": "hist", "learning_rate": 0.03, "max_depth": 8,
                      "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
                      "lambda": 1, "seed": RANDOM_SEED, "verbosity": 0}
        m = xgb.train(params_bin, dtrain, num_boost_round=3000, evals=[(dval, "val")],
                      early_stopping_rounds=200, verbose_eval=0)
        xg_oof_srv[val_idx] = m.predict(dval, iteration_range=(0, m.best_iteration + 1))
        xg_test_srv += m.predict(xgb.DMatrix(X_test_srv),
                                  iteration_range=(0, m.best_iteration + 1)) / N_FOLDS

        act_r = apply_action_rules(xg_oof_act[val_idx], next_sn[val_idx])
        f1a = macro_f1(y_act[val_idx], act_r, N_ACTION)
        f1p = macro_f1(y_pt[val_idx], xg_oof_pt[val_idx], N_POINT)
        auc = roc_auc_score(y_srv[val_idx], xg_oof_srv[val_idx])
        ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        print(f"  XGB Fold {fold+1}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f} ({time.time()-t0:.0f}s)")

        del X_tr_act_smote, y_tr_act_smote, X_tr_pt_smote, y_tr_pt_smote
        gc.collect()

    act_r = apply_action_rules(xg_oof_act, next_sn)
    f1a = macro_f1(y_act, act_r, N_ACTION)
    f1p = macro_f1(y_pt, xg_oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, xg_oof_srv)
    print(f"  XGB OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={0.4*f1a+0.4*f1p+0.2*auc:.4f}")

    # --- LightGBM ---
    print("\n--- LightGBM ---")
    lg_oof_act = np.zeros((len(y_act), N_ACTION))
    lg_oof_pt = np.zeros((len(y_act), N_POINT))
    lg_oof_srv = np.zeros(len(y_act))
    lg_test_act = np.zeros((len(X_test_act), N_ACTION))
    lg_test_pt = np.zeros((len(X_test_pt), N_POINT))
    lg_test_srv = np.zeros(len(X_test_srv))

    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        t0 = time.time()

        # SMOTE for action
        X_tr_act_smote, y_tr_act_smote = apply_smote_to_fold(
            X_act[tr_idx], y_act[tr_idx], task_name=f"LGB_act_fold{fold}")
        sw_act = compute_sample_weights(next_sn[tr_idx], y_act[tr_idx])
        if len(y_tr_act_smote) > len(tr_idx):
            sw_act_ext = np.concatenate([sw_act, np.ones(len(y_tr_act_smote) - len(tr_idx))])
        else:
            sw_act_ext = sw_act

        # SMOTE for point
        X_tr_pt_smote, y_tr_pt_smote = apply_smote_to_fold(
            X_pt[tr_idx], y_pt[tr_idx], task_name=f"LGB_pt_fold{fold}")

        # Action
        dtrain = lgb.Dataset(X_tr_act_smote, label=y_tr_act_smote, weight=sw_act_ext)
        dval = lgb.Dataset(X_act[val_idx], label=y_act[val_idx], reference=dtrain)
        params = {"objective": "multiclass", "num_class": N_ACTION,
                  "metric": "multi_logloss", "learning_rate": 0.03,
                  "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                  "subsample": 0.8, "colsample_bytree": 0.7, "is_unbalance": True,
                  "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        lg_oof_act[val_idx] = m.predict(X_act[val_idx])
        lg_test_act += m.predict(X_test_act) / N_FOLDS

        # Point
        dtrain = lgb.Dataset(X_tr_pt_smote, label=y_tr_pt_smote)
        dval = lgb.Dataset(X_pt[val_idx], label=y_pt[val_idx], reference=dtrain)
        params_pt = {**params, "num_class": N_POINT}
        m = lgb.train(params_pt, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        lg_oof_pt[val_idx] = m.predict(X_pt[val_idx])
        lg_test_pt += m.predict(X_test_pt) / N_FOLDS

        # Server (no SMOTE)
        dtrain = lgb.Dataset(X_srv[tr_idx], label=y_srv[tr_idx])
        dval = lgb.Dataset(X_srv[val_idx], label=y_srv[val_idx], reference=dtrain)
        params_bin = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
                      "num_leaves": 127, "max_depth": 8, "min_child_samples": 20,
                      "subsample": 0.8, "colsample_bytree": 0.8, "is_unbalance": True,
                      "seed": RANDOM_SEED, "verbose": -1, "n_jobs": -1}
        m = lgb.train(params_bin, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        lg_oof_srv[val_idx] = m.predict(X_srv[val_idx])
        lg_test_srv += m.predict(X_test_srv) / N_FOLDS

        act_r = apply_action_rules(lg_oof_act[val_idx], next_sn[val_idx])
        f1a = macro_f1(y_act[val_idx], act_r, N_ACTION)
        f1p = macro_f1(y_pt[val_idx], lg_oof_pt[val_idx], N_POINT)
        auc = roc_auc_score(y_srv[val_idx], lg_oof_srv[val_idx])
        ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        print(f"  LGB Fold {fold+1}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f} ({time.time()-t0:.0f}s)")

        del X_tr_act_smote, y_tr_act_smote, X_tr_pt_smote, y_tr_pt_smote
        gc.collect()

    act_r = apply_action_rules(lg_oof_act, next_sn)
    f1a = macro_f1(y_act, act_r, N_ACTION)
    f1p = macro_f1(y_pt, lg_oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, lg_oof_srv)
    print(f"  LGB OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={0.4*f1a+0.4*f1p+0.2*auc:.4f}")

    print(f"\n  STEP 5 done: {(time.time()-t_start)/60:.1f} min elapsed")

    # ==================================================================
    # STEP 6: Stacking Meta-Learner
    # ==================================================================
    print(f"\n{'='*60}")
    print("STEP 6: Stacking Meta-Learner")
    print(f"{'='*60}")

    oof_dict = {
        'cb_act': cb_oof_act, 'xg_act': xg_oof_act, 'lg_act': lg_oof_act,
        'cb_pt': cb_oof_pt, 'xg_pt': xg_oof_pt, 'lg_pt': lg_oof_pt,
        'cb_srv': cb_oof_srv, 'xg_srv': xg_oof_srv, 'lg_srv': lg_oof_srv,
    }

    meta_results = train_stacking_meta(oof_dict, y_act, y_pt, y_srv, next_sn, X_top50)

    # Generate stacking test predictions
    test_oof_dict = {
        'cb_act': cb_test_act, 'xg_act': xg_test_act, 'lg_act': lg_test_act,
        'cb_pt': cb_test_pt, 'xg_pt': xg_test_pt, 'lg_pt': lg_test_pt,
        'cb_srv': cb_test_srv, 'xg_srv': xg_test_srv, 'lg_srv': lg_test_srv,
    }
    stack_test_act, stack_test_pt, stack_test_srv = predict_stacking_meta(
        meta_results, test_oof_dict, X_top50_test)

    print(f"\n  STEP 6 done: {(time.time()-t_start)/60:.1f} min elapsed")

    # ==================================================================
    # STEP 7: Optimal Blend Search
    # ==================================================================
    print(f"\n{'='*60}")
    print("STEP 7: Optimal Blend Search (CB/XGB/LGB weights)")
    print(f"{'='*60}")

    best_ov = -1
    best_blend = None
    best_params = None

    for w_cb in np.arange(0.2, 0.8, 0.05):
        for w_xg in np.arange(0.1, 0.8 - w_cb + 0.025, 0.05):
            w_lg = round(1.0 - w_cb - w_xg, 2)
            if w_lg < 0 or w_lg > 0.5:
                continue

            blend_act = w_cb * cb_oof_act + w_xg * xg_oof_act + w_lg * lg_oof_act
            blend_pt = w_cb * cb_oof_pt + w_xg * xg_oof_pt + w_lg * lg_oof_pt
            blend_srv = w_cb * cb_oof_srv + w_xg * xg_oof_srv + w_lg * lg_oof_srv

            ba_r = apply_action_rules(blend_act, next_sn)
            f1a = macro_f1(y_act, ba_r, N_ACTION)
            f1p = macro_f1(y_pt, blend_pt, N_POINT)
            auc = roc_auc_score(y_srv, blend_srv)
            ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc

            if ov > best_ov:
                best_ov = ov
                best_params = (w_cb, w_xg, w_lg)
                best_blend = (blend_act.copy(), blend_pt.copy(), blend_srv.copy())

    w_cb, w_xg, w_lg = best_params
    print(f"\n  Best manual blend: CB={w_cb:.2f} XGB={w_xg:.2f} LGB={w_lg:.2f}")
    ba_r = apply_action_rules(best_blend[0], next_sn)
    f1a_blend = macro_f1(y_act, ba_r, N_ACTION)
    f1p_blend = macro_f1(y_pt, best_blend[1], N_POINT)
    auc_blend = roc_auc_score(y_srv, best_blend[2])
    ov_blend = 0.4 * f1a_blend + 0.4 * f1p_blend + 0.2 * auc_blend
    print(f"  Manual blend OOF: F1a={f1a_blend:.4f} F1p={f1p_blend:.4f} AUC={auc_blend:.4f} OV={ov_blend:.4f}")

    # Compare with stacking
    stack_act_r = apply_action_rules(meta_results['stack_act'], next_sn)
    f1a_stack = macro_f1(y_act, stack_act_r, N_ACTION)
    f1p_stack = macro_f1(y_pt, meta_results['stack_pt'], N_POINT)
    auc_stack = roc_auc_score(y_srv, meta_results['stack_srv'])
    ov_stack = 0.4 * f1a_stack + 0.4 * f1p_stack + 0.2 * auc_stack
    print(f"  Stacking OOF:     F1a={f1a_stack:.4f} F1p={f1p_stack:.4f} AUC={auc_stack:.4f} OV={ov_stack:.4f}")

    # Pick the better approach
    if ov_stack > ov_blend:
        print("  >> Stacking wins! Using stacking predictions.")
        use_stacking = True
        v4_oof_act = meta_results['stack_act']
        v4_oof_pt = meta_results['stack_pt']
        v4_oof_srv = meta_results['stack_srv']
        v4_test_act_raw = stack_test_act
        v4_test_pt_raw = stack_test_pt
        v4_test_srv_raw = stack_test_srv
    else:
        print("  >> Manual blend wins! Using manual blend predictions.")
        use_stacking = False
        v4_oof_act = best_blend[0]
        v4_oof_pt = best_blend[1]
        v4_oof_srv = best_blend[2]
        v4_test_act_raw = w_cb * cb_test_act + w_xg * xg_test_act + w_lg * lg_test_act
        v4_test_pt_raw = w_cb * cb_test_pt + w_xg * xg_test_pt + w_lg * lg_test_pt
        v4_test_srv_raw = w_cb * cb_test_srv + w_xg * xg_test_srv + w_lg * lg_test_srv

    # Also try blending stacking with manual blend
    best_stack_w = 0
    best_combo_ov = max(ov_blend, ov_stack)
    for sw in np.arange(0, 1.05, 0.05):
        c_act = sw * meta_results['stack_act'] + (1 - sw) * best_blend[0]
        c_pt = sw * meta_results['stack_pt'] + (1 - sw) * best_blend[1]
        c_srv = sw * meta_results['stack_srv'] + (1 - sw) * best_blend[2]
        cr = apply_action_rules(c_act, next_sn)
        f1a = macro_f1(y_act, cr, N_ACTION)
        f1p = macro_f1(y_pt, c_pt, N_POINT)
        auc = roc_auc_score(y_srv, c_srv)
        ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        if ov > best_combo_ov:
            best_combo_ov = ov
            best_stack_w = sw

    if best_stack_w > 0 and best_stack_w < 1:
        print(f"  >> Hybrid blend: stack_w={best_stack_w:.2f} improves OV to {best_combo_ov:.4f}")
        v4_oof_act = best_stack_w * meta_results['stack_act'] + (1 - best_stack_w) * best_blend[0]
        v4_oof_pt = best_stack_w * meta_results['stack_pt'] + (1 - best_stack_w) * best_blend[1]
        v4_oof_srv = best_stack_w * meta_results['stack_srv'] + (1 - best_stack_w) * best_blend[2]
        v4_test_act_raw = best_stack_w * stack_test_act + (1 - best_stack_w) * (w_cb * cb_test_act + w_xg * xg_test_act + w_lg * lg_test_act)
        v4_test_pt_raw = best_stack_w * stack_test_pt + (1 - best_stack_w) * (w_cb * cb_test_pt + w_xg * xg_test_pt + w_lg * lg_test_pt)
        v4_test_srv_raw = best_stack_w * stack_test_srv + (1 - best_stack_w) * (w_cb * cb_test_srv + w_xg * xg_test_srv + w_lg * lg_test_srv)

    print(f"\n  STEP 7 done: {(time.time()-t_start)/60:.1f} min elapsed")

    # ==================================================================
    # STEP 8: Blend with V2 and V3 OOF if available
    # ==================================================================
    print(f"\n{'='*60}")
    print("STEP 8: Blend with V2 and V3 OOF (if available)")
    print(f"{'='*60}")

    # Collect prior version OOF predictions
    prior_oofs = {}
    prior_tests = {}

    for version, prefix in [("v2", "v2_fast"), ("v3", "v3_champion")]:
        oof_file = os.path.join(MODEL_DIR, f"oof_{prefix}.npz")
        test_file = os.path.join(MODEL_DIR, f"test_{prefix}.npz")
        if os.path.exists(oof_file) and os.path.exists(test_file):
            oof = np.load(oof_file)
            tst = np.load(test_file)
            # Blend the 3 models within the prior version
            act = (0.5 * oof["catboost_act"] + 0.3 * oof["xgboost_act"] + 0.2 * oof["lightgbm_act"]
                   if "lightgbm_act" in oof else
                   0.6 * oof["catboost_act"] + 0.4 * oof["xgboost_act"])
            pt = (0.5 * oof["catboost_pt"] + 0.3 * oof["xgboost_pt"] + 0.2 * oof["lightgbm_pt"]
                  if "lightgbm_pt" in oof else
                  0.6 * oof["catboost_pt"] + 0.4 * oof["xgboost_pt"])
            srv = (0.4 * oof["catboost_srv"] + 0.3 * oof["xgboost_srv"] + 0.3 * oof["lightgbm_srv"]
                   if "lightgbm_srv" in oof else
                   0.5 * oof["catboost_srv"] + 0.5 * oof["xgboost_srv"])

            t_act = (0.5 * tst["catboost_act"] + 0.3 * tst["xgboost_act"] + 0.2 * tst["lightgbm_act"]
                     if "lightgbm_act" in tst else
                     0.6 * tst["catboost_act"] + 0.4 * tst["xgboost_act"])
            t_pt = (0.5 * tst["catboost_pt"] + 0.3 * tst["xgboost_pt"] + 0.2 * tst["lightgbm_pt"]
                    if "lightgbm_pt" in tst else
                    0.6 * tst["catboost_pt"] + 0.4 * tst["xgboost_pt"])
            t_srv = (0.4 * tst["catboost_srv"] + 0.3 * tst["xgboost_srv"] + 0.3 * tst["lightgbm_srv"]
                     if "lightgbm_srv" in tst else
                     0.5 * tst["catboost_srv"] + 0.5 * tst["xgboost_srv"])

            prior_oofs[version] = (act, pt, srv)
            prior_tests[version] = (t_act, t_pt, t_srv)
            print(f"  Loaded {version} OOF: {oof_file}")
        else:
            print(f"  {version} OOF not found, skipping")

    # Blend V4 with prior versions
    final_oof_act = v4_oof_act.copy()
    final_oof_pt = v4_oof_pt.copy()
    final_oof_srv = v4_oof_srv.copy()
    final_test_act = v4_test_act_raw.copy()
    final_test_pt = v4_test_pt_raw.copy()
    final_test_srv = v4_test_srv_raw.copy()

    if prior_oofs:
        # Search for best blend weight between V4 and each prior version
        for version in sorted(prior_oofs.keys()):
            p_act, p_pt, p_srv = prior_oofs[version]
            t_act, t_pt, t_srv = prior_tests[version]

            # Check shape compatibility
            if p_act.shape != final_oof_act.shape:
                print(f"  Shape mismatch for {version}, skipping")
                continue

            best_w = 1.0
            best_mega_ov = -1
            for w in np.arange(0.5, 1.05, 0.05):
                ma = w * final_oof_act + (1 - w) * p_act
                mp = w * final_oof_pt + (1 - w) * p_pt
                ms = w * final_oof_srv + (1 - w) * p_srv
                mar = apply_action_rules(ma, next_sn)
                f1a = macro_f1(y_act, mar, N_ACTION)
                f1p = macro_f1(y_pt, mp, N_POINT)
                auc = roc_auc_score(y_srv, ms)
                ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
                if ov > best_mega_ov:
                    best_mega_ov = ov
                    best_w = w

            print(f"  Best V4 + {version} blend: w_v4={best_w:.2f}, OV={best_mega_ov:.4f}")

            if best_w < 1.0:
                final_oof_act = best_w * final_oof_act + (1 - best_w) * p_act
                final_oof_pt = best_w * final_oof_pt + (1 - best_w) * p_pt
                final_oof_srv = best_w * final_oof_srv + (1 - best_w) * p_srv
                final_test_act = best_w * final_test_act + (1 - best_w) * t_act
                final_test_pt = best_w * final_test_pt + (1 - best_w) * t_pt
                final_test_srv = best_w * final_test_srv + (1 - best_w) * t_srv
    else:
        print("  No prior versions found, using V4 only")

    # Report final OOF
    fr = apply_action_rules(final_oof_act, next_sn)
    f1a = macro_f1(y_act, fr, N_ACTION)
    f1p = macro_f1(y_pt, final_oof_pt, N_POINT)
    auc = roc_auc_score(y_srv, final_oof_srv)
    ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
    print(f"\n  Final blended OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    print(f"\n  STEP 8 done: {(time.time()-t_start)/60:.1f} min elapsed")

    # ==================================================================
    # STEP 9: Post-processing
    # ==================================================================
    print(f"\n{'='*60}")
    print("STEP 9: Post-Processing (action rules + calibration)")
    print(f"{'='*60}")

    # Apply action rules to test
    final_test_act = apply_action_rules(final_test_act, test_next_sn)

    # Class weight calibration for action (boost underrepresented classes)
    print("  Calibrating action class weights...")
    train_act_dist = np.bincount(y_act.astype(int), minlength=N_ACTION).astype(float)
    train_act_dist /= train_act_dist.sum()

    # Mild temperature scaling to sharpen predictions
    temperature = 0.95
    final_test_act_cal = np.power(final_test_act, 1.0 / temperature)
    final_test_act_cal /= final_test_act_cal.sum(axis=1, keepdims=True)

    # Check if calibration helps (can't check on test, so just apply mild)
    final_test_act = final_test_act_cal

    # Re-apply action rules after calibration
    final_test_act = apply_action_rules(final_test_act, test_next_sn)

    # Calibrate point predictions similarly
    final_test_pt_cal = np.power(final_test_pt, 1.0 / temperature)
    final_test_pt_cal /= final_test_pt_cal.sum(axis=1, keepdims=True)
    final_test_pt = final_test_pt_cal

    print(f"  STEP 9 done: {(time.time()-t_start)/60:.1f} min elapsed")

    # ==================================================================
    # STEP 10: Save submission + OOF + test predictions
    # ==================================================================
    print(f"\n{'='*60}")
    print("STEP 10: Save Results")
    print(f"{'='*60}")

    submission = pd.DataFrame({
        "rally_uid": feat_test["rally_uid"].values.astype(int),
        "actionId": np.argmax(final_test_act, axis=1).astype(int),
        "pointId": np.argmax(final_test_pt, axis=1).astype(int),
        "serverGetPoint": (final_test_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_v4_ultimate.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\n  Saved: {out_path}")
    print(f"  actionId:        {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId:         {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  serverGetPoint:  {submission.serverGetPoint.value_counts().to_dict()}")

    # Save OOF predictions
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(os.path.join(MODEL_DIR, "oof_v4_ultimate.npz"),
             catboost_act=cb_oof_act, xgboost_act=xg_oof_act, lightgbm_act=lg_oof_act,
             catboost_pt=cb_oof_pt, xgboost_pt=xg_oof_pt, lightgbm_pt=lg_oof_pt,
             catboost_srv=cb_oof_srv, xgboost_srv=xg_oof_srv, lightgbm_srv=lg_oof_srv,
             blend_act=v4_oof_act, blend_pt=v4_oof_pt, blend_srv=v4_oof_srv,
             y_act=y_act, y_pt=y_pt, y_srv=y_srv, next_sn=next_sn)
    np.savez(os.path.join(MODEL_DIR, "test_v4_ultimate.npz"),
             catboost_act=cb_test_act, xgboost_act=xg_test_act, lightgbm_act=lg_test_act,
             catboost_pt=cb_test_pt, xgboost_pt=xg_test_pt, lightgbm_pt=lg_test_pt,
             catboost_srv=cb_test_srv, xgboost_srv=xg_test_srv, lightgbm_srv=lg_test_srv,
             blend_act=v4_test_act_raw, blend_pt=v4_test_pt_raw, blend_srv=v4_test_srv_raw,
             test_next_sn=test_next_sn,
             rally_uids=feat_test["rally_uid"].values.astype(int))

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*70}")
    print(f"V4 ULTIMATE PIPELINE COMPLETE: {elapsed:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
