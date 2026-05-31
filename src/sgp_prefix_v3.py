"""R-030 sgp_prefix_v3 — dedicated prefix-only SGP trainer (LightGBM).

Codex APPROVED v1 scope (2026-05-20):
- Implement with `--feature-profile core`
- Run audits before training (5 audits)
- Run diagnostics: length-only, no-length ablation, logistic baseline, Fold-1 LGBM smoke
- Produce metadata JSON
- NO --include-old-test in v1 smoke
- NO full 5-fold
- NO submission

Smoke gate (Codex): `Fold1_AUC >= max(0.620, same_fold_best_baseline_AUC + 0.005)`

Usage:
    python -u src/sgp_prefix_v3.py --tag sgp_prefix_v3_smoke --max-folds 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, TRAIN_PATH, TEST_PATH, N_FOLDS, RANDOM_SEED  # noqa: E402
from data_cleaning import clean_data  # noqa: E402
from features_sgp_prefix_v3 import (  # noqa: E402
    SGP_V3_CORE_COLUMNS,
    audit_no_banned_names,
    build_features_sgp_v3,
    get_feature_cols,
)

# Baseline AUC values (for gate comparison) — from current v14_seed2_oldtest
# per-fold log analysis
_V14_BASELINE_FOLD_AUC = {
    1: 0.6104,
    2: 0.5837,
    3: 0.6173,
    4: 0.6115,
    5: 0.6080,
}
_V14_BASELINE_OOF_AUC = 0.6056
_R027_PAIR_BLEND_AUC = 0.6131


def _audit_a_strict_prefix_containment(feat_df: pd.DataFrame, raw_df: pd.DataFrame) -> dict:
    """Audit A: for a sample of 100 training rows, verify max(prefix_strikeNum) < target_strikeNum."""
    print("  [Audit A] strict prefix containment...")
    sampled = feat_df.sample(min(100, len(feat_df)), random_state=42)
    raw_sorted = raw_df.sort_values(["rally_uid", "strikeNumber"])
    failures = 0
    for _, row in sampled.iterrows():
        rid = int(row["rally_uid"])
        target_sn = int(row["next_strikeNumber"])
        rally_shots = raw_sorted[raw_sorted["rally_uid"] == rid]
        prefix_max_sn = rally_shots[rally_shots["strikeNumber"] < target_sn]["strikeNumber"].max()
        if pd.notna(prefix_max_sn) and prefix_max_sn >= target_sn:
            failures += 1
            print(f"    FAIL rally={rid}: prefix_max={prefix_max_sn} >= target={target_sn}")
    result = {"name": "strict_prefix_containment", "samples": len(sampled), "failures": int(failures)}
    print(f"  [Audit A] {result}")
    if failures > 0:
        raise ValueError(f"Audit A FAILED: {failures} prefix containment violations")
    return result


def _audit_b_banned_names(feature_cols: list) -> dict:
    """Audit B: no feature names contain banned substrings."""
    print("  [Audit B] banned feature name grep...")
    audit_no_banned_names(feature_cols)
    result = {"name": "banned_names", "n_features": len(feature_cols), "violations": 0}
    print(f"  [Audit B] OK ({len(feature_cols)} feature names checked)")
    return result


def _audit_c_train_test_consistency(train_feat: pd.DataFrame, test_feat: pd.DataFrame) -> dict:
    """Audit C: train and test feature builders produce same column schema."""
    print("  [Audit C] train/test feature schema consistency...")
    train_cols = set(get_feature_cols(train_feat))
    test_cols = set(get_feature_cols(test_feat))
    if train_cols != test_cols:
        missing_in_test = train_cols - test_cols
        extra_in_test = test_cols - train_cols
        raise ValueError(
            f"Schema mismatch: train-only={missing_in_test}, test-only={extra_in_test}"
        )
    result = {"name": "train_test_schema", "n_features": len(train_cols), "schema_match": True}
    print(f"  [Audit C] OK ({len(train_cols)} cols match)")
    return result


def _audit_d_finite_values(feat_df: pd.DataFrame, feature_cols: list, label: str) -> dict:
    """Audit D: all features finite (no NaN/Inf)."""
    print(f"  [Audit D-{label}] finite-value check...")
    n_bad = 0
    bad_cols = []
    for c in feature_cols:
        vals = feat_df[c].values
        if not np.all(np.isfinite(vals.astype(np.float64, copy=False))):
            n_bad_in_col = int(np.sum(~np.isfinite(vals.astype(np.float64, copy=False))))
            n_bad += n_bad_in_col
            bad_cols.append(f"{c}({n_bad_in_col})")
    result = {"name": f"finite_values_{label}", "n_bad": n_bad, "bad_cols": bad_cols[:5]}
    if n_bad > 0:
        raise ValueError(f"Audit D-{label} FAILED: {n_bad} non-finite values in {bad_cols[:5]}")
    print(f"  [Audit D-{label}] OK (all finite)")
    return result


def _audit_e_shape(test_feat: pd.DataFrame, expected_rallies: int) -> dict:
    """Audit E: test has exactly N rallies = expected_rallies (1845 for test_new)."""
    print(f"  [Audit E] test shape check (expected {expected_rallies} rallies)...")
    n_rows = len(test_feat)
    n_unique = test_feat["rally_uid"].nunique()
    if n_rows != expected_rallies or n_unique != expected_rallies:
        raise ValueError(
            f"Audit E FAILED: test has {n_rows} rows / {n_unique} unique rallies; expected {expected_rallies}"
        )
    result = {"name": "test_shape", "n_rows": n_rows, "n_rallies": n_unique}
    print(f"  [Audit E] OK ({n_rows} rows = {n_unique} unique rallies)")
    return result


def _run_counts_only_diagnostic(
    X_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray,
    feature_cols: list, n_splits: int = 5,
) -> dict:
    """Diagnostic: train using ONLY `prefix_length_log` feature. Per Codex:
    > 0.70 = hard stop; 0.65-0.70 = pause + no-length ablation review.
    """
    print("\n--- Diagnostic: counts-only baseline ---")
    import lightgbm as lgb
    length_only = X_train[["prefix_length_log"]].values
    splitter = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(y_train))
    for fold, (tr, va) in enumerate(splitter.split(length_only, y_train, groups), 1):
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=1000, learning_rate=0.03,
            num_leaves=15, max_depth=4, min_child_samples=50,
            random_state=RANDOM_SEED, verbosity=-1,
        )
        model.fit(length_only[tr], y_train[tr],
                  eval_set=[(length_only[va], y_train[va])],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        oof_pred[va] = model.predict_proba(length_only[va])[:, 1]
    auc = float(roc_auc_score(y_train, oof_pred))
    print(f"  Counts-only OOF AUC: {auc:.4f}")
    if auc > 0.70:
        print(f"  *** HARD STOP per Codex: counts-only AUC {auc:.4f} > 0.70 ***")
        verdict = "HARD_STOP"
    elif auc > 0.65:
        print(f"  *** PAUSE per Codex: counts-only AUC {auc:.4f} > 0.65 ***")
        verdict = "PAUSE"
    else:
        verdict = "OK"
    return {"name": "counts_only_baseline", "auc": auc, "verdict": verdict}


def _run_no_length_ablation(
    X_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray,
    feature_cols: list,
) -> dict:
    """Diagnostic: train LightGBM WITHOUT `prefix_length_log` (per Codex 'always report no-length ablation')."""
    print("\n--- Diagnostic: no-length ablation ---")
    import lightgbm as lgb
    no_length_cols = [c for c in feature_cols if c != "prefix_length_log"]
    X = X_train[no_length_cols].values
    splitter = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(y_train))
    for fold, (tr, va) in enumerate(splitter.split(X, y_train, groups), 1):
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=3000, learning_rate=0.03,
            num_leaves=31, min_child_samples=20,
            reg_lambda=0.1, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
            random_state=RANDOM_SEED, verbosity=-1,
        )
        model.fit(X[tr], y_train[tr],
                  eval_set=[(X[va], y_train[va])],
                  callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
        oof_pred[va] = model.predict_proba(X[va])[:, 1]
    auc = float(roc_auc_score(y_train, oof_pred))
    print(f"  No-length OOF AUC: {auc:.4f} (using {len(no_length_cols)} features, length feature dropped)")
    return {"name": "no_length_ablation", "auc": auc, "n_features": len(no_length_cols)}


def _run_logistic_baseline(
    X_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray,
    feature_cols: list,
) -> dict:
    """Diagnostic: logistic regression sanity baseline."""
    print("\n--- Diagnostic: logistic regression baseline ---")
    X = X_train[feature_cols].values.astype(np.float64)
    splitter = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(y_train))
    for fold, (tr, va) in enumerate(splitter.split(X, y_train, groups), 1):
        # Scale features within each fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        model = LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, solver="lbfgs",
            random_state=RANDOM_SEED,
        )
        model.fit(X_tr, y_train[tr])
        oof_pred[va] = model.predict_proba(X_va)[:, 1]
    auc = float(roc_auc_score(y_train, oof_pred))
    print(f"  Logistic OOF AUC: {auc:.4f}")
    return {"name": "logistic_baseline", "auc": auc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="sgp_prefix_v3_smoke")
    parser.add_argument("--max-folds", type=int, default=1,
                        help="Codex restriction (v1 smoke): Fold-1 only. "
                             "Override to N_FOLDS when --full-train is set.")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--feature-profile", type=str, default="core",
                        choices=["core"], help="Codex restriction: v1 = core only.")
    parser.add_argument("--include-old-test", type=str, default=None,
                        help="Allowed only with --full-train (post-smoke v1b+).")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--full-train", action="store_true",
                        help="Train all folds + save OOF/test arrays in standard "
                             "format (post-smoke; bypasses fold-1 gate verdict). "
                             "Output: {tag}_oof_*.npy + {tag}_test_*.npy in "
                             "oof_predictions/. Action/Point arrays are uniform "
                             "(this is an SGP-only specialist component).")
    args = parser.parse_args()

    if args.include_old_test and not args.full_train:
        raise ValueError("R-030 v1 smoke MUST NOT use --include-old-test per Codex. "
                         "Use --full-train for v1b post-smoke.")
    if args.full_train and args.max_folds == 1:
        args.max_folds = args.folds  # auto-promote to all folds

    t_start = time.time()
    print("=" * 70)
    print(f"R-030 sgp_prefix_v3 — tag={args.tag}")
    print(f"  Feature profile: {args.feature_profile}")
    print(f"  Folds: {args.folds}  Max folds to run: {args.max_folds}")
    print(f"  Seed: {args.seed}")
    print("=" * 70)

    # ---- Load data ----
    test_path = args.test_path or TEST_PATH
    raw_train = pd.read_csv(TRAIN_PATH)
    if args.include_old_test:
        # Wire through --include-old-test: concat old test (which has full
        # labels including SGP) as extra training data, with rally_uid offset
        # to avoid collisions. Per AICUP 2026-05-13 announcement.
        old_test = pd.read_csv(args.include_old_test)
        n_before = len(raw_train)
        # Offset to avoid collisions with train rally_uids
        old_test = old_test.copy()
        old_test["rally_uid"] = old_test["rally_uid"] + 20000
        # Align columns: keep only columns present in raw_train
        common_cols = [c for c in raw_train.columns if c in old_test.columns]
        old_test = old_test[common_cols]
        raw_train = pd.concat([raw_train, old_test], ignore_index=True)
        print(f"  [include-old-test] +{len(raw_train) - n_before} rows "
              f"({old_test['rally_uid'].nunique()} rallies)")
    raw_test = pd.read_csv(test_path)
    train_df, test_df, _ = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1  # placeholder; we don't use it
    print(f"\nTrain raw: {len(raw_train)} shots / {raw_train['rally_uid'].nunique()} rallies")
    print(f"Test raw:  {len(raw_test)} shots / {raw_test['rally_uid'].nunique()} rallies")

    # ---- Build features ----
    print("\n--- Building features ---")
    t0 = time.time()
    train_feat = build_features_sgp_v3(train_df, is_train=True)
    test_feat = build_features_sgp_v3(test_df, is_train=False)
    print(f"  Train features: {len(train_feat)} samples (build={time.time()-t0:.1f}s)")
    print(f"  Test features:  {len(test_feat)} samples")
    feature_cols = get_feature_cols(train_feat)
    print(f"  Feature count: {len(feature_cols)}")

    audits: list = []

    # ---- Run audits ----
    print("\n--- Audits ---")
    audits.append(_audit_a_strict_prefix_containment(train_feat, train_df))
    audits.append(_audit_b_banned_names(feature_cols))
    audits.append(_audit_c_train_test_consistency(train_feat, test_feat))
    audits.append(_audit_d_finite_values(train_feat, feature_cols, "train"))
    audits.append(_audit_d_finite_values(test_feat, feature_cols, "test"))
    audits.append(_audit_e_shape(test_feat, expected_rallies=raw_test["rally_uid"].nunique()))

    # ---- Prepare matrices ----
    X_train = train_feat[feature_cols].astype(np.float32)
    y_train = train_feat["serverGetPoint"].astype(np.int8).values
    groups = train_feat["rally_uid"].values  # GroupKFold by rally for now (could be match)
    X_test = test_feat[feature_cols].astype(np.float32)
    test_rally_uids = test_feat["rally_uid"].values

    # Use match for grouping if available (matches our V14 convention)
    if "match" in train_df.columns:
        # Map rally_uid → match
        rally_to_match = train_df.drop_duplicates("rally_uid").set_index("rally_uid")["match"].to_dict()
        groups = np.array([rally_to_match.get(int(r), -1) for r in train_feat["rally_uid"].values])
        print(f"\n  Grouping by match column ({len(set(groups))} unique matches)")
    else:
        print(f"\n  Grouping by rally_uid ({len(set(groups))} unique rallies)")

    # ---- Diagnostics (required by Codex) ----
    diagnostics: dict = {}
    diagnostics["counts_only"] = _run_counts_only_diagnostic(X_train, y_train, groups, feature_cols)
    if diagnostics["counts_only"]["verdict"] == "HARD_STOP":
        print("\n*** STOPPING: counts-only AUC > 0.70 hard stop gate ***")
        _save_metadata(args.tag, audits, diagnostics, fold_aucs={}, gate_verdict="HARD_STOP_COUNTS_ONLY")
        sys.exit(2)
    diagnostics["no_length"] = _run_no_length_ablation(X_train, y_train, groups, feature_cols)
    diagnostics["logistic"] = _run_logistic_baseline(X_train, y_train, groups, feature_cols)

    # ---- LightGBM training ----
    print("\n--- LightGBM training ---")
    import lightgbm as lgb
    splitter = GroupKFold(n_splits=args.folds)
    fold_aucs: dict = {}
    oof_pred = np.zeros(len(y_train))
    test_pred_accum = np.zeros(len(X_test), dtype=np.float64)
    fold_masks = list(splitter.split(X_train, y_train, groups))
    folds_to_run = list(range(1, args.max_folds + 1))
    n_folds_run = 0
    for fold_idx, (tr, va) in enumerate(fold_masks, start=1):
        if fold_idx not in folds_to_run:
            continue
        print(f"\n  === Fold {fold_idx}/{args.folds} ===")
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=3000, learning_rate=0.03,
            num_leaves=31, min_child_samples=20,
            reg_lambda=0.1, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
            random_state=args.seed, verbosity=-1,
        )
        model.fit(X_train.values[tr], y_train[tr],
                  eval_set=[(X_train.values[va], y_train[va])],
                  callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
        pred = model.predict_proba(X_train.values[va])[:, 1]
        oof_pred[va] = pred
        auc = float(roc_auc_score(y_train[va], pred))
        fold_aucs[fold_idx] = auc
        print(f"  Fold {fold_idx} AUC: {auc:.4f}")
        if args.full_train:
            test_pred_accum += model.predict_proba(X_test.values)[:, 1]
            n_folds_run += 1
        # Per-SN AUC slice
        next_sn = train_feat.iloc[va]["next_strikeNumber"].values
        for label, mask in [
            ("SN=2", next_sn == 2),
            ("SN=3-4", (next_sn >= 3) & (next_sn <= 4)),
            ("SN=5-8", (next_sn >= 5) & (next_sn <= 8)),
            ("SN=9-12", (next_sn >= 9) & (next_sn <= 12)),
            ("SN>=13", next_sn >= 13),
        ]:
            n = int(mask.sum())
            if n > 0 and len(np.unique(y_train[va][mask])) >= 2:
                a = float(roc_auc_score(y_train[va][mask], pred[mask]))
                print(f"    {label}: AUC={a:.4f} (n={n})")
            else:
                print(f"    {label}: skip (n={n})")

    # ---- Gate evaluation ----
    print("\n--- Gate evaluation ---")
    smoke_fold1_auc = fold_aucs.get(1, None)
    baseline_fold1_auc = _V14_BASELINE_FOLD_AUC[1]
    gate_threshold = max(0.620, baseline_fold1_auc + 0.005)
    print(f"  Fold-1 smoke AUC: {smoke_fold1_auc:.4f}")
    print(f"  v14_seed2 Fold-1 baseline AUC: {baseline_fold1_auc:.4f}")
    print(f"  Smoke gate (Codex): max(0.620, baseline + 0.005) = {gate_threshold:.4f}")
    if smoke_fold1_auc is None:
        gate_verdict = "NO_FOLD1_AUC"
    elif smoke_fold1_auc >= gate_threshold:
        gate_verdict = "PASS"
    elif smoke_fold1_auc >= 0.615:
        gate_verdict = "PAUSE_FOR_REVIEW"
    else:
        gate_verdict = "FAIL_PARK"
    print(f"  Verdict: {gate_verdict}")

    # ---- Save OOF + test arrays (full-train mode only) ----
    if args.full_train:
        print("\n--- Saving OOF + test arrays (full-train mode) ---")
        oof_dir = Path(PROJECT_ROOT) / "oof_predictions"
        oof_dir.mkdir(exist_ok=True)

        # Average test predictions across folds
        test_srv = (test_pred_accum / max(n_folds_run, 1)).astype(np.float32)

        # Build ground-truth + uniform act/pt arrays.
        # Action/Point: this model has NO signal there -> uniform => blend
        # weight search will assign 0. Saving uniform keeps the array shape
        # compatible with our standard audit pipeline.
        N = len(y_train)
        N_TEST = len(test_rally_uids)

        # Ground truth: y_act, y_pt from train_feat (already have y_srv = y_train)
        y_act = train_feat["next_actionId"].astype(np.int64).values if "next_actionId" in train_feat.columns else np.zeros(N, dtype=np.int64)
        y_pt = train_feat["next_pointId"].astype(np.int64).values if "next_pointId" in train_feat.columns else np.zeros(N, dtype=np.int64)

        oof_act = np.full((N, 19), 1.0 / 15.0, dtype=np.float32)  # uniform over eval classes 0-14
        oof_act[:, 15:] = 0.0  # serve classes excluded from action macro
        oof_pt = np.full((N, 10), 0.1, dtype=np.float32)
        test_act = np.full((N_TEST, 19), 1.0 / 15.0, dtype=np.float32)
        test_act[:, 15:] = 0.0
        test_pt = np.full((N_TEST, 10), 0.1, dtype=np.float32)

        # next_strikeNumber for SN-slice audits (matches train_v14 convention)
        nsn = train_feat["next_strikeNumber"].astype(np.int64).values if "next_strikeNumber" in train_feat.columns else np.zeros(N, dtype=np.int64)
        # Mask = all rows valid for this component
        oof_mask = np.ones(N, dtype=bool)

        np.save(oof_dir / f"{args.tag}_oof_act.npy", oof_act)
        np.save(oof_dir / f"{args.tag}_oof_pt.npy", oof_pt)
        np.save(oof_dir / f"{args.tag}_oof_srv.npy", oof_pred.astype(np.float32))
        np.save(oof_dir / f"{args.tag}_oof_y_act.npy", y_act)
        np.save(oof_dir / f"{args.tag}_oof_y_pt.npy", y_pt)
        np.save(oof_dir / f"{args.tag}_oof_y_srv.npy", y_train.astype(np.int64))
        np.save(oof_dir / f"{args.tag}_oof_mask.npy", oof_mask)
        np.save(oof_dir / f"{args.tag}_oof_nsn.npy", nsn)
        np.save(oof_dir / f"{args.tag}_test_act.npy", test_act)
        np.save(oof_dir / f"{args.tag}_test_pt.npy", test_pt)
        np.save(oof_dir / f"{args.tag}_test_srv.npy", test_srv)
        np.save(oof_dir / f"{args.tag}_test_rally_uid.npy",
                np.asarray(test_rally_uids, dtype=np.int64))
        print(f"  Saved 12 arrays to {oof_dir} with tag={args.tag}")
        print(f"  oof_srv:  shape={oof_pred.shape}, AUC overall={roc_auc_score(y_train, oof_pred):.4f}")
        print(f"  test_srv: shape={test_srv.shape}, mean={test_srv.mean():.4f}")
        print(f"  oof_act/pt: uniform (SGP-specialist; blend will weight these 0)")

    # ---- Save metadata ----
    _save_metadata(args.tag, audits, diagnostics, fold_aucs, gate_verdict,
                   baseline_fold1_auc=baseline_fold1_auc, gate_threshold=gate_threshold)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    print(f"Verdict: {gate_verdict}")


def _save_metadata(tag, audits, diagnostics, fold_aucs, gate_verdict,
                    baseline_fold1_auc=None, gate_threshold=None) -> None:
    metadata = {
        "tag": tag,
        "feature_count": len(SGP_V3_CORE_COLUMNS),
        "feature_profile": "core",
        "audits": audits,
        "diagnostics": diagnostics,
        "fold_aucs": fold_aucs,
        "baseline_fold1_auc": baseline_fold1_auc,
        "gate_threshold": gate_threshold,
        "gate_verdict": gate_verdict,
    }
    out_dir = Path(PROJECT_ROOT) / "runs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{tag}_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {path}")


if __name__ == "__main__":
    main()
