"""
K-fold cross-validation for AutoGluon models using rally-disjoint
GroupKFold splits.

Each fold trains a fresh TabularPredictor per target on the training rallies
and scores it on the held-out rallies. We do NOT use AutoGluon's bagging here
— the whole point of CV is to get an honest estimate of generalization with a
proper rally-level split. Bagging on top would either re-introduce leakage or
multiply training time without benefit.

Usage:
    uv run python -m src.cv [--n-splits 5] [--time-limit 60] [--presets ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import GroupKFold

from src.evaluate.metrics import evaluate_predictions, optimize_multiclass_thresholds
from src.features.engineering import (
    compute_player_profiles,
    compute_transition_tables,
    merge_player_profiles,
    merge_transition_features,
    prepare_train_test,
)
from src.models.autogluon_model import _proba_df_to_array
from src.train import TrainConfig, snapshot_src, write_version_metadata

ROOT = Path(__file__).resolve().parent.parent
CV_MODEL_DIR = ROOT / "models" / "cv"

logger = logging.getLogger(__name__)


# AutoGluon hyperparameter presets tuned for small-data regimes (~1000 samples).
# Forces shallower trees, larger min-leaf samples, and stronger regularization
# than AutoGluon's defaults — fights overfitting at the cost of model capacity.
SMALL_DATA_HYPERPARAMETERS = {
    "GBM": {
        "num_leaves": 7,
        "max_depth": 4,
        "min_data_in_leaf": 20,
        "learning_rate": 0.03,
        "num_boost_round": 500,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
    },
    "CAT": {
        "depth": 4,
        "l2_leaf_reg": 5.0,
        "iterations": 500,
        "learning_rate": 0.03,
    },
    "XGB": {
        "max_depth": 4,
        "min_child_weight": 5,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "learning_rate": 0.03,
        "n_estimators": 500,
    },
}

# Deterministic hyperparameters: explicit per-model configs so training
# outcome doesn't depend on wall-clock budget (time_limit) or AG's
# internal hyperparameter search ordering. Set large *_round/iterations
# caps; the per-model early_stopping_rounds (default 50, see
# ag_args_fit) decides actual termination based on validation metric.
#
# These mirror the moderate-capacity settings AG would normally land
# on in medium_quality on this dataset, but with hard-coded values
# so the same config trains identically on any machine.
DETERMINISTIC_HYPERPARAMETERS = {
    "GBM": {
        "num_boost_round": 2000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 20,
    },
    # CAT deliberately omitted: on 19-class actionId CPU it takes 40+
    # minutes per (fold, seed) at iterations=500, which makes the full
    # 5×5 grid impractical. v6's medium_quality at time_limit=300s
    # included CAT but rarely got past ~50 iterations, so the
    # contribution to the final weighted ensemble was small. Trade -0.005
    # OOF for a deterministic and fast pipeline.
    "XGB": {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
    },
}


def _make_balanced_sample_weights(y_target: pd.Series) -> np.ndarray:
    """Inverse-frequency class weights so each class contributes equally
    regardless of base rate. Useful for macro F1."""
    counts = y_target.value_counts()
    n_classes = len(counts)
    total = len(y_target)
    class_weight = {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}
    return y_target.map(class_weight).to_numpy()


def cross_validate(
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,
    cfg: TrainConfig,
    n_splits: int = 5,
    n_seeds: int = 1,
    regularized: bool = False,
    deterministic: bool = False,
    class_weight: bool = False,
    X_test: pd.DataFrame | None = None,
    player_profile: bool = False,
    transition: bool = False,
    raw_train_df: pd.DataFrame | None = None,
    produce_submission: bool = False,
    threshold_optimize: bool = False,
) -> dict:
    """
    Run k-fold GroupKFold cross-validation. Returns a dict with per-fold
    scores and aggregate statistics.

    Args:
        X: feature DataFrame (shot-level)
        y: target DataFrame with actionId, pointId, serverGetPoint
        groups: rally_uid Series aligned with X (used for GroupKFold)
        cfg: training configuration (presets, time_limit, etc.)
        n_splits: number of folds (k)
        n_seeds: number of seeds to ensemble per fold. When > 1, each fold
            trains `n_seeds` independent predictors per target (using
            seeds cfg.seed, cfg.seed+1, ...) and averages their
            predict_proba before argmax. Reduces fold-level variance; cost
            is roughly n_seeds × training time.
        X_test: test features for submission generation (when produce_submission).
        player_profile: if True, compute fold-internal player profiles and merge.
        produce_submission: if True, predict X_test on every (fold, seed) and
            return averaged proba in ``submission_probs``.

    Returns:
        {
            "n_splits": int,
            "n_seeds": int,
            "folds": [{"fold": int, "actionId_macro_f1": float, ...}],
            "aggregate": {"actionId_macro_f1": {"mean", "std", "min", "max"}, ...}
        }
    """
    if groups is None or len(groups) != len(X):
        raise ValueError("groups must be a Series aligned with X")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")

    target_metrics = {
        "actionId": cfg.action_metric,
        "pointId": cfg.point_metric,
        "serverGetPoint": cfg.server_metric,
    }

    splitter = GroupKFold(n_splits=n_splits)
    folds = list(splitter.split(X, y, groups=groups))

    fold_scores: list[dict] = []
    # Submission accumulator: for each target, list of per-fold test probas
    # (already averaged across seeds within the fold). Final submission is the
    # mean across folds → an ensemble of (n_splits × n_seeds) predictors.
    do_submission = produce_submission and X_test is not None
    submission_acc: dict[str, list[np.ndarray]] = {
        "actionId": [], "pointId": [], "serverGetPoint": [],
    }
    # OOF accumulators for threshold optimization
    oof_action_proba: list[np.ndarray] = []
    oof_point_proba: list[np.ndarray] = []
    oof_action_true: list[np.ndarray] = []
    oof_point_true: list[np.ndarray] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        X_tr = X.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)
        n_train_rallies = int(groups.iloc[train_idx].nunique())
        n_val_rallies = int(groups.iloc[val_idx].nunique())

        # Player profiles: compute from train fold's raw data, merge into both
        # (and also the submission X_test, if producing one).
        X_test_fold = X_test  # alias to the version with this fold's profiles
        train_rally_uids = set(groups.iloc[train_idx].unique())
        raw_train_fold = None
        if raw_train_df is not None:
            raw_train_fold = raw_train_df[raw_train_df["rally_uid"].isin(train_rally_uids)]

        if player_profile and raw_train_fold is not None:
            profiles = compute_player_profiles(raw_train_fold)
            X_tr = merge_player_profiles(X_tr, profiles)
            X_val = merge_player_profiles(X_val, profiles)
            if X_test_fold is not None:
                X_test_fold = merge_player_profiles(X_test_fold, profiles)

        # Transition features: compute from train fold's raw data
        if transition and raw_train_fold is not None:
            trans_tables = compute_transition_tables(raw_train_fold)
            X_tr = merge_transition_features(X_tr, trans_tables)
            X_val = merge_transition_features(X_val, trans_tables)
            if X_test_fold is not None:
                X_test_fold = merge_transition_features(X_test_fold, trans_tables)

        logger.info(
            "===== Fold %d/%d: %d/%d shots, %d/%d rallies, %d seeds =====",
            fold_idx, n_splits,
            len(train_idx), len(val_idx),
            n_train_rallies, n_val_rallies, n_seeds,
        )
        print(f"\n===== Fold {fold_idx}/{n_splits} =====")

        eval_action_pred = None
        eval_point_pred = None
        eval_server_prob = None

        # Optionally constrain AutoGluon to a small-data hyperparameters
        # preset (regularized GBM/CAT/XGB only), or use deterministic
        # explicit hyperparameters (large num_boost_round + early stop).
        if deterministic:
            hyperparameters = DETERMINISTIC_HYPERPARAMETERS
        elif regularized:
            hyperparameters = SMALL_DATA_HYPERPARAMETERS
        else:
            hyperparameters = None

        for target, metric in target_metrics.items():
            logger.info("  Training %s (%d seeds)...", target, n_seeds)
            train_data = X_tr.copy()
            train_data[target] = y_tr[target].values

            if class_weight:
                train_data["sample_weight"] = _make_balanced_sample_weights(
                    pd.Series(y_tr[target].values)
                )

            seed_probas: list[np.ndarray] = []
            submission_seed_probas: list[np.ndarray] = []
            for seed_idx in range(n_seeds):
                seed_value = cfg.seed + seed_idx
                target_dir = CV_MODEL_DIR / f"fold{fold_idx}" / f"seed{seed_idx}" / target
                shutil.rmtree(target_dir, ignore_errors=True)

                predictor = TabularPredictor(
                    label=target,
                    eval_metric=metric,
                    path=str(target_dir),
                    verbosity=cfg.verbosity,
                    sample_weight="sample_weight" if class_weight else None,
                    weight_evaluation=False,
                )
                fit_kwargs = dict(
                    num_bag_folds=cfg.num_bag_folds,
                    num_stack_levels=cfg.num_stack_levels,
                    excluded_model_types=cfg.excluded_model_types or None,
                    num_cpus=cfg.num_cpus if cfg.num_cpus != -1 else "auto",
                    num_gpus=cfg.num_gpus,
                    fit_weighted_ensemble=cfg.fit_weighted_ensemble,
                    ag_args_fit={
                        "early_stop": cfg.early_stopping_rounds,
                        "seed_value": seed_value,
                    },
                )
                if deterministic:
                    # No preset (avoids AG's internal hyperparam search).
                    # Large time cap; actual termination is driven by
                    # per-model early stopping on validation metric.
                    fit_kwargs["time_limit"] = 99999
                else:
                    fit_kwargs["presets"] = cfg.presets
                    fit_kwargs["time_limit"] = cfg.time_limit
                if hyperparameters is not None:
                    fit_kwargs["hyperparameters"] = hyperparameters
                predictor.fit(train_data, **fit_kwargs)

                proba_df = predictor.predict_proba(X_val)
                if target == "serverGetPoint":
                    seed_probas.append(proba_df[1].values)
                elif target == "actionId":
                    seed_probas.append(_proba_df_to_array(proba_df, 19))
                else:  # pointId
                    seed_probas.append(_proba_df_to_array(proba_df, 10))

                if do_submission and X_test_fold is not None:
                    sub_proba = predictor.predict_proba(X_test_fold)
                    if target == "serverGetPoint":
                        submission_seed_probas.append(sub_proba[1].values)
                    elif target == "actionId":
                        submission_seed_probas.append(_proba_df_to_array(sub_proba, 19))
                    else:
                        submission_seed_probas.append(_proba_df_to_array(sub_proba, 10))

            mean_proba = np.mean(seed_probas, axis=0)
            if target == "serverGetPoint":
                eval_server_prob = mean_proba
            elif target == "actionId":
                eval_action_pred = mean_proba
            else:
                eval_point_pred = mean_proba

            if do_submission and submission_seed_probas:
                submission_acc[target].append(
                    np.mean(submission_seed_probas, axis=0),
                )

        action_pred_labels = eval_action_pred.argmax(axis=1)
        point_pred_labels = eval_point_pred.argmax(axis=1)

        # Collect OOF predictions for threshold optimization
        oof_action_proba.append(eval_action_pred)
        oof_point_proba.append(eval_point_pred)
        oof_action_true.append(y_val["actionId"].values)
        oof_point_true.append(y_val["pointId"].values)

        scores = evaluate_predictions(
            action_true=y_val["actionId"].values,
            action_pred=action_pred_labels,
            point_true=y_val["pointId"].values,
            point_pred=point_pred_labels,
            server_true=y_val["serverGetPoint"].values,
            server_prob=eval_server_prob,
        )
        scores["fold"] = fold_idx
        scores["n_train_rallies"] = n_train_rallies
        scores["n_val_rallies"] = n_val_rallies
        fold_scores.append(scores)
        logger.info("  Fold %d total_score=%.4f", fold_idx, scores["total_score"])
        print(f"  Fold {fold_idx} scores: {scores}")

    metric_keys = [
        "actionId_macro_f1",
        "pointId_macro_f1",
        "serverGetPoint_auc",
        "total_score",
    ]
    aggregate = {}
    for k in metric_keys:
        vals = [s[k] for s in fold_scores]
        aggregate[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    # Cross-fold ensemble: mean each target's per-fold test probs across all
    # folds → final submission probabilities (n_splits × n_seeds models averaged).
    submission_probs: dict[str, np.ndarray] | None = None
    if do_submission and all(submission_acc[t] for t in submission_acc):
        submission_probs = {
            t: np.mean(submission_acc[t], axis=0)
            for t in submission_acc
        }

    # --- Threshold optimization on OOF predictions ---
    optimized_scales: dict[str, np.ndarray] | None = None
    if threshold_optimize and oof_action_proba:
        all_action_proba = np.concatenate(oof_action_proba, axis=0)
        all_point_proba = np.concatenate(oof_point_proba, axis=0)
        all_action_true = np.concatenate(oof_action_true)
        all_point_true = np.concatenate(oof_point_true)

        action_scales = optimize_multiclass_thresholds(
            all_action_true, all_action_proba, 19,
        )
        point_scales = optimize_multiclass_thresholds(
            all_point_true, all_point_proba, 10,
        )

        # Report improvement on OOF
        baseline_action_f1 = float(np.mean([
            s["actionId_macro_f1"] for s in fold_scores
        ]))
        baseline_point_f1 = float(np.mean([
            s["pointId_macro_f1"] for s in fold_scores
        ]))
        from src.evaluate.metrics import macro_f1
        opt_action_f1 = macro_f1(
            all_action_true,
            (all_action_proba * action_scales).argmax(axis=1),
        )
        opt_point_f1 = macro_f1(
            all_point_true,
            (all_point_proba * point_scales).argmax(axis=1),
        )
        print("\n=== Threshold Optimization (OOF) ===")
        print(f"  actionId macro F1: {baseline_action_f1:.4f} → {opt_action_f1:.4f} "
              f"(Δ={opt_action_f1 - baseline_action_f1:+.4f})")
        print(f"  pointId  macro F1: {baseline_point_f1:.4f} → {opt_point_f1:.4f} "
              f"(Δ={opt_point_f1 - baseline_point_f1:+.4f})")
        print(f"  action_scales: {action_scales.tolist()}")
        print(f"  point_scales:  {point_scales.tolist()}")

        optimized_scales = {
            "actionId": action_scales,
            "pointId": point_scales,
        }

    return {
        "n_splits": n_splits,
        "n_seeds": n_seeds,
        "folds": fold_scores,
        "aggregate": aggregate,
        "submission_probs": submission_probs,
        "optimized_scales": optimized_scales,
    }


def _write_versioned_submission(
    save_dir: Path,
    submission_probs: dict[str, np.ndarray],
    test_rally_uids: pd.Series,
    cfg: TrainConfig,
    run_name: str,
    args: argparse.Namespace,
    raw_train_df: pd.DataFrame | None,
    optimized_scales: dict[str, np.ndarray] | None = None,
) -> None:
    """Materialize the CV ensemble as a self-contained versioned folder."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Apply per-class threshold scales if available
    action_proba = submission_probs["actionId"]
    point_proba = submission_probs["pointId"]
    if optimized_scales is not None:
        action_proba = action_proba * optimized_scales["actionId"]
        point_proba = point_proba * optimized_scales["pointId"]
        # Save scales for reproducibility
        scales_data = {
            "actionId": optimized_scales["actionId"].tolist(),
            "pointId": optimized_scales["pointId"].tolist(),
        }
        (save_dir / "optimized_scales.json").write_text(
            json.dumps(scales_data, indent=2)
        )
        print("  Threshold scales applied and saved to optimized_scales.json")

    submission = pd.DataFrame({
        "rally_uid": test_rally_uids.values,
        "actionId": action_proba.argmax(axis=1),
        "pointId": point_proba.argmax(axis=1),
        "serverGetPoint": (submission_probs["serverGetPoint"] > 0.5).astype(int),
    })
    sub_path = save_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)
    logger.info("CV submission saved: %s (%d rows, %d-fold × %d-seed ensemble)",
                sub_path, len(submission), args.n_splits, args.n_seeds)
    print(f"\nCV submission saved: {sub_path} ({len(submission)} rows, "
          f"{args.n_splits}-fold × {args.n_seeds}-seed ensemble)")

    # Also write a timestamped copy into data/submissions/ for historical record.
    hist_path = ROOT / "data" / "submissions" / f"submission_cv_{run_name}.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(hist_path, index=False)

    # Save full-data player profiles for later re-prediction
    # (during CV each fold uses its own profiles to avoid leakage; here we
    # snapshot the profiles computed from *all* training data, matching what
    # predict.py expects at inference time).
    if raw_train_df is not None:
        profiles = compute_player_profiles(raw_train_df)
        profiles.to_csv(save_dir / "player_profiles.csv")

    # Source snapshot + metadata (config.json, commit.txt, REPRODUCE.md)
    snapshot_src(save_dir)
    import sys
    write_version_metadata(save_dir, cfg, run_name, args.n_seeds, sys.argv[1:])


def _save_cv_results(results: dict, run_name: str) -> Path:
    out_dir = ROOT / "data" / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cv_{run_name}.json"
    # Strip non-JSON-serializable fields (numpy arrays)
    skip_keys = {"submission_probs", "optimized_scales"}
    json_ready = {}
    for k, v in results.items():
        if k in skip_keys:
            continue
        json_ready[k] = v
    # Serialize optimized_scales if present
    if results.get("optimized_scales") is not None:
        json_ready["optimized_scales"] = {
            t: s.tolist() for t, s in results["optimized_scales"].items()
        }
    with open(path, "w") as f:
        json.dump(json_ready, f, indent=2)
    return path


def main():  # pragma: no cover
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="K-fold rally-disjoint cross-validation for AutoGluon models.",
    )
    parser.add_argument("--train-path", default=defaults.train_path)
    parser.add_argument("--test-path", default=defaults.test_path,
                        help="Unused for CV; kept for symmetry with src.train.")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of GroupKFold folds (default: 5).")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Seeds to ensemble per fold; averages predict_proba. "
                             "Roughly n_seeds × training time. Default: 1.")
    parser.add_argument("--num-bag-folds", type=int, default=0,
                        help="AutoGluon bagging folds within each CV fold's "
                             "training set. Safe because outer CV is rally-disjoint. "
                             "0 = off (default), 5-8 recommended.")
    parser.add_argument("--num-stack-levels", type=int, default=0,
                        help="AutoGluon stacking levels (0 = off). Needs more "
                             "time_limit to train the meta-layer.")
    parser.add_argument("--regularized", action="store_true",
                        help="Use SMALL_DATA_HYPERPARAMETERS preset (shallow trees, "
                             "high min-leaf samples, L1/L2 regularization).")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use DETERMINISTIC_HYPERPARAMETERS with large num_boost_round "
                             "caps + early-stopping. Removes wall-clock time_limit dependency "
                             "for reproducibility across machines. Disables --presets / --time-limit.")
    parser.add_argument("--class-weight", action="store_true",
                        help="Pass inverse-frequency class weights as sample_weight "
                             "(rebalances macro F1 even when classes are near-uniform).")
    parser.add_argument("--player-profile", action="store_true",
                        help="Add cross-rally player profile features (win rate, "
                             "action/point distributions) computed from train fold.")
    parser.add_argument("--transition", action="store_true",
                        help="Add transition matrix features (empirical conditional "
                             "distributions of next shot given last shot context).")
    parser.add_argument("--threshold-optimize", action="store_true",
                        help="Optimize per-class probability scales on OOF predictions "
                             "to maximize macro F1 before argmax. Free lunch.")
    parser.add_argument("--presets", default=defaults.presets,
                        choices=["medium_quality", "good_quality", "high_quality",
                                 "best_quality", "extreme_quality",
                                 "extreme_quality_v140"])
    parser.add_argument("--time-limit", type=int, default=120,
                        help="Per-target, per-fold time limit (total ≈ 3 × n_splits × this).")
    parser.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--excluded-model-types", nargs="*",
                        default=defaults.excluded_model_types)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--action-metric", default=defaults.action_metric,
                        choices=["f1_macro", "accuracy", "balanced_accuracy", "log_loss"])
    parser.add_argument("--point-metric", default=defaults.point_metric,
                        choices=["f1_macro", "accuracy", "balanced_accuracy", "log_loss"])
    parser.add_argument("--server-metric", default=defaults.server_metric,
                        choices=["roc_auc", "accuracy", "f1"])
    parser.add_argument("--num-cpus", type=int, default=defaults.num_cpus)
    parser.add_argument("--num-gpus", type=int, default=defaults.num_gpus)
    parser.add_argument("--early-stopping-rounds", type=int,
                        default=defaults.early_stopping_rounds)
    parser.add_argument("--model-dir", default=None,
                        help="If set, also produce a submission.csv via "
                             "(n_splits × n_seeds)-model ensemble on test.csv, "
                             "and write src_snapshot / config.json / commit.txt / "
                             "REPRODUCE.md into this folder. Use e.g. models/v4.")
    args = parser.parse_args()

    cfg = TrainConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        presets=args.presets,
        num_bag_folds=args.num_bag_folds,
        num_stack_levels=args.num_stack_levels,
        time_limit=args.time_limit,
        verbosity=args.verbosity,
        excluded_model_types=args.excluded_model_types or [],
        seed=args.seed,
        action_metric=args.action_metric,
        point_metric=args.point_metric,
        server_metric=args.server_metric,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    X_train, y_train, groups, X_test, test_rally_uids = prepare_train_test(
        cfg.train_path, cfg.test_path,
    )
    logger.info(
        "CV setup: n_splits=%d, n_seeds=%d, %d shots, %d rallies, presets=%s, time_limit=%ds/target",
        args.n_splits, args.n_seeds, len(X_train),
        groups.nunique(), cfg.presets, cfg.time_limit,
    )

    # Load raw train data for player profiles / transition features
    _raw_train_df = None
    if args.player_profile or args.transition:
        import pandas as _pd
        _raw_train_df = _pd.read_csv(cfg.train_path)

    produce_submission = args.model_dir is not None
    results = cross_validate(
        X_train, y_train, groups, cfg,
        n_splits=args.n_splits, n_seeds=args.n_seeds,
        regularized=args.regularized, deterministic=args.deterministic,
        class_weight=args.class_weight,
        X_test=X_test if produce_submission else None,
        player_profile=args.player_profile,
        transition=args.transition,
        raw_train_df=_raw_train_df,
        produce_submission=produce_submission,
        threshold_optimize=args.threshold_optimize,
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _save_cv_results(results, run_name)
    print(f"\nCV results saved: {path}")
    print(f"\n=== {args.n_splits}-fold × {args.n_seeds}-seed CV (mean ± std) ===")
    for k, v in results["aggregate"].items():
        print(f"  {k:25s} {v['mean']:.4f} ± {v['std']:.4f}  "
              f"(min={v['min']:.4f}, max={v['max']:.4f})")

    # --model-dir: persist submission + snapshot + metadata into a versioned folder.
    if produce_submission and results.get("submission_probs") is not None:
        _write_versioned_submission(
            Path(args.model_dir), results["submission_probs"],
            test_rally_uids, cfg, run_name, args, _raw_train_df,
            optimized_scales=results.get("optimized_scales"),
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
