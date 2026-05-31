"""
Main training script.
Usage:
    uv run python -m src.train [options]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineering import (
    compute_player_profiles,
    merge_player_profiles,
    prepare_train_test,
)
from src.models.autogluon_model import AutoGluonModel

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Data paths
    train_path: str = str(ROOT / "data" / "raw" / "train.csv")
    test_path: str = str(ROOT / "data" / "raw" / "test.csv")
    # AutoGluon
    presets: str = "medium_quality"
    num_bag_folds: int = 5
    num_bag_sets: int = 1
    num_stack_levels: int = 0
    time_limit: int = 3600
    verbosity: int = 2
    excluded_model_types: list[str] = field(default_factory=lambda: ["NN_TORCH", "FASTAI"])
    refit_full: bool = False
    set_best_to_refit_full: bool = False
    calibrate: bool = False
    auto_stack: bool = False
    holdout_frac: float = 0.2
    use_bag_holdout: bool = False
    keep_only_best: bool = False
    num_cpus: int = -1   # -1 = auto
    num_gpus: int = 0
    # Reproducibility
    seed: int = 0
    # Evaluation metrics per target
    action_metric: str = "f1_macro"   # f1_macro, accuracy, balanced_accuracy
    point_metric: str = "f1_macro"
    server_metric: str = "roc_auc"    # roc_auc, accuracy, f1
    # Ensemble
    fit_weighted_ensemble: bool = True
    # Per-model early stopping
    early_stopping_rounds: int = 50
    # Validation
    # When True, validation uses a rally-disjoint GroupShuffleSplit holdout
    # (avoids leakage from random bagging folds). Disables bagging/stacking.
    group_aware: bool = True
    # Inverse-frequency class weights. Rebalances macro F1 for imbalanced
    # targets (actionId has 15k:372 ratio between common/rare classes).
    class_weight: bool = False


def snapshot_src(dest: Path, src_root: Path | None = None) -> None:
    """Copy ``src/`` into ``dest/src_snapshot/`` (excluding __pycache__).

    Makes the version folder self-contained: future prediction can run
    against this frozen source tree via ``PYTHONPATH=<dest>/src_snapshot``
    without any git checkout — bugfixes and refactors on main stay isolated
    from the version this model was trained with.

    Silently skipped if the source tree does not exist (e.g. in pytest
    harnesses that monkeypatch ROOT to a temp dir). Production runs always
    have ``src/`` under ROOT.
    """
    src_root = src_root or (ROOT / "src")
    if not src_root.exists():
        logger.warning("snapshot_src: %s does not exist, skipping", src_root)
        return
    snap = dest / "src_snapshot"
    if snap.exists():
        shutil.rmtree(snap)
    shutil.copytree(
        src_root, snap,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _git_head_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def write_version_metadata(
    dest: Path,
    cfg: TrainConfig,
    run_name: str,
    n_seeds: int,
    cli_argv: list[str] | None = None,
) -> None:
    """Write config.json, commit.txt, and REPRODUCE.md to ``dest``."""
    config = {
        "run_name": run_name,
        "n_seeds": n_seeds,
        "cli_argv": cli_argv,
        "train_config": dataclasses.asdict(cfg),
    }
    (dest / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (dest / "commit.txt").write_text(_git_head_hash() + "\n")

    name = dest.name
    reproduce_md = f"""# Reproducing {name}

This folder is self-contained: model, player profile, submission, source snapshot.

## Direct submission
`submission.csv` is the ensemble output over `data/raw/test.csv`, ready for the
leaderboard (same columns as `data/raw/sample_submission.csv`).

## Re-predict with the frozen source
```bash
PYTHONPATH={dest}/src_snapshot \\
  python -m src.predict \\
    --model-path {dest} \\
    --test-path <path-to-your-test.csv> \\
    --run-name {name}_repredict
```

## Retrain from scratch
See `config.json` for the exact CLI args. The frozen source at `src_snapshot/`
matches git commit `commit.txt`.
"""
    (dest / "REPRODUCE.md").write_text(reproduce_md)


def train_autogluon(
    cfg: TrainConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    test_rally_uids: pd.Series,
    run_name: str,
    groups: pd.Series | None = None,
    player_profiles: pd.DataFrame | None = None,
    model_dir: Path | None = None,
    n_seeds: int = 1,
    cli_argv: list[str] | None = None,
) -> dict:
    """Train AutoGluon models and produce submission.csv.

    For ``n_seeds > 1`` we train ``n_seeds`` independent AutoGluonModel
    instances (seeds ``cfg.seed``, ``cfg.seed+1``, ...), save each under
    ``{model_dir}/seed{i}/``, and ensemble their test predictions by
    averaging ``predict_proba``. The validation scores returned are also
    averaged across seeds, matching the CV script's ensemble behavior.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")

    save_dir = model_dir or (ROOT / "models" / "autogluon")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Freeze src/ + metadata so this version folder is self-contained and
    # can be re-predicted later even after main branch has drifted.
    snapshot_src(save_dir)
    write_version_metadata(save_dir, cfg, run_name, n_seeds, cli_argv)

    if player_profiles is not None:
        X_train = merge_player_profiles(X_train, player_profiles)
        X_test = merge_player_profiles(X_test, player_profiles)
        profile_path = save_dir / "player_profiles.csv"
        player_profiles.to_csv(profile_path)
        logger.info("Player profiles saved: %s (%d players)", profile_path, len(player_profiles))
        print(f"Player profiles saved: {profile_path} ({len(player_profiles)} players)")

    # Train one or more seeds. The first seed owns the ancillary outputs
    # (leaderboard, feature importances). All seeds contribute to the
    # averaged test predictions.
    all_probs: list[dict[str, np.ndarray]] = []
    all_scores: list[dict] = []
    primary_model: AutoGluonModel | None = None
    for seed_i in range(n_seeds):
        seed_cfg = dataclasses.replace(cfg, seed=cfg.seed + seed_i)
        seed_dir = save_dir / f"seed{seed_i}" if n_seeds > 1 else save_dir
        if n_seeds > 1:
            logger.info("--- Seed %d/%d (seed_value=%d, dir=%s) ---",
                        seed_i + 1, n_seeds, seed_cfg.seed, seed_dir)
            print(f"\n--- Seed {seed_i + 1}/{n_seeds} ---")
        model = AutoGluonModel(seed_cfg, model_dir=seed_dir)
        scores = model.fit(X_train, y_train, groups=groups)
        all_scores.append(scores)
        all_probs.append(model.predict_proba(X_test))
        if primary_model is None:
            primary_model = model

    # Average test probabilities across seeds → ensemble prediction
    ensemble_probs = {
        target: np.mean([p[target] for p in all_probs], axis=0)
        for target in all_probs[0]
    }
    submission = pd.DataFrame(
        {
            "rally_uid": test_rally_uids.values,
            "actionId": ensemble_probs["actionId"].argmax(axis=1),
            "pointId": ensemble_probs["pointId"].argmax(axis=1),
            "serverGetPoint": (ensemble_probs["serverGetPoint"] > 0.5).astype(int),
        }
    )
    # Averaged validation scores give a fair ensemble readout.
    scores = {
        k: float(np.mean([s[k] for s in all_scores]))
        for k in all_scores[0]
    }

    confidence = primary_model.get_confidence(ensemble_probs)
    _save_confidence(confidence, test_rally_uids, run_name)
    _save_leaderboard(
        primary_model.get_leaderboards(),
        primary_model.get_feature_importances(),
        run_name,
    )
    sub_path = _save_submission(submission, run_name)
    _append_score(scores, run_name)

    # Write a canonical copy inside the model dir so every versioned folder
    # is self-contained (model + profile + submission.csv).
    versioned_sub = save_dir / "submission.csv"
    submission.to_csv(versioned_sub, index=False)

    logger.info("Submission saved: %s (%d rows, ensemble of %d seeds)",
                sub_path, len(submission), n_seeds)
    logger.info("Submission also at: %s", versioned_sub)
    logger.info("Scores (ensemble mean): %s", scores)
    print(f"Submission saved: {sub_path} ({len(submission)} rows, "
          f"ensemble of {n_seeds} seed{'s' if n_seeds > 1 else ''})")
    print(f"Submission also at: {versioned_sub}")
    print(f"Scores (ensemble mean): {scores}")
    return scores


def _save_submission(df: pd.DataFrame, run_name: str) -> str:
    path = ROOT / "data" / "submissions"
    path.mkdir(parents=True, exist_ok=True)
    filename = path / f"submission_autogluon_{run_name}.csv"
    df.to_csv(filename, index=False)
    return str(filename)


def _save_confidence(confidence: dict, rally_uids: pd.Series, run_name: str):
    path = ROOT / "data" / "submissions"
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"rally_uid": rally_uids.values, **confidence})
    df.to_csv(path / f"confidence_{run_name}.csv", index=False)


def _save_leaderboard(leaderboards: dict, feature_importances: dict, run_name: str):
    path = ROOT / "data" / "submissions"
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"leaderboard_{run_name}.json", "w") as f:
        json.dump({
            "run_name": run_name,
            "leaderboards": leaderboards,
            "feature_importances": feature_importances,
        }, f, indent=2)


def _append_score(scores: dict, run_name: str):
    scores_path = ROOT / "data" / "submissions" / "scores.json"
    existing = []
    if scores_path.exists():
        with open(scores_path) as f:
            existing = json.load(f)
    existing.append({"model": "autogluon", "run_name": run_name, **scores})
    with open(scores_path, "w") as f:
        json.dump(existing, f, indent=2)


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default=defaults.train_path)
    parser.add_argument("--test-path", default=defaults.test_path)
    parser.add_argument("--presets", default=defaults.presets,
                        choices=["medium_quality", "good_quality", "high_quality",
                                 "best_quality", "extreme_quality",
                                 "extreme_quality_v140"])
    parser.add_argument("--num-bag-folds", type=int, default=defaults.num_bag_folds)
    parser.add_argument("--num-bag-sets", type=int, default=defaults.num_bag_sets)
    parser.add_argument("--num-stack-levels", type=int, default=defaults.num_stack_levels)
    parser.add_argument("--time-limit", type=int, default=defaults.time_limit)
    parser.add_argument("--verbosity", type=int, default=defaults.verbosity, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--excluded-model-types", nargs="*", default=defaults.excluded_model_types)
    parser.add_argument("--refit-full", action="store_true", default=defaults.refit_full)
    parser.add_argument("--set-best-to-refit-full", action="store_true", default=defaults.set_best_to_refit_full)
    parser.add_argument("--calibrate", action="store_true", default=defaults.calibrate)
    parser.add_argument("--auto-stack", action="store_true", default=defaults.auto_stack)
    parser.add_argument("--holdout-frac", type=float, default=defaults.holdout_frac)
    parser.add_argument("--use-bag-holdout", action="store_true", default=defaults.use_bag_holdout)
    parser.add_argument("--keep-only-best", action="store_true", default=defaults.keep_only_best)
    parser.add_argument("--num-cpus", type=int, default=defaults.num_cpus)
    parser.add_argument("--num-gpus", type=int, default=defaults.num_gpus)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--action-metric", default=defaults.action_metric,
                        choices=["f1_macro", "accuracy", "balanced_accuracy", "log_loss"])
    parser.add_argument("--point-metric", default=defaults.point_metric,
                        choices=["f1_macro", "accuracy", "balanced_accuracy", "log_loss"])
    parser.add_argument("--server-metric", default=defaults.server_metric,
                        choices=["roc_auc", "accuracy", "f1"])
    parser.add_argument("--no-weighted-ensemble", action="store_false", dest="fit_weighted_ensemble",
                        default=defaults.fit_weighted_ensemble)
    parser.add_argument("--early-stopping-rounds", type=int, default=defaults.early_stopping_rounds)
    parser.add_argument("--no-group-aware", action="store_false", dest="group_aware",
                        default=defaults.group_aware,
                        help="Disable rally-disjoint validation; use AutoGluon random bagging folds.")
    parser.add_argument("--player-profile", action="store_true",
                        help="Add cross-rally player profile features (win rate, "
                             "action/point distributions). Profiles are saved alongside "
                             "the model for later prediction.")
    parser.add_argument("--model-dir", default=None,
                        help="Directory to save models. Default: models/autogluon. "
                             "Use e.g. models/v2 to save a named version.")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Train an ensemble of N seeds (saved under "
                             "{model-dir}/seed{i}/). Averaged predict_proba "
                             "becomes submission.csv. Default: 1.")
    parser.add_argument("--class-weight", action="store_true",
                        help="Use inverse-frequency sample_weight to rebalance "
                             "macro F1 on imbalanced targets (actionId / pointId).")
    args = parser.parse_args()

    cfg = TrainConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        presets=args.presets,
        num_bag_folds=args.num_bag_folds,
        num_bag_sets=args.num_bag_sets,
        num_stack_levels=args.num_stack_levels,
        time_limit=args.time_limit,
        verbosity=args.verbosity,
        excluded_model_types=args.excluded_model_types or [],
        refit_full=args.refit_full,
        set_best_to_refit_full=args.set_best_to_refit_full,
        calibrate=args.calibrate,
        auto_stack=args.auto_stack,
        holdout_frac=args.holdout_frac,
        use_bag_holdout=args.use_bag_holdout,
        keep_only_best=args.keep_only_best,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        seed=args.seed,
        action_metric=args.action_metric,
        point_metric=args.point_metric,
        server_metric=args.server_metric,
        fit_weighted_ensemble=args.fit_weighted_ensemble,
        early_stopping_rounds=args.early_stopping_rounds,
        group_aware=args.group_aware,
        class_weight=args.class_weight,
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    X_train, y_train, groups, X_test, test_rally_uids = prepare_train_test(
        cfg.train_path, cfg.test_path
    )

    profiles = None
    if args.player_profile:
        raw_train = pd.read_csv(cfg.train_path)
        profiles = compute_player_profiles(raw_train)
        logger.info("Computed player profiles for %d players", len(profiles))

    model_dir = Path(args.model_dir) if args.model_dir else None

    import sys
    train_autogluon(
        cfg, X_train, y_train, X_test, test_rally_uids, run_name,
        groups=groups, player_profiles=profiles, model_dir=model_dir,
        n_seeds=args.n_seeds,
        cli_argv=sys.argv[1:],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
