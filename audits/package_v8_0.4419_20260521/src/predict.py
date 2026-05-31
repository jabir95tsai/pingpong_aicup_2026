"""
Prediction script using pre-saved AutoGluon models.
Loads models from a given path, runs feature engineering on test.csv,
and writes a submission CSV + confidence CSV.

Usage:
    uv run python -m src.predict [--model-path PATH] [--test-path PATH] [--run-name NAME]
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineering import (
    build_test_features,
    get_feature_cols,
    merge_player_profiles,
    validate_raw_dataframe,
)
from src.models.autogluon_model import AutoGluonModel
from src.train import TrainConfig, _save_confidence, _save_submission

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def _discover_seed_dirs(model_path: Path) -> list[Path]:
    """Return sorted list of seed{i}/ subdirs if this is a multi-seed
    ensemble layout; empty list if legacy single-dir layout."""
    seed_dirs = sorted(
        p for p in model_path.iterdir()
        if p.is_dir() and p.name.startswith("seed") and p.name[4:].isdigit()
    )
    return seed_dirs


def predict_from_model(model_path: str | Path, test_path: str, run_name: str) -> str:
    """Load saved models and generate predictions on test data.

    Supports two layouts:
    - Single-seed: ``{model_path}/{actionId,pointId,serverGetPoint}/``
    - Multi-seed:  ``{model_path}/seed{i}/{actionId,pointId,serverGetPoint}/``
      — auto-detected; test probabilities are averaged across seeds.
    """
    model_path = Path(model_path)
    seed_dirs = _discover_seed_dirs(model_path) if model_path.exists() else []

    if seed_dirs:
        logger.info("Detected multi-seed ensemble: %d seeds", len(seed_dirs))
        print(f"Detected multi-seed ensemble: {len(seed_dirs)} seeds")
        for sd in seed_dirs:
            for target in ["actionId", "pointId", "serverGetPoint"]:
                tp = sd / target
                if not tp.exists():
                    raise FileNotFoundError(
                        f"Model for '{target}' not found at {tp}. "
                        "Expected subdirectories: actionId/, pointId/, serverGetPoint/"
                    )
    else:
        for target in ["actionId", "pointId", "serverGetPoint"]:
            target_path = model_path / target
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Model for '{target}' not found at {target_path}. "
                    "Expected subdirectories: actionId/, pointId/, serverGetPoint/"
                )

    logger.info("Loading models from: %s", model_path)
    print(f"Loading models from: {model_path}")

    # Build test features
    test_raw = pd.read_csv(test_path)
    validate_raw_dataframe(test_raw, kind="test")
    logger.info("Test raw: %d shots, %d rallies", len(test_raw), test_raw["rally_uid"].nunique())
    print(f"Test raw: {len(test_raw)} shots, {test_raw['rally_uid'].nunique()} rallies")

    # We need feature columns — derive from test data
    test_feat = build_test_features(test_raw)
    feature_cols = get_feature_cols(test_feat)
    X_test = test_feat[feature_cols].reset_index(drop=True)
    test_rally_uids = test_feat["rally_uid"].reset_index(drop=True)
    logger.info("Test features: %d samples", len(X_test))
    print(f"Test features: {len(X_test)} samples")

    # Load player profiles if saved alongside model
    profile_path = model_path / "player_profiles.csv"
    if profile_path.exists():
        profiles = pd.read_csv(profile_path, index_col="player_id")
        X_test = merge_player_profiles(X_test, profiles)
        logger.info("Loaded player profiles: %d players, %d new features",
                     len(profiles), len(X_test.columns) - len(feature_cols))
        print(f"Loaded player profiles: {len(profiles)} players")

    # Load models — one AutoGluonModel per seed, then average predict_proba.
    cfg = TrainConfig()  # dummy config, only used to init AutoGluonModel
    if seed_dirs:
        all_probs: list[dict[str, np.ndarray]] = []
        for sd in seed_dirs:
            m = AutoGluonModel(cfg)
            m.load(sd)
            all_probs.append(m.predict_proba(X_test))
        probs = {
            target: np.mean([p[target] for p in all_probs], axis=0)
            for target in all_probs[0]
        }
        last_model = m  # keep last for get_confidence helper
    else:
        last_model = AutoGluonModel(cfg)
        last_model.load(model_path)
        probs = last_model.predict_proba(X_test)

    submission = pd.DataFrame(
        {
            "rally_uid": test_rally_uids.values,
            "actionId": probs["actionId"].argmax(axis=1),
            "pointId": probs["pointId"].argmax(axis=1),
            "serverGetPoint": (probs["serverGetPoint"] > 0.5).astype(int),
        }
    )

    confidence = last_model.get_confidence(probs)
    _save_confidence(confidence, test_rally_uids, run_name)
    sub_path = _save_submission(submission, run_name)

    # Also write a canonical copy inside the model dir so every versioned
    # folder (e.g. models/vN/) is self-contained — model + profile + submission
    # all in one place for reproduction / direct leaderboard submission.
    versioned_sub = model_path / "submission.csv"
    submission.to_csv(versioned_sub, index=False)

    n_seeds = len(seed_dirs) if seed_dirs else 1
    logger.info("Submission saved: %s (%d rows, %d seed%s)",
                sub_path, len(submission), n_seeds, "s" if n_seeds > 1 else "")
    logger.info("Submission also at: %s", versioned_sub)
    print(f"Submission saved: {sub_path} ({len(submission)} rows, "
          f"{n_seeds} seed{'s' if n_seeds > 1 else ''})")
    print(f"Submission also at: {versioned_sub}")
    return sub_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(ROOT / "models" / "autogluon"))
    parser.add_argument("--test-path", default=str(ROOT / "data" / "raw" / "test.csv"))
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or ("uploaded_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    predict_from_model(args.model_path, args.test_path, run_name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
