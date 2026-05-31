"""
AutoGluon TabularPredictor baseline.
Trains three separate predictors: actionId, pointId (multiclass), serverGetPoint (binary).

- actionId:       19 classes (0-18), Macro F1, weight 0.4
- pointId:        10 classes (0-9),  Macro F1, weight 0.4
- serverGetPoint: binary,            AUC-ROC,  weight 0.2

Validation strategy:
- Default (group_aware=True): split off a rally-disjoint holdout via
  GroupShuffleSplit, pass it as tuning_data to AutoGluon, and report scores
  on the holdout. This avoids rally-level leakage that random bagging causes.
- Legacy (group_aware=False): use AutoGluon's random bagging folds and OOF
  scores. May overestimate performance because shots from the same rally can
  appear in both train and validation folds.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import GroupShuffleSplit

from src.evaluate.metrics import evaluate_predictions

if TYPE_CHECKING:
    from src.train import TrainConfig

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models" / "autogluon"


def make_balanced_sample_weights(y_target: pd.Series) -> np.ndarray:
    """Inverse-frequency class weights so each class contributes equally
    regardless of base rate — helpful for macro F1 on imbalanced targets."""
    counts = y_target.value_counts()
    n_classes = len(counts)
    total = len(y_target)
    class_weight = {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}
    return y_target.map(class_weight).to_numpy()


class AutoGluonModel:
    def __init__(self, cfg: TrainConfig, model_dir: Path | None = None):
        self.cfg = cfg
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.predictors: dict[str, TabularPredictor] = {}
        self.feature_importances: dict[str, list] = {}
        self.oof_confidence: dict[str, np.ndarray] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        groups: pd.Series | None = None,
    ) -> dict:
        """
        Train one TabularPredictor per target. Returns competition scores.

        If `cfg.group_aware` is True and `groups` is provided, a rally-disjoint
        holdout is split via GroupShuffleSplit and passed to AutoGluon as
        tuning_data. Bagging is disabled (num_bag_folds=0) in this mode and
        scores are computed on the holdout instead of OOF predictions.
        """
        target_metrics = {
            "actionId": self.cfg.action_metric,
            "pointId": self.cfg.point_metric,
            "serverGetPoint": self.cfg.server_metric,
        }

        # Decide validation strategy
        use_group_holdout = bool(getattr(self.cfg, "group_aware", False)) and groups is not None
        if use_group_holdout:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self.cfg.holdout_frac,
                random_state=self.cfg.seed,
            )
            train_idx, val_idx = next(splitter.split(X, y, groups=groups))
            X_tr, X_val = X.iloc[train_idx].reset_index(drop=True), X.iloc[val_idx].reset_index(drop=True)
            y_tr, y_val = y.iloc[train_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)
            logger.info(
                "Group-aware holdout: %d train rows / %d val rows over %d / %d unique rallies",
                len(train_idx), len(val_idx),
                groups.iloc[train_idx].nunique(), groups.iloc[val_idx].nunique(),
            )
        else:
            X_tr, y_tr = X, y
            X_val = y_val = None

        eval_action_pred = None
        eval_point_pred = None
        eval_server_prob = None

        use_class_weight = bool(getattr(self.cfg, "class_weight", False))

        for target, metric in target_metrics.items():
            logger.info("=== Training AutoGluon for: %s ===", target)
            print(f"\n=== Training AutoGluon for: {target} ===")  # keep for Streamlit log parsing
            train_data = X_tr.copy()
            train_data[target] = y_tr[target].values
            if use_class_weight:
                train_data["sample_weight"] = make_balanced_sample_weights(
                    pd.Series(y_tr[target].values),
                )

            tuning_data = None
            if use_group_holdout:
                tuning_data = X_val.copy()
                tuning_data[target] = y_val[target].values
                if use_class_weight:
                    # Derive tuning weights from the same class-weight scheme
                    # (AutoGluon requires the column on tuning_data too).
                    tuning_data["sample_weight"] = make_balanced_sample_weights(
                        pd.Series(y_val[target].values),
                    )

            predictor = TabularPredictor(
                label=target,
                eval_metric=metric,
                path=str(self.model_dir / target),
                verbosity=self.cfg.verbosity,
                sample_weight="sample_weight" if use_class_weight else None,
                weight_evaluation=False,
            )
            fit_kwargs = dict(
                presets=self.cfg.presets,
                num_bag_folds=0 if use_group_holdout else self.cfg.num_bag_folds,
                num_bag_sets=self.cfg.num_bag_sets,
                num_stack_levels=0 if use_group_holdout else self.cfg.num_stack_levels,
                time_limit=self.cfg.time_limit,
                excluded_model_types=self.cfg.excluded_model_types or None,
                calibrate=self.cfg.calibrate,
                auto_stack=False if use_group_holdout else self.cfg.auto_stack,
                use_bag_holdout=self.cfg.use_bag_holdout,
                keep_only_best=self.cfg.keep_only_best,
                num_cpus=self.cfg.num_cpus if self.cfg.num_cpus != -1 else "auto",
                num_gpus=self.cfg.num_gpus,
                holdout_frac=(
                    None
                    if use_group_holdout
                    else (self.cfg.holdout_frac if self.cfg.num_bag_folds == 0 else None)
                ),
                fit_weighted_ensemble=self.cfg.fit_weighted_ensemble,
                ag_args_fit={
                    "early_stop": self.cfg.early_stopping_rounds,
                    "seed_value": self.cfg.seed,
                },
            )
            if tuning_data is not None:
                fit_kwargs["tuning_data"] = tuning_data

            predictor.fit(train_data, **fit_kwargs)

            # Score BEFORE any refit. In group-aware + refit_full mode the
            # production model below is trained on the holdout too, so scoring
            # it would leak data. Capture predictions from the validation
            # predictor while it still hasn't seen X_val.
            score_X = X_val if use_group_holdout else X
            eval_proba = predictor.predict_proba(score_X)
            if target == "serverGetPoint":
                eval_server_prob = eval_proba[1].values
                self.oof_confidence[target] = np.maximum(eval_proba[1].values, 1 - eval_proba[1].values)
            elif target == "actionId":
                eval_action_pred = _proba_df_to_array(eval_proba, 19)
                self.oof_confidence[target] = eval_action_pred.max(axis=1)
            else:  # pointId
                eval_point_pred = _proba_df_to_array(eval_proba, 10)
                self.oof_confidence[target] = eval_point_pred.max(axis=1)

            if self.cfg.refit_full and use_group_holdout:
                # Build a production model that uses *all* the data (train + tuning).
                # The validation predictor above is discarded; AutoGluon's own
                # `refit_full` would only re-train on `train_data`, so the
                # rally-disjoint holdout would be permanently excluded.
                full_train = pd.concat([train_data, tuning_data], ignore_index=True)
                logger.info(
                    "Refitting %s on combined train+tuning (%d rows)",
                    target, len(full_train),
                )
                print(f"  Refitting {target} on combined train+tuning ({len(full_train)} rows)...")
                target_dir = self.model_dir / target
                shutil.rmtree(target_dir, ignore_errors=True)
                production_predictor = TabularPredictor(
                    label=target,
                    eval_metric=metric,
                    path=str(target_dir),
                    verbosity=self.cfg.verbosity,
                )
                production_predictor.fit(
                    full_train,
                    presets=self.cfg.presets,
                    num_bag_folds=0,
                    num_stack_levels=0,
                    time_limit=self.cfg.time_limit,
                    excluded_model_types=self.cfg.excluded_model_types or None,
                    calibrate=self.cfg.calibrate,
                    keep_only_best=self.cfg.keep_only_best,
                    num_cpus=self.cfg.num_cpus if self.cfg.num_cpus != -1 else "auto",
                    num_gpus=self.cfg.num_gpus,
                    fit_weighted_ensemble=self.cfg.fit_weighted_ensemble,
                    ag_args_fit={"early_stop": self.cfg.early_stopping_rounds},
                )
                predictor = production_predictor
            elif self.cfg.refit_full:
                logger.info("Refitting %s on full data...", target)
                print(f"  Refitting {target} on full data...")
                predictor.refit_full(
                    set_best_to_refit_full=self.cfg.set_best_to_refit_full,
                )

            self.predictors[target] = predictor

            # AutoGluon's feature_importance() needs an explicit dataset when
            # there's no OOF data (i.e. when bagging is off — which is always
            # the case in group-aware mode). For legacy bagging mode it can
            # use the internal OOF predictions, so we omit `data`.
            try:
                fi_kwargs = {"silent": True}
                if use_group_holdout:
                    fi_data = tuning_data
                elif self.cfg.num_bag_folds == 0:
                    fi_data = train_data
                else:
                    fi_data = None
                if fi_data is not None:
                    fi_kwargs["data"] = fi_data
                imp = predictor.feature_importance(**fi_kwargs)
                self.feature_importances[target] = (
                    imp[["importance"]].head(20).reset_index()
                    .rename(columns={"index": "feature"})
                    .to_dict(orient="records")
                )
            except Exception as e:
                logger.warning("Feature importance failed for %s: %s", target, e)
                self.feature_importances[target] = []

        score_y = y_val if use_group_holdout else y
        scores = evaluate_predictions(
            action_true=score_y["actionId"].values,
            action_pred=eval_action_pred.argmax(axis=1),
            point_true=score_y["pointId"].values,
            point_pred=eval_point_pred.argmax(axis=1),
            server_true=score_y["serverGetPoint"].values,
            server_prob=eval_server_prob,
        )
        score_label = "Group-holdout" if use_group_holdout else "OOF"
        logger.info("%s Scores: %s", score_label, scores)
        print(f"\n{score_label} Scores: {scores}")
        return scores

    def predict_proba(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        results = {}
        for target, predictor in self.predictors.items():
            proba = predictor.predict_proba(X)
            if target == "serverGetPoint":
                results[target] = proba[1].values
            elif target == "actionId":
                results[target] = _proba_df_to_array(proba, 19)
            else:  # pointId
                results[target] = _proba_df_to_array(proba, 10)
        return results

    def get_leaderboards(self) -> dict:
        """Return leaderboard for each target as a list of dicts (JSON-serialisable)."""
        result = {}
        for target, predictor in self.predictors.items():
            lb = predictor.leaderboard(silent=True)
            result[target] = lb[["model", "score_val", "fit_time", "pred_time_val"]].to_dict(orient="records")
        return result

    def get_feature_importances(self) -> dict:
        return self.feature_importances

    def get_confidence(self, probs: dict) -> dict:
        """Return max prediction confidence per target for test set."""
        result = {}
        for target, pred in probs.items():
            if target == "serverGetPoint":
                arr = pred  # already 1D probability of positive class
                result[f"{target}_conf"] = np.maximum(arr, 1 - arr)
            else:
                result[f"{target}_conf"] = pred.max(axis=1)
        return result

    def load(self, path: str | Path):
        path = Path(path)
        for target in ["actionId", "pointId", "serverGetPoint"]:
            self.predictors[target] = TabularPredictor.load(str(path / target))
        logger.info("Models loaded from %s", path)
        print(f"Models loaded from {path}")


def _proba_df_to_array(proba_df: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Convert AutoGluon predict_proba DataFrame to (n_samples, n_classes) array."""
    out = np.zeros((len(proba_df), n_classes))
    for cls in proba_df.columns:
        idx = int(cls)
        if 0 <= idx < n_classes:
            out[:, idx] = proba_df[cls].values
    return out
