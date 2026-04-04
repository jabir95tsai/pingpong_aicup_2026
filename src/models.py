"""Model definitions for the three prediction tasks."""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
from config import N_ACTION_CLASSES, N_POINT_CLASSES, RANDOM_SEED, SERVE_ACTION_IDS, RETURN_FORBIDDEN_ACTIONS


def get_lgb_params_multiclass(n_classes: int) -> dict:
    return {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": RANDOM_SEED,
        "verbose": -1,
        "is_unbalance": True,
    }


def get_lgb_params_binary() -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": RANDOM_SEED,
        "verbose": -1,
        "is_unbalance": True,
    }


def train_lgb_multiclass(X_train, y_train, X_val, y_val, n_classes, num_boost_round=1000):
    params = get_lgb_params_multiclass(n_classes)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return model


def train_lgb_binary(X_train, y_train, X_val, y_val, num_boost_round=1000):
    params = get_lgb_params_binary()
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return model


def predict_multiclass(model, X) -> np.ndarray:
    """Predict class probabilities, return shape (n_samples, n_classes)."""
    return model.predict(X, num_iteration=model.best_iteration)


def predict_binary(model, X) -> np.ndarray:
    """Predict probability of class 1."""
    return model.predict(X, num_iteration=model.best_iteration)


def apply_action_constraints(probs: np.ndarray, next_strike_numbers: np.ndarray) -> np.ndarray:
    """Apply domain constraints to actionId predictions.

    - strikeNumber=1 (serve): only allow {0, 15, 16, 17, 18}
    - strikeNumber=2 (return): forbid {15, 16, 17, 18}
    """
    preds = probs.copy()
    for i in range(len(preds)):
        sn = next_strike_numbers[i]
        if sn == 1:
            # Only serve actions allowed
            mask = np.zeros(preds.shape[1])
            for aid in SERVE_ACTION_IDS:
                if aid < preds.shape[1]:
                    mask[aid] = 1.0
            preds[i] *= mask
        elif sn == 2:
            # Forbid serve actions on return
            for aid in RETURN_FORBIDDEN_ACTIONS:
                if aid < preds.shape[1]:
                    preds[i, aid] = 0.0

        # Renormalize
        total = preds[i].sum()
        if total > 0:
            preds[i] /= total
        else:
            preds[i] = np.ones(preds.shape[1]) / preds.shape[1]

    return preds


def eval_macro_f1(y_true, y_pred_probs, n_classes):
    """Compute macro F1 from probability matrix."""
    y_pred = np.argmax(y_pred_probs, axis=1)
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def eval_auc(y_true, y_pred_prob):
    return roc_auc_score(y_true, y_pred_prob)
