"""Post-processing: calibrate probabilities and optimize per-class thresholds
to maximize Macro F1.

Macro F1 = mean of per-class F1 scores. Since rare classes are weighted equally,
we need to carefully tune decision thresholds to find the sweet spot where
rare class recall doesn't hurt majority class precision too much.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_DIR, SUBMISSION_DIR

N_ACTION, N_POINT = 19, 10
SERVE_OK = {0, 15, 16, 17, 18}
SERVE_FORBIDDEN = {15, 16, 17, 18}


def macro_f1(y_true, y_pred, n_classes):
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def apply_action_rules_preds(preds, next_sns):
    """Apply rules to predicted class labels."""
    out = preds.copy()
    for i in range(len(out)):
        sn = next_sns[i]
        if sn == 1 and out[i] not in SERVE_OK:
            out[i] = 0  # default serve
        elif sn == 2 and out[i] in SERVE_FORBIDDEN:
            out[i] = 0
    return out


def apply_action_rules_probs(probs, next_sns):
    preds = probs.copy()
    for i in range(len(preds)):
        sn = next_sns[i]
        if sn == 1:
            mask = np.zeros(preds.shape[1])
            for a in SERVE_OK:
                if a < preds.shape[1]: mask[a] = 1.0
            preds[i] *= mask
        elif sn == 2:
            for a in SERVE_FORBIDDEN:
                if a < preds.shape[1]: preds[i, a] = 0.0
        total = preds[i].sum()
        if total > 0: preds[i] /= total
        else: preds[i] = np.ones(preds.shape[1]) / preds.shape[1]
    return preds


def optimize_temperature(probs, y_true, n_classes, next_sn=None, is_action=False):
    """Find optimal temperature for probability calibration."""
    best_score = -1
    best_t = 1.0

    for t in np.arange(0.1, 5.0, 0.1):
        scaled = probs ** (1/t)
        scaled /= scaled.sum(axis=1, keepdims=True)
        if is_action and next_sn is not None:
            scaled = apply_action_rules_probs(scaled, next_sn)
        preds = np.argmax(scaled, axis=1)
        score = macro_f1(y_true, preds, n_classes)
        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


def optimize_class_weights(probs, y_true, n_classes, next_sn=None, is_action=False):
    """Optimize per-class multiplicative weights to maximize Macro F1."""
    best_score = -1
    best_weights = np.ones(n_classes)

    # Strategy: boost underperforming classes
    # First compute per-class F1 with current probs
    if is_action and next_sn is not None:
        probs_ruled = apply_action_rules_probs(probs, next_sn)
    else:
        probs_ruled = probs

    base_preds = np.argmax(probs_ruled, axis=1)
    base_score = macro_f1(y_true, base_preds, n_classes)
    print(f"    Base Macro F1: {base_score:.4f}")

    # Per-class F1
    per_class_f1 = f1_score(y_true, base_preds, labels=list(range(n_classes)),
                            average=None, zero_division=0)
    print(f"    Per-class F1: {[f'{x:.3f}' for x in per_class_f1]}")

    # Find underperforming classes (F1 < mean)
    mean_f1 = np.mean(per_class_f1)
    weak_classes = np.where(per_class_f1 < mean_f1 * 0.5)[0]
    print(f"    Weak classes (F1 < {mean_f1*0.5:.3f}): {weak_classes.tolist()}")

    # Grid search class boosts
    weights = np.ones(n_classes)
    for cls in weak_classes:
        best_cls_score = base_score
        best_w = 1.0
        for w in np.arange(1.0, 5.0, 0.5):
            test_weights = weights.copy()
            test_weights[cls] = w
            adjusted = probs * test_weights[np.newaxis, :]
            adjusted /= adjusted.sum(axis=1, keepdims=True)
            if is_action and next_sn is not None:
                adjusted = apply_action_rules_probs(adjusted, next_sn)
            preds = np.argmax(adjusted, axis=1)
            score = macro_f1(y_true, preds, n_classes)
            if score > best_cls_score:
                best_cls_score = score
                best_w = w
        weights[cls] = best_w

    # Evaluate with all adjusted weights
    adjusted = probs * weights[np.newaxis, :]
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    if is_action and next_sn is not None:
        adjusted = apply_action_rules_probs(adjusted, next_sn)
    preds = np.argmax(adjusted, axis=1)
    final_score = macro_f1(y_true, preds, n_classes)
    print(f"    After class weight opt: {final_score:.4f} (weights: {weights})")

    return weights, final_score


def main():
    print("=" * 70)
    print("PROBABILITY CALIBRATION + THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # Load V2 fast predictions (our best individual model outputs)
    v2_path = os.path.join(MODEL_DIR, "oof_v2_fast.npz")
    v2_test_path = os.path.join(MODEL_DIR, "test_v2_fast.npz")
    sn_path = os.path.join(MODEL_DIR, "oof_sn_cond.npz")
    sn_test_path = os.path.join(MODEL_DIR, "test_sn_cond.npz")

    if not os.path.exists(v2_path):
        print("Run train_fast_v2.py first!")
        return

    d = np.load(v2_path)
    # Best V2 blend
    oof_act = 0.6 * d["catboost_act"] + 0.4 * d["xgboost_act"]
    oof_pt = 0.6 * d["catboost_pt"] + 0.3 * d["xgboost_pt"] + 0.1 * d["lightgbm_pt"]
    oof_srv = 0.3 * d["catboost_srv"] + 0.4 * d["xgboost_srv"] + 0.3 * d["lightgbm_srv"]
    y_act = d["y_act"]
    y_pt = d["y_pt"]
    y_srv = d["y_srv"]
    next_sn = d["next_sn"]

    # Add SN-conditioned
    if os.path.exists(sn_path):
        d_sn = np.load(sn_path)
        sn_act = 0.5 * d_sn["oof_act_sn"] + 0.5 * d_sn["oof_act_global"]
        sn_pt = 0.5 * d_sn["oof_pt_sn"] + 0.5 * d_sn["oof_pt_global"]
        # Mega-blend
        oof_act = 0.6 * oof_act + 0.4 * sn_act
        oof_pt = 0.9 * oof_pt + 0.1 * sn_pt

    # 1. Temperature scaling
    print("\n--- Temperature Scaling ---")
    print("  Action:")
    best_t_act, score_act = optimize_temperature(oof_act, y_act, N_ACTION, next_sn, True)
    print(f"    Best temp: {best_t_act:.1f}, F1: {score_act:.4f}")

    print("  Point:")
    best_t_pt, score_pt = optimize_temperature(oof_pt, y_pt, N_POINT)
    print(f"    Best temp: {best_t_pt:.1f}, F1: {score_pt:.4f}")

    # 2. Class weight optimization
    print("\n--- Class Weight Optimization ---")
    print("  Action:")
    act_scaled = oof_act ** (1/best_t_act)
    act_scaled /= act_scaled.sum(axis=1, keepdims=True)
    act_weights, act_f1 = optimize_class_weights(act_scaled, y_act, N_ACTION, next_sn, True)

    print("\n  Point:")
    pt_scaled = oof_pt ** (1/best_t_pt)
    pt_scaled /= pt_scaled.sum(axis=1, keepdims=True)
    pt_weights, pt_f1 = optimize_class_weights(pt_scaled, y_pt, N_POINT)

    # 3. Final evaluation
    print("\n--- Final Evaluation ---")
    final_act = act_scaled * act_weights[np.newaxis, :]
    final_act /= final_act.sum(axis=1, keepdims=True)
    final_act = apply_action_rules_probs(final_act, next_sn)
    final_pt = pt_scaled * pt_weights[np.newaxis, :]
    final_pt /= final_pt.sum(axis=1, keepdims=True)

    f1a = macro_f1(y_act, np.argmax(final_act, axis=1), N_ACTION)
    f1p = macro_f1(y_pt, np.argmax(final_pt, axis=1), N_POINT)
    auc = roc_auc_score(y_srv, oof_srv)
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc
    print(f"  CALIBRATED OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # 4. Generate calibrated submission
    dt = np.load(v2_test_path)
    test_act = 0.6 * dt["catboost_act"] + 0.4 * dt["xgboost_act"]
    test_pt = 0.6 * dt["catboost_pt"] + 0.3 * dt["xgboost_pt"] + 0.1 * dt["lightgbm_pt"]
    test_srv = 0.3 * dt["catboost_srv"] + 0.4 * dt["xgboost_srv"] + 0.3 * dt["lightgbm_srv"]
    test_next_sn = dt["test_next_sn"]
    rally_uids = dt["rally_uids"]

    if os.path.exists(sn_test_path):
        d_sn_t = np.load(sn_test_path)
        sn_t_act = 0.5 * d_sn_t["test_act_sn"] + 0.5 * d_sn_t["test_act_global"]
        sn_t_pt = 0.5 * d_sn_t["test_pt_sn"] + 0.5 * d_sn_t["test_pt_global"]
        test_act = 0.6 * test_act + 0.4 * sn_t_act
        test_pt = 0.9 * test_pt + 0.1 * sn_t_pt

    # Apply calibration
    test_act_cal = test_act ** (1/best_t_act)
    test_act_cal /= test_act_cal.sum(axis=1, keepdims=True)
    test_act_cal = test_act_cal * act_weights[np.newaxis, :]
    test_act_cal /= test_act_cal.sum(axis=1, keepdims=True)
    test_act_cal = apply_action_rules_probs(test_act_cal, test_next_sn)

    test_pt_cal = test_pt ** (1/best_t_pt)
    test_pt_cal /= test_pt_cal.sum(axis=1, keepdims=True)
    test_pt_cal = test_pt_cal * pt_weights[np.newaxis, :]
    test_pt_cal /= test_pt_cal.sum(axis=1, keepdims=True)

    submission = pd.DataFrame({
        "rally_uid": rally_uids.astype(int),
        "actionId": np.argmax(test_act_cal, axis=1).astype(int),
        "pointId": np.argmax(test_pt_cal, axis=1).astype(int),
        "serverGetPoint": (test_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_calibrated.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nSaved: {out_path} ({submission.shape})")
    print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId: {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  serverGetPoint: {submission.serverGetPoint.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
