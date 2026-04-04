"""Post-hoc calibration and class weight optimization for V3 predictions.
Run after train_v3_champion.py completes.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_DIR, SUBMISSION_DIR

N_ACTION, N_POINT = 19, 10
SERVE_OK = {0, 15, 16, 17, 18}
SERVE_FORBIDDEN = {15, 16, 17, 18}


def macro_f1(y_true, y_probs, n_classes):
    y_pred = np.argmax(y_probs, axis=1)
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def apply_action_rules(probs, next_sns):
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


def temperature_scale(probs, temp):
    """Apply temperature scaling to probability predictions."""
    if temp <= 0:
        return probs
    log_probs = np.log(probs + 1e-10) / temp
    log_probs -= log_probs.max(axis=1, keepdims=True)
    exp_probs = np.exp(log_probs)
    return exp_probs / exp_probs.sum(axis=1, keepdims=True)


def optimize_class_weights(y_true, probs, n_classes, n_trials=200):
    """Optimize per-class multiplicative weights to maximize macro F1."""
    best_f1 = macro_f1(y_true, probs, n_classes)
    best_weights = np.ones(n_classes)

    np.random.seed(42)
    for trial in range(n_trials):
        weights = np.ones(n_classes)
        # Random perturbation for 1-3 classes
        n_perturb = np.random.randint(1, 4)
        for _ in range(n_perturb):
            cls = np.random.randint(0, n_classes)
            weights[cls] = np.random.uniform(0.5, 5.0)

        weighted = probs * weights[np.newaxis, :]
        weighted /= weighted.sum(axis=1, keepdims=True)
        f1 = macro_f1(y_true, weighted, n_classes)

        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights.copy()
            print(f"    Trial {trial}: F1={f1:.4f}, weights={dict((i,round(w,2)) for i,w in enumerate(weights) if w != 1.0)}")

    return best_weights, best_f1


def main():
    t0 = time.time()
    print("=" * 70)
    print("V3 CALIBRATION & THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # Load V3 OOF
    v3_file = os.path.join(MODEL_DIR, "oof_v3_champion.npz")
    if not os.path.exists(v3_file):
        print(f"ERROR: {v3_file} not found. Run train_v3_champion.py first.")
        return

    v3 = np.load(v3_file)
    y_act = v3["y_act"]
    y_pt = v3["y_pt"]
    y_srv = v3["y_srv"]
    next_sn = v3["next_sn"]

    # Best V3 blend (will be updated from training output)
    # Default: CB=0.5, XGB=0.3, LGB=0.2
    blend_configs = [
        (0.5, 0.3, 0.2),
        (0.6, 0.3, 0.1),
        (0.6, 0.2, 0.2),
        (0.4, 0.4, 0.2),
        (0.5, 0.4, 0.1),
        (0.7, 0.2, 0.1),
    ]

    best_overall = -1
    best_cfg = None

    for w_cb, w_xg, w_lg in blend_configs:
        act = w_cb * v3["catboost_act"] + w_xg * v3["xgboost_act"] + w_lg * v3["lightgbm_act"]
        pt = w_cb * v3["catboost_pt"] + w_xg * v3["xgboost_pt"] + w_lg * v3["lightgbm_pt"]
        srv = w_cb * v3["catboost_srv"] + w_xg * v3["xgboost_srv"] + w_lg * v3["lightgbm_srv"]

        act_r = apply_action_rules(act, next_sn)
        f1a = macro_f1(y_act, act_r, N_ACTION)
        f1p = macro_f1(y_pt, pt, N_POINT)
        auc = roc_auc_score(y_srv, srv)
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc

        if ov > best_overall:
            best_overall = ov
            best_cfg = (w_cb, w_xg, w_lg)
        print(f"  CB={w_cb:.1f} XGB={w_xg:.1f} LGB={w_lg:.1f}: OV={ov:.4f}")

    w_cb, w_xg, w_lg = best_cfg
    print(f"\n  Best blend: CB={w_cb:.1f} XGB={w_xg:.1f} LGB={w_lg:.1f} OV={best_overall:.4f}")

    oof_act = w_cb * v3["catboost_act"] + w_xg * v3["xgboost_act"] + w_lg * v3["lightgbm_act"]
    oof_pt = w_cb * v3["catboost_pt"] + w_xg * v3["xgboost_pt"] + w_lg * v3["lightgbm_pt"]
    oof_srv = w_cb * v3["catboost_srv"] + w_xg * v3["xgboost_srv"] + w_lg * v3["lightgbm_srv"]

    # --- Temperature scaling ---
    print("\n--- Temperature Scaling ---")
    for temp in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        act_t = temperature_scale(oof_act, temp)
        pt_t = temperature_scale(oof_pt, temp)
        act_r = apply_action_rules(act_t, next_sn)
        f1a = macro_f1(y_act, act_r, N_ACTION)
        f1p = macro_f1(y_pt, pt_t, N_POINT)
        auc = roc_auc_score(y_srv, oof_srv)
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        print(f"  T={temp:.1f}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # --- Class weight optimization for action ---
    print("\n--- Action Class Weight Optimization ---")
    act_ruled = apply_action_rules(oof_act, next_sn)
    act_weights, act_f1 = optimize_class_weights(y_act, act_ruled, N_ACTION, n_trials=500)
    print(f"  Best action F1: {act_f1:.4f}")

    # --- Class weight optimization for point ---
    print("\n--- Point Class Weight Optimization ---")
    pt_weights, pt_f1 = optimize_class_weights(y_pt, oof_pt, N_POINT, n_trials=500)
    print(f"  Best point F1: {pt_f1:.4f}")

    # Apply optimized weights
    act_opt = act_ruled * act_weights[np.newaxis, :]
    act_opt /= act_opt.sum(axis=1, keepdims=True)
    pt_opt = oof_pt * pt_weights[np.newaxis, :]
    pt_opt /= pt_opt.sum(axis=1, keepdims=True)

    f1a_opt = macro_f1(y_act, act_opt, N_ACTION)
    f1p_opt = macro_f1(y_pt, pt_opt, N_POINT)
    auc_opt = roc_auc_score(y_srv, oof_srv)
    ov_opt = 0.4*f1a_opt + 0.4*f1p_opt + 0.2*auc_opt
    print(f"\n  CALIBRATED OOF: F1a={f1a_opt:.4f} F1p={f1p_opt:.4f} AUC={auc_opt:.4f} OV={ov_opt:.4f}")

    # --- Also blend with V2 OOF ---
    print("\n--- Blend calibrated V3 with V2 ---")
    v2_file = os.path.join(MODEL_DIR, "oof_v2_fast.npz")
    if os.path.exists(v2_file):
        v2 = np.load(v2_file)
        v2_act = 0.6 * v2["catboost_act"] + 0.4 * v2["xgboost_act"]
        v2_pt = 0.6 * v2["catboost_pt"] + 0.3 * v2["xgboost_pt"] + 0.1 * v2["lightgbm_pt"]
        v2_srv = 0.3 * v2["catboost_srv"] + 0.4 * v2["xgboost_srv"] + 0.3 * v2["lightgbm_srv"]

        best_mega = -1
        best_w = 0
        for w in np.arange(0, 1.05, 0.05):
            ma = w * oof_act + (1-w) * v2_act
            mp = w * oof_pt + (1-w) * v2_pt
            ms = w * oof_srv + (1-w) * v2_srv
            # Apply rules and weights
            mar = apply_action_rules(ma, next_sn)
            mar = mar * act_weights[np.newaxis, :]
            mar /= mar.sum(axis=1, keepdims=True)
            mpr = mp * pt_weights[np.newaxis, :]
            mpr /= mpr.sum(axis=1, keepdims=True)

            f1a = macro_f1(y_act, mar, N_ACTION)
            f1p = macro_f1(y_pt, mpr, N_POINT)
            auc = roc_auc_score(y_srv, ms)
            ov = 0.4*f1a + 0.4*f1p + 0.2*auc
            if ov > best_mega:
                best_mega = ov
                best_w = w

        print(f"  Best mega: w_v3={best_w:.2f}, OV={best_mega:.4f}")

        # Generate final submission
        v3t = np.load(os.path.join(MODEL_DIR, "test_v3_champion.npz"))
        v2t = np.load(os.path.join(MODEL_DIR, "test_v2_fast.npz"))

        t_act_v3 = w_cb * v3t["catboost_act"] + w_xg * v3t["xgboost_act"] + w_lg * v3t["lightgbm_act"]
        t_pt_v3 = w_cb * v3t["catboost_pt"] + w_xg * v3t["xgboost_pt"] + w_lg * v3t["lightgbm_pt"]
        t_srv_v3 = w_cb * v3t["catboost_srv"] + w_xg * v3t["xgboost_srv"] + w_lg * v3t["lightgbm_srv"]

        t_act_v2 = 0.6 * v2t["catboost_act"] + 0.4 * v2t["xgboost_act"]
        t_pt_v2 = 0.6 * v2t["catboost_pt"] + 0.3 * v2t["xgboost_pt"] + 0.1 * v2t["lightgbm_pt"]
        t_srv_v2 = 0.3 * v2t["catboost_srv"] + 0.4 * v2t["xgboost_srv"] + 0.3 * v2t["lightgbm_srv"]

        final_act = best_w * t_act_v3 + (1-best_w) * t_act_v2
        final_pt = best_w * t_pt_v3 + (1-best_w) * t_pt_v2
        final_srv = best_w * t_srv_v3 + (1-best_w) * t_srv_v2

        test_next_sn = v3t["test_next_sn"]
        final_act = apply_action_rules(final_act, test_next_sn)
        final_act = final_act * act_weights[np.newaxis, :]
        final_act /= final_act.sum(axis=1, keepdims=True)
        final_pt = final_pt * pt_weights[np.newaxis, :]
        final_pt /= final_pt.sum(axis=1, keepdims=True)

        submission = pd.DataFrame({
            "rally_uid": v3t["rally_uids"].astype(int),
            "actionId": np.argmax(final_act, axis=1).astype(int),
            "pointId": np.argmax(final_pt, axis=1).astype(int),
            "serverGetPoint": (final_srv >= 0.5).astype(int),
        })

        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        out = os.path.join(SUBMISSION_DIR, "submission_v3_calibrated.csv")
        submission.to_csv(out, index=False, lineterminator="\n", encoding="utf-8")
        print(f"\nSaved: {out}")
        print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
        print(f"  pointId: {submission.pointId.value_counts().sort_index().to_dict()}")

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
