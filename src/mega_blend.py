"""Mega-blend: combine all available model predictions with optimal weights.

Sources:
1. V2 Fast Ensemble (CB+XGB+LGB with V2 features + sample weights)
2. SN-Conditioned (global + SN-specific blend)
3. V1 Ensemble (CB+XGB+LGB with V1 features) - from earlier run
"""
import sys, os, warnings
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


def load_predictions():
    """Load all saved predictions."""
    preds = {}

    # V2 Fast ensemble
    v2_path = os.path.join(MODEL_DIR, "oof_v2_fast.npz")
    if os.path.exists(v2_path):
        d = np.load(v2_path)
        # Blend the 3 GBDT models from V2
        # Best weights from V2: act(CB:0.6, XGB:0.4), pt(CB:0.6, XGB:0.3, LGB:0.1), srv(CB:0.3, XGB:0.4, LGB:0.3)
        preds["v2_oof_act"] = 0.6 * d["catboost_act"] + 0.4 * d["xgboost_act"]
        preds["v2_oof_pt"] = 0.6 * d["catboost_pt"] + 0.3 * d["xgboost_pt"] + 0.1 * d["lightgbm_pt"]
        preds["v2_oof_srv"] = 0.3 * d["catboost_srv"] + 0.4 * d["xgboost_srv"] + 0.3 * d["lightgbm_srv"]
        preds["y_act"] = d["y_act"]
        preds["y_pt"] = d["y_pt"]
        preds["y_srv"] = d["y_srv"]
        preds["next_sn"] = d["next_sn"]
        print(f"  Loaded V2 OOF: {preds['v2_oof_act'].shape}")

    v2_test_path = os.path.join(MODEL_DIR, "test_v2_fast.npz")
    if os.path.exists(v2_test_path):
        d = np.load(v2_test_path)
        preds["v2_test_act"] = 0.6 * d["catboost_act"] + 0.4 * d["xgboost_act"]
        preds["v2_test_pt"] = 0.6 * d["catboost_pt"] + 0.3 * d["xgboost_pt"] + 0.1 * d["lightgbm_pt"]
        preds["v2_test_srv"] = 0.3 * d["catboost_srv"] + 0.4 * d["xgboost_srv"] + 0.3 * d["lightgbm_srv"]
        preds["test_next_sn"] = d["test_next_sn"]
        preds["rally_uids"] = d["rally_uids"]
        print(f"  Loaded V2 Test: {preds['v2_test_act'].shape}")

    # SN-conditioned
    sn_path = os.path.join(MODEL_DIR, "oof_sn_cond.npz")
    if os.path.exists(sn_path):
        d = np.load(sn_path)
        # Best blend w_sn=0.5
        preds["sn_oof_act"] = 0.5 * d["oof_act_sn"] + 0.5 * d["oof_act_global"]
        preds["sn_oof_pt"] = 0.5 * d["oof_pt_sn"] + 0.5 * d["oof_pt_global"]
        preds["sn_oof_srv"] = d["oof_srv_global"]
        print(f"  Loaded SN-Cond OOF: {preds['sn_oof_act'].shape}")

    sn_test_path = os.path.join(MODEL_DIR, "test_sn_cond.npz")
    if os.path.exists(sn_test_path):
        d = np.load(sn_test_path)
        preds["sn_test_act"] = 0.5 * d["test_act_sn"] + 0.5 * d["test_act_global"]
        preds["sn_test_pt"] = 0.5 * d["test_pt_sn"] + 0.5 * d["test_pt_global"]
        preds["sn_test_srv"] = d["test_srv_global"]
        print(f"  Loaded SN-Cond Test: {preds['sn_test_act'].shape}")

    return preds


def main():
    print("=" * 70)
    print("MEGA-BLEND: Combining All Models")
    print("=" * 70)

    print("\nLoading predictions...")
    preds = load_predictions()

    if "v2_oof_act" not in preds or "sn_oof_act" not in preds:
        print("ERROR: Missing predictions. Run train_fast_v2.py and train_sn_conditioned.py first.")
        return

    y_act = preds["y_act"]
    y_pt = preds["y_pt"]
    y_srv = preds["y_srv"]
    next_sn = preds["next_sn"]

    # Available OOF predictions
    models = {
        "v2": {"act": preds["v2_oof_act"], "pt": preds["v2_oof_pt"], "srv": preds["v2_oof_srv"]},
        "sn": {"act": preds["sn_oof_act"], "pt": preds["sn_oof_pt"], "srv": preds["sn_oof_srv"]},
    }

    # Individual scores
    print("\n--- Individual Model Scores ---")
    for name, m in models.items():
        act_r = apply_action_rules(m["act"], next_sn)
        f1a = macro_f1(y_act, act_r, N_ACTION)
        f1p = macro_f1(y_pt, m["pt"], N_POINT)
        auc = roc_auc_score(y_srv, m["srv"])
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc
        print(f"  {name:10s}: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # Search best blend
    print("\n--- Blend Search ---")
    best_ov = -1
    best_w = {}

    weight_grid = np.arange(0, 1.05, 0.1)
    model_names = list(models.keys())

    for task, n_cls, y_true in [("act", N_ACTION, y_act), ("pt", N_POINT, y_pt)]:
        best_score = -1
        best_task_w = 0.5
        for w in weight_grid:
            blend = w * models[model_names[0]][task] + (1-w) * models[model_names[1]][task]
            if task == "act":
                blend = apply_action_rules(blend, next_sn)
            score = macro_f1(y_true, blend, n_cls)
            if score > best_score:
                best_score = score
                best_task_w = w
        best_w[task] = best_task_w
        print(f"  {task}: F1={best_score:.4f}, w_v2={best_task_w:.1f}")

    # Server blend
    best_score = -1
    best_srv_w = 0.5
    for w in weight_grid:
        blend = w * models[model_names[0]]["srv"] + (1-w) * models[model_names[1]]["srv"]
        score = roc_auc_score(y_srv, blend)
        if score > best_score:
            best_score = score
            best_srv_w = w
    best_w["srv"] = best_srv_w
    print(f"  srv: AUC={best_score:.4f}, w_v2={best_srv_w:.1f}")

    # Final blend evaluation
    blend_act = best_w["act"] * models["v2"]["act"] + (1-best_w["act"]) * models["sn"]["act"]
    blend_pt = best_w["pt"] * models["v2"]["pt"] + (1-best_w["pt"]) * models["sn"]["pt"]
    blend_srv = best_w["srv"] * models["v2"]["srv"] + (1-best_w["srv"]) * models["sn"]["srv"]

    blend_act_r = apply_action_rules(blend_act, next_sn)
    f1a = macro_f1(y_act, blend_act_r, N_ACTION)
    f1p = macro_f1(y_pt, blend_pt, N_POINT)
    auc = roc_auc_score(y_srv, blend_srv)
    ov = 0.4*f1a + 0.4*f1p + 0.2*auc
    print(f"\n  MEGA-BLEND OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # Generate submission
    test_next_sn = preds["test_next_sn"]
    blend_test_act = best_w["act"] * preds["v2_test_act"] + (1-best_w["act"]) * preds["sn_test_act"]
    blend_test_pt = best_w["pt"] * preds["v2_test_pt"] + (1-best_w["pt"]) * preds["sn_test_pt"]
    blend_test_srv = best_w["srv"] * preds["v2_test_srv"] + (1-best_w["srv"]) * preds["sn_test_srv"]

    blend_test_act = apply_action_rules(blend_test_act, test_next_sn)

    submission = pd.DataFrame({
        "rally_uid": preds["rally_uids"].astype(int),
        "actionId": np.argmax(blend_test_act, axis=1).astype(int),
        "pointId": np.argmax(blend_test_pt, axis=1).astype(int),
        "serverGetPoint": (blend_test_srv >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, "submission_mega_blend.csv")
    submission.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\nSaved: {out_path} ({submission.shape})")
    print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId: {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  serverGetPoint: {submission.serverGetPoint.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
