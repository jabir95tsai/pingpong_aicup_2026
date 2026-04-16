"""Error analysis on V5 OOF predictions.
Produces per-class F1, per-SN breakdown, confusion matrices.
"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_DIR
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

N_ACTION, N_POINT = 19, 10
SERVE_FORBIDDEN = {15, 16, 17, 18}

ACTION_NAMES = {
    0: "none", 1: "pull", 2: "counter", 3: "smash", 4: "flick",
    5: "fast_block", 6: "push", 7: "lob", 8: "arc", 9: "bump",
    10: "chop", 11: "short_block", 12: "slice", 13: "block", 14: "high_ball",
    15: "serve_std", 16: "serve_hook", 17: "serve_rev", 18: "serve_squat"
}
POINT_NAMES = {
    0: "miss/net", 1: "FH_short", 2: "mid_short", 3: "BH_short",
    4: "FH_mid", 5: "mid_mid", 6: "BH_mid",
    7: "FH_long", 8: "mid_long", 9: "BH_long"
}

def apply_action_rules(probs, next_sns):
    preds = probs.copy()
    for i in range(len(preds)):
        if next_sns[i] == 2:
            for a in SERVE_FORBIDDEN:
                if a < preds.shape[1]: preds[i, a] = 0.0
        total = preds[i].sum()
        if total > 0: preds[i] /= total
        else: preds[i] = np.ones(preds.shape[1]) / preds.shape[1]
    return preds


def main():
    oof_path = os.path.join(MODEL_DIR, "oof_v5_clean.npz")
    if not os.path.exists(oof_path):
        print(f"ERROR: {oof_path} not found")
        return

    data = np.load(oof_path, allow_pickle=True)
    print("Keys:", list(data.keys()))

    y_act = data["y_act"]
    y_pt = data["y_pt"]
    y_srv = data["y_srv"]
    next_sn = data["next_sn"]

    # Load all model OOF predictions
    models = {}
    for prefix in ["CB", "XGB", "LGB"]:
        models[prefix] = {
            "act": data[f"{prefix}_act"],
            "pt": data[f"{prefix}_pt"],
            "srv": data[f"{prefix}_srv"],
        }

    n_samples = len(y_act)
    print(f"\nTotal samples: {n_samples}")
    print(f"SN distribution: {dict(zip(*np.unique(next_sn, return_counts=True)))}")

    # ================================================================
    # 1. PER-CLASS F1 BREAKDOWN
    # ================================================================
    print("\n" + "=" * 80)
    print("1. PER-CLASS F1 BREAKDOWN (actionId)")
    print("=" * 80)

    for model_name in ["CB", "XGB", "LGB"]:
        act_probs = apply_action_rules(models[model_name]["act"], next_sn)
        act_preds = np.argmax(act_probs, axis=1)
        per_class_f1 = f1_score(y_act, act_preds, labels=list(range(N_ACTION)),
                                average=None, zero_division=0)
        macro_f1_val = np.mean(per_class_f1)

        print(f"\n{model_name} actionId (Macro F1 = {macro_f1_val:.4f}):")
        print(f"{'Class':>6} {'Name':>15} {'F1':>8} {'Support':>8} {'Predicted':>10}")
        print("-" * 55)
        for c in range(N_ACTION):
            support = (y_act == c).sum()
            predicted = (act_preds == c).sum()
            print(f"{c:>6} {ACTION_NAMES.get(c,'?'):>15} {per_class_f1[c]:>8.4f} {support:>8} {predicted:>10}")

    print("\n" + "=" * 80)
    print("2. PER-CLASS F1 BREAKDOWN (pointId)")
    print("=" * 80)

    for model_name in ["CB", "XGB", "LGB"]:
        pt_probs = models[model_name]["pt"]
        pt_preds = np.argmax(pt_probs, axis=1)
        per_class_f1 = f1_score(y_pt, pt_preds, labels=list(range(N_POINT)),
                                average=None, zero_division=0)
        macro_f1_val = np.mean(per_class_f1)

        print(f"\n{model_name} pointId (Macro F1 = {macro_f1_val:.4f}):")
        print(f"{'Class':>6} {'Name':>15} {'F1':>8} {'Support':>8} {'Predicted':>10}")
        print("-" * 55)
        for c in range(N_POINT):
            support = (y_pt == c).sum()
            predicted = (pt_preds == c).sum()
            print(f"{c:>6} {POINT_NAMES.get(c,'?'):>15} {per_class_f1[c]:>8.4f} {support:>8} {predicted:>10}")

    # ================================================================
    # 3. PER-SN BREAKDOWN (best model = CB)
    # ================================================================
    print("\n" + "=" * 80)
    print("3. PER-STRIKENUMBER BREAKDOWN (CatBoost)")
    print("=" * 80)

    cb_act = apply_action_rules(models["CB"]["act"], next_sn)
    cb_pt = models["CB"]["pt"]
    cb_srv = models["CB"]["srv"]

    print(f"\n{'SN':>4} {'Count':>7} {'F1_act':>8} {'F1_pt':>8} {'AUC':>8} {'OV':>8}")
    print("-" * 50)
    for sn in sorted(np.unique(next_sn)):
        mask = next_sn == sn
        if mask.sum() < 10:
            continue
        f1a = f1_score(y_act[mask], np.argmax(cb_act[mask], axis=1),
                       labels=list(range(N_ACTION)), average="macro", zero_division=0)
        f1p = f1_score(y_pt[mask], np.argmax(cb_pt[mask], axis=1),
                       labels=list(range(N_POINT)), average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_srv[mask], cb_srv[mask])
        except:
            auc = 0.5
        ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        print(f"{sn:>4} {mask.sum():>7} {f1a:>8.4f} {f1p:>8.4f} {auc:>8.4f} {ov:>8.4f}")

    # ================================================================
    # 4. CONFUSION MATRIX (top confusions)
    # ================================================================
    print("\n" + "=" * 80)
    print("4. TOP CONFUSIONS (CatBoost)")
    print("=" * 80)

    # ActionId confusions
    act_preds = np.argmax(cb_act, axis=1)
    cm_act = confusion_matrix(y_act, act_preds, labels=list(range(N_ACTION)))
    print("\nactionId - Top 15 confusions (true -> pred, count):")
    confusions = []
    for i in range(N_ACTION):
        for j in range(N_ACTION):
            if i != j and cm_act[i, j] > 0:
                confusions.append((i, j, cm_act[i, j]))
    confusions.sort(key=lambda x: -x[2])
    for true_c, pred_c, cnt in confusions[:15]:
        print(f"  {ACTION_NAMES[true_c]:>15} -> {ACTION_NAMES[pred_c]:>15}: {cnt:>5}")

    # PointId confusions
    pt_preds = np.argmax(cb_pt, axis=1)
    cm_pt = confusion_matrix(y_pt, pt_preds, labels=list(range(N_POINT)))
    print("\npointId - Top 15 confusions (true -> pred, count):")
    confusions = []
    for i in range(N_POINT):
        for j in range(N_POINT):
            if i != j and cm_pt[i, j] > 0:
                confusions.append((i, j, cm_pt[i, j]))
    confusions.sort(key=lambda x: -x[2])
    for true_c, pred_c, cnt in confusions[:15]:
        print(f"  {POINT_NAMES[true_c]:>15} -> {POINT_NAMES[pred_c]:>15}: {cnt:>5}")

    # ================================================================
    # 5. BLEND OOF SCORES
    # ================================================================
    print("\n" + "=" * 80)
    print("5. BLEND OV SCORES")
    print("=" * 80)

    best_ov = -1
    best_w = None
    for w_cb in np.arange(0.2, 0.8, 0.05):
        for w_xg in np.arange(0.0, 0.5, 0.05):
            w_lg = 1.0 - w_cb - w_xg
            if w_lg < 0 or w_lg > 0.6:
                continue
            blend_act = w_cb * models["CB"]["act"] + w_xg * models["XGB"]["act"] + w_lg * models["LGB"]["act"]
            blend_pt = w_cb * models["CB"]["pt"] + w_xg * models["XGB"]["pt"] + w_lg * models["LGB"]["pt"]
            blend_srv = w_cb * models["CB"]["srv"] + w_xg * models["XGB"]["srv"] + w_lg * models["LGB"]["srv"]

            blend_act = apply_action_rules(blend_act, next_sn)
            f1a = f1_score(y_act, np.argmax(blend_act, axis=1),
                           labels=list(range(N_ACTION)), average="macro", zero_division=0)
            f1p = f1_score(y_pt, np.argmax(blend_pt, axis=1),
                           labels=list(range(N_POINT)), average="macro", zero_division=0)
            try:
                auc = roc_auc_score(y_srv, blend_srv)
            except:
                auc = 0.5
            ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
            if ov > best_ov:
                best_ov = ov
                best_blend_w = (w_cb, w_xg, w_lg)

    print(f"Best blend: CB={best_blend_w[0]:.2f} XGB={best_blend_w[1]:.2f} LGB={best_blend_w[2]:.2f} OV={best_ov:.4f}")

    # ================================================================
    # 6. THRESHOLD OPTIMIZATION SIMULATION
    # ================================================================
    print("\n" + "=" * 80)
    print("6. PER-CLASS THRESHOLD OPTIMIZATION (on CB OOF)")
    print("=" * 80)

    # ActionId threshold optimization
    act_probs = apply_action_rules(models["CB"]["act"], next_sn)
    base_preds = np.argmax(act_probs, axis=1)
    base_f1 = f1_score(y_act, base_preds, labels=list(range(N_ACTION)), average="macro", zero_division=0)
    print(f"\nactionId base Macro F1 (argmax): {base_f1:.4f}")

    weights_act = np.ones(N_ACTION)
    for cls in range(N_ACTION):
        best_w = 1.0
        best_score = base_f1
        for w in np.arange(0.5, 8.0, 0.25):
            test_w = weights_act.copy()
            test_w[cls] = w
            adj = act_probs * test_w[np.newaxis, :]
            adj /= adj.sum(axis=1, keepdims=True)
            preds = np.argmax(adj, axis=1)
            score = f1_score(y_act, preds, labels=list(range(N_ACTION)), average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best_w = w
        weights_act[cls] = best_w

    adj_act = act_probs * weights_act[np.newaxis, :]
    adj_act /= adj_act.sum(axis=1, keepdims=True)
    opt_preds = np.argmax(adj_act, axis=1)
    opt_f1 = f1_score(y_act, opt_preds, labels=list(range(N_ACTION)), average="macro", zero_division=0)
    print(f"actionId optimized Macro F1: {opt_f1:.4f} (delta: +{opt_f1 - base_f1:.4f})")
    print(f"actionId weights: {weights_act}")

    # Per-class F1 after optimization
    per_class_f1_opt = f1_score(y_act, opt_preds, labels=list(range(N_ACTION)), average=None, zero_division=0)
    per_class_f1_base = f1_score(y_act, base_preds, labels=list(range(N_ACTION)), average=None, zero_division=0)
    print(f"\nactionId per-class F1 change:")
    for c in range(N_ACTION):
        delta = per_class_f1_opt[c] - per_class_f1_base[c]
        if abs(delta) > 0.001:
            print(f"  {c:>2} {ACTION_NAMES[c]:>15}: {per_class_f1_base[c]:.4f} -> {per_class_f1_opt[c]:.4f} ({delta:+.4f})")

    # PointId threshold optimization
    pt_probs = models["CB"]["pt"]
    base_preds_pt = np.argmax(pt_probs, axis=1)
    base_f1_pt = f1_score(y_pt, base_preds_pt, labels=list(range(N_POINT)), average="macro", zero_division=0)
    print(f"\npointId base Macro F1 (argmax): {base_f1_pt:.4f}")

    weights_pt = np.ones(N_POINT)
    for cls in range(N_POINT):
        best_w = 1.0
        best_score = base_f1_pt
        for w in np.arange(0.5, 8.0, 0.25):
            test_w = weights_pt.copy()
            test_w[cls] = w
            adj = pt_probs * test_w[np.newaxis, :]
            adj /= adj.sum(axis=1, keepdims=True)
            preds = np.argmax(adj, axis=1)
            score = f1_score(y_pt, preds, labels=list(range(N_POINT)), average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best_w = w
        weights_pt[cls] = best_w

    adj_pt = pt_probs * weights_pt[np.newaxis, :]
    adj_pt /= adj_pt.sum(axis=1, keepdims=True)
    opt_preds_pt = np.argmax(adj_pt, axis=1)
    opt_f1_pt = f1_score(y_pt, opt_preds_pt, labels=list(range(N_POINT)), average="macro", zero_division=0)
    print(f"pointId optimized Macro F1: {opt_f1_pt:.4f} (delta: +{opt_f1_pt - base_f1_pt:.4f})")
    print(f"pointId weights: {weights_pt}")

    per_class_f1_opt_pt = f1_score(y_pt, opt_preds_pt, labels=list(range(N_POINT)), average=None, zero_division=0)
    per_class_f1_base_pt = f1_score(y_pt, base_preds_pt, labels=list(range(N_POINT)), average=None, zero_division=0)
    print(f"\npointId per-class F1 change:")
    for c in range(N_POINT):
        delta = per_class_f1_opt_pt[c] - per_class_f1_base_pt[c]
        if abs(delta) > 0.001:
            print(f"  {c:>2} {POINT_NAMES[c]:>15}: {per_class_f1_base_pt[c]:.4f} -> {per_class_f1_opt_pt[c]:.4f} ({delta:+.4f})")

    # ================================================================
    # 7. OVERALL OV WITH THRESHOLD OPTIMIZATION
    # ================================================================
    print("\n" + "=" * 80)
    print("7. OVERALL OV COMPARISON")
    print("=" * 80)

    # Best blend with threshold
    w_cb, w_xg, w_lg = best_blend_w
    blend_act = w_cb * models["CB"]["act"] + w_xg * models["XGB"]["act"] + w_lg * models["LGB"]["act"]
    blend_pt = w_cb * models["CB"]["pt"] + w_xg * models["XGB"]["pt"] + w_lg * models["LGB"]["pt"]
    blend_srv = w_cb * models["CB"]["srv"] + w_xg * models["XGB"]["srv"] + w_lg * models["LGB"]["srv"]

    # Without threshold
    blend_act_ruled = apply_action_rules(blend_act, next_sn)
    f1a_base = f1_score(y_act, np.argmax(blend_act_ruled, axis=1),
                        labels=list(range(N_ACTION)), average="macro", zero_division=0)
    f1p_base = f1_score(y_pt, np.argmax(blend_pt, axis=1),
                        labels=list(range(N_POINT)), average="macro", zero_division=0)
    auc_base = roc_auc_score(y_srv, blend_srv)
    ov_base = 0.4 * f1a_base + 0.4 * f1p_base + 0.2 * auc_base

    # With threshold (optimize on blend)
    print("\nOptimizing thresholds on blended predictions...")
    blend_act_ruled2 = apply_action_rules(blend_act, next_sn)

    weights_act_blend = np.ones(N_ACTION)
    f1a_current = f1a_base
    for cls in range(N_ACTION):
        best_w_cls = 1.0
        best_score_cls = f1a_current
        for w in np.arange(0.5, 8.0, 0.25):
            test_w = weights_act_blend.copy()
            test_w[cls] = w
            adj = blend_act_ruled2 * test_w[np.newaxis, :]
            adj /= adj.sum(axis=1, keepdims=True)
            preds = np.argmax(adj, axis=1)
            score = f1_score(y_act, preds, labels=list(range(N_ACTION)), average="macro", zero_division=0)
            if score > best_score_cls:
                best_score_cls = score
                best_w_cls = w
        weights_act_blend[cls] = best_w_cls
        f1a_current = best_score_cls

    adj_blend_act = blend_act_ruled2 * weights_act_blend[np.newaxis, :]
    adj_blend_act /= adj_blend_act.sum(axis=1, keepdims=True)
    f1a_opt = f1_score(y_act, np.argmax(adj_blend_act, axis=1),
                       labels=list(range(N_ACTION)), average="macro", zero_division=0)

    weights_pt_blend = np.ones(N_POINT)
    f1p_current = f1p_base
    for cls in range(N_POINT):
        best_w_cls = 1.0
        best_score_cls = f1p_current
        for w in np.arange(0.5, 8.0, 0.25):
            test_w = weights_pt_blend.copy()
            test_w[cls] = w
            adj = blend_pt * test_w[np.newaxis, :]
            adj /= adj.sum(axis=1, keepdims=True)
            preds = np.argmax(adj, axis=1)
            score = f1_score(y_pt, preds, labels=list(range(N_POINT)), average="macro", zero_division=0)
            if score > best_score_cls:
                best_score_cls = score
                best_w_cls = w
        weights_pt_blend[cls] = best_w_cls
        f1p_current = best_score_cls

    adj_blend_pt = blend_pt * weights_pt_blend[np.newaxis, :]
    adj_blend_pt /= adj_blend_pt.sum(axis=1, keepdims=True)
    f1p_opt = f1_score(y_pt, np.argmax(adj_blend_pt, axis=1),
                       labels=list(range(N_POINT)), average="macro", zero_division=0)

    ov_opt = 0.4 * f1a_opt + 0.4 * f1p_opt + 0.2 * auc_base

    print(f"\n{'Metric':<25} {'Base (argmax)':<15} {'Optimized':<15} {'Delta':<10}")
    print("-" * 65)
    print(f"{'actionId Macro F1':<25} {f1a_base:<15.4f} {f1a_opt:<15.4f} {f1a_opt-f1a_base:+.4f}")
    print(f"{'pointId Macro F1':<25} {f1p_base:<15.4f} {f1p_opt:<15.4f} {f1p_opt-f1p_base:+.4f}")
    print(f"{'serverGetPoint AUC':<25} {auc_base:<15.4f} {auc_base:<15.4f} {0.0:+.4f}")
    print(f"{'Overall OV':<25} {ov_base:<15.4f} {ov_opt:<15.4f} {ov_opt-ov_base:+.4f}")

    print(f"\nactionId class weights: {weights_act_blend}")
    print(f"pointId class weights:  {weights_pt_blend}")


if __name__ == "__main__":
    main()
