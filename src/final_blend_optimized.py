"""Final tagged blend with per-task threshold optimization on OOF.

For each task (action / point):
  1. Search blend alpha
  2. Apply temperature scaling
  3. Apply class-weight greedy + scipy optimization
For server: continuous probs (best by AUC)

Default output: ``submission_v12_v11_optblend.csv``.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SUBMISSION_DIR

ACTION_EVAL = list(range(15))
POINT_EVAL  = list(range(10))
N_ACTION = 19
N_POINT = 10

ACTION_CW = {
    0: 1.5, 1: 0.6, 2: 0.9, 3: 1.5, 4: 1.2, 5: 1.0,
    6: 0.8, 7: 1.8, 8: 14.0, 9: 8.0, 10: 0.6, 11: 1.2,
    12: 0.9, 13: 0.7, 14: 10.0,
    15: 0.01, 16: 0.01, 17: 0.01, 18: 0.01,
}
POINT_CW = {
    0: 0.5, 1: 12.0, 2: 2.5, 3: 22.0, 4: 2.0,
    5: 0.9, 6: 1.5, 7: 0.8, 8: 0.7, 9: 0.6,
}


def macro_f1(y, probs, labels):
    return f1_score(y, probs.argmax(axis=1), labels=labels,
                    average="macro", zero_division=0)


def load_tagged_array(base_dir, tag, suffix):
    return np.load(os.path.join(base_dir, f"{tag}_{suffix}.npy"))


def optimize_thresholds(probs, y_true, labels, init_cw, n_classes):
    best_t, best_f1 = 1.0, -1.0
    for t in np.arange(0.2, 3.5, 0.1):
        scaled = probs ** (1.0 / t)
        scaled /= scaled.sum(axis=1, keepdims=True)
        s = f1_score(y_true, scaled.argmax(axis=1), labels=labels,
                     average="macro", zero_division=0)
        if s > best_f1:
            best_f1 = s; best_t = t
    probs_t = probs ** (1.0 / best_t)
    probs_t /= probs_t.sum(axis=1, keepdims=True)
    print(f"    Temp={best_t:.1f} -> F1={best_f1:.4f}")

    w = np.array([init_cw.get(c, 1.0) for c in range(n_classes)])
    cur_f1 = f1_score(y_true, (probs_t * w).argmax(axis=1), labels=labels,
                       average="macro", zero_division=0)
    for c in range(n_classes):
        best_wc, best_local = w[c], cur_f1
        for wc in np.concatenate([np.arange(0.05, 1.0, 0.1), np.arange(1.0, 40.0, 1.0)]):
            trial = w.copy(); trial[c] = wc
            f = f1_score(y_true, (probs_t * trial).argmax(axis=1), labels=labels,
                         average="macro", zero_division=0)
            if f > best_local:
                best_local = f; best_wc = wc
        w[c] = best_wc; cur_f1 = best_local
    print(f"    Greedy -> F1={cur_f1:.4f}")

    def neg_f1(log_w):
        ww = np.exp(np.clip(log_w, -5, 5))
        return -f1_score(y_true, (probs_t * ww).argmax(axis=1), labels=labels,
                         average="macro", zero_division=0)
    try:
        res = minimize(neg_f1, np.log(np.clip(w, 0.01, 100)),
                        method="Powell", options={"maxiter": 100})
        if -res.fun > cur_f1:
            w = np.exp(np.clip(res.x, -5, 5))
            cur_f1 = -res.fun
            print(f"    Scipy -> F1={cur_f1:.4f} (improved)")
    except Exception as e:
        print(f"    Scipy failed: {e}")
    return best_t, w, cur_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default=os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions"))
    ap.add_argument("--primary-tag", default="v12")
    ap.add_argument("--aux-tag", default="v11")
    ap.add_argument("--aux-submission", default=os.path.join(SUBMISSION_DIR, "submission_v11_transformer.csv"))
    ap.add_argument("--output-tag", default=None)
    ap.add_argument("--output-tag-auxsrv", default=None)
    args = ap.parse_args()

    OOD = args.oof_dir
    primary_tag = args.primary_tag
    aux_tag = args.aux_tag
    aux_submission = args.aux_submission
    output_tag = args.output_tag or f"{primary_tag}_{aux_tag}_optblend"
    output_tag_auxsrv = args.output_tag_auxsrv or f"{primary_tag}_{aux_tag}_optblend_{aux_tag}srv"

    v12_act = load_tagged_array(OOD, primary_tag, "oof_act")
    v12_pt  = load_tagged_array(OOD, primary_tag, "oof_pt")
    v12_srv = load_tagged_array(OOD, primary_tag, "oof_srv")
    v11_act = load_tagged_array(OOD, aux_tag, "oof_act")
    v11_pt  = load_tagged_array(OOD, aux_tag, "oof_pt")
    v11_srv = load_tagged_array(OOD, aux_tag, "oof_srv")
    y_a = load_tagged_array(OOD, primary_tag, "oof_y_act")
    y_p = load_tagged_array(OOD, primary_tag, "oof_y_pt")
    y_s = load_tagged_array(OOD, primary_tag, "oof_y_srv")
    nsn = load_tagged_array(OOD, primary_tag, "oof_nsn")

    # V11 has 15-class action; pad to 19
    if v11_act.shape[1] != v12_act.shape[1]:
        pad = np.zeros((len(v11_act), v12_act.shape[1]), dtype=v11_act.dtype)
        pad[:, :v11_act.shape[1]] = v11_act
        v11_act = pad

    # Search per-task alpha (blend before threshold opt)
    print("=== Search alpha (then optimize) ===")
    best_a_act, best_f1_a = 1.0, -1.0
    for a in np.arange(0, 1.05, 0.05):
        blend = a * v12_act + (1 - a) * v11_act
        f = macro_f1(y_a, blend, ACTION_EVAL)
        if f > best_f1_a:
            best_f1_a, best_a_act = f, a
    print(f"  Action: best alpha_v12={best_a_act:.2f}  F1={best_f1_a:.4f}")

    best_a_pt, best_f1_p = 1.0, -1.0
    for a in np.arange(0, 1.05, 0.05):
        blend = a * v12_pt + (1 - a) * v11_pt
        f = macro_f1(y_p, blend, POINT_EVAL)
        if f > best_f1_p:
            best_f1_p, best_a_pt = f, a
    print(f"  Point : best alpha_v12={best_a_pt:.2f}  F1={best_f1_p:.4f}")

    best_a_srv, best_auc = 1.0, -1.0
    for a in np.arange(0, 1.05, 0.05):
        blend = a * v12_srv + (1 - a) * v11_srv
        auc = roc_auc_score(y_s, blend)
        if auc > best_auc:
            best_auc, best_a_srv = auc, a
    print(f"  Server: best alpha_v12={best_a_srv:.2f}  AUC={best_auc:.4f}")

    # Apply blend + optimize thresholds
    blend_act = best_a_act * v12_act + (1 - best_a_act) * v11_act
    blend_pt  = best_a_pt  * v12_pt  + (1 - best_a_pt)  * v11_pt
    blend_srv = best_a_srv * v12_srv + (1 - best_a_srv) * v11_srv

    print("\n=== Optimize blended action ===")
    t_a, w_a, f1_a_opt = optimize_thresholds(blend_act, y_a, ACTION_EVAL, ACTION_CW, N_ACTION)
    print("\n=== Optimize blended point ===")
    t_p, w_p, f1_p_opt = optimize_thresholds(blend_pt, y_p, POINT_EVAL, POINT_CW, N_POINT)

    auc_blend = roc_auc_score(y_s, blend_srv)
    ov_opt = 0.4 * f1_a_opt + 0.4 * f1_p_opt + 0.2 * auc_blend
    print(f"\nFINAL OPTIMIZED BLEND OV={ov_opt:.4f}  "
          f"(F1_a={f1_a_opt:.4f}  F1_p={f1_p_opt:.4f}  AUC={auc_blend:.4f})")

    # Apply same recipe to test
    v12_t_act = load_tagged_array(OOD, primary_tag, "test_act")
    v12_t_pt  = load_tagged_array(OOD, primary_tag, "test_pt")
    v12_t_srv = load_tagged_array(OOD, primary_tag, "test_srv")
    v12_t_uid = load_tagged_array(OOD, primary_tag, "test_rally_uid")

    v11_sub = pd.read_csv(aux_submission)
    v11_t_act = load_tagged_array(OOD, aux_tag, "test_act")
    v11_t_pt  = load_tagged_array(OOD, aux_tag, "test_pt")
    v11_t_srv = load_tagged_array(OOD, aux_tag, "test_srv")

    # align V11 to V12 rally order
    v11_uid = v11_sub["rally_uid"].values
    if len(v11_uid) != len(v11_t_act):
        raise ValueError("V11 sub/npy length mismatch")
    uid_to_i = {int(u): i for i, u in enumerate(v11_uid)}
    align = np.array([uid_to_i[int(u)] for u in v12_t_uid])
    if v11_t_act.shape[1] != v12_t_act.shape[1]:
        pad = np.zeros((len(v11_t_act), v12_t_act.shape[1]), dtype=v11_t_act.dtype)
        pad[:, :v11_t_act.shape[1]] = v11_t_act
        v11_t_act = pad
    v11_t_act = v11_t_act[align]
    v11_t_pt  = v11_t_pt[align]
    v11_t_srv = v11_t_srv[align]

    test_act = best_a_act * v12_t_act + (1 - best_a_act) * v11_t_act
    test_pt  = best_a_pt  * v12_t_pt  + (1 - best_a_pt)  * v11_t_pt
    test_srv = best_a_srv * v12_t_srv + (1 - best_a_srv) * v11_t_srv

    # Apply temperature + weight to test
    test_act_t = test_act ** (1.0 / t_a)
    test_act_t /= test_act_t.sum(axis=1, keepdims=True)
    test_act_w = test_act_t * w_a[np.newaxis, :]
    pred_act = test_act_w.argmax(axis=1)

    test_pt_t = test_pt ** (1.0 / t_p)
    test_pt_t /= test_pt_t.sum(axis=1, keepdims=True)
    test_pt_w = test_pt_t * w_p[np.newaxis, :]
    pred_pt = test_pt_w.argmax(axis=1)

    pred_srv = test_srv  # continuous

    sub = pd.DataFrame({
        "rally_uid": v12_t_uid,
        "actionId": pred_act,
        "pointId": pred_pt,
        "serverGetPoint": pred_srv,
    })
    out = os.path.join(SUBMISSION_DIR, f"submission_{output_tag}.csv")
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"  actionId dist top: {dict(pd.Series(pred_act).value_counts().sort_values(ascending=False).head(5))}")
    print(f"  pointId dist:      {dict(pd.Series(pred_pt).value_counts().sort_index())}")
    print(f"  srv mean={pred_srv.mean():.4f}  std={pred_srv.std():.4f}")

    # Also save with V11 cont srv (proven LB winner pattern)
    sub2 = sub.copy()
    sub2["serverGetPoint"] = v11_t_srv
    out2 = os.path.join(SUBMISSION_DIR, f"submission_{output_tag_auxsrv}.csv")
    sub2.to_csv(out2, index=False)
    print(f"Saved (V11 srv variant): {out2}")


if __name__ == "__main__":
    main()
