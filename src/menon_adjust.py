"""Menon logit adjustment post-processing.

For each task (action / point), at inference compute:
  P_adj(c) ∝ P(c) / prior(c)^τ
where prior(c) is computed from OOF labels (matches training distribution).

Search τ in [0, 1] on OOF for max macro-F1, then apply to test predictions.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SUBMISSION_DIR

ACTION_EVAL = list(range(15))
POINT_EVAL  = list(range(10))


def macro_f1(y, probs, labels):
    return f1_score(y, probs.argmax(axis=1), labels=labels,
                    average="macro", zero_division=0)


def menon_adjust(probs, prior, tau):
    """P_adj ∝ P / prior^τ"""
    eps = 1e-9
    adj = probs / np.power(prior + eps, tau)
    adj /= adj.sum(axis=1, keepdims=True)
    return adj


def search_tau(probs, y_true, labels, prior, name):
    base_f1 = macro_f1(y_true, probs, labels)
    best_t, best_f1 = 0.0, base_f1
    for t in np.arange(0, 1.05, 0.05):
        adj = menon_adjust(probs, prior, t)
        f = macro_f1(y_true, adj, labels)
        if f > best_f1:
            best_t, best_f1 = t, f
    print(f"  {name}: τ={best_t:.2f}  F1={best_f1:.4f}  (vs base={base_f1:.4f}, gain={best_f1-base_f1:+.4f})")
    return best_t, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-sub", type=str,
                     default=os.path.join(SUBMISSION_DIR, "submission_v12_v11srv.csv"))
    ap.add_argument("--output-sub", type=str, default="")
    args = ap.parse_args()

    OOD = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")
    v12_act = np.load(os.path.join(OOD, "v12_oof_act.npy"))
    v12_pt  = np.load(os.path.join(OOD, "v12_oof_pt.npy"))
    v12_mask = np.load(os.path.join(OOD, "v12_oof_mask.npy")).astype(bool)
    y_a = np.load(os.path.join(OOD, "v12_oof_y_act.npy"))
    y_p = np.load(os.path.join(OOD, "v12_oof_y_pt.npy"))

    # Compute prior from OOF labels
    prior_a = np.bincount(y_a[v12_mask], minlength=v12_act.shape[1]).astype(np.float64)
    prior_a /= prior_a.sum()
    prior_p = np.bincount(y_p[v12_mask], minlength=v12_pt.shape[1]).astype(np.float64)
    prior_p /= prior_p.sum()

    print("=== OOF prior ===")
    print(f"  action: {dict(zip(range(15), prior_a[:15].round(4)))}")
    print(f"  point:  {dict(zip(range(10), prior_p[:10].round(4)))}")

    print("\n=== Menon adjustment search on V12 OOF ===")
    tau_a, f1a_adj = search_tau(v12_act[v12_mask], y_a[v12_mask],
                                  ACTION_EVAL, prior_a, "Action")
    tau_p, f1p_adj = search_tau(v12_pt[v12_mask],  y_p[v12_mask],
                                  POINT_EVAL,  prior_p, "Point ")

    if tau_a == 0 and tau_p == 0:
        print("\n  No improvement found. Skipping submission update.")
        return

    # Apply to test predictions
    v12_t_act = np.load(os.path.join(OOD, "v12_test_act.npy"))
    v12_t_pt  = np.load(os.path.join(OOD, "v12_test_pt.npy"))
    v12_t_uid = np.load(os.path.join(OOD, "v12_test_rally_uid.npy"))

    if tau_a > 0:
        v12_t_act = menon_adjust(v12_t_act, prior_a, tau_a)
    if tau_p > 0:
        v12_t_pt  = menon_adjust(v12_t_pt,  prior_p, tau_p)

    pred_act = v12_t_act.argmax(axis=1)
    pred_pt  = v12_t_pt.argmax(axis=1)

    # Pull server from input submission (key-based)
    in_sub = pd.read_csv(args.input_sub)
    srv_lookup = in_sub.set_index("rally_uid")["serverGetPoint"].to_dict()
    pred_srv = np.array([srv_lookup[int(u)] for u in v12_t_uid])

    out_path = args.output_sub or os.path.join(
        SUBMISSION_DIR, f"submission_v12_menon_a{tau_a:.2f}_p{tau_p:.2f}.csv")
    sub = pd.DataFrame({
        "rally_uid": v12_t_uid,
        "actionId": pred_act,
        "pointId": pred_pt,
        "serverGetPoint": pred_srv,
    })
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  actionId dist: {dict(pd.Series(pred_act).value_counts().sort_index().head(10))}")
    print(f"  pointId dist:  {dict(pd.Series(pred_pt).value_counts().sort_index())}")


if __name__ == "__main__":
    main()
