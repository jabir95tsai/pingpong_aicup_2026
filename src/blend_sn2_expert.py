"""SN=2 Expert Hybrid Blender.

Combines:
  - V12 + V11 optblend (current best baseline)
  - SN=2 expert predictions (only for SN=2 rows)

Strategy: for SN=2 rows, blend or fully replace baseline predictions with
expert predictions. For non-SN=2 rows, keep baseline unchanged.

Usage:
  python src/blend_sn2_expert.py --base v12       # blend with V12 baseline
  python src/blend_sn2_expert.py --base v12aug    # blend with V12aug baseline
  python src/blend_sn2_expert.py --mode replace   # fully replace SN=2 rows
  python src/blend_sn2_expert.py --mode blend     # search optimal alpha

Outputs:
  submissions/submission_sn2_expert_<base>_<mode>.csv
  Reports OOF metrics: SN=2 slice, non-SN=2 slice, overall.
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
N_POINT  = 10

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


def optimize_thresholds(probs, y_true, labels, init_cw, n_classes, name=""):
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
    print(f"  [{name}] Temp={best_t:.1f} -> F1={best_f1:.4f}")

    w = np.array([init_cw.get(c, 1.0) for c in range(n_classes)])
    cur_f1 = f1_score(y_true, (probs_t * w).argmax(axis=1), labels=labels,
                       average="macro", zero_division=0)
    for c in range(n_classes):
        best_wc, best_local = w[c], cur_f1
        for wc in np.concatenate([np.arange(0.05, 1.0, 0.1),
                                   np.arange(1.0, 40.0, 1.0)]):
            trial = w.copy(); trial[c] = wc
            f = f1_score(y_true, (probs_t * trial).argmax(axis=1), labels=labels,
                         average="macro", zero_division=0)
            if f > best_local:
                best_local = f; best_wc = wc
        w[c] = best_wc; cur_f1 = best_local
    print(f"  [{name}] Greedy -> F1={cur_f1:.4f}")

    def neg_f1(log_w):
        ww = np.exp(np.clip(log_w, -5, 5))
        return -f1_score(y_true, (probs_t * ww).argmax(axis=1), labels=labels,
                         average="macro", zero_division=0)
    try:
        res = minimize(neg_f1, np.log(np.clip(w, 0.01, 100)),
                        method="Powell", options={"maxiter": 150})
        if -res.fun > cur_f1:
            w = np.exp(np.clip(res.x, -5, 5))
            cur_f1 = -res.fun
            print(f"  [{name}] Scipy -> F1={cur_f1:.4f} (improved)")
    except Exception as e:
        print(f"  [{name}] Scipy failed: {e}")
    return best_t, w, cur_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",  default="v12",
                    help="Baseline GBM tag for non-SN=2 rows (v12 / v12aug / v13)")
    ap.add_argument("--mode",  default="blend",
                    choices=["replace", "blend"],
                    help="replace: SN=2 expert fully overrides baseline. "
                         "blend: search optimal alpha between expert and baseline+V11.")
    ap.add_argument("--expert-tag", default="sn2_expert")
    ap.add_argument("--no-v11",  action="store_true")
    ap.add_argument("--out",     default="")
    args = ap.parse_args()

    use_v11 = not args.no_v11
    oof_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")

    # ── Load baseline GBM OOF ─────────────────────────────────────────────────
    print(f"\n=== Loading {args.base} OOF ===")
    base_act  = np.load(os.path.join(oof_dir, f"{args.base}_oof_act.npy"))
    base_pt   = np.load(os.path.join(oof_dir, f"{args.base}_oof_pt.npy"))
    base_srv  = np.load(os.path.join(oof_dir, f"{args.base}_oof_srv.npy"))
    base_mask = np.load(os.path.join(oof_dir, f"{args.base}_oof_mask.npy")).astype(bool)
    y_a = np.load(os.path.join(oof_dir, f"{args.base}_oof_y_act.npy"))
    y_p = np.load(os.path.join(oof_dir, f"{args.base}_oof_y_pt.npy"))
    y_s = np.load(os.path.join(oof_dir, f"{args.base}_oof_y_srv.npy"))
    nsn = np.load(os.path.join(oof_dir, f"{args.base}_oof_nsn.npy"))

    # ── Load V11 OOF ──────────────────────────────────────────────────────────
    if use_v11:
        v11_act  = np.load(os.path.join(oof_dir, "v11_oof_act.npy"))
        v11_pt   = np.load(os.path.join(oof_dir, "v11_oof_pt.npy"))
        v11_srv  = np.load(os.path.join(oof_dir, "v11_oof_srv.npy"))
        v11_mask = np.load(os.path.join(oof_dir, "v11_oof_mask.npy")).astype(bool)
        if v11_act.shape[1] < N_ACTION:
            pad = np.zeros((len(v11_act), N_ACTION), dtype=v11_act.dtype)
            pad[:, :v11_act.shape[1]] = v11_act
            v11_act = pad

    # ── Load SN=2 expert OOF ──────────────────────────────────────────────────
    print(f"=== Loading {args.expert_tag} OOF ===")
    exp_act  = np.load(os.path.join(oof_dir, f"{args.expert_tag}_oof_act.npy"))
    exp_pt   = np.load(os.path.join(oof_dir, f"{args.expert_tag}_oof_pt.npy"))
    exp_srv  = np.load(os.path.join(oof_dir, f"{args.expert_tag}_oof_srv.npy"))
    exp_mask = np.load(os.path.join(oof_dir, f"{args.expert_tag}_oof_mask.npy")).astype(bool)

    sn2_mask = (nsn == 2)
    common_mask = base_mask & exp_mask & sn2_mask
    if use_v11:
        common_mask = common_mask & v11_mask

    print(f"\n  Total SN=2 OOF rows: {sn2_mask.sum()}")
    print(f"  After mask intersection: {common_mask.sum()}")

    # ── Compare standalone metrics on SN=2 ────────────────────────────────────
    print("\n=== Standalone SN=2 metrics ===")
    for tag, act, pt, srv in [
        ("baseline", base_act, base_pt, base_srv),
        ("expert  ", exp_act, exp_pt, exp_srv),
    ]:
        m = common_mask
        f1a = macro_f1(y_a[m], act[m], ACTION_EVAL)
        f1p = macro_f1(y_p[m], pt[m],  POINT_EVAL)
        if y_s[m].std() < 1e-9: au = 0.5
        else: au = roc_auc_score(y_s[m], srv[m])
        print(f"  {tag}  F1_a={f1a:.4f}  F1_p={f1p:.4f}  AUC={au:.4f}  "
              f"OV={0.4*f1a+0.4*f1p+0.2*au:.4f}")

    # ── Build baseline + V11 blend on SN=2 (current best path) ────────────────
    if use_v11:
        # Use the same alphas as in final_blend_optimized: act=0.6, pt=0.55, srv=0.95
        # These were tuned globally; they should still be reasonable for SN=2.
        a_a, a_p, a_s = 0.60, 0.55, 0.95
        bv_act = a_a * base_act + (1-a_a) * v11_act
        bv_pt  = a_p * base_pt  + (1-a_p) * v11_pt
        bv_srv = a_s * base_srv + (1-a_s) * v11_srv
    else:
        bv_act, bv_pt, bv_srv = base_act, base_pt, base_srv

    # ── For SN=2 rows: blend or replace ───────────────────────────────────────
    print(f"\n=== SN=2 strategy: {args.mode} ===")
    if args.mode == "replace":
        sn2_act_used = exp_act
        sn2_pt_used  = exp_pt
        sn2_srv_used = exp_srv
        chosen_alpha = (1.0, 1.0, 1.0)  # 100% expert
    else:
        # Search optimal alpha (expert weight) for action and point on SN=2
        m = common_mask
        best_a_a, best_f1_a = 1.0, -1.0
        for a in np.arange(0, 1.05, 0.05):
            bl = a * exp_act[m] + (1-a) * bv_act[m]
            f = macro_f1(y_a[m], bl, ACTION_EVAL)
            if f > best_f1_a: best_f1_a, best_a_a = f, a
        best_a_p, best_f1_p = 1.0, -1.0
        for a in np.arange(0, 1.05, 0.05):
            bl = a * exp_pt[m] + (1-a) * bv_pt[m]
            f = macro_f1(y_p[m], bl, POINT_EVAL)
            if f > best_f1_p: best_f1_p, best_a_p = f, a
        if y_s[m].std() < 1e-9:
            best_a_s, best_au = 0.0, 0.5
        else:
            best_a_s, best_au = 1.0, -1.0
            for a in np.arange(0, 1.05, 0.05):
                bl = a * exp_srv[m] + (1-a) * bv_srv[m]
                au = roc_auc_score(y_s[m], bl)
                if au > best_au: best_au, best_a_s = au, a

        print(f"  SN=2 expert weight: act α={best_a_a:.2f} (F1={best_f1_a:.4f})  "
              f"pt α={best_a_p:.2f} (F1={best_f1_p:.4f})  "
              f"srv α={best_a_s:.2f} (AUC={best_au:.4f})")

        sn2_act_used = best_a_a * exp_act + (1-best_a_a) * bv_act
        sn2_pt_used  = best_a_p * exp_pt  + (1-best_a_p) * bv_pt
        sn2_srv_used = best_a_s * exp_srv + (1-best_a_s) * bv_srv
        chosen_alpha = (best_a_a, best_a_p, best_a_s)

    # ── Hybrid: use expert blend on SN=2 rows, baseline+V11 elsewhere ─────────
    hybrid_act = bv_act.copy()
    hybrid_pt  = bv_pt.copy()
    hybrid_srv = bv_srv.copy()
    hybrid_act[sn2_mask] = sn2_act_used[sn2_mask]
    hybrid_pt [sn2_mask] = sn2_pt_used [sn2_mask]
    hybrid_srv[sn2_mask] = sn2_srv_used[sn2_mask]

    # ── Threshold optimisation on hybrid ──────────────────────────────────────
    full_mask = base_mask & (v11_mask if use_v11 else np.ones_like(base_mask, bool))
    full_mask = full_mask & ~(sn2_mask & ~common_mask)  # exclude SN=2 rows where expert missing

    print("\n=== Optimize hybrid action ===")
    t_a, w_a, f1_a_opt = optimize_thresholds(
        hybrid_act[full_mask], y_a[full_mask], ACTION_EVAL, ACTION_CW, N_ACTION, "Action")
    print("\n=== Optimize hybrid point ===")
    t_p, w_p, f1_p_opt = optimize_thresholds(
        hybrid_pt[full_mask], y_p[full_mask], POINT_EVAL, POINT_CW, N_POINT, "Point")

    auc_h = roc_auc_score(y_s[full_mask], hybrid_srv[full_mask])
    ov_h  = 0.4*f1_a_opt + 0.4*f1_p_opt + 0.2*auc_h
    print(f"\nHYBRID OV={ov_h:.4f}  (F1_a={f1_a_opt:.4f}  F1_p={f1_p_opt:.4f}  AUC={auc_h:.4f})")
    print(f"  vs V12+V11 optblend baseline OOF=0.3734")

    # ── Per-slice breakdown ───────────────────────────────────────────────────
    hybrid_act_t = hybrid_act ** (1.0/t_a)
    hybrid_act_t /= hybrid_act_t.sum(axis=1, keepdims=True)
    hybrid_act_w = hybrid_act_t * w_a
    hybrid_pt_t  = hybrid_pt ** (1.0/t_p)
    hybrid_pt_t /= hybrid_pt_t.sum(axis=1, keepdims=True)
    hybrid_pt_w  = hybrid_pt_t * w_p

    print(f"\n{'Slice':<10} {'n':>6} {'F1_a':>7} {'F1_p':>7} {'AUC':>7} {'OV':>7}")
    for sname, smask in [("SN=2",   nsn==2),
                          ("SN=3-4", (nsn>=3)&(nsn<=4)),
                          ("SN=5-8", (nsn>=5)&(nsn<=8)),
                          ("SN=9-12",(nsn>=9)&(nsn<=12)),
                          ("SN>=13", nsn>=13)]:
        m = full_mask & smask
        if m.sum() < 5: continue
        fa = f1_score(y_a[m], hybrid_act_w[m].argmax(1),
                      labels=ACTION_EVAL, average="macro", zero_division=0)
        fp = f1_score(y_p[m], hybrid_pt_w[m].argmax(1),
                      labels=POINT_EVAL, average="macro", zero_division=0)
        if y_s[m].std() < 1e-9: au = 0.5
        else: au = roc_auc_score(y_s[m], hybrid_srv[m])
        print(f"{sname:<10} {m.sum():>6} {fa:>7.4f} {fp:>7.4f} {au:>7.4f} "
              f"{0.4*fa+0.4*fp+0.2*au:>7.4f}")

    # ── Build test submission ────────────────────────────────────────────────
    print("\n=== Building test submission ===")
    base_t_act = np.load(os.path.join(oof_dir, f"{args.base}_test_act.npy"))
    base_t_pt  = np.load(os.path.join(oof_dir, f"{args.base}_test_pt.npy"))
    base_t_srv = np.load(os.path.join(oof_dir, f"{args.base}_test_srv.npy"))
    base_t_uid = np.load(os.path.join(oof_dir, f"{args.base}_test_rally_uid.npy"))

    if use_v11:
        v11_t_act = np.load(os.path.join(oof_dir, "v11_test_act.npy"))
        v11_t_pt  = np.load(os.path.join(oof_dir, "v11_test_pt.npy"))
        v11_t_srv = np.load(os.path.join(oof_dir, "v11_test_srv.npy"))
        v11_t_uid = pd.read_csv(os.path.join(SUBMISSION_DIR,
                                              "submission_v11_transformer.csv"))["rally_uid"].values
        if v11_t_act.shape[1] < N_ACTION:
            pad = np.zeros((len(v11_t_act), N_ACTION), dtype=v11_t_act.dtype)
            pad[:, :v11_t_act.shape[1]] = v11_t_act
            v11_t_act = pad
        # Align V11 to base uid order
        v11_uid_map = {int(u): i for i, u in enumerate(v11_t_uid)}
        align = np.array([v11_uid_map[int(u)] for u in base_t_uid])
        v11_t_act = v11_t_act[align]
        v11_t_pt  = v11_t_pt[align]
        v11_t_srv = v11_t_srv[align]
        bv_t_act = a_a * base_t_act + (1-a_a) * v11_t_act
        bv_t_pt  = a_p * base_t_pt  + (1-a_p) * v11_t_pt
        bv_t_srv = a_s * base_t_srv + (1-a_s) * v11_t_srv
    else:
        bv_t_act, bv_t_pt, bv_t_srv = base_t_act, base_t_pt, base_t_srv

    # Load expert test predictions (full-length, zero for non-SN=2)
    exp_t_act  = np.load(os.path.join(oof_dir, f"{args.expert_tag}_test_act.npy"))
    exp_t_pt   = np.load(os.path.join(oof_dir, f"{args.expert_tag}_test_pt.npy"))
    exp_t_srv  = np.load(os.path.join(oof_dir, f"{args.expert_tag}_test_srv.npy"))
    exp_t_uid  = np.load(os.path.join(oof_dir, f"{args.expert_tag}_test_rally_uid.npy"))
    sn2_test_m = np.load(os.path.join(oof_dir, f"{args.expert_tag}_test_sn2_mask.npy"))

    # Align expert test to base uid order
    exp_uid_map = {int(u): i for i, u in enumerate(exp_t_uid)}
    align_e = np.array([exp_uid_map[int(u)] for u in base_t_uid])
    exp_t_act = exp_t_act[align_e]
    exp_t_pt  = exp_t_pt[align_e]
    exp_t_srv = exp_t_srv[align_e]
    sn2_test_m_aligned = sn2_test_m[align_e]

    # Apply SN=2 strategy on test
    if args.mode == "replace":
        sn2_t_act_used = exp_t_act
        sn2_t_pt_used  = exp_t_pt
        sn2_t_srv_used = exp_t_srv
    else:
        a_a_e, a_p_e, a_s_e = chosen_alpha
        sn2_t_act_used = a_a_e * exp_t_act + (1-a_a_e) * bv_t_act
        sn2_t_pt_used  = a_p_e * exp_t_pt  + (1-a_p_e) * bv_t_pt
        sn2_t_srv_used = a_s_e * exp_t_srv + (1-a_s_e) * bv_t_srv

    hybrid_t_act = bv_t_act.copy()
    hybrid_t_pt  = bv_t_pt.copy()
    hybrid_t_srv = bv_t_srv.copy()
    hybrid_t_act[sn2_test_m_aligned] = sn2_t_act_used[sn2_test_m_aligned]
    hybrid_t_pt [sn2_test_m_aligned] = sn2_t_pt_used [sn2_test_m_aligned]
    hybrid_t_srv[sn2_test_m_aligned] = sn2_t_srv_used[sn2_test_m_aligned]

    # Apply optimised thresholds
    h_act_t = hybrid_t_act ** (1.0/t_a)
    h_act_t /= h_act_t.sum(axis=1, keepdims=True)
    pred_act = (h_act_t * w_a).argmax(axis=1)
    h_pt_t  = hybrid_t_pt ** (1.0/t_p)
    h_pt_t /= h_pt_t.sum(axis=1, keepdims=True)
    pred_pt  = (h_pt_t * w_p).argmax(axis=1)
    pred_srv = hybrid_t_srv

    out_name = args.out or f"submission_sn2_expert_{args.base}_{args.mode}"
    out_path = os.path.join(SUBMISSION_DIR, f"{out_name}.csv")
    sub = pd.DataFrame({
        "rally_uid":      base_t_uid,
        "actionId":       pred_act,
        "pointId":        pred_pt,
        "serverGetPoint": pred_srv,
    })
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  actionId top: {dict(pd.Series(pred_act).value_counts().head(5))}")
    print(f"  pointId dist: {dict(pd.Series(pred_pt).value_counts().sort_index())}")
    print(f"  srv mean={pred_srv.mean():.4f}  std={pred_srv.std():.4f}")
    print(f"\nHYBRID OOF OV: {ov_h:.4f}  (V12+V11 optblend baseline: 0.3734)")


if __name__ == "__main__":
    main()
