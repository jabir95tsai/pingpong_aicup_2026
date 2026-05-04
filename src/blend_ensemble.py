"""Generic ensemble blend + threshold optimization.

Usage:
  python src/blend_ensemble.py --v1 v12aug --v2 v13                       # + V11 (default aux)
  python src/blend_ensemble.py --v1 v12aug --aux-tag v11plus               # use V11+ as aux
  python src/blend_ensemble.py --v1 v14 --no-aux                           # solo (no transformer)

Loads OOF npys from oof_predictions/<name>_oof_*.npy
Searches per-task alpha, then runs temperature + greedy + scipy optimization.
Saves submission_<v1>_<v2>_<aux>_optblend.csv by default.

NOTE: --v11 / --no-v11 flags are deprecated aliases for --aux-tag / --no-aux.
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


def load_oof(tag, n_act=N_ACTION, n_pt=N_POINT):
    """Load OOF npys for <tag>. Returns (act, pt, srv, y_act, y_pt, y_srv, mask, nsn).
    y_* and nsn are None if not found (e.g. for v11 which uses v12 labels).
    """
    oof_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")
    act  = np.load(os.path.join(oof_dir, f"{tag}_oof_act.npy"))
    pt   = np.load(os.path.join(oof_dir, f"{tag}_oof_pt.npy"))
    srv  = np.load(os.path.join(oof_dir, f"{tag}_oof_srv.npy"))
    mask = np.load(os.path.join(oof_dir, f"{tag}_oof_mask.npy")).astype(bool)

    def _maybe(fname):
        p = os.path.join(oof_dir, fname)
        return np.load(p) if os.path.exists(p) else None

    y_a = _maybe(f"{tag}_oof_y_act.npy")
    y_p = _maybe(f"{tag}_oof_y_pt.npy")
    y_s = _maybe(f"{tag}_oof_y_srv.npy")
    nsn = _maybe(f"{tag}_oof_nsn.npy")

    # Pad action dim if needed
    if act.shape[1] < n_act:
        pad = np.zeros((len(act), n_act), dtype=act.dtype)
        pad[:, :act.shape[1]] = act
        act = pad
    return act, pt, srv, y_a, y_p, y_s, mask, nsn


def load_test(tag, n_act=N_ACTION, n_pt=N_POINT):
    oof_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")
    act = np.load(os.path.join(oof_dir, f"{tag}_test_act.npy"))
    pt  = np.load(os.path.join(oof_dir, f"{tag}_test_pt.npy"))
    srv = np.load(os.path.join(oof_dir, f"{tag}_test_srv.npy"))
    uid_path = os.path.join(oof_dir, f"{tag}_test_rally_uid.npy")
    if os.path.exists(uid_path):
        uid = np.load(uid_path)
    else:
        # Fall back to submission CSV rally_uid (e.g. v11 which has no uid npy)
        sub_csv = os.path.join(SUBMISSION_DIR, f"submission_{tag}_transformer.csv")
        if not os.path.exists(sub_csv):
            sub_csv = os.path.join(SUBMISSION_DIR, f"submission_{tag}.csv")
        uid = pd.read_csv(sub_csv)["rally_uid"].values
    if act.shape[1] < n_act:
        pad = np.zeros((len(act), n_act), dtype=act.dtype)
        pad[:, :act.shape[1]] = act
        act = pad
    return act, pt, srv, uid


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
    ap.add_argument("--v1", required=True,
                    help="Primary GBM tag (e.g. v12aug, v13)")
    ap.add_argument("--v2", default="",
                    help="Optional second GBM tag (blended with v1 first)")
    ap.add_argument("--v11", action="store_true", default=True,
                    help="[DEPRECATED] Include aux model in ensemble (default: True)")
    ap.add_argument("--no-v11", action="store_true",
                    help="[DEPRECATED] alias of --no-aux")
    ap.add_argument("--no-aux", action="store_true",
                    help="Exclude aux transformer model from ensemble")
    ap.add_argument("--aux-tag", default="v11",
                    help="Tag of aux transformer to blend (default: v11). "
                         "Use 'v11plus' to blend V11+ instead.")
    ap.add_argument("--out", default="",
                    help="Output csv basename (auto if empty). Trailing .csv is stripped.")
    args = ap.parse_args()

    use_aux = args.v11 and not (args.no_v11 or args.no_aux)
    aux_tag = args.aux_tag
    # Backwards-compat alias
    use_v11 = use_aux
    oof_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")

    # ── Load primary GBM ─────────────────────────────────────────────────────
    print(f"\n=== Loading OOF: {args.v1} ===")
    v1_act, v1_pt, v1_srv, y_a, y_p, y_s, v1_mask, nsn = load_oof(args.v1)
    if y_a is None:
        raise ValueError(f"Primary tag {args.v1} must have _oof_y_*.npy files")
    base_act, base_pt, base_srv = v1_act, v1_pt, v1_srv
    base_mask = v1_mask

    # ── Optional second GBM blend ─────────────────────────────────────────────
    if args.v2:
        print(f"=== Loading OOF: {args.v2} ===")
        v2_act, v2_pt, v2_srv, _, _, _, v2_mask, _ = load_oof(args.v2)
        common_mask = v1_mask & v2_mask
        print(f"  {args.v1} mask={v1_mask.sum()}  {args.v2} mask={v2_mask.sum()}  "
              f"common={common_mask.sum()}")
        # Search blend alpha between v1 and v2
        best_a12, best_f1_a12 = 1.0, -1.0
        for a in np.arange(0, 1.05, 0.1):
            bl = a * v1_act[common_mask] + (1-a) * v2_act[common_mask]
            f  = macro_f1(y_a[common_mask], bl, ACTION_EVAL)
            if f > best_f1_a12: best_f1_a12, best_a12 = f, a
        best_b12, best_f1_p12 = 1.0, -1.0
        for a in np.arange(0, 1.05, 0.1):
            bl = a * v1_pt[common_mask] + (1-a) * v2_pt[common_mask]
            f  = macro_f1(y_p[common_mask], bl, POINT_EVAL)
            if f > best_f1_p12: best_f1_p12, best_b12 = f, a
        print(f"  V1/V2 blend: act α={best_a12:.2f} F1={best_f1_a12:.4f}  "
              f"pt α={best_b12:.2f} F1={best_f1_p12:.4f}")
        base_act  = best_a12 * v1_act  + (1-best_a12) * v2_act
        base_pt   = best_b12 * v1_pt   + (1-best_b12) * v2_pt
        base_srv  = 0.5 * v1_srv + 0.5 * v2_srv
        base_mask = common_mask

    # ── Aux (transformer) blend ──────────────────────────────────────────────
    if use_aux:
        print(f"=== Loading OOF: {aux_tag} ===")
        v11_act, v11_pt, v11_srv, _, _, _, v11_mask, _ = load_oof(aux_tag)
        common = base_mask & v11_mask
        print(f"  common={common.sum()}")
    else:
        common = base_mask

    # Search per-task alpha for aux blend
    if use_aux:
        best_a_act, best_f1_a = 1.0, -1.0
        for a in np.arange(0, 1.05, 0.05):
            bl = a * base_act[common] + (1-a) * v11_act[common]
            f  = macro_f1(y_a[common], bl, ACTION_EVAL)
            if f > best_f1_a: best_f1_a, best_a_act = f, a
        best_a_pt, best_f1_p = 1.0, -1.0
        for a in np.arange(0, 1.05, 0.05):
            bl = a * base_pt[common] + (1-a) * v11_pt[common]
            f  = macro_f1(y_p[common], bl, POINT_EVAL)
            if f > best_f1_p: best_f1_p, best_a_pt = f, a
        best_a_srv, best_auc = 1.0, -1.0
        for a in np.arange(0, 1.05, 0.05):
            bl = a * base_srv[common] + (1-a) * v11_srv[common]
            auc = roc_auc_score(y_s[common], bl)
            if auc > best_auc: best_auc, best_a_srv = auc, a
        print(f"\nBlend alphas (GBM weight):")
        print(f"  act  α={best_a_act:.2f}  raw_F1={best_f1_a:.4f}")
        print(f"  pt   α={best_a_pt:.2f}  raw_F1={best_f1_p:.4f}")
        print(f"  srv  α={best_a_srv:.2f}  AUC={best_auc:.4f}")
        blend_act = best_a_act * base_act + (1-best_a_act) * v11_act
        blend_pt  = best_a_pt  * base_pt  + (1-best_a_pt)  * v11_pt
        blend_srv = best_a_srv * base_srv + (1-best_a_srv)  * v11_srv
    else:
        blend_act, blend_pt, blend_srv = base_act, base_pt, base_srv
        best_a_act = best_a_pt = best_a_srv = 1.0
        best_auc = roc_auc_score(y_s[common], blend_srv[common])

    # ── Threshold optimization ─────────────────────────────────────────────────
    print("\n=== Optimize blended action ===")
    t_a, w_a, f1_a_opt = optimize_thresholds(
        blend_act[common], y_a[common], ACTION_EVAL, ACTION_CW, N_ACTION, "Action")
    print("\n=== Optimize blended point ===")
    t_p, w_p, f1_p_opt = optimize_thresholds(
        blend_pt[common], y_p[common], POINT_EVAL, POINT_CW, N_POINT, "Point")

    auc_blend = roc_auc_score(y_s[common], blend_srv[common])
    ov_opt = 0.4 * f1_a_opt + 0.4 * f1_p_opt + 0.2 * auc_blend
    print(f"\nFINAL BLEND OV={ov_opt:.4f}  "
          f"(F1_a={f1_a_opt:.4f}  F1_p={f1_p_opt:.4f}  AUC={auc_blend:.4f})")

    # ── Per-SN slice ──────────────────────────────────────────────────────────
    if nsn is not None and nsn.sum() > 0:
        print(f"\n{'Slice':<10} {'n':>6} {'F1_a':>7} {'F1_p':>7} {'AUC':>7} {'OV':>7}")
        blend_act_t = blend_act ** (1.0 / t_a)
        blend_act_t /= blend_act_t.sum(axis=1, keepdims=True)
        blend_act_w = blend_act_t * w_a
        blend_pt_t  = blend_pt ** (1.0 / t_p)
        blend_pt_t /= blend_pt_t.sum(axis=1, keepdims=True)
        blend_pt_w  = blend_pt_t * w_p
        for sname, smask in [
            ("SN=2",   nsn==2),
            ("SN=3-4", (nsn>=3)&(nsn<=4)),
            ("SN=5-8", (nsn>=5)&(nsn<=8)),
            ("SN=9-12",(nsn>=9)&(nsn<=12)),
            ("SN>=13", nsn>=13)
        ]:
            m = common & smask
            if m.sum() < 5: continue
            fa = f1_score(y_a[m], blend_act_w[m].argmax(axis=1),
                          labels=ACTION_EVAL, average="macro", zero_division=0)
            fp = f1_score(y_p[m], blend_pt_w[m].argmax(axis=1),
                          labels=POINT_EVAL, average="macro", zero_division=0)
            if y_s[m].std() < 1e-9: au = 0.5
            else: au = roc_auc_score(y_s[m], blend_srv[m])
            print(f"{sname:<10} {m.sum():>6} {fa:>7.4f} {fp:>7.4f} {au:>7.4f} "
                  f"{0.4*fa+0.4*fp+0.2*au:>7.4f}")

    # ── Apply to test ─────────────────────────────────────────────────────────
    print("\n=== Building test submission ===")
    t1_act, t1_pt, t1_srv, t_uid = load_test(args.v1)
    if args.v2:
        t2_act, t2_pt, t2_srv, _ = load_test(args.v2)
        t_act  = best_a12 * t1_act + (1-best_a12) * t2_act
        t_pt   = best_b12 * t1_pt  + (1-best_b12) * t2_pt
        t_srv_ = 0.5 * t1_srv + 0.5 * t2_srv
    else:
        t_act, t_pt, t_srv_ = t1_act, t1_pt, t1_srv

    if use_aux:
        v11_t_act, v11_t_pt, v11_t_srv, v11_t_uid = load_test(aux_tag)
        # Align V11 test to v1 uid order
        v11_uid_map = {int(u): i for i, u in enumerate(v11_t_uid)}
        align = np.array([v11_uid_map[int(u)] for u in t_uid])
        v11_t_act_al = v11_t_act[align] if v11_t_act.shape[1] == N_ACTION else \
                       np.pad(v11_t_act, ((0,0),(0,N_ACTION-v11_t_act.shape[1])))[align]
        v11_t_pt_al  = v11_t_pt[align]
        v11_t_srv_al = v11_t_srv[align]
        t_act  = best_a_act * t_act   + (1-best_a_act) * v11_t_act_al
        t_pt   = best_a_pt  * t_pt    + (1-best_a_pt)  * v11_t_pt_al
        t_srv_ = best_a_srv * t_srv_  + (1-best_a_srv) * v11_t_srv_al

    # Temperature + weight
    t_act_t = t_act ** (1.0 / t_a)
    t_act_t /= t_act_t.sum(axis=1, keepdims=True)
    pred_act = (t_act_t * w_a).argmax(axis=1)
    t_pt_t  = t_pt ** (1.0 / t_p)
    t_pt_t  /= t_pt_t.sum(axis=1, keepdims=True)
    pred_pt  = (t_pt_t * w_p).argmax(axis=1)
    pred_srv = t_srv_

    # Output name (strip trailing .csv to prevent .csv.csv)
    if args.out:
        out_name = args.out
        if out_name.lower().endswith(".csv"):
            out_name = out_name[:-4]
    else:
        parts = [args.v1]
        if args.v2: parts.append(args.v2)
        if use_aux: parts.append(aux_tag)
        out_name = f"submission_{'_'.join(parts)}_optblend"
    out_path = os.path.join(SUBMISSION_DIR, f"{out_name}.csv")

    sub = pd.DataFrame({
        "rally_uid":       t_uid,
        "actionId":        pred_act,
        "pointId":         pred_pt,
        "serverGetPoint":  pred_srv,
    })
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"  actionId top: {dict(pd.Series(pred_act).value_counts().head(5))}")
    print(f"  pointId dist: {dict(pd.Series(pred_pt).value_counts().sort_index())}")
    print(f"  srv mean={pred_srv.mean():.4f}  std={pred_srv.std():.4f}")
    print(f"\nOOF OV={ov_opt:.4f}  (previous best V12+V11: 0.3734)")


if __name__ == "__main__":
    main()
