"""Final V12 ensemble: V12 GBM + V11 transformer + V12 hierarchical pointId.

Searches optimal blend weights on OOF (action / point / hier-point / server),
emits multiple submission candidates plus a slice report.

Inputs:
  oof_predictions/
    v12_oof_act.npy, v12_oof_pt.npy, v12_oof_srv.npy, v12_oof_mask.npy
    v12_oof_y_act.npy, v12_oof_y_pt.npy, v12_oof_y_srv.npy, v12_oof_nsn.npy
    v12_test_act.npy, v12_test_pt.npy, v12_test_srv.npy, v12_test_rally_uid.npy

    v11_oof_act.npy, v11_oof_pt.npy, v11_oof_srv.npy, v11_oof_mask.npy
    v11_test_act.npy, v11_test_pt.npy, v11_test_srv.npy

    v12_hier_oof_valid.npy, v12_hier_oof_depth.npy, v12_hier_oof_side.npy
    v12_hier_oof_mask.npy
    v12_hier_test_valid.npy, v12_hier_test_depth.npy, v12_hier_test_side.npy
    v12_hier_test_rally_uid.npy

Outputs (in submissions/):
  submission_v12_alone.csv
  submission_v12_v11_blend.csv
  submission_v12_v11_hier.csv  (best of all)
  submission_v12_v11_hier_report.txt
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SUBMISSION_DIR

N_ACTION_TRAIN = 15
N_POINT = 10
ACTION_EVAL = list(range(N_ACTION_TRAIN))
POINT_EVAL  = list(range(N_POINT))

DEPTH_BUCKET = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3}
SIDE_BUCKET  = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3, 7: 1, 8: 2, 9: 3}


def macro_f1(y, probs, labels):
    return f1_score(y, probs.argmax(axis=1), labels=labels,
                    average="macro", zero_division=0)


def load_tagged_array(base_dir, tag, suffix):
    return np.load(os.path.join(base_dir, f"{tag}_{suffix}.npy"))


def joint_reconstruct(p_v, p_d, p_s):
    n = len(p_v)
    out = np.zeros((n, N_POINT), dtype=np.float32)
    out[:, 0] = 1.0 - p_v
    for k in range(1, N_POINT):
        d = DEPTH_BUCKET[k]
        s = SIDE_BUCKET[k]
        out[:, k] = p_v * p_d[:, d] * p_s[:, s]
    s_pos = out[:, 1:].sum(axis=1)
    s_pos = np.where(s_pos < 1e-9, 1.0, s_pos)
    scale = p_v / s_pos
    out[:, 1:] = out[:, 1:] * scale[:, np.newaxis]
    return np.clip(out, 1e-9, 1.0)


def slice_metrics(act, pt, srv, y_a, y_p, y_s, nsn, mask, name="ALL", report=None):
    if mask.sum() < 5:
        return None
    f1_a = f1_score(y_a[mask], act[mask].argmax(axis=1), labels=ACTION_EVAL,
                     average="macro", zero_division=0)
    f1_p = f1_score(y_p[mask], pt[mask].argmax(axis=1), labels=POINT_EVAL,
                     average="macro", zero_division=0)
    if y_s[mask].std() < 1e-9:
        auc = 0.5
    else:
        auc = roc_auc_score(y_s[mask], srv[mask])
    ov = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
    line = (f"  {name:<10} n={mask.sum():>5}  F1_a={f1_a:.4f}  F1_p={f1_p:.4f}  "
            f"AUC={auc:.4f}  OV={ov:.4f}")
    print(line)
    if report is not None:
        report.append(line)
    return ov, f1_a, f1_p, auc


def run_slices(act, pt, srv, y_a, y_p, y_s, nsn, mask, label, report):
    print(f"\n=== {label} slices ===")
    report.append(f"\n=== {label} slices ===")
    slice_metrics(act, pt, srv, y_a, y_p, y_s, nsn, mask, "ALL", report)
    for sn_name, sn_m in [
        ("SN=2",   nsn == 2),
        ("SN=3-4", (nsn >= 3) & (nsn <= 4)),
        ("SN=5-8", (nsn >= 5) & (nsn <= 8)),
        ("SN=9-12",(nsn >= 9) & (nsn <= 12)),
        ("SN>=13", nsn >= 13),
    ]:
        m = mask & sn_m
        slice_metrics(act, pt, srv, y_a, y_p, y_s, nsn, m, sn_name, report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default=os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions"))
    ap.add_argument("--primary-tag", default="v12")
    ap.add_argument("--aux-tag", default="v11")
    ap.add_argument("--aux-submission", default=os.path.join(SUBMISSION_DIR, "submission_v11_transformer.csv"))
    ap.add_argument("--hier-tag", default="v12_hier")
    ap.add_argument("--output-tag", default=None)
    ap.add_argument("--report-name", default=None)
    args = ap.parse_args()
    OOD = args.oof_dir
    primary_tag = args.primary_tag
    aux_tag = args.aux_tag
    aux_submission = args.aux_submission
    hier_tag = args.hier_tag
    output_tag = args.output_tag or f"{primary_tag}_{aux_tag}_hier"
    report_name = args.report_name or f"submission_{output_tag}_report.txt"
    report = []

    # ── Load primary OOF ─────────────────────────────────────────────────────
    v12_act = load_tagged_array(OOD, primary_tag, "oof_act")
    v12_pt  = load_tagged_array(OOD, primary_tag, "oof_pt")
    v12_srv = load_tagged_array(OOD, primary_tag, "oof_srv")
    v12_mask = load_tagged_array(OOD, primary_tag, "oof_mask")
    y_a = load_tagged_array(OOD, primary_tag, "oof_y_act")
    y_p = load_tagged_array(OOD, primary_tag, "oof_y_pt")
    y_s = load_tagged_array(OOD, primary_tag, "oof_y_srv")
    nsn = load_tagged_array(OOD, primary_tag, "oof_nsn")
    print(f"Loaded {primary_tag} OOF: act={v12_act.shape}  pt={v12_pt.shape}  mask={v12_mask.sum()}")

    # ── Load auxiliary OOF ───────────────────────────────────────────────────
    v11_act = load_tagged_array(OOD, aux_tag, "oof_act")
    v11_pt  = load_tagged_array(OOD, aux_tag, "oof_pt")
    v11_srv = load_tagged_array(OOD, aux_tag, "oof_srv")
    v11_mask = load_tagged_array(OOD, aux_tag, "oof_mask").astype(bool)
    print(f"Loaded {aux_tag} OOF: act={v11_act.shape}  pt={v11_pt.shape}  mask={v11_mask.sum()}")

    # Hard length-equality check (cf. blend_v10_v11.py blocker fix)
    if len(v12_act) != len(v11_act):
        raise ValueError(f"OOF length mismatch V12={len(v12_act)} vs V11={len(v11_act)}")

    # V11 only outputs 15-class action; pad to 19
    if v11_act.shape[1] != v12_act.shape[1]:
        v11_act_full = np.zeros((len(v11_act), v12_act.shape[1]), dtype=v11_act.dtype)
        v11_act_full[:, :v11_act.shape[1]] = v11_act
        v11_act = v11_act_full

    common = v12_mask.astype(bool) & v11_mask
    print(f"Common OOF samples: {common.sum()}")

    # ── Baseline slice reports ───────────────────────────────────────────────
    run_slices(v12_act, v12_pt, v12_srv, y_a, y_p, y_s, nsn, common, "V12 alone", report)
    run_slices(v11_act, v11_pt, v11_srv, y_a, y_p, y_s, nsn, common, "V11 alone", report)

    # ── Search alpha per task (action / point / server) ──────────────────────
    print("\n=== Grid search blend (V12 + V11) ===")
    best = (-1, 0.5, 0.5, 0.5)
    for aa, ap_, asrv in product(np.arange(0, 1.05, 0.1),
                                   np.arange(0, 1.05, 0.1),
                                   np.arange(0, 1.05, 0.1)):
        b_act = aa * v12_act + (1 - aa) * v11_act
        b_pt  = ap_ * v12_pt + (1 - ap_) * v11_pt
        b_srv = asrv * v12_srv + (1 - asrv) * v11_srv
        f1a = macro_f1(y_a[common], b_act[common], ACTION_EVAL)
        f1p = macro_f1(y_p[common], b_pt[common],  POINT_EVAL)
        if y_s[common].std() < 1e-9:
            auc = 0.5
        else:
            auc = roc_auc_score(y_s[common], b_srv[common])
        ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        if ov > best[0]:
            best = (ov, aa, ap_, asrv)
    print(f"  Best OV={best[0]:.4f}  alpha_act={best[1]:.1f}  alpha_pt={best[2]:.1f}  alpha_srv={best[3]:.1f}")
    report.append(f"\nBest V12+V11 blend OV={best[0]:.4f}  α_a={best[1]:.1f} α_p={best[2]:.1f} α_s={best[3]:.1f}")

    aa, ap_, asrv = best[1], best[2], best[3]
    b_act = aa * v12_act + (1 - aa) * v11_act
    b_pt  = ap_ * v12_pt + (1 - ap_) * v11_pt
    b_srv = asrv * v12_srv + (1 - asrv) * v11_srv
    run_slices(b_act, b_pt, b_srv, y_a, y_p, y_s, nsn, common, "Blend (best alpha)", report)

    # ── Try hierarchical point if available ──────────────────────────────────
    hier_v_p = os.path.join(OOD, f"{hier_tag}_oof_valid.npy")
    hier_used = False
    if os.path.exists(hier_v_p):
        print("\n=== Try hierarchical pointId ===")
        h_v = load_tagged_array(OOD, hier_tag, "oof_valid")
        h_d = load_tagged_array(OOD, hier_tag, "oof_depth")
        h_s = load_tagged_array(OOD, hier_tag, "oof_side")
        h_m = load_tagged_array(OOD, hier_tag, "oof_mask").astype(bool)
        if len(h_v) == len(v12_pt):
            h_pt = joint_reconstruct(h_v, h_d, h_s)
            h_common = common & h_m

            best_a = 0.0
            base_f1p = macro_f1(y_p[h_common], b_pt[h_common], POINT_EVAL)
            best_f1p = base_f1p
            for a in np.arange(0, 1.05, 0.05):
                cand = a * h_pt + (1 - a) * b_pt
                f = macro_f1(y_p[h_common], cand[h_common], POINT_EVAL)
                if f > best_f1p:
                    best_a, best_f1p = a, f
            print(f"  Hier alpha={best_a:.2f}  F1_p={best_f1p:.4f}  (vs blend={base_f1p:.4f})")
            report.append(f"\nHier α_hier={best_a:.2f}  F1_p_hier={best_f1p:.4f}  vs blend F1_p={base_f1p:.4f}")
            if best_a > 0:
                b_pt = best_a * h_pt + (1 - best_a) * b_pt
                hier_used = True
                run_slices(b_act, b_pt, b_srv, y_a, y_p, y_s, nsn, h_common,
                            f"Blend+Hier α={best_a:.2f}", report)

    # ── Test predictions (key-aligned by rally_uid) ───────────────────────────
    v12_t_act = load_tagged_array(OOD, primary_tag, "test_act")
    v12_t_pt  = load_tagged_array(OOD, primary_tag, "test_pt")
    v12_t_srv = load_tagged_array(OOD, primary_tag, "test_srv")
    v12_t_uid = load_tagged_array(OOD, primary_tag, "test_rally_uid")
    v11_t_act = load_tagged_array(OOD, aux_tag, "test_act")
    v11_t_pt  = load_tagged_array(OOD, aux_tag, "test_pt")
    v11_t_srv = load_tagged_array(OOD, aux_tag, "test_srv")

    # Need v11_test rally_uid for proper alignment — fall back to V11 submission CSV
    v11_uid_order = pd.read_csv(aux_submission)["rally_uid"].values
    if len(v11_uid_order) != len(v11_t_act):
        raise ValueError(f"V11 test rally_uid length mismatch: sub={len(v11_uid_order)} npy={len(v11_t_act)}")

    # Re-order V11 test arrays into V12 rally_uid order
    uid_to_idx = {int(u): i for i, u in enumerate(v11_uid_order)}
    align_idx = []
    missing = []
    for u in v12_t_uid:
        if int(u) in uid_to_idx:
            align_idx.append(uid_to_idx[int(u)])
        else:
            missing.append(u)
    if missing:
        raise ValueError(f"V11 test missing {len(missing)} rally_uid (first: {missing[:5]})")
    align_idx = np.array(align_idx, dtype=int)

    if v11_t_act.shape[1] != v12_t_act.shape[1]:
        pad = np.zeros((len(v11_t_act), v12_t_act.shape[1]), dtype=v11_t_act.dtype)
        pad[:, :v11_t_act.shape[1]] = v11_t_act
        v11_t_act = pad
    v11_t_act = v11_t_act[align_idx]
    v11_t_pt  = v11_t_pt[align_idx]
    v11_t_srv = v11_t_srv[align_idx]

    test_b_act = aa * v12_t_act + (1 - aa) * v11_t_act
    test_b_pt  = ap_ * v12_t_pt + (1 - ap_) * v11_t_pt
    test_b_srv = asrv * v12_t_srv + (1 - asrv) * v11_t_srv

    if hier_used:
        h_t_v = load_tagged_array(OOD, hier_tag, "test_valid")
        h_t_d = load_tagged_array(OOD, hier_tag, "test_depth")
        h_t_s = load_tagged_array(OOD, hier_tag, "test_side")
        h_t_uid = load_tagged_array(OOD, hier_tag, "test_rally_uid")
        if (h_t_uid == v12_t_uid).all():
            h_t_pt = joint_reconstruct(h_t_v, h_t_d, h_t_s)
            test_b_pt = best_a * h_t_pt + (1 - best_a) * test_b_pt

    # Final test predictions
    pred_act = test_b_act.argmax(axis=1)
    pred_pt  = test_b_pt.argmax(axis=1)
    pred_srv = test_b_srv  # continuous for AUC

    out_path = os.path.join(SUBMISSION_DIR, f"submission_{output_tag}.csv")
    sub = pd.DataFrame({
        "rally_uid": v12_t_uid,
        "actionId": pred_act,
        "pointId": pred_pt,
        "serverGetPoint": pred_srv,
    })
    sub.to_csv(out_path, index=False)
    print(f"\nSaved final submission: {out_path}")

    # Pure V12 test (action/point) + V11 continuous srv
    sub_alone = pd.DataFrame({
        "rally_uid": v12_t_uid,
        "actionId": v12_t_act.argmax(axis=1),
        "pointId": v12_t_pt.argmax(axis=1),
        "serverGetPoint": v11_t_srv,
    })
    sub_alone.to_csv(os.path.join(SUBMISSION_DIR, f"submission_{primary_tag}_alone_{aux_tag}srv.csv"),
                      index=False)
    print(f"Saved {primary_tag}+{aux_tag}srv: {os.path.join(SUBMISSION_DIR, f'submission_{primary_tag}_alone_{aux_tag}srv.csv')}")

    # Save text report
    rep_path = os.path.join(SUBMISSION_DIR, report_name)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Saved report: {rep_path}")


if __name__ == "__main__":
    main()
