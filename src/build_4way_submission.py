"""Build test submission for 4-way blend (V12cb + V12_5f + V12 + V11)
using the OOF-optimized weights and thresholds saved by the 4-way search.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SUBMISSION_DIR

OOF = "oof_predictions"
N_ACTION = 19
N_POINT  = 10


def pad19(arr):
    if arr.shape[1] < N_ACTION:
        out = np.zeros((len(arr), N_ACTION), dtype=arr.dtype)
        out[:, :arr.shape[1]] = arr
        return out
    return arr


def load_test(tag):
    a = pad19(np.load(f"{OOF}/{tag}_test_act.npy"))
    p = np.load(f"{OOF}/{tag}_test_pt.npy")
    s = np.load(f"{OOF}/{tag}_test_srv.npy")
    uid_path = f"{OOF}/{tag}_test_rally_uid.npy"
    if os.path.exists(uid_path):
        u = np.load(uid_path)
    else:
        # v11
        u = pd.read_csv(os.path.join(SUBMISSION_DIR,
                                     "submission_v11_transformer.csv"))["rally_uid"].values
    return a, p, s, u


def align_to(probs_act, probs_pt, probs_srv, src_uid, dst_uid):
    m = {int(u): i for i, u in enumerate(src_uid)}
    idx = np.array([m[int(u)] for u in dst_uid])
    return probs_act[idx], probs_pt[idx], probs_srv[idx]


def main():
    # Load all 4 sources
    cb_a, cb_p, cb_s, cb_u = load_test("v12cb")
    f5_a, f5_p, f5_s, f5_u = load_test("v12_5f")
    v12_a, v12_p, v12_s, v12_u = load_test("v12")
    v11_a, v11_p, v11_s, v11_u = load_test("v11")

    # Align all to v12cb's uid order
    base_uid = cb_u
    f5_a, f5_p, f5_s   = align_to(f5_a, f5_p, f5_s,  f5_u, base_uid)
    v12_a, v12_p, v12_s = align_to(v12_a, v12_p, v12_s, v12_u, base_uid)
    v11_a, v11_p, v11_s = align_to(v11_a, v11_p, v11_s, v11_u, base_uid)

    # Load weights
    w_a = np.load(f"{OOF}/4way_w_a.npy")  # [cb, 5f, v12, v11] for action
    w_p = np.load(f"{OOF}/4way_w_p.npy")
    w_s = np.load(f"{OOF}/4way_w_s.npy")
    print(f"Action weights:  cb={w_a[0]:.2f}  5f={w_a[1]:.2f}  v12={w_a[2]:.2f}  v11={w_a[3]:.2f}")
    print(f"Point  weights:  cb={w_p[0]:.2f}  5f={w_p[1]:.2f}  v12={w_p[2]:.2f}  v11={w_p[3]:.2f}")
    print(f"Server weights:  cb={w_s[0]:.2f}  5f={w_s[1]:.2f}  v12={w_s[2]:.2f}  v11={w_s[3]:.2f}")

    blend_act = w_a[0]*cb_a + w_a[1]*f5_a + w_a[2]*v12_a + w_a[3]*v11_a
    blend_pt  = w_p[0]*cb_p + w_p[1]*f5_p + w_p[2]*v12_p + w_p[3]*v11_p
    blend_srv = w_s[0]*cb_s + w_s[1]*f5_s + w_s[2]*v12_s + w_s[3]*v11_s

    # Apply temperature + class weights from optimization
    t_a = float(np.load(f"{OOF}/4way_thresh_t_a.npy")[0])
    cw_a = np.load(f"{OOF}/4way_thresh_w_a.npy")
    t_p = float(np.load(f"{OOF}/4way_thresh_t_p.npy")[0])
    cw_p = np.load(f"{OOF}/4way_thresh_w_p.npy")
    print(f"\nThresholds: t_a={t_a:.2f}  t_p={t_p:.2f}")

    act_t = blend_act ** (1.0/t_a)
    act_t /= act_t.sum(axis=1, keepdims=True)
    pred_act = (act_t * cw_a).argmax(axis=1)

    pt_t = blend_pt ** (1.0/t_p)
    pt_t /= pt_t.sum(axis=1, keepdims=True)
    pred_pt = (pt_t * cw_p).argmax(axis=1)

    pred_srv = blend_srv

    out_path = os.path.join(SUBMISSION_DIR, "submission_4way_optblend.csv")
    sub = pd.DataFrame({
        "rally_uid":      base_uid,
        "actionId":       pred_act,
        "pointId":        pred_pt,
        "serverGetPoint": pred_srv,
    })
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  actionId top: {dict(pd.Series(pred_act).value_counts().head(5))}")
    print(f"  pointId dist: {dict(pd.Series(pred_pt).value_counts().sort_index())}")
    print(f"  srv mean={pred_srv.mean():.4f}  std={pred_srv.std():.4f}")
    print(f"\nExpected LB ~0.362 based on OOF=0.3809 minus 0.019 CV-LB gap")


if __name__ == "__main__":
    main()
