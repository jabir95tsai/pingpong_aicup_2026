"""Quick V12 + V11 hybrid submission generator.

Combines:
  - V12 actionId / pointId predictions (from submission_v12.csv)
  - V11 continuous serverGetPoint probs (from oof_predictions/v11_test_srv.npy
    aligned by submission_v11_transformer.csv rally_uid order)

Outputs:
  submission_v12_v11srv.csv  (key-aligned hybrid)

Also runs OOF evaluation showing V12+V11srv combined OV.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SUBMISSION_DIR

ACTION_EVAL = list(range(15))
POINT_EVAL  = list(range(10))


def macro_f1(y, probs_or_labels, labels, is_label=False):
    if is_label:
        return f1_score(y, probs_or_labels, labels=labels,
                         average="macro", zero_division=0)
    return f1_score(y, probs_or_labels.argmax(axis=1), labels=labels,
                     average="macro", zero_division=0)


def main():
    OOD = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")

    # ── OOF evaluation: V12 alone, V11 alone, hybrid ─────────────────────────
    v12_act = np.load(os.path.join(OOD, "v12_oof_act.npy"))
    v12_pt  = np.load(os.path.join(OOD, "v12_oof_pt.npy"))
    v12_srv = np.load(os.path.join(OOD, "v12_oof_srv.npy"))
    v12_mask = np.load(os.path.join(OOD, "v12_oof_mask.npy"))
    y_a = np.load(os.path.join(OOD, "v12_oof_y_act.npy"))
    y_p = np.load(os.path.join(OOD, "v12_oof_y_pt.npy"))
    y_s = np.load(os.path.join(OOD, "v12_oof_y_srv.npy"))
    nsn = np.load(os.path.join(OOD, "v12_oof_nsn.npy"))

    v11_srv = np.load(os.path.join(OOD, "v11_oof_srv.npy"))
    v11_mask = np.load(os.path.join(OOD, "v11_oof_mask.npy")).astype(bool)

    common = v12_mask.astype(bool) & v11_mask
    print(f"V12 OOF mask: {v12_mask.sum()}/{len(v12_mask)}")
    print(f"V11 OOF mask: {v11_mask.sum()}/{len(v11_mask)}")
    print(f"Common: {common.sum()}")

    # V12 alone
    f1a = macro_f1(y_a[common], v12_act[common], ACTION_EVAL)
    f1p = macro_f1(y_p[common], v12_pt[common], POINT_EVAL)
    auc12 = roc_auc_score(y_s[common], v12_srv[common])
    auc11 = roc_auc_score(y_s[common], v11_srv[common])
    ov_v12_alone   = 0.4 * f1a + 0.4 * f1p + 0.2 * auc12
    ov_v12_v11_srv = 0.4 * f1a + 0.4 * f1p + 0.2 * auc11
    print(f"\nV12 alone:        F1_a={f1a:.4f} F1_p={f1p:.4f} AUC={auc12:.4f}  OV={ov_v12_alone:.4f}")
    print(f"V12 + V11 srv:    F1_a={f1a:.4f} F1_p={f1p:.4f} AUC={auc11:.4f}  OV={ov_v12_v11_srv:.4f}")

    # Server blend search
    best_a, best_auc = 0.0, auc12
    for a in np.arange(0, 1.05, 0.05):
        b = a * v11_srv + (1 - a) * v12_srv
        auc = roc_auc_score(y_s[common], b[common])
        if auc > best_auc:
            best_a, best_auc = a, auc
    ov_v12_blendsrv = 0.4 * f1a + 0.4 * f1p + 0.2 * best_auc
    print(f"V12 + blend srv:  α={best_a:.2f}  AUC={best_auc:.4f}  OV={ov_v12_blendsrv:.4f}")

    # ── Slice analysis (per next-strikeNumber) ───────────────────────────────
    print("\n=== Per-SN slice metrics (V12 + V11 srv) ===")
    print(f"{'slice':<10} {'n':>6} {'F1_a':>7} {'F1_p':>7} {'AUC':>7} {'OV':>7}")
    for sn_name, sn_m in [("SN=2",   nsn == 2),
                           ("SN=3-4", (nsn >= 3) & (nsn <= 4)),
                           ("SN=5-8", (nsn >= 5) & (nsn <= 8)),
                           ("SN=9-12",(nsn >= 9) & (nsn <= 12)),
                           ("SN>=13", nsn >= 13)]:
        m = common & sn_m
        if m.sum() < 5:
            continue
        f1a_s = macro_f1(y_a[m], v12_act[m], ACTION_EVAL)
        f1p_s = macro_f1(y_p[m], v12_pt[m], POINT_EVAL)
        if y_s[m].std() < 1e-9:
            auc_s = 0.5
        else:
            auc_s = roc_auc_score(y_s[m], v11_srv[m])
        ov_s = 0.4 * f1a_s + 0.4 * f1p_s + 0.2 * auc_s
        print(f"{sn_name:<10} {m.sum():>6} {f1a_s:>7.4f} {f1p_s:>7.4f} {auc_s:>7.4f} {ov_s:>7.4f}")

    # ── Build hybrid test submission ─────────────────────────────────────────
    v12_path = os.path.join(SUBMISSION_DIR, "submission_v12.csv")
    v11_path = os.path.join(SUBMISSION_DIR, "submission_v11_transformer.csv")
    v12_sub = pd.read_csv(v12_path)
    v11_sub = pd.read_csv(v11_path)
    v11_srv_arr = np.load(os.path.join(OOD, "v11_test_srv.npy"))

    if len(v11_sub) != len(v11_srv_arr):
        raise ValueError(f"V11 sub/npy length mismatch: {len(v11_sub)} vs {len(v11_srv_arr)}")

    v11_lookup = pd.DataFrame({
        "rally_uid": v11_sub["rally_uid"].values,
        "v11_srv":   v11_srv_arr,
    })
    if v11_lookup["rally_uid"].nunique() != len(v11_lookup):
        raise ValueError("V11 has duplicate rally_uid")
    if v12_sub["rally_uid"].nunique() != len(v12_sub):
        raise ValueError("V12 has duplicate rally_uid")

    merged = v12_sub.merge(v11_lookup, on="rally_uid", how="inner")
    if len(merged) != len(v12_sub):
        miss = set(v12_sub["rally_uid"]) - set(v11_lookup["rally_uid"])
        raise ValueError(f"V12 has {len(miss)} rally_uids missing from V11 (first: {list(miss)[:5]})")

    out_path = os.path.join(SUBMISSION_DIR, "submission_v12_v11srv.csv")
    sub = merged[["rally_uid", "actionId", "pointId"]].copy()
    sub["serverGetPoint"] = merged["v11_srv"].values  # continuous
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  (n={len(sub)})")
    print(f"  serverGetPoint mean={sub['serverGetPoint'].mean():.4f}  std={sub['serverGetPoint'].std():.4f}")

    # Also save blended-server variant
    if best_a > 0:
        # Need V12 test srv; load from oof_predictions
        v12_test_srv = np.load(os.path.join(OOD, "v12_test_srv.npy"))
        v12_test_uid = np.load(os.path.join(OOD, "v12_test_rally_uid.npy"))
        if (v12_test_uid == v12_sub["rally_uid"].values).all():
            v12_srv_arr = v12_test_srv
        else:
            uid_map = {int(u): i for i, u in enumerate(v12_test_uid)}
            v12_srv_arr = np.array([v12_test_srv[uid_map[int(u)]] for u in v12_sub["rally_uid"]])

        # align v11_srv to v12 rally order
        v11_map = {int(u): i for i, u in enumerate(v11_sub["rally_uid"].values)}
        v11_aligned = np.array([v11_srv_arr[v11_map[int(u)]] for u in v12_sub["rally_uid"]])
        b_srv = best_a * v11_aligned + (1 - best_a) * v12_srv_arr
        sub2 = v12_sub[["rally_uid", "actionId", "pointId"]].copy()
        sub2["serverGetPoint"] = b_srv
        out2 = os.path.join(SUBMISSION_DIR, f"submission_v12_blendsrv{best_a:.2f}.csv")
        sub2.to_csv(out2, index=False)
        print(f"Saved: {out2}  (alpha_v11={best_a:.2f})")


if __name__ == "__main__":
    main()
