"""Slice analysis on V11 OOF predictions.

Computes overall + per-SN-bucket F1/AUC/OV from V11 OOF predictions saved in
oof_predictions/v11_oof_*.npy.

Each OOF row corresponds to one (rally_uid, target_strikeNumber) sample.
We rebuild the sample index by re-running build_samples on the cleaned train
data and align with the OOF arrays positionally.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH
from data_cleaning import clean_data
from train_v11_transformer import build_samples

ACTION_EVAL = list(range(15))   # exclude serve classes when scored as 0
POINT_EVAL  = list(range(10))


def macro_f1(y_true, probs, labels):
    return f1_score(y_true, probs.argmax(axis=1), labels=labels,
                    average="macro", zero_division=0)


def slice_metrics(act_p, pt_p, srv_p, y_a, y_p, y_s, mask):
    if mask.sum() < 5:
        return None
    f1_a = macro_f1(y_a[mask], act_p[mask], ACTION_EVAL)
    f1_p = macro_f1(y_p[mask], pt_p[mask], POINT_EVAL)
    if y_s[mask].nunique() if hasattr(y_s[mask], "nunique") else len(set(y_s[mask].tolist())) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(y_s[mask], srv_p[mask])
    return {
        "n": int(mask.sum()),
        "f1_a": f1_a,
        "f1_p": f1_p,
        "auc": auc,
        "ov": 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc,
    }


def main():
    oof_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oof_predictions")
    act = np.load(os.path.join(oof_dir, "v11_oof_act.npy"))
    pt  = np.load(os.path.join(oof_dir, "v11_oof_pt.npy"))
    srv = np.load(os.path.join(oof_dir, "v11_oof_srv.npy"))
    msk = np.load(os.path.join(oof_dir, "v11_oof_mask.npy"))
    print(f"Loaded V11 OOF: act={act.shape} pt={pt.shape} srv={srv.shape} mask={msk.sum()}/{len(msk)}")

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(os.path.join(os.path.dirname(TRAIN_PATH), "test.csv"))
    train_df, _, _ = clean_data(raw_train, raw_test)
    samples = build_samples(train_df, is_train=True)
    print(f"Rebuilt {len(samples)} samples")

    if len(samples) != len(act):
        print(f"  WARN: sample count {len(samples)} != OOF len {len(act)}")
        return

    y_a = np.array([s["y_action"] for s in samples])
    y_p = np.array([s["y_point"]  for s in samples])
    y_s = np.array([s["y_server"] for s in samples])
    nsn = np.array([s["next_sn"]  for s in samples])

    msk = msk.astype(bool)

    print("\n=== OVERALL (where mask=1) ===")
    overall = slice_metrics(act, pt, srv, y_a, y_p, y_s, msk)
    if overall:
        print(f"  n={overall['n']}  F1_a={overall['f1_a']:.4f}  F1_p={overall['f1_p']:.4f}  "
              f"AUC={overall['auc']:.4f}  OV={overall['ov']:.4f}")

    print("\n=== Per next-strikeNumber slice ===")
    print(f"{'slice':<10} {'n':>6} {'F1_a':>7} {'F1_p':>7} {'AUC':>7} {'OV':>7}")
    slices = [
        ("SN=1",   nsn == 1),
        ("SN=2",   nsn == 2),
        ("SN=3-4", (nsn >= 3) & (nsn <= 4)),
        ("SN=5-8", (nsn >= 5) & (nsn <= 8)),
        ("SN=9-12",(nsn >= 9) & (nsn <= 12)),
        ("SN>=13", nsn >= 13),
    ]
    for name, sn_mask in slices:
        m = sn_mask & msk
        r = slice_metrics(act, pt, srv, y_a, y_p, y_s, m)
        if r:
            print(f"{name:<10} {r['n']:>6} {r['f1_a']:>7.4f} {r['f1_p']:>7.4f} {r['auc']:>7.4f} {r['ov']:>7.4f}")

    # Point class breakdown
    print("\n=== PointId per-class F1 ===")
    pt_pred = pt.argmax(axis=1)
    for cls in range(10):
        mc = (y_p == cls) & msk
        if mc.sum() < 5:
            continue
        f1c = f1_score(y_p == cls, pt_pred == cls, zero_division=0)
        n_correct = ((pt_pred == cls) & mc).sum()
        n_total = mc.sum()
        print(f"  pt={cls}: n={n_total:>4}  F1={f1c:.4f}  recall={n_correct/n_total:.4f}")

    # Confusion: 0 vs 7/8/9
    print("\n=== Point 0-vs-long confusion ===")
    long_mask = msk & np.isin(y_p, [0, 7, 8, 9])
    if long_mask.sum() > 0:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_p[long_mask], pt_pred[long_mask], labels=[0, 7, 8, 9])
        print("       pred=0  pred=7  pred=8  pred=9")
        for i, c in enumerate([0, 7, 8, 9]):
            print(f"true={c}  {cm[i,0]:>6}  {cm[i,1]:>6}  {cm[i,2]:>6}  {cm[i,3]:>6}")

    # Action per-class
    print("\n=== ActionId per-class F1 (0..14) ===")
    act_pred = act.argmax(axis=1)
    for cls in range(15):
        mc = (y_a == cls) & msk
        if mc.sum() < 5:
            continue
        f1c = f1_score(y_a == cls, act_pred == cls, zero_division=0)
        print(f"  act={cls}: n={mc.sum():>4}  F1={f1c:.4f}")


if __name__ == "__main__":
    main()
