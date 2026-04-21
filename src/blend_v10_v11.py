"""Blend V10 GBM + V11 Transformer OOF predictions to find optimal weights.

Usage:
  python src/blend_v10_v11.py
  python src/blend_v10_v11.py --alpha-act 0.4 --alpha-pt 0.4 --alpha-srv 0.6

The script:
  1. Loads V11 OOF predictions (from oof_predictions/)
  2. Loads V10 GBM OOF data (re-runs lightweight predictions if needed)
     — OR can accept V10 OOF npz files if saved separately
  3. Grid-searches blend weights on OOF to maximize OV
  4. Applies optimal blend weights to test predictions
  5. Saves blended submission

Note: V10 GBM does NOT save OOF predictions by default.  Run with --gbm-oof-dir
to point to a directory with v10_oof_act.npy etc. if you saved them separately.
Otherwise uses V11 transformer predictions directly.
"""
import sys, os, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SUBMISSION_DIR

N_ACTION_TRAIN = 15
N_POINT        = 10
ACTION_EVAL    = list(range(15))
POINT_EVAL     = list(range(10))


def macro_f1(y_true, probs, labels):
    return f1_score(y_true, probs.argmax(axis=1),
                    labels=labels, average="macro", zero_division=0)


def compute_ov(act_probs, pt_probs, srv_probs, y_act, y_pt, y_srv, oof_mask=None):
    if oof_mask is not None:
        act_probs = act_probs[oof_mask]
        pt_probs  = pt_probs[oof_mask]
        srv_probs = srv_probs[oof_mask]
        y_act = y_act[oof_mask]
        y_pt  = y_pt[oof_mask]
        y_srv = y_srv[oof_mask]
    f1_a = macro_f1(y_act, act_probs, ACTION_EVAL)
    f1_p = macro_f1(y_pt,  pt_probs,  POINT_EVAL)
    auc  = roc_auc_score(y_srv, srv_probs)
    return 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc, f1_a, f1_p, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-dir",   type=str,
                        default=os.path.join(os.path.dirname(SUBMISSION_DIR),
                                             "oof_predictions"),
                        help="Directory with v11_oof_*.npy and v11_test_*.npy")
    parser.add_argument("--gbm-oof-dir", type=str, default="",
                        help="Optional directory with v10_oof_*.npy for GBM OOF")
    parser.add_argument("--v11-sub",   type=str,
                        default=os.path.join(SUBMISSION_DIR,
                                             "submission_v11_transformer.csv"),
                        help="V11 test predictions (submission CSV)")
    parser.add_argument("--v10-sub",   type=str,
                        default=os.path.join(SUBMISSION_DIR, "submission_v10.csv"),
                        help="V10 GBM test predictions (submission CSV)")
    parser.add_argument("--alpha-act", type=float, default=-1,
                        help="Manual blend weight for transformer action (-1=search)")
    parser.add_argument("--alpha-pt",  type=float, default=-1)
    parser.add_argument("--alpha-srv", type=float, default=-1)
    args = parser.parse_args()

    oof_dir = args.oof_dir
    print(f"Loading V11 OOF from {oof_dir}")

    oof_act11  = np.load(os.path.join(oof_dir, "v11_oof_act.npy"))
    oof_pt11   = np.load(os.path.join(oof_dir, "v11_oof_pt.npy"))
    oof_srv11  = np.load(os.path.join(oof_dir, "v11_oof_srv.npy"))
    oof_mask11 = np.load(os.path.join(oof_dir, "v11_oof_mask.npy"))
    test_act11 = np.load(os.path.join(oof_dir, "v11_test_act.npy"))
    test_pt11  = np.load(os.path.join(oof_dir, "v11_test_pt.npy"))
    test_srv11 = np.load(os.path.join(oof_dir, "v11_test_srv.npy"))
    print(f"  V11 OOF shape: act={oof_act11.shape}  mask={oof_mask11.sum()}/{len(oof_mask11)}")

    # V11 standalone OV
    from config import TRAIN_PATH
    from data_cleaning import clean_data
    raw_train = pd.read_csv(TRAIN_PATH)
    train_df, _, _ = clean_data(raw_train, pd.DataFrame())
    # Need y_act, y_pt, y_srv labels in same order as V11 OOF
    # V11 OOF follows the same sample order as build_samples(train_df)
    from train_v11_transformer import build_samples
    all_samples = build_samples(train_df, is_train=True)
    y_a_all = np.array([s["y_action"] for s in all_samples])
    y_p_all = np.array([s["y_point"]  for s in all_samples])
    y_s_all = np.array([s["y_server"] for s in all_samples])

    ov11, f1a11, f1p11, auc11 = compute_ov(
        oof_act11, oof_pt11, oof_srv11,
        y_a_all, y_p_all, y_s_all, oof_mask11)
    print(f"\nV11 Transformer standalone: action={f1a11:.4f}  point={f1p11:.4f}  AUC={auc11:.4f}  OV={ov11:.4f}")

    if args.gbm_oof_dir:
        oof_act10  = np.load(os.path.join(args.gbm_oof_dir, "v10_oof_act.npy"))
        oof_pt10   = np.load(os.path.join(args.gbm_oof_dir, "v10_oof_pt.npy"))
        oof_srv10  = np.load(os.path.join(args.gbm_oof_dir, "v10_oof_srv.npy"))
        oof_mask10 = np.load(os.path.join(args.gbm_oof_dir, "v10_oof_mask.npy"))

        ov10, f1a10, f1p10, auc10 = compute_ov(
            oof_act10, oof_pt10, oof_srv10,
            y_a_all, y_p_all, y_s_all, oof_mask10)
        print(f"V10 GBM standalone:        action={f1a10:.4f}  point={f1p10:.4f}  AUC={auc10:.4f}  OV={ov10:.4f}")

        # Find common OOF mask
        common_mask = oof_mask11 & oof_mask10

        if args.alpha_act < 0:
            print("\nSearching blend weights on OOF...")
            best_ov, best_aa, best_ap, best_as = -1.0, 0.5, 0.5, 0.5
            for aa, ap, as_ in product(np.arange(0, 1.05, 0.1),
                                        np.arange(0, 1.05, 0.1),
                                        np.arange(0, 1.05, 0.1)):
                blend_act = aa * oof_act11 + (1 - aa) * oof_act10[:len(oof_act11)]
                blend_pt  = ap * oof_pt11  + (1 - ap) * oof_pt10[:len(oof_pt11)]
                blend_srv = as_ * oof_srv11 + (1 - as_) * oof_srv10[:len(oof_srv11)]
                ov, _, _, _ = compute_ov(blend_act, blend_pt, blend_srv,
                                          y_a_all, y_p_all, y_s_all, common_mask)
                if ov > best_ov:
                    best_ov, best_aa, best_ap, best_as = ov, aa, ap, as_
            print(f"  Best OV={best_ov:.4f}  alpha_act={best_aa:.1f}  alpha_pt={best_ap:.1f}  alpha_srv={best_as:.1f}")
            alpha_act, alpha_pt, alpha_srv = best_aa, best_ap, best_as
        else:
            alpha_act, alpha_pt, alpha_srv = args.alpha_act, args.alpha_pt, args.alpha_srv
            blend_act = alpha_act * oof_act11 + (1 - alpha_act) * oof_act10[:len(oof_act11)]
            blend_pt  = alpha_pt * oof_pt11   + (1 - alpha_pt) * oof_pt10[:len(oof_pt11)]
            blend_srv = alpha_srv * oof_srv11 + (1 - alpha_srv) * oof_srv10[:len(oof_srv11)]
            ov, f1a, f1p, auc = compute_ov(blend_act, blend_pt, blend_srv,
                                             y_a_all, y_p_all, y_s_all, common_mask)
            print(f"\nManual blend OV={ov:.4f}  F1_a={f1a:.4f}  F1_p={f1p:.4f}  AUC={auc:.4f}")

        # Build blended test predictions using V10/V11 submissions
        v11_sub = pd.read_csv(args.v11_sub)
        v10_sub = pd.read_csv(args.v10_sub)

        # Align V10 server predictions to V11 rally_uid order (key-based, not positional).
        # V10 submission has binary 0/1 serverGetPoint; use that as the V10 component.
        v11_uid_order = v11_sub["rally_uid"].values
        v10_srv_lookup = v10_sub.set_index("rally_uid")["serverGetPoint"].to_dict()
        test_srv10 = np.array([v10_srv_lookup.get(uid, 0.5) for uid in v11_uid_order],
                               dtype=np.float32)
        n_missing_v10 = np.sum(test_srv10 == 0.5)
        if n_missing_v10:
            print(f"  WARNING: {n_missing_v10} test rally_uids not in V10 sub; using 0.5")

        # alpha_srv was searched on OOF where both models had real predictions;
        # test-time blend must use the same formula so OOF calibration is valid.
        blend_test_srv = alpha_srv * test_srv11 + (1 - alpha_srv) * test_srv10

        print("Using V11 action/point with rally-aligned blended server.")
        pred_act = test_act11.argmax(axis=1)
        pred_pt  = test_pt11.argmax(axis=1)
        pred_srv = blend_test_srv
    else:
        # No V10 OOF → just emit V11 standalone submission
        v11_sub = pd.read_csv(args.v11_sub)
        print("\nNo V10 OOF provided — using V11 predictions only.")
        pred_act = test_act11.argmax(axis=1)
        pred_pt  = test_pt11.argmax(axis=1)
        pred_srv = test_srv11

    # Emit final blended submission (key-based: use V11 rally_uid order)
    rally_uid_test = pd.read_csv(args.v11_sub)["rally_uid"].values
    sub = pd.DataFrame({
        "rally_uid":      rally_uid_test,
        "actionId":       pred_act,
        "pointId":        pred_pt,
        "serverGetPoint": pred_srv,
    })
    out_path = os.path.join(SUBMISSION_DIR, "submission_v11_blend.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nBlended submission saved: {out_path}")


if __name__ == "__main__":
    main()
