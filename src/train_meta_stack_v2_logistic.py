"""meta_stack v2 — logistic / linear stack diagnostic.

Same protocol as `train_meta_stack.py` (R-005), but using
`LogisticRegression` (multinomial for action/point, binary for server) instead
of shallow LightGBM. Codex's APPROVE_WITH_FIXES explicitly allowed
"linear/logistic stack" as an alternative to the shallow LightGBM.

Why this v2: the v1 LGBM significantly underfit (F1_p −0.029 vs best single
component). A linear stack is the closest possible model to what the
zoo_v2 Dirichlet weight search already does (global linear combination of
component probabilities), so its OOF tells us how much per-row blending
adds when the function class is restricted to row-conditional linear.

Same Codex constraints as v1:
- Outer: GroupKFold(5) by match, hard no-overlap assertion.
- Inputs: ONLY component probability arrays.
- Mask-false rows excluded.
- HELD as standalone diagnostic, not zoo-eligible without separate review.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, SUBMISSION_DIR, PROJECT_ROOT, RANDOM_SEED
from data_cleaning import clean_data
from features_v9 import build_features_v9, compute_global_stats_v9

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")

COMPONENTS = [
    "v16_testhist_aug", "v16_avg3", "v16_seed1", "v16_seed2",
    "v14_avg3", "v14_seed0", "v14_seed1", "v14_seed2", "v14_recvhand",
    "v12_5f",
    "v11", "v11plus", "v11_aug",
    "v13",
]

N_ACTION = 19
N_ACTION_EVAL = 15
N_POINT = 10
ACTION_EVAL_LABELS = list(range(15))
POINT_EVAL_LABELS = list(range(10))

# Logistic regression config — strong L2 to keep this clearly "regularized".
LR_C = 1.0  # inverse regularization strength
LR_MAX_ITER = 200


def pad19(arr):
    if arr.shape[1] >= N_ACTION:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], N_ACTION), dtype=np.float32)
    out[:, :arr.shape[1]] = arr
    return out


def fast_macro_f1(y_true, y_pred, labels, n_total):
    cm = np.bincount(y_true.astype(np.int64) * n_total + y_pred.astype(np.int64),
                     minlength=n_total * n_total).reshape(n_total, n_total)
    col_sum = cm.sum(axis=0)
    row_sum = cm.sum(axis=1)
    diag = np.diag(cm)
    f1s = []
    for c in labels:
        tp = diag[c]; fp = col_sum[c] - tp; fn = row_sum[c] - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom <= 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


def load_components():
    ref = "v16_testhist_aug"
    mask = np.load(f"{OOF_DIR}/{ref}_oof_mask.npy")
    y_a = np.load(f"{OOF_DIR}/{ref}_oof_y_act.npy")
    y_p = np.load(f"{OOF_DIR}/{ref}_oof_y_pt.npy")
    y_s = np.load(f"{OOF_DIR}/{ref}_oof_y_srv.npy")
    nsn = np.load(f"{OOF_DIR}/{ref}_oof_nsn.npy")
    test_uid = np.load(f"{OOF_DIR}/{ref}_test_rally_uid.npy")
    comp = {}
    for tag in COMPONENTS:
        oa = pad19(np.load(f"{OOF_DIR}/{tag}_oof_act.npy"))
        op = np.load(f"{OOF_DIR}/{tag}_oof_pt.npy").astype(np.float32, copy=False)
        srv = np.load(f"{OOF_DIR}/{tag}_oof_srv.npy").astype(np.float32, copy=False)
        ta = pad19(np.load(f"{OOF_DIR}/{tag}_test_act.npy"))
        tp = np.load(f"{OOF_DIR}/{tag}_test_pt.npy").astype(np.float32, copy=False)
        ts = np.load(f"{OOF_DIR}/{tag}_test_srv.npy").astype(np.float32, copy=False)
        comp[tag] = {"oa": oa, "op": op, "srv": srv, "ta": ta, "tp": tp, "ts": ts}
    return comp, mask, y_a, y_p, y_s, nsn, test_uid


def build_groups():
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, _ = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1
    gs = compute_global_stats_v9(train_df)
    feat = build_features_v9(train_df, is_train=True,
                             global_stats_v9=gs, raw_df=train_df)
    rally_uids = feat["rally_uid"].values
    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    return np.array([rally_to_match[r] for r in rally_uids])


def best_single(comp, mask, y_a, y_p, y_s):
    best_a = 0.0; best_a_t = ""
    best_p = 0.0; best_p_t = ""
    best_s = 0.0; best_s_t = ""
    for tag in COMPONENTS:
        f1a = fast_macro_f1(y_a[mask], comp[tag]["oa"][mask].argmax(axis=1),
                            ACTION_EVAL_LABELS, N_ACTION)
        f1p = fast_macro_f1(y_p[mask], comp[tag]["op"][mask].argmax(axis=1),
                            POINT_EVAL_LABELS, N_POINT)
        try:
            auc = roc_auc_score(y_s[mask], comp[tag]["srv"][mask])
        except Exception:
            auc = 0.5
        if f1a > best_a: best_a, best_a_t = f1a, tag
        if f1p > best_p: best_p, best_p_t = f1p, tag
        if auc > best_s: best_s, best_s_t = auc, tag
    return best_a, best_a_t, best_p, best_p_t, best_s, best_s_t


def stack(comp, key, axis=1):
    if key in ("srv", "ts"):
        return np.stack([comp[t][key] for t in COMPONENTS], axis=1).astype(np.float32)
    return np.concatenate([comp[t][key] for t in COMPONENTS], axis=1).astype(np.float32)


def main():
    t_start = time.time()
    out_tag = "meta_stack_v2_logistic"
    print(f"=== {out_tag}: logistic-regression stack diagnostic ===")
    print(f"Components: {COMPONENTS}")
    print(f"LR config: C={LR_C}  max_iter={LR_MAX_ITER}")

    comp, mask, y_a, y_p, y_s, nsn, test_uid = load_components()
    best_a, best_a_t, best_p, best_p_t, best_s, best_s_t = \
        best_single(comp, mask, y_a, y_p, y_s)
    print(f"Best F1_a: {best_a:.4f} ({best_a_t})")
    print(f"Best F1_p: {best_p:.4f} ({best_p_t})")
    print(f"Best AUC : {best_s:.4f} ({best_s_t})")

    match_all = build_groups()
    keep = mask.copy()
    print(f"Kept rows: {int(keep.sum())} / {len(keep)}")

    X_a = stack(comp, "oa")
    X_p = stack(comp, "op")
    X_s = stack(comp, "srv")
    Xt_a = stack(comp, "ta")
    Xt_p = stack(comp, "tp")
    Xt_s = stack(comp, "ts")

    n = len(y_a)
    n_test = len(test_uid)
    oof_act = np.zeros((n, N_ACTION_EVAL), dtype=np.float32)
    oof_pt = np.zeros((n, N_POINT), dtype=np.float32)
    oof_srv = np.zeros(n, dtype=np.float32)
    test_act = np.zeros((n_test, N_ACTION_EVAL), dtype=np.float32)
    test_pt = np.zeros((n_test, N_POINT), dtype=np.float32)
    test_srv = np.zeros(n_test, dtype=np.float32)

    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(np.arange(n), groups=match_all))
    fold_metrics = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        tr_idx = tr_idx[keep[tr_idx]]
        val_idx = val_idx[keep[val_idx]]
        tr_m = set(match_all[tr_idx].tolist())
        val_m = set(match_all[val_idx].tolist())
        assert not (tr_m & val_m), f"fold {fold_idx}: match overlap"
        t_fold = time.time()
        print(f"\n=== Fold {fold_idx+1}/{n_folds}  train={len(tr_idx)}  val={len(val_idx)} ===")

        # ACTION (multinomial logistic)
        lr_a = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, solver="lbfgs",
                                   random_state=RANDOM_SEED,
                                   n_jobs=-1)
        lr_a.fit(X_a[tr_idx], y_a[tr_idx])
        pa_val = lr_a.predict_proba(X_a[val_idx])
        # Map LR's reduced class space to full N_ACTION_EVAL
        pa_full = np.zeros((len(val_idx), N_ACTION_EVAL), dtype=np.float32)
        for i, c in enumerate(lr_a.classes_):
            if 0 <= int(c) < N_ACTION_EVAL:
                pa_full[:, int(c)] = pa_val[:, i]
        oof_act[val_idx] = pa_full
        pa_test = lr_a.predict_proba(Xt_a)
        pa_test_full = np.zeros((n_test, N_ACTION_EVAL), dtype=np.float32)
        for i, c in enumerate(lr_a.classes_):
            if 0 <= int(c) < N_ACTION_EVAL:
                pa_test_full[:, int(c)] = pa_test[:, i]
        test_act += pa_test_full / n_folds
        f1a = fast_macro_f1(y_a[val_idx], pa_full.argmax(axis=1),
                            ACTION_EVAL_LABELS, N_ACTION_EVAL)

        # POINT
        lr_p = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, solver="lbfgs",
                                   random_state=RANDOM_SEED,
                                   n_jobs=-1)
        lr_p.fit(X_p[tr_idx], y_p[tr_idx])
        pp_val = lr_p.predict_proba(X_p[val_idx])
        pp_full = np.zeros((len(val_idx), N_POINT), dtype=np.float32)
        for i, c in enumerate(lr_p.classes_):
            if 0 <= int(c) < N_POINT:
                pp_full[:, int(c)] = pp_val[:, i]
        oof_pt[val_idx] = pp_full
        pp_test = lr_p.predict_proba(Xt_p)
        pp_test_full = np.zeros((n_test, N_POINT), dtype=np.float32)
        for i, c in enumerate(lr_p.classes_):
            if 0 <= int(c) < N_POINT:
                pp_test_full[:, int(c)] = pp_test[:, i]
        test_pt += pp_test_full / n_folds
        f1p = fast_macro_f1(y_p[val_idx], pp_full.argmax(axis=1),
                            POINT_EVAL_LABELS, N_POINT)

        # SERVER
        lr_s = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, solver="lbfgs",
                                   random_state=RANDOM_SEED, n_jobs=-1)
        lr_s.fit(X_s[tr_idx], y_s[tr_idx])
        ps_val = lr_s.predict_proba(X_s[val_idx])[:, 1]
        oof_srv[val_idx] = ps_val.astype(np.float32)
        ps_test = lr_s.predict_proba(Xt_s)[:, 1]
        test_srv += ps_test.astype(np.float32) / n_folds
        try:
            auc = float(roc_auc_score(y_s[val_idx], ps_val))
        except Exception:
            auc = 0.5
        ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        print(f"  F1_a={f1a:.4f}  F1_p={f1p:.4f}  AUC={auc:.4f}  OV={ov:.4f}  "
              f"[{time.time()-t_fold:.1f}s]")
        fold_metrics.append({"fold": fold_idx + 1, "F1_a": f1a, "F1_p": f1p,
                             "AUC": auc, "OV": ov})

    print("\n=== meta_stack_v2 OOF (all kept rows) ===")
    f1_a = fast_macro_f1(y_a[keep], oof_act[keep].argmax(axis=1),
                          ACTION_EVAL_LABELS, N_ACTION_EVAL)
    f1_p = fast_macro_f1(y_p[keep], oof_pt[keep].argmax(axis=1),
                          POINT_EVAL_LABELS, N_POINT)
    auc = float(roc_auc_score(y_s[keep], oof_srv[keep]))
    ov = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
    print(f"  F1_a   = {f1_a:.4f}  (best single = {best_a:.4f}  Δ = {f1_a - best_a:+.4f})")
    print(f"  F1_p   = {f1_p:.4f}  (best single = {best_p:.4f}  Δ = {f1_p - best_p:+.4f})")
    print(f"  AUC    = {auc:.4f}  (best single = {best_s:.4f}  Δ = {auc - best_s:+.4f})")
    print(f"  OV     = {ov:.4f}")

    print("\n=== Stop-gate evaluation (Codex strict bar) ===")
    combined_threshold = 0.3775 + 0.003
    print(f"  per-task F1_a >= {best_a + 0.001:.4f} ? "
          f"{'PASS' if f1_a >= best_a + 0.001 else 'FAIL'}")
    print(f"  per-task F1_p >= {best_p + 0.001:.4f} ? "
          f"{'PASS' if f1_p >= best_p + 0.001 else 'FAIL'}")
    print(f"  per-task AUC  >= {best_s + 0.001:.4f} ? "
          f"{'PASS' if auc >= best_s + 0.001 else 'FAIL'}")
    print(f"  combined OV   >= {combined_threshold:.4f} ? "
          f"{'PASS' if ov >= combined_threshold else 'FAIL'}")

    # Save artifacts
    oof_act_19 = np.zeros((n, N_ACTION), dtype=np.float32)
    oof_act_19[:, :N_ACTION_EVAL] = oof_act
    test_act_19 = np.zeros((n_test, N_ACTION), dtype=np.float32)
    test_act_19[:, :N_ACTION_EVAL] = test_act
    np.save(f"{OOF_DIR}/{out_tag}_oof_act.npy", oof_act_19)
    np.save(f"{OOF_DIR}/{out_tag}_oof_pt.npy", oof_pt)
    np.save(f"{OOF_DIR}/{out_tag}_oof_srv.npy", oof_srv)
    np.save(f"{OOF_DIR}/{out_tag}_oof_mask.npy", mask)
    np.save(f"{OOF_DIR}/{out_tag}_oof_y_act.npy", y_a)
    np.save(f"{OOF_DIR}/{out_tag}_oof_y_pt.npy", y_p)
    np.save(f"{OOF_DIR}/{out_tag}_oof_y_srv.npy", y_s)
    np.save(f"{OOF_DIR}/{out_tag}_oof_nsn.npy", nsn)
    np.save(f"{OOF_DIR}/{out_tag}_test_act.npy", test_act_19)
    np.save(f"{OOF_DIR}/{out_tag}_test_pt.npy", test_pt)
    np.save(f"{OOF_DIR}/{out_tag}_test_srv.npy", test_srv)
    np.save(f"{OOF_DIR}/{out_tag}_test_rally_uid.npy", test_uid)

    meta = {
        "tag": out_tag,
        "components": COMPONENTS,
        "model": "LogisticRegression(multinomial)",
        "lr_C": LR_C,
        "lr_max_iter": LR_MAX_ITER,
        "n_folds": n_folds,
        "outer_cv": "GroupKFold(5) by match",
        "best_single_metrics": {
            "F1_a": {"value": float(best_a), "tag": best_a_t},
            "F1_p": {"value": float(best_p), "tag": best_p_t},
            "AUC": {"value": float(best_s), "tag": best_s_t},
        },
        "meta_stack_oof_metrics": {
            "F1_a": float(f1_a), "F1_p": float(f1_p),
            "AUC": float(auc), "OV": float(ov),
        },
        "fold_metrics": fold_metrics,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "submission_status": "HELD — no T3 approval implied.",
    }
    with open(f"{OOF_DIR}/{out_tag}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nTotal time: {(time.time() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
