"""meta_stack — LightGBM stacking meta-learner over 14 component OOF probability arrays.

R-005 implementation per Codex APPROVE_WITH_FIXES (2026-05-09) + Jabir constraints:

- Outer split: GroupKFold(5) by `match`, with hard no-overlap assertion. Treated
  as an OOF diagnostic, not a leakage-free LB-transfer proof.
- Inputs: ONLY component probability arrays. No rally_uid, match, player IDs,
  row index, fold id, target labels, next_strikeNumber, or submission-derived
  labels.
- Per-task models: shallow / regularized LightGBM (num_leaves=8,
  min_data_in_leaf=200, subsampling, early stopping). One model per task.
- Mask-false rows excluded from outer CV (defensive — current mask is fully True).
- Output: standalone diagnostic artifact set in `oof_predictions/meta_stack_*`
  + `oof_predictions/meta_stack_metadata.json`. NOT added to the zoo as a
  normal component without a separate Codex review.
- Stop gates (per Codex strict revision):
  - Per-task: meta OOF metric >= exact best-single component metric + 0.001.
  - Combined: meta OV >= zoo_v10 elig1 OOF (0.3775) + 0.003 = 0.3805.

This script does NOT approve any submission. Any meta_stack-based submission
requires a separate T3 artifact review.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, SUBMISSION_DIR, PROJECT_ROOT, RANDOM_SEED
from data_cleaning import clean_data
from features_v9 import build_features_v9, compute_global_stats_v9

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")

# 14 menu components per Codex / T1 correlation analysis.
COMPONENTS = [
    "v16_testhist_aug", "v16_avg3", "v16_seed1", "v16_seed2",
    "v14_avg3", "v14_seed0", "v14_seed1", "v14_seed2", "v14_recvhand",
    "v12_5f",
    "v11", "v11plus", "v11_aug",
    "v13",
]

N_ACTION = 19
N_ACTION_EVAL = 15  # train action labels are 0..14
N_POINT = 10
ACTION_EVAL_LABELS = list(range(15))
POINT_EVAL_LABELS = list(range(10))

# Hyperparameters (Codex: shallow / regularized).
HP = {
    "num_leaves": 8,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "learning_rate": 0.05,
    "n_boost": 1000,
    "es": 80,
    "seed": RANDOM_SEED,
}


def pad19(arr: np.ndarray) -> np.ndarray:
    if arr.shape[1] >= N_ACTION:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], N_ACTION), dtype=np.float32)
    out[:, :arr.shape[1]] = arr
    return out


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
                  labels: list, n_total: int) -> float:
    cm = np.bincount(y_true.astype(np.int64) * n_total + y_pred.astype(np.int64),
                     minlength=n_total * n_total).reshape(n_total, n_total)
    col_sum = cm.sum(axis=0)
    row_sum = cm.sum(axis=1)
    diag = np.diag(cm)
    f1s = []
    for c in labels:
        tp = diag[c]
        fp = col_sum[c] - tp
        fn = row_sum[c] - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom <= 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


def load_components_and_meta():
    """Load all 14 components + reference metadata, with hard alignment checks."""
    print(f"Loading {len(COMPONENTS)} components from {OOF_DIR}")
    ref = "v16_testhist_aug"
    mask = np.load(f"{OOF_DIR}/{ref}_oof_mask.npy")
    y_a = np.load(f"{OOF_DIR}/{ref}_oof_y_act.npy")
    y_p = np.load(f"{OOF_DIR}/{ref}_oof_y_pt.npy")
    y_s = np.load(f"{OOF_DIR}/{ref}_oof_y_srv.npy")
    nsn = np.load(f"{OOF_DIR}/{ref}_oof_nsn.npy")
    test_uid = np.load(f"{OOF_DIR}/{ref}_test_rally_uid.npy")

    assert mask.sum() == len(y_a), f"mask sum {mask.sum()} != y len {len(y_a)}"
    assert len(y_a) == 69712, f"unexpected n_train {len(y_a)}"

    comp_data = {}
    for tag in COMPONENTS:
        oa = pad19(np.load(f"{OOF_DIR}/{tag}_oof_act.npy"))
        op = np.load(f"{OOF_DIR}/{tag}_oof_pt.npy").astype(np.float32, copy=False)
        srv = np.load(f"{OOF_DIR}/{tag}_oof_srv.npy").astype(np.float32, copy=False)
        ta = pad19(np.load(f"{OOF_DIR}/{tag}_test_act.npy"))
        tp = np.load(f"{OOF_DIR}/{tag}_test_pt.npy").astype(np.float32, copy=False)
        ts = np.load(f"{OOF_DIR}/{tag}_test_srv.npy").astype(np.float32, copy=False)

        m = np.load(f"{OOF_DIR}/{tag}_oof_mask.npy")
        assert np.array_equal(m, mask), f"mask mismatch for {tag}"
        for suf, ref_arr in [("oof_y_act", y_a), ("oof_y_pt", y_p),
                              ("oof_y_srv", y_s), ("oof_nsn", nsn)]:
            path = f"{OOF_DIR}/{tag}_{suf}.npy"
            if os.path.exists(path):
                arr = np.load(path)
                assert np.array_equal(arr, ref_arr), f"{suf} mismatch for {tag}"
        tu_path = f"{OOF_DIR}/{tag}_test_rally_uid.npy"
        if os.path.exists(tu_path):
            tu = np.load(tu_path)
            assert np.array_equal(tu, test_uid), f"test_rally_uid mismatch for {tag}"

        comp_data[tag] = {"oa": oa, "op": op, "srv": srv,
                           "ta": ta, "tp": tp, "ts": ts}

    print("All component arrays aligned (mask, y, nsn, test_uid byte-equal).")
    return comp_data, mask, y_a, y_p, y_s, nsn, test_uid


def build_groups():
    """Reconstruct match group per OOF row by replicating the v14 row order."""
    print("\nReconstructing match-group array (replicates build_features_v9 row order)...")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, _ = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1
    gs = compute_global_stats_v9(train_df)
    feat = build_features_v9(train_df, is_train=True,
                             global_stats_v9=gs, raw_df=train_df)
    rally_uids = feat["rally_uid"].values
    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_all = np.array([rally_to_match.get(r, -1) for r in rally_uids])
    assert (match_all >= 0).all(), \
        f"unmapped rallies: {(match_all < 0).sum()}"
    assert len(match_all) == 69712, f"unexpected n_rows {len(match_all)}"
    n_unique_matches = len(np.unique(match_all))
    print(f"  match_all: {len(match_all)} rows, {n_unique_matches} unique matches")
    return match_all


def best_single_baselines(comp_data, mask, y_a, y_p, y_s):
    print("\nBest-single baselines per task (used for stop gates):")
    best_a_f1 = 0.0; best_a_tag = ""
    best_p_f1 = 0.0; best_p_tag = ""
    best_s_auc = 0.0; best_s_tag = ""
    for tag in COMPONENTS:
        oa = comp_data[tag]["oa"]
        op = comp_data[tag]["op"]
        srv = comp_data[tag]["srv"]
        f1_a = fast_macro_f1(y_a[mask], oa[mask].argmax(axis=1),
                             ACTION_EVAL_LABELS, N_ACTION)
        f1_p = fast_macro_f1(y_p[mask], op[mask].argmax(axis=1),
                             POINT_EVAL_LABELS, N_POINT)
        try:
            auc = roc_auc_score(y_s[mask], srv[mask])
        except Exception:
            auc = 0.5
        if f1_a > best_a_f1:
            best_a_f1, best_a_tag = f1_a, tag
        if f1_p > best_p_f1:
            best_p_f1, best_p_tag = f1_p, tag
        if auc > best_s_auc:
            best_s_auc, best_s_tag = auc, tag
        print(f"  {tag:18s}  F1_a={f1_a:.4f}  F1_p={f1_p:.4f}  AUC={auc:.4f}")
    print(f"  Best F1_a: {best_a_f1:.4f} ({best_a_tag})")
    print(f"  Best F1_p: {best_p_f1:.4f} ({best_p_tag})")
    print(f"  Best AUC : {best_s_auc:.4f} ({best_s_tag})")
    return best_a_f1, best_a_tag, best_p_f1, best_p_tag, best_s_auc, best_s_tag


def stack_features(comp_data, key):
    if key == "srv":
        return np.stack([comp_data[t][key] for t in COMPONENTS], axis=1).astype(np.float32)
    return np.concatenate([comp_data[t][key] for t in COMPONENTS], axis=1).astype(np.float32)


def stack_test(comp_data, key):
    if key == "ts":
        return np.stack([comp_data[t][key] for t in COMPONENTS], axis=1).astype(np.float32)
    return np.concatenate([comp_data[t][key] for t in COMPONENTS], axis=1).astype(np.float32)


def train_lgb_multiclass(X_tr, y_tr, X_val, y_val, n_classes, hp):
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "num_leaves": hp["num_leaves"],
        "min_data_in_leaf": hp["min_data_in_leaf"],
        "feature_fraction": hp["feature_fraction"],
        "bagging_fraction": hp["bagging_fraction"],
        "bagging_freq": hp["bagging_freq"],
        "learning_rate": hp["learning_rate"],
        "verbosity": -1,
        "seed": hp["seed"],
    }
    model = lgb.train(params, train_set, num_boost_round=hp["n_boost"],
                      valid_sets=[val_set],
                      callbacks=[lgb.early_stopping(hp["es"], verbose=False)])
    return model


def train_lgb_binary(X_tr, y_tr, X_val, y_val, hp):
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": hp["num_leaves"],
        "min_data_in_leaf": hp["min_data_in_leaf"],
        "feature_fraction": hp["feature_fraction"],
        "bagging_fraction": hp["bagging_fraction"],
        "bagging_freq": hp["bagging_freq"],
        "learning_rate": hp["learning_rate"],
        "verbosity": -1,
        "seed": hp["seed"],
    }
    model = lgb.train(params, train_set, num_boost_round=hp["n_boost"],
                      valid_sets=[val_set],
                      callbacks=[lgb.early_stopping(hp["es"], verbose=False)])
    return model


def main():
    t_start = time.time()
    out_tag = "meta_stack"
    print(f"=== {out_tag}: stacking meta-learner over {len(COMPONENTS)} components ===")
    print(f"Components: {COMPONENTS}")
    print(f"Hyperparameters: {HP}")

    comp_data, mask, y_a, y_p, y_s, nsn, test_uid = load_components_and_meta()
    best_a_f1, best_a_tag, best_p_f1, best_p_tag, best_s_auc, best_s_tag = \
        best_single_baselines(comp_data, mask, y_a, y_p, y_s)

    match_all = build_groups()
    assert len(match_all) == len(y_a), \
        f"match_all len {len(match_all)} != y len {len(y_a)}"

    # Codex constraint #3: exclude rows where reference oof_mask is False.
    keep = mask.copy()
    print(f"\nKept rows (mask=True): {int(keep.sum())} / {len(keep)}")

    X_a = stack_features(comp_data, "oa")
    X_p = stack_features(comp_data, "op")
    X_s = stack_features(comp_data, "srv")
    Xt_a = stack_test(comp_data, "ta")
    Xt_p = stack_test(comp_data, "tp")
    Xt_s = stack_test(comp_data, "ts")
    print(f"\nFeature shapes: X_a={X_a.shape}  X_p={X_p.shape}  X_s={X_s.shape}")
    print(f"Test shapes:    Xt_a={Xt_a.shape}  Xt_p={Xt_p.shape}  Xt_s={Xt_s.shape}")

    n = len(y_a)
    n_test = len(test_uid)
    oof_act_15 = np.zeros((n, N_ACTION_EVAL), dtype=np.float32)
    oof_pt = np.zeros((n, N_POINT), dtype=np.float32)
    oof_srv = np.zeros(n, dtype=np.float32)
    test_act_15 = np.zeros((n_test, N_ACTION_EVAL), dtype=np.float32)
    test_pt = np.zeros((n_test, N_POINT), dtype=np.float32)
    test_srv = np.zeros(n_test, dtype=np.float32)

    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(np.arange(n), groups=match_all))

    fold_metrics = []
    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        # Codex constraint #3: exclude mask-false rows.
        tr_idx = tr_idx[keep[tr_idx]]
        val_idx = val_idx[keep[val_idx]]
        # Codex constraint #1: assert no match overlap.
        tr_matches = set(match_all[tr_idx].tolist())
        val_matches = set(match_all[val_idx].tolist())
        overlap = tr_matches & val_matches
        assert len(overlap) == 0, f"fold {fold_idx}: match overlap {overlap}"

        t_fold = time.time()
        print(f"\n=== Fold {fold_idx+1}/{n_folds}  train={len(tr_idx)}  val={len(val_idx)} ===")

        # ACTION (15-class targets)
        m_act = train_lgb_multiclass(X_a[tr_idx], y_a[tr_idx],
                                      X_a[val_idx], y_a[val_idx],
                                      n_classes=N_ACTION_EVAL, hp=HP)
        pa_val = m_act.predict(X_a[val_idx])
        oof_act_15[val_idx] = pa_val.astype(np.float32)
        pa_test = m_act.predict(Xt_a)
        test_act_15 += pa_test.astype(np.float32) / n_folds
        f1a = fast_macro_f1(y_a[val_idx], pa_val.argmax(axis=1),
                            ACTION_EVAL_LABELS, N_ACTION_EVAL)

        # POINT
        m_pt = train_lgb_multiclass(X_p[tr_idx], y_p[tr_idx],
                                     X_p[val_idx], y_p[val_idx],
                                     n_classes=N_POINT, hp=HP)
        pp_val = m_pt.predict(X_p[val_idx])
        oof_pt[val_idx] = pp_val.astype(np.float32)
        pp_test = m_pt.predict(Xt_p)
        test_pt += pp_test.astype(np.float32) / n_folds
        f1p = fast_macro_f1(y_p[val_idx], pp_val.argmax(axis=1),
                            POINT_EVAL_LABELS, N_POINT)

        # SERVER
        m_srv = train_lgb_binary(X_s[tr_idx], y_s[tr_idx],
                                  X_s[val_idx], y_s[val_idx], hp=HP)
        ps_val = m_srv.predict(X_s[val_idx])
        oof_srv[val_idx] = ps_val.astype(np.float32)
        ps_test = m_srv.predict(Xt_s)
        test_srv += ps_test.astype(np.float32) / n_folds
        try:
            auc = float(roc_auc_score(y_s[val_idx], ps_val))
        except Exception:
            auc = 0.5

        ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
        print(f"  F1_a={f1a:.4f}  F1_p={f1p:.4f}  AUC={auc:.4f}  OV={ov:.4f}  "
              f"[{time.time()-t_fold:.1f}s]")
        fold_metrics.append({"fold": fold_idx + 1, "F1_a": f1a, "F1_p": f1p,
                             "AUC": auc, "OV": ov,
                             "n_train": len(tr_idx), "n_val": len(val_idx)})

    # Aggregate OOF metrics on all kept rows.
    print("\n=== meta_stack OOF (all kept rows) ===")
    f1_a_oof = fast_macro_f1(y_a[keep], oof_act_15[keep].argmax(axis=1),
                              ACTION_EVAL_LABELS, N_ACTION_EVAL)
    f1_p_oof = fast_macro_f1(y_p[keep], oof_pt[keep].argmax(axis=1),
                              POINT_EVAL_LABELS, N_POINT)
    auc_oof = float(roc_auc_score(y_s[keep], oof_srv[keep]))
    ov_oof = 0.4 * f1_a_oof + 0.4 * f1_p_oof + 0.2 * auc_oof
    print(f"  F1_a   = {f1_a_oof:.4f}  (best single = {best_a_f1:.4f}  "
          f"Δ = {f1_a_oof - best_a_f1:+.4f})")
    print(f"  F1_p   = {f1_p_oof:.4f}  (best single = {best_p_f1:.4f}  "
          f"Δ = {f1_p_oof - best_p_f1:+.4f})")
    print(f"  AUC    = {auc_oof:.4f}  (best single = {best_s_auc:.4f}  "
          f"Δ = {auc_oof - best_s_auc:+.4f})")
    print(f"  OV     = {ov_oof:.4f}")

    # Stop-gate evaluation (Codex strict bar).
    print("\n=== Stop-gate evaluation (Codex strict bar) ===")
    zoo_v10_elig1_ov = 0.3775  # RESULTS §22
    combined_threshold = zoo_v10_elig1_ov + 0.003
    f1a_pass = f1_a_oof >= best_a_f1 + 0.001
    f1p_pass = f1_p_oof >= best_p_f1 + 0.001
    auc_pass = auc_oof >= best_s_auc + 0.001
    combined_pass = ov_oof >= combined_threshold
    print(f"  per-task F1_a >= {best_a_f1 + 0.001:.4f} ? "
          f"{'PASS' if f1a_pass else 'FAIL'}")
    print(f"  per-task F1_p >= {best_p_f1 + 0.001:.4f} ? "
          f"{'PASS' if f1p_pass else 'FAIL'}")
    print(f"  per-task AUC  >= {best_s_auc + 0.001:.4f} ? "
          f"{'PASS' if auc_pass else 'FAIL'}")
    print(f"  combined OV   >= {combined_threshold:.4f} ? "
          f"{'PASS' if combined_pass else 'FAIL'}")

    # Save artifacts (pad action 15 -> 19 for blender compatibility).
    oof_act_19 = np.zeros((n, N_ACTION), dtype=np.float32)
    oof_act_19[:, :N_ACTION_EVAL] = oof_act_15
    test_act_19 = np.zeros((n_test, N_ACTION), dtype=np.float32)
    test_act_19[:, :N_ACTION_EVAL] = test_act_15

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

    # Optional submission CSV (held; not approved for upload by this script).
    sub = pd.DataFrame({
        "rally_uid": test_uid,
        "actionId": test_act_19.argmax(axis=1),
        "pointId":  test_pt.argmax(axis=1),
        "serverGetPoint": test_srv,
    })
    sub_path = os.path.join(SUBMISSION_DIR, f"submission_{out_tag}.csv")
    sub.to_csv(sub_path, index=False, lineterminator="\n")
    print(f"\nSubmission CSV (HELD, do not upload without T3 review): {sub_path}")

    # Metadata json (Codex requirement #6).
    meta = {
        "tag": out_tag,
        "components": COMPONENTS,
        "n_components": len(COMPONENTS),
        "hyperparameters": HP,
        "n_folds": n_folds,
        "outer_cv": "GroupKFold(5) by match",
        "outer_cv_optimism_caveat": ("Stack OOF on per-row OOF features can be "
                                       "optimistic; not a fully nested stack and "
                                       "not a clean LB-transfer proof."),
        "n_train_rows": int(n),
        "n_train_kept": int(keep.sum()),
        "n_test_rows": int(n_test),
        "best_single_metrics": {
            "F1_a": {"value": float(best_a_f1), "tag": best_a_tag},
            "F1_p": {"value": float(best_p_f1), "tag": best_p_tag},
            "AUC": {"value": float(best_s_auc), "tag": best_s_tag},
        },
        "meta_stack_oof_metrics": {
            "F1_a": float(f1_a_oof),
            "F1_p": float(f1_p_oof),
            "AUC": float(auc_oof),
            "OV": float(ov_oof),
        },
        "stop_gates": {
            "per_task_threshold_delta": 0.001,
            "combined_threshold": float(combined_threshold),
            "f1_a_pass": bool(f1a_pass),
            "f1_p_pass": bool(f1p_pass),
            "auc_pass": bool(auc_pass),
            "combined_pass": bool(combined_pass),
        },
        "fold_metrics": fold_metrics,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "submission_status": "HELD — no T3 approval implied by this script.",
    }
    with open(f"{OOF_DIR}/{out_tag}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nSaved artifacts:")
    print(f"  oof_predictions/{out_tag}_oof_*.npy (8 files)")
    print(f"  oof_predictions/{out_tag}_test_*.npy (4 files)")
    print(f"  oof_predictions/{out_tag}_metadata.json")
    print(f"  {sub_path} (HELD)")
    print(f"\nTotal time: {(time.time() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
