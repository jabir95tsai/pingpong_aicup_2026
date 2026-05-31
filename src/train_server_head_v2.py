"""train_server_head_v2 — v1 + last-K shots one-hot features.

Same pipeline + leak gates as train_server_head_v1, but uses
features_server_v2 (v1 + lag block).

If Fold 1 AUC > 0.80: HARD STOP (leak suspected).
If Fold 1 AUC 0.75-0.80: PAUSE.
If Fold 1+2 mean AUC < 0.62: WEAK STOP.
If final AUC >= 0.65 + counts not dominant: ELIGIBLE.
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
from features_server_v2 import (
    build_features_server_v2, build_test_per_rally_features_v2,
    feature_names, count_only_features,
)

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")

HP = {
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "learning_rate": 0.03,
    "n_boost": 2000,
    "es": 100,
    "seed": RANDOM_SEED,
}


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
    return lgb.train(params, train_set, num_boost_round=hp["n_boost"],
                     valid_sets=[val_set],
                     callbacks=[lgb.early_stopping(hp["es"], verbose=False)])


def main():
    t_start = time.time()
    out_tag = "server_head_v2"
    print(f"=== {out_tag}: server-head v2 with last-3 shots one-hot features ===")
    print(f"Hyperparameters: {HP}")

    ref = "v16_testhist_aug"
    ref_y_srv = np.load(f"{OOF_DIR}/{ref}_oof_y_srv.npy")
    ref_mask = np.load(f"{OOF_DIR}/{ref}_oof_mask.npy")
    ref_test_uid = np.load(f"{OOF_DIR}/{ref}_test_rally_uid.npy")
    print(f"Reference: y_srv shape {ref_y_srv.shape}, mask sum {int(ref_mask.sum())}, "
          f"test rallies {len(ref_test_uid)}")

    print("\n--- Loading raw data ---")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, _ = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1

    print("\n--- Building target-row metadata via features_v9 ---")
    gs_v9 = compute_global_stats_v9(train_df)
    feat_v9 = build_features_v9(train_df, is_train=True,
                                 global_stats_v9=gs_v9, raw_df=train_df)
    target_rows = feat_v9[["rally_uid", "next_strikeNumber"]].reset_index(drop=True)
    rally_uids = target_rows["rally_uid"].to_numpy()
    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_all = np.array([rally_to_match[r] for r in rally_uids])
    assert len(target_rows) == len(ref_y_srv)
    print(f"  target_rows: {len(target_rows)} rows, {len(np.unique(match_all))} matches")

    print("\n--- Building server_head_v2 train features ---")
    t0 = time.time()
    X_train_df = build_features_server_v2(target_rows, train_df, is_train=True)
    fnames = feature_names()
    X_train = X_train_df[fnames].to_numpy(dtype=np.float32)
    print(f"  X_train: {X_train.shape}  [{time.time()-t0:.1f}s]")

    print("\n--- Building server_head_v2 test features (per-rally) ---")
    t0 = time.time()
    X_test_df, test_rally_uids = build_test_per_rally_features_v2(test_df)
    X_test = X_test_df[fnames].to_numpy(dtype=np.float32)
    print(f"  X_test: {X_test.shape}  [{time.time()-t0:.1f}s]")

    assert set(test_rally_uids) == set(ref_test_uid)
    rid_to_idx = {rid: i for i, rid in enumerate(test_rally_uids)}
    perm = np.array([rid_to_idx[r] for r in ref_test_uid])
    X_test = X_test[perm]
    test_rally_uids = ref_test_uid

    y_srv = ref_y_srv.astype(np.int32)

    # Counts-only diagnostic
    print("\n--- Counts-only AUC diagnostic ---")
    co_cols = count_only_features()
    co_idx = [fnames.index(c) for c in co_cols]
    X_co = X_train[:, co_idx]
    print(f"  Counts-only features: {co_cols}")
    gkf_co = GroupKFold(n_splits=5)
    co_aucs = []
    for f, (tr, val) in enumerate(gkf_co.split(np.arange(len(y_srv)), groups=match_all)):
        tr = tr[ref_mask[tr]]; val = val[ref_mask[val]]
        m_co = train_lgb_binary(X_co[tr], y_srv[tr], X_co[val], y_srv[val],
                                 hp={**HP, "n_boost": 500, "es": 50})
        p = m_co.predict(X_co[val])
        co_aucs.append(float(roc_auc_score(y_srv[val], p)))
        print(f"    fold {f+1}: counts-only AUC = {co_aucs[-1]:.4f}")
    co_mean = float(np.mean(co_aucs))
    print(f"  Counts-only mean AUC: {co_mean:.4f}")
    if co_mean > 0.65:
        print("  *** WARNING: counts-only AUC > 0.65 → leak suspected. STOP.")
        sys.exit(2)

    print("\n--- Outer GroupKFold(5) by match ---")
    n = len(y_srv)
    n_test = len(test_rally_uids)
    oof_srv = np.zeros(n, dtype=np.float32)
    test_srv_acc = np.zeros(n_test, dtype=np.float32)

    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(np.arange(n), groups=match_all))
    fold_aucs = []
    fold_importances = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        tr_idx = tr_idx[ref_mask[tr_idx]]
        val_idx = val_idx[ref_mask[val_idx]]
        tr_m = set(match_all[tr_idx].tolist())
        val_m = set(match_all[val_idx].tolist())
        assert not (tr_m & val_m), f"fold {fold_idx}: match overlap"

        t_fold = time.time()
        print(f"\n=== Fold {fold_idx+1}/{n_folds}  train={len(tr_idx)}  val={len(val_idx)} ===")

        m = train_lgb_binary(X_train[tr_idx], y_srv[tr_idx],
                              X_train[val_idx], y_srv[val_idx], HP)
        ps_val = m.predict(X_train[val_idx])
        oof_srv[val_idx] = ps_val.astype(np.float32)
        ps_test = m.predict(X_test)
        test_srv_acc += ps_test.astype(np.float32) / n_folds
        auc = float(roc_auc_score(y_srv[val_idx], ps_val))
        fold_aucs.append(auc)
        print(f"  AUC = {auc:.4f}  [{time.time()-t_fold:.1f}s]  best_iter={m.best_iteration}")

        imp = m.feature_importance(importance_type="gain")
        order = np.argsort(-imp)[:15]
        top_feats = [(fnames[i], float(imp[i])) for i in order]
        print(f"  Top-15 feature importance:")
        for nm, val in top_feats:
            print(f"    {nm:32s}  {val:.2f}")
        fold_importances.append({"fold": fold_idx + 1, "top_15": top_feats})

        if fold_idx == 0:
            if auc > 0.80:
                print(f"\n*** HARD STOP: Fold 1 AUC {auc:.4f} > 0.80")
                sys.exit(3)
            elif auc > 0.75:
                print(f"\n*** PAUSE: Fold 1 AUC {auc:.4f} in [0.75, 0.80]")
                sys.exit(4)
            print(f"  Fold 1 AUC {auc:.4f} <= 0.75 → no leak signal, continuing.")
        if fold_idx == 1:
            mean_aucs = float(np.mean(fold_aucs))
            print(f"  Fold 1+2 mean AUC: {mean_aucs:.4f}")
            if mean_aucs < 0.62:
                print(f"  Mean < 0.62 — signal too weak.")
                meta = {"tag": out_tag, "status": "WEAK_STOP",
                        "reason": f"Fold 1+2 mean AUC {mean_aucs:.4f} < 0.62",
                        "fold_aucs": fold_aucs,
                        "fold_importances": fold_importances,
                        "counts_only_aucs": co_aucs,
                        "counts_only_mean": co_mean}
                with open(f"{OOF_DIR}/{out_tag}_metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)
                sys.exit(5)

    print("\n=== server_head_v2 OOF AUC (all kept rows) ===")
    auc_full = float(roc_auc_score(y_srv[ref_mask], oof_srv[ref_mask]))
    mean_fold = float(np.mean(fold_aucs))
    print(f"  Per-fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"  Per-fold mean: {mean_fold:.4f}")
    print(f"  Full OOF AUC: {auc_full:.4f}")
    print(f"  Counts-only mean AUC: {co_mean:.4f}  Δ = {auc_full - co_mean:+.4f}")
    print(f"  Best per-shot single component AUC (baseline): 0.6117 (v14_avg3)")

    final_pass = auc_full >= 0.65
    counts_dominate = co_mean / max(auc_full, 1e-6) > 0.95
    print(f"\n=== Final intake gate ===")
    print(f"  AUC >= 0.65 ? {'PASS' if final_pass else 'FAIL'} ({auc_full:.4f})")
    print(f"  Counts NOT dominant ? "
          f"{'PASS' if not counts_dominate else 'FAIL'} ({co_mean / max(auc_full, 1e-6):.3f})")
    overall_pass = final_pass and not counts_dominate
    print(f"  Overall: {'ELIGIBLE for server-channel review' if overall_pass else 'PARK'}")

    np.save(f"{OOF_DIR}/{out_tag}_oof_srv.npy", oof_srv)
    np.save(f"{OOF_DIR}/{out_tag}_oof_mask.npy", ref_mask)
    np.save(f"{OOF_DIR}/{out_tag}_oof_y_srv.npy", y_srv)
    np.save(f"{OOF_DIR}/{out_tag}_test_srv.npy", test_srv_acc)
    np.save(f"{OOF_DIR}/{out_tag}_test_rally_uid.npy", test_rally_uids)

    meta = {
        "tag": out_tag, "status": "COMPLETE",
        "model": "LightGBM(binary)",
        "hyperparameters": HP,
        "feature_set": "features_server_v2 (v1 + last-3-shots one-hot)",
        "n_features": len(fnames),
        "n_train_rows": int(n), "n_test_rallies": int(n_test),
        "n_folds": n_folds, "outer_cv": "GroupKFold(5) by match",
        "fold_aucs": fold_aucs, "fold_aucs_mean": mean_fold,
        "full_oof_auc": auc_full,
        "counts_only_aucs": co_aucs, "counts_only_mean_auc": co_mean,
        "fold_importances": fold_importances,
        "intake_gate": {
            "auc_ge_065": bool(final_pass),
            "counts_not_dominant": bool(not counts_dominate),
            "overall_eligible": bool(overall_pass),
        },
        "submission_status": "HELD — server-channel integration requires T3 review.",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{OOF_DIR}/{out_tag}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nSaved artifacts.")
    print(f"Total time: {(time.time() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
