"""train_server_head_v1 — dedicated rally-level serverGetPoint classifier.

R-006 implementation per Codex APPROVE_WITH_FIXES (2026-05-09):

- Uses `features_server_v1` (prefix-only safe features; no player IDs).
- Outer CV: GroupKFold(5) by `match`.
- Per-fold leak gates (Codex strict):
  - Fold 1 AUC > 0.80  → HARD STOP, write status, exit non-zero.
  - Fold 1 AUC 0.75–0.80 → PAUSE, write feature importance + counts-only AUC,
    exit non-zero pending Codex review.
  - Fold 1+2 mean AUC < 0.62 → STOP (signal too weak; not worth zoo intake).
  - Final AUC ≥ 0.65 AND feature importance not dominated by prefix counts /
    next-strike proxies → eligible for zoo (server-channel) intake; still
    requires T3 review.
- Output:
  - Per-row OOF server prediction shape (69712,)
  - Per-rally test prediction shape (1845,) — one per test rally.
  - Metadata json with feature importance, counts-only AUC, gates passed.
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
from features_server_v1 import (
    build_features_server_v1, build_test_per_rally_features,
    feature_names, count_only_features,
)

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")

# Standard LightGBM hyperparameters (not over-shallow — server head is a
# direct task, not a meta-learner).
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


def fast_macro_f1(*args, **kwargs):
    return 0.0  # unused; kept for symmetry


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
    out_tag = "server_head_v1"
    print(f"=== {out_tag}: dedicated rally-level serverGetPoint classifier ===")
    print(f"Hyperparameters: {HP}")

    # --- Reference metadata (from existing aligned OOF artifacts) ---
    ref = "v16_testhist_aug"
    ref_y_srv = np.load(f"{OOF_DIR}/{ref}_oof_y_srv.npy")
    ref_mask = np.load(f"{OOF_DIR}/{ref}_oof_mask.npy")
    ref_test_uid = np.load(f"{OOF_DIR}/{ref}_test_rally_uid.npy")
    print(f"Reference: y_srv shape {ref_y_srv.shape}, mask sum {int(ref_mask.sum())}, "
          f"test rallies {len(ref_test_uid)}")

    # --- Load + clean raw data ---
    print("\n--- Loading raw data ---")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, _ = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1

    # --- Build target row metadata via features_v9 (replicates standard order) ---
    print("\n--- Building target-row metadata via features_v9 (for row order + groups) ---")
    gs_v9 = compute_global_stats_v9(train_df)
    feat_v9 = build_features_v9(train_df, is_train=True,
                                 global_stats_v9=gs_v9, raw_df=train_df)
    target_rows = feat_v9[["rally_uid", "next_strikeNumber"]].reset_index(drop=True)
    rally_uids = target_rows["rally_uid"].to_numpy()
    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_all = np.array([rally_to_match[r] for r in rally_uids])
    assert len(target_rows) == len(ref_y_srv), \
        f"row count mismatch {len(target_rows)} vs {len(ref_y_srv)}"
    print(f"  target_rows: {len(target_rows)} rows, {len(np.unique(match_all))} matches")

    # --- Build server-head features ---
    print("\n--- Building server_head_v1 train features ---")
    t0 = time.time()
    X_train_df = build_features_server_v1(target_rows, train_df, is_train=True)
    fnames = feature_names()
    X_train = X_train_df[fnames].to_numpy(dtype=np.float32)
    print(f"  X_train: {X_train.shape}  [{time.time()-t0:.1f}s]")

    print("\n--- Building server_head_v1 test features (per-rally) ---")
    t0 = time.time()
    X_test_df, test_rally_uids = build_test_per_rally_features(test_df)
    X_test = X_test_df[fnames].to_numpy(dtype=np.float32)
    print(f"  X_test: {X_test.shape}  [{time.time()-t0:.1f}s]")

    # Sanity: test_rally_uids should match ref_test_uid (same set, same first-appearance order in test_new.csv)
    assert set(test_rally_uids) == set(ref_test_uid), "test rally set mismatch"
    # Reorder X_test to match ref_test_uid order
    rid_to_idx = {rid: i for i, rid in enumerate(test_rally_uids)}
    perm = np.array([rid_to_idx[r] for r in ref_test_uid])
    X_test = X_test[perm]
    test_rally_uids = ref_test_uid
    print(f"  X_test reordered to ref_test_uid: {X_test.shape}")

    y_srv = ref_y_srv.astype(np.int32)

    # --- Counts-only diagnostic AUC (Codex requirement) ---
    print("\n--- Counts-only AUC diagnostic (Codex leak check) ---")
    co_cols = count_only_features()
    co_idx = [fnames.index(c) for c in co_cols]
    X_co = X_train[:, co_idx]
    print(f"  Counts-only features: {co_cols}")
    # Use single 5-fold CV with same splits to estimate counts-only AUC
    gkf_co = GroupKFold(n_splits=5)
    co_aucs = []
    for f, (tr, val) in enumerate(gkf_co.split(np.arange(len(y_srv)), groups=match_all)):
        tr = tr[ref_mask[tr]]
        val = val[ref_mask[val]]
        m_co = train_lgb_binary(X_co[tr], y_srv[tr], X_co[val], y_srv[val],
                                 hp={**HP, "n_boost": 500, "es": 50})
        p = m_co.predict(X_co[val])
        co_aucs.append(float(roc_auc_score(y_srv[val], p)))
        print(f"    fold {f+1}: counts-only AUC = {co_aucs[-1]:.4f}")
    co_mean = float(np.mean(co_aucs))
    print(f"  Counts-only mean AUC: {co_mean:.4f}")
    if co_mean > 0.65:
        print(f"  *** WARNING: counts-only AUC > 0.65 — feature set is leak-prone. "
              "STOP and review with Codex before continuing.")
        # Save partial result and exit
        meta = {"tag": out_tag, "status": "STOPPED", "reason":
                f"counts-only AUC {co_mean:.4f} > 0.65 (leak suspected)",
                "counts_only_aucs": co_aucs, "counts_only_features": co_cols}
        with open(f"{OOF_DIR}/{out_tag}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        sys.exit(2)

    # --- Outer 5-fold CV with leak gates ---
    print("\n--- Outer GroupKFold(5) by match ---")
    n = len(y_srv)
    n_test = len(test_rally_uids)
    oof_srv = np.zeros(n, dtype=np.float32)
    test_srv_acc = np.zeros(n_test, dtype=np.float32)

    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(np.arange(n), groups=match_all))
    fold_aucs = []
    fold_metrics = []
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
        print(f"  AUC = {auc:.4f}  [{time.time()-t_fold:.1f}s]  "
              f"best_iter={m.best_iteration}")

        # Top-10 feature importance (gain)
        imp = m.feature_importance(importance_type="gain")
        order = np.argsort(-imp)[:10]
        top_feats = [(fnames[i], float(imp[i])) for i in order]
        print(f"  Top-10 feature importance:")
        for nm, val in top_feats:
            print(f"    {nm:30s}  {val:.2f}")
        fold_importances.append({"fold": fold_idx + 1, "top_10": top_feats})

        # Codex leak gate evaluation
        if fold_idx == 0:
            if auc > 0.80:
                print(f"\n*** HARD STOP: Fold 1 AUC {auc:.4f} > 0.80. "
                      "Strong leak suspected. Saving partial result and exiting.")
                meta = {"tag": out_tag, "status": "HARD_STOP",
                        "reason": f"Fold 1 AUC {auc:.4f} > 0.80",
                        "fold_aucs": fold_aucs,
                        "fold_importances": fold_importances,
                        "counts_only_aucs": co_aucs}
                with open(f"{OOF_DIR}/{out_tag}_metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)
                sys.exit(3)
            elif auc > 0.75:
                print(f"\n*** PAUSE: Fold 1 AUC {auc:.4f} in [0.75, 0.80]. "
                      "Codex requires feature importance + counts-only AUC review "
                      "before continuing. Saving partial result and exiting.")
                meta = {"tag": out_tag, "status": "PAUSE",
                        "reason": f"Fold 1 AUC {auc:.4f} in [0.75, 0.80]",
                        "fold_aucs": fold_aucs,
                        "fold_importances": fold_importances,
                        "counts_only_aucs": co_aucs}
                with open(f"{OOF_DIR}/{out_tag}_metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)
                sys.exit(4)
            print(f"  Fold 1 AUC {auc:.4f} <= 0.75 → no leak signal, continuing.")
        if fold_idx == 1:
            mean_aucs = float(np.mean(fold_aucs))
            print(f"  Fold 1+2 mean AUC: {mean_aucs:.4f}")
            if mean_aucs < 0.62:
                print(f"  Mean < 0.62 — signal too weak to justify continuing.")
                meta = {"tag": out_tag, "status": "WEAK_STOP",
                        "reason": f"Fold 1+2 mean AUC {mean_aucs:.4f} < 0.62",
                        "fold_aucs": fold_aucs,
                        "fold_importances": fold_importances,
                        "counts_only_aucs": co_aucs,
                        "counts_only_mean": co_mean}
                with open(f"{OOF_DIR}/{out_tag}_metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)
                sys.exit(5)

    # Aggregate OOF AUC
    print("\n=== server_head_v1 OOF AUC (all kept rows) ===")
    auc_full = float(roc_auc_score(y_srv[ref_mask], oof_srv[ref_mask]))
    mean_fold = float(np.mean(fold_aucs))
    print(f"  Per-fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"  Per-fold mean AUC: {mean_fold:.4f}")
    print(f"  Full OOF AUC: {auc_full:.4f}")
    print(f"  Counts-only mean AUC: {co_mean:.4f}  Δ = {auc_full - co_mean:+.4f}")
    print(f"  Best per-shot single component AUC (baseline): 0.6117 (v14_avg3)")

    # Final intake gate
    final_pass = auc_full >= 0.65
    counts_dominate = co_mean / max(auc_full, 1e-6) > 0.95
    print(f"\n=== Final intake gate ===")
    print(f"  AUC >= 0.65 ? {'PASS' if final_pass else 'FAIL'} ({auc_full:.4f})")
    print(f"  Counts NOT dominant (counts/full < 0.95) ? "
          f"{'PASS' if not counts_dominate else 'FAIL'} ({co_mean / max(auc_full, 1e-6):.3f})")
    overall_pass = final_pass and not counts_dominate
    print(f"  Overall: {'ELIGIBLE for server-channel review' if overall_pass else 'PARK'}")

    # --- Save per-row OOF + per-rally test predictions ---
    np.save(f"{OOF_DIR}/{out_tag}_oof_srv.npy", oof_srv)
    np.save(f"{OOF_DIR}/{out_tag}_oof_mask.npy", ref_mask)
    np.save(f"{OOF_DIR}/{out_tag}_oof_y_srv.npy", y_srv)
    np.save(f"{OOF_DIR}/{out_tag}_test_srv.npy", test_srv_acc)
    np.save(f"{OOF_DIR}/{out_tag}_test_rally_uid.npy", test_rally_uids)

    meta = {
        "tag": out_tag,
        "status": "COMPLETE",
        "model": "LightGBM(binary)",
        "hyperparameters": HP,
        "feature_set": "features_server_v1",
        "n_features": len(fnames),
        "n_train_rows": int(n),
        "n_test_rallies": int(n_test),
        "n_folds": n_folds,
        "outer_cv": "GroupKFold(5) by match",
        "fold_aucs": fold_aucs,
        "fold_aucs_mean": mean_fold,
        "full_oof_auc": auc_full,
        "counts_only_aucs": co_aucs,
        "counts_only_mean_auc": co_mean,
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

    print(f"\nSaved artifacts:")
    print(f"  oof_predictions/{out_tag}_oof_srv.npy / _oof_mask / _oof_y_srv")
    print(f"  oof_predictions/{out_tag}_test_srv.npy / _test_rally_uid.npy")
    print(f"  oof_predictions/{out_tag}_metadata.json")
    print(f"\nTotal time: {(time.time() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
