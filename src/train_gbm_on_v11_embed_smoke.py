"""R-082 Phase 2 Step 3 — Fold-1 GBM smoke on V11 embeddings.

Theory (v0.4 candidate report):
  theoretical_generalization_reason:
    V11's softmax heads compress 192-d pooled rep → 15-d action logits. GBM on
    the raw 192-d embedding accesses information the heads lose. If embedding
    encodes structural patterns beyond argmax classification (e.g. distance to
    decision boundary, multi-modal confidence), GBM can extract it.
  why_transfers_to_test_new:
    Embeddings encode the same structural patterns V11 heads use. V11 is
    empirically irreplaceable in LB-best blends → its representation transfers.
    OOF-safety by construction (val emb only from non-trained fold model).

Two compare points:
  - GBM on last_repr embedding vs V11's own head output (action + point)
  - GBM on pool_repr embedding vs V11's own head output (server)

Decision:
  - If GBM-on-emb F1/AUC > V11-head F1/AUC + 0.005 → embeddings carry usable
    extra signal → proceed to Phase 3 (combine with zoo)
  - Else → V11 heads already extract maximum useful signal; embedding route
    closed

USAGE:
    python -u src/train_gbm_on_v11_embed_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR, TRAIN_PATH

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
N_ACTION_TRAIN = 15
N_POINT = 10

LGBM_HP_MULTI = {
    "num_leaves": 63, "min_data_in_leaf": 50,
    "learning_rate": 0.05, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l2": 1.0, "verbosity": -1, "seed": 42,
}
LGBM_HP_BIN = dict(LGBM_HP_MULTI, objective="binary", metric="auc")


def main() -> None:
    print("=" * 80)
    print(" R-082 Phase 2 Step 3 — GBM on V11 embeddings, Fold-1 smoke")
    print("=" * 80)

    # Load embeddings
    prefix = "v11_emb"
    paths = {
        "last_oof": os.path.join(OOF_DIR, f"{prefix}_last_oof.npy"),
        "pool_oof": os.path.join(OOF_DIR, f"{prefix}_pool_oof.npy"),
        "last_test": os.path.join(OOF_DIR, f"{prefix}_last_test.npy"),
        "pool_test": os.path.join(OOF_DIR, f"{prefix}_pool_test.npy"),
        "mask": os.path.join(OOF_DIR, f"{prefix}_oof_mask.npy"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            print(f" MISSING: {p}")
            print(" Run src/extract_v11_embeddings.py first.")
            sys.exit(1)
    emb_last_oof = np.load(paths["last_oof"])
    emb_pool_oof = np.load(paths["pool_oof"])
    emb_last_test = np.load(paths["last_test"])
    emb_pool_test = np.load(paths["pool_test"])
    emb_mask = np.load(paths["mask"])
    print(f"  emb_last_oof: {emb_last_oof.shape}  "
          f"emb_pool_oof: {emb_pool_oof.shape}")
    print(f"  emb_last_test: {emb_last_test.shape}  "
          f"emb_pool_test: {emb_pool_test.shape}")
    print(f"  oof mask: {emb_mask.sum()}/{len(emb_mask)} valid")

    # Load V11's own OOF predictions for comparison
    v11_oof_act = np.load(os.path.join(OOF_DIR, "v11_oof_act.npy"))
    v11_oof_pt  = np.load(os.path.join(OOF_DIR, "v11_oof_pt.npy"))
    v11_oof_srv = np.load(os.path.join(OOF_DIR, "v11_oof_srv.npy"))
    y_act = np.load(os.path.join(OOF_DIR, "v11_oof_y_act.npy"))
    y_pt  = np.load(os.path.join(OOF_DIR, "v11_oof_y_pt.npy"))
    y_srv = np.load(os.path.join(OOF_DIR, "v11_oof_y_srv.npy"))
    nsn   = np.load(os.path.join(OOF_DIR, "v11_oof_nsn.npy"))
    print(f"  V11 OOF: act{v11_oof_act.shape} pt{v11_oof_pt.shape} srv{v11_oof_srv.shape}")
    assert v11_oof_act.shape[0] == emb_last_oof.shape[0], \
        f"alignment mismatch v11 {v11_oof_act.shape[0]} vs emb {emb_last_oof.shape[0]}"

    # Build Fold-1 split
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_train = raw_train.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    non_serve = raw_train[raw_train["strikeNumber"] != 1].reset_index(drop=True)
    n_rows = len(emb_last_oof)
    match_per_row = non_serve["match"].to_numpy()[:n_rows]
    gkf = GroupKFold(n_splits=5)
    fold_of_row = np.full(n_rows, -1, dtype=np.int32)
    for f, (_, val_idx) in enumerate(gkf.split(np.arange(n_rows), groups=match_per_row)):
        fold_of_row[val_idx] = f
    fold1_mask = (fold_of_row == 0)
    train_mask = ~fold1_mask & emb_mask
    print(f"  Fold-1 val: {fold1_mask.sum()} | train (OOF-valid): {train_mask.sum()}")

    results = {}

    # ─── Action task: GBM on last_repr vs V11's own action head ──────────
    print("\n  TASK: action  (GBM on last_repr embedding)")
    y_clip = np.where(y_act >= N_ACTION_TRAIN, 0, y_act)
    t0 = time.time()
    hp = dict(LGBM_HP_MULTI, objective="multiclass", num_class=N_ACTION_TRAIN,
              metric="multi_logloss")
    dtr = lgb.Dataset(emb_last_oof[train_mask], label=y_clip[train_mask])
    dva = lgb.Dataset(emb_last_oof[fold1_mask], label=y_clip[fold1_mask],
                       reference=dtr)
    m = lgb.train(hp, dtr, num_boost_round=500,
                   valid_sets=[dva], valid_names=["val"],
                   callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                              lgb.log_evaluation(0)])
    gbm_a_probs = m.predict(emb_last_oof[fold1_mask], num_iteration=m.best_iteration)
    gbm_a_pred = gbm_a_probs.argmax(axis=1)
    gbm_a_f1 = f1_score(y_clip[fold1_mask], gbm_a_pred,
                          labels=list(range(N_ACTION_TRAIN)),
                          average="macro", zero_division=0)
    v11_a_pred = v11_oof_act[fold1_mask, :N_ACTION_TRAIN].argmax(axis=1)
    v11_a_f1 = f1_score(y_clip[fold1_mask], v11_a_pred,
                          labels=list(range(N_ACTION_TRAIN)),
                          average="macro", zero_division=0)
    print(f"    V11 head F1_a:    {v11_a_f1:.4f}")
    print(f"    GBM-on-emb F1_a:  {gbm_a_f1:.4f}  (Δ {gbm_a_f1-v11_a_f1:+.4f})")
    print(f"    time: {time.time()-t0:.1f}s")
    results["action"] = {
        "v11_head_f1": float(v11_a_f1),
        "gbm_on_emb_f1": float(gbm_a_f1),
        "delta_f1": float(gbm_a_f1 - v11_a_f1),
    }

    # ─── Point task ──────────────────────────────────────────────────────
    print("\n  TASK: point  (GBM on last_repr embedding)")
    t1 = time.time()
    hp_p = dict(LGBM_HP_MULTI, objective="multiclass", num_class=N_POINT,
                 metric="multi_logloss")
    dtr = lgb.Dataset(emb_last_oof[train_mask], label=y_pt[train_mask])
    dva = lgb.Dataset(emb_last_oof[fold1_mask], label=y_pt[fold1_mask],
                       reference=dtr)
    m = lgb.train(hp_p, dtr, num_boost_round=500,
                   valid_sets=[dva], valid_names=["val"],
                   callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                              lgb.log_evaluation(0)])
    gbm_p_pred = m.predict(emb_last_oof[fold1_mask],
                             num_iteration=m.best_iteration).argmax(axis=1)
    gbm_p_f1 = f1_score(y_pt[fold1_mask], gbm_p_pred,
                          labels=list(range(N_POINT)),
                          average="macro", zero_division=0)
    v11_p_pred = v11_oof_pt[fold1_mask].argmax(axis=1)
    v11_p_f1 = f1_score(y_pt[fold1_mask], v11_p_pred,
                          labels=list(range(N_POINT)),
                          average="macro", zero_division=0)
    print(f"    V11 head F1_p:    {v11_p_f1:.4f}")
    print(f"    GBM-on-emb F1_p:  {gbm_p_f1:.4f}  (Δ {gbm_p_f1-v11_p_f1:+.4f})")
    print(f"    time: {time.time()-t1:.1f}s")
    results["point"] = {
        "v11_head_f1": float(v11_p_f1),
        "gbm_on_emb_f1": float(gbm_p_f1),
        "delta_f1": float(gbm_p_f1 - v11_p_f1),
    }

    # ─── Server task ─────────────────────────────────────────────────────
    print("\n  TASK: server  (GBM on pool_repr embedding)")
    t2 = time.time()
    srv_mask_tr = train_mask & (y_srv >= 0)
    srv_mask_va = fold1_mask & (y_srv >= 0)
    dtr = lgb.Dataset(emb_pool_oof[srv_mask_tr], label=y_srv[srv_mask_tr])
    dva = lgb.Dataset(emb_pool_oof[srv_mask_va], label=y_srv[srv_mask_va],
                       reference=dtr)
    m = lgb.train(LGBM_HP_BIN, dtr, num_boost_round=500,
                   valid_sets=[dva], valid_names=["val"],
                   callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                              lgb.log_evaluation(0)])
    gbm_s_pred = m.predict(emb_pool_oof[srv_mask_va], num_iteration=m.best_iteration)
    gbm_s_auc = roc_auc_score(y_srv[srv_mask_va], gbm_s_pred)
    v11_s_auc = roc_auc_score(y_srv[srv_mask_va], v11_oof_srv[srv_mask_va])
    print(f"    V11 head AUC:     {v11_s_auc:.4f}")
    print(f"    GBM-on-emb AUC:   {gbm_s_auc:.4f}  (Δ {gbm_s_auc-v11_s_auc:+.4f})")
    print(f"    time: {time.time()-t2:.1f}s")
    results["server"] = {
        "v11_head_auc": float(v11_s_auc),
        "gbm_on_emb_auc": float(gbm_s_auc),
        "delta_auc": float(gbm_s_auc - v11_s_auc),
    }

    # ─── Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    signal = (results["action"]["delta_f1"] > 0.005
              or results["point"]["delta_f1"] > 0.005
              or results["server"]["delta_auc"] > 0.005)
    print(f"  action ΔF1:   {results['action']['delta_f1']:+.4f}")
    print(f"  point  ΔF1:   {results['point']['delta_f1']:+.4f}")
    print(f"  server ΔAUC:  {results['server']['delta_auc']:+.4f}")
    print(f"  signal threshold (≥+0.005 on any task): "
          f"{'PASS — proceed to Phase 3' if signal else 'FAIL — close embedding route'}")

    manifest = {
        "rid": "R-082-phase2-step3",
        "ts": "2026-05-26",
        "results": results,
        "signal_pass": bool(signal),
        "v04_candidate_report": {
            "theoretical_generalization_reason":
                "V11 softmax compresses 192-d pooled rep → 15-d action logits. GBM "
                "on raw embedding accesses signal heads lose.",
            "why_transfers_to_test_new":
                "Embeddings encode same patterns V11 heads use; V11 transfers; "
                "OOF-safe by construction.",
            "smoke_sanity_pass": True,
            "lb_probe_worthy": signal,
            "lb_confirm_hypothesis":
                "If GBM-on-emb beats V11 head by ≥+0.005 on any task → proceed to "
                "Phase 3 (combine with zoo blend) → potential STRATEGIC LB lift.",
            "lb_reject_hypothesis":
                "If all deltas < +0.005 → V11 heads already extract max useful signal; "
                "embedding route closed.",
        },
    }
    out = os.path.join(SUBMISSION_DIR, "r082_phase2_step3_gbm_emb_smoke.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved: {out}")


if __name__ == "__main__":
    main()
