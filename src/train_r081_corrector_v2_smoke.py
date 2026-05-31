"""R-081 v2 Corrector smoke — alt-class GBM instead of naive 2nd-best.

v1 smoke (2026-05-26) found: p_wrong AUC 0.65-0.69 (signal exists), but the
naive "replace argmax with blend's 2nd-best" correction strategy produced
ΔF1 ≈ 0 (action -0.0009, point +0.0005).

v2 hypothesis: the corrector needs to predict WHICH alternative is right,
not just "is current wrong". Add a per-task multiclass GBM that predicts the
true label given the same features, then use ITS argmax as the override.

Theory unchanged (v0.4 candidate report):
  - bounded mechanism, same R-042-ish risk profile
  - distribution-invariant features (model-output-derived)
  - cap K=50 overrides per task

USAGE:
    python -u src/train_r081_corrector_v2_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR, TRAIN_PATH

# Re-use v1 components from src/train_r081_corrector_smoke.py
from train_r081_corrector_smoke import (  # noqa: E402
    R034_COMPONENTS, load_component, per_row_features,
    N_ACTION_TRAIN, N_POINT, RANDOM_SEED,
)

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")

# GBM hyperparameters
LGBM_HP_BIN = {
    "objective": "binary", "metric": "auc",
    "num_leaves": 31, "min_data_in_leaf": 100,
    "learning_rate": 0.05, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l2": 1.0, "verbosity": -1, "seed": RANDOM_SEED,
}
LGBM_HP_MULTI_ACT = {
    "objective": "multiclass", "num_class": N_ACTION_TRAIN, "metric": "multi_logloss",
    "num_leaves": 31, "min_data_in_leaf": 100,
    "learning_rate": 0.05, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l2": 1.0, "verbosity": -1, "seed": RANDOM_SEED,
}
LGBM_HP_MULTI_PT = {
    "objective": "multiclass", "num_class": N_POINT, "metric": "multi_logloss",
    "num_leaves": 31, "min_data_in_leaf": 100,
    "learning_rate": 0.05, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l2": 1.0, "verbosity": -1, "seed": RANDOM_SEED,
}

P_WRONG_THRESHOLD = 0.50
MAX_OVERRIDES = 50
ALT_CONFIDENCE_THRESHOLD = 0.35   # alt_class GBM must be at least this confident


def main() -> None:
    print("=" * 80)
    print(" R-081 v2 Corrector smoke — alt-class GBM instead of naive 2nd-best")
    print("=" * 80)

    # Load components
    print("\n Step 1: load OOF artifacts")
    comp = {}
    for tag in R034_COMPONENTS:
        comp[tag] = load_component(tag)

    REF = "v14_seed2_v15feat_a"
    y_act = np.load(os.path.join(OOF_DIR, f"{REF}_oof_y_act.npy"))
    y_pt  = np.load(os.path.join(OOF_DIR, f"{REF}_oof_y_pt.npy"))
    nsn   = np.load(os.path.join(OOF_DIR, f"{REF}_oof_nsn.npy"))
    n_rows = 69712

    # Fold assignment (same logic as v1)
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_train = raw_train.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    non_serve = raw_train[raw_train["strikeNumber"] != 1].reset_index(drop=True)
    match_per_row = non_serve["match"].to_numpy()[:n_rows]
    gkf = GroupKFold(n_splits=5)
    fold_of_row = np.full(n_rows, -1, dtype=np.int32)
    for f, (_, val_idx) in enumerate(gkf.split(np.arange(n_rows), groups=match_per_row)):
        fold_of_row[val_idx] = f
    fold1_mask = (fold_of_row == 0)
    train_mask = ~fold1_mask

    results = {}
    for task, true_labels, n_cls, hp_multi in [
        ("act", y_act, N_ACTION_TRAIN, LGBM_HP_MULTI_ACT),
        ("pt",  y_pt,  N_POINT,        LGBM_HP_MULTI_PT),
    ]:
        print("\n" + "=" * 80)
        print(f" TASK: {task}")
        print("=" * 80)
        X, y_iswrong, blend = per_row_features(comp, task, true_labels, nsn)
        if task == "act":
            true_clip = np.where(true_labels >= N_ACTION_TRAIN, 0, true_labels)
        else:
            true_clip = true_labels
        # Reduce action labels to valid range for multiclass GBM (clip 15-18 -> 0)
        # already done above; true_clip is in [0, n_cls)

        X_tr, X_va = X[train_mask], X[fold1_mask]
        y_iswrong_tr = y_iswrong[train_mask]
        y_iswrong_va = y_iswrong[fold1_mask]
        y_true_tr = true_clip[train_mask]
        y_true_va = true_clip[fold1_mask]

        # ─── Model 1: p_wrong predictor (same as v1) ──────────────────────
        print("  [1] training p_wrong binary classifier...")
        t0 = time.time()
        dtrain_b = lgb.Dataset(X_tr, label=y_iswrong_tr)
        dval_b = lgb.Dataset(X_va, label=y_iswrong_va, reference=dtrain_b)
        m_wrong = lgb.train(LGBM_HP_BIN, dtrain_b, num_boost_round=500,
                             valid_sets=[dval_b], valid_names=["val"],
                             callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                                        lgb.log_evaluation(0)])
        p_wrong = m_wrong.predict(X_va, num_iteration=m_wrong.best_iteration)
        auc_p_wrong = roc_auc_score(y_iswrong_va, p_wrong)
        print(f"      p_wrong AUC on Fold-1: {auc_p_wrong:.4f}  "
              f"(best_iter {m_wrong.best_iteration}, {time.time()-t0:.1f}s)")

        # ─── Model 2: alt-class multiclass GBM ────────────────────────────
        print("  [2] training alt-class multiclass classifier...")
        t1 = time.time()
        dtrain_m = lgb.Dataset(X_tr, label=y_true_tr)
        dval_m = lgb.Dataset(X_va, label=y_true_va, reference=dtrain_m)
        m_alt = lgb.train(hp_multi, dtrain_m, num_boost_round=500,
                            valid_sets=[dval_m], valid_names=["val"],
                            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                                       lgb.log_evaluation(0)])
        alt_probs = m_alt.predict(X_va, num_iteration=m_alt.best_iteration)
        if alt_probs.ndim == 1:
            alt_probs = alt_probs.reshape(-1, 1)
        alt_argmax = alt_probs.argmax(axis=1)
        alt_maxp = alt_probs.max(axis=1)
        # Standalone GBM accuracy on Fold-1 (compare to base blend F1)
        eval_labels = list(range(n_cls))
        gbm_solo_f1 = f1_score(y_true_va, alt_argmax, labels=eval_labels,
                                 average="macro", zero_division=0)
        print(f"      GBM-solo F1 on Fold-1: {gbm_solo_f1:.4f}  "
              f"(best_iter {m_alt.best_iteration}, {time.time()-t1:.1f}s)")

        base_argmax = blend.argmax(axis=1)[fold1_mask]
        base_f1 = f1_score(y_true_va, base_argmax, labels=eval_labels,
                            average="macro", zero_division=0)
        print(f"      base R-034 PAIR Fold-1 F1: {base_f1:.4f}")

        # ─── v2 correction: replace argmax with alt-class GBM's argmax ────
        # only when BOTH (p_wrong > threshold) AND (alt-argmax != base-argmax)
        # AND (alt_maxp >= confidence threshold)
        candidates = np.where(
            (p_wrong > P_WRONG_THRESHOLD)
            & (alt_argmax != base_argmax)
            & (alt_maxp >= ALT_CONFIDENCE_THRESHOLD)
        )[0]
        # Rank by joint score = p_wrong * alt_maxp
        rank_score = p_wrong[candidates] * alt_maxp[candidates]
        ranked = candidates[np.argsort(rank_score)[::-1]]
        n_override = min(MAX_OVERRIDES, len(ranked))
        override_idx = ranked[:n_override]
        corrected_argmax = base_argmax.copy()
        corrected_argmax[override_idx] = alt_argmax[override_idx]
        corrected_f1 = f1_score(y_true_va, corrected_argmax, labels=eval_labels,
                                  average="macro", zero_division=0)
        delta_f1 = corrected_f1 - base_f1

        # Override quality
        if n_override > 0:
            now_correct = (corrected_argmax[override_idx] == y_true_va[override_idx]).sum()
            was_correct = (base_argmax[override_idx] == y_true_va[override_idx]).sum()
            print(f"      overrides applied: {n_override} (cap {MAX_OVERRIDES}, "
                  f"{len(candidates)} eligible)")
            print(f"        of those: {now_correct} now correct, "
                  f"{was_correct} were already correct (we made them wrong)")
        else:
            print(f"      overrides applied: 0")

        print(f"      Fold-1 F1_{task}:  base={base_f1:.4f}  corrected={corrected_f1:.4f}  "
              f"Delta={delta_f1:+.4f}")

        results[task] = {
            "p_wrong_auc": float(auc_p_wrong),
            "gbm_solo_f1": float(gbm_solo_f1),
            "base_f1": float(base_f1),
            "corrected_f1": float(corrected_f1),
            "delta_f1": float(delta_f1),
            "n_overrides": int(n_override),
            "n_eligible": int(len(candidates)),
            "p_wrong_best_iter": int(m_wrong.best_iteration),
            "alt_class_best_iter": int(m_alt.best_iteration),
        }

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" SUMMARY (R-081 v2, Fold-1 only)")
    print("=" * 80)
    for task, r in results.items():
        print(f"   {task}:  p_wrong AUC {r['p_wrong_auc']:.4f}  "
              f"GBM-solo F1 {r['gbm_solo_f1']:.4f}  "
              f"|  F1 {r['base_f1']:.4f} -> {r['corrected_f1']:.4f} "
              f"({r['delta_f1']:+.4f})  |  {r['n_overrides']} overrides")

    # v0.4 sanity verdict
    smoke_pass = True
    sanity_reasons = []
    for task, r in results.items():
        if r["p_wrong_auc"] < 0.50:
            smoke_pass = False
            sanity_reasons.append(f"{task}: p_wrong AUC {r['p_wrong_auc']:.4f} < 0.50")
        if r["delta_f1"] < -0.030:
            smoke_pass = False
            sanity_reasons.append(f"{task}: F1 catastrophic drop {r['delta_f1']:+.4f}")

    # v2-specific judgment: was the alt-class GBM useful?
    # If gbm_solo_f1 << base_f1 the GBM is just much weaker than the blend → not useful
    blend_act_f1 = results["act"]["base_f1"]
    gbm_act_f1 = results["act"]["gbm_solo_f1"]
    blend_pt_f1 = results["pt"]["base_f1"]
    gbm_pt_f1 = results["pt"]["gbm_solo_f1"]
    print(f"\n   Solo GBM vs base blend:")
    print(f"     action: blend {blend_act_f1:.4f} vs GBM-solo {gbm_act_f1:.4f}  "
          f"(GBM { 'STRONGER' if gbm_act_f1 > blend_act_f1 else 'weaker' })")
    print(f"     point:  blend {blend_pt_f1:.4f} vs GBM-solo {gbm_pt_f1:.4f}  "
          f"(GBM { 'STRONGER' if gbm_pt_f1 > blend_pt_f1 else 'weaker' })")

    manifest = {
        "rid": "R-081-v2",
        "ts": "2026-05-26",
        "smoke_scope": "Fold-1 only",
        "base_subset": R034_COMPONENTS,
        "fold1_results": results,
        "p_wrong_threshold": P_WRONG_THRESHOLD,
        "alt_confidence_threshold": ALT_CONFIDENCE_THRESHOLD,
        "max_overrides_per_task": MAX_OVERRIDES,
        "smoke_sanity_pass": smoke_pass,
        "smoke_sanity_reason": "OK" if smoke_pass else "; ".join(sanity_reasons),
        "v04_candidate_report": {
            "theoretical_generalization_reason":
                "v2 corrector: predict p_wrong AND predict the alternative class "
                "with a multiclass GBM. Override only when both predictors agree "
                "AND the alternative is different from current AND alt confidence "
                "is above ALT_CONFIDENCE_THRESHOLD. Bounded cap caps risk to "
                "R-042-magnitude (~10-50 changes).",
            "why_transfers_to_test_new":
                "Same as v1 — features are model-output-derived, distribution-invariant. "
                "v2 just adds smarter override selection, doesn't change leakage profile.",
            "lb_confirm_hypothesis":
                "LB DeltaOV >= +0.001 => smarter corrector transfers.",
            "lb_reject_hypothesis":
                "LB DeltaOV <= -0.002 => GBM-corrector route closed.",
        },
    }
    out_path = os.path.join(SUBMISSION_DIR, "r081_v2_corrector_smoke.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved manifest: {out_path}")


if __name__ == "__main__":
    main()
