"""R-081 Corrector smoke — Fold-1 OOF diagnostic.

Tests whether a LightGBM corrector can predict "is R-034 PAIR's argmax wrong?"
with above-chance AUC. If yes → apply conservative correction logic
(bounded override count) → measure F1 delta.

Theory (v0.4 candidate report):
  theoretical_generalization_reason:
    Correct only LOW-confidence rows where GBM has a high-confidence
    alternative. Bounded override count (cap K=50/task) caps risk.
    Mechanism closer to R-042 rule_override (1.0 LB transfer) than to
    R-054r meta_stack (-0.0103 LB).
  why_transfers_to_test_new:
    Where blend is confident, we don't touch. Where uncertain, GBM uses
    agreement/entropy features that are distribution-invariant. Override
    cap means a wrong call loses at most ~R-072 magnitude.
  smoke_sanity_pass: TBD by this script
  lb_probe_worthy: only if smoke OK AND p_wrong AUC clearly > 0.55
  lb_confirm_hypothesis:
    LB DeltaOV >= +0.001 => bounded conditional correction transfers.
  lb_reject_hypothesis:
    LB DeltaOV <= -0.002 => even conditional correction overfits OOF.

USAGE:
    python -u src/train_r081_corrector_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

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
RANDOM_SEED = 42

# R-034 PAIR component list (current LB-best subset, pre-server-head-blend)
R034_COMPONENTS = ["v11_aug_oldtest", "v11plus", "v13_oldtest",
                    "v14_seed2_v15feat_a", "v16_avg3"]

# Per-task simple-average weights (smoke; full Dirichlet search would change
# this slightly but per-component balance is roughly equal in R-034 PAIR).
SUBSET_WEIGHT = 1.0 / len(R034_COMPONENTS)

# GBM hyperparameters — keep conservative for smoke
LGBM_HP = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": RANDOM_SEED,
}

# Correction logic
P_WRONG_THRESHOLD = 0.50    # GBM-predicted P(wrong) above this is candidate
MAX_OVERRIDES = 50           # bounded override count per task (safety cap)


def pad19(arr: np.ndarray, n: int = 19) -> np.ndarray:
    """Zero-pad action arrays from 15-class to n-class for alignment."""
    if arr.shape[1] >= n:
        return arr.astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], n), dtype=np.float32)
    out[:, :arr.shape[1]] = arr
    return out


def load_component(tag: str) -> Dict[str, np.ndarray]:
    """Load OOF arrays for one component. Slices 72065→69712 for oldtest tags."""
    base = os.path.join(OOF_DIR, tag)
    oof_act = pad19(np.load(f"{base}_oof_act.npy"))
    oof_pt  = np.load(f"{base}_oof_pt.npy").astype(np.float32)
    oof_srv = np.load(f"{base}_oof_srv.npy").astype(np.float32)
    # Slice oldtest variants down to canonical 69712 rows (drop the last 2353)
    if oof_act.shape[0] == 72065:
        oof_act = oof_act[:69712]
        oof_pt = oof_pt[:69712]
        oof_srv = oof_srv[:69712]
    return {"oof_act": oof_act, "oof_pt": oof_pt, "oof_srv": oof_srv}


def per_row_features(comp: Dict[str, Dict[str, np.ndarray]],
                      task: str,
                      true_labels: np.ndarray,
                      nsn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y_iswrong) per-row features for one task.

    Features per row (~16):
      Per-component max-prob (5)
      Per-component entropy (5)
      Per-component argmax (5)  — used for agreement
      Cross-component top-1 agreement count (1)
      R-034 PAIR margin top1-top2 (1)
      SN bucket one-hot SN<=2 / SN 3-4 / SN>=5 (3)

    Target: binary 1 if R-034 PAIR argmax != true label.
    """
    key = f"oof_{task}"  # "oof_act" or "oof_pt"
    n = nsn.shape[0]
    tags = R034_COMPONENTS
    n_comp = len(tags)
    n_cls = comp[tags[0]][key].shape[1]

    # Stack: (n_comp, n_rows, n_cls)
    stack = np.stack([comp[t][key] for t in tags], axis=0)
    # Per-row blend (uniform weights)
    blend = stack.mean(axis=0)  # (n_rows, n_cls)
    blend_argmax = blend.argmax(axis=1)
    # Top-2 margin
    sorted_probs = np.sort(blend, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]   # top1 - top2

    # Per-component features
    per_comp_maxp = stack.max(axis=2)              # (n_comp, n_rows)
    per_comp_entropy = -np.sum(stack * np.log(stack + 1e-12), axis=2)  # (n_comp, n_rows)
    per_comp_argmax = stack.argmax(axis=2)         # (n_comp, n_rows)

    # Cross-component agreement count: how many components match blend_argmax
    agreement = (per_comp_argmax == blend_argmax[None, :]).sum(axis=0)  # (n_rows,)

    # SN bucket one-hot
    sn_le2 = (nsn <= 2).astype(np.float32)
    sn_3to4 = ((nsn >= 3) & (nsn <= 4)).astype(np.float32)
    sn_ge5 = (nsn >= 5).astype(np.float32)

    feats = np.column_stack([
        per_comp_maxp.T,            # (n_rows, n_comp)
        per_comp_entropy.T,         # (n_rows, n_comp)
        per_comp_argmax.T.astype(np.float32),   # (n_rows, n_comp)
        agreement[:, None],         # (n_rows, 1)
        margin[:, None],            # (n_rows, 1)
        sn_le2[:, None],
        sn_3to4[:, None],
        sn_ge5[:, None],
    ])

    # Clip true labels to eval range for action
    if task == "act":
        y_clip = np.where(true_labels >= N_ACTION_TRAIN, 0, true_labels)
    else:
        y_clip = true_labels
    y_iswrong = (blend_argmax != y_clip).astype(np.int32)

    return feats, y_iswrong, blend


def main() -> None:
    print("=" * 80)
    print(" R-081 Corrector smoke — Fold-1 diagnostic")
    print(f" Base: R-034 PAIR ({len(R034_COMPONENTS)}-component uniform blend)")
    print("=" * 80)

    # Load components
    print("\n Step 1: load OOF artifacts")
    comp = {}
    for tag in R034_COMPONENTS:
        comp[tag] = load_component(tag)
        print(f"   {tag:<30} act{comp[tag]['oof_act'].shape}  "
              f"pt{comp[tag]['oof_pt'].shape}  srv{comp[tag]['oof_srv'].shape}")

    # Load labels + fold info (use v14_seed2_v15feat_a as reference; 69712-row aligned)
    REF = "v14_seed2_v15feat_a"
    y_act = np.load(os.path.join(OOF_DIR, f"{REF}_oof_y_act.npy"))
    y_pt  = np.load(os.path.join(OOF_DIR, f"{REF}_oof_y_pt.npy"))
    nsn   = np.load(os.path.join(OOF_DIR, f"{REF}_oof_nsn.npy"))
    print(f"   labels: y_act{y_act.shape}  y_pt{y_pt.shape}  nsn{nsn.shape}")

    # Build train.csv match assignment for GroupKFold split
    print("\n Step 2: build fold assignment (GroupKFold by match)")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_train = raw_train.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    # V14 trainer filters to positions with prev shot → drop strikeNumber==1
    # Actually V14 keeps all rows but the OOF index maps to positions 2..N within each rally
    # The 69712 length = 84707 - 14995 rallies (drop one position per rally — the serve)
    n_rows = 69712
    if len(raw_train) != n_rows + 14995:
        print(f"   WARN: raw_train {len(raw_train)} != expected {n_rows + 14995}")
    # Drop strikeNumber==1 (serves) → 69712 rows
    non_serve = raw_train[raw_train["strikeNumber"] != 1].reset_index(drop=True)
    if len(non_serve) != n_rows:
        print(f"   WARN: non-serve count {len(non_serve)} != {n_rows}; alignment may drift")
    match_per_row = non_serve["match"].to_numpy()[:n_rows]
    print(f"   match assignments: {len(match_per_row)} rows, "
          f"{len(np.unique(match_per_row))} unique matches")

    gkf = GroupKFold(n_splits=5)
    fold_of_row = np.full(n_rows, -1, dtype=np.int32)
    for f, (_, val_idx) in enumerate(gkf.split(np.arange(n_rows), groups=match_per_row)):
        fold_of_row[val_idx] = f
    print(f"   fold sizes: {np.bincount(fold_of_row).tolist()}")
    fold1_mask = (fold_of_row == 0)
    train_mask = ~fold1_mask
    print(f"   Fold-1 val: {int(fold1_mask.sum())} rows  |  Train: {int(train_mask.sum())} rows")

    results = {}
    for task, true_labels, n_cls in [("act", y_act, N_ACTION_TRAIN),
                                       ("pt", y_pt, N_POINT)]:
        print("\n" + "=" * 80)
        print(f" TASK: {task}")
        print("=" * 80)
        X, y_iswrong, blend = per_row_features(comp, task, true_labels, nsn)
        print(f"   features: X{X.shape}  y_iswrong sum: {int(y_iswrong.sum())} "
              f"(base error rate {y_iswrong.mean():.4f})")

        # Train corrector on folds 1-4
        X_tr, y_tr = X[train_mask], y_iswrong[train_mask]
        X_va, y_va = X[fold1_mask], y_iswrong[fold1_mask]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        t0 = time.time()
        model = lgb.train(LGBM_HP, dtrain, num_boost_round=500,
                          valid_sets=[dtrain, dval], valid_names=["train", "val"],
                          callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                                     lgb.log_evaluation(100)])
        print(f"   training time: {time.time() - t0:.1f}s, "
              f"best_iteration: {model.best_iteration}")

        # Predict P(wrong) on Fold-1
        p_wrong = model.predict(X_va, num_iteration=model.best_iteration)
        auc_p_wrong = roc_auc_score(y_va, p_wrong)
        print(f"   p_wrong AUC on Fold-1: {auc_p_wrong:.4f}  "
              f"(>= 0.55 = corrector has signal)")

        # Baseline F1 (no correction)
        if task == "act":
            true_clip = np.where(true_labels[fold1_mask] >= N_ACTION_TRAIN,
                                 0, true_labels[fold1_mask])
            eval_labels = list(range(N_ACTION_TRAIN))
        else:
            true_clip = true_labels[fold1_mask]
            eval_labels = list(range(N_POINT))
        base_argmax = blend.argmax(axis=1)[fold1_mask]
        base_f1 = f1_score(true_clip, base_argmax, labels=eval_labels,
                            average="macro", zero_division=0)

        # Apply correction logic: for rows where p_wrong > threshold,
        # find GBM's "alternative" — actually GBM predicts WRONG/NOT-WRONG,
        # not the alternative class. For the smoke we use: replace argmax
        # with second-best class from the BLEND when p_wrong > threshold.
        # This is a conservative correction: replace with the next-most-likely
        # class according to the SAME blend's softmax.
        blend_f1 = blend[fold1_mask]
        # Top-2: sort by probability descending; pick index 1 (second-best)
        sorted_idx = np.argsort(blend_f1, axis=1)[:, ::-1]
        second_best = sorted_idx[:, 1]
        # Threshold + cap
        candidates = np.where(p_wrong > P_WRONG_THRESHOLD)[0]
        ranked = candidates[np.argsort(p_wrong[candidates])[::-1]]
        n_override = min(MAX_OVERRIDES, len(ranked))
        override_idx = ranked[:n_override]
        corrected_argmax = base_argmax.copy()
        corrected_argmax[override_idx] = second_best[override_idx]
        corrected_f1 = f1_score(true_clip, corrected_argmax, labels=eval_labels,
                                  average="macro", zero_division=0)

        # How many overrides went the right way?
        if n_override > 0:
            override_correct = (corrected_argmax[override_idx] == true_clip[override_idx]).sum()
            override_was_originally_correct = (base_argmax[override_idx] == true_clip[override_idx]).sum()
            print(f"   overrides applied: {n_override} (cap {MAX_OVERRIDES})")
            print(f"     of those: {override_correct} now correct, "
                  f"{override_was_originally_correct} were already correct (we made them wrong)")
        else:
            print(f"   overrides applied: 0 (no rows had p_wrong > {P_WRONG_THRESHOLD})")

        delta_f1 = corrected_f1 - base_f1
        print(f"   Fold-1 F1_{task}:  base={base_f1:.4f}  corrected={corrected_f1:.4f}  "
              f"Delta={delta_f1:+.4f}")

        results[task] = {
            "p_wrong_auc": float(auc_p_wrong),
            "base_f1": float(base_f1),
            "corrected_f1": float(corrected_f1),
            "delta_f1": float(delta_f1),
            "n_overrides": int(n_override),
            "best_iteration": int(model.best_iteration),
            "feature_importance": dict(zip(
                ["maxp_v11augOLD", "maxp_v11plus", "maxp_v13OLD", "maxp_v14v15a", "maxp_v16avg3",
                 "ent_v11augOLD", "ent_v11plus", "ent_v13OLD", "ent_v14v15a", "ent_v16avg3",
                 "amax_v11augOLD", "amax_v11plus", "amax_v13OLD", "amax_v14v15a", "amax_v16avg3",
                 "agreement", "margin", "sn_le2", "sn_3to4", "sn_ge5"],
                model.feature_importance(importance_type="gain").tolist()
            )),
        }

    # Save manifest
    print("\n" + "=" * 80)
    print(" SUMMARY (Fold-1 only)")
    print("=" * 80)
    for task, r in results.items():
        print(f"   {task}:  p_wrong AUC {r['p_wrong_auc']:.4f}  |  "
              f"F1 {r['base_f1']:.4f} -> {r['corrected_f1']:.4f} "
              f"({r['delta_f1']:+.4f})  |  {r['n_overrides']} overrides")

    # v0.4 sanity verdict
    smoke_pass = True
    sanity_reasons = []
    for task, r in results.items():
        if r["p_wrong_auc"] < 0.50:
            smoke_pass = False
            sanity_reasons.append(f"{task}: p_wrong AUC {r['p_wrong_auc']:.4f} < 0.50 (worse than chance)")
        if r["delta_f1"] < -0.030:
            smoke_pass = False
            sanity_reasons.append(f"{task}: F1 catastrophic drop {r['delta_f1']:+.4f} <= -0.030")

    manifest = {
        "rid": "R-081",
        "ts": "2026-05-26",
        "smoke_scope": "Fold-1 only",
        "base_subset": R034_COMPONENTS,
        "fold1_results": results,
        "p_wrong_threshold": P_WRONG_THRESHOLD,
        "max_overrides_per_task": MAX_OVERRIDES,
        "smoke_sanity_pass": smoke_pass,
        "smoke_sanity_reason": "OK" if smoke_pass else "; ".join(sanity_reasons),
        "v04_candidate_report": {
            "theoretical_generalization_reason":
                "Conditional correction on R-034 PAIR; bounded override count caps "
                "downside risk to ~R-072 magnitude (-0.003). Mechanism closer to "
                "R-042 rule_override (1.0 LB transfer) than R-054r meta_stack "
                "(-0.0103 LB).",
            "why_transfers_to_test_new":
                "Where blend is confident we don't touch. Where uncertain GBM uses "
                "agreement/entropy features that are distribution-invariant. "
                "Override cap means wrong call loses bounded amount.",
            "lb_confirm_hypothesis":
                "LB DeltaOV >= +0.001 => bounded conditional correction transfers; "
                "mechanism distinct from pure meta-stacking.",
            "lb_reject_hypothesis":
                "LB DeltaOV <= -0.002 => even conditional correction overfits OOF; "
                "GBM-corrector route closed for this base model family.",
        },
    }
    out_path = os.path.join(SUBMISSION_DIR, "r081_corrector_smoke.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved manifest: {out_path}")
    print(f" smoke_sanity_pass = {smoke_pass}")


if __name__ == "__main__":
    main()
