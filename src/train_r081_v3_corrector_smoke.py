"""R-081 v3 Corrector smoke — corrector aligned to R-034 PAIR Dirichlet blend.

v2 trained corrector on uniform-blend wrongness but applied it to R-067cr's
stored Dirichlet-weighted predictions → target/inference mismatch. This may
have diluted the signal.

v3 fix: train corrector on the ACTUAL R-034 PAIR Dirichlet-weighted blend's
argmax wrongness (same blend that produces R-067cr's stored test predictions).

USAGE:
    python -u src/train_r081_v3_corrector_smoke.py
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
from analyze_oldtest_blend import load_components, evaluate_subset_none

from train_r081_corrector_smoke import (  # noqa: E402
    R034_COMPONENTS, load_component, per_row_features,
    N_ACTION_TRAIN, N_POINT, RANDOM_SEED,
)
from train_r081_corrector_v2_smoke import (  # noqa: E402
    LGBM_HP_BIN, LGBM_HP_MULTI_ACT, LGBM_HP_MULTI_PT,
    P_WRONG_THRESHOLD, MAX_OVERRIDES, ALT_CONFIDENCE_THRESHOLD,
)

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")


def build_r034_pair_blend_oof():
    """Compute R-034 PAIR Dirichlet-weighted blend OOF on 69712-row aligned subset.

    Returns:
      blend_act:  (69712, 19) softmax-weighted blend action probs
      blend_pt:   (69712, 10) blend point probs
      blend_srv:  (69712,) blend SGP probs
      weights:    {"w_a": (5,), "w_p": (5,), "w_s": (5,)}
      y_act, y_pt, y_srv, nsn
    """
    print(" Loading components via analyze_oldtest_blend...")
    comp, y_a, y_p, y_s, mask, test_uid = load_components(R034_COMPONENTS)
    print(f"   loaded {len(R034_COMPONENTS)} comps, mask sum {mask.sum()}/{len(mask)}")

    print(" Running Dirichlet search to derive R-034 PAIR weights...")
    base = evaluate_subset_none(R034_COMPONENTS, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=300, seed=20260524)
    print(f"   R-034 PAIR OOF: F1_a={base['F1_a']:.4f}  F1_p={base['F1_p']:.4f}  "
          f"AUC={base['AUC']:.4f}  OV={base['OV']:.4f}")

    # Recompute blend probs using returned weights
    act_stack = np.stack([comp[t]["oof_act"] for t in R034_COMPONENTS], axis=0)  # (5, n, 19)
    pt_stack = np.stack([comp[t]["oof_pt"] for t in R034_COMPONENTS], axis=0)
    srv_stack = np.stack([comp[t]["oof_srv"] for t in R034_COMPONENTS], axis=0)
    blend_act = (base["w_a"][:, None, None] * act_stack).sum(axis=0)
    blend_pt = (base["w_p"][:, None, None] * pt_stack).sum(axis=0)
    blend_srv = (base["w_s"][:, None] * srv_stack).sum(axis=0)
    print(f"   blend shapes: act{blend_act.shape}  pt{blend_pt.shape}  srv{blend_srv.shape}")

    # nsn from any reference
    nsn = np.load(os.path.join(OOF_DIR, "v14_seed2_v15feat_a_oof_nsn.npy"))
    return blend_act, blend_pt, blend_srv, base, y_a, y_p, y_s, nsn, comp


def per_row_features_v3(blend_act, blend_pt, comp, task, nsn):
    """Same shape as v1 features but using R-034 PAIR blend's per-row metrics.

    Per-component features still computed for diversity signal (which component
    disagrees most with the blend = useful corrector input).
    """
    n = nsn.shape[0]
    if task == "act":
        blend = blend_act
        per_comp_arr = [comp[t]["oof_act"] for t in R034_COMPONENTS]
    else:
        blend = blend_pt
        per_comp_arr = [comp[t]["oof_pt"] for t in R034_COMPONENTS]
    n_cls = blend.shape[1]
    stack = np.stack(per_comp_arr, axis=0)  # (5, n, n_cls)

    blend_argmax = blend.argmax(axis=1)
    sorted_probs = np.sort(blend, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    per_comp_maxp = stack.max(axis=2)
    per_comp_entropy = -np.sum(stack * np.log(stack + 1e-12), axis=2)
    per_comp_argmax = stack.argmax(axis=2)
    agreement = (per_comp_argmax == blend_argmax[None, :]).sum(axis=0)

    sn_le2 = (nsn <= 2).astype(np.float32)
    sn_3to4 = ((nsn >= 3) & (nsn <= 4)).astype(np.float32)
    sn_ge5 = (nsn >= 5).astype(np.float32)

    feats = np.column_stack([
        per_comp_maxp.T, per_comp_entropy.T,
        per_comp_argmax.T.astype(np.float32),
        agreement[:, None], margin[:, None],
        sn_le2[:, None], sn_3to4[:, None], sn_ge5[:, None],
    ])
    return feats, blend, blend_argmax


def main() -> None:
    print("=" * 80)
    print(" R-081 v3 Corrector smoke — R-034 PAIR Dirichlet-aligned target")
    print("=" * 80)

    blend_act, blend_pt, blend_srv, weights, y_act, y_pt, y_srv, nsn, comp = build_r034_pair_blend_oof()

    # Fold split (same as v1/v2)
    print("\n Step: build fold assignment")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_train = raw_train.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    non_serve = raw_train[raw_train["strikeNumber"] != 1].reset_index(drop=True)
    n_rows = 69712
    match_per_row = non_serve["match"].to_numpy()[:n_rows]
    gkf = GroupKFold(n_splits=5)
    fold_of_row = np.full(n_rows, -1, dtype=np.int32)
    for f, (_, val_idx) in enumerate(gkf.split(np.arange(n_rows), groups=match_per_row)):
        fold_of_row[val_idx] = f
    fold1_mask = (fold_of_row == 0)
    train_mask = ~fold1_mask
    print(f"   Fold-1 val: {int(fold1_mask.sum())} | train: {int(train_mask.sum())}")

    results = {}
    for task, true_labels, n_cls, hp_multi in [
        ("act", y_act, N_ACTION_TRAIN, LGBM_HP_MULTI_ACT),
        ("pt",  y_pt,  N_POINT,        LGBM_HP_MULTI_PT),
    ]:
        print("\n" + "=" * 80)
        print(f" TASK: {task}")
        print("=" * 80)

        X, blend, blend_argmax = per_row_features_v3(blend_act, blend_pt, comp, task, nsn)
        if task == "act":
            true_clip = np.where(true_labels >= N_ACTION_TRAIN, 0, true_labels)
        else:
            true_clip = true_labels
        # Target: is R-034 PAIR blend's argmax wrong?
        y_iswrong = (blend_argmax != true_clip).astype(np.int32)
        print(f"   features X{X.shape}; iswrong rate {y_iswrong.mean():.4f} "
              f"(blend baseline F1 reflects this)")

        X_tr, X_va = X[train_mask], X[fold1_mask]
        y_iw_tr = y_iswrong[train_mask]
        y_iw_va = y_iswrong[fold1_mask]
        y_true_tr = true_clip[train_mask]
        y_true_va = true_clip[fold1_mask]

        # Train both classifiers
        t0 = time.time()
        dtr_b = lgb.Dataset(X_tr, label=y_iw_tr)
        dva_b = lgb.Dataset(X_va, label=y_iw_va, reference=dtr_b)
        m_wrong = lgb.train(LGBM_HP_BIN, dtr_b, num_boost_round=500,
                             valid_sets=[dva_b], valid_names=["val"],
                             callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                                        lgb.log_evaluation(0)])
        p_wrong = m_wrong.predict(X_va, num_iteration=m_wrong.best_iteration)
        auc_p_wrong = roc_auc_score(y_iw_va, p_wrong)

        dtr_m = lgb.Dataset(X_tr, label=y_true_tr)
        dva_m = lgb.Dataset(X_va, label=y_true_va, reference=dtr_m)
        m_alt = lgb.train(hp_multi, dtr_m, num_boost_round=500,
                            valid_sets=[dva_m], valid_names=["val"],
                            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                                       lgb.log_evaluation(0)])
        alt_probs = m_alt.predict(X_va, num_iteration=m_alt.best_iteration)
        if alt_probs.ndim == 1:
            alt_probs = alt_probs.reshape(-1, 1)
        alt_argmax = alt_probs.argmax(axis=1)
        alt_maxp = alt_probs.max(axis=1)
        print(f"   train time: {time.time()-t0:.1f}s  p_wrong AUC {auc_p_wrong:.4f}")

        eval_labels = list(range(n_cls))
        base_argmax = blend_argmax[fold1_mask]
        base_f1 = f1_score(y_true_va, base_argmax, labels=eval_labels,
                            average="macro", zero_division=0)
        gbm_solo_f1 = f1_score(y_true_va, alt_argmax, labels=eval_labels,
                                 average="macro", zero_division=0)

        candidates = np.where(
            (p_wrong > P_WRONG_THRESHOLD)
            & (alt_argmax != base_argmax)
            & (alt_maxp >= ALT_CONFIDENCE_THRESHOLD)
        )[0]
        rank_score = p_wrong[candidates] * alt_maxp[candidates]
        ranked = candidates[np.argsort(rank_score)[::-1]]
        n_override = min(MAX_OVERRIDES, len(ranked))
        override_idx = ranked[:n_override]
        corrected_argmax = base_argmax.copy()
        corrected_argmax[override_idx] = alt_argmax[override_idx]
        corrected_f1 = f1_score(y_true_va, corrected_argmax, labels=eval_labels,
                                  average="macro", zero_division=0)

        if n_override > 0:
            now_correct = (corrected_argmax[override_idx] == y_true_va[override_idx]).sum()
            was_correct = (base_argmax[override_idx] == y_true_va[override_idx]).sum()
        else:
            now_correct = was_correct = 0
        delta_f1 = corrected_f1 - base_f1

        print(f"   base F1 {base_f1:.4f}  corrected F1 {corrected_f1:.4f}  "
              f"Delta {delta_f1:+.4f}  ({n_override} overrides "
              f"of {len(candidates)} eligible)")
        print(f"   override net: {now_correct} now-correct - {was_correct} was-correct "
              f"= {now_correct - was_correct:+d}")
        print(f"   GBM-solo F1 {gbm_solo_f1:.4f}  (vs base {base_f1:.4f})")

        results[task] = {
            "p_wrong_auc": float(auc_p_wrong),
            "base_f1": float(base_f1),
            "gbm_solo_f1": float(gbm_solo_f1),
            "corrected_f1": float(corrected_f1),
            "delta_f1": float(delta_f1),
            "n_overrides": int(n_override),
            "n_eligible": int(len(candidates)),
            "override_now_correct": int(now_correct),
            "override_was_correct": int(was_correct),
        }

    print("\n" + "=" * 80)
    print(" R-081 v3 SUMMARY")
    print("=" * 80)
    for task, r in results.items():
        print(f"  {task}: p_wrong AUC {r['p_wrong_auc']:.4f}  "
              f"F1 {r['base_f1']:.4f} -> {r['corrected_f1']:.4f} "
              f"({r['delta_f1']:+.4f})  net {r['override_now_correct']-r['override_was_correct']:+d}")

    # Compare to v2
    try:
        with open(os.path.join(SUBMISSION_DIR, "r081_v2_corrector_smoke.json")) as f:
            v2_data = json.load(f)
        print(f"\n  v3 vs v2 comparison (Fold-1):")
        for task in ["act", "pt"]:
            v3_d = results[task]["delta_f1"]
            v2_d = v2_data["fold1_results"][task]["delta_f1"]
            improvement = v3_d - v2_d
            print(f"    {task}: v2 ΔF1 {v2_d:+.4f}  vs  v3 ΔF1 {v3_d:+.4f}  "
                  f"(v3 improvement {improvement:+.4f})")
    except FileNotFoundError:
        pass

    # Manifest
    smoke_pass = all(r["delta_f1"] > -0.030 for r in results.values())
    manifest = {
        "rid": "R-081-v3",
        "ts": "2026-05-26",
        "smoke_scope": "Fold-1 only",
        "base_subset": R034_COMPONENTS,
        "r034_pair_weights": {
            "w_a": weights["w_a"].tolist(),
            "w_p": weights["w_p"].tolist(),
            "w_s": weights["w_s"].tolist(),
        },
        "r034_pair_full_oof": {
            "F1_a": float(weights["F1_a"]),
            "F1_p": float(weights["F1_p"]),
            "AUC": float(weights["AUC"]),
            "OV": float(weights["OV"]),
        },
        "fold1_results": results,
        "p_wrong_threshold": P_WRONG_THRESHOLD,
        "alt_confidence_threshold": ALT_CONFIDENCE_THRESHOLD,
        "max_overrides_per_task": MAX_OVERRIDES,
        "smoke_sanity_pass": smoke_pass,
        "v04_candidate_report": {
            "theoretical_generalization_reason":
                "v3 fix: corrector trained on the same Dirichlet-weighted R-034 PAIR "
                "blend used at inference. Removes the train/inference target mismatch "
                "that diluted v2's signal. Same bounded-override mechanism (cap 50/task).",
            "why_transfers_to_test_new":
                "Same as v2 — model-output-derived features (distribution-invariant). "
                "Target alignment fix means corrector learns on the SAME prediction "
                "distribution it will see at inference, which should sharpen the signal.",
            "lb_confirm_hypothesis":
                "LB DeltaOV >= +0.001 => v3 alignment fix improves transfer over v2.",
            "lb_reject_hypothesis":
                "LB DeltaOV <= -0.002 => corrector mechanism overfits OOF regardless of "
                "target alignment; GBM-corrector route confirmed closed.",
        },
    }
    out_path = os.path.join(SUBMISSION_DIR, "r081_v3_corrector_smoke.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved: {out_path}")


if __name__ == "__main__":
    main()
