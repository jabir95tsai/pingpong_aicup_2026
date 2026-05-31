"""Build R-081 v2 LB candidate CSV.

Trains corrector (p_wrong + alt-class) on FULL OOF, applies to TEST
predictions, builds candidate CSV on top of R-067cr (preserves SGP).

Per v0.4: this is artifact materialization. ARTIFACT_READY_FOR_JABIR_UPLOAD_REVIEW.
Jabir decides LB upload manually.

USAGE:
    python -u src/build_r081_v2_lb_candidate.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR

from train_r081_corrector_smoke import (  # noqa: E402
    R034_COMPONENTS, load_component, per_row_features,
    N_ACTION_TRAIN, N_POINT, RANDOM_SEED,
)
from train_r081_corrector_v2_smoke import (  # noqa: E402
    LGBM_HP_BIN, LGBM_HP_MULTI_ACT, LGBM_HP_MULTI_PT,
    P_WRONG_THRESHOLD, MAX_OVERRIDES, ALT_CONFIDENCE_THRESHOLD,
)

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
R067CR_BASE = os.path.join(SUBMISSION_DIR,
                            "submission_R067cr_alpha030_v22_blend_PLUS_RULE.csv")
OUT_CSV = os.path.join(SUBMISSION_DIR,
                        "submission_R081v2_R067cr_PLUS_CORRECTOR.csv")
MANIFEST = os.path.join(SUBMISSION_DIR, "r081_v2_lb_candidate_manifest.json")


def load_test_components() -> Dict[str, Dict[str, np.ndarray]]:
    """Load TEST per-component prediction arrays."""
    comp = {}
    for tag in R034_COMPONENTS:
        base = os.path.join(OOF_DIR, tag)
        ta = np.load(f"{base}_test_act.npy").astype(np.float32)
        tp = np.load(f"{base}_test_pt.npy").astype(np.float32)
        # Pad action to 19-class if needed
        if ta.shape[1] < 19:
            padded = np.zeros((ta.shape[0], 19), dtype=np.float32)
            padded[:, :ta.shape[1]] = ta
            ta = padded
        comp[tag] = {"oof_act": ta, "oof_pt": tp, "oof_srv": None}
    return comp


def test_features(comp_test: Dict, task: str, nsn_test: np.ndarray) -> tuple:
    """Build test features (same shape as OOF features)."""
    # Re-use OOF feature builder; just pass test predictions as 'oof_*'
    # The function only needs oof_act/oof_pt and doesn't use true labels for X.
    n = nsn_test.shape[0]
    feats, _y_iswrong, blend = per_row_features(
        comp_test, task,
        true_labels=np.zeros(n, dtype=np.int64),  # placeholder; not used for X
        nsn=nsn_test,
    )
    return feats, blend


def main() -> None:
    print("=" * 80)
    print(" R-081 v2 LB candidate builder")
    print("=" * 80)

    # Load OOF artifacts (training data for corrector)
    print("\n Step 1: load OOF artifacts (for corrector training)")
    comp_oof = {}
    for tag in R034_COMPONENTS:
        comp_oof[tag] = load_component(tag)
    REF = "v14_seed2_v15feat_a"
    y_act = np.load(os.path.join(OOF_DIR, f"{REF}_oof_y_act.npy"))
    y_pt  = np.load(os.path.join(OOF_DIR, f"{REF}_oof_y_pt.npy"))
    nsn   = np.load(os.path.join(OOF_DIR, f"{REF}_oof_nsn.npy"))
    print(f"   OOF labels: {y_act.shape}, nsn {nsn.shape}")

    # Load TEST artifacts (inference target)
    print("\n Step 2: load TEST artifacts")
    comp_test = load_test_components()
    n_test = comp_test[R034_COMPONENTS[0]]["oof_act"].shape[0]
    print(f"   test predictions: {n_test} rows per component")

    # Test NSN: each rally has one row at the LAST shot. Reconstruct from test.csv
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test_new.csv"))
    test_df = test_df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)
    # next_strikeNumber for the last-visible-shot is = strikeNumber + 1
    # (the unseen target shot)
    last_per_rally = test_df.groupby("rally_uid", sort=False).tail(1)
    nsn_test = (last_per_rally["strikeNumber"].to_numpy() + 1).astype(np.int64)
    print(f"   test NSN reconstructed: {len(nsn_test)} rallies, "
          f"nsn range [{nsn_test.min()}, {nsn_test.max()}]")
    if len(nsn_test) != n_test:
        print(f"   WARN: nsn_test length {len(nsn_test)} != n_test {n_test}")

    # Load R-067cr base CSV (we'll override actionId/pointId only; SGP preserved)
    print("\n Step 3: load R-067cr base CSV")
    r067cr = pd.read_csv(R067CR_BASE)
    print(f"   {len(r067cr)} rows; columns: {list(r067cr.columns)}")
    assert len(r067cr) == n_test, "rally count mismatch"

    results = {}
    final_action_overrides = []
    final_point_overrides = []

    for task, true_labels, n_cls, hp_multi in [
        ("act", y_act, N_ACTION_TRAIN, LGBM_HP_MULTI_ACT),
        ("pt",  y_pt,  N_POINT,        LGBM_HP_MULTI_PT),
    ]:
        print(f"\n TASK: {task}")
        # Build OOF features
        X_oof, y_iswrong, blend_oof = per_row_features(comp_oof, task, true_labels, nsn)
        if task == "act":
            true_clip = np.where(true_labels >= N_ACTION_TRAIN, 0, true_labels)
        else:
            true_clip = true_labels

        # Train p_wrong on FULL OOF
        print(f"   [1] training p_wrong on full OOF ({len(X_oof)} rows)...")
        t0 = time.time()
        dtr = lgb.Dataset(X_oof, label=y_iswrong)
        m_wrong = lgb.train(LGBM_HP_BIN, dtr, num_boost_round=200)
        # Train alt-class on FULL OOF
        print(f"   [2] training alt-class on full OOF...")
        dtm = lgb.Dataset(X_oof, label=true_clip)
        m_alt = lgb.train(hp_multi, dtm, num_boost_round=200)
        print(f"      train time: {time.time()-t0:.1f}s")

        # Build TEST features
        X_test, blend_test = test_features(comp_test, task, nsn_test)
        # IMPORTANT: the "base" for override decisions is R-067cr's stored
        # predictions, NOT the uniform-blend argmax. Only flip rows the
        # corrector confidently wants to change.
        r067cr_col = "actionId" if task == "act" else "pointId"
        r067cr_argmax = r067cr[r067cr_col].to_numpy().astype(np.int64)

        # Predict on test
        p_wrong_test = m_wrong.predict(X_test)
        alt_probs_test = m_alt.predict(X_test)
        if alt_probs_test.ndim == 1:
            alt_probs_test = alt_probs_test.reshape(-1, 1)
        alt_argmax_test = alt_probs_test.argmax(axis=1)
        alt_maxp_test = alt_probs_test.max(axis=1)

        # Apply correction logic — only flip when corrector disagrees with R-067cr
        candidates = np.where(
            (p_wrong_test > P_WRONG_THRESHOLD)
            & (alt_argmax_test != r067cr_argmax)
            & (alt_maxp_test >= ALT_CONFIDENCE_THRESHOLD)
        )[0]
        rank_score = p_wrong_test[candidates] * alt_maxp_test[candidates]
        ranked = candidates[np.argsort(rank_score)[::-1]]
        n_override = min(MAX_OVERRIDES, len(ranked))
        override_idx = ranked[:n_override]
        # Start from R-067cr's stored predictions; only flip overrides
        corrected_argmax = r067cr_argmax.copy()
        corrected_argmax[override_idx] = alt_argmax_test[override_idx]

        # Diff vs R-067cr base — should equal n_override exactly
        diff = (corrected_argmax != r067cr_argmax).sum()
        print(f"   eligible: {len(candidates)}, applied: {n_override} overrides "
              f"(cap {MAX_OVERRIDES})")
        print(f"   diff vs R-067cr base: {diff} rows differ in {task}")

        # Collect change log
        change_log = []
        for i in override_idx:
            change_log.append({
                "rally_uid": int(r067cr.iloc[i]["rally_uid"]),
                "task": task,
                "from": int(r067cr_argmax[i]),
                "to": int(alt_argmax_test[i]),
                "p_wrong": float(p_wrong_test[i]),
                "alt_maxp": float(alt_maxp_test[i]),
            })
        if task == "act":
            final_action_overrides = change_log
        else:
            final_point_overrides = change_log

        # Update CSV column
        r067cr["actionId" if task == "act" else "pointId"] = corrected_argmax

        results[task] = {
            "n_eligible": int(len(candidates)),
            "n_overrides": int(n_override),
            "diff_vs_r067cr": int(diff),
        }

    # Sanity: SGP unchanged
    r067cr_orig = pd.read_csv(R067CR_BASE)
    assert np.array_equal(r067cr["serverGetPoint"].to_numpy(),
                           r067cr_orig["serverGetPoint"].to_numpy()), \
        "R-081 v2 MUST preserve SGP from R-067cr"

    # Save CSV
    r067cr.to_csv(OUT_CSV, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\n Saved candidate CSV: {OUT_CSV}")

    manifest = {
        "rid": "R-081-v2",
        "ts": "2026-05-26",
        "base_csv": R067CR_BASE,
        "output_csv": OUT_CSV,
        "fold1_smoke_delta_f1_act": +0.0003,   # from v2 smoke run
        "fold1_smoke_delta_f1_pt":  +0.0003,
        "results_per_task": results,
        "n_action_overrides": len(final_action_overrides),
        "n_point_overrides":  len(final_point_overrides),
        "sample_action_overrides": final_action_overrides[:10],
        "sample_point_overrides":  final_point_overrides[:10],
        "v04_candidate_report": {
            "theoretical_generalization_reason":
                "Bounded conditional correction on R-067cr LB-best. Train p_wrong + "
                "alt-class GBMs on FULL OOF (fold-safe by base-model construction). "
                "Apply only when both predictors agree, alt is different, and alt "
                "confidence >= 0.35. Cap 50 overrides per task (action/point only; "
                "SGP preserved from R-067cr).",
            "why_transfers_to_test_new":
                "Features (entropy, margin, agreement) are model-output-derived = "
                "distribution-invariant. Override cap caps wrong-direction LB damage "
                "near R-072 magnitude (-0.003). Mechanism distinct from R-054r meta_stack "
                "(replaced components vs additive bounded corrector).",
            "smoke_sanity_pass": True,
            "smoke_sanity_reason": "OK — Fold-1 ΔF1 act +0.0003 / pt +0.0003; "
                                    "p_wrong AUC 0.69/0.65; no leak; no catastrophe.",
            "lb_probe_worthy": True,
            "lb_confirm_hypothesis":
                "LB DeltaOV >= +0.001 (R-067cr + 0.001 = 0.388+) => bounded conditional "
                "correction transfers; mechanism distinct from pure meta-stacking.",
            "lb_reject_hypothesis":
                "LB DeltaOV <= -0.002 (R-067cr - 0.002 = 0.385-) => GBM-corrector route "
                "closed for this base model family. Treat as 2nd LB datapoint that "
                "GBM-meta is hard to make work for our zoo.",
            "predicted_lb_delta": "+0.0003 (Fold-1 smoke implies; small with high uncertainty)",
        },
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f" Saved manifest: {MANIFEST}")

    print("\n" + "=" * 80)
    print(" ARTIFACT_READY_FOR_JABIR_UPLOAD_REVIEW")
    print("=" * 80)
    print(f" File: {OUT_CSV}")
    print(f" Action overrides: {len(final_action_overrides)}")
    print(f" Point overrides:  {len(final_point_overrides)}")
    print(f" SGP: unchanged from R-067cr")
    print(f" Expected LB delta: +0.0003 (tiny; v2 smoke implies)")
    print(f" Risk: bounded at ~-0.003 worst case (override cap mechanism)")


if __name__ == "__main__":
    main()
