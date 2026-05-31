"""Holdout-aware blend-swap audit — read-only diagnostic.

Per user 2026-05-24:
- Run blend-swap audit using HOLDOUT as additional column (NOT a hard gate).
- Evaluate at BLEND level, not standalone.
- Include R-058/R-059/R-060/R-061/R-068/R-070 candidates if artifacts exist.
- Report full OOF, holdout OV, holdout AUC, F1_a, F1_p, known toxic class.
- Compare against R-067c LB = 0.3870095, NOT old R-042 0.3866550.
- Do not materialize/upload anything; pure diagnostic.

Baseline: R-034 PAIR (Dirichlet n=300 weights) — same as R-042's base.

Output:
  submissions/holdout_aware_blend_audit.json (full results)
  submissions/holdout_aware_blend_audit.txt (human-readable ranking)

USAGE:
    python -u src/holdout_aware_blend_audit.py
"""
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa: E402
from analyze_oldtest_blend import (  # noqa: E402
    load_components, evaluate_subset_none, N_ACTION, N_POINT,
    ACTION_EVAL, POINT_EVAL, fast_macro_f1,
)

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
HOLDOUT_PATH = os.path.join(PROJECT_ROOT, "data", "player_holdout_idx.npy")

R034_PAIR = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]
R042_LB = 0.3866550
R067cr_LB = 0.3870095
RATIO_CONS = 1.0035
RATIO_OPT = 1.0142
RULE_LIFT_LB = 0.0028

# Candidates per user instruction
CANDIDATES = [
    # (rid, slot_to_swap, swap_in, toxic_class_warning, note)
    ("R-058", "v14_seed2_v15feat_a", "v14_seed2_v15feat_c_oldtest_avg3", "B-feature (avg3)", "v15feat_c 3-seed avg + oldtest"),
    ("R-059", "v14_seed2_v15feat_a", "v14_seed2_v15feat_a_oldtest_avg3", "B-feature (avg3)", "v15feat_a 3-seed avg + oldtest"),
    ("R-060", "v14_seed2_v15feat_a", "v14_recvprofile",                  "B-player-style risk", "receiver-profile (no oldtest)"),
    ("R-061", "v14_seed2_v15feat_a", "v14_recvhand",                     "B-player-style risk", "receiver-hand (no oldtest)"),
    ("R-058oldtest_swap", "v14_seed2_v15feat_a", "v14_seed2_v15feat_c_oldtest", "B-feature", "v15feat_c single seed + oldtest (Codex-favored on holdout AUC)"),
]


def compute_blend_metrics(subset: List[str], comp: Dict, y_a, y_p, y_s,
                           holdout_mask: np.ndarray,
                           w_a=None, w_p=None, w_s=None,
                           n_samples: int = 300, seed: int = 20260524) -> Dict:
    """Compute full + holdout blend metrics.

    If weights are None, uses Dirichlet search to find optimal NONE-blend weights.
    """
    if w_a is None:
        m = evaluate_subset_none(subset, comp, y_a, y_p, y_s,
                                  optimize=True, n_samples=n_samples, seed=seed)
        w_a, w_p, w_s = m["w_a"], m["w_p"], m["w_s"]

    act_stack = np.stack([comp[t]["oof_act"] for t in subset], axis=0)
    pt_stack = np.stack([comp[t]["oof_pt"] for t in subset], axis=0)
    srv_stack = np.stack([comp[t]["oof_srv"] for t in subset], axis=0)
    blend_a = (w_a[:, None, None] * act_stack).sum(axis=0)
    blend_p = (w_p[:, None, None] * pt_stack).sum(axis=0)
    blend_s = (w_s[:, None] * srv_stack).sum(axis=0)

    # Full OOF metrics
    pred_a = blend_a.argmax(axis=1)
    pred_p = blend_p.argmax(axis=1)
    f1_a_full = fast_macro_f1(y_a, pred_a, ACTION_EVAL, N_ACTION)
    f1_p_full = fast_macro_f1(y_p, pred_p, POINT_EVAL, N_POINT)
    auc_full = roc_auc_score(y_s, blend_s)
    ov_full = 0.4 * f1_a_full + 0.4 * f1_p_full + 0.2 * auc_full

    # Holdout-only metrics
    m = holdout_mask
    f1_a_hd = fast_macro_f1(y_a[m], pred_a[m], ACTION_EVAL, N_ACTION)
    f1_p_hd = fast_macro_f1(y_p[m], pred_p[m], POINT_EVAL, N_POINT)
    auc_hd = roc_auc_score(y_s[m], blend_s[m])
    ov_hd = 0.4 * f1_a_hd + 0.4 * f1_p_hd + 0.2 * auc_hd

    return {
        "full": {
            "OV": float(ov_full), "F1_a": float(f1_a_full),
            "F1_p": float(f1_p_full), "AUC": float(auc_full),
        },
        "holdout": {
            "OV": float(ov_hd), "F1_a": float(f1_a_hd),
            "F1_p": float(f1_p_hd), "AUC": float(auc_hd),
        },
        "w_a": w_a.tolist(), "w_p": w_p.tolist(), "w_s": w_s.tolist(),
    }


def main() -> None:
    print("=" * 95)
    print(" HOLDOUT-AWARE BLEND-SWAP AUDIT — read-only diagnostic")
    print(" Holdout used as ADVISORY signal only (not a hard LB gate)")
    print(f" Compare to R-067cr LB-best = {R067cr_LB} (NOT old R-042 = {R042_LB})")
    print("=" * 95)

    holdout = np.load(HOLDOUT_PATH)
    print(f"\n P11 holdout: {holdout.sum()}/{len(holdout)} rows = "
          f"{100*holdout.sum()/len(holdout):.1f}% of OOF")

    # Load all unique components
    all_tags = list(set(R034_PAIR + [c[2] for c in CANDIDATES]))
    print(f"\n Loading {len(all_tags)} components ...")
    comp, y_a, y_p, y_s, _, _ = load_components(all_tags)
    available = [t for t in all_tags if t in comp]
    print(f" Loaded: {len(available)}/{len(all_tags)}")

    # ─── Baseline: R-034 PAIR Dirichlet ──────────────────────────────────────
    print("\n" + "=" * 95)
    print(" BASELINE: R-034 PAIR (Dirichlet n=300, same as R-042's base before rule_override)")
    print("=" * 95)
    base = compute_blend_metrics(R034_PAIR, comp, y_a, y_p, y_s, holdout)
    print(f"\n   FULL    OV={base['full']['OV']:.4f}  F1a={base['full']['F1_a']:.4f}  "
          f"F1p={base['full']['F1_p']:.4f}  AUC={base['full']['AUC']:.4f}")
    print(f"   HOLDOUT OV={base['holdout']['OV']:.4f}  F1a={base['holdout']['F1_a']:.4f}  "
          f"F1p={base['holdout']['F1_p']:.4f}  AUC={base['holdout']['AUC']:.4f}")
    print(f"   Holdout vs Full Δ:  OV={base['holdout']['OV'] - base['full']['OV']:+.4f}  "
          f"F1a={base['holdout']['F1_a'] - base['full']['F1_a']:+.4f}  "
          f"F1p={base['holdout']['F1_p'] - base['full']['F1_p']:+.4f}  "
          f"AUC={base['holdout']['AUC'] - base['full']['AUC']:+.4f}")
    base_full_ov = base["full"]["OV"]
    base_holdout_ov = base["holdout"]["OV"]

    # ─── R-068: Bayes weights on R-034 (from saved manifest) ─────────────────
    bayes_path = os.path.join(SUBMISSION_DIR, "bayes_r034_safe_search.json")
    bayes_result = None
    if os.path.exists(bayes_path):
        with open(bayes_path) as f:
            bj = json.load(f)
        w_a = np.array(bj["weights"]["w_a"])
        w_p = np.array(bj["weights"]["w_p"])
        w_s = np.array(bj["weights"]["w_s"])
        bayes_result = compute_blend_metrics(R034_PAIR, comp, y_a, y_p, y_s, holdout,
                                              w_a=w_a, w_p=w_p, w_s=w_s)
        print(f"\n R-068 Bayes weights on R-034 PAIR:")
        print(f"   FULL    OV={bayes_result['full']['OV']:.4f}  dOV={bayes_result['full']['OV']-base_full_ov:+.4f}  "
              f"AUC={bayes_result['full']['AUC']:.4f}")
        print(f"   HOLDOUT OV={bayes_result['holdout']['OV']:.4f}  dOV={bayes_result['holdout']['OV']-base_holdout_ov:+.4f}  "
              f"AUC={bayes_result['holdout']['AUC']:.4f}")

    # ─── Candidate swap audits ──────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(" CANDIDATE SWAPS (blend-level; not standalone)")
    print("=" * 95)
    print(f"\n {'R-id':<24} {'Class':<22} {'Full dOV':<10} {'HoldOut dOV':<13} {'Full AUC':<10} {'Hold AUC':<10}")
    print(" " + "-" * 92)

    rows = []
    for rid, slot, swap_in, cls, note in CANDIDATES:
        if swap_in not in comp:
            print(f"   {rid:<24} {cls:<22}   SKIP — {swap_in} not loaded")
            continue
        new_subset = [swap_in if t == slot else t for t in R034_PAIR]
        m = compute_blend_metrics(new_subset, comp, y_a, y_p, y_s, holdout)
        d_full = m["full"]["OV"] - base_full_ov
        d_holdout = m["holdout"]["OV"] - base_holdout_ov
        marker_f = "*" if d_full >= 0.001 else (" " if d_full >= 0 else "-")
        marker_h = "*" if d_holdout >= 0.001 else (" " if d_holdout >= 0 else "-")
        print(f"   {rid:<24} {cls:<22}   {d_full:+.4f}{marker_f}    "
              f"{d_holdout:+.4f}{marker_h}     {m['full']['AUC']:.4f}     {m['holdout']['AUC']:.4f}")
        rows.append({
            "rid": rid, "slot": slot, "swap_in": swap_in,
            "class": cls, "note": note,
            "full": m["full"], "holdout": m["holdout"],
            "dOV_full": float(d_full), "dOV_holdout": float(d_holdout),
            "AUC_delta_holdout_vs_full": float(m["holdout"]["AUC"] - m["full"]["AUC"]),
        })

    # ─── Predicted LB comparison (against R-067cr) ──────────────────────────
    print("\n" + "=" * 95)
    print(f" PREDICTED LB vs R-067cr (= {R067cr_LB})")
    print(" Note: R-067cr already has +0.000355 LB lift from server-blend stacked on R-034")
    print(" Predicted LB+rule for these candidates uses R-034 baseline ratio, no v22 SGP blend")
    print("=" * 95)
    print(f"\n {'R-id':<24} {'Pred LB+rule (cons-opt)':<30} {'vs R-067cr midpoint':<22} {'Notes':<30}")
    print(" " + "-" * 95)
    for r in rows:
        ov = r["full"]["OV"]
        pred_lo = ov * RATIO_CONS + RULE_LIFT_LB
        pred_hi = ov * RATIO_OPT + RULE_LIFT_LB
        pred_mid = (pred_lo + pred_hi) / 2
        delta_mid = pred_mid - R067cr_LB
        sign = "+" if delta_mid >= 0 else ""
        print(f"   {r['rid']:<24} {pred_lo:.4f} - {pred_hi:.4f}      "
              f"{sign}{delta_mid:.4f}              {r['class']:<30}")

    # ─── User instruction: include R-068 (Bayes) and R-070 (not yet built) ──
    if bayes_result is not None:
        print(f"\n R-068 Bayes-on-R-034-safe (weight refinement, NOT a swap):")
        bayes_ov = bayes_result["full"]["OV"]
        pred_lo = bayes_ov * RATIO_CONS + RULE_LIFT_LB
        pred_hi = bayes_ov * RATIO_OPT + RULE_LIFT_LB
        pred_mid = (pred_lo + pred_hi) / 2
        print(f"   Pred LB+rule: {pred_lo:.4f} - {pred_hi:.4f} (mid {pred_mid:.4f}, "
              f"{pred_mid - R067cr_LB:+.4f} vs R-067cr)")
        print(f"   Holdout dOV vs baseline: {bayes_result['holdout']['OV'] - base_holdout_ov:+.4f}")

    print(f"\n R-070 v15feat_e (not yet implemented; build per Codex fixes after this audit)")

    # ─── Save manifest ──────────────────────────────────────────────────────
    manifest = {
        "ts": "2026-05-24",
        "purpose": "Holdout-aware blend-swap audit (read-only diagnostic, NOT a hard LB gate)",
        "p11_holdout_rows": int(holdout.sum()),
        "lb_best_R067cr": R067cr_LB,
        "lb_R042_baseline": R042_LB,
        "rule_lift_LB": RULE_LIFT_LB,
        "transfer_ratio_conservative": RATIO_CONS,
        "transfer_ratio_optimistic": RATIO_OPT,
        "R034_PAIR_baseline": base,
        "R068_bayes_safe": bayes_result,
        "swap_candidates": rows,
        "interpretation": (
            "Holdout dOV > 0 AND full dOV ≥ -0.002 → candidate worth Codex review "
            "for LB upload. Holdout-negative candidates should be deprioritized. "
            "Holdout is ADVISORY only per user 2026-05-24."
        ),
    }
    out_path = os.path.join(SUBMISSION_DIR, "holdout_aware_blend_audit.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved: {out_path}")

    # Human-readable text
    txt_path = os.path.join(SUBMISSION_DIR, "holdout_aware_blend_audit.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("HOLDOUT-AWARE BLEND-SWAP AUDIT (2026-05-24)\n")
        f.write("=" * 80 + "\n")
        f.write(f"R-067cr LB-best: {R067cr_LB}\n")
        f.write(f"R-042 baseline:  {R042_LB}\n")
        f.write(f"Baseline R-034 PAIR full OV: {base_full_ov:.4f}, holdout OV: {base_holdout_ov:.4f}\n\n")
        f.write("Candidates (sorted by holdout dOV):\n")
        for r in sorted(rows, key=lambda r: -r["dOV_holdout"]):
            f.write(f"  {r['rid']:<28} class={r['class']:<24}  "
                    f"full dOV={r['dOV_full']:+.4f}  holdout dOV={r['dOV_holdout']:+.4f}  "
                    f"({r['note']})\n")
        if bayes_result is not None:
            f.write(f"  {'R-068 Bayes-safe':<28} class=weight-refinement     "
                    f"full dOV={bayes_result['full']['OV']-base_full_ov:+.4f}  "
                    f"holdout dOV={bayes_result['holdout']['OV']-base_holdout_ov:+.4f}\n")
    print(f" Saved: {txt_path}")


if __name__ == "__main__":
    main()
