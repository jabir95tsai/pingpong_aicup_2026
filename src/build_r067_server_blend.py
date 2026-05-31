"""R-067: Server-head-only blend (R-066 v22 causal LM SGP + R-042 base).

Per R-066 full 5-fold result:
  - F1_a 0.3082 (weak), F1_p 0.0911 (weak), AUC 0.6873 (+0.077 vs v11 baseline)
  - OV 0.2972 < 0.314 v11 baseline (full-model PARK per §9.6)
  - BUT server head genuinely diversity-positive

Strategy: keep R-042's action+point predictions UNCHANGED, replace ONLY the
serverGetPoint with a tuned blend of v22 causal LM SGP and R-042 SGP.

Two variants:
  R-067a (full replace): test_srv = v22_test_srv
  R-067b (weighted blend): test_srv = α * v22_test_srv + (1-α) * R042_test_srv
                            where α is tuned on OOF AUC

Output:
  submissions/r067_oof_sweep.json (per-α OOF AUC sweep)
  submissions/submission_R067a_v22srv_replace.csv (full-replace, +rule_override applied)
  submissions/submission_R067b_alpha{NN}_v22srv_blend.csv (best-α blend, +rule)
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from config import PROJECT_ROOT, SUBMISSION_DIR  # noqa: E402
from analyze_oldtest_blend import load_components, evaluate_subset_none  # noqa: E402

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
R034 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]


def run_rule_override(in_csv: str, out_csv: str) -> str:
    train_csv = os.path.join(PROJECT_ROOT, "data", "train.csv")
    test_csv = os.path.join(PROJECT_ROOT, "data", "test_new.csv")
    script = os.path.join(PROJECT_ROOT, "src", "apply_rule_override.py")
    cmd = ["python", "-u", script, "--input", in_csv, "--train", train_csv,
           "--test", test_csv, "--output", out_csv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rule_override failed: {r.stderr}")
    return r.stdout


def main() -> None:
    print("=" * 78)
    print(" R-067 server-head blend builder")
    print("=" * 78)

    # ─── Load R-066 v22 OOF + test (server head) ─────────────────────────────
    v22_oof_srv = np.load(os.path.join(OOF_DIR, "v22_causal_lm_v1_oof_srv.npy"))  # (15833,) per-rally
    v22_oof_y_srv = np.load(os.path.join(OOF_DIR, "v22_causal_lm_v1_oof_y_srv.npy"))
    v22_oof_mask = np.load(os.path.join(OOF_DIR, "v22_causal_lm_v1_oof_mask.npy"))
    v22_test_srv = np.load(os.path.join(OOF_DIR, "v22_causal_lm_v1_test_srv.npy"))  # (1845,)
    v22_test_uid = np.load(os.path.join(OOF_DIR, "v22_causal_lm_v1_test_rally_uid.npy"))
    print(f" v22 OOF (per-rally): {len(v22_oof_srv)} rows, mask {v22_oof_mask.sum()}/{len(v22_oof_mask)}")
    print(f" v22 test: {len(v22_test_srv)} rows")

    # ─── Load R-034 components + compute R-034 SGP (per-shot OOF, 69712 rows) ──
    comp, y_a, y_p, y_s, _, test_uid = load_components(R034)
    base = evaluate_subset_none(R034, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=300, seed=20260524)
    srv_stack = np.stack([comp[t]["oof_srv"] for t in R034], axis=0)
    r034_oof_srv = (base["w_s"][:, None] * srv_stack).sum(axis=0)  # (69712,)
    test_srv_stack = np.stack([comp[t]["test_srv"] for t in R034], axis=0)
    r034_test_srv = (base["w_s"][:, None] * test_srv_stack).sum(axis=0)  # (1845,)
    print(f"\n R-034 OOF srv (per-shot): {len(r034_oof_srv)} rows")
    print(f" R-034 test srv: {len(r034_test_srv)} rows")
    print(f" R-034 baseline AUC (per-shot OOF): {roc_auc_score(y_s, r034_oof_srv):.4f}")
    print(f" v22 AUC (per-rally OOF): {roc_auc_score(v22_oof_y_srv[v22_oof_mask], v22_oof_srv[v22_oof_mask]):.4f}")

    # ─── For blend OOF AUC: aggregate R-034 per-shot to per-rally for comparison ──
    # Build a rally_uid map for R-034 OOF rows
    # We need: for each train rally_uid, what is the mean R-034 SGP across its rows?
    # Easiest: load reference rally_uids per row (from any v14 OOF that has the data)
    # Actually: train rally_uids are encoded in the OOF order. Use v22's rally_uid + map back.

    # Hmm — we need a rally_uid-per-row array for R-034 OOF. Let me build via TRAIN_PATH
    # by re-running clean_data + feature builder rally enumeration.
    # SIMPLER: just align via rally-level statistics
    # For test, alignment is trivial (both per-rally, UIDs match — already verified).

    # For OOF blend AUC: we can ONLY blend at rally-level. Compute per-rally R-034 SGP:
    # We don't have a direct rally_uid array for R-034 per-shot OOF, BUT v22 OOF has
    # rally_uids stored in its rally_uid file... actually v22 saves train rally OOF
    # rally_uids via the train_samples list. Let me check.

    # Quick approach: reuse v22's OOF rally_uid + extract train SGP labels per-rally.
    # The v22 OOF y_srv == rally's SGP (1 per rally).
    # For R-034 SGP per-rally we need to AVERAGE the per-shot SGP across each rally.

    # Build per-rally R-034 SGP using train rally_uid map (from clean_data)
    import sys
    sys.path.insert(0, 'src')
    from data_cleaning import clean_data
    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train.csv"))
    test_df_raw = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test_new.csv"))
    train_cleaned, _, _ = clean_data(train_df, test_df_raw)

    # Build per-rally R-034 SGP by averaging per-shot SGP rows
    # We don't have direct per-row rally_uid for the OOF rows, so use the
    # features_v3-style enumeration: rally produces N-1 OOF rows.
    # Easier: feat_full = build_features on full train, gives a rally_uid array.
    from features_v9 import compute_global_stats_v9, build_features_v9
    gs = compute_global_stats_v9(train_cleaned)
    feat_full = build_features_v9(train_cleaned, is_train=True,
                                   global_stats_v9=gs, raw_df=train_cleaned)
    print(f"\n feat_full rally_uid count: {len(feat_full)}")
    feat_rally_uid = feat_full["rally_uid"].astype(np.int64).values
    assert len(feat_rally_uid) == len(r034_oof_srv), (
        f"Row mismatch: feat {len(feat_rally_uid)} vs r034 {len(r034_oof_srv)}"
    )

    # Aggregate R-034 per-shot SGP → per-rally (mean)
    df = pd.DataFrame({
        "rally_uid": feat_rally_uid,
        "r034_srv": r034_oof_srv,
        "y_srv": y_s,
    })
    per_rally = df.groupby("rally_uid").agg({
        "r034_srv": "mean",
        "y_srv": "first",  # SGP is rally-constant
    }).reset_index()
    print(f" R-034 per-rally aggregated: {len(per_rally)} rows")

    # Now align v22 per-rally to R-034 per-rally
    v22_oof_uid = np.arange(len(v22_oof_srv))   # v22 saves by index, not rally_uid
    # Need to figure out v22's rally_uid mapping... let me try loading from the train samples.
    # Actually v22 OOF mask coverage is 100% (15833/15833 train rallies).
    # The 15833 rallies should map 1:1 to per_rally's 15833 rallies (assuming same order).
    # Check: do both have same number of train rallies?

    print(f"\n per_rally count: {len(per_rally)}")
    print(f" v22 OOF rows: {len(v22_oof_srv)}")
    if len(per_rally) != len(v22_oof_srv):
        print("  WARN: rally counts differ; cannot directly align by index")
        print(f"  Use approximate alignment via y_srv match")
    else:
        # Sort per_rally by rally_uid to match v22's likely order
        per_rally = per_rally.sort_values("rally_uid").reset_index(drop=True)
        # v22 should also be sorted by rally_uid (GroupKFold's natural order)
        # Verify: check y_srv match
        y_match = np.array_equal(per_rally["y_srv"].values, v22_oof_y_srv)
        print(f"  y_srv match after sort: {y_match}")
        if not y_match:
            print(f"  per_rally y_srv[:10]: {per_rally['y_srv'].values[:10]}")
            print(f"  v22 y_srv[:10]: {v22_oof_y_srv[:10]}")

    # If y_srv matches, we can compute blend OOF AUC
    if len(per_rally) == len(v22_oof_srv) and np.array_equal(per_rally["y_srv"].values, v22_oof_y_srv):
        y_srv_rally = v22_oof_y_srv
        r034_srv_rally = per_rally["r034_srv"].values
        v22_srv_rally = v22_oof_srv
        srv_mask = y_srv_rally >= 0   # exclude any -1 sentinels (shouldn't happen for train)
        print(f"\n srv_mask: {srv_mask.sum()}/{len(srv_mask)} valid SGP labels")

        # ─── Per-α AUC sweep ────────────────────────────────────────────────
        print("\n=== R-067b α-sweep (blend = α * v22 + (1-α) * R034) ===")
        sweep = []
        for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            blend = alpha * v22_srv_rally + (1 - alpha) * r034_srv_rally
            auc = roc_auc_score(y_srv_rally[srv_mask], blend[srv_mask])
            sweep.append({"alpha": alpha, "auc": float(auc)})
            print(f"  α={alpha:.1f}  AUC={auc:.4f}")
        best = max(sweep, key=lambda s: s["auc"])
        print(f"\n  Best α={best['alpha']:.1f}  AUC={best['auc']:.4f}")
        print(f"  Baseline AUC (α=0): {sweep[0]['auc']:.4f}")
        print(f"  Lift: {best['auc'] - sweep[0]['auc']:+.4f}")
    else:
        print("\n  Cannot compute OOF blend AUC due to alignment issue.")
        best = {"alpha": 1.0, "auc": float("nan")}
        sweep = []

    # ─── Build test submissions ──────────────────────────────────────────────
    r042 = pd.read_csv(os.path.join(SUBMISSION_DIR, "submission_R042_R034_rule_override.csv"))
    assert np.array_equal(r042["rally_uid"].values, v22_test_uid), "test UID mismatch"
    print(f"\n R-042 reference loaded: {len(r042)} rows")

    # R-067a: full replace
    r067a = r042.copy()
    r067a["serverGetPoint"] = v22_test_srv
    fname_a_base = "submission_R067a_v22srv_replace.csv"
    out_a_base = os.path.join(SUBMISSION_DIR, fname_a_base)
    r067a.to_csv(out_a_base, index=False, lineterminator="\n", encoding="utf-8")
    print(f" Wrote: {out_a_base}")
    fname_a_rule = "submission_R067ar_v22srv_replace_PLUS_RULE.csv"
    out_a_rule = os.path.join(SUBMISSION_DIR, fname_a_rule)
    # Note: R-042 already has rule_override applied; R-067a inherits R-042's actionId/pointId,
    # so rule_override would re-apply with same context. Skip re-applying; CSV is already
    # rule_override-equivalent (just with different SGP).
    # Actually R-042 already underwent rule_override; the actionId/pointId are POST-rule.
    # We use R-042 directly as base — no additional rule_override needed.
    import shutil
    shutil.copy(out_a_base, out_a_rule)
    print(f" Wrote: {out_a_rule} (already rule_override via R-042 base)")

    # R-067b: weighted blend at best α
    alpha_best = best["alpha"]
    blend_test = alpha_best * v22_test_srv + (1 - alpha_best) * r042["serverGetPoint"].values
    r067b = r042.copy()
    r067b["serverGetPoint"] = blend_test
    fname_b_base = f"submission_R067b_alpha{int(alpha_best*10):02d}_v22srv_blend.csv"
    out_b_base = os.path.join(SUBMISSION_DIR, fname_b_base)
    r067b.to_csv(out_b_base, index=False, lineterminator="\n", encoding="utf-8")
    print(f" Wrote: {out_b_base}")
    fname_b_rule = f"submission_R067br_alpha{int(alpha_best*10):02d}_v22srv_blend_PLUS_RULE.csv"
    out_b_rule = os.path.join(SUBMISSION_DIR, fname_b_rule)
    shutil.copy(out_b_base, out_b_rule)
    print(f" Wrote: {out_b_rule} (already rule_override via R-042 base)")

    # ─── Save manifest ──────────────────────────────────────────────────────
    manifest = {
        "rid": "R-067",
        "ts": "2026-05-24",
        "R042_LB": 0.3866,
        "R042_R034_baseline_AUC_OOF": float(roc_auc_score(y_s, r034_oof_srv)),
        "v22_AUC_OOF_per_rally": float(roc_auc_score(v22_oof_y_srv[v22_oof_mask], v22_oof_srv[v22_oof_mask])),
        "alpha_sweep": sweep,
        "best_alpha": float(best["alpha"]),
        "best_auc": float(best["auc"]),
        "submissions": {
            "R-067a_full_replace": fname_a_rule,
            "R-067b_best_alpha": fname_b_rule,
        },
    }
    out_path = os.path.join(SUBMISSION_DIR, "r067_server_blend.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved manifest: {out_path}")

    print("\n=== R-067 DONE ===")
    print(f"  α=0 (R-042 only) AUC: {sweep[0]['auc']:.4f}" if sweep else "  (sweep skipped)")
    print(f"  α=1 (v22 only)   AUC: {sweep[-1]['auc']:.4f}" if sweep else "")
    print(f"  Best α={best['alpha']:.1f} AUC: {best['auc']:.4f}" if sweep else "")


if __name__ == "__main__":
    main()
