"""R-067 server-head blend builder — v2 (fix alignment via train_samples rebuild).

The v1 script tried to align R-034 per-shot OOF (69712 rows) with v22 per-rally
OOF (15833 rows) by aggregating R-034 per-rally and comparing y_srv. The
alignment failed because per-rally ordering differs.

v2 fix: REBUILD train_samples using exactly the same code path as
train_causal_lm_v1.py, then derive the canonical v22 rally_uid sequence. Use
this to join R-034 per-shot SGP → per-rally SGP indexed by rally_uid.

USAGE:
    python -u src/build_r067_server_blend_v2.py
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from config import PROJECT_ROOT, SUBMISSION_DIR, TRAIN_PATH, TEST_PATH  # noqa: E402
from data_cleaning import clean_data  # noqa: E402
from train_causal_lm_v1 import build_rally_samples  # noqa: E402
from analyze_oldtest_blend import load_components, evaluate_subset_none  # noqa: E402

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
R034 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]

# The R-066 v22 trainer was invoked with --include-old-test data/test.csv
# so we must replicate that data prep here to get matching rally_uids.
INCLUDE_OLD_TEST = os.path.join(PROJECT_ROOT, "data", "test.csv")


def main() -> None:
    print("=" * 78)
    print(" R-067 v2 — server-head blend with proper rally_uid alignment")
    print("=" * 78)

    # ─── Reproduce v22's train_samples to get rally_uid sequence ────────────
    print("\n Step 1: rebuild train_samples (same path as train_causal_lm_v1.py)")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test_new.csv"))
    n_before = len(raw_train)
    if os.path.exists(INCLUDE_OLD_TEST):
        old_test = pd.read_csv(INCLUDE_OLD_TEST)
        required_cols = list(raw_train.columns)
        raw_train = pd.concat([raw_train, old_test[required_cols]], ignore_index=True)
        print(f"   include-old-test: added {len(raw_train) - n_before} rows")

    train_df, _, _ = clean_data(raw_train, raw_test)
    train_samples = build_rally_samples(train_df, is_aug=False)
    v22_rally_uids = np.array([s["rally_uid"] for s in train_samples], dtype=np.int64)
    print(f"   v22 train rally_uids: {len(v22_rally_uids)}")

    # ─── Load v22 server OOF + ground-truth ──────────────────────────────────
    v22_oof_srv = np.load(os.path.join(OOF_DIR, "v22_causal_lm_v1_oof_srv.npy"))
    v22_oof_y_srv = np.load(os.path.join(OOF_DIR, "v22_causal_lm_v1_oof_y_srv.npy"))
    v22_oof_mask = np.load(os.path.join(OOF_DIR, "v22_causal_lm_v1_oof_mask.npy"))
    assert len(v22_oof_srv) == len(v22_rally_uids), (
        f"v22 OOF length {len(v22_oof_srv)} != rebuilt rally count {len(v22_rally_uids)}"
    )
    print(f"   v22 OOF aligned: {len(v22_oof_srv)} rows ({v22_oof_mask.sum()} valid)")

    # ─── Compute R-034 per-shot OOF SGP + map to per-rally ──────────────────
    print("\n Step 2: compute R-034 PAIR SGP + map per-shot → per-rally")
    comp, y_a, y_p, y_s, _, test_uid = load_components(R034)
    base = evaluate_subset_none(R034, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=300, seed=20260524)
    srv_stack = np.stack([comp[t]["oof_srv"] for t in R034], axis=0)
    r034_oof_srv_pershot = (base["w_s"][:, None] * srv_stack).sum(axis=0)  # 69712
    test_srv_stack = np.stack([comp[t]["test_srv"] for t in R034], axis=0)
    r034_test_srv = (base["w_s"][:, None] * test_srv_stack).sum(axis=0)
    print(f"   R-034 per-shot OOF: {len(r034_oof_srv_pershot)} rows  "
          f"AUC={roc_auc_score(y_s, r034_oof_srv_pershot):.4f}")

    # We need rally_uid per shot for R-034 OOF. Use features_v9 on CANONICAL
    # train only (no oldtest) so feat rally_uids match the 69712-row OOF set.
    raw_train_canonical = pd.read_csv(TRAIN_PATH)
    train_df_canonical, _, _ = clean_data(raw_train_canonical, raw_test)
    from features_v9 import compute_global_stats_v9, build_features_v9
    gs = compute_global_stats_v9(train_df_canonical)
    feat_full = build_features_v9(train_df_canonical, is_train=True,
                                   global_stats_v9=gs, raw_df=train_df_canonical)
    feat_rally_uid = feat_full["rally_uid"].astype(np.int64).values
    assert len(feat_rally_uid) == len(r034_oof_srv_pershot), (
        f"feat rally_uid count {len(feat_rally_uid)} != R-034 OOF {len(r034_oof_srv_pershot)}"
    )
    df = pd.DataFrame({
        "rally_uid": feat_rally_uid,
        "r034_srv": r034_oof_srv_pershot,
        "y_srv": y_s,
    })
    # Per-rally R-034 SGP = mean of per-shot predictions (SGP is rally-constant
    # but predictions differ slightly per shot because feature context differs)
    per_rally = df.groupby("rally_uid").agg({
        "r034_srv": "mean",
        "y_srv": "first",
    })
    print(f"   R-034 per-rally aggregated: {len(per_rally)} rallies")

    # ─── Align R-034 per-rally to v22's rally_uid order ──────────────────────
    print("\n Step 3: join R-034 per-rally to v22 rally_uid sequence")
    # v22_rally_uids may include duplicates? Check.
    if len(set(v22_rally_uids.tolist())) != len(v22_rally_uids):
        print(f"   WARN: v22 has duplicate rally_uids! "
              f"{len(set(v22_rally_uids))} unique vs {len(v22_rally_uids)}")
    # Lookup
    r034_per_rally_aligned = per_rally.reindex(v22_rally_uids)
    if r034_per_rally_aligned["r034_srv"].isna().any():
        n_missing = r034_per_rally_aligned["r034_srv"].isna().sum()
        print(f"   WARN: {n_missing} rally_uids missing from R-034 OOF (likely oldtest "
              f"rallies not in canonical 69712 OOF set); will mask those out")
    r034_srv_rally = r034_per_rally_aligned["r034_srv"].values  # (15833,)
    r034_y_rally = r034_per_rally_aligned["y_srv"].values

    # Validity mask: drop NaN + drop rows where v22 mask is False
    valid = (
        v22_oof_mask
        & ~np.isnan(r034_srv_rally)
        & (v22_oof_y_srv >= 0)
    )
    print(f"   Joint valid rows: {valid.sum()} / {len(valid)}")

    # Sanity check: y_srv should match (rally-constant)
    y_match = (r034_y_rally[valid] == v22_oof_y_srv[valid]).mean()
    print(f"   y_srv match rate: {y_match:.4f}")
    if y_match < 0.99:
        # Print mismatches
        ix = np.where((r034_y_rally != v22_oof_y_srv) & valid)[0][:10]
        for i in ix:
            print(f"     rally_uid={v22_rally_uids[i]}  R-034 y={r034_y_rally[i]}  v22 y={v22_oof_y_srv[i]}")

    # ─── α-sweep on per-rally OOF AUC ────────────────────────────────────────
    print(f"\n Step 4: α-sweep (blend = α * v22 + (1-α) * R-034)")
    sweep = []
    y_srv_valid = v22_oof_y_srv[valid]
    r034_valid = r034_srv_rally[valid]
    v22_valid = v22_oof_srv[valid]
    for alpha in np.linspace(0.0, 1.0, 21):
        blend = alpha * v22_valid + (1 - alpha) * r034_valid
        auc = float(roc_auc_score(y_srv_valid, blend))
        sweep.append({"alpha": float(alpha), "auc": auc})
    best = max(sweep, key=lambda s: s["auc"])
    base_auc = sweep[0]["auc"]   # α=0 → R-034 only
    v22_auc = sweep[-1]["auc"]   # α=1 → v22 only
    print(f"   α=0 (R-034 only) AUC: {base_auc:.4f}")
    print(f"   α=1 (v22 only)   AUC: {v22_auc:.4f}")
    print(f"   Best α={best['alpha']:.2f}  AUC={best['auc']:.4f}")
    print(f"   Lift vs R-034 baseline: {best['auc'] - base_auc:+.4f}")

    # Print full sweep
    print("\n   Per-α sweep:")
    for s in sweep:
        marker = " *BEST*" if s["alpha"] == best["alpha"] else ""
        print(f"     α={s['alpha']:.2f}  AUC={s['auc']:.4f}{marker}")

    # ─── Build test submissions with best α ──────────────────────────────────
    print(f"\n Step 5: build test submission with best α={best['alpha']:.2f}")
    v22_test_srv = np.load(os.path.join(OOF_DIR, "v22_causal_lm_v1_test_srv.npy"))
    v22_test_uid = np.load(os.path.join(OOF_DIR, "v22_causal_lm_v1_test_rally_uid.npy"))
    r042 = pd.read_csv(os.path.join(SUBMISSION_DIR, "submission_R042_R034_rule_override.csv"))
    assert np.array_equal(r042["rally_uid"].values, v22_test_uid), "test UID mismatch"

    alpha_best = best["alpha"]
    # blend test SGP at best α (using R-042's SGP as baseline, not raw R-034)
    blend_test = alpha_best * v22_test_srv + (1 - alpha_best) * r042["serverGetPoint"].values
    r067 = r042.copy()
    r067["serverGetPoint"] = blend_test

    fname = f"submission_R067cr_alpha{int(alpha_best*100):03d}_v22_blend_PLUS_RULE.csv"
    out_path = os.path.join(SUBMISSION_DIR, fname)
    r067.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"   Wrote: {out_path}")

    # Also save full-replace (α=1.0) as comparison
    fname_replace = "submission_R067cr_alpha100_v22_full_replace_PLUS_RULE.csv"
    out_replace = os.path.join(SUBMISSION_DIR, fname_replace)
    r067_full = r042.copy()
    r067_full["serverGetPoint"] = v22_test_srv
    r067_full.to_csv(out_replace, index=False, lineterminator="\n", encoding="utf-8")
    print(f"   Wrote: {out_replace} (α=1.0 reference)")

    # ─── Manifest ────────────────────────────────────────────────────────────
    manifest = {
        "rid": "R-067c",
        "ts": "2026-05-24",
        "alignment_fix": "v2 — rebuild train_samples to derive v22's rally_uid order, join R-034 per-rally via reindex",
        "y_srv_match_rate": float(y_match),
        "valid_oof_rows": int(valid.sum()),
        "R042_R034_baseline_AUC_per_rally_OOF": base_auc,
        "v22_AUC_per_rally_OOF": v22_auc,
        "alpha_sweep": sweep,
        "best_alpha": float(best["alpha"]),
        "best_auc": float(best["auc"]),
        "AUC_lift_vs_R034_baseline": float(best["auc"] - base_auc),
        "submissions": {
            "R-067c_best_alpha": fname,
            "R-067c_full_replace": fname_replace,
        },
    }
    out_json = os.path.join(SUBMISSION_DIR, "r067c_server_blend_alpha_sweep.json")
    with open(out_json, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved manifest: {out_json}")

    # ─── Predicted LB estimate ───────────────────────────────────────────────
    # If AUC lift transfers 1:1 → OV lift = (AUC_lift) * 0.2 (server weight in OV)
    # R-042 LB = 0.3866
    auc_lift = best["auc"] - base_auc
    ov_lift_full_transfer = auc_lift * 0.2
    ov_lift_partial_transfer = auc_lift * 0.2 * 0.5  # half-transfer rule of thumb
    print(f"\n=== Predicted LB (R-067c at best α={best['alpha']:.2f}) ===")
    print(f"  AUC lift (OOF): {auc_lift:+.4f}")
    print(f"  OV lift (full transfer):    {ov_lift_full_transfer:+.4f} → LB ≈ {0.3866 + ov_lift_full_transfer:.4f}")
    print(f"  OV lift (partial transfer): {ov_lift_partial_transfer:+.4f} → LB ≈ {0.3866 + ov_lift_partial_transfer:.4f}")
    print(f"  OV lift (no transfer):       0.0000          → LB ≈ {0.3866:.4f}")


if __name__ == "__main__":
    main()
