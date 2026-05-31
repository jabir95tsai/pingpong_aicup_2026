"""R-075 server-head blend builder — uses R-071 v4 (focal+CB) server head.

Analog of R-067 v2, but blends R-071 v4's improved server head instead of R-066
v3's. R-071 v4 had AUC 0.6804 on smoke vs v3's 0.6759 = +0.0045 improvement.
If the full 5-fold preserves this AUC lift, expected LB gain via the same R-067
mechanism is ~+0.0004 (= +0.0045 × 0.2 OV-weight × 0.05 transfer multiplier
per candidate_goal's server-head-blend prior).

PRE-REQUISITE: R-071 v4 full 5-fold completed and OOF arrays pulled to
`oof_predictions/v22_causal_lm_v4_full_*.npy`.

USAGE:
    python -u src/build_r075_server_blend_v4.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["ALLOW_UID_MISMATCH"] = "1"
from config import PROJECT_ROOT, SUBMISSION_DIR, TRAIN_PATH  # noqa: E402
from data_cleaning import clean_data  # noqa: E402
from train_causal_lm_v1 import build_rally_samples  # noqa: E402
from analyze_oldtest_blend import load_components, evaluate_subset_none  # noqa: E402

OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")
R034 = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]

INCLUDE_OLD_TEST = os.path.join(PROJECT_ROOT, "data", "test.csv")
V4_TAG = "v22_causal_lm_v4_full"


def main() -> None:
    print("=" * 78)
    print(" R-075 — server-head blend using R-071 v4 (focal+CB) server head")
    print("=" * 78)

    # Verify the v4 full 5-fold artifacts are present
    required = [
        f"{V4_TAG}_oof_srv.npy", f"{V4_TAG}_oof_y_srv.npy",
        f"{V4_TAG}_oof_mask.npy", f"{V4_TAG}_test_srv.npy",
        f"{V4_TAG}_test_rally_uid.npy",
    ]
    for fname in required:
        path = os.path.join(OOF_DIR, fname)
        if not os.path.exists(path):
            print(f" MISSING: {path}")
            print(" Run R-071 v4 full 5-fold on Kaggle first, then pull outputs:")
            print(f"   kaggle kernels output jabir95tsai/aicup-r-071-causal-lm-v4-focal-full5fold "
                  f"-p kaggle_pulls/r071_full/")
            print(f"   cp kaggle_pulls/r071_full/oof_predictions/{V4_TAG}_*.npy {OOF_DIR}/")
            sys.exit(1)

    # Rebuild v4 train_samples rally_uid sequence (same code path as trainer)
    print("\n Step 1: rebuild train_samples for v4 rally_uid alignment")
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
    v4_rally_uids = np.array([s["rally_uid"] for s in train_samples], dtype=np.int64)
    print(f"   v4 train rally_uids: {len(v4_rally_uids)}")

    # Load v4 OOF
    v4_oof_srv = np.load(os.path.join(OOF_DIR, f"{V4_TAG}_oof_srv.npy"))
    v4_oof_y_srv = np.load(os.path.join(OOF_DIR, f"{V4_TAG}_oof_y_srv.npy"))
    v4_oof_mask = np.load(os.path.join(OOF_DIR, f"{V4_TAG}_oof_mask.npy"))
    assert len(v4_oof_srv) == len(v4_rally_uids), (
        f"v4 OOF length {len(v4_oof_srv)} != rebuilt rally count {len(v4_rally_uids)}"
    )
    print(f"   v4 OOF: {len(v4_oof_srv)} rows ({v4_oof_mask.sum()} valid)")

    # R-034 per-shot OOF → per-rally aggregation
    print("\n Step 2: compute R-034 PAIR baseline SGP per-rally")
    comp, y_a, y_p, y_s, _, test_uid = load_components(R034)
    base = evaluate_subset_none(R034, comp, y_a, y_p, y_s,
                                 optimize=True, n_samples=300, seed=20260524)
    srv_stack = np.stack([comp[t]["oof_srv"] for t in R034], axis=0)
    r034_oof_srv_pershot = (base["w_s"][:, None] * srv_stack).sum(axis=0)
    print(f"   R-034 per-shot OOF AUC: {roc_auc_score(y_s, r034_oof_srv_pershot):.4f}")

    raw_train_canonical = pd.read_csv(TRAIN_PATH)
    train_df_canonical, _, _ = clean_data(raw_train_canonical, raw_test)
    from features_v9 import compute_global_stats_v9, build_features_v9
    gs = compute_global_stats_v9(train_df_canonical)
    feat_full = build_features_v9(train_df_canonical, is_train=True,
                                   global_stats_v9=gs, raw_df=train_df_canonical)
    feat_rally_uid = feat_full["rally_uid"].astype(np.int64).values
    df = pd.DataFrame({
        "rally_uid": feat_rally_uid,
        "r034_srv": r034_oof_srv_pershot,
        "y_srv": y_s,
    })
    per_rally = df.groupby("rally_uid").agg({"r034_srv": "mean", "y_srv": "first"})

    # Align to v4 rally order
    print("\n Step 3: join R-034 per-rally to v4 rally_uid sequence")
    r034_aligned = per_rally.reindex(v4_rally_uids)
    valid = (
        v4_oof_mask
        & ~np.isnan(r034_aligned["r034_srv"].values)
        & (v4_oof_y_srv >= 0)
    )
    print(f"   Joint valid rows: {valid.sum()} / {len(valid)}")
    y_match = (r034_aligned["y_srv"].values[valid] == v4_oof_y_srv[valid]).mean()
    print(f"   y_srv match rate: {y_match:.4f}")

    # α-sweep on per-rally OOF AUC
    print(f"\n Step 4: α-sweep (blend = α * v4 + (1-α) * R-034)")
    sweep = []
    y_srv_valid = v4_oof_y_srv[valid]
    r034_valid = r034_aligned["r034_srv"].values[valid]
    v4_valid = v4_oof_srv[valid]
    for alpha in np.linspace(0.0, 1.0, 21):
        blend = alpha * v4_valid + (1 - alpha) * r034_valid
        auc = float(roc_auc_score(y_srv_valid, blend))
        sweep.append({"alpha": float(alpha), "auc": auc})
    best = max(sweep, key=lambda s: s["auc"])
    base_auc = sweep[0]["auc"]   # R-034 only
    v4_auc = sweep[-1]["auc"]    # v4 only
    print(f"   α=0 (R-034 only) AUC: {base_auc:.4f}")
    print(f"   α=1 (v4 only)    AUC: {v4_auc:.4f}")
    print(f"   Best α={best['alpha']:.2f}  AUC={best['auc']:.4f}")
    print(f"   Lift vs R-034 baseline: {best['auc'] - base_auc:+.4f}")

    # Compare to R-067cr (v3 best α=0.30, AUC was 0.7680 with v3)
    print("\n   For reference: R-067cr (v3) was α=0.30 with per-rally AUC=0.7680")

    # Build test submission
    print(f"\n Step 5: build R-075 test submission with α={best['alpha']:.2f}")
    v4_test_srv = np.load(os.path.join(OOF_DIR, f"{V4_TAG}_test_srv.npy"))
    v4_test_uid = np.load(os.path.join(OOF_DIR, f"{V4_TAG}_test_rally_uid.npy"))
    r042 = pd.read_csv(os.path.join(SUBMISSION_DIR,
                                      "submission_R042_R034_rule_override.csv"))
    assert np.array_equal(r042["rally_uid"].values, v4_test_uid), "test UID mismatch"

    alpha_best = best["alpha"]
    blend_test = alpha_best * v4_test_srv + (1 - alpha_best) * r042["serverGetPoint"].values
    r075 = r042.copy()
    r075["serverGetPoint"] = blend_test

    # Sanity assertions: actionId / pointId should be identical to R-042 base
    assert (r075["actionId"].values == r042["actionId"].values).all(), \
        "R-075 must not change actionId — only serverGetPoint"
    assert (r075["pointId"].values == r042["pointId"].values).all(), \
        "R-075 must not change pointId — only serverGetPoint"

    fname = f"submission_R075_R067cr_v4blend_alpha{int(alpha_best*100):03d}_PLUS_RULE.csv"
    out_path = os.path.join(SUBMISSION_DIR, fname)
    r075.to_csv(out_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"   Wrote: {out_path}")

    # Manifest
    manifest = {
        "rid": "R-075",
        "ts": "2026-05-25",
        "v4_source_tag": V4_TAG,
        "best_alpha": alpha_best,
        "best_oof_auc": best["auc"],
        "baseline_oof_auc": base_auc,
        "oof_auc_lift_vs_r034_baseline": best["auc"] - base_auc,
        "v4_only_oof_auc": v4_auc,
        "alpha_sweep": sweep,
        "output_csv": out_path,
        "valid_rallies_for_alpha_sweep": int(valid.sum()),
    }
    manifest_path = os.path.join(SUBMISSION_DIR, "r075_server_blend_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
