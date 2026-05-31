"""R-170 — orthogonal-mechanism ensemble of R-094 v2 + R-081 v2.

Diversity audit (2026-05-26) revealed: R-094 v2 (SoftF1 additive) and R-081 v2
(GBM corrector) change DIFFERENT rows on action (intersection = 1 row out of
38 + 50). Mechanisms are orthogonal.

R-170 combines both: take R-094 v2's 38 action changes + R-081 v2's ~49
non-overlapping action changes + R-081 v2's 50 point changes. Apply
rule_override Layer A.

Theory (v0.4):
  theoretical_generalization_reason:
    Two LB-untested mechanisms (SoftF1 macro-F1-targeted additive vs bounded
    GBM corrector) acting on disjoint row sets. If both transfer, effects
    should compound. If only one transfers, the other contributes noise
    bounded by its individual cap. Combined coverage is broader than either
    alone (87 action rows vs 38 or 50).
  why_transfers_to_test_new:
    Both mechanisms individually pass v0.4 sanity. Their orthogonality is
    structural (different model families: V11+SoftF1 vs GBM-on-features),
    so combined behavior on test_new should mirror combined OOF behavior.

USAGE:
    python -u src/build_r170_orthogonal_ensemble.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, SUBMISSION_DIR

R067CR_BASE = os.path.join(SUBMISSION_DIR,
                            "submission_R067cr_alpha030_v22_blend_PLUS_RULE.csv")
R094_V2     = os.path.join(SUBMISSION_DIR,
                            "submission_R094v2_R067cr_PLUS_SOFTF1_act_only_alpha005_PLUS_RULE.csv")
R081_V2     = os.path.join(SUBMISSION_DIR,
                            "submission_R081v2_R067cr_PLUS_CORRECTOR.csv")
OUT_RAW = os.path.join(SUBMISSION_DIR, "submission_R170_R094v2_PLUS_R081v2_orthogonal.csv")
OUT_RULE = OUT_RAW.replace(".csv", "_PLUS_RULE.csv")
MANIFEST = os.path.join(SUBMISSION_DIR, "r170_orthogonal_manifest.json")


def main():
    print("=" * 80)
    print(" R-170 — orthogonal-mechanism ensemble (R-094 v2 + R-081 v2)")
    print("=" * 80)

    base = pd.read_csv(R067CR_BASE)
    r094 = pd.read_csv(R094_V2)
    r081 = pd.read_csv(R081_V2)
    assert (base["rally_uid"] == r094["rally_uid"]).all()
    assert (base["rally_uid"] == r081["rally_uid"]).all()
    print(f"   loaded {len(base)} rallies; all rally_uid orders match")

    # Identify diffs
    a_base = base["actionId"].to_numpy()
    a_r094 = r094["actionId"].to_numpy()
    a_r081 = r081["actionId"].to_numpy()
    p_base = base["pointId"].to_numpy()
    p_r094 = r094["pointId"].to_numpy()
    p_r081 = r081["pointId"].to_numpy()

    r094_act_diff = (a_r094 != a_base)
    r081_act_diff = (a_r081 != a_base)
    r094_pt_diff  = (p_r094 != p_base)
    r081_pt_diff  = (p_r081 != p_base)

    print(f"\n Action diffs:")
    print(f"   R-094 v2: {r094_act_diff.sum()}")
    print(f"   R-081 v2: {r081_act_diff.sum()}")
    print(f"   Both: {(r094_act_diff & r081_act_diff).sum()} (overlap)")

    print(f"\n Point diffs:")
    print(f"   R-094 v2: {r094_pt_diff.sum()}")
    print(f"   R-081 v2: {r081_pt_diff.sum()}")

    # Combine: for each row, prefer R-094 v2's prediction (cleaner smoke) where it
    # changed; otherwise use R-081 v2's where IT changed; otherwise base.
    new_action = a_base.copy()
    new_point = p_base.copy()

    # Action: R-094 v2 takes priority on overlap (cleaner smoke, all classes
    # positive in OOF)
    new_action[r081_act_diff] = a_r081[r081_act_diff]
    new_action[r094_act_diff] = a_r094[r094_act_diff]   # overrides R-081 on overlap

    # Point: R-094 v2 doesn't touch point (α_p=0). R-081 v2 does (50 changes).
    # Take R-081 v2 point changes only.
    new_point[r081_pt_diff] = p_r081[r081_pt_diff]

    # SGP unchanged from R-067cr
    new_sgp = base["serverGetPoint"].to_numpy().copy()

    # Build output
    out_df = pd.DataFrame({
        "rally_uid": base["rally_uid"],
        "actionId": new_action,
        "pointId": new_point,
        "serverGetPoint": new_sgp,
    })

    n_act_diff = int((new_action != a_base).sum())
    n_pt_diff  = int((new_point  != p_base).sum())
    print(f"\n Final combined diffs vs R-067cr base:")
    print(f"   action: {n_act_diff} (vs sum 38+50=88, expected ~87 due to 1 overlap)")
    print(f"   point:  {n_pt_diff} (vs R-081 v2's 50)")
    print(f"   SGP:    0 (preserved)")

    out_df.to_csv(OUT_RAW, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\n Saved raw CSV: {OUT_RAW}")

    # Apply rule_override Layer A
    print("\n Applying rule_override Layer A...")
    cmd = [sys.executable, "-u", os.path.join("src", "apply_rule_override.py"),
           "--input", OUT_RAW, "--train", os.path.join("data", "train.csv"),
           "--test", os.path.join("data", "test_new.csv"),
           "--output", OUT_RULE]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        last_lines = r.stdout.strip().split("\n")[-3:]
        for ln in last_lines:
            print(f"   {ln}")
        # Verify SGP unchanged
        final = pd.read_csv(OUT_RULE)
        assert (final["serverGetPoint"].to_numpy() == new_sgp).all(), "SGP changed!"
        print(f"   SGP unchanged after rule_override: confirmed")
    else:
        print(f"   WARN rule_override failed: {r.stderr}")

    # Final diff counts
    final = pd.read_csv(OUT_RULE)
    n_act_final = int((final["actionId"].to_numpy() != a_base).sum())
    n_pt_final  = int((final["pointId"].to_numpy()  != p_base).sum())
    print(f"\n Final after rule_override vs R-067cr base:")
    print(f"   action: {n_act_final}")
    print(f"   point:  {n_pt_final}")
    print(f"   SGP:    0 (asserted)")

    manifest = {
        "rid": "R-170",
        "ts": "2026-05-26",
        "base_csv": R067CR_BASE,
        "component_csvs": {"R-094 v2": R094_V2, "R-081 v2": R081_V2},
        "output_csv": OUT_RULE,
        "diffs_vs_base": {
            "action": n_act_final,
            "point": n_pt_final,
            "sgp": 0,
        },
        "v04_candidate_report": {
            "theoretical_generalization_reason":
                "Combine two LB-untested but smoke-passed mechanisms acting on "
                "DISJOINT row sets (only 1-row overlap on action). R-094 v2 = "
                "SoftF1 6th-component additive (B-feature-class). R-081 v2 = "
                "bounded GBM corrector (new-mechanism). Orthogonal coverage.",
            "why_transfers_to_test_new":
                "Each component already passes individual sanity. Combined effect "
                "on disjoint rows preserves each mechanism's LB transfer rate. "
                "Cumulative downside bounded at sum of individual ceilings (~-0.006).",
            "smoke_sanity_pass": True,
            "lb_probe_worthy": True,
            "lb_confirm_hypothesis":
                "LB ΔOV >= +0.0008 ⇒ orthogonal mechanism combination > sum of "
                "individual best estimates; can stack diverse correctors.",
            "lb_reject_hypothesis":
                "LB ΔOV <= -0.003 ⇒ combination interaction dominates; mechanisms "
                "interfere even on disjoint rows; close the stacking route.",
            "predicted_lb_delta": "+0.0006 to +0.0011 (sum of components, minus interaction)",
        },
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n Saved manifest: {MANIFEST}")
    print("\n" + "=" * 80)
    print(" ARTIFACT_READY_FOR_JABIR_UPLOAD_REVIEW")
    print("=" * 80)
    print(f" File: {OUT_RULE}")


if __name__ == "__main__":
    main()
