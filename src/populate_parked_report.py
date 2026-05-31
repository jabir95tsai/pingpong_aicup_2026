"""Populate PARKED_AUDIT_REPORT.md with audit results.

Reads:
  - submissions/parked_audit_summary.csv  (best slot per component)
  - submissions/parked_audit_full_ranking.csv (all swap attempts)

Replaces sections 2-5 in PARKED_AUDIT_REPORT.md with concrete data.
"""
import os
import sys

import numpy as np
import pandas as pd

PROJ = "C:/Users/jabir/Hacker_J/pingpong_aicup_2026"
REPORT = f"{PROJ}/PARKED_AUDIT_REPORT.md"
FULL_CSV = f"{PROJ}/submissions/parked_audit_full_ranking.csv"
SUM_CSV = f"{PROJ}/submissions/parked_audit_summary.csv"

# Class assignment heuristic.
# Reference R-034 baseline slots:
#   v11_aug_oldtest: transformer V11 + aug + oldtest
#   v11plus       : transformer V11plus (different transformer)
#   v13_oldtest   : GBM V13 + oldtest
#   v14_seed2_v15feat_a: GBM V14 + v15feat_a hand-crafted features (R-034 win)
#   v16_avg3      : GBM V16 + testhist aug, averaged 3 seeds
def assign_class(cand: str, slot: str) -> str:
    # NEW SIGNAL CLASS: stacking ensembles
    if cand.startswith("meta_stack"):
        return "B-meta (NEW SIGNAL CLASS — stacking ensemble, never LB-tested)"
    # NEW SIGNAL CLASS: sn2 expert
    if cand == "sn2_expert":
        return "B-meta (SN=2 specialist, partial coverage)"
    # CLASS B-feature: same arch + same data, new features
    # - v15feat: hand-crafted aggregates added to v14
    # - recvhand/recvprofile: receiver-relative features added to v14
    if "_v15feat" in cand or cand.startswith("v14_recv"):
        return "B-feature (R-034 LB-WIN class)"
    # CLASS B-impure when swapping mulminet (different transformer arch) into v11/v11plus slots
    if slot in {"v11plus", "v11_aug_oldtest"} and "mulminet" in cand:
        return "B-impure (R-028 LB-failed at this pattern)"
    # CLASS B-impure when swapping mulminet (transformer) into v13_oldtest (GBM) slot
    if slot == "v13_oldtest" and "mulminet" in cand:
        return "B-impure (transformer→GBM cross-family)"
    # CLASS B-impure for v11_big into v11plus (different arch)
    if slot == "v11plus" and ("v11_big" in cand or "v11_aug_big" in cand):
        return "B-impure (bigger arch into transformer slot)"
    # CLASS B-impure for v12 into v14-family
    if slot.startswith("v14") and cand.startswith("v12"):
        return "B-impure (v12 GBM family swap)"
    # CLASS B-seedavg: within-family seed avg / single seed of same arch+data
    if slot.endswith("_oldtest") and cand.startswith(slot.replace("_oldtest", "")) and ("_avg" in cand or "_seed" in cand or "_oldtest_avg" in cand):
        return "B-seedavg (R-033 LB-failed)"
    if slot == "v13_oldtest" and cand.startswith("v13_oldtest"):
        return "B-seedavg (R-033 LB-failed)"
    if slot == "v11_aug_oldtest" and cand.startswith("v11_aug_oldtest"):
        return "B-seedavg (R-033 LB-failed)"
    if slot == "v16_avg3" and cand.startswith("v16"):
        return "B-seedavg (R-033 LB-failed)"
    # CLASS B-pure (ADD oldtest, same arch)
    if slot.endswith("_oldtest") and cand.endswith("_oldtest"):
        return "B-pure (R-027 PAIR-class)"
    if "oldtest" in cand and "oldtest" not in slot:
        return "B-pure (ADD oldtest)"
    # CLASS A: re-arrangement only
    return "A or other (R-007 LB-failed pattern; needs strong signal proof)"


def main() -> None:
    if not os.path.exists(SUM_CSV) or not os.path.exists(FULL_CSV):
        print(f"Missing input CSVs: run audit_all_parked_components.py first")
        sys.exit(1)

    sumdf = pd.read_csv(SUM_CSV)
    fulldf = pd.read_csv(FULL_CSV)

    sumdf = sumdf.sort_values("OV", ascending=False).reset_index(drop=True)
    sumdf["transfer_class"] = sumdf.apply(lambda r: assign_class(r["component"], r["best_slot"]), axis=1)

    # Build sections
    lines = []
    lines.append("## 2. Global ranking (top 30 swap attempts by OOF)")
    lines.append("")
    lines.append("| # | swap_label | OV | dOV vs R-034 | pred_LB (lo–hi) |")
    lines.append("|---|---|---:|---:|---:|")
    for i, r in enumerate(fulldf.head(30).itertuples(), 1):
        lines.append(f"| {i} | `{r.label}` | {r.OV:.4f} | {r.delta_OV:+.4f} | "
                     f"{r.pred_LB_lo:.4f}–{r.pred_LB_hi:.4f} |")
    lines.append("")

    lines.append("## 3. Best slot per parked component (all 60)")
    lines.append("")
    lines.append("| # | component | best_slot | OV | dOV | pred_LB (lo–hi) | class |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for i, r in enumerate(sumdf.itertuples(), 1):
        lines.append(f"| {i} | `{r.component}` | `{r.best_slot}` | {r.OV:.4f} | "
                     f"{r.delta_OV:+.4f} | {r.pred_LB_lo:.4f}–{r.pred_LB_hi:.4f} | {r.transfer_class} |")
    lines.append("")

    # Two-stage gate framework
    lines.append("## 4. Two-stage gate framework classification")
    lines.append("")
    s1 = sumdf[sumdf["delta_OV"] >= 0]
    s2 = sumdf[(sumdf["delta_OV"] >= -0.002) & (sumdf["delta_OV"] < 0)]
    s3 = sumdf[(sumdf["delta_OV"] >= -0.005) & (sumdf["delta_OV"] < -0.002)]
    park = sumdf[sumdf["delta_OV"] < -0.005]

    lines.append(f"### STAGE 1 — STRONG/TIED (dOV ≥ 0): {len(s1)} candidates")
    lines.append("*ELIGIBLE for direct LB upload (existing standalone fast-track).*")
    lines.append("")
    if len(s1):
        lines.append("| component | best_slot | dOV | pred_LB (lo–hi) | class |")
        lines.append("|---|---|---:|---:|---|")
        for r in s1.itertuples():
            lines.append(f"| `{r.component}` | `{r.best_slot}` | {r.delta_OV:+.4f} | "
                         f"{r.pred_LB_lo:.4f}–{r.pred_LB_hi:.4f} | {r.transfer_class} |")
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append(f"### STAGE 2 — NEAR-TIED (-0.002 ≤ dOV < 0): {len(s2)} candidates")
    lines.append("*ELIGIBLE for blend-swap diagnostic upload (NEW gate, post-R-034).*")
    lines.append("")
    if len(s2):
        lines.append("| component | best_slot | dOV | pred_LB (lo–hi) | class |")
        lines.append("|---|---|---:|---:|---|")
        for r in s2.itertuples():
            lines.append(f"| `{r.component}` | `{r.best_slot}` | {r.delta_OV:+.4f} | "
                         f"{r.pred_LB_lo:.4f}–{r.pred_LB_hi:.4f} | {r.transfer_class} |")
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append(f"### STAGE 3 — MARGINAL (-0.005 ≤ dOV < -0.002): {len(s3)} candidates")
    lines.append("*DIAGNOSTIC ONLY — hold unless new-signal-class evidence.*")
    lines.append("")
    if len(s3):
        lines.append("| component | best_slot | dOV | pred_LB (lo–hi) | class |")
        lines.append("|---|---|---:|---:|---|")
        for r in s3.head(15).itertuples():
            lines.append(f"| `{r.component}` | `{r.best_slot}` | {r.delta_OV:+.4f} | "
                         f"{r.pred_LB_lo:.4f}–{r.pred_LB_hi:.4f} | {r.transfer_class} |")
        if len(s3) > 15:
            lines.append(f"| ... | ... | ... | ... | _({len(s3) - 15} more)_ |")
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append(f"### PARKED (dOV < -0.005): {len(park)} candidates")
    lines.append("*No LB evidence either way; user may still override and upload to disprove the gate.*")
    lines.append("")
    if len(park):
        lines.append("Components in this tier (sorted by dOV, closest to threshold first):")
        for r in park.sort_values("delta_OV", ascending=False).head(20).itertuples():
            lines.append(f"  - `{r.component}` (best slot `{r.best_slot}`, dOV {r.delta_OV:+.4f})")
        if len(park) > 20:
            lines.append(f"  - ... + {len(park) - 20} more (see CSV)")
    lines.append("")

    # Section 5: class-based transfer risk
    lines.append("## 5. Class-based transfer risk for Stage 1+2 candidates")
    lines.append("")
    s12 = pd.concat([s1, s2])
    if len(s12):
        lines.append("Sorted by predicted LB (optimistic, then conservative). Use this to pick the most")
        lines.append("LB-likely upload candidate; the user makes the final call.")
        lines.append("")
        lines.append("| # | component | best_slot | dOV | conservative LB | optimistic LB | class | LB risk |")
        lines.append("|---|---|---|---:|---:|---:|---|---|")
        s12 = s12.copy()
        s12 = s12.sort_values(["pred_LB_hi", "pred_LB_lo"], ascending=[False, False])
        for i, r in enumerate(s12.itertuples(), 1):
            cls = r.transfer_class
            risk = "HIGH" if "impure" in cls else ("MED" if "seedavg" in cls else "LOW")
            lines.append(f"| {i} | `{r.component}` | `{r.best_slot}` | {r.delta_OV:+.4f} | "
                         f"{r.pred_LB_lo:.4f} | {r.pred_LB_hi:.4f} | {cls} | {risk} |")
    else:
        lines.append("_(none)_")
    lines.append("")

    # Section 6: actionable recommendation
    lines.append("## 6. Final list — components that have NEVER been LB-submitted")
    lines.append("")
    lines.append(f"**Total parked components: {len(sumdf)}**. None of these have been")
    lines.append("LB-uploaded. They are organized below by gate status. The user makes the")
    lines.append("final upload decision.")
    lines.append("")
    lines.append(f"- **STAGE 1 (dOV ≥ 0)**: {len(s1)} components — direct-upload eligible")
    lines.append(f"- **STAGE 2 (-0.002 ≤ dOV < 0)**: {len(s2)} components — blend-diagnostic eligible")
    lines.append(f"- **STAGE 3 (-0.005 ≤ dOV < -0.002)**: {len(s3)} components — diagnostic only")
    lines.append(f"- **Below threshold (dOV < -0.005)**: {len(park)} components — no clear blend benefit")
    lines.append("")
    lines.append("Per LESSONS 2026-05-21: standalone gates over-reject. The new blend-swap")
    lines.append("gate is the post-R-034 fix. Predicted-LB ranges use ratios derived from")
    lines.append("R-027 (1.0035, conservative) and R-034 (1.0151, optimistic).")
    lines.append("")
    lines.append("Class transfer hazards (from LESSONS, 2026-05-21):")
    lines.append("- CLASS B-impure (architecture change): R-028 LB-FAILED at ratio 0.9768.")
    lines.append("  HIGH LB risk even when OOF dOV is strongly positive.")
    lines.append("- CLASS B-seedavg (within-family seed avg only): R-033 LB-FAILED at ratio 1.0005.")
    lines.append("  MED LB risk.")
    lines.append("- CLASS B-pure (ADD oldtest, same arch): R-027 PAIR LB-WON at ratio 1.0035.")
    lines.append("  LOW LB risk.")
    lines.append("- CLASS B-feature (same arch + same data, new features): R-034 LB-WON at ratio 1.0121.")
    lines.append("  LOW LB risk.")
    lines.append("")

    # Now read existing report, replace from section 1 onwards
    with open(REPORT, "r", encoding="utf-8") as f:
        text = f.read()

    # Find the boundary at "## 2. Global ranking"
    head_anchor = "## 2. Global ranking"
    head_idx = text.find(head_anchor)
    if head_idx == -1:
        # Fallback: append
        new_text = text + "\n" + "\n".join(lines) + "\n"
    else:
        # Replace from section 2 to section 7 (appendix)
        appendix_anchor = "## Appendix:"
        app_idx = text.find(appendix_anchor)
        if app_idx == -1:
            new_text = text[:head_idx] + "\n".join(lines) + "\n"
        else:
            new_text = text[:head_idx] + "\n".join(lines) + "\n\n---\n\n" + text[app_idx:]

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(new_text)

    print(f"Populated {REPORT}")
    print(f"  STAGE 1: {len(s1)}")
    print(f"  STAGE 2: {len(s2)}")
    print(f"  STAGE 3: {len(s3)}")
    print(f"  PARKED:  {len(park)}")


if __name__ == "__main__":
    main()
