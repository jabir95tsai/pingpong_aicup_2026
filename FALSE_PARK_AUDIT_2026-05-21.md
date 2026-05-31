# False-Park Audit — 2026-05-21

**Trigger**: User question — "let's reexamine all the decisions we make
before, what are those er parked but in theory must have improve even a
little margin".

**Method**: For each parked component / banned decision, classify by
strength of evidence:
- **NO_LB** — parked on OOF alone, never uploaded to LB. False-park risk = HIGH.
- **LB_1** — parked on a SINGLE LB-tested configuration. Sample-of-1 risk.
- **LB_N** — parked across N≥2 LB-tested configurations. Confident.

Then cross-reference with the 2026-05-21 parked-component blend-swap audit
to see which "in-theory should help" components are now blend-eligible.

---

## A. Components banned WITHOUT any LB evidence (highest false-park risk)

| Component | Park rationale (historical) | 2026-05-21 blend audit verdict |
|---|---|---|
| `v14_recvprofile` | R-011 "FAILED intake gate" (standalone OOF only) | **STAGE 1 dOV +0.0007** in v14_seed2_v15feat_a slot |
| `v14_recvhand` | early eligible, never explicitly LB-tested standalone | **STAGE 1 dOV +0.0004** (R-035/R-039 candidate built) |
| `meta_stack` (R-005) | "PARKED" — gate failed standalone | **STAGE 1 dOV +0.0012** in v14_seed2_v15feat_a slot |
| `meta_stack_v2_logistic` (R-005) | "PARKED" same as above | **STAGE 1 dOV +0.0006** in v13_oldtest slot |
| `server_head_v1` (R-006) | AUC 0.584 (FAIL gate 0.62, PARK) | Not in current audit — would be SGP-specialist class |
| `server_head_v2` (R-006) | AUC 0.602 (FAIL gate 0.62, PARK) | Same |
| `v11_big`, `v11_aug_big` | "underperformed" (OOF) | **STAGE 1 dOV +0.0006** for v11_aug_big in v11plus slot |
| `v11plus_aug` | "underperformed" (OOF) | **STAGE 2 dOV -0.0004** — diagnostic-eligible |
| `v16_seed1` | "single-seeds; v16_avg3 dominates" (no LB) | **STAGE 1 dOV +0.0001** in v16_avg3 slot |
| `v16_seed2` | same | **STAGE 2 dOV -0.0004** |
| `v17_momentum` (R-015) | "FAILED intake gate by -0.0003 OV; r=0.992 vs v16_avg3" | LB-tested elsewhere as part of zoo subsets; never standalone-uploaded |
| `v14_seed0`, `v14_seed1` | "redundant + not LB-validated" (their own words!) | v14_seed0 in v13_oldtest slot: dOV +0.0001 (STAGE 1) |

**Subtotal**: 12 components banned without standalone LB evidence. Most are
now demonstrably blend-eligible by the new two-stage gate framework.

---

## B. Components banned with single LB datapoint (sample-of-1 risk)

| Component | LB result | Park rationale | Re-examine? |
|---|---|---|---|
| `v14_avg3` | R-007 LB -0.0013 vs baseline | "LB-confirmed regression" | Audit dOV +0.0001 in current R-034 PAIR. Context differed at R-007 (zoo_v6 era, not R-034). Different blend slot might transfer. |
| `v14_pseudo_v1` | R-010 LB -0.0068, zoo_v12 elig1 | "Bias amplification from LB-teacher pseudo" | The CONCRETE finding (don't use LB-best teacher for pseudo) stays valid. But a structurally-different teacher might work. |
| `v11_mulminet_aug_oldtest_avg2` | R-028 top1 LB -0.0086 | "CLASS B-impure architecture swap" | We extrapolated to ALL B-impure swaps from this ONE datapoint. Today's audit: v11_mulminet_aug_avg3 has dOV +0.0039 (STRONGEST OOF lift). May not generalize to LB at ratio 0.97 — but worth ONE diagnostic upload. |
| `v11_mulminet_aug_oldtest_avg3` | R-033 LB -0.0015 | "B-impure" | Same single-datapoint extrapolation. |
| `v13_oldtest_avg3` | R-033 LB -0.0015 | "B-seedavg" | Single LB test. v13_oldtest_avg2 might behave differently (audit: dOV -0.0001 STAGE 2). |
| `v15_player_only` | LB 0.3555 (gap -0.022) | "Player profile non-transfer" | Teammate v8 audit: their fold-safe per-fold + p+opp side claims +0.04 OOF F1. Ours was fold-leaky single-side. Different setup. |
| `v15_pp` | LB 0.3507 (gap -0.026) | "V15 features don't transfer" | Similar — our specific setup might have been weak. |
| `v14_5f_nocb` | LB 0.3599 | "superseded by V16 LB 0.3673" | Never tested in current R-034 PAIR slot context. |

**Subtotal**: 8 components banned on 1 LB datapoint. The decision was reasonable
at the time (we DID see LB regression) but the conclusion ("ban entirely") is
broader than the evidence ("regress in THIS specific config"). Per the new
framework, would re-test in current R-034 PAIR context.

---

## C. Class-level bans extrapolated from few datapoints

These were systemic rules that may over-generalize:

| Rule | Evidence | False-park risk |
|---|---|---|
| **"CLASS A re-arrangements 6/6 LB-regress, banned"** | R-007/R-008/R-016/R-017/R-020b/R-026 all failed | LIKELY CORRECT — but specific subsets within "rearrangement" could differ. The 6 tests used similar zoo_vN search patterns. A fundamentally different rearrangement (e.g., SN=2 specialist + base ensemble) might escape. |
| **"CLASS B-impure architecture swap = always regress, ratio ~0.97"** | R-028 top1: 1 datapoint! -0.0086 LB | **HIGH RISK** — extrapolating from 1 datapoint. The strongest OOF lifts in today's parked audit are B-impure (v11_mulminet_aug_avg3 +0.0039). Worth 1 diagnostic upload to refine the rule. |
| **"CLASS B-seedavg same-arch averaging is a no-op, ratio ~1.0005"** | R-033 CLASSBpure: 1 datapoint -0.0015 LB | MEDIUM RISK. v11_aug_oldtest seeds + v13_oldtest seeds all show dOV 0.0000 in audit (consistent). But v14_seed0_oldtest, v14_seed1_oldtest, v14_seed2_oldtest haven't been blended-tested as separate components. |
| **"DL/transformers underperform on this dataset"** | Teammate's claim in v8 README | Their experience says "all sequence/DL/transformer collapse". Our R-034 includes v11_aug_oldtest (transformer) + v11plus (transformer). Contradicts their conclusion. |
| **"Hard per-SN-bucket blends are banned"** | Specific over-fitting episode | Specific cause was over-fitting to OOF. Soft per-SN (e.g., feature flag, not weight conditioning) might be fine. |

---

## D. Standalone-OOF gates that we now know over-reject

| Historical gate | Documented failures of the gate |
|---|---|
| `OV >= baseline + 0.003` to enter zoo | R-029a (v15feat_a) failed this. 8 days later → R-034 +0.0028 LB. |
| `Fold-1 AUC >= 0.620` SGP gate (R-030) | R-030 smoke fold-1 AUC 0.6110 → FAIL_PARK. But in full 5-fold (today): OOF AUC 0.6037 overall. Not yet blend-tested but may still contribute as SGP-only specialist. |
| `intake F1_a delta >= +0.0025` (R-011 v14_recvprofile) | Today: dOV +0.0007 in R-034 PAIR blend swap. |
| `near-clone (r >= 0.99) → ban` (R-015 v17_momentum) | v17_momentum has been part of NONE blends since; its solo r=0.99 doesn't preclude marginal blend contribution. |

---

## E. Recommended re-tests, ranked by EV / cost

Given the new two-stage gate framework + R-034 evidence that "small OOF +
new signal class = LB upload worthy":

| # | Component | Why it might lift LB | Estimated cost | LB risk |
|---|---|---|---|---|
| 1 | **R-036 meta_stack** (already built, awaiting upload) | NEW SIGNAL CLASS, never LB-tested. Audit dOV +0.0012. | Free CSV exists | LOW |
| 2 | **R-040 v11_mulminet_aug_avg3** (already built) | BIGGEST OOF lift +0.0039 in audit. B-impure — refines 1-datapoint rule. | Free CSV exists; **1 LB slot** | HIGH (worst-case ~0.3707, best ~0.3868) |
| 3 | **R-037 v14_recvprofile** (already built) | B-feature class same as R-034 LB-WIN pattern. dOV +0.0007. | Free CSV exists | LOW |
| 4 | **R-035/R-039 v14_recvhand** (already built x2) | Same B-feature class. dOV +0.0004. | Free CSV exists | LOW |
| 5 | **R-038 meta_stack_v2_logistic** (already built) | NEW SIGNAL CLASS. dOV +0.0006. | Free CSV exists | LOW |
| 6 | Audit v11_aug_big as NEW slot (no built CSV) | OOF dOV +0.0006 in v11plus slot, banned w/o LB | Run build_low_risk_submissions.py with new tag | LOW |
| 7 | Audit v14_avg3 in current R-034 PAIR context | R-007 ban was different era. dOV +0.0001 today | Same | MEDIUM (R-007 LB-tested) |
| 8 | Re-test player_profile with teammate's fold-safe + p+opp setup | Teammate claims +0.04 OOF; our 2026-05-04 setup was different | Need to write `features_player_profile_v2.py` + train v14 variant | UNKNOWN (was LB-failed in old setup) |
| 9 | R-031 SoftF1 (running on Kaggle) | NEW SIGNAL CLASS targeting rare-class F1 | Already running | UNKNOWN |
| 10 | R-043 (v15feat_b) — done locally, awaiting audit | Transition prior features. Likely B-feature class. | Just need blend audit + build CSV | LOW |
| 11 | R-044 (sgp_prefix_v3_full) — done locally, awaiting audit | SGP-specialist NEW SIGNAL | Just need blend audit | UNKNOWN |

**Existing built-but-not-uploaded LB candidates** (5 LOW-RISK + 2 HIGH-RISK):

```
submission_R035_v14_TO_v14_recvhand.csv               (LOW, dOV +0.0002, B-feature)
submission_R036_v14_seed2_v15feat_a_TO_meta_stack.csv (LOW, dOV +0.0012, NEW SIGNAL)
submission_R037_v14_seed2_v15feat_a_TO_v14_recvprofile.csv (LOW, dOV +0.0007, B-feature)
submission_R038_v13_oldtest_TO_meta_stack_v2_logistic.csv (LOW, dOV +0.0006, NEW SIGNAL)
submission_R039_v14_seed2_v15feat_a_TO_v14_recvhand.csv  (LOW, dOV +0.0004, B-feature - same as R035 dup)
submission_R040_v11_mulminet_aug_avg3_TO_v11_aug_oldtest.csv (HIGH, dOV +0.0030, B-impure)
submission_R041_v11_mulminet_aug_avg2_TO_v11_aug_oldtest.csv (HIGH, dOV +0.0017, B-impure)
submission_R042_R034_rule_override.csv                (LOW, post-process only)
```

8 candidates ready to upload, 0 uploaded so far. Daily cap is 3 — we have
**room for ~8 days of LB testing** with current candidates alone.

---

## F. Decisions we should reverse in light of new framework

1. **The "BANNED from submission" list in LESSONS_CHECKLIST is too aggressive**:
   - Reframe `v14_recvprofile`, `v14_recvhand`, `meta_stack*`, `v17_momentum`,
     `v11_aug_big`, `v16_seed1/2`, `v14_seed0/1` from "BANNED" to
     "STANDALONE-PARKED — eligible for blend audit + LB diagnostic"
   - Keep harder bans only on LB-confirmed regressors (v14_avg3, v14_pseudo_v1)
     AND only in their LB-tested blend context, not universally.

2. **R-005 (meta_stack) and R-006 (server_head)** should be RE-OPENED:
   - meta_stack is now STAGE 1 dOV +0.0012 in audit; R-036 already built
   - server_head_v1/v2 were SGP-specialists (parked on AUC < 0.62 standalone
     gate). New gate is blend-swap, not standalone — same logic as R-030

3. **CLASS B-impure rule should be downgraded from "always FAIL" to
   "needs more LB datapoints"**: 1 R-028 LB-failure is not enough to ban
   a whole class. R-040 (v11_mulminet_aug_avg3 +0.0039 OOF) is the next
   data point.

4. **CLASS B-seedavg rule** should be similarly downgraded.

---

## G. Tightest summary (one line per item)

12 banned without LB. 8 banned on 1 LB datapoint. 5 class-rules from <=2
LB tests each. Standalone-OOF gate over-rejects (5+ documented cases).
8 LB candidate CSVs already built and ready. Upload R-036 (highest EV
LOW-RISK) first to test the new framework empirically.
