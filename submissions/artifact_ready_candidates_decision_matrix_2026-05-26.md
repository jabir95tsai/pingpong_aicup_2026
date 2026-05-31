# ARTIFACT_READY Candidates Decision Matrix (2026-05-26)

**Current LB-best**: R-067cr 0.3870095. Target 0.4000. Gap +0.013.
**Daily LB upload cap**: 3 (per AGENTS.md).

Three ARTIFACT_READY candidates await your manual LB decision. Goal Function v0.4
required reports below. Pick by risk/reward profile, slot availability, and
information value.

---

## Decision matrix

| Candidate | act diffs | pt diffs | sgp diffs | total cells | predicted LB Δ | downside ceiling | info value if LB-wins | info value if LB-fails |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **R-094 v2** SoftF1 action-only | 38 | **0** | 0 | **38** | +0.0003 to +0.0008 | ~−0.003 | confirms size-6 cap relax + SoftF1 additive transfers | size-6 + SoftF1 additive route closed |
| R-094 v1 SoftF1 shared-α | 38 | 218 | 0 | 256 | +0.0005 to +0.0010 | ~−0.005 | same + confirms point predictions tolerate small SoftF1 mixing | mixed-task ambiguity (was it action or point that hurt?) |
| R-081 v2 GBM corrector | 50 | 50 | 0 | 100 | +0.0003 | ~−0.003 | bounded GBM correction works (mechanism distinct from R-054r) | R-081 family confirmed unpromising |

Note: R-094 v1's point diffs (218) come from shared α=0.05 affecting point
predictions even though Fold-1 OOF showed F1_p basically flat. v2 specifically
eliminates this risk by setting α_point=0.00.

---

## v0.4 Candidate reports

### R-094 v2 (recommended first probe)

| Field | Value |
|---|---|
| theoretical_generalization_reason | R-031 SoftF1 (`v11_mulminet_aug_oldtest_softf1_phaseB`) targets macro-F1 directly. Adding it as 6th component AT ACTION-ONLY α=0.05 is structurally novel: "size-6 cap relax" is a documented LESSONS lever never LB-tested. Point F1 didn't benefit in OOF sweep → kept at α=0.00 (no point change). |
| why_transfers_to_test_new | SoftF1 = training-objective change, not feature change. Same data distribution behavior. v2 conservatism: only action changed, point + SGP exactly preserved from LB-best. |
| smoke_sanity_pass | TRUE (Fold-1 OOF: F1_a +0.0006 across ALL 15 action classes; zero canary drops; rule_override Layer A re-applied successfully) |
| lb_probe_worthy | TRUE |
| lb_confirm_hypothesis | LB ΔOV ≥ +0.0003 ⇒ size-6 cap relax + SoftF1 additive transfers; opens cap-relax as new strategic lever |
| lb_reject_hypothesis | LB ΔOV ≤ -0.003 ⇒ additive B-impure at low weight also fails; cap relax closed |

**File**: `submission_R094v2_R067cr_PLUS_SOFTF1_act_only_alpha005_PLUS_RULE.csv`

### R-094 v1 (alternative — shared α, more aggressive)

| Field | Value |
|---|---|
| theoretical_generalization_reason | Same as v2 — SoftF1 additive at α=0.05, but applied to BOTH action and point. Point F1 flat in OOF but argmax-shifts occur on near-boundary rows (hence 218 point diffs). |
| why_transfers_to_test_new | Same as v2; slightly more LB risk because point predictions are also changing on 218 rows even though OOF didn't show point improvement. |
| smoke_sanity_pass | TRUE (zero canary drops; ALL 15 action classes positive/neutral; point F1 within noise) |
| lb_probe_worthy | TRUE but slightly less safe than v2 |
| lb_confirm_hypothesis | Same as v2; if v1 wins and v2 also wins, the point changes are tolerable. |
| lb_reject_hypothesis | If v1 loses and v2 wins, the 218 point diffs were the problem (informative). |

**File**: `submission_R094_R067cr_PLUS_SOFTF1_alpha005_PLUS_RULE.csv`

### R-081 v2 (bounded corrector — different mechanism family)

| Field | Value |
|---|---|
| theoretical_generalization_reason | GBM corrector predicts (a) is R-067cr's argmax wrong (p_wrong AUC 0.69/0.65 = signal exists) AND (b) which alternative class is right (multiclass GBM). Override only when both predictors agree AND alt confidence ≥ 0.35. Cap 50 per task. Mechanism closer to R-042 rule_override (proven 1.0 LB) than R-054r meta_stack (LB -0.0103). |
| why_transfers_to_test_new | Features (entropy, margin, agreement, SN bucket) are model-output-derived → distribution-invariant. Bounded override count = bounded LB damage. Distinct from pure meta-stacking. |
| smoke_sanity_pass | TRUE (Fold-1 ΔF1 act +0.0003 pt +0.0003 — small but positive; AUC signal extracts; no catastrophe) |
| lb_probe_worthy | TRUE |
| lb_confirm_hypothesis | LB ΔOV ≥ +0.001 ⇒ bounded conditional correction transfers; mechanism distinct from meta-stacking |
| lb_reject_hypothesis | LB ΔOV ≤ -0.002 ⇒ even bounded correction overfits OOF; GBM-corrector route closed |

**File**: `submission_R081v2_R067cr_PLUS_CORRECTOR.csv`

---

## Suggested upload sequence

If you have 3 slots available today:

**Conservative sequence** (maximize info / minimize risk):
1. **R-094 v2** first — smallest diff, cleanest signal, lowest downside. If LB +Δ ≥ +0.0005 → confirms cap relax lever works.
2. Then **R-094 v1** — if v2 won, v1 tells you whether point changes hurt or help.
3. **R-081 v2** last — different mechanism family; informative either way.

**Aggressive sequence** (maximize upside chance):
1. **R-094 v1** first — biggest expected lift (+0.0005 to +0.0010).
2. **R-094 v2** if v1 hurt — diagnose whether point or action caused damage.
3. **R-081 v2** — bounded correction probe.

**Single-slot** (just one upload today):
- **R-094 v2** — best risk-adjusted EV; 38 cells changed, all on action where OOF showed clean gain across all 15 classes.

---

## Open expectations

| Outcome | Implication for goal |
|---|---|
| Any of 3 produces LB ≥ +0.0005 | Cap relax / GBM correction works → can iterate (R-094 v3 with higher α, R-081 v4 with more sophisticated corrector) |
| All 3 produce LB ≤ 0 | All non-architectural mechanisms exhausted at R-067cr; goal achievable only via R-082 Phase 2 (V11 embeddings) or a NEW STRATEGIC mechanism |
| Mixed (some +, some −) | Per-mechanism analysis: which families transfer for our component set |

---

## What's NOT in this list

- R-082 Phase 4 candidate (would be 4th ARTIFACT_READY) — pending R-082 Phase 2 kernel completion (~9-27h)
- R-080 (probability stack) — explicitly gated by user policy
- R-077 (v14 + focal loss) — NORMAL priority, blocked while R-082 STRATEGIC in flight
