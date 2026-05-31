# ⛔ DEFERRED — single-candidate search CLOSED (2026-05-31)

The team has pivoted to a final AutoGluon ensemble of all members' components.
Per Jabir's direction, the autonomous single-candidate STRATEGIC search is
CLOSED. R-200–R-205 below are DEFERRED (not abandoned — preserved for a future
cycle if the AutoGluon ensemble does not clear 0.4000 and the team reopens
single-model work). Final-phase summary:
- LB-best: R-067cr = 0.3870095 (unbeaten).
- STRATEGIC results this cycle: R-094v2 LB-FAIL, R-170 LB-FAIL, R-082 NO-GO,
  R-203 soft-NO-GO, R-210 NO-GO, R-202 NO-GO, R-211 NO-GO.
- Gap diagnosis: lives in POINT (F1_p 0.229); FH/BH handedness axis is the
  residual hard bucket (recvhand/recvside only ~+0.0005).
- Path to 0.4000 now = team AutoGluon ensemble (see
  audits/AUTOGLUON_COMPONENT_MANIFEST_2026-05-31.md).

---

# STRATEGIC Candidate Backlog (post-R-082 horizon)

**Created**: 2026-05-26 during autonomous /goal mode while R-082 Phase 2 ran.
**Purpose**: pre-designed STRATEGIC menu so when R-082 lands (success or fail),
the next step is unambiguous. Each candidate has a v0.4 candidate report stub.

**All candidates here**:
- new-mechanism + novelty=high (qualify as STRATEGIC per v0.4)
- Need either Kaggle GPU OR substantial engineering (not local-CPU quick-wins)
- Each is genuinely structurally different from existing R-034 PAIR / R-067cr

---

## R-200 — V11 retrain with multi-task auxiliary heads (regularization)

| Field | Value |
|---|---|
| Class | new-mechanism (architectural augmentation) |
| Novelty | high |
| Predicted LB Δ | +0.005 to +0.012 |
| Compute cost | ~10 GPU-hr Kaggle |
| Theoretical reason | Add auxiliary heads: predict (a) rally length remaining, (b) who won the rally, (c) next-shot's `handId`/`strengthId`/`spinId`. Auxiliary signal forces V11's 192-d representation to encode richer rally dynamics. Standard MTL regularization technique (Caruana 1997). Expected: improved primary-task generalization, especially on long rallies (74% of hard rows per diversity audit). |
| Why transfers | Auxiliary supervision uses train.csv labels only; same training distribution. The improved representation helps both train+test_new equally. |
| Smoke criteria | Fold-1 OV ≥ V11 baseline + 0.005; no canary class drop > -0.025; push-family F1 mean ≥ 0.35 |
| LB confirm | +0.005 ⇒ MTL regularization transfers; opens MTL-heavy architectures path |
| LB reject | ≤-0.005 ⇒ aux heads overfit at expense of primary tasks |
| Leakage safety | All aux labels from train.csv only; no test SGP; no player IDs in aux |

## R-201 — Bidirectional encoder + causal decoder hybrid

| Field | Value |
|---|---|
| Class | new-mechanism |
| Novelty | high |
| Predicted LB Δ | +0.008 to +0.015 (highest upside, highest risk) |
| Compute cost | ~12-15 GPU-hr Kaggle |
| Theoretical reason | V11 = bidirectional encoder, full attention over context. R-066/R-071 = causal LM, multi-position prediction. Hybrid: bidirectional encoder for context understanding + causal decoder for last-shot prediction. Standard seq-to-seq pattern. Combines both architectures' strengths: V11's rich context + causal LM's exposure to all-position prediction. |
| Why transfers | Both base architectures transfer (V11 LB-irreplaceable, R-066 v3 AUC +0.066). Hybrid combines proven strengths without inventing new mechanisms. |
| Smoke criteria | Fold-1 OV ≥ R-066 v3 + 0.010 (target competitive with V11 family); AUC ≥ 0.65 |
| LB confirm | +0.008 ⇒ hybrid architecture works; biggest STRATEGIC win since R-027 PAIR |
| LB reject | ≤-0.003 ⇒ architectural complexity overfits; close hybrid route |
| Leakage safety | Same as V11/R-066 individually; no new feature sources |

## R-202 — Long-rally specialist + R-067cr ensemble

| Field | Value |
|---|---|
| Class | new-mechanism (data-filtered specialist + blend) |
| Novelty | medium-high |
| Predicted LB Δ | +0.005 to +0.008 |
| Compute cost | ~6 GPU-hr Kaggle |
| Theoretical reason | Diversity audit: 74% of hardest rows are SN≥5. Train a SPECIALIST V11 ONLY on SN≥3 training rows (longer rallies). Theory: model can focus capacity on long-rally patterns that are currently underfit. NOTE: this is NOT hard per-SN gating at inference (LB-toxic); the specialist trained on long-rally data is BLENDED with general v14/v11/v16 at inference. |
| Why transfers | Train-time data filtering on SN≥3 is legal (using train.csv subset). Test-time inference uses the model on ALL test rows (no SN gating). Blend with base R-067cr balances specialty vs generality. |
| Smoke criteria | Standalone OV on SN≥5 rows ≥ R-067cr SN≥5 + 0.010; standalone OV on SN≤2 not catastrophically worse |
| LB confirm | +0.005 ⇒ specialist-blend approach transfers; open data-filtered specialist track |
| LB reject | ≤-0.003 ⇒ specialist overfits its data subset; route closed |
| Leakage safety | SN filtering uses strikeNumber from train.csv (legal feature); no test SGP; no player IDs |

## R-203 — V14 GBM with custom focal+CB loss objective

| Field | Value |
|---|---|
| Class | B-feature (same arch, new training objective) |
| Novelty | medium |
| Predicted LB Δ | +0.003 to +0.008 |
| Compute cost | ~8h local CPU (3 estimators × focal impl) |
| Theoretical reason | V14 is LightGBM + XGBoost + CatBoost ensemble. Replace standard CE multiclass objective with custom focal CE (γ=2) + Cui et al. class-balanced weights for push family (act5/6/13) and Loop (act1). R-094 v2 showed SoftF1-trained model improves these classes; v14 is the LB-best blend's biggest GBM contributor. Direct attack on action-class imbalance. |
| Why transfers | Same train data, same arch, only loss function changes. B-feature class has 0.9 LB transfer empirically (R-034). |
| Smoke criteria | Fold-1 OV ≥ v14 baseline + 0.003; push-family F1 mean ≥ +0.005 vs baseline |
| LB confirm | +0.003 ⇒ B-feature focal-loss variant works; could swap into R-034 PAIR slot |
| LB reject | ≤-0.003 ⇒ focal loss hurts more than helps; close GBM-focal route |
| Leakage safety | Same as base v14; no leakage vectors introduced |

## R-204 — Cross-architecture distillation ensemble

| Field | Value |
|---|---|
| Class | new-mechanism (with explicit pseudo-label review per LESSONS) |
| Novelty | high |
| Predicted LB Δ | +0.005 to +0.012 |
| Compute cost | ~8-10 GPU-hr Kaggle |
| Theoretical reason | Combine R-071 v4 (causal LM with focal+CB; AUC 0.6994) + R-067cr (best blend) + V11 family soft labels → train a new STUDENT model that takes the BEST signal from each teacher. Critical: teacher = ENSEMBLE of decorrelated models (NOT a single LB-best teacher like R-010 failed pattern). LESSONS says "future pseudo-label experiment uses a STRUCTURALLY DIFFERENT teacher". This satisfies that. |
| Why transfers | Distilling from diverse teachers should produce a generalist student. Risk: still pseudo-label adjacent → could repeat R-010 failure if student over-fits to teacher predictions. Mitigation: regularize student with mixed real + soft labels. |
| Smoke criteria | Fold-1 OV ≥ best individual teacher; per-class F1 closer to teacher-best-per-class than teacher-average |
| LB confirm | +0.005 ⇒ ensemble distillation transfers; opens distillation as new mechanism class |
| LB reject | ≤-0.005 ⇒ pseudo-label monoculture failure recurs even with diverse teachers; close distillation route |
| Leakage safety | Need explicit Codex new-mechanism review (per goal toxic-class override) |

## R-205 — Cross-rally context model (within-match)

| Field | Value |
|---|---|
| Class | new-mechanism |
| Novelty | high (untested) |
| Predicted LB Δ | +0.004 to +0.010 |
| Compute cost | ~10 GPU-hr Kaggle |
| Theoretical reason | Current models predict shot t using only shots 1..t-1 within the current rally. But the SAME MATCH has prior rallies whose strategies inform current rally tactics. Use the LAST K rallies of the same match (player-disjoint? need to think) as additional context. Risk: depends on whether match-level context features are player-style adjacent (which is toxic). |
| Why transfers | Match context is inherent to the data (no leak). Test_new matches are different from train matches, but the WITHIN-MATCH inter-rally dynamics generalize. |
| Smoke criteria | Fold-1 OV ≥ V11 baseline + 0.004; specifically check no B-player-style failure mode (per-player aggregates excluded from features) |
| LB confirm | +0.004 ⇒ cross-rally signal transfers; opens match-level context as new mechanism class |
| LB reject | ≤-0.005 ⇒ match-context reproduces B-player-style toxicity; close cross-rally route |
| Leakage safety | **HIGH SCRUTINY** — match-context features could carry player-style signal indirectly. Codex review required. Hard rules: no per-player aggregates, no player IDs, only shot-content features from prior rallies. |

---

## Priority ranking for post-R-082 launch (if needed)

If R-082 Phase 3 PASSES (signal ≥ +0.005 on any task):
1. Phase 4 LB candidate (build & mark ARTIFACT_READY)
2. R-201 (hybrid architecture) — highest upside, next-best STRATEGIC

If R-082 Phase 3 FAILS (no task lift ≥ +0.005):
1. R-201 (hybrid) — different mechanism class than R-082's failure mode
2. R-200 (V11 + MTL aux heads) — milder retrain ask
3. R-202 (long-rally specialist) — targeted at audit-revealed weakness

If user wants LOCAL-only work (no further Kaggle):
1. R-203 (V14 + focal) — only local-CPU candidate in the backlog (~8h)
2. None others are local-only

## Refused / not-included

- Any per-player aggregate (R-062r/R-072 toxic)
- Pure replacement meta-stack (R-054r toxic)
- Hard per-SN gating
- Pseudo-consensus with LB-best teacher (R-010 toxic)
- V15 hist/streak/player-profile family
- LB upload sequencing (Jabir's decision domain)

---

## Stop condition impact

Per /goal directive, stop auto-clears when:
> "all STRATEGIC + HIGH candidates are exhausted (in_progress queue empty
> AND no new mechanism in design)"

This backlog (R-200 through R-205) explicitly keeps "new mechanism in design"
NON-empty for the duration the user wishes to pursue 0.4000. Stop only
auto-clears if:
- LB ≥ 0.4000 reached, OR
- User explicitly `/goal clear`, OR
- All 6 backlog items are PROVISIONAL_FAIL'd via LB

The backlog is intentionally rich enough to support pursuing the LB ≥ 0.4000
goal through multiple iterations.
