# STRATEGY
## Round: 2026-05-05 — zoo_v2 LB win + v16_avg3 LB regression

---

## Current State

| Item | Value |
|---|---|
| **Current best submission** | `submission_zoo_v6_elig1_none_v11_aug_v11plus_v13_v14s2_v16.csv` |
| **Current best Public LB** | **0.3748577** (2026-05-06) |
| **Current best OOF (matched)** | 0.3794 (NONE calibration; reported eligible top-1 in zoo_v6) |
| **Current best P11 holdout** | 0.3873 |
| **Components** | `v11_aug + v11plus + v13 + v14_seed2 + v16_testhist_aug` (5-model global blend, **NONE calibration**) |
| **Prior best (zoo_v2 top-1)** | LB 0.3733788 — was previous current best (THR edge) |
| **Prior-prior best (zoo_v16_fast_01)** | LB 0.3694863 — fallback |
| **Backup (single-family best)** | `submission_v16_testhist_aug_v11_optblend.csv` LB 0.3673269 |
| **Old V14+V11 stable** | LB 0.3598509 — deep fallback |
| Daily LB submission limit | 3/day (2026-05-05 used: 2/3) |
| OOF→LB gap (current best, zoo_v2 5-model+v16) | −0.0095 |
| OOF→LB gap (zoo_v3 6-model + **v16_avg3**) | **−0.0164** (REGRESSION; LB 0.3675453, see What Failed) |
| OOF→LB gap (zoo_v16_fast_01 4-model) | −0.0105 |
| OOF→LB gap (V16+V11) | −0.0070 (OOF underestimated LB) |
| OOF→LB gap (V14+V11, clean) | −0.0155 (canonical baseline) |
| OOF→LB gap (per-SN bucket zoo #4) | −0.0197 (overfit, REJECTED) |

Bottlenecks (ranked by leverage):
- **pointId** F1 ≈ 0.235 — largest headroom; class 0 (off-grid) and short classes (1/2/3) carry most of the loss. Largely unchanged by zoo iteration; structural attack (P3 hierarchical point head) is the next lever.
- **serverGetPoint** AUC ≈ 0.61 — rally-level label is structurally under-modelled (per-shot prediction is noisy).
- **SN=2** slice OV ≈ 0.28 — early-rally; partly addressed by V16 test-history aug, do not chase further with hard SN gating.
- **zoo search OOF→LB transfer fragility** — newly observed; large blends (n≥6) and grid-edge calibration both inflate OOF without LB benefit.

---

## Locked Rules (do not violate)

1. **NEVER** use test.csv `serverGetPoint` as feature, target, or supervision. All training scripts overwrite it with −1; `build_test_history_pairs.py` discards real values before saving.
2. **NEVER** include SGP-derived player win-rate features.
3. **NEVER** include raw player-profile features (`player_action_freq`, `opp_action_freq`, per-player ID stats). V15 family is permanently rejected — non-transfer confirmed across two LB tests (gaps −0.022 and −0.026).
4. **NEVER** include `hist_action_freq` / `hist_point_freq` / `streak_*` (V15 hist+streak group). Inert vs V14, no LB signal.
5. **NEVER** use **hard per-SN-bucket gating** in blend weight search. zoo_v16_fast_04 proved this overfits OOF and loses −0.0098 on LB vs the global zoo.
6. Validation: `GroupKFold(n_splits=5)` by **match** for any new training script. Test-history augmented rows never enter validation.
7. Submission gate: a candidate may be submitted only if it has a credible path to beating the current best **0.3733788** — not merely beating an internal OOF threshold. Use multi-signal judgment (OOF + per-SN slice + OOF→LB gap of *structurally similar* prior submission), not OOF alone. The gap is NOT a fixed quantity — it grows with blend size and aggressive calibration (see Rules 8–9).
8. **Blend-size cap**: `blend_zoo_v2.py` and successors must search subsets of size **≤ 5** unless a size-6 candidate has been LB-validated. zoo_v3 top-1 (n=6, OOF 0.3839) lost −0.0058 on LB vs zoo_v2 top-1 (n=5, OOF 0.3829) — one extra component near-doubled the OOF→LB gap.
9. **Calibration grid edge guard** (REVISED 2026-05-06): THR-EDGE candidates STILL transferred to LB (zoo_v2 won), but **NONE-calibration candidates transfer BETTER** (zoo_v6 elig1 NONE, LB 0.3749 > zoo_v2 0.3734). Going forward: prefer NONE / TEMP / CW over THR-EDGE when both are available, especially if the NONE candidate has higher P11 holdout OV.
10. **`v16_avg3` is provisionally suspect** as a zoo component until a controlled (size-≤5, temperature-interior) v16_avg3 blend beats its single-seed v16 counterpart on LB. Per-fold OV variance reduction (v16_avg3 OOF 0.3597 base ≈ +0.0014 vs v16) does NOT imply LB transfer — zoo_v3 top-1 with v16_avg3 lost −0.0058 LB.
11. **NONE-calibration is LB-validated** (RESULTS §19/§20) BUT ONLY when paired with ≥ 2 Transformer-family components. Holdout-LB gap is calibration-arm-AND-subset-dependent: THR has LB > holdout (+0.007); NONE varies wildly (−0.012 with 2 transformers, −0.034 with 1 transformer). **P11 holdout magnitude is not a reliable LB-delta predictor for NONE candidates** — use directional ranking only, and only when subsets are structurally similar.
12. **v11_aug is STRUCTURALLY CRITICAL** for NONE blends (RESULTS §20). Removing v11_aug from a winning NONE blend (zoo_v6 elig1 → zoo_holdout_top1) lost 0.020 LB. v11_aug is no longer "optional" — it is a required component for any NONE-calibration submission with v11plus.
13. **NONE blends require ≥ 2 Transformer-family components** (any 2 of {v11, v11plus, v11_aug}). Single-Transformer NONE blends (e.g., v11plus alone + GBM mix) lose ≥ 0.018 LB. THR blends may differ — this rule is for NONE specifically.

---

## What Worked

| Direction | Evidence | Status |
|---|---|---|
| **V16 test-history augmentation** | LB +0.0075 vs V14+V11; OOF→LB gap shrank to −0.007 | Backbone — single-seed transfers reliably |
| **Global 5-model zoo blend (zoo_v2)** | LB 0.3733788 (+0.00389 vs zoo_v16_fast_01); per-task independent weights, THR calibration | **Current best recipe**; cap at size ≤ 5 |
| **V11 + V11plus together in zoo** | zoo_v2 top-1 includes both transformer variants | Use both as candidate slots in zoo menu |
| **v13 (legacy) as diversity component** | All zoo_v2 top-5 included v13; current best uses it with weight 0.346 in action | Keep in component menu |
| **Multi-seed V14 averaging (v14_avg3)** | Component-quality artifact; v14_seed0 won the zoo selection (not avg3) | Available, but v14_seed0 is the LB-validated representative |

## What Failed (do not retry without new evidence)

| Direction | Evidence | Status |
|---|---|---|
| **V15 player profile (any form)** | Two LB tests, OOF→LB gap −0.022 / −0.026 | PERMANENTLY REJECTED |
| **V15 hist freq + streak** | OOF flat (−0.0013); LB −0.0024 vs V14 | PERMANENTLY REJECTED |
| **Hard per-SN-bucket blend weights (zoo_v16_fast_04)** | OOF 0.37936 (≈ best), LB 0.3596738 (−0.0098 vs zoo #1) | REJECTED — non-transfer |
| **6-model zoo blend with v16_avg3 (zoo_v3 top-1)** | OOF 0.3839 (+0.0010 vs zoo_v2 top-1), LB **0.3675453** (−0.0058 LB), gap −0.0164 (vs zoo_v2 −0.0095) | REJECTED — confounds blend-size growth and v16_avg3 swap; both factors implicated |
| **THR temperature at grid edge (t=0.5)** | Both zoo_v2 and zoo_v3 top-1 hit t_a=t_p=0.5; zoo_v3 transferred badly | Treat as suspect (Rule 9); widen grid or pick interior-temperature alternate |
| **CatBoost in final blend** | OOF +0.006, LB −0.001 | Excluded from all final candidates |
| **V11+ class-weight + larger transformer** | OOF flat to negative | CLOSED |
| **Plain hierarchical point head (V12 era, hard decode)** | F1_p 0.158 vs flat 0.210 | Use only as soft-decoded variant in P3 |
| **Flat SN-bucket per-target weighting** | Inflates OOF, degrades LB | Avoid |

---

## Round Objective (2026-05-05 → next 1–2 days)

**Primary:** Find a candidate that beats the current best **LB 0.3733788** without burning a submission slot on a low-confidence file. Same-menu zoo iteration is hitting diminishing returns (zoo_v3 with v16_avg3 regressed). Need either a structurally new component (P3) or a diagnostic that isolates the v16_avg3 transfer issue.

**Secondary:** Diagnose the v16_avg3 vs blend-size confound. Either (a) a controlled re-run of `blend_zoo_v2.py` with size cap = 5 AND temperature grid extended to t≥0.3 (cheap, no training), or (b) a single LB probe of a 5-model v16_avg3 blend with an interior temperature.

**Tertiary:** Add a structurally distinct component to the zoo menu. Hierarchical point head (P3, soft-decoded) is the highest-upside / lowest-risk option — orthogonal to V16 family, attacks the largest task bottleneck.

---

## High-ROI Hypotheses (this round)

### H1 (P1, COMPLETED) — Blend Zoo v2

**Outcome (2026-05-05):** Top-1 (5-model: `v11+v11plus+v13+v14_seed0+v16_testhist_aug`, THR) OOF 0.3829 → **LB 0.3733788** (+0.0039 vs zoo_v16_fast_01). Current best.

Lessons retained for successors:
- 5 models was the sweet spot; size-6 candidates (zoo_v3) regressed on LB.
- THR calibration won (no TEMP/CW variant cleared 0.378 OOF gate).
- v13 (legacy) and v11plus (transformer variant) both contributed — keep in menu.
- Random Dirichlet n=300 underperformed exhaustive grid alpha for n=4 case (ref subset OOF 0.3785 vs historical 0.37998); upside larger blend spaces compensate, but small n=3-4 subsets may benefit from a finer search.

### H2 (P2, COMPLETED, REGRESSED) — V16 multi-seed ensemble

**Outcome (2026-05-05):**
- `v16_seed1` opt OV 0.3667 (vs v16 0.3677, −0.0010); `v16_seed2` opt OV 0.3674 (−0.0003). Per-seed variance very small (~0.001).
- `v16_avg3` averaged base OV 0.3597 (+0.0014 vs single-seed v16 base 0.3583). Solo OV gate barely passed.
- zoo_v3 with v16_avg3 swap: top-1 (n=6) OOF 0.3839 → **LB 0.3675453** (−0.0058 vs zoo_v2 top-1, gap −0.0164).

Conclusion: V16 is **seed-insensitive** (per-seed OV variance ≪ 0.005). The averaging gain on OOF (+0.0014) is mostly noise; the LB regression is large. v16_avg3 transfer is suspect (Locked Rule #10) and the size-6 search overfit OOF (Locked Rule #8). H2 is closed for direct submission; v16_avg3 may still serve as a *component* in size-≤5 blends if a controlled probe confirms transfer.

### H3 (P3, COMPLETED, FAILED) — Hierarchical point head (soft-decoded)

**Outcome (2026-05-05, see RESULTS.md §12):** v18 full 5-fold ran with on-grid SUBSET
training (Codex-approved) and soft product reconstruction. Both Codex gates failed:
cls0 F1 −0.0172 vs V14 baseline (gate ≥ −0.01); short F1 (cls 1/2/3 mean) −0.0392 vs
V14 (gate ≥ +0.03). Solo OOF opt OV 0.3595 vs V14 solo 0.3661 (−0.0066). v18 PARKED.
Do NOT blend `v18_*.npy` into any zoo. Codex's deferred fallback `P(side|depth)` is the
only structural rescue path; not scheduled for this round.

Diagnosis: product-of-marginals (`p_valid × p_depth × p_side`) is too restrictive vs the
flat 10-class joint head — depth and side are not independent given on-grid.

### H4 (P4) — Rally-level Server head

Core idea: SGP is rally-constant. Build a separate model that pools per-shot features into a single rally embedding (mean+max pool, plus rally-level meta features like rally length, score diff at end of rally history, last-shot action) and predicts SGP once per rally; broadcast to per-shot rows.

Why it could beat current best: AUC=0.61 is suspiciously low for a label that does not vary within a rally. A direct +0.04 AUC gain = +0.008 score, plus this is a structurally orthogonal source of OOF signal for the zoo blend.

Cost: low–medium; can be implemented as a small post-hoc module reading existing OOF features (no need to re-train action/point bases).

Risk: the per-shot model may already implicitly use most of the rally context. Quick OOF check before committing engineering.

Success signal:
- OOF rally-AUC ≥ 0.65 in 1 fold.
- When swapped into the zoo blend's server channel, OOF OV improves.

Failure signal: rally-AUC ≤ 0.62 → existing per-shot pipeline already captures the signal. Park.

### H5 (P5, deferred) — Autoregressive multi-task sequence model

Core idea: Causal Transformer that predicts (action, point, sgp) at every position in the rally; pretrain LM-style on union of train+test rallies (using observable history only, no test SGP); fine-tune supervised on real train positions. Multi-task heads with rally-pooled SGP head.

Why it might beat current best: 5–10× more supervised positions per rally; pretraining on test rallies generalises the V16 trick across all positions; structurally distinct from V11 so should add blend diversity.

Cost: high engineering; full run ≈ 8–10 h. Therefore start with a 1-fold smoke (≤90 min, hidden=256, 6 layers).

Risk: redundant with V11. Smoke must show ≥ V14 solo (≈ 0.36) AND non-trivial blend diversity vs V11 (Pearson correlation of OOF probs < 0.95).

Success signal: smoke solo OOF ≥ V14 solo and OOF probs decorrelate from V11; commit to full run.

Failure signal: smoke OOF in low 0.34s, or OOF probs ≈ V11 → abort.

---

## H6–H12 Breakthrough hypotheses (from 2026-05-05 deep research memo + Codex review)

### Component diversity ceiling — root-cause finding (NEW 2026-05-05)

Pairwise Pearson correlation of OOF point predictions (full matrix in RESULTS.md §12):
- v16_avg3 ↔ v16_testhist_aug = **0.994** (averaging across V16 seeds added near-zero diversity)
- v14_seed0/1/2 pairwise = 0.977; v14_avg3 ↔ each seed = 0.992
- v14 ↔ v12_5f = 0.95
- Cross-cluster GBM ↔ Transformer = 0.65–0.78
- v11 ↔ v11plus = 0.83

**Implication.** The zoo's "9-component" menu is effectively **2 clusters** (GBM and Transformer)
with strong intra-cluster correlation. Dirichlet random search exploits OOF noise differences
between near-identical components — explains the OOF→LB gap variability (zoo_v2 −0.0095 vs
zoo_v3 −0.0164). **Future LB lift requires adding a genuinely uncorrelated component** (target
cross-cluster correlation ≤ 0.78), NOT more same-architecture seeds.

### H6 (NEW) — V11 + test-history augmentation (HIGHEST priority next)

Plumb `data/test_history_pairs.parquet` (2,353 pairs) into V11's training data so the
Transformer learns the same test-distribution shifts that V16 caught on the GBM side.

Why: V16 → V14 LB delta was +0.0075 from this exact mechanism on GBM. V11 is the most
uncorrelated component (0.65–0.78 cross-cluster). Combining the two should produce real
diversity gain without LB-transfer risk.

**Codex sign-off (2026-05-05): 1-fold smoke APPROVED, with HARD implementation constraint.**
`data/test_history_pairs.parquet` carries `serverGetPoint = -1` placeholders. Current
`src/train_v11_transformer.py` server head computes BCE over ALL samples; feeding aug rows
in unchanged would treat −1 as a label and poison the SGP head. Required fix: **mask the
server loss for aug rows** (implementation choice: zero sample weight on aug rows for the
server head, OR compute server BCE only over `is_aug == 0` rows). Action and point losses
on aug rows are fine.

Smoke gate: 1-fold solo action F1 not regressed vs V11 baseline; no NaN/Inf; verified
server BCE excludes aug rows.
Full gate: solo action F1 ≥ V11 + 0.005 AND OOF correlation v11_aug ↔ v16 ≤ 0.78 (no worse
than current v11 cross-cluster correlation).

### H7 (NEW) — GBM/zoo soft-label distillation into V11

Train V11 with auxiliary KL loss using zoo_v2 top-1 OOF probabilities as soft labels.
Compresses ensemble knowledge into the most-uncorrelated component.

Risk: collapses to teacher mimicry → loses V11 orthogonal signal. Mitigate by capping α ≤ 0.5
and gating on cross-correlation gate (distilled v11 ↔ zoo_v2 OOF ≤ 0.95).

Schedule: after H6 outcome, reuses same training infrastructure.

### H8 (NEW) — Pseudo-labelled test rallies (POLICY-GATED, NOT submission-approved yet)

Use zoo_v2 top-1's high-confidence test predictions (action prob > 0.6, point prob > 0.4)
as pseudo-labels; append to V14/V16 training set. ~1,000 rows expected after gating.

**Codex sign-off (2026-05-05): NOT approved for submission training.** Distinct from V16
test-history augmentation — those rows are organiser-confirmed observable shots
(history_visible). Pseudo-labels are model-generated test targets and walk close to the
"no manual correction of test outputs" rule. Offline label generation and design exploration
acceptable. Any pseudo-label-trained submission requires explicit Jabir policy approval
BEFORE training begins. Treat as a parked direction until that approval lands.

### H9 (NEW) — Geometry-aware pointId loss (label-smoothing on 3×3 grid)

Distance-weighted soft labels: each true on-grid label distributes ε mass to its 4 spatial
neighbours. Direct attack on FH_short / mid_short / BH_short failures.

**Codex sign-off (2026-05-05): difficulty under-estimated.** GBM multiclass cannot trivially
absorb soft labels — implementation requires either (a) sample expansion (one shot becomes
multiple weighted (X, neighbour-class) rows) or (b) a custom loss / objective hook. Plan
cost rises from "low" to "medium". Safer than v18 hier because the joint head is preserved.

### H10 (NEW) — Rally-pooled SGP head (was P4 in TRAIN_PLAN.md)

SGP is rally-constant; predict once per rally and broadcast. Pool features across observable
shots (mean/max/last) plus rally-level meta. Targets the AUC=0.61 ceiling.

Cost: low. Smoke: 1-fold ≤ 30 min, gate rally-AUC ≥ 0.65. Orthogonal to point/action so does
not interfere with H6/H7.

### H11 (NEW) — Player-disjoint holdout (validation diagnostic)

Build a held-out fold whose primary players don't appear in training. Compute "player-disjoint
OOF" alongside standard match-OOF. Use the gap to predict LB transfer.

**Codex sign-off (2026-05-05): APPROVED to prioritise BUT initially advisory only.** With
≤ 5 LB-tested points (V12+V11, V14+V11, V16+V11, zoo_v16_fast_01, zoo_v2, zoo_v3), Pearson
> 0.85 is dominated by single points and submissions are not independent. Initial gate:
leave-one-out / rank-consistency check — does the holdout correctly predict why zoo_v2 won
and zoo_v3 / per-SN bucket / V15 lost? Hard gate only after that lands.

### H12 (NEW) — Anchor-perturbation zoo search

Restrict the search to weight perturbations from zoo_v2 top-1 (the LB-tested winner)
instead of fresh Dirichlet. Search space drastically narrower; OOF overfit risk drops.

Cost: ~30 min impl + ~50 min CPU. Smoke: top-1 OOF ≥ zoo_v2 + 0.001 AND drift < 0.2 from
anchor. Submission only after H6 / H10 produce a new component to perturb the menu with.

### Other ideas considered (deferred)

- Causal Transformer with LM pretraining on train+test (was P5; high cost ~10h, not in
  remaining 8h budget).
- Soft mixture-of-experts on strikeNumber (close to STRATEGY rule 5 boundary; needs Codex
  sign-off on the soft-vs-hard distinction).
- Contrastive rally embeddings without raw player IDs (large engineering, deferred).
- Flip-TTA at inference: difficulty under-estimated by Codex — must rebuild flipped raw
  context features and flip action/point posteriors back, not just relabel submission
  outputs. Plan cost rises from "trivial" to "low-medium".
- Rule-based posterior projection (action grammar): cheap, tried via apply_action_rules
  already; minor expected lift.

---

## Priority Order This Round (revised 2026-05-05 post-deep-memo + Codex review)

| Priority | Hypothesis | Status | Risk | Cost | Ceiling |
|---|---|---|---|---|---|
| **P0** | Hold current best 0.3733788; protect submission slots | active | — | none | — |
| **P1** | Blend Zoo v2 (5-model + THR) | ✅ COMPLETED, LB 0.3733788 | — | done | — |
| **P2** | V16 multi-seed (`v16_avg3`) → re-run zoo blend | ❌ COMPLETED, LB regressed | — | done | closed |
| **P1.5** | Diagnostic re-run with size cap = 5 + temp ≥ 0.3 (zoo_v4a) | ✅ COMPLETED; eligible top-1 OOF 0.3771 < gate; slot-3 SKIPPED | — | done | closed |
| **P3** | Hierarchical point head (soft-decoded), `train_v18_hier_point.py` | ❌ COMPLETED, gates failed (cls0 −0.0172, short −0.0392); v18 PARKED | — | done | closed |
| **P6** (NEW) | **V11 + test-history augmentation** (H6) | ✅ COMPLETED 2026-05-06; v11_aug solo OV 0.3247. zoo_v6 elig1 NONE with v11_aug → **LB 0.3748577 NEW CURRENT BEST** (RESULTS §19). Prior parking decision REVERSED — v11_aug helps when NONE-calibrated. | done | done | NEW BEST |
| **P10** (NEW) | Rally-pooled SGP head (H10, was P4) | ❌ COMPLETED 2026-05-06; AUC=0.998 leaked via n_shots parity (table-tennis alternation rule). PARKED — see RESULTS §15 | done | done | closed |
| **P11** (NEW) | Player-disjoint holdout (H11, validation diagnostic) | ✅ COMPLETED 2026-05-06; rank-consistency holds at zoo level (zoo_v2 > zoo_v3); flagged zoo_v6 candidates as likely regressions. Advisory signal. | done | done | retained for future submissions |
| **P12** (NEW) | Anchor-perturbation zoo search (H12) | ❌ COMPLETED 2026-05-06 (zoo_v7, zoo_v7b); anchor at local OOF optimum, perturbation cannot improve. PARKED — see RESULTS §16 | done | done | closed |
| **P7** (NEW) | GBM/zoo soft-label distillation into V11 (H7) | not started; pending H6 outcome | medium | ~2 h training | depends on H6 |
| **P9** (NEW) | Geometry-aware point loss (H9) | not started; Codex flagged difficulty under-estimated | medium (was low) | ~1–2 h impl + 80 min training | +0.005 point F1 if smoothing works |
| **P8** (PARKED) | Pseudo-labelled test rallies (H8) | NOT approved for submission; offline design only | high (rule risk) | varies | requires explicit Jabir policy approval before any submission training |
| **P5** | Autoregressive sequence model (smoke first) | deferred (cost > remaining budget) | high | 1.5 h smoke / 8–10 h full | step change to 0.38+ if it works |

---

## Submission Hypothesis (next slot, revised 2026-05-05)

The next submission slot must clear the bar `LB > 0.3733788`. Eligible candidates, ranked:

1. **H6 v11_aug zoo top-1** — after `v11_aug` (V11 + test-history aug, server-mask) passes
   smoke + full gates, swap into zoo as the v11 slot and re-run `blend_zoo_v2.py
   --max-models 5`. Expected first viable submission for slot 1 of 2026-05-06. Required:
   v11_aug solo gates pass, OOF correlation v11_aug ↔ v16 ≤ 0.78, zoo top-1 OOF ≥ zoo_v2
   top-1 OOF, edge-rejection passes.
2. **H10 rally-SGP-augmented zoo top-1** — after rally-SGP head passes its 1-fold smoke
   (rally-AUC ≥ 0.65) and full 5-fold (rally-AUC ≥ 0.62), swap into the SGP channel of the
   zoo blend; OOF must improve.
3. **H12 anchor-perturbation zoo top-1** — only viable AFTER H6 or H10 produces a new
   component to perturb the zoo menu with. Cheap top-up, not a standalone breakthrough.

**Today's slot 3 (2026-05-05): SKIP** (P1.5 Run A failed gate; v18 failed gates;
no eligible candidate). Preserve for 2026-05-06 if any 2026-05-05 work lands new
artifacts.

**Codex submission discipline (2026-05-05)**: P1.5 Run B is OOF-only diagnostic; its
output **does NOT** rehabilitate v16_avg3 for direct submission. v16_avg3 may only
re-enter the candidate pool after a separately LB-tested controlled probe.

---

## Forbidden / Deferred

- Hard per-SN-bucket weight conditioning (zoo_v16_fast_04 failure).
- All V15 / player-profile / hist-freq / streak features.
- CatBoost in final blends.
- Plain V11+ class-weight escalation.
- Hard-decoded hierarchical point heads.
- Any feature derived from test.csv `serverGetPoint`.
- **Blends of size ≥ 6** without LB validation (Locked Rule 8).
- **THR candidates with edge-grid temperature** (t = lower bound) without re-running with a wider grid (Locked Rule 9).
- **`v16_avg3` as a zoo component in the next submission** unless a controlled probe (size ≤ 5, interior temperature) confirms LB transfer (Locked Rule 10).
- Submitting the next-slot candidate before Codex reviews this STRATEGY.md and TRAIN_PLAN.md.

---

## Open Questions for Codex

1. **OOF→LB gap modeling for size-N blends**: zoo_v2 (n=5) had gap −0.0095; zoo_v3 (n=6, with v16_avg3) had gap −0.0164. Is the gap growth attributable to (a) blend-size alone, (b) v16_avg3 alone, or (c) both compounding? A controlled P1.5 re-run (n≤5, with v16_avg3 in menu) should disambiguate. What additional features should the spread-penalised score include to catch this — e.g., a size penalty `−0.001 × (n_models − 4)`?
2. **Temperature grid bound**: both LB-tested top-1s hit t=0.5 (lower edge). Should the grid be widened (down to 0.3 or 0.2) AND should we add a "no edge" filter that rejects candidates where the chosen t lies on the grid boundary?
3. **`--seed` plumbing audit**: I patched `train_v16_testhist_aug.py` (9 model-init sites + `np.random.seed(seed)`); flip-aug is deterministic, GroupKFold is deterministic. Did I miss a path? Was the v16_avg3 LB regression caused by a subtle seed-handling bug on my part, or by genuine v16_avg3 transfer failure?
4. **Hierarchical point reconstruction (P3)**: code skeleton in TRAIN_PLAN.md §P3 uses on-grid SUBSET for depth/side heads (per the prior Codex revision). Confirm sign-off before I write `train_v18_hier_point.py`.

---

## Carried Anchors

- V14 5-fold no-CB OOF: action 0.3793 / point 0.2162 / AUC 0.6101 / **OV 0.3602 base, 0.3754 +V11**
- V14 solo opt OV (post threshold-opt, before V11): 0.3661
- V16 testhist aug solo opt OV: 0.3677; +V11 OOF 0.3743; **LB 0.3673269**
- zoo_v16_fast_01 OOF 0.37998; **LB 0.3694863** (gap −0.0105)
- **zoo_v2 top-1** (n=5: v11+v11plus+v13+v14_seed0+v16) OOF 0.3829, F1_a 0.4145, F1_p 0.2362, AUC 0.6132, spread 0.0924, t_a=t_p=0.5; **LB 0.3733788** (gap −0.0095) — CURRENT BEST
- **zoo_v3 top-1** (n=6: v11+v11plus+v12_5f+v13+v14_seed0+v16_avg3) OOF 0.3839, F1_a 0.4172, F1_p 0.2349, AUC 0.6150, spread 0.0913, t_a=t_p=0.5; **LB 0.3675453** (gap −0.0164) — REGRESSION
- v16_seed1 (seed 48879) opt OV 0.3667; v16_seed2 (51966) opt OV 0.3674; v16_avg3 averaged base OV 0.3597
- V14_avg3 solo OOF 0.3623; +V11 OOF 0.3765
- SN=2 slice (V14+V11): n=14995, F1_a=0.243, F1_p=0.161, AUC=0.539, OV=0.270
- SN buckets (zoo_v2 top-1): SN=2 0.279 / SN=3-4 0.357 / SN=5-8 0.371 / SN=9-12 0.353 / SN≥13 0.341
- **zoo_v4a (P1.5 Run A, 2026-05-05, size≤5, temp≥0.3)**: 198/396 entries eligible (50% rejected as edge). Global top-5 by spread_penalised_score are ALL THR with t_a=t_p=0.3 (hit the new lower edge — THR fundamentally wants sharper temperatures). Eligible top-1 = NONE calibration, OOF 0.3771 (well below zoo_v2 top-1 0.3829). Slot-3 gate FAILED — submission skipped; current best preserved at LB 0.3733788. Implication: lowering the temp grid alone cannot rehabilitate edge candidates without LB validation, since the optimum sits past the lower bound.
