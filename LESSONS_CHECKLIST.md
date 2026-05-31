# Lessons Checklist

Pre-flight checklist for **T1+ actions**. Claude runs through this and writes a
3-line "checked X, Y, Z — green" before launching anything > 30 min wall, before
opening a `R-NNN` entry, and before submitting any LB candidate. New entries are
added when Codex flags the same class of bug a second time.

> **POLICY UPDATE 2026-05-22**: Codex `ARTIFACT_OK` pre-approval for LB
> upload is **REMOVED**. LB upload is Jabir's decision alone. Component
> design (T2: new trainer, feature module, schema change) still requires
> Codex review via `REVIEW_QUEUE.md`. Post-run artifact integrity is
> Jabir's check.
>
> Rationale: after R-034 (+0.0028 win after 8 days of false-park) and
> R-042 (same-day LB-tested +0.0028 win), holding built candidates for
> serial Codex review costs more in slot-burn opportunity than in
> regression risk. The blend-swap diagnostic + new-signal-class
> framework (2026-05-21 lesson) is the recommended pre-upload heuristic.

Each entry has the form:

> **Rule** — short statement.
>
> Why it's here: 1-line context, usually a past failure.
>
> How to verify: concrete check Claude can run in seconds.

---

## SGP / target leakage

> **`serverGetPoint` is never used as input or target proxy.** Clean train
> `serverGetPoint` may be used only as the supervised server-head label.
>
> Past failure: v19_rally_srv pulled SGP via `n_shots` parity → AUC=0.998 leak.
>
> Verify: grep training code for `serverGetPoint` / `srv_label` / `sgp` outside
> the **target** position; confirm the training loop's `y_srv` only flows to the
> server head loss, never to feature builders.

> **Old-test / teammate leak artifacts are quarantined.** Any file trained with
> `data_old/test.csv` SGP or any submission that copies old-test SGP is not a legal
> component, even if a `NOLEAK` version exists.
>
> Past failure: `AICUP_v1_LB0.4304.zip` reached LB 0.4304 by overwriting 1236
> overlapping `test_new` rallies with old-test `serverGetPoint` truth.
>
> Verify: compare any external artifact against `data_old/test.csv`; LEAK variants
> must not enter `oof_predictions/`, the zoo menu, or submission candidates.
>
> **2026-05-13 UPDATE — partial unblock.** AICUP organizers' announcement
> permits `data/test.csv` (the current OLD test) as additional training data.
> Models trained via `--include-old-test data/test.csv` (concatenated to
> `train.csv` BEFORE clean_data; full labels including real SGP) are LEGAL,
> NOT quarantined. Final submissions must still use `test_new.csv` as test.
> The pre-2026-05-13 quarantine rule still applies to two specific cases:
> (a) submissions that overwrite `test_new` rows with `data/test.csv` SGP
> ground truth (prediction-side leak), and (b) the LEAK variants in
> `AICUP_v1_LB0.4304.zip`. Training-side use of old test as additional
> training data is the legal path; `LEAK_SGP_*` overwriting submissions
> are NOT (unless the user explicitly authorizes it as a one-shot
> diagnostic — and even then, the result is not a "blend best" but a
> known-leak ceiling).
>
> Verify: tagging convention `*_oldtest` for any model trained with
> `--include-old-test`. The trainer logs `[include-old-test] Added 3589
> rows from data/test.csv (1236 rallies, 55 matches)` exactly once.

> **Test rallies' SGP is not visible.** Test-history augmentation may use visible
> `actionId` and `pointId` of test rallies, but `serverGetPoint` must be either
> absent or masked with a sentinel (e.g. `-1`) and excluded from BCE.
>
> Past failure: P6 v11+test-history aug; first revision did not mask SGP.
>
> Verify: in the trainer, confirm `aug_rows_in_server_loss == 0` is logged each
> epoch. The aug parquet's SGP column should be sentinel/-1.

> **Aug rows do not enter the server BCE loss.** Even when SGP is sentinel, BCE
> must be masked by `(is_aug == 0)`.
>
> Verify: trainer prints `aug_rows_in_server_loss=0` per epoch.

> **`n_shots` / rally length / terminal-shot parity are SGP-leaky.** The
> alternation rule deterministically encodes who hits the decider once you know
> rally parity. Any feature aggregating across rally end is high risk.
>
> Verify: feature builders aggregate only over **prefix** rows of the current
> shot, never the rally suffix or the rally total.

## Pseudo-labels and external data

> **Pseudo-label test targets require explicit Jabir approval.** Training a model
> that will be submitted on test-derived labels is T3 and must go through
> `REVIEW_QUEUE.md` with Codex sign-off **and** Jabir confirmation.
>
> Past failure: none yet, but a tempting failure mode.

> **External data sources require T3 review.** Same as pseudo-label.

## Calibration / submission gating

> **Edge rejection must affect file materialisation, not just annotation.** A
> THR/TEMP candidate at the temperature lower bound is suspect overfit; the
> blender's eligible-only top-K must skip it before writing the submission CSV.
>
> Past failure: zoo_v2 first version annotated edge candidates but still wrote
> their CSVs.
>
> Verify: in `blend_zoo_v2.py`, eligible top-K loop iterates only over rows
> whose `temp_at_edge == False`.

> **Zoo top rank does not equal submittable.** A blend at THR rank 1 can be
> edge-rejected (no submission CSV exists). Always confirm the materialised
> file path before considering a submission.
>
> Verify: `ls submissions/<expected_filename>.csv` succeeds; the CSV's
> `actionId`/`pointId` columns are non-empty integers.

> **NONE blends require ≥ 2 transformers** (Locked Rule 9). A NONE blend with
> only one of {v11, v11plus, v11_aug} is unsafe — confirmed −0.020 LB hit on
> 2026-05-06 single-variable test.
>
> Verify: count distinct elements in the blend subset that come from
> `{"v11","v11plus","v11_aug"}`; must be ≥ 2 for any NONE candidate proposed
> for LB upload.

> **NONE blends should NOT exceed 2 transformers, AND should retain v13.**
> A NONE blend with 3 transformers (v11+v11_aug+v11plus) AND no v13 lost
> −0.0043 LB on a single-variable test (zoo_v11 elig1, 2026-05-10) vs the
> known-good 2-transformer + v13 baseline (zoo_v8 elig3, LB 0.3688). The
> two factors are confounded; the safe interpretation is to (a) cap
> transformer count at 2 of {v11, v11plus, v11_aug} and (b) keep v13 in any
> NONE submission candidate until a future single-variable test isolates
> which factor matters.
>
> Verify: count(v11/v11plus/v11_aug) <= 2 AND v13 in subset for any NONE
> candidate proposed for LB upload.

> **Always specify SOLO vs BLEND when comparing LB scores.**
>
> Past failure: 2026-05-11, v17_momentum SOLO LB 0.3601 was reported
> as "−0.0093 vs LB best 0.3694", but 0.3694 is a 5-component blend.
> Solo-vs-blend comparison conflates ensemble-averaging benefit with
> feature design. v17_momentum's OOF→LB ratio (0.9833) was actually
> normal for V16-family solo submissions; the apparent regression
> was the typical solo-vs-blend gap.
>
> Verify: any LB delta claim must compare like-to-like — solo vs solo,
> 5-blend vs 5-blend. Single-model penalty (vs n-component blend) is
> typically 0.005-0.020 on LB, baked into ensemble averaging. If no
> directly-comparable solo LB exists for the baseline, say so
> explicitly rather than inventing a misleading delta.

> **Rule #12 LB-CONFIRMED (2026-05-11).** v11plus in NONE blend
> WITHOUT v11_aug → LB regression beyond typical blender-search noise.
>
> Past failure: 2026-05-11, R-017 elig1 (v11+v11plus+v13+v14_recvhand+v16_avg3)
> uploaded → LB 0.3615 (ratio 0.958). LB-best (v11_aug+v11plus, ratio
> 0.981) and R-016 (v11+v11_aug only, ratio 0.970) prove that the rule
> violation costs ~0.012 OOF→LB ratio = ~−0.005 LB on top of the
> typical blender-search regression.
>
> Verify before LB upload: any NONE 5-blend including v11plus MUST
> also include v11_aug. The original holdout-derived rule #12 is now
> empirically LB-validated.

> **PUBLIC-LB vs PRIVATE-LB framework (added 2026-05-12, Jabir
> strategic insight).** Components with strong PUBLIC-LB performance
> may overfit the public test set; components with high OOF→LB ratio
> (≥ 1.0) signal STRONG GENERALIZATION and may win on PRIVATE LB even
> if they regress on PUBLIC LB substitution tests.
>
> Empirical evidence (2026-05-12):
> - v11_mulminet_aug solo: OOF 0.3299 → LB 0.3518, **ratio 1.066**
>   (LB > OOF; strongest generalization signal we have)
> - LB-best zoo_v10 elig2: OOF 0.3712 → LB 0.3694, ratio 0.995
>   (mild OOF overfit, typical for tuned blends)
> - R-020b SAFE blend (v11_mulminet_aug into LB-best): OOF 0.3738 →
>   LB 0.3645, ratio 0.975 (averaging diluted the generalization
>   advantage)
>
> **Strategic implication**:
> - Public LB rule: keep current LB-best subset rigid; substitutions
>   systematically lose public LB.
> - Private LB rule: PRESERVE high-ratio components in the model zoo
>   for FINAL SUBMISSION SELECTION even if their public-LB
>   substitution test fails. They are insurance against public LB
>   shake-up.
>
> Components currently in the "private LB candidate" pool:
> - v11_mulminet_aug (ratio 1.066, strongest)
> - LB-best zoo_v10 elig2 (ratio 0.995, public-LB-safe)
> - Future: R-020c, R-019 + MuLMINet, Path B causal LM (if R-014 succeeds)
>
> Verify before final-submission selection: include AT LEAST ONE
> high-ratio component in the diverse final set, not just the
> public-LB-best.

> **Blender-search OOF gains do NOT transfer to PUBLIC LB — 6 confirmed
> instances 2026-05-10/11/12/13. New components, new techniques, AND
> seed-averaging ALL failed to escape this pattern.**
>
> Update 2026-05-13: R-026 SAFE used seed-averaged v11_mulminet_aug_avg2
> (which boosted STANDALONE OV by +0.013, a real gain). Blend OOF +0.0013
> over single-seed; blend LB regressed −0.0017 worse. **Seed averaging
> didn't escape the pattern.** OOF→LB ratio degraded 0.975 → 0.967.
>
> The LB-best subset (zoo_v10 elig2) is empirically a hard local
> optimum on PUBLIC LB. Substitution paths exhausted. Future LB-lift
> attempts must use ENTIRELY DIFFERENT mechanisms:
> 1. Add components without substituting (size-6 cap relax)
> 2. Use a different calibration arm (TEMP/CW instead of NONE)
> 3. Solo upload a strong-generalization component (private LB hedge)
> 4. NEW model class outside V11/V14/V16 family entirely
>
> **Critical update 2026-05-12**: R-020b SAFE candidate added the
> structurally NEW v11_mulminet_aug component to the LB-best subset
> (single swap: v11plus → v11_mulminet_aug). OOF +0.0026, **actual LB
> −0.0049**. The earlier reasoning that "new components should transfer
> differently than re-arrangements" is WRONG. Even new components
> following solid solo LB results (v11_mulminet_aug solo ratio 1.066)
> regress when substituted into LB-best.
>
> **v11plus is empirically irreplaceable in the LB-best subset.**
> Whatever v11plus contributes to LB private set is NOT captured by
> any other component including v11_mulminet_aug.
>
> Past failures (5 instances):
> - R-007 v14_avg3 substitution: LB −0.0013
> - R-008 drop-v13 + 3-transformer: LB −0.0043
> - R-016 v11+v11_aug+v16_testhist swap: OOF +0.0019, LB −0.0022
> - R-017 elig1 (rule#12 violation): LB −0.0079
> - R-020b SAFE (NEW component swap): OOF +0.0026, LB −0.0049
>
> **Hard lesson**: any modification to the LB-best subset
> (zoo_v10 elig2: v11_aug+v11plus+v13+v14_seed2+v16_avg3 NONE)
> systematically loses LB. The path forward is NOT subset modification
> with current components. New paths must:
> (a) Add components as INDEPENDENT solo submissions (v11_mulminet_aug
>     solo at LB 0.3518 was a fine standalone; the failure is purely
>     in blend substitution), OR
> (b) Add a 6th component WITH a way to validate the size-6 cap relax
>     (Codex review required), OR
> (c) Try a different calibration arm (not NONE) that has different
>     transfer characteristics.
>
> When exhaustive subset search over already-trained components finds an
> OOF-better arrangement, that arrangement systematically REGRESSES on LB.
>
> Past failures:
> - R-007 v14_avg3 substitution: LB −0.0013
> - R-008 drop-v13 + 3-transformer: LB −0.0043
> - R-016 v11+v11_aug+v16_testhist swap: OOF +0.0019, LB **−0.0022**
>
> The current LB-best subset (zoo_v10 elig2: v11_aug+v11plus+v13+
> v14_seed2+v16_avg3 NONE) is a **local optimum** that exhaustive
> blender search cannot improve via component rearrangement. The
> blender overfits subtle OOF interaction effects that don't exist on
> the LB private set.
>
> Verify before LB upload: any subset that swaps components vs the
> current LB-best MUST be justified by either (a) a STRUCTURALLY NEW
> component in the swap (not a near-clone like v17_momentum), (b) a
> NEW calibration arm (not NONE), or (c) Codex review on why this
> swap differs structurally from the 3 failed cases above. Pure
> rearrangement is BANNED as an LB-probe path.

> **Component status — REFRAMED 2026-05-21 (post-R-034 lesson + FALSE_PARK_AUDIT).**
>
> Previously this section listed components as "BANNED from submission".
> That framing was too strict — R-034's +0.0028 LB win came from a
> "BANNED" / standalone-OOF-parked component (v15feat_a). The full audit
> (see `FALSE_PARK_AUDIT_2026-05-21.md`) found 12 components banned
> WITHOUT any LB evidence, several of which are now demonstrably
> blend-eligible.
>
> New tiers:
>
> **HARD-BANNED — LB-confirmed regression AND mechanism understood:**
> - `v14_pseudo_v1` (R-010 LB regression −0.0068; LB-best teacher
>   monoculture causes bias amplification. Banned UNIVERSALLY because
>   the failure mechanism is mechanistic, not blend-context-specific.
>   A future pseudo-label experiment with a STRUCTURALLY DIFFERENT
>   teacher is fine — that's a different component, not this one.)
>
> **CONTEXT-PARKED — LB-failed in 1 specific blend context:**
> - `v14_avg3` (R-007 LB −0.0013 in zoo_v6 era; never tested in current
>   R-034 PAIR slot. Audit dOV +0.0001 in R-034 PAIR. Worth diagnostic
>   LB upload as R-046 if higher-EV candidates exhaust.)
> - `v11_mulminet_aug_oldtest_avg2` (R-028 LB −0.0086 in v11plus slot;
>   in v11_aug_oldtest slot today's audit shows dOV +0.0027. Different
>   context. R-041 candidate built.)
> - `v11_mulminet_aug_oldtest_avg3` (R-033 LB −0.0015. Same caveat as above.)
> - `v13_oldtest_avg3` (R-033 LB −0.0015 in v13_oldtest slot.)
> - `v15_player_only` (LB 0.3555. Teammate v8 uses fold-safe per-fold
>   profile + p+opp side; ours was likely fold-leaky single-side.
>   Re-test with teammate's exact setup is a NEW component.)
> - `v15_pp` (LB 0.3507. Same caveat as v15_player_only.)
> - `v14_5f_nocb` (LB 0.3599, superseded by V16. Never tested in current
>   R-034 PAIR context.)
>
> **STANDALONE-PARKED — failed standalone OOF gate, never LB-tested,
> ELIGIBLE for blend audit + diagnostic LB upload:**
> - `v14_recvprofile` (R-011 standalone FAIL; audit dOV +0.0007 STAGE 1.
>   R-037 candidate built.)
> - `v14_recvhand` (informally banned; audit dOV +0.0004 STAGE 1.
>   R-035/R-039 candidates built.)
> - `meta_stack` (R-005 PARKED on standalone OOF; audit dOV +0.0012
>   STAGE 1, NEW SIGNAL CLASS. R-036 candidate built.)
> - `meta_stack_v2_logistic` (R-005 PARKED; audit dOV +0.0006 STAGE 1,
>   NEW SIGNAL CLASS. R-038 candidate built.)
> - `server_head_v1`, `server_head_v2` (R-006 PARKED on AUC < 0.62
>   standalone gate; SGP-specialists, blend audit not yet run on them.)
> - `v11_aug_big`, `v11_big` (banned "underperformed" without LB;
>   v11_aug_big audit dOV +0.0006 STAGE 1.)
> - `v11plus_aug` (banned without LB; audit dOV −0.0004 STAGE 2.)
> - `v16_seed1` (audit dOV +0.0001 STAGE 1.)
> - `v16_seed2` (audit dOV −0.0004 STAGE 2.)
> - `v14_seed0`, `v14_seed1` (banned "redundant + not LB-validated"
>   — verbatim, the rationale literally said "not LB-validated".
>   v14_seed0 in v13_oldtest slot: audit dOV +0.0001 STAGE 1.)
> - `v17_momentum` (R-015 PARKED on r=0.99 near-clone heuristic +
>   −0.0003 OV opt. The −0.0003 is below noise; the high correlation
>   doesn't preclude marginal blend contribution. Has been in zoo
>   subsets but never standalone LB-tested.)
>
> **CLASS-LEVEL RULES:**
> - **CLASS B-impure (architecture swap) — HARD-CONFIRMED 2026-05-22**:
>   2 LB datapoints, 2 different blend slots, same ratio 0.97-0.98:
>     - R-028 top1 (v11plus → v11_mulminet_aug_oldtest_avg2): OOF +0.001,
>       LB −0.0086, ratio 0.97
>     - R-040 (v11_aug_oldtest → v11_mulminet_aug_avg3): OOF +0.0030
>       (BIGGEST audit OOF lift), LB −0.0094, ratio 0.98
>   Even the largest OOF margin gets cancelled by the ratio drop. All 14
>   v11_mulminet variants in parked-audit are now NOT blend-eligible as
>   swap candidates. **Architecture-change swaps are banned for blend
>   intake until a new mechanism is identified that breaks this pattern.**
> - **CLASS B-seedavg (within-family averaging)** — extrapolated from
>   R-033 (1 datapoint, LB −0.0015). v14_seed0/1/2_oldtest haven't been
>   blend-tested as separate components yet.
> - **CLASS B-feature (R-034 LB-WIN class)** — CONFIRMED 2 datapoints:
>   R-034 (+0.0028) + the rule_override stack on R-042 (+0.0028). Same
>   architecture, same data, NEW features. Ratio 1.01+ to 1.0151.
> - **POST-PROCESS (rule_override) — CONFIRMED 2026-05-22**: R-042 = R-034
>   + apply_rule_override (10 row changes) → +0.0028 LB. Teammate claimed
>   +0.0014, ours delivered 2×. Should be applied to ALL future LB
>   candidates as a final post-process step. Stacks with any other lever.
>
> **CURRENTLY LB-WINNING (R-034 PAIR, LB 0.3838):**
> v11_aug_oldtest, v11plus, v13_oldtest, v14_seed2_v15feat_a, v16_avg3
>
> Verify before parking: run `python -u src/audit_all_parked_components.py`
> for blend dOV. The new two-stage gate (standalone + blend-swap) is the
> minimum bar for PARK_HARD. LB upload is the only authoritative verdict.

> **Pseudo-label V1 (current-best teacher) does NOT transfer to LB.**
> Path A V1 trained v14 with 274 high-confidence pseudo rows from the
> LB-best blend (zoo_v10 elig2, OOF 0.3771). v14_pseudo_v1 OOF (opt)
> +0.0021 vs v14_seed2 baseline. zoo_v12 elig1 (v14_pseudo_v1 +
> v16_avg3) OOF +0.0002 vs current LB best. Actual LB **−0.0068**
> (R-010 result). OOF→LB ratio collapsed 0.978→0.961.
>
> Why it's here: training on the LB-best teacher's confident predictions
> narrows the model toward the teacher's specific overfit patterns
> instead of generalising. Pseudo influence was small in sample-weight
> mass (~0.1%) but the model still over-fit to the teacher's distribution.
>
> Verify: any future pseudo-label experiment uses a STRUCTURALLY
> DIFFERENT teacher (e.g. ensemble of decorrelated models, NOT a
> known-LB-best blend). Per-task subset training is OK; the failure
> mode is teacher monoculture, not the trainer plumbing.

> **Submission CSVs**: UTF-8 (no BOM), LF line endings, one row per unique
> `rally_uid` in the first-appearance order from `data/test_new.csv`.
>
> Verify: `python -c "import pandas as pd; t=pd.read_csv('data/test_new.csv'); s=pd.read_csv(<sub>); u=t.rally_uid.drop_duplicates().to_numpy(); assert len(s)==len(u) and (u==s.rally_uid.to_numpy()).all()"`.

## Architecture / head structure

> **Hierarchical point head requires a `cls0` regression gate.** A
> `is_valid × depth × side` factorisation can improve some short classes while
> silently damaging the cls0 (out-of-grid) mass.
>
> Past failure: v18_hier_point — even with an `is_valid` head, cls0 regressed by
> 0.017 and short-class F1 regressed by 0.039.
>
> Verify: any new hierarchical or factored point head includes a binary
> `is_valid` head with a calibrated cls0 prior, and the reconstruction
> renormalises after the gate.

> **Subset heads cannot use placeholder labels with `sample_weight=0`.** Models
> learn to ignore the weight and predict the placeholder.
>
> Past risk: Codex caught this before implementation for v18; the final design used
> on-grid subset training, which is the required pattern.
>
> Verify: for any subset head, the loss is computed only on rows where the
> label is genuinely defined; do not pad with `0` plus zero weight.

## Feature engineering

> **Raw player-profile / player-ID-frequency features cannot be revived raw.**
> They overfit to train identities and don't transfer to de-identified test
> players (post-2026-05-06 reset).
>
> Verify: any feature derived from `gamePlayerId` must be either rally-internal
> (rolling stats from this rally only) or fold-safe target encoding with proper
> CV split.

> **Hard per-SN-bucket blends are banned.** Per-SN weight conditioning overfit the
> blender to OOF and did not transfer.
>
> Past failure: `submission_zoo_v16_fast_04_per_sn_bucket.csv` dropped to LB
> 0.3597 despite strong OOF; zoo_v3 was a separate v16_avg3 / size-6 regression,
> not the canonical hard-bucket failure.
>
> Verify: blender's per-task weight search is global (not bucketed by
> `nsn`). Diagnostic SN-spread is a *report*, not a search variable.

> **`pointId` 正手/反手 axis is receiver-relative.** Court coordinates alone
> don't define FH/BH — receiver dominant hand does. De-identified test players
> mean the model can't recover this from player_id.
>
> Verify: any new pointId feature derives receiver dominant hand from observable
> `handId` distribution (rally history or match aggregate), not from
> `gamePlayerOtherId` lookup.

## Validation infrastructure

> **`avg_oof.py` must validate** `mask`, `test_rally_uid`, `oof_y_act`,
> `oof_y_pt`, `oof_y_srv`, `oof_nsn` are byte-equal across source tags before
> averaging.

> **CLASS A/B/C framework (REFINED 2026-05-18 after R-028 top1 LB regression).**
> Blender swap candidates fall into three classes:
>
> - **CLASS A — RE-ARRANGEMENT** (same trained components, different selection/
>   weights). 6/6 transfer FAILURES (R-007/R-008/R-016/R-017/R-020b/R-026).
>   OOF→LB ratio 0.95-0.98. Don't upload.
>
> - **CLASS B-pure — LIKE-FOR-LIKE oldtest swap** (same component architecture,
>   ONLY change is ADDING oldtest training data to a previously-non-oldtest
>   component). 1/1 SUCCESS so far (R-027 PAIR v11_aug+v13 → oldtest, LB 0.3810,
>   ratio 1.0037, LB > OOF). These DO transfer. Upload.
>
> - **CLASS B-impure — STRUCTURAL ARCHITECTURE swap with oldtest** (different
>   model architecture, even if also oldtest-trained, even if also seed-averaged).
>   1/1 FAILURE so far (R-028 top1: v11plus → v11_mulminet_aug_oldtest_avg2,
>   predicted LB 0.3826, actual 0.3724, ratio 0.9768 = back to CLASS A territory).
>   These do NOT reliably transfer. v11plus appears empirically IRREPLACEABLE
>   in the LB-best subset across 3 attempts now (R-020b, R-026, R-028 top1).
>
> - **CLASS B-seedavg — WITHIN-FAMILY seed-averaging on already-oldtest
>   components** (no architecture change, no new data class, just averaging
>   more seeds of the SAME oldtest component). 1/1 FAILURE (R-033 2026-05-20:
>   v13_oldtest → v13_oldtest_avg3, OOF Δ −0.0001, LB Δ **−0.0015**, ratio
>   1.0005). Effectively a no-op or slight regression for the blend. Seed
>   averaging within an already-trained-on-oldtest family adds no structural
>   information — variance reduction is already captured in OOF; LB sees no
>   bonus and the slight blend correlation increase hurts.
>
> - **CLASS B-feature — SAME-DATA, SAME-ARCH, NEW-FEATURES swap** (no new
>   training data, no architecture change, but the component uses an
>   EXTENDED feature set). 1/1 SUCCESS (R-034 2026-05-21: v14_seed2 →
>   v14_seed2_v15feat_a, OOF Δ −0.0005 in blend, LB Δ **+0.0028**, ratio
>   **1.0121** — highest blend-swap ratio ever observed). The component was
>   STANDALONE-rejected (OV 0.3655 vs gate 0.3717 = −0.0062 below gate), but
>   in blend its OOF tied baseline AND LB transferred at +0.0028. The
>   diversity added by new features (36 prefix aggregates: per-class freqs,
>   entropy, dominance, streaks) was net-positive on LB despite being
>   net-neutral on OOF.
>
> Verify before LB upload: ask "does the swap ADD a NEW training data class
> (oldtest where there was none, or new component class) to the blend, OR is
> it just averaging more of what's already in the blend?" Only the former
> (CLASS B-pure ADD) transfers. CLASS B-seedavg, CLASS B-impure, and CLASS A
> all fail.
>
> Updated swap-experiment scoreboard (2026-05-24 — REVISED after R-067cr WIN):
> Wins / Losses on LB-tested swaps + adds:
> - R-027 PAIR (CLASS B-pure ADD-oldtest): +0.0116 LB, ratio 1.0037 ✓
> - R-034 (CLASS B-feature, new-features same-data: v15feat_a prefix aggregates): +0.0028 LB, ratio 1.0121 ✓
> - R-042 (rule_override post-process on R-034): +0.0028 LB, stacks ✓
> - R-028 top1 (CLASS B-impure SWAP, mulminet_avg2): −0.0086 LB, ratio 0.9768 ✗
> - R-033 (CLASS B-seedavg of B-impure family, v13_oldtest_avg3): −0.0015 LB ✗
> - R-040 (CLASS B-impure SWAP, mulminet_avg3): −0.0094 LB, ratio 0.98 ✗
> - R-055 (CLASS B-impure ADD with Bayes weights, R-052 7-comp): **−0.0141 LB, ratio 0.969** ✗
> - R-062r (CLASS B-player-style swap, v16match_v2 LORO): **−0.0057 LB, ratio 0.996** ✗
> - R-054r (CLASS B-meta + B-player-style 8-comp, meta_stack_v2 + v11_aug_big + recvprofile): **−0.0103 LB, ratio 0.9848** ✗
> - **R-067cr (CLASS server-head-blend, 30% v22 + 70% R-042 SGP): +0.000355 LB ✓** (NEW LB-BEST 0.3870)
>
> Net since R-034: **3 wins** (R-034, R-042, R-067cr), 6 losses. **R-067cr 0.3870095 is LB-best 2026-05-24.**

> **CLASS server-head-blend — LB-VALIDATED 2026-05-24 (R-067cr = +0.000355 LB).**
> Path B causal LM (R-066) full-model OV failed §9.6 gate at 0.2972 < 0.314.
> But its server head had AUC 0.6873 on per-rally OOF, +0.077 above v11 baseline.
> R-067cr replaces R-042's SGP with α=0.30 × v22_SGP + 0.70 × R-042_SGP,
> keeping action+point UNCHANGED. LB result: 0.3870095 vs R-042 0.3866550 =
> **+0.000355 LB** = first non-rule-override LB win since R-034 (2026-05-21).
>
> AUC OOF→LB transfer rate: 5.4% (OOF AUC lift +0.0326 → LB OV lift +0.000355
> after 0.2 OV weight on AUC). Much weaker than action/point B-feature ratios
> (~1.01) but POSITIVE and reproducible.
>
> Mechanistic lesson: **server-head-only blends from cross-architecture
> components transfer LB-positive when (a) action+point are NOT swapped,
> (b) the new server head has a clear OOF AUC lift, (c) the blend weight is
> moderate (30% new, 70% old) to inject diversity without over-replacing
> a proven signal.**
>
> Rule: future "diversity-only zoo addition" candidates per STRATEGY §9.6 are
> eligible for SERVER-HEAD-ONLY blend testing even if their full-model OV
> fails the 0.314 gate. The server head can be diversity-positive
> independently. ETA per candidate: ~1 hr local CPU + 1 LB slot.
>
> Verify before any server-head-blend candidate: per-rally OOF AUC must be
> > R-042 baseline (~0.6134 per-shot or ~0.7355 per-rally). α-sweep should
> show smooth peak in [0.2, 0.4]; flat or monotone curve indicates no
> diversity benefit — PARK.

> **CLASS B-meta (stacking ensembles) — HARD-CONFIRMED 2026-05-24 (R-054r = −0.0103 LB).**
> R-055 (−0.0141) bundled meta_stack_v2_logistic with v11_mulminet_aug_avg3,
> so we couldn't isolate which component caused the failure. R-054r tested
> meta_stack_v2_logistic WITHOUT v11_mulminet (replaced with v11_aug_big +
> v14_recvprofile). LB came back at 0.3763 = −0.0103. OOF→LB ratio 0.9848
> matches the B-impure / B-player-style transfer pattern.
>
> Conclusion: meta_stack v1 / meta_stack_v2_logistic / future stacking heads
> are STRUCTURALLY TOXIC for this dataset. Stacking-ensemble OOF features
> overfit to train-side predictor distributions in a way that does not
> transfer to held-out test predictions. **All `meta_stack*` components are
> now BLEND-INELIGIBLE for any LB candidate, including ADDs.**
>
> R-054r also contained `v14_recvprofile` and `v11_aug_big`. Confounding
> means we cannot fully isolate meta_stack alone. But the cleanest reading
> is "meta_stack v2 in a blend with otherwise-safe components still tanked".

> **Path B causal LM (multi-position objective) — PARKED at smoke 2026-05-24 (R-066 v3).**
> User-authorized 2026-05-23 after teammate package_v8 SGP-leak quarantine.
> Two smoke runs on Kaggle T4 (~13-32 min each):
>
> - **v2 (initial)**: Fold-1 OV 0.2002. Had a classic autoregressive
>   label-alignment bug: `multi_position_loss` compared `action_logits[t]`
>   against `y_action[t]` (same position), letting the model trivially copy
>   the current shot's action from its own input embedding under causal mask.
>   AUC was inflated to 0.7945 by the same bug (server label is rally-
>   constant; positions later in the rally saw enough prefix to memorise).
>   F1_a / F1_p collapsed to 0.08 / 0.02 because eval extracted predictions
>   for shot t-1 but compared against shot t labels (mismatch).
> - **v3 (fixed)**: Standard causal-LM left-shift. Compared `logits[:, :-1]`
>   against `y[:, 1:]` (predict NEXT shot from prefix). Eval positions were
>   already aligned correctly under the shifted-target paradigm. Result:
>   Fold-1 OV 0.2885. F1_a recovered to 0.29, F1_p to 0.09, AUC settled at
>   0.6759 (still +0.066 above v11 baseline).
>
> Per STRATEGY §9.6 stop gate (OV 0.2885 < 0.295) → PARK.
>
> Mechanistic lesson: multi-position causal LM with d=192/4L converges at a
> lower last-position OV than v11's bidirectional single-target objective.
> Spreading training signal across all positions trades sharpness for
> regularization; in our setup the regularization doesn't pay back.
>
> Verify before any future causal-LM experiment: explicitly assert
> "loss compares output[t] against label[t+1]" in a unit test. The 2026-05-23
> bug took 1 wasted GPU hour to catch.
>
> **Notable partial finding**: AUC 0.6759 is +0.066 above v11 baseline. The
> SERVER HEAD alone is diversity-positive even when action/point are weak.
> R-067 (server-head-only blend) is the follow-up experiment. Path B as a
> full model is PARKED; Path B server head as a single-task blend component
> is the open question.
>
> Success/failure rule (REVISED 2026-05-23 after R-062r):
> **ADD or SWAP a new signal class → uploadable** ONLY when (a) blend-swap OOF
> tied-or-positive AND (b) the added component is in a class that is NOT
> B-impure, NOT B-seedavg-of-B-impure, NOT B-meta, **NOT B-player-style**.
> Bayes/COBYLA weight refinement is BANNED until we identify a fix that doesn't
> amplify toxic components — see B-impure ADD rule below.

> **CLASS B-player-style (per-player aggregate features) — HARD-CONFIRMED 2026-05-23 (R-062r = −0.0057 LB).**
> A new class identified after R-062r: features that aggregate observations
> **per player** (even when computed within-match and leave-one-rally-out)
> encode an effective player-style signal. Test players are de-identified
> (40/71 overlap with train per CLAUDE.md), so per-player aggregates compute
> on a DIFFERENT distribution at test time → non-transfer.
>
> R-062r evidence: v14_seed2_v16match_v2 (R-032 v2.1, Codex-APPROVED LORO
> match-context features, cap K=22, Family A only). OOF dOV +0.0037 (largest
> non-toxic blend-swap lift recorded). Actual LB 0.3809 vs R-042 0.3866 =
> −0.0057 LB. OOF→LB ratio 0.996 (B-impure territory).
>
> The match-pair aggregates were intended to provide cross-rally context that
> doesn't depend on player ID. In practice, since each match is by a SPECIFIC
> pair of players, the aggregated action/point distributions are an effective
> per-player signature, even if `gamePlayerId` is never read.
>
> Banned: any feature module that aggregates over rallies sharing the same
> `match` value (test matches are disjoint AND their players are 56% novel),
> or that aggregates over `gamePlayerId` directly (already covered by V15
> player-only ban).
>
> Verify before LB upload: ask "does the feature module produce different
> values when the rally's MATCH changes but the visible prefix stays the
> same?" If yes, it's B-player-style. Park.
>
> Related parked: v15_player_only (LB 0.3555), v15_pp (LB 0.3507), now
> v16match_v2 family. The match-disjoint train/test split + low player
> overlap (56%) makes this class structurally non-transferring.

> **CLASS B-impure ADD — HARD-CONFIRMED 2026-05-23 (R-055 = −0.0141 LB).**
> Previously we hypothesised that B-impure (architecture-swap) components
> might still help if ADDED to a blend rather than SWAPPED into it. R-055
> tested this with R-052 = R-034 + `meta_stack_v2_logistic` +
> `v11_mulminet_aug_avg3` (7-comp) under Bayes-optimised per-task weights +
> rule_override. OOF was +0.0008 above Dirichlet R-052 (Bayes refine real)
> and +0.0058 above R-042 base. **LB came back at 0.3725 — full −0.0141
> regression vs R-042 (0.3866).** OOF→LB ratio collapsed 1.014 → 0.969.
>
> The failure mechanism: Bayes weight search put 35% of the action-F1
> weight on `v11_mulminet_aug_avg3` because that component had the best
> per-task OOF F1. v11_mulminet is the SAME B-impure family that already
> LB-failed in R-028 (−0.0086) and R-040 (−0.0094). When you let a
> weight-search algorithm see that component's OOF, it concentrates mass
> there and amplifies the non-transfer.
>
> Rule: **ANY component in the v11_mulminet family** (single-seed, avg2,
> avg3, oldtest, no-oldtest, all of them) is now BLEND-INELIGIBLE for any
> LB candidate, including ADDs and seedavg derivatives. The B-impure
> architecture itself does not transfer to LB, regardless of incorporation
> method, regardless of weight strategy.
>
> Verify before any blend build: if `v11_mulminet*` or any new "different
> architecture, similar features" component is in the subset, the blend is
> not uploadable until that family produces a clean ratio≥1.0 LB datapoint
> on a STANDALONE upload (which has never happened in 14 audit entries).

> **CLASS B-meta (stacking ensembles) — PRESUMED TOXIC 2026-05-23.** R-055
> bundled `meta_stack_v2_logistic` together with `v11_mulminet_aug_avg3`,
> so we can't isolate which component caused the −0.0141 LB loss. Both
> were never standalone-LB-tested. Until a clean B-meta-only LB datapoint
> exists, treat stacking-ensemble components (meta_stack v1,
> meta_stack_v2_logistic, future stacking heads) as toxic-by-association.
> They are STAGE 1-positive in the audit (+0.0006 to +0.0012 OOF) but the
> OOF→LB transfer pattern of stacking models is unknown and the only LB
> evidence we have is negative.
>
> Verify: do not include any `meta_stack*` component in an LB candidate
> blend until either (a) a standalone B-meta-only LB upload (low-risk
> diagnostic, R-034 with one B-meta swap, ~+0.0006 OOF) is run AND
> transfers, OR (b) the audit framework adds a B-meta-vs-baseline
> empirical ratio.

> **Bayes / COBYLA weight refinement is BLEND-DANGEROUS when toxic
> components are in the candidate pool (2026-05-23 lesson).** R-055 used
> `bayes_blend_search.py` which combined Dirichlet random search (500
> samples) with `scipy.optimize.minimize(method="COBYLA")` refinement
> from top-30 seeds. The refinement found a per-task OOF maximum that
> assigned 35% action weight to v11_mulminet_aug_avg3 and 29% point
> weight to the same — this is a cliff in OOF→LB space.
>
> Dirichlet random search with uniform-ish weights (~14% each for 7
> components) would have produced a much milder OOF result AND likely a
> milder LB regression. The Bayes refinement converted a "small OOF
> regression risk" into a "−0.0141 LB cliff".
>
> Rule: **before applying Bayes/COBYLA refinement to a blend, every
> component must already be LB-validated as non-toxic on a STANDALONE
> upload**. If any component is "OOF-strong but LB-untested", do not
> Bayes-optimise; use uniform or Dirichlet weights to limit exposure.
> Equivalently: **Bayes weight search is a LB-amplifier of whatever the
> component pool gives it**. Garbage in → cliff out.
>
> Verify: any blend script with `scipy.optimize.minimize` or similar
> weight-search-beyond-Dirichlet must enforce a "components-LB-validated"
> check on the subset before running.

> **Higher-order blends (9-comp, 10-comp) inherit the toxicity of any
> single included component (2026-05-23 lesson).** `higher_order_blend_search.py`
> found +0.0013 OOF for 9c/10c configurations adding e.g.
> `v16_testhist_aug_oldtest_avg3 + v11_mulminet_aug_oldtest` on top of
> R-052. Every top-ranked 9c/10c blend included v11_mulminet family
> components → they are all expected to LB-fail in the same way as R-055.
>
> Rule: a blend's LB transfer ratio is bounded above by its WORST
> component's transfer ratio, not its best. Adding good components to a
> blend that already contains a toxic component does not dilute the
> toxicity enough to recover ratio>1.0. Higher-order search must be run
> AFTER toxic components are removed from the pool, not as a way to "wash
> out" their negative LB contribution.

> **Standalone OOF gates over-reject blend-useful components (Gate refinement
> 2026-05-21).** R-029a was standalone-rejected at OV 0.3655 (gate 0.3717,
> −0.0062 below). But its blend-swap OOF Δ was only −0.0005 (essentially
> tied), and the LB upload transferred at ratio **1.0121 → +0.0028 LB**.
>
> The standalone OV measures "how good is this component alone". The LB
> impact depends on "how does this component change the blend". These are
> different. A diversity-positive component can be standalone-weak but
> blend-positive.
>
> **Refined two-stage gate framework**:
>
> 1. **Standalone fast-track gate** (existing): OV ≥ baseline + 0.003.
>    PASSED → eligible for direct blend-swap test.
>
> 2. **Blend-swap diagnostic gate (NEW)**: even if standalone FAILS, run a
>    blend-swap analyzer test. If blend-swap OOF Δ ≥ −0.002 AND component
>    is in a NEW signal class (new features, new arch, new data, NOT pure
>    seed averaging), the component is ELIGIBLE for diagnostic LB upload.
>
> 3. **Park gate** (existing): OV < baseline − 0.010 standalone AND
>    blend-swap OOF Δ < −0.005 → park hard.
>
> Verify before parking a component: run the blend-swap analyzer test (~30s)
> BEFORE committing to PARK. Cost is trivial; the false-park rate for
> diversity-positive components was demonstrably non-zero (R-029a) and
> resulted in leaving real LB on the table.

> **`_oldtest` OOF arrays are LONGER than standard OOF arrays.** Trainers that
> received `--include-old-test data/test.csv` produce OOF arrays of shape
> ~72065 vs the standard 69712 (the extra ~2353 rows are out-of-fold predictions
> for the 2353 valid prediction samples among the 3589 added old-test rows).
> Test arrays are unchanged at 1845 rows.
>
> Past failure: 2026-05-13 first analyzer run hit `ValueError: all input arrays
> must have the same shape` in `np.stack` because v11_mulminet_aug_oldtest
> (72065 rows) was stacked with v11_aug (69712 rows).
>
> Verify: blend / analysis code that compares oldtest vs standard OOFs must
> slice the first `N_REF` rows of oldtest arrays (verified `oof_y_act[:N_REF]`
> == reference `y_act`). See `src/analyze_oldtest_blend.py:load_components`
> for the canonical slice pattern.

> **PowerShell `$Args` is a reserved auto-variable; never use it as a
> function parameter name.** Defining `param([string[]]$Args)` collides with
> the automatic `$Args` variable and breaks argument binding — the function
> sees an empty array no matter what the caller passes.
>
> Past failure: 2026-05-13 orchestrator launched v14 with EMPTY args because
> `Run-Training -Args $v14Args` silently dropped the args. v14 fell into the
> Python REPL and spammed `OSError: [WinError 123]` to stderr (144 MB of
> error log over ~1 hour before detected).
>
> Verify: in any PowerShell `param()` block intended to forward args to
> `Start-Process -ArgumentList`, name the parameter `$PyArgs` or `$CmdArgs`
> — not `$Args`. Also `$Input`, `$MyInvocation`, `$PSCmdlet`, `$PSBoundParameters`
> are reserved.

> **TIMING_TABLE estimates were 1.7-2.3× too optimistic under concurrent
> GPU+CPU load.** Single-job benchmarks underestimate wall time when both
> lanes run together — CPU contention from PyTorch dataloader workers + disk
> I/O contention with LightGBM nearly doubles per-job time.
>
> Past observation: 2026-05-19 Phase 3 orchestrator with GPU+CPU parallel:
> - v13 CPU: TIMING_TABLE estimated 87m, actual 170-182m (2.0×)
> - v11_aug GPU: estimated 110m, actual 180-206m (1.8×)
> - v16 CPU: estimated 85m, actual 200m (2.3×)
> - v11_mulminet GPU: estimated 110m, actual 230-238m (2.1×)
> - v14 CPU: estimated 134m, actual 170-182m (1.3×)
>
> Verify: when planning orchestrator deadline budgets, multiply
> single-job TIMING_TABLE estimates by ~1.8× for mixed GPU+CPU workloads.
> Update TIMING_TABLE entries with `actual_parallel` columns after each
> multi-job batch run.

> **Some component families show ZERO seed variance.** Seed averaging is a
> no-op (or worse, identical output) for these families.
>
> Empirically confirmed 2026-05-19/20 Phase 3:
> - **v11_aug_oldtest**: 4 seeds (42, 31337, 51966, 7) → ALL OV = 0.3253
>   exactly. Seed averaging cannot help; saves CPU.
> - **v11plus_oldtest**: 2 seeds (31337, 51966) → both OV = 0.3212. Same.
>
> Variance-positive families (seed averaging may help):
> - v11_mulminet_aug_oldtest: 0.3284-0.3340 spread, ~0.006 range
> - v13_oldtest: 0.3681-0.3700 spread, ~0.002 range (tight, may also be no-op)
> - v14_oldtest: 0.3680-0.3687, ~0.001 range (likely no-op)
> - v16_testhist_aug_oldtest: 0.3739-0.3747, ~0.001 range (likely no-op)
>
> Verify: before queueing seed-averaging jobs for a family, check if 2 seeds
> have already produced identical OV. If yes, don't burn CPU on more seeds —
> seed averaging will produce the same OOF arrays element-wise (up to
> numerical noise).

> **`train_v11_uncertainty.py` lacks `--include-old-test` flag.** Phase 3
> J012 failed with `unrecognized arguments: --include-old-test` (rc=2).
>
> Cause: the trainer was created before the 2026-05-13 announcement and
> wasn't retrofitted with the flag when the others (v11_mulminet, v14, v16,
> v11_transformer, v13) were.
>
> Verify: any future trainer to be used with oldtest must have the
> `--include-old-test` arg + the 11-line concat block in main(). Pattern is
> in `src/train_v11_transformer.py` lines 565-584 and others.

> **Queued-ready work auto-promotes ahead of newer proposals.** If an R-### is
> Codex-approved, implementation-complete, and blocked only on compute, it
> takes priority over a newer R-### that requires fresh dev work or has
> equivalent/lower EV. Newer proposals only jump the queue when they
> explicitly (a) have higher expected LB lift AND (b) block downstream work
> that's needed for the existing queue.
>
> Past failure (2026-05-20): R-029a (Codex-APPROVED, code+tests ready, waiting
> on Phase 3 CPU since 2026-05-18) was about to be skipped because I proposed
> launching R-030's smoke immediately after Phase 3 instead. R-030 needed
> ~2-3h of dev work BEFORE any compute could start. R-029a's compute would
> have happened anyway during the R-030 dev window — running them serially
> wasted both lanes. User caught the mistake.
>
> Verify: at every "what's next?" decision, enumerate (a) what's
> Codex-approved and waiting, (b) how long it's been waiting, (c) what its
> compute resource is (GPU/CPU). Anything ready+waiting >24h on the
> matching resource lane goes FIRST, unless explicitly justified
> otherwise. Pure dev work (no compute) for the newer R-### fills the
> compute window of the older R-### — no contention.
>
> Verify: the averaging script prints "Consistency checks passed" line; if a
> source tag was retrained on different test data (e.g. old vs new test), the
> assertion fires.

> **GroupKFold by `match`** — never row-level split, never random shuffle.

> **Fold-derived statistics are fold-safe.** Player profiles, target encoding,
> any leak-prone statistic is computed only on each fold's train rows.

## P11 player-disjoint holdout

> **P11 holdout is advisory, not LB-gating.** Confirmed 2026-05-06: holdout
> magnitude does not predict LB delta for NONE-vs-NONE comparisons. Only useful
> for THR-vs-THR ranking direction.

## Stop gates (Workflow v2.1 — TIGHTENED 2026-05-10)

> **Standard T2-component stop gate**: per-task OOF metric must beat best-
> single component by **≥ +0.003** (not +0.001). Combined OV must beat
> zoo top-1 OOF by **≥ +0.005** before a T3 submission review is opened.
>
> Why tightened: real OOF→LB ratio is ~0.978 on safe substitutions, so
> +0.001 OOF ≈ +0.001 LB — well below the +0.002 LB lift required to
> spend a submission slot (per `COLLABORATION_WORKFLOW.md` §4.6). The
> +0.001 gate produced false positives (e.g. v14_recvhand passed but
> didn't survive LB validation).
>
> Verify: any T2-component R-### preflight states this gate explicitly.

> **T2-diagnostic stop gate**: 1 h compute total (per
> `COLLABORATION_WORKFLOW.md` §2.1). At the 30-min mark, the smoke check
> must show the paradigm is plausibly working (e.g. Fold 1 metric within
> noise of best-single). If not → kill the run, document, PARK.
>
> Verify: T2-diagnostic R-### preflight states the 30-min smoke check
> AND the kill rule.

> **T2-exploration stop gate**: per `COLLABORATION_WORKFLOW.md` §4.5,
> looser gates (no +0.003 minimum). Just "not catastrophically broken"
> (e.g. AUC > random + 0.05; F1_a > 0.30) AND a useful artifact produced
> (diversity component, diagnostic, etc.).
>
> Verify: exploration R-### entry includes pre-mortem (success criterion +
> plan B for failure).

> **Per-class regression canaries** (Codex R-011 fix #3, 2026-05-11):
> the named canary classes (point cls9 BH_long, point cls5 mid_half,
> action cls1 Loop) are **tightened to 0.015 F1** drop max. Other
> meaningful-support classes keep the **0.020 F1** cap from earlier
> rules. The canaries were chosen empirically: cls9 has the largest n
> (16073), cls5 was the largest pseudo-V1 regressor (−0.0184), and cls1
> Loop was Codex's bias-amplification flag from R-009. Future feature
> experiments should report per-class deltas at all 5 fold completions.

---

## Self-check template

Before any T1+ launch, paste this into your scratchpad and fill it in:

```
LESSONS_CHECKLIST self-check for <action>:
- SGP leakage: <green / N/A / specific risk>
- Pseudo-label / external data: <green / N/A>
- Edge-rejection / submission gate: <green / N/A>
- Architecture (hier head, subset head): <green / N/A>
- Feature engineering (player-ID, per-SN buckets, pointId axis): <green / N/A>
- Validation infra (avg_oof, GroupKFold, fold-safe stats): <green / N/A>
```

Anything not "green" must either be addressed before launch or escalated to a
`R-NNN` entry in `REVIEW_QUEUE.md`.
