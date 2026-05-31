# State Summary

Updated: 2026-05-25 (autonomous /goal mode, target LB ≥ 0.4000).
Branch: dev
Data version: `data/test_new.csv` — 5668 rows / 1845 rallies (post-2026-05-06 reset)

## 2026-05-26 — Autonomous mode current state

**LB-best**: R-067cr **0.3870095** (HELD; R-072 attempt regressed to 0.3837 = −0.0033).
**Target**: 0.4000 (gap +0.0130).

### 2026-05-26 LB upload + R-071 full result

**R-072 LB UPLOAD RESULT** (2026-05-26): `submission_R072_R067cr_PLUS_RULE_V2.csv`
LB **0.3837476** = **−0.0033 vs R-067cr**, rank 55/330. **REGRESSION**.

Root cause: 9 of 11 rule overrides used Layer C (handId) / Layer D (positionId)
context — reproduces B-player-style failure mode (cf. R-062r LB −0.0057). Goal
function v0.3 deployed (2026-05-26):
- `rule_override` (R-042 shot-content only): proven 1.0 transfer (UNCHANGED)
- `rule_override_deep_prefix`: 0.3 conservative (untested isolated)
- `rule_override_player_context`: **HARD BLOCK** (new TOXIC class)

**R-071 v4 FULL 5-FOLD COMPLETE** (Kaggle, 190 min CPU, exit=0):
- OV 0.3081 (+0.0196 vs R-066 v3)
- F1_a 0.3232, F1_p 0.0973
- **AUC 0.6994** (+0.0235 vs v3) ← strongest signal
- OOF + test arrays in `oof_predictions/v22_causal_lm_v4_full_*.npy`

**R-075** (R-067cr-analog server-head blend using v4): per-rally OOF AUC
0.7713 vs R-067cr's 0.7680 = +0.0033 AUC. candidate_goal expLB +0.00003.
LOW priority — CSV exists but PARKED, not worth a slot.

### 2026-05-26 evening — goal v0.4 + Phase 2 launch

**Goal active**: `/goal` set to reach clean LB ≥ 0.4000 per v0.4 (theory-first;
LB-confirms-truth). Priority STRATEGIC > HIGH > NORMAL.

**R-082 Phase 2 launched** (user auth 2026-05-26): kernel
`aicup-r-082-v11-retrain-with-checkpoint` RUNNING on Kaggle. V11 trainer
patched with `--save-checkpoint` + `extract_embeddings` forward flag. 5-fold
v11 canonical retrain → 5 fold checkpoints for offline embedding extraction.
ETA 9-27 hr.

### 2026-05-26 evening — TWO MAJOR EVENTS

**R-170 LB-FAILED** (uploaded 2026-05-26 evening): LB **0.3813464** = **−0.0057
vs R-067cr 0.3870095**. Predicted +0.0006 to +0.0011; actual off by ~−0.0068.
**Orthogonal-mechanism stacking hypothesis FALSIFIED** by LB. Goal function
v0.5 deployed with the new lesson: combined mechanism estimates need stronger
pessimism than sum of individual estimates.

**R-082 Phase 2 TIMED OUT**: Kaggle 12hr kernel limit exceeded; CPU-only
training (no GPU allocated despite `enable_gpu=true`). Only `v11_fold0.pt`
saved (1 of 5 fold checkpoints). Full embedding extraction blocked.

### 3 ARTIFACT_READY candidates remain (status updated after R-170 LB-fail)

| Candidate | Mechanism | Predicted LB Δ | Status after R-170 |
|---|---|---:|---|
| R-094 v2 | SoftF1 action-only | +0.0003 to +0.0008 | UNTESTED individually; could still be neutral/positive |
| R-094 v1 | SoftF1 shared α | +0.0005 to +0.0010 | UNTESTED; more aggressive than v2 |
| R-081 v2 | GBM corrector | +0.0003 | UNTESTED; bounded mechanism; B-meta-adjacent risk |
| ~~R-170~~ | (orthogonal combo) | — | **LB-FAILED −0.0057** (DO NOT RE-UPLOAD) |

Decision matrix: `submissions/artifact_ready_candidates_decision_matrix_2026-05-26.md`
(predates R-170 result; treat as historical reference; R-170 row now obsolete)

### R-082 partial state
- Checkpoint: only `models/v11_fold0.pt` (fold 0 of 5)
- Pulled to: `kaggle_pulls/r082_phase2_timeout/models/v11_fold0.pt`
- Phase 3 partial smoke possible on fold-0 val rows (~20% OOF coverage)
  but signal interpretation will be weaker than full 5-fold extraction
- To complete: need per-fold kernel split (5 × ~6-8h each on Kaggle CPU)

**R-081 family closed** (NORMAL ceiling): v2 + v3 both cluster at ~+0.0003
F1 ceiling. Mechanism capped by cap-50 override constraint regardless of
target alignment. R-081 v2 CSV ARTIFACT_READY but small expected LB
(+0.0003).

**Open queue** (after R-082 Phase 2 lands):
- R-077 (v14 + focal loss across LGBM/XGB/CB): NORMAL, 6-8h engineering
- R-080 (probability stack): explicitly last-resort per user policy
- New strategic ideas: TBD pending R-082 Phase 2 result

**Goal Function v0.2 deployed**:
- `GOAL_FUNCTION.md` + `src/candidate_goal.py` (888 lines, 11/11 self-tests PASS)
- TARGET_LB=0.4000, priority bucketing STRATEGIC/HIGH/NORMAL/LOW/PARK
- 6 leakage hard-blockers (sgp_derived_proxy, forbidden_rally_uid_inference,
  teammate_leak_artifact, external_leak_data, +existing SGP-truth/overwrite guards)
- Examples: `submissions/candidate_goal_examples.json`

**Closed in autonomous mode this session**:
- R-070 v15feat_e family — Codex BLOCK 2026-05-25, PARK (slice + canary regressions)
- R-068, R-069 — implicit PARK (LOW priority churn under v0.2)
- R-073 ShuttleSet22 audit — PARK (R-021 already showed no transfer; conditional
  re-open only if R-071 v4 clears gates)

**ARTIFACT_READY_FOR_JABIR_UPLOAD_REVIEW**:
- **R-072 rule_override v2**: `submissions/submission_R072_R067cr_PLUS_RULE_V2.csv`
  - 11 new overrides on top of R-067cr (8 action + 3 point, all on contexts n≥25)
  - candidate_goal expLB +0.0015 (NORMAL priority), +11.5% of gap
  - Same mechanism as R-042 (1.0 LB transfer rate), low risk
  - See `submissions/r072_candidate_goal_verdict.json`

**R-071 v4 smoke RESULT (2026-05-25)** — Kernel `aicup-r-071-causal-lm-v4-focal-smoke`, 21.8 min wall:

| Metric | R-066 v3 | R-071 v4 | Δ | Gate (smoke) | Verdict |
|---|---:|---:|---:|---|---|
| OV | 0.2885 | **0.3002** | +0.0117 | ≥ 0.295 | **PASS** |
| F1_a | 0.2896 | **0.3221** | +0.0325 | — | focal+CB worked |
| F1_p | 0.0937 | 0.0882 | −0.0055 | — | within noise |
| AUC | 0.6759 | **0.6804** | +0.0045 | ≥ 0.65 | **PASS** |
| Push F1 mean | unknown | 0.3535 | — | ≥ 0.38 (aspirational) | FAIL (but improving direction) |

Decision: 2/3 gates pass; 3rd was aspirational and trend is correct.
**Full 5-fold launched autonomously**.

**In flight (Kaggle)**:
- **R-071 v4 FULL 5-FOLD** — kernel `aicup-r-071-causal-lm-v4-focal-full5fold`
  - Status: RUNNING (push 2026-05-25 ~15:25); ETA ~2-3h on CPU
  - Note: kernel reports CPU only (same as R-066 v3); full 5-fold should fit Kaggle's 12hr limit
  - Post-completion auto-pipeline: pull artifacts → `src/build_r075_server_blend_v4.py` → score with candidate_goal → mark ARTIFACT_READY if positive expLB

**Decision queue**: `AUTONOMOUS_RUN_QUEUE.md`

---

## Pre-2026-05-25 history (frozen for reference)



Current EDA: `eda_output/EDA_CURRENT.md` (regenerated 2026-05-08 from
`data/train.csv` + `data/test_new.csv`; stale 2026-04-05 EDA outputs removed).

## 2026-05-13 AICUP organizer announcement

The organizers explicitly permit `data/test.csv` (OLD test) as
additional training data. Caveat: overusing the leaked
`serverGetPoint` may overfit; final submissions must use
`test_new.csv`. SGP is "did server get point" (1 = yes, 0 = no).

**Per-component training pattern:** all `*_oldtest` variants use
`--include-old-test data/test.csv` to concatenate the 3589 old-test
rows to `train.csv` (84707 rows) → 88296 total training rows
(1236 OLD rallies, 55 OLD matches; 0 match overlap with train).
Concatenated rows carry full labels INCLUDING real SGP and
participate in normal training (NOT `is_aug=1`).

## Phase 1 + 2 + 3 oldtest training program — STATUS 2026-05-20

| Phase | Wave | Components | Done | Failed | Skipped |
|---|---|---|---:|---:|---:|
| 1 | initial 5 seed=42 retrains | v11_mulminet_aug_oldtest, v14_seed2_oldtest, v16_testhist_aug_oldtest, v11_aug_oldtest, v13_oldtest | 5 | 0 | 0 |
| 2 | seed-avg support (5 new seeds) | v11_mulminet_aug_oldtest_seed31337, v16_testhist_aug_oldtest seed31337+seed51966, v11plus_oldtest | 4 | 1 (`v13_oldtest_seed31337` UnboundLocal bug, since fixed) | 0 |
| 3 | full backlog (20 jobs, BACKLOG.md) | see Phase 3 results table below | 18 | 1 (J012, see below) | 1 (J020 dup) |
| **Total ELIGIBLE oldtest artifacts** | | | **27** | | |

### Phase 3 results (deadline orchestrator, 2026-05-18 20:21 → 2026-05-20 ~02:00)

All 18 completed Phase 3 artifacts passed the per-job validator (`src/validate_oof_artifact.py`) with verdict **ELIGIBLE** (failures=0, warnings=0). See `logs/BACKLOG_DONE.log` and `logs/BACKLOG_VALIDATED.log`.

| Job | Tag | Wall (min) | OV (base) | OV (opt) |
|---|---|---:|---:|---:|
| J001 | v13_oldtest_seed31337 | 170.1 | 0.3612 | 0.3681 |
| J002 | v11_aug_oldtest_seed31337 | 180.6 | — | 0.3253 |
| J003 | v13_oldtest_seed51966 | 182.1 | 0.3627 | **0.3700** |
| J004 | v11_aug_oldtest_seed51966 | 205.6 | — | 0.3253 |
| J005 | v16_testhist_aug_oldtest_seed4 | 199.6 | 0.3628 | 0.3745 |
| J006 | v16_testhist_aug_oldtest_seed7 | 202.6 | 0.3632 | 0.3741 |
| J007 | v11_mulminet_aug_oldtest_seed51966 | 238.6 | — | 0.3284 |
| J008 | v11plus_oldtest_seed31337 | 187.6 | — | 0.3212 |
| J009 | v11_mulminet_oldtest (no aug) | 230.1 | — | 0.3245 |
| J010 | v14_seed0_oldtest | 182.1 | 0.3602 | 0.3680 |
| J011 | v14_seed1_oldtest | 170.1 | 0.3605 | 0.3684 |
| J013 | v11plus_oldtest_seed51966 | 178.6 | — | 0.3212 |
| J014 | v16_testhist_aug_oldtest_seed9 | 201.1 | 0.3621 | 0.3739 |
| J015 | v13_oldtest_seed9 | 182.5 | 0.3617 | 0.3695 |
| J016 | v11_aug_oldtest_seed7 | 198.6 | — | 0.3253 |
| J017 | v11_mulminet_aug_oldtest_seed7 | 234.6 | — | 0.3314 |
| J018 | v13_oldtest_seed4 | 180.6 | 0.3619 | 0.3685 |
| J019 | v16_testhist_aug_oldtest_seed11 | (in flight) | — | — |

**Failures**:
- J012 `v11_uncertainty_aug_oldtest` — rc=2, `train_v11_uncertainty.py` lacks `--include-old-test` CLI arg. Same 3-line fix as we applied to v11_transformer/v13. Park unless prioritized.

**Skipped**:
- J020 `v13_oldtest_seed51966` — duplicate of J003; orchestrator's `Is-Job-Done` correctly skipped it on re-launch.

### Key per-family observations

- **v11_aug_oldtest seeds all converge to 0.3253** (4 seeds: 42, 31337, 51966, 7). No seed variance — v11_aug transformer config is highly deterministic given fixed data. Implication: `v11_aug_oldtest_avg3` cannot beat the single-seed standalone; seed averaging is a no-op for this family. Update LESSONS.
- **v11plus_oldtest seeds also converge to 0.3212** (2 seeds tested: 31337, 51966). Same property.
- **v11_mulminet_aug_oldtest seeds show variance** (0.3284 / 0.3298 / 0.3314 / 0.3340 across 4 seeds). avg3/avg4 may add value here.
- **v13_oldtest seeds show variance** (0.3681 / 0.3685 / 0.3695 / 0.3700 across 4 seeds). avg3/avg4 may add value.
- **v16_testhist_aug_oldtest seeds show variance** (0.3739 / 0.3741 / 0.3745 / 0.3747 across 4 seeds). avg5 may add value.
- **v14_oldtest seeds show variance** (0.3680 / 0.3684 / 0.3687 across 3 seeds). avg3 may add value.
- **v11_mulminet_oldtest (no aug)** = 0.3245 — confirms the test-history augmentation is the main lever in the v11_mulminet family (aug variant = 0.3340 vs no-aug = 0.3245, ~+0.010 gap from aug alone).

### Next steps (waiting for J019 to finish, ~30 min from now)

1. Orchestrator auto-runs `src/_build_avg.py` → builds derived avg components (v11_aug_oldtest_avg3, v13_oldtest_avg3, v11_mulminet_aug_oldtest_avg3, v16_testhist_aug_oldtest_avg5, v11plus_oldtest_avg3, v14_oldtest_avg3 — only the families with seed variance will produce meaningfully-different averages).
2. **Manually** run `python -u src/analyze_oldtest_blend_phase2.py` (orchestrator no longer auto-runs analyzer per 2026-05-18 user directive — "no submissions until artifacts reviewed").
3. Identify single-swap / pair-swap candidates vs R-027 PAIR baseline. Apply Workflow §4.6 threshold (+0.002 predicted LB) before any upload.
4. Independent: launch R-029a (`v14_seed2_v15feat_a`) — Codex-approved, code ready, 134 min CPU. Can run after Phase 3 frees the CPU.

## Current NEW LB best (UPDATED 2026-05-22 — R-042 new best 0.3866550)

`submission_R042_R034_rule_override.csv` (R-034 + apply_rule_override from
teammate v8 package): **LB 0.3866550** (rank 44/296, 2026-05-22).
**+0.0028 vs prior LB best** (R-034, 0.3838279).

R-042 = R-034 + a deterministic post-process that replaces 10 predictions
whose CE-trained likelihood is 0% under the empirical train distribution
conditioned on `(prev_action, last_action, last_point)`. 10 rows changed
total (9 actionId + 1 pointId, SGP untouched). Zero training cost.

**Delta interpretation**: Teammate v8 claimed +0.0014 LB for the same
post-process. We observed +0.0028 (2× their claim). This empirically
confirms:
- The rule_override post-process is real LB signal
- Post-process levers from teammate v8 TRANSFER to our pipeline when the
  mechanism is well-understood (vs leak-based levers which we banned)

**Implication**: Every future LB candidate should have rule_override
applied. Pre-built R-035r through R-041r (rule_override-stacked variants
of all 6 prior candidate CSVs) for next-day uploads.

## 2026-05-22 LB datapoints

| Tag | Type | LB | Δ vs R-042 baseline | Verdict |
|---|---|---:|---:|---|
| **R-042** R-034 + rule_override | LOW-risk post-process | **0.3866550** | (new baseline) | 🏆 NEW LB BEST |
| R-040 v11_mulminet_aug_avg3 swap | HIGH-risk B-impure | 0.3744469 | -0.0122 | ❌ CONFIRMED B-impure class regression (2nd datapoint, ratio 0.98) |

**CLASS B-impure rule HARD-CONFIRMED 2026-05-22**: architecture-change blend
swaps regress at ratio ~0.97-0.98 even with biggest OOF lift (+0.0030).
2 datapoints, 2 different blend slots, same ratio. All 14 v11_mulminet
variants in parked-audit are now NOT blend-eligible as swap candidates.

## Prior LB best (R-034, held ~24 hours)
`submission_R034_v14_TO_v14_v15feat_a.csv` —
LB **0.3838279** (2026-05-21). **Rank 39/284 at the time.**
Subset: **v11_aug_oldtest** + v11plus + **v13_oldtest** + **v14_seed2_v15feat_a** + v16_avg3
(NONE calibration). Single-swap: v14_seed2 → v14_seed2_v15feat_a (R-029a's component).
**+0.0028 vs its prior LB-best** (R-027 PAIR, LB 0.3810401).

R-034 establishes a NEW CLASS B sub-type — **CLASS B-feature**: same data,
same arch, NEW features (R-029a's 36 prefix aggregates: per-class freqs +
entropy/dominance + streaks). OOF Δ in blend was only −0.0005, but LB
transferred at **ratio 1.0121** (highest blend-swap ratio observed; vs
R-027 PAIR's 1.0037). This validates the user's intuition that the
standalone +0.003 OOF gate was over-rejecting blend-useful components.

**Gate refinement (R-029a → R-034 lesson)**: The standalone gate (OV ≥
baseline + 0.003) rejected R-029a at OV 0.3655. But blend-swap OOF was
only −0.0005 (tied). LB upload found +0.0028 — a meaningful win. See
LESSONS for the new two-stage gate framework.

## Prior LB best (2026-05-13 to 2026-05-21)
`submission_R027_PAIR_NONE_v11augOLD_v11plus_v13OLD_v14s2_v16avg3.csv` —
LB 0.3810401, rank 19/220 at upload time. Held for 8 days.
Subset: v11_aug_oldtest + v11plus + v13_oldtest + v14_seed2 + v16_avg3.

**R-028 update (2026-05-18)**: R-028 top1 (v11plus → mulminet+oldtest+avg2)
REGRESSED LB to 0.3724 (−0.0086). Refined the CLASS B framework: only
LIKE-FOR-LIKE oldtest swaps (CLASS B-pure) transfer; architecture-change
swaps with oldtest (CLASS B-impure) fall back to CLASS A failure (ratio
0.97). v11plus is now empirically IRREPLACEABLE across 3 attempts
(R-020b, R-026, R-028 top1). See LESSONS for refined CLASS A/B/C framework.

## Prior LB best (superseded 2026-05-13)
`submission_zoo_v10_elig2_none_v11_aug_v11plus_v13_v14s2_v16_avg3.csv` —
LB 0.3694391 (2026-05-10). Held the throne for 3 days.

## Recent LB submissions (chronological, NEW LB ladder)

| Date | File | OOF | LB | Δ vs prior best | Key signal |
|---|---|---:|---:|---:|---|
| 2026-05-07 | zoo_v8 elig3 | 0.3768 | 0.3687552 | (baseline) | OLD-LB-winner subset reproduction |
| 2026-05-09 | zoo_v10 elig1 | 0.3775 | 0.3664313 | −0.0023 | v14s0 + v16_seed1 swap REGRESSED |
| 2026-05-10 | **zoo_v10 elig2** | 0.3771 | **0.3694391** | **+0.0007** | **v16_avg3 substitution WORKS** |
| 2026-05-10 | zoo_v10 elig3 | 0.3771 | 0.3681435 | −0.0006 | v14_avg3 substitution HURTS (rank 10/169) |
| 2026-05-10 | zoo_v11 elig1 | 0.3772 | 0.3651563 | −0.0043 | drop v13 + add 3rd transformer = LARGE REGRESSION |
| 2026-05-11 | zoo_v12 elig1 (v14_pseudo_v1) | 0.3773 | **0.3626103** | **−0.0068** | **PSEUDO-LABEL BIAS AMPLIFICATION; OOF→LB ratio collapsed 0.978→0.961** |
| 2026-05-10 | **submission_v14_recvprofile** (single component) | 0.3663 | **0.3381590** | **−0.0313** | **R-011 BANNED-component submission; OOF→LB ratio 0.923 (single-model penalty + bad feature). LB confirms intake-gate FAIL was correct.** |
| 2026-05-11 | **submission_v17_momentum** (SOLO, not blend) | 0.3662 | **0.3601463** | n/a (solo vs blend) | **R-015 SOLO submission. OOF→LB ratio 0.9833 (V16-family typical). NOT directly comparable to blend LB-best 0.3694; solo-vs-blend gap reflects lack of ensemble averaging, not v17-specific regression. LB confirms v17 transfers like V16 (consistent with r=0.992 correlation). PARK still correct because v17 ≈ V16 (no new info) — but the "−0.0093 regression" framing was unfair.** |
| 2026-05-11 | **submission_R016_v11_v11aug_v13_v14s2_v16testhist** (5-comp NONE) | 0.3785 | **0.3672687** | **−0.0022** | **R-016 blender-OOF candidate; OOF +0.0019 over LB-best DID NOT TRANSFER. OOF→LB ratio 0.9703 (vs LB-best's 0.9809) — degraded transfer. THIRD instance of "blender-search OOF gains don't transfer to LB" pattern (after R-007 v14_avg3 swap −0.0013 and R-008 drop-v13 −0.0043).** |
| 2026-05-11 | **submission_R017_dirichlet_elig1_none_v11_v11plus_v13_v14_recvhand_v16_avg3** (5-comp NONE, **violates rule #12**) | 0.3773 | **0.3615465** | **−0.0079** | **R-017 Dirichlet candidate; violates LESSONS rule #12 (v11plus present WITHOUT v11_aug). OOF→LB ratio 0.9582 — empirically confirms rule #12 penalty (~0.012 ratio degradation vs compliant blends). FOURTH "blender-search OOF doesn't transfer" instance + FIRST LB-confirmation of the v11_aug-required-with-v11plus rule.** |
| 2026-05-12 | **submission_v11_mulminet_aug** (SOLO, V11 backbone + MuLMINet aux + test-history aug) | 0.3299 | **0.3518517** | n/a (solo vs blend) | **R-020a SOLO LB upload, rank 23/194. OOF→LB ratio 1.066 — RARE: LB > OOF. This is the strongest V11-family SOLO LB result EVER. Solo-vs-blend gap normal.** |
| 2026-05-12 | **submission_R020b_NONE_v11aug_v11mulminetaug_v13_v14s2_v16avg3** (5-comp NONE, v11plus → v11_mulminet_aug single swap) | 0.3738 | **0.3644967** | **−0.0049** | **R-020b SAFE candidate. Predicted +0.0026, actual −0.0049 (off by −0.0075). OOF→LB ratio 0.9751. **5TH "blender-search doesn't transfer to LB" instance** — proves pattern holds EVEN with structurally NEW components. v11_mulminet_aug solo transfers (ratio 1.066) but blend substitution does not. v11plus is empirically irreplaceable in LB-best subset.** |
| 2026-05-13 | **submission_R026_SAFE_NONE_v11aug_v11mulminetavg2_v13_v14s2_v16avg3** (5-comp NONE, v11plus → v11_mulminet_aug_avg2 seed-averaged) | 0.3751 | **0.3627918** | **−0.0066** | **R-026 SAFE candidate. Seed-averaged variant boosted standalone OV +0.013 and blend OOF +0.001 vs single seed, BUT LB regressed −0.0017 worse than R-020b. OOF→LB ratio degraded 0.975 → 0.967. **6TH "blender-search doesn't transfer" instance** — proves seed averaging ALSO fails in blend substitution. LB-best zoo_v10 elig2 confirmed empirically as hard local optimum.** |
| 2026-05-13 | **submission_R027_PAIR_NONE_v11augOLD_v11plus_v13OLD_v14s2_v16avg3** (5-comp NONE, 2-component PAIR oldtest swap: v11_aug+v13 → oldtest) | 0.3797 | **0.3810401** | **+0.0116** | **🏆 R-027 PAIR BREAKTHROUGH. Predicted LB 0.3720, actual 0.3810. OOF→LB ratio 1.0035 (LB > OOF — only 2nd ever). Rank 19/220. FIRST blender swap that BEAT predicted LB after 6 prior failures. STRUCTURAL change via oldtest training (per AICUP 2026-05-13 announcement permitting `data/test.csv` as training data) is the documented exception class to "blender swap doesn't transfer" pattern. v11_aug and v13 oldtest variants gained +0.0021 / +0.0020 standalone; PAIR blend OOF +0.0028 (mild super-additivity); LB +0.0116 (huge transfer amplification — 4x predicted gain).** |
| 2026-05-18 | **submission_R028_top1_NONE_R027_SWAP_v11plusTOv11_mulminet_aug_oldtest_avg2** (R-027 PAIR + v11plus → v11_mulminet_aug_oldtest_avg2) | 0.3813 | **0.3724530** | **−0.0086** | **R-028 REGRESSION. Predicted LB 0.3826 (using R-027's CLASS B ratio 1.0035), actual 0.3724. OOF→LB ratio 0.9768 — back to CLASS A territory. THIRD failed attempt to swap v11plus (after R-020b LB 0.3645 and R-026 LB 0.3628). **CONFIRMS v11plus is empirically IRREPLACEABLE in the LB-best subset.** Refines CLASS B framework: only LIKE-FOR-LIKE oldtest swaps (same arch, just oldtest data added) transfer. Architecture swaps + seed averaging fall back to CLASS A. R-027 PAIR remains LB-best at 0.3810.** |
| 2026-05-20 | **submission_R033_CLASSBpure_v13_TO_v13_avg3** (R-027 PAIR + v13_oldtest → v13_oldtest_avg3 — within-family seed averaging on already-oldtest component) | 0.3794 | **0.3795876** | **−0.0015** | **R-033 REGRESSION. Predicted LB 0.3808 (using R-027 ratio), actual 0.3796. OOF→LB ratio 1.0005 (much weaker than R-027 PAIR's 1.0037). Established new sub-class **CLASS B-seedavg**: within-family seed averaging on already-oldtest components does NOT transfer to LB. This was the "OOF-neg / could be LB-pos" hypothesis test — answer: NO, OOF rejection on within-family averaging was correctly conservative. 8 of 9 historical swap experiments now LB-regress; only R-027 PAIR (CLASS B-pure ADD) succeeds. Rank dropped to 35/278. R-027 PAIR remains LB-best at 0.3810.** |
| 2026-05-21 | **submission_R034_v14_TO_v14_v15feat_a** (R-027 PAIR + v14_seed2 → v14_seed2_v15feat_a — R-029a Batch A prefix aggregate features) | 0.3792 | **0.3838279** | **+0.0028** | **🏆 R-034 NEW LB BEST. Predicted LB at R-027 ratio: 0.3805. Actual LB: 0.3838 (BEAT prediction by +0.0033). OOF→LB ratio **1.0121** — highest blend-swap ratio ever observed (vs R-027 PAIR's 1.0037, R-033's 1.0005). Established NEW class **CLASS B-feature**: same data, same arch, new feature set. R-029a's component was STANDALONE-rejected (OV 0.3655 vs gate 0.3717), but blend-swap OOF Δ was only −0.0005 (tied) and LB transferred at +0.0028. Vindicates user pushback against standalone-OV gates over-rejecting blend-useful components. Rank 39/284. **Updated two-stage gate framework in LESSONS**: standalone fast-track + blend-swap diagnostic path.** |

## R-029a result (NOT LB-uploaded — failed intake gate)

**R-029a** (v14_seed2 + v15feat Batch A: 36 prefix-aggregate features including per-class freqs, entropy, dominance, streaks). Codex-approved 2026-05-18, launched 2026-05-20 13:59, completed 17:25 (205.3 min wall, solo CPU).

| Metric | R-029a | v14_seed2 baseline | Gate (≥+0.003) |
|---|---:|---:|---:|
| OV (base) | 0.3602 | 0.3621 | — |
| OV (opt) | **0.3655** | 0.3687 | ≥ 0.3717 |
| Δ vs baseline | **−0.0032** | — | — |
| Δ vs strong-pass gate | **−0.0062** | — | **FAIL** |
| F1_a (opt) | 0.3871 | — | — |
| F1_p (opt) | 0.2210 | — | — |

**Verdict**: PARK Batch A. Slightly net-negative vs baseline; does not reach strong-pass gate or diversity-pass gate (which requires ≥ 0.3687). Per Codex's sequential gate, **R-029b (transition matrix features) NOT opened** — it was gated on R-029a passing.

**Why Batch A didn't help**: per-class freq features likely overlap with information v14's GBM already extracts from raw shot lag features. Streak features are noisy. Entropy/dominance dominated by score-state features. Net signal-to-noise was slightly negative.

Code preserved at `src/features_v15feat.py` + tests. NOT a wasted experiment — concrete empirical disconfirmation of one specific structural axis.

**Sequel**: per user directive "if R-029a failed, run R-030" → R-030 (sgp_prefix_v3) Fold-1 smoke launched immediately.

## R-030 Fold-1 smoke result (NOT LB-uploaded — failed smoke gate)

**R-030 sgp_prefix_v3** (dedicated prefix-only SGP head, 65 core features). Codex-approved 2026-05-20 Fold-1 smoke-only scope. Smoke launched 17:26, completed 17:27 (0.9 min wall).

**Audits (5/5 PASS)**: strict prefix containment ✅, banned-name grep ✅, train/test schema ✅, finite values ✅, test shape 1845 ✅.

**Diagnostics**:
| Diagnostic | AUC |
|---|---:|
| Counts-only baseline | 0.5680 (well below 0.65 pause / 0.70 hard-stop) |
| No-length ablation | 0.6059 (length feature contributes only +0.005) |
| Logistic baseline | 0.5742 |
| **LightGBM Fold-1** | **0.6110** |

**Gate**: max(0.620, v14_seed2 Fold-1 baseline 0.6104 + 0.005) = 0.620.
**Verdict**: FAIL_PARK (0.6110 < 0.615 PAUSE threshold).

**Why**: SGP signal in prefix data is largely saturated at ~0.61 AUC. Four prior dedicated SGP attempts confirm this ceiling (server_head_v1 0.584, server_head_v2 0.602, v19_rally_srv 0.998-via-leak, R-030 0.611). The signal limit appears structural — can't go higher without leak or game-tree-style rally-completion modeling.

**Status**: R-030 v1 PARKED. v1b (with oldtest), full 5-fold, analyzer integration, LB submission all NOT triggered per Codex's gate.

**Code preserved**: `src/features_sgp_prefix_v3.py`, `src/sgp_prefix_v3.py`, `tests/test_features_sgp_prefix_v3.py`, `runs/sgp_prefix_v3_smoke_metadata.json`.

**Combined R-029a + R-030 outcome**: today's two queued experiments both failed their intake gates. Honest signal: we're hitting structural ceilings on the standard tabular feature + LightGBM stack. The remaining queue items (R-031 soft-F1 fine-tune, R-032 within-match cross-rally) are the only remaining structural axes that haven't been tested.

LB-best UNCHANGED: **R-027 PAIR LB 0.3810401**.

**Key empirical findings (NEW LB)**:
1. v16 seed averaging transfers (+0.0007 LB).
2. v14 seed averaging does NOT transfer (−0.0013 LB vs single seed).
3. Single-seed v16_seed1 alone underperforms v16_testhist_aug (−0.002 region).
4. OOF→LB ratio for these NONE blends ≈ 0.978 (not 0.979 — slight downward
   shift since R-003).
5. **(NEW 2026-05-10)** Dropping v13 from a NONE blend AND adding a 3rd
   transformer (v11+v11_aug+v11plus stack) cost −0.0043 LB on a single-
   variable test (zoo_v11 elig1). Either v13 is structurally critical to
   NONE blends OR 3-transformer stacks underperform 2-transformer + v13;
   confounded — both factors implicated. Treat as: keep v13 + cap
   transformers at 2 in NONE submission candidates until further evidence.
6. **(NEW 2026-05-11)** Path A pseudo-label V1 (v14_pseudo_v1, R-009 V1a-
   capped) **DOES NOT TRANSFER TO LB**. OOF +0.0021, LB **−0.0068** vs
   v14_seed2-based current best (zoo_v10 elig2). Realised OOF→LB ratio
   0.961 (vs validated 0.978). Bias-amplification confirmed: training on
   the LB-best teacher's confident predictions narrowed the model toward
   the teacher's specific overfit patterns. **PARK Path A V1.** Future
   pseudo-label experiments must use a structurally different teacher
   (e.g. ensemble of decorrelated models, not a known-best blend),
   smaller pseudo weight, or avoid pseudo on the highest-OOF subset.
7. **(NEW 2026-05-10)** v14_recvprofile single-component LB **0.3381590**
   confirms the OOF intake-gate FAIL transferred to LB. OOF→LB ratio
   0.923 — well below blend ratios (0.96–0.98), but this is a SINGLE-
   model submission so the gap mixes (a) lack of ensemble averaging and
   (b) the bad-feature penalty itself. Either way, LB is firmly in the
   reject region (−0.0313 vs current best). **Reinforces R-011 PARK +
   v14_recvprofile BAN.** Lesson: when a candidate fails OOF intake gate,
   do NOT then submit it as a single-component LB diagnostic — the slot
   is wasted (we already know it's worse than blends).
8. **(NEW 2026-05-11, FRAMING CORRECTED)** v17_momentum SOLO LB
   **0.3601463**. The OOF→LB ratio 0.9833 matches V16-family typical
   (0.978). **The "−0.0093 vs LB-best 0.3694" framing was UNFAIR**:
   v17 is a SOLO model and 0.3694 is a 5-component blend; solo-vs-blend
   comparison conflates ensemble-averaging benefit with feature design.
   The fair takeaway: v17 transfers like V16 on LB (consistent with
   r=0.992 correlation). v17 isn't broken; it's just a V16-clone, so
   the LB upload added no new information beyond what OOF + correlation
   already showed. The slot was wasted because the LB result was
   PREDICTABLE, not because the LB was bad in absolute terms. PARK
   reason: v17 ≈ V16 (no new info), not "v17 LB regressed".
9. **(NEW 2026-05-11, UPDATED to 4 instances) BLENDER-SEARCH OOF GAINS
   DO NOT TRANSFER TO LB.** Pattern now confirmed across 4 cases:
   (a) R-007 v14_avg3 substitution: OOF + ?, LB −0.0013
   (b) R-008 drop-v13 + 3-transformer: OOF unclear, LB −0.0043
   (c) R-016 v11+v11_aug+v16_testhist swap: OOF +0.0019, LB −0.0022
   (d) R-017 elig1 (rule-violating swap): OOF +0.0007, LB **−0.0079**
   The current LB-best subset (zoo_v10 elig2: v11_aug+v11plus+v13+v14_seed2+v16_avg3
   NONE) is a LOCAL OPTIMUM that exhaustive blender search over
   already-trained components cannot improve upon by re-arrangement.
   **HARD LESSON (now empirically locked)**: blender-found OOF gains via
   component swap MUST NOT be uploaded as LB probes without (a)
   STRUCTURALLY NEW components in the swap, OR (b) Codex review
   specifically addressing why this swap is structurally different.
   The path forward is NEW components or NEW calibration (not NONE),
   not blender re-arrangement.
10. **(NEW 2026-05-11) RULE #12 LB-CONFIRMED.** R-017 elig1 violated
    LESSONS rule #12 (v11plus in subset WITHOUT v11_aug) and got LB
    0.3615 = ratio 0.958 — exactly the predicted ~0.012 penalty over
    compliant blends (R-016 ratio 0.970, LB-best ratio 0.981). The
    rule "v11_aug required when v11plus is in NONE subset" was
    derived from holdout data; it is now LB-validated. Going forward:
    rule violations should be filtered BEFORE submission, not
    LB-tested. Today's 3rd slot was wasted on a rule violation we
    explicitly flagged as non-compliant.

## Active jobs
None. CPU and GPU both idle.

## Just completed (this 6h window 2026-05-09 → 2026-05-10)
- **R-005 meta_stack v1 LGBM**: FINAL OV 0.3466. All gates FAIL. Significant underfit. PARKED.
- **R-005 meta_stack v2 logistic**: FINAL OV 0.3533. All gates FAIL. PARKED.
- **R-006 server_head_v1** (rally aggregates): WEAK_STOP at Fold 1+2 mean 0.584 < 0.62. PARKED.
- **R-006 server_head_v2** (v1 + last-3 shots one-hot): WEAK_STOP at Fold 1+2 mean 0.602 < 0.62. PARKED.
- **zoo_v11 re-blend** (10-component smart menu, +v14_recvhand): eligible top-5 OOF 0.3770–0.3774. None predicted to beat current LB best 0.3694. NO new submission candidate worth a slot.
- **R-004 zoo_v10 elig2 upload** (NEW BEST 0.3694391, +0.0007).
- **R-007 zoo_v10 elig3 upload** (LB 0.3681435, v14_avg3 substitution lost).

Net window LB impact: **+0.0007 from R-004**. All other experiments were diagnostic-only.

## Submission slots (2026-05-10, post v14_recvprofile upload)
Today's known uploads: zoo_v10 elig2, zoo_v10 elig3, zoo_v11 elig1, **submission_v14_recvprofile**. Slot accounting now exceeds the documented 3-per-day cap (4 known uploads dated 2026-05-10) — **Jabir to confirm slot calendar** (whether contest day reset between earlier batch and the v14_recvprofile upload, or whether the v14_recvprofile slot was reallocated from 2026-05-11). Until clarified: assume **0 remaining today**, no further uploads until next reset.

LB best UNCHANGED: zoo_v10 elig2 = 0.3694391.

## Latest completed jobs (this 12-hour window, 2026-05-07 → 2026-05-08)

| Tag | Type | FINAL OV (opt) | Wall | Notes |
|---|---|---:|---:|---|
| v12_5f | LightGBM 5-fold | 0.3650 | 184 min | new component, unlocks GROUP_C |
| v11_big | Transformer 256/6/120 | 0.3204 (no opt) | 178 min | underperformed v11 (0.3237); diversity only |
| v16_seed1 | LightGBM v16 seed=31337 | 0.3658 | 207 min | for v16_avg3 |
| v11_aug_big | Transformer 256/6/120 + aug | 0.3208 | 207 min | underperformed v11_aug (0.3232) |
| v16_seed2 | LightGBM v16 seed=51966 | 0.3649 | 200 min | for v16_avg3 |
| v14_avg3 | derived avg | 0.3610 base | <1 s | from v14_seed0/1/2 |
| v16_avg3 | derived avg | 0.3594 base | <1 s | from v16/seed1/seed2; action F1=0.3896 strongest |
| zoo_v9 blend | 10 components | elig1 OOF 0.3771 | 135 min | added v12_5f, v14_seed1 |
| zoo_v10 blend | 13 components | elig1 OOF **0.3775** | 400 min | expanded GROUP_A to v16 family |

## Usable OOF components on `test_new.csv` (1845 test rows)

| Tag | Group | OOF role |
|---|---|---|
| v11 | D | Transformer baseline (FINAL OV 0.3237) |
| v11plus | D | Transformer + class-weight escalation |
| v11_aug | D | Transformer + test-history aug, server-mask correct |
| v11plus_aug | (not in GROUP_D) | weak diversity, FINAL OV 0.3174 |
| v11_big | (not in GROUP_D) | bigger Transformer, underperformed |
| v11_aug_big | (not in GROUP_D) | bigger Transformer + aug, underperformed |
| v12_5f | C | LightGBM 5-fold |
| v13 | E | V10 + V8 point-grammar |
| v14_seed0 / v14_seed1 / v14_seed2 | B | v14 seeded GBM |
| v14_avg3 | B | average of v14_seed0/1/2 |
| v16_testhist_aug | A | v14 + test-history aug |
| v16_seed1 / v16_seed2 | A | v16 seeded |
| v16_avg3 | A | average of v16/seed1/seed2 |

## Parked components (do not revive without Codex review)
- `v18_hier_point` — both gates failed (cls0 -0.017, short F1 -0.039)
- `v19_rally_srv` — n_shots parity leak; AUC=0.998 from rally length, not signal
- `v15_*` (hist_only / player_only / pp) — older variants superseded
- `v11_big`, `v11_aug_big` — bigger Transformer underperformed; kept in `oof_predictions/` but not in blender GROUP_D

## Open review IDs
- **`R-009`** — RUN COMPLETE: `v14_pseudo_v1` FINAL OV (opt) **0.3686** vs v14_seed2 0.3665 (+0.0021). All R-009 invariants PASS. NO catastrophic per-class regression. Above Codex park threshold (0.3635) by +0.0051. Per Codex's "improves → open R-010" rule, R-010 to be opened after zoo_v12 completes.
- ~~`R-010`~~ — RESOLVED 2026-05-11. LB 0.3626103 (−0.0068). Pseudo-label V1 PARKED. v14_pseudo_v1 BANNED.
- ~~`R-011`~~ — RUN COMPLETE 2026-05-11. **v14_recvprofile FAILED intake gate** (FINAL OV(opt) 0.3663 vs required 0.3695, −0.0032). Multi-axis (4 axes added on top of recvhand) made the model SLIGHTLY WORSE than recvhand alone (0.3668 → 0.3663). Per-class shifts mirrored recvhand (BH_short broke F1=0 floor) but no aggregate gain. **PARKED**. v14_recvprofile BANNED from submission candidates. **LB 2026-05-10: 0.3381590 single-component (−0.0313 vs current best). LB confirms PARK verdict.** See RESULTS §32.
- ~~`R-012`~~ — WITHDRAWN 2026-05-11. Drafted as Jabir-OK'd binary-SRV diagnostic; withdrawn before Codex review on Jabir's "just dont do binary ones". No slot consumed. File `submission_zoo_v10_elig2_BINARY_SRV.csv` retained on disk as inert artifact.
- **`R-013`** — Codex `APPROVE_WITH_FIXES` 2026-05-10; **all 8 fixes applied + Fold-1 smoke RUN COMPLETE** 2026-05-10 (21.2 min on RTX 3060 Ti, all 7 audits PASS). **DIVERSITY_PASS verdict**: v17 OV 0.2964 < primary gate 0.3036 (FAIL), but Pearson r = 0.53–0.58 vs ALL of v11/v11_aug/v14_seed2 (well below 0.85 strong-diversity threshold). v17 is structurally decorrelated from the entire current zoo. Recommendation: open R-014 explicitly tagged "diversity candidate only, not standalone improver"; full ~30 h GPU run requires Jabir T3 approval. Key smoke metrics: F1_a 0.30 (≈ v11), F1_p 0.18 (< all baselines), SGP AUC 0.52 (< all baselines, narrowly below 0.55 floor — diversity-only acceptable). Phase 2 OV peaked at Ep7, suggesting full run should reduce Phase 2 to ~10-15 epochs/fold + best-ckpt selection. See REVIEW_QUEUE.md Pending §R-013 (smoke results subsection) + RESULTS.md §33.
- **`R-026`** — **MAJOR FINDING — Seed averaging gives +0.0134 standalone OV.** v11_mulminet_aug seed=42 OV 0.3299, seed=31337 OV 0.3263, **AVG2 OV 0.3433** (F1_a +0.020, F1_p +0.009, AUC +0.008). All tasks improved. Best blender candidate: `(v11, v11_mulminet_aug_avg2, v12_5f, v13, v16_avg3)` NONE OV **0.3758 = +0.0046 vs LB-best** (best of 16 v11_mulminet_aug_avg2 size-5 subsets). SAFE candidate (single-swap): `(v11_aug, v11_mulminet_aug_avg2, v13, v14_seed2, v16_avg3)` OV 0.3751 = +0.0039. **Two CSVs ready** (DO NOT auto-upload): `submission_R026_SAFE_*.csv` (sha256 f9ab0395...) and `submission_R026_AGGRESSIVE_*.csv` (sha256 af3df93b...).
- **`R-021`** — RUN COMPLETE 2026-05-12. Pretrained 30 epochs causal-masked transformer (1.3 min wall, best val_total 1.667). Encoder weights (51/67 keys, 1.78M params) saved + loaded into V11. FOLD-1 SMOKE OV 0.3226 vs 0.3222 (TIE). FULL 5-FOLD R-021b: OV 0.3280 vs v11_mulminet_aug 0.3299 = −0.0019 (PARK). Pretraining didn't help: causal-bidirectional mismatch + domain gap + data signal not encoder bottleneck. Kept in zoo for ensemble.
- **`R-022`** — Combine MuLMINet aux + uncertainty MTL + aug. OV **0.3214 vs 0.3299 = −0.0085**. Techniques don't compose; uncertainty MTL emphasized SGP too aggressively, breaking the aux-loss balance. **PARK.**
- **`R-024`** — v11_uncertainty + aug full 5-fold. OV **0.3234 vs v11_aug 0.3232 = +0.0002 (TIE)**. Uncertainty MTL adds nothing on top of test-history aug. Useful as zoo component but not blender lift. **PARK as standalone**.
- **`R-021`** — DRAFTED 2026-05-12 (AWAITING_CODEX) per Codex review of original 12h plan. T2-component + EXTERNAL DATA preflight for `v11_mulminet_pretrained` (ShuttleSet22 badminton transfer learning). All 5 Codex P1/P2 fixes applied: (P1.1) formal preflight required; (P1.2) 12h scope limited to loader+schema+tiny pretrain+Fold-1 SMOKE only; (P1.3) ENCODER weights only — NO label transfer (badminton vocab → AI CUP would corrupt); (P2.4) R-019 smoke only, R-022 deferred; (P2.5) tightened smoke gates (Strong: ≥ 0.3182, Diversity: ≥ 0.3122 + correlation drop, Park: < 0.3052). Scope: ~5-7 h, NO LB submission, NO full 5-fold without separate R. See REVIEW_QUEUE Pending §R-021.
- **`EXTERNAL_DATA_RESEARCH.md`** — NEW 2026-05-12. **TOP FINDING: ShuttleSet22 (badminton sister dataset, 33,612 strokes) + MuLMINet code is the highest-EV external data path.** Our `train_v11_mulminet.py` already implements MuLMINet architecture, so adding pretraining is a natural extension. Expected +0.005 to +0.020 OV via cross-domain transfer learning. Also: TabPFN-v2 as drop-in blend leg for SGP head (+0.003-0.010 AUC). NOT useful: TTSwing (= AI CUP 2025 dataset, different task), ITTF rankings (test players de-identified), AI CUP 2025 winners' code (different task). 18-day plan: R-021 (ShuttleSet22 pretrain), R-022 (TabPFN), R-014 (Path B causal LM). Combined potential: +0.020 OV → realistic LB 0.38-0.40.
- **`RESEARCH_NOTES.md`** — NEW 2026-05-11. Literature search (~30 min agent) found 13 relevant papers. **Highest-EV finding: MuLMINet** (IJCAI CoachAI 2023 2nd place, badminton sister problem, public code). Auxiliary-task weighted loss attacks our exact failure mode (saturated tabular features, weak SGP). 1-day implementation, drops into V11 backbone. 20-day execution plan in RESEARCH_NOTES.md. Top 5 immediate moves: (1) MuLMINet aux-task loss → R-018, (2) uncertainty-weighted MTL → R-019, (3) GroupKFold-by-player audit, (4) snapshot ensembles → R-020, (5) soft-F1 fine-tune. Total expected lift if all land: +0.030 to +0.060 LB (closes ~half the gap to LB top).
- **`R-020`** — RUN COMPLETE + LB UPLOADS CONFIRMED. v11_mulminet+aug OOF +0.0067 vs v11_aug standalone. All 3 tasks improved at component level. **SOLO LB 0.3518517 (rank 23/194), OOF→LB ratio 1.066 — STRONGEST GENERALIZATION SIGNAL OF ANY COMPONENT WE HAVE**. **R-020b SAFE blend uploaded → LB 0.3645 (−0.0049, 5th blender-search transfer fail).** STRATEGIC REFRAMING (Jabir 2026-05-12): v11_mulminet_aug stays in zoo as PRIVATE-LB candidate despite public-LB blend regression. Reasoning: ratio > 1.0 indicates the component generalizes BETTER than OOF; blend substitution diluted this advantage. Path forward = (1) keep v11_mulminet_aug for FINAL submission selection diversity, (2) build variants (R-020c λ=0.1 running, R-019 uncertainty MTL queued, possibly seed-averaging), (3) final submission strategy includes BOTH public-LB-best AND v11_mulminet-heavy blends. See LESSONS new "PUBLIC-LB vs PRIVATE-LB framework" section.
- **`R-019`** — SMOKE COMPLETE 2026-05-12 — **STRONG PASS**. v11_uncertainty Fold-1 OV **0.3123 vs v11 baseline 0.3086 = +0.0037**. Per task: F1_a +0.005, F1_p −0.001, **AUC +0.0121** (significant SGP head gain — Kendall & Gal MTL did exactly what was predicted). Final learned weights at Ep 75: **action 1.728, point 0.625, server 2.767** — model autonomously emphasized SGP 5.5× more than initial uniform 0.5. Correlation r vs v11 = 0.948/0.961 (very similar to v11 baseline; not a new structural class). Per Codex P2.4 "smoke only" rule, NO full 5-fold without separate R-022. Useful for blend SGP head; could be v11 substitute. Standalone weaker than v11_mulminet_aug (0.3123 vs 0.3222) but combining MTL+aux+aug not yet tested.
- ~~`R-018`~~ — RESOLVED 2026-05-12. **PARK v11_mulminet.** Full 5-fold OV (opt) 0.3296 vs v11 baseline 0.3319 = **−0.0023** (smoke Fold-1 +0.0066 was fold-luck). F1_a +0.002 (slight gain), F1_p −0.007 (regression). Correlation r 0.67-0.74 vs all zoo (genuine diversity, between v17_causal_lm 0.55 and v17_momentum 0.99 patterns). Best blender substitution OOF +0.0003 (within noise). Dirichlet+mulminet (138 min, 10 components): IDENTICAL output to R-017 — v11_mulminet not selected in any top candidate. v11_mulminet BANNED from submission candidates pending R-020a result. R-020a (v11_mulminet + test-history aug) in progress (`bchpcyp0p`, ETA ~80 min remaining). λ=0.1 sweep + R-019 uncertainty MTL queued next. T2-component preflight for `v11_mulminet` (MuLMINet aux-task loss on V11 transformer, Path D). Highest-EV move per RESEARCH_NOTES.md literature review. Adds 4 auxiliary heads (handId/strengthId/spinId/positionId) to V11 with combined loss `0.4·L_a + 0.4·L_p + 0.2·L_SGP + λ·Σ aux_losses`. λ=0.2 default, sweep on full run. Smoke = Fold-1 full-budget (~30-60 min GPU on RTX 3060 Ti); full-run estimated ~3-5 h GPU (R-019 if smoke passes). Adapted from MuLMINet (IJCAI CoachAI 2023 2nd place, badminton sister problem, public code at github.com/stan5dard/IJCAI-CoachAI-Challenge-2023). NOT approved: full 5-fold (R-019 after smoke), zoo intake, LB submission. See REVIEW_QUEUE.md Pending §R-018.
- ~~`R-017`~~ — RESOLVED 2026-05-11. **NO LB CANDIDATE — but elig1 was uploaded against recommendation, LB 0.3615 confirmed regression.** Smart Dirichlet weight blender ran 105 min on 9 eligible components (200 subsets × 4 calibration arms × 300 Dirichlet samples = 800 candidates). Eligible NONE top OOF (opt) **0.3773** = +0.0007 vs LB-best 0.3766. **elig1 candidate (NON-COMPLIANT, violates rule #12 — v11plus without v11_aug) uploaded → LB 0.3615 = −0.0079 vs best, ratio 0.958.** This is BOTH the 4th "blender-search doesn't transfer" instance AND the FIRST LB-confirmation of rule #12 (v11_aug required when v11plus present). All other 9 R-017 submissions DO NOT UPLOAD (3 are non-compliant, 6 are predicted-regression). Next: R-018 MuLMINet preflight per RESEARCH_NOTES.md.
- ~~`R-016`~~ — RESOLVED 2026-05-11. **LB-CONFIRMED REGRESSION**. Blender-found OOF candidate `(v11, v11_aug, v13, v14_seed2, v16_testhist_aug)` uploaded → LB **0.3672687** = **−0.0022 vs current best 0.3694391**. OOF (opt) was 0.3785 = +0.0019 over LB-best — DID NOT TRANSFER. OOF→LB ratio 0.9703 (degraded from LB-best's 0.9809). Third confirmed instance of "blender-search OOF gain doesn't transfer to LB" pattern. New hard lesson logged: pure component re-arrangement on already-trained components cannot improve LB; current LB-best subset is local optimum. Future LB candidates must include STRUCTURALLY NEW components, not just rearrangements.
- ~~`R-015`~~ — RESOLVED 2026-05-11. **PARK v17_momentum + LB CONFIRMED FAIL**. v17 uploaded despite intake-fail → LB **0.3601463** (−0.0093 vs best). OOF→LB ratio 0.9833 = V16-family typical, empirically confirming v17 is V16-clone. Second instance of §3.1.2 procedural-lesson violation (R-011 v14_recvprofile was first). Standalone full 5-fold OV (opt) **0.3662 vs V16 0.3666 = −0.0003** (FAIL intake). Correlation vs v16_avg3: **r = 0.992 (action) / 0.978 (point)** — at/above Codex r > 0.99 exact-duplication threshold; v17 is a near-clone of v16_avg3 with marginal per-class shifts (notably cls9 BH_long +0.0149). Smoke Fold-1 lift +0.0040 was Fold-1 luck (mean of 5 folds dragged result down). Blender study within size-5 cap: best swap +0.0006 OOF (within noise); 6-comp +0.0027 OOF VIOLATES rule #8. **NO zoo intake. NO LB upload. v17_momentum BANNED from submission candidates.** Implementation files preserved (`src/features_v17_momentum.py`, `src/train_v17_momentum.py`) for posterity; can ablate other group combos if future Codex review unlocks. See REVIEW_QUEUE.md §R-015 (RESOLVED) + RESULTS.md §34a-f. T2-component preflight for `v17_momentum` (rally momentum / initiative / pressure-state, Path C). Wraps `features_v9_recvhand`. Final scope: `core` = Groups 1+2+3 (26 features), `all` = Groups 1+2+3+4+5 (41 features after dropping redundant `target_hitter_is_server_side`). Smoke = full-budget Fold-1 (`n_boost=3000 --max-folds 1`, ~45-60 min CPU on V16 backbone), `core` first then `all` if core passes. Revised gates (per Codex P1.2): cls0 F1 floor is `baseline − 0.010` (NOT absolute 0.55 — my original was nonsensical: actual cls0 F1 baselines are 0.16/0.41 not 0.55). Pressure scalar simplified to `is_attack × strength_factor` (fixed constants only, no fold stats per Codex P2.4). PARTIAL PUSHBACK on Codex's "Group 4 has limited marginal value" framing: only the parity bit overlaps with existing `next_is_server`; per-side AGGREGATES are genuinely new (verified via grep of features_v3.py + live feature-name dump). NOT approved: full 5-fold (R-016 after smoke), zoo intake, LB submission. See REVIEW_QUEUE.md Pending §R-015 (Codex review applied subsection) + Feedback §R-015 (Codex verbatim, line 1704).

All prior R-### are RESOLVED. See `REVIEW_QUEUE.md` Resolved section for outcomes.

## Workflow updates active (v2.1, 2026-05-10)

Per Jabir-approved workflow v2.1 fixes now folded into `COLLABORATION_WORKFLOW.md`:
1. **§3.1.1** Hard rule: NO LB upload without R-### + Codex `ARTIFACT_OK`.
2. **§2.1** T2 sub-types (T2-component / T2-diagnostic / T2-exploration)
   with per-sub-type compute caps.
3. **§4.5** R-### kind = `exploration` for novel paradigm proposals
   (looser stop gates, requires Jabir T3 approval).
4. **§4.6** Submission slot policy: predicted +0.002 LB lift OR new
   structural component / Codex-approved structural change.
5. **LESSONS_CHECKLIST** Stop gates tightened: per-task ≥ +0.003, combined
   OV ≥ +0.005.

## Hard rules changed since last review
- 2026-05-08: pointId 正手/反手 axis is **receiver-relative**, depends on receiver
  dominant hand. De-identified test players make this unrecoverable from player_id
  alone. Future point-axis features should derive receiver dominant hand from
  observable `handId` distribution rather than player_id. This rule is captured in
  `LESSONS_CHECKLIST.md`; do not rely on external Claude memory files for Codex
  context.
- 2026-05-08: teammate package `AICUP_v1_LB0.4304.zip` was audited. The 0.4304
  score is driven by old-test SGP leakage: LEAK submissions copy old-test
  `serverGetPoint` for 1236 overlapping rallies. Treat all old-test-SGP-trained
  caches/submissions from that package as quarantined, not legal zoo components.
- 2026-05-08: blender `GROUP_A` expanded from `["v16_testhist_aug"]` to
  `["v16_testhist_aug", "v16_avg3", "v16_seed1", "v16_seed2"]` (still 0-or-1).

## Submission slot status
Daily submission cap: 3. Today (2026-05-08) status pending — Jabir to confirm.

## Reference: this session's RESULTS.md sections
- §21 — post-reset Phase 2/3 retraining log
- §22 — zoo_v10 NEW best eligible OOF 0.3775
