# STRATEGY (v3 — 2026-05-10, post zoo_v11 round)

> Live state: see `STATE_SUMMARY.md`. This file holds active strategy + plan.
> Locked rules / hard rules: see `LESSONS_CHECKLIST.md` and
> `COLLABORATION_WORKFLOW.md`.

---

## 0. Context

### 2026-05-24 UPDATE — R-067cr LB WIN: NEW LB-BEST 0.3870095

- **NEW LB-BEST: 0.3870095 (R-067cr = R-042 + α=0.30 v22 server-head blend)**
- R-067cr LB +0.000355 vs R-042 (0.3866550). Tiny but positive lift.
- Path B server head transfer rate: 5.4% of OOF AUC delta. Structurally weaker
  than action-head transfer (~100% for R-034 B-feature) but POSITIVE.
- **5-fail streak broken**: first non-rule-override LB win since R-034 on 2026-05-21.
- Net since R-034: **3 wins (R-034, R-042, R-067cr) / 6 losses**.

### 2026-05-23 UPDATE — R-055 (−0.0141) and R-062r (−0.0057) both regressed; R-042 holds

- **NEW LB best (us)**: **0.3866550 (R-042 = R-034 PAIR + rule_override post-process)** — UNCHANGED.
- R-055 (R-052 7-comp Bayes ADD + rule_override): LB **0.3725440** = −0.0141 vs R-042. B-impure ADD HARD-CONFIRMED toxic.
- R-062r (v14_seed2_v16match_v2 LORO swap + rule_override): LB **0.3809371** = −0.0057 vs R-042. **B-player-style HARD-CONFIRMED toxic** (new class added to LESSONS_CHECKLIST 2026-05-23). v16match_v2 family banned despite Codex-approved design + +0.0037 OOF lift.
- All v14+new-features SWAP candidates against R-034 (v15feat_c, v15feat_a+oldtest+avg3, testhist+v15feat_a, v16match_v2) have now either failed OOF or failed LB. R-034's v15feat_a swap was a SPECIFIC local maximum; the B-feature swap recipe on this slot is EXHAUSTED.
- 2 wins / 5 losses on LB-tested experiments since R-027 PAIR. R-042 remains undefeated.

### Post-R-062r strategic verdict

The realistic LB ceiling within the current paradigm is ≈ R-042 0.3866 (+ maybe +0.001 from a not-yet-found safe swap). Top-10 is 0.4445+; top-3 is 0.49+. The gap cannot be closed by more B-feature swap experiments.

Pending levers status (post-R-062r):

| Lever | Status | Decision |
|---|---|---|
| R-064 v15feat_d spin features | Fold-1 smoke base dOV −0.0001, opt −0.0045, AUC −0.0063 (marginal) | PARK pending higher-confidence signal — recent LB pattern says SKIP 5-fold |
| R-065c Consensus Pseudo V2c | Codex `BLOCK / ABANDON` (point pool 33 < 50 floor) | ABANDONED |
| R-060r v14_recvprofile swap (no-oldtest) | Built; predicted 0.3830-0.3870; **same B-player-style risk class as R-062r** | DO NOT UPLOAD |
| R-061r v14_recvhand swap (no-oldtest) | Built; predicted 0.3837-0.3877; **same B-player-style risk class as R-062r** | DO NOT UPLOAD |
| Path B causal LM | Untested; only unmapped structural lever | NEEDS JABIR DECISION |
| Re-examine teammate package_v8 (LB 0.4419) | Audited; rule_override extracted (R-042) | Possible final pass for unused levers |

Re-confirmed taxonomy verdicts (see `LESSONS_CHECKLIST.md`):
  - B-impure (v11_mulminet family, any incorporation) — TOXIC.
  - B-meta (stacking ensembles, meta_stack v1/v2_logistic) — PRESUMED TOXIC.
  - **B-player-style (per-player / per-match aggregates, incl. v16match_v2) — HARD-CONFIRMED TOXIC 2026-05-23.**
  - Bayes/COBYLA weight refinement — DISABLED on any blend containing LB-untested components.
  - 9-comp / 10-comp higher-order search — DISABLED until pool is filtered of v11_mulminet variants.
  - Consensus pseudo-labelling — ABANDONED per R-065c stop gate.

Sections 1-9 below were written 2026-05-10 and reflect zoo_v10 LB 0.3694 era; many specific component recommendations are now stale. The CLASS taxonomy in `LESSONS_CHECKLIST.md` is the authoritative rule book post-R-062r.

### Original 2026-05-10 baseline (kept for historical context)

- **NEW LB best (us)**: 0.3694391 (zoo_v10 elig2, 2026-05-10).
- **NEW LB top**: 0.4459209.
- **Gap**: −0.0765.
- **Days remaining**: ~20.
- **Submission slots**: 3/day = ~60 LB tries available.
- **Compute**: GPU + CPU, ~12h-window cadence.

The gap is huge. Our incremental wins this past 4 days have been
+0.0007 LB (R-004 v16_avg3 substitution). At that rate, even 60 LB tries
won't close 0.076 — and many of our tries will be neutral or negative.

Closing the gap requires **structurally new directions**, not iteration on
the v9-features × v11/v14/v16 paradigm we've exhausted.

---

## 1. What we've already exhausted (do NOT retry)

Locked from last 5+ days of experiments:

| Direction | Outcome | LB delta | Locked |
|---|---|---:|---|
| Larger transformers (v11_big, v11_aug_big) | Underfit / overfit | 0 (parked) | YES |
| Smaller transformer (v11_small) | Below default v11 | 0 (parked at gate) | YES |
| Multi-seed v14 averaging (v14_avg3) | Hurts LB | −0.0013 | YES |
| Multi-seed v16 averaging (v16_avg3) | **Helps LB** | **+0.0007** | **VALIDATED** |
| Single-seed v16 substitution (v16_seed1) | Hurts LB | −0.0023 | YES |
| LightGBM stacking meta-learner | Underfits, no signal | 0 | YES |
| Logistic stacking meta-learner | Underfits, no signal | 0 | YES |
| Rally-level prefix-only server head v1 | WEAK_STOP gate | 0 | YES |
| Rally-level prefix-only server head v2 (+ lag) | WEAK_STOP gate | 0 | YES |
| Per-SN bucket blend weights | OOF→LB regression | (old test) | YES |
| Hard hierarchical point head | Both gates failed | (old test) | YES |
| 3+ transformers in NONE blend (no v13) | Major regression | −0.0043 | YES |
| Player-profile / V15 family | Non-transfer | (old test) | YES |
| n_shots / parity / rally-length features | SGP leak | (old test) | YES |

Per `LESSONS_CHECKLIST.md`, none of these revive without explicit Codex re-review.

---

## 2. The honest LB ceiling for our current paradigm

Best blend OOF: 0.3775. OOF→LB ratio for safe substitutions: 0.978.
Implied LB ceiling **from our current paradigm**: ~0.3692. We're already
at it (0.3694).

No incremental tweak inside this paradigm will give a breakthrough. All
the "+0.001 OOF" improvements we might find translate to ~+0.001 LB and
will not close 0.076.

---

## 3. Three structural paths (this is the plan)

### Path A — Test-set pseudo-labeling (HIGHEST EV, T3, requires Jabir approval)

**Hypothesis**: top teams (LB 0.40+) almost certainly use some form of
pseudo-labeling or self-training on the test set. We have not tried this.

**Codex P1 fix (2026-05-10)**: V1 must NOT pseudo-label `serverGetPoint`.
Our server AUC is only ~0.61 — SGP pseudo-labels would amplify errors at
high rate. V1 covers `actionId` + `pointId` only; SGP rows are masked out
of the server BCE loss exactly as the test-history aug rows are. SGP
pseudo-labeling is a separate V1b experiment (after V1 result is known).

**Design (V1, action + point only)** — to be detailed in R-009 preflight:
1. Source file: predict `actionId` + `pointId` per test rally using current
   best blend (zoo_v10 elig2, OOF 0.3771). Save to `data/pseudo_v1.parquet`
   with explicit columns: `rally_uid`, `next_strikeNumber`,
   `pseudo_actionId`, `pseudo_pointId`, `act_top1_p`, `pt_top1_p`,
   `pt_is_cls0`, `is_pseudo=1`, `serverGetPoint=-1` (sentinel).
2. Filter to high-confidence test rows:
   - `act_top1_p > 0.5` (action confidence)
   - `pt_top1_p > 0.5` (point confidence) AND `pt_is_cls0 == 0` (drop
     pseudo-cls0 rows; they're "off-grid / unobserved" and would noise
     the BH/FH per-class learning)
   - Row cap: max 1500 pseudo rows (reduces bias amplification surface)
3. Sample weight `w_pseudo = 0.3` for pseudo rows; `w_real = 1.0` for
   train rows.
4. Retrain `v14_pseudo_v1` with `--feature-set v9` (no recvhand yet — keep
   single-variable). All original v9 features + pseudo rows. SGP loss
   masked exactly like P6 test-history-aug pattern: `is_pseudo == 1` rows
   excluded from server BCE.
5. Naming convention: `oof_predictions/v14_pseudo_v1_*.npy`,
   `submissions/submission_v14_pseudo_v1.csv`. NO `_pseudo_` infix in zoo
   menu / blender — keep blender unchanged in V1.
6. Compare per-fold OOF + per-class delta vs v14_seed2 baseline.

**Expected lift (action + point only)**: +0.003 to +0.012 LB if pseudo
labels are net-correct on action/point. Smaller than my prior estimate
(which assumed SGP was included).

**Risk**:
- Bias-amplification on the action/point space (still real even without SGP).
- LB→OOF gap may widen or invert.
- Hard rule §4: "Pseudo-label runs that train on predicted test labels
  require explicit Jabir approval." → T3 gate.

**Required approvals BEFORE training**:
- Jabir explicit T3 approval ("Open R-009 preflight for pseudo-labeling V1
  design.")
- R-009 preflight in REVIEW_QUEUE.md with the exact spec (source file,
  thresholds, sample weights, row cap, cls0 exclusion, SGP masking,
  artifact naming) — Codex APPROVE_WITH_FIXES or APPROVE.
- Jabir's decision on LB upload (Codex `ARTIFACT_OK` requirement REMOVED 2026-05-22).

**Cost**: ~3-4h CPU per training run. Could iterate 4-5 versions in 20 days.

### Path B — Causal autoregressive rally LM with multi-position objective (HIGH dev)

**Codex P2 fix (2026-05-10)**: a "small transformer encoder" is structurally
the same family as v11/v11_small/v11_big — we've shown this family doesn't
break out. Path B must be **structurally different**: a causal /
autoregressive sequence model with a multi-position objective, not just
another encoder.

**Hypothesis**: predict shot N from a learned representation of shots 1..N-1
where the model is trained to predict EVERY position (not just the last).
This is autoregressive language-model-style training over the rally
sequence. Test-rally visible action/point can also serve as additional
LM-pretraining data (no SGP needed).

**Design**:
- Input: full rally history as a token sequence. Each shot is a token
  with categorical embeddings (actionId × handId × spinId × strengthId ×
  pointId × positionId). Concatenate per-shot embeddings.
- Architecture: causal Transformer decoder, d=192, 4 layers, 4 heads
  (matches v11's capacity to control for hyperparameter confound).
- Objective: predict EACH shot in the rally given prefix. Per-position
  cross-entropy on action + point. Server head only on the predict-the-
  next-shot row, masked exactly per P6 (is_aug rows + is_train rows
  contribute to action/point loss but not to server BCE unless the row
  has a real SGP label).
- LM pre-training: visible test action+point shots can be used as
  additional autoregressive training data (no SGP). This lets the model
  learn rally-language structure from BOTH train + test, similar to the
  P6 test-history aug but at every position not just last.
- Train: GroupKFold(5) by match. Server BCE strictly masked.

**Stop gates (Codex P2 requirements)**:
- 1-fold smoke first (~1h GPU). Must report:
  - Fold 1 OV vs v11 baseline (must be >= v11 - 0.005 to continue)
  - Per-task F1_a / F1_p / AUC
  - OOF correlation with v11 and v14_seed2 (target: < 0.85 to be a
    diverse zoo addition; > 0.95 means redundant)
  - Whether the candidate would be eligible for the zoo blender at all
- If smoke passes, run full 5-fold (~5-6h GPU).
- If smoke fails: PARK as inert, no zoo intake.

**Expected lift**: +0.003 to +0.015 LB if it produces a diverse component
that augments the existing zoo. Lower bound is "expensive diversity-only
addition that doesn't beat v11 standalone".

**Risk**:
- Implementation cost: ~3-4 days dev work for the causal LM head + per-
  position loss + LM pre-training pipeline.
- Could still replicate v11's behavior if the data signal saturates at the
  v11 level (then it's just diversity).

**Cost**: ~1h GPU smoke + ~5-6h GPU per full training run. Iterate 2-3
versions over a week.

### Path C — New feature engineering (incremental but reliable)

**Hypothesis**: features_v9 has been our backbone. Adding genuinely
orthogonal feature classes could decorrelate from existing v14 family.

**Codex P2 fix (2026-05-10)**: cross-rally / cross-match priors are
DANGEROUS. The organizers explicitly noted that `rally_uid` ordering is
randomized in test — we cannot use rally order to infer match progression.
Any per-match prior using "what player did in prior rallies" requires
proof that the rally history is legally observable in test, otherwise
T2 review before implementation.

**Safe candidate feature classes (designed for in-rally / single-shot
features only)**:
1. **Prefix-in-rally aggregates** (always safe): features_v9 patterns
   computed over prefix shots of CURRENT rally only. Already similar
   to recvhand. Add new aggregates: e.g. mode of receiver's `pointId`
   over prefix, server's mean `strengthId` over prefix.
2. **Score-conditional features** within current rally (safe): score
   delta at shot N, score parity, score-sum thresholds — using only
   the current row's `scoreSelf` / `scoreOther`.
3. **Receiver-hand × shot-type interactions**: extension of
   `recv_hand_est` to per-shot interaction features (e.g. recv_hand × N-1
   actionId, recv_hand × N-1 strengthId). Same prefix-only safety as
   v14_recvhand.
4. **PositionId × actionId interaction**: positional play patterns
   (current row's `positionId` and prior-shot `actionId`). Single-row
   lookup, no cross-rally / cross-match.

**Banned without explicit Codex T2 review** (Codex P2):
- Cross-rally aggregates (would assume rally order is observable)
- Cross-match priors (per-player aggregates across matches)
- Anything that uses test rally ordering as a feature

**Expected lift per new feature class**: +0.0003 to +0.003 LB.

**Risk**: low (for the safe classes above). Each is a small additive
improvement, well-understood methodology. But cumulative gain is bounded
by the paradigm ceiling (~0.005 max).

**Cost**: ~1-2h dev + 3h CPU per new feature class.

### Path D — External / different architectures (LONG-shot, T3, last resort)

**Hypothesis**: top teams may use approaches outside our toolkit.
Candidates:
- Graph neural network (rally as a graph)
- Reinforcement-learning-style sequence prediction
- Pre-trained large model fine-tuning (if competition rules allow)

**Status**: Not pursuing in next 20 days unless Paths A/B/C all stall.
Implementation cost too high to fit timeline.

---

## 4. Recommended priority for next 20 days

Day-by-day budget:

| Phase | Days | Path | Goal |
|---|---|---|---|
| Phase 1 | 1–3 | **Path A pseudo-labeling V1** (assuming Jabir approves T3) | Establish whether pseudo-labeling lifts LB at all. Single iteration: predict labels with current best, filter at conf>0.6, retrain v14_pseudo, evaluate. |
| Phase 2 | 4–7 | **Path B sequence transformer V1** (parallel on GPU) | Build minimum-viable rally-sequence model. Even if it just matches v11, it's a structurally different blend component. |
| Phase 3 | 8–12 | Path A iteration: refine confidence threshold, add multi-task loss interactions | If V1 lifted LB, iterate. If V1 regressed, fall back to Path C. |
| Phase 4 | 13–16 | Path C: 2–3 new feature classes added to v14_pseudo / v14 base | Top up with incremental gains. |
| Phase 5 | 17–20 | Final blend search + ensemble of best-of-each path + LB optimization | Use remaining slots to find optimal ensemble. |

LB-target (realistic):
- Phase 1 outcome: 0.370–0.385
- Phase 2 outcome: +0.000 to +0.005 incremental
- Phase 3-5 outcome: +0.005 to +0.015 cumulative

End-of-window LB target: **0.380–0.400**. Closes ~half the gap. Not rank-1
but a major jump.

---

## 5. Decision gates (when to pivot)

After each 3-day phase, evaluate:
- LB lift achieved
- Compute used vs remaining budget
- Paradigm health (is the path still producing wins?)

Pivot rules:
- If Path A V1 LB regression > −0.005: park Path A entirely, double down on
  Paths B+C.
- If Path B doesn't beat v11 in a 3-day window: park as diversity-only
  component, focus on Paths A+C.
- If we're still at LB 0.370 by Day 10: emergency pivot to Path D
  exploration.

---

## 6. Submission slot policy (TIGHTER than before, Codex P2 update)

Past 4 days we've used ~12 slots for ~+0.0007 cumulative LB. Per slot
cost-benefit: must require predicted LB lift > +0.002 (with confidence)
to spend a slot.

**Diagnostic-slot exception (Codex P2 narrowed)**: ONLY the following
can use a slot below the +0.002 predicted-lift bar:
- A NEW structural component (Path A pseudo-label V1, Path B sequence LM,
  a new feature class first LB validation)
- A Codex-approved structural change (e.g. R-004 v16_avg3 substitution
  was Codex-vetted)

**NOT eligible for diagnostic-slot exception** (zoo_v11 elig1 lesson,
LB −0.0043 cost):
- Seed-variant substitutions (single-seed swaps)
- Average vs. single-seed substitutions (we already validated v16_avg3
  helps, v14_avg3 hurts — no further "single-variable" probes on this axis)
- Zoo-search blend-structure variations (changing transformer count,
  dropping v13, adding v11, etc. — all in the same paradigm)

For non-exception submissions: predicted LB lift > +0.002 with reasonable
confidence is required. Hold the slot otherwise.

---

## 7. What requires Jabir approval before this plan starts

1. **Path A T3 approval**: explicit "go ahead with pseudo-labeling design,
   you may open R-### preflight."
2. **Path B compute commitment**: ~30h GPU over 4-7 days. Confirm OK.
3. **Workflow updates** (see workflow re-examination memo in next section):
   tighter T3 gate, diagnostic-budget cap, stop-gate tightening.

---

## 8. Component menu freeze for the next round (Codex P3 — HARDENED)

Per current LB findings — these are now LOCKED rules for any submission
candidate, not just zoo defaults:

### Submission-candidate rules (any NONE blend bound for LB)
- **MUST contain v13** — empirical (R-008 zoo_v11 elig1 dropped v13 → −0.0043 LB).
- **MUST cap transformers at 2 of {v11, v11plus, v11_aug}** — empirical
  (R-008 added 3rd transformer + dropped v13 → −0.0043 LB combined).
- **v16_avg3 is the primary v16 component** — LB-validated (R-004 +0.0007 vs
  v16_testhist_aug).
- **v16_testhist_aug is backup only** — used in current LB-best (zoo_v8 elig3
  pre-R-004) but superseded by v16_avg3.
- **v14_seed2 is the canonical v14** — v14_avg3 hurt LB (R-007 −0.0013).
  v14_seed0 / v14_seed1 also redundant per T1 correlations.

### Eligible components for ANY submission candidate
- v13 (REQUIRED in NONE)
- v14_seed2
- v16_avg3 (primary) OR v16_testhist_aug (backup)
- Choose ≤ 2 of {v11_aug, v11plus, v11} (v11_aug + v11plus is current best
  pair)
- v12_5f (optional)
- v14_recvhand (optional, small diversity)

### Banned from submission-candidate consideration
- v14_avg3 (hurts LB)
- v14_seed0, v14_seed1 (redundant + not LB-validated)
- v16_seed1 (single-seed worse than v16_avg3 average)
- v16_seed2 (same)
- v11_big, v11_aug_big, v11plus_aug (underperformed)
- meta_stack v1, meta_stack v2_logistic (**HARD-PARKED 2026-05-23 after R-055
  −0.0141 LB; B-meta class presumed toxic by association**)
- server_head_v1, server_head_v2 (PARKED)
- **v11_mulminet entire family** (single-seed, avg2, avg3, oldtest, no-oldtest,
  all variants — HARD-PARKED 2026-05-23 after 3 LB datapoints: R-028 −0.0086,
  R-040 −0.0094, R-055 −0.0141. B-impure architecture-swap does not transfer
  to LB regardless of incorporation method)
- **R-052/R-053/R-054 blend designs** (all share v11_mulminet_aug_avg3 +
  meta_stack_v2_logistic; toxic by composition)
- **Bayes/COBYLA weight refinement on blends with LB-untested components**
  (R-055 lesson: weight search amplifies toxic components into LB cliffs)

### Future blender (zoo_v12+)
Future zoo runs use the eligible menu above. Adding v14_pseudo_v1 (Path A)
or sequence_lm_v1 (Path B) requires explicit Codex review of how they
slot into the GROUP_A/B/D/E + new GROUP_PSEUDO / GROUP_LM rules per
COLLABORATION_WORKFLOW.md §6.

---

## 9. Path B design draft (NOT implementing yet — Jabir not yet approved)

Per Jabir 2026-05-10: Path B design / smoke plan only; no implementation
or training without Path A blocked OR explicit Path B approval.

### 9.1 Architecture sketch

- Token = one shot in a rally. Token embedding is the concatenation of
  learnable categorical embeddings: `actionId` (15 bins, 15-dim each),
  `pointId` (10 bins, 10-dim each), `handId` (3 bins, 8-dim each),
  `spinId` (6 bins, 8-dim each), `strengthId` (4 bins, 8-dim each),
  `positionId` (4 bins, 8-dim each), `numberGame` (small lookup),
  `sex` (2 bins, 4-dim each), `strikeNumber` positional (sinusoidal,
  d=192). Total token dim ≈ 192 after a learned linear projection.
- Encoder: causal Transformer decoder, d=192, 4 layers, 4 heads,
  feed-forward dim 768, dropout 0.1.
- Output heads: action (15-class) + point (10-class) + server (binary).
- Causal mask: position `t` can attend to `1..t` only.

### 9.2 Multi-position objective (the Codex P2 differentiator)

For each rally with shots `1..N`, the model is trained to predict EVERY
position from its causal prefix:

```
loss = sum over t in 2..N of:
   alpha * CE(action_pred_t, action_t)
 + beta  * CE(point_pred_t,  point_t)
 + gamma * BCE(server_pred_t, server_t)         # only if server_t is real
```

`alpha`, `beta`, `gamma` follow the competition score weighting (0.4, 0.4, 0.2).
Server BCE is masked at any position where server_t is `-1` (test-history
aug rows) or where pseudo-labeling masking applies — same convention as
P6.

This is fundamentally different from v11/v14/v16 (which predict only the
LAST shot from prefix). The model learns rally-language structure at
every position, not just the prediction target.

### 9.3 LM pre-training on visible test action+point

Visible test action+point shots can be used as additional autoregressive
training data (no SGP). Treat each test rally's shots `1..M-1` as
unsupervised LM tokens with action+point self-supervision (no server).
This is a legal extension of P6's test-history-aug paradigm.

### 9.4 Inference

For each rally to predict, the model takes shots `1..N-1` as causal
input and outputs action+point+server probabilities at position N.
Same per-rally-1-prediction output shape as v11.

### 9.5 Smoke plan (~1 h GPU, before requesting full commitment)

Single-fold dry-run:
- Train `causal_lm_smoke` on Fold 1 only, ~20 epochs (truncated).
- Report:
  - Fold 1 OV vs v11 baseline (Fold 1 OV ≈ 0.314).
  - Per-task F1_a / F1_p / AUC.
  - OOF correlation with v11 (target < 0.85 to be diverse) and with
    v14_seed2 (target < 0.85).
  - Per-position loss curve (does the model use the full sequence?).

### 9.6 Stop gates for Path B

Smoke result decides whether to request the full ~30h GPU commitment:
- **If smoke Fold 1 OV >= v11 baseline (0.314)**: request full commitment.
- **If smoke Fold 1 OV in [0.295, 0.314] AND OOF correlation with v11 < 0.85**:
  request commitment for diversity-only zoo addition.
- **If smoke Fold 1 OV < 0.295 OR OOF correlation > 0.95**: PARK.

### 9.7 Required approvals BEFORE smoke

- Jabir explicit "Open R-010 exploration entry for Path B smoke" (current
  status: NOT given; per Jabir's 2026-05-10 decision Path B design only).
- R-010 exploration entry per `COLLABORATION_WORKFLOW.md` §4.5 (kind =
  `exploration`, includes pre-mortem).
- Codex APPROVE / APPROVE_WITH_FIXES on R-010 design.
- Then ~1h GPU smoke under T2-exploration budget.
- Then if smoke passes: separate Jabir approval for full ~30h GPU
  commitment.

This entry is the design draft only. Awaiting Path A outcome OR Jabir's
explicit Path B unlock.
