# TRAIN_PLAN

## Round: 2026-05-10 — Path A pseudo-label V1 (post zoo_v11 round)

Active strategy: STRATEGY.md v3 §3 Path A. Workflow rules: workflow v2.1
(see `COLLABORATION_WORKFLOW.md`; review outcomes are now folded into the stable workflow).

### Active R-### entries
- **R-009** (preflight, T2-component) — Path A pseudo-label V1
  (action + point only, NO SGP). Drafted in `REVIEW_QUEUE.md` Pending.
  Awaiting Codex APPROVE / APPROVE_WITH_FIXES, then Jabir's separate
  explicit go-ahead to launch training. T0 prerequisite already complete:
  `data/pseudo_v1.parquet` + `data/pseudo_v1_distribution.json`.

### Held / not-yet-approved
- Path B causal LM exploration (STRATEGY §9). Design only; no R-010 open
  per Jabir 2026-05-10. Smoke + commitment require Jabir explicit unlock.
- Path C feature engineering (STRATEGY §3.C). Pending Path A outcome.

### Once R-009 approved + Jabir greenlight: launch sequence
1. Add `--pseudo-parquet`, `--pseudo-mode`, `--pseudo-weight` flags to
   `src/train_v14.py`.
2. Regenerate `data/pseudo_v1.parquet` with the Codex-picked thresholds
   (V1a or V1b).
3. Smoke (Fold 1 only, ~30 min CPU): F1_a vs v14_seed2 baseline.
4. If smoke passes (≥ +0.005 F1_a): full 5-fold (~3-4 h CPU).
5. Post-train ARTIFACT_OK review for any submission candidate.
6. T3 review for LB upload.

### Submission slots policy
Per `COLLABORATION_WORKFLOW.md` §4.6 (workflow v2.1):
- Slot eligibility = predicted +0.002 LB lift OR new structural component
  first LB validation OR Codex-approved structural change.
- v14_pseudo_v1 first upload qualifies as new structural component.

---

## Historical: Round 2026-05-06 — POST-LB-RESET regeneration + re-validation
(content below archived for reference; superseded by 2026-05-10 round above)


⚠ **Competition reset 2026-05-06**: organisers released `data/test_new.csv` (1,845
rallies vs old 1,236) and wiped the public leaderboard. ALL prior LB results in
RESULTS.md are invalid on the new LB. See RESULTS.md top section for the schema
diff. Locked Rules 9–13 in STRATEGY.md are now PROVISIONAL pending NEW-LB
re-validation.

**This round's primary objectives** (in order):
1. Regenerate test predictions for current-best blend components on `test_new.csv`.
2. Re-establish a baseline NEW-LB score by submitting the OLD-LB-best subset
   (zoo_v6 elig1: NONE, v11_aug + v11plus + v13 + v14_seed2 + v16) on new test.
3. Re-validate the two strongest OLD-LB findings on NEW LB:
   (a) NONE > THR (Locked Rule 11, PROVISIONAL)
   (b) v11_aug structurally critical (Locked Rule 12, PROVISIONAL)
4. ONLY AFTER baseline + re-validation: pursue step-change bets (C / B / A from
   prior round's Codex-revised plan).

---

## Codex implementation update (2026-05-06)

Applied local support for the reset without renaming data files:

- `src/config.py` now automatically prefers `data/test_new.csv` when present.
  Override with `PINGPONG_TEST_FILE` if old-test forensics are needed.
- `src/build_test_history_pairs.py` now writes `data/test_history_pairs_new.parquet`
  by default when the active test is `test_new.csv`.
- `data/test_history_pairs_new.parquet` has been rebuilt and verified:
  5,668 rows / 1,845 rallies / 3,823 aug pairs, all `serverGetPoint=-1`,
  all `is_aug=1`.
- `src/train_v16_testhist_aug.py` and `src/train_v11_transformer.py` now compute
  expected aug rows/pairs dynamically from the parquet, so they no longer fail on
  the old 3,589/2,353 constants.
- `src/blend_zoo_v2.py` now supports `--only-tags`, so we can regenerate and blend
  only the old-best 5 components instead of retraining the full 10-tag zoo menu.
- Direct V11/V16 CSV writers now use Unix LF via `lineterminator="\n"`.

Important local fact: the 1,236 overlapping old-test rally histories are identical
on the 17 shared columns, but new test adds 609 rallies. Old submissions remain
invalid because they have only 1,236 rows.

---

## Concrete Phase 0 — Active-test verification + doc sync (2026-05-06, no compute, ~15 min)

Steps (will be executed at start of next session):

```powershell
python -c "import sys; sys.path.insert(0,'src'); import config; print(config.TEST_FILE); print(config.TEST_PATH)"
python -u src\build_test_history_pairs.py
python -m py_compile src\config.py src\build_test_history_pairs.py src\train_v16_testhist_aug.py src\train_v11_transformer.py src\blend_zoo_v2.py
```

Acceptance:
- `config.TEST_FILE == "test_new.csv"`
- `data/test_history_pairs_new.parquet` exists with 5,668 rows / 3,823 pairs
- All training scripts pick up new test automatically via `TEST_PATH = data/test_new.csv`

Do not rename `data/test.csv`; keep it as old-test forensic reference.

---

## Concrete Phase 1 — Rebuild aug parquet from test_new (~5 min, no training)

```powershell
python -u src\build_test_history_pairs.py > logs\build_test_history_pairs_new.log 2>&1
```

Acceptance:
- New `data/test_history_pairs_new.parquet` written
- Raw rows: 5,668; expected pairs: 3,823
- Log shows `NO_TRUE_TEST_SGP_USED = True`
- No constant edit required; V16/V11 aug guards are now dynamic

This is a one-time pure-data step. No model training.

---

## Concrete Phase 2 — Retrain the current-best blend's components

The 5 components in zoo_v6 elig1 (OLD-LB best 0.3749) need re-trained test
predictions because:
- v11_aug + v16: depend on aug parquet (rebuilt in Phase 1)
- v11plus + v14_seed2 + v13: train on TRAIN only, but their test predictions
  are length-1236 arrays in `oof_predictions/{tag}_test_*.npy` — must regenerate
  to length-1,845 over `test_new.csv`

Re-training is required because no model checkpoints were saved during the
original training. (Future: save model state to enable inference-only re-runs.)

### 2a. v14_seed2 retrain (CPU, ~200 min)

```powershell
python -u src\train_v14.py --folds 5 --skip-cb --tag v14_seed2 --seed 51966 > logs\v14_seed2_newtest.log 2>&1
```

OOF should reproduce (deterministic with seed); test predictions will be
length-1,845. Existing `oof_predictions/v14_seed2_*.npy` will be OVERWRITTEN.

### 2b. v13 retrain (CPU, ~60 min)

```powershell
python -u src\train_v13.py --folds 5 --skip-cb --tag v13 > logs\v13_newtest.log 2>&1
```

`train_v13.py` accepts `--folds`, `--skip-cb`, and `--tag`.

### 2c. v16_testhist_aug retrain (CPU, ~180 min)

```powershell
python -u src\train_v16_testhist_aug.py --aug data\test_history_pairs_new.parquet --folds 5 --skip-cb --tag v16_testhist_aug --seed 42 > logs\v16_testhist_aug_newtest.log 2>&1
```

Aug guards are dynamic now; expected pairs should print as 3,823.

### 2d. v11plus retrain (GPU, ~80 min)

`v11plus` is V11 with class-weight escalation. Confirm the exact original command
before starting; current best guess:

```powershell
python -u src\train_v11_transformer.py --tag v11plus --point-w-scale 2.0 > logs\v11plus_newtest.log 2>&1
```

### 2e. v11_aug retrain (GPU, ~80 min)

```powershell
python -u src\train_v11_transformer.py --aug-parquet data\test_history_pairs_new.parquet --tag v11_aug > logs\v11_aug_newtest.log 2>&1
```

After Phase 1's aug parquet rebuild, this will generate v11_aug trained on
new-test-history aug. server-mask diagnostic must read 0 (Codex P6 requirement).

### Optional 2f. v11 retrain only if we decide to re-test THR/zoo_v2 family

```powershell
python -u src\train_v11_transformer.py --tag v11 > logs\v11_newtest.log 2>&1
```

Do not run this for the first baseline unless slot planning explicitly needs a
v11-based comparison. The first NEW-LB baseline can be produced from the old-best
5 tags without v11.

### Total Phase 2 cost

- Sequential (CPU + GPU not parallel): ~600 min ≈ 10 h
- Parallel (CPU on Track A, GPU on Track B): max(440, 160) ≈ 7.5 h
  - Track A (CPU): v14_seed2 + v13 + v16 = 200 + 60 + 180 = 440 min
  - Track B (GPU): v11plus + v11_aug = 80 + 80 = 160 min

**Recommended: parallel where the user's hardware allows.**

---

## Concrete Phase 3 — Regenerate zoo blends for current-best subset (~15 min)

Once Phase 2 completes, the 5 old-best component artifacts have new length-1,845
test predictions. Use `--only-tags` so stale 1,236-row artifacts from unrelated
tags do not break hard alignment.

```powershell
python -u src\blend_zoo_v2.py --only-tags v11_aug,v11plus,v13,v14_seed2,v16_testhist_aug --max-models 5 --temp-min 0.5 --prefix zoo_v8 --ranking-out submissions\zoo_v8_ranking.csv > logs\zoo_v8.log 2>&1
```

Acceptance:
- `Test n=1845` in log
- All materialized submission files validate against `data/test_new.csv`
- At least one materialized candidate exactly matches the old-best subset
  `v11_aug+v11plus+v13+v14_seed2+v16_testhist_aug`, preferably NONE calibration

---

## Concrete Phase 4 — NEW LB submission order

| Slot | Candidate | Tests |
|---|---|---|
| 1 | zoo_v8 NONE matching OLD-LB winner (v11_aug+v11plus+v13+v14_seed2+v16) | Re-establishes baseline on reset LB |
| 2 | Best THR/TEMP candidate from the SAME refreshed 5-tag set, only if OOF/P11 is close | Re-validates calibration transfer with less risk than no-v11_aug |
| 3 | Hold for post-slot-1 result OR run Bet B/C smoke winner | Do not burn on no-v11_aug unless slot-1 is strong and we explicitly want a structural diagnostic |

Do **not** prioritize the old no-v11_aug diagnostic as slot 2. It already failed
badly on the old LB (0.3547), and the reset means the first priority is to regain
a valid score, not to spend slots on a likely-low-EV ablation.

---

## Phase 5 — Step-change bets (after Phase 4 baseline established)

After the diagnostic gauntlet locks in transfer patterns on NEW LB, resume the
prior round's plan from the Codex-revised version:

- **Bet C (rally-SGP v2 with prefix-level role-aware features)**: 6.5h impl + train
- **Bet B (V11 distillation with diversity gate)**: 3h
- **Bet A (P5 causal Transformer, smoke-only commit)**: 6h smoke

These are described in detail in the prior round's planning thread and remain
applicable — the failure modes (parity leak, mimicry collapse) and gates are
distribution-independent.

---

## Total compute estimate this round

| Phase | Cost | Type |
|---|---:|---|
| 0 — active-test verification + doc sync | 15 min | manual |
| 1 — rebuild aug parquet | 5 min | CPU (one-shot) |
| 2 — retrain 5 components | 7–10 h | CPU + GPU (parallel) |
| 3 — re-run zoo + materialise candidates | 15 min | CPU |
| 4 — submit 3 candidates (LB external) | varies | none (manual upload) |
| 5 — step-change bets (C / B / A) | 12–15 h | next session |

**Phase 0–3 fits one 12 h compute day.** Phase 5 is its own session after Phase 4
LB results land.

---

## Old TRAIN_PLAN content below (HISTORICAL — applied to OLD test/LB)



This plan supersedes the 2026-05-04 round. **Current best is `zoo_v2` top-1 at LB
0.3733788** (5-model: `v11+v11plus+v13+v14_seed0+v16_testhist_aug`, THR calibration).
Future submissions must aim to beat it.

**Round 2026-05-05 LB outcomes:**

1. ✅ P1 zoo_v2 top-1 → LB **0.3733788** (+0.00389 vs zoo_v16_fast_01, gap −0.0095).
2. ❌ P2 zoo_v3 top-1 (with v16_avg3, n=6) → LB **0.3675453** (−0.00583 LB, gap **−0.0164**).
   Two changes (5→6 models AND v16→v16_avg3) confound — both implicated. Locked Rules 8/9/10
   added in STRATEGY.md to prevent recurrence.
3. ✅ V16 `--seed` plumbing patched (9 model-init sites + `np.random.seed(seed)`); flip-aug
   and GroupKFold are deterministic. v16_seed1 opt OV 0.3667, v16_seed2 opt OV 0.3674;
   per-seed variance ≈ 0.001 — V16 is **seed-insensitive**.

**Codex execution order revised: P1.5 (diagnostic re-run) → P3 → P4 → P5.** P2 closed for
direct submission; v16_avg3 may still serve as a *component* in size-≤5 controlled probes.

**Codex review status (2026-05-05): conditional sign-off pending re-review.** Open items:

1. Did the size-6 search OR the v16_avg3 swap (or both) cause the −0.0058 LB regression?
   P1.5 diagnostic should disambiguate.
2. Should the spread-penalised score include a blend-size penalty (e.g.,
   `−0.001 × (n_models − 4)`) to discourage over-large blends in the OOF ranking?
3. Should THR temperature grid lower bound move from 0.5 to 0.3, AND should we add an
   "edge rejection" filter that rejects candidates whose chosen t lies on the boundary?
4. P3 (`train_v18_hier_point.py`) skeleton — sign off before implementation.

---

## Status snapshot

| Component | Status | Notes |
|---|---|---|
| `submission_zoo_v2_top1_thr_v11_v11plus_v13_v14s0_v16.csv` | ✅ **Current best**, LB **0.3733788** | 5-model: v11+v11plus+v13+v14_seed0+v16, THR, t_a=t_p=0.5 |
| `submission_zoo_v16_fast_01_…csv` | ✅ Fallback, LB 0.3694863 | 4-model global blend |
| `submission_zoo_v3_top1_thr_v11_v11plus_v125f_v13_v14s0_v16_avg3.csv` | ❌ REGRESSED, LB 0.3675453 | n=6 with v16_avg3, gap −0.0164 |
| `submission_v16_testhist_aug_v11_optblend.csv` | ✅ Backup, LB 0.3673269 | Single-family backbone |
| `submission_v14_5f_nocb_v11_optblend.csv` | ✅ Deep fallback, LB 0.3598509 | Stable transfer |
| `src/blend_zoo_v2.py` | ✅ Implemented & LB-validated | Add `--max-models` flag for P1.5 |
| `src/train_v16_testhist_aug.py --seed` | ✅ Patched and validated | Used for v16_seed1, v16_seed2 |
| `src/avg_oof.py` | ✅ Used for V16 seeds | Built `v16_avg3` |
| `oof_predictions/v16_seed1_*.npy` | ✅ Built (opt OV 0.3667) | LB-untested as solo |
| `oof_predictions/v16_seed2_*.npy` | ✅ Built (opt OV 0.3674) | LB-untested as solo |
| `oof_predictions/v16_avg3_*.npy` | ⚠ Built but transfer-suspect | Locked Rule 10 |
| `src/train_v18_hier_point.py` | ❌ Not implemented | Required for P3 (next priority) |

---

## Priority Order (revised 2026-05-05 post-deep-memo + Codex review)

| ID | Title | Status | Cost | Risk | Submission slot? |
|---|---|---|---|---|---|
| **P0** | Hold and protect current best (LB 0.3733788) | active | 0 | — | Today's slot 3 SKIPPED (no eligible candidate) |
| **P1** | Blend Zoo v2 broader search | ✅ done, LB 0.3733788 | — | — | — |
| **P2** | V16 multi-seed (`v16_seed1`, `v16_seed2`, `v16_avg3`) | ❌ done, LB regressed (−0.0058) | done | — | closed |
| **P1.5** | Diagnostic re-run with `--max-models 5` + `--temp-min 0.3` (zoo_v4a) | ✅ done; eligible top-1 OOF 0.3771 < gate; slot-3 SKIPPED | done | — | closed |
| **P3** | Hierarchical point head (`train_v18_hier_point.py`, v18) | ❌ done, gates failed (cls0 −0.0172, short −0.0392); v18 PARKED | done | — | closed (do NOT blend v18) |
| **P6** (NEW) | **V11 + test-history augmentation** (H6) | not started; Codex APPROVED 1-fold smoke with **server-mask requirement** | ~30 min impl + 30 min smoke + 90–120 min full | Medium | Yes, after smoke + full + correlation gate pass |
| **P10** (NEW) | Rally-pooled SGP head (H10, was P4) | not started | ~30 min smoke + ~2 h full | Low–med | Indirect — feeds zoo SGP channel |
| **P11** (NEW) | Player-disjoint holdout diagnostic (H11) | not started; Codex APPROVED as ADVISORY signal initially (NOT hard gate) | ~30 min impl + reuse OOFs | Low | Indirect — gates future submissions |
| **P12** (NEW) | Anchor-perturbation zoo search (H12) | not started | ~30 min impl + 50 min CPU | Low | Yes, AFTER P6/P10 add new component |
| **P7** (NEW) | GBM/zoo distillation into V11 (H7) | pending P6 outcome | ~2 h training | Medium | Conditional on P6 success |
| **P9** (NEW) | Geometry-aware point loss (H9) | not started; Codex flagged **difficulty under-estimated** | medium impl + 80 min training | Medium | Yes if smoothing implementation works |
| **P8** (PARKED) | Pseudo-labelled test rallies (H8) | NOT approved for submission | varies | High (rule risk) | NO — requires explicit Jabir policy approval before any submission training |
| **P5** | Autoregressive sequence model (P5 smoke) | deferred (cost > remaining budget) | 1.5 h smoke / 8–10 h full | High | Deferred until P6/P10 settle |

---

## P0 — Protect current best

Rules (no training; pure discipline):

- **Do not** submit any candidate whose expected LB < 0.3733788 + 0.001 cushion.
- **Do not** delete or move the current-best submission file (`submission_zoo_v2_top1_thr_v11_v11plus_v13_v14s0_v16.csv`).
- **Do not** mutate any artifact under `oof_predictions/` for tags listed in §"Component menu" below.
- Today (2026-05-05) submission slots: **2/3 used; slot 3 SKIPPED** (P1.5 Run A failed gate AND v18 failed gates; no eligible candidate). Next submission ≥ 2026-05-06.

Component menu (frozen — these are the inputs for P1/P2):

| Tag | What | OOF mask | Notes |
|---|---|---|---|
| `v16_testhist_aug` | V14 + 2353 test-history aug pairs | 69712/69712 | Backbone of current best |
| `v14_seed0` | V14 seed=42 (= v14_5f_nocb) | 69712/69712 | Solo opt 0.3661 |
| `v14_seed1` | V14 seed=48879 | 69712/69712 | Component of current best |
| `v14_seed2` | V14 seed=51966 | 69712/69712 | Solo opt 0.3665 |
| `v14_avg3` | Avg of v14_seed0/1/2 | 69712/69712 | Solo OOF 0.3623; +V11 0.3765 |
| `v14_5f_nocb` | (= v14_seed0; alias) | 69712/69712 | Keep as named anchor |
| `v12_5f` | V12 5-fold features_v7 | 69712/69712 | Component of current best |
| `v11` | Transformer (3-fold) | 69712/69712 | Aux blend partner |
| `v11plus` | V11+ class-weight escalation | 69712/69712 | Available but historically inert |
| `v13` | (legacy) | 69712/69712 | Diversity option |

Explicitly excluded from any final candidate (per STRATEGY hard rules):
`v15_pp`, `v15_player_only`, `v15_hist_only`, `v12cb`, `sn2_expert`, all `*_smoke` tags.

---

## P1 — Blend Zoo v2 ✅ COMPLETED (2026-05-05)

### Outcome

Top-1 (5-model: `v11+v11plus+v13+v14_seed0+v16_testhist_aug`, THR, t_a=t_p=0.5):
OOF 0.3829, F1_a 0.4145, F1_p 0.2362, AUC 0.6132, spread 0.0924 → **LB 0.3733788**
(gap −0.0095). +0.0039 vs zoo_v16_fast_01. **Current best.**

### Goal (historical, preserved for reference)

Search a broader, calibration-aware set of global multi-model blends than `zoo_v16_fast_01`.
Find a candidate whose OOF improves on 0.37998 **without** widening per-SN slice variance,
i.e., that likely transfers to LB.

### Search design

Component menu (P1 OOF artifacts only — no per-SN gating):

```
Group A (V16 family):    v16_testhist_aug
Group B (V14 family):    v14_avg3, v14_seed0, v14_seed1, v14_seed2
Group C (V12 family):    v12_5f
Group D (Transformer):   v11, v11plus
Group E (legacy diversity, optional): v13
```

Constraints (Codex-clarified):
- Choose at most 1 representative from Group B (avoid V14 self-collinearity).
- Group D selection: include **at least one** of {`v11`, `v11plus`}; both may be included
  (they are different transformer variants, not seed replicas). Default anchor is `v11`;
  `v11plus` is allowed as either a replacement or an addition.
- Total models per blend: **3, 4, 5, 6** (with both Group D members included, the max is 6:
  1×A + 1×B + 1×C + 2×D + 1×E).
- Per-task α weights: action / point / server **independent** (3 free vectors).
- α grid: {0.0, 0.1, …, 1.0} per slot (renormalised across blend members).
- **No per-SN-bucket conditioning.**

Calibration variants (cross-product with the above):
1. **THR** — current threshold-opt path (greedy + scipy). Same as zoo_v16_fast_01.
2. **TEMP** — temperature-only on the post-blend probs (T per task, no per-class threshold).
3. **CW** — global class-weight-only (per-task, no scipy threshold).
4. **NONE** — argmax of post-blend probs, no calibration.

### Implementation (new file `src/blend_zoo_v2.py`) — Codex-revised

`src/final_blend_optimized.py` is a **2-model** (primary + aux) blender. It cannot be
wrapped to do an N-way zoo. `blend_zoo_v2.py` must be a **purpose-built N-way blender**.
Required structure:

1. **Per-tag UID alignment**:
   - Load every tag's `_oof_mask.npy`, `_test_rally_uid.npy`, `_oof_y_*.npy`.
   - Assert that all tags share the **same** OOF mask (69712/69712) and the **same** test
     `rally_uid` ordering. Hard-fail if mismatched.
   - Use the intersection-of-masks only if a future tag has a smaller mask (currently all
     listed tags are full 69712).

2. **Action-dimension padding**:
   - Some bases (V11/V11+) may emit a 15-class action prob; GBM bases emit 19-class
     (with serve channels 15–18 set to ~0 for non-serve rows). Before averaging, **pad to
     19** by zero-filling the missing serve channels and renormalising over the 0–14 macro
     evaluation labels (or, equivalently, blend on the 15-dim macro space and pad after).
   - Document the chosen convention in the script header.

3. **Independent per-task weight vectors**:
   - Action weights: vector of length `n_models`, summing to 1.
   - Point weights: independent vector of length `n_models`, summing to 1.
   - Server weights: independent vector of length `n_models`, summing to 1.
   - Search per-task weights independently (no shared α).

4. **Search routine**:
   - For each model subset (size 3–6): random search per task, **n=300** weight samples
     (Dirichlet draws), evaluate OOF metric per task, take per-task argmax independently.
   - For each best-weight point, evaluate every calibration variant THR / TEMP / CW / NONE.
   - Fixed `np.random.seed(20260504)` for reproducibility.

5. **Output**:
   - CSV `submissions/zoo_v2_ranking.csv` with columns: `rank, models, action_weights,
     point_weights, server_weights, calibration, oof_ov, f1_a, f1_p, auc,
     per_sn_spread, spread_penalised_score, file`.
   - Write the top-5 submissions to `submissions/zoo_v2_top<k>_<provenance>.csv` using
     the same SUBMISSION schema as existing submissions.
   - Print a summary table sorted by `spread_penalised_score`.

6. **Hard checks (must run before any output is written)**:
   - All tags share the same OOF mask.
   - All tags share the same test `rally_uid` ordering.
   - All weight vectors sum to 1.0 ± 1e-6.
   - No NaN/inf in any blended OOF or test prob.

### Selection rule for the next submission slot

Pick the candidate that maximises:
```
score = OOF_OV - 0.5 * max(0, per_SN_spread - zoo_v16_fast_01_spread)
```
where `per_SN_spread = max(slice OV) - min(slice OV)` across {SN=2, SN=3-4, SN=5-8, SN=9-12, SN≥13}.

This penalises any blend that achieves OOF gains by widening per-SN variance — the
zoo_v16_fast_04 failure mode.

### Gates

| Gate | Pass | Fail action |
|---|---|---|
| OOF OV ≥ 0.37998 (zoo_v16_fast_01 anchor) | Continue | Park; do not submit |
| per_SN_spread ≤ zoo_v16_fast_01_spread + 0.005 | Continue | Reject; pick next-best |
| Calibration variant ∈ {TEMP, CW} OOF ≥ 0.378 exists | Prefer it as the safer LB transfer | Use THR variant only as a secondary candidate |

### Risk

- OOF overfitting from the broader grid. Mitigation: random-search cap + per-SN spread penalty above.
- Component collinearity (v14_seed0/1/2 all similar). Mitigation: cap to 1 V14 representative.

### Cost: 30–60 min CPU. Re-runs are cheap.

### Codex review checklist for P1

1. Confirm `src/blend_zoo_v2.py` does **not** introduce per-SN-bucket conditioning anywhere
   in the search (only post-hoc evaluation).
2. Confirm the OOF mask used is identical across all components (69712/69712 — assert at load).
3. Confirm class-weight calibration uses the same `ACTION_CW`/`POINT_CW` baselines as
   `final_blend_optimized.py` (do not re-tune these on OOF).
4. Confirm random-search seed is fixed for reproducibility.

---

## P1.5 — Diagnostic re-run of `blend_zoo_v2.py` (NEW, 2026-05-05)

### Goal

OOF-only diagnostic to disambiguate the −0.0058 LB regression of zoo_v3 top-1 between
two suspects: (a) blend-size growth (5 → 6) and (b) the v16 → v16_avg3 swap.

**Codex sign-off (2026-05-05) — important framing constraint:** P1.5 cannot *prove* LB
transfer for any candidate. It can only narrow which suspect is implicated based on OOF
behavior under the same restricted search space. Run B's outcome **does not rehabilitate
v16_avg3 for direct submission**; v16_avg3 may only re-enter the submission pool after a
separately LB-tested controlled probe.

### Implementation — edits to `src/blend_zoo_v2.py`

Add three CLI flags:

- `--max-models INT` (default 6, set to **5** for P1.5): cap the subset enumeration.
  Update the `enumerate_subsets()` filter to `len(subset) <= max_models`.
- `--temp-min FLOAT` (default 0.5, set to **0.3** for P1.5): lower bound for the THR/TEMP
  temperature search grid. Replace `np.arange(0.5, 3.55, 0.1)` with
  `np.arange(temp_min, 3.55, 0.1)` in `calib_thr` and `calib_temp`.
- `--edge-cushion FLOAT` (default 0.05): tolerance below `temp_min` that still counts as
  edge (e.g., t_a ≤ temp_min + 0.05 → edge).

Add a `temp_at_edge` annotation column to `zoo_v2_ranking.csv`:

- `temp_at_edge`: bool, True iff `min(t_a, t_p) <= temp_min + edge_cushion`.

**Codex fix (2026-05-05) — file materialization MUST respect eligibility:**
Currently `blend_zoo_v2.py` materializes `rows[:top_k]` (top-K by spread_penalised_score).
This breaks if the rank-1 candidate is at-edge — the eligible non-edge candidate could be
rank > top_k and would have no submission file. Fix:

1. Compute `eligible_mask = (temp_at_edge == False)` over all 412 entries.
2. Materialize the top-K by spread_penalised_score among **eligible** entries (a separate
   "eligible-rank" track), in addition to the global top-K (which keeps the full ranking
   inspectable).
3. The full `zoo_v2_ranking.csv` retains *all* entries with their `rank` (global) plus a
   new `eligible_rank` column (NaN for ineligible entries).
4. Submission filenames carry the eligible rank: `submission_{prefix}_elig_top{k}_…`.

### Run plan

Two paired runs (~50 min CPU each):

1. **Run A (single-seed v16, size ≤ 5, temp ≥ 0.3):**
   `python src/blend_zoo_v2.py --max-models 5 --temp-min 0.3 --prefix zoo_v4a
    --ranking-out submissions/zoo_v4a_ranking.csv`.
   - Eligible top-1 is the candidate for today's slot 3 IF gates pass (see below).
2. **Run B (v16_avg3, size ≤ 5, temp ≥ 0.3) — OOF-ONLY DIAGNOSTIC, NO SUBMISSION:**
   `python src/blend_zoo_v2.py --max-models 5 --temp-min 0.3
    --replace v16_testhist_aug:v16_avg3 --prefix zoo_v4b
    --ranking-out submissions/zoo_v4b_ranking.csv`.
   - Run B exists to ANSWER ONE QUESTION: under the same size-5 / temp≥0.3 restriction,
     does v16_avg3 land at OOF ≈ Run A or significantly below? This is informative about
     the OOF mechanics, NOT about LB transfer. Do not submit Run B's output without a
     separately LB-validated controlled probe.

### Selection rule for slot 3 (today, 2026-05-05)

Submit Run A's eligible top-1 ONLY if **all** of:
- OOF ≥ 0.3829 + 0.001 (clearly improves on zoo_v2 top-1)
- `temp_at_edge == False` (interior temperature)
- spread ≤ 0.0924 + 0.005
- The candidate is *structurally distinct* from zoo_v2 top-1 (different model subset OR
  meaningfully different per-task weights)
- Codex artifact review of the chosen submission CSV

If any gate fails: **skip slot 3 today**, preserve for tomorrow's strongest
P3-augmented candidate.

### Risk

Low — pure post-processing on existing OOF artifacts, no training. Does not commit any
slot unless ALL gates clear including Codex sign-off on the artifact.

### Cost

~30 min implementation + 2 × 50 min CPU = ~2 h total. P1.5 is the cheapest informative
move available before P3.

---

## P2 — V16-centered seed/ensemble ❌ COMPLETED, REGRESSED (2026-05-05)

### Outcome

- `--seed` patch applied (9 model-init sites + `np.random.seed(seed)`); flip-aug and
  GroupKFold are deterministic. Validated.
- v16_seed1 (seed 48879): solo opt OV **0.3667** (vs v16 0.3677, −0.0010).
- v16_seed2 (seed 51966): solo opt OV **0.3674** (vs v16 0.3677, −0.0003).
- Per-seed OV variance ≈ 0.001 — V16 is **seed-insensitive**.
- v16_avg3 (avg of seed42 + seed48879 + seed51966): averaged base OV **0.3597** (+0.0014
  vs single-seed v16 base 0.3583).
- zoo_v3 (with v16_avg3, n=6) top-1: OOF 0.3839 → **LB 0.3675453** (gap −0.0164, regression
  −0.0058 vs zoo_v2 top-1).

### Lessons

1. V16 seed insensitivity ≪ V14 seed sensitivity. Multi-seed averaging gives much smaller
   OOF lift on V16 than the spec assumed.
2. The zoo's spread-penalty did NOT catch the LB regression — top-1 spread (0.0913) was
   *better* than reference (0.0937).
3. The combination of (a) size-6 blend, (b) v16_avg3 swap, (c) THR with grid-edge t=0.5
   compounded to widen the OOF→LB gap from −0.0095 to −0.0164. P1.5 is needed to
   disambiguate which factor is dominant.

### Status

Closed for direct submission. v16_seed1, v16_seed2, v16_avg3 artifacts retained in
`oof_predictions/` for component-level use *only* in size-≤5 controlled probes (P1.5
Run B, future zoo iterations after Codex sign-off).

### Historical implementation (preserved for reference)

#### Step 1 — Add `--seed` to `train_v16_testhist_aug.py`

### Step 1 — Add `--seed` to `train_v16_testhist_aug.py`

`train_v16_testhist_aug.py` currently does **not** expose `--seed`. Required edits (mirror
the changes already applied to `train_v14.py`):

- Add `parser.add_argument("--seed", type=int, default=RANDOM_SEED)`.
- Set `seed = args.seed` and `np.random.seed(seed)` in `main()`.
- Replace `random_state=RANDOM_SEED` and `random_seed=RANDOM_SEED` with `…=seed` everywhere
  (LGB action, XGB action, LGB point, XGB point, LGB server, XGB server, optional CB).
- Audit the **flip-augmentation pair sampling**: if it uses `random.shuffle` or `np.random`
  without a seeded RNG instance, route it through `np.random.RandomState(seed)`.
- Update the pipeline banner to print `seed=…`.

### Step 2 — Smoke test seed1 (1 fold)

```
python -u src/train_v16_testhist_aug.py \
    --aug data/test_history_pairs.parquet \
    --folds 1 --skip-cb --tag v16_seed1_smoke --seed 48879 \
    2>&1 | tee logs/v16_seed1_smoke.log
```

Acceptance:
- `aug_pairs_per_fold == 2353` ✅
- `NO_TRUE_TEST_SGP_USED == True` ✅
- `server_aug_rows == 0` ✅
- Fold-1 solo OV within ±0.01 of v16 fold-1 (≈ 0.358 base / 0.367 opt is the V16 fold-3 high; fold-1 was 0.356 base, so accept ≥ 0.350).

### Step 3 — Two seed runs (full 5-fold, sequential)

```
# seed=48879
python -u src/train_v16_testhist_aug.py \
    --aug data/test_history_pairs.parquet \
    --folds 5 --skip-cb --tag v16_seed1 --seed 48879 \
    2>&1 | tee logs/v16_seed1.log

# seed=51966
python -u src/train_v16_testhist_aug.py \
    --aug data/test_history_pairs.parquet \
    --folds 5 --skip-cb --tag v16_seed2 --seed 51966 \
    2>&1 | tee logs/v16_seed2.log
```

Each ≈ 180 min. Total ≈ 6 h. **Do not run in parallel** (CPU contention).

The original V16 run (`v16_testhist_aug`) was at the legacy default seed 42; treat it as
`v16_seed0` for averaging purposes (no rename — just use the existing artifacts).

### Step 4 — Average the three V16 seeds

Reuse `src/avg_oof.py` (tag-agnostic):

```
python src/avg_oof.py \
    --tags v16_testhist_aug v16_seed1 v16_seed2 \
    --out-tag v16_avg3
```

(Do **not** pass `--blend-v11` here — we want the raw averaged artifacts. The blend step
happens via the zoo search in the next step.)

### Step 5 — Re-run zoo blend search with `v16_avg3` swapped in

```
# Re-run blend_zoo_v2.py with v16_testhist_aug → v16_avg3 in Group A
python src/blend_zoo_v2.py --replace v16_testhist_aug:v16_avg3
```

Submission candidates produced:
- `submissions/submission_v16_avg3_v11_optblend.csv` (single-pair sanity)
- `submissions/submission_zoo_v2_<rank>_v16avg3_…csv` (zoo top-K)

### Gates

| Gate | Pass | Fail action |
|---|---|---|
| `v16_avg3` solo OOF ≥ V16 solo (0.3677) | Continue | Park; investigate seed divergence |
| Per-fold solo OV variance smaller than single-seed V16 | Continue | Marginal; still proceed |
| `v16_avg3 + v14_avg3 + v12_5f + v11` OOF ≥ zoo_v16_fast_01 OOF (0.37998) | Submit candidate | Hold; pick best zoo v2 candidate from P1 instead |

### Risk

- V16 may already be near a per-seed ceiling. If seed1 smoke OV ≈ V16 seed0 (within 0.001),
  V16 is seed-insensitive and avg3 won't help. Abort early in that case.

---

## P3 — Hierarchical point head (soft-decoded) ❌ COMPLETED, FAILED (2026-05-05)

### Outcome (2026-05-05)

`src/train_v18_hier_point.py` ran full 5-fold (seed=42, --skip-cb, 94.7 min). Both Codex
gates failed:

| Gate | Threshold | V18 result | Pass/Fail |
|---|---|---:|---|
| cls0 F1 ≥ V14 cls0 F1 − 0.01 | ≥ 0.4279 | 0.4208 | ❌ FAIL by −0.0072 |
| short F1 (cls 1/2/3 mean) ≥ V14 short + 0.03 | ≥ 0.1511 | 0.0818 | ❌ FAIL by −0.0693 |
| F1_p ≥ 0.235 | ≥ 0.235 | 0.2066 | ❌ FAIL |
| Solo OOF OV ≥ V14 solo (0.3661) | ≥ 0.3661 | 0.3595 | ❌ FAIL |

V18 PARKED. Do NOT blend `oof_predictions/v18_*.npy` into any zoo. The product-of-marginals
factorisation (`p_valid × p_depth × p_side`) is too restrictive vs the flat 10-class joint
head — depth and side are not independent given on-grid. Soft recombination did not rescue.

Codex's deferred fallback `P(side|depth)` would be the only structural rescue; not
scheduled for this round. See RESULTS.md §12 for full per-class breakdown.

### Goal (historical, preserved)

Attack pointId F1 (0.23 → 0.27 target) by decomposing the 10-class point head into three
soft heads.

### Design

Build `src/train_v18_hier_point.py` from `train_v14.py`, replacing the point-model section.

**Codex revision: depth/side heads must be trained on the on-grid SUBSET, not via
sample_weight=0.** Sample-weight zero still feeds placeholder labels into the trainer and
can perturb class priors / leaf-stat estimates in subtle, learner-specific ways. Use
explicit row subsetting:

```python
# Inside the fold loop (per fold, per pass):
on_grid_tr = (y_pt_tr != 0)               # boolean mask on TRAIN rows of this fold
X_tr_on    = X_tr_pt[on_grid_tr]           # subset features
y_pt_tr_on = y_pt_tr[on_grid_tr]           # subset point labels (1..9 only)
y_depth_tr = np.array([DEPTH_OF[k] for k in y_pt_tr_on])  # 0/1/2
y_side_tr  = np.array([SIDE_OF[k]  for k in y_pt_tr_on])  # 0/1/2

# Three heads:
head_valid : binary CE on (y_pt_tr != 0)        # uses ALL train rows
head_depth : 3-class CE, fit on (X_tr_on, y_depth_tr)   # ON-GRID SUBSET ONLY
head_side  : 3-class CE, fit on (X_tr_on, y_side_tr)    # ON-GRID SUBSET ONLY

# OOF reconstruction (inside the fold loop, on the FULL val set):
p_valid = head_valid.predict_proba(val_X)[:, 1]    # P(point != 0), shape (n_val,)
p_depth = head_depth.predict_proba(val_X)          # shape (n_val, 3)
p_side  = head_side .predict_proba(val_X)          # shape (n_val, 3)

oof_pt[val_idx, 0] = 1 - p_valid                   # P(point = 0)
for k in 1..9:
    d, s = DEPTH_OF[k], SIDE_OF[k]
    oof_pt[val_idx, k] = p_valid * p_depth[:, d] * p_side[:, s]
# Renormalise each row to sum to 1 (corrects for product-of-marginals approx error).
oof_pt[val_idx] /= oof_pt[val_idx].sum(axis=1, keepdims=True)
```

Notes:
- `head_valid` always uses **all** train rows (binary "off-grid vs on-grid" is meaningful
  for every row).
- `head_depth` and `head_side` see **only on-grid rows** at training time, but predict on
  the **full val set** at OOF time (their probs are then weighted by `p_valid`).
- Reconstruction happens **inside** the fold loop, not after — no cross-fold contamination.

Class mapping:
```
depth_of: {1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:2, 8:2, 9:2}
side_of : {1:0, 4:0, 7:0, 2:1, 5:1, 8:1, 3:2, 6:2, 9:2}
```

All other components (action model, server model, stacking) are **unchanged** from V14.

### Gates (revised 2026-05-05 with Codex cls0 gate)

| Gate | Pass | Fail action |
|---|---|---|
| OOF F1 on cls 1/2/3 (short) ≥ V14 cls 1/2/3 + 0.03 | Continue | Park; structural change ineffective |
| **OOF F1 on cls 0 (off-grid) ≥ V14 cls 0 F1 − 0.01** (NEW, Codex) | Continue | Park; the hierarchical separation can silently damage cls 0, must guard explicitly |
| OOF F1_p ≥ 0.235 | Continue | Park |
| Solo OOF OV ≥ V14 solo (0.3661) | Continue | Park |
| When swapped into the zoo, OOF improves | Submit (after P1/P2) | Keep as component for future blends only |

The cls0 gate is required because the hierarchical head explicitly factors `P(cls=0) =
1 − P(valid=1)`, decoupling off-grid prediction from on-grid placement. Short-class
gains can mask a cls0 regression that would tank the macro F1_p. The −0.01 tolerance
allows minor noise; anything worse signals the off-grid head is mis-calibrated.

### Risk

- Hierarchical models can compound errors. Soft-product reconstruction (vs. hard decode)
  mitigates this. The V12-era hard-decode failure does **not** apply to this design.
- Codex must verify reconstruction happens **inside** the CV loop (not post-hoc on
  cross-fold predictions, which would leak).
- **Do NOT softmax over the 9 reconstructed `oof_pt[k]` terms** (Codex 2026-05-05). The
  product `p_valid * p_depth[d] * p_side[s]` plus row renormalisation is the correct
  baseline; softmax would break the probability scale and inflate the on-grid mass.

### Cost (revised 2026-05-05)

Per-fold V16 cadence on this hardware is ~15-17 min (much faster than the spec's 36-40
min/fold budget). At V14-class architecture (P3 base), expect ~12-15 min/fold × 5 folds =
~60-75 min training. Plus ~30-50 min implementation. **Total ~1.5-2 h per full P3 run.**

### Codex review checklist for P3 (must sign off before implementation)

1. Confirm the on-grid SUBSET design (not `sample_weight=0`) is the right call given the
   class-weight schedule used in V14 (depth/side may need re-balanced sample weights even
   inside the subset — defer to Codex).
2. Confirm the soft reconstruction `oof_pt[k] = p_valid * p_depth[d] * p_side[s]` is the
   correct factorisation given that depth and side are conditionally dependent on
   on-grid (the product-of-marginals approximation may understate certain combinations).
   Renormalisation per row partially corrects, but Codex should review whether a softmax
   over the 9 reconstructed terms would be cleaner.
3. Confirm `head_valid` should be trained with class-balanced weights (off-grid rate is
   ~50%; binary cross-entropy without rebalancing is fine, but document the choice).
4. Confirm v18's solo OOF must clear 0.366 (V14 solo) AND F1_p ≥ 0.235 BEFORE plugging
   into the zoo. Don't blend a worse-than-V14 component just because it's structurally new.

---

## P6 — V11 + test-history augmentation (NEW, top priority for 2026-05-06 slot 1)

### Goal

Add the V16 test-history aug mechanism (which transferred well, gap −0.007 vs OOF) to the
V11 Transformer — the only structurally distinct model in the zoo (cross-cluster correlation
0.65–0.78 vs GBM family). Combine the LB-best technique with the most-orthogonal component.

### Codex sign-off (2026-05-05): 1-fold smoke APPROVED with HARD implementation constraint

`data/test_history_pairs.parquet` carries `serverGetPoint = -1` placeholders (per
`build_test_history_pairs.py`). `src/train_v11_transformer.py` currently computes
`server_head` BCE over **all** samples — feeding aug rows in unchanged would treat −1 as
a label and poison the SGP head.

**Required implementation fix (must be done before smoke):**
- Tag each sample with `is_aug ∈ {0, 1}`. Aug rows get `is_aug=1`.
- In the loss step: either
  (a) zero the server-head sample weight on `is_aug == 1` rows, OR
  (b) compute `F.binary_cross_entropy(server_logits[is_aug==0], y_server[is_aug==0])`
      restricted to non-aug rows.
- Action and point losses on aug rows are fine (the parquet has valid action/point labels).
- Verification: print `aug_rows_in_server_loss = 0` per epoch.

### Implementation steps

1. **(impl, ~30 min)** Edit `src/train_v11_transformer.py`:
   - Add `--aug-parquet` flag.
   - Extend `build_samples` to load aug parquet rows when flag is set; tag `is_aug=1`.
   - Add server-mask logic per Codex constraint above.
   - Save artifacts under tag `v11_aug` (mirror existing v11 naming).
2. **(smoke, ~30 min)** 1-fold smoke on GPU:
   ```
   python -u src/train_v11_transformer.py --aug-parquet data/test_history_pairs.parquet \
       --folds 1 --tag v11_aug_smoke 2>&1 | tee logs/v11_aug_smoke.log
   ```
3. **(full, ~90–120 min)** Full V11 training (existing 3-fold or 5-fold protocol):
   ```
   python -u src/train_v11_transformer.py --aug-parquet data/test_history_pairs.parquet \
       --tag v11_aug 2>&1 | tee logs/v11_aug.log
   ```
4. **(zoo re-run, ~50 min)** Replace v11 with v11_aug in zoo menu:
   ```
   python -u src/blend_zoo_v2.py --replace v11:v11_aug --max-models 5 \
       --prefix zoo_v6 --ranking-out submissions/zoo_v6_ranking.csv 2>&1 | tee logs/zoo_v6.log
   ```

### Gates

| Stage | Gate | Pass | Fail action |
|---|---|---|---|
| Smoke | aug_rows_in_server_loss == 0 verified | continue | fix mask, re-smoke |
| Smoke | 1-fold action F1 not regressed vs V11 baseline (within −0.005) | continue | abort, debug |
| Full | Solo action F1 ≥ V11 + 0.005 OR solo point F1 ≥ V11 + 0.005 | continue | park |
| Full | OOF correlation (v11_aug ↔ v16_testhist_aug on point) ≤ 0.78 | continue | park (no diversity gain) |
| Zoo | zoo_v6 top-1 OOF ≥ zoo_v2 top-1 OOF (0.3829) AND non-edge top-1 OOF improves vs zoo_v4a non-edge (0.3771) | candidate for slot 1 | hold |

### Risk

- V11's player embedding (`pid_self`/`pid_other`) may react badly to aug rows whose
  player IDs may not be in the train vocab. Mitigate: cap unknown IDs at the OOV index
  (V11 already does `min(pid, n_players-1)`).
- Aug rows are short-context (2..n test shots) — V11 may over-weight short-context style.
  Quick check: per-SN slice breakdown in v11_aug OOF vs v11.

### Cost: ~30 min impl + 30 min smoke + 90–120 min full + 50 min zoo = ~3.5 h total.

### Expected LB upside: +0.002 to +0.005

Combines the +0.0075 V16-aug effect (proven on GBM) with the most-uncorrelated component.

---

## P10 — Rally-pooled SGP head (was P4)

### Goal

Lift AUC from 0.61 toward 0.65+ by exploiting the rally-constant nature of SGP.

### Design

New script `src/train_v19_rally_srv.py`:
- Group rows by `rally_uid`.
- Per rally, compute pooled features:
  - mean / max / min / last of selected V9 features
  - rally length, score diff at end of history, last-shot action one-hot
- Train LGB+XGB (binary, AUC objective) on rally-level rows.
- Predict per-rally; broadcast SGP prob to all per-shot rows in that rally.
- Save artifacts: `oof_predictions/v19_rally_srv_oof_srv.npy`, `…_test_srv.npy`.
  (Do **not** overwrite action/point arrays — this model only contributes to the SGP channel.)

### Gates

| Gate | Pass | Fail action |
|---|---|---|
| OOF rally-AUC ≥ 0.65 in 1-fold smoke | Continue full 5-fold | Park |
| Full 5-fold OOF AUC ≥ V14 AUC (0.610) + 0.020 | Use as SGP channel in zoo | Park |
| When swapped into zoo, OOF OV ≥ zoo without it | Adopt | Park |

### Cost: ~2 h. Optional fork after P6 settles.

### Risk

- Existing per-shot AUC may already capture most of the rally-level signal; gain may be small.
- Cheap to test (1-fold smoke ≤ 30 min).

---

## P11 — Player-disjoint holdout diagnostic (NEW, validation infrastructure)

### Goal

Build a held-out fold whose primary players don't appear in any training fold. Compute
"player-disjoint OOF" alongside standard match-OOF. Use the gap to predict LB transfer
gap. The match-disjoint GroupKFold systematically over-credits player memorization (V15
proved this); a player-disjoint slice should bring OOF closer to LB behaviour.

### Codex sign-off (2026-05-05): APPROVED, but ADVISORY signal only initially

With ≤ 5 LB-tested points (V12+V11, V14+V11, V16+V11, zoo_v16_fast_01, zoo_v2, zoo_v3),
Pearson > 0.85 is dominated by single points and submissions are not independent. Initial
gate: leave-one-out / rank-consistency check — does the holdout correctly predict why
zoo_v2 won and zoo_v3 / per-SN bucket / V15 lost? Hard gate only after that lands.

### Implementation

1. New script `src/build_player_disjoint_holdout.py`:
   - Read train.csv, group by `match`, compute primary players (most frequent
     `gamePlayerId` and `gamePlayerOtherId` per match).
   - Greedy partition: assign matches to "holdout" until ~15% of distinct players are
     fully held out.
   - Save `data/player_holdout_idx.npy` (boolean mask over the 69712 OOF rows).
2. New script `src/eval_player_disjoint.py`:
   - Load any tag's `_oof_pt.npy`, `_oof_act.npy`, `_oof_srv.npy`, `_oof_y_*.npy`.
   - Compute OOF OV restricted to the holdout indices.
   - Emit a CSV: `player_disjoint_eval.csv` with columns `tag, full_OV, holdout_OV, gap`.
3. Apply to all 5 LB-tested submissions; compute Pearson(holdout_OV, LB) and per-pair
   rank consistency.

### Gates (advisory only initially)

| Gate | Pass | Fail action |
|---|---|---|
| Holdout exists with ≥ 5000 rows AND ≥ 50 distinct players | Continue | Re-balance partition |
| Rank consistency: zoo_v2_top1 holdout_OV > zoo_v3_top1 holdout_OV (matches LB) AND > V15_pp+V11 holdout_OV | Continue | Holdout signal weak; do NOT use for gating |
| Pearson(holdout_OV, LB) ≥ 0.6 across 5 LB points | Promote to advisory hard-gate | Keep as advisory only |

### Cost: ~30 min impl + ~30 min eval = ~1 h. Pure post-processing on existing OOFs.

### Risk: low. Doesn't lift OOF, just gates submissions. Worst case: weak signal, no harm.

---

## P12 — Anchor-perturbation zoo search (NEW)

### Goal

Restrict the zoo search to weight perturbations from the LB-tested winner (zoo_v2 top-1)
instead of fresh Dirichlet draws. Drastically narrows the search space, fights OOF
overfit pattern that produced zoo_v3.

### Implementation

Add `--anchor-from <ranking_csv>:<rank>` flag to `src/blend_zoo_v2.py`:
- Load anchor weights from the ranking row.
- Per task: sample `w_new = (1 - δ) * w_anchor + δ * Dirichlet(α=1, n)` with `δ ~ Uniform(0, 0.3)`.
- Reject candidates with `|w_new - w_anchor|_L1 > 0.4`.
- Add component menu only if a NEW component (e.g., v11_aug, v19_rally_srv) is present
  beyond the anchor's components.

### Gates

| Gate | Pass | Fail action |
|---|---|---|
| Anchor reproducibility: re-running with δ=0 reproduces anchor's OOF within 1e-4 | Continue | Debug |
| Top-1 OOF ≥ zoo_v2 top-1 OOF (0.3829) + 0.001 | Submission candidate | Park |
| Top-1 component drift |w_new - w_anchor|_L1 ≤ 0.2 | Submission-eligible | Reject; too far from validated anchor |
| Top-1 includes the new component | Eligible | Reject (no new info) |

### Cost: ~30 min impl + ~50 min CPU = ~1.5 h.

### Risk

- If the LB-best truly requires a structurally different blend, perturbation can't find it.
  Mitigation: only run after P6/P10 add a new component to the menu.

---

## P5 — Autoregressive sequence model (smoke only this round)

### Goal

Step change to 0.38+. **Do not commit to a full run before the smoke gate passes.**

### Smoke design

New script `src/train_v20_seq_smoke.py`:
- Causal Transformer, 6 layers, d_model=256, 4 heads.
- Input: per-shot token = sum of categorical embeddings (strikeId, handId, strengthId, spinId,
  positionId, actionId, pointId) + linear projection of continuous features + SN positional emb.
- LM objective: predict next-shot (actionId, pointId) at every position.
- Joint multi-task: action / point heads at every position; rally-pooled SGP head.
- Pretrain corpus: union of train + test rallies (test SGP is `-1`, **not** used as label).
- 1-fold only; ≤ 30 epochs; expected ~90 min.
- Aggregate OOF predictions only on the 69712 V14 evaluation positions.

### Smoke gates

| Gate | Pass | Fail action |
|---|---|---|
| Smoke solo OOF ≥ V14 solo (≈ 0.36) on the matched eval positions | Plan full 5-fold | **Abort. Do not proceed.** |
| Pearson correlation of smoke OOF action probs vs V11 action probs ≤ 0.95 | Plan full 5-fold (adds blend diversity) | Abort. Redundant with V11. |

If both gates pass: write up a separate `TRAIN_PLAN_P5.md` for the full run, get Codex sign-off.

### Cost (smoke only): ~90 min training + ~4 h implementation.

---

## Concrete plan for the next 12 hours (2026-05-06)

P3/v18 parked (gates failed); P1.5 closed (slot-3 skipped); v16_avg3 closed for direct
submission. Active priorities for this 12h window are P6 (v11_aug) → P11 (validation
diagnostic) → P10 (rally-SGP) → P12 (anchor-perturbation), with H7 distillation as a
stretch. 2026-05-06 slots: 3/3 available at the start of the window.

**User-set rules for this window (2026-05-06):**
- During the 12-hour window, no per-step permission required (user is AFK; CPU/RAM
  monopolised is acceptable). Just run.
- Compute estimate MUST be announced in chat immediately before each training kicks off
  so the user knows when the PC will be free.
- LB submissions still require user's manual upload (Codex/Claude do not submit).
- All state changes (gate outcomes, parked directions, completed runs) persist to
  RESULTS.md / STRATEGY.md / TRAIN_PLAN.md immediately, no permission required.

| Block | Step | Compute estimate | Submission slot? |
|---|---|---:|---|
| 0:00 – 0:30 | **(impl)** P6 server-mask in `src/train_v11_transformer.py` (`--aug-parquet`, `is_aug` tag, server-loss mask) | 0 (engineering) | — |
| 0:30 – 1:00 | **(train)** P6 v11_aug 1-fold smoke; verify `aug_rows_in_server_loss == 0` AND fold-1 action F1 not regressed vs V11 baseline | ~30 min | — |
| 1:00 – 3:00 | **(train)** P6 v11_aug full 3-fold; gate solo action F1 ≥ V11 + 0.005 AND OOF correlation v11_aug↔v16 ≤ 0.78 | ~90–120 min | — |
| 3:00 – 3:30 | **(impl)** P11 player-disjoint holdout build + eval script | 0 | — |
| 3:30 – 3:50 | **(eval)** Apply P11 to 5 LB-tested submissions; rank-consistency check (zoo_v2 > zoo_v3 > V15_pp) | ~10–20 min CPU | — |
| 3:50 – 4:50 | **(zoo)** Re-run `blend_zoo_v2.py --replace v11:v11_aug --max-models 5` | ~50–60 min CPU | — |
| 4:50 – 5:00 | **(eval)** Apply P11 to zoo top-1 candidate; gate decision | ~5 min | — |
| 5:00 – 5:15 | **DECISION POINT — slot 1**: submit if P6 zoo top-1 OOF ≥ 0.3829 + 0.001 AND interior temp AND P11 rank-consistency holds | 0 | **slot 1/3** if approved |
| 5:15 – 6:15 | **(impl)** P10 `src/train_v19_rally_srv.py` | 0 | — |
| 6:15 – 6:45 | **(train)** P10 1-fold smoke; gate rally-AUC ≥ 0.65 | ~30 min | — |
| 6:45 – 8:45 | **(train)** P10 full 5-fold (only if smoke gate passes; else skip to step 9) | ~90–120 min | — |
| 8:45 – 9:45 | **(zoo)** Re-run zoo with v11_aug + v19_rally_srv (server channel) | ~50–60 min CPU | — |
| 9:45 – 10:00 | **DECISION POINT — slot 2**: submit if zoo with v19_rally_srv beats slot-1 candidate AND clears P11 | 0 | **slot 2/3** if approved |
| 10:00 – 10:30 | **(impl)** P12 `--anchor-from` flag in `blend_zoo_v2.py` | 0 | — |
| 10:30 – 11:30 | **(train)** P12 anchor-perturbation zoo run | ~50–60 min CPU | — |
| 11:30 – 11:45 | **DECISION POINT — slot 3**: submit if P12 produces structurally distinct candidate that clears all gates | 0 | **slot 3/3** if approved |
| 11:45 – 12:00 | (Stretch buffer) — H7 distillation impl + smoke if any of P6/P10/P12 ran short | varies | — |

Total compute budget: ~7–8 h (with ~4 h slack for debugging / re-runs). The slack covers
the P10-skip case (saves ~2 h) and any failed-gate retries.

**Submission gates for ALL slots this window:**
- Zoo top-1 OOF ≥ zoo_v2 top-1 OOF (0.3829) + 0.001
- Interior temperature (`temp_at_edge == False`)
- P11 rank-consistency check holds for the new candidate
- Otherwise SKIP that slot — current best 0.3733788 is hard to beat by chance.

---

## Top 3 candidates for 2026-05-06 (revised)

Ranked by expected probability of beating LB 0.3733788, conditional on the above plan
executing as scheduled.

### #1 — v11_aug zoo top-1 (P6 + P12)

- File (will be): `submissions/submission_zoo_v6_<elig_or_anchored>_v11aug_<provenance>.csv`
- Components: zoo_v2 menu with `v11` replaced by `v11_aug`; size ≤ 5; temp grid ≥ 0.3;
  edge-rejection on; P12 anchor-perturbation if v11_aug-only candidate doesn't clear gate.
- Expected OOF: 0.385 – 0.388 if v11_aug solo lifts by +0.005 over V11.
- Expected LB: ~0.374 – 0.378 if gap matches zoo_v2 top-1's −0.0095.
- Why it beats current best: V11 is the most uncorrelated component (cross-cluster 0.65–0.78
  with GBM family). Adding the V16-aug mechanism to V11 should produce real diversity gain
  without LB-transfer risk. This is the FIRST genuinely new component since V16.
- Confidence: **medium-high**, conditional on P6 server-mask correctness and gate clearance.

### #2 — Anchor-perturbation zoo around zoo_v2 top-1 (P12 fallback)

- File (will be): `submissions/submission_zoo_v7_anchor_<provenance>.csv`
- Components: zoo_v2 top-1 + perturbed weights (drift ≤ 0.2 L1) + (optional) v11_aug or
  v19_rally_srv if available.
- Expected OOF: 0.383 – 0.385.
- Expected LB: ~0.374 – 0.376.
- Why it beats current best: tightens around the LB-tested winner. Lower upside than #1
  but lower variance.
- Confidence: **medium**, conditional on perturbation finding a top-1 with OOF + 0.001
  improvement.

### #3 — Hold zoo_v2 top-1 (no submit) — defensive fallback

- Already at LB 0.3733788. If #1 and #2 both fail gates, **DO NOT submit anything** for
  slot 1. Preserve slots for late-day candidates from P10 (rally-SGP) or a re-tried P6.
- The candidate space has nothing safer than the current best.

---

## Hard rules (carried + added 2026-05-05)

1. No `serverGetPoint` from test.csv anywhere in features, training labels, or supervision.
2. No SGP-derived player win-rate.
3. No raw player profile (`player_action_freq`, `opp_action_freq`, per-player ID stats).
4. No `hist_action_freq` / `hist_point_freq` / `streak_*`.
5. No hard per-SN-bucket weight conditioning in blend search.
6. No CatBoost in any final blend.
7. Validation: `GroupKFold(n_splits=5)` by **match**.
8. OOF artifacts saved as individual `.npy` files with the established naming pattern
   (`{tag}_oof_act.npy`, `_pt.npy`, `_srv.npy`, `_mask.npy`, `_y_act.npy`, `_y_pt.npy`,
   `_y_srv.npy`, `_nsn.npy`, `_test_act.npy`, `_test_pt.npy`, `_test_srv.npy`,
   `_test_rally_uid.npy`).
9. **No heavy training before Codex sign-off on this plan and STRATEGY.md.**
10. **NEW (Locked Rule 8 in STRATEGY)**: blend size ≤ 5 unless LB-validated.
11. **NEW (Locked Rule 9 in STRATEGY)**: THR temperatures hitting the grid lower bound are
    suspect; widen grid (down to 0.3) and prefer interior-temperature top candidates.
12. **NEW (Locked Rule 10 in STRATEGY)**: `v16_avg3` is provisionally suspect as a zoo
    component until a controlled probe (size ≤ 5, interior temperature) confirms transfer.

---

## Codex review (resolved 2026-05-04, partial 2026-05-05)

| # | Issue | Status |
|---|---|---|
| 1 | P1 zoo search blocks `v11plus` and caps blend at 5 | ✅ Fixed: Group D selection rule rewritten; max blend size 6 (now revised to 5 in P1.5) |
| 2 | `final_blend_optimized.py` is only a 2-model blender | ✅ Fixed: `blend_zoo_v2.py` spec is now a purpose-built N-way blender (UID alignment, action-dim padding, independent per-task weight vectors) |
| 3 | V16 `--seed` plumbing missing | ✅ DONE: 9 model-init sites + `np.random.seed(seed)`; flip-aug and GroupKFold deterministic |
| 4 | P3 depth/side heads must use on-grid SUBSET (not sample_weight=0) | ✅ Fixed: code skeleton updated to subset `X[point != 0]` |
| 5 (2026-05-05) | zoo_v3 LB regression −0.0058 with v16_avg3 + n=6 | ✅ Acknowledged in P1.5 framing: P1.5 is OOF-only diagnostic, cannot prove LB transfer |
| 6 (2026-05-05) | THR temperatures at grid edge t=0.5 in both zoo_v2 and zoo_v3 top-1 | ✅ Resolved: `--temp-min 0.3` + `temp_at_edge` flag + eligible-only file materialization |
| 7 (2026-05-05) | P1.5 Run B cannot prove v16_avg3 transfer, must not rehabilitate for submission | ✅ Fixed: Run B explicitly framed as OOF-only diagnostic in §P1.5 |
| 8 (2026-05-05) | Edge-rejection must affect file materialization (not just annotate the rank) | ✅ Fixed: `--edge-cushion`, `eligible_rank` column, eligible-only top-K materialization documented in §P1.5 |
| 9 (2026-05-05) | P3 missing cls0 F1 regression gate | ✅ Fixed: cls0 gate (≥ V14 cls0 F1 − 0.01) added to §P3 Gates |
| 10 (2026-05-05 deep memo review) | H6 V11+test-history aug — server head computes BCE on aug rows where SGP=−1, would poison SGP head | ✅ Captured in §P6: server-mask required (zero sample weight on `is_aug==1` rows OR restrict BCE to non-aug). Smoke gate verifies `aug_rows_in_server_loss == 0` |
| 11 (2026-05-05 deep memo review) | H8 pseudo-label test rallies — distinct from V16 aug (organiser-confirmed history vs model-generated targets); rule-sensitive | ✅ Captured: H8 NOT approved for submission training; offline design only; explicit Jabir policy approval required before any submission training |
| 12 (2026-05-05 deep memo review) | H9 (was H11 in memo) player-disjoint holdout — Pearson > 0.85 over ≤5 LB points is dominated by single points | ✅ Captured in §P11: advisory signal initially; first gate is leave-one-out / rank-consistency (zoo_v2 > zoo_v3 > V15); hard gate only after that lands |
| 13 (2026-05-05 deep memo review) | H4 geometry smoothing — GBM multiclass needs sample expansion or custom objective, not just loss tweak | ✅ Captured in STRATEGY H9: difficulty raised from "low" to "medium" |
| 14 (2026-05-05 deep memo review) | H11 flip-TTA — must rebuild flipped raw context features and flip posteriors back, not just relabel | ✅ Captured in STRATEGY (deferred ideas list): difficulty raised from "trivial" to "low-medium" |

### Codex sign-off summary (2026-05-05)

**Direction:** Codex agrees with promoting zoo_v2 to current best, downgrading
zoo_v3/v16_avg3 to suspect, and pivoting to P3.

- **P1.5**: APPROVED for implementation, conditional on the three fixes above.
- **P3**: APPROVED to start `train_v18_hier_point.py` with the cls0 gate.
- **Today's slot 3**: NOT recommended for P1.5 unless Run A produces a non-edge,
  size ≤ 5, OOF clearly > 0.3839 candidate AND the submission CSV passes Codex artifact
  review. Default = preserve slot.

**P3 technical guidance (Codex 2026-05-05):**
- On-grid SUBSET is the correct choice; do NOT use `sample_weight=0` placeholders.
- Reconstruction `oof_pt[k] = p_valid * p_depth[d] * p_side[s] + row renorm` is the
  correct baseline. Do NOT softmax over the 9 reconstructed terms — softmax breaks the
  probability scale.
- Possible future upgrade (deferred past P3): model `P(side | depth)` instead of the
  product of marginals. Out of scope for v18.

---

## Status tracker

- [x] STRATEGY.md updated (zoo_v2 LB win, P2 LB regression, locked rules 8–10 added; deep memo + Codex review reflected)
- [x] TRAIN_PLAN.md updated (this file; new §P6 / §P10 / §P11 / §P12)
- [x] RESULTS.md updated through §12 (v18 hier failure + Codex review of memo + correlation matrix)
- [x] `src/blend_zoo_v2.py` implemented and LB-validated
- [x] P1 zoo v2 search executed (LB 0.3733788)
- [x] `train_v16_testhist_aug.py --seed` flag added
- [x] V16 seed1 smoke test
- [x] V16 seed1 full 5-fold (opt OV 0.3667)
- [x] V16 seed2 full 5-fold (opt OV 0.3674)
- [x] `v16_avg3` artifact built (averaged base OV 0.3597)
- [x] zoo_v3 with v16_avg3 swapped (top-1 OOF 0.3839, LB 0.3675453 — REGRESSION)
- [x] P1.5 implementation: `--max-models`, `--temp-min`, edge-rejection annotation
- [x] P1.5 Run A (zoo_v4a): 198/396 eligible; non-edge top-1 OOF 0.3771; slot-3 SKIPPED
- [ ] P1.5 Run B (v16_avg3, size ≤ 5, temp ≥ 0.3) — OOF-only diagnostic; not run (would not unblock submission per Codex; deferred)
- [x] `train_v18_hier_point.py` implemented (Codex-approved design)
- [x] P3 hierarchical point full 5-fold (v18) — gates failed (cls0 −0.0172, short −0.0392); v18 PARKED
- [ ] **P6 V11 + test-history aug** (NEW, top priority): server-mask impl, smoke, full, zoo re-run
- [ ] P10 rally-pooled SGP head (was P4): smoke ≥0.65 rally-AUC
- [ ] P11 player-disjoint holdout (NEW): impl + apply to 5 LB-tested submissions for rank-consistency check
- [ ] P12 anchor-perturbation zoo search (NEW): impl + run after P6/P10 land a new component
- [ ] P7 GBM/zoo distillation into V11 (NEW): pending P6 outcome
- [ ] P9 geometry-aware point loss (NEW): difficulty raised; needs sample expansion or custom objective
- [ ] P8 pseudo-labelled test rallies (NEW): NOT approved for submission — explicit Jabir policy required first
- [ ] P5 autoregressive smoke (deferred — cost > remaining budget this round)
