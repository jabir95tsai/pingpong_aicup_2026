# TRAIN_PLAN
## Round: 2026-05-04 — V16 + zoo-blend backbone (post-LB sync)

This plan supersedes prior V14 / V15 / V11+ plans. All V15 lines and per-SN bucket gating
plans are closed. Current best is `zoo_v16_fast_01` at LB **0.3694863**; future submissions
must aim to beat it.

**Codex review status (2026-05-04): conditional sign-off.** Four issues raised, all
resolved in this revision:

1. ✅ P1 Group-D constraint contradiction → fixed (v11plus now selectable; max blend size 6).
2. ✅ P1 `final_blend_optimized.py` is 2-model only → spec rewritten as purpose-built N-way
   blender (UID alignment, action-dim padding, independent per-task weight vectors).
3. ✅ P2 V16 `--seed` plumbing missing → confirmed; patch must precede any V16 seed run.
4. ✅ P3 depth/side heads must use on-grid SUBSET, not `sample_weight=0` → fixed.

Codex execution order: **P1 → patch V16 `--seed` → P2 → P3 → P4 → P5**. P3 stays parked
until P1 lands.

---

## Status snapshot

| Component | Status | Notes |
|---|---|---|
| `submission_zoo_v16_fast_01_…csv` | ✅ Current best, LB 0.3694863 | 4-model global blend |
| `submission_v16_testhist_aug_v11_optblend.csv` | ✅ Backup, LB 0.3673269 | Single-family backbone |
| `submission_v14_5f_nocb_v11_optblend.csv` | ✅ Deep fallback, LB 0.3598509 | Stable transfer |
| `train_v16_testhist_aug.py --seed` | ❌ Not implemented | Required for P2 |
| `src/avg_oof.py` | ✅ Works for V14 seeds | Reusable for V16 seeds (tag-agnostic) |
| Zoo blend search script | ❌ Not in repo | Must build for P1 (new file `src/blend_zoo_v2.py`) |
| `train_v18_hier_point.py` | ❌ Not implemented | Required for P3 |

---

## Priority Order

| ID | Title | Cost | Risk | Submission slot? |
|---|---|---|---|---|
| **P0** | Hold and protect current best | 0 | — | None — slots exhausted today |
| **P1** | Blend Zoo v2 broader search | 30–60 min CPU | Low | Yes, top candidate |
| **P2** | V16 multi-seed (`v16_seed1`, `v16_seed2`, `v16_avg3`) | ~6 h | Medium | Yes, after P1 |
| **P3** | Hierarchical point head (soft-decoded, on-grid SUBSET) | ~3.5 h | Medium | **Hold** until P1 lands; design now Codex-approved |
| **P4** | Rally-level Server head | ~2 h | Low–med | Indirect — feeds zoo |
| **P5** | Autoregressive sequence model (smoke first) | 1.5 h smoke / 8–10 h full | High | Deferred until P2 settles |

---

## P0 — Protect current best

Rules (no training; pure discipline):

- **Do not** submit any candidate whose expected LB < 0.3694863 + 0.001 cushion.
- **Do not** delete or move the current-best submission file.
- **Do not** mutate any artifact under `oof_predictions/` for tags listed in §"Component menu" below.
- Today's (2026-05-04) submission slots are **exhausted**. Next submission ≥ 2026-05-05.

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

## P1 — Blend Zoo v2

### Goal

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

## P2 — V16-centered seed/ensemble

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

## P3 — Hierarchical point head (soft-decoded)

### Goal

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

### Gates

| Gate | Pass | Fail action |
|---|---|---|
| OOF F1 on cls 1/2/3 (short) ≥ V14 cls 1/2/3 + 0.03 | Continue | Park; structural change ineffective |
| OOF F1_p ≥ 0.235 | Continue | Park |
| Solo OOF OV ≥ V14 solo (0.3661) | Continue | Park |
| When swapped into the zoo, OOF improves | Submit (after P1/P2) | Keep as component for future blends only |

### Risk

- Hierarchical models can compound errors. Soft-product reconstruction (vs. hard decode)
  mitigates this. The V12-era hard-decode failure does **not** apply to this design.
- Codex must verify reconstruction happens **inside** the CV loop (not post-hoc on
  cross-fold predictions, which would leak).

### Cost: ~3.5 h training + ~30 min implementation. One full 5-fold run.

---

## P4 — Rally-level Server head

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

### Cost: ~2 h. Optional fork after P1/P2 settle.

### Risk

- Existing per-shot AUC may already capture most of the rally-level signal; gain may be small.
- Cheap to test (1-fold smoke ≤ 30 min).

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

## Concrete plan for the next 6 hours

Assumes the next 6 hours are CPU-budget-constrained (single CPU box). Two heavy V16-seed runs
need ~6 h alone. Ordering prioritises P0/P1 first (zero training risk), then P2 (high LB upside)
in parallel with P3 implementation.

| Hour | Track A (CPU) | Track B (engineering, no GPU/CPU) |
|---|---|---|
| 0:00 – 0:30 | Idle | Codex reviews STRATEGY.md and TRAIN_PLAN.md |
| 0:30 – 1:30 | Implement & run `src/blend_zoo_v2.py` | Sketch `train_v18_hier_point.py` skeleton |
| 1:30 – 2:00 | Inspect P1 zoo v2 results (rank by spread-penalised score) | Codex reviews `blend_zoo_v2.py` output |
| 2:00 – 2:15 | — | Add `--seed` flag to `train_v16_testhist_aug.py`; Codex audits the flip-aug RNG |
| 2:15 – 3:00 | P2 Step 2: smoke `v16_seed1_smoke` (≤45 min) | Implement `train_v18_hier_point.py` |
| 3:00 – 6:00 | P2 Step 3a: `v16_seed1` full 5-fold (~180 min) | Codex reviews `train_v18_hier_point.py`; prepare `train_v19_rally_srv.py` skeleton |

After this 6-hour window:
- We have 1 zoo v2 ranking (best candidate identified, not submitted).
- We have `v16_seed1` complete (1 of 2 needed for `v16_avg3`).
- `train_v18_hier_point.py` is implementation-complete and ready to launch.
- `train_v19_rally_srv.py` is sketched.

The remaining `v16_seed2` (~180 min) and the P3 hierarchical-point full run (~210 min) are
sequential next-steps; one of them runs overnight.

**No LB submission happens inside this 6-hour window.** Slots are exhausted for 2026-05-04.
The first 2026-05-05 submission goes to whichever of {P1 best zoo, P2 v16_avg3 zoo} has the
highest spread-penalised OOF score by then.

---

## Top 3 candidates for next submission day (2026-05-05)

Ranked by expected probability of beating LB 0.3694863, conditional on the above plan
executing as scheduled.

### #1 — Best zoo v2 candidate (P1)

- File: `submissions/submission_zoo_v2_top1_<provenance>.csv`
- Components: 3–6 from {v16_testhist_aug, v14_avg3, v14_seed*, v12_5f, v11, v11plus}, no per-SN gating.
- Expected OOF: 0.380 – 0.382.
- Why it beats current best: same backbone family as zoo_v16_fast_01 but with calibration-aware
  variants (TEMP / CW) which historically transfer with smaller OOF→LB gap. Submission risk is
  bounded by the spread-penalised gate.
- Confidence: **medium-high** — the zoo v1 search was narrow, so an honest broader search
  almost always finds a small improvement.

### #2 — `v16_avg3 + v14_avg3 + v12_5f + v11` (P2 follow-up)

- File: `submissions/submission_zoo_v2_v16avg3_<rank>.csv`
- Components: V16 multi-seed average instead of single-seed V16, plugged into the same zoo
  structure that produced current best.
- Expected OOF: 0.380 – 0.383.
- Why it beats current best: variance reduction on the V16 channel; matches the same
  structural blend that already won on LB. Most direct extension of the proven recipe.
- Confidence: **medium**, conditional on V16 seed1/seed2 actually diverging from seed0
  (smoke test gate).

### #3 — `v14_avg3 + V11` standalone fallback (already complete)

- File: `submissions/submission_v14_avg3_v11_optblend.csv` (already exists, OOF 0.3765)
- Components: 3-seed V14 average + V11.
- Expected LB: ≈ 0.36 – 0.365 (V14 family LB regime).
- Why submit: defensive fallback if both #1 and #2 fail their gates; it's the strongest
  V14-family candidate not yet submitted. Likely **does not beat current best** but bounds
  the worst case for the day.
- Confidence: **low** — submit only as the day's third slot if the first two are needed for
  zoo iteration probes.

---

## Hard rules (carried)

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

---

## Codex review (resolved 2026-05-04)

| # | Issue | Status |
|---|---|---|
| 1 | P1 zoo search blocks `v11plus` and caps blend at 5 | ✅ Fixed: Group D selection rule rewritten; max blend size 6 |
| 2 | `final_blend_optimized.py` is only a 2-model blender | ✅ Fixed: `blend_zoo_v2.py` spec is now a purpose-built N-way blender (UID alignment, action-dim padding, independent per-task weight vectors) |
| 3 | V16 `--seed` plumbing missing | ✅ Acknowledged: P2 Step 1 explicitly required before any V16 seed run |
| 4 | P3 depth/side heads must use on-grid SUBSET (not sample_weight=0) | ✅ Fixed: code skeleton updated to subset `X[point != 0]` |

Open items still requiring Codex sign-off:

1. **STRATEGY OOF gate revision**: should the V14-era OOF gate (≥ 0.3764) be replaced with a
   per-family or multi-signal gate? V16+V11 failed the gate but won on LB.
2. **Zoo search overfit guard**: confirm the `score = OOF_OV − 0.5 × max(0, spread − ref)`
   penalty is appropriate; suggest alternative if not.
3. **V16 `--seed` plumbing audit**: when the patch lands, confirm the flip-aug RNG path is
   covered (in addition to LGB/XGB random_state).
4. **CPU contention**: confirm that running zoo v2 search in parallel with v16_seed1 training
   in the 0:30–1:30 hour window is acceptable.

---

## Status tracker

- [x] STRATEGY.md updated (V16 + zoo backbone, V15 family closed, per-SN bucket closed)
- [x] TRAIN_PLAN.md updated (this file)
- [ ] Codex review of STRATEGY.md and TRAIN_PLAN.md
- [ ] `src/blend_zoo_v2.py` implemented
- [ ] P1 zoo v2 search executed
- [ ] `train_v16_testhist_aug.py --seed` flag added
- [ ] V16 seed1 smoke test
- [ ] V16 seed1 full 5-fold
- [ ] V16 seed2 full 5-fold
- [ ] `v16_avg3` artifact built
- [ ] `train_v18_hier_point.py` implemented
- [ ] P3 hierarchical point full 5-fold
- [ ] P4 rally-level server head smoke
- [ ] P5 autoregressive smoke (deferred until P2 settles)
