# NEXT_PLAN
## Round: 2026-05-04 — V16 success, V16-aware ensemble next

---

## Status snapshot

- Current best: `submission_zoo_v16_fast_01_v16_v14_seed1_v12_5f_v11.csv`, **LB = 0.3694863**
- P1 V15 ablation: ✅ closed; player profile / hist / streak permanently excluded
- New official clarification (organizer, 2026-05-03):
  - Test.csv prior-shot `actionId` / `pointId` within the same rally are **not leakage** —
    they are observable history at inference time.
  - Test.csv `serverGetPoint` is **still excluded** from features and training.

P2.0 has been submitted and is now the new best. This means test-history augmentation transfers better to Public LB than match-GroupKFold OOF predicted.

---

## Priority ranking

| Rank | ID | Title | Risk | Expected gain (OOF) | Runtime |
|---|---|---|---|---|---|
| 1 | Blend Zoo | Continue V16-centered multi-model/per-SN blends | low | already +0.00216 LB | no training |
| 2 | P2.1 | Finish V14 avg3 as diagnostic / possible blend component | low | +0.001…+0.003 vs V14, may help V16 blend | 1 remaining seed + blend |
| 3 | P2.2 | Heterogeneous clean ensemble | medium (CB calibration) | +0.001…+0.005 | 2–3× ~90–120 min |
| 4 | P3 | Architecture breakthrough | high | step change to 0.38+ | weeks |

Rule: P2.0 is run first IF Codex approves the augmentation design. If Codex flags a leakage or
implementation risk, fall through to P2.1 as the conservative default.

---

## P2.0 — V16 test-history augmentation

### Goal

Use known test-rally history (`actionId` / `pointId` of shots before the target shot in the same
test rally) as additional supervised training pairs for the action and point models.

### Augmentation source (Codex pre-counts)

| Quantity | Count |
|---|---|
| Test rows | 3,589 |
| Test rallies | 1,236 |
| Augmented action+point pairs (history shots `< k` predicting known shot `k`) | 2,353 |
| Relative increase vs current train target rows | +3.38% |

Per-SN distribution of augmented rows:
- SN=2: 838
- SN=3: 541
- SN=4: 335
- SN=5+: 639

Rare-point augmentation (vs current train counts):
- pointId=1 (FH_short): +41 / 582 (+7.0%)
- pointId=2 (mid_short): +117 / 1920 (+6.1%)
- pointId=3 (BH_short): +12 / 203 (+5.9%)

The early-SN and rare-point coverage is exactly where V14 currently underperforms (SN=2 OV=0.2696,
BH_short F1=0.0000). This is the strongest theoretical case for augmentation.

### Strict rules

1. **Pair generation**: for each test rally with shots `s_1 … s_n`, build pairs
   `(history shots < k, target = (actionId_k, pointId_k))` for `k = 2 … n`. Only `actionId` and
   `pointId` of `k` are used as supervision; **never** the model's target shot 
` answer.
2. **serverGetPoint guard** (Codex-approved contract):
   - The raw augmentation source must never source true test `serverGetPoint`.
   - Before building aug features, set `test_aug_raw["serverGetPoint"] = -1` as a dummy placeholder
     if the feature builder (`build_features_v9`) requires the column when `is_train=True`.
   - Immediately drop / ignore `y_serverGetPoint` from all augmented feature rows after building.
   - Assert all augmented rows have `is_aug == 1`.
   - Assert no augmented rows enter server model training, AUC logging, OOF, threshold opt, or
     `y_s` arrays.
   - Assert that if `serverGetPoint` exists in the aug raw copy, **all values are exactly −1**.
   - Log: `NO_TRUE_TEST_SGP_USED = True`.
   - **Do NOT use `sample_weight=0` for server — full data-level exclusion only.**
3. **global_stats fold-safety**:
   - Per fold: `fold_stats = compute_global_stats_v9(tr_raw)` uses real train fold rows **only**.
   - Do NOT include test-history augmented rows in `compute_global_stats_v9`.
   - Build augmented features using the fold_stats computed from real train fold rows.
   - Append augmented feature rows only to action/point model train matrices.
   - Validation features and OOF must remain real-train validation rows only.
   - Rationale: keeping fold_stats train-only ensures OOF remains comparable to V14 and prevents
     test-history labels from leaking into validation-side prior features.
4. **Validation isolation**: test-augmented rows **never** enter OOF validation. They are appended
   only to the `train` side of each fold's split.
5. **Match-disjoint folds**: real train rows continue to use `GroupKFold(n_splits=5)` by `match`.
6. **Feature parity**: build augmented features with fold_stats from real train fold + `features_v9`
   pipeline (no test SGP, no player profile).
7. **Reproducibility**: pair-generation is deterministic; output the augmented row count, per-SN
   breakdown, and per-class label histogram in training logs.
8. **Assertions** (must appear in training logs):
   - `assert aug_rows == 2353`
   - `assert oof_mask.sum() == 69712`
   - `assert (aug_raw["serverGetPoint"] == -1).all()`
   - `assert NO_TRUE_TEST_SGP_USED == True`

### Tag and artifacts

- Training script: `src/train_v16_testhist_aug.py` (new)
- Tag: `v16_testhist_aug`
- OOF artifacts (`.npy`, not `.npz`):
  - `oof_predictions/v16_testhist_aug_oof_act.npy`
  - `oof_predictions/v16_testhist_aug_oof_pt.npy`
  - `oof_predictions/v16_testhist_aug_oof_srv.npy`
  - `oof_predictions/v16_testhist_aug_oof_mask.npy`
  - `oof_predictions/v16_testhist_aug_oof_y_act.npy`
  - `oof_predictions/v16_testhist_aug_oof_y_pt.npy`
  - `oof_predictions/v16_testhist_aug_oof_y_srv.npy`
  - `oof_predictions/v16_testhist_aug_oof_nsn.npy`
- Test artifacts (`.npy`):
  - `oof_predictions/v16_testhist_aug_test_act.npy`
  - `oof_predictions/v16_testhist_aug_test_pt.npy`
  - `oof_predictions/v16_testhist_aug_test_srv.npy`
- Solo submission (sanity): `submissions/submission_v16_testhist_aug.csv`
- Blend submission (final candidate): `submissions/submission_v16_testhist_aug_v11_optblend.csv`

### Launch sequence (Codex-approved)

```
# Step 1 — Build augmented pair index (deterministic, one-shot)
python src/build_test_history_pairs.py \
    --test data/test.csv \
    --out data/test_history_pairs.parquet

# Step 2 — Smoke test ONLY (1-fold, fast)
python src/train_v16_testhist_aug.py \
    --aug data/test_history_pairs.parquet \
    --folds 1 --n-boost 200 --es 30 --skip-cb \
    --tag v16_testhist_aug_smoke

# Step 3 — Pause: review smoke assertions and OV before proceeding.
# If assertions pass and smoke OV ≥ V14 smoke baseline, proceed to full:

# Step 4 — Full 5-fold (only after smoke review)
python src/train_v16_testhist_aug.py \
    --aug data/test_history_pairs.parquet \
    --folds 5 --skip-cb --tag v16_testhist_aug

# Step 5 — Blend with V11
python src/blend_ensemble.py \
    --v1 v16_testhist_aug --aux-tag v11 \
    --out submission_v16_testhist_aug_v11_optblend.csv
```

Smoke report must include (before full-fold launch approval):
- `aug_rows == 2353` assertion result
- `oof_mask.sum() == 69712` assertion result
- `NO_TRUE_TEST_SGP_USED = True` log line
- SGP exclusion confirmation (no aug rows in server path)
- Fold-1 solo OV vs V14 fold-1 baseline
- NaN/inf check on feature matrix and predictions

### Success / failure gates

| Gate | Pass | Fail action |
|---|---|---|
| OOF solo opt ≥ V14 solo opt (≈ **0.3661**) | proceed to blend | discard |
| Historical V16 OOF gate | failed locally, but Public LB succeeded | revise future gates with caution |
| OOF−LB gap ≤ 0.020 | adopt as new best if LB > 0.360 | analyse leakage; do not reuse |
| Per-SN F1 on SN=2 increases vs V14 baseline | strong evidence augmentation works | weak signal, keep V14+V11 |

Note: the solo gate (0.3661) is the V14 threshold-optimised solo OOF, **before** V11 blend.
The blend gate (0.3754) is V14+V11 final. A V16 solo below 0.3661 does not proceed to blending.

### Risks / open questions for Codex

1. Augmentation may introduce label noise if test history is heavily biased toward common classes
   (skews toward rare classes per count — favourable, but verify per-class histogram in logs).
2. Server-model exclusion implementation: augmented rows must be silently dropped before
   the SGP training slice; confirm the training script splits cleanly on an `is_aug` flag.
3. Class-weight interaction: `ACTION_CW` / `POINT_CW` tuned for V14 row distribution.
   Adding ~3.4% rare-skewed rows may shift the optimum slightly.
4. Test rallies are de-identified — augmented rows with unknown `gamePlayerId` must not break
   `features_v9` (confirm unknown-player fallback path exists in feature builder).

---

## P2.1 — Multi-seed V14

### Goal

Reduce LGB+XGB seed variance by averaging probabilities across 3 seed-randomised V14 runs.

### Rules

- Same `features_v9` pipeline as V14 (no player profile, no hist/streak).
- 3 seeds: 42 (existing V14), 48879 (0xBEEF), 51966 (0xCAFE) — fixed for reproducibility.
- Each seed runs full 5-fold `--skip-cb`.
- Average predicted probabilities (not argmaxed labels), then apply existing threshold-opt routine
  on the averaged OOF.
- Final blend with V11 uses the averaged V14 OOF.

### Tag and artifacts

- Training script: `src/train_v14.py` with `--seed` flag.
  **Codex: verify `--seed` is wired to both LGB `random_state`, XGB `seed`, and numpy/data shuffle
  before P2.1 is launched. Add the flag if absent.**
- Tags: `v14_seed0`, `v14_seed1`, `v14_seed2`
- Per-seed OOF artifacts (`.npy`, not `.npz`):
  - `oof_predictions/v14_seed{N}_oof_act.npy`, `_pt.npy`, `_srv.npy`, `_mask.npy`, `_y_act.npy`,
    `_y_pt.npy`, `_y_srv.npy`, `_nsn.npy`
  - `oof_predictions/v14_seed{N}_test_act.npy`, `_pt.npy`, `_srv.npy`
- Average artifact: `oof_predictions/v14_avg3_oof_act.npy`, etc. (same schema, averaged)
- Submission: `submissions/submission_v14_avg3_v11_optblend.csv`

### Concrete commands (DRAFT)

```
# Verify --seed flag first:
python src/train_v14.py --help | grep seed

# Three seed runs
python src/train_v14.py --folds 5 --skip-cb --tag v14_seed0 --seed 42
python src/train_v14.py --folds 5 --skip-cb --tag v14_seed1 --seed 48879
python src/train_v14.py --folds 5 --skip-cb --tag v14_seed2 --seed 51966

# Average probability arrays (not argmax)
python src/avg_oof.py --tags v14_seed0 v14_seed1 v14_seed2 --out v14_avg3

# Blend with V11 (reads .npy artifacts)
python src/blend_ensemble.py \
    --v1 v14_avg3 --aux-tag v11 \
    --out submission_v14_avg3_v11_optblend.csv
```

### Success / failure gates

| Gate | Pass | Fail action |
|---|---|---|
| OOF solo opt (averaged) ≥ V14 solo opt (≈ **0.3661**) | proceed to blend | discard, investigate seed-sensitivity |
| OOF (avg+V11) ≥ V14+V11 (**0.3754**) + 0.0015 | submit to LB | hold and run P2.2 |
| OOF−LB gap ≤ 0.018 | adopt as new best if LB > 0.361 | mark as variance-bound, escalate to P3 |

Note: averaging 3 seeds should not meaningfully change solo OOF vs a single seed; the main benefit
appears in reduced LB variance. Solo gate at 0.3661 (V14 level) is effectively a sanity check
that no seed diverged pathologically.

### Risks

- Cost: 3× runtime; each seed run ≈ 120 min on CPU-only setup → ~6 hours total.
- Diminishing returns: variance reduction tops out around 3 seeds for LGB+XGB; 5 seeds rarely beats 3.
- LB transfer: should be near-1.0 (no player features), so OOF gain should translate.

---

## P2.2 — Heterogeneous clean ensemble

### Goal

Add model diversity beyond LGB+XGB without leakage. Three diversity sources:

1. **LGB-only on V9**: drop XGB, retune LGB hyperparams (different bias/variance trade-off).
2. **XGB-only on V9**: drop LGB, slightly different inductive biases.
3. **Conservative CatBoost on V9**: shallow CB with no leaf-wise growth, gated against V14
   OOF-LB calibration before blending.

Then OOF-blend the four bases (LGB, XGB, LGB-only retuned, XGB-only retuned, optional CB)
and final blend with V11.

### Rules

- No CatBoost without explicit OOF−LB gap check vs V14 baseline (CB has historically widened gap).
- All bases trained on identical fold splits and identical `features_v9` rows.
- No raw player profile.
- No rally-disjoint CV.

### Tag and artifacts

- Tags: `v17a_lgb_only`, `v17b_xgb_only`, `v17c_cb_conservative`
- Submission: `submissions/submission_v17_hetero_v11_optblend.csv`

### Concrete commands (DRAFT — only after P2.1 outcome is known)

```
python src/train_v17_lgb_only.py --folds 5 --skip-cb --tag v17a_lgb_only
python src/train_v17_xgb_only.py --folds 5 --skip-cb --tag v17b_xgb_only
# Optional CB — only if OOF-LB gap sanity check passes:
python src/train_v17_cb_conservative.py --folds 5 --tag v17c_cb_conservative
python src/blend_hetero.py --tags v17a_lgb_only v17b_xgb_only v17c_cb_conservative \
    --aux-tag v11 --out submission_v17_hetero_v11_optblend.csv
```

### Success / failure gates

| Gate | Pass | Fail action |
|---|---|---|
| Each base's solo OOF ≥ 0.36 | include in blend | drop that base |
| CB OOF−LB gap ≤ 0.020 (sanity check vs V14) | include CB | drop CB; LGB+XGB only |
| Final blend OOF ≥ V14+V11 + 0.001 | submit to LB | escalate to P3 |

### Risks

- CatBoost previously caused OOF-LB mismatch. Need a separate single-fold sanity LB check before
  trusting it.
- Overlap of LGB/XGB on identical features may yield insufficient diversity.

---

## P3 — Architecture breakthrough (deferred)

Do **not** start before P2 outcomes. Reserved sketches:

1. **Self-supervised rally embedding**: pre-train Transformer/GRU/TCN on rally sequences with
   masked-shot prediction; freeze and append the embeddings as features to GBM. Decouples
   representation learning from final prediction.
2. **Structured point decoder**: model `pointId` as `(depth ∈ {short, half, long}) × (side ∈ {FH,
   mid, BH})` with shared latent. May lift mid-frequency point classes.
3. **Phase-aware MoE**: per-phase experts (SN=1, SN=2, SN=3-4, SN=5+) gated by SN bucket.

Forbidden:
- Plain supervised Transformer simply replacing V11's job (V11+ Gate 2 evidence shows capacity-bound).
- Class-weight escalation past POINT_W cls3=22.0.

---

## Submission budget plan (next 3 days)

| Day | Slot 1 | Slot 2 | Slot 3 |
|---|---|---|---|
| 2026-05-03 (today) | ✅ used: V15_pp diagnostic, LB=0.3506750 | ✅ used: V15_player_only diagnostic, LB=0.3555110 | ✅ used: V15_hist_only diagnostic, LB=0.3574287 |
| 2026-05-04 | ✅ used: V16+V11, LB=0.3673269 | ✅ used: zoo_v16_fast_01, LB=0.3694863 | ❌ used: per-SN bucket, LB=0.3596738 |
| 2026-05-05 | depends on 2026-05-04 outcomes | depends | depends |

Daily limit: 3. **Do NOT submit any more V15 variants. New LB bar is 0.3694863. Per-SN bucket blend failed badly; prefer global multi-model blends and V16-centered seed/heterogeneous ensembles.**

---

## P2.1 note

`train_v14.py` currently does **not** expose `--seed`. Do not start multi-seed runs until the flag
is added and verified to seed LGB `random_state`, XGB `seed`, and data-shuffle. P2.1 is on hold
until P2.0 smoke completes and Codex adds the `--seed` flag.

---

## What needs Codex review before training

1. **P2.0 augmentation correctness**:
   - Pair generation logic (`history shots < k → predict known shot k actionId/pointId`).
   - Server model exclusion: confirm `is_aug` flag cleanly splits augmented rows from all server
     model paths (training, AUC logging, OOF accumulation, threshold opt).
   - Assert in training log: `aug_rows == 2353`, `oof_mask.sum() == 69712`,
     `"serverGetPoint" not in aug_df.columns`.
   - Validation OOF isolation: confirm augmented rows cannot appear in any fold's validation slice.
   - Feature pipeline: confirm `features_v9` handles unknown `gamePlayerId` without crashing.

2. **P2.1 seed-flag plumbing**:
   - Verify `train_v14.py --seed` seeds LGB `random_state`, XGB `seed`, AND 
p.random.seed` /
     data-shuffle; add the flag if absent.
   - Confirm `avg_oof.py` averages probability arrays (not argmaxed predictions) before threshold-opt.

3. **Artifact contract (applies to all P2 candidates)**:
   - All OOF and test artifacts must be saved as individual `.npy` files with the naming pattern:
     `{tag}_oof_act.npy`, `_pt.npy`, `_srv.npy`, `_mask.npy`, `_y_act.npy`, `_y_pt.npy`,
     `_y_srv.npy`, `_nsn.npy`; `{tag}_test_act.npy`, `_pt.npy`, `_srv.npy`.
   - Do **not** use `.npz` unless `blend_ensemble.py` is updated to read `.npz` for all tags.

4. **Gate values (corrected)**:
   - Solo gate: ≥ V14 solo opt ≈ **0.3661** (threshold-optimised V14 before V11 blend).
   - Blend gate: ≥ V14+V11 = **0.3754**.
   - Confirm `blend_ensemble.py` reports OOF OV on the 69712-row real train mask only.

5. **Implementation order**:
   - P2.0 is higher expected-value but higher implementation risk; P2.1 is lower risk.
   - Recommended: implement P2.0 first; if Codex flags any assertion failure or leakage risk,
     fall back to P2.1 immediately.

---

## Pause point

**No training will be launched until Codex completes review of this plan.**

After review, request user confirmation with:
- Which candidate(s) to launch first
- Any modifications to commands / gates / flags
- Any additional safety rails

Once approved: execute according to launch order, monitor logs, report OOF on completion, request
LB submission slot.





