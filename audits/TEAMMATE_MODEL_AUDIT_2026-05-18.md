# Teammate Model Audit — 2026-05-18

**Package**: `table-tennis-prediction-main.zip` (260 KB, 58 entries)
**Source**: `C:/Users/jabir/Downloads/table-tennis-prediction-main.zip`
**Extracted to**: `audits/teammate_table_tennis_2026-05-18/table-tennis-prediction-main/`
**Date received**: 2026-05-14 (file mtime)
**Claimed LB**: **0.4597134** (v6, test_new)

---

## EXECUTIVE VERDICT

**Score is leakage-driven, not real.** The teammate's v6 LB 0.4597 is built on top of `apply_server_leak.py`, which overwrites the submission's `serverGetPoint` column with ground-truth values from `data/raw/test.csv` for the 1236 rally_uids that overlap between old test and `test_new`. Per the teammate's own README, this leak adds **+0.058 LB** (v2→v3 on old test) and **~+0.05 LB** (v5→v6 on test_new). After subtracting the leak, the underlying model is ~LB 0.40–0.41 — modestly above our R-027 PAIR (0.3810) but not the headline 0.4597.

**The leak pattern is the SAME as the previously-banned `AICUP_v1_LB0.4304.zip`.** This package therefore falls under the existing quarantine rule in `LESSONS_CHECKLIST.md` ("Old-test / teammate leak artifacts are quarantined").

**However**, the model code itself (minus the leak post-processing) contains 2-3 legitimate feature ideas that we have NOT tested and should consider stealing legally.

---

## 1. PACKAGE INVENTORY

### Top-level structure
| Path | Size | Note |
|---|---:|---|
| `README.md` | 10,230 B | Documents the leak openly |
| `Makefile` | 3,172 B | Standard build targets |
| `pyproject.toml` | 2,174 B | Dependencies (autogluon 1.5+, streamlit, etc.) |
| `uv.lock` | 508,070 B | Pinned dependency lockfile |
| `app/` | 71 KB | Streamlit UI (not relevant to ML audit) |
| `docker/` | 762 B | Dockerfile + compose |
| `src/` | 99 KB | Source code |
| `tests/` | 132 KB | Unit tests |

### Source files (`src/`)
| File | Size | Purpose | Audit flag |
|---|---:|---|---|
| `train.py` | 16,149 B | Single-run training entrypoint | OK |
| `cv.py` | 25,318 B | 5-fold × 5-seed CV ensemble | OK |
| `predict.py` | 6,446 B | Predict using saved model | OK |
| **`apply_server_leak.py`** | **3,102 B** | **OVERWRITES SGP WITH OLD TEST TRUTH** | **🚨 LEAK** |
| `apply_rule_override.py` | 6,078 B | 0%-prob rule postprocessing | OK |
| `build_augmented_train.py` | 3,339 B | Concat old test → train (legal post-2026-05-13) | OK |
| `features/engineering.py` | 24,035 B | Feature builder + transition tables + player profiles | OK |
| `models/autogluon_model.py` | 13,732 B | AutoGluon TabularPredictor wrapper | OK |
| `evaluate/metrics.py` | 2,488 B | Macro F1 + AUC + per-class threshold optimizer | OK |

### NOT present in the package
- ❌ No model weights (`.pkl`, `.cbm`, `.txt`, `.pt`, `.bin`)
- ❌ No cached predictions / OOF arrays
- ❌ No submission CSVs (the actual leak-applied submission is not shipped)
- ❌ No data files
- ❌ No external dataset

So the package is **code-only**. To reproduce the 0.4597 LB, a user must run the full 6.5h CV + apply the leak themselves.

### Suspicious filename red flags
- `src/apply_server_leak.py` — name explicitly says "leak"
- `tests/test_apply_server_leak.py` — leak code has unit tests, indicating it's treated as a first-class feature
- `README.md` claims "v3 + serverGetPoint test-leak: +0.058" openly

---

## 2. LEAKAGE AUDIT

| Check | Status | Evidence |
|---|---|---|
| Uses `serverGetPoint` from `test.csv` as submission output | **🚨 CONFIRMED LEAK** | `src/apply_server_leak.py` line 35-61. README v3 entry: "+0.058 LB". |
| Uses `serverGetPoint` from `test_new.csv` | NOT POSSIBLE | test_new.csv has no SGP column (acknowledged in their code line 16) |
| Old test.csv overlap exploit | **🚨 CONFIRMED** | Per README: "test_new 中與舊 test 重疊的 1236 個 rally 直接套用真值" (apply truth to overlapping rallies) |
| Copies SGP directly into submission | **🚨 CONFIRMED** | `apply_server_leak.py:58-60` writes `srv_true` into submission's `serverGetPoint` column |
| Target leakage from actionId/pointId target row | NO | `_extract_features` uses only `history.iloc[:i]` shots; current shot excluded |
| Uses future rows beyond visible prefix | NO | Training: shot i uses shots `0..i-1`. Test: uses all available history shots. |
| Uses rally_uid order as temporal signal | NO | `sort_values(["rally_uid", "strikeNumber"])` sorts within rally; rally_uid is used as group key only |
| Player-ID memorization | LOW RISK | Per-player aggregate stats (win rate, action freq, point freq) merged via `gamePlayerId`; relies on player presence in train. De-identified test players get `defaults` (0.5 win rate, 0 freqs). |
| Cached predictions trained with leaked labels | N/A | No cached predictions shipped |
| External data | NONE | No external sources; only competition CSVs (train, test, test_new) |
| Rally-disjoint CV | OK | `GroupKFold(n_splits=5)` with `groups=rally_uid` — rally-disjoint splits |

### Leak quantification (per README)
- v2 (no leak): old-test LB 0.3822214
- v3 (= v2 + SGP leak): old-test LB **0.4401335** → leak alone = **+0.058 LB**
- v5 (= v4 + augmented train, new test era): LB 0.4465332
- v6 (= v5 + transition matrices + threshold opt): LB **0.4597134**

The teammate explicitly states the leak survives the test_new migration: "leak 來源換成舊 test ... 覆蓋 1236/1845 個 rally". So v6's 0.4597 includes ~+0.05 from leak.

**Estimated non-leak LB**: 0.4597 − 0.05 ≈ **0.41** (compared to our R-027 PAIR LB 0.3810).

---

## 3. REPRODUCIBILITY AUDIT

### Entrypoints
- `python -m src.build_augmented_train` — concat old test → train
- `python -m src.cv` — 5-fold × 5-seed AutoGluon CV (6.5h CPU)
- `python -m src.apply_server_leak` — **LEAK STEP (DO NOT RUN)**
- `python -m src.apply_rule_override` — 0%-prob postprocessing
- Streamlit UI: `streamlit run app/streamlit_app.py`

### Expected files
- `data/raw/train.csv` — competition train (we have)
- `data/raw/test.csv` — old test with SGP (we have)
- `data/raw/test_new.csv` — new test (we have)

### Targets
- Per README v6 CLI: `--test-path data/raw/test_new.csv` — targets **new** test
- Submission shape: 1 row per `rally_uid` in test_new (1845 rows)

### Runs on our project?
- **Schema-compatible**: their feature engineering reads the same column names we use
- **Output incompatible**: their pipeline emits ONE row per rally (`build_test_features` does `groupby rally_uid` then aggregates), whereas our submissions emit one row per RALLY too (1845 rows). So output shape matches.
- **Dependencies**: requires `autogluon>=1.5,<1.6` (~3GB install) and Python 3.12
- **Runtime**: 6.5h CPU for full v6; 5 min for toy mode (`time_limit=60 n_seeds=1 n_splits=2`)

### Lightweight dry-run safety
A pure static-analysis pass (no execution) reveals all the leak entrypoints. A dry-run of just `src.build_augmented_train` would write `data/raw/train_augmented.csv` — **DO NOT RUN** because it would mutate our `data/` folder.

A dry-run of just the feature engineering (call `compute_transition_tables` on our `train.csv`) is safe — pure function, no side effects. **Recommend**: import-only inspection without executing the full pipeline.

---

## 4. MODEL / ARCHITECTURE ANALYSIS

### Model family
**AutoGluon TabularPredictor × 3 (one per target)** × 5 folds × 5 seeds = 75 trained models per submission.
- `actionId`: 19-class, `eval_metric=macro_f1`, weighted-ensemble of {LightGBM, CatBoost, XGBoost, RandomForest, ExtraTrees, NN_TORCH, FASTAI}. NN_TORCH + FASTAI excluded in default config.
- `pointId`: same architecture, 10-class
- `serverGetPoint`: same, binary AUC

### Feature engineering (`src/features/engineering.py`)

| Feature group | Count | Notes |
|---|---:|---|
| Match-state context | ~10 | sex, numberGame, scores, score_diff, score_pressure, is_serve_side, rally_phase, points_to_win |
| Last-shot features | 7 | `last_{actionId, pointId, handId, strengthId, spinId, positionId, strikeId}` |
| 2nd-to-last shot features | 7 | `prev2_{...}` |
| Action-point combo | 2 | `last_action_point_combo`, `prev2_action_point_combo` (`a*10+p` interaction) |
| Hist aggregates | ~30 | mode, nunique, mean, std, last3_mean for SEQ_COLS |
| Per-class freqs | 29 | `hist_action_freq_{0..18}`, `hist_point_freq_{0..9}` |
| Entropy / dominance | 4 | shannon entropy + max-class-frequency for actions and points |
| Streak features | 3 | streak_action, streak_point, consecutive_same_player |
| Player profile (cross-rally) | ~31 | per-player win_rate, action/point freq distributions for top-k classes (self + opponent + diff) |
| **Transition matrix priors** | **33** | `P(next_action \| last_action, is_serve_side)` × 19 + `P(next_point \| last_action, last_point)` × 10 + 4 entropy/top1 |

### Validation strategy
- `GroupKFold(n_splits=5)` with `groups=rally_uid` — rally-disjoint
- 5 seeds per fold → 25 trained models per target → mean of predict_proba → argmax
- All cross-rally features (player profiles, transition tables) are computed **per fold** using only the train fold's data — proper leak-safe

### Calibration / post-processing
- `optimize_multiclass_thresholds` — greedy coordinate ascent over per-class scale factors to maximize macro F1 on OOF predictions (3 passes, 14 candidate scales)
- `apply_server_leak` — **LEAK**
- `apply_rule_override` — replaces predictions with train mode when the predicted class has **0 probability** in train under same `(prev_action, last_action, last_point)` context (and context has ≥30 samples)

### Comparison vs our stack

| Dimension | Teammate | Us | Verdict |
|---|---|---|---|
| Backbone | AutoGluon (auto-ensemble GBM+RF+ET+NN) | V11 transformer + V14/V16 LightGBM stack + V13 GBM | Different paradigms — could be complementary |
| Feature builder | Single mono-script with 100+ features | V8/V9/V10/V14 hand-engineered features + transformer raw seq | We're more modular |
| CV split | GroupKFold by rally_uid | GroupKFold by **match** | Ours is stricter (match-disjoint > rally-disjoint) |
| Seed averaging | 5 seeds × 5 folds | 1-3 seeds × 5 folds | Theirs has more variance reduction |
| Aug data (old test) | Yes (~8% boost) | Yes (since 2026-05-13, R-027) | Equivalent |
| Transition matrix features | **YES** (their +0.0132) | **NO** | Worth stealing |
| Player profiles | YES (per-player win/action/point distributions) | Partial (some v9_recvhand player features) | Mostly orthogonal |
| Per-class threshold opt | YES (greedy ascent) | YES (via blend_zoo_v2 calib_thr CW grid) | Equivalent |
| 0%-prob rule override | YES (+0.0014) | NO | Marginal but cheap |
| Server leak | YES (+0.05) | **NO** (banned per LESSONS) | Don't touch |

---

## 5. SUBMISSION ANALYSIS

**No submission CSV is shipped in the package** — only the code that generates one. Cannot inspect distributions, line endings, or SGP values directly. The README claims their submissions are well-formed CSVs with 1845 rows for test_new.

To verify, we would have to run their full v6 pipeline (~6.5h CPU + GPU-free) and inspect the output. This is the only way to confirm the actual SGP distribution would be a binary copy from old test (which would be detectable: SGP would be exactly 0/1 for 1236 rows, continuous probability for 609 rows).

---

## 6. LEGAL REUSE CLASSIFICATION

| Component | Class | Action |
|---|---|---|
| `apply_server_leak.py` + tests | **DO_NOT_USE** | Quarantine. Banned by LESSONS rule (old-test SGP overwrite). Same pattern as `AICUP_v1_LB0.4304.zip`. |
| `build_augmented_train.py` | **SAFE_TO_REUSE_NOW** (but redundant) | Same pattern as our `--include-old-test` flag. We already do this since 2026-05-13. No reuse needed. |
| `apply_rule_override.py` (0%-prob override) | **SAFE_AFTER_REIMPLEMENTATION** | Cheap post-processing. Marginal +0.0014 LB. Re-implement in our framework as a calibration option. |
| `optimize_multiclass_thresholds` | **SAFE_AFTER_REIMPLEMENTATION** (redundant) | We have equivalent in `blend_zoo_v2.calib_thr`. Their greedy ascent over scale factors is similar to our CW grid search. No reuse needed. |
| **`compute_transition_tables` + `merge_transition_features`** | **NEEDS_CODEX_REVIEW** | **Highest-EV stealable idea (+0.0132 in their LB)**. Empirical conditional priors as model features. Per-fold computation (leak-safe). Reimplement as new feature option for our V14/V16. |
| `compute_player_profiles` + `merge_player_profiles` | **NEEDS_CODEX_REVIEW** | Per-player aggregate stats (win_rate uses train SGP, which is legal supervised). Risk: de-identified test players get defaults; could be hurt by domain shift. We have similar in v9_recvhand. Compare carefully before integrating. |
| AutoGluon ensemble | **NEEDS_CODEX_REVIEW** | Different framework from our LightGBM stack. Could be a new diversity component. Setup overhead: ~3 GB install + AutoGluon version lock. |
| `GroupKFold(rally_uid)` | **SAFE_TO_REUSE_NOW** (redundant) | We already use GroupKFold by `match` (stricter). No change needed. |
| Streamlit UI (`app/`) | **DO_NOT_USE** | Not relevant to our LB workflow. |
| Tests | OK to read for documentation | Don't import. |

---

## 7. USEFUL IDEAS WE CAN STEAL (LEGALLY)

### High-EV (worth a Codex review + implementation)

1. **Transition matrix features (+0.0132 in their LB)** — `src/features/engineering.py:458-558`
   - `P(next_action | last_action, is_serve_side)` × 19 = 19 prior features
   - `P(next_point | last_action, last_point)` × 10 = 10 prior features
   - Plus 4 summary stats: entropy + top1 dominance for each
   - **Total: 33 new features**
   - Per-fold computation from `raw_train_fold` only → leak-safe
   - Estimated EV: +0.005 to +0.015 OOF if it composes with our V14/V16 stack
   - **Implementation cost**: ~1 day. Adapt to our `train_v14.py` / `train_v16_testhist_aug.py` per-fold feature computation
   - **Risk**: low. Pure empirical prior, no novel architecture, leak-safe by construction.

### Medium-EV

2. **0%-probability rule override (+0.0014 in their LB)** — `src/apply_rule_override.py`
   - Post-processing layer applied to final predictions
   - Build `(prev_action, last_action, last_point) → P(next_action), P(next_point)` table from train
   - For each test rally, if predicted class has **zero observations** in train under same context AND context has ≥30 samples, replace prediction with train mode
   - Estimated EV: +0.001 to +0.005 LB (their measured +0.0014 was on 7/1236 rows)
   - **Implementation cost**: ~2 hours. Wrap as a calibration option in our blender.
   - **Risk**: very low. Postprocessing only, leak-safe.

### Low-EV / partial reuse

3. **Player profile features** — already in our v9_recvhand. Compare implementations:
   - Their `_PROFILE_ACTION_TOPK = (0,1,2,5,6,10,13,15)` chooses top-k action classes for per-player rate features
   - Their `win_rate_diff = p_player_win_rate - opp_player_win_rate` interaction
   - These specific choices might add marginal lift if not in our v9_recvhand. Worth a 30-min diff.
   - **Risk**: medium. De-identified test players require `defaults` fallback — same risk we already have.

### NOT useful (already have / known dead-end)

- AutoGluon ensemble — different framework, integration overhead too high vs expected diversity. Skip.
- Augmented train (old test → train concat) — same as our `--include-old-test`. Skip.
- Per-class threshold optimization — same as our `calib_thr`. Skip.
- GroupKFold by rally_uid — we use stricter match-level. Skip.

---

## 8. PARTS TO QUARANTINE

1. **`src/apply_server_leak.py`** — DELETE if integrated. Or keep in `audits/teammate_table_tennis_2026-05-18/` as historical reference of banned pattern.
2. **`tests/test_apply_server_leak.py`** — same.
3. **Any submission CSV produced by their v3+ pipeline** — not in package but DO NOT regenerate.

---

## 9. RECOMMENDED NEXT ACTIONS

### Immediate (no Codex review needed)
- ✅ This audit doc written
- ✅ Quarantine folder created (`audits/teammate_table_tennis_2026-05-18/`)
- ✅ Package extracted but not copied into `src/` or `data/`

### Before any integration (REQUIRES CODEX REVIEW)
Open R-029 in `REVIEW_QUEUE.md` requesting Codex review of:
1. **Transition matrix features**: integration plan into our V14/V16 GBM stack. Discuss whether to add as 33 new features in `features_v9.py` or as a separate feature module. Per-fold computation pattern to avoid leakage. Expected OOF lift bounds.
2. **0%-prob rule override**: integration as a post-processing calibration option. Threshold for "minimum context samples" (their default = 30). Whether to apply only to argmax or to soft predictions.

### What NOT to do
- ❌ Do NOT copy `apply_server_leak.py` into our src/
- ❌ Do NOT run their full pipeline (would consume 6.5h CPU and produce leak-contaminated output)
- ❌ Do NOT upload any submission derived from their leak pattern
- ❌ Do NOT compare their 0.4597 LB as a benchmark — it's leakage-driven

---

## 10. CODEX REVIEW REQUIRED?

**YES** for:
- Transition matrix features (R-029 proposal)
- Player profile feature diff vs our v9_recvhand
- AutoGluon as alternative GBM (lower priority — Codex can pre-approve or block)

**NO** for:
- 0%-prob rule override re-implementation (small enough to be a regular T2-component PR with self-review)

---

## FINAL ANSWERS (1-LINE EACH)

1. **Is the teammate model legal?** Partly. The CV training pipeline is legal; the `apply_server_leak.py` post-processing is **NOT** (overwrites SGP with old-test truth, banned by our LESSONS rule and matches the precedent of `AICUP_v1_LB0.4304.zip`).

2. **Is the score (0.4597) real or leakage-driven?** **Leakage-driven**. Per teammate's own README, the SGP overwrite alone contributes ~+0.05 LB. Removing the leak puts their underlying model at ~LB 0.41 — modestly above our R-027 PAIR (0.3810) but nowhere near 0.4597.

3. **What parts are reusable?**
   - **Transition matrix features** (NEEDS_CODEX_REVIEW) — their +0.0132 LB lift, leak-safe per-fold computation, ~1 day to reimplement.
   - **0%-prob rule override** (SAFE_AFTER_REIMPLEMENTATION) — marginal +0.001-0.005 LB, ~2 hours.
   - Player profile diff (NEEDS_CODEX_REVIEW) — possibly small gains over our v9_recvhand.
   - **DO NOT** reuse: `apply_server_leak.py`, augmented train (redundant), AutoGluon (overhead too high).

4. **What should we ask Codex to review?** Open R-029 with integration plan for:
   - Transition matrix features as a new feature module for V14/V16 GBM (highest EV)
   - Player profile feature diff vs our existing v9_recvhand
   - Expected OOF lift bounds + leak-safety verification of per-fold computation

5. **Next safest experiment?** **Re-implement transition matrix features in our framework, train one V14 variant with them, measure OOF delta.** Estimated cost: ~3-4 hours of implementation + ~134 min CPU for v14 training. If OOF improves by ≥+0.003 over v14_seed2 baseline (0.3687), open R-029 formally and proceed. Wait for the current 48h deadline orchestrator to finish so we don't contend for CPU.
