# Package v8 Audit — teammate LB 0.4419

Source zip: `C:\Users\jabir\Downloads\package_v8_0.4419.zip`

Extracted to: `audits/package_v8_0.4419_20260521`

## Executive verdict

The package claims Public LB `0.4419`, but it is not a clean/legal reference for our pipeline.

Two separate leakage surfaces are present:

1. `src/apply_server_leak.py` explicitly overwrites `serverGetPoint` for 1,236 overlapping old-test rallies using old `test.csv` truth.
2. `data/train_pseudo.csv` contains shifted `test_new` rows whose overlap `serverGetPoint` values exactly match old-test SGP truth, including the synthetic target rows.

Therefore, do not use the final submission, leak-postprocessed submissions, or `train_pseudo.csv` as-is.

## Submission comparison

Files inspected:

- `submissions/v8_raw_submission.csv`
- `submissions/v8_after_server_leak.csv`
- `submissions/sub_v8_FULL_FINAL_LB0.4419.csv`

All have 1,845 unique `rally_uid` rows and binary `serverGetPoint`.

Diff summary:

- raw -> leak: `serverGetPoint` changed on 127 rows; `actionId` and `pointId` unchanged.
- leak -> final: `actionId` changed on 2 rows; `pointId` changed on 1 row; `serverGetPoint` unchanged.
- final overlap with old test: all 1,236 overlapping SGP labels exactly equal old `test.csv` truth.

Important implication: the final 0.4419 is materially server-leak boosted. Also, raw SGP already matches old overlap truth at about 89.7%, likely because leaked SGP is already present inside `train_pseudo.csv`.

## Data audit

Package data shapes:

- `data/train.csv`: 84,707 rows / 14,995 rallies / 216 matches
- `data/test.csv`: 3,589 rows / 1,236 rallies / 55 matches, includes SGP truth
- `data/test_new.csv`: 5,668 rows / 1,845 rallies / 79 matches, no SGP column
- `data/train_pseudo.csv`: 95,809 rows / 18,076 rallies / 295 matches

`train_pseudo.csv` includes:

- original train rows
- old `test.csv` shifted by +20,000 rally_uid
- `test_new.csv` shifted by +60,000 rally_uid
- one synthetic target row per `test_new` rally

For the 1,236 old/new overlapping rallies, all shifted `test_new` visible rows and synthetic target rows have SGP equal to old-test truth. This is not usable under our rules.

## Code audit

Useful, but must be reimplemented cleanly:

- `features/engineering.py` uses prefix-only history for action/point/context features.
- It computes per-class frequency, entropy, dominance, streak, transition, and player-profile features.
- `cv.py` computes player profile and transition tables from each training fold only, which is the right fold-safety pattern.
- `apply_rule_override.py` is a cheap action/point postprocess and changed only 3 final rows in v8.
- `autogluon_model.py` wraps AutoGluon with grouped validation and best-quality style ensembling.

Not usable:

- `apply_server_leak.py`
- `v8_after_server_leak.csv`
- `sub_v8_FULL_FINAL_LB0.4419.csv`
- `train_pseudo.csv` as-is
- any shifted `test_new` SGP labels
- any pseudo target rows without a separate T3 pseudo-label approval

## Legal ideas worth extracting

Priority candidates:

1. Clean AutoGluon component: train only on legal data, no `train_pseudo.csv`, no server overwrite, output OOF/test probability arrays for blending.
2. Fold-safe transition priors: reimplement from `cv.py` / `features/engineering.py` without teammate code copy if needed.
3. Rule override diagnostic: apply action/point-only grammar/context override to our current best OOF/test predictions and verify no OOF regression.
4. Prefix frequency / entropy / streak features: already aligned with our R-029 feature direction.
5. Player-profile top-k features: potentially useful but high overfit risk; only behind separate review and player-disjoint checks.

## Recommended next step

Open a new review item, e.g. `R-031`, for a clean AutoGluon no-leak smoke:

- source data: `data/train.csv` plus only explicitly approved legal augmentation
- no `train_pseudo.csv`
- no `apply_server_leak.py`
- match/group-aware CV
- SGP predicted by model only, not overwritten
- output OOF probabilities and test probabilities, not just hard labels

Separately, keep R-029 transition/prefix feature work alive, but do not mix it with leaked teammate artifacts.

---

## Deep-dive update (2026-05-21, post Claude full file scan)

### Teammate's published lever-by-lever LB attribution

From `README.md` §2 (verified LB 0.4419):

| # | Lever | Claimed lift | Legal? | Comments |
|---|---|---:|---|---|
| 1 | augmented train (data/test.csv as legal labels) | +0.005 LB | ✅ | Already in our `--include-old-test` (R-027 PAIR) |
| 2 | player_profile (16-col, top-k action/point + win_rate) | **+0.04 OOF F1** | ⚠ | OOF claim only — our V15_player_only LB-FAILED 0.3555 |
| 3 | class_weight (AutoGluon sample weight) | +0.005 LB | ✅ | NEW for us — try via AutoGluon or our GBM weights |
| 4 | server_leak (SGP truth overwrite) | **+0.022 LB** | ❌ | The biggest lift — but LEAK; not legal under our rules per CLAUDE.md |
| 5 | rule_override (0%-prob → train mode) | +0.0014 LB | ✅ | Cheap post-process — apply to R-034 immediately |
| 6 | best_quality preset (AG: LGB+XGB+CAT+NN+KNN) | +0.01 LB | ✅ | We can replicate via separate model stack |
| 7 | 5-seed × 5-fold | +0.01 LB | ✅ | We already do this for some components |
| 8 | train_pseudo.csv (synthetic last-shot pseudo) | +0.005 LB | ❌ | Last-shot pseudo from leaked SGP base — illegal |
| 9 | GroupKFold by rally_uid | safety | ✅ | We use match-grouped (stricter) — keep |

**Legal-only LB ceiling estimate** (subtracting 4+8 illegal):
- v8 LB 0.4419 − 0.022 − 0.005 = **~0.4149** as a pure-legal upper bound
- That's still +0.031 over our R-034 0.3838 → meaningful headroom exists

### Notable inconsistency vs our 2026-05-04 V15_player_only LB-FAIL

The teammate claims `player_profile` is +0.04 OOF F1, "safety: 高". This contradicts our
2026-05-04 finding: `V15_player_only + V11 = OOF 0.3777, LB 0.3555` (gap −0.022). That gap is
classic player-profile non-transfer because test players are de-identified (IDs 199-206
never seen in train).

Possible reconciliations:
1. Teammate's version is fold-safe (per-fold profile from train fold only, not whole-train).
   Ours may have been fold-leaky.
2. Teammate uses BOTH `p_*` (player) and `opp_*` (opponent) — provides 2× signal vs single-side.
3. Teammate stratifies by top-k classes only (action=[0,1,2,5,6,10,13,15], point=[0,4,5,8,9]).
4. Their LB claim is "+0.04 OOF" — not LB-verified standalone. The OOF→LB ratio for
   player_profile may still be poor under our match-disjoint CV.

**Action**: re-test player_profile with fold-safe per-fold computation + p+opp side, isolated
from other levers, OOF gate ≥ +0.003 OV AND blend-swap (NEW gate, 2026-05-21).

### Feature-by-feature legal extraction map

| Teammate feature group | Our equivalent | Status |
|---|---|---|
| `hist_action_freq_{0..18}`, `hist_point_freq_{0..9}` | v15feat_a `hist_action/point_freq_*` | ✅ already in R-034 |
| `hist_action_entropy`, `hist_point_entropy`, `*_dominance` | v15feat_a same | ✅ in R-034 |
| `streak_action`, `streak_point`, `consecutive_same_player` | v15feat_a | ✅ in R-034 |
| Lag-2 columns `last_*`, `prev2_*` (7 cols each × 2) | v15feat_a lag-1/lag-2 | ✅ in R-034 |
| `last_action_point_combo`, `prev2_action_point_combo` (×10 categorical) | — | ❌ NEW for us, cheap add |
| `is_serve_side`, `rally_phase`, `next_strikeNumber` | partial | ⚠ partial — `is_serve_side` cheap to add |
| `is_deuce`, `match_point_self/other`, `total_points` | — | ❌ NEW for us, score-pressure features |
| `points_to_win_self/other`, `score_lead_abs` | — | ❌ NEW for us |
| `trans_action_prior_{0..18}` × 19 (P(next_action \| last_action, is_serve_side)) | features_v15feat_b (R-029b, never trained) | 🟡 designed, never deployed |
| `trans_point_prior_{0..9}` × 10 | features_v15feat_b | 🟡 designed, never deployed |
| `trans_action/point_entropy/top1` (4 summary) | features_v15feat_b | 🟡 designed, never deployed |
| `compute_player_profiles` (16-col top-k) | features_v9_recvprofile (partial) | 🟡 different shape; re-test |

### Postprocess `apply_rule_override.py` deep look

Conditional table: `(prev_action, last_action, last_point) → P(next_actionId)` and
`→ P(next_pointId)`. Only overrides predictions with **0%** empirical probability under the
context (and require ≥30 samples in context for trust). Replaces with train-mode.

In v8, this changed 3 final rows out of 1,845 — extremely conservative. Teammate's claim:
+0.0014 LB lift on v3 → v4 (a 7-row delta on 1,236 old-test).

**Applicability to our R-034**: The probability we have ≥1 row in our 1,845-row prediction
that is empirically impossible under (prev_action, last_action, last_point) is non-zero. The
post-process is FREE compute and zero risk for the changed rows (they were impossible per
training data anyway). Build R-042 as a diagnostic upload:
`apply_rule_override(R-034 submission, train_full)` → expect 1-10 row changes, pred LB
~0.3838 + ~0.0008 = 0.3846 if it transfers like teammate's claim.

### What's BANNED to copy verbatim

- `src/apply_server_leak.py` (SGP truth from test.csv — banned by our CLAUDE.md no-leak policy)
- `data/train_pseudo.csv` (contains leaked pseudo labels for test_new)
- `submissions/v8_after_server_leak.csv`, `sub_v8_FULL_FINAL_LB0.4419.csv` (leak-boosted)
- Any "synthetic last-shot pseudo label" workflow without an independent T3 pseudo-label review

### Recommended action items (ranked by LB-EV / effort)

1. **R-042: apply_rule_override post-process on R-034 submission** — 5 min, pred lift +0.0008 LB
2. **R-031 (already opened): clean AutoGluon component** — separate model class for blend slot
3. **R-043: train v15feat_b (R-029b transition priors) standalone**, blend-test vs R-034
4. **R-044: add 6 cheap score-pressure features** (is_serve_side, is_deuce, match_point_*, total_points, points_to_win_*) to v14_seed2 → new feature variant; blend-test
5. **R-045: fold-safe player_profile re-test** with p+opp sides and top-k stratification; new
   blend slot — strictly OOF + blend-swap gated since LB-fail history exists.
6. **R-046: action-point interaction combos** (last_action_point_combo, prev2_action_point_combo) — 2 cheap categorical features
7. SKIP: any train_pseudo, server_leak, or synthetic-target lever (banned).

### LB headroom estimate

If R-042 + R-043 + R-044 + R-045 each lifted +0.001-0.003 LB (modest interpretation), cumulative
LB ~0.3838 + 0.006-0.012 = **0.39-0.40 LB**. Still below teammate's legal ceiling 0.4149, but
meaningfully closer.

The teammate's pure-AutoGluon stack delta (best_quality preset, +0.01 LB) is the largest
single legal lever they have. We should consider building an AutoGluon-based component into
our blend as a separate model class.
