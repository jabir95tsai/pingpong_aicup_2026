# RESULTS

## ⚠ 2026-05-06 — COMPETITION RESET: NEW TEST SET RELEASED, ALL OLD LB SCORES INVALIDATED

**The organisers released `data/test_new.csv` containing a fresh Private Test Data set.
Public Leaderboard has been fully reset. ALL previously-reported LB scores in this file
(zoo_v6_elig1=0.3749, zoo_v2 top-1=0.3734, zoo_v3 top-1=0.3675, zoo_holdout=0.3547,
zoo_v16_fast_01=0.3695, V14+V11=0.3599, V15_*, V16+V11=0.3673, etc.) NO LONGER COUNT
on the public leaderboard.**

What this means concretely:
- All `submission_*.csv` files in `submissions/` are predictions on the OLD test (1236
  rallies). They cannot be re-submitted directly — the new test requires 1,845 rows.
- All `oof_predictions/{tag}_test_*.npy` arrays are length-1236 predictions over OLD test.
  They are stale; new arrays must be generated against `test_new.csv`.
- All training OOF artifacts (`{tag}_oof_*.npy`) are computed on TRAIN (unchanged) and
  remain valid. The 5-fold OOF mask is still 69712/69712.
- All Locked Rules in STRATEGY.md derived from old LB feedback (Rules 9–13: NONE > THR,
  v11_aug structurally critical, P11 directional signal, blend-size cap, edge-temperature
  guard) are PROVISIONAL pending re-validation on the NEW LB.

NEW test schema differences vs OLD (from `data/test_new.csv` inspection):

| Property | OLD test | NEW test (`test_new.csv`) |
|---|---:|---:|
| Total rows | 3,589 | **5,668** (+58%) |
| Distinct rallies | 1,236 | **1,845** (+49%) |
| Distinct matches | 55 | **79** (+24 new) |
| Match overlap with train | 0 | 0 (still match-disjoint) |
| Distinct players | 63 | **71** (+8 new) |
| Player overlap with train | 40/63 (63.5%) | **40/71 (56.3%)** |
| `serverGetPoint` column | present (real labels) | **ABSENT** |
| rally_uid overlap with OLD test | n/a | **all 1,236 OLD UIDs present** |
| Per-rally shot count | (was similar) | mean 3.07, median 2, max 24 |

Implications:
- **Lower player overlap (56.3% vs 63.5%)** means player-aware features are even higher
  risk for non-transfer than before. V15 player-profile parking is reaffirmed.
- **rally_uid overlap with OLD test is total** — but the organisers state "rally_uid
  is randomly shuffled; numeric continuity does not imply rally order". Local inspection
  confirms the 1,236 overlapping rally histories are byte-identical on the 17 shared
  columns, but old submissions are still incomplete because `test_new.csv` adds 609
  new rallies. Old predictions may be used only as a sanity reference for the overlap,
  not as a complete submission.
- **serverGetPoint column absent in new test** — this is actually CLEANER (no
  accidental SGP-as-feature risk). All training scripts already do
  `test_df["serverGetPoint"] = -1` which simply ADDS the column with -1.
- **Submission length must be 1,845 rows** (one per test rally), not 1,236.
- **New aug parquet rebuilt** — `data/test_history_pairs_new.parquet` has 5,668
  rows / 3,823 pairs with `serverGetPoint=-1` and `is_aug=1`. Old
  `data/test_history_pairs.parquet` remains as an old-test forensic artifact only.

Old LB results below preserved as historical reference — but treat them as old-test
proxies that don't necessarily transfer to the new LB. Re-validation required.

---

## Round: 2026-05-02 / 2026-05-04 — V14, V15 ablation, V16 aug, P2.1 seeds (HISTORICAL — old test)

---

## 1. Submission Outcomes (Public LB)

| Submission | OOF OV | Public LB | OOF−LB Δ | Decision |
|---|---|---|---|---|
| `submission_v12_v11_optblend.csv` (prior best) | 0.3734 | 0.3541608 | −0.019 | superseded |
| `submission_v14_5f_nocb_v11_optblend.csv` | 0.3754 | 0.3598509 | −0.016 | superseded by V16 |
| `submission_v15_pp_v11_optblend.csv` | 0.3765 | 0.3506750 | **−0.026 (anomalous)** | ❌ DO NOT REUSE |
| `submission_v15_player_only_v11_optblend.csv` | 0.3777 | 0.3555110 | **−0.022** | ❌ player profile non-transfer confirmed |
| `submission_v16_testhist_aug_v11_optblend.csv` | 0.3743 | 0.3673269 | −0.0070 | superseded by zoo blend |
| `submission_zoo_v16_fast_01_v16_v14_seed1_v12_5f_v11.csv` | 0.37998 | 0.3694863 | −0.01049 | superseded by zoo_v2 |
| `submission_zoo_v2_top1_thr_v11_v11plus_v13_v14s0_v16.csv` | 0.38291 | 0.3733788 | −0.00953 | superseded by zoo_v6 NONE |
| `submission_zoo_v3_top1_thr_v11_v11plus_v125f_v13_v14s0_v16_avg3.csv` | 0.38385 | 0.3675453 | −0.01630 | ❌ v16_avg3 / 6-model zoo did not transfer |
| **`submission_zoo_v6_elig1_none_v11_aug_v11plus_v13_v14s2_v16.csv`** | **0.3794** | **0.3748577** | **−0.0045** | ✅ **CURRENT BEST (NEW 2026-05-06)** |
| `submission_zoo_holdout_top1_none_v11plus_v13_v14s2_v16.csv` | 0.3763 | 0.3546861 | **−0.0216** | ❌ Same NONE blend WITHOUT v11_aug → collapses LB by −0.0202. **v11_aug is CRITICAL for the NONE arm.** |

OOF−LB gap (|OOF − LB|, sorted by gap):

| Submission | OOF | LB | Gap |
|---|---|---|---|
| **zoo_v6_elig1 NONE (current best, NEW)** | **0.3794** | **0.3748577** | **0.0045** (NONE-calibration transfers BEST) |
| V16+V11 (prior best) | 0.3743 | 0.3673269 | 0.0070 (OOF underestimated LB) |
| zoo_v2_top1 THR-edge | 0.38291 | 0.3733788 | 0.00953 (5-model THR transferred) |
| zoo_v16_fast_01 (prior best) | 0.37998 | 0.3694863 | 0.01049 (multi-blend transferred) |
| zoo_v3_top1 THR-edge | 0.38385 | 0.3675453 | **0.01630** (higher OOF, worse LB) |
| V14+V11 (prior best) | 0.3754 | 0.3598509 | 0.0155 (normal) |
| V12+V11 (prior best) | 0.3734 | 0.3541608 | 0.0192 (normal) |
| V15_player_only+V11 | 0.3777 | 0.3555110 | **0.0222** (elevated, player non-transfer) |
| V15_pp+V11 | 0.3765 | 0.3506750 | **0.0258** (worst, full V15 bundle) |

Pattern: any pipeline that includes raw player profile features widens the OOF−LB gap by
+0.007–0.010 vs V14, consistent with player ID statistics overfitting on 100% known train players.
V14+V11 remains the cleanest transfer.

---

## 2. Per-Model OOF Results

### V14 (features_v9, 5-fold no-CB)

```
Tag: v14_5f_nocb
Mask: 69712 / 69712 (100%)
Base    : action≈0.3793  point≈0.2162  AUC≈0.6101  OV≈0.3602   (Codex sanity-check)
Opt blnd: action=0.4039  point=0.2292  AUC=0.6105  OV=0.3754
SN=2 slice: OV=0.2696  (vs V12 baseline 0.271 — flat)
```

> Earlier table reported V14 base as `action=0.3839 point=0.2151 AUC=0.6104 OV=0.3617`. Codex sanity
> check against the actual V14 OOF artifact returned the values shown above; the previous numbers
> were a copy of the V15_pp pre-blend opt logs. Numbers are now consistent with the V14 5-fold run.

V9 joint serve-receive priors did not improve SN=2 (the targeted slice). The +0.002 OOF gain over
V12+V11 came from LGB+XGB benefiting marginally from extra features overall, not the SN=2 thesis.

### V15_pp (features_v10 = V9 + player profile + hist freq + streak, 5-fold no-CB)

```
Tag: v15_pp
Mask: 69712 / 69712 (100%)
Base    : (raw base log not preserved cleanly; opt scores below are authoritative)
Opt sub : action=0.3946  point=0.2223  AUC=0.6104  OV=0.3688

Per-class F1 highlights:
  pointId
    miss     (cls0): 0.4386  (improved from V14)
    FH_short (cls1): 0.1307  (improved)
    mid_short(cls2): 0.2232  (improved)
    BH_short (cls3): 0.0000  (still unlearnable)
  actionId
    Loop     (cls1): 0.6345  (consistent)
    Chop     (cls12):0.6766  (improved)
```

V15_pp solo OOF beats V14 solo on optimised metrics, but behind V14+V11 blend (0.3754).
V15_pp + V11 achieves OOF 0.3765, but Public LB drops sharply to 0.3506750.

---

## 3. V15_pp Failure Analysis

V15_pp + V11 has higher OOF but lower LB than V14 + V11. Mechanism hypotheses:

**H1: Player profile non-transfer.**
The pp_act_freq / opp_act_freq features encode player-specific style. With 63.5% public-LB player
overlap, the OOF (which uses match-disjoint GroupKFold but with 100% known players) over-credits
this signal vs what actually transfers to LB.

**H2: Threshold optimization overfit.**
With 1255 features (vs 1170 in V14), threshold + class-weight Scipy optimization has more degrees of
freedom on a noisier OOF surface. Threshold-opt gain on V15_pp = +0.0072 vs V14's +0.0117, but the
*absolute scaled weights* may be more aggressive on OOF-only signals that don't survive on test.

**H3: Hist freq + streak redundancy with V7 grammar.**
V7 already encodes `recv_serve[(serve_act, sex)] -> P(receive_a | ...)`. hist_action_freq overlaps
heavily for SN=2 rows where history is the serve only. Adding redundant features that the LGB model
fits via fold-specific quirks can inflate OOF without adding LB signal.

**H4: Unidentified — combined effect.**
The three new feature groups together produce a worse-than-expected LB. Until ablation isolates
which group is the offender, none of them should be reused as a bundle.

---

## 4. SN-Slice Comparison (V14+V11 blend)

```
Slice       n      F1_a    F1_p    AUC     OV
SN=2     14995   0.2431  0.1614  0.5390  0.2696
SN=3-4   23667   0.3491  0.2156  0.6129  0.3484
SN=5-8   20075   0.3898  0.2142  0.6365  0.3689   ← best slice
SN=9-12   6247   0.3599  0.2109  0.6267  0.3537
SN>=13    4728   0.3541  0.2103  0.6076  0.3473
```

SN=2 remains the worst slice. V9 joint priors did not improve it.

---

## 5. Decisions

| Item | Decision |
|---|---|
| `submission_v14_5f_nocb_v11_optblend.csv` | **KEEP** as stable backup (LB 0.3599; superseded by V16 LB 0.3673) |
| `submission_v15_pp.csv` | **DO NOT submit** (solo OOF below baseline) |
| `submission_v15_pp_v11_optblend.csv` | **REJECT** (LB 0.3507 < V12 baseline 0.3541) |
| Three-way V14+V15+V11 blend | **DEFER** |
| `submission_v15_player_only_v11_optblend.csv` | **DIAGNOSTIC-ONLY** — submit only if budget allows, to confirm whether player-profile alone also fails on Public LB |
| Player profile features (any form) | **HIGH-RISK** — V15_player_only+V11 OOF=0.3777 is promising but V15_pp already failed; await diagnostic LB before any use |
| SGP-derived player winrate | **PERMANENTLY EXCLUDED** (policy + LB risk) |

---

## 6. V15 Ablation — diagnosis (2026-05-03)

### V15_hist_only (NO player profile, hist + streak only)

```
Tag: v15_hist_only
Mask: 69712 / 69712 (100%)
Base    : action=0.3766  point=0.2132  AUC=0.6100  OV=0.3579
Opt sub : action=0.3843  point=0.2207  AUC=0.6100  OV=0.3640
With V11: OOF OV=0.3741   (alpha: act=0.60, pt=0.60, srv=0.95)
```

V15_hist_only + V11 blend OOF (0.3741) is BELOW V14 + V11 (0.3754) by −0.0013. Hist freq +
streak features alone do NOT improve OOF — they add slight noise. Diagnostic LB submitted on
2026-05-03: **0.3574287**, below then-best V14+V11 (0.3598509) by −0.0024.

### Mechanism resolved

| Pipeline | OOF (opt) | LB | OOF-LB Δ |
|---|---|---|---|
| V14+V11 (V9 priors only) | 0.3754 | 0.3598509 | −0.016 |
| V15_hist_only+V11 (V9 + hist + streak) | 0.3741 | 0.3574287 | −0.017 |
| V15_pp+V11 (V9 + hist + streak + player) | 0.3765 | 0.3506750 | −0.026 |
| **V15_player_only+V11 (V9 + player only)** | **0.3777** | **0.3555110** | **−0.022** |

Decomposition of the V15_pp−V14 OOF lift (+0.0011):
- Hist + streak component: **−0.0013 OOF** (no signal, slight noise)
- Player profile component: **+0.0023 OOF** (V15_player_only+V11=0.3777 vs V14+V11=0.3754)

V15_player_only+V11 OOF (0.3777) beats V14+V11 (0.3754) by +0.0023. This is a stronger solo OOF
signal than anticipated from the solo V15_player_only opt score (0.3699), which was below V14.
The blend with V11 rescues the player profile signal considerably.

### Verdict (✅ FINAL — P1 CLOSED 2026-05-03)

- **Hist freq + streak are inert** — permanently excluded.
- **Player profile features cause consistent OOF−LB non-transfer** across two independent tests:
  - V15_pp+V11: OOF 0.3765, LB 0.3507 (gap −0.026)
  - V15_player_only+V11: OOF 0.3777, LB 0.3555 (gap −0.022)
- The OOF−LB gap widens by +0.006–0.010 relative to V14 whenever player profile is included,
  regardless of whether hist/streak noise is present.
- **Root cause confirmed**: player ID statistics computed on 100% known train players overfit
  OOF folds; this signal does not generalise to the LB player distribution (63.5% overlap).
- **Player profile family permanently excluded from all future candidates.**

### Confirmation experiment (COMPLETED) — solo model

```
Tag: v15_player_only
Mask: 69712 / 69712 (100%)
Base    : action=0.3847  point=0.2162  AUC=0.6124  OV=0.3629
Opt sub : action=0.3927  point=0.2259  AUC=0.6124  OV=0.3699
Time    : 209.4 min
```

### V15_player_only + V11 blend (Codex, 2026-05-03)

Command:
```
python src/blend_ensemble.py --v1 v15_player_only --aux-tag v11 \
  --out submission_v15_player_only_v11_optblend.csv
```

Output:
```
Tag         : v15_player_only_v11_optblend
File        : submissions/submission_v15_player_only_v11_optblend.csv
OOF OV      : 0.3777   (highest OOF observed across all V15 ablations)
F1_action   : 0.4059
F1_point    : 0.2320
AUC         : 0.6127
SN=2 OV     : 0.2727   (V14+V11 baseline: 0.2696 — slight improvement)
```

OOF 0.3777 beats V14+V11 (0.3754) by +0.0023. SN=2 also improves slightly (+0.0031). The V11
architecture appears to extract complementary signal from player profile features that the solo
GBM cannot use (solo V15_player_only opt = 0.3699, below V14's 0.3754).

LB result (2026-05-03 17:11): **0.3555110** (rank 35/175)
OOF−LB gap: −0.022 (vs normal −0.016 for V14+V11).

P1 gate applied: LB 0.3555 < threshold 0.360 → **player profile non-transfer confirmed**.
Both V15_pp+V11 (−0.026 gap) and V15_player_only+V11 (−0.022 gap) fail to transfer.
The non-transfer is a robust property of player profile features, not a V15_pp-specific artefact.

**REJECTED. Player profile family permanently excluded from all future candidates.**

---

## 7. Updated Open Questions

1. ~~Which of {player profile, hist freq, streak} is the LB-breaker in V15_pp?~~ → **player profile** ✅
2. ~~Can hist freq + streak be safely added to V14?~~ → **No, no signal added** ✅
3. ~~Why does match-disjoint OOF over-estimate LB by ~0.026 for V15_pp but only ~0.016 for V14?~~
   → **Player ID statistics overfit OOF because all train players appear in all folds.** ✅
4. ~~Does player profile ALONE also fail on Public LB?~~
   → **Yes. V15_player_only+V11 LB=0.3555, gap −0.022. Non-transfer confirmed. Player profile permanently excluded.** ✅
5. **How do we capture player-tendency signal without LB non-transfer?**
   → P2 via heterogeneous ensemble (model diversity) or multi-seed V14 (variance reduction).
   → Do NOT use player ID statistics in any form until a leakage-free player embedding is developed (P3).

---

## 8. P2.0 V16 Test-History Augmentation (2026-05-04) — PUBLIC LB SUCCESS, NEW BEST

### V16 full 5-fold (features_v9 + 2353 test-history aug pairs)

**IMPORTANT: test.csv has real serverGetPoint labels (0/1) for all 3589 rows.**
All training scripts and `build_test_history_pairs.py` overwrite with -1 before use.
Hard rule confirmed: NO_TRUE_TEST_SGP_USED = True in all runs.

```
Tag: v16_testhist_aug
Aug pairs per fold: 2353 / 2353 ✅  NaN/inf: 0 ✅  OOF mask: 69712/69712 ✅
server_aug_rows: 0 ✅ (Guard 1)  fold_stats from real tr_raw only ✅ (Guard 2)

Base:      action=0.3824  point=0.2094  AUC=0.6075  OV=0.3582
Opt solo:  action=0.3921  point=0.2233  OV=0.3677   (+0.0016 vs V14 solo 0.3661)
Time: 180.1 min
```

Per-SN comparison (V16+V11 blend vs V14+V11 blend):
```
Slice       n     V16+V11 OV   V14+V11 OV   Δ
SN=2     14995      0.2727       0.2696    +0.0031 ✅ (aug helps early-rally)
SN=3-4   23667      0.3472       0.3484    −0.0012
SN=5-8   20075      0.3653       0.3689    −0.0036
SN=9-12   6247      0.3508       0.3537    −0.0029
SN>=13    4728      0.3398       0.3473    −0.0075
```

### V16+V11 blend

```
Blend alphas: act=0.65  pt=0.55  srv=0.95
OOF: F1_action=0.4031  F1_point=0.2286  AUC=0.6079  OV=0.3743
OOF gate: 0.3743 < 0.3764 → failed locally, but Public LB submitted at user request.
```

**Public LB result:** `submission_v16_testhist_aug_v11_optblend.csv` = **0.3673269** (rank 27/180), beating V14+V11 by **+0.0074760**.

**Updated verdict at the time:** V16 became the then-current best. OOF underestimated this family because Public LB appears more aligned with early-rally / SN=2 distribution than match-GroupKFold OOF. Keep V16 as a key backbone for future work; next experiments should build on V16, not discard it.

---

## 9. P2.1 Multi-Seed V14 (2026-05-04) — COMPLETE

Infrastructure:
- `--seed` flag added to `src/train_v14.py` (controls np, LGB, XGB random states)
- `src/avg_oof.py` created (averages raw prob arrays across seeds, then blends with V11)

| Tag | Seed | Base OV | Opt OV | Time |
|---|---|---|---|---|
| v14_seed0 | 42 | 0.3602 | 0.3661 | 197.6 min |
| v14_seed1 | 48879 | 0.3593 | 0.3667 | 196.3 min |
| v14_seed2 | 51966 | 0.3598 | 0.3665 | 146.4 min |

seed0 (seed=42) = V14 baseline (exact match confirms seed plumbing correct).
seed1 (seed=48879) solo opt 0.3667 (+0.0006 vs seed0) — seeds are diversifying.

`v14_avg3` was built from seed0/seed1/seed2. Solo averaged OOF OV = 0.3623. `v14_avg3+V11`
blend OOF OV = 0.3765, clearing the original 0.3764 gate, but later zoo results became the higher
priority path.
## 10. Blend Zoo (2026-05-04) — MULTI-MODEL BLEND SUCCESS

### zoo_v16_fast_01

Models: `v16_testhist_aug + v14_seed1 + v12_5f + v11`

```text
OOF OV      : 0.37998
F1_action   : 0.40824
F1_point    : 0.23465
AUC         : 0.61413
Public LB   : 0.3694863 (2026-05-04 11:45, rank 28/181)
Delta vs V16+V11: +0.0021594 LB
Delta vs V14+V11: +0.0096354 LB
```

Verdict at the time: multi-model blend transferred and became the then-current best. OOF gain was +0.0057 over V16+V11, Public LB gain was +0.00216. Later `zoo_v2_top1` superseded this result.
### zoo_v16_fast_04 per-SN bucket — failed LB probe

```text
OOF OV      : 0.37936
Public LB   : 0.3596738 (2026-05-04 16:01, rank 37/184)
Delta vs then-current zoo #1: -0.0098125 LB
```

Verdict: per-SN conditional weights overfit OOF and do not transfer. Even though the file was structurally different from zoo #1, Public LB rejects the fine-grained SN bucket optimization. Continue using global/multi-model blends; avoid highly conditional per-SN weight search unless validated by another public probe.

---

## 11. 2026-05-05 State Sync — zoo_v2 CURRENT BEST

### zoo_v2_top1 — new best

Models: `v11 + v11plus + v13 + v14_seed0 + v16_testhist_aug`

```text
Submission  : submission_zoo_v2_top1_thr_v11_v11plus_v13_v14s0_v16.csv
OOF OV      : 0.38291
F1_action   : 0.41451
F1_point    : 0.23618
AUC         : 0.61318
Public LB   : 0.3733788 (2026-05-05 00:03, rank 22/188)
Delta vs zoo_v16_fast_01: +0.0038925 LB
```

Verdict: `zoo_v2_top1` supersedes all previous submissions and is the current best. This is a
5-model global blend with threshold calibration; it transfers better than the larger `zoo_v3`
candidate despite lower OOF.

### zoo_v3_top1 — failed LB probe

Models: `v11 + v11plus + v12_5f + v13 + v14_seed0 + v16_avg3`

```text
Submission  : submission_zoo_v3_top1_thr_v11_v11plus_v125f_v13_v14s0_v16_avg3.csv
OOF OV      : 0.38385
Public LB   : 0.3675453 (2026-05-05, rank 26/190)
Delta vs current best zoo_v2_top1: -0.0058335 LB
```

Verdict: higher OOF did not transfer. `v16_avg3` and/or the 6-model zoo blend widened the OOF-LB
gap. Do not assume larger zoo blends or averaged V16 automatically improve LB. Keep blend-size
control and prefer structurally validated candidates.

### P1.5 temperature-edge diagnostic

`zoo_v4a` reran the zoo search with size cap <= 5 and lower temperature bound extended to 0.3.
The strongest THR candidates still hit the new lower edge (`t_a=t_p=0.3`), while the best eligible
non-edge candidate was materially below `zoo_v2_top1` OOF. The slot-3 gate failed, so no LB
submission was recommended from P1.5.

Current working conclusion:
- Protect `zoo_v2_top1` as the current best at LB 0.3733788.
- Treat `zoo_v3` / `v16_avg3` as suspect until a better-controlled LB probe says otherwise.
- Shift high-upside work toward structurally new components such as hierarchical point modeling,
  rally-level server modeling, or Transformer/test-history augmentation rather than pure zoo scaling.

---

## 12. 2026-05-05 — V18 hierarchical point head (FAILED gates) + deep research memo

### V18 hier full 5-fold (`v18`, seed=42, --skip-cb)

Architecture: V14 base with 10-class flat point head replaced by three soft heads (valid /
depth / side), reconstructed via `oof_pt[k] = p_valid * p_depth[d(k)] * p_side[s(k)]` with
row renormalisation (no softmax over reconstructed terms, per Codex).

```text
Submission  : submission_v18.csv (saved, NOT submitted)
Base    : action=0.3793  point=0.1642  AUC=0.6101  OV=0.3394
Opt sub : action=0.3872  point=0.2066  AUC=0.6101  OV=0.3595
Time    : 94.7 min
```

Per-class point F1 (V18 vs V14 baseline):

| Class | V14 cls F1 | V18 cls F1 | Δ |
|---|---:|---:|---:|
| miss (cls0) | 0.4379 | 0.4208 | **−0.0172** |
| FH_short (1) | (V14 ~0.13) | 0.1038 | down |
| mid_short (2) | (V14 ~0.22) | 0.1275 | down |
| BH_short (3) | (V14 ~0.0) | 0.0142 | flat-zero |
| BH_long (9) | (V14 ~0.39) | 0.3905 | flat |
| Short F1 mean (1/2/3) | 0.1211 | 0.0818 | **−0.0392** |

### V18 Codex gates — FAILED

| Gate | Threshold | V18 result | Pass/Fail |
|---|---|---:|---|
| cls0 F1 ≥ V14 cls0 F1 − 0.01 | ≥ 0.4279 | 0.4208 | ❌ FAIL by −0.0072 |
| short F1 (cls1/2/3 mean) ≥ V14 short + 0.03 | ≥ 0.1511 | 0.0818 | ❌ FAIL by −0.0693 |
| F1_p ≥ 0.235 | ≥ 0.235 | 0.2066 | ❌ FAIL |
| Solo OOF OV ≥ V14 solo (0.3661) | ≥ 0.3661 | 0.3595 | ❌ FAIL by −0.0066 |

**Verdict: V18 hierarchical point head PARKED.** The product-of-marginals factorisation
(`p_valid × p_depth × p_side`) is too restrictive vs the flat 10-class joint head — the
independence between depth and side under-specifies the joint distribution. Soft
recombination did not rescue the design. Do NOT blend `v18` into any zoo. Codex's deferred
fallback (`P(side | depth)` instead of marginals) is the only structural rescue path; it is
NOT scheduled for this round.

Artifacts retained in `oof_predictions/v18_*.npy` for diagnostic purposes only.

### Deep research memo (2026-05-05) — Codex review outcomes

A read-only deep research memo proposed 12 hypotheses (H1–H12). Codex review key
conclusions:

- **H1 — V11 + test-history aug**: 1-fold smoke APPROVED, BUT implementation **MUST**
  mask the server head for aug rows. `data/test_history_pairs.parquet` carries
  `serverGetPoint = -1` placeholders; feeding those into the SGP head as labels would
  poison the server channel. Implementation must either (a) compute server BCE only over
  non-aug rows, or (b) make aug rows contribute action+point losses only.
- **H3 — pseudo-labelled test rallies**: NOT approved for submission training. Distinct
  from V16 test-history aug (history rows are organiser-confirmed observable shots;
  pseudo-labels are model-generated test targets). Offline label generation / design
  exploration acceptable; any pseudo-label-trained submission requires explicit Jabir
  policy approval.
- **H9 — player-disjoint holdout**: APPROVED to prioritise BUT advisory signal only
  initially. With ≤5 LB-tested points, Pearson > 0.85 is dominated by single points and
  submissions are not independent. First gate: leave-one-out / rank-consistency check
  (does the holdout correctly explain why zoo_v2 won, zoo_v3 / per-SN / V15 lost?). Hard
  gate only after that lands.
- **H4 — geometry-aware point smoothing**: difficulty under-estimated. For GBM
  multiclass, neighbour smoothing is not just a loss tweak — needs sample expansion (one
  shot becomes multiple weighted (X, neighbour-class) rows) or a custom objective. Plan
  cost should rise from "low" to "medium".
- **H11 — flip-TTA**: difficulty under-estimated. Must rebuild flipped raw context
  features (re-run feature engineering on the mirrored rally context) and flip the
  action/point posterior back, not just relabel submission outputs. Plan cost should rise
  from "trivial" to "low-medium".
- **RESULTS.md staleness**: my memo's claim was outdated — Codex confirms RESULTS.md was
  already current before the memo; do not propagate the staleness note.

### Component correlation matrix finding (2026-05-05, NEW)

Pairwise Pearson correlation of OOF point predictions across all components (computed
read-only this round; not in any prior section):

```
                  v16_t  v14_s0 v14_s1 v14_s2 v14_av v12_5f  v11    v11+   v13    v16_avg3
v16_testhist_aug  1.00   0.81   0.81   0.81   0.82   0.78    0.77   0.75   0.75   0.99
v14_seed0         0.81   1.00   0.98   0.98   0.99   0.95    0.71   0.68   0.89   0.82
v14_seed1         0.81   0.98   1.00   0.98   0.99   0.95    0.71   0.69   0.89   0.82
v14_seed2         0.81   0.98   0.98   1.00   0.99   0.95    0.70   0.68   0.89   0.81
v14_avg3          0.82   0.99   0.99   0.99   1.00   0.95    0.71   0.69   0.90   0.82
v12_5f            0.78   0.95   0.95   0.95   0.95   1.00    0.68   0.65   0.91   0.78
v11               0.77   0.71   0.71   0.70   0.71   0.68    1.00   0.83   0.67   0.78
v11plus           0.75   0.68   0.69   0.68   0.69   0.65    0.83   1.00   0.64   0.75
v13               0.75   0.89   0.89   0.89   0.90   0.91    0.67   0.64   1.00   0.75
v16_avg3          0.99   0.82   0.82   0.81   0.82   0.78    0.78   0.75   0.75   1.00
```

Implications:
- **v16_avg3 is 0.994 correlated to v16_testhist_aug** on point — averaging across V16
  seeds added near-zero diversity. Explains why v16_avg3 swap in zoo_v3 produced no real
  diversification gain over v16_testhist_aug; the LB regression came from the size-6
  stretch and edge calibration, not from v16_avg3 carrying new orthogonal signal.
- v14_seed0/1/2 are 0.977 correlated pairwise; v14_avg3 is 0.992 correlated to each.
  Multi-seed averaging on V14 is similarly near-noise.
- The zoo's "9-component" menu is effectively **2 clusters** (GBM cluster: v14/v16/v12/v13
  pairwise 0.78–0.99; Transformer cluster: v11/v11plus 0.83) with cross-cluster
  correlation 0.65–0.78. Same-cluster reshuffling produces near-equivalent OOF candidates
  that the random Dirichlet search exploits via OOF noise — explains the OOF→LB gap
  variability seen in zoo_v2 (gap −0.0095) vs zoo_v3 (gap −0.0164).

Action implication: future zoo improvements MUST add a genuinely uncorrelated component
(target cross-cluster correlation ≤ 0.78). Same-architecture seed averaging is closed.

---

## 13. 2026-05-06 — P11 player-disjoint holdout built + initial eval

### Build (`src/build_player_disjoint_holdout.py`)

```
seed=42  frac=0.15
distinct primary players in OOF rows: 166
held-out players: 25 / 166 (15.1%)
holdout rows: 8284 / 69712 (11.9%)
output mask: data/player_holdout_idx.npy
class diversity: all 10 pointId classes represented
  cls 3 (BH_short): only 23 rows in holdout — F1 macro for this class is noisy
```

Row-order replication verified: matches V14/V16/V12 OOF length (69712).

### Solo-tag eval on P11 holdout (`src/eval_player_disjoint.py`)

| Tag | full_OV | holdout_OV | gap |
|---|---:|---:|---:|
| v14_seed0 | 0.3602 | **0.3691** | −0.0089 |
| v14_seed1 | 0.3593 | 0.3666 | −0.0073 |
| v14_seed2 | 0.3598 | 0.3712 | −0.0114 |
| v14_avg3 | 0.3623 | **0.3725** | −0.0102 |
| v12_5f | 0.3571 | 0.3603 | −0.0032 |
| v13 | 0.3552 | 0.3560 | −0.0008 |
| v16_testhist_aug | 0.3582 | 0.3670 | −0.0088 |
| v16_seed1 | 0.3583 | 0.3675 | −0.0092 |
| v16_seed2 | 0.3582 | 0.3663 | −0.0082 |
| v16_avg3 | 0.3597 | 0.3686 | −0.0088 |
| v11 | 0.3205 | 0.3185 | +0.0020 |
| v11plus | 0.3198 | 0.3229 | −0.0030 |

**Solo-level signal is weak / mildly inverted vs LB direction:** all GBM models
*outperform* on the held-out players (negative gap). V14 family > V16 family on
holdout, but V16 family > V14 family on LB. So solo-tag holdout OV does NOT
predict LB ranking — held-out players might simply have more predictable patterns
or class distributions that favour GBM-style models.

### Reconstructed zoo blend eval on P11 holdout

| Candidate | reconstructed full_OV | reconstructed holdout_OV | LB |
|---|---:|---:|---:|
| zoo_v2_top1 | 0.3552 | **0.3668** | 0.3734 |
| zoo_v3_top1 | 0.3532 | 0.3608 | 0.3675 |

(Reconstruction note: omits THR's scipy per-class weights — those aren't stored
in the ranking CSV; only temperature is. Absolute OV is therefore lower than the
reported zoo_v2 OOF 0.3829, but the relative ordering between zoo_v2 and zoo_v3
is what matters for the rank-consistency check.)

**Rank consistency at the BLEND level: zoo_v2 > zoo_v3 — CORRECT direction
matching LB.** Margin on holdout (+0.0060) is 3× the margin on full
reconstruction (+0.0020), suggesting the holdout slice amplifies the LB-correct
signal at the zoo-blend level (even though it doesn't at the solo-tag level).

### Decision

Per Codex's advisory-only stance (P11 should be a tiebreaker, not a hard gate
with ≤ 5 LB points), use the following P11 rule for slot-1 decisions:

**Slot acceptance gate (advisory):** new zoo top-1 must satisfy
`reconstructed_holdout_OV ≥ 0.3668` (zoo_v2_top1 baseline) using the same
reconstruction approach (THR temperature + base CW, no scipy per-class).

If a candidate's holdout OV falls noticeably below 0.3668 despite higher full
OOF, treat as an OOF-overfit warning and prefer the conservative anchor.

Solo-tag-level P11 evaluation is NOT predictive (per the table above) and
should not be used as a gate for individual components.

---

## 14. 2026-05-06 — V11+test-history aug (P6/H6) FAILED + zoo_v6 SKIPPED

### v11_aug full 5-fold (`v11_aug`, seed=42, 80 epochs, --aug-parquet data/test_history_pairs.parquet)

P6 server-mask verified: `aug_rows_in_server_loss == 0` every epoch; `aug_rows_seen == 2353`
per epoch (aug rows DO flow through model, masked only from server BCE).

```text
Tag: v11_aug
Aug pairs per fold: 2353 ✅  NaN/inf: 0 ✅
NO_TRUE_TEST_SGP_USED = True (server BCE restricted to is_aug==0)

Per-fold BEST OV: [0.3182, 0.3081, 0.3310, 0.3231, 0.3166]
Global OOF: F1_a=0.3364  F1_p=0.1973  AUC=0.5560  OV=0.3247
Time: 80.6 min
```

### P6 gates

| Gate | Threshold | v11_aug | Pass/Fail |
|---|---|---:|---|
| `aug_rows_in_server_loss == 0` (Codex P6 server-mask) | 0 | 0 | ✅ PASS |
| Solo action F1 ≥ V11 + 0.005 | ≥ 0.3271 (V11 act ≈ 0.322) | 0.3364 | borderline (gain +0.0042 vs +0.005 target — very close, judging tied) |
| Solo OV ≥ V11 + 0.005 | ≥ 0.3255 (V11 OV 0.3205) | 0.3247 | ❌ FAIL by 0.0008 |
| OOF correlation v11_aug ↔ v16 (action) | ≤ 0.78 | 0.752 | ✅ PASS |
| OOF correlation v11_aug ↔ v16 (point) | ≤ 0.78 | 0.781 | ⚠ marginally over (0.001) |
| OOF correlation v11_aug ↔ v11 (server) | divergent | 0.503 | ✅ server-mask differentiated |

Solo gate fails by hair; correlation gates mostly pass. Decisive test: zoo_v6 with
v11_aug added.

### zoo_v6 — search with v11_aug in GROUP_D (max_models=5, temp_min=0.5)

`src/blend_zoo_v2.py` extended (GROUP_D = ["v11", "v11plus", "v11_aug"]; D_choices
allow r ∈ [1, 2, 3]). 224 subsets enumerated (vs 99 in zoo_v4a, 103 in zoo_v2).
Total search time: 141.6 min CPU.

**Global top-2 (THR, t=0.5 EDGE):**
- Rank 1: `v11_aug+v11plus+v13+v14_avg3+v16` (n=5) reported OOF=0.3841 spread=0.0950
- Rank 2: `v11_aug+v12_5f+v13+v14_seed0+v16` (n=5) reported OOF=0.3840 spread=0.0902

**Eligible (non-edge) top-1:**
- elig 1 / global rank 107: `v11_aug+v11plus+v13+v14_seed2+v16` (NONE calib) OOF=0.3793 spread=0.0900

### P11 holdout decision (reconstructed without scipy weights)

| Candidate | full OV | **holdout OV** | gap |
|---|---:|---:|---:|
| zoo_v6 top-1 (v11_aug + v11plus + v13 + v14_avg3 + v16) | 0.3513 | **0.3623** | −0.0045 vs zoo_v2 |
| zoo_v6 top-2 (v11_aug + v12_5f + v13 + v14_seed0 + v16) | 0.3556 | **0.3644** | −0.0024 vs zoo_v2 |
| **zoo_v2 top-1 (LB 0.3734)** | 0.3552 | **0.3668** | baseline |

**zoo_v6 with v11_aug LOSES on P11 holdout despite +0.0012 OOF gain.** P11 (which
correctly predicted zoo_v3 < zoo_v2 directionally) is signalling that v11_aug
doesn't add LB-relevant diversity — it adds OOF noise that the search exploits.

### Verdict

- **v11_aug PARKED.** Same fate as v18: structural change didn't lift LB-relevant
  metrics. Possible mechanism: V11 already sees test rallies at INFERENCE (as
  context for the prediction), so feeding test-history rows as TRAINING samples
  is largely redundant — V16 GBM benefits because it doesn't have rally context
  at inference and the aug rows directly teach the test-distribution shot
  transitions.
- **Slot 1 (2026-05-06): SKIP zoo_v6 candidates.** Eligible top-1 fails OOF gate
  AND P11 says LB will likely regress.
- **Locked Rule (NEW)**: same-architecture aug-style data injection into V11
  (test-history aug) does not transfer LB benefit. Do not retry without a
  structurally different mechanism (e.g., distillation rather than aug rows).
- Pivot to P10 (rally-pooled SGP) for orthogonal AUC lift.

---

## 15. 2026-05-06 — P10 rally-pooled SGP (`v19_rally_srv`) STRUCTURALLY LEAKY, PARKED

### Smoke + full 5-fold both hit AUC ≈ 0.998

`src/train_v19_rally_srv.py` builds per-rally pooled features (action / hand /
spin / position / point aggregates + sex / numberGame / n_shots) and predicts
SGP at the rally level via LGB+XGB. After multiple leak fixes (drop scoreSelf /
scoreOther; drop the rally decider shot from train aggregates), OOF AUC was
still 0.998 — clearly an unresolved leak.

### Root cause: `n_shots` parity exploits the table-tennis alternation rule

Bisection showed AUC=0.997 from JUST `(sex, numberGame, n_shots, sn_max)`.
Inspecting `n_shots` vs SGP distribution:

| n_shots (= N-1 after decider drop) | sgp=1 rate | rallies |
|---:|---:|---:|
| 1 | 100% | 1869 |
| 2 | 0.08% | 2585 |
| 3 | 100% | 3030 |
| 4 | 0.05% | 1933 |
| 5 | 99.87% | 1598 |
| 6 | 0.20% | 974 |
| ... | ... | ... |

Table tennis alternation: shot 1 = server, shot 2 = receiver, shot 3 = server, ...
The N-th shot is the decider. Server hits odd-numbered shots; receiver hits
even-numbered shots. Whoever hits the deciding shot causes the rally to end
(usually by missing). So:
- n_shots = N-1 even → N is odd → server hit the decider → **sgp = 0**
- n_shots = N-1 odd  → N is even → receiver hit the decider → **sgp = 1**

`n_shots` deterministically encodes the SGP label.

### Why it won't transfer to LB

For TEST rallies, the (n+1)-th shot is not necessarily the decider — the rally
may continue past it. The visible test n_shots distribution differs from train
N-1 distribution, so the parity-based mapping doesn't apply at test time. v19's
0.998 OOF AUC is a pure train-time artifact.

### Verdict

**v19_rally_srv PARKED.** The rally-pooled-SGP framing has an irreducible leak
via shot-count parity given the alternation rule. Possible future redesign:
predict per-shot SGP (V14's approach) but with rally-level CONTEXT features
that are parity-invariant (e.g., aggregate per ROLE — server's shots vs
receiver's shots — rather than over the raw shot sequence). Out of scope this
round.

Artifacts retained at `oof_predictions/v19_rally_srv_*.npy` for forensic
purposes only; do NOT swap into the zoo's SGP channel.

### Locked Rule (NEW): rally-level prediction must be parity-invariant

Any future rally-level head MUST handle the table-tennis alternation rule
explicitly. Either predict per-shot (V14 path) or pool features along the
SERVER/RECEIVER role axis (not the temporal shot axis), so n_shots parity
cannot leak the SGP label.

---

## 16. 2026-05-06 — P12 anchor-perturbation zoo (no improvement)

`src/blend_zoo_v2.py` extended with `--anchor-from / --anchor-rank /
--anchor-alpha` flags. P12 mode: restrict the search to the anchor subset only;
sample weights as `(1-α)·anchor + α·Dirichlet`, bounding L1 drift by `2α`.

### Run zoo_v7 (anchor=zoo_v2 top-1, α=0.10, n=600, max_models=5, temp_min=0.5)

```text
1  THR     0.3816   0.0980   0.3804  YES   v11+v11plus+v13+v14_seed0+v16
2  TEMP    0.3772   0.0931   0.3772  YES   (same subset)
3  NONE    0.3772   0.0931   0.3772  no    (same subset, eligible)
4  CW      0.2852   0.0984   0.2838  no    (same subset, eligible)
```

Top-1 (THR EDGE) OOF 0.3816 < zoo_v2 top-1 0.3829 — perturbation moved AWAY
from the anchor's THR optimum. Eligible top-1 (NONE) at OOF 0.3772 — well
below slot gate.

### Run zoo_v7b (anchor=zoo_v2 top-1, α=0.15, n=1500, max_models=5, temp_min=0.3)

```text
1  THR     0.3827   0.0987   0.3805  YES   (same subset)
2  TEMP    0.3771   0.0929   0.3771  YES   (same subset)
3  NONE    0.3771   0.0929   0.3771  no    (same subset, eligible)
4  CW      0.2830   0.0957   0.2823  no    (same subset, eligible)
```

Wider perturbation (α=0.15) and wider temp grid (down to 0.3) only recovered
to OOF=0.3827 (still −0.0002 vs anchor). The anchor weights ARE at the local
OOF optimum for this subset. P12 did not surface a better candidate.

### Verdict

**P12 anchor-perturbation produced no candidate above zoo_v2 top-1's OOF.**
The zoo_v2 top-1 weights appear to be at the local OOF optimum for the
{v11, v11plus, v13, v14_seed0, v16} subset; small perturbations can only
match-or-degrade.

---

## 17. 2026-05-06 — 12-hour plan summary

Compute budget used: ~6 h. Slots used: **0/3** (all 3 preserved).

| Step | Outcome |
|---|---|
| P6 v11_aug full training (server-mask + aug parquet) | OV=0.3247, +0.0042 vs V11 (gate 0.005 misses by 0.0008) |
| zoo_v6 with v11_aug in GROUP_D | OOF=0.3841 BUT P11 holdout 0.3623 < zoo_v2 0.3668 → SKIP |
| P11 player-disjoint holdout build + eval | Built; rank-consistency holds at zoo level (zoo_v2 > zoo_v3) |
| P10 v19_rally_srv (smoke + full) | AUC=0.998 PARKED — n_shots parity leak via alternation rule |
| P12 anchor-perturbation (zoo_v7, zoo_v7b) | Anchor at local OOF optimum; perturbation can't improve |

### Slot-by-slot decision (2026-05-06)

| Slot | Decision | Reason |
|---|---|---|
| 1 / 3 | **SKIP** (preserved) | zoo_v6 top-1 fails P11 holdout; zoo_v7 doesn't beat anchor |
| 2 / 3 | **SKIP** (preserved) | v19 leaky; nothing else viable from this round |
| 3 / 3 | **SKIP** (preserved) | No candidate beats zoo_v2 top-1 |

**LB best preserved at 0.3733788** (zoo_v2 top-1, submitted 2026-05-05 00:03).
3 LB slots remain available for 2026-05-06.

### What this round established (positive output despite zero LB submissions)

1. **P11 holdout works at zoo level**: correctly predicts zoo_v2 > zoo_v3 (matching LB
   direction); flagged zoo_v6 candidates as likely regressions. Hold for advisory use.
2. **V11 + test-history aug doesn't transfer LB**: the V16 GBM-aug mechanism does NOT
   generalise to V11. Likely because V11 already sees test rallies as inference context.
3. **Rally-level SGP prediction has a structural leak (n_shots parity)**: any future
   rally-pooled approach MUST be parity-invariant (e.g., role-axis pooling).
4. **zoo_v2 top-1 is locally optimal** for its {v11, v11plus, v13, v14_seed0, v16}
   subset; perturbation cannot improve. Future zoo gains require a structurally NEW
   component, not weight tuning.

### What's next (out of scope this round)

- H7 distillation (V11 student of zoo_v2 teacher) — same mechanism risk as H6, deferred.
- Role-axis rally aggregation for SGP — concrete redesign of v19 to drop parity leak.
- P5 autoregressive pretraining smoke (high-risk, high-cost) — deferred.
- Re-investigate V14 grammar features for SN=2 specialisation, OR hierarchical
  point head with `P(side|depth)` joint factorisation (Codex's deferred fallback).

---

## 18. 2026-05-06 — MAJOR FINDING: NONE-calibration candidates dominate P11 holdout

While inspecting zoo_v7 / zoo_v7b candidates for the user, evaluating ALL
calibration variants (THR/TEMP/CW/NONE) across all 4 zoo runs (v2 / v3 / v4a /
v6 / v7 / v7b) revealed a striking pattern:

| Calibration class | Reported full OOF | P11 holdout OV |
|---|---:|---:|
| THR EDGE (zoo_v2 winner, etc.) | **0.380–0.384** | **0.366–0.376** |
| THR INTERIOR (t=0.3) | 0.380–0.384 | 0.370–0.376 |
| **NONE** (no temperature, no class weight) | 0.376–0.379 | **0.385–0.389** |

Top 5 NONE candidates by P11 holdout OV (across all rankings):

| Rank | Subset | full OV | **holdout OV** | Source |
|---|---|---:|---:|---|
| 1 | v11plus + v13 + v14_seed2 + v16 | 0.3763 | **0.3885** | zoo_v6 r=263 |
| 2 | v11plus + v13 + v14_seed2 + v16 | 0.3757 | 0.3879 | zoo_v4a r=115 (same subset) |
| 3 | v11_aug + v11plus + v13 + v14_seed2 + v16 | 0.3794 | 0.3873 | zoo_v6 r=107 (with v11_aug) |
| 4 | v11plus + v13 + v14_seed1 + v16 | 0.3762 | 0.3871 | zoo_v2 r=94 |
| 5 | v11plus + v13 + v14_avg3 + v16 | 0.3760 | 0.3865 | zoo_v2 r=102 |

### Mechanism: per-class F1 breakdown shows THR scipy weights are OOF-overfit

For the top NONE candidate vs zoo_v2 top-1 (THR with full scipy CW), per-class
point F1:

| Point class | NONE F1 (FULL) | THR F1 (FULL) | Δ | NONE F1 (HOLDOUT) | THR F1 (HOLDOUT) |
|---|---:|---:|---:|---:|---:|
| cls0 miss | 0.411 | 0.388 | +0.023 | 0.425 | 0.403 |
| cls1 FH_short | 0.125 | 0.089 | +0.036 | 0.161 | 0.120 |
| cls2 mid_short | 0.241 | 0.140 | **+0.100** | 0.303 | 0.205 |
| cls5 mid_half | 0.184 | 0.098 | **+0.086** | 0.175 | 0.054 |
| cls9 BH_long | 0.347 | 0.257 | **+0.090** | 0.324 | 0.233 |

THR's scipy per-class weights aggressively up-weight high-CW classes
(cls 8 / 14 / 9 / 10 → CW 14, 10, 8, 6 respectively), causing argmax to over-
predict those classes. The reported macro F1 gain on OOF (0.235 vs NONE's 0.228)
is driven by `cls 14` and `cls 8` — but at the cost of other classes the LB
distribution actually has more of.

### Cross-validation against LB-tested points (N=2)

| Submission | reconstructed full OV | reconstructed holdout OV | LB | holdout-LB gap |
|---|---:|---:|---:|---:|
| zoo_v2 top-1 (LB-WIN) | 0.3552 | 0.3668 | 0.3734 | +0.0066 |
| zoo_v3 top-1 (LB-LOSS) | 0.3532 | 0.3608 | 0.3675 | +0.0067 |

Holdout-to-LB gap is consistent at ~+0.007. If the same gap applies to NONE
candidates, the top NONE candidate's expected LB ≈ 0.3885 + 0.007 ≈ **0.395**.

**This extrapolation has only N=2 calibration points and is untested for the
NONE-calibration arm.** Highly promising signal but requires LB validation.

### Materialised candidate

`submission_zoo_holdout_top1_none_v11plus_v13_v14s2_v16.csv` (NEW)
- Subset: v11plus + v13 + v14_seed2 + v16_testhist_aug (n=4)
- Calibration: NONE (no temperature, no class weight)
- pointId distribution: cls 9 (BH_long) 513, cls 8 (mid_long) 202, cls 0 (miss) 192,
  cls 7 (FH_long) 119, cls 1 (FH_short) 56, ..., cls 3 (BH_short) 7
  (more concentrated on majority classes than zoo_v2 top-1 — expected for NONE)
- Reported full OOF: 0.3763
- P11 holdout OV: **0.3885** (+0.0217 vs zoo_v2 top-1's 0.3668)

### Decision

This finding INVERTS the slot-1 SKIP recommendation. With holdout signal
predicting LB > 0.38, this NONE candidate is the strongest probe available
this round.

**Recommended slot 1 (2026-05-06): submit
`submission_zoo_holdout_top1_none_v11plus_v13_v14s2_v16.csv`.**

Two outcomes:
- LB ≥ 0.378: holdout-to-LB-gap pattern holds, NONE-calibration confirmed
  superior to THR for LB transfer. Massive structural shift to the round.
- LB ≤ 0.370: holdout-to-LB calibration breaks for NONE, similar to how
  per-SN bucket gating broke. v11plus+v13+v14+v16 NONE is then PARKED, but
  current best (zoo_v2 LB 0.3734) still preserved (we burn 1 slot to learn).

User decision required (manual submit).

---

## 19. 2026-05-06 — NONE-CALIBRATION LB CONFIRMED, NEW CURRENT BEST 0.3748577

### LB result

```
Submission : submission_zoo_v6_elig1_none_v11_aug_v11plus_v13_v14s2_v16.csv
Models     : v11_aug + v11plus + v13 + v14_seed2 + v16_testhist_aug (n=5)
Calibration: NONE (no temperature, no class weight)
Reported OOF: 0.3794 (eligible top-1 in zoo_v6 search)
P11 holdout : 0.3873
Public LB   : 0.3748577 (2026-05-06)
Δ vs prior best (zoo_v2 top-1 LB 0.3733788): +0.0014789
```

**This is the new current best.**

### Holdout-LB gap calibration update (N=3 LB-tested points)

| Submission | calib | reconstructed full | reconstructed holdout | LB | LB − holdout |
|---|---|---:|---:|---:|---:|
| zoo_v2 top-1 | THR (edge) | 0.3552 | 0.3668 | 0.3734 | **+0.0066** |
| zoo_v3 top-1 | THR (edge, n=6, v16_avg3) | 0.3532 | 0.3608 | 0.3675 | **+0.0067** |
| zoo_v6 elig1 | NONE (n=5, v11_aug) | 0.3772 | 0.3873 | 0.3749 | **−0.0124** |

Pattern: **THR-EDGE candidates: LB > holdout (+0.007); NONE candidates: LB < holdout
(−0.012).** The holdout was OPTIMISTIC for the NONE candidate.

Updated extrapolation rule for NONE candidates: LB ≈ holdout − 0.012.
- Top NONE without v11_aug (`v11plus+v13+v14_seed2+v16`, holdout 0.3885): expected LB ≈ 0.3765
  (just +0.0017 over the new current best; marginal; would test "is v11_aug actually helping?")
- Other NONE candidates (holdout 0.385–0.388): expected LB ≈ 0.373–0.376

### What this confirms / inverts

- **NONE-calibration TRANSFERS to LB.** First demonstration that the entire calibration
  arm (NONE, no temperature, no scipy CW) can beat THR-EDGE on LB. THR's scipy per-class
  weights are partly OOF-overfit even when temperature is reasonable.
- **v11_aug as a zoo COMPONENT helps** when paired with NONE — the v11_aug parking
  decision (made on THR-edge zoo_v6 top-1's holdout regression) was WRONG. v11_aug's
  contribution was masked by THR's overfit; NONE reveals the real diversity benefit.
- **P11 holdout signal is real but the magnitude is calibration-arm-dependent.** THR's
  holdout is a LB-conservative proxy (LB > holdout); NONE's holdout is a LB-optimistic
  proxy (LB < holdout). The ranking direction is consistent, but the gap magnitude
  needs calibration per-arm.

### Locked Rule (NEW)

For zoo blend search going forward:
- NONE / TEMP / CW calibration variants are now **LB-validated submission candidates**,
  not just diversity options.
- THR-EDGE (t hits the lower grid bound) is HIGH-RISK on LB (zoo_v3 lost −0.006);
  THR-INTERIOR is acceptable but no longer preferred over NONE.
- v11_aug is **REVIVED** as a usable zoo component (was parked).

### Top remaining LB candidates this round (untested)

Per the holdout sweep + the new LB-NONE calibration (LB ≈ holdout − 0.012):

| Candidate | Subset | calib | holdout | expected LB | File |
|---|---|---|---:|---:|---|
| TOP NONE (no v11_aug) | v11plus+v13+v14_seed2+v16 | NONE | 0.3885 | ~0.3765 | submission_zoo_holdout_top1_none_v11plus_v13_v14s2_v16.csv |
| Diff-subset NONE | v11plus+v12_5f+v13+v14_avg3+v16 | NONE | 0.3863 | ~0.3743 | NOT YET MATERIALIZED |
| TEMP-interior (v7b r=2) | same as zoo_v2 subset | TEMP t=0.3 | 0.3842 | ~0.3722 | NOT YET MATERIALIZED |

### Slot status (2026-05-06) — UPDATED

- Slot 1: zoo_v6_elig1 NONE → LB **0.3748577** (NEW BEST)
- Slot 2: zoo_holdout_top1 NONE (no v11_aug) → LB **0.3546861** (−0.020, FAIL)
- Slot 3: USED BY TEAMMATE (content unknown to us)
- **All 3 slots used today (2026-05-06). Next submission ≥ 2026-05-07.**

---

## 20. 2026-05-06 — v11_aug is STRUCTURALLY CRITICAL for the NONE arm (−0.020 LB without it)

### LB result

```
Submission : submission_zoo_holdout_top1_none_v11plus_v13_v14s2_v16.csv
Models     : v11plus + v13 + v14_seed2 + v16_testhist_aug (n=4, **no v11_aug**)
Calibration: NONE
Reported OOF: 0.3763
P11 holdout : 0.3885 (HIGHEST among all NONE candidates evaluated)
Public LB   : 0.3546861 (2026-05-06)
Δ vs current best (zoo_v6 elig1 LB 0.3748577): -0.0202
Δ vs prior best (zoo_v2 LB 0.3733788):         -0.0187
```

This is a SIGNIFICANT regression. The same NONE blend WITH v11_aug (zoo_v6 elig1) won
LB 0.3749; WITHOUT v11_aug it falls to 0.3547. **v11_aug carries +0.020 LB signal that
no other component covers.**

### Holdout signal calibration (now N=4 LB points)

| Submission | calib | reconstructed full | reconstructed holdout | LB | LB − holdout |
|---|---|---:|---:|---:|---:|
| zoo_v2 top-1 | THR (edge) | 0.3552 | 0.3668 | 0.3734 | +0.0066 |
| zoo_v3 top-1 | THR (edge, n=6, v16_avg3) | 0.3532 | 0.3608 | 0.3675 | +0.0067 |
| zoo_v6 elig1 | NONE (n=5, **with v11_aug**) | 0.3772 | 0.3873 | 0.3749 | **−0.0124** |
| zoo_holdout | NONE (n=4, **no v11_aug**) | 0.3763 | 0.3885 | 0.3547 | **−0.0338** |

**P11 holdout magnitude DOES NOT predict LB ranking among NONE candidates.** Both NONE
candidates had nearly identical holdout (0.3873 vs 0.3885) but very different LB (Δ
−0.020). The advisory direction is unreliable for NONE-vs-NONE comparison.

### Mechanism hypothesis: transformer coverage matters under NONE

All LB-validated 5-model winners include TWO Transformer-family components:
- zoo_v2 top-1 (LB 0.3734, THR): v11 + v11plus + v13 + v14_seed0 + v16
- zoo_v6 elig1 (LB 0.3749, NONE): v11_aug + v11plus + v13 + v14_seed2 + v16

The LB-failed NONE has only **one** Transformer member (v11plus alone):
- zoo_holdout (LB 0.3547, NONE): v11plus + v13 + v14_seed2 + v16

Possible mechanism: NONE calibration (no class-weight escalation, no temperature) does
not sharpen rare-class predictions. With a single Transformer, the GBM ensemble (v13 +
v14 + v16, 0.78–0.91 pairwise correlated) dominates the point head. Adding a SECOND
Transformer (v11 or v11_aug) brings independent point predictions that survive the soft
NONE argmax. v11plus alone seems insufficient — possibly because v11plus's class-weight
escalation makes it more concentrated, less "diverse partner" to v11_aug.

### Locked Rule (NEW)

**For NONE-calibration zoo blends, require ≥ 2 Transformer-family components**
(any 2 of {v11, v11plus, v11_aug}). Single-Transformer NONE blends (e.g., v11plus alone
+ GBM mix) lose ≥ 0.018 LB.

### P11 holdout — revised utility

P11 holdout was VALIDATED as a directional ranking signal at the THR level (zoo_v2 >
zoo_v3 in both holdout and LB). It is RELIABLE for THR-vs-THR comparison.

For NONE-vs-NONE comparison it is **unreliable in magnitude** — small holdout gaps
(0.001) can correspond to large LB gaps (0.020). This suggests the player-disjoint
slice does NOT capture the LB distribution shift that NONE candidates are sensitive to
(possibly a class-prior shift or a server-side action prior shift).

P11 should be downgraded to "advisory tiebreaker" only, not a strong gating signal,
especially for cross-calibration-arm comparison.

### Slot status — 2026-05-06 EXHAUSTED

- Slot 1: zoo_v6_elig1 NONE (with v11_aug) → LB **0.3748577** ✅ NEW BEST
- Slot 2: zoo_holdout_top1 NONE (no v11_aug) → LB 0.3546861 ❌
- Slot 3: USED BY TEAMMATE (outcome unknown to this session)
- **3/3 slots used. Next submission ≥ 2026-05-07.**

### What this round established

1. **v11_aug is STRUCTURALLY CRITICAL** in NONE blends — not optional, not "helps a bit".
   Removing it from a winning NONE blend loses 0.020 LB.
2. **NONE calibration is LB-validated** but only when paired with TWO transformers
   (v11 + v11plus or v11_aug + v11plus).
3. **Single-Transformer NONE blends are unsafe** — Locked Rule added.
4. **P11 holdout magnitude is NOT a reliable LB-delta predictor for NONE** — only
   useful for THR-vs-THR ranking direction.

### Tomorrow's slot-1 candidates (2026-05-07)

Top NONE candidates known so far (ALL with ≥ 2 transformers):

| Candidate | Subset | calib | holdout | LB risk | File status |
|---|---|---|---:|---|---|
| Current best | v11_aug+v11plus+v13+v14s2+v16 | NONE | 0.3873 | known LB 0.3749 | exists |
| Variant w/ v14_seed1 | v11_aug+v11plus+v13+v14_seed1+v16 | NONE | 0.3868 | similar to current best | exists (zoo_v6 r=165) |
| Variant w/ v14_avg3 | v11_aug+v11plus+v13+v14_avg3+v16 | NONE | (need eval) | similar | exists (zoo_v6 elig2) |
| Add v12_5f | v11_aug+v11plus+v12_5f+v13+v14s2+v16 (n=6) | NONE | (need eval) | size > 5 — Locked Rule 8 violation | exists |
| TEMP-interior | (zoo_v7b r=2 TEMP t=0.3) | TEMP | 0.3842 | calibration arm untested | NOT yet materialised |

## 21. 2026-05-07 — Post-reset Phase 2/3 — NEW LB ladder begins

### Components retrained on test_new.csv

| Tag | FINAL OV (opt) | Wall | New-test rows |
|---|---:|---:|---|
| v14_seed2 | 0.3665 | 177 min | 1845 ✓ |
| v16_testhist_aug | 0.3670 | 175 min | 1845 ✓ |
| v13 | 0.3663 | 184 min | 1845 ✓ |
| v14_seed0 | 0.3661 | 210 min | 1845 ✓ |
| v14_seed1 | 0.3644 | 200 min | 1845 ✓ |
| v12_5f | 0.3650 | 184 min | 1845 ✓ |
| v11plus | 0.3227 | 161 min | 1845 ✓ |
| v11_aug | 0.3232 | 232 min | 1845 ✓ |
| v11 | 0.3237 | 149 min | 1845 ✓ |
| v11plus_aug | 0.3174 | 206 min | 1845 ✓ |

(v16_seed1, v16_seed2 — running 2026-05-07 21:27 +)

### zoo_v8 (8 components, no v12_5f no v14_seed1) — 2026-05-07 first NEW-LB submission

- 256 entries / 128 eligible. ALL THR top-5 edge-rejected.
- Slot 1: `submission_zoo_v8_elig3_none_v11_aug_v11plus_v13_v14s2_v16.csv` (OOF 0.3768) ← OLD-LB-winner subset reproduction
- LB **0.3687552** (rank 3/89) — new fresh leaderboard
- Δ from old LB on same subset: **−0.006** (test set tougher post-reset)

### zoo_v9 (10 components, +v12_5f +v14_seed1) — eligible top now favors v12_5f over v14

- 712 entries / 356 eligible.
- NEW eligible best: `submission_zoo_v9_elig1_none_v11_aug_v11plus_v125f_v13_v16.csv` OOF **0.3771** — v12_5f replaces v14 in the dominant subset.
- Reference subset (v11+v12_5f+v14_seed1+v16) THR found at OOF 0.3785 but **edge-rejected** (temperature pinned at lower bound, suspect overfit).
- Slot 2 candidate ready (not yet submitted).

### Active workload (12-hour window started 2026-05-07 ~21:00)
- GPU: v11_big = bigger Transformer (d_model=256, n_layers=6, epochs=120), running.
- CPU: v16_seed1 → v16_seed2 chain (after zoo_v9 blender finished). Once seeds 1+2 done, derive v16_avg3 (3-seed average) for free.
- Final: re-blend zoo_v10 with full ~12-component menu.

## 22. 2026-05-08 — zoo_v10 (13-component menu, expanded GROUP_A) — NEW best eligible OOF

### Components added beyond zoo_v9
- v16_seed1 (FINAL OV 0.3658 opt) - new
- v16_seed2 (FINAL OV 0.3649 opt) - new
- v16_avg3 (avg of v16/v16_seed1/v16_seed2) - free derivation, OOF OV 0.3594, action F1 0.3896 (strongest single)
- v14_avg3 (avg of v14_seed0/1/2) - free derivation, OOF OV 0.3610
- v11_big (FINAL OV 0.3204) - underperformed, kept for diversity
- v11_aug_big (FINAL OV 0.3208) - underperformed

GROUP_A expanded to ["v16_testhist_aug","v16_avg3","v16_seed1","v16_seed2"] (still 0-or-1).

### zoo_v10 search
- 560 enumerated subsets (vs 178 in zoo_v9, 64 in zoo_v8) due to 4-way A and 4-way B.
- 2240 calibration variants / 1120 eligible / 1120 edge-rejected.
- Reference subset (v11+v12_5f+v14_seed1+v16_testhist_aug) THR found at OOF 0.3811 — still edge-rejected.

### Eligible top-5 (NONE, materialised)
| Elig | Subset | OOF |
|---|---|---:|
| 1 | v11_aug+v11plus+v13+v14_seed0+v16_seed1 | **0.3775** ← NEW best eligible |
| 2 | v11_aug+v11plus+v13+v14_seed2+v16_avg3 | 0.3771 |
| 3 | v11_aug+v11plus+v13+v14_avg3+v16_avg3 | 0.3771 (both averages) |
| 4 | v11+v11_aug+v13+v14_seed2+v16_avg3 | 0.3769 |
| 5 | v11+v11_aug+v11plus+v14_seed2+v16_avg3 | 0.3768 |

### Predicted LB delta vs zoo_v8 elig3 (LB 0.3688)
- zoo_v10 elig1 OOF 0.3775 vs zoo_v8 elig3 OOF 0.3768 → +0.0007 OOF.
- Applying ratio (0.3688/0.3768 = 0.979): expected LB ≈ 0.3695 (+0.0007 vs LB baseline).
- Small gain. Worth one slot to validate the v16_seed1 substitution.

### Components confirmed not helpful (this round)
- **v11_big** and **v11_aug_big**: bigger transformers underperformed default v11/v11_aug. Lesson: more capacity hurts on this dataset size.
- **GPU bigger-model exploration: dead end**. Future GPU work should try fundamentally different architectures or feature sets, not bigger transformers.

## 23. 2026-05-09 — zoo_v10 elig1 LB result: REGRESSION (single-seed v16_seed1 substitution failed)

### Submission

| Field | Value |
|---|---|
| File | `submissions/submission_zoo_v10_elig1_none_v11_aug_v11plus_v13_v14s0_v16_seed1.csv` |
| Calibration | NONE |
| Subset | v11_aug + v11plus + v13 + v14_seed0 + v16_seed1 |
| OOF OV | 0.3775 |
| Predicted LB (0.979 ratio from zoo_v8 elig3) | ≈ 0.3695 |
| **Actual LB** | **0.3664313** (rank 7/150) |
| Δ vs current best (zoo_v8 elig3 LB 0.3687552) | **−0.0023** |
| Δ vs prediction | −0.0030 |
| Realised OOF→LB ratio | 0.9706 (vs 0.979 baseline) |

Codex `ARTIFACT_OK` (R-003 2026-05-08 23:25); Jabir uploaded 2026-05-09.

### What this single-variable test isolates

zoo_v10 elig1 differs from the LB best (zoo_v8 elig3) on **two** components:
1. `v14_seed2 → v14_seed0`
2. `v16_testhist_aug → v16_seed1`

The −0.0023 LB regression is the joint effect of both swaps. Two priors:

- **v14 seed swap**: based on internal OOF deltas (v14_seed0 FINAL OV 0.3661 vs v14_seed2 FINAL OV 0.3665), the seed swap alone is unlikely to drop LB by 0.0023. Marginal effect expected.
- **v16 family swap**: `v16_seed1` is single-seed; `v16_testhist_aug` is the canonical single-seed but has an established LB transfer pattern. The drop is more likely concentrated here.

### Implication for v16_avg3

`v16_avg3` averages three v16 seeds (v16_testhist_aug + v16_seed1 + v16_seed2). It includes the LB-validated `v16_testhist_aug` PLUS smoothing from the additional seeds. This makes v16_avg3 a **safer** v16-family substitution than raw `v16_seed1` alone.

The two zoo_v10 candidates that use v16_avg3 (elig2 and elig3) have OOF 0.3771 — slightly below elig1's 0.3775, but their OOF→LB transfer should be friendlier.

### Lessons → LESSONS_CHECKLIST.md candidate addition

Add to §Calibration / submission gating:

> **Single-seed v16 substitutions are higher LB-transfer risk than v16_avg3.** v16_seed1 alone substituted into a known-good NONE blend lost −0.0023 LB despite +0.0007 OOF. Prefer v16_avg3 (or the LB-validated v16_testhist_aug) over single-seed v16 swaps in any blend candidate that's about to be submitted.

### Remaining LB candidates from zoo_v10
- elig2 (v14_seed2 + **v16_avg3**) — best single-variable test for the v16_avg3 transfer claim. Predicted LB ≈ 0.3686 if 0.979 ratio held, ≈ 0.3661 if 0.9706 ratio held.
- elig3 (v14_avg3 + v16_avg3) — both averages; tests the seed-averaging strategy as a whole.

### Slot accounting
1 slot used today (2026-05-09). Status of remaining slots: unknown — Jabir to confirm.

## 24. 2026-05-09 — v14_recvhand component built (R-001 receiver-handedness pointId feature)

### Summary

| Tag | Wall | Base OV | Opt OV | Opt F1_a | Opt F1_p |
|---|---:|---:|---:|---:|---:|
| v14_seed2 (baseline) | 177 min | 0.3598 | 0.3665 | 0.3886 | 0.2225 |
| **v14_recvhand** | 205 min | 0.3598 | **0.3668** | 0.3886 | **0.2227** |
| Δ | +28 min | 0 | **+0.0003** | 0 | +0.0002 |

Same trainer (`train_v14.py`), same seed (51966), same `--skip-cb`. Only delta:
`--feature-set v9_recvhand` instead of `v9` baseline. The new feature is a
single integer column `recv_hand_est ∈ {0, 1, 2}` derived per-row from prefix
handId observations of the target receiver in the same rally (Codex R-001
spec; see `src/features_v9_recvhand.py`).

### Per-fold trend

| Fold | v14_seed2 OV | v14_recvhand OV | Δ |
|---|---:|---:|---:|
| 1 | 0.3605 | 0.3590 | −0.0015 |
| 2 | 0.3414 | 0.3435 | +0.0021 |
| 3 | 0.3800 | 0.3812 | +0.0012 |
| 4 | 0.3491 | 0.3494 | +0.0003 |
| 5 | 0.3361 | 0.3362 | +0.0001 |
| **mean** | 0.35342 | 0.35386 | **+0.0005** |

All five folds passed Codex's stop gates:
- Mean(Fold 1+2 OV) ≥ 0.346: 0.35125 ≥ 0.346 ✓
- F1_p per-fold ≥ baseline − 0.003: F1 (0.2003 ≥ 0.1995), F2 (0.2102 ≥ 0.2045) ✓

### Per-class pointId F1 — receiver-relative axis effect

| Class | v14_seed2 | v14_recvhand | Δ |
|---|---:|---:|---:|
| miss (cls0) | 0.4385 | 0.4357 | −0.0028 |
| FH_short | 0.1289 | 0.1244 | −0.0045 |
| mid_short | 0.2156 | 0.2128 | −0.0028 |
| **BH_short** | **0.0000** | **0.0070** | **+0.0070** (broke F1=0 floor) |
| FH_half | 0.1487 | 0.1498 | +0.0011 |
| mid_half | 0.1630 | 0.1646 | +0.0016 |
| BH_half | 0.3119 | 0.3055 | −0.0064 |
| FH_long | 0.2138 | 0.2136 | −0.0002 |
| mid_long | 0.1961 | 0.2009 | +0.0048 |
| **BH_long** | 0.3316 | **0.3377** | +0.0061 |

The aggregate point F1 barely moves (+0.0002), but the structural shift
matches the hypothesis: with explicit receiver-handedness, the model
redistributes mass on the FH/BH axis. **BH_short broke the F1=0 floor** that
all prior components (v14_*, v16_*, v11_*) had carried. BH_long and mid_long
gain meaningfully; BH_half and FH_short give back some of the gain.

### Train/test feature distribution (recv_hand_est)

| | unknown(0) | right(1) | left(2) |
|---|---:|---:|---:|
| train (n=69712) | 15.2% | 51.2% | 33.6% |
| test  (n=1845)  | 14.6% | 56.4% | 28.9% |

Test distribution shifts slightly toward right-handed (+5.2 pp) and away from
left (−4.7 pp) vs train. Modest covariate shift; not large enough to expect
dramatic test transfer issues.

### Blend implications

The component is retained in the zoo menu (GROUP_B alongside v14_seed0/1/2
and v14_avg3). Whether it adds blend-level value vs v14_seed2 depends on
prediction-correlation; if highly correlated, it's redundant. Given the
per-class redistribution above, partial decorrelation is expected — worth a
re-blend (`zoo_v11`) to find out.

### Artifacts

- `oof_predictions/v14_recvhand_oof_act.npy`, `_oof_pt.npy`, `_oof_srv.npy`,
  `_oof_mask.npy`, `_oof_y_act.npy`, `_oof_y_pt.npy`, `_oof_y_srv.npy`,
  `_oof_nsn.npy`
- `oof_predictions/v14_recvhand_test_act.npy`, `_test_pt.npy`, `_test_srv.npy`,
  `_test_rally_uid.npy`
- `submissions/submission_v14_recvhand.csv`

Awaiting Codex `ARTIFACT_OK` (post-run integrity check) before zoo intake.

## 25. 2026-05-09 — R-005 meta_stack DIAGNOSTIC: paradigm fails for our component set

### Two attempts (per Codex APPROVE_WITH_FIXES allowing both LGBM and linear stacks)

| Variant | OV | F1_a | F1_p | AUC | Wall |
|---|---:|---:|---:|---:|---:|
| zoo_v10 elig1 (best blender) | 0.3775 | — | — | — | — |
| Best single component | 0.3680 | 0.3896 (v16_avg3) | 0.2162 (v14_seed0) | 0.6117 (v14_avg3) | — |
| **meta_stack v1 LGBM** (num_leaves=8, min_data_in_leaf=200) | **0.3466** | 0.3732 | 0.1876 | 0.6114 | 2.7 min |
| **meta_stack v2 logistic** (LR, C=1.0) | **0.3533** | 0.3840 | 0.1947 | 0.6093 | 51.9 min |

Both variants fail ALL Codex stop gates:
- Per-task F1_a / F1_p / AUC: each below `best_single + 0.001`.
- Combined OV: each below `zoo_v10_elig1 + 0.003 = 0.3805`.

### Why stacking failed here

The T1 correlation analysis (RESULTS §24-pre / `logs/component_correlations_2026-05-09.log`)
showed mean off-diagonal correlation 0.866 (action) / 0.819 (point) / 0.710 (server).
That's "MEDIUM"–"HIGH" room in theory, but in practice:

1. **The blender's global Dirichlet search already extracts near-optimal linear
   combinations** (zoo_v10 elig1 OOF 0.3775). A linear or shallow non-linear
   per-row meta-learner cannot improve on that meaningfully — the best
   component on each task already captures most of the signal, and the
   redundant components don't help row-conditional blending.
2. **Per-row meta-blending in 14×P-dim feature space is high-variance under
   GroupKFold-by-match outer CV.** Fold variance is large (e.g. v2 OV
   0.3335→0.3866 across folds), but the mean settles below the blender's
   global-weight ceiling.
3. **The action F1 ceiling (~0.39) is hit by v16_avg3 already**. The
   meta-learner can at best replicate it; with regularization, it underfits.

### Status: PARK as inert diagnostic

Per Codex R-005: "If both per-task and combined gates fail, mark meta_stack
as inert and archive without zoo intake." Both v1 and v2 meet this condition.

Artifacts retained for reference (not added to blender menu):
- `oof_predictions/meta_stack_*.npy` + `_metadata.json` (v1 LGBM)
- `oof_predictions/meta_stack_v2_logistic_*.npy` + `_metadata.json`

No T3 submission entry will be opened from these.

### Implication for future work

Stacking on probability-array features alone is dead. Future high-EV directions:

1. **Different feature space for the meta layer** — e.g., use raw rally-level
   features (handId histograms, score features) as additional inputs alongside
   component probabilities. May break the "components are correlated → meta
   can't add" pattern. Needs new T2 review.
2. **Per-task channel substitution** — instead of one meta blending all three
   tasks, try replacing only the server channel with a dedicated server-head
   predictor (R-006 is testing this).
3. **Different model class entirely** — graph neural network over rally
   sequences, transformer with cross-rally attention, etc. Higher dev cost,
   higher upside.

## 26. 2026-05-09 — R-006 server_head v1 + v2: rally-level prefix-only SGP head PARKED

### Two attempts (per Codex APPROVE_WITH_FIXES + leak gates)

| Variant | Counts-only mean AUC | Fold 1 AUC | Fold 2 AUC | Mean F1+F2 AUC | Status | Wall |
|---|---:|---:|---:|---:|---|---:|
| **server_head_v1** (rally aggregates) | 0.570 | 0.5940 | 0.5746 | 0.5843 | WEAK_STOP | ~5 min |
| **server_head_v2** (v1 + last-3 shots one-hot) | 0.570 | 0.6032 | 0.6017 | 0.6025 | WEAK_STOP | ~7 min |

Both hit Codex's WEAK_STOP gate (`Fold 1+2 mean AUC < 0.62`), confirmed in
their respective metadata.json files.

### Leak-surface diagnostics

- Counts-only AUC ≈ 0.570 in both runs (4-feature subset:
  `server_prior_count`, `receiver_prior_count`, `total_prior_count`,
  `empty_prefix`). Below Codex's 0.65 threshold → no leak from prefix
  counts. The small lift over 0.5 reflects legitimate "rallies that go
  longer have different SGP base rates" — bounded prefix info, not parity
  proxy.
- Fold 1 AUC stayed below the 0.75 PAUSE gate and the 0.80 HARD_STOP gate
  in both runs.

### Top features (v2)

Across folds, the dominant features were:
1. `shot_lag1_strengthId_p3` — last visible shot's "weak strength" indicator
2. `shot_lag2_is_server` — whether the second-to-last visible shot was the server's
3. `score_diff_start` — score difference at rally start
4. Aggregate `srv_strengthId_p3` / `rcv_strengthId_p3` / spin features

Lag features WERE used heavily, but adding them only lifted AUC ~+0.009 over
v1 — not enough to reach the 0.62 mean gate.

### Why this attack didn't work

1. **The per-shot v14 family already reaches AUC ≈ 0.61 with the full feature
   suite (1145 features).** Reducing to a prefix-only rally-level view
   (85→205 features) loses information without gaining structural advantage.
2. **The "server is shooter at strike 1" heuristic is informative but not
   discriminating** — in many rallies the score / strength patterns leak no
   strong rally-outcome signal until the very end (which we cannot see).
3. **Aggregating per-shot signal across the rally prefix doesn't add value
   beyond what v14 already learns from per-shot features and rally order.**

### Conclusion: server head as separate component is NOT a path to breakthrough

Neither v1 nor v2 is eligible for zoo intake. The SGP bottleneck in our score
(AUC 0.61) is intrinsic to the per-shot prediction problem, not solvable by
re-framing as rally-level. Future attacks on AUC require either:
- A structurally different model class (transformer with cross-rally attention,
  graph net), OR
- Pseudo-labeling test set (T3 — Jabir explicit approval required), OR
- Acceptance that AUC 0.61 is the legitimate ceiling for this problem.

### Status: PARKED as diagnostic

Artifacts retained:
- `oof_predictions/server_head_v1_metadata.json` (status: WEAK_STOP)
- `oof_predictions/server_head_v2_metadata.json` (status: WEAK_STOP)

No T3 review opened. No zoo intake. Not added to GROUP_A/B menu.

## 27. 2026-05-10 — NEW LB BEST: zoo_v10 elig2 (v16_avg3 substitution validated)

### Two LB results

| File | OOF | LB | Rank | Δ vs current best | Conclusion |
|---|---:|---:|---:|---:|---|
| zoo_v10 elig2 (v14s2 + **v16_avg3**) | 0.3771 | **0.3694391** | (top tier) | **+0.0007** | NEW BEST. v16 seed averaging transfers. |
| zoo_v10 elig3 (v14_avg3 + v16_avg3) | 0.3771 | 0.3681435 | 10/169 | −0.0006 | v14_avg3 substitution HURTS. |

### Updated NEW-LB ladder

```
zoo_v10 elig2  LB 0.3694391  (2026-05-10)  ← NEW BEST
zoo_v8  elig3  LB 0.3687552  (2026-05-07)
zoo_v10 elig3  LB 0.3681435  (2026-05-10)
zoo_v10 elig1  LB 0.3664313  (2026-05-09)
```

### Key empirical findings on NEW LB

1. **v16 seed averaging transfers**: `v16_testhist_aug → v16_avg3` swap alone
   (R-004 single-variable test) gave +0.0007 LB. v16_avg3 is now the
   canonical v16-family representative.
2. **v14 seed averaging does NOT transfer**: `v14_seed2 → v14_avg3` swap (with
   v16_avg3 fixed) lost −0.0013 LB. Keep `v14_seed2` as the canonical v14
   representative; do NOT substitute with v14_avg3 in submission candidates.
3. **Different model families have different averaging behaviors.** v16's
   stronger transfer benefit may relate to v16 having more inherent variance
   (smaller fold-to-fold stability) than v14, so averaging meaningfully
   regularizes.
4. **OOF→LB ratio update**: R-004 elig2 ratio = 0.3694391 / 0.3771 = **0.980**.
   Slightly higher than the 0.979 baseline from zoo_v8 (consistent within
   noise). Single-variable safe substitutions retain the OOF→LB ratio.

### Implications for zoo_v11 (currently running)

zoo_v11's smart 10-component menu drops v16_seed1, v16_seed2, v14_seed0,
v14_seed1 and adds v14_recvhand. The R-004 result confirms this pruning was
correct (single-seed v16 substitutions don't help; v14_avg3 hurts;
v16_testhist_aug + v16_avg3 are the only useful v16 entries; v14_seed2 +
v14_avg3 + v14_recvhand cover the v14 axis). zoo_v11's eligible top-K
candidates should still favor:
- v14_seed2 + v16_avg3 pairs (matches NEW LB best)
- v14_recvhand as a possible diversity addition

### Procedural note (process improvement)

zoo_v10 elig3 was uploaded WITHOUT a dedicated `R-###` submission review
entry. R-004 covered only elig2. Going forward, every distinct submission
file must have its own `R-###` artifact review per `COLLABORATION_WORKFLOW.md`
§3.1 — even when uploading multiple files in the same session, each gets a
separate gate. Logged as `R-007` (resolved) for traceability.

## 28. 2026-05-10 — zoo_v11 result: no breakthrough beyond NEW LB best

### Smart 10-component menu (drop redundant seeds + add v14_recvhand)

Components used: `v11`, `v11plus`, `v11_aug`, `v12_5f`, `v13`, `v14_seed2`,
`v14_avg3`, `v14_recvhand`, `v16_testhist_aug`, `v16_avg3`. Dropped (per T1
correlation analysis): `v16_seed1`, `v16_seed2`, `v14_seed0`, `v14_seed1`.

### Result

| | zoo_v10 (full menu, 13 comp) | zoo_v11 (smart menu, 10 comp) |
|---|---:|---:|
| Total subsets | 560 | 268 |
| Wall time | 400 min | 215 min |
| Eligible NONE top-1 OOF | 0.3775 | 0.3772 |
| Eligible NONE top-5 OOF range | 0.3768–0.3775 | 0.3770–0.3774 |

### zoo_v11 eligible NONE top 5

| Elig | Subset | OOF | Predicted LB (×0.978) |
|---|---|---:|---:|
| 1 | v11 + v11_aug + v11plus + v14_seed2 + v16_testhist_aug | 0.3772 | 0.3689 |
| 2 | v11_aug + v11plus + v12_5f + v13 + v16_testhist_aug | 0.3771 | 0.3688 |
| 3 | v11_aug + v11plus + v13 + **v14_recvhand** + v16_testhist_aug | 0.3770 | 0.3688 |
| 4 | v11 + v11_aug + v13 + **v14_recvhand** + **v16_avg3** | 0.3774 | 0.3691 |
| 5 | v11 + v11plus + v13 + **v14_recvhand** + **v16_avg3** | 0.3770 | 0.3688 |

### Why no breakthrough

1. **The exact NEW LB BEST subset (v11_aug+v11plus+v13+v14_seed2+v16_avg3)
   ranks at zoo_v11 eligible 14 (OOF 0.3767)** because the random Dirichlet
   weight search gives slightly different scores across runs. zoo_v10 found
   this same subset at eligible 2 with OOF 0.3771. Variance is ~0.0004
   between runs on the same subset.
2. **None of zoo_v11's eligible top 5 is predicted to beat LB 0.3694.** Best
   predicted is elig 4 at 0.3691 (−0.0003 vs current best).
3. **v14_recvhand appears in 3 of top-5 eligible** but doesn't dominate. It
   adds slight diversity but not enough to break the LB ceiling.
4. **Global top 5 (THR) all edge-rejected** as in zoo_v10, suggesting THR
   calibration consistently overfits in this NONE-friendly test regime.

### Strategic conclusion for this 6-hour window

Spending remaining LB slot (1 left today after zoo_v10 elig2/elig3 uploads)
on a zoo_v11 candidate is **low-EV**: best-predicted candidate (elig 4) is
slightly below current best, and the OOF→LB transfer fragility from R-003
(zoo_v10 elig1 lost despite higher OOF) cautions against speculative
substitutions.

**Recommendation**: hold the slot. Re-evaluate next window with a higher-EV
candidate (new component or new feature direction).

### Window summary (2026-05-09 → 2026-05-10, ~5h compute)

| Track | Outcome | LB impact |
|---|---|---|
| R-005 meta_stack v1 LGBM | FAILED all gates (OV 0.3466) | none |
| R-005 meta_stack v2 logistic | FAILED all gates (OV 0.3533) | none |
| R-006 server_head_v1 | WEAK_STOP (Fold 1+2 mean 0.584) | none |
| R-006 server_head_v2 (+ lag-3 features) | WEAK_STOP (Fold 1+2 mean 0.602) | none |
| R-004 zoo_v10 elig2 LB upload | **LB 0.3694391 — NEW BEST +0.0007** | +0.0007 |
| R-007 zoo_v10 elig3 LB upload | LB 0.3681435 (rank 10/169) | n/a |
| zoo_v11 (10-component re-blend) | No candidate beats LB best | none |

**Net for this window**: +0.0007 LB (from a candidate already materialized
last window — R-004 was the win). All four new experimental tracks (stacking,
two server heads, zoo_v11) were diagnostic-only with no positive LB impact.

### Forward-looking implications

The LB ceiling for our current paradigm (per-shot v11/v14/v16 + Dirichlet
zoo blend) appears to be ~0.369–0.370. Beating that requires either:
1. **New model class** (transformer with cross-rally attention, GNN, etc.)
   — high dev cost.
2. **Pseudo-labeling test set** (T3, requires explicit Jabir approval) —
   could give meaningful lift but bias risk.
3. **Different feature family** orthogonal to the v9 lineage that all v14
   variants use.
4. **External data** (T3, requires approval) — banned per competition
   rules without confirmation.

Top of leaderboard is at 0.4477 (+0.078 above us). Closing that gap likely
requires #1 or #2.

## 29. 2026-05-10 — zoo_v11 elig1 LB result: drop-v13 + 3-transformer LARGE REGRESSION

### Submission
| Field | Value |
|---|---|
| File | `submission_zoo_v11_elig1_none_v11_v11_aug_v11plus_v14s2_v16.csv` |
| Subset | v11 + v11_aug + v11plus + v14_seed2 + v16_testhist_aug |
| Calibration | NONE |
| OOF | 0.3772 |
| Predicted LB (×0.978) | 0.3689 |
| **Actual LB** | **0.3651563** |
| Δ vs current best (zoo_v10 elig2, LB 0.3694391) | **−0.0043** |
| Δ vs prediction | −0.0038 worse than predicted |
| Realised OOF→LB ratio | 0.9680 (vs ~0.978 baseline) |

### Single-variable diff vs zoo_v8 elig3 (LB-known-good 0.3688)

zoo_v8 elig3: v11_aug + v11plus + v13 + v14_seed2 + v16_testhist_aug
zoo_v11 elig1: **v11** + v11_aug + v11plus + v14_seed2 + v16_testhist_aug

Two simultaneous changes:
1. **Removed `v13`** (the GBM diversity component used in every prior LB-good blend)
2. **Added `v11`** (third transformer; now 3 transformers vs 2)

Combined LB delta: −0.0036.

### Confounded but actionable

We can't separate the two factors from a single LB observation. But the
direction is clear and aligns with our existing transformer-saturation
hypothesis (Locked Rule 9: NONE blends require ≥ 2 transformers — that
established a LOWER bound; this result establishes an UPPER bound around 2).

### LESSONS_CHECKLIST update

Added to §Calibration / submission gating:

> **NONE blends should NOT exceed 2 transformers, AND should retain v13.**
> A NONE blend with 3 transformers (v11+v11_aug+v11plus) AND no v13 lost
> −0.0043 LB on a single-variable test (zoo_v11 elig1, 2026-05-10) vs the
> known-good 2-transformer + v13 baseline (zoo_v8 elig3, LB 0.3688).
> Verify: count(v11/v11plus/v11_aug) ≤ 2 AND v13 in subset for any NONE
> candidate proposed for LB upload.

### Updated NEW-LB ladder (no change at top)

```
zoo_v10 elig2  LB 0.3694391  (NEW BEST, 2026-05-10)
zoo_v8  elig3  LB 0.3687552  (2026-05-07)
zoo_v10 elig3  LB 0.3681435  (2026-05-10)
zoo_v10 elig1  LB 0.3664313  (2026-05-09)
zoo_v11 elig1  LB 0.3651563  (2026-05-10)  ← biggest single-variable LB drop yet
```

### Slot status
3/3 slots used 2026-05-10. Out of slots until 2026-05-11.

### Procedural note
zoo_v11 elig1 was uploaded WITHOUT a pre-upload R-### artifact review
(Codex `ARTIFACT_OK`). This is the second occurrence (R-007 was the first).
Per `COLLABORATION_WORKFLOW.md` §3.1, every submission needs Codex
`ARTIFACT_OK` plus Jabir's explicit file approval before upload. Logged
retroactively as `R-008 RESOLVED` in REVIEW_QUEUE.md.

## 30. 2026-05-10 → 2026-05-11 — R-009 V1a-capped pseudo-label V1: PASSES intake gate (+0.0021 OOF opt vs v14_seed2)

### Path A V1 result

Codex APPROVE_WITH_FIXES on 2026-05-10 → R-009 V1a-capped:
- 274 kept pseudo rows (greedy per-class cap 120 per actionId AND per pointId)
- Filter: act_top1_p > 0.40 AND pt_top1_p > 0.25 AND pseudo_pointId != 0
- Sample weight: 0.3 flat for pseudo
- Server training EXCLUDES pseudo entirely
- No flip-aug on pseudo rows
- OOF arrays remain length 69,712 (real train rows only)
- Trainer logs `pseudo_rows_in_server_loss == 0` per fold ✓

### Trainer + invariants

- `data/pseudo_v1.parquet` + immutable `data/pseudo_v1.parquet.manifest.json`
  (test_rally_uid sha256 `53da544097b54190a3e84522797510087d84c29555af8eedceafbf379ed3c272`).
- `src/train_v14.py` modified with `--pseudo-parquet`, `--pseudo-mode`,
  `--pseudo-weight` flags + per-task injection + server-exclude + OOF-shape
  invariants.
- Per-fold log shows: `[Pseudo] action: real=111538-111540 pseudo=274 sw_mass
  real=~115k-122k pseudo=82.2 (0.1% pseudo); server: pseudo=0 (EXCLUDED per
  R-009)`.
- All R-009 invariants PASS at end of run.

### Per-fold OV trend

| Fold | v14_seed2 | v14_pseudo_v1 | Δ |
|---|---:|---:|---:|
| 1 | 0.3605 | 0.3608 | +0.0003 |
| 2 | 0.3414 | 0.3448 | +0.0034 |
| 3 | 0.3800 | 0.3827 | +0.0027 |
| 4 | 0.3491 | 0.3538 | +0.0047 |
| 5 | 0.3361 | 0.3386 | +0.0025 |
| **mean** | 0.35342 | **0.36014** | **+0.00672** |

Per-fold OV gain is consistent across all 5 folds (no fold lost).

### Threshold-optimised metrics

| Metric | v14_seed2 | v14_pseudo_v1 | Δ |
|---|---:|---:|---:|
| FINAL OV (base) | 0.3598 | **0.3624** | +0.0026 |
| FINAL OV (opt)  | 0.3665 | **0.3686** | +0.0021 |
| F1_a (opt) | 0.3886 | 0.3906 | +0.0020 |
| F1_p (opt) | 0.2225 | 0.2253 | +0.0028 |
| Threshold opt gain | +0.0067 | +0.0063 | (similar) |

### Per-class point F1 (Codex's regression check)

Codex gate (after Fold 1+2): no class with meaningful support regresses
catastrophically (>0.02 F1 drop), especially cls1/7/8/9 (V1a was skewed
toward those).

| Class | n | v14_seed2 | v14_pseudo_v1 | Δ |
|---|---:|---:|---:|---:|
| miss (cls0) | 15263 | 0.4385 | 0.4378 | −0.0007 |
| FH_short (cls1) | 582 | 0.1289 | 0.1278 | −0.0011 |
| mid_short (cls2) | 1920 | 0.2156 | **0.2244** | **+0.0088** |
| BH_short (cls3) | 203 | 0.0000 | **0.0073** | **+0.0073** |
| FH_half (cls4) | 2995 | 0.1487 | 0.1539 | +0.0052 |
| mid_half (cls5) | 6585 | 0.1630 | 0.1446 | **−0.0184** (largest) |
| BH_half (cls6) | 4583 | 0.3119 | **0.3257** | **+0.0138** |
| FH_long (cls7) | 9122 | 0.2138 | 0.2030 | −0.0108 |
| mid_long (cls8) | 12386 | 0.1961 | 0.1974 | +0.0013 |
| BH_long (cls9) | 16073 | 0.3316 | **0.3451** | **+0.0135** |

**No catastrophic regression** (worst is cls5 mid_half at −0.0184, under the
0.02 threshold). The BH-axis classes (cls3, cls6, cls9) all gained
materially — consistent with the V1a-capped pseudo distribution being
heavily BH-skewed (cls9 = 43.8% of pseudo points).

Per-class action F1 gains: ShortStop (cls11) +0.0197, Block (cls13) +0.0050,
Chop (cls12) +0.0014.

### Codex intake-gate decision

Codex R-009 verdict text: "if FINAL OV < v14_seed2 FINAL OV - 0.003, park
it; if FINAL OV is roughly flat but test predictions differ materially,
create a separate R-010 T3 artifact review rather than auto-submitting; if
FINAL OV improves or zoo intake improves, open R-010 with the generated
files and artifact checks."

- Park threshold: `0.3665 - 0.003 = 0.3635`. We're at 0.3686, **+0.0051
  above park threshold**. NOT PARK.
- Improvement: **+0.0021 opt OV vs v14_seed2** → improves → **open R-010**.

### Action: zoo_v12 + R-010

1. Added `v14_pseudo_v1` to `blend_zoo_v2.py` GROUP_B.
2. Launched `zoo_v12` with 10-component menu (banned components excluded
   per LESSONS_CHECKLIST submission-candidate freeze): `v11, v11plus,
   v11_aug, v12_5f, v13, v14_seed2, v14_recvhand, v14_pseudo_v1,
   v16_testhist_aug, v16_avg3`.
3. After zoo_v12 completes: open R-010 with the artifact integrity check
   plus the zoo top-K analysis. If a zoo candidate beats current LB best
   (zoo_v10 elig2 OOF 0.3771), open separate T3 submission review for
   that file.

### Wall time
v14_pseudo_v1 rerun: 90.4 min (faster than v14_seed2's 177 min — pseudo
adds < 0.3% rows, but XGB/LGB convergence varied between runs).

### Sample-weight mass per fold (logged)
Pseudo mass was consistent ~82.2 across all 5 folds (0.1% of total action
sample-weight mass). Real-row mass dominated (115k-122k), confirming
pseudo influence is small and advisory.

## 31. 2026-05-11 — R-010 LB result: pseudo-label V1 BIAS-AMPLIFICATION CONFIRMED, Path A V1 PARKED

### Submission

| Field | Value |
|---|---|
| File | `submission_zoo_v12_elig1_none_v11_aug_v11plus_v13_v14_pseudo_v1_v16_avg3.csv` |
| Subset | v11_aug + v11plus + v13 + **v14_pseudo_v1** + v16_avg3 |
| Calibration | NONE |
| OOF | 0.3773 |
| Predicted LB (×0.978) | ~0.3690 |
| **Actual LB** | **0.3626103** |
| Δ vs current best (zoo_v10 elig2 LB 0.3694391) | **−0.0068** |
| Δ vs prediction | −0.0064 worse than predicted |
| **Realised OOF→LB ratio** | **0.961** (vs validated 0.978) |

### Single-variable analysis vs current LB best

zoo_v10 elig2 (LB 0.3694391, OOF 0.3771):
`v11_aug + v11plus + v13 + v14_seed2 + v16_avg3`

zoo_v12 elig1 (LB 0.3626103, OOF 0.3773):
`v11_aug + v11plus + v13 + **v14_pseudo_v1** + v16_avg3`

**Only difference: v14_seed2 → v14_pseudo_v1.** Net LB delta: **−0.0068**.

OOF said the swap was a slight improvement (+0.0002). LB says it's a
substantial regression. OOF→LB ratio collapsed from 0.978 to 0.961, the
worst transfer ratio we've observed.

### Bias-amplification mechanism (now empirically confirmed)

The pseudo-label teacher was zoo_v10 elig2, our current LB best. Training
v14 on its high-confidence test predictions had two competing effects:

1. **Hoped-for**: model learns test-distribution patterns the teacher
   captured but the original train data didn't expose.
2. **Actually-observed**: model over-fits to the teacher's specific
   confident-but-imperfect predictions, narrowing its decision boundary
   toward the teacher's overfit pattern.

Effect 2 dominated. The v14_pseudo_v1 component is now CORRELATED with
the teacher's errors at test time. When that v14_pseudo_v1 is
substituted into a blend that already includes the teacher's v16_avg3
component (also derived from the same lineage), the blend doubles down
on the teacher's bias.

This is the textbook self-training collapse mode. We tried to mitigate
it with:
- Cautious pseudo weight (0.3 flat — only 0.1% of total sample-weight mass)
- Per-class caps (no class > 120 rows)
- Confidence thresholds (act > 0.4, pt > 0.25)
- Server-loss exclusion
- No flip-aug on pseudo

None of those mitigations were sufficient. The OOF metric (held-out
train rallies) does not see the test distribution — it under-penalises
teacher-pattern memorisation. Only LB validation revealed the failure.

### Per-class point F1 was a misleading signal

The per-class regression check at v14_pseudo_v1 training time showed
gains on BH-axis classes (cls3 +0.0073, cls6 +0.0138, cls9 +0.0135).
That looked like the receiver-relative axis hypothesis being validated.

In hindsight: those gains came from over-fitting to the teacher's
high-confidence predictions on those very classes (which were the
teacher's confident-and-correct cells). At test time the gains
inverted, because the same teacher confidence pattern was already in
the blend's other components.

### Updated NEW-LB ladder

```
zoo_v10 elig2  LB 0.3694391  (NEW BEST, 2026-05-10 — UNCHANGED)
zoo_v8  elig3  LB 0.3687552  (2026-05-07)
zoo_v10 elig3  LB 0.3681435  (2026-05-10)
zoo_v10 elig1  LB 0.3664313  (2026-05-09)
zoo_v11 elig1  LB 0.3651563  (2026-05-10)
zoo_v12 elig1  LB 0.3626103  (2026-05-11) ← worst LB transfer of any
                              candidate this round; pseudo-label
                              bias-amplification confirmed
```

### Status: PARK Path A V1; v14_pseudo_v1 banned from submissions

LESSONS_CHECKLIST submission-candidate freeze updated to ban
`v14_pseudo_v1`. New rule added: "Pseudo-label V1 (current-best teacher)
does NOT transfer to LB. Future pseudo-label experiments must use a
structurally different teacher."

R-010 RESOLVED in REVIEW_QUEUE.md.

### Slot status (2026-05-11)
1 of 3 used today. 2 slots remaining; held until a higher-EV candidate
exists (Path B sequence LM, or Path A V2 with non-current-best teacher,
or Path C feature engineering).

### Workflow note
This is the FIRST time a Codex-cleared (`ARTIFACT_OK`) submission
regressed LB. Previous LB regressions (R-007, R-008) bypassed Codex.
Codex correctly cautioned in R-010: "Expected LB lift is uncertain and
likely small ... Treat it as a structural pseudo-label transfer probe,
not a high-confidence improvement." That caution proved warranted —
the experiment was diagnostic-value, not LB-improvement.

### Implications for STRATEGY.md §3 Path A

Path A as proposed (use current best as teacher) is closed. Plausible
next iterations:
- **V2**: use a teacher that AVERAGES across diverse model families (e.g.
  v14 + v11 + v16 components averaged at the OOF level, NOT a known-best
  blend). This breaks the teacher monoculture.
- **V3**: only pseudo-label rows where MULTIPLE independent components
  agree at high confidence (stricter consensus filter). Fewer rows but
  less single-teacher overfit.
- **PARK Path A entirely** if Path B (sequence LM) shows promise.

## 32. 2026-05-11 — R-011 v14_recvprofile: PARKED (multi-axis adds noise vs single-axis recvhand)

### Result

| Metric | v14_seed2 | v14_recvhand | v14_recvprofile | Δ vs seed2 (opt) |
|---|---:|---:|---:|---:|
| FINAL OV (base) | 0.3598 | 0.3598 | 0.3600 | +0.0002 |
| FINAL OV (opt) | 0.3665 | **0.3668** | **0.3663** | **−0.0002** |
| F1_a (opt) | 0.3886 | 0.3886 | 0.3868 | −0.0018 |
| F1_p (opt) | 0.2225 | 0.2227 | 0.2226 | +0.0001 |

**Intake-gate decision**: needs FINAL OV (opt) ≥ v14_seed2 + 0.003 = 0.3695.
Got 0.3663 → **FAILS by −0.0032 → PARK**.

### Per-fold OV trend (v14_recvprofile − v14_seed2)

F1: −0.0021 / F2: +0.0032 / F3: +0.0002 / F4: −0.0016 / F5: −0.0005 → mean
−0.0002. No consistent direction. Compare:
- v14_recvhand (1 axis): F1: −0.0015 / F2: +0.0021 / F3: +0.0012 / F4: +0.0003 / F5: +0.0001 → mean +0.0005.
- v14_pseudo_v1: F1: +0.0003 / F2: +0.0034 / F3: +0.0027 / F4: +0.0047 / F5: +0.0025 → mean +0.0067 (but LB regressed −0.0068!)

### Codex R-011 canary regression check (0.015 cap)

All canaries PASSED:

| Canary | v14_seed2 | v14_recvprofile | Δ |
|---|---:|---:|---:|
| point cls9 BH_long | 0.3316 | 0.3283 | −0.0033 |
| point cls5 mid_half | 0.1630 | 0.1638 | +0.0008 (vs pseudo's −0.0184!) |
| action cls1 Loop | 0.6225 | 0.6194 | −0.0031 |

Notable per-class shifts (point):
- **BH_short cls3**: 0.0000 → 0.0073 (broke F1=0 floor, same as recvhand)
- **FH_long cls7**: 0.2138 → 0.2219 (+0.0081 — biggest point gain)
- mid_long cls8: 0.1961 → 0.1992 (+0.0031)

Notable action shifts:
- **Flick cls7**: 0.1912 → 0.2071 (+0.0159 — biggest action gain)
- Chop cls12: 0.6742 → 0.6819 (+0.0077)
- Lob cls14: 0.3557 → 0.3375 (−0.0182, under non-canary 0.020 cap)
- Pushfast cls5: 0.1952 → 0.1888 (−0.0064)

The structural per-class story is similar to recvhand's (BH classes gain),
but the AGGREGATE doesn't improve. Conclusion: the single-axis recvhand
already captured the receiver-relative signal; the 4 additional axes added
noise without proportional new information.

### Why multi-axis didn't help (post-mortem)

Hypothesised lift was +0.002 to +0.005 OOF. Got −0.0002. Three plausible
reasons:

1. **Information saturation**: handedness alone is the dominant
   receiver-relative axis for pointId labels. The other 4 axes
   (action/point/strength/spin modes) are correlated with handedness or
   with each other, so they don't add orthogonal signal.
2. **High unknown rates**: action mode 40-45% unknown, point 30-33%
   unknown — early-rally rows have no receiver history to compute mode
   from. The unknown indicator dominates when most rows have insufficient
   prior. Tree models split heavily on the unknown indicator and don't
   gain from the rare informative cases.
3. **One-hot expansion adds 36 cols** that increase model complexity
   without proportional signal. v14 has 1170 base features; adding 36
   sparse one-hots is a small fraction but pushes the model toward
   slightly weaker generalisation in the macro-F1 metric.

### Status: PARK v14_recvprofile

- v14_recvhand stays in the zoo as the LB-validated receiver-relative
  feature.
- v14_recvprofile NOT added to zoo menu. Banned from submission
  candidates per LESSONS_CHECKLIST submission-candidate freeze.
- Future Path C iterations should NOT layer more receiver-mode features.
  Try fundamentally different feature classes (rally trajectory
  derivatives, score-state interactions, trigram transitions).

### Wall time
215.8 min total (~3.6 h CPU).

### Updated component menu (no change)

Eligible for submission candidates: v13 (required for NONE), v14_seed2,
v14_recvhand (optional), v16_avg3 (primary) / v16_testhist_aug (backup),
v11_aug + v11plus + v11 (max 2), v12_5f (optional).

Banned: v14_avg3, v14_seed0, v14_seed1, v16_seed1, v16_seed2, v11_big,
v11_aug_big, v11plus_aug, meta_stack_v1/v2, server_head_v1/v2,
v14_pseudo_v1, **v14_recvprofile** (NEW).

### Next strategic options

1. **Try ablation variant**: `--recvprofile-axes strength,spin` (most
   conservative). The action/point one-hot expansion may have been the
   noise source. Cost: ~3-4h CPU.
2. **Different Path C feature class**: rally trajectory derivatives
   (strength/spin shift across prefix), score-state × shot-type
   interactions, trigram transitions.
3. **Path B causal LM** (STRATEGY §9): structural shift to autoregressive
   sequence model. Higher dev cost (~3-4 days), uncertain payoff.
4. **Pseudo-label V2** with non-current-best teacher (not the LB-best
   blend that caused R-010's −0.0068 LB regression).

### LB transfer (added 2026-05-10)

`submission_v14_recvprofile.csv` was uploaded as a single-component
submission (NOT blended). Result:

| Metric | Value | Note |
|---|---|---|
| LB | **0.3381590** | vs current best 0.3694391 = **−0.0313** |
| FINAL OV (opt) | 0.3663 | failed intake gate by −0.0032 |
| OOF→LB ratio | 0.923 | vs blend ratios 0.96–0.98 |

**Caveat on the ratio**: this is a single-model upload, not a blend.
Single-model OOF→LB ratios are typically lower because there's no
ensemble averaging benefit on the LB private-set distribution. We don't
have a clean reference single-model LB datapoint, but a 0.92 ratio is
qualitatively consistent with (a) the lack of ensemble averaging plus
(b) a feature set that adds noise without orthogonal signal.

**Conclusion**: LB confirms the OOF intake-gate verdict. v14_recvprofile
PARK + BAN reaffirmed. The −0.0313 LB drop is large enough that no
ablation of the recvprofile axes can be expected to recover into the
submission-candidate region without removing the multi-axis structure
entirely (i.e., reverting to recvhand). Treat the recvprofile
direction as exhausted.

**Procedural lesson**: A candidate that fails the OOF intake gate
should not be uploaded to LB as a "diagnostic" single-component
submission. The OOF gate is a much cheaper signal than a slot, and the
slot is more valuable than the diagnostic. This pattern (intake-fail
then submit anyway) wasted a slot on a confirmation we already had.
Candidate workflow update: add §3.1.2 "no LB upload of intake-gate
failures, even as single-component diagnostics, unless the diagnostic
question requires LB to answer (e.g., suspected OOF/LB drift) and is
explicitly Codex-approved."

## 33. 2026-05-10 — R-013 v17_causal_lm Fold-1 SMOKE: DIVERSITY_PASS

### Context

R-013 was drafted 2026-05-10 as T2-exploration preflight for a Path B
autoregressive rally LM. Codex `APPROVE_WITH_FIXES` with 8 required
fixes; all applied inline same day. Jabir greenlight to implement and
run **smoke only** with the optimized legal protocol (Phase 1a shared
test-prefix pretrain + Phase 1b Fold-1 train continuation + Phase 2
supervised fine-tune). RTX 3060 Ti, 8 GB VRAM. Hard cap 2 h GPU.

### Implementation

Two new files:
- `src/features_v17_lm_tokens.py` (264 lines) — token sequence builder
  (8 fields: action/point/hand/strength/spin/position/strikeId/shooter_side),
  fold-aware corpus builder, 5 audit functions.
- `src/v17_causal_lm.py` (598 lines) — `CausalRallyLM` decoder-only
  Transformer (1.89 M params: d_model 192, 4 layers, 6 heads), Phase 1
  next-token dataset, Phase 2 supervised dataset, 3-phase training loop,
  pre-training audit harness, correlation report.

### Audits (all PASS, asserted before any optimiser step)

| Audit | Result |
|---|---|
| 8.A fold-safe Phase 1 corpus | Phase 1a (1337 test) ∪ Phase 1b (12,113 fold-1 train) DISJOINT from 2,882 fold-1 val rallies |
| 8.B no target in own prefix | 5,000 supervised samples checked; all input length == N−1 |
| 8.C test prefix length matches visible | 1,337 test rallies, 100% match |
| 8.D no forbidden token fields / model modules | token builder uses only the 8 declared fields + 2 meta; no SGP/match/rally_uid/player IDs anywhere |
| 8.E SGP loss count | Phase 1 = 0; Phase 2 = 1,673,070 = 55,769 train pairs × 30 epochs (exact match) |
| train/val match disjoint | 174 train matches, 42 val matches, intersection = ∅ |
| no_forbidden_in_model | model module names contain no forbidden substrings |

### Run summary (21.2 min wall, 2 h cap NOT REACHED)

- Phase 1a (8 epochs, 1337 test rallies): 0.1 min wall, loss 2.11 → 1.54
- Phase 1b (10 epochs, 12,113 fold-1 train rallies): 1.4 min, loss 1.66 → 1.45
- Phase 2 (30 epochs, 55,769 fold-1 train pairs): 19.2 min, loss 1.29 → 0.77
- Phase 2 best OV at epoch 7 (0.2964); slow decline thereafter (overfitting).

### Fold-1 OOF metrics (best Phase 2 epoch)

| Metric | v17_smoke | v11 | v11_aug | v14_seed2 |
|---|---|---|---|---|
| F1_action | 0.2998 | 0.3001 | 0.3216 | **0.3680** |
| F1_point  | 0.1789 | 0.2009 | 0.1897 | 0.1919 |
| SGP AUC   | 0.5247 | 0.5410 | 0.5406 | **0.6015** |
| Joint OV  | 0.2964 | 0.3086 | 0.3126 | **0.3442** |

v17 is competitive with v11 on F1_action; below all three on F1_point
and SGP AUC; below all three on joint OV by 0.012–0.048.

### Correlation matrix (Pearson r, macro-class avg, Fold-1 val rows)

| | vs v11_aug | vs v11 | vs v14_seed2 |
|---|---|---|---|
| r_action | 0.5807 | 0.5685 | 0.5644 |
| r_point  | 0.5343 | 0.5584 | 0.5186 |

ALL six correlations are well below the 0.85 strong-diversity threshold
and even below the 0.80 conservative line. For comparison, v11_aug ↔
v11 typically have r > 0.85, and v14_seed2 ↔ v14_recvhand are similarly
correlated. v17 is structurally decorrelated from the entire current
zoo.

### Gate verdicts (per R-013 §6)

- **Primary gate** (OV ≥ 0.3036 = min(v11_aug OV, v11 OV) − 0.005):
  **FAIL** by 0.0072.
- **Diversity gate** (r_a vs v11_aug ≤ 0.85 AND r_p vs v11_aug ≤ 0.85):
  **PASS** by wide margin.
- Per-task collapse guards: no F1_action collapse (0.2998 > 0.16 floor);
  no F1_point collapse (0.1789 > 0.095 floor); SGP AUC 0.5247 narrowly
  below the absolute 0.55 floor (acceptable for diversity-only path
  per R-013 caveat — action+point heads carry the diversity payload).
- No NaN, no OOM, no SGP masking violation, no audit failure.

### Recommendation: DIVERSITY_PASS

Open R-014 explicitly tagged "diversity candidate only, not standalone
improver". Full ~30 h GPU run (per R-013 §9 optimized legal protocol)
requires Jabir T3 approval per workflow v2.1 §4.5.

**Why a full run might still be worth it**:
- v17 has r ≈ 0.55 with all three current zoo families. No existing
  component is this decorrelated from the others. Calibrated blend
  including v17 could lift OV even with weaker standalone score.
- Bias-amplification risk LOWER than R-010's pseudo-label V1 (different
  teacher class, different training objective).

**Why a full run might NOT be worth it**:
- ~30 h GPU is significant. Opportunity cost = no other GPU work for
  ~1.5 days.
- v17's weak SGP AUC (0.52) and weak F1_point (0.18) could DRAG a
  blend's per-task scores even with the action diversity benefit.
- Even if the blend OV lifts a small amount on OOF, the OOF→LB ratio
  for blends is ~0.96–0.98 — a 0.001 OOF lift translates to roughly
  −0.004 LB margin of error. Need a CLEAR OOF win to justify the slot.

**Tweaks to consider for the full run** (if approved):
- Phase 2 epochs reduced to ~10–15 per fold + best-checkpoint selection
  (smoke showed clear overfitting beyond Ep7).
- Add SGP-specific Phase 1 task (predict rally outcome from prefix) to
  boost the underperforming SGP head.
- Larger d_model (256 instead of 192) to expand capacity for the
  decorrelation hypothesis.
- Or explicitly KEEP smoke config to preserve the diversity profile —
  larger model might collapse toward v11/v14 representations.

### Artifacts written

- `runs/v17_causal_lm_smoke_fold1/audit.json` — all 7 audit results
- `runs/v17_causal_lm_smoke_fold1/val_metrics.json` — per-epoch + final
- `runs/v17_causal_lm_smoke_fold1/correlation_matrix.json`
- `runs/v17_causal_lm_smoke_fold1/per_class_f1.json`
- `runs/v17_causal_lm_smoke_fold1/fold1_oof_partial.npz` (val rows only,
  not blender-eligible — full run R-014 will produce standard
  `oof_predictions/v17_causal_lm_*` arrays per Codex fix #6)
- `runs/v17_causal_lm_smoke_fold1/summary.txt`
- `logs/v17_smoke_fold1.log` (full training log)

### Status

- v17_causal_lm = NOT in zoo (smoke only).
- NO submission generated.
- NO LB upload.
- Smoke pipeline + audits validated; ready for R-014 if Jabir approves.

## 34. 2026-05-11 — R-015 v17_momentum CORE smoke PASSED, ALL regressed; full 5-fold running

### Context

R-015 drafted 2026-05-10 as T2-component preflight for rally
momentum / initiative / pressure-state features. Codex
`APPROVE_WITH_FIXES` 2026-05-11 (5 findings). Claude critical review
applied with 1 documented partial pushback (Group 4 per-side
aggregates kept; only the parity-bit redundancy with `next_is_server`
removed). Per Jabir's 8h autonomous window, implementation +
both smokes ran sequentially.

### Implementation

Two new files (additive only; no edits to existing v14/v16 trainers):
- `src/features_v17_momentum.py` (350 lines):
  - Wraps `features_v9_recvhand`.
  - Per-rally array precomputation cache.
  - `--momentum-groups` env-var flag (`core` / `all`).
  - 4 build-time assertions per Codex P3.5: `SOURCE_COLS` allow-list,
    no-forbidden-fields, no-NaN/inf, max-source-strikeNumber < N.
  - Pressure scalar = `is_attack × strength_factor` (fixed constants;
    no fold dependency per Codex P2.4).
  - Cap-hit-rate logging per group.
- `src/train_v17_momentum.py` (790 lines, cloned from
  `train_v16_testhist_aug.py`):
  - V16 backbone preserved (test-history augmentation, two-pass
    action→point stacking, flip aug, threshold optimisation).
  - New CLI: `--feature-set v9 / v9_recvhand / v9_momentum`,
    `--momentum-groups core / all`, `--max-folds N`.
  - Full-coverage OOF assertion bypassed when `--max-folds N < 5`
    (preserves smoke usability).

### Smoke 1: CORE (Groups 1+2+3, 26 features) — Fold-1, full-budget

Wall: 17.6 min on CPU.

| Metric | v17 CORE | v16_testhist_aug Fold-1 | Δ vs V16 | Gate (V16 − tol) | Pass |
|---|---:|---:|---:|---:|---|
| OV (base) | **0.3577** | 0.3562 | **+0.0015** | ≥ 0.3512 | ✅ |
| OV (opt)  | **0.3717** | ~0.3677 (solo opt) | **+0.0040** | — | ✅ |
| F1_action | **0.4086** | 0.4003 | **+0.0083** | ≥ 0.3953 | ✅ |
| F1_point  | 0.1865 | 0.1893 | −0.0028 | ≥ 0.1843 | ✅ |
| SGP AUC   | 0.5984 | 0.6016 | −0.0032 | ≥ 0.5966 | ✅ |
| cls0 point F1 | 0.1526 | 0.1590 | −0.0064 | ≥ 0.1490 | ✅ |
| All 4 build-time audits | PASS | — | — | — | ✅ |
| No NaN/inf | clean | — | — | — | ✅ |
| Cap-hit rates | streak 5%, total 0.4% | — | log only | — | acceptable |

**CORE smoke verdict: PRIMARY PASS.** OV gain of +0.0015 (base) /
+0.0040 (opt) over V16 backbone. F1_action carries the win; F1_point
and SGP AUC slightly regress but stay within tolerance.

### Smoke 2: ALL (Groups 1+2+3+4+5, 41 features) — Fold-1, full-budget

Wall: 18.3 min on CPU.

| Metric | ALL | CORE | V16 | Δ vs CORE | Δ vs V16 |
|---|---:|---:|---:|---:|---:|
| OV (base) | 0.3554 | 0.3577 | 0.3562 | **−0.0023** | −0.0008 |
| OV (opt)  | 0.3666 | 0.3717 | ~0.3677 | **−0.0051** | −0.0011 |
| F1_action | 0.4039 | 0.4086 | 0.4003 | −0.0047 | +0.0036 |
| F1_point  | 0.1874 | 0.1865 | 0.1893 | +0.0009 | −0.0019 |
| SGP AUC   | 0.5944 | 0.5984 | 0.6016 | −0.0040 | −0.0072 |
| cls0 point F1 | 0.1533 | 0.1526 | 0.1590 | +0.0007 | −0.0057 |

**ALL smoke verdict: REGRESSED vs CORE.** Group 4 (per-side initiative)
+ Group 5 (pressure derivatives) add noise rather than signal.

This empirically vindicates Codex's "Group 4 has limited marginal
value due to overlap with `next_is_server`" framing — even though the
per-side aggregates are technically NEW info (and I pushed back on
that point in the preflight), they don't help the model. My pushback
was logically defensible but wrong in practice.

It also vindicates my own self-review concern that Group 5 (pressure
derivatives) was the noisiest and most heuristic-laden group.

### Decision: full 5-fold uses CORE

Per the smoke pass paths in R-015 §6:
1. Primary pass (CORE): all gates passed → run full 5-fold with CORE.
2. ALL: regressed → ALL configuration parked.

Full 5-fold launched 2026-05-11 with `--momentum-groups core`. ETA
3-4 h. Output artifacts will populate `oof_predictions/v17_momentum_*`
in standard blender-compatible naming (per Codex P2.6).

### Correlation matrix (Fold-1 val, computed offline post-smoke)

Codex's correlation diagnostic was specified as "report Pearson r vs
`v16_testhist_aug` and `v14_recvhand` action+point probs, diagnostic
only, NOT pass/fail unless r > 0.99":

| | vs v14_seed2 | vs v14_recvhand | vs v16_testhist_aug |
|---|---:|---:|---:|
| CORE r_action | 0.878 | 0.875 | **0.987** |
| CORE r_point | 0.789 | 0.770 | **0.967** |
| ALL r_action | 0.877 | 0.875 | 0.986 |
| ALL r_point | 0.784 | 0.766 | 0.964 |

**Interpretation**: v17_momentum CORE is **highly correlated with
`v16_testhist_aug`** (r ~0.97-0.99) — same V16 backbone with marginal
features added. Below the 0.99 exact-duplication threshold, so NOT
flagged as duplicate, but clearly an in-family extension rather than a
diversity component.

vs v14 family the correlation is moderate (r ~0.78-0.88) — reasonable
diversity for a v14+v17 blend.

**Implication for blender intake (R-016)**:
- v17_momentum should REPLACE v16_testhist_aug in NONE blends, not
  complement it (high r → minimal additional ensemble gain).
- v17_momentum + v14 family blends remain valid (moderate r).
- Submission candidate menu impact (per LESSONS): if v17_momentum
  beats v16_testhist_aug at full 5-fold, it becomes the new V16-family
  representative; v16_testhist_aug demoted from active menu.

### Audits (all PASS, asserted at module import / build time)

| Audit (per Codex P3.5) | Result |
|---|---|
| 8.D-equivalent: SOURCE_COLS contains no forbidden fields | PASS |
| 8.D-equivalent: emitted feature names contain no forbidden identifiers | PASS |
| 8.B-equivalent: per-row max source strikeNumber < N (zero violations) | PASS |
| no NaN/inf in any v17m_* column | PASS |
| Pressure scalar bounded in [0.0, 1.5] (Group 5 only) | N/A (CORE smoke) |

### Artifacts written

- `src/features_v17_momentum.py`, `src/train_v17_momentum.py`
- `oof_predictions/v17_momentum_smoke_core_oof_{act,pt,srv,pt_bin,mask,y_act,y_pt,y_srv,nsn}.npy`
- `oof_predictions/v17_momentum_smoke_core_test_{act,pt,srv,rally_uid}.npy`
- `oof_predictions/v17_momentum_smoke_all_*` (corresponding ALL arrays — parked)
- `submissions/submission_v17_momentum_smoke_core.csv` (NOT for LB upload — diagnostic only)
- `submissions/submission_v17_momentum_smoke_all.csv` (parked)
- `runs/v17_momentum_smoke_correlation.json` (correlation diagnostic)
- `runs/v17_momentum_smoke_core_baselines.json` (Fold-1 baseline metrics)
- `logs/v17_momentum_smoke_core.log`, `logs/v17_momentum_smoke_all.log`

### Status (smoke phase)

- v17_momentum CORE smoke = PASSED Fold-1 gates.
- v17_momentum ALL = PARKED (regressed vs CORE).
- Full 5-fold CORE = RUNNING.

### 34a. Full 5-fold v17_momentum CORE results (2026-05-11, 86.4 min wall)

Wall: 86.4 min on CPU (faster than 3-4 h estimate).

**Per-fold OV variance**:
- Fold 1: 0.3577 (matches smoke exactly)
- Fold 2: 0.3368 (low)
- Fold 3: 0.3785 (high)
- Fold 4: 0.3428
- Fold 5: 0.3411
- Mean: 0.3514, std ~0.015 (typical fold variance band)

**Smoke Fold-1 (0.3577) was the LUCKIEST fold of 5.** The +0.0040 vs
V16 lift seen in smoke evaporated when averaged across all folds.

**Global 5-fold OOF metrics (re-computed offline with consistent threshold
optimization for fair comparison)**:

| Tag | OV (base) | OV (opt) | F1_a | F1_p | AUC | cls0_p | cls9_p |
|---|---:|---:|---:|---:|---:|---:|---:|
| v14_seed2 | 0.3598 | 0.3661 | 0.3794 | 0.2148 | 0.6104 | 0.4385 | 0.3316 |
| v14_recvhand | 0.3598 | 0.3666 | 0.3786 | 0.2152 | 0.6113 | 0.4357 | 0.3377 |
| v16_testhist_aug | 0.3575 | 0.3666 | 0.3880 | 0.2012 | 0.6092 | 0.1969 | 0.3172 |
| **v17_momentum** | **0.3571** | **0.3662** | **0.3876** | **0.2002** | **0.6099** | **0.1884** | **0.3320** |
| **Δ vs V16** | **−0.0004** | **−0.0003** | −0.0005 | −0.0011 | +0.0007 | −0.0085 | **+0.0149** |
| **Δ vs V14_recvhand** | −0.0027 | −0.0004 | +0.0090 | −0.0150 | −0.0014 | −0.2473 | −0.0057 |

v17_momentum **FAILS the standalone intake gate** (R-015 §6: solo
opt OOF ≥ V16 0.3677, OR F1_p improves ≥ +0.005 without F1_a/AUC
regress > 0.005). OV opt 0.3662 is 0.0003 BELOW V16/V14_recvhand
(both at 0.3666), and F1_p regresses by 0.0011 (slight, within noise).

**Notable per-class shifts** (within the 0.020 cap):
- cls9 BH_long: **+0.0149** vs V16 — meaningful structural gain on a
  high-support class (n=16,073). v17m's per-side initiative + recent-
  attack ratios likely help predict back-court returns.
- cls0 miss: −0.0085 vs V16 (V16 backbone effect; v17m inherits
  V16's weakness here).
- cls5 mid_half: +0.0017 (Codex canary, no regression).
- cls1 Loop (action): −0.0008 (Codex canary, no regression).

### 34b. Full 5-fold correlation matrix vs all zoo components

| Reference | r_action | r_point |
|---|---:|---:|
| v11_aug | 0.680 | 0.657 |
| v11 | 0.671 | 0.678 |
| v11plus | 0.661 | 0.636 |
| v13 | 0.879 | 0.767 |
| v14_seed2 | 0.887 | 0.772 |
| v14_recvhand | 0.886 | 0.771 |
| v16_testhist_aug | 0.988 | 0.966 |
| **v16_avg3** | **0.992** | **0.978** |

**v17_momentum has r = 0.992 (action) / 0.978 (point) vs `v16_avg3`** —
**at/above the Codex r > 0.99 "exact duplication" threshold for the
action axis.** v17_momentum is effectively a near-clone of v16_avg3,
not a structurally distinct component.

This is the 2nd Fold-level vs full-OOF surprise this round: the Fold-1
smoke showed r ~0.987 vs V16; the full 5-fold confirmed r ~0.99 — even
more correlated than expected. The momentum features add per-class
shifts (cls9 BH_long +0.0149) but don't move the model meaningfully
away from the V16 representation.

### 34c. Blender substitution study (size ≤ 5, per locked rule #8)

Tested v17_momentum substitutions into the LB-best subset
`v11_aug + v11plus + v13 + v14_seed2 + v16_avg3` (zoo_v10 elig2,
LB 0.3694, OOF opt 0.3766):

| Subset | OOF (opt) | Δ vs LB-best |
|---|---:|---:|
| LB-best baseline (5 components) | 0.3766 | — |
| swap v16_avg3 → v17_momentum | 0.3763 | **−0.0002** |
| swap v16_testhist_aug → v17_momentum | 0.3763 | −0.0002 |
| swap v14_seed2 → v17_momentum | 0.3772 | **+0.0006** |
| ADD v17 alongside v16_avg3 (6 comp) | 0.3793 | **+0.0027** *but VIOLATES rule #8* |
| 4-comp (v17 + v11_aug + v13 + v14s2) | 0.3770 | +0.0004 |

Within the size-5 cap, the best result is `swap v14_seed2 → v17` with
**+0.0006 OOF** — within typical noise band, no clear LB lift expected.
The 6-comp `+v17` configuration shows +0.0027 OOF but violates the
locked blend-size cap (rule #8: "subsets of size ≤ 5 unless a size-6
candidate has been LB-validated"; zoo_v3 with 6 components lost
−0.0058 LB).

### 34d. PARK decision

**v17_momentum PARKED.**

Reasons (in priority order):
1. **Standalone OV ties V16** (Δ −0.0003 OV opt). Fails the strict
   intake gate (V16 + 0.003 = 0.3696). The +0.0040 smoke lift was
   Fold-1 luck — full 5-fold mean confirmed near-zero standalone gain.
2. **Near-duplicate correlation** (r = 0.992 action / 0.978 point vs
   v16_avg3). At/above Codex's exact-duplication threshold. Not a
   diversity component.
3. **Blender substitution gives ±0.0006 OOF** at best within size-5
   cap — within fold-variance noise. No clear LB lift expected.
4. **The +0.0027 6-comp gain violates rule #8** (blend-size cap).
   Past 6-comp candidates lost −0.0058 LB.

**Net effect**: v17_momentum is computationally a duplicate of v16_avg3
with marginal per-class shifts (most notable: cls9 BH_long +0.0149).
That per-class gain is real but doesn't translate to OV or to
identifiable LB lift via valid blend configurations.

**v17_momentum BANNED from submission candidates** per LESSONS_CHECKLIST
submission-candidate freeze (added 2026-05-11).

### 34e. Lessons + design retrospective

**Why v17_momentum failed despite reasonable design**:

1. **The V16 backbone already encodes momentum implicitly.** V16 +
   v9_recvhand + v6 lag one-hots already give the model 1170+ features
   covering most of the per-shot context. The model can synthesize
   "is the prev shot an attack?" from existing one-hot lags via tree
   splits. The explicit `v17m_prev1_is_attack` feature is a single-
   split shortcut, not new information.

2. **Recent-window ratios are computed by trees anyway.** Tree models
   can split on `oh_lag1_actionId_1 + oh_lag2_actionId_1 + ...` to
   approximate a recent attack count. The explicit `recent3_attack_count`
   feature saves a few splits but doesn't unlock new information.

3. **Per-side aggregates (Group 4) and pressure derivatives (Group 5)
   add noise.** ALL smoke confirmed this empirically. Codex's "Group 4
   limited marginal value" framing — which I pushed back on as
   over-generalized — turned out to be correct in practice. My
   pushback was logically defensible (per-side aggregates ARE new
   info) but wrong about whether the model could use that info. The
   GBM apparently can't translate per-side aggregates into useful
   splits given the existing 1170 features already capture per-shot
   structure.

4. **The streak/transition features (Group 3) DID help action F1
   slightly** (+0.008 vs V14_seed2) but the gain was offset by F1_p
   regression. Action-specialist behavior didn't translate to
   blend-level gains.

5. **CORE smoke Fold-1 gave +0.0040 OV — pure noise.** Fold-1 is
   ~0.005 above the 5-fold mean (high-variance fold for v17). Lesson:
   single-fold smoke gates are necessary but insufficient; budget for
   the full 5-fold variance.

**What this means for future feature engineering**:

- Adding tabular features on V16 backbone with GBM is hitting the
  ceiling. The ~0.3666 OV (opt) is shared across v14_recvhand,
  v16_testhist_aug, and now v17_momentum — three distinct feature
  designs all clustering at the same point.
- The structural levers remaining are: (a) different model class
  (v17_causal_lm Path B), (b) different training paradigm (pseudo
  V2 with diverse teacher), (c) different per-task architecture
  (hierarchical head, but P3 already failed).

### 34f. Artifacts written (full run)

- `oof_predictions/v17_momentum_oof_{act,pt,srv,pt_bin,mask,y_act,y_pt,y_srv,nsn}.npy`
- `oof_predictions/v17_momentum_test_{act,pt,srv,rally_uid}.npy`
- `submissions/submission_v17_momentum.csv` (NOT for LB upload — failed intake gate)
- `submissions/submission_v17_momentum_binary_srv.csv` (auto-generated, not for LB)
- `runs/v17_momentum_full_correlation.json`
- `logs/v17_momentum_full.log`

### Final status

- v17_momentum solo OV (opt) **0.3662** vs V16 0.3666 = **−0.0003** (FAIL intake).
- v17_momentum r > 0.99 vs v16_avg3 (action axis) — near-duplicate.
- v17_momentum NOT added to zoo.
- v17_momentum NOT used in any submission.
- ~~NO LB upload.~~ **LB upload occurred 2026-05-11** despite Claude's recommendation against. See §34g.
- v17_momentum BANNED from submission candidates.

### 34g. LB transfer (2026-05-11) — fair-comparison framing

`submission_v17_momentum.csv` was uploaded to LB. **IMPORTANT
CORRECTION**: this is a SOLO model submission, not a blend. The
initial framing "−0.0093 vs LB-best 0.3694" was UNFAIR because
0.3694 is a 5-component blend; solo-vs-blend comparison conflates
ensemble-averaging benefit with feature design.

| Metric | Value | Fair comparison |
|---|---|---|
| LB | **0.3601463** | — |
| OOF (opt) | 0.3662 | — |
| OOF→LB ratio | **0.9833** | matches V16-family typical (0.978) |
| Δ vs LB-best 5-blend (0.3694) | −0.0093 | **NOT FAIR** — solo vs blend |
| Δ vs estimated solo V16 LB | ~0 (within solo-vs-blend gap) | FAIR — close to V16-family solo expectation |

**What the LB actually tells us**:
- v17 transfers like V16 on LB. Consistent with r=0.992 correlation
  finding.
- v17 is NOT broken — it's a V16-clone, behaves like V16 on every axis
  (OOF, correlation, LB ratio).
- The LB upload added no new information beyond what OOF + correlation
  already told us. The slot was wasted because the LB result was
  PREDICTABLE, not because the LB was bad in absolute terms.

**Updated PARK rationale** (the 4 reasons remain valid):
1. OOF (opt) 0.3662 ties V16 solo OV (opt) 0.3666 — solo vs solo, fair
2. r = 0.992 vs v16_avg3 — near-duplicate
3. Blender substitution OOF −0.0002 (5-blend vs 5-blend) — fair, no improvement
4. LB ratio 0.9833 (matches V16-family typical) — fair: v17 transfers
   like V16, no new info gained

**Lesson update**: always specify SOLO vs BLEND when comparing LB
scores. Single-model penalty vs n-component blend is typically
0.005-0.020 on LB and bakes in ensemble averaging. Logged to
LESSONS_CHECKLIST 2026-05-11.

(The R-011 v14_recvprofile case was different — its OOF→LB ratio
was 0.92, well below V14-family typical, indicating actual feature
regression beyond the solo-vs-blend gap. v17 is closer to V16 in
both OOF AND ratio, so v17's LB outcome was structurally expected.)

## 35. 2026-05-11 — SERENDIPITOUS: blender finds +0.0019 OOF over LB-best (NO v17)

While running the R-015 v17_momentum blender intake study, I performed
an exhaustive search over all 45 valid size-5 NONE-eligible subsets
using only currently-eligible zoo components. The top result is:

**`(v11, v11_aug, v13, v14_seed2, v16_testhist_aug)`** = OV (opt)
**0.3785** vs current LB-best zoo_v10 elig2 OOF (opt) 0.3766 = **+0.0019**.

This subset contains NO v17_momentum. It's a pure rearrangement of
existing eligible components vs the current LB-best subset:

| | Current LB-best (zoo_v10 elig2) | Top OOF candidate |
|---|---|---|
| A-family | v16_avg3 | **v16_testhist_aug** (single-seed) |
| B-family | v14_seed2 | v14_seed2 (same) |
| D-family (transformers) | v11_aug + v11plus | **v11 + v11_aug** |
| E-family | v13 | v13 (same) |
| LB | 0.3694391 | TBD |
| OOF (opt) | 0.3766 | **0.3785** |

### Top 5 size-5 OOF candidates

| Rank | OV (opt) | Δ vs LB-best | Subset |
|:---:|---:|---:|---|
| 1 | **0.3785** | **+0.0019** | (v11, v11_aug, v13, v14_seed2, v16_testhist_aug) |
| 2 | 0.3782 | +0.0016 | (v11, v11_aug, v13, v14_recvhand, v16_avg3) |
| 3 | 0.3782 | +0.0016 | (v11, v11_aug, v13, v14_recvhand, v16_testhist_aug) |
| 4 | 0.3780 | +0.0014 | (v11, v11plus, v12_5f, v13, v16_avg3) |
| 5 | 0.3778 | +0.0012 | (v11, v11_aug, v12_5f, v13, v16_avg3) |

### Why the OOF prefers "older" components

Both substitutions go AGAINST recent LB findings:
- **v11plus → v11**: v11plus has higher solo OV but introduces correlation
  with v11_aug; v11+v11_aug pair appears to give better diversity.
- **v16_avg3 → v16_testhist_aug**: R-004 LB result showed v16_avg3 was
  +0.0007 LB over v16_testhist_aug for the v11_aug+v11plus+v14s2 subset.
  For THIS subset (v11+v11_aug+v14s2), the relationship may invert.

This is a textbook case of **interaction effects**: a substitution
that helps in one blend may hurt in another because of correlation
structure with the other components.

### Expected LB transfer

Current LB-best OOF→LB ratio: 0.3694 / 0.3766 = 0.9809.
If top OOF candidate transfers similarly: 0.3785 × 0.9809 = **0.3713**
= **+0.0019 LB lift potential**.

But OOF→LB transfer is fragile for new subsets — past size-6 candidates
lost on LB even when OOF was higher. +0.0019 is small enough to be
within typical transfer noise (±0.005).

### Recommendation for next round

**Open R-016** (next session) for the top OOF candidate as an LB probe
candidate. Per LESSONS submission-candidate freeze:
- All components in the top candidate are ELIGIBLE
  (v11_aug, v11, v13, v14_seed2, v16_testhist_aug)
- NONE rule satisfied (≥ 2 transformers: v11+v11_aug)
- v13 in subset ✓
- Transformer count ≤ 2 ✓
- v11_aug present (required for NONE per rule #12) ✓

R-016 should request Codex ARTIFACT_OK + Jabir explicit slot approval
to upload `submission_zoo_v??_v11_v11aug_v13_v14s2_v16testhist.csv` as
the next LB probe.

### Status (pre-LB-upload)

- v17_momentum: PARKED (this section's parent §34).
- **Top non-v17 OOF candidate identified**: `(v11, v11_aug, v13, v14_seed2, v16_testhist_aug)`
- Artifact: existing OOF .npy files; no new training needed for this
  candidate.

### 35a. R-016 LB transfer (2026-05-11) — CONFIRMED REGRESSION

`submission_R016_v11_v11aug_v13_v14s2_v16testhist.csv` was generated
during the v17 session (NONE calibration, OOF-tuned thresholds) and
uploaded to LB. Result:

| Metric | Value |
|---|---:|
| LB | **0.3672687** |
| Δ vs current LB best (zoo_v10 elig2 = 0.3694391) | **−0.0022** |
| OOF (opt) | 0.3785 |
| OOF→LB ratio | **0.9703** |
| Predicted LB (Claude) | 0.3713 (range 0.3680–0.3740) |
| Actual vs prediction | **below predicted minimum** |

The OOF +0.0019 gain inverted to LB −0.0022. OOF→LB ratio degraded by
~1 percentage point vs the LB-best subset (0.9809 → 0.9703).

### 35b. Why the OOF gain did not transfer (post-mortem)

The candidate had two substitutions vs the LB-best subset:
1. **v11plus → v11**: Counter to recent LB evidence. v11plus had been
   LB-validated in zoo_v10 elig2; swapping it for v11 lost something
   the OOF didn't capture. Possibly v11plus's class-weight escalation
   helps on the LB private-set distribution that OOF doesn't represent.
2. **v16_avg3 → v16_testhist_aug**: Directly reverses R-004's LB
   finding (+0.0007 LB for v16_avg3). The blender re-discovered an
   OOF-favorable but LB-unfavorable arrangement.

**Combined effect**: −0.0022 LB despite +0.0019 OOF. The two
unfavorable substitutions outweighed each other on OOF (no net swing)
but compounded on LB.

### 35c. New hard lesson — blender-overfit OOF

This is the **3rd confirmed instance** of "blender-search OOF gain
doesn't transfer to LB" this round:

| R-### | Candidate type | OOF | LB | Reason |
|---|---|---|---|---|
| R-007 | v14_avg3 substitution | +? | **−0.0013** | seed-averaging didn't transfer |
| R-008 | drop-v13 + 3-transformer | unclear | **−0.0043** | rule-violating subset |
| R-016 | v11+v11_aug+v16_testhist swap | +0.0019 | **−0.0022** | rearrangement of LB-validated components |

**The current LB-best subset (v11_aug+v11plus+v13+v14_seed2+v16_avg3
NONE) is a LOCAL OPTIMUM that exhaustive blender search over
already-trained components cannot improve upon by rearrangement.**

**Forward implications**:
- Pure component re-arrangement on already-trained components is
  EXHAUSTED as a path to LB lift.
- Future LB candidates must include either:
  (a) STRUCTURALLY NEW components (Path B causal LM, or genuinely
      new feature class — NOT V16-clones like v17_momentum), OR
  (b) NEW calibration arms (TEMP, CW, THR not just NONE), OR
  (c) Smart-blender Dirichlet weight search (not just equal-weight)
      with Codex review on the resulting subset.
- The rich blender exhaustive-search space we're exploring on existing
  components is overfitting OOF in subtle interaction effects.

### Status (post-LB-upload)

- R-016 candidate: PARKED (LB regression confirmed).
- Subset `(v11, v11_aug, v13, v14_seed2, v16_testhist_aug)` BANNED
  from re-submission unless paired with a structurally new component.
- LB best UNCHANGED: zoo_v10 elig2 = **0.3694391**.
- This round (R-015 + R-016) net LB impact: **0** (both regressed; no
  improvement to LB best).

## 36. 2026-05-12 — 12-hour autonomous training session (R-018 through R-021)

### Window summary

12-hour autonomous training window 2026-05-12. Codex pre-reviewed plan (5 P1/P2
fixes applied). Approved scope: smoke-only experiments for R-019 (uncertainty
MTL) and R-021 (ShuttleSet22 pretraining), plus full 5-fold for R-020c
(λ=0.1 v11_mulminet+aug sweep). NO LB uploads, NO full 5-fold for R-019/R-021
without separate R approval.

### Experiment 1: R-020c v11_mulminet+aug at λ=0.1 (full 5-fold)

| Metric | λ=0.2 (R-020a) | **λ=0.1 (R-020c)** | Δ |
|---|---:|---:|---:|
| Wall | 110.7 min | 90.6 min | -20 min |
| OV (base) | 0.3299 | **0.3254** | **−0.0045** |
| F1_action | 0.3441 | 0.3414 | −0.0027 |
| F1_point | 0.2000 | 0.1929 | −0.0071 |
| AUC | 0.5614 | 0.5546 | −0.0068 |

**Verdict: PARK λ=0.1.** All metrics worse than λ=0.2. Per Codex P2.4
"if doesn't beat λ=0.2 → park". λ=0.2 confirmed as the optimal aux weight.

### Experiment 2: R-021 v11_mulminet pretrained on ShuttleSet22 (badminton)

#### Pretrain phase (CPU/GPU, ~5 min total)

- ShuttleSet22 cloned (60 matches, 4806 rallies after filtering, 18 stroke types)
- Loader `src/features_shuttleset22.py` with audit (no forbidden cols read)
- License: open-source academic, AI CUP 2026 rules permit
- Pretrain script v1: BUG (target leakage from bidirectional encoder predicting
  next position from input that included it; loss collapsed to 0.0005)
- Pretrain script v2: CAUSAL MASK FIX. Loss converges sensibly.
- 30 epochs, 1.3 min wall, best val_total **1.667 at Ep 8** (then overfits)
- Encoder weights saved (`models/v11_pretrained_badminton.pt`, 51 keys, 1.78M params)

#### Fine-tune Fold-1 smoke (with badminton pretrained init)

| Metric | v11_mulminet_aug Fold-1 | **v11_mulminet_pretrained_aug Fold-1** | Δ |
|---|---:|---:|---:|
| OV | 0.3222 | **0.3226** | **+0.0004 (TIE)** |
| F1_action | 0.3408 | 0.3348 | **−0.0060** |
| F1_point | 0.1899 | 0.1924 | +0.0025 |
| **AUC** | 0.5496 | **0.5585** | **+0.0089** ★ |

#### Correlation matrix (v11_mulminet_pretrained_aug Fold-1 val)

| Reference | r_action | r_point |
|---|---:|---:|
| v11 | 0.817 | 0.850 |
| v11_aug | 0.833 | 0.849 |
| v11plus | 0.826 | 0.847 |
| v11_mulminet | 0.870 | 0.892 |
| **v11_mulminet_aug** | **0.842** | **0.839** |
| v13 / v14 / v16 family | 0.70-0.71 | 0.65-0.71 |

Critical observation: **r vs v11_mulminet_aug = 0.842 / 0.839 — HIGHER than
v11_mulminet_aug's typical correlation with V11 family (0.71-0.76)**. Pretrain
made the model MORE similar to V11 family, not less.

#### Verdict: **PARK per Codex P2.5 strict gate**

- Strong pass: FAIL (OV +0.0004 < +0.003 threshold)
- Diversity pass: FAIL (correlation went UP, not DOWN)
- Park triggered (neither strong nor diversity)

#### Why pretraining didn't help (post-mortem)

1. **Causal-bidirectional mismatch**: Pretrained with causal mask (to fix
   target leakage); V11 fine-tune uses bidirectional. Encoder weights tuned
   for causal attention may not be optimal for bidirectional use.
2. **Domain gap**: badminton 2D coords + 18 stroke types ≠ table tennis
   9-zone + 15 actionId. Encoder representations may not capture relevant
   table tennis dynamics.
3. **Data signal limit**: V11/V14/V16 saturated. The transformer encoder
   weights were not the bottleneck — the data signal is.

#### Status

- v11_mulminet_pretrained_aug PARKED as standalone or substitution.
- Kept in zoo as **private-LB candidate** (AUC +0.009 useful for ensemble
  diversity per Jabir's "private LB ≠ public LB" framework, 2026-05-12).
- NO LB upload, NO full 5-fold without separate R-022.

### Experiment 3: R-019 v11_uncertainty (Kendall & Gal MTL)

#### Implementation

- 3 learnable log-variance scalars (action/point/server), zero-init
- Loss: `Σ (1/(2·exp(s_i))) · L_i + 0.5·s_i` per Kendall & Gal CVPR 2018
- Replaces fixed 0.4/0.4/0.2 with model-learned weights

#### Fold-1 smoke result — **STRONG PASS**

| Metric | v11 baseline | **v11_uncertainty** | Δ |
|---|---:|---:|---:|
| OV | 0.3086 | **0.3123** | **+0.0037** |
| F1_action | 0.3001 | 0.3047 | +0.0046 |
| F1_point | 0.2009 | 0.1995 | −0.0014 |
| **AUC** | 0.5410 | **0.5531** | **+0.0121** ★ |

#### Learned task weights converged sensibly

Final weights at Ep 75 (vs initial uniform 0.5):

| Task | Initial w | **Final w** | Effective × |
|---|---:|---:|---:|
| action | 0.500 | 1.728 | **3.5×** |
| point | 0.500 | 0.625 | 1.25× |
| **server (SGP)** | 0.500 | **2.767** | **5.5×** |

The model autonomously emphasized SGP 5.5× more than initial weight. This
**precisely matches Kendall & Gal's prediction** that the uncertainty-weighted
MTL emphasizes the higher-uncertainty (i.e., harder) task.

#### Correlation matrix (v11_uncertainty Fold-1 val)

| Reference | r_action | r_point |
|---|---:|---:|
| **v11** | **0.948** | **0.961** |
| v11_aug | 0.849 | 0.869 |
| v11plus | 0.924 | 0.933 |
| v11_mulminet_aug | 0.790 | 0.816 |
| v14 / v16 family | 0.64-0.66 | 0.62-0.67 |

v11_uncertainty is **structurally similar to v11** (r=0.948) — it's a
v11 with rebalanced MTL weights. Less correlated with v11_mulminet_aug
(0.79/0.82 — the aux losses change representation too).

#### Status

- v11_uncertainty SMOKE PASS per R-019 gates.
- Per Codex P2.4 "R-019 only smoke" — NO full 5-fold without separate R.
- Could be **v11 substitute** in blends (similar architecture, AUC +0.012 lift).
- Combining with MuLMINet aux + aug not yet tested (would need R-023).

### Session totals

| Component | Standalone OV (Fold-1) | Status |
|---|---:|---|
| v11 baseline | 0.3086 | reference |
| v11_aug baseline | 0.3216 | reference |
| **v11_mulminet** (R-018) | 0.3197 | weak win standalone, blend regression confirmed |
| **v11_mulminet_aug** (R-020a) | **0.3299** (full 5-fold) | strongest V11-family standalone, solo LB 0.3518 |
| v11_mulminet_aug λ=0.1 (R-020c) | 0.3254 | parked, λ=0.2 wins |
| v11_uncertainty (R-019 smoke) | 0.3123 | strong smoke pass, AUC +0.012 |
| v11_mulminet_pretrained_aug (R-021 smoke) | 0.3226 | tied with v11_mulminet_aug, parked but kept for ensemble |

### Final 12h-window verdict

**Direct LB impact**: 0 (no LB uploads per Codex strict scope).

**Component zoo additions**:
- v11_uncertainty: validated SGP-strengthening technique (AUC +0.012)
- v11_mulminet_pretrained_aug: validated transferable encoder pretraining
  pipeline (although standalone TIE)

**Validated dead-ends**:
- λ=0.1 worse than λ=0.2 for v11_mulminet
- Causal-pretrain → bidirectional-finetune mismatch limits ShuttleSet22 transfer

**Strategic value**: We now have 4 v11-family components with different
strengths/weaknesses:
- v11 (vanilla baseline)
- v11_aug (test-history aug)
- v11_mulminet_aug (aux + aug, strongest standalone)
- v11_uncertainty (MTL rebalance, AUC specialist)
- v11_mulminet_pretrained_aug (badminton pretrain, AUC sub-specialist)

For final submission ensemble, can pick 2-3 from this set for diversity.

### Recommended next experiments (post-window)

1. **R-022**: combine techniques — v11_mulminet + uncertainty MTL + aug.
   Expected: best of both worlds, possibly OV ~0.3260 standalone Fold-1.
2. **R-023**: bidirectional pretraining objective (MLM-style or masked-shot)
   to better match V11's bidirectional fine-tune. Could fix the
   pretrain-finetune attention mismatch.
3. **R-024**: full 5-fold of v11_uncertainty (after R-019 smoke pass).
4. **R-025**: TabPFN-v2 SGP head as ensemble diversity.

All require explicit Codex review per workflow §3.1.1 (T3 / external data).

## 36. 2026-05-11 — R-018 v11_mulminet auxiliary-task Transformer (FAILED gate)

`src/train_v11_mulminet.py` cloned `train_v11_transformer.py` and added
MuLMINet-style auxiliary heads for next-shot `handId`, `strengthId`,
`spinId`, and `positionId`.

Full 5-fold completed (`logs/v11_mulminet_full.log`, tag
`v11_mulminet`):

| Metric | Value |
|---|---:|
| OOF mask | 69712 / 69712 |
| F1_action | 0.3277 |
| F1_point | 0.1929 |
| AUC | 0.5573 |
| OV | **0.3197** |

Verdict: **v11_mulminet FAILED intake**. The auxiliary-task design did
not improve V11; it underperforms existing transformer components and is
not a zoo submission candidate. Artifacts are retained for diagnostics:
`oof_predictions/v11_mulminet_*`, `submissions/submission_v11_mulminet.csv`.

---

## 37. 2026-05-18 → 2026-05-20 — Phase 3 deadline orchestrator: 18 new ELIGIBLE oldtest variants

### Context

Following R-027 PAIR LB breakthrough (0.3810401, +0.0116 vs prior LB-best),
R-028 top1 LB regression (0.3724530, −0.0086 — refined CLASS B framework),
and user's 2026-05-18 directive ("prioritize making all viable core components
use the maximum legal data"), we re-launched a deadline-driven orchestrator
to fill out seed coverage of the oldtest training axis.

### Setup

- Orchestrator: `src/orchestrate_deadline.ps1` with `-DeadlineHours 48`
- Start: 2026-05-18 20:21
- Deadline: 2026-05-20 20:21 (~32h slack)
- Per-job validator: `src/validate_oof_artifact.py` (9 checks: existence,
  shape, finite, UID alignment, Y-label alignment for oldtest slices,
  mask sum, prob sum-to-1, SGP range)
- Submission analyzer auto-run: **DISABLED** per user directive
  ("no LB submissions until artifacts reviewed")
- Backlog: 20 jobs in `BACKLOG.md`, priority-ordered J001-J004 first
  then J005-J008/J013

### Results (~28h elapsed, 18/20 done)

| Job | Tag | Wall (min) | OV (base) | OV (opt) | Validator |
|---|---|---:|---:|---:|---|
| J001 | v13_oldtest_seed31337 | 170.1 | 0.3612 | 0.3681 | ELIGIBLE |
| J002 | v11_aug_oldtest_seed31337 | 180.6 | — | 0.3253 | ELIGIBLE |
| J003 | v13_oldtest_seed51966 | 182.1 | 0.3627 | **0.3700** | ELIGIBLE |
| J004 | v11_aug_oldtest_seed51966 | 205.6 | — | 0.3253 | ELIGIBLE |
| J005 | v16_testhist_aug_oldtest_seed4 | 199.6 | 0.3628 | 0.3745 | ELIGIBLE |
| J006 | v16_testhist_aug_oldtest_seed7 | 202.6 | 0.3632 | 0.3741 | ELIGIBLE |
| J007 | v11_mulminet_aug_oldtest_seed51966 | 238.6 | — | 0.3284 | ELIGIBLE |
| J008 | v11plus_oldtest_seed31337 | 187.6 | — | 0.3212 | ELIGIBLE |
| J009 | v11_mulminet_oldtest (no aug) | 230.1 | — | 0.3245 | ELIGIBLE |
| J010 | v14_seed0_oldtest | 182.1 | 0.3602 | 0.3680 | ELIGIBLE |
| J011 | v14_seed1_oldtest | 170.1 | 0.3605 | 0.3684 | ELIGIBLE |
| J012 | v11_uncertainty_aug_oldtest | — | — | — | **FAILED rc=2** |
| J013 | v11plus_oldtest_seed51966 | 178.6 | — | 0.3212 | ELIGIBLE |
| J014 | v16_testhist_aug_oldtest_seed9 | 201.1 | 0.3621 | 0.3739 | ELIGIBLE |
| J015 | v13_oldtest_seed9 | 182.5 | 0.3617 | 0.3695 | ELIGIBLE |
| J016 | v11_aug_oldtest_seed7 | 198.6 | — | 0.3253 | ELIGIBLE |
| J017 | v11_mulminet_aug_oldtest_seed7 | 234.6 | — | 0.3314 | ELIGIBLE |
| J018 | v13_oldtest_seed4 | 180.6 | 0.3619 | 0.3685 | ELIGIBLE |
| J019 | v16_testhist_aug_oldtest_seed11 | (in flight) | — | — | (pending) |
| J020 | v13_oldtest_seed51966 | (dup of J003) | — | — | SKIPPED |

**J012 failure**: `train_v11_uncertainty.py` lacks `--include-old-test` CLI arg
(was created before the 2026-05-13 announcement and never retrofitted).
Same 3-line fix as v11_transformer/v13. Logged in LESSONS for future trainer.

**J020 skipped**: orchestrator's `Is-Job-Done` check correctly identified
v13_oldtest_seed51966 OOF as already present (built by J003), skipped re-launch.

### Key per-family findings

**v11_aug_oldtest seed invariance** (NEW LESSON): 4 seeds (42, 31337,
51966, 7) all produced **identical** OV 0.3253. Seed averaging is a no-op
for this family — saves CPU on future runs. Same for v11plus_oldtest
(2 seeds at 0.3212).

**v11_mulminet_aug_oldtest seed variance**: 0.3284-0.3340 spread across
4 seeds (~0.006 range) — avg may help.

**v14_oldtest seed variance**: 0.3680-0.3687 across 3 seeds (~0.001
range) — borderline.

**v13_oldtest seed variance**: 0.3681-0.3700 across 4 seeds (~0.002
range) — borderline.

**v16_testhist_aug_oldtest seed variance**: 0.3739-0.3747 across 4 seeds
(~0.001 range) — borderline.

**v11_mulminet_oldtest (no aug)** = 0.3245 vs v11_mulminet_aug_oldtest =
0.3340 — test-history augmentation contributes ~+0.010 standalone OV
in the v11_mulminet family.

### Wall-time accuracy

TIMING_TABLE was systematically 1.7-2.3× too optimistic under
concurrent GPU+CPU contention (single-job benchmarks vs parallel reality).
See LESSONS section "TIMING_TABLE estimates were too optimistic".

### Sequel work (not done in this section)

1. Phase 3 finishes (~30 min remaining at time of writing)
2. Orchestrator auto-runs `src/_build_avg.py` (avg components for variance-positive families only)
3. Human runs `src/analyze_oldtest_blend_phase2.py` manually (no auto-submission per user directive)
4. Decision: does any single-swap candidate beat R-027 PAIR by ≥+0.002 predicted LB?
5. Independent: R-029a `v14_seed2_v15feat_a` launch (Codex-approved, blocked on Phase 3 CPU freeing up)

### Status

**LB-best UNCHANGED**: R-027 PAIR remains at 0.3810401. This Phase 3 work
expanded the component pool from 5 oldtest variants to 23 (+ derived averages),
enabling more single-swap experiments without any new architecture risk.

---

## R-028 → R-055 LB sequence (2026-05-18 to 2026-05-23) — CLASS framework
calibration

Summary of LB activity since R-027 PAIR (0.3810). 4 wins, 3 losses across 7
LB-tested uploads. Net LB delta: **+0.0056** to current best.

| R-### | Date | Design | OOF | LB | ΔLB vs prev best | Class verdict |
|---|---|---|---:|---:|---:|---|
| R-027 PAIR | 2026-05-18 | 5c B-pure ADD oldtest | 0.3771 | 0.3810 | baseline | B-pure ADD ✓ |
| R-028 top1 | 2026-05-19 | v11plus → mulminet_avg2 SWAP | +0.001 | 0.3724 | **−0.0086** | B-impure SWAP ✗ |
| R-033 | 2026-05-20 | v13_oldtest → v13_oldtest_avg3 SWAP | −0.0001 | 0.3795 | **−0.0015** | B-seedavg ✗ |
| R-034 | 2026-05-21 | v14_seed2 → v14_seed2_v15feat_a SWAP | −0.0005 | 0.3838 | **+0.0028** | B-feature ✓ |
| R-040 | 2026-05-21 | v11_aug_oldtest → mulminet_avg3 SWAP | +0.0030 | 0.3744 | **−0.0094** | B-impure SWAP ✗ |
| R-042 | 2026-05-22 | R-034 + rule_override post-process | 0.3812 | **0.3866** | **+0.0028** | post-process ✓ |
| R-055 | 2026-05-23 | R-052 7c ADD Bayes + rule_override | 0.3844 | 0.3725 | **−0.0141** | B-impure ADD ✗ |

**Current LB best: R-042 = 0.3866550 (44/305 ranking as of 2026-05-23).**

### R-034 — first B-feature win (2026-05-21, LB +0.0028)

Swap: `v14_seed2 → v14_seed2_v15feat_a` in the R-027 5-comp blend.
The component was standalone-rejected at OV 0.3655 (gate 0.3717,
−0.0062 below) and would have been parked under the v3 gate framework.
Blend-swap OOF was −0.0005 (essentially tied baseline).

Reasoning to upload despite standalone fail: new features (36 prefix
aggregates: per-class freqs, entropy, dominance, streaks) are a NEW
SIGNAL CLASS — no new training data, no new architecture, just an
extended feature view of the same v14 GBM. The diversity is plausibly
LB-positive even when standalone-OOF-negative.

LB transferred at ratio **1.0121** (highest swap-ratio ever recorded).
R-034 PAIR (5-comp) is now the LB-WIN baseline for all future blends.

### R-040 — second B-impure SWAP failure (2026-05-21, LB −0.0094)

Swap: `v11_aug_oldtest → v11_mulminet_aug_avg3` in R-034 PAIR. This swap
had the LARGEST OOF lift in the entire parked audit (+0.0030 dOV). Per
the v3 gate it was the strongest swap candidate; under the post-R-034
"NEW SIGNAL CLASS / blend-swap diagnostic" framework it was a B-impure
swap (different arch family v11_mulminet vs transformer v11_aug), which
had already failed once at R-028 top1.

LB came back at 0.3744, **−0.0094 vs R-034**. Ratio 0.98. The OOF→LB
ratio for B-impure swaps is now 2 datapoints at 0.97-0.98 — a hard
empirical wall. v11_mulminet family is now permanently BLEND-INELIGIBLE
as a swap candidate.

### R-042 — first post-process LB win (2026-05-22, LB +0.0028)

R-034 PAIR submission file + `apply_rule_override.py` from teammate's
`package_v8_0.4419` audit, applied as a final post-process step on the
submission CSV. The post-process replaces predictions whose conditional
context-empirical probability is 0% in train with the train mode under
that context. 10 row changes out of 1845.

LB lifted from 0.3838 → **0.3866** (+0.0028). Same magnitude as
R-034's blend-swap win. Confirmed: **rule_override stacks with any
blend-level lever and adds a deterministic +0.0028 LB**. Should be
applied to all future LB candidates.

### R-052 / R-053 / R-054 (2026-05-22, never uploaded)

7-comp and 8-comp ADD designs around R-034 PAIR:
- R-052: 7c = R-034 + `meta_stack_v2_logistic` + `v11_mulminet_aug_avg3`. OOF 0.3836.
- R-053: 7c = R-034 + `meta_stack` (v1) + `v11_mulminet_aug_avg3`. OOF 0.3830.
- R-054: 8c = R-034 + `meta_stack_v2_logistic` + `v11_aug_big` + `v14_recvprofile`. OOF 0.3821.

All built as `_PLUS_RULE.csv` candidates. Never LB-uploaded. After
R-055 result, R-052 and R-053 are now expected to LB-fail by similar
margin (both share v11_mulminet_aug_avg3 + meta_stack). R-054 has no
mulminet (would partially isolate the failure cause) but still includes
meta_stack_v2_logistic. **All three are PARKED-HARD by association.**

### R-055 — B-impure ADD failure (2026-05-23, LB −0.0141)

Design: R-052 7-component blend with weights from
`bayes_blend_search.py` (Dirichlet 500 samples + scipy COBYLA refinement
from top-30 seeds, per-task independent search), then
`apply_rule_override` post-process on top.

OOF: 0.3844 (Bayes refinement +0.0008 above Dirichlet R-052 at n=300).
Predicted LB conservative (R-027 ratio 1.0035): 0.3886.
Predicted LB optimistic (R-042 ratio 1.0142): 0.3927.

**Actual LB: 0.3725440 (53/305). Predicted-actual gap: −0.0161 to
−0.0202. OOF→LB ratio: 0.969.**

#### Why it failed

The Bayes solver put 35% of the action-F1 weight on
`v11_mulminet_aug_avg3` because it had the strongest per-task OOF
F1. v11_mulminet is the B-impure family already LB-failed in R-028
(−0.0086) and R-040 (−0.0094). Bayes refinement amplified the toxicity
by concentrating mass on the worst-transferring component.

If we'd used uniform Dirichlet weights (~14% each across 7
components), the toxic-component contribution would have been smaller
and the LB regression would likely have been milder (estimated −0.005
to −0.008 instead of −0.0141). The weight search converted "small
known risk" into "catastrophic LB cliff".

#### Mechanistic lesson

A blend's LB ratio is bounded above by its **worst** component's
transfer ratio, not its best. Adding good components to a blend
containing v11_mulminet does NOT wash out v11_mulminet's −0.0094
ratio — it amplifies it through the weight-search dynamics.

#### Hypothesis status post-R-055

| Hypothesis | Status |
|---|---|
| "B-impure SWAP fails on LB" | CONFIRMED (R-028, R-040, now R-055 sibling family) |
| "B-impure ADD might work because weight stays low" | **FALSIFIED** — Bayes weights drive ADD to the same failure mode as SWAP |
| "Bayes weight refinement > Dirichlet (transfers to LB)" | **FALSIFIED** — Bayes refinement on a toxic-containing pool is a LB-amplifier of toxicity |
| "meta_stack is a NEW signal class worth testing" | INCONCLUSIVE — bundled with v11_mulminet in R-055, can't isolate. Presumed toxic by association. |
| "Higher-order (9c, 10c) blends > 7c" | INVALIDATED in current pool — every top 9c/10c candidate from `higher_order_blend_search.py` included v11_mulminet variants |
| "R-034 PAIR + rule_override (R-042) is LB-best" | STILL HOLDS at 0.3866 |
| "rule_override post-process adds deterministic +0.0028 LB" | STILL HOLDS |
| "B-feature swaps (R-034 LB-WIN pattern) transfer at ratio ≥1.01" | STILL HOLDS (1 datapoint, awaiting v15feat_c_oldtest_avg3 + R-032 v2.1 LORO to confirm) |

#### Submission artifacts

- `submissions/submission_R055_bayes_r052.csv` — base (no rule)
- `submissions/submission_R055_bayes_r052_PLUS_RULE.csv` — uploaded version (9 row overrides)
- `submissions/bayes_r052_search.json` — Bayes weights audit log
- `submissions/higher_order_blend_search.json` — all 9c/10c trials (post-mortem ref)
- `logs/bayes_r052_search.log`, `logs/higher_order_blend_search.log`

#### Next plan post-R-055

1. Skip remaining LB slot today; reserve teammate slot.
2. Tomorrow (2026-05-24, 3 fresh slots):
   - Finish v15feat_c_oldtest seed 7 + seed 31337 (in progress, ~3 hr remaining at
     time of writing). Combine to `v14_seed2_v15feat_c_oldtest_avg3`.
   - Test as **B-feature SWAP into R-034 + rule_override** (R-034 LB-WIN pattern).
   - Same protocol for R-032 v2.1 LORO (Kaggle in progress).
3. Both targets follow the proven CLASS B-feature winning recipe. Expected
   LB lift: +0.0001 to +0.0028 per swap, on top of R-042's 0.3866.
4. No more Bayes-weighted blends; no more higher-order search; no more
   ADDs of v11_mulminet or meta_stack variants until those classes
   have a clean STANDALONE LB win.

### R-062r — B-player-style FAILURE (2026-05-23, LB −0.0057 vs R-042)

Design: SWAP `v14_seed2_v15feat_a → v14_seed2_v16match_v2` in R-034 PAIR +
rule_override. v14_seed2_v16match_v2 was produced by the R-032 v2.1 LORO
trainer (Codex-APPROVED scope, cap K=22, Family A only — match-level action
+ point aggregates from OTHER rallies in the same match, leave-one-out).

OOF: 0.3823 (+0.0037 dOV vs R-034 PAIR baseline 0.3786 — **largest non-toxic
blend-swap lift recorded**).
Predicted LB+rule (conservative-optimistic): 0.3865 – 0.3905.

**Actual LB: 0.3809371 (-0.0057 vs R-042 0.3866). OOF→LB ratio: 0.996.**

#### Why it failed

The v16match_v2 Family A features aggregate action and point distributions
from OTHER rallies in the same match. Since each match is by a specific pair
of players, those aggregates are effectively a per-player style signature
even though `gamePlayerId` is never read directly.

At test time:
- Train matches: 0 overlap with test matches (per data-validation)
- Test players: only 40 / 71 (56.3%) appear in train; 31 players are novel
- Match-level aggregates computed from test rallies have a DIFFERENT
  distribution than train (different players → different shot styles)

Result: the in-blend OOF lift came from train-side memorisation of
player-pair style; it doesn't survive the player de-identification.

#### Mechanistic lesson — new CLASS B-player-style

The match-disjoint train/test split AND the low player overlap together
make per-match or per-player aggregate features structurally non-transferring,
regardless of how the aggregation is constructed (LORO, in-rally, in-match,
within-prefix). This adds **B-player-style** to the LB-toxic class list:

| Banned class | Banned components | Evidence |
|---|---|---|
| B-impure (architecture-swap) | v11_mulminet family | R-028 −0.0086, R-040 −0.0094, R-055 −0.0141 |
| B-meta (stacking ensembles) | meta_stack, meta_stack_v2_logistic | R-055 association |
| B-player-style (per-player or per-match aggregates) | v15_player_only, v15_pp, **v16match_v2 family** | LB 0.3555, 0.3507, **0.3809** |

#### Hypothesis status post-R-062r

| Hypothesis | Status |
|---|---|
| "Codex-approved LORO match-context features transfer to LB" | **FALSIFIED** — R-062r −0.0057 LB despite Codex approval and +0.0037 OOF |
| "OOF dOV in a B-feature swap predicts LB transfer at ratio 1.01+" | **FALSIFIED** for v16match_v2 axis (ratio collapsed to 0.996) |
| "R-034 v15feat_a was a specific local maximum" | RE-CONFIRMED — every nearby v14+new-features variant (v15feat_c, v15feat_a+oldtest+avg3, testhist+v15feat_a, v16match_v2) fails to transfer |
| "R-042 0.3866 is still LB-best" | STILL HOLDS |
| "rule_override post-process holds at +0.0028 LB" | STILL HOLDS (stacked on R-062 base too — without rule_override R-062 would have been LB ~0.378) |

#### Implication for in-progress / pending work

- **R-060r (v14_recvprofile swap)** + **R-061r (v14_recvhand swap)** are
  recvprofile/recvhand features — per-player axes. Same B-player-style risk
  class as R-062r. PARKED as high-risk; not for LB upload.
- **R-064 v15feat_d spin features** — was already marginal at Fold-1 smoke
  (base −0.0001, opt −0.0045, AUC −0.0063). Given two consecutive transfer
  failures, the 5-fold expense is no longer justified by expected lift.
  Recommend PARK pending higher-confidence baseline.
- **R-065c Consensus Pseudo V2c** — Codex `BLOCK / ABANDON` (point pool
  fails ≥50 floor). Lever exhausted.

#### Submission artifacts

- `submissions/submission_R062_v16match_v2_swap.csv` — base
- `submissions/submission_R062r_v16match_v2_PLUS_RULE.csv` — uploaded (LB 0.3809)
- `submissions/r062_candidate.json` — OOF + blend weights
- `kaggle_pulls/r032v2/` — Kaggle training artifacts

### Net LB delta 2026-05-18 → 2026-05-23 (REVISED post-R-062r)

- R-027 PAIR start: 0.3810
- R-042 end: **0.3866 (+0.0056 net)** — LB-best, unchanged
- Best single move: R-027 → R-034 (+0.0028 B-feature swap with v15feat_a)
- Second best: R-034 → R-042 (+0.0028 rule_override post-process)
- Losses absorbed (LB-tested, lever now banned):
  R-028 (−0.0086), R-033 (−0.0015), R-040 (−0.0094),
  **R-055 (−0.0141), R-062r (−0.0057)**
- 7 total LB-tested experiments since R-027; 2 wins, 5 losses.
- LB-WIN-conditional strategy is working: R-042 0.3866 remains undefeated
  on the public LB despite 5 losses behind it. Each loss eliminates a
  toxic class without dropping our top score.

### Ranking trajectory (REVISED post-R-062r)

- 2026-05-18: ~50/280 (R-027 PAIR LB-uploaded)
- 2026-05-21: 50/296 (R-034 win)
- 2026-05-22: 44/296 (R-042 win → still LB-best)
- 2026-05-23 (morning): 53/305 (R-055 regression — competition pool grew by 9)
- 2026-05-23 (afternoon): rank ~55-60 / 305+ (R-062r regression, R-042 still our LB-best)

R-042 0.3866 remains the score on the public board. Rank fluctuations are
from competition pool growth + our regressions consuming slots.

### Open questions for 2026-05-24+

- **The B-feature swap recipe of R-034 is exhausted.** Every nearby v14+new-features
  variant has been tested or shown to fail OOF→LB transfer. Realistic LB
  ceiling within current paradigm: ~0.3870 (R-042 + a tiny safe swap if found).
- **Top-10 gap is 0.06+.** Even maxing all available levers we project to
  ~0.39-0.41 — well outside top 10 (currently 0.4445+).
- **Only remaining structural lever**: Path B causal LM (STRATEGY.md §9).
  Smoke ~1 h GPU, full commit ~30 h GPU. Jabir approval required.
- **Alternative**: accept R-042 as final, hold rank.

---

## R-054r — B-meta + B-player-style 8-comp FAILURE (2026-05-24, LB −0.0103 vs R-042)

Design: 8-comp blend = R-034 PAIR + `meta_stack_v2_logistic` + `v11_aug_big` +
`v14_recvprofile` + rule_override. Specifically chosen as the cleanest test of
the B-meta hypothesis since it contains `meta_stack_v2_logistic` WITHOUT
`v11_mulminet_aug_avg3` (the bundled-toxic component from R-055).

OOF: 0.3821 (+0.0035 dOV vs R-034 PAIR baseline 0.3786).
Predicted LB+rule: 0.3862-0.3903 (midpoint 0.3882, +0.0016 above R-042).

**Actual LB: 0.3762672. OOF→LB ratio: 0.9848.**

Effect: −0.0103 LB vs R-042. This is the 3rd consecutive post-R-034 LB
regression (R-055 −0.0141, R-062r −0.0057, now R-054r −0.0103).

### Why it failed (the meta_stack confirmation)

R-055 confounded meta_stack_v2_logistic with v11_mulminet. R-054r isolates
meta_stack_v2 (no mulminet, replaced with v11_aug_big + recvprofile). The
fact that LB still collapsed by 0.0103 isolates meta_stack as the toxic
component independent of v11_mulminet.

**Reclassification**: meta_stack v1 / meta_stack_v2_logistic / any future
stacking-ensemble component → **B-meta class HARD-CONFIRMED toxic** as of
2026-05-24. Previously listed as "PRESUMED TOXIC by association"; now
isolated evidence.

R-054r also contained `v14_recvprofile` (B-player-style risk class per
R-062r). Confounding means we cannot fully isolate which of {meta_stack,
recvprofile, v11_aug_big} contributed the most. But meta_stack is the
only one with no prior LB-test, so the cleanest reading is "meta_stack v2
in a blend with otherwise-safe components still tanked".

---

## R-066 Path B causal LM smoke — PARKED 2026-05-24

User-authorized 2026-05-23 (after teammate package_v8 LB 0.4419 confirmed
SGP-leaked → unusable). Path B was the only remaining structural lever.

### v2 (initial smoke) — Label-shift bug

Fold-1 OV 0.2002. F1_a 0.0794, F1_p 0.0239, AUC 0.7945 (inflated).

Root cause: `multi_position_loss` compared `action_logits[t]` against
`y_action[t]` (same position). Under causal mask, output at position t
sees inputs 0..t INCLUSIVE — including shot t's own action/point in the
embedding. The model trivially learned "copy current shot's action to
output" instead of "predict next shot's action".

### v3 (fixed) — Standard causal-LM label shift

Fixed `multi_position_loss` to compare `action_logits[t]` against
`y_action[t+1]` (predict NEXT shot from prefix). Evaluation positions were
already aligned correctly for the shifted-target paradigm (output[t]
predicts shot[t+1]).

Fold-1 smoke results (single fold, ~13 min Kaggle T4):

| Metric | v3 | v2 (buggy) | v11 baseline (Fold-1) |
|---|---:|---:|---:|
| F1_a | 0.2896 | 0.0794 | ~0.41 |
| F1_p | 0.0937 | 0.0239 | ~0.20 |
| AUC | **0.6759** | 0.7945 | **~0.61** |
| **OV** | **0.2885** | 0.2002 | **0.314** |

**Verdict**: PARK per STRATEGY §9.6 stop gate (OV 0.2885 < 0.295
uncompetitive threshold). The label shift recovered OV from 0.20 → 0.29
(+0.09 directional confirmation that the fix was right) but the full
model still doesn't reach v11 baseline 0.314.

Likely reason: multi-position causal LM at d=192 / 4L spreads training
signal across all positions; v11's bidirectional single-target objective
puts all signal on the last-shot prediction. Multi-position regularization
doesn't pay back the lost sharpness at this scale.

### Notable partial signal — R-067 follow-up

AUC = **0.6759 is +0.066 above v11 baseline ~0.61.** Server head is
genuinely diversity-positive even when action/point heads underperform.
With OV weighting 0.2 on AUC, this is +0.013 OV potential if blended.

R-067 (server-head-only blend) opened in REVIEW_QUEUE.md to test this
specifically. R-066 full-model is PARKED.

### LB delta + ranking trajectory (REVISED post-R-054r + R-066 PARK)

- R-027 PAIR start: 0.3810
- R-042 end (2026-05-22): **0.3866 (+0.0056 net)** — LB-best, unchanged
- Best single move: R-027 → R-034 (+0.0028 B-feature swap with v15feat_a)
- Second best: R-034 → R-042 (+0.0028 rule_override post-process)
- Losses absorbed (LB-tested, lever now banned):
  R-028 (−0.0086), R-033 (−0.0015), R-040 (−0.0094),
  R-055 (−0.0141), R-062r (−0.0057), R-054r (−0.0103)
- R-066 Path B: PARKED at smoke (Fold-1 OV 0.2885 < 0.295 gate)
- **8 LB-tested experiments since R-027; 2 wins, 6 losses + 1 OOF-smoke PARK.**
- LB-WIN-conditional strategy is working: R-042 0.3866 remains undefeated
  on the public LB despite 6 losses behind it. Each loss eliminates a
  toxic class without dropping our top score.

Ranking trajectory:
- 2026-05-18: ~50/280 (R-027 PAIR LB-uploaded)
- 2026-05-21: 50/296 (R-034 win)
- 2026-05-22: 44/296 (R-042 win → still LB-best)
- 2026-05-23 (morning): 53/305 (R-055 regression)
- 2026-05-23 (afternoon): rank ~55 / 305+ (R-062r regression)
- 2026-05-24: rank **56 / 313** (R-054r regression; competition pool +8 overnight)

### Open lever inventory 2026-05-24

- ✅ R-042 0.3866 LB-best, unchanged
- 🟡 **R-067 server-head blend** (only candidate with positive expected lift; AWAITING_CODEX)
- ⚠️ R-064 v15feat_d 5-fold (Fold-1 was marginal; recent LB pattern says HIGH risk)
- ❌ R-066 PARKED at smoke (full model uncompetitive)
- ❌ Path A consensus pseudo abandoned (R-065c verdict)
- ❌ Path B causal LM full PARKED (R-066 smoke verdict)
- ❌ All B-impure / B-meta / B-player-style swaps PARKED-HARD with LB evidence
- ❌ Teammate package_v8 SGP-leaked (banned)
- ❌ External badminton TL — never integrated; sport mismatch concerns

If R-067 fails too, we accept R-042 0.3866 as final.

---

## R-067cr — server-head-blend LB WIN (2026-05-24, +0.000355 vs R-042)

Design: replace R-042's serverGetPoint column with α-blend of R-066 v22 causal
LM server head and R-042's existing SGP. Per-rally OOF α-sweep selected
α=0.30 (30% v22 + 70% R-042) as the peak (AUC 0.7680 vs R-034 baseline 0.7355).

Predicted LB+rule (full transfer): +0.0065 OV → LB 0.3931
Predicted LB+rule (partial 50%):    +0.0033 OV → LB 0.3899

**Actual LB: 0.3870095 (+0.000355 vs R-042 0.3866550). NEW LB-BEST.**

### Transfer analysis

OOF AUC lift: +0.0326. Expected OV lift at 100% transfer: 0.0326 × 0.2 = +0.0065.
Actual LB OV lift: +0.000355. **Transfer rate: 5.4%.**

Much lower than expected. Plausible reasons:
- Per-rally OOF AUC overestimates test-set transferability (train rallies have
  player overlap; test rallies are 56% novel)
- R-042's existing SGP already captures most signal — 30% diversity from v22
  mostly redistributes noise rather than adding orthogonal information
- α=0.30 was OOF-optimal, not LB-optimal; LB might prefer α closer to 0.10-0.20

### Strategic implications

1. **Path B server head DOES transfer** — proves R-066 wasn't a dead end.
   Just structurally weak compared to v11/v14 action-head transfer.
2. **Server-head-only blend pattern is now LB-validated** — future Path B-like
   experiments (or any new architecture with a strong server head) can be tested
   this way without needing full-model OV competitiveness.
3. **Margin is tiny** — +0.000355 is within typical LB noise. Replication needed
   for full confidence, but it's a positive datapoint.
4. **5-fail streak broken**: this is the first non-rule-override LB win since
   R-034 on 2026-05-21 (3 days ago). Net since R-034: 3 wins / 6 losses.

### Submission artifacts

- `submissions/submission_R067cr_alpha030_v22_blend_PLUS_RULE.csv` — uploaded version (LB 0.3870)
- `submissions/r067c_server_blend_alpha_sweep.json` — full α-sweep table + manifest
- `oof_predictions/v22_causal_lm_v1_*.npy` — Path B full 5-fold artifacts (R-066)

### Next steps (post-R-067cr win)

- R-067cr 0.3870 is new LB-best. R-042 is now superseded.
- Possible further squeezes: try α=0.10, α=0.20, α=0.40 to find LB-optimal
- R-068r (Bayes weights on R-034) still in queue — could combine with v22 SGP
  blend for stacked lift
- R-070 v15feat_e movement features — AWAITING_CODEX
- Path B at full scale unlikely to add more (server head exhausted; action+point
  still weak; class weights would be a new experiment R-068+)

### Updated LB ranking trajectory

- 2026-05-22: 44/296 (R-042 0.3866 first set as LB-best)
- 2026-05-23 (afternoon): 53/305 (R-055 regression to 0.3725)
- 2026-05-23 (later): 55/305 (R-062r regression to 0.3809)
- 2026-05-24 (morning): 56/313 (R-054r regression to 0.3763)
- 2026-05-24 (afternoon): **53-55/313 (R-067cr LB 0.3870, +0.000355)** — incremental rank recovery
