# RESULTS
## Round: 2026-05-02 / 2026-05-04 — V14, V15 ablation, V16 aug, P2.1 seeds

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
