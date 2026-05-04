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
| **`submission_zoo_v16_fast_01_v16_v14_seed1_v12_5f_v11.csv`** | **0.37998** | **0.3694863** | **−0.01049** | ✅ **CURRENT BEST** |

OOF−LB gap (|OOF − LB|, sorted by gap):

| Submission | OOF | LB | Gap |
|---|---|---|---|
| **zoo_v16_fast_01 (current best)** | **0.37998** | **0.3694863** | **0.01049** (multi-blend transfers) |
| V16+V11 (prior best) | 0.3743 | 0.3673269 | 0.0070 (OOF underestimated LB) |
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

**Updated verdict:** V16 is now the current best. OOF underestimated this family because Public LB appears more aligned with early-rally / SN=2 distribution than match-GroupKFold OOF. Keep V16 as the main backbone for future work; next experiments should build on V16, not discard it.

---

## 9. P2.1 Multi-Seed V14 (2026-05-04) — IN PROGRESS

Infrastructure:
- `--seed` flag added to `src/train_v14.py` (controls np, LGB, XGB random states)
- `src/avg_oof.py` created (averages raw prob arrays across seeds, then blends with V11)

| Tag | Seed | Base OV | Opt OV | Time |
|---|---|---|---|---|
| v14_seed0 | 42 | 0.3602 | 0.3661 | 197.6 min |
| v14_seed1 | 48879 | 0.3593 | 0.3667 | 196.3 min |
| v14_seed2 | 51966 | — | — | NOT STARTED |

seed0 (seed=42) = V14 baseline (exact match confirms seed plumbing correct).
seed1 (seed=48879) solo opt 0.3667 (+0.0006 vs seed0) — seeds are diversifying.

**Next:** run seed2, then `python src/avg_oof.py --tags v14_seed0 v14_seed1 v14_seed2 --out-tag v14_avg3 --blend-v11`.
Gate for submission: avg3+V11 OOF ≥ 0.3764.
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

Verdict: multi-model blend transfers and becomes the new current best. OOF gain was +0.0057 over V16+V11, Public LB gain is +0.00216. Continue blend-zoo exploration; next best low-cost probe is `submission_zoo_v16_fast_04_per_sn_bucket.csv` because it is structurally different from #1 while still strong OOF (0.37936).
### zoo_v16_fast_04 per-SN bucket — failed LB probe

```text
OOF OV      : 0.37936
Public LB   : 0.3596738 (2026-05-04 16:01, rank 37/184)
Delta vs current best zoo #1: -0.0098125 LB
```

Verdict: per-SN conditional weights overfit OOF and do not transfer. Even though the file was structurally different from zoo #1, Public LB rejects the fine-grained SN bucket optimization. Continue using global/multi-model blends; avoid highly conditional per-SN weight search unless validated by another public probe.

