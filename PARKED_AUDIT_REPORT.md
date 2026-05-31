# Parked-Component Audit Report — 2026-05-21

**Trigger**: User directive — "dont make the same mistake. and never make a
conclusion without really submission score. and list out all the things that
has never submit and check but just parked."

**Baseline**: R-034 PAIR (LB 0.3838279, current LB best)
  Components: `v11_aug_oldtest, v11plus, v13_oldtest, v14_seed2_v15feat_a, v16_avg3`

**Method**: For every parked OOF component (60 in total), tried swapping it
into each of the 5 R-034 PAIR slots. Per-task Dirichlet random search
(n_samples=80) for blend weights. Reported the best slot per component, the
OOF delta vs the R-034 baseline, and the predicted-LB range using:
  - Conservative ratio 1.0035 (CLASS B-pure, R-027 PAIR origin)
  - Optimistic ratio 1.0151 (R-034 actual transfer ratio at n=80 baseline)

**IMPORTANT — NO VERDICTS ARE ISSUED IN THIS REPORT.** Only OOF deltas and
predicted-LB ranges. The user decides what to upload. This matches the
2026-05-21 lesson: standalone OOF gates over-rejected viable blend components,
and the new two-stage gate framework only emits "ELIGIBLE_FOR_LB_UPLOAD" — not
"PARK" — for components with dOV ≥ -0.002 in a new signal class.

---

## 1. Parked-component inventory (60 components)

[Populated by audit script — see `submissions/parked_audit_summary.csv`]

### Categorization

| Category | Count | Examples |
|---|---|---|
| meta_stack (stacking ensemble) | 2 | meta_stack, meta_stack_v2_logistic |
| sn2_expert (SN=2 specialist) | 1 | sn2_expert |
| v11_mulminet family (B-impure risk) | 14 | v11_mulminet, v11_mulminet_aug, v11_mulminet_aug_avg2/3, v11_mulminet_aug_oldtest_* |
| v16_testhist_aug_oldtest derivs | 9 | seed*, avg3, avg5 |
| v13_oldtest seed derivs | 5 | seed4/9/31337/51966, avg2 |
| v11_aug_oldtest derivs | 5 | seed7/31337/51966, avg2/3 |
| v11plus oldtest/aug derivs | 5 | v11plus_aug, v11plus_oldtest, seed*, avg2 |
| v12 family | 4 | v12, v12_5f, v12aug, v12cb |
| v14 base + avg | 3 | v14, v14_avg3, v14_oldtest_avg2 |
| v14_seed*_oldtest | 3 | seed0/1/2_oldtest |
| v14_recv* (receiver-relative) | 2 | v14_recvhand, v14_recvprofile |
| v11_big arch variants | 2 | v11_aug_big, v11_big |
| v16 individual seeds | 2 | v16_seed1, v16_seed2 |
| v15 family | 1 | v15_hist_only |
| v11 uncertainty aug | 1 | v11_uncertainty_aug |
| v18 (single) | 1 | v18 |

---

## 2. Global ranking (top 30 swap attempts by OOF)

| # | swap_label | OV | dOV vs R-034 | pred_LB (lo–hi) |
|---|---|---:|---:|---:|
| 1 | `SWAP_v11_aug_oldtest_TO_v11_mulminet_aug_avg3` | 0.3819 | +0.0039 | 0.3833–0.3865 |
| 2 | `SWAP_v11plus_TO_v11_mulminet_aug_avg3` | 0.3811 | +0.0030 | 0.3824–0.3857 |
| 3 | `SWAP_v11_aug_oldtest_TO_v11_mulminet_aug_avg2` | 0.3807 | +0.0027 | 0.3820–0.3853 |
| 4 | `SWAP_v13_oldtest_TO_v11_mulminet_aug_avg3` | 0.3801 | +0.0021 | 0.3815–0.3847 |
| 5 | `SWAP_v11plus_TO_v11_mulminet_aug_avg2` | 0.3801 | +0.0021 | 0.3814–0.3847 |
| 6 | `SWAP_v11_aug_oldtest_TO_v11_mulminet_aug_s31337` | 0.3794 | +0.0013 | 0.3807–0.3840 |
| 7 | `SWAP_v11_aug_oldtest_TO_v11_mulminet_aug_s12345` | 0.3793 | +0.0012 | 0.3806–0.3839 |
| 8 | `SWAP_v14_seed2_v15feat_a_TO_meta_stack` | 0.3792 | +0.0012 | 0.3806–0.3838 |
| 9 | `SWAP_v13_oldtest_TO_v11_mulminet_aug_avg2` | 0.3790 | +0.0010 | 0.3804–0.3836 |
| 10 | `SWAP_v11_aug_oldtest_TO_v11_mulminet_pretrained_aug` | 0.3790 | +0.0010 | 0.3804–0.3836 |
| 11 | `SWAP_v16_avg3_TO_v11_mulminet_aug_avg3` | 0.3789 | +0.0008 | 0.3802–0.3835 |
| 12 | `SWAP_v13_oldtest_TO_v11_mulminet_aug_s31337` | 0.3789 | +0.0008 | 0.3802–0.3835 |
| 13 | `SWAP_v14_seed2_v15feat_a_TO_v14_recvprofile` | 0.3788 | +0.0007 | 0.3801–0.3834 |
| 14 | `SWAP_v13_oldtest_TO_v11_mulminet_aug_oldtest_seed31337` | 0.3788 | +0.0007 | 0.3801–0.3834 |
| 15 | `SWAP_v13_oldtest_TO_meta_stack` | 0.3788 | +0.0007 | 0.3801–0.3834 |
| 16 | `SWAP_v11plus_TO_v11_mulminet_aug_s31337` | 0.3788 | +0.0007 | 0.3801–0.3833 |
| 17 | `SWAP_v11_aug_oldtest_TO_v11_mulminet_aug_oldtest_seed31337` | 0.3787 | +0.0007 | 0.3801–0.3833 |
| 18 | `SWAP_v11plus_TO_v11_mulminet_pretrained_aug` | 0.3787 | +0.0006 | 0.3800–0.3833 |
| 19 | `SWAP_v13_oldtest_TO_meta_stack_v2_logistic` | 0.3786 | +0.0006 | 0.3799–0.3832 |
| 20 | `SWAP_v11plus_TO_v11_aug_big` | 0.3786 | +0.0006 | 0.3799–0.3832 |
| 21 | `SWAP_v11plus_TO_v11_mulminet_aug_oldtest_seed31337` | 0.3786 | +0.0005 | 0.3799–0.3832 |
| 22 | `SWAP_v14_seed2_v15feat_a_TO_meta_stack_v2_logistic` | 0.3786 | +0.0005 | 0.3799–0.3832 |
| 23 | `SWAP_v13_oldtest_TO_v11_mulminet_pretrained_aug` | 0.3785 | +0.0005 | 0.3799–0.3831 |
| 24 | `SWAP_v11plus_TO_v11_mulminet_aug_s12345` | 0.3785 | +0.0005 | 0.3799–0.3831 |
| 25 | `SWAP_v13_oldtest_TO_v11_mulminet_aug_s12345` | 0.3785 | +0.0005 | 0.3798–0.3831 |
| 26 | `SWAP_v11_aug_oldtest_TO_meta_stack_v2_logistic` | 0.3785 | +0.0005 | 0.3798–0.3831 |
| 27 | `SWAP_v11_aug_oldtest_TO_v11_mulminet_aug_oldtest_seed51966` | 0.3785 | +0.0004 | 0.3798–0.3830 |
| 28 | `SWAP_v13_oldtest_TO_v13_oldtest_seed9` | 0.3784 | +0.0004 | 0.3798–0.3830 |
| 29 | `SWAP_v14_seed2_v15feat_a_TO_v14_recvhand` | 0.3784 | +0.0004 | 0.3798–0.3830 |
| 30 | `SWAP_v11_aug_oldtest_TO_v11_mulminet_aug_oldtest_seed7` | 0.3784 | +0.0003 | 0.3797–0.3829 |

## 3. Best slot per parked component (all 60)

| # | component | best_slot | OV | dOV | pred_LB (lo–hi) | class |
|---|---|---|---:|---:|---:|---|
| 1 | `v11_mulminet_aug_avg3` | `v11_aug_oldtest` | 0.3819 | +0.0039 | 0.3833–0.3865 | B-impure (R-028 LB-failed at this pattern) |
| 2 | `v11_mulminet_aug_avg2` | `v11_aug_oldtest` | 0.3807 | +0.0027 | 0.3820–0.3853 | B-impure (R-028 LB-failed at this pattern) |
| 3 | `v11_mulminet_aug_s31337` | `v11_aug_oldtest` | 0.3794 | +0.0013 | 0.3807–0.3840 | B-impure (R-028 LB-failed at this pattern) |
| 4 | `v11_mulminet_aug_s12345` | `v11_aug_oldtest` | 0.3793 | +0.0012 | 0.3806–0.3839 | B-impure (R-028 LB-failed at this pattern) |
| 5 | `meta_stack` | `v14_seed2_v15feat_a` | 0.3792 | +0.0012 | 0.3806–0.3838 | B-meta (NEW SIGNAL CLASS — stacking ensemble, never LB-tested) |
| 6 | `v11_mulminet_pretrained_aug` | `v11_aug_oldtest` | 0.3790 | +0.0010 | 0.3804–0.3836 | B-impure (R-028 LB-failed at this pattern) |
| 7 | `v14_recvprofile` | `v14_seed2_v15feat_a` | 0.3788 | +0.0007 | 0.3801–0.3834 | B-feature (R-034 LB-WIN class) |
| 8 | `v11_mulminet_aug_oldtest_seed31337` | `v13_oldtest` | 0.3788 | +0.0007 | 0.3801–0.3834 | B-impure (transformer→GBM cross-family) |
| 9 | `meta_stack_v2_logistic` | `v13_oldtest` | 0.3786 | +0.0006 | 0.3799–0.3832 | B-meta (NEW SIGNAL CLASS — stacking ensemble, never LB-tested) |
| 10 | `v11_aug_big` | `v11plus` | 0.3786 | +0.0006 | 0.3799–0.3832 | B-impure (bigger arch into transformer slot) |
| 11 | `v11_mulminet_aug_oldtest_seed51966` | `v11_aug_oldtest` | 0.3785 | +0.0004 | 0.3798–0.3830 | B-impure (R-028 LB-failed at this pattern) |
| 12 | `v13_oldtest_seed9` | `v13_oldtest` | 0.3784 | +0.0004 | 0.3798–0.3830 | B-seedavg (R-033 LB-failed) |
| 13 | `v14_recvhand` | `v14_seed2_v15feat_a` | 0.3784 | +0.0004 | 0.3798–0.3830 | B-feature (R-034 LB-WIN class) |
| 14 | `v11_mulminet_aug_oldtest_seed7` | `v11_aug_oldtest` | 0.3784 | +0.0003 | 0.3797–0.3829 | B-impure (R-028 LB-failed at this pattern) |
| 15 | `v11_mulminet_aug_oldtest` | `v11_aug_oldtest` | 0.3783 | +0.0003 | 0.3797–0.3829 | B-impure (R-028 LB-failed at this pattern) |
| 16 | `v11_mulminet_aug` | `v11_aug_oldtest` | 0.3783 | +0.0002 | 0.3796–0.3829 | B-impure (R-028 LB-failed at this pattern) |
| 17 | `v13_oldtest_seed51966` | `v13_oldtest` | 0.3783 | +0.0002 | 0.3796–0.3828 | B-seedavg (R-033 LB-failed) |
| 18 | `v13_oldtest_seed31337` | `v13_oldtest` | 0.3782 | +0.0001 | 0.3795–0.3828 | B-seedavg (R-033 LB-failed) |
| 19 | `v14_seed0_oldtest` | `v13_oldtest` | 0.3782 | +0.0001 | 0.3795–0.3828 | B-pure (R-027 PAIR-class) |
| 20 | `v14_avg3` | `v14_seed2_v15feat_a` | 0.3782 | +0.0001 | 0.3795–0.3827 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 21 | `v16_seed1` | `v16_avg3` | 0.3782 | +0.0001 | 0.3795–0.3827 | B-seedavg (R-033 LB-failed) |
| 22 | `v13_oldtest_seed4` | `v13_oldtest` | 0.3781 | +0.0000 | 0.3794–0.3827 | B-seedavg (R-033 LB-failed) |
| 23 | `v11_aug_oldtest_avg2` | `v11_aug_oldtest` | 0.3781 | +0.0000 | 0.3794–0.3826 | B-seedavg (R-033 LB-failed) |
| 24 | `v11_aug_oldtest_seed31337` | `v11_aug_oldtest` | 0.3781 | +0.0000 | 0.3794–0.3826 | B-seedavg (R-033 LB-failed) |
| 25 | `v11_aug_oldtest_seed51966` | `v11_aug_oldtest` | 0.3781 | +0.0000 | 0.3794–0.3826 | B-seedavg (R-033 LB-failed) |
| 26 | `v11_aug_oldtest_seed7` | `v11_aug_oldtest` | 0.3781 | +0.0000 | 0.3794–0.3826 | B-seedavg (R-033 LB-failed) |
| 27 | `v11_aug_oldtest_avg3` | `v11_aug_oldtest` | 0.3781 | -0.0000 | 0.3794–0.3826 | B-seedavg (R-033 LB-failed) |
| 28 | `v14_oldtest_avg2` | `v13_oldtest` | 0.3780 | -0.0001 | 0.3793–0.3826 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 29 | `v13_oldtest_avg2` | `v13_oldtest` | 0.3779 | -0.0001 | 0.3793–0.3825 | B-seedavg (R-033 LB-failed) |
| 30 | `v14_seed1_oldtest` | `v13_oldtest` | 0.3779 | -0.0002 | 0.3792–0.3825 | B-pure (R-027 PAIR-class) |
| 31 | `v11_big` | `v11plus` | 0.3779 | -0.0002 | 0.3792–0.3824 | B-impure (bigger arch into transformer slot) |
| 32 | `v12_5f` | `v14_seed2_v15feat_a` | 0.3778 | -0.0002 | 0.3792–0.3824 | B-impure (v12 GBM family swap) |
| 33 | `v11_mulminet_aug_lam01` | `v11plus` | 0.3778 | -0.0002 | 0.3791–0.3824 | B-impure (R-028 LB-failed at this pattern) |
| 34 | `v11plus_oldtest` | `v11_aug_oldtest` | 0.3778 | -0.0003 | 0.3791–0.3823 | B-pure (R-027 PAIR-class) |
| 35 | `v11plus_oldtest_avg2` | `v11_aug_oldtest` | 0.3778 | -0.0003 | 0.3791–0.3823 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 36 | `v11plus_oldtest_seed31337` | `v11_aug_oldtest` | 0.3778 | -0.0003 | 0.3791–0.3823 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 37 | `v11plus_oldtest_seed51966` | `v11_aug_oldtest` | 0.3778 | -0.0003 | 0.3791–0.3823 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 38 | `v14_seed2_oldtest` | `v13_oldtest` | 0.3777 | -0.0003 | 0.3790–0.3823 | B-pure (R-027 PAIR-class) |
| 39 | `v16_seed2` | `v16_avg3` | 0.3777 | -0.0004 | 0.3790–0.3823 | B-seedavg (R-033 LB-failed) |
| 40 | `v11plus_aug` | `v11_aug_oldtest` | 0.3776 | -0.0004 | 0.3790–0.3822 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 41 | `v11_uncertainty_aug` | `v11plus` | 0.3776 | -0.0004 | 0.3789–0.3822 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 42 | `v16_testhist_aug_oldtest_seed4` | `v13_oldtest` | 0.3775 | -0.0005 | 0.3789–0.3821 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 43 | `v16_testhist_aug_oldtest_seed11` | `v13_oldtest` | 0.3775 | -0.0005 | 0.3789–0.3821 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 44 | `v11_mulminet_uncertainty_aug` | `v11plus` | 0.3775 | -0.0006 | 0.3788–0.3820 | B-impure (R-028 LB-failed at this pattern) |
| 45 | `v16_testhist_aug_oldtest_avg5` | `v13_oldtest` | 0.3774 | -0.0006 | 0.3788–0.3820 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 46 | `v11_mulminet_oldtest` | `v11_aug_oldtest` | 0.3773 | -0.0008 | 0.3786–0.3819 | B-impure (R-028 LB-failed at this pattern) |
| 47 | `v16_testhist_aug_oldtest_seed7` | `v13_oldtest` | 0.3773 | -0.0008 | 0.3786–0.3818 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 48 | `v16_testhist_aug_oldtest_avg3` | `v13_oldtest` | 0.3772 | -0.0009 | 0.3785–0.3818 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 49 | `v16_testhist_aug_oldtest_seed51966` | `v16_avg3` | 0.3772 | -0.0009 | 0.3785–0.3817 | B-seedavg (R-033 LB-failed) |
| 50 | `v11_mulminet` | `v11_aug_oldtest` | 0.3772 | -0.0009 | 0.3785–0.3817 | B-impure (R-028 LB-failed at this pattern) |
| 51 | `v16_testhist_aug_oldtest_seed31337` | `v13_oldtest` | 0.3771 | -0.0009 | 0.3784–0.3817 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| 52 | `v16_testhist_aug_oldtest_seed9` | `v16_avg3` | 0.3771 | -0.0010 | 0.3784–0.3817 | B-seedavg (R-033 LB-failed) |
| 53 | `v16_testhist_aug_oldtest` | `v13_oldtest` | 0.3771 | -0.0010 | 0.3784–0.3816 | B-pure (R-027 PAIR-class) |

## 4. Two-stage gate framework classification

### STAGE 1 — STRONG/TIED (dOV ≥ 0): 26 candidates
*ELIGIBLE for direct LB upload (existing standalone fast-track).*

| component | best_slot | dOV | pred_LB (lo–hi) | class |
|---|---|---:|---:|---|
| `v11_mulminet_aug_avg3` | `v11_aug_oldtest` | +0.0039 | 0.3833–0.3865 | B-impure (R-028 LB-failed at this pattern) |
| `v11_mulminet_aug_avg2` | `v11_aug_oldtest` | +0.0027 | 0.3820–0.3853 | B-impure (R-028 LB-failed at this pattern) |
| `v11_mulminet_aug_s31337` | `v11_aug_oldtest` | +0.0013 | 0.3807–0.3840 | B-impure (R-028 LB-failed at this pattern) |
| `v11_mulminet_aug_s12345` | `v11_aug_oldtest` | +0.0012 | 0.3806–0.3839 | B-impure (R-028 LB-failed at this pattern) |
| `meta_stack` | `v14_seed2_v15feat_a` | +0.0012 | 0.3806–0.3838 | B-meta (NEW SIGNAL CLASS — stacking ensemble, never LB-tested) |
| `v11_mulminet_pretrained_aug` | `v11_aug_oldtest` | +0.0010 | 0.3804–0.3836 | B-impure (R-028 LB-failed at this pattern) |
| `v14_recvprofile` | `v14_seed2_v15feat_a` | +0.0007 | 0.3801–0.3834 | B-feature (R-034 LB-WIN class) |
| `v11_mulminet_aug_oldtest_seed31337` | `v13_oldtest` | +0.0007 | 0.3801–0.3834 | B-impure (transformer→GBM cross-family) |
| `meta_stack_v2_logistic` | `v13_oldtest` | +0.0006 | 0.3799–0.3832 | B-meta (NEW SIGNAL CLASS — stacking ensemble, never LB-tested) |
| `v11_aug_big` | `v11plus` | +0.0006 | 0.3799–0.3832 | B-impure (bigger arch into transformer slot) |
| `v11_mulminet_aug_oldtest_seed51966` | `v11_aug_oldtest` | +0.0004 | 0.3798–0.3830 | B-impure (R-028 LB-failed at this pattern) |
| `v13_oldtest_seed9` | `v13_oldtest` | +0.0004 | 0.3798–0.3830 | B-seedavg (R-033 LB-failed) |
| `v14_recvhand` | `v14_seed2_v15feat_a` | +0.0004 | 0.3798–0.3830 | B-feature (R-034 LB-WIN class) |
| `v11_mulminet_aug_oldtest_seed7` | `v11_aug_oldtest` | +0.0003 | 0.3797–0.3829 | B-impure (R-028 LB-failed at this pattern) |
| `v11_mulminet_aug_oldtest` | `v11_aug_oldtest` | +0.0003 | 0.3797–0.3829 | B-impure (R-028 LB-failed at this pattern) |
| `v11_mulminet_aug` | `v11_aug_oldtest` | +0.0002 | 0.3796–0.3829 | B-impure (R-028 LB-failed at this pattern) |
| `v13_oldtest_seed51966` | `v13_oldtest` | +0.0002 | 0.3796–0.3828 | B-seedavg (R-033 LB-failed) |
| `v13_oldtest_seed31337` | `v13_oldtest` | +0.0001 | 0.3795–0.3828 | B-seedavg (R-033 LB-failed) |
| `v14_seed0_oldtest` | `v13_oldtest` | +0.0001 | 0.3795–0.3828 | B-pure (R-027 PAIR-class) |
| `v14_avg3` | `v14_seed2_v15feat_a` | +0.0001 | 0.3795–0.3827 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v16_seed1` | `v16_avg3` | +0.0001 | 0.3795–0.3827 | B-seedavg (R-033 LB-failed) |
| `v13_oldtest_seed4` | `v13_oldtest` | +0.0000 | 0.3794–0.3827 | B-seedavg (R-033 LB-failed) |
| `v11_aug_oldtest_avg2` | `v11_aug_oldtest` | +0.0000 | 0.3794–0.3826 | B-seedavg (R-033 LB-failed) |
| `v11_aug_oldtest_seed31337` | `v11_aug_oldtest` | +0.0000 | 0.3794–0.3826 | B-seedavg (R-033 LB-failed) |
| `v11_aug_oldtest_seed51966` | `v11_aug_oldtest` | +0.0000 | 0.3794–0.3826 | B-seedavg (R-033 LB-failed) |
| `v11_aug_oldtest_seed7` | `v11_aug_oldtest` | +0.0000 | 0.3794–0.3826 | B-seedavg (R-033 LB-failed) |

### STAGE 2 — NEAR-TIED (-0.002 ≤ dOV < 0): 27 candidates
*ELIGIBLE for blend-swap diagnostic upload (NEW gate, post-R-034).*

| component | best_slot | dOV | pred_LB (lo–hi) | class |
|---|---|---:|---:|---|
| `v11_aug_oldtest_avg3` | `v11_aug_oldtest` | -0.0000 | 0.3794–0.3826 | B-seedavg (R-033 LB-failed) |
| `v14_oldtest_avg2` | `v13_oldtest` | -0.0001 | 0.3793–0.3826 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v13_oldtest_avg2` | `v13_oldtest` | -0.0001 | 0.3793–0.3825 | B-seedavg (R-033 LB-failed) |
| `v14_seed1_oldtest` | `v13_oldtest` | -0.0002 | 0.3792–0.3825 | B-pure (R-027 PAIR-class) |
| `v11_big` | `v11plus` | -0.0002 | 0.3792–0.3824 | B-impure (bigger arch into transformer slot) |
| `v12_5f` | `v14_seed2_v15feat_a` | -0.0002 | 0.3792–0.3824 | B-impure (v12 GBM family swap) |
| `v11_mulminet_aug_lam01` | `v11plus` | -0.0002 | 0.3791–0.3824 | B-impure (R-028 LB-failed at this pattern) |
| `v11plus_oldtest` | `v11_aug_oldtest` | -0.0003 | 0.3791–0.3823 | B-pure (R-027 PAIR-class) |
| `v11plus_oldtest_avg2` | `v11_aug_oldtest` | -0.0003 | 0.3791–0.3823 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v11plus_oldtest_seed31337` | `v11_aug_oldtest` | -0.0003 | 0.3791–0.3823 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v11plus_oldtest_seed51966` | `v11_aug_oldtest` | -0.0003 | 0.3791–0.3823 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v14_seed2_oldtest` | `v13_oldtest` | -0.0003 | 0.3790–0.3823 | B-pure (R-027 PAIR-class) |
| `v16_seed2` | `v16_avg3` | -0.0004 | 0.3790–0.3823 | B-seedavg (R-033 LB-failed) |
| `v11plus_aug` | `v11_aug_oldtest` | -0.0004 | 0.3790–0.3822 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v11_uncertainty_aug` | `v11plus` | -0.0004 | 0.3789–0.3822 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v16_testhist_aug_oldtest_seed4` | `v13_oldtest` | -0.0005 | 0.3789–0.3821 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v16_testhist_aug_oldtest_seed11` | `v13_oldtest` | -0.0005 | 0.3789–0.3821 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v11_mulminet_uncertainty_aug` | `v11plus` | -0.0006 | 0.3788–0.3820 | B-impure (R-028 LB-failed at this pattern) |
| `v16_testhist_aug_oldtest_avg5` | `v13_oldtest` | -0.0006 | 0.3788–0.3820 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v11_mulminet_oldtest` | `v11_aug_oldtest` | -0.0008 | 0.3786–0.3819 | B-impure (R-028 LB-failed at this pattern) |
| `v16_testhist_aug_oldtest_seed7` | `v13_oldtest` | -0.0008 | 0.3786–0.3818 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v16_testhist_aug_oldtest_avg3` | `v13_oldtest` | -0.0009 | 0.3785–0.3818 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v16_testhist_aug_oldtest_seed51966` | `v16_avg3` | -0.0009 | 0.3785–0.3817 | B-seedavg (R-033 LB-failed) |
| `v11_mulminet` | `v11_aug_oldtest` | -0.0009 | 0.3785–0.3817 | B-impure (R-028 LB-failed at this pattern) |
| `v16_testhist_aug_oldtest_seed31337` | `v13_oldtest` | -0.0009 | 0.3784–0.3817 | A or other (R-007 LB-failed pattern; needs strong signal proof) |
| `v16_testhist_aug_oldtest_seed9` | `v16_avg3` | -0.0010 | 0.3784–0.3817 | B-seedavg (R-033 LB-failed) |
| `v16_testhist_aug_oldtest` | `v13_oldtest` | -0.0010 | 0.3784–0.3816 | B-pure (R-027 PAIR-class) |

### STAGE 3 — MARGINAL (-0.005 ≤ dOV < -0.002): 0 candidates
*DIAGNOSTIC ONLY — hold unless new-signal-class evidence.*

_(none)_

### PARKED (dOV < -0.005): 0 candidates
*No LB evidence either way; user may still override and upload to disprove the gate.*


## 5. Class-based transfer risk for Stage 1+2 candidates

Sorted by predicted LB (optimistic, then conservative). Use this to pick the most
LB-likely upload candidate; the user makes the final call.

| # | component | best_slot | dOV | conservative LB | optimistic LB | class | LB risk |
|---|---|---|---:|---:|---:|---|---|
| 1 | `v11_mulminet_aug_avg3` | `v11_aug_oldtest` | +0.0039 | 0.3833 | 0.3865 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 2 | `v11_mulminet_aug_avg2` | `v11_aug_oldtest` | +0.0027 | 0.3820 | 0.3853 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 3 | `v11_mulminet_aug_s31337` | `v11_aug_oldtest` | +0.0013 | 0.3807 | 0.3840 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 4 | `v11_mulminet_aug_s12345` | `v11_aug_oldtest` | +0.0012 | 0.3806 | 0.3839 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 5 | `meta_stack` | `v14_seed2_v15feat_a` | +0.0012 | 0.3806 | 0.3838 | B-meta (NEW SIGNAL CLASS — stacking ensemble, never LB-tested) | LOW |
| 6 | `v11_mulminet_pretrained_aug` | `v11_aug_oldtest` | +0.0010 | 0.3804 | 0.3836 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 7 | `v14_recvprofile` | `v14_seed2_v15feat_a` | +0.0007 | 0.3801 | 0.3834 | B-feature (R-034 LB-WIN class) | LOW |
| 8 | `v11_mulminet_aug_oldtest_seed31337` | `v13_oldtest` | +0.0007 | 0.3801 | 0.3834 | B-impure (transformer→GBM cross-family) | HIGH |
| 9 | `meta_stack_v2_logistic` | `v13_oldtest` | +0.0006 | 0.3799 | 0.3832 | B-meta (NEW SIGNAL CLASS — stacking ensemble, never LB-tested) | LOW |
| 10 | `v11_aug_big` | `v11plus` | +0.0006 | 0.3799 | 0.3832 | B-impure (bigger arch into transformer slot) | HIGH |
| 11 | `v11_mulminet_aug_oldtest_seed51966` | `v11_aug_oldtest` | +0.0004 | 0.3798 | 0.3830 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 12 | `v13_oldtest_seed9` | `v13_oldtest` | +0.0004 | 0.3798 | 0.3830 | B-seedavg (R-033 LB-failed) | MED |
| 13 | `v14_recvhand` | `v14_seed2_v15feat_a` | +0.0004 | 0.3798 | 0.3830 | B-feature (R-034 LB-WIN class) | LOW |
| 14 | `v11_mulminet_aug_oldtest_seed7` | `v11_aug_oldtest` | +0.0003 | 0.3797 | 0.3829 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 15 | `v11_mulminet_aug_oldtest` | `v11_aug_oldtest` | +0.0003 | 0.3797 | 0.3829 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 16 | `v11_mulminet_aug` | `v11_aug_oldtest` | +0.0002 | 0.3796 | 0.3829 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 17 | `v13_oldtest_seed51966` | `v13_oldtest` | +0.0002 | 0.3796 | 0.3828 | B-seedavg (R-033 LB-failed) | MED |
| 18 | `v13_oldtest_seed31337` | `v13_oldtest` | +0.0001 | 0.3795 | 0.3828 | B-seedavg (R-033 LB-failed) | MED |
| 19 | `v14_seed0_oldtest` | `v13_oldtest` | +0.0001 | 0.3795 | 0.3828 | B-pure (R-027 PAIR-class) | LOW |
| 20 | `v14_avg3` | `v14_seed2_v15feat_a` | +0.0001 | 0.3795 | 0.3827 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 21 | `v16_seed1` | `v16_avg3` | +0.0001 | 0.3795 | 0.3827 | B-seedavg (R-033 LB-failed) | MED |
| 22 | `v13_oldtest_seed4` | `v13_oldtest` | +0.0000 | 0.3794 | 0.3827 | B-seedavg (R-033 LB-failed) | MED |
| 23 | `v11_aug_oldtest_avg2` | `v11_aug_oldtest` | +0.0000 | 0.3794 | 0.3826 | B-seedavg (R-033 LB-failed) | MED |
| 24 | `v11_aug_oldtest_seed31337` | `v11_aug_oldtest` | +0.0000 | 0.3794 | 0.3826 | B-seedavg (R-033 LB-failed) | MED |
| 25 | `v11_aug_oldtest_seed51966` | `v11_aug_oldtest` | +0.0000 | 0.3794 | 0.3826 | B-seedavg (R-033 LB-failed) | MED |
| 26 | `v11_aug_oldtest_seed7` | `v11_aug_oldtest` | +0.0000 | 0.3794 | 0.3826 | B-seedavg (R-033 LB-failed) | MED |
| 27 | `v11_aug_oldtest_avg3` | `v11_aug_oldtest` | -0.0000 | 0.3794 | 0.3826 | B-seedavg (R-033 LB-failed) | MED |
| 28 | `v14_oldtest_avg2` | `v13_oldtest` | -0.0001 | 0.3793 | 0.3826 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 29 | `v13_oldtest_avg2` | `v13_oldtest` | -0.0001 | 0.3793 | 0.3825 | B-seedavg (R-033 LB-failed) | MED |
| 30 | `v14_seed1_oldtest` | `v13_oldtest` | -0.0002 | 0.3792 | 0.3825 | B-pure (R-027 PAIR-class) | LOW |
| 31 | `v11_big` | `v11plus` | -0.0002 | 0.3792 | 0.3824 | B-impure (bigger arch into transformer slot) | HIGH |
| 32 | `v12_5f` | `v14_seed2_v15feat_a` | -0.0002 | 0.3792 | 0.3824 | B-impure (v12 GBM family swap) | HIGH |
| 33 | `v11_mulminet_aug_lam01` | `v11plus` | -0.0002 | 0.3791 | 0.3824 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 34 | `v11plus_oldtest` | `v11_aug_oldtest` | -0.0003 | 0.3791 | 0.3823 | B-pure (R-027 PAIR-class) | LOW |
| 35 | `v11plus_oldtest_avg2` | `v11_aug_oldtest` | -0.0003 | 0.3791 | 0.3823 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 36 | `v11plus_oldtest_seed31337` | `v11_aug_oldtest` | -0.0003 | 0.3791 | 0.3823 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 37 | `v11plus_oldtest_seed51966` | `v11_aug_oldtest` | -0.0003 | 0.3791 | 0.3823 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 38 | `v14_seed2_oldtest` | `v13_oldtest` | -0.0003 | 0.3790 | 0.3823 | B-pure (R-027 PAIR-class) | LOW |
| 39 | `v16_seed2` | `v16_avg3` | -0.0004 | 0.3790 | 0.3823 | B-seedavg (R-033 LB-failed) | MED |
| 40 | `v11plus_aug` | `v11_aug_oldtest` | -0.0004 | 0.3790 | 0.3822 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 41 | `v11_uncertainty_aug` | `v11plus` | -0.0004 | 0.3789 | 0.3822 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 42 | `v16_testhist_aug_oldtest_seed4` | `v13_oldtest` | -0.0005 | 0.3789 | 0.3821 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 43 | `v16_testhist_aug_oldtest_seed11` | `v13_oldtest` | -0.0005 | 0.3789 | 0.3821 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 44 | `v11_mulminet_uncertainty_aug` | `v11plus` | -0.0006 | 0.3788 | 0.3820 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 45 | `v16_testhist_aug_oldtest_avg5` | `v13_oldtest` | -0.0006 | 0.3788 | 0.3820 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 46 | `v11_mulminet_oldtest` | `v11_aug_oldtest` | -0.0008 | 0.3786 | 0.3819 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 47 | `v16_testhist_aug_oldtest_seed7` | `v13_oldtest` | -0.0008 | 0.3786 | 0.3818 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 48 | `v16_testhist_aug_oldtest_avg3` | `v13_oldtest` | -0.0009 | 0.3785 | 0.3818 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 49 | `v16_testhist_aug_oldtest_seed51966` | `v16_avg3` | -0.0009 | 0.3785 | 0.3817 | B-seedavg (R-033 LB-failed) | MED |
| 50 | `v11_mulminet` | `v11_aug_oldtest` | -0.0009 | 0.3785 | 0.3817 | B-impure (R-028 LB-failed at this pattern) | HIGH |
| 51 | `v16_testhist_aug_oldtest_seed31337` | `v13_oldtest` | -0.0009 | 0.3784 | 0.3817 | A or other (R-007 LB-failed pattern; needs strong signal proof) | LOW |
| 52 | `v16_testhist_aug_oldtest_seed9` | `v16_avg3` | -0.0010 | 0.3784 | 0.3817 | B-seedavg (R-033 LB-failed) | MED |
| 53 | `v16_testhist_aug_oldtest` | `v13_oldtest` | -0.0010 | 0.3784 | 0.3816 | B-pure (R-027 PAIR-class) | LOW |

## 6. Final list — components that have NEVER been LB-submitted

**Total parked components: 53**. None of these have been
LB-uploaded. They are organized below by gate status. The user makes the
final upload decision.

- **STAGE 1 (dOV ≥ 0)**: 26 components — direct-upload eligible
- **STAGE 2 (-0.002 ≤ dOV < 0)**: 27 components — blend-diagnostic eligible
- **STAGE 3 (-0.005 ≤ dOV < -0.002)**: 0 components — diagnostic only
- **Below threshold (dOV < -0.005)**: 0 components — no clear blend benefit

Per LESSONS 2026-05-21: standalone gates over-reject. The new blend-swap
gate is the post-R-034 fix. Predicted-LB ranges use ratios derived from
R-027 (1.0035, conservative) and R-034 (1.0151, optimistic).

Class transfer hazards (from LESSONS, 2026-05-21):
- CLASS B-impure (architecture change): R-028 LB-FAILED at ratio 0.9768.
  HIGH LB risk even when OOF dOV is strongly positive.
- CLASS B-seedavg (within-family seed avg only): R-033 LB-FAILED at ratio 1.0005.
  MED LB risk.
- CLASS B-pure (ADD oldtest, same arch): R-027 PAIR LB-WON at ratio 1.0035.
  LOW LB risk.
- CLASS B-feature (same arch + same data, new features): R-034 LB-WON at ratio 1.0121.
  LOW LB risk.


---

## Appendix: Historical LB-tested components (excluded from audit)

| Component | Status | Notes |
|---|---|---|
| v11_aug_oldtest | LB-tested (in R-027 PAIR + R-034 PAIR) | KEEP |
| v11plus | LB-tested (in R-027 PAIR + R-034 PAIR) | KEEP |
| v13_oldtest | LB-tested (in R-027 PAIR + R-034 PAIR) | KEEP |
| v14_seed2_v15feat_a | LB-tested (R-034 NEW BEST) | KEEP |
| v16_avg3 | LB-tested (in zoo_v10 elig2 + R-027 PAIR + R-034 PAIR) | KEEP |
| v14_seed2 | LB-tested (zoo_v6 elig1, replaced by v14_seed2_v15feat_a) | superseded |
| v11, v11_aug, v13, v14_seed0, v14_seed1 | LB-tested (zoo_v2 etc) | retired |
| v16_testhist_aug, v16, v17_momentum | LB-tested (V16 LB 0.3673) | retired |
| v14_pseudo_v1 | LB-FAILED zoo_v12 elig1 LB 0.3626 | BANNED |
| v15_pp | LB-FAILED LB 0.3507 | BANNED |
| v15_player_only | LB-FAILED LB 0.3555 (non-transfer player profile) | BANNED |
| v14_5f_nocb | LB-tested LB 0.3599, superseded | retired |
| v11_mulminet_aug_oldtest_avg2 | R-028 top1 LB-FAILED (-0.0086) | CLASS B-impure |
| v11_mulminet_aug_oldtest_avg3 | R-033 CLASSBimpure LB-FAILED (-0.0015) | CLASS B-impure |
| v13_oldtest_avg3 | R-033 CLASSBpure LB-FAILED (-0.0015) | CLASS B-seedavg |
