# R-067cr Gap Reassessment (auto-generated)

```
==============================================================================
 R-067cr GAP REASSESSMENT — per-class macro-F1 decomposition (OOF)
==============================================================================
 Blend OV=0.3792  F1_a=0.4130  F1_p=0.2285  AUC=0.6129
 LB anchor (R-067cr) = 0.3870095   target = 0.4000   gap = +0.0130

------------------------------------------------------------------------------
 ACTION (weight 0.4, 15 eval classes)
------------------------------------------------------------------------------
 cls name                       F1   prec    rec   supp   pred  topConfusions
   0 none                    0.168  0.240  0.130   2052   1109  block:666, counter-loop:329, loop:205
   8 arc/hook                0.190  0.178  0.204    372    428  chop-push:89, loop:73, tap:50
   5 fast-drive              0.250  0.296  0.217   4192   3074  block:776, push-block:769, loop:658
   7 flick                   0.285  0.311  0.264   1413   1200  chop-push:455, loop:201, short-stop/short-chop:201
   3 smash                   0.369  0.394  0.346   2129   1869  loop:619, chop-push:211, chop:111
  14 lob                     0.385  0.357  0.418    613    717  block:113, counter-loop:70, loop:53
   9 tap                     0.395  0.359  0.440    794    972  chop-push:106, loop:74, chop:64
   6 push-block              0.417  0.440  0.396   6635   5980  block:1259, loop:911, counter-loop:679
  11 short-stop/short-chop   0.429  0.409  0.451   3522   3886  chop-push:956, twist:410, loop:350
   4 twist                   0.439  0.424  0.456   2638   2840  chop-push:504, short-stop/short-chop:476, loop:296
   2 counter-loop            0.468  0.482  0.456   6339   5990  block:1566, push-block:698, loop:347
  13 block                   0.496  0.454  0.548   7848   9462  counter-loop:1108, push-block:856, fast-drive:482
  10 chop-push               0.573  0.573  0.572  11208  11198  loop:1643, short-stop/short-chop:1257, twist:673
   1 loop                    0.659  0.639  0.681  15435  16445  chop-push:1461, push-block:673, fast-drive:507
  12 chop                    0.671  0.669  0.672   4522   4542  chop-push:610, block:297, loop:192

  HEADROOM (lift each weak class -> F1=0.30, OV impact = 0.4 * dF1/15):
    cls  0 none                   F1 0.168->0.30  +0.0088 F1  => +0.0035 OV  (supp=2052)
    cls  8 arc/hook               F1 0.190->0.30  +0.0073 F1  => +0.0029 OV  (supp=372)
    cls  5 fast-drive             F1 0.250->0.30  +0.0033 F1  => +0.0013 OV  (supp=4192)
    cls  7 flick                  F1 0.285->0.30  +0.0010 F1  => +0.0004 OV  (supp=1413)

------------------------------------------------------------------------------
 POINT  (weight 0.4, 10 eval classes)
------------------------------------------------------------------------------
 cls name                       F1   prec    rec   supp   pred  topConfusions
   3 BH-short                0.033  0.036  0.030    203    166  mid-short:36, mid-half:34, FH-short:28
   1 FH-short                0.119  0.073  0.321    582   2566  mid-short:109, BH-long:77, FH-half:65
   4 FH-half                 0.172  0.139  0.224   2995   4836  FH-long:421, BH-long:411, BH-half:363
   5 mid-half                0.176  0.240  0.139   6585   3811  BH-long:1127, miss/net:851, mid-long:829
   7 FH-long                 0.230  0.207  0.259   9122  11428  BH-long:2272, miss/net:1599, mid-long:1360
   8 mid-long                0.230  0.259  0.207  12386   9926  BH-long:3212, miss/net:2212, FH-long:2122
   2 mid-short               0.247  0.208  0.305   1920   2822  FH-short:360, mid-half:275, BH-long:207
   9 BH-long                 0.334  0.326  0.343  16073  16908  FH-long:2864, mid-long:2397, miss/net:2166
   6 BH-half                 0.335  0.372  0.304   4583   3746  BH-long:818, FH-half:717, FH-long:491
   0 miss/net                0.410  0.436  0.386  15263  13503  BH-long:3254, FH-long:2281, mid-long:2151

  HEADROOM (lift each weak class -> F1=0.30, OV impact = 0.4 * dF1/10):
    cls  3 BH-short               F1 0.033->0.30  +0.0267 F1  => +0.0107 OV  (supp=203)
    cls  1 FH-short               F1 0.119->0.30  +0.0181 F1  => +0.0072 OV  (supp=582)
    cls  4 FH-half                F1 0.172->0.30  +0.0128 F1  => +0.0051 OV  (supp=2995)
    cls  5 mid-half               F1 0.176->0.30  +0.0124 F1  => +0.0050 OV  (supp=6585)
    cls  7 FH-long                F1 0.230->0.30  +0.0070 F1  => +0.0028 OV  (supp=9122)
    cls  8 mid-long               F1 0.230->0.30  +0.0070 F1  => +0.0028 OV  (supp=12386)
    cls  2 mid-short              F1 0.247->0.30  +0.0053 F1  => +0.0021 OV  (supp=1920)

==============================================================================
 CUMULATIVE OV HEADROOM (all sub-0.30, supp>=30 classes -> 0.30)
==============================================================================
   action: +0.0082 OV    point: +0.0357 OV    total: +0.0439 OV
   (gap to 0.4000 is +0.0130; realistic capture is a fraction of this ceiling)
```

## Strategic conclusions (2026-05-29)

**The gap lives almost entirely in POINT, not action or server.**
- F1_a = 0.4130 (healthy), **F1_p = 0.2285 (the bottleneck)**, AUC = 0.6129.
- Both action & point are weighted 0.4, so point has ~2x the leverage.
- Cumulative OV headroom: action +0.0082, **point +0.0357**. Gap to 0.4000 is
  only +0.0130, so capturing even a third of point headroom clears it.

**Within point, two failure modes dominate:**
1. **FH/BH axis is broken (handedness).** FH-short (cls 1) is over-predicted
   4.4x (2566 preds for 582 support, precision 0.073); BH-short (cls 3) is
   nearly dead (F1=0.033, 166 preds for 203 support). This matches the known
   structural fact: pointId's FH/BH axis is *receiver-relative* (depends on the
   receiver's dominant hand), which is not recoverable from de-identified
   player IDs. The model defaults to spamming FH-short and never commits to
   BH-short. **Top single levers:** BH-short +0.0107 OV, FH-short +0.0072 OV.
2. **Argmax calibration bias.** mid-half (supp 6585, recall 0.139) and mid-long
   (supp 12386, recall 0.207) are badly *under*-predicted while FH-short is
   over-predicted. The blend uses RAW argmax — no per-class threshold/prior
   calibration on point. (V14's own threshold-opt gave +0.0105 on point.)
   BH-long/FH-long/miss-net act as universal attractor sinks.

**Action is near its stochastic ceiling** (F1=0.413). Weakest: none (0.168),
arc/hook (0.190), fast-drive (0.250) — all bleed into block/push-block/loop.
Low, hard-won headroom (+0.0082 total). Not where to spend effort.

**Recommended next directions, ranked by leverage x safety x novelty:**
1. **Point per-class threshold/prior calibration on the R-034 blend**
   (HIGH leverage, LOW risk, in-distribution). The blend currently argmaxes
   raw probs; correcting the FH-short over-prediction + mid under-prediction
   is free headroom never applied to the blend's point head. Est +0.003..+0.008
   OV. Must be calibrated on OOF and checked for collapse, not fit to LB.
2. **Handedness-latent point model** (HIGH leverage, HIGH novelty, MEDIUM risk).
   Infer receiver handedness from *in-rally shot signatures* (spin/position
   patterns), NOT player ID — to unlock the FH/BH short/half classes. Must
   avoid the hard-ruled-out player-profile family. Genuinely new generalizing
   mechanism if the latent is rally-derived.
3. **Point-specialist learner** with loss focused on short/half zones — lower
   novelty, overlaps R-203's focal direction (which gave only +0.0029).

## R-210 point calibration — NO-GO (OOF smoke, 2026-05-29)

Prior-shift correction P'(c|x) ∝ P(c|x)·(π_true(c)/π_pred(c))^β swept β∈[0,1.2].
**β=0 (no correction) is optimal; F1_p decreases monotonically with β** (−0.0049
at β=0.3, −0.0392 at β=1.0). Calibration/threshold tricks do NOT unlock point.

**Interpretation (important):** the FH-short 4.4× over-prediction is a genuine
*discriminability* failure, not a prior/calibration mismatch. The blend assigns
high-confidence FH-short to instances that are truly mid-short/BH-long, so
re-weighting toward the true prior only relocates errors. **Conclusion: the
point gap requires NEW discriminative signal (handedness), not recalibration.**
This is a generalization-positive negative result — falsified at OOF, zero LB
spend. Pivot to the handedness-latent mechanism (R-211).

## R-211 within-rally side-consistency probe — signal VALIDATED (moderate)

Tested whether a striker's OWN prior point-side (shots n-2,n-4,... — same
strikeNumber parity, recoverable positionally with NO player ID, hard-rule
clean) predicts their next point-side. Train.csv, 41,718 FH/BH-side targets.

- Base rate P(FH)=0.412
- prior-majority FH → P(next=FH)=0.474 (+0.062)
- prior-majority BH → P(next=FH)=0.327 (−0.085)
- **SIGNAL SPREAD = +0.147** (just under the 0.15 "strong" bar)

**Verdict:** real, legal, transferable side-consistency signal exists — but
moderate, and only ~42% of targets have a non-tie prior (most rallies short →
striker has 0–1 own prior shots). NOTE: this is within-rally side-consistency
(handedness OR court-geometry OR tactical autocorrelation — indistinguishable
here, but all legal). OPEN QUESTION: the V11/V14 models already see prior
pointId in-sequence; the marginal value of an EXPLICIT same-striker-grouped
side feature over what the transformer already extracts is unproven and needs
a controlled training smoke to settle. Estimated upside if it helps: +0.002..
+0.005 OV on point; could also be ~0 if already captured.

## R-202 long-rally SN>=3 specialist — NO-GO (dropped 2026-05-31)

Ran fold-1 on Kaggle GPU; cancelled at ep75/80 (never completed 5 folds, so no
usable full OOF). Decisive negative signal in the fold-1 log: the SN>=3 filter
shrinks the training set, causing HARD overfit after ~ep20 — train loss falls
0.76->0.29 while val OV degrades 0.286 (ep20) -> 0.265 (ep75). The specialist's
aggregate val OV (~0.286 peak) is well below R-067cr components, and the
overfit makes the long-rally specialization fragile. User decision: DROP.
Lesson: data-subset specialists (SN-filtered) overfit fast at full epoch budget;
if revisited, need <=25 epochs + stronger regularization, but priority is now
the higher-novelty backlog (R-200 MTL aux heads, R-201 encoder-decoder hybrid).

## R-211 recv_side_est — NO-GO (dropped 2026-05-31)
Built features_v9_recvside.py + V14 branch; Kaggle full-5-fold kernel ERRORed
on a trivial argparse `choices` omission (v9_recvside not whitelisted; runtime
patch added the branch but not the choice). Not re-run: expected upside tiny
(prior recvhand handId-mode = +0.0005 OV) and the team is moving to a final
AutoGluon ensemble, so a marginal decorrelated feature isn't worth the cycle.
Code retained (features_v9_recvside.py, train_v14.py branch) if revisited; the
argparse choices list still needs v9_recvside added to be runnable.

## AutoGluon meta-stack — LB_FAIL 0.3152 (2026-06-02)
Clean AutoGluon stack of the 6 components (f1_macro/roc_auc metrics correct,
v22 server @0.30). AutoGluon internal validation OV = 0.4149 — but real LB
= **0.3152 (267/423)**, a −0.10 collapse. CAUSE: AutoGluon's random (non-
match-grouped) holdout let the stacker memorize within-match correlation in
the OOF meta-features that does not exist train→test → catastrophic overfit;
worse than any single component (~0.38). point class-9 over-predicted 54%
(sanity warning) confirmed. LESSON: meta-stacking OOF predictions REQUIRES
match-grouped CV (pass groups to AutoGluon); random holdout is wildly
optimistic (+0.10) here. FINAL DECISION: keep R-067cr (0.3870095, clean,
LB-verified) as the final submission.
