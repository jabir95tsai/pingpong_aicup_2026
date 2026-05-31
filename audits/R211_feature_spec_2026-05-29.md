# R-211 — Same-striker within-rally point-side feature: SPEC + leakage analysis

**Status:** designed, NOT yet smoked. Awaiting Jabir's call on where to test
(Kaggle GPU vs local) before any compute is spent.

## Motivation
Gap reassessment (2026-05-29) localized the +0.013 LB gap to POINT (F1_p=0.229).
R-210 proved it is a *discriminability* problem, not calibration. R-211 probe
showed a striker's own prior point-side predicts their next side (signal spread
+0.147), recoverable positionally with no player ID. This spec turns that into
an explicit feature the sequence model may under-exploit.

## Core idea
The striker alternates every shot, so the striker of target shot n also struck
shots n-2, n-4, ... (same strikeNumber parity). Those are the striker's OWN
prior shots, identifiable WITHOUT player ID. Summarize their point-side/zone
tendency from the visible context only.

## Feature definitions (per target shot n)
Let P = {pointId of prior shots j<n with strikeNumber[j] %2 == strikeNumber[n] %2}.
Sides: FH={1,4,7}, BH={3,6,9}, mid={2,5,8}, net={0}. Depth: short={1,2,3},
half={4,5,6}, long={7,8,9}.

1. `ss_prior_n`        — |P| (same-striker prior depth)
2. `ss_fh_frac`       — #FH / (#FH+#BH), fallback 0.5 if denom 0
3. `ss_last_side`     — side of most-recent prior same-striker shot (FH=+1,BH=-1,else 0)
4. `ss_short_frac`    — #short / |P|, fallback 0
5. `ss_long_frac`     — #long / |P|, fallback 0
6. `ss_net_frac`      — #net / |P|, fallback 0
7. `ss_fh_count`,`ss_bh_count` — raw counts (let GBM see support, not just ratio)

(Depth features 4-5 included because the same within-rally autocorrelation
applies to the short/half/long axis, which is ALSO badly under-predicted and is
NOT a handedness issue — potential bonus lift independent of FH/BH.)

## Leakage analysis (hard-rule audit)
- Uses ONLY shots strictly before n (visible context) → no target leakage. PASS
- No player ID, no cross-rally aggregation, no profile → not player-profile. PASS
- Grouping key = strikeNumber parity (a provided feature); within-rally order
  only, no rally_uid/global-order inference. PASS
- No SGP, no test-truth, no teammate parquet. PASS
- Fully rally-local → transfers to de-identified test_new. PASS

## Test plan (cheapest A/B)
- V14 (feature/GBM model) baseline vs baseline + the 8 ss-features.
- Fold-1 OOF only. Metric: point macro-F1 on Fold-1 val.
- **GO bar:** Δpoint-F1 ≥ +0.003 (≈ +0.0012 OV) AND no action-F1 regression
  AND no canary-class collapse.
- First check what V14 already builds — if it already groups striker history,
  expected Δ≈0 (signal already captured) and we drop R-211.
- If GO → full 5-fold, then blend-swap audit into R-034 PAIR → ARTIFACT.

## 6-field candidate report (v0.4)
- theoretical_generalization_reason: within-rally side/zone autocorrelation is a
  rally-local regularity (tactical + handedness + court geometry) present in any
  match; not tied to identities, so it generalizes.
- why_transfers_to_test_new: feature derived purely from visible context shots;
  no ID/profile dependence; test_new has the same strike-alternation structure.
- smoke_sanity_pass: PENDING (Fold-1 A/B not yet run).
- lb_probe_worthy: only if smoke Δ ≥ +0.003 point-F1.
- lb_confirm_hypothesis: LB rises ≥ +0.003 → explicit striker-history feature
  adds signal beyond the sequence model.
- lb_reject_hypothesis: LB flat/down → transformer already captured it, or the
  signal doesn't transfer to test class balance.

## UPDATE 2026-05-29 — R-211-on-V14 is REDUNDANT by construction (free NO-GO)

Checked V14 features before spending a smoke: V14 builds one-hot pointId at
lags {1,2,3,4,5,6,8,10} (train_v14.py:77). The EVEN lags (2,4,6,8,10) are the
same-striker prior shots (parity). So V14 already has the striker's last five
prior point-sides as explicit features. The R-211 probe showed the signal is
dominated by the IMMEDIATE prior same-striker shot (prev-side shift ±0.06-0.08,
~most of the +0.147 majority spread) = exactly `oh_lag2_pointId`, already a
feature. A GBM can already derive ss_fh_frac / side-consistency from these
one-hots. => Adding R-211 summary features to V14 has expected lift ~0.
**R-211-on-V14: NO-GO, no smoke needed.**

Remaining live sub-question: R-211-on-V11 (transformer). V11 sees pointId in
the interleaved A/B/A/B sequence but has NO explicit parity grouping, so its
attention may not cleanly isolate same-striker history. An explicit parity
feature/embedding COULD help V11 — but that is a Kaggle-GPU retrain (the
R-082-class cost) with uncertain upside given the moderate, partly-captured
signal. Defer unless Jabir wants to chase it.

## UPDATE 2 — R-211-on-V11 also NO-GO (capture probe, no GPU spent)

Resolved the V11 question WITHOUT retraining, via V11's existing OOF point
preds + rebuilt samples (alignment self-validated: rebuilt y_point == stored
OOF y exactly). For 33,558 FH/BH-side targets, read the target striker's own
prior context shots (indices k-2,k-4,...) for prior-side majority:

- V11 side-accuracy ALL rows: 0.324
- on rows where prior evidence is CORRECT (n=10,527): **0.422** (vs 0.324)
  => V11 ALREADY partially uses the signal (not ignoring it).
- prior reliability when present: 10,527 correct / 19,826 present = **53%**
  (barely above chance — the per-row signal is weak).
- recoverable rows (prior correct but V11 wrong) = 2,415 = 7.2% of side-targets
  — an UPPER bound ignoring the symmetric new-error cost on the 47% of
  present-priors that are WRONG.

**Verdict: R-211-on-V11 NO-GO.** V11 already extracts the weak same-striker
signal; the prior is only 53% reliable per-row, so an explicit feature offers
marginal, likely net-zero-or-negative value — not worth a GPU retrain.

## POINT BOTTLENECK — CEILING CONFIRMED
All four point mechanisms now closed by analysis, zero LB/GPU spend:
  R-082 (emb-GBM) dominated · R-210 (calibration) hurts ·
  R-211-on-V14 redundant · R-211-on-V11 weak+already-captured.
The FH/BH point gap is at its de-identification ceiling: the disambiguating
signal (receiver handedness) is only weakly recoverable within-rally (~53%),
already partially used by the models, and not cleanly extractable. The clean
+0.013 LB gap is NOT reachable via point-side mechanisms.
