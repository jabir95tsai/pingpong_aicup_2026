# R-082 Phase 3 Verdict — GBM-on-V11-embeddings: NO-GO (mechanism falsified)

**Date:** 2026-05-29
**Status:** STRATEGIC line CLOSED. Do NOT build Phase 4 LB candidate.

## What R-082 tested
Hypothesis: a gradient-boosted meta-model trained on V11's frozen 192-d
internal embeddings (last-position repr for action/point, mean-pool repr for
server) can extract signal the V11 linear heads miss — yielding either a
stronger single learner or a diverse ensemble component to blend into R-067cr.

## Pipeline executed
- **Phase 1** (audit): no fold checkpoints existed; patched trainer with
  `--save-checkpoint` + `--fold-only N`.
- **Phase 2** (Kaggle retrain): 5 per-fold kernels, ~8.6 hr/fold CPU. All 4
  fold-kernels showed "ERROR" status but that was a trivial post-training
  `mkdir`-missing crash at `submissions/` to_csv — checkpoints + OOF/test
  preds were saved intact. All 5 recovered: `models/v11_fold{0..4}.pt`.
- **Phase 3 Step 2** (extract): OOF-safe embedding extraction, 69,712 rows
  100% fold-safe coverage; 1,845 test rows averaged across 5 fold models.
  (Three extraction-harness bugs fixed: n_players dict→count; test-set
  serverGetPoint dummy column; collate key names cat_seq→cat.)
- **Phase 3 Step 3** (GBM smoke, Fold-1 OOF):

| Task   | V11 head | GBM-on-emb | Δ        |
|--------|----------|------------|----------|
| action | 0.3055   | 0.2547     | −0.0508  |
| point  | 0.2066   | 0.1416     | −0.0650  |
| server | 0.5447 (AUC) | 0.5272 | −0.0175  |

## Verdict (Goal Function v0.4 candidate report)
- **theoretical_generalization_reason:** embeddings are a lossy bottleneck of
  the same features the heads see; a GBM re-reading them has strictly less
  information than the end-to-end-trained head. No new generalizing signal.
- **why_transfers_to_test_new:** N/A — fails at the source (OOF), nothing to
  transfer.
- **smoke_sanity_pass:** **FAIL** — worse than the head on all 3 tasks by
  large margins (−0.05 action, −0.065 point, −0.018 AUC).
- **lb_probe_worthy:** **NO.** A component strictly dominated by an existing
  blend member cannot improve a weighted ensemble.
- **lb_confirm_hypothesis:** n/a (not probed)
- **lb_reject_hypothesis:** n/a (rejected pre-LB by smoke)

## Lesson for GOAL_FUNCTION
"Re-learn a meta-model on a model's own frozen embeddings" is a dominated
mechanism when the source model already has trained task heads on those same
embeddings — the head is the Bayes-optimal readout given the representation;
a GBM only loses information. Add to NORMAL-priority anti-patterns: do not
revisit GBM/MLP-on-own-embeddings for V11-class models. (Distinct from
stacking on *independent* models' outputs, which remains untested.)

## Salvage value
The 5 recovered `v11_fold{0..4}.pt` checkpoints + 100%-coverage OOF embeddings
are reusable for any FUTURE mechanism that needs per-fold V11 inference
(e.g. cross-model stacking with genuinely independent learners). Not wasted.
