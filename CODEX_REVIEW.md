# CODEX REVIEW

## Overall Judgment

Summary: The strategy is directionally sound, but P0 has already been tested and should be closed. The next real high-upside run is `V11+`, because the current best blends depend on V11 for point prediction and pointId remains the main scoring bottleneck.

Most promising direction: Hypothesis B, `V11+ Transformer`, with a staged run plan and explicit early-stop gates.

Weakest direction: Hypothesis A as a future action. `V12_5f + V11` has already produced OOF `0.3732`, slightly below the current `V12 + V11` baseline `0.3734`, so it should not consume more time or a submission slot.

Main missing piece: The `V11+` plan needs a smaller pilot before the full 5-fold long run. We need to verify that the larger model and new point weights improve point F1 without damaging action F1 before spending the full GPU window.

## Evidence Check

Current baseline used by Claude: `submission_v12_v11_optblend.csv`, LB `0.3541608`, OOF about `0.3734`.

Is the baseline correct: Yes.

Consistent with existing OOF: Mostly yes. The strategy correctly identifies pointId and `SN=2` as bottlenecks. It also correctly marks CatBoost, `SN=2` expert, flip augmentation, V8 point grammar, and hierarchical point as already tested or risky.

Consistent with local benchmark: Partially. The local benchmark showed `V13` only had a small overall gain and did not improve `SN=2`; this supports deprioritizing generic point-grammar work.

Consistent with slice evidence: Yes. `SN=2` remains structurally weak, but the overnight `SN=2 expert` result suggests a separate expert is not enough without better signal.

Conflicts with prior experiments: The workflow file still lists CatBoost as a current high-priority direction, but the latest strategy and LB evidence say CatBoost overfits LB despite OOF gains. For this round, trust the newer strategy: do not make CatBoost the next long run.

## ROI Ranking

### P0 Keep

Experiment: `V11+ Transformer`.

Reason: V11 is the strongest complementary point model in the current ensemble. The best 4-way weighting also confirms V11 has high point weight and zero server value, so improving V11 point predictions is the cleanest route to a meaningful score jump.

Expected upside: `F1_point +0.005` to `+0.015`, which maps to roughly `+0.002` to `+0.006` OV before blend effects. If it improves point while keeping action stable, it is submission-relevant.

Risk: Medium. A wider/deeper transformer may overfit, class weights may damage majority classes, and a 5-fold GPU run is costly.

Required validation: Run a pilot first. Suggested gate: one fold or reduced epochs with the new architecture must show point F1 improvement over comparable V11 fold behavior and action F1 drop no worse than `0.02`. Full run only if the pilot is not clearly worse.

### P1 Keep If Resources Allow

Experiment: `features_v9` SN=2 joint serve-receive features.

Reason: It attacks the worst slice with a more specific feature than V8. The idea is different enough from previous point grammar because it uses joint serve context rather than marginal priors.

Expected upside: `+0.003` to `+0.007` OOF if it lifts `SN=2` point/action without hurting global rows.

Risk: Medium. Sparse lookup cells can overfit or collapse to zero-fill. It may repeat the V8 failure pattern if GBM already learns the interaction.

Required validation: Before training, print coverage and zero-fill rates for `(serve_action, serve_point, sex)` keys by fold. Stop if validation fallback/zero-fill is high or if `SN=2` OOF gain is below `+0.003`.

### P2 Delay Or Drop

Experiment: More work on `V12_5f + V11` clean blend.

Reason: P0 has already been run as a quick validation and returned OOF `0.3732`, slightly lower than baseline `0.3734`.

Why it should not be prioritized now: It does not clear the current best OOF and should not be submitted unless Public LB evidence or an error in the blend evaluation appears.

## Risk Review

Leakage risk: No obvious leakage in the strategy. `features_v9` must be reviewed carefully because fold-safe lookup tables are easy to get subtly wrong.

Player identity risk: Low. The proposed features do not depend on raw test player identity.

Fold safety risk: Medium for `features_v9`; low for `V11+` if it keeps existing `GroupKFold(match)` behavior.

OOF/test alignment risk: Medium. New `v11plus_*` outputs must follow the artifact contract exactly so existing blend tools can align them with `rally_uid`.

Submission contract risk: Low if `validate_submission_contract.py` is run before any upload.

Public LB overfitting risk: Medium. The 4-way result showed that extra blend freedom can widen the OOF-LB gap. Keep `V11+` blending simple: one alpha per task, or a constrained blend using existing tools.

## Missing Validation

Missing slice reports: `V11+` plan must require per-SN slice output, especially `SN=2`, `SN=3-4`, and `SN>=13`.

Missing class reports: Need pointId per-class F1 for `0`, `1`, `3`, `8`, `9`; and actionId per-class F1 for rare classes `8`, `9`, `14`.

Missing benchmark: No local benchmark requirement is specified for `V11+`. Add it if runtime allows after OOF, but do not block the first OOF decision on local benchmark.

Missing baseline comparison: `V11+` must compare against both V11 standalone and `V12 + V11` blend, not only against global best ensemble.

Missing artifact: `train_v11_transformer.py` may not currently expose `--d_model`, `--n_heads`, or `--n_layers`. Add these args before training and record them in `RESULTS.md`.

## Recommended Final Plan

First: Implement `V11+` trainer arguments and run a pilot.

Why: This is the highest-upside path and avoids spending the full GPU window blindly.

Need: Add CLI args for model size, point class weights, tag, folds, epochs if missing.

Success condition: Pilot does not regress point F1 and does not destabilize action F1.

Second: If pilot passes, run full `v11plus` 5-fold training.

Why: A better V11 point model is directly useful in the current blend structure.

Need: Full artifact contract under `oof_predictions/v11plus_*` and `submissions/submission_v11plus.csv`.

Success condition: `v11plus` standalone point F1 beats V11, and `V12/V12_5f/V12cb + v11plus` constrained blend beats the current best OOF without excessive free parameters.

Third: Develop `features_v9` in parallel only if it does not compete for GPU.

Why: It is a plausible next source of `SN=2` signal, but less proven than `V11+`.

Need: Coverage report before training. Smoke train only after fold-safe review.

Stop conditions:

- Stop `features_v9` if validation fallback rate is high.
- Stop `V11+` architecture escalation if the pilot has `F1_point <= V11` and action F1 drops materially.
- Do not submit `V12_5f_v11_blend` unless new evidence changes its standing.

## Required Artifact Checks

Expected OOF files:

- `oof_predictions/v11plus_oof_act.npy`
- `oof_predictions/v11plus_oof_pt.npy`
- `oof_predictions/v11plus_oof_srv.npy`
- `oof_predictions/v11plus_oof_mask.npy`
- `oof_predictions/v11plus_oof_y_act.npy`
- `oof_predictions/v11plus_oof_y_pt.npy`
- `oof_predictions/v11plus_oof_y_srv.npy`
- `oof_predictions/v11plus_oof_nsn.npy`

Expected test prediction files:

- `oof_predictions/v11plus_test_act.npy`
- `oof_predictions/v11plus_test_pt.npy`
- `oof_predictions/v11plus_test_srv.npy`
- `oof_predictions/v11plus_test_rally_uid.npy`

Expected submission files:

- `submissions/submission_v11plus.csv`
- one or more constrained blend submissions using `v11plus` if OOF supports it

Expected report files:

- `TRAIN_PLAN.md`
- `RESULTS.md`
- per-class and per-SN slice report embedded in `RESULTS.md` or written separately

## Submission Decision Notes

Likely first submission: Not `V12_5f_v11_blend`; it underperformed baseline OOF. Wait for `V11+` or a constrained blend with `V11+`.

Conservative backup: Existing `submission_v12_v11_optblend.csv` remains the validated baseline.

High-risk candidate: A constrained blend using `V11+` if it improves point F1 and preserves action F1.

Reasoning: The next submission should be based on a new model signal, not a blend that already failed to beat the current OOF baseline.

## Final Recommendation

Proceed as written: No.

Proceed with modifications: Yes.

Do not run yet: Do not start the full `V11+` 5-fold run until the trainer args and pilot gate are defined in `TRAIN_PLAN.md`.

Required changes before training:

- Close Hypothesis A as completed and not submission-worthy.
- Add a pilot stage before full `V11+`.
- Add explicit `V11+` CLI arg changes to `TRAIN_PLAN.md`.
- Add a coverage audit for `features_v9` before any V14 training.
- Keep blending constrained: one alpha per task or a small fixed model set, no large grid with many degrees of freedom.
