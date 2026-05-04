# Collaboration Workflow

## Goal

This workflow keeps Claude and Codex coordinated during AI CUP experiments.

Claude owns strategy generation, model design, long training runs, and nightly reports.
Codex owns strategy review, benchmark checks, slice analysis, leakage/validation guards,
artifact integration, and submission decisions.

The point is not to keep both agents busy. The point is to spend training time on
experiments that can plausibly move the score, then use consistent evidence to decide
what to keep.

## Public LB vs Local Validation

Public LB is useful external feedback, but it should not replace local validation.

Use Public LB as:

- A sanity check
- A tie-breaker between close candidates
- A signal that distribution shift may exist

Do not use Public LB as the only reason to keep a model. A model should first have
support from OOF, local benchmark, slice analysis, or a clearly stated high-upside
hypothesis.

## High ROI Strategy

High ROI does not only mean cheap or fast.

A high ROI experiment may be expensive if it has a realistic chance of creating a
large score jump. Score potential, bottleneck relevance, risk, and runtime all matter.

Good high ROI ideas usually satisfy at least one of these:

- They directly attack the current worst slice or task.
- They add a genuinely different modeling view.
- They can produce OOF/test probabilities useful for blending.
- They have high upside even if the run is long.
- They answer a strategically important question.

## Standard Loop

1. Claude writes `STRATEGY.md`.
2. Codex reviews it and writes `CODEX_REVIEW.md`.
3. Claude finalizes `TRAIN_PLAN.md`.
4. Claude runs training and writes `RESULTS.md`.
5. Codex verifies artifacts, runs slice analysis, and writes `SUBMISSION_DECISION.md`.

## Required Artifact Contract

Each model line should produce:

- `src/train_<tag>.py` or a clear command using an existing trainer
- `oof_predictions/<tag>_oof_act.npy`
- `oof_predictions/<tag>_oof_pt.npy`
- `oof_predictions/<tag>_oof_srv.npy`
- `oof_predictions/<tag>_oof_mask.npy`
- `oof_predictions/<tag>_oof_y_act.npy`
- `oof_predictions/<tag>_oof_y_pt.npy`
- `oof_predictions/<tag>_oof_y_srv.npy`
- `oof_predictions/<tag>_oof_nsn.npy`
- `oof_predictions/<tag>_test_act.npy`
- `oof_predictions/<tag>_test_pt.npy`
- `oof_predictions/<tag>_test_srv.npy`
- `oof_predictions/<tag>_test_rally_uid.npy`
- `submissions/submission_<tag>.csv`

If a line cannot produce all of these, the report should explicitly say why.

## Fixed Validation Report

Every serious line should report:

- Overall OV
- `F1_action`
- `F1_point`
- `AUC_server`
- `SN=2`
- `SN=3-4`
- `SN=5-8`
- `SN=9-12`
- `SN>=13`
- `actionId` per-class F1
- `pointId` per-class F1
- Major point confusion, especially `0` vs `7/8/9`

## Current AI CUP Priorities

As of 2026-04-30, the strongest known line is:

- `submission_v12_v11_optblend.csv`
- LB: `0.3541608`
- OOF: about `0.3734`

Current bottleneck:

- `SN=2` receive slice
- `pointId` remains the main task bottleneck

Current high-priority directions:

- `SN=2` specialist or receive-specific calibration
- CatBoost diversity on the V12-style pipeline
- Stronger, well-validated blends using full OOF artifacts

Lower priority unless new evidence appears:

- Hierarchical point head already tested poorly
- Flip augmentation was nearly flat
- Generic point grammar features were marginal

## Safety Rules

- Do not use `serverGetPoint` as an input feature or target proxy.
- Do not depend on test player identity overlap.
- Use `GroupKFold` by match.
- Keep fold-derived statistics fold-safe.
- Keep submission files traceable to their training command and artifacts.
