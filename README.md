# pingpong_aicup_2026

AI CUP 2026 Spring table-tennis temporal prediction project.

## Active Entry Points

- `STATE_SUMMARY.md` — current project state; read first.
- `COLLABORATION_WORKFLOW.md` — Claude / Codex / Jabir workflow rules.
- `REVIEW_QUEUE.md` — T2/T3 review requests for Codex.
- `LESSONS_CHECKLIST.md` — pre-flight leakage and validation checklist.
- `CLAUDE_TRAINING_WINDOW_PROMPT_TEMPLATE.md` — reusable prompt for new training windows.
- `STRATEGY.md`, `TRAIN_PLAN.md`, `RESULTS.md` — strategy, live plan, and run history.

## Current Data

- Active test file: `data/test_new.csv`
- Old `data/test.csv` may be used only through approved `--include-old-test`
  training paths. Never copy old-test `serverGetPoint` into `test_new`
  predictions or submissions.

## EDA

- Current EDA report: `eda_output/EDA_CURRENT.md`
- Regeneration script: `src/run_current_eda.py`
