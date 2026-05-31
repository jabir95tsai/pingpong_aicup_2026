# Prompt For Claude: Training Window Planning Template

We are continuing the AI CUP 2026 Spring table-tennis temporal prediction project.
Do not rely on stale memory. First read the local project state.

Important: the training budget is a **per-session input**, not a permanent rule.
Future training windows may be shorter or longer. Always use the current budget
Jabir gives for this specific session.

Before planning, use these session inputs:

```text
TRAINING_WINDOW_HOURS = <Jabir fills this in for this session>
GPU_AVAILABLE = <yes/no/unknown>
CPU_AVAILABLE = <yes/no/unknown>
TODAY_SUBMISSION_SLOTS = <0/1/2/3/unknown>
PRIMARY_GOAL = <e.g. improve NEW LB, build one new component, verify artifacts>
SPECIAL_INSTRUCTIONS = <optional>
```

Project path:

```text
C:\Users\jabir\Hacker_J\pingpong_aicup_2026
```

## Required First Reads

Read these files before proposing anything:

1. `COLLABORATION_WORKFLOW.md`
2. `STATE_SUMMARY.md`
3. `LESSONS_CHECKLIST.md`
4. `REVIEW_QUEUE.md`
5. `STRATEGY.md`
6. `TRAIN_PLAN.md`
7. `RESULTS.md`

Also run:

```powershell
git status --short --branch
git log --oneline --decorate -n 8
```

## Collaboration Rules

Follow the new workflow v2:

- `COLLABORATION_WORKFLOW.md` is the stable protocol.
- `STATE_SUMMARY.md` is the current state. Keep it updated.
- `REVIEW_QUEUE.md` is the Claude-to-Codex review queue.
- `LESSONS_CHECKLIST.md` is mandatory before T1+ actions.
- Do not use `CODEX_REVIEW.md` as an authority.

Tier rules:

- T0: read-only analysis or very cheap eval. You may run directly.
- T1: cheap blend variants or <30 min smoke. Run only after self-checking against `LESSONS_CHECKLIST.md`.
- T2: new trainer, >30 min training, new feature, CV/OOF schema change. You must update `STATE_SUMMARY.md`, open an `R-NNN` preflight entry in `REVIEW_QUEUE.md`, and wait for Codex review.
- T3: LB submission, pseudo-labels, external data, non-train SGP, SGP-derived features/proxies, or test-target labels. Requires Codex sign-off and Jabir explicit approval.

Default ambiguity rule:

If Jabir says "go ahead" for T2/T3, interpret it as "first pass the gate, then go", unless he explicitly says to skip Codex review. Hard rules can never be overridden.

## Hard Safety Rules

These are non-negotiable:

1. Do not use test or old-test `serverGetPoint` truth. Clean train `serverGetPoint` may only be used as the supervised server-head label.
2. Do not use SGP-derived proxies such as full-rally `n_shots`, terminal parity, rally length, or any feature that reveals who hit the decider.
3. Test-history augmentation may use visible test action/point history only. Its SGP must be absent/masked and excluded from server loss.
4. Pseudo-labeling test targets is not approved for submission training. It is design-only unless Jabir explicitly approves it as T3.
5. Teammate leak artifacts are quarantined. `AICUP_v1_LB0.4304.zip` reached 0.4304 by copying old-test SGP for 1236 overlapping rallies. Do not import its cache, submissions, SGP predictions, or old-test-SGP-trained outputs into our legal zoo.
6. Do not revive raw player profile / player-ID frequency features unchanged.
7. Do not repeat hard per-SN bucket blends unchanged.
8. Submission files must use `data/test_new.csv`, one row per unique `rally_uid` in first-appearance order, UTF-8 without BOM, LF line endings.

## Current Situation To Account For

Use `STATE_SUMMARY.md` as the source of truth. Do not hard-code old LB scores,
old best submissions, or old component status from memory.

In your response, first restate the current state you read from `STATE_SUMMARY.md`:

- Data version and test row/rally count
- Current NEW LB best
- Best unsubmitted legal candidate, if any
- Active jobs
- Usable OOF components
- Parked components / no-go directions
- Open review IDs
- Today's submission slot status, if known

## Your Task

Your first task is to propose a concrete plan for `TRAINING_WINDOW_HOURS`, not to
launch long training immediately.

Please produce a plan that:

1. Fits within `TRAINING_WINDOW_HOURS` of wall-clock time, using CPU and GPU in parallel where safe.
2. Separates T0/T1 actions you can run immediately from T2/T3 actions that require review.
3. Prioritizes legal, high-upside work that can improve NEW LB, not old-LB assumptions.
4. Includes stop gates for every expensive branch.
5. Includes expected artifacts for every proposed component.
6. Includes how each branch will be evaluated locally before any LB submission.
7. Includes which `R-NNN` review entries you will open for Codex before T2/T3 work.
8. Explicitly says what you will not do, especially pseudo-label training, old-test SGP usage, hard per-SN bucket replay, and leaked teammate artifacts.

## Preferred Plan Shape

Use this format:

```md
# Training Window Plan Proposal

## Starting Assumptions
- ...

## Budget
- Wall-clock:
- CPU:
- GPU:
- Submission slots:

## Candidate Tracks

### Track A: <name>
- Tier:
- Goal:
- Files touched:
- Runtime:
- Self-check:
- Codex review needed:
- Stop gate:
- Artifacts:
- Expected upside:
- Main risk:

### Track B: <name>
...

## Parallel Schedule
| Hour | CPU | GPU | Decision point |
|---|---|---|---|
| 0-1 | ... | ... | ... |

## Review Queue Entries To Open
- R-###: ...

## No-Go List
- ...

## Final Recommendation
- First action:
- What to ask Codex:
- What to ask Jabir:
```

## Important Instruction

Do not start any >30 min training, new trainer, new feature, or LB submission until the corresponding `REVIEW_QUEUE.md` entry has been reviewed by Codex and the verdict allows it.
