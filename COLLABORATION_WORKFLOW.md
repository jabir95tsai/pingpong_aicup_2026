# Collaboration Workflow (v2)

Stable rules for the Claude ↔ Codex ↔ Jabir loop. Day-to-day project state lives in
`STATE_SUMMARY.md`; review traffic lives in `REVIEW_QUEUE.md`; pre-flight checks live
in `LESSONS_CHECKLIST.md`. Do not put state into this file.

---

## 1. Roles

- **Claude (executor)**: strategy drafting, code, long training runs, OOF/test
  artifact generation, blender execution, RESULTS/STRATEGY updates, file I/O.
- **Codex (reviewer)**: leakage and validation guards, slice analysis, calibration
  arm calls, submission gates, artifact integrity, post-mortems on failed
  experiments.
- **Jabir (decider)**: final approval for irreversible actions (LB submissions,
  destructive ops), priority calls when Claude and Codex disagree, compute budget.

## 2. Tiered Review Gates

Before launching any work, classify the action.

| Tier | Examples | Required gate |
|---|---|---|
| **T0** | read-only analysis, artifact inspection, OOF inspection, < 10 min eval | Run directly. |
| **T1** | cheap blend variants on existing OOF, post-process tweaks, < 30 min smoke | Claude self-check vs `LESSONS_CHECKLIST.md`. |
| **T2** | new trainer, > 30 min training, new feature, CV/OOF schema change | self-check **+** open `R-NNN` in `REVIEW_QUEUE.md`. |
| **T3** | LB submission | **Jabir decision only** (Codex sign-off REMOVED 2026-05-22). Pseudo-label, external data, non-train SGP, SGP-derived features/proxies, or test-target labels still require Codex sign-off as T2. |

If unclear which tier, default to the higher one.

## 2.1. T2 sub-types (Workflow v2.1)

T2 is a wide tier covering "new trainer / new feature / > 30 min training".
Sub-typing helps gate the right experiment to the right compute budget:

| T2 sub-type | Intent | Compute cap | R-### kind |
|---|---|---|---|
| `T2-component` | Build a usable new component for the zoo (e.g. v14_recvhand) | Standard 30 min – 4 h training | `preflight` or `feature` |
| `T2-diagnostic` | Test whether a paradigm helps; expected outcome is "park if it doesn't" | **1 h compute total**; if 30-min smoke says paradigm is dead, kill | `preflight` |
| `T2-exploration` | Propose a NEW model class / paradigm (e.g. causal LM, GNN) | Up to 8 h training; looser stop gates | `exploration` (new kind, see §4.5) |

Empirical justification: `R-005 meta_stack` (5h CPU, 0 LB), `R-006
server_head` (15 min, 0 LB) consumed compute under "standard T2" but produced
only "paradigm parked" results. Should have run as T2-diagnostic with 1h cap.

The R-### preflight entry must declare which sub-type it is.

## 3. Default Ambiguity Rule

If Jabir says "跑吧" / "go ahead" / equivalent on a T2 or T3 task, the default
interpretation is **"first pass the gate, then go"** — not skip the gate. Claude
may skip the gate only when Jabir explicitly says "skip Codex review, run now"
**and** the action does not violate any hard rule in `LESSONS_CHECKLIST.md`.

For T0 / T1, "go ahead" is sufficient on its own.

## 3.1. T3 Submission Approval Ritual (HARDENED — Workflow v2.1)

LB submissions are uploaded **manually by Jabir** through the competition
website. Claude has no network access to the LB and never performs the upload
itself. The approval ritual must reflect this:

- **Approval format**: ``Approved — I'll upload submissions/<filename>.csv to LB.``
  - `Approved` confirms Jabir's decision to upload (Codex `ARTIFACT_OK`
    requirement REMOVED 2026-05-22).
  - `I'll upload ... myself` (or equivalent first-person upload language) makes
    explicit that Jabir is the actor, not Claude.
- Vague approvals like "好", "可以", "go ahead", "submit it", "do it" are NOT
  sufficient for T3 — the file name must be unambiguous and the upload subject
  must be clear (Jabir, not Claude).
- After Jabir uploads and the LB returns a score, Jabir reports the score back
  to Claude. Claude then moves the corresponding `R-NNN` entry to Resolved
  with the actual LB score recorded, and updates `STATE_SUMMARY.md` and
  `RESULTS.md` accordingly.

### 3.1.1. LB upload — Codex review REMOVED 2026-05-22 (user directive)

Previously this section required Codex `ARTIFACT_OK` before any LB
upload. The user cancelled this rule on 2026-05-22 — LB upload
decisions are now Jabir's call alone.

Recent context informing the change:
- After R-042 (+0.0028 LB) confirmed our post-process is real signal,
  the 7-comp blend audit (2026-05-22) produced R-052r with predicted
  LB ~0.391. Holding these candidates for serial Codex review is
  costing more in slot-burn opportunity than in regression risk.
- Lessons from R-034 (+0.0028 win after 8 days of false-park) and
  R-042 (LB-tested same day): we're better at recognizing LB-EV
  candidates now. The 2026-05-10 episode that motivated this rule
  (R-007/R-008 LB regressions) was pre-audit framework.
- Component-design Codex review (T2 gates) is unchanged. Only the
  per-upload Codex sign-off requirement is removed.

**New policy**:
1. Component design and training (T2) still go through Codex via
   `REVIEW_QUEUE.md` as before.
2. LB upload of a built submission CSV is Jabir's decision. No
   Codex pre-approval required.
3. Best practice (not enforced): note the upload in `STATE_SUMMARY.md`
   with the LB result for the lesson log.
4. The "blend-swap diagnostic + new-signal-class" framework from the
   2026-05-21 lesson remains the recommended pre-upload heuristic.

## 4. Codex Verdict Vocabulary

Codex feedback inside `REVIEW_QUEUE.md` uses these tags so Claude can dispatch
mechanically:

- `APPROVE` — proceed exactly as proposed.
- `APPROVE_WITH_FIXES` — proceed after applying the listed fixes; no re-review needed.
- `BLOCK` — do not run. Reasons listed.
- `NEEDS_INFO` — Codex needs more context before deciding. Claude should answer in the
  same review entry.
- `ARTIFACT_OK` — for T2/T3 post-run integrity checks (mask alignment, UID order,
  no NaN, OOF/test row counts match).
- `DO_NOT_SUBMIT` — for T3 LB submission gates. Final signal that this candidate must
  not be submitted, regardless of other approvals.

Each verdict is one of the above keywords on its own line; the prose explanation
follows below.

## 4.5. R-### kind = `exploration` (Workflow v2.1)

Added to the R-### kind enum (alongside `preflight`, `submission`,
`postmortem`, `artifact`, `feature`):

- `exploration` — proposes a NEW paradigm or model class outside the
  existing v9-features × {v11, v14, v16} family. Examples: causal
  autoregressive rally LM (Path B in STRATEGY.md), graph neural net,
  pseudo-labeling V1 (R-009 covered as `preflight` since infrastructure
  is the same trainer; future pseudo-label V1b/V2 would be `exploration`).

Differences from a standard `preflight`:
- **Higher compute budget**: up to 8 h training per attempt, not the
  standard 30 min – 4 h.
- **Looser stop gates**: no per-task `+0.003` minimum; just "not
  catastrophically broken" (e.g. AUC > random + 0.05; F1_a > 0.30).
- **Required Jabir explicit T3 approval** even though no LB submission
  is implied — because of the compute commitment and paradigm risk.
- **Required pre-mortem analysis in the R-### entry**:
  - "What would success look like?" (concrete LB / OOF / diversity target)
  - "What's plan B if it fails?" (does it produce ANY useful artifact —
    e.g. diversity-only blend addition, diagnostic insight)

`exploration` does not bypass other rules — hard rules in
`LESSONS_CHECKLIST.md` still apply (no SGP leak, no pseudo-label without
approval, etc.).

## 4.6. Submission slot policy (Workflow v2.1)

Each daily LB slot must clear ONE of these bars:
- **Predicted LB lift > +0.002** with reasonable confidence (e.g. derived
  from OOF→LB ratio of validated past submissions).
- **NEW structural component** first LB validation (e.g. v14_pseudo_v1
  first upload).
- **Codex-approved structural change** (e.g. R-004 v16_avg3 substitution).

NOT eligible for slot exception (Codex P2 narrowed, reflecting zoo_v11
elig1's −0.0043 cost):
- Seed-variant substitutions (e.g. v14_seed0 → v14_seed1)
- Average vs single-seed substitutions (v16_avg3 / v14_avg3 questions
  already validated)
- Zoo-search blend-structure variations (changing transformer count,
  dropping v13, adding v11 — same paradigm)

Hold the slot otherwise. Per STRATEGY.md §6.

## 5. Hard Rules (Always Active, Cannot Be Overridden by "go ahead")

These are the inviolable safety rules. Hard-rule violations override every approval.

1. **`serverGetPoint` is never an input feature or target proxy.** Clean train
   `serverGetPoint` may be used only as the supervised label for the server head.
   Test / old-test SGP, test-history SGP, and SGP-derived proxies are never visible.
   SGP-leaking features (n_shots parity, rally length, terminal-shot parity,
   role-pooled aggregates) are forbidden unless Codex has explicitly cleared the
   specific implementation.
2. **GroupKFold by `match`** — no row-level split, no random shuffle.
3. **Fold-derived statistics must be fold-safe.** Player profiles, target encoding,
   and any leak-prone statistic must be computed only on each fold's train rows.
4. **Test rows are inputs, not labels.** Aug parquets may use *visible* action /
   point history of test rallies, but never their SGP. Pseudo-label runs that train
   on predicted test labels require explicit Jabir approval.
5. **Submission CSVs**: UTF-8 (no BOM), LF line endings, one row per unique
   `rally_uid` in the first-appearance order from `data/test_new.csv`. Schema
   correctness is Jabir's check (Codex `ARTIFACT_OK` requirement REMOVED
   2026-05-22).
6. **Each model line ships the full OOF/test artifact set** (see §6) or the report
   explicitly says which artifacts are missing and why.

## 6. Required Artifact Contract

Every named model `<tag>` should produce:

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

For derived tags (e.g. `v14_avg3`, `v16_avg3` built by `src/avg_oof.py`), the
averaging step must validate `mask`, `test_rally_uid`, `oof_y_act`, `oof_y_pt`,
`oof_y_srv`, `oof_nsn` are byte-equal across source tags before averaging.

## 7. Fixed Validation Report

Every serious line reports at minimum:

- Overall OV (base + opt where applicable)
- `F1_action`, `F1_point`, `AUC_server`
- Per-SN buckets: `SN=2`, `SN=3-4`, `SN=5-8`, `SN=9-12`, `SN>=13`
- `actionId` per-class F1
- `pointId` per-class F1
- Major point confusion, especially `0` vs `7/8/9`

## 8. Public LB vs Local Validation

Public LB is one signal among many. Use it as:

- A sanity check
- A tie-breaker between close candidates
- A signal that distribution shift may exist

A model should first have support from OOF, local benchmark, slice analysis, or a
clearly stated high-upside hypothesis. After the 2026-05-06 test reset, NEW LB
calibration must be re-established (see `STATE_SUMMARY.md` for current best).

## 9. High-ROI Heuristics

A high-ROI experiment may be expensive if it has a realistic chance of creating a
large score jump. Bottleneck relevance, score potential, risk, and runtime all
matter. Good high-ROI candidates satisfy at least one of:

- Directly attacks the current worst slice or task.
- Adds a genuinely different modeling view (different feature set, different head
  structure, different decision boundary).
- Produces OOF/test probabilities useful for blending.
- Has high upside even if the run is long.
- Answers a strategically important open question.

## 10. File Map

| File | Purpose | Update cadence |
|---|---|---|
| `COLLABORATION_WORKFLOW.md` | This file. Stable rules only. | Rare (weeks–months). |
| `STATE_SUMMARY.md` | Current LB, active jobs, components, candidates. | Every long run, every LB result, every plan change. |
| `REVIEW_QUEUE.md` | Pending Codex reviews + verbatim feedback. | Every T2/T3 gate. |
| `LESSONS_CHECKLIST.md` | Pre-flight checklist Claude self-runs. | When Codex flags a new repeated pattern. |
| `STRATEGY.md` | Active strategy notes. | Free-form, Claude-owned. |
| `TRAIN_PLAN.md` | Live training queue. | Per phase. |
| `RESULTS.md` | Append-only history of completed runs. | Per finished run. |

## 11. Standard Loop (T2 / T3)

1. Claude updates `STATE_SUMMARY.md` and writes a proposal block in `STRATEGY.md`
   (T2) or `TRAIN_PLAN.md` (T3).
2. Claude opens an `R-NNN` entry in `REVIEW_QUEUE.md` Pending section with the
   self-check filled in.
3. Codex reads `STATE_SUMMARY.md`, then the relevant referenced files, then writes
   verdict + comments in the same `R-NNN` entry.
4. Claude reads verdict, applies fixes (`APPROVE_WITH_FIXES`) or proceeds (`APPROVE`)
   or pauses (`BLOCK` / `NEEDS_INFO` / `DO_NOT_SUBMIT`).
5. After the run, Claude appends results to `RESULTS.md`, updates `STATE_SUMMARY.md`,
   and (for T3) opens a follow-up `R-NNN` for `ARTIFACT_OK` integrity check.
6. Once resolved, the `R-NNN` is moved from Pending to Resolved in `REVIEW_QUEUE.md`.
