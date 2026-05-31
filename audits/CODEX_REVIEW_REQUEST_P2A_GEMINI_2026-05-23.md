# Codex Review Request — R-073 P2A → Gemini external-data pipeline

**Type**: pre-R-073 design + pilot toolkit review (no LB upload, no training).
**Author**: Claude (Opus 4.7, 1M context).
**Reviewer requested**: Codex.
**Verdict expected**: `APPROVE` / `APPROVE_WITH_FIXES` / `BLOCK` / `NEEDS_INFO`,
with specific concrete fixes if not full approve.
**Authorization to act on this**: Jabir confirms "let's develop by ourselves"
(2026-05-23, this session). NOTHING in this work has been trained, uploaded,
or merged into the existing pipeline yet. All artifacts live in a new
`p2a_pilot` namespace and are inert.

---

## 1. Context (5 bullets)

1. **Current LB best**: R-042 = 0.3866550 (R-034 PAIR + `apply_rule_override`).
   Top-10 floor 0.4445+. We are saturated on the current paradigm per the
   `AUDIT_2026-05-23.md` you APPROVED_WITH_FIXES earlier today.
2. **The audit's top-3 levers** are (1) R-066 AutoGluon clean component
   (highest-EV legal preflight, unverified), (2) R-069 Path B causal LM smoke,
   (3) R-068 action-point combos. Jabir now wants to explore a 4th lever:
   the P2A dataset.
3. **P2A audit history**: `audits/P2A_DATASET_AUDIT_2026-05-19.md` previously
   marked the 208 GB videos as "DO_NOT_USE" because we have no video-CV
   infrastructure. The new mechanism is to use **Gemini's native video API
   as the missing video-CV infrastructure**, which structurally bypasses
   that constraint.
4. **Legality position**: P2A is 2019 ITTF event clips (Ma Long, Fan
   Zhendong, etc.) — zero overlap with AICUP de-identified test_new players.
   AICUP rules permit `自製資料或開源資源`. The `嚴禁反向比對真實比賽影片`
   rule forbids back-matching test videos; it does NOT forbid using
   unrelated external video to generate training data. The LESSONS
   external-data rule requires T3 review (this document).
5. **Failure-mode pre-check**: this is NOT R-009-class pseudo-labelling
   (which LB-failed at −0.0068 due to LB-best-teacher monoculture). The
   teacher here is Gemini — structurally external to our zoo. It is also
   NOT R-021-class encoder pretraining (which LB-PARKED on ShuttleSet22)
   — we generate tabular AICUP-schema rows for normal supervised training,
   no weight transfer.

---

## 2. What was built today (file inventory)

### New module — `src/p2a_pilot/` (550 LOC, 50 unit tests passing)

| File | Purpose |
|---|---|
| `__init__.py` | namespace doc |
| `load_p2a.py` | parse P2A label JSON (both `v1.json` `{fps, gts}` dict shape AND `v1_renamed.json` bare-list shape); pick pilot videos; build per-video stroke-anchor table; pre-segment rallies by time-gap heuristic; **match-aware picker** that takes one chunk per match for diversity |
| `group_matches.py` | parse `proj.json`; extract `(match_name, chunk_idx, total_chunks)` via regex `^(.+?)-?(\d+)-of-(\d+)\.mp4$`; group all 2489 chunks into 290 distinct matches; detect near-duplicate match names with `LIKELY_SAME_MATCH` (separator-only diff, e.g. `4-2` vs `42`) vs `MANUAL_REVIEW` verdict |
| `build_prompt.py` | generate Gemini prompt text per video; anchors on exact stroke count + per-stroke `start/end` timestamps + known `hand`/`is_serve`/`action_type` from P2A JSON; asks Gemini ONLY for the 3 missing fields (landing_zone, player_position, server_won_rally) plus rally segmentation verification |
| `parse_response.py` | robust JSON extraction (strips markdown fences, prose preamble); strict schema validation (stroke count must match, rally coverage exact, all enums in valid sets) |
| `accuracy.py` | per-field accuracy vs hand-truth; per-confidence breakdown (high/medium/low); built-in `PILOT_GATES` constants with documented thresholds |
| `cli.py` | 3 subcommands: `group-matches`, `prepare`, `parse`, `report` |

### New tests — `tests/test_p2a_pilot.py` (50 tests, all pass)

Covered: vocab mapping (HAND/IS_SERVE/ACTION dicts); anchor table construction;
stroke malformed-row handling; rally pre-segmentation; prompt invariants
(stroke count, indices, rule text); JSON parser happy path + 7 failure
paths (markdown fences, prose preamble, wrong video_id, wrong stroke count,
invalid enum, rally coverage misses, out-of-range outcome); chunk filename
regex (with/without dash separator, with score-hyphen in match name,
with underscore in match name, with match-name-ending-in-digit); match
grouping (basic, missing chunks, sort, unmatched reason, unknown split);
lookup helpers (per-split shape AND combined multi-split shape);
near-duplicate detection (LIKELY_SAME_MATCH vs MANUAL_REVIEW verdicts);
match-aware picker (one per match invariant, smallest_index strategy,
too-few-matches error, invalid strategy error).

### New CLI artifacts

| File | Bytes/rows | Purpose |
|---|---|---|
| `data/p2a_match_groups.json` | 290 matches across 2 splits, 0 unmatched, 118 incomplete, 2 near-dups | Phase-1 input — primary match-grouping reference |
| `runs/p2a_pilot/pilot_manifest.json` | 10 videos, all from distinct matches | which videos to label in pilot |
| `runs/p2a_pilot/prompts/*.prompt.txt` | 10 files, ~5 KB each | exact text to paste into Gemini after uploading the matching .mp4 |
| `runs/p2a_pilot/anchors/*.anchors.json` | 10 files | stroke timestamps + known fields per video |
| `runs/p2a_pilot/responses/` | empty | Jabir will fill with Gemini outputs |
| `runs/p2a_pilot/parsed/` | empty | filled by `cli.py parse` |

### New protocol docs — `scripts/p2a_pilot/`

| File | Purpose |
|---|---|
| `PILOT_PROTOCOL.md` | end-to-end workflow (prepare → run on web Gemini or Antigravity → parse → hand-truth → report) |
| `HAND_TRUTH_SCHEMA.md` | what the human labels manually for accuracy validation |
| `PROMPT_SAMPLE.txt` | rendered example prompt (synthetic 6-stroke video) |
| `hand_truth/strokes.csv`, `hand_truth/rallies.csv` | empty templates with headers |

### Files NOT touched (deliberate)

- No `src/train_v14.py` edits. The eventual `--include-external` flag is
  Phase-1 (post-pilot), requires its own R-### preflight.
- No `data/train.csv` modification.
- No new `oof_predictions/` files.
- No `submissions/` files.
- No model checkpoints.
- No LB upload.

---

## 3. The Gemini-labelling design (the part where opinion matters most)

### Field scope per Jabir's instruction

| Field | Source | Why this choice |
|---|---|---|
| `handId` | P2A JSON `label_names[0]` (正手/反手) | already in P2A; no Gemini call needed |
| `strikeId` | P2A JSON `label_names[1]` (是/否 for serve) | already in P2A; maps to AICUP `{1, 2, 4}` |
| `actionId` | P2A JSON `label_names[2]` via vocab table (10/15 map cleanly) | already in P2A |
| `pointId` | **Gemini** — asks for 9-grid `landing_zone` (FH/mid/BH × short/half/long) plus `off_grid` for unobservable | hard but high-value |
| `positionId` | **Gemini** — `player_position` (left/center/right/unknown) | easy spatial reasoning |
| `serverGetPoint` | **Gemini** — `server_won_rally` per rally (0/1/-1) | easy if rally end is visible |
| `strengthId`, `spinId` | NOT asked of Gemini; left as 0 for V1 | spin from broadcast video is unreliable even for humans; strength is subjective |

### Critical prompt design choices

1. **Anchoring**: every prompt embeds the exact stroke count + per-stroke
   timestamps + already-known fields. Gemini is told "output EXACTLY N
   stroke objects in this order; do NOT invent or skip strokes." Parser
   enforces this.
2. **Rally pre-segmentation**: time-gap heuristic (default 4.0s) provides
   a sensible default; Gemini asked to verify/adjust.
3. **Receiver-handedness assumption**: prompt assumes right-handed
   receiver. We accept ~10% systematic noise for left-handed receivers
   (Fan Zhendong, Harimoto, etc.). Documented in `HAND_TRUTH_SCHEMA.md`.
4. **Confidence levels per field**: high/medium/low. Downstream filtering
   can keep only `high` rows.
5. **JSON-only output rule** + parser robustness against markdown fences
   and prose preamble (because web Gemini in practice doesn't always
   obey).
6. **Pilot gates** (`src/p2a_pilot/accuracy.py:PILOT_GATES`):

   | Field | Overall acc ≥ | High-conf acc ≥ |
   |---|---:|---:|
   | landing_zone (pointId proxy) | 0.50 | 0.65 |
   | player_position (positionId) | 0.70 | 0.85 |
   | server_won_rally (SGP) | 0.75 | 0.90 |

### Match-aware grouping

P2A videos are 5-15 min chunks of full ITTF matches. The original audit
missed that. Now built:
- 290 distinct matches across v1+v2 (was reported as "2489 unrelated videos")
- 118 matches with at least one missing chunk (rally history will have gaps)
- 2 near-duplicate pairs with distinct verdicts:
  - `LIKELY_SAME_MATCH` — `樊振东4-2张本智和` vs `樊振东42张本智和` (typo, same match)
  - `MANUAL_REVIEW` — `第二盘郑培锋vs林高远` vs `第四盘郑培锋vs赛林威`
    (different opponents in same team round — DO NOT merge)
- Match-aware pilot picker (`pick_pilot_videos_match_aware`) ensures 1 chunk
  per match for max diversity

---

## 4. The 8 questions where Codex's view matters

### Q1 — Workflow compliance

Per the AUDIT §6.1 "Stop immediately" list, you blocked R-058/R-059 re-runs,
R-064 5-fold, R-065 pseudo, etc. Should this P2A-Gemini work also be
HARD-blocked at the design stage until a formal R-073 preflight is opened
in `REVIEW_QUEUE.md`? Or is the current "no training, no upload, inert
toolkit only" scope acceptable as exploratory work that can be reviewed
out-of-band?

### Q2 — Classification under the CLASS framework

The eventual training run would be on the AICUP `train.csv` + a new
`data/p2a_gemini.parquet` of pseudo-labeled rows. Which CLASS does this
fit?

- It is NOT B-impure (no architecture change).
- It is NOT B-pure ADD (oldtest-like — that was AICUP-internal real data).
- It is NOT B-seedavg, NOT B-meta, NOT B-player-style.
- It is most similar to **R-009 V1 pseudo-label class** structurally, but
  with a structurally-different teacher (external foundation model on
  external data, not LB-best zoo on test_new). Does this deserve a NEW
  class label (e.g. B-external-pseudo)? Or does the R-009 ban still apply
  on the basis of "noisy labels in training set = bias amplification"
  even when source is external?

### Q3 — pointId label-noise tolerance

Gemini's `landing_zone` accuracy on broadcast video is genuinely uncertain.
If the pilot lands at e.g. 55% overall / 70% high-conf, ~64,914 strokes ×
55% = ~36,000 "correctly labeled" + ~29,000 noisy pointId rows would
augment our 84,707-row train. Does this risk hurting OOF on the rare
pointId classes (BH_short cls3 is already F1=0 — could go negative)?
Suggested mitigations:
- (a) drop all medium/low confidence rows → ~15k high-conf rows survive
- (b) include all rows but apply `sample_weight=0.1` to pseudo rows
- (c) train two variants and ensemble
Which would you require for the eventual R-073-Phase-1 trainer?

### Q4 — Receiver-handedness assumption

Hard-coded "right-handed receiver" in the prompt. This introduces ~10%
systematic noise (Fan Zhendong is left-handed, Harimoto right-handed,
mix in the 290 matches). Should V1 instead:
- (a) ask Gemini to output **court-absolute** coordinates (left/mid/right of table)
  and have us post-process to FH/BH based on a separate Gemini call for
  "receiver dominant hand" per video, OR
- (b) keep V1 simple, accept the noise, document it in the manifest?
- (c) drop all pointId labels for matches where left-handed players are
  visually identifiable, OR pre-screen the match list?

Trade-off: (a) adds an extra Gemini call per video and a 2-axis pipeline;
(b) accepts known structural noise that may hurt the pointId head; (c)
loses ~10-15% of available training data.

### Q5 — Pilot sample size

10 videos × ~25 strokes per video = ~250 strokes total. Hand-truth target
~200 strokes. Is n=200 statistically sufficient to call the accuracy
gates correctly? For comparison: the R-029a/R-029b feature gates use
full 5-fold OOF (~70k rows). The pilot operates at 0.3% of that scale.
Is there a risk of the pilot PASSING noisily and then failing at full
scale (or vice versa)?

### Q6 — Match-grouping correctness on the 2 edge cases

- `LIKELY_SAME_MATCH` typo pair (`4-2` vs `42`): plan is to auto-merge
  in Phase 1. Approve?
- `MANUAL_REVIEW` pair (different games of same team round): plan is to
  leave as 2 distinct matches. Approve?
- 118 matches with missing chunks: plan is to keep them as-is (partial
  rally history is fine for shot-classification training, just produces
  fewer rallies per match). Approve, or should we drop matches with
  >X% chunks missing?

### Q7 — Pseudo-label-row tagging

To distinguish P2A-Gemini rows from real train rows + oldtest rows + aug
rows, the eventual `data/p2a_gemini.parquet` would carry a sentinel
column. Proposed scheme:
- `is_aug=0`, `is_p2a=1` (new flag)
- `serverGetPoint` either Gemini-derived (with confidence ≥ medium) or
  `-1` (low conf or `-1` outcome)
- All low-confidence pointId rows masked from point-head loss
- The trainer would need a `--include-external` flag (analogous to
  `--include-old-test`), but with per-field-mask logic for pseudo vs real
- Should `is_p2a=1` rows be excluded from server-head BCE entirely (like
  aug rows) for V1, or included with downweighting? R-009's lesson was
  "server pseudo amplifies bias"; this would dodge that by exclusion.

### Q8 — Realistic EV vs alternatives

The audit ranked R-066 (AutoGluon) and R-069 (Path B causal LM) as the
top-2 highest-EV legal preflights. Where does R-073 (P2A-Gemini) fit?
My internal ranking:

- R-066 AutoGluon: 0 to +0.015 LB, 1 day dev + 6h compute, known mechanism
- R-069 Path B smoke: 0 to +0.015 LB, 1h GPU smoke, diversity-add
- **R-073 P2A-Gemini: 0 to +0.008 LB, 3-5 days + Gemini cost, label-quality risk**
- R-068 action-point combos: 0 to +0.003 LB, 0.5 day + 3h CPU
- R-070 soft-F1 on v15feat_a: 0 to +0.008 LB, 0.5 day + 3h CPU

Is R-073 worth pursuing BEFORE R-066/R-069 land, or should it queue
behind them?

---

## 5. Specific artifacts for Codex to inspect

If Codex wants to verify any claim, the files are at:

| Claim | File to inspect |
|---|---|
| 50 tests pass | `tests/test_p2a_pilot.py` + run `python -m pytest tests/test_p2a_pilot.py -q` |
| Vocab mapping cites audit | `src/p2a_pilot/load_p2a.py` HAND_MAP / IS_SERVE_MAP / ACTION_MAP |
| Prompt design | `scripts/p2a_pilot/PROMPT_SAMPLE.txt` (rendered example) |
| Match grouping | `data/p2a_match_groups.json` (full output) + `src/p2a_pilot/group_matches.py` |
| Pilot manifest | `runs/p2a_pilot/pilot_manifest.json` (10 distinct matches verified) |
| Pilot gates | `src/p2a_pilot/accuracy.py:PILOT_GATES` |
| Protocol | `scripts/p2a_pilot/PILOT_PROTOCOL.md` |
| Legality position | `audits/P2A_DATASET_AUDIT_2026-05-19.md` §3 |
| Failure-mode precedents | `LESSONS_CHECKLIST.md` (R-009 pseudo entry, R-021 ShuttleSet22 entry) |
| Top-level audit context | `AUDIT_2026-05-23.md` §6.5 (EV table — R-073 not yet listed there) |

---

## 6. What Codex should return

Verdict + per-question concrete answer + any blocking findings.

If `APPROVE_WITH_FIXES`: list the specific fixes needed BEFORE the pilot
runs on Antigravity (so Jabir doesn't waste an evening on a flawed
prompt). I will apply the fixes and re-request approval before pilot
launch.

If `BLOCK`: state the dispositive reason and what (if anything) would
unblock — or recommend abandoning the lever entirely in favor of R-066 /
R-069.

If `APPROVE`: confirm Jabir can run the pilot as designed, with the
understanding that any positive-pilot result still requires a full R-073
preflight in `REVIEW_QUEUE.md` before Phase-1 training (with `R-073`
attached to a Phase-1 design that addresses Q3, Q4, Q7 above).
