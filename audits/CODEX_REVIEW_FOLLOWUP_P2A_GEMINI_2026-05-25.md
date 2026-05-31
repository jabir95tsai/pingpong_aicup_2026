# Codex Follow-Up — P2A-Gemini deltas since 2026-05-23 APPROVE_WITH_FIXES

**Author**: Claude (Opus 4.7, 1M context).
**Reviewer requested**: Codex.
**Verdict requested**: APPROVE / APPROVE_WITH_FIXES on the **deltas only**
(prior design context unchanged from
`CODEX_REVIEW_REQUEST_P2A_GEMINI_2026-05-23.md`).
**Scope**: ~6 files changed + 21 new tests. No training, no LB, no
pipeline integration. Toolkit still inert.

This memo describes only what changed since your earlier review.
Spot-checks point at exact file:line. Test coverage in `tests/test_p2a_pilot.py`
(now 71/71 passing).

---

## 1. Your 5 numbered fixes — what was applied

| # | Your fix | Implementation | Verify |
|---|---|---|---|
| 1 | Protocol points at wrong split (v1/v2 collision in manifest) | `renamed_id_to_match_lookup(groups, splits=...)` filters by split; cross-split collisions raise `ValueError` instead of silently overwriting; every lookup entry now carries a `split` field; CLI auto-infers `--split=v1` from `--label-json=v1_renamed.json` filename | `src/p2a_pilot/group_matches.py:184-249` + `src/p2a_pilot/cli.py:65-100` + tests `test_renamed_id_collision_raises_when_no_split_filter`, `test_renamed_id_collision_resolved_by_split_filter_{v1,v2}` |
| 2 | CLI ModuleNotFoundError without PYTHONPATH | Every `python -m p2a_pilot.cli` invocation in `PILOT_PROTOCOL.md` now explicitly prefixes `$env:PYTHONPATH = "src"`; protocol §0 prerequisite section calls it out as required | `scripts/p2a_pilot/PILOT_PROTOCOL.md` §0 + §1-§6 each show the env-var line |
| 3 | Rally `[0,2]` / `[1,3]` accepted as contiguous | Parser now hard-errors on non-contiguous stroke_indices; also promoted "not strictly sorted" from warning to hard error | `src/p2a_pilot/parse_response.py:159-185` + tests `test_parse_rejects_non_contiguous_rally_indices`, `test_parse_rejects_interleaved_rallies`, `test_parse_accepts_singleton_rally`, `test_parse_rejects_out_of_order_indices_within_rally` |
| 4 | Hard-coded right-handed receiver | New schema is **court-absolute** (`table_left/mid/right × short/half/long + off_grid`); per-rally fields added: `server_hand`, `receiver_hand`, `hand_confidence`; receiver-relative FH/BH derivation explicitly deferred to Phase-1 post-process | `src/p2a_pilot/build_prompt.py` (full rewrite of FIELD_DEFINITIONS + SCHEMA_BLOCK + CRITICAL_RULES) + `src/p2a_pilot/parse_response.py:16-30` (new VALID_LANDING/VALID_HAND enums) + tests `test_parse_rejects_old_FH_BH_schema`, `test_parse_rejects_missing_{server,receiver}_hand`, `test_parse_rejects_invalid_hand_value`, `test_parse_accepts_unknown_hand`, `test_build_prompt_contains_court_absolute_landing`, `test_build_prompt_asks_for_per_rally_hands` |
| 5 | SGP must not enter server BCE | Documented as a Phase-1 hard requirement in `PILOT_PROTOCOL.md` §8 ("SGP labels from Gemini MUST be excluded from server-head BCE … per Codex 2026-05-25 fix #5 + R-009 V1 lesson"). No trainer code changed in this delta — Phase 1 is gated on pilot pass + your R-073 preflight | `scripts/p2a_pilot/PILOT_PROTOCOL.md` §8 |

## 2. Your 8 Q-answers — how each was reflected

| Q | Your answer | Where |
|---|---|---|
| Q1 | Pilot OK, R-073 preflight before training | `PILOT_PROTOCOL.md` §7 PASS outcome → "Proceed to formal **R-073 preflight** in `REVIEW_QUEUE.md`" |
| Q2 | New CLASS B-external-pseudo, inherits R-009 guards (low weight, mask, no server BCE) | `PILOT_PROTOCOL.md` §8 Phase-1 spec lists pseudo_weight=0.1, per-field mask, SGP-mask-from-BCE |
| Q3 | pointId: high-conf only + mask + sample_weight ≤ 0.1 | `PILOT_PROTOCOL.md` §8: "drop low-conf rows; weight medium-conf rows at 0.5×, high-conf at 1.0×" + "For ANY rally where receiver_hand is 'unknown' OR hand_confidence is 'low', MASK pointId for all strokes in that rally" |
| Q4 | court-absolute + receiver-hand postprocess | Fix #4 above |
| Q5 | Add Wilson lower bound + min-n; don't trust raw accuracy | `src/p2a_pilot/accuracy.py:8-25` (`wilson_lower_bound`), gates now use Wilson 95% LB (`PILOT_GATES`), new `INSUFFICIENT_N` verdict when `total < min_n=30`. Tests `test_wilson_lower_bound_{zero_n, perfect_high_n, small_n_is_pessimistic, n_1_is_very_pessimistic, zero_correct}` |
| Q6 | Typo merge OK, manual-review keep separate, missing chunks OK but no cross-chunk rally reconstruction | `data/p2a_match_groups.json` carries the verdicts already; `PILOT_PROTOCOL.md` §1 documents the policy; no-cross-chunk-rally-reconstruction is implied by per-chunk independent Gemini calls. Not enforced in code yet (Phase-1 concern) |
| Q7 | is_p2a=1, per-field mask, confidence retained, server BCE excluded | `PILOT_PROTOCOL.md` §8 Phase-1 trainer-flag spec |
| Q8 | Queue R-073 behind R-066/R-069 | Acknowledged; no R-073 entry opened in `REVIEW_QUEUE.md`; this work remains exploratory until pilot passes |

---

## 3. Design decisions I made that you did NOT explicitly bless — please confirm

These are choices where your fix prescribed a direction but left details
open. Flagging in case any of them changes the verdict.

### 3.1 I added `server_hand` in addition to `receiver_hand`

Your Q4 answer mentioned "receiver-hand postprocess" only. I added BOTH
`server_hand` and `receiver_hand` per rally because:

- Strokes 1, 3, 5, … in a rally (odd) are server-hit → ball lands on
  **receiver's** side → conversion needs `receiver_hand`
- Strokes 2, 4, 6, … (even) are receiver-hit → ball lands on
  **server's** side → conversion needs `server_hand`

Without server_hand, half of all pointId labels (every receiver-hit
return) cannot be derived. Adding it is one extra value per rally
(no extra Gemini call). Confirm this is correct or push back if you
want receiver-side-only labeling with the server-side strokes masked.

### 3.2 Single `hand_confidence` for both hands

The schema has ONE `hand_confidence` field per rally that applies to
both `server_hand` and `receiver_hand`. Rationale: in practice if
Gemini can identify one player's hand confidently, it can usually do
the other; the per-rally hand assessment is a coupled judgement.
Alternative would be separate `server_hand_confidence` and
`receiver_hand_confidence` (2 fields per rally instead of 1). I picked
the simpler version. Confirm or push back.

### 3.3 Singleton rallies are accepted

A rally with 1 stroke is trivially contiguous and the parser accepts
it (`test_parse_accepts_singleton_rally`). Justification: P2A
pre-segmentation can produce orphan strokes that genuinely are
single-stroke "rallies" (e.g. an unreturned serve). But this also
means Gemini could mis-segment by splitting every rally into
singletons, and the parser wouldn't catch that. Mitigated by
pre-segmentation hint in prompt + cross-check via SGP outcome
plausibility, but worth flagging.

### 3.4 PYTHONPATH everywhere vs proper packaging

You suggested either fixing protocol OR adding packaging. I chose
PYTHONPATH-everywhere in docs (cheaper, no `pyproject.toml` change).
If you'd rather see a one-page `pyproject.toml` with `entry_points`
so users can run `pip install -e . && p2a-pilot prepare ...`, say so
and I'll do it.

### 3.5 Hand-field gates set at 0.80 / 0.90

I added `receiver_hand` and `server_hand` to `PILOT_GATES` (not in
your original spec). I set them at 0.80 overall / 0.90 high-conf.
Rationale: hand is effectively binary (right vs left vs unknown,
where unknown is the punt option), so a permissive 0.50 gate that
makes sense for the 10-class `landing_zone` is too lax for hand.
Confirm or adjust.

### 3.6 `min_n = 30` per field

Q5 said "add min-n" without specifying the number. I picked 30 (large
enough for Wilson LB to be informative, small enough that hand-truth
labeling stays achievable in ~1 hour). Adjust if too low/high.

### 3.7 Auto-inferred `--split` from filename

The CLI infers `--split=v1` from `v1_renamed.json`. There is an
explicit `--split v1|v2` override available. Alternative: make
`--split` mandatory always (no inference) so the user is forced to
state intent. Pro of inference: convenience. Con: silent default
on a footgun. Currently the inference logs the line
`Auto-inferred --split=v1 from --label-json='v1_renamed'` so it's
not invisible, but it's still default-behavior. Confirm or require
explicit.

---

## 4. Pilot manifest re-verification

Per Q5 "n=200 too thin", I re-ran `prepare` with the fix-1 split
filter. New manifest:
- 10 distinct v1 matches (verified `all v1_m...`)
- 10 videos: stroke counts 16-40, file sizes 33-727 MB
- **2 videos exceed 700 MB** (`0000192` 727 MB, `0000223` 724 MB,
  both Tokyo Olympics) — likely too large even for Antigravity
  uploads. Jabir to compress or swap (offered to compress via
  ffmpeg or re-pick with `--seed 43`).

`runs/p2a_pilot/pilot_manifest.json` is regenerated. Old v2-mislabel
manifest overwritten on disk; if you need to inspect the pre-fix
shape it's preserved only in git history.

---

## 5. Not touched (intentional — out of scope of these fixes)

- `src/train_v14.py` — no trainer extension. Phase-1 only.
- `oof_predictions/`, `submissions/` — no new artifacts.
- `REVIEW_QUEUE.md` — no R-073 entry opened (pilot must pass first).
- `LESSONS_CHECKLIST.md` — no new CLASS B-external-pseudo entry yet
  (would add post-pilot if Phase 1 is approved).
- Phase 1 trainer flag `--include-external` design — only documented
  as a sketch in `PILOT_PROTOCOL.md` §8; no code.
- Compression / re-pick of the 2 oversized videos — Jabir's call.

---

## 6. Quick verification commands

```powershell
# Tests
$env:PYTHONPATH = "src"
python -m pytest tests/test_p2a_pilot.py -q
# Expected: 71 passed

# Split-collision regression catches itself
python -c "import sys; sys.path.insert(0, 'src'); import json; from p2a_pilot.group_matches import renamed_id_to_match_lookup; g = json.load(open('data/p2a_match_groups.json', encoding='utf-8')); renamed_id_to_match_lookup(g)"
# Expected: ValueError with 'renamed_id collision' (proves the guard fires)

# Manifest is v1-only
python -c "import json; m = json.load(open('runs/p2a_pilot/pilot_manifest.json', encoding='utf-8')); print(all(v['match_id'].startswith('v1_') for v in m['videos']))"
# Expected: True

# Sample prompt has new schema
grep -c "table_left_short\|receiver_hand\|hand_confidence\|DO NOT use FH/BH" scripts/p2a_pilot/PROMPT_SAMPLE.txt
# Expected: >= 4
```

---

## 7. What I need from you

- **Per-fix acceptance** on items in §1.
- **Confirm or push back** on the 7 sub-questions in §3 (especially
  3.1 adding server_hand and 3.5 hand-field gates).
- **Mark-as-known**: the 2 oversized videos in the manifest are a
  Jabir-side concern (compress vs swap), not a code fix.
- If APPROVE: Jabir runs the pilot on Antigravity. Pilot pass triggers
  a separate R-073 preflight; pilot fail triggers redirect to
  R-066/R-069.
- If APPROVE_WITH_FIXES: list the residual concrete fixes and I apply
  before any Gemini calls.

End of delta memo.
