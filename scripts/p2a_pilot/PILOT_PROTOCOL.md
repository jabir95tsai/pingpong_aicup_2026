# P2A-Gemini Pilot Protocol

Step-by-step instructions for the P2A → Gemini → AICUP-format pilot.

**Purpose**: validate whether Gemini can extract the 5 fields P2A JSON
labels lack (pointId axis, positionId, serverGetPoint, and the two
player hand designations needed to convert court-absolute landing zones
into receiver-relative FH/BH) accurately enough to use as supplemental
training data.

**Stop gate**: if pilot accuracy is below the thresholds in §5 (per
**Wilson 95% lower bound**, not raw point estimate), abandon and
redirect compute to R-066 (AutoGluon) or R-069 (Path B causal LM).

---

## 0. Prerequisites

1. **P2A dataset on disk** at `D:/P2A_dataset/dataset/`. Confirm layout:
   ```
   D:/P2A_dataset/dataset/
   ├── proj.json                # original ↔ renamed filename map (both splits)
   ├── label/
   │   ├── v1.json              # dict shape: {"fps": 25, "gts": [...]}, original Chinese names
   │   ├── v1_renamed.json      # list shape, numeric names — USE THIS FOR PILOT
   │   ├── v2.json
   │   └── v2_renamed.json
   └── video/
       ├── v1/                  # 1281 .mp4 files, 0000000.mp4 .. 0001280.mp4
       └── v2/                  # 1208 .mp4 files, 0000000.mp4 .. 0001207.mp4
   ```
   **CRITICAL**: v1 and v2 share the renamed_id namespace (both have
   `0000000.mp4`). The toolkit auto-infers `--split=v1` from the
   `--label-json` filename to avoid cross-split contamination. Verify
   the printed line `Auto-inferred --split=v1` appears.
2. **Web Gemini or Antigravity** access. Web Gemini upload cap is
   ~100-200 MB; for the larger pilot videos (>150 MB) use Antigravity
   if available, or compress with `ffmpeg -i in.mp4 -vcodec libx264
   -crf 28 out.mp4`.
3. **Python path**: every `python -m p2a_pilot.cli ...` invocation in
   this protocol requires `$env:PYTHONPATH = "src"` first (the toolkit
   is not yet packaged). Each command below shows it explicitly. Or set
   it once per shell session:
   ```powershell
   $env:PYTHONPATH = "src"
   ```

---

## 1. Build match groups (one-time, 2 sec)

P2A clips are 5-15 min chunks of full ITTF matches. Group them so AICUP
`match` IDs stay consistent across chunks and so the pilot picker can
sample 1 chunk per match (max diversity).

```powershell
$env:PYTHONPATH = "src"
python -m p2a_pilot.cli group-matches `
    --proj-json "D:/P2A_dataset/dataset/proj.json" `
    --out data/p2a_match_groups.json
```

Expected output:
```
Split: v1   Matches: 156   Videos: 1281   Unmatched: 0
Split: v2   Matches: 134   Videos: 1208   Unmatched: 0
```

Two near-duplicate pairs flagged with verdicts:
- `LIKELY_SAME_MATCH` — `4-2` vs `42` typo, plan to merge in Phase 1
- `MANUAL_REVIEW` — different games of same team round, keep separate

---

## 2. Prepare prompts (10 sec, no Gemini calls)

```powershell
$env:PYTHONPATH = "src"
python -m p2a_pilot.cli prepare `
    --label-json "D:/P2A_dataset/dataset/label/v1_renamed.json" `
    --match-groups data/p2a_match_groups.json `
    --split v1 `
    --n 10 `
    --seed 42 `
    --min-strokes 10 `
    --max-strokes 40 `
    --chunk-strategy smallest_index `
    --out runs/p2a_pilot
```

`--split v1` is explicit here (Codex 2026-05-25 polish suggestion).
Without it the CLI would auto-infer the same value from the
`v1_renamed.json` filename and log `Auto-inferred --split=v1`, but
explicit-over-implicit is preferred. Manifest has 10 v1 match_ids,
no v2 mixing.

Output layout:
```
runs/p2a_pilot/
├── pilot_manifest.json         # which 10 videos were picked + match context
├── prompts/<video_id>.prompt.txt   # paste into Gemini after uploading mp4
├── anchors/<video_id>.anchors.json # stroke timestamps + known fields
├── responses/                  # YOU save Gemini's responses here
└── parsed/                     # filled in by `parse` step
```

---

## 3. Run Gemini on each pilot video (~1.5-2 hours for 10 videos)

**Per video** (~10 min including Gemini wait time):

1. Open a **fresh chat tab** at https://gemini.google.com/app (model:
   2.5 Flash works; 2.5 Pro / Gemini 3 better) OR open Antigravity for
   larger videos (>100 MB).
2. Upload the matching .mp4 from `D:/P2A_dataset/dataset/video/v1/<video_id>.mp4`.
3. Wait for upload acknowledgement.
4. Copy the entire contents of `runs/p2a_pilot/prompts/<video_id>.prompt.txt`,
   paste as chat message, send.
5. Wait for response (~30 sec to 2 min).
6. Save Gemini's response as `runs/p2a_pilot/responses/<video_id>.json`.
   The parser strips markdown fences + prose automatically, but saving
   just the `{...}` block is cleaner.
7. Move to next video (new chat tab).

**Important**:
- New chat tab per video — Gemini will conflate stroke indices across
  videos in the same conversation.
- If 3 of the first 5 produce schema-invalid output even on retry,
  **stop** and report back; the prompt needs revision.

---

## 4. Parse + validate (1 sec)

```powershell
$env:PYTHONPATH = "src"
python -m p2a_pilot.cli parse --out runs/p2a_pilot
```

Writes:
- `parsed/<video_id>.parsed.json` for each response that passed schema
- `parse_summary.json` with per-video pass/fail + error list

Pass = at least 8/10 videos validated. Schema checks (any of these can
fail):
- video_id matches request
- stroke count matches input
- all landing_zone in {table_left/mid/right_short/half/long, off_grid}
- all player_position in {left, center, right, unknown}
- all server_hand / receiver_hand in {right, left, unknown}
- all confidence values in {low, medium, high}
- every stroke_idx in [0, N-1] appears in exactly ONE rally
- each rally's stroke_indices is a CONTIGUOUS range (no interleaving)

---

## 5. Hand-truth labeling (~1 hour manual)

Watch the same 10 videos and fill in:

**`scripts/p2a_pilot/hand_truth/strokes.csv`** — court-absolute landing
+ position per stroke:
```csv
video_id,stroke_idx,landing_zone,player_position
0000144,0,table_mid_long,center
0000144,1,table_left_short,left
0000144,2,off_grid,right
...
```

**`scripts/p2a_pilot/hand_truth/rallies.csv`** — outcome + both
players' hands per rally:
```csv
video_id,rally_id,server_won_rally,server_hand,receiver_hand
0000144,0,1,right,right
0000144,1,0,right,left
0000144,2,-1,unknown,unknown
...
```

**IMPORTANT — do not use Gemini's rally_id as the truth source**. When
hand-labeling rally outcomes, use YOUR OWN rally identification (look
at the video and decide where rallies start/end). The pre-segmented
rallies in `anchors/<video_id>.anchors.json` are a useful starting
guide but Gemini may have changed them. Compare against the parsed
Gemini response; if Gemini's rally segmentation differs from yours,
that itself is a rally-segmentation error worth recording.

See `HAND_TRUTH_SCHEMA.md` for full field definitions + speed tips.
Aim for ~200 stroke labels + ~30 rally labels minimum.

---

## 6. Accuracy report (1 sec)

```powershell
$env:PYTHONPATH = "src"
python -m p2a_pilot.cli report --out runs/p2a_pilot `
    --hand-truth-strokes scripts/p2a_pilot/hand_truth/strokes.csv `
    --hand-truth-rallies scripts/p2a_pilot/hand_truth/rallies.csv
```

Output: per-field accuracy with **Wilson 95% lower bound**, confidence
breakdown, and pass/fail verdict per gate.

### Pilot gates (per `src/p2a_pilot/accuracy.py:PILOT_GATES`)

| Field | Overall WLB ≥ | High-conf WLB ≥ | min n |
|---|---:|---:|---:|
| `landing_zone` (court-absolute pointId proxy) | 0.50 | 0.65 | 30 |
| `player_position` | 0.70 | 0.85 | 30 |
| `server_won_rally` | 0.75 | 0.90 | 30 |
| `receiver_hand` | 0.80 | 0.90 | 30 |
| `server_hand`   | 0.80 | 0.90 | 30 |

"WLB" = Wilson 95% lower bound on observed accuracy. Per Codex
2026-05-25 review Q5, point estimates on n=200 are too noisy; the
lower bound is the most pessimistic accuracy compatible with the data.

**ALL 5 fields must pass BOTH gates AND meet min_n** to proceed to
Phase 1. If any field shows `INSUFFICIENT_N`, label more strokes for
that field before declaring a verdict.

If receiver_hand or server_hand fails the gate, point-id derivation in
Phase 1 cannot work and pointId labels must be masked entirely.

---

## 7. Decision

| Outcome | Action |
|---|---|
| All 5 fields PASS (Wilson lower bound) | Proceed to formal **R-073 preflight** in `REVIEW_QUEUE.md` describing the Phase-1 trainer integration. NO training yet. |
| pointId fails, others pass | Re-evaluate with Codex: is supplemental data WITHOUT pointId worth +0.001-0.003 LB? |
| receiver_hand or server_hand fails | pointId derivation impossible; abandon pointId; reconsider whether 3-field supplemental data (positionId + serverGetPoint + actionId from P2A JSON) is enough EV. |
| server_won_rally fails | Full abandon — SGP is the hardest field and without it the data largely duplicates P2A JSON. |
| Any INSUFFICIENT_N | Label more strokes; do not declare verdict prematurely. |

The decision belongs to Jabir + Codex. This script produces evidence only.

---

## 8. Phase 1 preview (NOT in pilot scope)

If pilot passes, Phase 1 needs an R-073 preflight specifying:

1. **AICUP-format mapper** `src/p2a_pilot/to_aicup.py`:
   - For each stroke, compute receiver-relative pointId from court-absolute
     `landing_zone` + the **target-player hand**, where the target player
     is the one on whose side the ball LANDS. Stroke parity within a
     rally decides which side that is:
       - **Server-hit strokes** (1st, 3rd, 5th, … within rally) — ball
         lands on the **receiver's** side → target-player hand =
         `receiver_hand`
       - **Receiver-hit strokes** (2nd, 4th, 6th, …) — ball lands on the
         **server's** side → target-player hand = `server_hand`
   - **Per-stroke pointId masking rule** (Codex 2026-05-25 pre-Phase-1 fix):
       - If the stroke's target-player hand == `"unknown"` → mask pointId
         for THIS stroke only (the other side may still be derivable)
       - If the rally's `hand_confidence` == `"low"` → mask pointId for
         ALL strokes in the rally (one confidence covers both hands;
         a low value taints both sides)
   - Apply per-field confidence filter: drop low-conf rows; weight
     medium-conf rows at 0.5×, high-conf at 1.0×
2. **Trainer flag** `--include-external data/p2a_gemini.parquet`
   - Analogous to `--include-old-test`
   - Marks rows with `is_p2a=1` (distinct from `is_aug=1`)
   - **SGP labels from Gemini MUST be excluded from server-head BCE**
     (per Codex 2026-05-25 fix #5 + R-009 V1 lesson: even when Gemini
     achieves 90%+ SGP accuracy, training server head on Gemini's
     labels risks the same bias-amplification class as R-009)
   - Pseudo weight 0.1× for V1 (conservative)
3. **R-073 formal preflight** in `REVIEW_QUEUE.md` with: vocab mapping
   verification, leak audit (no SGP from P2A enters server BCE), per-fold
   isolation (P2A rows match-grouped under their own match IDs, no
   cross-leakage into AICUP train), gate spec.
4. **Gemini API harness** for batch processing (~2400 remaining videos
   at ~$72-216 total).
