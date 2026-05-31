# Hand-truth labeling guide (updated 2026-05-25 per Codex review)

You're labeling the same 10 videos Gemini already labeled so we can
measure Gemini's accuracy on the 5 fields. This is the ground-truth check.

## Tools you need

- 10 .mp4 pilot videos from `D:/P2A_dataset/dataset/video/v1/`
- Any video player with frame-step (VLC, MPV, browser HTML5)
- `runs/p2a_pilot/pilot_manifest.json` — which videos and how many strokes
- `runs/p2a_pilot/anchors/<video_id>.anchors.json` — per-stroke
  timestamps + the already-known fields (hand from P2A, action from P2A)
- Two empty CSVs to fill in:
  - `scripts/p2a_pilot/hand_truth/strokes.csv`
  - `scripts/p2a_pilot/hand_truth/rallies.csv`

## ⚠ Important: do NOT trust Gemini's rally segmentation as truth

When you label rally outcomes, identify rallies **yourself** by watching
the video. Don't just adopt Gemini's `rally_id` values — that would hide
rally-segmentation errors.

Workflow:
1. Decide for yourself where each rally starts/ends in the video
2. Number your rallies starting from 0 (sequential)
3. Record your rally outcomes against YOUR rally_ids
4. The accuracy script will compare YOUR rally_id N against Gemini's
   rally_id N — if Gemini's segmentation differs, that's an error that
   shows up as misaligned outcomes (correctly counted as a failure)

If Gemini segmented very differently from you, that itself is a finding
worth recording in a comment alongside the accuracy report.

---

## `strokes.csv` — one row per stroke (court-absolute, not FH/BH)

```csv
video_id,stroke_idx,landing_zone,player_position
0000144,0,table_mid_long,center
0000144,1,table_left_short,left
0000144,2,table_right_half,right
0000144,3,off_grid,unknown
...
```

### `landing_zone` values (10 choices, COURT-ABSOLUTE)

Where the ball LANDED on the table, in **camera-frame coordinates**
(NOT receiver-relative FH/BH). The 3×3 grid:

|  | Column → | | |
|---|---|---|---|
| **Row ↓** | table_left | table_mid | table_right |
| **short** (near net) | `table_left_short` | `table_mid_short` | `table_right_short` |
| **half** (middle of half) | `table_left_half` | `table_mid_half` | `table_right_half` |
| **long** (near baseline) | `table_left_long` | `table_mid_long` | `table_right_long` |
| **anything else** | `off_grid` | | |

The "short/half/long" axis is from net → baseline on whichever HALF of
the table the ball landed on.

`off_grid` = ball into net, off table, off-side, OR you can't see
clearly (broadcast angle obstructed, ball obscured). **Prefer `off_grid`
over guessing.**

Why court-absolute (not FH/BH): so we can convert to receiver-relative
FH/BH later using each rally's `receiver_hand` AND `server_hand` fields.
**Which hand to use depends on stroke parity within the rally**
(Codex 2026-05-25 spec):

  - **Server-hit strokes** (1st, 3rd, 5th, … within the rally) land on
    the **receiver's** side → use `receiver_hand`
  - **Receiver-hit strokes** (2nd, 4th, 6th, …) land on the **server's**
    side → use `server_hand`

This handles left-handed players (Fan Zhendong etc.) correctly on
both sides of the table. If the relevant hand is `unknown` for a
given stroke, that stroke's pointId is masked from training.

### `player_position` values (4 choices)

Where the HITTER of THIS stroke is standing on their own side, in 3 zones:

| Value | Meaning |
|---|---|
| `left`    | hitter standing in left third of own side |
| `center`  | hitter standing in middle third |
| `right`   | hitter standing in right third |
| `unknown` | obscured / can't tell |

### Speed tips

- You don't need to label every stroke. 20-30 per video (~200 total).
  Blank rows are skipped by the accuracy script.
- Pause at the moment the hitter CONTACTS the ball (use 0.5× playback).
- Use stroke start/end timestamps in `anchors/<video_id>.anchors.json`
  to jump to the right moment.
- For `landing_zone`: use the table center line and the sidelines as
  reference. Service-box markings help.

---

## `rallies.csv` — one row per rally (outcome + BOTH players' hands)

```csv
video_id,rally_id,server_won_rally,server_hand,receiver_hand
0000144,0,1,right,right
0000144,1,0,right,left
0000144,2,-1,unknown,unknown
...
```

### `server_won_rally` (3 choices)

| Value | Meaning |
|---|---|
| `1`  | The player who SERVED the first stroke of this rally won the point |
| `0`  | The receiver won the point (server lost) |
| `-1` | Rally not visibly completed in the clip, or you can't tell |

### `server_hand` / `receiver_hand` (3 choices each)

| Value | Meaning |
|---|---|
| `right` | right-handed |
| `left`  | left-handed |
| `unknown` | cannot determine from this rally |

The server is the player whose first stroke has the lowest stroke_idx
in this rally. The receiver is the other player.

Within a rally, the players don't switch hands. Once you've identified
each player's dominant hand in any rally where both are visible, you
can reuse those values across all rallies that involve the same two
players in the same video (P2A clips are single matches).

### Identifying rallies

Two heuristics:
1. **Time gap**: more than ~3-5 seconds between strokes usually = new rally
2. **Visual cues**: ball stops bouncing; players reset position; brief
   replay before the serve

The `anchors/<video_id>.anchors.json` includes `pre_segment` (our
time-gap heuristic's guess). It's a starting point but trust your eyes
over the heuristic.

### Quickest signal for `server_won_rally`

- Watch the last 2-3 seconds after the final stroke
- Who celebrates / pumps fist / nods to themselves?
- Look at the scorebar if visible (most ITTF broadcasts show it
  shortly after the rally)

If the rally ends with a clear "ball off the table", the LAST PLAYER
TO HIT lost the point.

---

## How accuracy is computed

For each field:
1. If the truth row is blank, that stroke/rally is skipped.
2. If the truth has a value, accuracy = exact match.
3. Wilson 95% lower bound on accuracy (per-field, per-confidence-bucket)
   is computed and compared to the gate.

Pilot pass = Wilson lower bound ≥ gate AND total n ≥ 30 per field.

See `src/p2a_pilot/accuracy.py:PILOT_GATES` for the live gate values.

## How many labels do I need?

Minimum: **200 stroke labels and 30 rally labels** across all 10 videos.
This is borderline — Wilson 95% lower bound is sensitive at this n.
If you can comfortably do 300 strokes and 50 rallies, please do.

Specifically: each of the 5 fields needs **n ≥ 30** to receive any
verdict (otherwise reports as `INSUFFICIENT_N`). Since landing_zone
and player_position are per-stroke (so ~200 each is easy with 200
stroke labels), but receiver_hand and server_hand are per-rally
(so ~30 means labeling 30 rallies). Plan for 30+ rally labels.

## If you really can't tell

Leave the row blank rather than guess. The accuracy script ignores
blank truth rows. **Your own noise will make Gemini look worse than
it actually is.** Better to skip than to add noise.
