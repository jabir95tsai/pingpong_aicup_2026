"""Generate the Gemini prompt text for one P2A video.

The prompt is designed for the **web Gemini UI** (gemini.google.com).
Workflow:
  1. Upload the .mp4 to the web Gemini chat
  2. Paste the prompt text (output of ``build_prompt``) as the user message
  3. Gemini responds with JSON
  4. Save the JSON response to ``runs/p2a_pilot/<video_id>.gemini.json``
  5. Run ``parse_response.py`` to validate the response

Design notes:
  - We tell Gemini the exact stroke count + timestamps so it can't drift on
    stroke counting (a real problem with raw video CV).
  - We pre-segment rallies via time-gap heuristic and ask Gemini to verify.
  - We ask for confidence labels per field so we can filter low-confidence
    rows downstream.
  - We instruct "JSON only, no markdown fences, no preamble" — but the
    parser handles fence-stripping anyway as a robustness measure.

Usage:
    from p2a_pilot.load_p2a import load_p2a_videos, pick_pilot_videos, video_anchor_table
    from p2a_pilot.build_prompt import build_prompt

    videos = load_p2a_videos("path/to/v1.json")
    pilot  = pick_pilot_videos(videos, n=10)
    for v in pilot:
        anchors = video_anchor_table(v)
        prompt  = build_prompt(anchors)
        print(prompt)   # paste into web Gemini after uploading the .mp4
"""
from __future__ import annotations

import json
from typing import Iterable

from .load_p2a import pre_segment_rallies


PROMPT_HEADER = """\
You are analysing a table-tennis (ping-pong) match video clip.

I have already detected the stroke events in this clip with their start/end
timestamps, hitter's HAND (forehand/backhand), and whether each stroke is a
serve. That data is provided below as INPUT.

Your task: for each detected stroke, watch the video at that timestamp and
identify THREE additional fields. You also need to verify the rally
segmentation (which strokes belong to the same point).

OUTPUT: return ONLY a valid JSON object that conforms to the schema below.
Do NOT wrap in markdown code fences. Do NOT include any preamble or
explanation. JSON only.
"""

SCHEMA_BLOCK = """\
JSON SCHEMA (your response must match this shape exactly):

{
  "video_id": "<the video_id I gave you>",
  "rallies": [
    {
      "rally_id": 0,
      "stroke_indices": [<CONTIGUOUS 0-indexed stroke positions in this rally>],
      "server_won_rally": 0 | 1 | -1,
      "outcome_confidence": "low" | "medium" | "high",
      "outcome_evidence": "<one short sentence explaining how you decided>",
      "server_hand": "right" | "left" | "unknown",
      "receiver_hand": "right" | "left" | "unknown",
      "hand_confidence": "low" | "medium" | "high"
    },
    ...
  ],
  "strokes": [
    {
      "stroke_idx": 0,
      "landing_zone": "table_left_short" | "table_mid_short" | "table_right_short" | "table_left_half" | "table_mid_half" | "table_right_half" | "table_left_long" | "table_mid_long" | "table_right_long" | "off_grid",
      "landing_confidence": "low" | "medium" | "high",
      "player_position": "left" | "center" | "right" | "unknown",
      "position_confidence": "low" | "medium" | "high"
    },
    ...
  ]
}
"""

FIELD_DEFINITIONS = """\
FIELD DEFINITIONS:

LANDING_ZONE — where the ball LANDED on the table, in CAMERA-FRAME
COORDINATES (NOT relative to either player's forehand/backhand).

  Imagine the full table viewed from the broadcast camera angle. Split the
  table-top into a 3x3 grid:

    Columns (left to right from camera POV):
      - table_left   = leftmost third of the table
      - table_mid    = center third
      - table_right  = rightmost third

    Rows (front to back relative to the camera, but applied to whichever
    HALF of the table the ball landed on):
      - short = within ~30 cm of the NET (i.e. close to the table center line)
      - half  = middle third between net and baseline
      - long  = within ~30 cm of the BASELINE (far end of that half)

  Combine column + row, e.g. "table_left_short" = ball landed on the left
  side of the table, close to the net (on whichever half the ball was
  hit into).

  Special:
    - "off_grid" = ball into the net, off the table, off-side, OR if you
      cannot reliably tell where it landed (broadcast angle obstructed,
      ball obscured by player, etc.). When unsure, prefer "off_grid" with
      confidence "low" over guessing.

  IMPORTANT: DO NOT use FH/BH terminology. Use ONLY the table_left /
  table_mid / table_right axis from camera POV. We post-process to
  receiver-relative FH/BH later using receiver_hand from the rally object.

PLAYER_POSITION — where the HITTER of this stroke is standing on their own
side of the table, in a 3-zone partition of THEIR own half (from the
hitter's POV, but you can read from camera frame):
  - "left"    = the hitter is standing on their own-side left third
  - "center"  = standing in the middle third
  - "right"   = standing in the right third
  - "unknown" = position is obscured or not clear

SERVER_WON_RALLY — for each rally, whether the player who SERVED the first
stroke of that rally won the point:
  - 1  = server won (server scored)
  - 0  = receiver won (server lost the point)
  - -1 = unclear (rally not visibly completed in clip, or outcome unobservable)

SERVER_HAND / RECEIVER_HAND — per-rally dominant playing hand of the
server (player who serves the rally's first stroke) and the receiver
(the other player).
  - "right"   = right-handed
  - "left"    = left-handed
  - "unknown" = cannot determine confidently from this rally
  Use any visual evidence: which hand holds the racket, swing direction,
  body orientation when waiting to receive. If both players are visible
  in any one stroke, that's usually enough.
  These hands apply for the WHOLE rally — within a rally each player
  always uses the same hand. ONE rally object covers BOTH players' hands.

CONFIDENCE LEVELS (apply to each field independently):
  - "high"   = certain; clear visual evidence
  - "medium" = probable but some ambiguity
  - "low"    = guessing; prefer "off_grid" / "unknown" / -1 over guessing
"""

CRITICAL_RULES = """\
CRITICAL RULES:

1. Output EXACTLY {n_strokes} stroke objects in "strokes" — one per input
   stroke. Do NOT invent extra strokes. Do NOT skip any.

2. Strokes in your output must be in the SAME ORDER as the input stroke
   list (by stroke_idx).

3. Each "stroke_idx" in your output must match the input stroke_idx exactly.

4. Verify rally segmentation. I have pre-segmented the strokes into
   tentative rallies based on time gaps; please adjust if the video shows
   a different rally structure (e.g., merge two pre-segments that are
   actually one rally; split a pre-segment that contains two rallies).

   Each rally's "stroke_indices" MUST be a CONTIGUOUS, STRICTLY
   INCREASING run of integers (e.g. [4,5,6,7] is valid; [4,6,7] and
   [4,5,7] are INVALID — strokes cannot interleave between rallies).
   Every stroke_idx in [0, {n_strokes_minus_one}] must appear in
   EXACTLY ONE rally.

5. For each rally, ALSO output server_hand, receiver_hand, and
   hand_confidence. These describe the dominant playing hand of the
   two players involved. The server is the player whose first stroke
   has the lowest stroke_idx in this rally. The receiver is the other
   player. If you cannot tell from the video, use "unknown" rather
   than guessing.

6. Output VALID JSON only. No markdown code fences (```), no preamble,
   no explanation, no trailing commentary. The first character of your
   response must be '{{' and the last must be '}}'.

7. When visual evidence is ambiguous, prefer "off_grid" / "unknown" /
   -1 with confidence "low" over a guess.
"""


def _format_strokes_for_prompt(strokes: list[dict]) -> str:
    """Render the input stroke list as a compact human-readable block."""
    lines = []
    for s in strokes:
        idx = s.get("stroke_idx")
        start = s.get("start")
        end = s.get("end")
        hand = s.get("hand_raw") or "?"
        is_serve = s.get("is_serve")
        is_serve_str = "SERVE" if is_serve else ("rally-shot" if is_serve is False else "?")
        action = s.get("action_raw") or "?"
        action_aicup = s.get("action_aicup")
        action_str = f"{action}" + (f" (AICUP actionId={action_aicup})" if action_aicup else " (mapping unresolved)")
        lines.append(
            f"  stroke_idx={idx:3d}  t=[{start:7.2f}, {end:7.2f}]s  "
            f"hand={hand}  type={is_serve_str}  action={action_str}"
        )
    return "\n".join(lines)


def _format_rallies_for_prompt(rallies: list[dict]) -> str:
    """Render the pre-segmented rally hypothesis."""
    if not rallies:
        return "  (no rallies — empty input)"
    lines = []
    for r in rallies:
        rid = r.get("rally_id")
        idxs = r.get("stroke_indices", [])
        if not idxs:
            lines.append(f"  rally_id={rid}  (empty)")
            continue
        lines.append(f"  rally_id={rid}  stroke_indices={idxs}")
    return "\n".join(lines)


def build_prompt(
    anchor_table: dict,
    rally_gap_seconds: float = 4.0,
    pre_segment: list[dict] | None = None,
) -> str:
    """Build the full Gemini prompt text for one P2A video.

    Parameters
    ----------
    anchor_table : dict
        Output of ``load_p2a.video_anchor_table``.
    rally_gap_seconds : float
        Threshold for the pre-segmentation heuristic. Default 4.0s.
    pre_segment : list[dict] | None
        Optional pre-segmented rallies to use. If None, computed from
        ``pre_segment_rallies``.

    Returns
    -------
    str
        The full prompt text. Paste this into the web Gemini chat AFTER
        uploading the corresponding .mp4 file.
    """
    strokes = anchor_table.get("strokes", [])
    n_strokes = len(strokes)
    if n_strokes == 0:
        raise ValueError(f"Anchor table for {anchor_table.get('video_id')} has 0 strokes")

    if pre_segment is None:
        pre_segment = pre_segment_rallies(anchor_table, rally_gap_seconds=rally_gap_seconds)

    rules = CRITICAL_RULES.format(
        n_strokes=n_strokes,
        n_strokes_minus_one=n_strokes - 1,
    )

    parts = [
        PROMPT_HEADER,
        "",
        f"VIDEO_ID: {anchor_table.get('video_id')}",
        f"TOTAL_STROKES: {n_strokes}",
        "",
        "INPUT STROKES (known fields from prior detection):",
        _format_strokes_for_prompt(strokes),
        "",
        f"PRE-SEGMENTED RALLIES (hypothesis from time-gap < {rally_gap_seconds}s; verify and adjust):",
        _format_rallies_for_prompt(pre_segment),
        "",
        FIELD_DEFINITIONS,
        "",
        SCHEMA_BLOCK,
        "",
        rules,
        "",
        "Now analyse the uploaded video and return the JSON.",
    ]
    return "\n".join(parts)
