"""Parse and validate Gemini's JSON response against the pilot schema.

Web Gemini sometimes wraps JSON in ```json ... ``` fences or adds a brief
preamble like "Here is the JSON:". This module:
  1. Strips common preamble/fence noise
  2. Parses the JSON
  3. Validates against the strict schema documented in build_prompt
  4. Checks invariants (stroke count matches, rally coverage is exact, etc.)
  5. Returns a (ParseResult, parsed_dict | None) tuple
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Court-absolute landing zones (NEW SCHEMA — see build_prompt.py FIELD_DEFINITIONS).
# The 9-grid is now described in camera-frame coordinates (table_left/mid/right
# from camera POV × short/half/long from net to baseline), NOT receiver-relative
# FH/BH. Receiver-relative FH/BH is derived post-hoc using `receiver_hand`
# which Gemini outputs per rally.
VALID_LANDING = {
    "table_left_short",  "table_mid_short",  "table_right_short",
    "table_left_half",   "table_mid_half",   "table_right_half",
    "table_left_long",   "table_mid_long",   "table_right_long",
    "off_grid",
}
VALID_POSITION = {"left", "center", "right", "unknown"}
VALID_HAND = {"right", "left", "unknown"}
VALID_CONF = {"low", "medium", "high"}
VALID_OUTCOME = {0, 1, -1}


@dataclass
class ParseResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _strip_preamble_and_fences(raw: str) -> str:
    """Robustly extract the JSON block from a Gemini response.

    Handles common patterns:
      - Wrapped in ```json ... ```
      - Wrapped in ``` ... ```
      - Preamble like "Here is the JSON:\n{...}"
      - Trailing commentary after the closing }
    """
    text = raw.strip()
    # Strip fenced code block if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    # Find the first { and the matching last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return text  # let json.loads raise
    return text[first_brace : last_brace + 1]


def parse_gemini_response(
    raw_text: str,
    expected_video_id: str,
    expected_n_strokes: int,
) -> tuple[ParseResult, dict | None]:
    """Parse and validate one Gemini response.

    Parameters
    ----------
    raw_text : str
        The exact text Gemini returned. May contain markdown fences or
        preamble — this function strips them.
    expected_video_id : str
        The video_id we asked Gemini to label (must match the response's
        video_id field).
    expected_n_strokes : int
        The number of strokes we provided as input. The response's "strokes"
        array must have exactly this length.

    Returns
    -------
    (ParseResult, parsed_dict | None)
        If parsing succeeds, parsed_dict is the validated JSON; otherwise None.
        ParseResult.ok is True iff there are no schema/invariant errors.
    """
    result = ParseResult(ok=True)
    cleaned = _strip_preamble_and_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        result.add_error(f"JSON parse failed: {e}")
        return result, None

    if not isinstance(data, dict):
        result.add_error(f"Top-level JSON must be an object, got {type(data).__name__}")
        return result, None

    # video_id check
    vid = data.get("video_id")
    if vid != expected_video_id:
        result.add_error(
            f"video_id mismatch: response={vid!r}, expected={expected_video_id!r}"
        )

    # strokes — count + schema
    strokes = data.get("strokes")
    if not isinstance(strokes, list):
        result.add_error("'strokes' must be a list")
        strokes = []
    else:
        if len(strokes) != expected_n_strokes:
            result.add_error(
                f"strokes count mismatch: response={len(strokes)}, "
                f"expected={expected_n_strokes}"
            )
        seen_idx = set()
        for i, s in enumerate(strokes):
            if not isinstance(s, dict):
                result.add_error(f"strokes[{i}] is not an object")
                continue
            idx = s.get("stroke_idx")
            if not isinstance(idx, int):
                result.add_error(f"strokes[{i}].stroke_idx must be int, got {idx!r}")
            elif idx in seen_idx:
                result.add_error(f"duplicate stroke_idx={idx}")
            else:
                seen_idx.add(idx)
                if idx != i:
                    result.add_warning(
                        f"strokes[{i}].stroke_idx={idx} (expected {i} for in-order)"
                    )
            if s.get("landing_zone") not in VALID_LANDING:
                result.add_error(
                    f"strokes[{i}].landing_zone={s.get('landing_zone')!r} "
                    f"not in {sorted(VALID_LANDING)}"
                )
            if s.get("landing_confidence") not in VALID_CONF:
                result.add_error(
                    f"strokes[{i}].landing_confidence={s.get('landing_confidence')!r} "
                    f"not in {sorted(VALID_CONF)}"
                )
            if s.get("player_position") not in VALID_POSITION:
                result.add_error(
                    f"strokes[{i}].player_position={s.get('player_position')!r} "
                    f"not in {sorted(VALID_POSITION)}"
                )
            if s.get("position_confidence") not in VALID_CONF:
                result.add_error(
                    f"strokes[{i}].position_confidence={s.get('position_confidence')!r} "
                    f"not in {sorted(VALID_CONF)}"
                )

    # rallies — coverage + schema
    rallies = data.get("rallies")
    if not isinstance(rallies, list):
        result.add_error("'rallies' must be a list")
        rallies = []
    else:
        covered_idx: set[int] = set()
        for i, r in enumerate(rallies):
            if not isinstance(r, dict):
                result.add_error(f"rallies[{i}] is not an object")
                continue
            rid = r.get("rally_id")
            if not isinstance(rid, int):
                result.add_error(f"rallies[{i}].rally_id must be int, got {rid!r}")
            idxs = r.get("stroke_indices")
            if not isinstance(idxs, list) or not all(isinstance(x, int) for x in idxs):
                result.add_error(f"rallies[{i}].stroke_indices must be list[int]")
                continue
            if idxs != sorted(idxs):
                result.add_error(
                    f"rallies[{i}].stroke_indices not strictly sorted: {idxs}"
                )
            # A rally must be a contiguous range of stroke indices —
            # rallies cannot interleave. ``[0, 2]`` is invalid when 1 exists
            # because stroke 1 chronologically belongs to one of these rallies
            # or to a third, but never neither.
            if len(idxs) >= 2:
                expected = list(range(idxs[0], idxs[0] + len(idxs)))
                if idxs != expected:
                    result.add_error(
                        f"rallies[{i}].stroke_indices must be a contiguous "
                        f"range; got {idxs} (expected {expected} starting at "
                        f"{idxs[0]})"
                    )
            for x in idxs:
                if x in covered_idx:
                    result.add_error(f"stroke_idx={x} appears in multiple rallies")
                covered_idx.add(x)
            if r.get("server_won_rally") not in VALID_OUTCOME:
                result.add_error(
                    f"rallies[{i}].server_won_rally={r.get('server_won_rally')!r} "
                    f"not in {sorted(VALID_OUTCOME)}"
                )
            if r.get("outcome_confidence") not in VALID_CONF:
                result.add_error(
                    f"rallies[{i}].outcome_confidence={r.get('outcome_confidence')!r} "
                    f"not in {sorted(VALID_CONF)}"
                )
            # NEW per-rally fields (Codex 2026-05-25 fix #4): receiver-relative
            # axis is computed post-hoc from these instead of being hard-coded
            # in the prompt as "assume right-handed receiver".
            if r.get("server_hand") not in VALID_HAND:
                result.add_error(
                    f"rallies[{i}].server_hand={r.get('server_hand')!r} "
                    f"not in {sorted(VALID_HAND)}"
                )
            if r.get("receiver_hand") not in VALID_HAND:
                result.add_error(
                    f"rallies[{i}].receiver_hand={r.get('receiver_hand')!r} "
                    f"not in {sorted(VALID_HAND)}"
                )
            if r.get("hand_confidence") not in VALID_CONF:
                result.add_error(
                    f"rallies[{i}].hand_confidence={r.get('hand_confidence')!r} "
                    f"not in {sorted(VALID_CONF)}"
                )
        # Coverage: every stroke must be in exactly one rally
        expected_set = set(range(expected_n_strokes))
        missing = expected_set - covered_idx
        extra = covered_idx - expected_set
        if missing:
            result.add_error(f"rallies miss stroke_idx: {sorted(missing)}")
        if extra:
            result.add_error(f"rallies contain out-of-range stroke_idx: {sorted(extra)}")

    return result, (data if result.ok or data else None)
