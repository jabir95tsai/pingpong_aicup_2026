"""Load P2A label JSON, pick pilot videos, build stroke-anchor tables.

The P2A JSON (per audit ``audits/P2A_DATASET_AUDIT_2026-05-19.md``) is shaped::

    [
      {
        "url": "0000000.mp4",
        "total_frames": null,
        "actions": [
          {
            "label_ids": null,
            "label_names": ["正手", "否", "控制"],
            "start_id": 106.34,
            "end_id":   106.82
          },
          ...
        ]
      },
      ...
    ]

``label_names`` is ``[hand, is_serve, action_type]`` per audit §2.

This module:
  - Loads a P2A JSON file
  - Maps Chinese labels to AICUP-aligned codes
  - Picks a deterministic pilot subset of N videos
  - Emits a per-video "stroke anchor table" used by build_prompt

Usage:
    from p2a_pilot.load_p2a import load_p2a_videos, pick_pilot_videos, video_anchor_table
    videos = load_p2a_videos("C:/path/to/P2A/dataset/label/v1.json")
    pilot = pick_pilot_videos(videos, n=10, seed=42)
    for v in pilot:
        anchors = video_anchor_table(v)
        # feed anchors to build_prompt.build_prompt(...)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

# Vocab mapping from P2A Chinese labels to AICUP IDs.
# Source: audits/P2A_DATASET_AUDIT_2026-05-19.md §2.

HAND_MAP = {
    "正手": 1,  # forehand
    "反手": 2,  # backhand
}

IS_SERVE_MAP = {
    "是": True,   # yes — serve
    "否": False,  # no — not serve
}

# 10 of 15 P2A action types map cleanly to AICUP actionId (audit §2 field 3).
# The 5 "UNRESOLVED" types are left as None — caller decides whether to drop or
# keep them with actionId=0 (other/unknown).
ACTION_MAP = {
    "拉":     1,   # loop / topspin pull -> AICUP 1 拉球
    "侧旋":   15,  # sidespin (serve)    -> AICUP 15 傳統 (closest serve variant)
    "摆短":   11,  # short push          -> AICUP 11 擺短
    "侧身拉": 1,   # sidestep loop       -> AICUP 1 拉球
    "控制":   None,  # generic 'control' -> ambiguous (could be 8/9/10/11)
    "拧":     4,   # twist / wrist flick -> AICUP 4 擰球
    "劈长":   10,  # long push           -> AICUP 10 搓球 (closest)
    "转不转": 15,  # spin/no-spin (serve)-> AICUP 15 傳統 (variant)
    "逆旋转": 17,  # reverse sidespin    -> AICUP 17 逆旋轉
    "挑":     7,   # flick               -> AICUP 7 挑撥
    "勾球":   16,  # hook (serve)        -> AICUP 16 勾手
    "普通":   15,  # traditional         -> AICUP 15 傳統
    "下蹲":   18,  # squat (serve)       -> AICUP 18 下蹲式
    "中性":   None,  # neutral           -> ambiguous
    "":       None,  # empty             -> noise
}


def load_p2a_videos(label_json_path: str | Path) -> list[dict]:
    """Load the P2A JSON file and return the list of video dicts.

    Supports both observed P2A label schemas:
      - **list** at top level (``v1_renamed.json``, ``v2_renamed.json``):
        each item has ``url``, ``total_frames``, ``actions``.
      - **dict** at top level (``v1.json``, ``v2.json``):
        ``{"fps": <int>, "gts": [<video dict>, ...]}``.

    The audit (2026-05-19) documented only the list form. The dict-wrapped
    form was discovered on 2026-05-23 in the actual dataset; both are now
    handled.

    Returns the raw video list. No filtering or mapping is applied here.
    """
    path = Path(label_json_path)
    if not path.exists():
        raise FileNotFoundError(f"P2A label JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "gts" in data and isinstance(data["gts"], list):
            return data["gts"]
        raise ValueError(
            f"Top-level dict at {path} has no 'gts' list key. "
            f"Keys: {list(data.keys())}"
        )
    if isinstance(data, list):
        return data
    raise ValueError(
        f"Expected list or dict-with-'gts' at top-level of {path}, "
        f"got {type(data).__name__}"
    )


def pick_pilot_videos(
    videos: list[dict],
    n: int = 10,
    seed: int = 42,
    min_strokes: int = 10,
    max_strokes: int = 40,
) -> list[dict]:
    """Deterministically pick N pilot videos.

    Filters to videos whose stroke count is in [min_strokes, max_strokes] —
    we want enough strokes per video for the rally-segmentation task to be
    meaningful, but not so many that the video is too long for the web Gemini
    UI to upload comfortably.

    Returns a list of video dicts (subset of input).

    **For match-aware picking (one chunk per match for diversity)**, use
    ``pick_pilot_videos_match_aware`` instead.
    """
    rng = random.Random(seed)
    eligible = [
        v for v in videos
        if min_strokes <= len(v.get("actions", [])) <= max_strokes
    ]
    if len(eligible) < n:
        raise ValueError(
            f"Only {len(eligible)} videos meet the stroke-count filter "
            f"[{min_strokes}, {max_strokes}]; cannot pick {n} pilot videos."
        )
    return rng.sample(eligible, n)


def pick_pilot_videos_match_aware(
    videos: list[dict],
    renamed_to_match: dict[str, dict],
    n: int = 10,
    seed: int = 42,
    min_strokes: int = 10,
    max_strokes: int = 40,
    chunk_strategy: str = "smallest_index",
) -> list[dict]:
    """Pick N pilot videos with at most ONE chunk per match (max diversity).

    Parameters
    ----------
    videos : list[dict]
        Output of ``load_p2a_videos``.
    renamed_to_match : dict
        Output of ``group_matches.renamed_id_to_match_lookup``. Maps each
        renamed_id (e.g. ``"0000244"``) to its match metadata.
    n : int
        Number of pilot videos (= number of matches sampled).
    seed : int
        RNG seed for deterministic match selection.
    min_strokes, max_strokes : int
        Stroke-count filter applied per chunk, same as ``pick_pilot_videos``.
    chunk_strategy : str
        Which chunk to pick from each sampled match. One of:
          - ``"smallest_index"``  : pick the eligible chunk with the lowest
            ``chunk_idx`` (often the first portion of the match, usually
            cleanest pacing).
          - ``"middle"``           : pick the eligible chunk closest to
            ``ceil(n_chunks/2)``.
          - ``"random"``           : pick a uniformly random eligible chunk.

    Returns
    -------
    list[dict]
        Same shape as ``pick_pilot_videos``. Each item is a video dict from
        ``videos``. Guaranteed: every item belongs to a distinct match.
    """
    if chunk_strategy not in {"smallest_index", "middle", "random"}:
        raise ValueError(f"Unknown chunk_strategy: {chunk_strategy!r}")

    rng = random.Random(seed)

    # Index videos by renamed_id for fast match lookup
    by_id: dict[str, dict] = {}
    for v in videos:
        url = v.get("url", "")
        rid = url.rsplit(".", 1)[0] if "." in url else url
        by_id[rid] = v

    # Group eligible videos by match_id
    by_match: dict[str, list[tuple[str, dict]]] = {}
    for rid, vid in by_id.items():
        meta = renamed_to_match.get(rid)
        if meta is None:
            continue  # unmatched (no -N-of-M pattern) — skip for pilot
        n_strokes = len(vid.get("actions", []))
        if not (min_strokes <= n_strokes <= max_strokes):
            continue
        by_match.setdefault(meta["match_id"], []).append((rid, vid))

    if len(by_match) < n:
        raise ValueError(
            f"Only {len(by_match)} matches have an eligible chunk in "
            f"[{min_strokes}, {max_strokes}] strokes; cannot pick {n} matches."
        )

    chosen_match_ids = rng.sample(sorted(by_match.keys()), n)
    picked: list[dict] = []
    for mid in chosen_match_ids:
        chunks = by_match[mid]
        # chunks is list[(renamed_id, video_dict)]
        if chunk_strategy == "smallest_index":
            chunks_sorted = sorted(
                chunks,
                key=lambda rv: renamed_to_match[rv[0]]["chunk_idx"],
            )
            picked.append(chunks_sorted[0][1])
        elif chunk_strategy == "middle":
            chunks_sorted = sorted(
                chunks,
                key=lambda rv: renamed_to_match[rv[0]]["chunk_idx"],
            )
            picked.append(chunks_sorted[len(chunks_sorted) // 2][1])
        else:  # random
            picked.append(rng.choice(chunks)[1])
    return picked


def video_anchor_table(video: dict) -> dict:
    """Build the stroke-anchor table for one P2A video.

    Returns a dict suitable for feeding to ``build_prompt.build_prompt``::

        {
          "video_id": "0000123",     # url without extension
          "url": "0000123.mp4",
          "strokes": [
            {
              "stroke_idx": 0,
              "start": 106.34,
              "end": 106.82,
              "hand_raw": "正手",
              "hand_aicup": 1,
              "is_serve": False,
              "action_raw": "拉",
              "action_aicup": 1,
              "action_aicup_known": True,
            },
            ...
          ]
        }
    """
    url = video.get("url", "")
    video_id = url.rsplit(".", 1)[0] if "." in url else url
    strokes = []
    for i, action in enumerate(video.get("actions", [])):
        names = action.get("label_names", [])
        if not isinstance(names, list) or len(names) < 3:
            # Malformed entry — keep stroke_idx alignment by emitting a stub
            strokes.append({
                "stroke_idx": i,
                "start": action.get("start_id"),
                "end": action.get("end_id"),
                "hand_raw": None,
                "hand_aicup": None,
                "is_serve": None,
                "action_raw": None,
                "action_aicup": None,
                "action_aicup_known": False,
                "malformed": True,
            })
            continue
        hand_raw, is_serve_raw, action_raw = names[0], names[1], names[2]
        action_aicup = ACTION_MAP.get(action_raw)
        strokes.append({
            "stroke_idx": i,
            "start": action.get("start_id"),
            "end": action.get("end_id"),
            "hand_raw": hand_raw,
            "hand_aicup": HAND_MAP.get(hand_raw),
            "is_serve": IS_SERVE_MAP.get(is_serve_raw),
            "action_raw": action_raw,
            "action_aicup": action_aicup,
            "action_aicup_known": action_aicup is not None,
        })
    return {
        "video_id": video_id,
        "url": url,
        "n_strokes": len(strokes),
        "strokes": strokes,
    }


def pre_segment_rallies(
    anchor_table: dict,
    rally_gap_seconds: float = 4.0,
) -> list[dict]:
    """Pre-segment strokes into candidate rallies using a time-gap heuristic.

    Two consecutive strokes belong to the same rally iff their inter-stroke
    gap (gap = next_start - this_end) is < ``rally_gap_seconds``.

    Returns a list of rally dicts::

        [
          {"rally_id": 0, "stroke_indices": [0, 1, 2, 3]},
          {"rally_id": 1, "stroke_indices": [4, 5]},
          ...
        ]

    Gemini is asked to VERIFY/ADJUST these boundaries in the response, so this
    is just a sensible default to anchor the prompt on.
    """
    strokes = anchor_table.get("strokes", [])
    if not strokes:
        return []
    rallies: list[dict] = []
    current_indices: list[int] = [strokes[0]["stroke_idx"]]
    for prev, curr in zip(strokes[:-1], strokes[1:]):
        prev_end = prev.get("end")
        curr_start = curr.get("start")
        gap = None
        if prev_end is not None and curr_start is not None:
            gap = curr_start - prev_end
        if gap is None or gap >= rally_gap_seconds:
            rallies.append({"rally_id": len(rallies), "stroke_indices": current_indices})
            current_indices = []
        current_indices.append(curr["stroke_idx"])
    rallies.append({"rally_id": len(rallies), "stroke_indices": current_indices})
    return rallies
