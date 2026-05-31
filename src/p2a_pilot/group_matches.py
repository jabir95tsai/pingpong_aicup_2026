"""Group P2A videos by source match.

P2A videos are chunked recordings of full ITTF matches. Filenames follow::

    <match_name>-<chunk_idx>-of-<total_chunks>.mp4

For example, the 2019 World Cup final between Fan Zhendong and Harimoto is
14 chunks named ``2019男子世界杯决赛及颁奖樊振东4-2张本智和-{1..14}-of-14.mp4``,
which proj.json maps to renamed IDs 0000017–0000030 (not always sequential).

This module:
  - Loads proj.json (original_filename -> renamed_filename per split)
  - Parses the chunk-naming pattern to extract (match_name, chunk_idx, total_chunks)
  - Groups all chunks belonging to the same match
  - Flags videos that DON'T match the chunk pattern (single-clip videos, etc.)
  - Flags potential duplicate matches with near-identical names (e.g. typo
    variants like ``樊振东4-2张本智和`` vs ``樊振东42张本智和``)

Output schema (the JSON written by ``build_match_groups``):

    {
      "split": "v1",
      "n_matches": 253,
      "n_videos": 1281,
      "n_unmatched": 4,
      "matches": [
        {
          "match_id": "v1_m000",
          "match_name": "2019男子世界杯铜牌战马龙vs林昀儒",
          "n_chunks_expected": 12,
          "n_chunks_found": 11,
          "chunks_missing": [1],
          "chunks": [
            {"chunk_idx": 2, "renamed_id": "0000007", "original_filename": "..."},
            ...
          ]
        }
      ],
      "unmatched": [
        {"renamed_id": "...", "original_filename": "...", "reason": "no -N-of-M pattern"}
      ],
      "near_duplicate_matches": [
        {
          "group": ["match_id_1", "match_id_2"],
          "names": ["...4-2...", "...42..."],
          "similarity_hint": "differ only in '-' separators"
        }
      ]
    }
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

# ``<match_name>[-]<chunk_idx>-of-<total_chunks>.mp4``
# The separator before chunk_idx is OPTIONAL (handles both
# ``match-9-of-11.mp4`` and ``match9-of-11.mp4`` — the latter pattern is used
# by some 2019 Chengdu World Cup clips in P2A v1).
# Non-greedy prefix + ``-?`` lets the regex find the first viable split when
# match_name itself ends in a hyphen or digit (e.g. ``4-2`` scores).
CHUNK_PATTERN = re.compile(r"^(.+?)-?(\d+)-of-(\d+)\.mp4$")

NEAR_DUP_SIMILARITY_THRESHOLD = 0.92


def load_proj_json(proj_json_path: str | Path) -> dict[str, dict[str, str]]:
    """Load proj.json. Returns ``{split: {original_filename: renamed_filename}}``.

    Splits are typically ``"v1"`` and ``"v2"``.
    """
    path = Path(proj_json_path)
    if not path.exists():
        raise FileNotFoundError(f"proj.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at top of {path}, got {type(data).__name__}")
    for split, mapping in data.items():
        if not isinstance(mapping, dict):
            raise ValueError(f"proj['{split}'] must be a dict, got {type(mapping).__name__}")
    return data


def parse_chunk_filename(filename: str) -> tuple[str, int, int] | None:
    """Parse ``<match_name>-<chunk>-of-<total>.mp4``.

    Returns (match_name, chunk_idx, total_chunks) or None if the pattern
    doesn't match.
    """
    m = CHUNK_PATTERN.match(filename)
    if not m:
        return None
    match_name, chunk_idx, total_chunks = m.group(1), int(m.group(2)), int(m.group(3))
    return match_name, chunk_idx, total_chunks


def _find_near_duplicates(matches: list[dict]) -> list[dict]:
    """Heuristic detection of near-duplicate match groups.

    Each output dict carries a ``verdict`` so the user knows what action to
    take:

      - ``LIKELY_SAME_MATCH``: names are identical after stripping all
        ``-``, ``_``, whitespace. Almost always a typo variant of the same
        physical match. Safe to auto-merge.
      - ``MANUAL_REVIEW``: high string similarity but separator-stripped
        forms differ. Often a false positive (different games of the same
        team match, sequel matches between same players, etc.). Do NOT
        auto-merge; flag for human inspection.
    """
    near_dups: list[dict] = []
    n = len(matches)
    for i in range(n):
        for j in range(i + 1, n):
            a = matches[i]["match_name"]
            b = matches[j]["match_name"]
            if a == b:
                continue
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio >= NEAR_DUP_SIMILARITY_THRESHOLD:
                a_strip = re.sub(r"[-_\s]", "", a)
                b_strip = re.sub(r"[-_\s]", "", b)
                if a_strip == b_strip:
                    verdict = "LIKELY_SAME_MATCH"
                    hint = "differ only in separators (e.g. 4-2 vs 42)"
                else:
                    verdict = "MANUAL_REVIEW"
                    hint = "near-identical names but content differs — likely distinct matches"
                near_dups.append({
                    "group": [matches[i]["match_id"], matches[j]["match_id"]],
                    "names": [a, b],
                    "similarity": round(ratio, 4),
                    "similarity_hint": hint,
                    "verdict": verdict,
                })
    return near_dups


def build_match_groups(
    proj_data: dict[str, dict[str, str]],
    split: str,
) -> dict:
    """Group one split's videos by source match.

    Parameters
    ----------
    proj_data : dict
        Loaded proj.json (output of ``load_proj_json``).
    split : str
        Which split to group (``"v1"`` or ``"v2"``).

    Returns
    -------
    dict
        See module docstring for schema.
    """
    if split not in proj_data:
        raise KeyError(f"split={split!r} not in proj.json (have: {list(proj_data.keys())})")

    mapping = proj_data[split]
    # Group by match_name; collect (original_filename, renamed_id, chunk_idx, total) tuples
    by_match: dict[str, list[dict]] = defaultdict(list)
    unmatched: list[dict] = []
    for original_filename, renamed_filename in mapping.items():
        renamed_id = renamed_filename.rsplit(".", 1)[0]
        parsed = parse_chunk_filename(original_filename)
        if parsed is None:
            unmatched.append({
                "renamed_id": renamed_id,
                "original_filename": original_filename,
                "reason": "no -N-of-M.mp4 pattern",
            })
            continue
        match_name, chunk_idx, total_chunks = parsed
        by_match[match_name].append({
            "chunk_idx": chunk_idx,
            "total_chunks": total_chunks,
            "renamed_id": renamed_id,
            "original_filename": original_filename,
        })

    # Build the match list with consistent IDs + diagnostics
    matches: list[dict] = []
    for i, (match_name, chunks) in enumerate(sorted(by_match.items())):
        # Sort chunks by chunk_idx
        chunks_sorted = sorted(chunks, key=lambda c: c["chunk_idx"])
        # All chunks should agree on total_chunks; if not, flag
        totals = {c["total_chunks"] for c in chunks_sorted}
        if len(totals) != 1:
            n_expected = max(totals)
            total_mismatch = sorted(totals)
        else:
            n_expected = chunks_sorted[0]["total_chunks"]
            total_mismatch = None
        present_idxs = {c["chunk_idx"] for c in chunks_sorted}
        missing = sorted(set(range(1, n_expected + 1)) - present_idxs)
        matches.append({
            "match_id": f"{split}_m{i:03d}",
            "match_name": match_name,
            "n_chunks_expected": n_expected,
            "n_chunks_found": len(chunks_sorted),
            "chunks_missing": missing,
            "chunks": [
                {
                    "chunk_idx": c["chunk_idx"],
                    "renamed_id": c["renamed_id"],
                    "original_filename": c["original_filename"],
                }
                for c in chunks_sorted
            ],
            **({"total_mismatch": total_mismatch} if total_mismatch else {}),
        })

    near_dups = _find_near_duplicates(matches)

    return {
        "split": split,
        "n_matches": len(matches),
        "n_videos": sum(m["n_chunks_found"] for m in matches),
        "n_unmatched": len(unmatched),
        "matches": matches,
        "unmatched": unmatched,
        "near_duplicate_matches": near_dups,
    }


def renamed_id_to_match_lookup(
    groups: dict,
    splits: list[str] | None = None,
) -> dict[str, dict]:
    """Build a fast lookup: renamed_id -> {match_id, match_name, chunk_idx, split, ...}.

    Used to enrich the pilot manifest with match context.

    Accepts both shapes:
      - **Per-split groups** (output of ``build_match_groups``): has a
        top-level ``"matches"`` key. Single split, no collision risk.
      - **Combined multi-split groups** (output of the ``group-matches``
        CLI command): has a top-level ``"splits": {split_name: per_split_dict}``.

    **CRITICAL — split namespace collision** (2026-05-25 Codex finding):
    v1 and v2 in the P2A dataset BOTH use renamed IDs ``0000000..0001207``
    pointing to DIFFERENT physical .mp4 files. Naively merging both splits
    into one dict silently overwrites v1 entries with v2 entries.

    Defenses:
      1. ``splits`` parameter filters to a specified subset of splits.
         When None on a multi-split input, ALL splits are included AND
         collisions raise ``ValueError`` (forces caller to disambiguate).
      2. Every returned value carries a ``"split"`` key so the caller can
         always tell which split a video came from.

    Parameters
    ----------
    groups : dict
        Either per-split or combined-multi-split as documented above.
    splits : list[str] | None
        Optional whitelist of splits to include. If None on a combined
        input, includes all splits AND raises on cross-split renamed_id
        collisions.
    """
    # Combined multi-split shape — iterate selected splits with collision guard
    if "splits" in groups and "matches" not in groups:
        out: dict[str, dict] = {}
        all_splits = list(groups["splits"].keys())
        included = all_splits if splits is None else list(splits)
        for sn in included:
            if sn not in groups["splits"]:
                raise KeyError(
                    f"split={sn!r} requested but not in combined groups "
                    f"(have: {all_splits})"
                )
        for sn in included:
            sub = renamed_id_to_match_lookup(groups["splits"][sn], splits=None)
            # sub entries already carry "split"
            collisions = sorted(set(sub.keys()) & set(out.keys()))
            if collisions:
                # When the caller explicitly asked for multiple splits but
                # they collide, error loudly — there's no safe merge.
                example = collisions[0]
                raise ValueError(
                    f"renamed_id collision across splits when including "
                    f"{included}: {len(collisions)} ids appear in multiple "
                    f"splits (e.g. {example!r} is in both "
                    f"{out[example]['split']!r} and {sub[example]['split']!r}). "
                    f"Pass `splits=['{included[0]}']` (or one split) to "
                    f"restrict the lookup, or build per-split lookups "
                    f"separately."
                )
            out.update(sub)
        return out

    # Per-split shape
    if "matches" not in groups:
        raise ValueError(
            "Expected 'matches' key (per-split groups) or 'splits' key "
            "(combined groups) in input dict."
        )
    split_name = groups.get("split", "unknown")
    out = {}
    for m in groups["matches"]:
        for c in m["chunks"]:
            out[c["renamed_id"]] = {
                "match_id": m["match_id"],
                "match_name": m["match_name"],
                "chunk_idx": c["chunk_idx"],
                "n_chunks_in_match": m["n_chunks_expected"],
                "original_filename": c["original_filename"],
                "split": split_name,
            }
    return out


def summarize_groups(groups: dict) -> str:
    """Human-readable summary."""
    lines = []
    lines.append(f"Split: {groups['split']}")
    lines.append(f"  Matches:   {groups['n_matches']}")
    lines.append(f"  Videos:    {groups['n_videos']}")
    lines.append(f"  Unmatched: {groups['n_unmatched']}")
    # Distribution of chunks-per-match
    chunk_counts = [m["n_chunks_found"] for m in groups["matches"]]
    if chunk_counts:
        chunk_counts.sort()
        median = chunk_counts[len(chunk_counts) // 2]
        lines.append(
            f"  Chunks per match: min={min(chunk_counts)} "
            f"median={median} max={max(chunk_counts)}"
        )
    # Matches with missing chunks
    matches_with_missing = [m for m in groups["matches"] if m["chunks_missing"]]
    if matches_with_missing:
        lines.append(f"  Matches with missing chunks: {len(matches_with_missing)}")
    # Near-duplicate clusters
    if groups["near_duplicate_matches"]:
        lines.append(f"  Near-duplicate match pairs: {len(groups['near_duplicate_matches'])}")
    return "\n".join(lines)
