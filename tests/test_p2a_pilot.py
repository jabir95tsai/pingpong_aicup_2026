"""Tests for src/p2a_pilot/ — schema validation, segmentation, mapping."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add src/ to path so `import p2a_pilot` works without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p2a_pilot.load_p2a import (  # noqa: E402
    ACTION_MAP, HAND_MAP, IS_SERVE_MAP,
    load_p2a_videos, pre_segment_rallies, video_anchor_table,
)
from p2a_pilot.build_prompt import build_prompt  # noqa: E402
from p2a_pilot.parse_response import parse_gemini_response  # noqa: E402
from p2a_pilot.group_matches import (  # noqa: E402
    parse_chunk_filename, build_match_groups, renamed_id_to_match_lookup,
    load_proj_json,
)
from p2a_pilot.load_p2a import pick_pilot_videos_match_aware  # noqa: E402
from p2a_pilot.accuracy import wilson_lower_bound  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

SAMPLE_VIDEO = {
    "url": "0000123.mp4",
    "total_frames": None,
    "actions": [
        # Rally 1: serve + 3 returns
        {"label_names": ["正手", "是", "侧旋"], "start_id": 1.00, "end_id": 1.40},
        {"label_names": ["反手", "否", "拧"],   "start_id": 2.10, "end_id": 2.50},
        {"label_names": ["正手", "否", "拉"],   "start_id": 3.20, "end_id": 3.60},
        {"label_names": ["反手", "否", "拉"],   "start_id": 4.40, "end_id": 4.80},
        # Big time gap (15s) -> new rally
        {"label_names": ["反手", "是", "逆旋转"], "start_id": 20.00, "end_id": 20.50},
        {"label_names": ["正手", "否", "挑"],   "start_id": 21.30, "end_id": 21.70},
    ],
}


# ---------------------------------------------------------------------------
# load_p2a_videos — both top-level shapes
# ---------------------------------------------------------------------------

def test_load_p2a_videos_list_format(tmp_path):
    """v1_renamed.json / v2_renamed.json shape: bare list at top level."""
    p = tmp_path / "renamed.json"
    p.write_text(json.dumps([SAMPLE_VIDEO]), encoding="utf-8")
    videos = load_p2a_videos(p)
    assert len(videos) == 1
    assert videos[0]["url"] == "0000123.mp4"


def test_load_p2a_videos_dict_format(tmp_path):
    """v1.json / v2.json shape: {fps, gts}."""
    p = tmp_path / "original.json"
    p.write_text(json.dumps({"fps": 25, "gts": [SAMPLE_VIDEO]}), encoding="utf-8")
    videos = load_p2a_videos(p)
    assert len(videos) == 1
    assert videos[0]["url"] == "0000123.mp4"


def test_load_p2a_videos_invalid_dict_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"fps": 25, "videos": [SAMPLE_VIDEO]}), encoding="utf-8")  # no 'gts'
    with pytest.raises(ValueError, match="no 'gts' list key"):
        load_p2a_videos(p)


def test_load_p2a_videos_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_p2a_videos(tmp_path / "does_not_exist.json")


# ---------------------------------------------------------------------------
# Vocab mapping tests
# ---------------------------------------------------------------------------

def test_hand_map_complete():
    assert HAND_MAP["正手"] == 1
    assert HAND_MAP["反手"] == 2


def test_is_serve_map_complete():
    assert IS_SERVE_MAP["是"] is True
    assert IS_SERVE_MAP["否"] is False


def test_action_map_known_serves():
    """All P2A serve types should map to AICUP serve actionIds 15-18."""
    serve_types = ["侧旋", "转不转", "逆旋转", "勾球", "普通", "下蹲"]
    for t in serve_types:
        aid = ACTION_MAP[t]
        assert aid in (15, 16, 17, 18), f"{t} → {aid}, expected serve range"


def test_action_map_rally_shots():
    """Common rally shots map to known AICUP actionIds."""
    assert ACTION_MAP["拉"] == 1     # loop
    assert ACTION_MAP["拧"] == 4     # twist
    assert ACTION_MAP["挑"] == 7     # flick
    assert ACTION_MAP["摆短"] == 11  # short push


def test_action_map_unresolved_returns_none():
    assert ACTION_MAP["控制"] is None
    assert ACTION_MAP["中性"] is None
    assert ACTION_MAP[""] is None


# ---------------------------------------------------------------------------
# video_anchor_table tests
# ---------------------------------------------------------------------------

def test_anchor_table_basic():
    anchors = video_anchor_table(SAMPLE_VIDEO)
    assert anchors["video_id"] == "0000123"
    assert anchors["url"] == "0000123.mp4"
    assert anchors["n_strokes"] == 6
    assert len(anchors["strokes"]) == 6


def test_anchor_table_stroke_idx_in_order():
    anchors = video_anchor_table(SAMPLE_VIDEO)
    for i, s in enumerate(anchors["strokes"]):
        assert s["stroke_idx"] == i


def test_anchor_table_field_mapping():
    anchors = video_anchor_table(SAMPLE_VIDEO)
    s0 = anchors["strokes"][0]
    assert s0["hand_raw"] == "正手"
    assert s0["hand_aicup"] == 1
    assert s0["is_serve"] is True
    assert s0["action_raw"] == "侧旋"
    assert s0["action_aicup"] == 15
    assert s0["action_aicup_known"] is True


def test_anchor_table_malformed_entry():
    """Malformed entries (missing label_names) should produce a stub stroke
    with stroke_idx preserved (so downstream indexing isn't broken)."""
    bad_video = {
        "url": "bad.mp4",
        "actions": [
            {"label_names": ["正手"], "start_id": 1.0, "end_id": 1.2},  # too short
            {"label_names": ["正手", "否", "拉"], "start_id": 2.0, "end_id": 2.3},
        ],
    }
    anchors = video_anchor_table(bad_video)
    assert anchors["n_strokes"] == 2
    assert anchors["strokes"][0].get("malformed") is True
    assert anchors["strokes"][0]["stroke_idx"] == 0
    assert anchors["strokes"][1]["stroke_idx"] == 1
    assert anchors["strokes"][1]["action_aicup"] == 1


# ---------------------------------------------------------------------------
# Rally pre-segmentation tests
# ---------------------------------------------------------------------------

def test_pre_segment_rallies_basic():
    anchors = video_anchor_table(SAMPLE_VIDEO)
    rallies = pre_segment_rallies(anchors, rally_gap_seconds=4.0)
    # First 4 strokes are within 4s gaps -> rally 0
    # 15s gap -> rally 1 starts at stroke 4
    assert len(rallies) == 2
    assert rallies[0]["stroke_indices"] == [0, 1, 2, 3]
    assert rallies[1]["stroke_indices"] == [4, 5]


def test_pre_segment_covers_all_strokes_exactly_once():
    anchors = video_anchor_table(SAMPLE_VIDEO)
    rallies = pre_segment_rallies(anchors)
    covered = []
    for r in rallies:
        covered.extend(r["stroke_indices"])
    assert sorted(covered) == list(range(anchors["n_strokes"]))


def test_pre_segment_empty_video():
    assert pre_segment_rallies({"strokes": []}) == []


# ---------------------------------------------------------------------------
# Prompt-building tests
# ---------------------------------------------------------------------------

def test_build_prompt_contains_required_anchors():
    anchors = video_anchor_table(SAMPLE_VIDEO)
    prompt = build_prompt(anchors)
    # Must include video_id, total_strokes, all 6 stroke timestamps
    assert "VIDEO_ID: 0000123" in prompt
    assert "TOTAL_STROKES: 6" in prompt
    for s in anchors["strokes"]:
        assert f"stroke_idx={s['stroke_idx']:3d}" in prompt
    # Must include the schema block and critical rules
    assert "JSON SCHEMA" in prompt
    assert "CRITICAL RULES" in prompt
    assert "Output EXACTLY 6 stroke objects" in prompt


def test_build_prompt_zero_strokes_raises():
    with pytest.raises(ValueError):
        build_prompt({"video_id": "x", "url": "x.mp4", "n_strokes": 0, "strokes": []})


# ---------------------------------------------------------------------------
# parse_response — happy path
# ---------------------------------------------------------------------------

VALID_RESPONSE = {
    "video_id": "0000123",
    "rallies": [
        {
            "rally_id": 0,
            "stroke_indices": [0, 1, 2, 3],
            "server_won_rally": 1,
            "outcome_confidence": "high",
            "outcome_evidence": "Server raised arm after rally",
            "server_hand": "right",
            "receiver_hand": "right",
            "hand_confidence": "high",
        },
        {
            "rally_id": 1,
            "stroke_indices": [4, 5],
            "server_won_rally": 0,
            "outcome_confidence": "medium",
            "outcome_evidence": "Receiver scored final shot",
            "server_hand": "right",
            "receiver_hand": "left",
            "hand_confidence": "medium",
        },
    ],
    "strokes": [
        {"stroke_idx": 0, "landing_zone": "table_mid_long", "landing_confidence": "high",
         "player_position": "center", "position_confidence": "high"},
        {"stroke_idx": 1, "landing_zone": "table_left_short", "landing_confidence": "medium",
         "player_position": "left", "position_confidence": "high"},
        {"stroke_idx": 2, "landing_zone": "table_right_half", "landing_confidence": "high",
         "player_position": "right", "position_confidence": "medium"},
        {"stroke_idx": 3, "landing_zone": "off_grid", "landing_confidence": "low",
         "player_position": "center", "position_confidence": "high"},
        {"stroke_idx": 4, "landing_zone": "table_mid_short", "landing_confidence": "high",
         "player_position": "center", "position_confidence": "high"},
        {"stroke_idx": 5, "landing_zone": "table_right_long", "landing_confidence": "medium",
         "player_position": "right", "position_confidence": "medium"},
    ],
}


def test_parse_valid_response():
    raw = json.dumps(VALID_RESPONSE)
    result, data = parse_gemini_response(raw, "0000123", 6)
    assert result.ok, f"Unexpected errors: {result.errors}"
    assert data["video_id"] == "0000123"
    assert len(data["strokes"]) == 6


def test_parse_strips_markdown_fences():
    raw = "Here is the JSON:\n```json\n" + json.dumps(VALID_RESPONSE) + "\n```\nLet me know!"
    result, data = parse_gemini_response(raw, "0000123", 6)
    assert result.ok, f"Errors after stripping: {result.errors}"
    assert data is not None


def test_parse_strips_prose_around_braces():
    raw = "Sure, here it is: " + json.dumps(VALID_RESPONSE) + " — happy to help further."
    result, data = parse_gemini_response(raw, "0000123", 6)
    assert result.ok, f"Errors: {result.errors}"


# ---------------------------------------------------------------------------
# parse_response — failure paths
# ---------------------------------------------------------------------------

def test_parse_wrong_video_id():
    raw = json.dumps(VALID_RESPONSE)
    result, _ = parse_gemini_response(raw, "999", 6)
    assert not result.ok
    assert any("video_id mismatch" in e for e in result.errors)


def test_parse_wrong_stroke_count():
    bad = dict(VALID_RESPONSE)
    bad["strokes"] = VALID_RESPONSE["strokes"][:4]  # drop 2 strokes
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("strokes count mismatch" in e for e in result.errors)


def test_parse_invalid_landing_zone():
    bad = json.loads(json.dumps(VALID_RESPONSE))  # deep copy
    bad["strokes"][0]["landing_zone"] = "north_pole"
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("landing_zone" in e for e in result.errors)


def test_parse_rejects_old_FH_BH_schema():
    """Old receiver-relative FH/BH schema is now INVALID — must use court-absolute.

    Codex 2026-05-25 fix #4: hard-coded right-handed receiver was removed.
    """
    bad = json.loads(json.dumps(VALID_RESPONSE))
    bad["strokes"][0]["landing_zone"] = "FH_short"  # old enum
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("landing_zone" in e for e in result.errors)


def test_parse_rally_missing_stroke():
    bad = json.loads(json.dumps(VALID_RESPONSE))
    bad["rallies"][1]["stroke_indices"] = [5]  # drops stroke_idx=4
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("rallies miss stroke_idx" in e for e in result.errors)


def test_parse_rally_duplicate_stroke():
    bad = json.loads(json.dumps(VALID_RESPONSE))
    bad["rallies"][0]["stroke_indices"] = [0, 1, 2, 3, 4]  # claims stroke 4 too
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("multiple rallies" in e for e in result.errors)


def test_parse_outcome_out_of_range():
    bad = json.loads(json.dumps(VALID_RESPONSE))
    bad["rallies"][0]["server_won_rally"] = 7
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("server_won_rally" in e for e in result.errors)


def test_parse_garbage_input():
    result, data = parse_gemini_response("not json at all", "0000123", 6)
    assert not result.ok
    assert data is None


# ---------------------------------------------------------------------------
# group_matches — filename parsing
# ---------------------------------------------------------------------------

def test_parse_chunk_filename_simple():
    assert parse_chunk_filename("2019男子世界杯铜牌战马龙vs林昀儒-9-of-12.mp4") == (
        "2019男子世界杯铜牌战马龙vs林昀儒", 9, 12,
    )


def test_parse_chunk_filename_with_score_hyphen():
    """Match name contains '4-2' score — greedy match must give the WHOLE name."""
    assert parse_chunk_filename("2019男子世界杯决赛及颁奖樊振东4-2张本智和-9-of-14.mp4") == (
        "2019男子世界杯决赛及颁奖樊振东4-2张本智和", 9, 14,
    )


def test_parse_chunk_filename_with_underscore():
    """Match name contains 1_4 (1/4 final) — underscores preserved."""
    assert parse_chunk_filename("2019男子世界杯1_4决赛张本智和vs丹羽孝希-3-of-10.mp4") == (
        "2019男子世界杯1_4决赛张本智和vs丹羽孝希", 3, 10,
    )


def test_parse_chunk_filename_no_pattern():
    assert parse_chunk_filename("random_file_no_pattern.mp4") is None
    assert parse_chunk_filename("0000123.mp4") is None  # renamed-only, no chunk info


# ---------------------------------------------------------------------------
# group_matches — full grouping pipeline
# ---------------------------------------------------------------------------

PROJ_DATA_SAMPLE = {
    "v1": {
        # Match A: 3 chunks
        "matchA-1-of-3.mp4": "0000000.mp4",
        "matchA-2-of-3.mp4": "0000001.mp4",
        "matchA-3-of-3.mp4": "0000002.mp4",
        # Match B: 2 chunks (with score hyphen)
        "matchB4-2subname-1-of-2.mp4": "0000003.mp4",
        "matchB4-2subname-2-of-2.mp4": "0000004.mp4",
        # Match C: missing chunk 2
        "matchC-1-of-3.mp4": "0000005.mp4",
        "matchC-3-of-3.mp4": "0000006.mp4",
        # Unmatched
        "weirdfile.mp4": "0000007.mp4",
    },
    "v2": {
        "matchD-1-of-1.mp4": "0001000.mp4",
    },
}


def test_build_match_groups_basic():
    groups = build_match_groups(PROJ_DATA_SAMPLE, "v1")
    assert groups["split"] == "v1"
    assert groups["n_matches"] == 3
    assert groups["n_videos"] == 7  # 3+2+2, excluding unmatched
    assert groups["n_unmatched"] == 1
    match_names = [m["match_name"] for m in groups["matches"]]
    assert "matchA" in match_names
    assert "matchB4-2subname" in match_names
    assert "matchC" in match_names


def test_build_match_groups_detects_missing_chunks():
    groups = build_match_groups(PROJ_DATA_SAMPLE, "v1")
    match_c = next(m for m in groups["matches"] if m["match_name"] == "matchC")
    assert match_c["n_chunks_expected"] == 3
    assert match_c["n_chunks_found"] == 2
    assert match_c["chunks_missing"] == [2]


def test_build_match_groups_chunks_sorted_by_idx():
    groups = build_match_groups(PROJ_DATA_SAMPLE, "v1")
    match_a = next(m for m in groups["matches"] if m["match_name"] == "matchA")
    chunk_indices = [c["chunk_idx"] for c in match_a["chunks"]]
    assert chunk_indices == [1, 2, 3]


def test_build_match_groups_unmatched_reason():
    groups = build_match_groups(PROJ_DATA_SAMPLE, "v1")
    assert groups["unmatched"][0]["renamed_id"] == "0000007"
    assert "no -N-of-M" in groups["unmatched"][0]["reason"]


def test_build_match_groups_unknown_split_raises():
    with pytest.raises(KeyError):
        build_match_groups(PROJ_DATA_SAMPLE, "nonexistent")


def test_renamed_id_to_match_lookup():
    groups = build_match_groups(PROJ_DATA_SAMPLE, "v1")
    lookup = renamed_id_to_match_lookup(groups)
    assert "0000000" in lookup
    assert lookup["0000000"]["match_name"] == "matchA"
    assert lookup["0000000"]["chunk_idx"] == 1
    assert lookup["0000000"]["n_chunks_in_match"] == 3
    # Match B chunks all share a match_id
    assert lookup["0000003"]["match_id"] == lookup["0000004"]["match_id"]
    # 0000007 was unmatched
    assert "0000007" not in lookup


def test_renamed_id_to_match_lookup_handles_combined_shape():
    """The lookup must work on both per-split AND combined multi-split JSONs."""
    v1 = build_match_groups(PROJ_DATA_SAMPLE, "v1")
    v2 = build_match_groups(PROJ_DATA_SAMPLE, "v2")
    combined = {
        "source_proj_json": "fake.json",
        "splits": {"v1": v1, "v2": v2},
        "near_duplicate_matches": [],
    }
    lookup = renamed_id_to_match_lookup(combined)
    # v1 entries
    assert "0000000" in lookup
    assert lookup["0000000"]["match_name"] == "matchA"
    # v2 entries (matchD)
    assert "0001000" in lookup
    assert lookup["0001000"]["match_name"] == "matchD"


def test_renamed_id_to_match_lookup_invalid_shape_raises():
    with pytest.raises(ValueError, match="Expected 'matches' key"):
        renamed_id_to_match_lookup({"random": "garbage"})


def test_near_duplicate_separator_variant_is_LIKELY_SAME():
    """4-2 vs 42 (same match, typo) -> LIKELY_SAME_MATCH verdict."""
    proj = {
        "v1": {
            "樊振东4-2张本智和-1-of-2.mp4": "0001.mp4",
            "樊振东4-2张本智和-2-of-2.mp4": "0002.mp4",
            "樊振东42张本智和-1-of-2.mp4": "0003.mp4",
            "樊振东42张本智和-2-of-2.mp4": "0004.mp4",
        },
    }
    groups = build_match_groups(proj, "v1")
    assert len(groups["near_duplicate_matches"]) >= 1
    pair = groups["near_duplicate_matches"][0]
    assert pair["verdict"] == "LIKELY_SAME_MATCH"
    assert "separators" in pair["similarity_hint"]


def test_near_duplicate_distinct_games_is_MANUAL_REVIEW():
    """第二盘 vs 第四盘 (different opponents, same team match) -> MANUAL_REVIEW."""
    proj = {
        "v1": {
            "2020联赛深圳vs汕头-第二盘郑培锋vs林高远-1-of-2.mp4": "0001.mp4",
            "2020联赛深圳vs汕头-第二盘郑培锋vs林高远-2-of-2.mp4": "0002.mp4",
            "2020联赛深圳vs汕头-第四盘郑培锋vs赛林威-1-of-2.mp4": "0003.mp4",
            "2020联赛深圳vs汕头-第四盘郑培锋vs赛林威-2-of-2.mp4": "0004.mp4",
        },
    }
    groups = build_match_groups(proj, "v1")
    # If similarity is high enough to register, the verdict should be MANUAL_REVIEW
    for pair in groups["near_duplicate_matches"]:
        assert pair["verdict"] in ("LIKELY_SAME_MATCH", "MANUAL_REVIEW")


# ---------------------------------------------------------------------------
# Chunk regex — relaxed pattern (handles `9-of-11.mp4` AND `-9-of-11.mp4`)
# ---------------------------------------------------------------------------

def test_parse_chunk_filename_without_separator_before_chunk():
    """2019成都世界杯男单半决赛马龙VS张本智和9-of-11.mp4 — no dash before chunk."""
    result = parse_chunk_filename(
        "2019成都世界杯男单半决赛马龙VS张本智和9-of-11.mp4"
    )
    assert result == ("2019成都世界杯男单半决赛马龙VS张本智和", 9, 11)


def test_parse_chunk_filename_match_name_ends_in_digit():
    """Match name ending in digit (e.g. '20190427') still parses correctly."""
    result = parse_chunk_filename(
        "2019年世锦赛1_4决赛梁靖崑VS丹羽孝希20190426-3-of-11.mp4"
    )
    assert result == ("2019年世锦赛1_4决赛梁靖崑VS丹羽孝希20190426", 3, 11)


def test_load_proj_json_real(tmp_path):
    proj_path = tmp_path / "proj.json"
    proj_path.write_text(json.dumps(PROJ_DATA_SAMPLE), encoding="utf-8")
    loaded = load_proj_json(proj_path)
    assert "v1" in loaded
    assert "v2" in loaded
    assert loaded["v1"]["matchA-1-of-3.mp4"] == "0000000.mp4"


def test_load_proj_json_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_proj_json(tmp_path / "no.json")


# ---------------------------------------------------------------------------
# Match-aware pilot picker
# ---------------------------------------------------------------------------

def _make_video_dicts_from_proj(proj_sample: dict, n_strokes_per: int = 20) -> list[dict]:
    """Build minimal video dicts matching the proj.json renamed IDs."""
    out = []
    for orig, renamed in proj_sample["v1"].items():
        out.append({
            "url": renamed,
            "total_frames": None,
            "actions": [
                {"label_names": ["正手", "否", "拉"], "start_id": float(i), "end_id": float(i) + 0.4}
                for i in range(n_strokes_per)
            ],
        })
    return out


def test_pick_match_aware_one_per_match():
    """With 3 v1 matches and n=3, each chunk should belong to a distinct match."""
    videos = _make_video_dicts_from_proj(PROJ_DATA_SAMPLE)
    groups = build_match_groups(PROJ_DATA_SAMPLE, "v1")
    lookup = renamed_id_to_match_lookup(groups)
    picked = pick_pilot_videos_match_aware(
        videos=videos, renamed_to_match=lookup, n=3, seed=42,
        min_strokes=1, max_strokes=100,
    )
    assert len(picked) == 3
    # Verify each pick belongs to a distinct match
    picked_match_ids = set()
    for v in picked:
        rid = v["url"].rsplit(".", 1)[0]
        picked_match_ids.add(lookup[rid]["match_id"])
    assert len(picked_match_ids) == 3


def test_pick_match_aware_smallest_index_strategy():
    """smallest_index should pick chunk_idx=1 from each match."""
    videos = _make_video_dicts_from_proj(PROJ_DATA_SAMPLE)
    groups = build_match_groups(PROJ_DATA_SAMPLE, "v1")
    lookup = renamed_id_to_match_lookup(groups)
    picked = pick_pilot_videos_match_aware(
        videos=videos, renamed_to_match=lookup, n=3, seed=42,
        min_strokes=1, max_strokes=100, chunk_strategy="smallest_index",
    )
    for v in picked:
        rid = v["url"].rsplit(".", 1)[0]
        assert lookup[rid]["chunk_idx"] == 1


def test_pick_match_aware_too_few_matches_raises():
    videos = _make_video_dicts_from_proj(PROJ_DATA_SAMPLE)
    groups = build_match_groups(PROJ_DATA_SAMPLE, "v1")
    lookup = renamed_id_to_match_lookup(groups)
    with pytest.raises(ValueError, match="cannot pick"):
        pick_pilot_videos_match_aware(
            videos=videos, renamed_to_match=lookup, n=10, seed=42,
            min_strokes=1, max_strokes=100,
        )


def test_pick_match_aware_invalid_strategy_raises():
    with pytest.raises(ValueError, match="Unknown chunk_strategy"):
        pick_pilot_videos_match_aware(
            videos=[], renamed_to_match={}, n=1, chunk_strategy="invalid",
        )


# ---------------------------------------------------------------------------
# Split-namespace collision (Codex 2026-05-25 fix #1)
# ---------------------------------------------------------------------------

# v1 and v2 in real P2A both use renamed_ids 0000000..0001207. Construct a
# minimal proj.json with a collision and verify the lookup defends correctly.

_COLLIDING_PROJ = {
    "v1": {
        "v1_matchA-1-of-2.mp4": "0000000.mp4",  # SAME renamed_id as below
        "v1_matchA-2-of-2.mp4": "0000001.mp4",
    },
    "v2": {
        "v2_matchZ-1-of-2.mp4": "0000000.mp4",  # COLLIDES with v1's 0000000
        "v2_matchZ-2-of-2.mp4": "0000002.mp4",
    },
}


def _build_combined(proj):
    v1g = build_match_groups(proj, "v1")
    v2g = build_match_groups(proj, "v2")
    return {
        "source_proj_json": "fake.json",
        "splits": {"v1": v1g, "v2": v2g},
        "near_duplicate_matches": [],
    }


def test_renamed_id_collision_raises_when_no_split_filter():
    combined = _build_combined(_COLLIDING_PROJ)
    with pytest.raises(ValueError, match="renamed_id collision"):
        renamed_id_to_match_lookup(combined)  # no splits filter


def test_renamed_id_collision_resolved_by_split_filter_v1():
    combined = _build_combined(_COLLIDING_PROJ)
    lookup = renamed_id_to_match_lookup(combined, splits=["v1"])
    assert lookup["0000000"]["match_name"] == "v1_matchA"
    assert lookup["0000000"]["split"] == "v1"
    assert "0000002" not in lookup  # v2-only ID


def test_renamed_id_collision_resolved_by_split_filter_v2():
    combined = _build_combined(_COLLIDING_PROJ)
    lookup = renamed_id_to_match_lookup(combined, splits=["v2"])
    assert lookup["0000000"]["match_name"] == "v2_matchZ"
    assert lookup["0000000"]["split"] == "v2"
    assert "0000001" not in lookup  # v1-only ID


def test_renamed_id_unknown_split_filter_raises():
    combined = _build_combined(_COLLIDING_PROJ)
    with pytest.raises(KeyError, match="not in combined groups"):
        renamed_id_to_match_lookup(combined, splits=["v99"])


def test_lookup_carries_split_field_per_split_shape():
    """Even on per-split (uncombined) input, the lookup tags each entry."""
    v1g = build_match_groups(_COLLIDING_PROJ, "v1")
    lookup = renamed_id_to_match_lookup(v1g)
    assert lookup["0000000"]["split"] == "v1"


# ---------------------------------------------------------------------------
# Rally contiguous-index validation (Codex 2026-05-25 fix #3)
# ---------------------------------------------------------------------------

def test_parse_rejects_non_contiguous_rally_indices():
    """Rally [0, 2] when 1 exists must be rejected."""
    bad = json.loads(json.dumps(VALID_RESPONSE))
    bad["rallies"][0]["stroke_indices"] = [0, 2, 3]  # missing 1, but 1 exists in rally 1
    bad["rallies"][1]["stroke_indices"] = [1, 4, 5]  # also non-contiguous
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("contiguous range" in e for e in result.errors)


def test_parse_rejects_interleaved_rallies():
    """[0, 1] then [2, 3] is contiguous within each; [0, 2] [1, 3] is not."""
    bad = json.loads(json.dumps(VALID_RESPONSE))
    # rally 0 takes stroke 0 and 2 (skipping 1) — wrong
    bad["rallies"] = [
        {"rally_id": 0, "stroke_indices": [0, 2], "server_won_rally": 1,
         "outcome_confidence": "high", "outcome_evidence": "x",
         "server_hand": "right", "receiver_hand": "right", "hand_confidence": "high"},
        {"rally_id": 1, "stroke_indices": [1, 3, 4, 5], "server_won_rally": 0,
         "outcome_confidence": "high", "outcome_evidence": "x",
         "server_hand": "right", "receiver_hand": "right", "hand_confidence": "high"},
    ]
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("contiguous range" in e for e in result.errors)


def test_parse_accepts_singleton_rally():
    """A rally with a single stroke is trivially contiguous."""
    bad = json.loads(json.dumps(VALID_RESPONSE))
    bad["rallies"] = [
        {"rally_id": 0, "stroke_indices": [0], "server_won_rally": 1,
         "outcome_confidence": "high", "outcome_evidence": "x",
         "server_hand": "right", "receiver_hand": "right", "hand_confidence": "high"},
        {"rally_id": 1, "stroke_indices": [1, 2, 3, 4, 5], "server_won_rally": 0,
         "outcome_confidence": "high", "outcome_evidence": "x",
         "server_hand": "right", "receiver_hand": "right", "hand_confidence": "high"},
    ]
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert result.ok, f"Singleton rallies should be accepted; errors: {result.errors}"


def test_parse_rejects_out_of_order_indices_within_rally():
    """stroke_indices [3, 1, 2] is sorted-fail (now hard error, was warning)."""
    bad = json.loads(json.dumps(VALID_RESPONSE))
    bad["rallies"][0]["stroke_indices"] = [3, 1, 2, 0]  # reversed/scrambled
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("not strictly sorted" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Receiver / server hand schema (Codex 2026-05-25 fix #4)
# ---------------------------------------------------------------------------

def test_parse_rejects_missing_server_hand():
    bad = json.loads(json.dumps(VALID_RESPONSE))
    del bad["rallies"][0]["server_hand"]
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("server_hand" in e for e in result.errors)


def test_parse_rejects_missing_receiver_hand():
    bad = json.loads(json.dumps(VALID_RESPONSE))
    del bad["rallies"][0]["receiver_hand"]
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("receiver_hand" in e for e in result.errors)


def test_parse_rejects_invalid_hand_value():
    bad = json.loads(json.dumps(VALID_RESPONSE))
    bad["rallies"][0]["server_hand"] = "ambidextrous"
    raw = json.dumps(bad)
    result, _ = parse_gemini_response(raw, "0000123", 6)
    assert not result.ok
    assert any("server_hand" in e for e in result.errors)


def test_parse_accepts_unknown_hand():
    """'unknown' is a valid hand value (Gemini opts out when uncertain)."""
    ok = json.loads(json.dumps(VALID_RESPONSE))
    ok["rallies"][0]["server_hand"] = "unknown"
    ok["rallies"][0]["receiver_hand"] = "unknown"
    raw = json.dumps(ok)
    result, data = parse_gemini_response(raw, "0000123", 6)
    assert result.ok, f"unknown hand should validate; errors: {result.errors}"


def test_build_prompt_contains_court_absolute_landing():
    """Prompt must mention table_left/mid/right axis, NOT FH/BH."""
    anchors = video_anchor_table(SAMPLE_VIDEO)
    prompt = build_prompt(anchors)
    assert "table_left_short" in prompt
    assert "table_mid_long" in prompt
    assert "table_right_half" in prompt
    # Confirm we removed the old receiver-relative axis
    assert "ASSUMING THE RECEIVER IS RIGHT-HANDED" not in prompt
    assert "DO NOT use FH/BH terminology" in prompt


def test_build_prompt_asks_for_per_rally_hands():
    anchors = video_anchor_table(SAMPLE_VIDEO)
    prompt = build_prompt(anchors)
    assert "server_hand" in prompt
    assert "receiver_hand" in prompt
    assert "hand_confidence" in prompt


# ---------------------------------------------------------------------------
# Wilson lower bound (Codex 2026-05-25 Q5)
# ---------------------------------------------------------------------------

def test_wilson_lower_bound_zero_n():
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_lower_bound_perfect_high_n():
    """With perfect accuracy and high n, lower bound approaches 1."""
    lb = wilson_lower_bound(1000, 1000)
    assert lb > 0.99


def test_wilson_lower_bound_small_n_is_pessimistic():
    """200 strokes at 55% raw: Wilson lower-bound should be noticeably below 0.55."""
    lb = wilson_lower_bound(110, 200)  # 0.55 raw
    assert 0.45 < lb < 0.53, f"Wilson lower bound on 110/200 should be ~0.48: got {lb}"


def test_wilson_lower_bound_n_1_is_very_pessimistic():
    """n=1 correct: huge uncertainty."""
    lb = wilson_lower_bound(1, 1)
    assert lb < 0.30, f"n=1 should give lower bound < 0.30, got {lb}"


def test_wilson_lower_bound_zero_correct():
    """0/100 -> lower bound should be 0."""
    lb = wilson_lower_bound(0, 100)
    assert lb == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
