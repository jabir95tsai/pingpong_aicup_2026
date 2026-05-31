"""CLI driver for the P2A-Gemini pilot.

Three commands:

  1. ``prepare``  — pick N pilot videos from P2A JSON, write a prompt file
                    per video plus a manifest. After this, you upload each
                    video to web Gemini + paste the corresponding prompt.

  2. ``parse``    — read Gemini's saved responses, validate against schema,
                    write a per-video parsed JSON + summary log.

  3. ``report``   — read the parsed JSONs + the hand-truth CSVs, compute
                    per-field accuracy + pilot pass/fail verdict.

Usage examples:

    # Step 1 — pick 10 pilot videos and generate prompts
    python -m p2a_pilot.cli prepare \\
        --label-json "C:/path/to/P2A/dataset/label/v1.json" \\
        --n 10 --seed 42 \\
        --out runs/p2a_pilot

    # (manual) For each prompts/<video_id>.prompt.txt:
    #   - upload runs/p2a_pilot/prompts/<video_id>.mp4-path.txt into Gemini
    #     (or upload the .mp4 file directly)
    #   - paste prompts/<video_id>.prompt.txt as the chat message
    #   - save the JSON response to runs/p2a_pilot/responses/<video_id>.json

    # Step 2 — parse + validate all responses
    python -m p2a_pilot.cli parse --out runs/p2a_pilot

    # (manual) Fill in hand-truth CSVs at scripts/p2a_pilot/hand_truth/{strokes,rallies}.csv

    # Step 3 — accuracy report
    python -m p2a_pilot.cli report --out runs/p2a_pilot \\
        --hand-truth-strokes scripts/p2a_pilot/hand_truth/strokes.csv \\
        --hand-truth-rallies scripts/p2a_pilot/hand_truth/rallies.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .load_p2a import (
    load_p2a_videos, pick_pilot_videos, pick_pilot_videos_match_aware,
    video_anchor_table, pre_segment_rallies,
)
from .build_prompt import build_prompt
from .parse_response import parse_gemini_response
from .accuracy import compare_video, aggregate_stats, accuracy_report
from .group_matches import (
    load_proj_json, build_match_groups, renamed_id_to_match_lookup,
    summarize_groups,
)


def cmd_prepare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    prompts_dir = out_dir / "prompts"
    anchors_dir = out_dir / "anchors"
    responses_dir = out_dir / "responses"
    parsed_dir = out_dir / "parsed"
    for d in [prompts_dir, anchors_dir, responses_dir, parsed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    videos = load_p2a_videos(args.label_json)
    print(f"Loaded {len(videos)} videos from {args.label_json}")

    # Optional match-aware picking
    renamed_to_match: dict[str, dict] = {}
    if args.match_groups:
        groups_path = Path(args.match_groups)
        if not groups_path.exists():
            print(f"ERROR: --match-groups path not found: {groups_path}", file=sys.stderr)
            return 2
        groups = json.loads(groups_path.read_text(encoding="utf-8"))

        # Determine which split(s) to include in the lookup.
        # v1 and v2 share the renamed_id namespace (0000000..0001207 in both)
        # so an unfiltered multi-split lookup silently corrupts manifests.
        # Defaults:
        #   --split <name>  : explicit, use this split only
        #   (omitted)       : auto-infer from --label-json filename if possible;
        #                     otherwise require explicit --split
        split_filter: list[str] | None = None
        if args.split:
            split_filter = [args.split]
        else:
            lj_name = Path(args.label_json).stem.lower()
            # Recognise patterns like "v1", "v2", "v1_renamed", "v2_renamed"
            inferred = None
            for candidate in ("v1", "v2"):
                if candidate in lj_name.split("_") or lj_name.startswith(candidate):
                    inferred = candidate
                    break
            if inferred is not None:
                split_filter = [inferred]
                print(f"Auto-inferred --split={inferred} from --label-json={lj_name!r}")
            else:
                # If groups has a single split, use it; otherwise error
                if "splits" in groups and len(groups["splits"]) == 1:
                    only = next(iter(groups["splits"].keys()))
                    split_filter = [only]
                    print(f"Using sole split {only!r} from match-groups file")

        renamed_to_match = renamed_id_to_match_lookup(groups, splits=split_filter)
        if split_filter:
            print(f"Loaded {len(renamed_to_match)} renamed_id → match entries "
                  f"(split filter: {split_filter})")
        else:
            print(f"WARNING: no split filter applied. If renamed_id collisions "
                  f"exist between splits this will raise on lookup.")

        pilot = pick_pilot_videos_match_aware(
            videos,
            renamed_to_match=renamed_to_match,
            n=args.n,
            seed=args.seed,
            min_strokes=args.min_strokes,
            max_strokes=args.max_strokes,
            chunk_strategy=args.chunk_strategy,
        )
        print(
            f"Picked {len(pilot)} pilot videos (match-aware, seed={args.seed}, "
            f"strategy={args.chunk_strategy})"
        )
    else:
        pilot = pick_pilot_videos(
            videos,
            n=args.n,
            seed=args.seed,
            min_strokes=args.min_strokes,
            max_strokes=args.max_strokes,
        )
        print(
            f"Picked {len(pilot)} pilot videos (flat, seed={args.seed}) — "
            f"pass --match-groups for match-aware diversity"
        )

    manifest: list[dict] = []
    for v in pilot:
        anchors = video_anchor_table(v)
        pre_seg = pre_segment_rallies(anchors, rally_gap_seconds=args.rally_gap)
        prompt = build_prompt(anchors, pre_segment=pre_seg)
        vid = anchors["video_id"]
        match_meta = renamed_to_match.get(vid) if renamed_to_match else None

        (anchors_dir / f"{vid}.anchors.json").write_text(
            json.dumps({"anchors": anchors, "pre_segment": pre_seg}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (prompts_dir / f"{vid}.prompt.txt").write_text(prompt, encoding="utf-8")
        entry = {
            "video_id": vid,
            "url": anchors["url"],
            "n_strokes": anchors["n_strokes"],
            "n_pre_rallies": len(pre_seg),
            "prompt_file": str((prompts_dir / f"{vid}.prompt.txt").relative_to(out_dir)),
            "anchors_file": str((anchors_dir / f"{vid}.anchors.json").relative_to(out_dir)),
            "response_file_expected": str((responses_dir / f"{vid}.json").relative_to(out_dir)),
        }
        if match_meta is not None:
            entry["match_id"] = match_meta["match_id"]
            entry["match_name"] = match_meta["match_name"]
            entry["chunk_idx"] = match_meta["chunk_idx"]
            entry["n_chunks_in_match"] = match_meta["n_chunks_in_match"]
            entry["original_filename"] = match_meta["original_filename"]
        manifest.append(entry)

    (out_dir / "pilot_manifest.json").write_text(
        json.dumps(
            {
                "n_videos": len(manifest),
                "seed": args.seed,
                "match_aware": bool(renamed_to_match),
                "chunk_strategy": args.chunk_strategy if renamed_to_match else None,
                "videos": manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {len(manifest)} prompts to {prompts_dir}/")
    print(f"Wrote manifest to {out_dir/'pilot_manifest.json'}")
    print(f"\nNext steps — see scripts/p2a_pilot/PILOT_PROTOCOL.md")
    return 0


def cmd_group_matches(args: argparse.Namespace) -> int:
    """Build the renamed↔match mapping artifact from proj.json."""
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    proj = load_proj_json(args.proj_json)
    print(f"Loaded proj.json with splits: {list(proj.keys())}")

    splits_to_process: list[str] = (
        args.splits.split(",") if args.splits else list(proj.keys())
    )

    combined = {
        "source_proj_json": str(Path(args.proj_json).resolve()),
        "splits": {},
        "near_duplicate_matches": [],
    }
    for split in splits_to_process:
        split = split.strip()
        if not split:
            continue
        groups = build_match_groups(proj, split)
        combined["splits"][split] = groups
        print()
        print(summarize_groups(groups))
        combined["near_duplicate_matches"].extend([
            {**nd, "split": split} for nd in groups["near_duplicate_matches"]
        ])

    out_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote match groups to {out_path}")
    if combined["near_duplicate_matches"]:
        print(
            f"NOTE: {len(combined['near_duplicate_matches'])} near-duplicate "
            f"match-name pairs detected. Review them in the output JSON; "
            f"these may be the same physical match with typo variants."
        )
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    anchors_dir = out_dir / "anchors"
    responses_dir = out_dir / "responses"
    parsed_dir = out_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "pilot_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run `prepare` first.", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    summary = {"n_videos": 0, "n_parsed_ok": 0, "n_errors": 0, "per_video": []}
    for v in manifest["videos"]:
        vid = v["video_id"]
        n_strokes = v["n_strokes"]
        response_path = out_dir / v["response_file_expected"]
        per = {"video_id": vid, "n_strokes": n_strokes}
        if not response_path.exists():
            per["status"] = "MISSING"
            per["error"] = f"response file not found: {response_path}"
            summary["per_video"].append(per)
            summary["n_videos"] += 1
            continue
        raw = response_path.read_text(encoding="utf-8")
        result, data = parse_gemini_response(
            raw_text=raw,
            expected_video_id=vid,
            expected_n_strokes=n_strokes,
        )
        per["status"] = "OK" if result.ok else "ERROR"
        per["errors"] = result.errors
        per["warnings"] = result.warnings
        summary["n_videos"] += 1
        if result.ok:
            summary["n_parsed_ok"] += 1
            (parsed_dir / f"{vid}.parsed.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            summary["n_errors"] += 1
        summary["per_video"].append(per)

    (out_dir / "parse_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Parsed {summary['n_parsed_ok']}/{summary['n_videos']} responses successfully.")
    if summary["n_errors"] > 0:
        print(f"  {summary['n_errors']} responses had schema errors — see parse_summary.json")
    return 0 if summary["n_parsed_ok"] == summary["n_videos"] else 1


def _load_hand_truth_strokes(csv_path: Path) -> dict[str, dict[int, dict]]:
    """Returns {video_id: {stroke_idx: {field: value}}}"""
    out: dict[str, dict[int, dict]] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "").strip()
            try:
                idx = int(row["stroke_idx"])
            except (KeyError, ValueError, TypeError):
                continue
            entry = out.setdefault(vid, {}).setdefault(idx, {})
            for k in ("landing_zone", "player_position"):
                v = (row.get(k) or "").strip()
                if v:
                    entry[k] = v
    return out


def _load_hand_truth_rallies(csv_path: Path) -> dict[str, dict[int, dict]]:
    """Returns {video_id: {rally_id: {field: value}}}

    Schema: video_id, rally_id, server_won_rally, server_hand, receiver_hand
    """
    out: dict[str, dict[int, dict]] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "").strip()
            try:
                rid = int(row["rally_id"])
            except (KeyError, ValueError, TypeError):
                continue
            entry = out.setdefault(vid, {}).setdefault(rid, {})
            v = (row.get("server_won_rally") or "").strip()
            if v:
                try:
                    entry["server_won_rally"] = int(v)
                except ValueError:
                    pass
            for hk in ("server_hand", "receiver_hand"):
                val = (row.get(hk) or "").strip().lower()
                if val in ("right", "left", "unknown"):
                    entry[hk] = val
    return out


def cmd_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    parsed_dir = out_dir / "parsed"
    parsed_files = sorted(parsed_dir.glob("*.parsed.json"))
    if not parsed_files:
        print(f"ERROR: no parsed JSON files in {parsed_dir}. Run `parse` first.", file=sys.stderr)
        return 2

    ht_strokes = _load_hand_truth_strokes(Path(args.hand_truth_strokes))
    ht_rallies = _load_hand_truth_rallies(Path(args.hand_truth_rallies))
    if not ht_strokes and not ht_rallies:
        print(
            f"ERROR: no hand-truth rows loaded. "
            f"Check {args.hand_truth_strokes} and {args.hand_truth_rallies}.",
            file=sys.stderr,
        )
        return 2

    per_video_stats = []
    for pf in parsed_files:
        data = json.loads(pf.read_text(encoding="utf-8"))
        vid = data.get("video_id", pf.stem.replace(".parsed", ""))
        stats = compare_video(
            gemini_data=data,
            hand_truth_strokes=ht_strokes.get(vid, {}),
            hand_truth_rallies=ht_rallies.get(vid, {}),
        )
        per_video_stats.append(stats)

    agg = aggregate_stats(per_video_stats)
    report = accuracy_report(agg)
    print(report)

    (out_dir / "accuracy_report.txt").write_text(report, encoding="utf-8")
    (out_dir / "accuracy_report.json").write_text(
        json.dumps({name: s.to_dict() for name, s in agg.items()}, indent=2),
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2A-Gemini pilot CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_group = sub.add_parser(
        "group-matches",
        help="group P2A chunks by source match (using proj.json)",
    )
    p_group.add_argument(
        "--proj-json", required=True,
        help="path to proj.json (e.g. D:/P2A_dataset/dataset/proj.json)",
    )
    p_group.add_argument(
        "--splits", default=None,
        help="comma-separated splits to process (default: all in proj.json)",
    )
    p_group.add_argument(
        "--out", default="data/p2a_match_groups.json",
        help="output path for the combined match-groups JSON",
    )
    p_group.set_defaults(func=cmd_group_matches)

    p_prep = sub.add_parser("prepare", help="pick pilot videos + generate prompts")
    p_prep.add_argument("--label-json", required=True, help="path to P2A label JSON (e.g. v1_renamed.json)")
    p_prep.add_argument(
        "--match-groups", default=None,
        help="optional path to match-groups JSON (output of `group-matches`). "
             "When set, picks one chunk per match for max diversity.",
    )
    p_prep.add_argument(
        "--split", default=None,
        help="which split (v1/v2) to restrict the match-groups lookup to. "
             "If omitted, inferred from --label-json filename. v1 and v2 "
             "share the renamed_id namespace (0000000..0001207 in both), so "
             "the filter is REQUIRED for collision safety on multi-split "
             "match-groups files.",
    )
    p_prep.add_argument(
        "--chunk-strategy", default="smallest_index",
        choices=["smallest_index", "middle", "random"],
        help="which chunk to pick from each sampled match (only with --match-groups)",
    )
    p_prep.add_argument("--n", type=int, default=10, help="number of pilot videos")
    p_prep.add_argument("--seed", type=int, default=42)
    p_prep.add_argument("--min-strokes", type=int, default=10)
    p_prep.add_argument("--max-strokes", type=int, default=40)
    p_prep.add_argument("--rally-gap", type=float, default=4.0, help="rally segmentation gap (s)")
    p_prep.add_argument("--out", default="runs/p2a_pilot")
    p_prep.set_defaults(func=cmd_prepare)

    p_parse = sub.add_parser("parse", help="parse + validate Gemini responses")
    p_parse.add_argument("--out", default="runs/p2a_pilot")
    p_parse.set_defaults(func=cmd_parse)

    p_report = sub.add_parser("report", help="accuracy report vs hand-truth")
    p_report.add_argument("--out", default="runs/p2a_pilot")
    p_report.add_argument(
        "--hand-truth-strokes",
        default="scripts/p2a_pilot/hand_truth/strokes.csv",
    )
    p_report.add_argument(
        "--hand-truth-rallies",
        default="scripts/p2a_pilot/hand_truth/rallies.csv",
    )
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
