"""Compare Gemini's pilot output against hand-truth, report per-field accuracy.

Hand-truth CSV schema (see scripts/p2a_pilot/HAND_TRUTH_TEMPLATE.csv):

    video_id,stroke_idx,landing_zone,player_position
    0000123,0,FH_short,center
    0000123,1,off_grid,left
    ...

Plus per-rally:
    video_id,rally_id,server_won_rally
    0000123,0,1
    0000123,1,0
    ...

Usage:
    from p2a_pilot.accuracy import compare_video, accuracy_report

    # gemini_data = parsed JSON from parse_response.parse_gemini_response
    # hand_truth = {"strokes": {...}, "rallies": {...}}
    # See cli.py for the full driver.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


def wilson_lower_bound(correct: int, total: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a binomial proportion.

    Use this instead of raw accuracy when n is small (e.g. pilot n=200).
    With z=1.96 the interval is 95% confidence; the lower bound is the
    most pessimistic accuracy compatible with the observation.

    Added 2026-05-25 per Codex Q5: "200 strokes / 30 rallies 太薄；至少
    加 min-n 與 Wilson lower-bound，不要只看 raw accuracy."
    """
    if total <= 0:
        return 0.0
    p = correct / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return max(0.0, centre - half)


MIN_N_FOR_VERDICT = 30  # below this, no PASS verdict regardless of point estimate


@dataclass
class FieldStats:
    name: str
    total: int = 0
    correct: int = 0
    n_low_conf: int = 0
    n_med_conf: int = 0
    n_high_conf: int = 0
    correct_low: int = 0
    correct_med: int = 0
    correct_high: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def accuracy_high_conf_only(self) -> float:
        return self.correct_high / self.n_high_conf if self.n_high_conf else 0.0

    @property
    def wilson_lower(self) -> float:
        """95% Wilson lower bound on overall accuracy."""
        return wilson_lower_bound(self.correct, self.total)

    @property
    def wilson_lower_high_conf(self) -> float:
        """95% Wilson lower bound on high-confidence-only accuracy."""
        return wilson_lower_bound(self.correct_high, self.n_high_conf)

    def to_dict(self) -> dict:
        return {
            "field": self.name,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "wilson_lower_95": round(self.wilson_lower, 4),
            "n_high_conf": self.n_high_conf,
            "accuracy_high_conf_only": round(self.accuracy_high_conf_only, 4),
            "wilson_lower_95_high_conf": round(self.wilson_lower_high_conf, 4),
            "n_medium_conf": self.n_med_conf,
            "n_low_conf": self.n_low_conf,
        }


def _record_field(stats: FieldStats, pred, truth, conf: str | None) -> None:
    stats.total += 1
    is_correct = pred == truth
    if is_correct:
        stats.correct += 1
    if conf == "high":
        stats.n_high_conf += 1
        if is_correct:
            stats.correct_high += 1
    elif conf == "medium":
        stats.n_med_conf += 1
        if is_correct:
            stats.correct_med += 1
    elif conf == "low":
        stats.n_low_conf += 1
        if is_correct:
            stats.correct_low += 1


def compare_video(
    gemini_data: dict,
    hand_truth_strokes: dict[int, dict],
    hand_truth_rallies: dict[int, dict],
) -> dict[str, FieldStats]:
    """Compare one video's Gemini output against its hand-truth.

    Parameters
    ----------
    gemini_data : dict
        Parsed + validated Gemini response (from parse_response).
    hand_truth_strokes : dict[int, dict]
        Mapping stroke_idx -> {"landing_zone": str, "player_position": str}
        ``landing_zone`` values are court-absolute (table_left_short, ...),
        matching the new schema.
    hand_truth_rallies : dict[int, dict]
        Mapping rally_id -> {"server_won_rally": int, "receiver_hand": str,
                             "server_hand": str}

    Returns
    -------
    dict[str, FieldStats]
        Keys: "landing_zone", "player_position", "server_won_rally",
              "receiver_hand", "server_hand".
        Each value is a FieldStats with per-confidence breakdown.
    """
    stats = {
        "landing_zone": FieldStats("landing_zone"),
        "player_position": FieldStats("player_position"),
        "server_won_rally": FieldStats("server_won_rally"),
        "receiver_hand": FieldStats("receiver_hand"),
        "server_hand": FieldStats("server_hand"),
    }

    for s in gemini_data.get("strokes", []):
        idx = s.get("stroke_idx")
        truth = hand_truth_strokes.get(idx)
        if truth is None:
            continue
        if "landing_zone" in truth:
            _record_field(
                stats["landing_zone"],
                s.get("landing_zone"),
                truth["landing_zone"],
                s.get("landing_confidence"),
            )
        if "player_position" in truth:
            _record_field(
                stats["player_position"],
                s.get("player_position"),
                truth["player_position"],
                s.get("position_confidence"),
            )

    for r in gemini_data.get("rallies", []):
        rid = r.get("rally_id")
        truth = hand_truth_rallies.get(rid)
        if truth is None:
            continue
        if "server_won_rally" in truth:
            _record_field(
                stats["server_won_rally"],
                r.get("server_won_rally"),
                truth["server_won_rally"],
                r.get("outcome_confidence"),
            )
        # Hand fields share the same hand_confidence
        hand_conf = r.get("hand_confidence")
        if "receiver_hand" in truth:
            _record_field(
                stats["receiver_hand"],
                r.get("receiver_hand"),
                truth["receiver_hand"],
                hand_conf,
            )
        if "server_hand" in truth:
            _record_field(
                stats["server_hand"],
                r.get("server_hand"),
                truth["server_hand"],
                hand_conf,
            )
    return stats


def aggregate_stats(per_video_stats: Iterable[dict[str, FieldStats]]) -> dict[str, FieldStats]:
    """Sum FieldStats across multiple videos for the same field name."""
    total = {
        "landing_zone": FieldStats("landing_zone"),
        "player_position": FieldStats("player_position"),
        "server_won_rally": FieldStats("server_won_rally"),
        "receiver_hand": FieldStats("receiver_hand"),
        "server_hand": FieldStats("server_hand"),
    }
    for video_stats in per_video_stats:
        for name, s in video_stats.items():
            agg = total.setdefault(name, FieldStats(name))
            agg.total += s.total
            agg.correct += s.correct
            agg.n_low_conf += s.n_low_conf
            agg.n_med_conf += s.n_med_conf
            agg.n_high_conf += s.n_high_conf
            agg.correct_low += s.correct_low
            agg.correct_med += s.correct_med
            agg.correct_high += s.correct_high
    return total


# Gates for proceeding past the pilot phase (see PILOT_PROTOCOL.md §5).
# NOTE per Codex 2026-05-25 Q5: gates are checked against the WILSON LOWER
# BOUND (95% CI), not raw accuracy. n=200 is too small to trust raw point
# estimates; Wilson lower bound is the most pessimistic accuracy compatible
# with the observation. Also a hard ``min_n`` per field prevents trivial
# PASS on n=2.
PILOT_GATES = {
    "landing_zone":     {"overall": 0.50, "high_conf_only": 0.65, "min_n": 30},
    "player_position":  {"overall": 0.70, "high_conf_only": 0.85, "min_n": 30},
    "server_won_rally": {"overall": 0.75, "high_conf_only": 0.90, "min_n": MIN_N_FOR_VERDICT},
    "receiver_hand":    {"overall": 0.80, "high_conf_only": 0.90, "min_n": MIN_N_FOR_VERDICT},
    "server_hand":      {"overall": 0.80, "high_conf_only": 0.90, "min_n": MIN_N_FOR_VERDICT},
}


def accuracy_report(agg: dict[str, FieldStats]) -> str:
    """Render a human-readable accuracy report with pass/fail per gate.

    The pass condition for each field is:
      Wilson 95% lower bound on overall accuracy >= overall gate
      AND Wilson 95% lower bound on high-conf-only accuracy >= high-conf gate
      AND total n >= min_n
    """
    lines = ["=" * 78, "P2A-Gemini Pilot — Accuracy Report (Wilson 95% lower-bound gates)", "=" * 78]
    overall_pass = True
    for name, stats in agg.items():
        d = stats.to_dict()
        gates = PILOT_GATES.get(name, {})
        gate_overall = gates.get("overall")
        gate_high = gates.get("high_conf_only")
        min_n = gates.get("min_n", MIN_N_FOR_VERDICT)

        n_ok = d["total"] >= min_n
        pass_overall = (
            d["wilson_lower_95"] >= gate_overall if gate_overall is not None else True
        )
        pass_high = (
            d["wilson_lower_95_high_conf"] >= gate_high if gate_high is not None else True
        )
        if not n_ok:
            verdict = "INSUFFICIENT_N"
            overall_pass = False
        else:
            verdict = "PASS" if (pass_overall and pass_high) else "FAIL"
            if verdict == "FAIL":
                overall_pass = False

        lines.append("")
        lines.append(f"Field: {name}  [{verdict}]")
        lines.append(f"  total predictions:  n={d['total']}  (min_n={min_n}: "
                     f"{'OK' if n_ok else 'TOO FEW'})")
        if gate_overall is not None:
            lines.append(
                f"  overall:    raw acc={d['accuracy']:.3f}  "
                f"Wilson 95% lower={d['wilson_lower_95']:.3f}  "
                f"(gate >= {gate_overall:.2f}: "
                f"{'PASS' if pass_overall and n_ok else 'FAIL'})"
            )
        if gate_high is not None:
            lines.append(
                f"  high-conf:  raw acc={d['accuracy_high_conf_only']:.3f}  "
                f"Wilson 95% lower={d['wilson_lower_95_high_conf']:.3f}  "
                f"(n={d['n_high_conf']}; gate >= {gate_high:.2f}: "
                f"{'PASS' if pass_high and n_ok else 'FAIL'})"
            )
        lines.append(
            f"  conf breakdown:  high={d['n_high_conf']} "
            f"med={d['n_medium_conf']} low={d['n_low_conf']}"
        )
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"PILOT VERDICT: {'PASS — proceed to Phase 1 R-073 preflight' if overall_pass else 'HOLD — see per-field FAIL/INSUFFICIENT_N above'}")
    lines.append("=" * 78)
    return "\n".join(lines)
