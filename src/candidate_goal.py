"""Candidate Goal Function v0.6 — heuristic decision aid (generalization-first).

Status: HEURISTIC v0.2, not mathematically proven. Empirically grounded on
post-2026-05-06 LB datapoints. Re-tune coefficients after every 3-5 new LB
results — see `GOAL_FUNCTION.md` for the rationale and re-calibration schedule.

Primary project goal:
  Reach clean NEW LB >= 0.4000 (target) while generalizing to the Private/Final
  distribution. We strictly forbid leakage and we deprioritize Public-LB
  overfit churn (weight refinement / zoo re-arrangement). Anchor = R-067c LB
  0.3870095 → remaining gap to TARGET_LB ≈ +0.0130.

Purpose:
  Score any proposed experiment / component / blend / submission candidate
  BEFORE spending compute or burning an LB slot. Returns a dict with:
    - hard_block:              True/False
    - block_reason:            str (if hard_block)
    - expected_lb_delta:       float (OV units, vs current LB-best)
    - goal_score:              float (priority signal, OV units)
    - risk_level:              LOW / MEDIUM / HIGH / BLOCKED
    - leakage_risk:            NONE / LOW / MEDIUM / HIGH / CRITICAL
    - public_lb_overfit_risk:  LOW / MEDIUM / HIGH
    - generalization_score:    float 0..1 (higher = more likely to transfer)
    - priority:                STRATEGIC / HIGH / NORMAL / LOW / PARK
    - target_progress_ratio:   expected_lb_delta / (TARGET_LB - anchor_lb)
    - recommended_action:      BLOCK / PARK / SMOKE_ONLY / FULL_5FOLD_REVIEW
                              / MATERIALIZE_FOR_REVIEW / SUBMIT_CANDIDATE
    - explanation:             short human-readable

Usage:
  from candidate_goal import score_candidate
  verdict = score_candidate(candidate_dict)

Coefficients are EXPLICIT (top of file) and easy to edit. Do NOT wire this
into training; it is a decision aid only. The user / Codex can override.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


# ─── Coefficients (heuristic v0.2, easy to edit) ────────────────────────────

# Project target — clean NEW LB >= 0.4000.
# Generalization-first: we are NOT optimizing Public LB; we are aiming for the
# Private / Final distribution. Public-LB-style tiny wins are low-priority.
TARGET_LB = 0.4000

# Default anchor LB-best (R-067c, 2026-05-24). Override per-candidate.
ANCHOR_LB_DEFAULT = 0.3870095
# Implied default gap: TARGET_LB - ANCHOR_LB_DEFAULT = +0.0130

# Empirical OOF→LB transfer multipliers per class
# (Calibrated on post-2026-05-06 LB history; see GOAL_FUNCTION.md §5)
TRANSFER_MULTIPLIER: Dict[str, float] = {
    # v0.3 SPLIT: rule_override is NOT a uniform 1.0 class. R-072 LB regression
    # (-0.0033) proved layers with handId/positionId in the context fail the same
    # way as B-player-style. Only the R-042 shot-content context (prev_action,
    # last_action, last_point) is empirically validated at 1.0 transfer.
    "rule_override":              1.0,    # R-042: +0.0028 OOF → +0.0028 LB
    "rule_override_deep_prefix":  0.3,    # NEW v0.3: deeper shot-context only
                                          # (prev_prev_action etc). Untested isolated;
                                          # included in R-072's failure → conservative.
    "B-pure":             1.0,   # R-027 PAIR: super-additive, conservatively 1.0
    "B-feature":          0.9,   # R-034: ratio 1.0121, 10% pessimism for unseen sets
    "server-head-blend":  0.05,  # R-067c: 5.4% of expected (0.0326 × 0.2 → +0.000355)
    "weight-refinement":  0.3,   # R-068: full +0.0012 but holdout −0.0001 → ~30%
    "A-rearrangement":    0.5,   # historical Class A ratio 0.95-0.98
    "new-mechanism":      0.8,   # default optimistic; tunes after first LB datapoint
    # Toxic classes — these never reach the multiplier (hard-blocked first):
    "B-impure":                     0.0,
    "B-meta":                       0.0,
    "B-player-style":               0.0,
    "rule_override_player_context": 0.0,   # v0.3 — R-072 LB -0.0033 with
                                            # handId/positionId in rule context.
    "B-impure-additive-low-weight": 0.0,   # NEW v0.6 — R-094 v2 LB -0.0040
                                            # at SoftF1 additive α=0.05 (action-only).
                                            # Confirms B-impure is toxic EVEN AT
                                            # 5% weight additive, not just full swap.
    "pseudo-consensus":             0.0,
    "hard-per-SN-blend":            0.0,
    "unknown":                      0.5,   # cautious default
}

# Classes that hard-block unless novelty=="high" + explicit new-mechanism review
TOXIC_CLASSES = {
    "B-impure", "B-meta", "B-player-style",
    "rule_override_player_context",     # v0.3
    "B-impure-additive-low-weight",      # NEW v0.6 — R-094 v2 LB -0.0040
    "pseudo-consensus", "hard-per-SN-blend",
}

# Slice-penalty thresholds (OV units)
SN_BUCKET_REGRESSION_THRESHOLD = -0.005       # per-bucket OV drop counted as regression
SN_BUCKET_REGRESSION_PENALTY = -0.001          # OV penalty per regressing bucket
CANARY_CLASS_DROP_THRESHOLD = -0.015           # per-class F1 drop counted as canary fail
CANARY_CLASS_DROP_PENALTY = -0.002             # OV penalty per failed canary

# Holdout advisory (small magnitude — never a hard gate)
HOLDOUT_POSITIVE_THRESHOLD = 0.001
HOLDOUT_POSITIVE_BONUS = 0.0002
HOLDOUT_NEGATIVE_THRESHOLD = -0.003
HOLDOUT_NEGATIVE_PENALTY = -0.0003

# Compute cost penalty (small; only kicks in past 5 hours)
COMPUTE_FREE_HOURS = 5.0
COMPUTE_PENALTY_PER_HOUR = -0.0005

# Novelty bonus
NOVELTY_BONUS = {"low": 0.0, "medium": 0.0003, "high": 0.0008}

# Recommended-action thresholds (expected_lb_delta in OV units)
SUBMIT_THRESHOLD = 0.0          # >=0 → eligible to upload
PARK_THRESHOLD = -0.005          # <=-0.005 → park
MARGINAL_LOWER = -0.005          # (-0.005, 0) → SMOKE_ONLY / FULL_5FOLD_REVIEW band

# ─── v0.4 POLICY (theory-first; LB-confirms-truth) ──────────────────────────
# Per user directive 2026-05-26:
# 1. Theoretical generalization mechanism comes first.
# 2. Fold-1 smoke is a sanity check, not the final judge.
# 3. Only manual LB submission can confirm real transfer.
# 4. Do not declare WIN/FAIL until LB result exists.
#
# For HIGH/STRATEGIC: a marginal Fold-1 OOF miss should NOT auto-kill a
# theoretically strong candidate. Smoke only blocks on:
#   (a) leakage / rule violation
#   (b) broken pipeline / alignment bug
#   (c) NaN / Inf / invalid submission
#   (d) catastrophic collapse
#   (e) severe per-class or per-SN regression
#
# For LOW priority churn, smoke gates stay strict.

# "Catastrophic collapse" thresholds (HIGH/STRATEGIC override smoke MUST trip)
CATASTROPHIC_OV_DROP        = -0.020   # full OV drops worse than this = collapse
CATASTROPHIC_AUC_DROP       = -0.030   # AUC alone dropping this much = collapse
CATASTROPHIC_F1_DROP        = -0.030   # F1_a or F1_p alone dropping this much = collapse

# "Severe per-class / per-SN regression" thresholds
SEVERE_CANARY_CLASS_DROP    = -0.025   # per-class F1 drop this large = severe
SEVERE_CANARY_CLASS_COUNT   = 3         # 3+ severe drops = severe pattern
SEVERE_SN_BUCKET_DROP       = -0.012   # per-bucket OV drop this large = severe
SEVERE_SN_BUCKET_COUNT      = 2         # 2+ severe buckets = severe pattern

# Verdict vocabulary (v0.4)
# Pre-LB:
#   PROVISIONAL_PASS  → smoke OK + theory strong; recommended for LB probe
#   PROVISIONAL_FAIL  → catastrophic smoke OR clear theory hole OR leak
#   PARK              → LOW-priority churn or expLB ≤ 0
#   BLOCK             → hard rule violation (leakage / toxic class etc.)
# Post-CSV-build:
#   ARTIFACT_READY_FOR_JABIR_UPLOAD_REVIEW
# Post-LB (set manually by Jabir/Codex/Claude after upload):
#   LB_WIN  / LB_FAIL  / LB_NOISE

# Daily LB upload cap (informational; not auto-enforced)
DAILY_LB_UPLOAD_CAP = 3

# ─── Priority bucketing (generalization-first) ──────────────────────────────
# Absolute expected-LB-delta thresholds, OV units. With target gap ~+0.0130:
#   +0.0010  ≈  8% of gap   → NORMAL (lowest tier of "worth a slot")
#   +0.0050  ≈ 38% of gap   → HIGH
#   +0.0100  ≈ 77% of gap   → STRATEGIC (one-shot bet to threshold)
#   +0.0200  ≈ 154% of gap  → STRATEGIC (justifies multi-day compute)
PRIORITY_NORMAL_THRESHOLD    = 0.001
PRIORITY_HIGH_THRESHOLD      = 0.005
PRIORITY_STRATEGIC_THRESHOLD = 0.010

# Target-progress thresholds (informational; not used for action directly)
TARGET_PROGRESS_TINY_THRESHOLD     = 0.05   # <5% of gap → tiny win
TARGET_PROGRESS_PRIORITY_THRESHOLD = 0.30   # >=30% of gap → STRATEGIC bet

# Classes that are typically OOF/Public-LB-tuned churn — low priority even if
# expected_lb_delta > 0, because they tend to overfit the Public-LB-style
# signal rather than add new structural information. They can still SUBMIT if
# already at ready-for-lb, but a fresh smoke / 5-fold launch is PARKed unless
# the lift is at least HIGH-band.
LOW_PRIORITY_CHURN_CLASSES = {
    "weight-refinement",   # Bayes / COBYLA on a fixed safe pool
    "A-rearrangement",     # pure re-weighting of already-trained components
    "server-head-blend",   # cross-arch SGP blend (transfer empirically ~5%)
}

# Classes considered structurally novel / generalization-positive. If novelty
# is "high" and the lift is in the HIGH band, these are escalated to STRATEGIC.
STRATEGIC_CLASSES = {
    "new-mechanism",       # genuinely new architecture / objective / data path
    "B-pure",              # ADD oldtest like-for-like (super-additive history)
}

# Public-LB overfit-risk per class (informational; helps the caller see WHY a
# small expLB might be deprioritized).
PUBLIC_LB_OVERFIT_RISK_HIGH = {
    "weight-refinement", "pseudo-consensus", "hard-per-SN-blend",
    "B-impure", "B-meta", "B-player-style",
}
PUBLIC_LB_OVERFIT_RISK_MEDIUM = {
    "server-head-blend", "A-rearrangement",
}
PUBLIC_LB_OVERFIT_RISK_LOW = {
    "rule_override", "B-pure", "B-feature",
}

# Generalization-score adjustments (0..1 bounded)
GEN_SCORE_BASE = 0.50
GEN_SCORE_NEW_MECHANISM_BONUS = +0.20
GEN_SCORE_CLEAN_FEATURE_BONUS = +0.10   # rule_override / B-pure / B-feature
GEN_SCORE_NOVELTY_BONUS = {"low": 0.0, "medium": +0.05, "high": +0.15}
GEN_SCORE_CHURN_PENALTY = -0.20         # LOW_PRIORITY_CHURN_CLASSES
GEN_SCORE_TOXIC_PENALTY = -0.40
GEN_SCORE_SLICE_PENALTY_SCALE = 50.0    # × abs(slice_pen), capped at -0.30
GEN_SCORE_SLICE_PENALTY_CAP = -0.30
GEN_SCORE_HOLDOUT_BONUS = +0.10
GEN_SCORE_HOLDOUT_PENALTY = -0.10

# Goal-score weighting: how much generalization deviation from 0.5 amplifies
# (positive) or dampens (negative) the raw expected_lb_delta.
GOAL_SCORE_GEN_WEIGHT = 0.5


# ─── Helpers ────────────────────────────────────────────────────────────────

def _compute_ov_lift(d: Dict[str, Any]) -> float:
    """Compute OOF OV lift from per-task deltas if `OV` not directly supplied."""
    if d.get("OV") is not None:
        return float(d["OV"])
    f1_a = float(d.get("F1_a") or 0.0)
    f1_p = float(d.get("F1_p") or 0.0)
    auc  = float(d.get("AUC")  or 0.0)
    return 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc


def _check_hard_blocks(cand: Dict[str, Any]) -> Optional[str]:
    """Return the FIRST hard-blocker reason found, or None."""
    guards = cand.get("guards", {}) or {}
    cls = cand.get("class", "unknown")
    stage = cand.get("stage", "preflight")
    novelty = cand.get("novelty", "low")

    # 1. Leakage (any form). Each maps to a hard block; the verdict's
    #    leakage_risk field will independently report CRITICAL.
    if guards.get("uses_test_sgp_truth"):
        return "uses_test_sgp_truth → test SGP truth leak"
    if guards.get("overwrites_test_sgp_truth"):
        return "overwrites_test_sgp_truth → direct test SGP overwrite"
    if guards.get("sgp_derived_proxy"):
        return "sgp_derived_proxy → indirect SGP-derived feature leak"
    if guards.get("forbidden_rally_uid_inference"):
        return "forbidden_rally_uid_inference → rally_uid / row-order leak"
    if guards.get("teammate_leak_artifact"):
        return "teammate_leak_artifact → contaminated teammate artifact used as label/feature"
    if guards.get("external_leak_data"):
        return "external_leak_data → external dataset overlaps test distribution"

    # 2. OOF/test alignment unverified.
    # For preflight/smoke a candidate may not have final test arrays yet, so a
    # missing flag is tolerated. Once a candidate reaches 5fold review or LB
    # materialization, the audit must be explicitly True.
    alignment = guards.get("oof_test_alignment_validated")
    if alignment is False:
        return "oof_test_alignment_validated=False → no alignment audit"
    if stage in {"5fold", "ready-for-lb"} and alignment is not True:
        return "oof_test_alignment_validated missing/unknown for materialization-stage candidate"

    # 3. Toxic class without explicit new-mechanism review
    if cls in TOXIC_CLASSES:
        if novelty != "high":
            return f"class={cls} in toxic set; novelty='{novelty}' (need 'high' + Codex new-mechanism review)"
        # Even with novelty=="high", require explicit Codex new-mechanism flag
        if not guards.get("codex_new_mechanism_reviewed", False):
            return f"class={cls} in toxic set; codex_new_mechanism_reviewed not set"

    # 4. 5-fold launch without Codex approval
    if stage == "5fold" and not guards.get("codex_5fold_approved", False):
        return "stage=5fold but codex_5fold_approved=False"

    # 5. LB candidate without rule_override (when applicable)
    if stage == "ready-for-lb":
        if guards.get("rule_override_applicable", True) and not guards.get("rule_override_applied", False):
            return "stage=ready-for-lb but rule_override_applied=False (applicable candidate)"

    # 6. P11 directly optimized
    if guards.get("p11_directly_optimized", False):
        return "p11_directly_optimized=True → holdout-overfit risk; needs separate review"

    return None


def _slice_penalty(cand: Dict[str, Any]) -> float:
    """Sum penalties for SN-bucket regressions + canary class drops."""
    sp = (cand.get("slice_penalties") or {})
    penalty = 0.0
    for bucket in sp.get("sn_bucket_regressions", []) or []:
        d = float(bucket.get("delta_OV", 0.0))
        if d <= SN_BUCKET_REGRESSION_THRESHOLD:
            penalty += SN_BUCKET_REGRESSION_PENALTY
    for cls in sp.get("canary_class_drops", []) or []:
        d = float(cls.get("delta_F1", 0.0))
        if d <= CANARY_CLASS_DROP_THRESHOLD:
            penalty += CANARY_CLASS_DROP_PENALTY
    return penalty


def _holdout_signal(cand: Dict[str, Any]) -> float:
    """Advisory — small bonus/penalty based on holdout OV delta."""
    hd = cand.get("holdout_delta") or {}
    ov = hd.get("OV")
    if ov is None:
        # Try to derive from per-task
        if any(hd.get(k) is not None for k in ("F1_a", "F1_p", "AUC")):
            ov = _compute_ov_lift(hd)
        else:
            return 0.0
    if ov >= HOLDOUT_POSITIVE_THRESHOLD:
        return HOLDOUT_POSITIVE_BONUS
    if ov <= HOLDOUT_NEGATIVE_THRESHOLD:
        return HOLDOUT_NEGATIVE_PENALTY
    return 0.0


def _compute_penalty(cand: Dict[str, Any]) -> float:
    hours = float(cand.get("compute_cost_hours") or 0.0)
    if hours <= COMPUTE_FREE_HOURS:
        return 0.0
    return COMPUTE_PENALTY_PER_HOUR * (hours - COMPUTE_FREE_HOURS)


def _novelty_bonus(cand: Dict[str, Any]) -> float:
    return NOVELTY_BONUS.get(cand.get("novelty", "low"), 0.0)


def _leakage_risk(cand: Dict[str, Any]) -> str:
    """Classify leakage risk. CRITICAL flags are also hard-blocked upstream."""
    g = cand.get("guards", {}) or {}
    critical_flags = (
        "uses_test_sgp_truth",
        "overwrites_test_sgp_truth",
        "sgp_derived_proxy",
        "forbidden_rally_uid_inference",
        "teammate_leak_artifact",
        "external_leak_data",
    )
    if any(g.get(k) for k in critical_flags):
        return "CRITICAL"
    align = g.get("oof_test_alignment_validated")
    if align is False:
        return "HIGH"
    if g.get("p11_directly_optimized"):
        return "HIGH"
    if align is None:
        return "MEDIUM"  # unverified, but not actively claimed leaky
    return "LOW"


def _public_lb_overfit_risk(cand: Dict[str, Any]) -> str:
    """Classify Public-LB overfit risk. Higher = candidate looks good on
    OOF/Public-LB-style signal but more likely to fail on Private/Final."""
    cls = cand.get("class", "unknown")
    novelty = cand.get("novelty", "low")
    if cls in PUBLIC_LB_OVERFIT_RISK_HIGH:
        return "HIGH"
    if cls in PUBLIC_LB_OVERFIT_RISK_MEDIUM:
        return "MEDIUM"
    if cls in PUBLIC_LB_OVERFIT_RISK_LOW:
        return "LOW"
    if cls == "new-mechanism":
        return "LOW" if novelty == "high" else "MEDIUM"
    return "MEDIUM"  # cautious default for "unknown"


def _generalization_score(
    cand: Dict[str, Any],
    slice_pen: float,
    holdout_sig: float,
) -> float:
    """0..1 score; higher = more likely to transfer to Private / Final.

    Combines class prior, novelty, slice-profile cleanliness, and holdout
    direction. The score is advisory and feeds the goal_score weighting.
    """
    cls = cand.get("class", "unknown")
    novelty = cand.get("novelty", "low")

    score = GEN_SCORE_BASE

    # Class contribution
    if cls == "new-mechanism":
        score += GEN_SCORE_NEW_MECHANISM_BONUS
    elif cls in {"B-pure", "B-feature", "rule_override"}:
        score += GEN_SCORE_CLEAN_FEATURE_BONUS
    elif cls in LOW_PRIORITY_CHURN_CLASSES:
        score += GEN_SCORE_CHURN_PENALTY
    elif cls in TOXIC_CLASSES:
        score += GEN_SCORE_TOXIC_PENALTY
    # server-head-blend (which is in churn set) already penalised above

    # Novelty
    score += GEN_SCORE_NOVELTY_BONUS.get(novelty, 0.0)

    # Slice-profile penalty (more / larger regressions → lower gen score)
    if slice_pen < 0:
        slice_adj = max(GEN_SCORE_SLICE_PENALTY_SCALE * slice_pen,
                        GEN_SCORE_SLICE_PENALTY_CAP)
        score += slice_adj

    # Holdout signal (advisory; never a gate)
    if holdout_sig > 0:
        score += GEN_SCORE_HOLDOUT_BONUS
    elif holdout_sig < 0:
        score += GEN_SCORE_HOLDOUT_PENALTY

    return max(0.0, min(1.0, score))


def _target_progress(expected_lb_delta: float, anchor_lb: float) -> float:
    """Fraction of remaining gap to TARGET_LB this candidate would cover.

    >0 if candidate makes progress, 0 if flat, <0 if it regresses, >1 if it
    overshoots the target on its own.
    """
    gap = TARGET_LB - anchor_lb
    if gap <= 0:
        return 1.0  # already at/past target — any further win is "bonus"
    return expected_lb_delta / gap


def _priority_level(
    cand: Dict[str, Any],
    expected_lb_delta: float,
    gen_score: float,
) -> str:
    """STRATEGIC / HIGH / NORMAL / LOW / PARK.

    Generalization-first: low-priority churn classes are capped at LOW even
    if they look mildly positive. Structurally novel high-novelty candidates
    in the HIGH band are promoted to STRATEGIC.
    """
    cls = cand.get("class", "unknown")
    novelty = cand.get("novelty", "low")

    if expected_lb_delta <= 0:
        return "PARK"

    # Tiny non-zero — only worth doing if zero-cost / diagnostic
    if expected_lb_delta < PRIORITY_NORMAL_THRESHOLD:
        return "LOW"

    # Low-priority churn classes capped unless lift is in HIGH+ band
    if cls in LOW_PRIORITY_CHURN_CLASSES and novelty != "high":
        if expected_lb_delta < PRIORITY_HIGH_THRESHOLD:
            return "LOW"
        if expected_lb_delta < PRIORITY_STRATEGIC_THRESHOLD:
            return "NORMAL"
        return "HIGH"

    # Structural-novel + high novelty + HIGH-band lift → STRATEGIC
    if cls in STRATEGIC_CLASSES and novelty == "high":
        if expected_lb_delta >= PRIORITY_HIGH_THRESHOLD:
            return "STRATEGIC"

    if expected_lb_delta >= PRIORITY_STRATEGIC_THRESHOLD:
        return "STRATEGIC"
    if expected_lb_delta >= PRIORITY_HIGH_THRESHOLD:
        return "HIGH" if gen_score >= GEN_SCORE_BASE else "NORMAL"
    if expected_lb_delta >= PRIORITY_NORMAL_THRESHOLD:
        return "NORMAL"
    return "LOW"


def _catastrophic_collapse_check(cand: Dict[str, Any]) -> Optional[str]:
    """v0.4 sanity check — does the smoke artifact show catastrophic failure?

    Returns a reason string if catastrophic, None if smoke is acceptable for
    HIGH/STRATEGIC LB probe. NORMAL/LOW candidates use the normal slice-penalty
    + holdout machinery instead.

    Catastrophic patterns (any one triggers PROVISIONAL_FAIL):
      - Full OV drops worse than CATASTROPHIC_OV_DROP (-0.020)
      - F1_a or F1_p drops worse than CATASTROPHIC_F1_DROP (-0.030)
      - AUC drops worse than CATASTROPHIC_AUC_DROP (-0.030)
      - SEVERE_CANARY_CLASS_COUNT+ classes drop > SEVERE_CANARY_CLASS_DROP
      - SEVERE_SN_BUCKET_COUNT+ SN buckets drop > SEVERE_SN_BUCKET_DROP
    """
    d = cand.get("oof_delta") or {}
    ov = d.get("OV")
    f1_a = d.get("F1_a")
    f1_p = d.get("F1_p")
    auc = d.get("AUC")

    if ov is not None and ov <= CATASTROPHIC_OV_DROP:
        return f"catastrophic OV drop: {ov:+.4f} ≤ {CATASTROPHIC_OV_DROP}"
    if f1_a is not None and f1_a <= CATASTROPHIC_F1_DROP:
        return f"catastrophic F1_a drop: {f1_a:+.4f} ≤ {CATASTROPHIC_F1_DROP}"
    if f1_p is not None and f1_p <= CATASTROPHIC_F1_DROP:
        return f"catastrophic F1_p drop: {f1_p:+.4f} ≤ {CATASTROPHIC_F1_DROP}"
    if auc is not None and auc <= CATASTROPHIC_AUC_DROP:
        return f"catastrophic AUC drop: {auc:+.4f} ≤ {CATASTROPHIC_AUC_DROP}"

    sp = (cand.get("slice_penalties") or {})
    severe_canaries = [
        c for c in (sp.get("canary_class_drops") or [])
        if float(c.get("delta_F1", 0.0)) <= SEVERE_CANARY_CLASS_DROP
    ]
    if len(severe_canaries) >= SEVERE_CANARY_CLASS_COUNT:
        names = ", ".join(c.get("class", "?") for c in severe_canaries[:5])
        return (f"severe canary regression: {len(severe_canaries)} class(es) ≤ "
                f"{SEVERE_CANARY_CLASS_DROP} F1 ({names})")

    severe_sn = [
        b for b in (sp.get("sn_bucket_regressions") or [])
        if float(b.get("delta_OV", 0.0)) <= SEVERE_SN_BUCKET_DROP
    ]
    if len(severe_sn) >= SEVERE_SN_BUCKET_COUNT:
        names = ", ".join(b.get("bucket", "?") for b in severe_sn[:5])
        return (f"severe SN regression: {len(severe_sn)} bucket(s) ≤ "
                f"{SEVERE_SN_BUCKET_DROP} OV ({names})")

    return None


def _classify_risk(expected_lb_delta: float, cls: str, stage: str) -> str:
    if expected_lb_delta <= PARK_THRESHOLD:
        return "HIGH"
    if expected_lb_delta < 0:
        return "MEDIUM"
    # Class-specific caution: server-head-blend has known weak transfer
    if cls == "server-head-blend":
        return "LOW" if expected_lb_delta >= 0 else "MEDIUM"
    return "LOW"


def _recommend_action(
    hard_block: bool,
    expected_lb_delta: float,
    stage: str,
    guards: Dict[str, Any],
    priority: str,
    cls: str,
    cand: Dict[str, Any] = None,
) -> str:
    """v0.4 theory-first action mapping.

    Key change vs v0.3: a marginal Fold-1 OOF miss does NOT auto-kill a
    HIGH/STRATEGIC candidate whose theoretical generalization story is sound.
    Only catastrophic collapse (5 specific patterns) blocks LB probe.

    Rules:
      * PARK if priority == PARK (expLB ≤ 0)
      * Low-priority churn classes: never burn fresh compute; SUBMIT only
        if already at ready-for-lb (sunk cost)
      * Tiny LOW lift any class: SUBMIT only at ready-for-lb
      * NORMAL: proceed through normal stage gate
      * HIGH/STRATEGIC: PROVISIONAL_PASS at smoke/5fold/ready-for-lb
        UNLESS catastrophic collapse detected → PROVISIONAL_FAIL
    """
    if hard_block:
        return "BLOCK"

    if priority == "PARK":
        return "PARK"

    # Low-priority churn: never escalate a fresh smoke/5fold; SUBMIT only sunk cost.
    if priority == "LOW" and cls in LOW_PRIORITY_CHURN_CLASSES:
        if stage == "ready-for-lb" and expected_lb_delta > SUBMIT_THRESHOLD:
            return "SUBMIT_CANDIDATE"
        return "PARK"

    if priority == "LOW":
        if stage == "ready-for-lb" and expected_lb_delta > SUBMIT_THRESHOLD:
            return "SUBMIT_CANDIDATE"
        return "PARK"

    # v0.4: HIGH / STRATEGIC use catastrophic-collapse sanity check, NOT expLB
    if priority in {"HIGH", "STRATEGIC"} and cand is not None:
        catastrophic = _catastrophic_collapse_check(cand)
        if catastrophic is not None:
            return "PROVISIONAL_FAIL"
        # Smoke is sanity-only for HIGH/STRATEGIC; theory deserves an LB probe
        if stage == "preflight":
            return "SMOKE_ONLY"
        # smoke / 5fold / ready-for-lb: all collapse onto "PROVISIONAL_PASS"
        # (sanity OK, materialize CSV, hand to Jabir for manual LB upload)
        return "PROVISIONAL_PASS"

    # NORMAL — keep v0.3 stage gate (smoke→5fold→materialize→submit)
    if stage == "preflight":
        return "SMOKE_ONLY"
    if stage == "smoke":
        if expected_lb_delta <= SUBMIT_THRESHOLD:
            return "PARK"
        return "FULL_5FOLD_REVIEW"
    if stage == "5fold":
        if expected_lb_delta <= SUBMIT_THRESHOLD:
            return "PARK"
        return "MATERIALIZE_FOR_REVIEW"
    if stage == "ready-for-lb":
        if expected_lb_delta <= SUBMIT_THRESHOLD:
            return "PARK"
        return "SUBMIT_CANDIDATE"
    return "SMOKE_ONLY"  # cautious default


# ─── Public API ─────────────────────────────────────────────────────────────

def score_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Score a candidate and return a verdict dict.

    See GOAL_FUNCTION.md §2 for the input schema and §3-§6 for the scoring
    formula. Heuristic v0.2; not mathematically proven. Generalization-first
    (target LB ≥ 0.4000), deprioritizes Public-LB overfit churn.
    """
    rid = cand.get("rid", "<unknown>")
    cls = cand.get("class", "unknown")
    stage = cand.get("stage", "preflight")
    anchor_lb = float(cand.get("anchor_lb", ANCHOR_LB_DEFAULT))
    gap_to_target = TARGET_LB - anchor_lb

    # 1. Hard blockers — leakage, alignment, toxic class, etc.
    block_reason = _check_hard_blocks(cand)
    if block_reason is not None:
        return {
            "rid": rid,
            "hard_block": True,
            "block_reason": block_reason,
            "expected_lb_delta": 0.0,
            "goal_score": 0.0,
            "risk_level": "BLOCKED",
            "leakage_risk": _leakage_risk(cand),
            "public_lb_overfit_risk": _public_lb_overfit_risk(cand),
            "generalization_score": 0.0,
            "priority": "PARK",
            "target_progress_ratio": 0.0,
            "recommended_action": "BLOCK",
            "explanation": f"BLOCKED: {block_reason}",
            "_diag": {
                "anchor_lb": float(anchor_lb),
                "target_lb": float(TARGET_LB),
                "current_gap_to_target": float(gap_to_target),
            },
        }

    # 2. Base lift
    oof_delta = cand.get("oof_delta") or {}
    base_lift_ov = _compute_ov_lift(oof_delta)

    # 3. Transfer multiplier (class-specific empirical prior)
    multiplier = TRANSFER_MULTIPLIER.get(cls, TRANSFER_MULTIPLIER["unknown"])
    expected_lb_delta_pre = base_lift_ov * multiplier

    # 4. Slice penalty
    slice_pen = _slice_penalty(cand)

    # 5. Holdout signal (advisory)
    holdout_sig = _holdout_signal(cand)

    # 6. Compute penalty
    comp_pen = _compute_penalty(cand)

    # 7. Novelty bonus
    nov_bonus = _novelty_bonus(cand)

    # 8. Final expected LB delta (in OV units)
    expected_lb_delta = expected_lb_delta_pre + slice_pen + holdout_sig

    # 9. Generalization + leakage + Public-LB-overfit signals
    gen_score   = _generalization_score(cand, slice_pen, holdout_sig)
    leak_risk   = _leakage_risk(cand)
    pub_overfit = _public_lb_overfit_risk(cand)

    # 10. Target progress + priority bucket
    target_progress = _target_progress(expected_lb_delta, anchor_lb)
    priority = _priority_level(cand, expected_lb_delta, gen_score)

    # 11. Goal score — generalization-weighted version of expected_lb_delta.
    #     Positive lift is amplified by (gen_score - 0.5); negative by the
    #     same factor (so a high gen_score cushions small regressions while a
    #     low gen_score penalises good-looking but flimsy candidates).
    gen_boost = GOAL_SCORE_GEN_WEIGHT * expected_lb_delta * (gen_score - GEN_SCORE_BASE)
    goal_score = expected_lb_delta + comp_pen + nov_bonus + gen_boost

    # 12. Risk + action (action uses priority + class for churn down-weighting)
    guards = cand.get("guards", {}) or {}
    risk = _classify_risk(expected_lb_delta, cls, stage)
    # v0.4: pass cand so HIGH/STRATEGIC can run the catastrophic-collapse check
    action = _recommend_action(False, expected_lb_delta, stage, guards,
                                priority, cls, cand=cand)

    # v0.4: sanity verdict for ALL stages — useful diagnostic, not blocking
    # for NORMAL/LOW (those use the standard slice penalty)
    catastrophic_reason = _catastrophic_collapse_check(cand)
    smoke_sanity_pass = (catastrophic_reason is None)

    # 13. Explanation (compact one-liner)
    parts = [
        f"class={cls}",
        f"OOF_OV={base_lift_ov:+.4f}",
        f"xfer={multiplier:.2f}",
        f"pre={expected_lb_delta_pre:+.4f}",
    ]
    if slice_pen != 0:
        parts.append(f"slice={slice_pen:+.4f}")
    if holdout_sig != 0:
        parts.append(f"holdout={holdout_sig:+.4f}")
    if comp_pen != 0:
        parts.append(f"compute={comp_pen:+.4f}")
    if nov_bonus != 0:
        parts.append(f"novelty={nov_bonus:+.4f}")
    parts.append(f"expLB={expected_lb_delta:+.4f}")
    parts.append(f"progress={target_progress*100:+.1f}%")
    parts.append(f"gen={gen_score:.2f}")
    parts.append(f"leak={leak_risk}")
    parts.append(f"pubLBoverfit={pub_overfit}")
    parts.append(f"priority={priority}")
    explanation = " | ".join(parts)

    # v0.4: lb_probe_worthy decision
    #   HIGH/STRATEGIC + sanity pass + slot available → True
    #   PROVISIONAL_PASS / SUBMIT_CANDIDATE / MATERIALIZE_FOR_REVIEW → True
    #   Otherwise → False
    lb_probe_actions = {
        "PROVISIONAL_PASS", "SUBMIT_CANDIDATE", "MATERIALIZE_FOR_REVIEW",
    }
    lb_probe_worthy = (action in lb_probe_actions) and smoke_sanity_pass

    return {
        "rid": rid,
        "hard_block": False,
        "block_reason": "",
        "expected_lb_delta": float(expected_lb_delta),
        "goal_score": float(goal_score),
        "risk_level": risk,
        "leakage_risk": leak_risk,
        "public_lb_overfit_risk": pub_overfit,
        "generalization_score": float(gen_score),
        "priority": priority,
        "target_progress_ratio": float(target_progress),
        "recommended_action": action,
        "explanation": explanation,
        # v0.4 report fields (free-text, populated by the caller — defaults if absent)
        "theoretical_generalization_reason":
            cand.get("theoretical_generalization_reason",
                     "(not provided — populate this field in the candidate dict)"),
        "why_transfers_to_test_new":
            cand.get("why_transfers_to_test_new",
                     "(not provided — populate this field in the candidate dict)"),
        "smoke_sanity_pass": bool(smoke_sanity_pass),
        "smoke_sanity_reason": catastrophic_reason or "OK",
        "lb_probe_worthy": bool(lb_probe_worthy),
        "lb_confirm_hypothesis":
            cand.get("lb_confirm_hypothesis",
                     "LB ΔOV ≥ expLB ⇒ mechanism transfers as predicted"),
        "lb_reject_hypothesis":
            cand.get("lb_reject_hypothesis",
                     "LB ΔOV < 0 ⇒ mechanism does NOT transfer (treat as toxic-class evidence)"),
        "lb_result": cand.get("lb_result", None),    # None until manual LB
        "final_verdict": cand.get("final_verdict", "PROVISIONAL"),
        "_diag": {
            "base_lift_OV": float(base_lift_ov),
            "transfer_multiplier": float(multiplier),
            "slice_penalty": float(slice_pen),
            "holdout_signal": float(holdout_sig),
            "compute_penalty": float(comp_pen),
            "novelty_bonus": float(nov_bonus),
            "anchor_lb": float(anchor_lb),
            "target_lb": float(TARGET_LB),
            "current_gap_to_target": float(gap_to_target),
            "target_progress_ratio": float(target_progress),
            "implied_predicted_lb": float(anchor_lb + expected_lb_delta),
            "generalization_score": float(gen_score),
            "gen_boost": float(gen_boost),
        },
    }


# ─── Self-tests ─────────────────────────────────────────────────────────────

EXAMPLES = [
    # 1. Historical R-034 (B-feature blend swap, ended up +0.0028 LB)
    {
        "rid": "R-034-historical",
        "name": "v14_seed2 → v14_seed2_v15feat_a swap (R-034)",
        "tier": "T2-component",
        "class": "B-feature",
        "stage": "ready-for-lb",
        "oof_delta": {"OV": -0.0005},   # in-blend OOF was actually tied/slightly negative
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "oof_test_alignment_validated": True,
            "rule_override_applied": True, "rule_override_applicable": True,
            "codex_smoke_approved": True, "codex_5fold_approved": True,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 3.0,
        "novelty": "low",
    },
    # 2. R-042 rule_override (proven +0.0028 LB)
    {
        "rid": "R-042-historical",
        "name": "R-034 + rule_override post-process",
        "tier": "T2-component",
        "class": "rule_override",
        "stage": "ready-for-lb",
        "oof_delta": {"OV": 0.0028},
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "oof_test_alignment_validated": True,
            "rule_override_applied": True, "rule_override_applicable": True,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 0.1,
        "novelty": "medium",
    },
    # 3. R-067c historical (server-head blend, +0.000355 LB)
    {
        "rid": "R-067c-historical",
        "name": "α=0.30 v22 SGP blend with R-042 base",
        "tier": "T2-component",
        "class": "server-head-blend",
        "stage": "ready-for-lb",
        "oof_delta": {"AUC": 0.0326},   # only AUC moved; F1_a/F1_p unchanged
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "oof_test_alignment_validated": True,
            "rule_override_applied": True, "rule_override_applicable": True,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 1.0,
        "novelty": "medium",
    },
    # 4. R-055 historical (B-impure ADD with Bayes — LB-disaster −0.0141)
    {
        "rid": "R-055-historical",
        "name": "R-052 7-comp Bayes + rule (mulminet ADD)",
        "tier": "T2-component",
        "class": "B-impure",
        "stage": "ready-for-lb",
        "oof_delta": {"OV": 0.0058},
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "oof_test_alignment_validated": True,
            "rule_override_applied": True, "rule_override_applicable": True,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 0.5,
        "novelty": "low",
    },
    # 5. R-062r historical (B-player-style v16match_v2 — LB −0.0057)
    {
        "rid": "R-062r-historical",
        "name": "v16match_v2 LORO swap + rule_override",
        "tier": "T2-component",
        "class": "B-player-style",
        "stage": "ready-for-lb",
        "oof_delta": {"OV": 0.0037},
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "oof_test_alignment_validated": True,
            "rule_override_applied": True, "rule_override_applicable": True,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 8.0,
        "novelty": "low",
    },
    # 6. R-070 current 7-feature smoke (mixed SN profile)
    {
        "rid": "R-070-7feat-smoke",
        "name": "v15feat_e 7-feature movement smoke",
        "tier": "T2-component",
        "class": "B-feature",
        "stage": "smoke",
        "oof_delta": {"OV": -0.0010, "F1_a": -0.0016, "F1_p": 0.0019, "AUC": -0.0058},
        "holdout_delta": {"OV": -0.0024},
        "slice_penalties": {
            "sn_bucket_regressions": [
                {"bucket": "SN<=2", "delta_OV": -0.0081},
                {"bucket": "SN>=5", "delta_OV": -0.0043},
            ],
            "canary_class_drops": [],   # cls1, cls9 within 0.015 cap
        },
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "oof_test_alignment_validated": True,
            "codex_smoke_approved": True, "codex_5fold_approved": False,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 0.5,
        "novelty": "medium",
    },
    # 7. R-070 no-mismatch ablation (hypothetical positive smoke)
    {
        "rid": "R-070-5feat-ablation",
        "name": "v15feat_e no-mismatch ablation smoke",
        "tier": "T2-component",
        "class": "B-feature",
        "stage": "smoke",
        "oof_delta": {"OV": 0.0022},
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "oof_test_alignment_validated": True,
            "codex_smoke_approved": True, "codex_5fold_approved": False,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 0.5,
        "novelty": "medium",
    },
    # 8. Hypothetical clean B-feature candidate (new feature module that lifts OOF by +0.0025)
    {
        "rid": "Hypothetical-clean-Bfeature",
        "name": "new B-feature swap into R-034",
        "tier": "T2-component",
        "class": "B-feature",
        "stage": "5fold",
        "oof_delta": {"OV": 0.0025, "F1_a": 0.0030, "F1_p": 0.0020, "AUC": 0.0010},
        "holdout_delta": {"OV": 0.0015},
        "slice_penalties": {
            "sn_bucket_regressions": [],
            "canary_class_drops": [],
        },
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "oof_test_alignment_validated": True,
            "codex_smoke_approved": True, "codex_5fold_approved": True,
            "rule_override_applied": True, "rule_override_applicable": True,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 3.0,
        "novelty": "low",
    },
    # 9. Hypothetical structural-novel mechanism with strong lift (+0.010 expLB)
    #    Should land STRATEGIC priority — covers ~77% of the gap to TARGET_LB.
    {
        "rid": "Hypothetical-structural-novel-+0.010",
        "name": "new causal LM head with per-shot training + clean slice profile",
        "tier": "T2-exploration",
        "class": "new-mechanism",
        "stage": "5fold",
        "oof_delta": {"OV": 0.0125, "F1_a": 0.0130, "F1_p": 0.0125, "AUC": 0.0110},
        "holdout_delta": {"OV": 0.0090},
        "slice_penalties": {
            "sn_bucket_regressions": [],
            "canary_class_drops": [],
        },
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "sgp_derived_proxy": False, "forbidden_rally_uid_inference": False,
            "teammate_leak_artifact": False, "external_leak_data": False,
            "oof_test_alignment_validated": True,
            "codex_smoke_approved": True, "codex_5fold_approved": True,
            "rule_override_applied": True, "rule_override_applicable": True,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 12.0,   # multi-hour bet — justified by +0.010 expLB
        "novelty": "high",
    },
    # 10. Hypothetical tiny weight-refinement churn (+0.0005 expLB on a smoke)
    #     Should LOW-prioritize and PARK — won't move us toward TARGET_LB.
    {
        "rid": "Hypothetical-weight-refinement-tiny",
        "name": "COBYLA on R-034 PAIR safe pool",
        "tier": "T2-component",
        "class": "weight-refinement",
        "stage": "smoke",
        "oof_delta": {"OV": 0.0005},
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "sgp_derived_proxy": False, "forbidden_rally_uid_inference": False,
            "teammate_leak_artifact": False, "external_leak_data": False,
            "oof_test_alignment_validated": True,
            "codex_smoke_approved": True, "codex_5fold_approved": False,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 0.2,
        "novelty": "low",
    },
    # 11. Hypothetical SGP-derived proxy leak — must hard-block.
    {
        "rid": "Hypothetical-sgp-proxy-leak",
        "name": "feature derived from test SGP via proxy",
        "tier": "T2-component",
        "class": "B-feature",
        "stage": "smoke",
        "oof_delta": {"OV": 0.0100},   # would look great on OOF — but leaked
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "sgp_derived_proxy": True,    # ← leak
            "oof_test_alignment_validated": True,
            "codex_smoke_approved": False, "codex_5fold_approved": False,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 1.0,
        "novelty": "medium",
    },
    # 12. NEW v0.3 — R-072 historical (LB -0.0033 from rule_override layers
    #     C/D using handId/positionId). After v0.3 reclassification this is
    #     class=rule_override_player_context → HARD BLOCK.
    {
        "rid": "R-072-historical-LBfailed",
        "name": "rule_override v2 with hand/position context layers (R-072)",
        "tier": "T1",
        "class": "rule_override_player_context",
        "stage": "ready-for-lb",
        "oof_delta": {"OV": 0.0015},   # what we predicted at time of upload
        "guards": {
            "uses_test_sgp_truth": False, "overwrites_test_sgp_truth": False,
            "sgp_derived_proxy": False, "forbidden_rally_uid_inference": False,
            "teammate_leak_artifact": False, "external_leak_data": False,
            "oof_test_alignment_validated": True,
            "rule_override_applied": True, "rule_override_applicable": True,
            "p11_directly_optimized": False,
        },
        "compute_cost_hours": 0.5,
        "novelty": "medium",
    },
]


def _self_test() -> None:
    # (action, priority) expected per example
    expected = {
        "R-034-historical":                  ("PARK",                    "PARK"),
        "R-042-historical":                  ("SUBMIT_CANDIDATE",        "NORMAL"),
        "R-067c-historical":                 ("SUBMIT_CANDIDATE",        "LOW"),
        "R-055-historical":                  ("BLOCK",                   "PARK"),
        "R-062r-historical":                 ("BLOCK",                   "PARK"),
        "R-070-7feat-smoke":                 ("PARK",                    "PARK"),
        "R-070-5feat-ablation":              ("FULL_5FOLD_REVIEW",       "NORMAL"),
        "Hypothetical-clean-Bfeature":       ("MATERIALIZE_FOR_REVIEW",  "NORMAL"),
        "Hypothetical-structural-novel-+0.010": ("PROVISIONAL_PASS",       "STRATEGIC"),
        "Hypothetical-weight-refinement-tiny":  ("PARK",                  "LOW"),
        "Hypothetical-sgp-proxy-leak":          ("BLOCK",                 "PARK"),
        "R-072-historical-LBfailed":            ("BLOCK",                 "PARK"),
    }
    print("=" * 175)
    print(" Candidate Goal Function v0.6 -- self-tests (target LB >= 0.4000)")
    print("=" * 175)
    header = (f"{'rid':<40} {'class':<20} {'expLB':>8} {'progress':>9} "
              f"{'gen':>5} {'priority':<10} {'action':<24} {'leak':<9} "
              f"{'pubLB-overfit':<14} score")
    print(header)
    print("-" * 175)
    failures = []
    for c in EXAMPLES:
        v = score_candidate(c)
        exp_action, exp_priority = expected[v["rid"]]
        if v["recommended_action"] != exp_action:
            failures.append(
                f"{v['rid']}: expected action={exp_action}, got {v['recommended_action']}"
            )
        if v["priority"] != exp_priority:
            failures.append(
                f"{v['rid']}: expected priority={exp_priority}, got {v['priority']}"
            )
        progress_pct = f"{v['target_progress_ratio']*100:+.1f}%"
        print(f"{v['rid']:<40} {c.get('class','?'):<20} "
              f"{v['expected_lb_delta']:+.4f} {progress_pct:>9} "
              f"{v['generalization_score']:.2f} {v['priority']:<10} "
              f"{v['recommended_action']:<24} {v['leakage_risk']:<9} "
              f"{v['public_lb_overfit_risk']:<14} {v['goal_score']:+.4f}")
    if failures:
        print()
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        raise AssertionError(f"{len(failures)} self-test failures")
    print()
    print(f"All {len(EXAMPLES)} examples passed action+priority assertions.")
    print(f"Anchor LB: {ANCHOR_LB_DEFAULT:.4f}   Target LB: {TARGET_LB:.4f}   "
          f"Gap: +{TARGET_LB - ANCHOR_LB_DEFAULT:.4f}")


if __name__ == "__main__":
    _self_test()
