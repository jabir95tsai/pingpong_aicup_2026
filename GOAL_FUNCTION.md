# Candidate Goal Function v0.6 (B-impure-additive-low-weight is toxic)

**Status**: HEURISTIC v0.4 — empirically grounded but not mathematically proven.
Coefficients should be re-tuned after every 3-5 new LB datapoints.

**v0.4 change (2026-05-26 — user policy directive)**: Decision priority is now:

> 1. Theoretical generalization mechanism comes first.
> 2. Fold-1 smoke is a sanity check, not the final judge.
> 3. Only manual LB submission can confirm real transfer.
> 4. Do not declare a candidate WIN/FAIL until LB result exists.

**For HIGH / STRATEGIC candidates**: the main decision point is "In theory, this
mechanism should improve real generalization ability." A marginal Fold-1 OOF
miss does NOT auto-kill a theoretically strong candidate. Smoke only blocks
on: (a) leakage / rule violation, (b) broken pipeline / alignment bug,
(c) NaN / Inf / invalid submission, (d) catastrophic collapse, (e) severe
per-class or per-SN regression.

**For LOW priority churn**: Fold-1 smoke gates stay strict.

**Verdict vocabulary** (extended):
- `PROVISIONAL_PASS` — sanity OK + theory strong; recommended for LB probe.
- `PROVISIONAL_FAIL` — catastrophic smoke OR clear theory hole OR leak.
- `ARTIFACT_READY_FOR_JABIR_UPLOAD_REVIEW` — CSV materialized, waiting for manual LB.
- `LB_WIN` / `LB_FAIL` / `LB_NOISE` — set ONLY after manual LB result.

**v0.3 change (2026-05-26)**: R-072 LB-failed (−0.0033) proved `rule_override`
is NOT a uniform 1.0 transfer class. Layers with `handId`/`positionId` in the
context reproduce the B-player-style failure mode. Split into:
- `rule_override` (R-042 shot-content only): proven 1.0 transfer.
- `rule_override_deep_prefix` (deeper shot-context): 0.3 conservative.
- `rule_override_player_context` (handId/positionId in context): **HARD BLOCK**.

**Purpose**: provide a reusable, transparent decision function for any proposed
experiment, component, blend, or submission candidate. Outputs a hard-block
boolean, a numeric priority score, a rough expected LB delta against the
current LB-best, a risk level, an explicit **leakage risk**, an explicit
**Public-LB-overfit risk**, a **generalization score**, a **target-progress
ratio**, a **priority bucket** (STRATEGIC / HIGH / NORMAL / LOW / PARK), and
a recommended next action.

This is a **decision aid**, not an oracle. It is calibrated on post-2026-05-06
LB history (the new `test_new.csv` era). It must be re-read and adjusted as
more LB datapoints land.

---

## 0a. Candidate report template (v0.4 required fields)

For every R-NNN candidate, Claude/Codex must report these six fields when
opening, materializing, or LB-uploading:

1. **theoretical_generalization_reason** — Why the mechanism *should* improve
   real generalization (not just OOF). One paragraph.
2. **why_transfers_to_test_new** — Why this should work on the post-2026-05-06
   `test_new.csv` distribution specifically (not just the OLD test). Address
   distribution-shift concerns explicitly.
3. **leakage_safety** — Explicit check against each leak rule:
   no test SGP truth, no SGP-derived proxy, no `rally_uid`/order inference, no
   teammate leak artifacts as labels, no player-profile features, no V15
   hist/streak family. Each gets `True/False/N/A`.
4. **fold1_smoke_result** — Numbers (global ΔOV, per-SN ΔOV, per-class F1 deltas,
   holdout ΔOV, AUC delta if applicable) **as sanity check only** for
   HIGH/STRATEGIC. For NORMAL/LOW it's still a decision input.
5. **lb_probe_worthy** — Boolean. True if (priority ≥ NORMAL) AND
   (smoke_sanity_pass) AND (slot available) AND (theoretical reason exists).
6. **lb_confirm_hypothesis** / **lb_reject_hypothesis** — Specific predictions:
   "LB ΔOV ≥ X ⇒ theory confirmed"; "LB ΔOV ≤ Y ⇒ theory rejected; treat as
   new toxic-class evidence."

The candidate dict input may include any of these as free-text fields; the
`score_candidate` verdict will populate them in the output even when not
supplied by the caller.

**Final verdict transitions** (audit trail per candidate):
- `PROVISIONAL` (pre-LB) → `LB_WIN` / `LB_FAIL` / `LB_NOISE` (post-LB only)
- A candidate is NEVER called WIN/FAIL until manual LB result lands.

---

## 0. Primary project goal

> **Reach clean NEW LB ≥ 0.4000**, but **not** by overfitting Public LB. We
> prioritize generalization to the Private / Final distribution and strictly
> forbid leakage.

* Current clean LB-best anchor: **R-067c = 0.3870095** (post-rule_override).
* Target: **0.4000**.
* Required gap: **+0.0130** (in OV units).
* Priority guidance (by **expected** LB delta, not OOF lift):
  * `< +0.001` (≈ <8% of gap) → **LOW** priority, only worth it if zero-cost or diagnostic.
  * `+0.001 … +0.005` → **NORMAL** (covers 8–38% of gap).
  * `+0.005 … +0.010` → **HIGH** (covers 38–77% of gap).
  * `≥ +0.010` → **STRATEGIC** (one-shot bet that could hit target on its own).
  * `≥ +0.020` → **STRATEGIC** and justifies multi-day compute.

A candidate is **not** promoted just because it gives a tiny OOF / Public-LB
gain. It needs evidence of generalizable signal:

1. structurally new mechanism, **or**
2. clean feature with stable slice profile, **or**
3. player-disjoint / P11 holdout not strongly negative, **or**
4. strong task-specific lift with plausible transfer, **or**
5. external / legal data that improves distribution coverage.

## 1. Purpose

Avoid the following recurring failure modes:

1. **OOF-overfit acceptance**: large OOF gain interpreted as guaranteed LB win
   (R-055 was +0.0058 OOF, LB −0.0141).
2. **Toxic-class re-litigation**: re-running B-impure / B-meta / B-player-style
   variants hoping a tweak rescues the family (R-040, R-054r).
3. **Slice-blind global pass**: global Fold-1 PASS while a major SN bucket
   regresses badly (R-070 7-feature smoke).
4. **Weight-refinement overfit**: Bayes/COBYLA picking weights that beat OOF
   then collapse on LB or P11 (R-055; R-068 dOV +0.0012 full / −0.0001 holdout).
5. **Holdout overpromotion**: turning P11 into a hard gate and rejecting
   in-blend-useful components (v11_aug_oldtest is standalone holdout-negative
   but works in R-034).
6. **Tiny-win churn** (new in v0.2): spending slots / compute on
   weight-refinement, A-rearrangement, or server-head sub-tuning when the
   expected LB delta cannot meaningfully close the gap to TARGET_LB.
7. **Leak slipping in through proxies** (new in v0.2): non-direct leakage
   such as SGP-derived features, rally_uid / row-order inference, contaminated
   teammate artifacts as training labels/features, or overlapping external
   datasets.

Every candidate the agent considers — feature module, blend variant, post-process,
pseudo-label, weight refinement, training plan — should be scored by this
function before compute is spent or a slot is burned.

## 2. Inputs

A candidate is a dict with the following fields (see `src/candidate_goal.py`
for the canonical schema). All fields optional unless noted; missing fields
default to neutral assumptions, and the recommended action becomes more
conservative when key fields are absent.

```python
{
  "rid":            "R-NNN | descriptive-name",
  "name":           "short human label",
  "tier":           "T1 | T2-component | T2-exploration | T3",
  "class":          "rule_override | B-pure | B-feature | server-head-blend "
                    "| weight-refinement | A-rearrangement | new-mechanism "
                    "| B-impure | B-meta | B-player-style | pseudo-consensus "
                    "| hard-per-SN-blend | unknown",
  "stage":          "preflight | smoke | 5fold | ready-for-lb",
  "anchor_lb":      0.3870095,   # default: R-067c LB-best
  "oof_delta": {                 # vs anchor; absolute units of the metric
      "F1_a": 0.0, "F1_p": 0.0, "AUC": 0.0, "OV": None   # OV computed if None
  },
  "holdout_delta": {             # OPTIONAL — advisory only
      "F1_a": None, "F1_p": None, "AUC": None, "OV": None
  },
  "slice_penalties": {           # OPTIONAL — per-SN / canary regressions
      "sn_bucket_regressions": [{"bucket": "SN<=2", "delta_OV": -0.0081}, ...],
      "canary_class_drops":    [{"class": "cls1_loop", "delta_F1": -0.005}, ...]
  },
  "guards": {                    # boolean preconditions
      # Leakage guards — any True → hard-block, leakage_risk=CRITICAL
      "uses_test_sgp_truth":             False,   # direct test SGP truth
      "overwrites_test_sgp_truth":       False,   # post-process overwrite of test SGP
      "sgp_derived_proxy":               False,   # indirect SGP-derived feature
      "forbidden_rally_uid_inference":   False,   # rally_uid / row-order inference
      "teammate_leak_artifact":          False,   # contaminated teammate artifact as label/feature
      "external_leak_data":              False,   # external dataset overlaps test distribution
      # Process / alignment guards
      "oof_test_alignment_validated":    True,    # 5fold/ready-for-lb MUST be True
      "rule_override_applied":           True,    # only relevant for LB candidates
      "rule_override_applicable":        True,    # set False when post-process N/A
      "codex_smoke_approved":            True,    # smoke is allowed
      "codex_5fold_approved":            False,   # full 5-fold blocked unless True
      "codex_new_mechanism_reviewed":    False,   # toxic-class override gate
      "p11_directly_optimized":          False,   # NEVER train against P11 directly
  },
  "compute_cost_hours":          1.0,             # rough wall-time
  "novelty":                     "low | medium | high"
}
```

## 3. Hard blockers

A `hard_block=True` verdict trumps everything. The candidate is rejected with
`recommended_action="BLOCK"`, `priority="PARK"`, and a `block_reason`. Any of:

**Leakage (all map to `leakage_risk="CRITICAL"`):**

1. `guards.uses_test_sgp_truth == True` → test SGP truth leak.
2. `guards.overwrites_test_sgp_truth == True` → direct test SGP overwrite.
3. `guards.sgp_derived_proxy == True` → indirect SGP-derived feature leak.
4. `guards.forbidden_rally_uid_inference == True` → rally_uid / row-order leak.
5. `guards.teammate_leak_artifact == True` → contaminated teammate artifact
   used as label/feature.
6. `guards.external_leak_data == True` → external dataset overlaps with test.

**Process / alignment:**

7. `guards.oof_test_alignment_validated == False` → no alignment audit
   (`leakage_risk="HIGH"`). For `stage in {"5fold", "ready-for-lb"}`, a
   missing/unknown alignment flag also blocks; materialization-stage
   candidates must explicitly prove OOF/test alignment.
8. `class in {B-impure, B-meta, B-player-style, pseudo-consensus, hard-per-SN-blend}`
   AND `novelty != "high"` AND no explicit new-mechanism Codex review
   (`guards.codex_new_mechanism_reviewed != True`).
9. `stage == "5fold"` (i.e. asking to launch full 5-fold) AND
   `guards.codex_5fold_approved == False`.
10. `stage == "ready-for-lb"` AND `guards.rule_override_applied == False`
    when applicable (i.e. when candidate is a R-034 / R-042-style blend).
11. `guards.p11_directly_optimized == True` → P11 overfit risk
    (`leakage_risk="HIGH"`; counts as holdout-adjacent leakage).

## 4. Scoring formula v0.2

After hard-blocker check, compute:

```
# 4.1 base lift in OV units (task-weighted)
base_lift_OV = 0.4 * d.F1_a + 0.4 * d.F1_p + 0.2 * d.AUC
# (if "OV" supplied directly, use that; otherwise derive)

# 4.2 transfer multiplier — class-specific empirical prior (§5 table)
expected_lb_delta_pre = base_lift_OV * transfer_multiplier[class]

# 4.3 slice penalty — major bucket / canary regressions
slice_penalty = sum(
    -0.001 for bucket in sn_bucket_regressions if bucket.delta_OV <= -0.005
) + sum(
    -0.002 for cls in canary_class_drops if cls.delta_F1 <= -0.015
)

# 4.4 holdout signal (advisory; small magnitude)
holdout_bonus = 0
if holdout_delta.OV is not None:
    if holdout_delta.OV >= 0.001:
        holdout_bonus = +0.0002
    elif holdout_delta.OV <= -0.003:
        holdout_bonus = -0.0003

# 4.5 final expected LB delta
expected_lb_delta = expected_lb_delta_pre + slice_penalty + holdout_bonus

# 4.6 compute cost penalty
compute_penalty = -0.0005 * max(0, compute_cost_hours - 5)

# 4.7 novelty bonus (encourages new mechanisms)
novelty_bonus = {"low": 0, "medium": 0.0003, "high": 0.0008}[novelty]

# 4.8 generalization score (0..1)
gen = 0.50
gen += {"new-mechanism": +0.20,
        "B-pure": +0.10, "B-feature": +0.10, "rule_override": +0.10,
        "weight-refinement": -0.20, "A-rearrangement": -0.20, "server-head-blend": -0.20,
        "B-impure": -0.40, "B-meta": -0.40, "B-player-style": -0.40,
        "pseudo-consensus": -0.40, "hard-per-SN-blend": -0.40}.get(class, 0.0)
gen += {"low": 0.0, "medium": +0.05, "high": +0.15}[novelty]
if slice_penalty < 0:
    gen += max(50.0 * slice_penalty, -0.30)
if holdout_bonus > 0: gen += +0.10
if holdout_bonus < 0: gen += -0.10
gen = clamp(gen, 0.0, 1.0)

# 4.9 target progress + priority
target_progress = expected_lb_delta / (TARGET_LB - anchor_lb)   # TARGET_LB = 0.4000
priority = bucket(expected_lb_delta, class, novelty, gen)        # §4.10

# 4.10 goal score — generalization-weighted priority signal
gen_boost = 0.5 * expected_lb_delta * (gen - 0.5)
goal_score = expected_lb_delta + compute_penalty + novelty_bonus + gen_boost
```

All terms are in **OV units** (same scale as LB delta). `goal_score` is now
amplified by `(gen - 0.5)` so a clean-feature candidate with the same expLB
as a churn candidate ranks higher.

### 4.10 Priority bucket

| Bucket | Rule |
|---|---|
| `PARK` | `expected_lb_delta <= 0` |
| `LOW`  | `0 < expected_lb_delta < +0.001`, OR `class in {weight-refinement, A-rearrangement, server-head-blend}` AND `expected_lb_delta < +0.005` AND `novelty != "high"` |
| `NORMAL` | `+0.001 <= expected_lb_delta < +0.005` (after churn-cap above) |
| `HIGH` | `+0.005 <= expected_lb_delta < +0.010` AND `gen >= 0.50` (else NORMAL) |
| `STRATEGIC` | `expected_lb_delta >= +0.010`, OR `class in {new-mechanism, B-pure}` AND `novelty=="high"` AND `expected_lb_delta >= +0.005` |

### 4.11 Action mapping

Given `priority` and `stage`:

| | preflight | smoke | 5fold | ready-for-lb |
|---|---|---|---|---|
| PARK | PARK | PARK | PARK | PARK |
| LOW (churn class) | PARK | PARK | PARK | SUBMIT_CANDIDATE (sunk cost; tiny but real win) |
| LOW (other) | PARK | PARK | PARK | SUBMIT_CANDIDATE |
| NORMAL+ | SMOKE_ONLY | FULL_5FOLD_REVIEW | MATERIALIZE_FOR_REVIEW | SUBMIT_CANDIDATE |

Key generalization-first principle: **a fresh smoke / 5-fold / multi-hour
training job is never launched for a `LOW` priority candidate**. R-067c-style
tiny wins are still uploadable when the artifact already exists, but they do
not justify spending new compute.

## 5. Transfer priors + per-class priority profile (CALIBRATED ON POST-2026-05-06 LB)

These multipliers convert OOF lift (in OV units) → expected LB lift. The
"churn?" column marks classes that are auto-capped at LOW priority unless
they reach the HIGH band (see §4.10).

| Class | Multiplier | Churn? | pub-LB-overfit-risk | Source of evidence |
|---|---:|---|---|---|
| `rule_override` (R-042 shot-context only: `prev_action, last_action, last_point`) | **1.0** | no | LOW | R-042 OOF +0.0028 → LB +0.0028 (exact 1.0) |
| `rule_override_deep_prefix` (e.g. `prev_prev_action+`) | **0.3** | no | MEDIUM | Untested isolated; included in R-072's −0.0033 failure. Conservative until separately validated. |
| `rule_override_player_context` (any `handId/positionId` in context) | **HARD BLOCK** | n/a | HIGH | **R-072 LB −0.0033 (2026-05-26)** with 9 of 11 overrides on Layer C/D (hand/position context). Same failure mode as R-062r B-player-style. |
| `B-pure` (ADD oldtest, like-for-like) | **1.0** | no | LOW | R-027 PAIR OOF +0.0028 → LB +0.0116 (super-additive but treated as 1.0 conservatively) |
| `B-feature` (new feature on same arch) | **0.9** | no | LOW | R-034 ratio 1.0121 with SN-clean profile; reduced by 10% pessimism for unseen feature sets |
| `server-head-blend` (cross-arch SGP blend) | **0.05** | **YES** | MEDIUM | R-067c OOF AUC +0.0326 × 0.2 OV weight → +0.000355 LB (5.4% of expected). Real but heavily attenuated. |
| `weight-refinement` (Bayes/COBYLA on safe pool) | **0.3** | **YES** | HIGH | R-068 +0.0012 full OOF → −0.0001 holdout; assumed to transfer ~30% if pool is clean. Classic Public-LB-style win that often fails Private. |
| `A-rearrangement` (re-weight already-trained safe components) | **0.5** | **YES** | MEDIUM | Class A historical LB ratio 0.95-0.98 |
| `new-mechanism` (no prior class match) | **0.8** + novelty bonus | no | LOW (if novelty=high) | Default optimistic; will be re-tuned after first LB datapoint. STRATEGIC when novelty=high and lift ≥ +0.005. |
| `B-impure` | **HARD BLOCK** | n/a | HIGH | R-028 −0.0086, R-040 −0.0094, R-055 −0.0141 |
| `B-meta` | **HARD BLOCK** | n/a | HIGH | R-054r −0.0103, R-055 (compounded) |
| `B-player-style` | **HARD BLOCK** | n/a | HIGH | R-062r −0.0057 |
| `pseudo-consensus` | **HARD BLOCK** | n/a | HIGH | R-010 V1 −0.0068; R-065/65b/65c abandoned |
| `hard-per-SN-blend` | **HARD BLOCK** | n/a | HIGH | `submission_zoo_v16_fast_04_per_sn_bucket.csv` LB 0.3597 |

## 6. Example evaluations

Hand-tuned for sanity check; see `_self_test()` in `src/candidate_goal.py`
for executable versions (run `python src/candidate_goal.py`).

Recall: gap to TARGET_LB = `0.4000 − 0.3870 = +0.0130`. `progress%` =
`expected_lb_delta / 0.0130`.

| R-id | Class | OOF lift OV | expLB | progress | gen | priority | action | rationale |
|---|---|---:|---:|---:|---:|---|---|---|
| R-034 (historical) | B-feature | −0.0005 | −0.0005 | −3.5% | 0.60 | PARK | PARK | Formula can't see the sub-additive LB win; expected_LB_delta is negative. Actual LB +0.0028 → known false-park; living with it to avoid overfit. |
| R-042 (historical) | rule_override | +0.0028 | +0.0028 | +21.6% | 0.65 | NORMAL | SUBMIT_CANDIDATE | Proven post-process. Covers ~22% of gap; still need much more. |
| **R-067c (historical, current LB-best)** | server-head-blend | AUC +0.0326 → OV +0.0065 | +0.0003 | +2.5% | 0.35 | **LOW** | SUBMIT_CANDIDATE | Valid small win and IS our current LB-best. But priority is LOW because (a) +0.0003 is <8% of gap and (b) server-head-blend is a churn class with MEDIUM pub-LB-overfit risk. **Do not spend more compute optimizing server-head sub-α**; chase larger wins. |
| R-055 (historical) | B-impure ADD | +0.0058 OOF | n/a | n/a | 0.00 | PARK | BLOCK | Toxic class (actual LB −0.0141). |
| R-062r (historical) | B-player-style | +0.0037 OOF | n/a | n/a | 0.00 | PARK | BLOCK | Toxic class (actual LB −0.0057). |
| R-070 7-feature smoke | B-feature | base −0.0010 | −0.0019 | −14.6% | 0.60 | PARK | PARK | Slice penalty (SN≤2 −0.0081, SN≥5 −0.0043). Mixed slice generalization → park even though global is only mildly negative. |
| R-070 5-feature ablation | B-feature | +0.0022 | +0.0020 | +15.2% | 0.65 | NORMAL | FULL_5FOLD_REVIEW | Clean ablation; clear positive lift, no slice regressions reported. Codex review pending. |
| Hypothetical clean B-feature (5fold) | B-feature | +0.0025 | +0.0025 | +18.9% | 0.70 | NORMAL | MATERIALIZE_FOR_REVIEW | Holdout +0.0015 confirms; no slice regressions. |
| **Hypothetical structural-novel +0.010 (5fold)** | new-mechanism | +0.0125 | +0.0102 | **+78.5%** | 0.95 | **STRATEGIC** | MATERIALIZE_FOR_REVIEW | One-shot bet — closes ~80% of the gap to TARGET_LB on its own. Multi-hour compute (12h) is justified at this priority. |
| **Hypothetical weight-refinement tiny churn** | weight-refinement | +0.0005 | +0.0001 | +1.2% | 0.30 | **LOW** | **PARK** | Class is in churn set; pub-LB-overfit risk HIGH; lift won't move us toward target. Do NOT spend a smoke slot. |
| Hypothetical SGP-derived proxy leak | B-feature | +0.0100 OOF | n/a | n/a | 0.00 | PARK | BLOCK | `sgp_derived_proxy=True` → leakage_risk=CRITICAL, hard-blocked. OOF looks great but cannot be trusted. |

**Pattern**: candidates ranked by `priority` (descending). One STRATEGIC and a
handful of NORMAL/HIGH candidates beat a long tail of LOW server-head /
weight-refinement churn even when those have non-negative expLB.

## 7. How Claude should use this BEFORE training/submission decisions

For every new candidate (feature module, blend, post-process, weight refinement,
pseudo-label, retrain, LB upload), before spending compute or a slot:

1. Construct the candidate dict from the available evidence.
2. Call `candidate_goal.score_candidate(cand)`.
3. Read `priority` FIRST, then `recommended_action`:
   - **`priority == "PARK"`** → do not pursue.
   - **`priority == "LOW"`** → unless artifact already exists (sunk cost),
     do not spend fresh compute. Even if `recommended_action == "SUBMIT_CANDIDATE"`,
     check whether the slot is better used for a HIGH/STRATEGIC candidate.
   - **`priority == "NORMAL"`** → proceed per stage gate.
   - **`priority == "HIGH"`** → proceed and queue with higher precedence than
     NORMAL.
   - **`priority == "STRATEGIC"`** → this is a candidate that could
     meaningfully close the gap to TARGET_LB on its own. Multi-day compute,
     Codex review, slot reservation are all justified.
4. Read `recommended_action`:
   - `BLOCK` → do not run; document the hard blocker; close the entry.
   - `PARK` → expected lift too small or risk too high; document and stop.
   - `SMOKE_ONLY` → allowed to run smoke under T2 budget; do NOT launch full.
   - `FULL_5FOLD_REVIEW` → smoke artifact ready; request Codex review for 5-fold launch.
   - `MATERIALIZE_FOR_REVIEW` → 5-fold artifact ready; build candidate CSV but DO NOT upload until Codex signs off.
   - `SUBMIT_CANDIDATE` → CSV materialized, rule_override applied, Codex review complete, OK to upload to LB.
5. Inspect `leakage_risk` and `public_lb_overfit_risk` independently:
   - `leakage_risk == "CRITICAL"` always co-occurs with `hard_block == True`.
   - `leakage_risk in {"HIGH", "MEDIUM"}` → flag in REVIEW_QUEUE entry.
   - `public_lb_overfit_risk == "HIGH"` → require Codex sign-off even if action
     is SUBMIT_CANDIDATE.
6. Quote the `goal_score`, `priority`, `expected_lb_delta`, `target_progress_ratio`,
   `generalization_score`, and `explanation` in the relevant `R-NNN` entry as
   a structured pre-commit summary. The user/Codex can override.

This file should be re-read at the start of any session where a new R-### is
being opened. The Python script is the authoritative implementation; this
document is the design rationale.

## 8. Re-calibration schedule

After every 3 new post-reset LB datapoints, update:
- `transfer_multiplier[class]` based on observed ratios
- The example evaluations table
- Any new hard-blocker classes discovered

Track these updates as `### Calibration log` appendices at the bottom of this file.

---

## Calibration log

### v0.1 (2026-05-24)

Initial. Anchored on R-067c LB-best 0.3870095. Includes 9 LB datapoints
since R-027 PAIR (2026-05-18): R-027 (+0.0116), R-028 (−0.0086), R-033
(−0.0015), R-034 (+0.0028), R-040 (−0.0094), R-042 (+0.0028), R-055
(−0.0141), R-062r (−0.0057), R-054r (−0.0103), R-067c (+0.000355).

### v0.6 (2026-05-26 late evening) — SoftF1 additive isolation: B-impure even at α=0.05

**New LB datapoint**: **R-094 v2 LB 0.3830398** (**−0.0040 vs R-067cr 0.3870095**).
Predicted +0.0003 to +0.0008; actual off by **~−0.0048**. Rank 70/364.

**Isolation analysis combining the 2 evening LB datapoints**:
- R-094 v2 alone: −0.0040 LB
- R-170 (R-094 v2 + R-081 v2 stacked): −0.0057 LB
- → R-081 v2 incremental contribution: ~−0.0017 (if approximately additive)
- **Both mechanisms are LB-toxic** for our component zoo

**Diagnosis**:
- R-094 v2 = SoftF1 component (R-031 `v11_mulminet_aug_oldtest_softf1_phaseB`) added
  as 6th additive at α_action=0.05, α_point=0.00. Action-only blend; SGP preserved.
- The OOF smoke had been **cleanest of the session**: ALL 15 action classes positive
  or neutral, ZERO canary drops, +0.0006 F1_a OOF lift, action-only diffs (38 cells).
- LB still went −0.0040.

**Lesson** (v0.6):
> B-impure (architecture-different component) is LB-toxic EVEN AS LOW-WEIGHT
> ADDITIVE, not just as full swap. R-094 v2 at α=0.05 (5% weight) reproduces
> the same B-impure failure mode that R-028 (LB −0.0086) and R-040 (LB −0.0094)
> showed at full swap weight. "Size-6 cap relax" hypothesis is FALSIFIED.

**New toxic class**: `B-impure-additive-low-weight` added to TOXIC_CLASSES set.
This blocks future SoftF1 / MuLMINet / other arch-different additive proposals
unless novelty=="high" with explicit Codex new-mechanism review.

**Implications for the strategic queue**:
- R-094 v1: NOT WORTH UPLOADING (shared-α version of v2; more aggressive; likely worse than -0.0040)
- R-081 v2: likely ~−0.0017 LB; tiny mechanism, not worth a slot
- Any future additive-low-weight architecture-different proposal: HARD BLOCK
- The remaining hope for clean LB +Δ is R-082 (V11 embeddings) — same architecture, not B-impure

**Open question**: Why did R-031 SoftF1 standalone OV transfer at LB-best blend ratio in OOF
but its low-weight ADDITIVE blend regressed by −0.0040? Hypothesis: SoftF1 macro-F1-targeted
training overfits to OOF in ways that show up as 5% weight blend perturbation. This is
distinct from B-impure SWAP failure (different component replaces existing one entirely).

### v0.5 (2026-05-26 evening) — Orthogonal-mechanism stacking caveat after R-170 LB regression

**New LB datapoint**: **R-170 LB 0.3813464** (**−0.0057 vs R-067cr 0.3870095**).
Predicted +0.0006 to +0.0011; actual off by **~−0.0068**.

**Diagnosis**:
- R-170 = R-094 v2 (SoftF1 additive action-only) + R-081 v2 (GBM corrector bounded)
- Diversity audit had shown action diff sets nearly disjoint (1-row overlap of 38+50)
- "Orthogonal mechanism stacking" hypothesis was: if mechanisms touch disjoint rows
  and each individually has positive smoke, stacking compounds.
- LB FALSIFIED this hypothesis.

**Lesson** (v0.5):
> Orthogonal mechanism stacking does NOT compound — it can interfere
> destructively. Two NORMAL-priority post-process mechanisms stacked may
> produce a STRATEGIC-magnitude LB regression. Stacking is HIGHER-risk than
> the sum of individual risk estimates suggests.

**Implications for v0.5**:
- Add new informational class `mechanism_stack`: when a candidate combines
  two or more LB-untested mechanisms additively, treat the combined risk as
  the SUM of individual downside ceilings (NOT sum of upside estimates).
- New advisory: prefer single-mechanism uploads to LB FIRST before stacking.
  Stacking should only happen after each individual mechanism is LB-confirmed
  positive.
- R-094 v2 and R-081 v2 individually are NOT YET LB-tested. Cannot conclude
  which is the toxic component (or if both are). To isolate: separate single-
  mechanism uploads required.

**Goal_function code changes**: minimal — the diagnostic message is the
calibration log entry. No new constants needed (the existing
`generalization_score` and `expected_lb_delta` framing still applies; the
issue is that COMBINED mechanism estimates need a STRONGER pessimism
multiplier than the SUM of individual estimates).

**Open question for v0.6** (after isolating R-094 v2 vs R-081 v2 via separate LB uploads):
- Which component is toxic? Both? Neither (interaction-only)?
- Does this update the GBM-corrector class transfer prior? (Currently set
  at "new-mechanism" / NORMAL priority.)

### v0.4 (2026-05-26) — Theory-first; smoke is sanity, LB is truth

**Trigger**: user policy directive 2026-05-26 — after R-072 LB regression and
several over-conservative auto-PARK calls.

**Diagnosis of the policy gap in v0.3**:
- v0.3's action mapping treated Fold-1 smoke OOF as quasi-truth (any negative
  expLB → PARK). This is conservative-safe but throws away theoretically
  strong candidates whose OOF noise happens to land marginally negative.
- It also pretends candidate_goal predictions of expLB are reliable. R-072
  proved they aren't (predicted +0.0015, actual −0.0033, off by −0.0048).
- Goal function should distinguish "smoke says don't waste compute" (NORMAL)
  from "smoke is just a sanity check; theory is what matters" (HIGH/STRATEGIC).

**Changes**:
- New verdict actions: `PROVISIONAL_PASS`, `PROVISIONAL_FAIL`.
- New helper `_catastrophic_collapse_check`: defines 5 patterns (OV/F1/AUC
  drop, severe canary class count, severe SN bucket count) that count as
  "sanity failure" worth stopping for. Anything milder is noise.
- `_recommend_action` v0.4: HIGH/STRATEGIC → `PROVISIONAL_PASS` unless
  catastrophic; NORMAL keeps the v0.3 stage gate; LOW unchanged.
- New verdict fields: `theoretical_generalization_reason`,
  `why_transfers_to_test_new`, `smoke_sanity_pass`, `smoke_sanity_reason`,
  `lb_probe_worthy`, `lb_confirm_hypothesis`, `lb_reject_hypothesis`,
  `lb_result`, `final_verdict`.
- §0a candidate report template added to GOAL_FUNCTION.md.
- Self-test 12/12 PASS (one expected-action updated for STRATEGIC class).

**Catastrophic thresholds** (v0.4 initial; subject to recalibration):
- OV drop ≤ −0.020
- F1_a or F1_p drop ≤ −0.030
- AUC drop ≤ −0.030
- 3+ classes with ΔF1 ≤ −0.025
- 2+ SN buckets with ΔOV ≤ −0.012

**Effect on past calls** (retroactive view):
- R-070 v15feat_e 7-feat smoke: ΔOV −0.0010 (well above −0.020 threshold).
  Under v0.4, would have been NORMAL+sanity-pass; Codex still BLOCKED it on
  slice + canary grounds (4 canary classes ≥ -0.015, but only 2 hit -0.025).
  Per v0.4 strict thresholds, R-070 would have been borderline:
  3 canaries at ≥ -0.025 = PROVISIONAL_FAIL only if at -0.025; actual deltas
  -0.0233/-0.0254/-0.0152, so just 1 trips -0.025 (action6 Push -0.0254).
  v0.4 would PROVISIONAL_PASS, recommending LB probe. Whether that's correct
  is unknown without an actual LB upload of R-070; this is the kind of call
  v0.4 is designed to give the human room to make.
- R-072 LB-failed: had `rule_override_player_context` toxic class, hard-blocked.
  Verdict unchanged under v0.4.

### v0.3 (2026-05-26) — `rule_override` sub-classed after R-072 LB regression

**New LB datapoint**: **R-072 LB 0.3837476** (−0.0033 vs R-067cr 0.3870095).
Predicted +0.0015 (per-override extrapolation halved); actual −0.0033. Off by
−0.0048.

**Diagnosis** (`submissions/submission_R072_R067cr_PLUS_RULE_V2_report.json`):
- R-072 applied 11 overrides on top of R-067cr (R-042's Layer A already in)
- Layer A (R-042 shot context): 0 new (already applied)
- Layer B (deeper shot context): 2 action overrides
- Layer C (hand-aware context, includes `handId`): 3 action + 1 point = 4
- Layer D (position-aware context, includes `positionId`): 3 action + 2 point = 5
- → 9 of 11 overrides used player-attribute context (handId or positionId)
- Same context-shape failure mode as R-062r B-player-style (−0.0057 LB)

**Reclassification**:
- `rule_override` (R-042's exact context: `prev_actionId, last_actionId, last_pointId`) — proven 1.0 transfer. UNCHANGED.
- `rule_override_deep_prefix` (deeper shot-only context, e.g. `prev_prev_actionId+`) — NEW class, 0.3 conservative transfer until isolated re-test.
- `rule_override_player_context` (any context with `handId`/`positionId`) — NEW TOXIC class. HARD BLOCK. Empirically reproduces B-player-style.

**Changes**:
- `TRANSFER_MULTIPLIER`: added `rule_override_deep_prefix` (0.3) and `rule_override_player_context` (0.0).
- `TOXIC_CLASSES`: added `rule_override_player_context`.
- `EXAMPLES`: added 12th case `R-072-historical-LBfailed` asserting `BLOCK`.
- Section 5 transfer-priors table updated.

**Open question for v0.4** (after next 3 LB datapoints): is Layer B alone (`prev_prev_action+`) safe? Currently 0.3 transfer is a guess based on R-072's combined failure. Could be tested by building rule_override_v3 with ONLY Layer A + Layer B (no C/D) and uploading.

### v0.2 (2026-05-25) — Generalization-first rewrite

No new LB datapoints since v0.1. This is a **design refactor** in response
to the user's clarification that the goal is clean **NEW LB ≥ 0.4000**
(not Public-LB chasing), with generalization to Private/Final as the
overriding criterion.

Changes:

* Added Section 0 (primary project goal + priority thresholds).
* Added new hard-blocker leakage guards: `sgp_derived_proxy`,
  `forbidden_rally_uid_inference`, `teammate_leak_artifact`,
  `external_leak_data`. All map to `leakage_risk="CRITICAL"`.
* Added new verdict fields:
  - `leakage_risk` (`NONE`/`LOW`/`MEDIUM`/`HIGH`/`CRITICAL`)
  - `public_lb_overfit_risk` (`LOW`/`MEDIUM`/`HIGH`)
  - `generalization_score` (0..1)
  - `priority` (`STRATEGIC`/`HIGH`/`NORMAL`/`LOW`/`PARK`)
  - `target_progress_ratio` (`expLB / (TARGET_LB - anchor_lb)`)
* `TARGET_LB = 0.4000` constant; gap-to-target driven by anchor_lb.
* New class buckets: `LOW_PRIORITY_CHURN_CLASSES =
  {weight-refinement, A-rearrangement, server-head-blend}` are capped at
  LOW priority unless they reach the HIGH band (expLB ≥ +0.005). They can
  still SUBMIT_CANDIDATE if already at ready-for-lb (sunk cost; e.g.
  R-067c was a real win worth uploading) but never burn fresh compute.
* `STRATEGIC_CLASSES = {new-mechanism, B-pure}` with `novelty=="high"`
  and expLB ≥ +0.005 are promoted to STRATEGIC.
* Goal score now amplified by `gen_boost = 0.5 * expLB * (gen - 0.5)`
  so clean-feature candidates outrank churn candidates at the same expLB.
* Action mapping rewritten: a fresh smoke / 5-fold / multi-hour training
  job is never launched for a LOW priority candidate. Tiny wins are still
  uploadable when the artifact already exists.
* Examples expanded with: hypothetical structural-novel +0.010 (STRATEGIC),
  hypothetical weight-refinement tiny churn (LOW → PARK), and hypothetical
  SGP-derived proxy leak (BLOCK with leakage_risk=CRITICAL).
