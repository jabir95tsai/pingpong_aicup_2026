# R-073 — `data/external/` audit (2026-05-25)

**Status**: AUDIT COMPLETE → PARK (with conditional re-open clause).
**Author**: Claude autonomous mode (per /goal directive).
**Anchor**: clean LB R-067cr 0.3870095, target 0.4000.

---

## What's on disk

`data/external/CoachAI-Projects/` is a git clone of the public CoachAI repository.
Two subdirectories:

### 1. CoachAI Badminton Environment
Small demo / fixture data (per-folder ~1-2 CSVs, ≤ a few hundred rows total):
- `First2Ball/demo.csv`
- `MovementForecasting/demo.csv`
- `StrategicEnvironment/Data/{ball_type_model.csv, hit_height.csv, hit_range.csv, player_speed_model.csv}`

These are model-fixture data for the CoachAI strategic-RL environment, NOT a
training dataset. Out of scope for our supervised setting.

### 2. CoachAI-Challenge-IJCAI2023 → ShuttleSet22
The well-known **ShuttleSet22** dataset (60 badminton matches, ~4806 rallies after
filtering, 18 stroke types). Per-set CSVs with columns:

```
rally, ball_round, time, frame_num, roundscore_A, roundscore_B, player, server,
type, aroundhead, backhand, hit_height, hit_area, hit_x, hit_y,
landing_height, landing_area, landing_x, landing_y, lose_reason, win_reason,
getpoint_player, flaw, player_location_area, player_location_x, player_location_y,
opponent_location_area, opponent_location_x, opponent_location_y, db
```

## Already-attempted (R-021, RESULTS.md §36)

**ShuttleSet22 was already used as a pretraining source in R-021** (2026-05-12,
12-hour autonomous session). Outcome:

- Pretrained v11_mulminet on ShuttleSet22 (causal next-shot LM pretraining)
- Then bidirectional-finetuned on table-tennis train.csv
- Result: smoke OV 0.3226 — TIED with v11_mulminet_aug baseline (no transfer benefit)
- Verdict at the time: **PARKED**. Documented root cause:
  > "Causal-pretrain → bidirectional-finetune mismatch limits ShuttleSet22 transfer"

## Why this is PARK in 2026-05-25 (independent of R-021)

1. **Domain gap is large**. Badminton ≠ table tennis:
   - 18 stroke types in BWF spec vs 19 actionIds (0-18) in our spec — vocabulary
     is non-overlapping (e.g. "rear court drop" vs "Loop").
   - Court geometry: rectangular badminton court with net at ~1.55m vs 9-area
     pingpong table grid. The `hit_area / landing_area` codes do not map.
   - Rally length distribution differs (badminton rallies are typically longer,
     with more defensive shots).
2. **R-021 empirical failure** is the strongest evidence — actual pretraining
   produced zero lift, not a small lift we could rescue.
3. **Stronger candidates exist** (R-071 in flight; R-072 already shipped).
   Spending time wrangling a domain-mismatched pretrain is LOW priority under
   `candidate_goal` v0.2 — even if it worked, expLB would be capped by the
   weakness of the transfer.
4. **No clean labelling that maps to our targets**: we'd need to construct
   pseudo-table-tennis labels from badminton strokes, which introduces a label
   distribution shift on top of the domain shift.

## Conditional re-open clause (R-073 v2)

There is **one scenario** where ShuttleSet22 deserves a second look, and only one:

> **If R-071 v4 (causal LM with focal + class-balanced sampling) passes its
> smoke gates** AND becomes a competitive standalone model (OV ≥ 0.30 with
> AUC ≥ 0.65), THEN ShuttleSet22 could be re-tried as a causal-LM pretrain
> source for R-071's architecture specifically. The R-021 failure mode
> ("causal pretrain → bidirectional finetune mismatch") would NOT apply,
> because R-071 is causal-LM throughout. The risk is then domain gap, not
> pretrain/finetune mismatch.

This is conditional on R-071 succeeding, which we'll know in ~2-3 hours.

## Leakage assessment

- ShuttleSet22 is a public dataset (BWF, IJCAI 2023 challenge); no overlap
  with our `data/test_new.csv` (which is post-2026-05-06 table tennis).
- No SGP-equivalent at row level (badminton's `getpoint_player` is at rally
  end only, different mechanic).
- `external_leak_data` guard = **False** (no test distribution overlap).

## Final action

- **PARK R-073**. Mark in AUTONOMOUS_RUN_QUEUE.
- Do NOT spend training compute on ShuttleSet22 pretraining unless R-071 v4
  smoke clears its gates AND we have spare GPU budget after pursuing R-071
  full 5-fold + any LB upload work.
- No code changes required.

## Compute saved

~4h research + (if attempted) 8-12h Kaggle GPU pretrain = ~12-16 hours of
compute redirected to higher-priority STRATEGIC candidates.
