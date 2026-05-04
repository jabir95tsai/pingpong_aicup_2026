# FINAL_PLAN
## Round: 2026-05-03 — Post-V15 closure, P2 design pending Codex review

---

## Current State

| Item | Value |
|---|---|
| Current best LB | **0.3694863** (zoo V16/V14_seed1/V12/V11 blend) |
| Short-term LB target | Stable > 0.36, then 0.38, then 0.40 |
| Daily submission limit | 3 / day |
| OOF-LB gap (V16+V11 current best) | 0.0070 (OOF underestimated LB) |
| OOF-LB gap (clean V14+V11 prior best) | 0.0155 |
| OOF-LB gap (player profile) | 0.022–0.026 (non-transfer, REJECTED) |

Locked rules:
- **NEVER** use test.csv `serverGetPoint` as truth or feature.
- **NEVER** include SGP-derived player winrate in any final candidate.
- **NEVER** include raw player profile features (pp_act_freq, opp_act_freq, per-player ID stats).
- **NEVER** include hist freq / streak features (inert, no signal vs V14).
- Validation: `GroupKFold(n_splits=5)` by **match** (NOT rally).

Organizer clarification (2026-05-03):
- **Test.csv `actionId` / `pointId` from prior shots within the same rally are NOT leakage.**
  They are observable history at inference time. Using them as supervised training augmentation
  is permitted.
- Test.csv `serverGetPoint` is still NOT to be used in features or training.

---

## Priority Roadmap

### P0 — Hold V16+V11 as current best

Submission `submission_zoo_v16_fast_01_v16_v14_seed1_v12_5f_v11.csv` is the current LB benchmark at **0.3694863**. V16+V11 remains the best single-family backup at 0.3673269.

### P1 — V15 ablation (✅ CLOSED 2026-05-03)

Final ablation table:

| Ablation | OOF Opt solo | OOF+V11 blend | LB | OOF−LB Δ | Decision |
|---|---|---|---|---|---|
| V15_hist_only | 0.3640 | 0.3741 | **0.3574287** | **−0.017** | hist+streak inert, REJECT |
| V15_pp (full) | 0.3688 | 0.3765 | 0.3506750 | −0.026 | REJECT |
| V15_player_only | 0.3699 | 0.3777 | **0.3555110** | **−0.022** | REJECT |

P1 gate result (2026-05-03): best V15 LB 0.3574 < V14+V11 0.3599 → **V15 family rejected**.

Closed conclusions:
- **Hist freq + streak**: permanently excluded (no OOF signal).
- **Player profile (all forms)**: permanently excluded. Non-transfer is robust — confirmed across
  V15_pp (gap −0.026) and V15_player_only (gap −0.022). Both beat V14 on OOF; both fail on LB.
- Root cause: player ID statistics computed on 100% known train players do not generalise to
  the LB player distribution (63.5% overlap).

**No V15-derived feature or submission may be used as a future candidate.**

### P2 — Detailed plan in [NEXT_PLAN.md](./NEXT_PLAN.md)

P2 candidates (ranked):
1. **P2.0 V16_test_history_aug** — ✅ submitted and became new best LB=0.3673269; use as new backbone.
2. **P2.1 Multi-seed V14** — variance reduction via 3-seed ensemble.
3. **P2.2 Heterogeneous clean ensemble** — model diversity (LGB-only/XGB-only/conservative CB).

Universal gates (each candidate):
- Must use `GroupKFold(n_splits=5)` by **match**.
- No raw player profile / hist / streak / SGP-derived features.
- OOF must beat V14+V11 (0.3754) before any LB submission.
- Test-augmented rows must never appear in validation OOF.

### P3 — Architecture breakthrough (deferred)

Reserved for after P2 settles. Sketch only:
- Self-supervised rally embedding (mask+predict pre-training, append to GBM).
- Structured point decoder (depth × side grid latent).
- Phase-aware MoE (per-SN-bucket experts).

Forbidden:
- Plain supervised Transformer replacing V11 (capacity-bound, evidence in V11+ Gate 2).
- Class-weight escalation past POINT_W cls3=22.0.

---

## Action Order This Round (✅ COMPLETE)

1. ✅ Update RESULTS.md
2. ✅ Update FINAL_PLAN.md
3. ✅ Implement `features_v10` ablation flags
4. ✅ V15_hist_only 5-fold (OV=0.3640)
5. ✅ Blend V15_hist_only + V11 (OOF=0.3741 < V14+V11=0.3754 → REJECT)
6. ✅ V15_player_only full 5-fold (solo=0.3699; +V11 blend=0.3777)
7. ✅ V15_player_only+V11 diagnostic LB → **0.3555110** (gap −0.022, player profile permanently closed)
8. ✅ V15_hist_only+V11 diagnostic LB → **0.3574287** (gap −0.017, clean transfer but weaker than V14)
9. ✅ P1 closed. Ready for P2.

## Next Round (P2 — see [NEXT_PLAN.md](./NEXT_PLAN.md))

P1 V15 ablation is fully closed. Detailed P2 design with concrete commands, runtimes, gates,
and artifact naming is in **NEXT_PLAN.md**, awaiting Codex review.

Headline:
- **P2.0 V16_test_history_aug** (submitted; new best LB=0.3673269)
- **P2.1 Multi-seed V14** (conservative default)
- **P2.2 Heterogeneous clean ensemble** (diversity backup)
- **P3 Architecture breakthrough** (deferred until P2 settles)

**Do NOT launch any training before Codex reviews NEXT_PLAN.md.**

---

## Submission Budget Tracking

Daily limit: 3 per day.

Used today (2026-05-03):
- 1 slot: `submission_v15_pp_v11_optblend.csv` (diagnostic, LB=0.3506750)
- 1 slot: `submission_v15_player_only_v11_optblend.csv` (diagnostic, LB=0.3555110)
- 1 slot: `submission_v15_hist_only_v11_optblend.csv` (diagnostic, LB=0.3574287)

Used today (2026-05-04):
- 1 slot: `submission_v16_testhist_aug_v11_optblend.csv` (then-best, LB=0.3673269)
- 1 slot: `submission_zoo_v16_fast_01_v16_v14_seed1_v12_5f_v11.csv` (new best, LB=0.3694863)
- 1 slot: `submission_zoo_v16_fast_04_per_sn_bucket.csv` (failed probe, LB=0.3596738)

Daily slots exhausted for 2026-05-04. Avoid per-SN bucket blend variants unless backed by new evidence. Do NOT use future slots on any V15 variants.

---

## Hard Rules (carried from CODEX_REVIEW)

1. No `serverGetPoint` test-csv leakage in any submission.
2. No SGP-derived player winrate features in any final candidate.
3. No submission based on OOF alone when features include player-level stats.
4. **No player profile features of any form** — pp_act_freq, opp_act_freq, or any per-player ID
   statistic. Non-transfer confirmed across two LB tests. Permanently excluded.
5. No history freq or streak features — inert (no OOF signal vs V14).
6. Threshold optimization must use the same routine as V14 to keep OOF-LB calibration stable.





