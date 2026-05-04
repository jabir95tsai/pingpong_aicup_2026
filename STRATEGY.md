# STRATEGY
## Round: 2026-05-04 — V16 + zoo-blend backbone (post-LB sync)

---

## Current State

| Item | Value |
|---|---|
| **Current best submission** | `submission_zoo_v16_fast_01_v16_v14_seed1_v12_5f_v11.csv` |
| **Current best Public LB** | **0.3694863** (rank 28/181, 2026-05-04) |
| **Current best OOF (matched)** | 0.37998 |
| **Components** | `v16_testhist_aug + v14_seed1 + v12_5f + v11` (global 4-model blend) |
| **Backup (single-family best)** | `submission_v16_testhist_aug_v11_optblend.csv` LB 0.3673269 |
| **Old V14+V11 stable** | LB 0.3598509 — keep as deep fallback |
| Daily LB submission limit | 3/day (2026-05-04 used: 3/3) |
| OOF→LB gap (current best) | −0.01049 |
| OOF→LB gap (V16+V11) | −0.0070 (OOF underestimated LB) |
| OOF→LB gap (V14+V11, clean) | −0.0155 (canonical baseline) |
| OOF→LB gap (per-SN bucket zoo #4) | −0.0197 (overfit, REJECTED) |

Bottlenecks (ranked by leverage):
- **pointId** F1 ≈ 0.23 — largest headroom; class 0 (off-grid) and short classes (1/2/3) carry most of the loss.
- **serverGetPoint** AUC ≈ 0.61 — rally-level label is structurally under-modelled (per-shot prediction is noisy).
- **SN=2** slice OV ≈ 0.27 — early-rally; partly addressed by V16 test-history aug, do not chase further with hard SN gating.

---

## Locked Rules (do not violate)

1. **NEVER** use test.csv `serverGetPoint` as feature, target, or supervision. All training scripts overwrite it with −1; `build_test_history_pairs.py` discards real values before saving.
2. **NEVER** include SGP-derived player win-rate features.
3. **NEVER** include raw player-profile features (`player_action_freq`, `opp_action_freq`, per-player ID stats). V15 family is permanently rejected — non-transfer confirmed across two LB tests (gaps −0.022 and −0.026).
4. **NEVER** include `hist_action_freq` / `hist_point_freq` / `streak_*` (V15 hist+streak group). Inert vs V14, no LB signal.
5. **NEVER** use **hard per-SN-bucket gating** in blend weight search. zoo_v16_fast_04 proved this overfits OOF and loses −0.0098 on LB vs the global zoo.
6. Validation: `GroupKFold(n_splits=5)` by **match** for any new training script. Test-history augmented rows never enter validation.
7. Submission gate: a candidate may be submitted only if it has a credible path to beating the current best **0.3694863** — not merely beating an internal OOF threshold. The OOF gate (≥0.3764) was calibrated to V14-era pipeline; V16 family's small OOF→LB gap means the gate over-rejects good V16 candidates. Use multi-signal judgment (OOF + per-SN slice + OOF→LB gap of similar prior submission), not OOF alone.

---

## What Worked

| Direction | Evidence | Status |
|---|---|---|
| **V16 test-history augmentation** | LB +0.0075 vs V14+V11; OOF→LB gap shrank to −0.007 | New backbone — extend with seeds and richer aug |
| **Global multi-model zoo blend** | LB +0.00216 vs V16+V11 single-pair blend | Continue exploration with broader model menu |
| **V11 transformer as aux blend partner** | Lifts every backbone by +0.005–0.008 OOF | Use in every final candidate |
| **Multi-seed V14 averaging** | Variance reduction; v14_avg3 OOF 0.3623 solo, +V11 OOF 0.3765 | Component-quality artifact for the zoo, not a standalone submission |

## What Failed (do not retry without new evidence)

| Direction | Evidence | Status |
|---|---|---|
| **V15 player profile (any form)** | Two LB tests, OOF→LB gap −0.022 / −0.026 | PERMANENTLY REJECTED |
| **V15 hist freq + streak** | OOF flat (−0.0013); LB −0.0024 vs V14 | PERMANENTLY REJECTED |
| **Hard per-SN-bucket blend weights (zoo_v16_fast_04)** | OOF 0.37936 (≈ best), LB 0.3596738 (−0.0098 vs zoo #1) | REJECTED — non-transfer |
| **CatBoost in final blend** | OOF +0.006, LB −0.001 | Excluded from all final candidates |
| **V11+ class-weight + larger transformer** | OOF flat to negative | CLOSED |
| **Plain hierarchical point head (V12 era, hard decode)** | F1_p 0.158 vs flat 0.210 | Use only as soft-decoded variant in P3 |
| **Flat SN-bucket per-target weighting** | Inflates OOF, degrades LB | Avoid |

---

## Round Objective (2026-05-04 → next 1–2 days)

**Primary:** Find a candidate that beats the current best **LB 0.3694863** without burning a submission slot on a low-confidence file.

**Secondary:** Build a wider menu of high-quality, diverse OOF/test artifacts so the zoo blend has more raw material to search over (V16 multi-seed in particular).

**Tertiary:** Take one cheap, high-upside structural shot at the pointId bottleneck (hierarchical point head, soft-decoded). Only one structural experiment in flight at a time.

---

## High-ROI Hypotheses (this round)

### H1 (P1) — Blend Zoo v2: broaden the global multi-model search

Core idea: With 7+ usable bases (v16_testhist_aug, v14_avg3, v14_seed0/1/2, v12_5f, v11, v11plus, optionally v13/v14_5f_nocb), search 3–6 model global blends with per-task weights (action / point / server independently), but **without** per-SN gating. Add calibration-aware variants:
  - temperature-only (no per-class threshold opt)
  - global class-weight only (no scipy per-class)
  - Mix-and-match against the standard threshold-opt path

Why it could beat current best: zoo_v16_fast_01 is the first multi-model blend and already exceeded the V16+V11 pair. The search space was narrow (≤4 models, single weight family, threshold-opt only). A wider OOF search with calibration variants is the cheapest way to find a candidate with a smaller OOF→LB gap than the current best.

Cost: pure post-processing — minutes per blend, no training. Most expensive step is the search loop itself (~30–60 min of CPU).

Risk: OOF overfitting if search space is too large. Mitigation: cap the search to a tractable grid (per-task α ∈ {0.0, 0.1, …, 1.0} + per-task temperature ∈ {1.0, 1.5, 2.0}); evaluate every candidate against per-SN slice stability (variance across SN buckets must not blow up — high SN-slice variance is the zoo_v16_fast_04 failure mode).

Success signal:
- Top-1 OOF candidate's per-SN slice F1 spread ≤ zoo_v16_fast_01's spread (no per-SN over-tuning).
- At least one variant with calibration-only (no scipy threshold) and OOF ≥ 0.378 — these are the safest LB transfers.

Failure signal: best OOF blend below zoo_v16_fast_01's 0.37998, or per-SN spread widens.

### H2 (P2) — V16-centered multi-seed ensemble

Core idea: Train `v16_testhist_aug` at two new seeds (48879, 51966) and build `v16_avg3`. Then blend `v16_avg3 + v14_avg3 + v12_5f + v11` (the zoo backbone with averaged V16 instead of single-seed V16).

Why it could beat current best: V16 transferred better than expected, but `zoo_v16_fast_01` used only one V16 seed. Averaging multiple V16 seeds should reduce LB variance directly, and the variance-reduced V16 should plug into the zoo blend cleanly.

Cost: prerequisite — add `--seed` flag to `train_v16_testhist_aug.py` (currently absent). Two full 5-fold runs ≈ 2 × 180 min = 6 h.

Risk: medium. V16's gain may be saturated; averaging seeds may not buy further LB. Mitigation: smoke-test seed1 on 1 fold first (≤45 min) to confirm per-fold OV is in V16's range; only then commit to seed2.

Success signal:
- `v16_avg3` solo OOF ≥ V16 solo (≈ 0.3677).
- `v16_avg3 + v14_avg3 + v12_5f + v11` OOF ≥ 0.380.
- Per-fold solo OV variance smaller than single-seed V16.

Failure signal: per-seed V16 OV varies by < 0.001 (no diversity to exploit); abort and pivot to H3.

### H3 (P3) — Hierarchical point head (soft-decoded)

Core idea: pointId is the largest leverage. Replace single 10-class head with three soft heads, recombined as probability product:
  - **valid head**: P(point=0) vs P(point∈{1..9})
  - **depth head** (trained on point≠0 only): P(short / half / long)
  - **side head** (trained on point≠0 only): P(FH / mid / BH)
- Reconstruct: `P(point=0) = P(valid=0)`, `P(point=k) = P(valid=1) · P(depth=d(k)) · P(side=s(k))` for k∈{1..9}.

Why it could beat current best: cls 0 (off-grid) is behaviourally a different process (player missed the intended grid) than the on-grid 9-class placement. Disentangling them frees the on-grid model from absorbing the cls-0 noise. Direct attack on the largest task bottleneck.

Cost: medium engineering (new training script `train_v18_hier_point.py` derived from `train_v14.py`); one full 5-fold run ≈ 200 min.

Risk: the V12-era hard-decoded hierarchical model failed badly (F1_p 0.158 vs 0.210). Soft probabilistic recombination is the safer formulation. Only run if Codex signs off on the OOF-side reconstruction (must happen inside the CV loop, not after).

Success signal:
- OOF F1 for cls 1/2/3 (short classes) improves by ≥ +0.03 vs V14.
- Overall OOF F1_p ≥ 0.235 (vs current 0.228).
- Solo OOF OV ≥ V14 solo (≈ 0.366).
- Plugs into the zoo as a new component.

Failure signal: cls 1/2/3 F1 unchanged or worse; cls 0 F1 regresses. Abort; do not blend.

### H4 (P4) — Rally-level Server head

Core idea: SGP is rally-constant. Build a separate model that pools per-shot features into a single rally embedding (mean+max pool, plus rally-level meta features like rally length, score diff at end of rally history, last-shot action) and predicts SGP once per rally; broadcast to per-shot rows.

Why it could beat current best: AUC=0.61 is suspiciously low for a label that does not vary within a rally. A direct +0.04 AUC gain = +0.008 score, plus this is a structurally orthogonal source of OOF signal for the zoo blend.

Cost: low–medium; can be implemented as a small post-hoc module reading existing OOF features (no need to re-train action/point bases).

Risk: the per-shot model may already implicitly use most of the rally context. Quick OOF check before committing engineering.

Success signal:
- OOF rally-AUC ≥ 0.65 in 1 fold.
- When swapped into the zoo blend's server channel, OOF OV improves.

Failure signal: rally-AUC ≤ 0.62 → existing per-shot pipeline already captures the signal. Park.

### H5 (P5, deferred) — Autoregressive multi-task sequence model

Core idea: Causal Transformer that predicts (action, point, sgp) at every position in the rally; pretrain LM-style on union of train+test rallies (using observable history only, no test SGP); fine-tune supervised on real train positions. Multi-task heads with rally-pooled SGP head.

Why it might beat current best: 5–10× more supervised positions per rally; pretraining on test rallies generalises the V16 trick across all positions; structurally distinct from V11 so should add blend diversity.

Cost: high engineering; full run ≈ 8–10 h. Therefore start with a 1-fold smoke (≤90 min, hidden=256, 6 layers).

Risk: redundant with V11. Smoke must show ≥ V14 solo (≈ 0.36) AND non-trivial blend diversity vs V11 (Pearson correlation of OOF probs < 0.95).

Success signal: smoke solo OOF ≥ V14 solo and OOF probs decorrelate from V11; commit to full run.

Failure signal: smoke OOF in low 0.34s, or OOF probs ≈ V11 → abort.

---

## Priority Order This Round

| Priority | Hypothesis | Risk | Cost | Ceiling |
|---|---|---|---|---|
| **P0** | Hold current best 0.3694863; protect submission slots | — | none | — |
| **P1** | Blend Zoo v2 (broader global search + calibration variants) | low | 30–60 min CPU | beat current best directly |
| **P2** | V16 multi-seed (`v16_avg3`) → re-run zoo blend | medium | ~6 h training | ≈ +0.001–0.003 LB |
| **P3** | Hierarchical point head (soft-decoded) | medium | ~3.5 h training | step change on cls 1/2/3 |
| **P4** | Rally-level Server head | low–medium | ~2 h | +0.004–0.008 score via AUC |
| **P5** | Autoregressive sequence model (smoke first) | high | 1.5 h smoke / 8–10 h full | step change to 0.38+ if it works |

---

## Submission Hypothesis (next slot)

The next submission slot must clear the bar `LB > 0.3694863`. Eligible candidates, ranked:

1. **Best zoo v2 candidate** — the OOF top-1 from H1 broader search **with** the smallest per-SN slice spread among the top-5 OOF candidates. This is the only candidate that can be evaluated end-to-end without new training.
2. **`v16_avg3 + v14_avg3 + v12_5f + v11` zoo blend** — only after H2 V16 multi-seed completes. Use the same blend search infrastructure as H1, locked to the same global-only weight family.
3. **V14_avg3 + V11 standalone** — already complete (OOF 0.3765). Treat as a *fallback* candidate, not a top pick: it is likely to score in the V14+V11 LB regime (≈ 0.36) which would not beat current best. Submit only if H1/H2 produce nothing.

---

## Forbidden / Deferred

- Hard per-SN-bucket weight conditioning (zoo_v16_fast_04 failure).
- All V15 / player-profile / hist-freq / streak features.
- CatBoost in final blends.
- Plain V11+ class-weight escalation.
- Hard-decoded hierarchical point heads.
- Any feature derived from test.csv `serverGetPoint`.
- Submitting the next-slot candidate before Codex reviews this STRATEGY.md and TRAIN_PLAN.md.

---

## Open Questions for Codex

1. **OOF gate calibration**: V16+V11 OOF (0.3743) failed the V14-era OOF gate (≥0.3764) but won on LB. Should the gate be replaced with a **per-family** gate (V16 family gate ≈ V16+V11 OOF − ε, V14 family gate ≈ V14+V11 OOF) or a **multi-signal** gate (OOF + OOF→LB gap of nearest prior)? The current single-threshold gate over-rejects V16 candidates.
2. **Zoo search overfit guard**: what is an acceptable OOF→LB gap budget for the broader H1 search? Suggest gating by max per-SN slice variance increase ≤ +0.005 vs zoo_v16_fast_01.
3. **V16 `--seed` plumbing**: confirm the location of every random-state in `train_v16_testhist_aug.py` (LGB `random_state`, XGB `random_state`, optional CB `random_seed`, `np.random.seed`, sklearn `GroupKFold` is deterministic but the **flip-aug pair sampling** path needs auditing).
4. **Hierarchical point reconstruction**: must the depth and side heads be trained on the point≠0 subset only, or on all rows with sample-weight = (point ≠ 0)? Either is defensible; choose one and keep it consistent with the OOF reconstruction inside the CV loop.

---

## Carried Anchors

- V14 5-fold no-CB OOF: action 0.3793 / point 0.2162 / AUC 0.6101 / **OV 0.3602 base, 0.3754 +V11**
- V14 solo opt OV (post threshold-opt, before V11): 0.3661
- V16 testhist aug solo opt OV: 0.3677; +V11 OOF 0.3743; **LB 0.3673269**
- zoo_v16_fast_01 OOF 0.37998; **LB 0.3694863**
- V14_avg3 solo OOF 0.3623; +V11 OOF 0.3765
- SN=2 slice (V14+V11): n=14995, F1_a=0.243, F1_p=0.161, AUC=0.539, OV=0.270
