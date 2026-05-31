# External Data Research — 2026-05-12

Research mission delivered ~30 min by general-purpose agent. Compiled after R-020 confirmed our internal V11/V14/V16 stack is saturated and we have ~18 days remaining vs LB top-10 cutoff at 0.40+.

## Top finding: ShuttleSet22 + MuLMINet pretraining (HIGHEST EV)

The single highest-EV move is **pretrain a sequence transformer on ShuttleSet22 (badminton) using MuLMINet code, then fine-tune on AI CUP train**.

### Why this is the right move

1. **Sister sport** — same rally structure (alternating turns, opening serve, position-dependent shot selection)
2. **Schema near-isomorphic** to AI CUP 2026:
   - Stroke-level records with player identity ✓
   - 10 shot-type classes ✓ (we have 15; remappable)
   - 2D landing coordinates ✓ (binnable to our 9-zone grid)
   - Forehand/backhand attributes ✓
3. **MuLMINet architecture** (IJCAI CoachAI 2023, 2nd place) is what our `src/train_v11_mulminet.py` already implements
4. **Public reference code** at https://github.com/stan5dard/IJCAI-CoachAI-Challenge-2023
5. **Realistic in 18 days** — ShuttleSet22 is small (33k strokes), pretraining = few hours on RTX 3060 Ti
6. **Expected gain** per published transfer-learning benchmarks: **+0.005 to +0.020 macro-F1**, especially on rare classes (we have BH_short F1=0; pretraining provides better attention/embedding inits)

### Sources
- ShuttleSet22 dataset: https://github.com/wywyWang/CoachAI-Projects (`CoachAI-Challenge-IJCAI2023/`)
- MuLMINet code (2nd place): https://github.com/stan5dard/IJCAI-CoachAI-Challenge-2023
- ShuttleNet (AAAI'22 baseline): https://github.com/wywyWang/ShuttleNet
- Team8 winning solution (Advanced ShuttleNet): https://arxiv.org/abs/2307.13715
- ShuttleSet22 paper: https://arxiv.org/pdf/2306.15664

### Implementation plan (R-021)

**Phase 1** (Day 1, ~4 h GPU): Clone CoachAI-Projects, load ShuttleSet22, reproduce MuLMINet baseline numbers locally.

**Phase 2** (Day 2, ~4 h CPU): Schema shim:
- Badminton 10 stroke types → AI CUP 15 actionId classes (lookup table; embedding init)
- Badminton 2D landing coords → 9-zone bins matching pointId
- Adapt our `src/train_v11_mulminet.py` to load pretrained encoder weights

**Phase 3** (Day 3, ~4 h GPU): Fine-tune v11_mulminet from pretrained init on AI CUP train. Compare to v11_mulminet from scratch (current 0.3299 OOF).

## Other candidates (lower priority)

### Extended OpenTTGames Dataset (rank 2)
- **Source**: https://github.com/moamal01/table_tennis_data, paper https://arxiv.org/abs/2512.19327
- **Type**: Actual table tennis (12 video files, 5 train + 7 test)
- **Strokes**: 6 technique types (serve, Loop, Block, Push, Flick, Lob — overlap with our actionId)
- **Issue**: Video-frame format requires ETL; small size; rally endings only; no spin/strength
- **Verdict**: Marginal in 18 days; lower than ShuttleSet22

### TTSwing (rank 3 — but actually NOT applicable)
- **Source**: https://datadryad.org/dataset/doi:10.5061/dryad.0zpc8677f
- **Critical**: This is the AI CUP 2025 dataset (Taiwan IMU paddle competition)
- **Schema**: 9-axis IMU sensor stream per stroke, NOT rally events
- **Verdict**: NOT applicable to AI CUP 2026 tactical prediction (different modality)

### ITTF / WTT public data (skip)
- Player rankings, handedness, country data
- Issue: Test players DE-IDENTIFIED — can't match to ITTF data
- Verdict: skip; only helps train, which we've maxed

### Tennis Match Charting Project (Jeff Sackmann)
- **Source**: https://github.com/JeffSackmann/tennis_MatchChartingProject
- **Size**: 17,633 matches, 10.4M shots (largest racket-sport dataset)
- **License**: CC BY-NC-SA 4.0
- **Verdict**: Tertiary pretraining stage if ShuttleSet22 underperforms; lower domain transfer than badminton

### RacketVision (cross-sport)
- **Source**: https://arxiv.org/html/2511.17045v3
- **Size**: 1,672 video clips (badminton + tennis + table tennis)
- **Verdict**: Too small + video-based; low EV

### TabPFN-v2 (different angle — model not data)
- **Source**: https://github.com/PriorLabs/TabPFN
- **Pitch**: Tabular foundation model; drop-in for our blend
- **Use case**: Especially good for SGP binary task (AUC-scored, smaller per-rally size = TabPFN's sweet spot)
- **Realistic**: Yes — install + try in an afternoon
- **Expected gain**: +0.003-0.010 AUC for SGP, decorrelated from our current models
- **Verdict**: Worth a R-022 entry as parallel work

### Synthetic data / simulator
- CoachAI-Plus exists for badminton (https://github.com/KuangDW/CoachAI-Plus) — not for table tennis
- Building a TT simulator from scratch in 18 days: not viable
- Verdict: skip

## What we will NOT use

- **AI CUP 2025 winners' code**: 2025 was a different task (IMU player-attribute), not transferable
- **Any video-based reverse-checking** (rules ban this)
- **Inter-team data sharing** (rules ban this)
- **ITTF rankings as test prior** (test players de-identified, can't match)

## Combined plan

If R-014 (Path B causal LM, ~30h GPU) AND R-021 (ShuttleSet22 pretrain, ~3 days) AND TabPFN-v2 (R-022, ~1 day) all land at midpoint:
- R-014 contribution: +0.005 OV via blend diversity
- R-021 contribution: +0.010 OV via pretraining
- TabPFN-v2 contribution: +0.005 OV via SGP improvement
- Combined: **+0.020 OV** (within striking distance of 0.39)

Plus our existing v11_mulminet_aug as private-LB candidate gives generalization headroom for the shake-up.

Realistic LB target by 2026-05-30: **0.38-0.40** (closes ~half the gap to top 10).
Top 5 (0.43+) likely requires either external pretrained models we don't have access to, or competitive insights we haven't found.
