# Research Notes — Literature for ceiling-breaking

Compiled 2026-05-11 after R-015 (v17_momentum) confirmed our V16-backbone GBM tabular features are saturated at OV ~0.3666. Gap to LB top: 0.076.

## Highest-EV finding: **MuLMINet auxiliary-task loss**

**Paper**: MuLMINet — Multi-Layer Multi-Input Transformer Network with Weighted Loss
**Source**: arXiv 2307.08262, IJCAI CoachAI Challenge 2023 (2nd place), code: https://github.com/stan5dard/IJCAI-CoachAI-Challenge-2023

**Why it's the #1 priority**:
- Same problem family (next stroke type + landing in racquet sport from short categorical sequence).
- Won 2nd at a real benchmark with public code we can read.
- Attacks our **exact failure mode** (saturated tabular features + weak SGP) with a low-risk training-time change.
- ~1-day implementation, drops into V11 transformer backbone.

**Technique**: Add 4 auxiliary heads predicting next-shot's `handId`, `spinId`, `strengthId`, `positionId`. Total loss:
```
L = 0.4·L_action + 0.4·L_point + 0.2·L_SGP + λ·Σ aux_losses
```
α≈0.3-0.45 (their value); tune on OOF. The aux signal regularizes the encoder toward predicting all aspects of the next shot, which forces a richer representation than just (action, point, SGP).

## High-EV cluster — 5 immediate wins

### 1. MuLMINet aux-task loss (A2 above)
- Cost: ~1 day
- Risk: low
- EV: high (direct match to failure mode)
- Open as R-018 preflight

### 2. Uncertainty-weighted MTL losses (Kendall & Gal 2018)
- Replace fixed 0.4/0.4/0.2 with learnable `nn.Parameter` log-vars
- ~2 hours impl, ~10 lines PyTorch
- Lets model learn σ_action / σ_point / σ_SGP — likely rebalances toward weak SGP (AUC 0.61)
- Open as R-019 preflight

### 3. GroupKFold-by-player audit
- AI CUP 2025 winner used StratifiedGroupKFold on player_id
- We currently use GroupKFold on `match`, which is correlated but not identical
- Audit: compare CV variance under match-grouped vs player-grouped split
- If significantly different, switch (closes CV-LB gap)
- Cost: trivial code change + 1 GPU rerun

### 4. Snapshot ensembles (Huang et al. ICLR 2017)
- Cosine LR with 3 cycles, save 3 snapshots per training run
- Counts as 1 model in our blend cap of 5 → effectively 3× diversity
- ~1 hour code change in V11 trainer
- Open as R-020 (V11 snapshot ensemble)

### 5. Soft-F1 fine-tuning (Surrogate Fβ, AnyLoss)
- After CE training, fine-tune last 5-10 epochs with `0.5·CE + 0.5·soft_macro_F1`
- Direct optimization of our actual eval metric
- Targets stuck rare classes (BH_short F1=0)
- Cost: ~half day

## Medium-EV cluster — structural moves

### 6. ShuttleNet 2-stream architecture (Wang et al. AAAI 2022)
- Rally encoder + per-receiver style encoder with position-aware gated fusion
- Style summary = receiver's first-N shot statistics (no player ID needed → de-id-safe)
- ~2-3 days implementation, novel architecture
- Higher risk than MuLMINet but bigger ceiling lift

### 7. DANN player-invariance (Cross-subject EEG generalization)
- Gradient-reversal layer over a player-ID classifier head during V11 training
- Forces rally encoding to be player-invariant
- Drop the head at inference; nothing leaks
- ~1.5 days
- Specifically attacks the de-identified-test-players generalization

### 8. ART adaptive resampling (arXiv 2509.00955, Sep 2025)
- Every 5 epochs, set sample weight ∝ (1 − F1_class) on val
- Specifically targets stuck rare classes (BH_short)
- ~half day, ~30 lines
- +2.64 macro-F1 reported on tabular benchmarks

### 9. BiLSTM SGP head (badminton outcome paper, IEEE 2023)
- Bi-directional pooling over rally prefix → MLP → sigmoid for SGP
- Targets weakest task (AUC 0.61); +0.05 AUC = +0.010 OV
- ~1 day

### 10. LGBM-distilled transformer (AMEX 14th place writeup)
- Use V14 OOF probs as soft targets when training V11
- Loss = α·CE + (1-α)·KL(student || V14_softmax)
- Bridges GBM + NN inductive priors
- Adds blend diversity even if standalone is similar

## Lower-priority but interesting

### 11. ShuttleFlow (normalizing flows, Springer ML 2024)
- Joint distribution of (action, point) via normalizing flow
- Captures action-point correlation that our dual-head softmax discards
- Risky, ~3-4 days
- Last-resort if everything else plateaus

### 12. LEM autoregressive soccer LM
- Confirms our Path B causal LM direction is well-grounded
- Use Path B as **blend ingredient** (not standalone), ≤0.15 weight
- Snapshot ensemble of 3 Path B runs at cyclic LR

### 13. Temperature scaling preference for private LB
- Literature: TEMP > NONE > CW > THR for private-set transfer
- We currently submit NONE blends; might switch to TEMP for safer transfer

## Suggested 20-day execution plan

| Day | Action | Source | Expected lift |
|---|---|---|---|
| 1-2 | GroupKFold-by-player audit + temperature-scaling consolidation | C3 + E3 | Free |
| 3 | Uncertainty-weighted MTL on V11 | D1 | +0.001-0.005 |
| 4-5 | **MuLMINet aux-task loss on V11** | A2 | **+0.002-0.010** |
| 6 | Soft-F1 fine-tune + ART resampling | D2 + D3 | +0.002-0.005 |
| 7-9 | ShuttleNet 2-stream architecture | A1 | +0.005-0.015 |
| 10 | BiLSTM SGP head | B3 | +0.005-0.010 |
| 11 | LGBM-distilled transformer | E1 | diversity for blend |
| 12-14 | DANN player-invariance | C1 | +0.002-0.008 |
| 15-17 | Path B causal LM full run + snapshot ensemble blend | E2 + Path B | diversity |
| 18-20 | Final ensemble tuning, NONE/TEMP blend selection | E3 | safety pick |

Total expected lift if all land near midpoint: **+0.030 to +0.060 LB** — closes ~half the gap to top.

## What NOT to do (locked from today's lessons)

- ❌ More tabular feature engineering on V16 backbone (saturated at 0.3666)
- ❌ Player-ID-dependent features (de-id breaks them)
- ❌ Pseudo-labeling from LB-best teacher (R-010 bias amplification)
- ❌ 6+ component blends (rule #8)
- ❌ Single-component LB submissions (§3.1.2)
- ❌ Equal-weight blender exhaustive search (R-007/R-008/R-016 all regressed)

## Reference list (full)

| # | Cluster | Title | Source |
|---|---|---|---|
| A1 | Direct analog | ShuttleNet | https://cdn.aaai.org/ojs/20341/20341-13-24354-1-2-20220628.pdf |
| A2 | **Direct analog** | **MuLMINet** | https://arxiv.org/html/2307.08262 + https://github.com/stan5dard/IJCAI-CoachAI-Challenge-2023 |
| A3 | Direct analog | Advanced ShuttleNet | https://arxiv.org/abs/2307.13715 |
| A4 | Direct analog | ShuttleFlow | https://link.springer.com/article/10.1007/s10994-024-06682-0 |
| B1 | Sister-domain LM | Seq2Event (soccer) | https://eprints.soton.ac.uk/458099/1/KDD22_paper_CReady_v20220606.pdf |
| B2 | Sister-domain LM | LEM (soccer) | https://link.springer.com/article/10.1007/s10994-024-06606-y |
| B3 | Sister-domain LM | Badminton BiLSTM | https://ieeexplore.ieee.org/document/10082764/ |
| B4 | Sister-domain LM | Tennis return decision | https://peerj.com/articles/cs-3439/ |
| C1 | De-id generalization | Cross-subject EEG survey | https://arxiv.org/html/2604.27033v1 |
| C2 | De-id generalization | player2vec | https://arxiv.org/html/2404.04234v1 |
| C3 | De-id generalization | AI CUP 2025 30th place | https://github.com/yuchen0515/AI_CUP_2025_Table_Tennis |
| D1 | Multi-task | Kendall & Gal MTL uncertainty | https://arxiv.org/abs/1705.07115 |
| D2 | Multi-task | Surrogate Fβ / AnyLoss | https://arxiv.org/pdf/2104.01459 |
| D3 | Multi-task | ART adaptive resampling | https://arxiv.org/html/2509.00955 |
| E1 | Ensembling | AMEX 14th LGBM-distill NN | https://www.kaggle.com/competitions/amex-default-prediction/writeups/chris-deotte-14th-place-gold-nn-transformer-using- |
| E2 | Ensembling | Snapshot Ensembles | https://arxiv.org/abs/1704.00109 |
| E3 | Ensembling | Temperature scaling (Guo et al.) | https://proceedings.mlr.press/v70/guo17a/guo17a.pdf |
| extra | Direct analog | MJSSM 2026 table tennis stroke forecasting | https://www.mjssm.me/clanci/MJSSM_March_2026_Wu.pdf |
| extra | CoachAI org | CoachAI Projects (badminton repos) | https://github.com/wywyWang/CoachAI-Projects |

## Plan rewrites needed

After R-017 (Dirichlet blender) completes, rewrite STRATEGY.md / TRAIN_PLAN.md to reflect:
1. The new top-priority queue (above table)
2. Acknowledgment that tabular feature work is parked
3. The MuLMINet → ShuttleNet → BiLSTM-SGP cascade as the main work plan
4. Path B causal LM (R-014) repositioned as a blend ingredient, not standalone
