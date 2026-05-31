# P2A Dataset Audit — 2026-05-19

**Source**: `C:/Users/jabir/Downloads/P2A_dataset/dataset/`
**Total size**: **208 GB** (mostly video)
**Date received**: 2026-05-18 (file mtime)

## Executive verdict

**P2A is genuinely table-tennis data with stroke-level labels**, BUT:
- **Schema overlap is partial**: only `(hand, is_serve, action_type)` per stroke. Missing `pointId` (40% of competition score) and `serverGetPoint` (20% of competition score) entirely.
- **Player identity gap**: original videos have named players (Ma Long, Lin Yun-Ju, Fan Zhendong, Harimoto, Calderano, etc.). AICUP test players are de-identified — no way to match.
- **Closest prior experiment (R-021 ShuttleSet22 pretraining) did NOT help** — pretrained transformer encoder on a sister sport failed to transfer.
- **Expected EV**: LOW for direct supervised use. MEDIUM-LOW for tiny transition-prior augmentation. Videos themselves are unusable without a video-processing pipeline we don't have.

**Recommendation**: **PARK the videos. Quarantine the label JSONs for future research. Do NOT auto-train on this.** A formal R-### proposal would only be justified if R-029b (teammate transition matrix features) lands well — then a P2A external transition prior could be a follow-on.

---

## 1. Inventory

```
P2A_dataset/dataset/
├── proj.json                    180 KB  — filename mapping (original ↔ renamed)
├── label/                       18 MB total
│   ├── v1.json                 4.8 MB  — original-filename labels
│   ├── v1_renamed.json         4.7 MB  — renamed-filename labels
│   ├── v2.json                 4.6 MB  — original-filename labels
│   └── v2_renamed.json         4.5 MB  — renamed-filename labels
└── video/                      208 GB
    ├── v1/                     1281 .mp4 files
    └── v2/                     1208 .mp4 files
```

**Original filenames** (from proj.json) reveal source: clips from 2019 ITTF Men's World Cup and other ITTF events. Real top players. Clips appear to be 1-of-N chunked match recordings.

## 2. Label schema

Each video has:
```json
{
  "url": "0000000.mp4",
  "total_frames": null,
  "actions": [
    {
      "label_ids": null,
      "label_names": ["正手", "否", "控制"],   // [hand, is_serve, action]
      "start_id": 106.34,                       // seconds
      "end_id":   106.82
    },
    ...
  ]
}
```

### Total annotated strokes
| Split | Videos | Strokes | Strokes/video |
|---|---:|---:|---:|
| v1 | 1281 | 33,321 | 26.0 |
| v2 | 1208 | 31,593 | 26.2 |
| **Total** | **2489** | **64,914** | 26.1 |

### Field 1: `hand`
| Value | Count (v1) | Meaning |
|---|---:|---|
| 正手 | 19,843 (60%) | Forehand → AICUP `handId=1` |
| 反手 | 13,476 (40%) | Backhand → AICUP `handId=2` |

### Field 2: `is_serve` (decoded from data)
Confirmed by per-action breakdown: serve action types (sidespin, reverse-sidespin, etc.) have ~96-100% "是" (yes), non-serve actions ~0-0.4%. So this column = **is_serve binary flag**.
- "否" = no  (non-serve, 80%) → AICUP `strikeId ∈ {2, 4}` (receive or 3rd+ ball)
- "是" = yes (serve,    20%) → AICUP `strikeId = 1` (serve)

### Field 3: `action_type` (15 distinct values)

| P2A label | English | v1 count | AICUP actionId mapping |
|---|---|---:|---|
| 拉 | loop / topspin pull | 12,467 | **1 拉球** — direct |
| 侧旋 | sidespin (serve) | 5,164 | **15 傳統** or similar serve variant |
| 摆短 | short push (placement) | 3,856 | **11 擺短** — direct |
| 侧身拉 | sidestep loop | 3,000 | 1 拉球 (variant) |
| 控制 | control (general) | 2,927 | 8/9/10/11 — UNRESOLVED |
| 拧 | twist / wrist flick | 2,136 | **4 擰球** — direct |
| 劈长 | long push | 1,421 | 10 搓球 (similar) |
| 转不转 | spin/no-spin (serve) | 747 | 15 傳統 (variant) |
| 逆旋转 | reverse sidespin (serve) | 648 | **17 逆旋轉** — direct |
| 挑 | flick | 640 | **7 挑撥** — direct |
| 勾球 | hook (serve) | 154 | **16 勾手** — direct |
| 普通 | traditional / normal (serve) | 101 | **15 傳統** — direct |
| 下蹲 | squat (serve) | 36 | **18 下蹲式** — direct |
| 中性 | neutral | 20 | UNRESOLVED |
| (empty) | — | 4 | NOISE |

**~10 of 15 P2A action types map cleanly to AICUP actionIds 1, 4, 7, 11, 15, 16, 17, 18.** 5 types are ambiguous or generic.

### What's MISSING vs AICUP

| AICUP field | In P2A? | Notes |
|---|---|---|
| `actionId` | Partial | Vocab maps ~10/15 cleanly; some ambiguity |
| `pointId` (landing zone) | **NO** | Would require court-aware video CV |
| `serverGetPoint` (rally outcome) | **NO** | Would require rally-end detection |
| `strikeId` | **Inferred** | Maps from is_serve flag, partial |
| `handId` | Yes | Direct |
| `strengthId` | NO | Not annotated |
| `spinId` | **Inferred** | Encoded in action_type for serves only |
| `positionId` | NO | Would need court coords |
| Per-player IDs | **Original-name only** | De-identified AICUP players can't be matched |
| Rally structure | **Inferred** | Single video may contain many rallies; need to segment by time gaps |

## 3. Leakage / legality audit

| Check | Status | Evidence |
|---|---|---|
| External data permitted by AICUP | **YES** | "可使用自製資料或開源資源" (own or open-source data allowed) |
| Same data as test set | NO | P2A is real-player ITTF match clips; AICUP test is de-identified |
| Reverse-matching test players via P2A | **NO PATH** | AICUP de-identifies players; P2A names them. Even if we identify the matches, we can't map them to AICUP player IDs. |
| Risk of training on test rallies | NONE | P2A is video clips of unrelated 2019 matches |
| Sgp/target leak | NO | P2A has no serverGetPoint or pointId labels at all |
| LESSONS rule on external data | **T3 review required** | Documented in `LESSONS_CHECKLIST.md` |

**Legality verdict**: USE-ALLOWED with proper citation + Codex T3 review per LESSONS rule.

## 4. Use-case assessment

### Option A: Pretrain a stroke-action classifier on P2A → fine-tune on AICUP
- 64k strokes with 3-field labels
- Direct analogue: **R-021 ShuttleSet22 pretraining FAILED**: full 5-fold OV 0.3280 vs v11_mulminet_aug 0.3299 = −0.0019 (PARK verdict). Causal-bidirectional architecture mismatch + domain gap.
- P2A has STRONGER domain match (same sport) but SMALLER vocab overlap (15 actions vs AICUP 19).
- **Estimated EV**: probably 0 to +0.003 OOF (best case beats R-021 marginally due to same-sport).
- **Cost**: ~10-15h dev (loader, vocab mapping, pretrain encoder) + ~3-5h GPU pretrain.
- **Verdict**: LOW EV. Not worth pursuing while R-029a/b (cheaper, higher-EV) are open.

### Option B: External transition prior — P(next_action | last_action) computed from P2A
- Need to segment P2A action lists into rallies (use time-gap heuristic; gaps > 3-5s = rally boundary)
- Build empirical transition matrix from rally-segmented P2A
- Add as new feature column to V14/V16 (like teammate's transition matrix in R-029)
- **EV**: would complement R-029b (teammate's transition prior built from AICUP train) by adding a domain-prior smoothing term. Could marginally improve generalization to the de-identified test.
- **Risk**: P2A vocab subset of AICUP — would need to map and 0-pad unmapped classes. Hyperparameter: how much weight to give P2A vs AICUP transitions.
- **Cost**: ~6h dev (rally segmentation + vocab mapping + feature integration) + 1 v14 retrain ~150 min CPU.
- **Verdict**: MEDIUM-LOW EV. **Only consider AFTER R-029b lands** to avoid confounding the variable.

### Option C: Use videos for visual stroke detection or motion features
- 208 GB video, no extraction pipeline, no GPU video-model training in our stack
- Would require building from scratch: video preprocessing, frame-level feature extraction (3D CNN / video transformer), then integration
- **Cost**: 50-100+ hours dev work to build infrastructure + significant GPU time
- **Verdict**: **DO NOT PURSUE** in this competition window. Vision is not our model class.

### Option D: Pseudo-labeling P2A with our V11 model → use as additional training
- Run V11 on P2A clips (after extracting tabular stroke sequences), generate predicted (actionId, pointId, serverGetPoint), add to train
- Direct analogue: **R-009 pseudo-label V1 BANNED** (LB regressed −0.0068 due to bias amplification). Pseudo-labeling already proven to fail in this competition.
- **Verdict**: BLOCKED by LESSONS pseudo-label rule. Don't pursue.

### Option E: Player profile from named players
- Original P2A filenames have player names; could extract per-player action distributions
- **BUT** AICUP test players are de-identified — no mapping
- **Verdict**: DEAD END.

## 5. Legal reuse classification

| Component | Class | Action |
|---|---|---|
| Video files (208 GB) | **DO_NOT_USE** (no infrastructure) | Park; don't process |
| Label JSONs (18 MB) | **NEEDS_CODEX_REVIEW** | Quarantine; potential Option B input |
| Pseudo-labeling on P2A | **BANNED** | Same class as R-009 (LB regressed) |
| Action vocab mapping table | **SAFE_AFTER_REIMPLEMENTATION** | Document in `audits/` for future use |
| Direct supervised pretrain (Option A) | **PARK** | Same failure class as R-021 |
| External transition prior (Option B) | **NEEDS_CODEX_REVIEW** | Future R-030 after R-029b lands |
| Video CV models (Option C) | **DO_NOT_USE** | Out of scope |
| Player profile (Option E) | **DO_NOT_USE** | No mapping path |

## 6. Recommended next actions

### Immediate
- ✅ This audit documented (`audits/P2A_DATASET_AUDIT_2026-05-19.md`)
- ✅ No data copied into `data/` or `src/`
- ✅ No training launched

### Short-term (after Phase 3 + R-029a/b)
- Decide whether Option B (external transition prior) is worth a formal R-030. Conditional on:
  - R-029b (teammate transition matrix) showing positive OOF lift in R-029a successor
  - User/Codex T3 approval for external data per LESSONS rule
  - Realistic EV bound of +0.001 to +0.005 OOF (smaller than R-029b's +0.005 to +0.015 expected)

### Long-term
- The 208 GB videos remain a potential resource if we ever build a video-processing pipeline. Not in scope for AI CUP 2026.

## Final summary (1-line per question)

1. **Is P2A useful?** Marginally. The labels are partial (no pointId, no serverGetPoint) and the videos are unusable without an infrastructure investment we won't make. Closest prior experiment (R-021 ShuttleSet22 pretrain) failed.
2. **Is the data legal?** YES — open-source external data is permitted by AICUP. T3 Codex review required before any training use (per LESSONS).
3. **What's the best use?** Option B (external transition-prior smoothing for action features), ONLY as a follow-on to R-029b if that lands. Not a high-EV move on its own.
4. **What's blocked?** Pseudo-labeling on P2A (banned by R-009 precedent), video CV (out of scope), player profile matching (de-identification blocks it).
5. **Next safest action?** None — keep auditing. The 208 GB videos and 18 MB labels stay in `C:/Users/jabir/Downloads/P2A_dataset/`. Not copied into our pipeline.
