# 新對話繼續使用 — AI CUP 2026 桌球預測

## 📊 當前進度（2026-04-04）

### 已完成
✅ 修復 6 個 critical bugs：
1. serve_mask bug（features_v2.py, v3.py）：`isin([0,1])` → `== 0`
2. Target encoding 序序問題（features_v4.py）：`actionId.mean()` → `attack_rate`（coarse category）
3. 原始 player IDs 洩漏（get_feature_names_v3/v4）：已排除原始 ID
4. Combo feature 對齐問題（train_v4_fast.py）：train/test 用同一 `top_indices`
5. Global stats CV leakage（train_v5_clean.py）：改為 per-fold 計算
6. Feature selection CV leakage（train_v5_clean.py）：改為 per-fold 選擇

✅ 特徵工程演進：148 → 871 → 875 base features
- V3: one-hot lag(4), transition prob vectors, cumulative stats, player distributions, momentum, bigram/trigram
- V4: +target encoding (coarse rates), rally-level SGP predictors, 3-way interactions

✅ 模型集成（3模型 soft vote）：CatBoost + XGBoost + LightGBM
✅ 後處理：溫度縮放 + per-task 類別權重校正
✅ Git commit & push：branch `claude/strange-merkle`

### 正在執行
🔄 **V5 fold-safe clean CV pipeline**（背景執行，估計 3-4 小時）
- Location: `src/train_v5_clean.py`
- Progress: Fold 1/2 完成，Fold 3/4/5 中...
- Fold 1 結果：CB OV=0.3256（誠實分數，無 leakage）

## 📈 歷史分數

| 模型 | OV 分數 | 備註 |
|------|---------|------|
| V4+V3 per-task blend + calibration | 0.3678 | 修 bug 前，可能有 leakage |
| V5 clean fold 1 (CatBoost) | **0.3256** | 誠實分數，無 leakage |
| V4 Fast calibrated | 0.3650 | 修 bug 前 |
| V3 Calibrated | 0.3639 | 修 bug 前 |

## 📁 關鍵檔案位置

```
src/
├── features_v2.py      # 148 features (基礎)
├── features_v3.py      # 871 features (lag, prob, momentum)
├── features_v4.py      # 871 + TE, rally stats, interactions
├── data_cleaning.py    # 前處理 (strikeId remap 等)
├── train_v3_champion.py        # V3 + combo features
├── train_v4_fast.py            # V4 無 SMOTE 版本 (OV=0.3650)
├── train_v4_ultimate.py        # V4 全功能 + SMOTE + stacking
└── train_v5_clean.py           # ⭐ 新：fold-safe CV，per-fold stats/selection
models/
├── oof_v3_champion.npz         # V3 OOF
├── oof_v4_fast.npz             # V4 OOF
└── oof_v5_clean.npz            # V5 OOF (執行中)
```

## 🎯 下一步任務

1. **等 V5 跑完** → 拿 5-fold 完整誠實分數
2. **加 combo features**（per-fold 計算 top_indices）→ 衝 0.33+
3. **任務特化優化**（隊友建議）：
   - Point 任務專用的特徵組合
   - Action 機率 → Point 特徵的 cross-task 串接
   - 階層式 Point 模型（長/短 → 細分位置）
   - Test-like next_sn 加權評估
4. **提交最佳模型** → 生成 final submission

## 🔧 重要細節

### 類別定義
- **actionId**: 19 類（0=無, 1-7=attack, 8-11=control, 12-14=defense, 15-18=serve）
- **pointId**: 10 類（0=無/掛網, 1-3=短, 4-6=半台, 7-9=長）
- **serverGetPoint**: 二元（發球者得分 0/1）

### 評估指標
- actionId: Macro F1 (weight 0.4)
- pointId: Macro F1 (weight 0.4)
- serverGetPoint: AUC-ROC (weight 0.2)
- **綜合評分**: 0.4×F1_a + 0.4×F1_p + 0.2×AUC

### 重要概念
- **strikeNumber=1** 必為發球（actionId ∈ {15,16,17,18}）
- **serverGetPoint** 是 rally 層級（同一 rally 所有拍相同）
- **Test set** 班長分佈偏斜（72.9% SN≤4 vs 55.5% train）→ test-weighted evaluation 更誠實
- **Fold-safe CV**: 所有計算（global stats, TE, feature selection）must be per-fold，避免 train/val leakage

## 🏗️ 架構邏輯

```
Raw Data
  ↓
data_cleaning.py (strikeId remap, player ID 匿名化)
  ↓
features_v{2,3,4}.py (build_features)
  ├─ Lag features (4 steps)
  ├─ Transition probabilities (19 dim for actions, 10 for points)
  ├─ Cumulative statistics (per-player, per-zone, etc.)
  ├─ Target encoding (per-fold, coarse rates)
  └─ Interactions (3-way, ratios, diffs)
  ↓
Fold-safe CV (train_v5_clean.py)
  ├─ Per-fold global stats
  ├─ Per-fold feature selection (XGBoost gain)
  ├─ Per-fold model training (CB, XGB, LGB × 2000 iter)
  ├─ Per-fold OOF predictions
  └─ Test-weighted OV evaluation
  ↓
Final model (on full train)
  ├─ Train CB/XGB/LGB with per-fold selected features
  └─ Predict on test
  ↓
submission.csv
```

## 🎓 從隊友 Code Review 學到的

1. **Global stats 洩漏**: 先在全資料算統計，再 CV → val predictions 會看到 test-like stats
2. **Feature selection 洩漏**: 在全資料選特徵，再 CV → CV 看到的特徵已被未來資料"汙染"
3. **Coarse encoding**: Nominal targets (actionId, pointId) 不能用 `.mean()`（把類別當順序），改用 `attack_rate`、`long_rate`
4. **Combo alignment**: Train/test 的組合特徵必須用相同的 top_indices
5. **Test-like evaluation**: 重視 strikeNumber 小的預測（競賽更關心），加權評估更現實

## 📝 怎麼用這份 Prompt

貼在新對話開頭，系統會立即有完整 context，不用重複說明，直接開始工作。

---

**Last updated**: 2026-04-04 (V5 fold 2 training)
**Branch**: `claude/strange-merkle`
**Commit**: c3ffdc5
