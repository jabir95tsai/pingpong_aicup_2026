# AI CUP 2026 — Overnight Report (2026-04-30)

**Submitted to user:** Codex 隔日分析
**Total overnight runtime**: ~3.5 小時（SN=2 expert 8 min + V12+CB 146 min + V12 5-fold 44 min + blend experiments）

---

## 🎯 TL;DR — 推薦提交順序

### 🥇 第一提交：`submission_4way_optblend.csv`
- **OOF OV = 0.3809**（+0.0075 vs 已提交 0.3734）
- **預期 LB ≈ 0.358-0.362**（基於 -0.019 CV-LB gap）
- 4-way 模型: V12cb + V12_5f + V12 + V11
- 找到的最佳權重不對稱（每個任務不同）：
  - Action: V12cb=0.4, V12_5f=0.4, V12=0.0, V11=0.2
  - Point:  V12cb=0.2, V12_5f=0.2, V12=0.2, V11=0.4 ⚠️ V11 主導 point
  - Server: V12cb=0.1, V12_5f=0.5, V12=0.4, V11=0.0 ⚠️ V11 srv 完全沒用

### 🥈 第二提交（保險）：`submission_v12cb_v11_optblend.csv`
- **OOF OV = 0.3771**（+0.0037）
- 預期 LB ≈ 0.358
- 較簡單的 2-model blend，過擬合風險低於 4-way

### 🥉 第三提交（高風險高報酬）：`submission_v12cb_v12_5f_v11_optblend.csv`
- **OOF OV = 0.3787**
- 介於兩者之間，3-model blend 平衡

---

## 各條實驗線設定 + 結果

| 實驗 | folds | aug | CB | features | 訓練時間 | 單模 OV (opt) |
|---|---|---|---|---|---|---|
| V12 (已存在) | 3 | ❌ | ❌ | V7 | ~24 min | 0.3632 |
| V12aug (今天先前) | 3 | ✅ | ❌ | V7 | ~50 min | 0.3733 (blend) |
| V13 (今天先前) | 3 | ❌ | ❌ | V8 | ~53 min | 0.3634 |
| **V12+CB** ⭐ | 3 | ❌ | ✅ | V7 | **146 min** | **0.3691** |
| **V12 5-fold** | 5 | ❌ | ❌ | V7 | **44 min** | **0.3646** |
| SN=2 expert | 3 | ❌ | ❌ | V8 | 8 min | 0.2645 (SN=2 only) |

### Raw OOF 指標（單模型，未 blend，未 threshold opt）

| Tag | F1_a | F1_p | AUC | OV(raw) |
|---|---|---|---|---|
| v12 | 0.3743 | 0.2102 | 0.6055 | 0.3549 |
| v12aug | 0.3747 | 0.2124 | 0.6042 | 0.3557 |
| **v12cb** | **0.3880** | **0.2138** | 0.6050 | **0.3617** |
| **v12_5f** | 0.3813 | 0.2058 | **0.6115** | 0.3571 |
| v13 | 0.3764 | 0.2092 | 0.6048 | 0.3552 |
| sn2_expert (SN=2 rows only) | 0.2147 | 0.1477 | 0.5267 | 0.2503 |

**關鍵觀察：**
- V12+CB 對 F1_a 大幅提升（+0.0137 vs v12），證實 CatBoost 對 ensemble 有獨立貢獻
- V12 5-fold 對 AUC 提升（+0.0060 vs v12），更穩定的 server 預測
- 兩者來自不同維度，所以 blend 才會疊加

---

## Ensemble 排行榜（OOF OV）

| Ensemble | F1_a | F1_p | AUC | **OOF OV** | Δ |
|---|---|---|---|---|---|
| 已提交：V12 + V11 optblend | 0.4022 | 0.2279 | 0.6066 | 0.3734 | LB=0.3541 |
| V12aug + V11 optblend | 0.4014 | 0.2292 | 0.6054 | 0.3733 | -0.0001 |
| V13 + V11 optblend | 0.4028 | 0.2278 | 0.6060 | 0.3734 | 0.0000 |
| V12cb + V11 optblend | 0.4106 | 0.2291 | 0.6062 | 0.3771 | **+0.0037** |
| SN=2 expert + V12 hybrid | 0.4022 | 0.2279 | 0.6071 | 0.3734 | +0.0000 (失敗) |
| SN=2 expert + V12cb hybrid | 0.4091 | 0.2314 | 0.6064 | 0.3775 | +0.0041 |
| V12cb + V12 + V11 | 0.4106 | 0.2313 | 0.6074 | 0.3782 | +0.0048 |
| V12cb + V12aug + V11 | 0.4106 | 0.2309 | 0.6066 | 0.3779 | +0.0045 |
| V12cb + V13 + V11 | 0.4106 | 0.2299 | 0.6076 | 0.3777 | +0.0043 |
| V12cb + V12_5f + V11 | 0.4100 | 0.2299 | 0.6139 | 0.3787 | +0.0053 |
| **🏆 4-way: V12cb+V12_5f+V12+V11** | **0.4128** | **0.2324** | **0.6144** | **0.3809** | **+0.0075** |

---

## SN=2 Slice 指標

baseline（V12+V11 optblend）vs 各候選：

| Submission | SN=2 F1_a | SN=2 F1_p | SN=2 AUC | SN=2 OV |
|---|---|---|---|---|
| V12+V11（已提交）| 0.2486 | 0.1606 | 0.5388 | 0.2714 |
| **4-way blend** | 待生成 | - | - | - |
| SN=2 expert hybrid | 0.2495 | 0.1634 | 0.5370 | 0.2726 |
| V12cb + V12_5f + V11 | - | - | - | - |

**SN=2 結論：** SN=2 仍是全域瓶頸。Expert 對 SN=2 slice 改善極小（+0.0012 OV），不值得投入更多。

---

## 每條實驗線的明確結論

### ✅ V12+CB — 高度成功，建議保留
- 單模 OV +0.006，blend 後 +0.0037
- CatBoost 在處理 categorical features 上補了 LGB+XGB 的盲點
- 訓練成本 2.5 小時，CP 值極高
- **結論：未來任何 V14+ 都應該包含 CatBoost**

### ✅ V12 5-fold — 中度成功，建議保留
- 單模 OV +0.001，blend 後 +0.005
- 5-fold 的主要貢獻是 AUC（0.6115 vs V12 3-fold 0.6055）
- **結論：5-fold 比 3-fold 穩，未來主線都用 5-fold**

### ❌ SN=2 Expert — 失敗
- 單模 OV 0.2645（vs baseline 0.2714，-0.007）
- 即使 hybrid 也只 +0.0041
- **失敗原因：**
  - SN=2 訓練樣本只有 ~10k，不足以訓出比全域模型更精準的 specialist
  - 全域 V12+V11 已透過 features_v7 receive priors 隱式處理 SN=2
  - LGB/XGB 樹結構天然能在 phase 特徵上分支 → 不需要顯式 expert
- **結論：MoE 方向暫不繼續，除非有更強的特徵或更多 SN=2 資料**

### ✅ 4-way blend — 最大突破
- OOF OV 0.3809（+0.0075 vs baseline）
- F1_a 從 0.4022 跳到 0.4128（+0.0106）
- 三個 GBM（V12cb, V12_5f, V12）+ V11 各有不同優勢，互補性強
- **結論：今晚最有效的突破**

### ❌ Aug + V8 features — 之前已失敗，今晚未繼續
- 已在前一份報告中分析

---

## 模型互補性分析

從 4-way 最佳權重看出每個模型的角色：

| 模型 | Action 權重 | Point 權重 | Server 權重 | 角色 |
|---|---|---|---|---|
| V12cb | 0.40 | 0.20 | 0.10 | Action 主力（CB 增加多樣性）|
| V12_5f | 0.40 | 0.20 | 0.50 | Action 副力 + **Server 主力** |
| V12 | 0.00 | 0.20 | 0.40 | Action 廢棄 + Server 副力 |
| V11 | 0.20 | 0.40 | 0.00 | **Point 主力** + Action 副力，Server 完全廢棄 |

**重大發現：V11 transformer 的 server 預測完全沒用（AUC 太低）**。這跟最佳 4-way blend 的 server 權重是 V12_5f=0.5 + V12=0.4 + V12cb=0.1 一致。

---

## Codex 建議的下一步分析方向

### 1. 先驗證 4-way 是否在 LB 真的有效
今日提交 4-way（slot 1）。如果 LB 有 ~0.36 預期值，繼續往 5-fold + CB 方向。如果 LB 跟 OOF 落差大（gap > 0.025），表示 4-way 有 OOF overfit 風險。

### 2. 如果 4-way 成功，下一步：V14 = V12 5-fold + CatBoost 全餐
- 5-fold + CB + V8 features 的單模型訓練
- 估計 8-10 小時（5-fold × 訓練時間 × 1.5 包含 CB）
- 預期 OOF +0.005~0.010 vs V12+CB

### 3. 加強 V11 transformer
- 目前 V11 在 ensemble 中扮演關鍵 point 角色
- 訓練更深的 transformer / 更多 epochs / 加 V8 features → 可能再 +0.005

### 4. Server-only 模型
- 今晚發現 V11 server 完全沒用
- 也許訓練一個專門的 server 模型（更深的網路/特徵工程）
- 預期 +0.002~0.005

---

## 已產生的可提交檔案

```
submissions/
├── submission_4way_optblend.csv                ⭐ OOF=0.3809 最佳
├── submission_v12cb_v12_5f_v11_optblend.csv    OOF=0.3787 三模型備用
├── submission_v12cb_v12_v11_optblend.csv       OOF=0.3782 (備選 1)
├── submission_v12cb_v12aug_v11_optblend.csv    OOF=0.3779 (備選 2)
├── submission_v12cb_v13_v11_optblend.csv       OOF=0.3777 (備選 3)
├── submission_sn2_expert_v12cb_blend.csv       OOF=0.3775 (SN=2 hybrid)
├── submission_v12cb_v11_optblend.csv           OOF=0.3771 (二模型)
├── submission_v12_v11_optblend.csv             OOF=0.3734 → LB=0.3541 (已提交)
└── submission_v12cb.csv                        V12+CB 單模 (V11 srv 變體)
```

---

## 已產生的 OOF 資產

```
oof_predictions/
├── v12_oof_*.npy           V12 baseline (3-fold)
├── v12aug_oof_*.npy        V12 + flip-aug (3-fold)
├── v13_oof_*.npy           V13 (features_v8)
├── v12cb_oof_*.npy         V12 + CatBoost (3-fold)         ⭐ NEW
├── v12_5f_oof_*.npy        V12 5-fold no-CB                 ⭐ NEW
├── sn2_expert_oof_*.npy    SN=2 specialist                  ⭐ NEW
└── v11_oof_*.npy           V11 transformer (existing)
```

對應 test 預測 + threshold params + 4-way 權重也都已存。

---

## 風險提示

### 4-way OOF Overfit 風險
- 4-way 用 grid search 找到的權重，理論上有 OOF overfit 風險
- 若 LB-OOF gap 從 -0.019 擴大到 -0.030，表示有過擬合
- **緩解：** 第二提交用 V12cb+V11 簡單 blend（OOF=0.3771）作為保險

### Per-class F1 不平衡
- BH_short (cls3) F1 仍 = 0.0163（102 樣本太少）
- Arch (cls8) F1 = 0.1243
- 這些罕見類別 LB 上可能波動

### Server AUC 高但低相關
- 4-way 的 server 完全是 V12_5f+V12 主導
- 若 LB test set 的 server 分佈不同，AUC 可能下滑

---

## 結論

**今晚最大發現：**
1. **CatBoost 是真正的 ensemble 多樣化來源**（V12+CB 單模就 +0.006）
2. **5-fold 對 AUC 有實質幫助**
3. **4-way blend 各任務用不同權重**才是最優解
4. **V11 transformer 的 server 完全無用**，但 point 主導性強
5. **SN=2 expert 失敗** — 全域模型已隱式處理 SN=2

**最終推薦：第一提交 4-way（OOF=0.3809），預期 LB 0.36+。**
