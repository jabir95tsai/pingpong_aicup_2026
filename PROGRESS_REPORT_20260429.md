# AI CUP 2026 進度報告
**日期**: 2026-04-29  
**供 Codex 分析下一步進程**

---

## 競賽目標

- 評分公式: `Score = 0.4 × F1_action + 0.4 × F1_point + 0.2 × AUC_server`
- Baseline: 0.28，目標衝 Top 3
- CV-LB gap: OOF 約比 LB 高 ~0.019（例如 OOF 0.3734 → LB 0.3541）

---

## 歷史 LB 成績（由高到低）

| 提交檔案 | LB 分數 | 排名 | 備註 |
|---|---|---|---|
| submission_v12_v11_optblend.csv | **0.3541608** | 26/156 | 當前最佳 |
| submission_v10_v11srv.csv | 0.3472 | 22/120 | V10+V11 srv |
| submission_v10.csv | 0.3379 | 28/127 | V10 alone |

---

## 模型架構演進

### 核心架構（V10 起不變）
- **Two-Pass Action→Point Stacking**: 先訓練 action 模型，再把 action probs 當特徵餵給 point 模型
- **LGB + XGB 各 3 fold**（+ 可選 CatBoost）
- **GroupKFold by match**（防 data leakage）
- **Ensemble**: GBM 混合 V11 Transformer（15-class action, 10-class point, binary server）

### 特徵版本

| 版本 | 特徵數 | 新增內容 |
|---|---|---|
| features_v6 | 1138 | 基礎版（時序 lag、one-hot、role features 等）|
| features_v7 | 1145 (+7) | Action-grammar priors（P(depth/side/valid \| prev_action, phase)）+ trigram + receive priors |
| features_v8 | 1175 (+30) | Point-grammar priors（P(pt_side \| prev_pt_side, action, phase)、ball physics、receive point priors、point trigram）|

---

## 今日訓練結果

### Raw OOF 比較（未做 threshold optimization）

| 模型 | F1_a | F1_p | AUC | OV(raw) |
|---|---|---|---|---|
| V12（V7 features, no-aug, no-CB, 3-fold）| 0.3743 | 0.2102 | 0.6055 | 0.3549 |
| V12aug（V7 + flip-aug, 3-fold）| 0.3747 | 0.2124 | 0.6042 | 0.3557 |
| V13（V8 features, no-aug, no-CB, 3-fold）| 0.3764 | 0.2092 | 0.6048 | 0.3552 |

### Blend + Threshold Optimization 後 OOF OV

| Ensemble | F1_a | F1_p | AUC | OOF OV |
|---|---|---|---|---|
| V12 + V11 optblend（**已提交**）| 0.4022 | 0.2279 | 0.6066 | **0.3734** |
| V12aug + V11 optblend | 0.4014 | 0.2292 | 0.6054 | 0.3733 |
| V13 + V11 optblend | 0.4028 | 0.2278 | 0.6060 | 0.3734 |
| 3-GBM(V12+V12aug+V13) 均等 blend raw | 0.3774 | 0.2096 | 0.6070 | 0.3562 |
| 3-GBM + V11 blend raw（alpha=0.6/0.55/0.95）| 0.3905 | 0.2235 | 0.6079 | 0.3672 |

---

## 今日驗證失敗的假設

1. **Flip Augmentation (FH↔BH)**: OV 從 0.3549 → 0.3557，進步幾乎為零。
   - **原因推測**: 桌球選手有慣用手，FH/BH 不對稱，翻轉會引入噪音
   - **結論**: 不繼續做 aug 訓練

2. **Features V8 (Point Grammar Priors)**: V13 raw OV = 0.3552 vs V12 = 0.3549，+0.0003
   - Point grammar 對 F1_p 幾乎無影響（0.2092 vs 0.2102，略退）
   - **原因推測**: LGB/XGB 本身已學習到 point transition patterns，顯式 lookup table 重複了已有的學習
   - **結論**: V8 特徵帶來的邊際效益接近零

3. **Hierarchical PointId（valid/depth/side 三 head）**: F1_p=0.158 vs flat 0.210（之前 V12 session 做的）
   - **結論**: 完全不行，Joint reconstruction 損失稀有類別資訊

---

## 當前瓶頸分析

### Per-SN Slice（V12+V11 blend）

| Slice | n | F1_a | F1_p | AUC | OV |
|---|---|---|---|---|---|
| **SN=2（接發球）** | 14995 | 0.2486 | 0.1606 | 0.5388 | **0.2714** |
| SN=3-4 | 23667 | 0.3470 | 0.2191 | 0.6105 | 0.3485 |
| SN=5-8 | 20075 | 0.3781 | 0.2175 | 0.6341 | 0.3651 |
| SN=9-12 | 6247 | 0.3537 | 0.2097 | 0.6185 | 0.3491 |
| SN>=13 | 4728 | 0.3496 | 0.2064 | 0.5702 | 0.3364 |

**SN=2 是最大瓶頸**（OV=0.2714，差全量 0.3652 約 10%）。SN=2 佔總數據 21.5%。

### Per-class F1 弱點

- **actionId**: 類別 8（拱球, F1≈0.02）、9（磕球, F1≈0.04）、14（放高球, F1≈0.05）極弱
- **pointId**: 類別 3（反手短, F1≈0.02）、1（正手短, F1≈0.13）稀少且難預測
- **整體**: F1_action（0.40）明顯優於 F1_point（0.23）；point 是主要瓶頸

---

## 現有未用的 Submission Slots

今日剩 **2 個 slots**：
- 手邊可立刻提交的：`submission_v12aug_v11_optblend.csv`、`submission_v13_v11_optblend.csv`
- OOF 均 = 0.3733~0.3734，與最佳持平，LB 效果未知

---

## Codex 請分析：最高 ROI 的下一步

以下幾個方向請評估可行性與預期增益：

### 方向 A：SN=2 Expert Model
- 只用 SN=2 的資料訓練獨立模型（約 15k 樣本）
- Features: serve_action（前一拍）、sex、score context + 物理特徵
- 預測: receive_action + receive_point
- 最終 blend: SN=2 rows 用 expert，其餘用 V12
- **預期增益**: SN=2 OV 從 0.27 → 0.30+，整體 OV ≈ +0.004

### 方向 B：5-fold 全量訓練（V12, 5-fold, no-aug, no-CB）
- 目前 V12 是 3-fold，V10 是 5-fold
- 更多 fold = 更穩定的 OOF estimate + 稍好的 ensemble
- **預估時間**: ~40 min（參考 V12 no-aug 3-fold 23.6 min，5-fold 約 39 min）
- **預期增益**: +0.002~0.005（更穩定的模型，非線性增益）

### 方向 C：CatBoost 加入（V12 + CB, 3-fold, no-aug）
- 目前 LGB+XGB，CB 可補足 ensemble 多樣性
- CB 在 categorical features 上有優勢（playerID, actionId lag 等）
- **預估時間**: ~2.5 小時
- **預期增益**: +0.005~0.010

### 方向 D：更好的 Point 特徵
- 目前 F1_p=0.228，是主要短板
- 測試集 player ID 去識別化，player-specific 特徵不可用
- 可能有效的方向：
  - P(pointId | prev_handId, prev_strengthId, prev_spinId) — 球的物理轉移
  - Rally-length features（長回合 vs 短回合的落點偏好）
  - Positional sequence（position → position → point）
- **注意**: features_v8 已試過類似方向，效果幾乎為零。深入分析原因前，此方向 ROI 不確定

### 方向 E：Model-Level Stacking（OOF 二階學習）
- 用 V12、V12aug、V13、V11 的 OOF 作為輸入，訓練 meta-learner（LR 或 LGB with few features）
- 預測最終 actionId/pointId 的 class probabilities
- **優點**: 自動學習最佳 blend weights
- **風險**: 小 holdout（69k samples × 3 folds），容易 overfit meta layer
- **預期增益**: +0.002~0.005（如果 meta 不 overfit）

### 方向 F：Test-Time Augmentation (TTA)
- 對 test data 做 FH/BH 翻轉，預測兩次取平均
- 即使 aug 對訓練沒用，TTA 可能對 test 有穩定效果
- **實作成本**: 低（約 30 min）
- **預期增益**: 不確定，aug 訓練沒用時 TTA 通常也沒用

---

## 可用的 OOF 資產（可供 meta-learning）

```
oof_predictions/
├── v11_oof_act.npy       (69712, 15)  V11 Transformer
├── v11_oof_pt.npy        (69712, 10)
├── v11_oof_srv.npy       (69712,)
├── v12_oof_act.npy       (69712, 19)  V12 no-aug no-CB 3-fold
├── v12_oof_pt.npy        (69712, 10)
├── v12_oof_srv.npy       (69712,)
├── v12aug_oof_act.npy    (69712, 19)  V12 + flip-aug 3-fold
├── v12aug_oof_pt.npy     (69712, 10)
├── v12aug_oof_srv.npy    (69712,)
├── v13_oof_act.npy       (69712, 19)  V13 (features_v8) no-aug 3-fold
├── v13_oof_pt.npy        (69712, 10)
└── v13_oof_srv.npy       (69712,)
```

對應 test:
```
v12_test_act/pt/srv/rally_uid.npy
v12aug_test_*.npy
v13_test_*.npy
v11_test_act/pt/srv.npy  (uid from submission_v11_transformer.csv)
```

---

## 目前 Submission 配額

- 每日上限: 3 次
- 今日已用: 1 次（V12+V11 optblend → LB 0.3541）
- 今日剩餘: **2 次**
- 建議用途: 提交最有機會突破的新模型組合

---

## 核心設計限制（請 Codex 遵守）

1. **不使用 SGP leakage**（serverGetPoint 不作為輸入特徵）
2. **Test set player ID 去識別化**：不可用 player-specific 特徵（只能用 global priors 或 rally-context features）
3. **GroupKFold by match**：OOF 必須按比賽分組，不可按 rally
4. **Fold-safe**: 所有統計 table（grammar priors 等）只能從 training fold 計算
5. **Serve class 規則**: strikeNumber=1 時 actionId 只能是 15~18（已 hard-coded 在 apply_action_rules()）
