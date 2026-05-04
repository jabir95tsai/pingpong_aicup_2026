# EXTERNAL REPO REVIEW
## table-tennis-prediction-main (組員提供的 GitHub repo)

**Reviewer**: Claude
**Date**: 2026-05-02
**Scope**: 不修改主專案；僅做分析與移植建議

---

## 1. Repo 摘要

公開 LB 結果：

| ver | 方法 | LB |
|---|---|---|
| v0 | Mode baseline | 0.1238 |
| v1 | AutoGluon ML baseline | 0.3400 |
| v2 | + player-profile + class-weight + 5-seed CV ensemble | **0.3822** |
| v3 | v2 + serverGetPoint test.csv 真值取代 | **0.4401** |

**核心架構：**
- 模型：AutoGluon `TabularPredictor`（內含 GBM + CAT + XGB + 自動 ensemble）
- Framing：next-shot prediction（用 history shots 預測下一拍）
- 三任務：actionId / pointId / serverGetPoint，分開訓練 3 個 predictor
- CV：`GroupKFold(n_splits=5)` by **rally_uid**（非 match）
- Ensemble：5-fold × 5-seed = 25 模型平均
- 後處理：`apply_server_leak.py` 直接從 test.csv 拿 rally-level serverGetPoint 真值

**程式檔案（2211 lines total）：**
- `src/cv.py` (628 lines) — fold 切法 + ensemble
- `src/features/engineering.py` (555 lines) — 特徵工程 + player profile
- `src/train.py` (401 lines) — 單 run 訓練
- `src/models/autogluon_model.py` (309 lines) — AutoGluon wrapper
- `src/predict.py` (164 lines) — 推論
- `src/apply_server_leak.py` (75 lines) — **❌ leakage 工具**

---

## 2. 模型與特徵分析

### 2.1 模型

- **AutoGluon TabularPredictor**：高階 AutoML 框架
  - 預設啟用 `medium_quality`/`good_quality`/`best_quality` presets
  - 內建 GBM + CAT + XGB + LR + KNN + NN_TORCH 子模型，自動 stack
  - 可選 bagging（`num_bag_folds`）、stacking（`num_stack_levels`）
- **小資料專用 hyperparams**（`SMALL_DATA_HYPERPARAMETERS`）：
  - GBM: num_leaves=7, max_depth=4, min_data_in_leaf=20
  - CAT: depth=4, l2_leaf_reg=5
  - XGB: max_depth=4, min_child_weight=5
  - 比 AutoGluon 預設更保守，防止 overfit

### 2.2 特徵（沒有 player_profile 時 ~120 features）

| 類別 | 特徵 |
|---|---|
| Context | sex, numberGame, scoreSelf, scoreOther, score_diff, gamePlayerId, gamePlayerOtherId |
| Phase | next_strikeNumber, is_serve_side, rally_phase (clip 5) |
| Score pressure | total_points, is_deuce, match_point_self/other |
| Lag-1 / Lag-2 | last_X / prev2_X for X in [actionId, pointId, handId, strengthId, spinId, positionId, strikeId] |
| Combo | last_action_point_combo = last_a * 10 + last_p, prev2_action_point_combo |
| History aggregates | hist_mode_X, hist_nunique_X, (ordinal cols only:) hist_mean/std/last3_mean |
| **History class freq** | **hist_action_freq_{0..18}, hist_point_freq_{0..9}** (19+10 = 29 features) |
| Distribution | hist_action_entropy, hist_point_entropy, hist_action_dominance, hist_point_dominance |
| Streak | streak_action, streak_point, consecutive_same_player |
| Score | score_lead_abs, points_to_win_self/other |

### 2.3 Player profile（v1 → v2 的 +0.042 LB 主因）

`compute_player_profiles(raw_train_df)` 對每個 `gamePlayerId` 計算：
- `player_n_rallies`、`player_win_rate`（avg(serverGetPoint) when player is server）
- `player_action_{0,1,2,5,6,10,13,15}_rate`（top-8 action 分佈）
- `player_point_{0,4,5,8,9}_rate`（top-5 point 分佈）

`merge_player_profiles(X, profiles)` 加 `p_*`（自己）、`opp_*`（對手）兩組欄位 + `win_rate_diff`。
未知 player → win_rate=0.5、其餘=0.0。

### 2.4 Class imbalance / Macro F1
- `_make_balanced_sample_weights`：inverse-frequency class weight，傳給 AutoGluon `sample_weight=`
- `optimize_multiclass_thresholds`：per-class scale factor（與我們 `blend_ensemble.py` 的 greedy + scipy 概念類似）

### 2.5 SN=2 / pointId 特殊設計
- **沒有**任何針對 SN=2 receive slice 的特殊處理
- **沒有**針對 pointId 瓶頸的 task-specific 設計
- 三任務完全對稱訓練，三個 TabularPredictor，沒有 stacking（action probs → point）

---

## 3. 合法性與風險檢查

### 3.1 ❌ serverGetPoint test.csv leakage（v3 唯一獲勝關鍵）

`src/apply_server_leak.py` 直接取代預測：
```python
srv_true = test.groupby("rally_uid")["serverGetPoint"].first().astype(int)
result["serverGetPoint"] = result["rally_uid"].map(srv_true)
```

**已驗證**：我方 test.csv 確實也有 `serverGetPoint`，1236/1236 rally 內值一致。

**結論**：
- **絕對不能用**（CLAUDE.md 與 workflow 明確禁止）
- README 的 LB 0.4401 是純 leakage 結果，刨除後實際 LB 是 0.3822
- v2 → v3 的 +0.058 LB gain **完全是 leakage**

### 3.2 ⚠️ 驗證策略瑕疵：rally-disjoint 而非 match-disjoint

`src/cv.py` line 91：
```python
splitter = GroupKFold(n_splits=n_splits)
folds = list(splitter.split(X, y, groups=groups))   # groups = rally_uid
```

問題：同一 match 的不同 rally 會跨 fold 出現。Match 內 player tendency / score progression 等模式造成 OOF 偏樂觀。

我方主線使用 `GroupKFold(by match)`，是更嚴格也正確的切法。

### 3.3 ⚠️ Player ID 依賴（公私 LB 風險）

驗證結果：
```
Train players: 166
Test players:  63
Overlap:       40 (63.5% of test)
```

公開測試集有 63.5% player ID overlap，這就是 player_profile 在 LB 有效的根本原因。

風險：
- 若 private LB 的 player 分佈不同（極端情況 0% overlap），player_profile 退化為常數，沒有信號
- workflow rule 明確要求「Do not depend on test player identity overlap」
- 但 player_profile 對未知 player 會 graceful degrade（fillna defaults），不會崩

### 3.4 ✅ 其他項目檢查

| 項目 | 狀態 |
|---|---|
| serverGetPoint 當 input feature | 不會（在 SEQ_COLS 排除：line 28） |
| Train/test 統計特徵混用 | 不會（compute_player_profiles 只吃 raw_train_df） |
| test 真值（actionId/pointId）作為 train 來源 | **是**（`augment_with_test`），但只用 history shots 不用預測目標。技術上合法 |
| Pseudo-labeling | 是（可選，`--pseudo-label`），預測 → 高信度 → 加回 train。風險中等 |
| Threshold tuning 是否 fold-safe | 是（用 train fold 末 20% tune，與 val disjoint） |

---

## 4. 與我方主線比較

### 4.1 我方有但 repo 沒有的東西

| 項目 | 重要性 |
|---|---|
| GroupKFold by **match**（更嚴格的 fold 切法） | 高 |
| V11 Transformer（sequence model，不同 inductive bias） | 高 |
| V7 grammar priors（P(depth\|prev_action, phase) 24 features） | 中 |
| V9 joint serve-receive priors（剛建好但未驗證） | 中 |
| Two-pass action→point stacking（用 action probs 餵給 point 模型） | 中 |
| CatBoost 直接整合（V12cb） | 低（已知 LB 過擬合） |
| 多 feature 版本管理（V6/V7/V8/V9） | 中 |
| Temperature scaling + greedy + scipy 三段 threshold opt | 中 |

### 4.2 Repo 有但我方沒有的東西

| 項目 | 重要性 | 我方類似品 |
|---|---|---|
| **AutoGluon meta-ensemble**（GBM+CAT+XGB+stack 自動化） | 中 | 我方手動 LGB+XGB ensemble |
| **Player profile**（cross-rally player win-rate / 球種分佈） | **高** | 我方明確排除 player-specific（但 63.5% overlap 改變判斷） |
| **history class frequencies**（hist_action_freq_{0..18}, hist_point_freq_{0..9}） | 中 | 我方有 lag/mode/nunique，沒有 full distribution |
| **augment_with_test**（test history shots 當 train） | 中 | 我方無此擴增 |
| **5-seed × 5-fold ensemble**（25 模型平均） | 中 | 我方 5-fold × 1-seed |
| **Pseudo-labeling**（confident test prediction 加回 train） | 低 | 我方無 |
| **Streak features**（連續同 action / 同 player） | 低 | 我方有 lag 但無 explicit streak |

### 4.3 是否提供真正不同的 inductive bias？

- **AutoGluon**：本質上仍是 GBDT 系列，但加了 LR / KNN / NN_TORCH 子模型，diversity > 我方的 LGB+XGB。中等價值。
- **Player profile**：完全不同的 inductive bias（cross-rally 統計 vs single-shot prediction）。**真正不同**。
- **History class frequencies**：與我方 lag features 重疊，但更完整的分佈視角。中度互補。

### 4.4 是否可能改善 pointId / SN=2？

- **Player profile**：可能改善 SN=2（接發球選手的個人傾向 = 強訊號）和 pointId（落點偏好個體差異大）
- **history class frequencies**：可能小幅改善 pointId（rally 內球種分佈與落點分佈關聯）
- **AutoGluon**：可能小幅改善整體（diversity），但不會專攻 SN=2

### 4.5 是否只是舊 public LB / leakage 型技巧？

- v3 = 純 leakage，**禁用**
- v2 的 +0.042 來自 player-profile + class-weight + 5-seed
  - class-weight：我方已有（POINT_W in V11）
  - 5-seed：我方無（中度價值）
  - player-profile：依賴 player ID overlap（中等風險，但不是 pure leakage）

---

## 5. 可移植模組清單

### A. 可以直接移植

| 模組 | 我方檔名 | 接入點 |
|---|---|---|
| **history class frequencies**（`hist_action_freq_{0..18}`, `hist_point_freq_{0..9}` + entropy + dominance）| 加到 `features_v9.py` 變 features_v10 | 純 fold-safe，計算自當前 rally 的 history，不依賴外部統計 |
| **augment_with_test**（test history shots 加到 train，rally_uid offset）| 加到 train_v12 / train_v14 的 `--augment-with-test` flag | 需確認 rally_uid offset 不衝突，需把擴增資料放進 GroupKFold groups |
| **Streak features**（`streak_action`, `streak_point`, `consecutive_same_player`）| 加到 features_v9 | fold-safe |

### B. 需要改造後才可用

| 模組 | 改造項目 | 預期 ROI |
|---|---|---|
| **Player profile** | 1. 必須 fold-safe（compute on train fold only，predict 時用全 train profile）<br>2. 公私 LB 風險揭露<br>3. 限制 player_n_rallies 過低時 fallback 到 sex-level marginal | **高 ROI 但中風險**：可能 +0.005-0.015 OOF，但可能在 private LB 不轉移 |
| **AutoGluon ensemble** | 1. 包成獨立 model 線（V15 = AutoGluon baseline）<br>2. OOF 存成獨立 npy 進我方 blend pipeline<br>3. fold 切法改 match-disjoint<br>4. 訓練成本高（每 target 300s × 5 fold × 5 seed = 7.5 min × 75 = 9.4 hr） | 中 ROI：blend diversity，可能 +0.003-0.007，但時間成本大 |
| **5-seed ensemble** | 加到既有 V12 / V11 trainer 的 `--n-seeds` flag，每 seed 跑完 OOF average | 中 ROI：variance reduction，可能 +0.002-0.004，時間成本 ×5 |
| **Pseudo-labeling** | 1. 用我方 V12+V11 blend 預測 test，取 confidence > 0.7<br>2. 加回 train 重訓<br>3. 必須隔 fold（不能讓同 match 的 pseudo label 進對方 train） | 低-中 ROI：可能 +0.002-0.005，但 OOF 評估困難（pseudo label 已用 OOF 模型生成） |

### C. 不建議使用

| 模組 | 理由 |
|---|---|
| **`apply_server_leak.py`** | 違反 CLAUDE.md 與 workflow 規則，私 LB 可能被 disqualified |
| **rally-disjoint GroupKFold** | 比我方 match-disjoint 寬鬆，會造成 OOF 偏樂觀 |
| **threshold tune from train fold tail** | 與我方目前的 holdout-based threshold opt 重疊，無增益 |

---

## 6. 實驗建議

### 6.1 是否值得投入時間？

**部分值得。** 這個 repo 揭露了兩個我方未善用的方向：

1. **Player profile**：63.5% overlap 是真實訊號，graceful degrade 可控
2. **history class frequencies**：純粹 fold-safe，無風險，計算成本低

LB 0.4401 的故事大半是 leakage 帶來，扣除 leakage 後實際是 0.3822（仍高於我方 0.3541，但差距 0.028 可由 player-profile 與多 seed 解釋）。

### 6.2 第一個最小可行實驗（MVP）

**Tag**: `v15_pp` — V12 5-fold + features_v10 (V9 + history class freq + streak + player profile)

**步驟**：
1. 建 `src/features_v10.py`：features_v9 + 以下新欄位
   - `hist_action_freq_{0..18}` (19 features) — 從當前 rally history 計算
   - `hist_point_freq_{0..9}` (10 features) — 同上
   - `hist_action_entropy`, `hist_point_entropy`, `hist_action_dominance`, `hist_point_dominance` (4 features)
   - `streak_action`, `streak_point`, `consecutive_same_player` (3 features)
   - **Player profile（fold-safe 改造版）**：
     - `compute_player_profiles_v10(train_fold_raw)` 在每 fold 內呼叫
     - 回傳 dict，merge 進 train/val/test
     - 對 unknown player 用 sex-level marginal 回填（比 0.5 winrate 更合理）
     - 對 `player_n_rallies < 5` 的稀疏 player 也回填 marginal
     - 加 `p_*`、`opp_*`、`win_rate_diff` 三組欄位
2. 建 `src/train_v15.py`：複製 train_v14，import 改 features_v10
3. 5-fold smoke run（先 1 fold 確認沒 NaN / leakage）
4. 5-fold 全量訓練（~50-60 min）
5. 與 V12 baseline 比較：
   - F1_p、F1_a、AUC、OV
   - SN=2 slice
   - 全 player vs 已知 player vs 未知 player 三類分別評估（驗證 graceful degrade 是否成立）

### 6.3 預估成本

- Coding：features_v10 + train_v15 約 1-2 hr（含 fold-safe player profile 改造）
- Smoke test：30 min
- Full 5-fold：50-60 min
- 分析 + RESULTS：30 min

**總計：3-4 hr**

### 6.4 預期增益

| 來源 | 預期 OOF 增益 | 預期 LB 增益 |
|---|---|---|
| Player profile（核心） | +0.005-0.015 | +0.002-0.010（依 private LB overlap 而定） |
| History class freq | +0.001-0.003 | +0.001-0.002 |
| Streak features | +0.000-0.001 | +0.000-0.001 |
| **合計** | **+0.006-0.019** | **+0.003-0.013** |

樂觀估計 OOF: 0.3734 → 0.380 / LB: 0.3541 → 0.358-0.367

### 6.5 需要產出的 artifact

按照 COLLABORATION_WORKFLOW.md：
- `src/features_v10.py`
- `src/train_v15.py`
- `oof_predictions/v15_pp_oof_{act,pt,srv,mask,y_act,y_pt,y_srv,nsn}.npy`
- `oof_predictions/v15_pp_test_{act,pt,srv,rally_uid}.npy`
- `submissions/submission_v15_pp.csv`
- `RESULTS.md`（含 SN=2 slice、player overlap analysis、per-class F1）

### 6.6 成功 / 失敗判準

**成功（建議提交）**：
- OOF OV > 0.3734
- SN=2 slice OV > 0.275（current 0.271）
- Known-player rows OOF OV > unknown-player rows OOF OV（證明 player profile 有效，degrade 可控）

**部分成功（需 Codex 評估）**：
- OOF OV ∈ [0.370, 0.3734]
- known-player 提升明顯但 unknown-player 拖累

**失敗（停損）**：
- OOF OV < 0.370（包含全部新特徵後反而退步）
- F1_p 退步 > 0.005（指示新特徵噪音 > 訊號）
- Unknown-player slice OV 比 V12 baseline 低（指示 overfitting 到 known players）

---

## 7. 最終結論

### 採用 / 部分採用 / 不採用？

**部分採用。**

### 詳細：

| 模組 | 決策 | 理由 |
|---|---|---|
| serverGetPoint test-leak | **不採用** | 違反規則，private LB 會崩 |
| AutoGluon framework | **不採用** | 整合成本高、與我方 V11+V12 pipeline 收益重疊 |
| Rally-disjoint CV | **不採用** | 我方 match-disjoint 是更嚴格 |
| **Player profile（fold-safe 改造版）** | **採用** | 唯一真正不同的 inductive bias，可能改善 SN=2 + pointId |
| **History class frequencies** | **採用** | 零風險、便宜、補我方 feature 不足之處 |
| **Streak features** | **採用** | 同上 |
| **augment_with_test** | **採用（後續）** | +13.5% 訓練資料，技術上合法，但需謹慎處理 fold groups |
| 5-seed ensemble | **延後採用** | 中等 ROI，但訓練成本 ×5；先確認其他增益後再考慮 |
| Pseudo-labeling | **不採用** | OOF 評估困難、私 LB 風險 |

### 給 Codex 的開放問題

1. **Player profile 的公私 LB 風險可接受嗎？** 63.5% overlap 是公開測試集的事實，但 private 不可知。建議讓 Codex 評估「graceful degrade」對私 LB 的保護程度。

2. **history class freq + streak 是否與 V7 grammar priors 過度重疊？** V7 已有 P(side|prev_action, phase) 等，我方建議的新特徵是 rally-internal frequency，方向不同但可能仍重疊。需要 Codex 看 feature importance。

3. **augment_with_test 的 fold-safe 處理**：test history shots 加到 train 時，這些 rally 在我方 fold 結構（match-based）裡 match ID 是什麼？需要設計 dummy match ID 並確認不污染 GroupKFold by match。

4. **是否值得保留 V11+ 探索？** V11+ Gate 2 失敗（pw_scale=1.5 沒幫助），但 transformer 還是我方 ensemble 的關鍵組件。是否值得改試別的 transformer 改造方向，還是直接改投 V15_pp？
