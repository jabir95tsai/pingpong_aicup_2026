# AI CUP 2026 桌球預測 — 專案規劃

## 目標
根據前 n-1 拍的球種與落點，預測：
1. **Task 1**: 下一拍球種 (`actionId`, 19類) → Macro F1-Score
2. **Task 2**: 下一拍落點 (`pointId`, 10類) → Macro F1-Score
3. **Task 3**: 發球者得分 (`serverGetPoint`, 二元) → AUC-ROC

綜合評分 = 0.4 × S1 + 0.4 × S2 + 0.2 × S3 (需超過 Baseline 0.28)

---

## 整體策略

### Phase 1: 資料探索與特徵工程（EDA）
- **資料分析**
  - 各類別分布（處理不平衡問題）
  - Rally 內時序特性
  - 球員差異性

- **特徵生成** (~950-1200 features)
  - **Statistical**: 均值、標準差、最值、四分位數
  - **Time Domain**: 時序變化趨勢、速度、加速度
  - **Frequency Domain**: FFT 特徵
  - **Multi-Scale**: 不同窗口大小的滑動統計
  - **Rally Context**: 當前得分、局數、拍數序列
  - **前歷史特徵**: 前 1~5 拍的動作、落點、力道、旋轉
  - **玩家特徵**: 玩家習慣（該玩家常用的球種、落點分布）

- **數據增強**
  - SMOTE 處理不平衡類別
  - 或自訂類別權重

### Phase 2: 特徵選擇（Feature Auto Selection）

**方式**：三階段篩選（參考冠軍方案）

1. **XGBoost Gain**（快速評估）
   - 訓練初版 XGBoost，提取 feature importance
   - 保留 Top 600 個特徵

2. **TreeSHAP**（精細評估）
   - 計算 SHAP values，評估每個特徵對預測的貢獻度
   - 篩選至 300 個

3. **Cross Importance**（多折驗證穩定性）
   - 5-Fold CV 中每 fold 都計算重要性
   - 最終保留在多個 fold 中都重要的特徵 (~200-300 個)

### Phase 3: 模型訓練

**多模型集成方案**：

1. **Task 1 & 2（多分類，Macro F1）**
   - **主模型**: XGBoost / LightGBM / CatBoost
     - 多類別損失函數
     - 類別權重調整（Macro F1 友好）
   - **輔助模型**:
     - Neural Network (3-4層 MLP，Dropout)
     - 可選：Stacking 或 Voting Ensemble
   - **後處理**:
     - 規則約束（如 strikeId=1 時 actionId 只能是 15-18）
     - 溫度調整 (Temperature Scaling)

2. **Task 3（二分類，AUC-ROC）**
   - **主模型**: XGBoost with scale_pos_weight
   - **評估指標**: AUC-ROC, AUC-PR
   - **閾值優化**: 根據訓練集找最優閾值

### Phase 4: 模型驗證與優化
- **驗證策略**: Stratified K-Fold (K=5)
- **超參數調優**:
  - 網格搜索 / 隨機搜索（小範圍）
  - 或 Optuna 進行貝葉斯優化
- **特徵互動**: 試驗手動設計的特徵交互（如 actionId × strikeId）

### Phase 5: 提交
- 在測試集上產生預測
- CSV 格式：UTF-8 無 BOM，Unix 換行符
- 每日上傳上限 3 次

---

## 技術棧

| 層級 | 工具 |
|------|------|
| 資料處理 | pandas, numpy, scikit-learn |
| 特徵工程 | pandas, scipy, librosa (信號處理) |
| 特徵選擇 | XGBoost, SHAP, scikit-learn |
| 建模 | XGBoost, LightGBM, CatBoost, PyTorch |
| 驗證 | scikit-learn (cross_val_score, metrics) |
| 可視化 | matplotlib, seaborn |

---

## 檔案結構

```
project/
├── data/
│   ├── train.csv          ← 訓練集（完整標註）
│   ├── test.csv           ← 測試集
│   └── sample_submission.csv
│
├── notebooks/
│   ├── 01_eda.ipynb              # 資料探索
│   ├── 02_feature_engineering.ipynb  # 特徵生成
│   ├── 03_feature_selection.ipynb    # 特徵篩選
│   └── 04_model_training.ipynb       # 模型訓練
│
├── src/
│   ├── features.py             # 特徵生成函數
│   ├── models.py               # 模型定義與訓練
│   ├── utils.py                # 工具函數
│   └── config.py               # 設定常數
│
├── models/
│   ├── xgb_task1.pkl
│   ├── xgb_task2.pkl
│   └── xgb_task3.pkl
│
├── submissions/
│   └── submission_v*.csv
│
├── PROJECT_PLAN.md             ← 本檔案
├── CLAUDE.md
├── requirements.txt
└── README.md
```

---

## 重點注意事項

1. **時序特性**
   - 同一 rally 內的球經常有相關性
   - 需要按 `rally_uid` + `strikeNumber` 排序
   - 避免時序洩露（測試集中不能有訓練集的後續球拍）

2. **類別不平衡**
   - actionId 和 pointId 都有少數類別
   - Macro F1-Score 自動給不同類別相同權重
   - 考慮 SMOTE、class_weight 或過採樣

3. **規則約束**
   - strikeId=1（發球）→ actionId ∈ {15, 16, 17, 18}
   - strikeNumber=1 一定是發球
   - serverGetPoint 是 rally 級別（同 rally 內所有拍相同）

4. **去識別化問題**
   - 測試集的 match ID、player ID 可能與訓練集不同
   - 不要過度依賴玩家特徵，應注重通用的動作特徵

5. **計算資源**
   - 特徵工程可能產生 1000+ 維
   - XGBoost + SHAP 計算量較大，考慮採樣或分批

---

## 預期時程（待資料公佈）

1. **EDA**: 1-2 天（理解資料分布）
2. **特徵工程**: 3-5 天（創造 950+ 特徵）
3. **特徵選擇**: 2-3 天（篩選至 200-300 個）
4. **模型訓練**: 3-5 天（超參數調優、集成）
5. **驗證與調整**: 2-3 天（交叉驗證、後處理）
6. **提交**: 1 天

---

## 成功指標

- Baseline: 0.28
- 目標: **0.45+**（相對於冠軍的成績有競爭力）
