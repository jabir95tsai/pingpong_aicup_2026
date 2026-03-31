# AI CUP 2026 春季賽 — 基於時序資料之桌球戰術與結果預測競賽

## 競賽概述
根據桌球比賽中前 n-1 拍的擊球資訊，預測第 n 拍的球種與落點，並推論該回合最終勝負。

## 預測目標（3 個任務）

| 任務 | 欄位名稱 | 說明 | 評估指標 | 權重 |
|------|---------|------|---------|------|
| 任務1 | `actionId` | 下一拍球種（19類，含0） | Macro F1-Score | 0.4 |
| 任務2 | `pointId` | 下一拍落點（10類，含0） | Macro F1-Score | 0.4 |
| 任務3 | `serverGetPoint` | 發球者是否得分（二元） | AUC-ROC | 0.2 |

**綜合評分：** `Score = 0.4 × S1 + 0.4 × S2 + 0.2 × S3`（各指標標準化至 0~1）

**Baseline score = 0.28**（需超過此分數才有獲獎資格）

## 資料結構

每一筆資料 = 某場比賽某個 rally 中的一次揮拍，按時間順序排列。

### 檔案
- `train.csv` — 訓練集（完整標註）
- `test.csv` — 測試集（不含預測目標）
- `sample_submission.csv` — 提交格式範例

### 欄位說明

| 欄位 | 說明 | 類型 |
|------|------|------|
| `rally_uid` | 小分唯一識別碼 | ID |
| `sex` | 比賽性別：1=男, 2=女 | 特徵 |
| `match` | 比賽唯一識別碼 | ID |
| `numberGame` | 第幾局 | 特徵 |
| `rally_id` | 局內小分編號 | 特徵 |
| `strikeNumber` | 小分內揮拍次序 | 特徵（重要） |
| `scoreSelf` | 主視角選手得分 | 特徵 |
| `scoreOther` | 對側選手得分 | 特徵 |
| `serverGetPoint` | 發球者是否得分（1/0） | **預測目標** |
| `gamePlayerId` | 主視角選手 ID | 特徵 |
| `gamePlayerOtherId` | 對側選手 ID | 特徵 |
| `strikeId` | 揮拍狀態 | 特徵 |
| `handId` | 正手/反手 | 特徵 |
| `strengthId` | 擊球力道 | 特徵 |
| `spinId` | 旋轉方式 | 特徵 |
| `pointId` | 落點位置（九宮格） | **預測目標** |
| `actionId` | 擊球方式 | **預測目標** |
| `positionId` | 球員站位區域 | 特徵 |

### 類別定義

#### strikeId（揮拍狀態）
- 1=發球, 2=接發球, 4=第三板之後, 8=無(未錄影), 16=暫停

#### handId（正反手）
- 0=無, 1=正拍, 2=反拍

#### strengthId（力道）
- 0=無, 1=強, 2=中, 3=弱

#### spinId（旋轉）
- 0=無, 1=上旋, 2=下旋, 3=不旋, 4=側上旋, 5=側下旋

#### pointId（落點，九宮格）
- 0=無/未落在九宮格（掛網出界等）
- 1=正手短, 2=中間短, 3=反手短
- 4=正手半出台, 5=中路半出台, 6=反手半出台
- 7=正手長, 8=中間長, 9=反手長

#### actionId（球種）
- 0=無/其他
- **進攻(Attack):** 1=拉球, 2=反拉, 3=殺球, 4=擰球, 5=快帶, 6=推擠, 7=挑撥
- **控制(Control):** 8=拱球, 9=磕球, 10=搓球, 11=擺短
- **防守(Defensive):** 12=削球, 13=擋球, 14=放高球
- **發球(Serve):** 15=傳統, 16=勾手, 17=逆旋轉, 18=下蹲式

#### positionId（站位）
- 0=無, 1=左, 2=中, 3=右

## 提交格式
- CSV 檔案，UTF-8（無BOM）編碼，Unix 換行字符
- 參考 `sample_submission.csv` 的格式
- 每日上傳上限 3 次

## 重要規則
- 必須使用 ML/DL 方法，禁止人工修正結果
- 可使用自製資料或開源資源
- 嚴禁反向比對真實比賽影片
- 各隊伍間不可私下共享程式及特徵值

## 建模注意事項
- 這是時序預測問題：用前 n-1 拍預測第 n 拍
- 類別不平衡嚴重（使用 Macro F1 評估）
- actionId 在 strikeId=1（發球）時只會是 15~18（Serve 類）
- strikeNumber=1 一定是發球
- serverGetPoint 是 rally 層級的預測，同一 rally 內所有拍的 serverGetPoint 相同
- 測試集已去識別化，match/player ID 可能與訓練集不同

## 建議的專案結構
```
project/
├── CLAUDE.md          # 本檔案
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/         # EDA 和實驗
├── src/               # 模型程式碼
├── models/            # 儲存訓練好的模型
├── submissions/       # 產生的提交檔案
└── requirements.txt
```
