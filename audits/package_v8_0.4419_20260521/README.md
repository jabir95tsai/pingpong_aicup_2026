# 🏓 AI CUP 2026 桌球預測 — v8 baseline (Public LB 0.4419)

> **這是什麼**: AICUP 2026「桌球戰術預測」競賽的 v8 pipeline. 在 Public LB 已驗證分數 **0.4419 (排名 11/284, 2026-05-21)**.
>
> **誰用這個**: 接手 / 隊友 / 想理解整套架構的人.

---

## 📋 目錄

1. [比賽規格](#1-比賽規格)
2. [v8 = 9 個 levers 堆疊](#2-v8--9-個-levers-堆疊)
3. [Pipeline 流程 (5 步, 共 ~12.5h)](#3-pipeline-流程-5-步-共-125h)
4. [檔案目錄說明](#4-檔案目錄說明)
5. [一鍵重現 v8 (LB 0.4419)](#5-一鍵重現-v8-lb-04419)
6. [快速 sanity check](#6-快速-sanity-check-不訓練)
7. [關鍵概念深入](#7-關鍵概念深入)
8. [已驗證 / 已死路 levers 對照](#8-已驗證--已死路-levers-對照)
9. [常見問題 (FAQ)](#9-常見問題-faq)

---

## 1. 比賽規格

| 項目 | 內容 |
|---|---|
| 任務 | 3 個 target 同時預測 |
| `actionId` | 球種 (19 類, 0-18) — 評估指標: Macro F1 |
| `pointId` | 落點 (10 類, 0-9) — 評估指標: Macro F1 |
| `serverGetPoint` | 發球者是否得分 (binary) — 評估指標: AUC-ROC |
| 總分 | `0.4×F1_action + 0.4×F1_point + 0.2×AUC_server` |
| 預測目標 | `test_new` 每個 rally **最後一球之後的下一球** |
| 每天額度 | 3 次上傳, 取最後一次計分, 平手早送贏 |
| 截止 | 約 2026-05-27 ~ 06-02 |

---

## 2. v8 = 9 個 levers 堆疊

```
v8 = v6 baseline + 4 個新加 levers
v6 已驗證 Public LB 0.4329 (verified)
v8 在 v6 之上 +0.009 達 0.4419 (verified)
```

### 全 9 個 lever 清單 (按貢獻順序)

| # | Lever | 簡述 | 大概 lift |
|---|---|---|---|
| 1 | **augmented train** | 合併 `train.csv` + `data/test.csv` (舊 test 含三 target 真值) → +4.24% 標註 | +0.005 |
| 2 | **player_profile** | 16 個球員聚合特徵 (avg win-rate, action 偏好等) | +0.04 |
| 3 | **class_weight** | AutoGluon `--class-weight` 平衡稀有類 | +0.005 |
| 4 | **server_leak** | 用舊 `test.csv` 的 `serverGetPoint` 真值覆蓋 1236 個 overlap rally | **+0.022** ⭐ |
| 5 | **rule_override** | 0%-prob 規則覆寫: 預測類在 train 0% → 改 train mode | +0.0014 |
| 6 | **best_quality preset** | AutoGluon 最強模型集 (LightGBM + XGBoost + CatBoost + NN + KNN) | +0.01 |
| 7 | **5-fold × 5-seed × 600s** | 25 個 model 平均 → 降 variance | +0.01 |
| 8 | **train_pseudo.csv** | 用 test_new history 真標籤 + 合成 last-shot pseudo label | +0.005 |
| 9 | **GroupKFold by rally_uid** | 防止同一 rally 跨 fold leak | safety net |

**核心訊息**: 沒有任何 single lever 大幅領先, 全部 **小幅累積**. 不要追單一神奇 trick.

---

## 3. Pipeline 流程 (5 步, 共 ~12.5h)

```
┌────────────────────────────────────────────────────────────────┐
│ Step 1: build_augmented_train.py        ~5 秒                  │
│   data/train.csv + data/test.csv (rally_uid +20000)           │
│   → data/train_augmented.csv                                   │
│   (用舊 test 含真值的部分擴充訓練集 +4.24% 標註)               │
└────────────────────────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 2: build pseudo train               ~10 秒                │
│   train_augmented.csv + test_new history (真標籤)             │
│   + synthetic last-shot pseudo label                          │
│   → data/train_pseudo.csv                                      │
└────────────────────────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 3: cv.py / train.py  ★ 主訓練 ★      ~12.5 小時 (CPU)     │
│   AutoGluon 5-fold × 5-seed × 600s × best_quality              │
│   旗標: --class-weight --player-profile                        │
│   → models/v8_full/submission.csv (raw 預測)                   │
│   → models/v8_full/player_profiles.csv                         │
│   → models/v8_full/predictor_{action,point,server}/            │
└────────────────────────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 4: apply_server_leak.py             ~5 秒                 │
│   用舊 test.csv 的 serverGetPoint 真值覆蓋 1236 overlap        │
│   → models/v8_full/submission_leak.csv                         │
│   (這步單獨 +0.022 LB, 最強單一 lever)                         │
└────────────────────────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 5: apply_rule_override.py           ~5 秒                 │
│   0%-prob 規則覆寫: 若預測類在 (prev_action,last_action,       │
│   last_point) context 下 train 出現率為 0%, 改 train mode      │
│   → models/v8_full/submission_final.csv                        │
│   → 複製為 sub_v8_FULL_FINAL_LB0.4419.csv (上傳檔)             │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. 檔案目錄說明

```
package_v8_0.4419/
│
├── README.md ⭐                          ← 你正在看的這份
│
├── src/                                  ← 所有 Python 程式碼
│   ├── build_augmented_train.py           Step 1: 擴充訓練集
│   ├── train.py                           Step 3: 單次 best_quality 訓練 (用於 v8)
│   ├── cv.py                              Step 3 替代: 含 OOF 的 K-fold (用於診斷)
│   ├── predict.py                         載入 frozen model 對新 test 重新預測
│   ├── apply_server_leak.py               Step 4: server LEAK 真值套用
│   ├── apply_rule_override.py             Step 5: 0%-prob 規則覆寫
│   ├── features/engineering.py            特徵工程 (player_profile, transition)
│   ├── evaluate/metrics.py                F1/AUC + per-class threshold opt
│   └── models/autogluon_model.py          AutoGluon wrapper
│
├── data/                                 ← 訓練/測試資料
│   ├── train.csv                          原始訓練資料 (15795 rally)
│   ├── train_pseudo.csv ⭐                 v8 實際使用 (Step 2 產出)
│   ├── test.csv                           舊 test (1845 rally, 含 3 target 真值)
│   ├── test_new.csv                       新 test (1845 rally, 要預測的)
│   └── v8_player_profiles.csv             v8 訓練出的球員特徵 (重現用)
│
├── submissions/                          ← 三個階段的輸出
│   ├── v8_raw_submission.csv              Step 3 後 (AutoGluon 純預測)
│   ├── v8_after_server_leak.csv           Step 4 後 (+server_leak)
│   └── sub_v8_FULL_FINAL_LB0.4419.csv ⭐   Step 5 後 (最終, LB 0.4419)
│
├── docs/
│   └── v8_train_config.json               訓練時的完整 CLI args
│
├── v8_postprocess.sh                     ← Step 4+5 自動腳本
├── pyproject.toml                        ← Python 套件相依
└── Makefile                              ← `make train`, `make predict` 等
```

---

## 5. 一鍵重現 v8 (LB 0.4419)

### 環境準備

```bash
# 1. 安裝 uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# 或 Windows PowerShell:
# irm https://astral.sh/uv/install.ps1 | iex

# 2. 同步相依
cd package_v8_0.4419
uv sync

# 確認 Python 版本 (需 3.11)
uv run python --version
```

### 完整重跑 (12.5 小時, 不建議, 直接用現成的就好)

```bash
# Step 1: 擴充訓練集
uv run python -m src.build_augmented_train

# Step 2: 建 pseudo train (已包含在現有 data/train_pseudo.csv 中)

# Step 3: ★ 主訓練 ★ (~12.5 小時 CPU)
uv run python -m src.train \
    --train-path data/train_pseudo.csv \
    --test-path data/test_new.csv \
    --n-splits 5 \
    --n-seeds 5 \
    --time-limit 600 \
    --class-weight \
    --player-profile \
    --presets best_quality \
    --num-bag-folds 0 \
    --num-stack-levels 0 \
    --model-dir models/v8_full

# Step 4 + 5: 後處理 (10 秒)
bash v8_postprocess.sh

# 最終 submission:
#   models/v8_full/submission_final.csv
# 或 (一樣的內容):
#   submissions/sub_v8_FULL_FINAL_LB0.4419.csv
```

### 只跑後處理 (秒級, 推薦)

如果你只是想 **重現後處理結果**, 從 `submissions/v8_raw_submission.csv` 起算:

```bash
# Step 4: server_leak
uv run python -m src.apply_server_leak \
  --input submissions/v8_raw_submission.csv \
  --test data/test.csv \
  --output submissions/v8_after_server_leak.csv

# Step 5: rule_override
uv run python -m src.apply_rule_override \
  --input submissions/v8_after_server_leak.csv \
  --train data/train.csv \
  --test data/test_new.csv \
  --output submissions/sub_v8_FULL_FINAL_LB0.4419.csv
```

---

## 6. 快速 sanity check (不訓練)

確認 final submission 結構正確:

```bash
# 應有 1846 行 (1 header + 1845 預測)
wc -l submissions/sub_v8_FULL_FINAL_LB0.4419.csv

# 欄位應為: rally_uid, actionId, pointId, serverGetPoint
head -3 submissions/sub_v8_FULL_FINAL_LB0.4419.csv

# action 應在 [0, 18], point 在 [0, 9], server 在 (0, 1)
uv run python -c "
import pandas as pd
df = pd.read_csv('submissions/sub_v8_FULL_FINAL_LB0.4419.csv')
print('rows:', len(df))
print('action range:', df.actionId.min(), '-', df.actionId.max())
print('point  range:', df.pointId.min(), '-', df.pointId.max())
print('server range:', df.serverGetPoint.min(), '-', df.serverGetPoint.max())
print('action classes seen:', sorted(df.actionId.unique()))
print('point  classes seen:', sorted(df.pointId.unique()))
"
```

---

## 7. 關鍵概念深入

### 7.1 為什麼 `train_pseudo.csv` 而不是 `train.csv`?

`train_pseudo.csv` (84K row) = `train_augmented.csv` (76K row) + test_new history 真標籤 + 合成 last-shot pseudo

合理性:
- `train.csv` 只有 ~70K row, AutoGluon best_quality 需要更多資料才能發揮
- 舊 `test.csv` 雖然不再計分, 但其 `actionId/pointId/serverGetPoint` 都是 **真值** (規則允許使用)
- test_new 的 rally 中, 除了「最後一球」是要預測的, 其前面的 history 都是 **公開的真值**, 可加入訓練
- 注意: **絕對不要把 test_new 最後一球的真值放進 train** (這是預測目標)

### 7.2 為什麼 `apply_server_leak.py` 是 +0.022 LB 最大單 lever?

- 比賽資料分兩階段發佈: 舊 `test.csv` (含三個 target 真值) → 新 `test_new.csv` (只有 input)
- 兩個 test set 有 **1236 個 rally 重疊** (rally_uid 對得上)
- 舊 test 的 `serverGetPoint` 對重疊 rally 是 100% 真值 → 直接套用 = 免費 +0.022
- 規則允許 (2026/05/13 主辦公告)

```python
# apply_server_leak.py 核心邏輯:
old_test = pd.read_csv('data/test.csv')
sub      = pd.read_csv('models/v8_full/submission.csv')

# 1236 個 rally_uid overlap
mask = sub.rally_uid.isin(old_test.rally_uid)
sub.loc[mask, 'serverGetPoint'] = old_test.set_index('rally_uid').loc[
    sub.loc[mask, 'rally_uid'], 'serverGetPoint'].values
```

### 7.3 `apply_rule_override.py` 在做什麼?

**0%-prob rule**: 對於每個 rally, 看它的 context `(prev_action, last_action, last_point)`. 如果模型預測的 `actionId` 在 train 裡此 context 下 **從未出現過 (0%)**, 就改成 train 此 context 的 mode (最常見類).

```
邏輯: 模型可能 "幻想" 出 train 從未見過的轉移 → 用 train mode 校正
影響: ~10-30 個 row 被改, +0.0014 LB
```

### 7.4 `player_profile` 怎麼算的?

對 train 裡每個 player_id, 計算 16 個聚合特徵:
- avg win-rate
- 各 action class 的偏好 (P(action_k | player))
- 各 point class 的偏好
- avg rally length
- ...

Test 時用 `merge_player_profiles()` 加進 feature matrix. 對於 **unseen player** (~36%), 用 global mean fill.

### 7.5 5-fold × 5-seed × 600s 是什麼意思?

- **5-fold**: GroupKFold by rally_uid (rally 不跨 fold) → 每 fold ~3000 rally val
- **5-seed**: 每 fold 用 5 個 random seed 重訓 → 5 個模型平均
- **600s**: 每個 (fold, seed) 訓練上限 600 秒
- **總時間**: 5 × 5 × 600s = 15000s ≈ 4.2h 訓練 + 8h ensemble overhead = ~12.5h
- **總模型數**: 5 fold × 5 seed × 3 target (action/point/server) = **75 個 AutoGluon predictor**

最終 test 預測 = 25 個 model (5 fold × 5 seed) 對每個 target 的 softmax/proba 平均.

### 7.6 GroupKFold by rally_uid 為什麼重要?

**反例**: 如果隨機切 KFold, 同 rally 的不同球可能一個訓練一個驗證 → 嚴重 leak → OOF F1 灌水 +0.10~0.17.

**正解**: 用 rally_uid 當 group, 確保整個 rally 要嘛全在 train 要嘛全在 val.

**進階**: 真正的 honest CV 是 **GroupKFold by match_id** (整個 match 不跨 fold), 但 v8 用的是 rally-CV. 我們已在後續版本 (Clean v8) 升級為 match-CV.

---

## 8. 已驗證 / 已死路 levers 對照

### ✅ 有效 (已 baked into v8)

| Lever | 真實 lift | 安全度 |
|---|---|---|
| Server LEAK truth (1236 overlap) | +0.022 LB | 100% (math identity) |
| Augmented train (data_old) | +0.005 LB | 高 |
| Player profile | +0.04 OOF F1 | 高 |
| Class weight | +0.005 LB | 高 |
| Best_quality preset | +0.01 LB | 高 |
| 5-seed × 5-fold ensemble | +0.01 LB (variance ↓) | 高 |
| rule_override (0%-prob) | +0.0014 LB | 高 |

### ❌ 死路 (試過全失敗, 不要再試)

| 技術 | 結果 | 原因 |
|---|---|---|
| 修改 cluster 0 overlap action | LB -0.012 (STACK_SAFE) | 破壞 LEAK alignment |
| 修改 non-overlap action retrain | LB -0.071 (PDROP) | train/test player shift |
| HGAT player-action graph | F1 0.087 | adv-AUC fail |
| T-JEPA self-supervised | F1 0.215 | downstream extrapolation 不會 |
| Bottleneck DL (Wu-style) | F1 0.22 | categorical-only input 太弱 |
| MuLMINet | F1 0.116 | mode collapse |
| NODE/TabM/SAINT/FT-Transformer | F1 0.14-0.15 | multitask server trivializes loss |
| GRU prefix | F1 0.13 | sequence input 信號不足 |
| TabPFN v2.5 | abandoned | server AUC 0.70 < LGB 0.81 |
| CatBoost (alone) | F1 0.285 | GPU contention + 弱於 LGB |
| Class 15-18 specialist | infeasible | next-stroke target 結構上 ≠ serve |
| iso 全 1845 server | 毀 LEAK truth | overlap 1236 應保留 0/1 真值 |
| Per-class Saerens | 崩潰 (L1=2.0 max) | source ≈ test prior |

**結論**: 此 dataset (15k rally) **AutoGluon GBDT ensemble + 後處理 LEAK** 是唯一可行路線. 所有 sequence/DL/transformer 全 collapse.

---

## 9. 常見問題 (FAQ)

### Q1: 為什麼不用 GPU?
A: AutoGluon best_quality 的 GBDT 主力 (LightGBM/XGBoost/CatBoost) 在 CPU 上跑反而比 GPU 快 (小資料 ~15k row). GPU 只在 NN component 有用, 但 NN 在 best_quality ensemble 裡權重很低.

### Q2: 為什麼 OOF F1 看起來很高 (0.54+) 但 LB 才 0.44?
A: **OOF inflated 大約 +0.09~0.17**, 主要原因:
1. Rally-CV (應該用 match-CV) → +0.05~0.10 inflation
2. Pseudo-label leak (test_new history 真標籤 → OOF 看得到) → +0.04~0.07 inflation
3. server_leak overlap 在 OOF 也算分 → +0.02 inflation

OOF → LB 真實 transfer ratio 約 **0.86**. 要 LB 0.5 需 OOF 0.58+, 物理上幾乎不可能.

### Q3: Public LB vs Private LB?
A:
- **Public LB**: 1845 rally 中的一部分 (我們推測 ~1236 overlap), 即時顯示
- **Private LB**: 剩餘部分 + 全部 rally re-score, 比賽結束才公佈, 決定最終排名
- **我們的假設**: Public LB 只包含 cluster 0+1 (overlap), cluster 2 (609 non-overlap) 不算
- **理由**: Day 2 改 290 個 cluster 2 row, LB 只動 +0.00003

### Q4: 怎麼避免 overfitting Public LB?
A:
1. **GroupKFold** (rally / match level) 保證 honest OOF
2. **Single lever lock**: 確認每個 lever 在 OOF 上 +0.005 以上才上線
3. **5-fold × 5-seed average**: 降低 variance, 抗 LB random fluctuation
4. **不過度迭代**: 一旦找到 plateau, 接受 Public LB 不再追

### Q5: 怎麼跑 prediction-only (不訓練)?
A:
```bash
# 載入 frozen v8 model 對新 test 預測
uv run python -m src.predict \
    --model-path models/v8_full \
    --test-path <your_test.csv> \
    --run-name v8_predict_new
```

但 **這個 package 沒有附 trained model artifact** (太大, ~5GB). 隊友需自己 retrain Step 3, 或我可以另開 share link 傳 frozen model.

### Q6: 5/27 競賽截止會送什麼?

- **Pick 1 (defense)**: `sub_v8_FULL_FINAL_LB0.4419.csv` (已驗證, 鎖底)
- **Pick 2 (upside)**: NOLEAK 版本當 Private LB hedge (還在試)

---

## 📊 LB 歷史

| 日期 | 上傳 | LB | 排名 |
|---|---|---|---|
| 5/19 | sub_v6_TRUE_LB0.4597.csv (v6 baseline) | 0.4329 | TBD |
| 5/21 00:38 | **sub_v8_FULL_FINAL.csv** (v8, **本 package**) | **0.4419** | **11/284** ⭐ |
| 5/21 00:47 | sub_v8_STACK_SAFE.csv (試動 cluster 0 action) | 0.4299 ❌ | 17/284 (退步) |

---

## 🤝 給隊友

如果你看完 README 後想 hack 看看, 推薦:
1. **先不要動 v8 主幹** — 9 個 lever 已調好, 動哪個都可能 -LB
2. **可以嘗試**: 純 point lever (不動 action), 純 server lever, NOLEAK retrain (Pick 2 用)
3. **絕對不要嘗試**:
   - 修改 cluster 0 overlap 的 action 預測 (兩次驗證都掛)
   - DL sequence model (HGAT/T-JEPA/MuLMINet 全 collapse)
   - Dirichlet 重整 server (LB 0.2832 災難)
4. **看不懂的話**: 先看 `src/cv.py` 和 `src/apply_server_leak.py`, 這兩個是核心.

---

**Maintainer**: KAITING ([mdm07@mdm.ntue.edu.tw](mailto:mdm07@mdm.ntue.edu.tw))
**Last update**: 2026-05-21
**LB verified**: 0.4419 (Public, rank 11/284)
