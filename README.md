# 🏓 AI CUP 2026 — 桌球戰術與結果序列預測

> 用一場 rally 的前 n−1 拍，預測下一拍的**球種**、**落點**與**該球勝負**。在「零資料洩漏」的紀律下，以多模型集成達成 **私人榜 22 / 423（前 5.2%）**，且在 public→private 洗牌中**逆勢上升 57 名**。

> **EN (TL;DR):** A multi-task time-series prediction project for AI CUP 2026 table-tennis tactics: given the first n−1 strokes of a rally, predict the next stroke's shot type, landing zone, and rally outcome. Built a clean, leak-free ensemble (bidirectional Transformers + gradient-boosted stacking + Dirichlet blending) that ranked **22 / 423 (top 5.2%)** on the private leaderboard — climbing 57 places from public to private as leakage-reliant teams collapsed.

![Python](https://img.shields.io/badge/Python-3.13-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Transformer-ee4c2c) ![Competition](https://img.shields.io/badge/AI%20CUP%202026-桌球戰術預測-green) ![Rank](https://img.shields.io/badge/Private%20LB-22%2F423%20(top%205.2%25)-orange)

---

## 專案簡介

AI CUP 2026 春季賽「基於時序資料之桌球戰術與結果預測」——這是一個**多任務時序預測**問題：根據桌球比賽中某一回合（rally）的前 n−1 拍擊球資訊，預測第 n 拍的：

- **`actionId`** 下一拍球種（多類別，Macro-F1）
- **`pointId`** 下一拍落點（10 類九宮格，Macro-F1）
- **`serverGetPoint`** 發球者是否得分（二元，AUC-ROC）

綜合分數 `OV = 0.4 × F1_action + 0.4 × F1_point + 0.2 × AUC_server`，參賽門檻 baseline = 0.28。

**挑戰性：**
- **類別嚴重不平衡**：球種最多／最少差 **41.5×**、落點差 **23.4×**，且評估採 Macro-F1，逼模型照顧稀有類。
- **時序依賴**：須從可變長度的擊球序列建模戰術轉移。
- **訓練／測試分布偏移 + 去識別化**：測試集球員 ID 與訓練集不同，使「相對於接球者慣用手」的落點正反手軸難以還原 → 結構性難點。
- **賽中存在資料洩漏陷阱**：能否抵抗洩漏誘惑、選擇會 generalize 的方法，是這場比賽真正的試煉。

---

## 我的角色與貢獻

> 學生組團隊專案（隊伍 TEAM_10220，共 6 人）。

**我主導並完成的部分（最終提交的模型主線）：**

- 🧠 **設計並訓練多模型集成（最終提交 R-067cr，乾淨 public LB 0.387）**：5 個模型的 Dirichlet 加權混合 —— 雙向 Transformer 系列 + 梯度提升（GBDT）特徵堆疊 + 測試史增強 Transformer。
- 🔬 **誤差分析定位瓶頸**：以 per-class F1 拆解，確認落點（`pointId`）為主瓶頸（F1≈0.23 vs 球種 F1≈0.41），並歸因到「正/反手軸 = 接球者慣用手相對、去識別化後不可知」的結構性天花板。
- 🧩 **後處理與資料增強**：規則修正（封死不合法預測）、左右鏡像增強、測試史增強對齊。
- 🛡️ **泛化優先的驗證紀律**：以 match 分組的 GroupKFold 避免 rally/match 跨折洩漏；以「OOF→LB 轉移率」而非單純最高分挑選元件。
- ⚖️ **關鍵工程判斷（拒絕資料洩漏）**：評估後**否決**一個靠覆蓋 `serverGetPoint` 真值（+0.05）的高分洩漏提交，理由為 private LB 崩盤風險與違規風險；改採乾淨、會轉移的模型 → 最終在洗牌中上升 57 名得到驗證。
- 🧪 **負面結果驗證**：實測 AutoGluon meta-stacking，因驗證未按 match 分組而過擬合（驗證 0.41 → 實際 LB 0.32），據此排除。

---

## 技術方法

```
原始擊球序列 → 清理/特徵 → 多模型 OOF 預測 → Dirichlet 加權混合 → 規則修正 → 最終提交
```

**1. 資料處理 → 監督樣本**
- 將「rally 序列」轉成監督樣本：前 n−1 拍為 context，第 n 拍的三個目標為標籤。
- 約 **84,700 筆擊球紀錄 → 69,712 個（rally, 目標拍）訓練樣本**；測試集 1,845 個 rally。

**2. 特徵工程與增強**
- 每拍類別嵌入（球種 / 落點 / 正反手 / 旋轉 / 力道 / 站位 / 揮拍狀態）＋ 數值特徵（拍序、比分差等）。
- **左右鏡像增強**（翻轉 handId / positionId / pointId）使資料翻倍。
- **測試史增強**（合法利用可見前幾拍對齊分布）。

**3. 模型**
- **雙向 Transformer encoder**（d_model = 192）：末位表徵接球種／落點頭，mean-pool 接發球頭。
- **兩階段 GBDT 堆疊**：先預測球種，再以其機率輔助落點與勝負。
- **因果語言模型發球頭**：專門強化 `serverGetPoint` 的 AUC，於集成中以 0.30 權重混合。

**4. 集成與驗證**
- **Dirichlet 隨機搜尋**逐任務找最佳混合權重。
- **match 分組 GroupKFold**：防止同一場比賽跨折洩漏，確保 OOF 反映真實泛化。
- 元件以**轉移性（OOF→LB 落差小）**而非單純最高分入選。

---

## 成果

| 指標 | 數值 |
|---|---|
| 私人榜（Private LB）排名 | **22 / 423（前 5.2%）** |
| 私人榜分數 | 0.3737427 |
| 公開榜（Public LB）排名 | 79 / 423 |
| 公開榜分數 | 0.3870095 |
| Public → Private 名次變化 | **▲ 上升 57 名** |
| 競賽 baseline | 0.28 |

> 評估指標：`OV = 0.4 × F1_action + 0.4 × F1_point + 0.2 × AUC_server`（Macro-F1 / AUC-ROC）。
> 亮點：在洗牌中名次大幅上升——靠資料洩漏衝高 public 分數的隊伍於 private 崩盤，乾淨且會轉移的方法守住名次。

---

## 技術棧

- **語言**：Python 3.13
- **深度學習**：PyTorch（Transformer encoder、因果 LM 發球頭）
- **梯度提升 / 集成**：LightGBM、XGBoost、CatBoost、AutoGluon（meta-stacking 評估）
- **資料 / 科學運算**：pandas、NumPy、scikit-learn（GroupKFold、指標）
- **訓練環境**：Kaggle Notebooks（GPU）、本機 CUDA
- **工具**：Git、Jupyter

---

## 專案結構

```
pingpong_aicup_2026/
├── src/                # 模型訓練、特徵工程、集成、後處理腳本
├── audits/             # 誤差分析、轉移率分析、失敗方向記錄
├── submissions/        # 各候選提交（最終 R-067cr）
├── oof_predictions/    # OOF / test 預測陣列（集成用，未納入版本控制）
├── eda_output/         # EDA 圖表
└── DEV_README.md       # 開發期內部指引
```

---

## 關鍵收穫

- **泛化優先 > 榜單分數**：以 match 分組驗證 + 轉移率挑選元件，避免被「驗證高、實測崩」的過擬合（如 AutoGluon 0.41→0.32）誤導；private 洗牌時逆勢上升 57 名直接印證此紀律的價值。
- **工程倫理與風險判斷**：辨識並**主動放棄**靠覆蓋測試標籤真值的洩漏分數，權衡 private LB 崩盤與違規風險後選擇乾淨解——這是一個可在面試深聊的「為什麼不選看似更高分的方案」的決策案例。
- **誤差驅動的問題定位**：用 per-class F1 把整體分數拆到「哪一類、哪一拍序最弱」，定位落點正反手為結構性瓶頸（接球者慣用手不可知），而非盲目調參。
- **多模型集成實務**：Dirichlet 逐任務權重搜尋、多視角模型（Transformer / GBDT / 因果 LM）的多樣性如何貢獻穩健度。

---

## 連結

- **GitHub Repo**：https://github.com/jabir95tsai/pingpong_aicup_2026
- **聯絡**：jabir95tsai@gmail.com
