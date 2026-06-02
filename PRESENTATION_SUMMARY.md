# AI CUP 2026 桌球戰術預測 — 專案總結(簡報用)

> 給組員做簡報的一頁式重點。日期 2026-06-03。

## 1. 題目
用一個 rally 前 n-1 拍,預測第 n 拍的:
- **actionId**(球種,15 類有效)— Macro-F1,權重 0.4
- **pointId**(落點,10 類)— Macro-F1,權重 0.4
- **serverGetPoint**(發球者得分,二元)— AUC,權重 0.2

**總分 OV = 0.4·F1_action + 0.4·F1_point + 0.2·AUC_server**;Baseline 0.28。

## 2. 最終提交(乾淨、LB 實證)
**R-067cr = LB 0.3870095** ← 最終上傳這個

組成:
- **R-034 PAIR**:5 個模型的 Dirichlet 加權混合
  - `v11_aug_oldtest`、`v11plus`、`v13_oldtest`(雙向 Transformer 系列)
  - `v14_seed2_v15feat_a`(GBM 特徵堆疊)
  - `v16_avg3`(test-history 增強 Transformer 多 seed 平均)
- **+ rule_override**:高信心規則修正 action/point
- **+ v22 causal-LM server head @0.30**:因果語言模型的發球得分機率混合

模型核心:雙向 Transformer encoder(d=192),序列 = rally 內每一拍的
類別嵌入(action/point/hand/spin/strength/position/strike)+ 數值特徵,
末位表徵接 action/point head、mean-pool 接 server head。

## 3. 關鍵發現:瓶頸在 pointId(落點)
誤差分析(OOF per-class F1):
- F1_action = **0.413**(接近隨機上限)
- F1_point  = **0.229** ← 主要瓶頸(2× 改進空間)
- AUC       = 0.613

pointId 最難的是 **FH/BH(正手/反手)軸** —— 它是**相對於接球者慣用手**的
(右手/左手選手的 FH/BH 落點是鏡像)。測試集去識別化後無法查 player 慣用手,
模型只能狂猜多數側 → FH-short 過度預測 4.4×、BH-short 幾乎預測不出來。
**這是結構性天花板,不是調參能解的。**

## 4. 試過但失敗的方向(誠實記錄)
| 方法 | 結果 | 原因 |
|---|---|---|
| AutoGluon meta-stack | **LB 0.3152** ❌ | 非 match-grouped 驗證 → 驗證 0.4149 但真實崩 0.10 |
| Prior 校正(point) | NO-GO | 落點是辨識度問題,非校正問題 |
| GBM-on-Transformer-embeddings | NO-GO | 被原模型 head 完全支配 |
| 長 rally 專家(SN≥3) | NO-GO | 資料變少 → 嚴重 overfit |
| SoftF1 additive / 正交集成 | LB-fail | 低權重加性混合在 LB 上有毒 |
| 慣用手 within-rally 特徵 | 訊號中等,未採用 | spread +0.147,覆蓋率僅 42% |

## 5. 重要決策:拒絕 server-leak 0.4419
有一個 0.4419 的高分包,但它靠 `apply_server_leak.py` 用**舊 test.csv 的
serverGetPoint 真值**覆蓋預測(+0.05~0.022)。**我們不用它**,因為:
- **Private LB 會崩**:leak 灌的是 public split 對得上的 rally,private 上退回模型預測 ~0.39
- **DQ 風險**:用測試集標籤真值 = 違規

→ 我們選擇**乾淨、會 generalize 的 R-067cr (0.387)** 作為最終答案。

## 6. 一句話總結
> 在嚴守「不用任何洩漏」的前提下,以雙向 Transformer + GBM 多模型 Dirichlet
> 混合 + 因果 LM 發球頭,達到乾淨 LB 0.387;誤差分析定位落點 FH/BH 為結構性
> 瓶頸(接球者慣用手不可知);AutoGluon 堆疊因驗證未分組而過擬合,證實乾淨
> 混合才是穩健解。
