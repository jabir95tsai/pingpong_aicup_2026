# Gemini Prompt: Table-Tennis Video To AI CUP CSV

Use this prompt with a single external/P2A table-tennis video. Do not use it on
AI CUP test videos or to identify real matches behind de-identified test rows.

```text
你是一個桌球影片標註模型。請只根據我上傳的影片畫面、聲音、可見比分板和球路來標註資料；不要上網搜尋、不要辨識選手真名、不要比對真實比賽影片，也不要嘗試反推任何 AI CUP 測試集答案。

任務：
把這支桌球影片轉成 AI CUP 2026 的 shot-level CSV。每一列是一個 rally 中的一次有效擊球，按時間順序排列。

輸出要求：
1. 先輸出一個 fenced CSV block，且 header 必須完全如下：
rally_uid,sex,match,numberGame,rally_id,strikeNumber,scoreSelf,scoreOther,serverGetPoint,gamePlayerId,gamePlayerOtherId,strikeId,handId,strengthId,spinId,pointId,actionId,positionId
2. CSV 只能包含上述 18 欄，不要加入 confidence 或註解欄。
3. CSV 後面再輸出一個簡短 `audit` JSON block，列出不確定的 rally_uid、低信心原因、以及你用了哪些保守預設。
4. 若無法可靠判斷某欄，使用該欄的「無/未知」代碼，不要幻想。

ID 與視角規則：
- `match`: 若我沒有提供數字 match_id，請用 1。
- `rally_uid`: 用 `match * 100000 + rally_id` 的整數格式。
- `numberGame`: 若比分板可判斷局數就填真實局數；否則填 1。
- `rally_id`: 從 1 開始，影片內每個 rally 遞增。
- `gamePlayerId`: 本列擊球者 ID。影片一開始的發球者若在畫面近端/下方，設為 1，對手設為 2；之後每拍依實際擊球者填 1 或 2。
- `gamePlayerOtherId`: 本列擊球者的對手 ID。
- `scoreSelf`: 本列 `gamePlayerId` 在該 rally 開始前的分數。
- `scoreOther`: 本列 `gamePlayerOtherId` 在該 rally 開始前的分數。
- 若比分板不可讀，`scoreSelf=0`, `scoreOther=0`，並在 audit 說明。

Rally 與勝負規則：
- `strikeNumber`: rally 內擊球序號，發球是 1，接發球是 2，依序遞增。
- `serverGetPoint`: 該 rally 發球者是否得分；發球者得分填 1，否則填 0。同一個 rally 內所有列必須相同。
- 若 rally 結束原因看不清楚，用球最後落點、掛網、出界、未接到等畫面線索判斷；仍不確定時保守填最可能值，並在 audit 標記低信心。

欄位代碼：

`sex`
- 1=男
- 2=女
- 看不出來時填 1，並在 audit 標記。

`strikeId`
- 1=發球
- 2=接發球
- 4=第三板之後
- 8=無/未錄影
- 16=暫停

`handId`
- 0=無/未知
- 1=正手
- 2=反手

`strengthId`
- 0=無/未知
- 1=強
- 2=中
- 3=弱

`spinId`
- 0=無/未知
- 1=上旋
- 2=下旋
- 3=不旋
- 4=側上旋
- 5=側下旋

`pointId`：球在對方球台的落點九宮格；掛網、出界、看不清楚或未落在九宮格填 0。
- 0=無/未知/掛網/出界
- 1=正手短
- 2=中間短
- 3=反手短
- 4=正手半出台
- 5=中路半出台
- 6=反手半出台
- 7=正手長
- 8=中間長
- 9=反手長

`actionId`
- 0=無/其他/未知
- 1=拉球
- 2=反拉
- 3=殺球
- 4=擰球
- 5=快帶
- 6=推擠
- 7=挑撥
- 8=拱球
- 9=磕球
- 10=搓球
- 11=擺短
- 12=削球
- 13=擋球
- 14=放高球
- 15=傳統發球
- 16=勾手發球
- 17=逆旋轉發球
- 18=下蹲式發球

`positionId`：擊球者擊球瞬間的站位區域，以擊球者自己半台左右為準。
- 0=無/未知
- 1=左
- 2=中
- 3=右

標註原則：
- 每個 rally 必須從 `strikeNumber=1` 的發球開始；如果影片片段從 rally 中間開始，跳過該不完整 rally，不要硬補。
- `strikeNumber=1` 時，`strikeId=1`，`actionId` 應為 15/16/17/18；若真的無法辨識發球型，填 15。
- `strikeNumber=2` 時，`strikeId=2`，不可填發球 actionId。
- `strikeNumber>=3` 時，`strikeId=4`。
- 看不清楚球種時優先用 0，不要過度推測。
- 看不清楚落點時優先用 0，不要硬猜九宮格。
- 一個 rally 中 `serverGetPoint` 必須一致。
- 不要輸出非 CSV 的說明文字在 CSV block 內。

請先完整觀看影片，再進行分段與標註。最後輸出：

```csv
rally_uid,sex,match,numberGame,rally_id,strikeNumber,scoreSelf,scoreOther,serverGetPoint,gamePlayerId,gamePlayerOtherId,strikeId,handId,strengthId,spinId,pointId,actionId,positionId
...
```

```json
{
  "audit": [
    {
      "rally_uid": 100001,
      "confidence": "medium",
      "notes": "scoreboard unreadable; scoreSelf/scoreOther set to 0"
    }
  ]
}
```
```

