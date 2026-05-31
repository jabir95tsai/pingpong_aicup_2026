# Current EDA Report

Generated: 2026-05-08 23:11

Data source: `data/train.csv` and active `data/test_new.csv`.

## Executive Findings

1. The active test set is `test_new.csv` with 5,668 visible rows and 1,845 rallies. All modeling and submission checks must target this file.
2. Train provides 84,707 raw shot rows and 69,712 supervised next-shot target rows (`strikeNumber >= 2`).
3. Test-history augmentation can legally produce 3,823 visible action/point history pairs from `test_new.csv`, but no SGP labels.
4. Old `test.csv` overlaps 1,236 of 1,845 new-test rallies. The visible histories are identical on shared columns: `True`. Because old test contains SGP, it is a leakage audit source only.
5. Full-rally length parity is a strong SGP leak diagnostic on train (`best-direction AUC=0.9990`). Do not use total rally length, terminal parity, or any full-rally aggregate for server prediction.
6. The strongest raw train-vs-test shifts are listed below. Use them as feature/model-risk hints, not as hard rules.

## Dataset Overview

| dataset           |   rows |   rallies |   matches |   columns |
|:------------------|-------:|----------:|----------:|----------:|
| train             |  84707 |     14995 |       216 |        18 |
| test_new          |   5668 |      1845 |        79 |        17 |
| sample_submission |      0 |         0 |           |         4 |
| old_test          |   3589 |      1236 |        55 |        18 |

## Old-Test Overlap / Leakage Audit

|   old_test_rows |   old_test_rallies |   overlap_rallies |   new_only_rallies |   overlap_rows | overlap_histories_equal_on_shared_cols   | old_has_serverGetPoint   |
|----------------:|-------------------:|------------------:|-------------------:|---------------:|:-----------------------------------------|:-------------------------|
|            3589 |               1236 |              1236 |                609 |           3589 | True                                     | True                     |

## Missing Values

| column            |   train_missing |   test_new_missing |
|:------------------|----------------:|-------------------:|
| actionId          |               0 |                  0 |
| gamePlayerId      |               0 |                  0 |
| gamePlayerOtherId |               0 |                  0 |
| handId            |               0 |                  0 |
| match             |               0 |                  0 |
| numberGame        |               0 |                  0 |
| pointId           |               0 |                  0 |
| positionId        |               0 |                  0 |
| rally_id          |               0 |                  0 |
| rally_uid         |               0 |                  0 |
| scoreOther        |               0 |                  0 |
| scoreSelf         |               0 |                  0 |

## Train Target Class Distributions

### actionId

|   class |   count |    pct | label        |
|--------:|--------:|-------:|:-------------|
|       0 |    2050 | 0.0294 | other        |
|       1 |   15435 | 0.2214 | loop         |
|       2 |    6339 | 0.0909 | counter_loop |
|       3 |    2129 | 0.0305 | smash        |
|       4 |    2638 | 0.0378 | banana       |
|       5 |    4192 | 0.0601 | drive        |
|       6 |    6635 | 0.0952 | push_press   |
|       7 |    1413 | 0.0203 | flick        |
|       8 |     372 | 0.0053 | arch         |
|       9 |     794 | 0.0114 | block_chop   |
|      10 |   11208 | 0.1608 | chop         |
|      11 |    3522 | 0.0505 | short_push   |
|      12 |    4522 | 0.0649 | def_chop     |
|      13 |    7848 | 0.1126 | block        |
|      14 |     613 | 0.0088 | lob          |
|      15 |       1 | 0      | serve_trad   |
|      16 |       1 | 0      | serve_hook   |

### pointId

|   class |   count |    pct | label     |
|--------:|--------:|-------:|:----------|
|       0 |   15263 | 0.2189 | off_grid  |
|       1 |     582 | 0.0083 | FH_short  |
|       2 |    1920 | 0.0275 | mid_short |
|       3 |     203 | 0.0029 | BH_short  |
|       4 |    2995 | 0.043  | FH_half   |
|       5 |    6585 | 0.0945 | mid_half  |
|       6 |    4583 | 0.0657 | BH_half   |
|       7 |    9122 | 0.1309 | FH_long   |
|       8 |   12386 | 0.1777 | mid_long  |
|       9 |   16073 | 0.2306 | BH_long   |

### serverGetPoint by rally

|   class |   count |    pct |
|--------:|--------:|-------:|
|       0 |    6749 | 0.4501 |
|       1 |    8246 | 0.5499 |

## Test Visible History Class Distributions

These are observed history rows only, not target labels for the hidden next shot.

### visible actionId

|   class |   count |    pct | label         |
|--------:|--------:|-------:|:--------------|
|       0 |     229 | 0.0404 | other         |
|       1 |     860 | 0.1517 | loop          |
|       2 |     223 | 0.0393 | counter_loop  |
|       3 |      72 | 0.0127 | smash         |
|       4 |     236 | 0.0416 | banana        |
|       5 |     289 | 0.051  | drive         |
|       6 |     356 | 0.0628 | push_press    |
|       7 |     104 | 0.0183 | flick         |
|       8 |      19 | 0.0034 | arch          |
|       9 |      89 | 0.0157 | block_chop    |
|      10 |     644 | 0.1136 | chop          |
|      11 |     380 | 0.067  | short_push    |
|      12 |     200 | 0.0353 | def_chop      |
|      13 |     333 | 0.0588 | block         |
|      14 |      17 | 0.003  | lob           |
|      15 |    1244 | 0.2195 | serve_trad    |
|      16 |     263 | 0.0464 | serve_hook    |
|      17 |      73 | 0.0129 | serve_reverse |
|      18 |      37 | 0.0065 | serve_squat   |

### visible pointId

|   class |   count |    pct | label     |
|--------:|--------:|-------:|:----------|
|       0 |      23 | 0.0041 | off_grid  |
|       1 |     265 | 0.0468 | FH_short  |
|       2 |     513 | 0.0905 | mid_short |
|       3 |      91 | 0.0161 | BH_short  |
|       4 |     457 | 0.0806 | FH_half   |
|       5 |     830 | 0.1464 | mid_half  |
|       6 |     369 | 0.0651 | BH_half   |
|       7 |     595 | 0.105  | FH_long   |
|       8 |    1048 | 0.1849 | mid_long  |
|       9 |    1477 | 0.2606 | BH_long   |

## Target next-strikeNumber Distribution

Train target rows use each observed train shot with `strikeNumber >= 2`.
Test target rows are one per rally, with target position `max(strikeNumber) + 1`.

| sn_bucket   |   train_target_pct |   test_next_target_pct |   delta_test_minus_train |
|:------------|-------------------:|-----------------------:|-------------------------:|
| 2           |             0.2151 |                 0.2753 |                   0.0602 |
| 3-4         |             0.3395 |                 0.4244 |                   0.0849 |
| 5-8         |             0.288  |                 0.2509 |                  -0.037  |
| 9-12        |             0.0896 |                 0.0352 |                  -0.0544 |
| 13+         |             0.0678 |                 0.0141 |                  -0.0537 |

## Rally Length Summary

| dataset              |   count |   mean |    std |   min |   25% |   50% |   75% |   90% |   95% |   max |
|:---------------------|--------:|-------:|-------:|------:|------:|------:|------:|------:|------:|------:|
| train                |   14995 | 5.649  | 3.9805 |     2 |     3 |     5 |     7 |    10 |    13 |    52 |
| test_new_history     |    1845 | 3.0721 | 2.4238 |     1 |     1 |     2 |     4 |     6 |     7 |    24 |
| test_new_next_target |    1845 | 4.0721 | 2.4238 |     2 |     2 |     3 |     5 |     7 |     8 |    25 |

## Train vs Test Raw Distribution Shift

Total variation is easier to read: 0 means identical marginal distribution, 1 means no overlap.

| column         |   total_variation |   js_divergence |
|:---------------|------------------:|----------------:|
| positionId     |            0.2261 |          0.0399 |
| strikeId       |            0.2074 |          0.0335 |
| sn_bucket      |            0.2074 |          0.0443 |
| actionId       |            0.1956 |          0.0383 |
| pointId        |            0.188  |          0.0883 |
| spinId         |            0.1058 |          0.0264 |
| score_diff_bin |            0.0817 |          0.0074 |
| strengthId     |            0.0644 |          0.0193 |
| handId         |            0.0623 |          0.0089 |
| numberGame     |            0.0468 |          0.0029 |
| sex            |            0.033  |          0.0008 |

## Player-ID Overlap

Player IDs are de-identified and should not be treated as stable identity priors. This table is diagnostic only.

| metric                   |    value |
|:-------------------------|---------:|
| train_unique_players     | 166      |
| test_new_unique_players  |  71      |
| overlap_unique_players   |  40      |
| test_player_overlap_rate |   0.5634 |

## Rare Target Classes

| task     |   class_id | label      |   count |    pct |
|:---------|-----------:|:-----------|--------:|-------:|
| actionId |         15 | serve_trad |       1 | 0      |
| actionId |         16 | serve_hook |       1 | 0      |
| actionId |          8 | arch       |     372 | 0.0053 |
| actionId |         14 | lob        |     613 | 0.0088 |
| actionId |          9 | block_chop |     794 | 0.0114 |
| actionId |          7 | flick      |    1413 | 0.0203 |
| actionId |          0 | other      |    2050 | 0.0294 |
| actionId |          3 | smash      |    2129 | 0.0305 |
| actionId |          4 | banana     |    2638 | 0.0378 |
| actionId |         11 | short_push |    3522 | 0.0505 |
| pointId  |          3 | BH_short   |     203 | 0.0029 |
| pointId  |          1 | FH_short   |     582 | 0.0083 |
| pointId  |          2 | mid_short  |    1920 | 0.0275 |
| pointId  |          4 | FH_half    |    2995 | 0.043  |
| pointId  |          6 | BH_half    |    4583 | 0.0657 |
| pointId  |          5 | mid_half   |    6585 | 0.0945 |
| pointId  |          7 | FH_long    |    9122 | 0.1309 |
| pointId  |          8 | mid_long   |   12386 | 0.1777 |
| pointId  |          0 | off_grid   |   15263 | 0.2189 |
| pointId  |          9 | BH_long    |   16073 | 0.2306 |

## Top action/point co-occurrences

|   actionId |   pointId |   count |    pct |
|-----------:|----------:|--------:|-------:|
|          1 |         9 |    4780 | 0.0686 |
|          1 |         8 |    3279 | 0.047  |
|         13 |         0 |    3244 | 0.0465 |
|          1 |         7 |    2935 | 0.0421 |
|          1 |         0 |    2776 | 0.0398 |
|         10 |         6 |    2384 | 0.0342 |
|         10 |         9 |    2112 | 0.0303 |
|          0 |         0 |    2046 | 0.0293 |
|          6 |         9 |    1921 | 0.0276 |
|         10 |         5 |    1895 | 0.0272 |
|          2 |         0 |    1745 | 0.025  |
|         10 |         8 |    1604 | 0.023  |
|          2 |         9 |    1556 | 0.0223 |
|          6 |         8 |    1552 | 0.0223 |
|         13 |         8 |    1519 | 0.0218 |
|          6 |         0 |    1389 | 0.0199 |
|         13 |         9 |    1385 | 0.0199 |
|          5 |         9 |    1328 | 0.019  |
|          2 |         8 |    1238 | 0.0178 |
|         11 |         2 |    1203 | 0.0173 |

## Top visible transition pairs

### train action transitions

|   prev_actionId |   actionId |   count |    pct |
|----------------:|-----------:|--------:|-------:|
|              10 |          1 |    4469 | 0.0641 |
|               1 |         13 |    4007 | 0.0575 |
|              10 |         10 |    3912 | 0.0561 |
|              15 |         10 |    2938 | 0.0421 |
|              13 |          1 |    2506 | 0.0359 |
|               1 |         12 |    2182 | 0.0313 |
|               1 |          2 |    2086 | 0.0299 |
|               2 |          2 |    2064 | 0.0296 |
|              12 |          1 |    2028 | 0.0291 |
|               1 |          6 |    1602 | 0.023  |
|               6 |          6 |    1579 | 0.0227 |
|              15 |          1 |    1549 | 0.0222 |
|               2 |         13 |    1377 | 0.0198 |
|              15 |         11 |    1336 | 0.0192 |
|              15 |          4 |    1252 | 0.018  |

### test_new visible action transitions

|   prev_actionId |   actionId |   count |    pct |
|----------------:|-----------:|--------:|-------:|
|              15 |         10 |     277 | 0.0725 |
|              10 |          1 |     243 | 0.0636 |
|              15 |         11 |     206 | 0.0539 |
|               1 |         13 |     198 | 0.0518 |
|              15 |          4 |     147 | 0.0385 |
|              15 |          1 |     146 | 0.0382 |
|              13 |          1 |     113 | 0.0296 |
|              10 |         10 |     102 | 0.0267 |
|               1 |          2 |      87 | 0.0228 |
|              11 |         10 |      85 | 0.0222 |
|               1 |          6 |      79 | 0.0207 |
|               6 |          6 |      79 | 0.0207 |
|              11 |         11 |      74 | 0.0194 |
|               5 |          5 |      71 | 0.0186 |
|               1 |         12 |      65 | 0.017  |

### train point transitions

|   prev_pointId |   pointId |   count |    pct |
|---------------:|----------:|--------:|-------:|
|              9 |         0 |    4950 | 0.071  |
|              9 |         9 |    4661 | 0.0669 |
|              9 |         8 |    3518 | 0.0505 |
|              8 |         9 |    3426 | 0.0491 |
|              7 |         0 |    3381 | 0.0485 |
|              8 |         8 |    3086 | 0.0443 |
|              8 |         0 |    2995 | 0.043  |
|              9 |         7 |    2354 | 0.0338 |
|              5 |         9 |    2118 | 0.0304 |
|              8 |         7 |    2083 | 0.0299 |
|              7 |         9 |    1878 | 0.0269 |
|              7 |         7 |    1605 | 0.023  |
|              7 |         8 |    1584 | 0.0227 |
|              5 |         8 |    1555 | 0.0223 |
|              5 |         5 |    1406 | 0.0202 |

### test_new visible point transitions

|   prev_pointId |   pointId |   count |    pct |
|---------------:|----------:|--------:|-------:|
|              9 |         9 |     377 | 0.0986 |
|              9 |         8 |     273 | 0.0714 |
|              8 |         9 |     252 | 0.0659 |
|              8 |         8 |     224 | 0.0586 |
|              5 |         9 |     173 | 0.0453 |
|              9 |         7 |     135 | 0.0353 |
|              5 |         8 |     118 | 0.0309 |
|              5 |         5 |     118 | 0.0309 |
|              8 |         7 |     118 | 0.0309 |
|              7 |         9 |     112 | 0.0293 |
|              7 |         8 |      98 | 0.0256 |
|              7 |         7 |      91 | 0.0238 |
|              4 |         9 |      89 | 0.0233 |
|              6 |         9 |      87 | 0.0228 |
|              2 |         2 |      84 | 0.022  |

## SGP Leakage Diagnostic

This is not a feature recommendation. It documents why terminal/full-rally aggregates are forbidden for SGP.

|   length_parity |   count |   serverGetPoint_rate |   auc_best_direction |
|----------------:|--------:|----------------------:|---------------------:|
|               0 |    8248 |                0.999  |                0.999 |
|               1 |    6747 |                0.0009 |                0.999 |

## Generated Tables

- `tables/dataset_overview.csv`
- `tables/missing_values.csv`
- `tables/train_target_action_distribution.csv`
- `tables/train_target_point_distribution.csv`
- `tables/train_rally_server_distribution.csv`
- `tables/test_visible_action_distribution.csv`
- `tables/test_visible_point_distribution.csv`
- `tables/target_next_sn_distribution.csv`
- `tables/rally_length_summary.csv`
- `tables/train_test_raw_shift.csv`
- `tables/player_overlap.csv`
- `tables/train_action_by_target_sn_bucket.csv`
- `tables/train_point_by_target_sn_bucket.csv`
- `tables/train_target_action_point_cooccurrence.csv`
- `tables/rare_target_classes.csv`
- `tables/train_action_transition.csv`
- `tables/test_new_visible_action_transition.csv`
- `tables/train_point_transition.csv`
- `tables/test_new_visible_point_transition.csv`
- `tables/leakage_rally_length_parity_vs_sgp.csv`

## Generated Figures

- `figures/train_target_action_distribution.png`
- `figures/train_target_point_distribution.png`
- `figures/test_visible_action_distribution.png`
- `figures/test_visible_point_distribution.png`
- `figures/target_next_sn_distribution.png`
- `figures/rally_length_distribution.png`
- `figures/train_test_shift_top.png`
- `figures/train_action_point_cooccurrence_heatmap.png`
- `figures/leakage_rally_length_parity_vs_sgp.png`

## Modeling Implications

- Treat `test_new.csv` as the only active submission target.
- Keep legal test-history augmentation, but verify `aug_rows_in_server_loss == 0`.
- Focus pointId improvements on rare/low-support classes, especially short/half zones.
- Avoid same-family seed averaging as a standalone thesis unless it adds transfer evidence.
- Never import old-test-SGP-trained external caches or submissions into the legal zoo.
- Any new SGP head must be audited for terminal-length and parity leakage before training.
