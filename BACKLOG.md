# BACKLOG.md

Prioritized backlog of training experiments. Deadline-driven orchestrator
pulls jobs from here in priority order until budget exhausted.

Format per job:
```
ID  | PRIO  | RES  | EST_MIN | TAG                                       | CMD
```
- PRIO: 1 (highest) → 5 (filler)
- RES: GPU / CPU
- EST_MIN: from TIMING_TABLE.md (NOT gut-feel)
- CMD: full command, args separated by `|`

Done jobs move to `BACKLOG_DONE.md`. Failed jobs move to `BACKLOG_FAILED.md`.

## Active backlog (2026-05-18)

### PRIO 1 (highest-EV — direct path to LB improvement)

```
J001 | 1 | CPU |  87 | v13_oldtest_seed31337                    | -u|src/train_v13.py|--tag|v13_oldtest_seed31337|--seed|31337|--skip-cb|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J002 | 1 | GPU | 110 | v11_aug_oldtest_seed31337                | -u|src/train_v11_transformer.py|--tag|v11_aug_oldtest_seed31337|--aug-parquet|data/test_history_pairs_new.parquet|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J003 | 1 | CPU |  87 | v13_oldtest_seed51966                    | -u|src/train_v13.py|--tag|v13_oldtest_seed51966|--seed|51966|--skip-cb|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J004 | 1 | GPU | 110 | v11_aug_oldtest_seed51966 (for avg3)     | -u|src/train_v11_transformer.py|--tag|v11_aug_oldtest_seed51966|--aug-parquet|data/test_history_pairs_new.parquet|--test-path|data/test_new.csv|--include-old-test|data/test.csv
```

After J001+J003 → build v13_oldtest_avg3. After J002+J004 → build v11_aug_oldtest_avg3. These plug into the proven R-027 CLASS B-pure swap pattern.

### PRIO 2 (medium-EV — additional ensemble depth)

```
J005 | 2 | CPU |  85 | v16_testhist_aug_oldtest_seed4           | -u|src/train_v16_testhist_aug.py|--tag|v16_testhist_aug_oldtest_seed4|--seed|4|--aug|data/test_history_pairs_new.parquet|--skip-cb|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J006 | 2 | CPU |  85 | v16_testhist_aug_oldtest_seed7           | -u|src/train_v16_testhist_aug.py|--tag|v16_testhist_aug_oldtest_seed7|--seed|7|--aug|data/test_history_pairs_new.parquet|--skip-cb|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J007 | 2 | GPU | 110 | v11_mulminet_aug_oldtest_seed51966       | -u|src/train_v11_mulminet.py|--tag|v11_mulminet_aug_oldtest_seed51966|--seed|51966|--aux-lambda|0.2|--epochs|80|--batch|256|--aug-parquet|data/test_history_pairs_new.parquet|--test-path|data/test_new.csv|--include-old-test|data/test.csv
```

After J005+J006 → build v16_testhist_aug_oldtest_avg5 (5 seeds). After J007 → v11_mulminet_aug_oldtest_avg3. After J008+J013 → v11plus_oldtest_avg3 (existing v11plus_oldtest is seed=42 default; +seed31337 +seed51966 gives 3 seeds).

**2026-05-18 priority bump (user directive)**: J008 and J013 lifted from PRIO 3/4 to PRIO 2 per user's "J001-J004, then J005-J008/J013" sequencing. v11plus is empirically irreplaceable in the LB-best subset (R-020b, R-026, R-028 top1 all regressed when v11plus was swapped) — seed averaging is the highest-EV remaining experiment for that slot.

```
J008 | 2 | GPU | 110 | v11plus_oldtest_seed31337                | -u|src/train_v11_transformer.py|--tag|v11plus_oldtest_seed31337|--point-w-scale|2.0|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J013 | 2 | GPU | 110 | v11plus_oldtest_seed51966                | -u|src/train_v11_transformer.py|--tag|v11plus_oldtest_seed51966|--point-w-scale|2.0|--test-path|data/test_new.csv|--include-old-test|data/test.csv
```

### PRIO 3 (architectural diversity — could surface new components)

```
J009 | 3 | GPU | 110 | v11_mulminet_oldtest (no aug)            | -u|src/train_v11_mulminet.py|--tag|v11_mulminet_oldtest|--aux-lambda|0.2|--epochs|80|--batch|256|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J010 | 3 | CPU | 134 | v14_seed0_oldtest                        | -u|src/train_v14.py|--tag|v14_seed0_oldtest|--skip-cb|--seed|42|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J011 | 3 | CPU | 134 | v14_seed1_oldtest                        | -u|src/train_v14.py|--tag|v14_seed1_oldtest|--skip-cb|--seed|31337|--test-path|data/test_new.csv|--include-old-test|data/test.csv
```

### PRIO 4 (long-tail — fill remaining time)

```
J012 | 4 | GPU | 110 | v11_uncertainty_aug_oldtest              | -u|src/train_v11_uncertainty.py|--tag|v11_uncertainty_aug_oldtest|--aug-parquet|data/test_history_pairs_new.parquet|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J014 | 4 | CPU |  85 | v16_testhist_aug_oldtest_seed9           | -u|src/train_v16_testhist_aug.py|--tag|v16_testhist_aug_oldtest_seed9|--seed|9|--aug|data/test_history_pairs_new.parquet|--skip-cb|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J015 | 4 | CPU |  87 | v13_oldtest_seed9                        | -u|src/train_v13.py|--tag|v13_oldtest_seed9|--seed|9|--skip-cb|--test-path|data/test_new.csv|--include-old-test|data/test.csv
```

### PRIO 5 (filler — only if everything else done)

```
J016 | 5 | GPU | 110 | v11_aug_oldtest_seed7                    | -u|src/train_v11_transformer.py|--tag|v11_aug_oldtest_seed7|--aug-parquet|data/test_history_pairs_new.parquet|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J017 | 5 | GPU | 110 | v11_mulminet_aug_oldtest_seed7           | -u|src/train_v11_mulminet.py|--tag|v11_mulminet_aug_oldtest_seed7|--seed|7|--aux-lambda|0.2|--epochs|80|--batch|256|--aug-parquet|data/test_history_pairs_new.parquet|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J018 | 5 | CPU |  87 | v13_oldtest_seed4                        | -u|src/train_v13.py|--tag|v13_oldtest_seed4|--seed|4|--skip-cb|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J019 | 5 | CPU |  85 | v16_testhist_aug_oldtest_seed11          | -u|src/train_v16_testhist_aug.py|--tag|v16_testhist_aug_oldtest_seed11|--seed|11|--aug|data/test_history_pairs_new.parquet|--skip-cb|--test-path|data/test_new.csv|--include-old-test|data/test.csv
J020 | 5 | CPU |  87 | v13_oldtest_seed51966                    | -u|src/train_v13.py|--tag|v13_oldtest_seed51966|--seed|51966|--skip-cb|--test-path|data/test_new.csv|--include-old-test|data/test.csv
```

## 48h budget math

- GPU jobs in backlog: 9 × ~110 min = **16.5h GPU**
- CPU jobs in backlog: 11 × ~95 min = **17.4h CPU**
- Parallel pipeline: max(16.5, 17.4) ≈ **17.5h** (need more if running 48h)

If runtime > backlog, orchestrator stops gracefully with `BACKLOG_EXHAUSTED` log line. Next session can extend backlog with new variants.

## Excluded from backlog (known regressions / dead-ends)

- `v17_momentum*` — confirmed V16-clone, no new info
- `v14_recvprofile*` — failed intake gate
- `v18_hier_point*` — both gates failed
- `v19_rally_srv*` — SGP leak
- Any `LEAK_SGP_*` submission overwriting test_new SGP
