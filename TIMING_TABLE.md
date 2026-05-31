# TIMING_TABLE.md

Calibrated wall-time estimates for trainer scripts, derived from actual
`Total time:` log lines in `logs/*_full.log`. Updated when new runs land.

**Always update this file when a new trainer variant or hardware setting
changes typical wall time.** Estimates in this file are the ONLY ones
allowed for planning purposes — gut-feel projections are banned.

## GPU jobs (RTX 3060 Ti, 8 GB VRAM)

| Trainer | Tag pattern | Wall (min) | Notes |
|---|---|---:|---|
| `train_v11_transformer.py` | `v11`, `v11plus`, `v11_aug` | **90–110** | 80 epochs × 5 folds |
| `train_v11_transformer.py` | `v11plus_oldtest` (with `--include-old-test`) | **89** | Adds ~3.5% more data |
| `train_v11_transformer.py` | `v11_aug_oldtest` (with aug-parquet + oldtest) | **107** | |
| `train_v11_mulminet.py` | `v11_mulminet*` (no aug) | **103** | MuLMINet aux heads |
| `train_v11_mulminet.py` | `v11_mulminet_aug*` | **90–141** | spread depends on seed |
| `train_v11_mulminet.py` | `v11_mulminet_aug_oldtest*` | **98–118** | |
| `train_v11_mulminet_pretrained.py` | `v11_mulminet_pretrained*` | **150** | longer due to load+init |
| `train_v11_uncertainty.py` | `v11_uncertainty*` | **102** | |
| `train_v11_mulminet_uncertainty.py` | `v11_mulminet_uncertainty*` | **139** | |

**GPU plan baseline**: assume **~110 min per job** unless specifically known otherwise.
In 48h, ~26 GPU job slots.

## CPU jobs

| Trainer | Tag pattern | Wall (min) | Notes |
|---|---|---:|---|
| `train_v13.py` | `v13*` (with `--skip-cb`) | **87** | LightGBM-only |
| `train_v14.py` | `v14_seed*` (with `--skip-cb`) | **130–135** | v9 features + GBM stack |
| `train_v16_testhist_aug.py` | `v16_testhist_aug*` (with `--skip-cb`) | **85–126** | spread due to aug processing |

**CPU plan baseline**: assume **~100 min per job**. In 48h, ~28 CPU job slots.

## Combined budget for 48h plan

- **GPU**: ~26 jobs × 110 min ≈ 47h utilization (leave 1h buffer)
- **CPU**: ~28 jobs × 100 min ≈ 47h utilization
- Total **~54 trainings** possible in 48h with parallel GPU+CPU pipeline

## Historical estimation errors (post-mortem)

| Date | Component | My estimate | Actual | Error | Mode |
|---|---|---:|---:|---:|---|
| 2026-05-15 | v16 oldtest (CPU) | 210 min | 85 min | 2.5× too slow | single-job |
| 2026-05-13 | v14_seed2 oldtest (CPU) | 190 min | 134 min | 1.4× too slow | single-job |
| 2026-05-13 | v11_mulminet_aug oldtest (GPU) | 110 min | 118 min | OK | single-job |
| **2026-05-18** | **v13 (CPU under GPU+CPU parallel load)** | 87 min | 170-182 min | **2.0× too fast** | **parallel** |
| **2026-05-18** | **v11_aug (GPU under parallel load)** | 110 min | 180-206 min | **1.8× too fast** | **parallel** |
| **2026-05-18** | **v16 (CPU under parallel load)** | 85 min | 199-203 min | **2.3× too fast** | **parallel** |
| **2026-05-18** | **v11_mulminet (GPU under parallel load)** | 110 min | 230-238 min | **2.1× too fast** | **parallel** |
| **2026-05-18** | **v14 (CPU under parallel load)** | 134 min | 170-182 min | **1.3× too fast** | **parallel** |

**Lesson 1 (2026-05-15)**: CPU estimates were too pessimistic when based on full
pipeline times — actual `--skip-cb` runs were ~half as long.

**Lesson 2 (2026-05-18, OPPOSITE DIRECTION)**: under GPU+CPU concurrent load,
single-job estimates were **1.7-2.3× too OPTIMISTIC**. CPU contention from
PyTorch dataloader workers + disk I/O contention with LightGBM nearly doubles
per-job time when both lanes run together. Multiply single-job estimates
by ~1.8× when planning a deadline orchestrator with mixed GPU+CPU workloads.

## Calibrated single-job vs parallel-load estimates

| Trainer | Single-job (min) | Parallel-load (min) | Multiplier |
|---|---:|---:|---:|
| `train_v11_transformer.py` v11_aug oldtest | ~110 | ~190 | 1.7× |
| `train_v11_mulminet.py` v11_mulminet_aug oldtest | ~110 | ~235 | 2.1× |
| `train_v11_transformer.py` v11plus oldtest | ~90 | ~180 | 2.0× |
| `train_v13.py` v13 oldtest | ~87 | ~180 | 2.1× |
| `train_v14.py` v14 oldtest | ~134 | ~175 | 1.3× |
| `train_v16_testhist_aug.py` v16 oldtest | ~85 | ~200 | 2.4× |

**Implication for deadline planning**: in a 48h window with full GPU+CPU
parallel use, expect ~30-40h of actual throughput (not the naive ~48h).
Always have a backlog ≥ 2× the expected job count to absorb slowdowns.
