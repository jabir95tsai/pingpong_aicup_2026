#!/usr/bin/env bash
# Chain: seed 31337 train -> avg3 combine of (51966, 7, 31337)
# Dispatched after seed 7 completes (already trained).
set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Train seed 31337 ==="
python -u src/train_v14.py \
  --feature-set v15feat_c \
  --tag v14_seed2_v15feat_c_oldtest_seed31337 \
  --seed 31337 \
  --folds 5 \
  --n-boost 3000 \
  --es 200 \
  --include-old-test data/test.csv \
  --test-path data/test_new.csv \
  > logs/r047_v15feat_c_oldtest_seed31337.log 2>&1

echo "=== Step 2: avg3 combine ==="
python -u src/avg_oof.py \
  --tags v14_seed2_v15feat_c_oldtest \
         v14_seed2_v15feat_c_oldtest_seed7 \
         v14_seed2_v15feat_c_oldtest_seed31337 \
  --out-tag v14_seed2_v15feat_c_oldtest_avg3 \
  > logs/v15feat_c_oldtest_avg3_combine.log 2>&1

echo "=== Chain complete ==="
tail -15 logs/v15feat_c_oldtest_avg3_combine.log
