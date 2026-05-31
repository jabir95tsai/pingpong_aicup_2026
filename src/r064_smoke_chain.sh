#!/usr/bin/env bash
# R-064 Fold-1 smoke chain (per Codex fix #1).
#
# 1. Run Fold-1 baseline: v14_seed2_v15feat_a (no oldtest, seed=2, --max-folds 1)
# 2. Run Fold-1 smoke:   v14_seed2_v15feat_d_core (same params, --feature-set v15feat_d)
# 3. Print delta + gate check (dOV_base >= -0.005 vs baseline).
#
# Each fold ~25-30 min standalone; with R-063 competing, ~40-50 min each.
# Total ETA: ~80-100 min from start.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

SUMMARY="logs/r064_smoke_summary.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }

log "=== R-064 Fold-1 smoke chain STARTED ==="
log "Per Codex APPROVE_WITH_FIXES 2026-05-23: gate = dOV_base >= -0.005 vs same-fold baseline"

# Step 1: Fold-1 baseline (v15feat_a, no oldtest)
log "Step 1/2: Fold-1 baseline v14_seed2_v15feat_a (seed=2, --max-folds 1)"
python -u src/train_v14.py \
    --feature-set v15feat \
    --tag v14_seed2_v15feat_a_fold1 \
    --seed 2 \
    --folds 5 \
    --max-folds 1 \
    --n-boost 3000 \
    --es 200 \
    --test-path data/test_new.csv \
    > logs/r064_baseline_v15feat_a_fold1.log 2>&1

BASE_OV=$(grep -E "FOLD OV=" logs/r064_baseline_v15feat_a_fold1.log | head -1 | sed 's/.*FOLD OV=//' | awk '{print $1}')
log "Step 1 done: v15feat_a Fold-1 OV = $BASE_OV"

# Step 2: Fold-1 smoke (v15feat_d)
log "Step 2/2: Fold-1 smoke v14_seed2_v15feat_d_core (seed=2, --max-folds 1)"
python -u src/train_v14.py \
    --feature-set v15feat_d \
    --tag v14_seed2_v15feat_d_fold1_smoke \
    --seed 2 \
    --folds 5 \
    --max-folds 1 \
    --n-boost 3000 \
    --es 200 \
    --test-path data/test_new.csv \
    > logs/r064_smoke_v15feat_d_fold1.log 2>&1

SMOKE_OV=$(grep -E "FOLD OV=" logs/r064_smoke_v15feat_d_fold1.log | head -1 | sed 's/.*FOLD OV=//' | awk '{print $1}')
log "Step 2 done: v15feat_d Fold-1 OV = $SMOKE_OV"

# Delta + gate check
log ""
log "=== R-064 Fold-1 smoke RESULT ==="
log "  baseline (v15feat_a): $BASE_OV"
log "  smoke    (v15feat_d): $SMOKE_OV"
DELTA=$(python -c "print(f'{$SMOKE_OV - $BASE_OV:+.4f}')")
log "  dOV_base:             $DELTA"
PASS=$(python -c "print('PASS' if ($SMOKE_OV - $BASE_OV) >= -0.005 else 'FAIL (Codex gate dOV>=-0.005)')")
log "  Gate (dOV >= -0.005): $PASS"
log ""

# Spin-prior coverage stats from v15feat_d training log (Codex fix #3)
log "=== Spin-prior coverage (Codex fix #3 audit) ==="
grep -E "spin_prior_(min|median|unseen)" logs/r064_smoke_v15feat_d_fold1.log | tee -a "$SUMMARY"

log ""
log "=== R-064 smoke chain DONE ==="
log "Next: stop here. Do NOT launch full 5-fold until Codex reviews this artifact."
log "Report back to Codex with: baseline OV, smoke OV, dOV, per-task F1 split, coverage stats."
