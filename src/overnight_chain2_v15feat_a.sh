#!/usr/bin/env bash
# Overnight chain 2 — fires after chain 1 (v15feat_c avg3) completes.
#
# Trains v15feat_a_oldtest seeds 7 + 31337 to build v14_seed2_v15feat_a_oldtest_avg3.
# v15feat_a was the R-034 LB-WIN feature set; avg3 may lift the SWAP further.
#
# Steps (~5.5 hr):
#   1. Wait for chain 1 to finish (poll overnight_summary.log for "OVERNIGHT CHAIN COMPLETE")
#   2. Train v15feat_a_oldtest seed 7 (~2.5 hr)
#   3. Train v15feat_a_oldtest seed 31337 (~2.5 hr)
#   4. avg3 combine (51966 + 7 + 31337)
#   5. Build R-059r candidate: v15feat_a_oldtest_avg3 SWAP into R-034 + rule_override
#   6. Append to logs/overnight_summary.log
#
# Total ETA: 5.5 hr from chain 1 completion (~03:15 + 5.5 = 08:45 Taiwan).
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

SUMMARY="logs/overnight_summary.log"

log_section() {
    echo "" | tee -a "$SUMMARY"
    echo "========================================================" | tee -a "$SUMMARY"
    echo "  [$(date +%H:%M:%S)] $1" | tee -a "$SUMMARY"
    echo "========================================================" | tee -a "$SUMMARY"
}

log_section "OVERNIGHT CHAIN 2 (v15feat_a_oldtest avg3) — START (chain 1 already done)"

# ---- Step 2: Train v15feat_a_oldtest seed 7 ----
log_section "Chain 2 Step 2/5: Train v15feat_a_oldtest seed 7"
python -u src/train_v14.py \
    --feature-set v15feat \
    --tag v14_seed2_v15feat_a_oldtest_seed7 \
    --seed 7 \
    --folds 5 \
    --n-boost 3000 \
    --es 200 \
    --include-old-test data/test.csv \
    --test-path data/test_new.csv \
    > logs/v15feat_a_oldtest_seed7.log 2>&1
SEED7_OV=$(grep -E "FINAL OV \((base|opt)\):" logs/v15feat_a_oldtest_seed7.log | tail -2 | tr '\n' ' ')
echo "  seed 7 finished: $SEED7_OV" | tee -a "$SUMMARY"

# ---- Step 3: Train v15feat_a_oldtest seed 31337 ----
log_section "Chain 2 Step 3/5: Train v15feat_a_oldtest seed 31337"
python -u src/train_v14.py \
    --feature-set v15feat \
    --tag v14_seed2_v15feat_a_oldtest_seed31337 \
    --seed 31337 \
    --folds 5 \
    --n-boost 3000 \
    --es 200 \
    --include-old-test data/test.csv \
    --test-path data/test_new.csv \
    > logs/v15feat_a_oldtest_seed31337.log 2>&1
SEED31337_OV=$(grep -E "FINAL OV \((base|opt)\):" logs/v15feat_a_oldtest_seed31337.log | tail -2 | tr '\n' ' ')
echo "  seed 31337 finished: $SEED31337_OV" | tee -a "$SUMMARY"

# ---- Step 4: avg3 combine ----
log_section "Chain 2 Step 4/5: avg3 combine v15feat_a_oldtest"
python -u src/avg_oof.py \
    --tags v14_seed2_v15feat_a_oldtest \
           v14_seed2_v15feat_a_oldtest_seed7 \
           v14_seed2_v15feat_a_oldtest_seed31337 \
    --out-tag v14_seed2_v15feat_a_oldtest_avg3 \
    > logs/v15feat_a_oldtest_avg3_combine.log 2>&1
AVG3_LINE=$(grep "Averaged OOF" logs/v15feat_a_oldtest_avg3_combine.log)
echo "  $AVG3_LINE" | tee -a "$SUMMARY"

# ---- Step 5: Build R-059r candidate ----
log_section "Chain 2 Step 5/5: Build R-059r (v15feat_a_oldtest_avg3 SWAP + rule_override)"
python -u src/build_r059_candidate.py 2>&1 | tee -a "$SUMMARY"

log_section "OVERNIGHT CHAIN 2 COMPLETE"
echo "  R-059r candidate added to submissions/" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "  Tomorrow's full LB-ready menu:" | tee -a "$SUMMARY"
ls submissions/submission_R0{58,35,37,59}*PLUS_RULE.csv 2>&1 | tee -a "$SUMMARY"
