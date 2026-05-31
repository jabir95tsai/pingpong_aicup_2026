#!/usr/bin/env bash
# Overnight chain — runs unattended after seed-7 v15feat_c training completes.
#
# Steps (serial, each ~1-2 hr):
#   1. Wait for seed 7 training to finish (poll log for "FINAL OV" line)
#   2. Train seed 31337
#   3. Build avg3 (51966 + 7 + 31337)
#   4. Build R-058r submission: v14_seed2_v15feat_c_oldtest_avg3 swap into R-034 + rule_override
#   5. Build R-035r (v14_recvhand_oldtest swap + rule)
#   6. Build R-037r (v14_recvprofile_oldtest swap + rule)
#   7. Print OOF + predicted-LB summary
#
# Logs each step to logs/overnight_*.log. Final summary at logs/overnight_summary.log.
#
# Total ETA: ~3-3.5 hr from seed-7 completion.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

SEED7_LOG="logs/r047_v15feat_c_oldtest_seed7.log"
SUMMARY="logs/overnight_summary.log"

log_section() {
    echo "" | tee -a "$SUMMARY"
    echo "========================================================" | tee -a "$SUMMARY"
    echo "  [$(date +%H:%M:%S)] $1" | tee -a "$SUMMARY"
    echo "========================================================" | tee -a "$SUMMARY"
}

log_section "OVERNIGHT CHAIN STARTED"

# ---- Step 1: Wait for seed 7 ----
log_section "Step 1/7: Wait for seed 7 (poll $SEED7_LOG for FINAL OV)"
while ! grep -q "FINAL OV" "$SEED7_LOG" 2>/dev/null; do
    sleep 60
done
SEED7_OV=$(grep -E "FINAL OV \((base|opt)\):" "$SEED7_LOG" | tail -2 | tr '\n' ' ')
echo "  seed 7 finished: $SEED7_OV" | tee -a "$SUMMARY"

# ---- Step 2: Train seed 31337 ----
log_section "Step 2/7: Train seed 31337"
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
SEED31337_OV=$(grep -E "FINAL OV \((base|opt)\):" logs/r047_v15feat_c_oldtest_seed31337.log | tail -2 | tr '\n' ' ')
echo "  seed 31337 finished: $SEED31337_OV" | tee -a "$SUMMARY"

# ---- Step 3: avg3 combine ----
log_section "Step 3/7: avg3 combine (51966 + 7 + 31337)"
python -u src/avg_oof.py \
    --tags v14_seed2_v15feat_c_oldtest \
           v14_seed2_v15feat_c_oldtest_seed7 \
           v14_seed2_v15feat_c_oldtest_seed31337 \
    --out-tag v14_seed2_v15feat_c_oldtest_avg3 \
    > logs/v15feat_c_oldtest_avg3_combine.log 2>&1
AVG3_LINE=$(grep "Averaged OOF" logs/v15feat_c_oldtest_avg3_combine.log)
echo "  $AVG3_LINE" | tee -a "$SUMMARY"

# ---- Step 4-6: Build candidate submissions ----
log_section "Step 4-6: Build B-feature swap candidate submissions"
python -u src/build_overnight_candidates.py 2>&1 | tee -a "$SUMMARY"

# ---- Step 7: Final summary ----
log_section "Step 7/7: OVERNIGHT CHAIN COMPLETE"
echo "  artifacts ready in submissions/:" | tee -a "$SUMMARY"
ls submissions/submission_R0{58,35,37}*PLUS_RULE.csv 2>&1 | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "  Next: Jabir wakes up, picks 2 of {R-058r, R-035r, R-037r} to upload." | tee -a "$SUMMARY"
echo "  R-058r is the freshest B-feature win candidate (v15feat_c_oldtest_avg3 swap)." | tee -a "$SUMMARY"
echo "  R-035r (recvhand) and R-037r (recvprofile) are backup B-feature swaps." | tee -a "$SUMMARY"
