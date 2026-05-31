#!/bin/bash
# Local chained retrain — use ALL legal data (--include-old-test)
# Runs each variant sequentially, logs to logs/<tag>.log, continues even if one fails
# Total expected: ~8 hr on local CPU
#
# Launch:
#   bash scripts/local_oldtest_chain.sh > logs/local_oldtest_chain.log 2>&1 &

set -u  # error on undefined vars (not -e: we want to continue past individual failures)

cd /c/Users/jabir/Hacker_J/pingpong_aicup_2026
mkdir -p logs

OLD_TEST="data/test.csv"
TEST_NEW="data/test_new.csv"

run_variant() {
    local tag="$1"
    local trainer="$2"
    shift 2
    local args="$@"

    echo "============================================================"
    echo "[$(date)] STARTING: $tag"
    echo "  trainer: $trainer"
    echo "  args: $args --tag $tag --test-path $TEST_NEW --include-old-test $OLD_TEST"
    echo "============================================================"

    local t0=$(date +%s)
    python -u "src/$trainer" \
        --tag "$tag" \
        --test-path "$TEST_NEW" \
        --include-old-test "$OLD_TEST" \
        $args > "logs/${tag}.log" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        echo "[$(date)] OK $tag  (${elapsed}s)"
    else
        echo "[$(date)] FAIL $tag exit=$rc  (${elapsed}s)  see logs/${tag}.log"
    fi
}

echo "############################################################"
echo "# LOCAL OLDTEST CHAIN — $(date)"
echo "############################################################"

# 1. v14_seed2_v15feat_a_oldtest — R-034 winner + oldtest (highest EV)
run_variant "v14_seed2_v15feat_a_oldtest" "train_v14.py" \
    --feature-set v15feat --seed 51966 --folds 5 --n-boost 3000 --es 200

# 2. sgp_prefix_v3_full_oldtest — SGP specialist + oldtest (fast)
run_variant "sgp_prefix_v3_full_oldtest" "sgp_prefix_v3.py" \
    --full-train --folds 5 --seed 51966

# 3. v14_seed2_v15feat_b_oldtest — R-029b features + oldtest
run_variant "v14_seed2_v15feat_b_oldtest" "train_v14.py" \
    --feature-set v15feat_b --seed 51966 --folds 5 --n-boost 3000 --es 200

# 4. v14_recvhand_oldtest — recvhand features + oldtest
run_variant "v14_recvhand_oldtest" "train_v14.py" \
    --feature-set v9_recvhand --seed 42 --folds 5 --n-boost 3000 --es 200

# 5. v14_recvprofile_oldtest — recvprofile features + oldtest
run_variant "v14_recvprofile_oldtest" "train_v14.py" \
    --feature-set v9_recvprofile --seed 42 --folds 5 --n-boost 3000 --es 200

echo ""
echo "############################################################"
echo "# CHAIN DONE at $(date)"
echo "############################################################"
echo "OOF arrays in oof_predictions/:"
ls -la oof_predictions/v14_*_oldtest_oof_act.npy oof_predictions/sgp_prefix_v3_full_oldtest_oof_act.npy 2>&1 | head -10
