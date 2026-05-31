# R-082 Phase 2 Post-Completion Runbook

When `aicup-r-082-v11-retrain-with-checkpoint` reaches `COMPLETE` status,
execute these steps in order. All scripts are pre-built and syntax-verified.

---

## Step 1: pull artifacts and verify checkpoint existence

```bash
mkdir -p kaggle_pulls/r082_phase2_final
kaggle kernels output jabir95tsai/aicup-r-082-v11-retrain-with-checkpoint -p kaggle_pulls/r082_phase2_final/

# Verify 5 fold checkpoints exist
ls -la kaggle_pulls/r082_phase2_final/models/v11_fold*.pt
# Expected: v11_fold0.pt through v11_fold4.pt (~10-50 MB each)
```

**Sanity checks**:
- Read the kernel log: `tail -100 kaggle_pulls/r082_phase2_final/aicup-r-082-v11-retrain-with-checkpoint.log`
- Look for `exit=0` and 5x `BEST FOLD: F1_a=... OV=...` lines
- Look for 5x `Saved checkpoint: models/v11_fold{0-4}.pt`

If any checkpoint missing or exit != 0 → diagnose error, decide whether to retry.

---

## Step 2: copy checkpoints to local models/

```bash
cp kaggle_pulls/r082_phase2_final/models/v11_fold*.pt models/
ls -la models/v11_fold*.pt
```

---

## Step 3: extract OOF + test embeddings (local, no GPU needed)

```bash
python -u src/extract_v11_embeddings.py --tag v11
```

**Expected outputs** in `oof_predictions/`:
- `v11_emb_last_oof.npy` shape `(69712, 192)`
- `v11_emb_pool_oof.npy` shape `(69712, 192)`
- `v11_emb_last_test.npy` shape `(1845, 192)`
- `v11_emb_pool_test.npy` shape `(1845, 192)`
- `v11_emb_oof_mask.npy` shape `(69712,)`

Time: ~5-15 minutes local CPU.

---

## Step 4: Phase 3 GBM-on-embedding Fold-1 smoke

```bash
python -u src/train_gbm_on_v11_embed_smoke.py
```

**Decision criterion** (from script's stop gate):
- Compare GBM-on-emb F1/AUC to V11's own head F1/AUC for each task
- **PASS if any task has Δ ≥ +0.005** (embeddings carry usable extra signal)
- **FAIL if all deltas < +0.005** (V11 heads already extract maximum)

Output: `submissions/r082_phase2_step3_gbm_emb_smoke.json`

---

## Step 5: branching based on Phase 3 result

### Step 5a (if FAIL — all deltas < +0.005)

- Mark R-082 family as **PROVISIONAL_FAIL** in REVIEW_QUEUE
- v0.4 verdict: embeddings overfit OOF / V11 heads already optimal
- Document in `audits/R082_post_completion_runbook_2026-05-26.md` as resolved
- Pivot priorities:
  - **R-080** (probability stack) now eligible per user policy ("R-082 blocked AND R-081 unpromising" both true)
  - OR design new STRATEGIC mechanism (architectural; needs Kaggle GPU)
- Goal stop condition reached if no new STRATEGIC mechanism in design

### Step 5b (if PASS — signal exists on at least one task)

For each passing task `<X>` (action/point/server):

```bash
python -u src/build_r082_phase4_lb_candidate.py --task <X> --alpha 0.10
```

This produces:
- `submissions/submission_R082phase4_R067cr_<X>_emb<last|pool>_alpha010_PLUS_RULE.csv`
- `submissions/r082_phase4_<X>_alpha010_manifest.json`

Optionally sweep alphas 0.05 / 0.10 / 0.15 / 0.20 to find best OOF lift.

Mark CSV as ARTIFACT_READY_FOR_JABIR_UPLOAD_REVIEW.

---

## Step 6: update state files

After Phase 4 CSV built (or Phase 3 fail), update:

- `STATE_SUMMARY.md` (latest state section)
- `REVIEW_QUEUE.md` (jump list + R-082 entry)
- `AUTONOMOUS_RUN_QUEUE.md` (run log + queue)
- `RESULTS.md` (full Phase 2-3-4 results writeup with v0.4 candidate report)

---

## Step 7: if Phase 3 PASS, consider Phase 5 — full ensemble

If R-082 Phase 4 produces ARTIFACT_READY CSVs for MULTIPLE tasks (action +
point + server all signal-passed), consider:

- Phase 5: combine all three task corrections into a single CSV
- This is a 3-way candidate that tests whether the embedding signal compounds

Caveat: each task-correction is mechanism-distinct from R-067cr base. Stacking
all 3 increases cumulative LB risk. Recommend uploading the strongest single
task first, then add others if LB+ confirmed.

---

## Stop-condition check on completion

Per active /goal directive:
> Stop conditions: LB reaches >= 0.4000, OR all STRATEGIC + HIGH candidates
> are exhausted (in_progress queue empty AND no new mechanism in design), OR
> user explicit clear.

After R-082 Phase 2 lands:
- If Phase 3 PASS → R-082 advances to Phase 4 → STRATEGIC still in_progress → goal continues
- If Phase 3 FAIL → R-082 STRATEGIC exhausted → if no new STRATEGIC in design AND HIGH queue empty → **goal auto-clears**

Either way, Jabir's LB uploads (independent of R-082 timeline) can directly
satisfy the LB ≥ 0.4000 stop condition if R-094/R-081/R-082-Phase-4 wins big
enough.
