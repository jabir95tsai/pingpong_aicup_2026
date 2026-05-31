# # R-066 Path B Causal LM — FULL 5-fold (for R-067 server-head blend)
#
# R-066 PARKED at smoke per STRATEGY §9.6 (Fold-1 OV 0.2885 < 0.295 gate).
# BUT smoke AUC = 0.6759 is +0.066 above v11 baseline → server head is
# diversity-positive. To build R-067 (server-head-only blend) we need
# FULL 5-fold OOF coverage + multi-fold averaged test predictions.
#
# This kernel runs the SAME train_causal_lm_v1.py (bug-fixed v3, label-shift
# applied) for 5 folds instead of 1. ~2 hr wall-clock on T4. Tag stays
# `v22_causal_lm_v1` (full), distinct from the `_smoke` tag.
#
# **Scope**:
# - Full 5-fold OOF coverage (matches canonical 14995-rally train set)
# - Test predictions averaged across 5 folds
# - NO architecture/code changes vs v3 smoke
# - NO class weights added (out of scope per user PARK decision; would be
#   a future R-068 if we revisit Path B)
#
# **Post-run policy**: do NOT mark R-066 as un-PARKED. The full-model OV
# may still fail the §9.6 gate. The artifact we care about is server head
# OOF + test for R-067 specifically.

# ## Setup

# +
import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

DATA_DIR = Path("/kaggle/input/datasets/jabir95tsai/aicup2026-pingpong-private")
if not DATA_DIR.exists():
    DATA_DIR = Path("/kaggle/input/aicup2026-pingpong-private")
OUT_DIR = Path("/kaggle/working")
RO_CODE_DIR = DATA_DIR / "code"
CODE_DIR = OUT_DIR / "src"
if not CODE_DIR.exists():
    shutil.copytree(RO_CODE_DIR, CODE_DIR)
    print(f"Copied source from {RO_CODE_DIR} to {CODE_DIR}")
sys.path.insert(0, str(CODE_DIR))

(OUT_DIR / "oof_predictions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "submissions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

os.environ["PINGPONG_DATA_DIR"] = str(DATA_DIR)
print(f"PINGPONG_DATA_DIR={DATA_DIR}")
print(f"CODE_DIR={CODE_DIR}")
print(f"OUT_DIR={OUT_DIR}")
# -

# ## Sanity check — files

# +
required = [
    "code/train_causal_lm_v1.py",
    "code/config.py",
    "code/data_cleaning.py",
    "train.csv",
    "test.csv",
    "test_new.csv",
    "sample_submission.csv",
    "test_history_pairs_new.parquet",
]
for r in required:
    p = DATA_DIR / r
    assert p.exists(), f"MISSING: {p}"
    print(f"  OK {r} ({p.stat().st_size} bytes)")
# -

# ## Verify label-shift fix is in trainer

# +
import re
with open(CODE_DIR / "train_causal_lm_v1.py") as f:
    src = f.read()
assert "action_logits[:, :-1, :]" in src, \
    "label-shift fix missing — trainer not v3"
assert "y_action[:, 1:]" in src, \
    "label-shift target missing — trainer not v3"
print("  v3 label-shift fix present in trainer ✓")
# -

# ## GPU check

# +
import torch
print(f"PyTorch {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"  total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# -

# ## Launch full 5-fold (~2 hr ETA on T4)

# +
cmd = [
    "python", "-u", str(CODE_DIR / "train_causal_lm_v1.py"),
    # NO --smoke (full mode)
    # NO --max-folds (defaults to 0 = run all --folds)
    "--folds", "5",
    "--epochs", "25",
    "--batch-size", "32",
    "--lr", "1e-4",
    "--weight-decay", "1e-2",
    "--patience", "5",
    "--d-model", "192",
    "--n-heads", "4",
    "--n-layers", "4",
    "--dropout", "0.1",
    "--tag", "v22_causal_lm_v1",   # distinct from smoke tag
    "--seed", "42",
    "--include-old-test", str(DATA_DIR / "test.csv"),
    "--include-test-history", str(DATA_DIR / "test_history_pairs_new.parquet"),
    "--test-path", str(DATA_DIR / "test_new.csv"),
]
print(f"Cmd: {' '.join(cmd)}")
env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"] = str(DATA_DIR / "test_new.csv")

t0 = time.time()
log_path = OUT_DIR / "logs" / "r066_causal_lm_v1_full5fold.log"
with open(log_path, "w") as f:
    proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
elapsed = (time.time() - t0) / 60
print(f"\nElapsed: {elapsed:.1f} min  exit={proc.returncode}")
print(f"Log: {log_path}")
assert proc.returncode == 0
# -

# ## Verify outputs

# +
import numpy as np

oof_dir = OUT_DIR / "oof_predictions"
tag = "v22_causal_lm_v1"
for suffix in ["oof_act", "oof_pt", "oof_srv", "oof_mask",
               "test_act", "test_pt", "test_srv", "test_rally_uid"]:
    fp = oof_dir / f"{tag}_{suffix}.npy"
    if fp.exists():
        arr = np.load(fp)
        finite = bool(np.isfinite(arr).all()) if arr.dtype.kind in "fc" else True
        print(f"  OK {fp.name}: shape={arr.shape}, finite={finite}")
    else:
        print(f"  MISSING {fp.name}")

mask = np.load(oof_dir / f"{tag}_oof_mask.npy")
print(f"\nOOF mask coverage (full 5-fold target = 100%): "
      f"{mask.sum()}/{len(mask)} = {mask.sum()/len(mask):.3%}")
assert mask.sum() == len(mask), "Full 5-fold should cover 100% of train rallies"
# -

# ## Final 5-fold metrics summary

# +
import re

with open(log_path) as f:
    log_txt = f.read()

print("=== Per-fold OVs ===")
for m in re.finditer(r"FOLD \d+ OV=(\d+\.\d+).*F1_a=(\d+\.\d+)\s+F1_p=(\d+\.\d+)\s+AUC=(\d+\.\d+)", log_txt):
    print(f"  {m.group(0)}")

print("\n=== Final OV (5-fold OOF) ===")
for m in re.finditer(r"FINAL OV.*?:\s+(\d+\.\d+)", log_txt):
    print(f"  {m.group(0)}")
for m in re.finditer(r"F1_a=(\d+\.\d+)\s+F1_p=(\d+\.\d+)\s+AUC=(\d+\.\d+)", log_txt):
    print(f"  {m.group(0)}")
# -

# ## Download instructions
#
# After this kernel runs:
# ```bash
# kaggle kernels output jabir95tsai/<full-5fold-slug> -p oof_predictions/
# # New artifacts at oof_predictions/v22_causal_lm_v1_*.npy
# ```
#
# Then build R-067 server-head blend locally:
# ```bash
# python -u src/build_r067_server_blend.py
# ```
