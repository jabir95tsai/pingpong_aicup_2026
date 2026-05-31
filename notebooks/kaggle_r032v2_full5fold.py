# # R-032 v2.1 Full 5-fold (Codex APPROVED 2026-05-22)
#
# Cross-rally LORO match-pair features. Full 5-fold v14_seed2 + v16match_v2.
# Codex-approved scope (do NOT modify):
#   - match_pair grouping
#   - MAX_OTHER_RALLIES=22, PREFIX_CAP_K=3, MIN_OTHER_RALLIES=3
#   - Family A only (Family B deferred)
#   - canonical train axis only (no oldtest, no v15feat compounding)
#   - tag: v14_seed2_v16match_v2 (distinct from smoke)
#
# Post-run policy (Codex):
#   - Low-cost analyzer / blend-swap diagnostics allowed
#   - NO direct LB upload until Codex reviews full-artifact metrics
#
# ETA on Kaggle CPU: ~6-8 hr (within 12-hr cap).

# ## Setup

# +
import os
import sys
import time
import subprocess
from pathlib import Path

DATA_DIR = Path("/kaggle/input/datasets/jabir95tsai/aicup2026-pingpong-private")
if not DATA_DIR.exists():
    DATA_DIR = Path("/kaggle/input/aicup2026-pingpong-private")
OUT_DIR = Path("/kaggle/working")
RO_CODE_DIR = DATA_DIR / "code"
CODE_DIR = OUT_DIR / "src"
import shutil
if not CODE_DIR.exists():
    shutil.copytree(RO_CODE_DIR, CODE_DIR)
    print(f"Copied source from {RO_CODE_DIR} to {CODE_DIR}")
sys.path.insert(0, str(CODE_DIR))

(OUT_DIR / "oof_predictions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "submissions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "runs").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

os.environ["PINGPONG_DATA_DIR"] = str(DATA_DIR)
print(f"PINGPONG_DATA_DIR={DATA_DIR}")
print(f"CODE_DIR={CODE_DIR}")
print(f"OUT_DIR={OUT_DIR} (writable)")
# -

# ## Sanity check — required files

# +
required = [
    "code/features_v16match_v2.py",
    "code/features_v9.py",
    "code/config.py",
    "code/data_cleaning.py",
    "code/train_v14.py",
    "train.csv",
    "test.csv",
    "test_new.csv",
    "sample_submission.csv",
]
for r in required:
    p = DATA_DIR / r
    assert p.exists(), f"MISSING: {p}"
    print(f"  OK {r} ({p.stat().st_size} bytes)")
# -

# ## Verify cap is wired (sanity)

# +
import features_v16match_v2 as v16
print(f"MAX_OTHER_RALLIES = {v16.MAX_OTHER_RALLIES}")
print(f"PREFIX_CAP_K = {v16.PREFIX_CAP_K}")
print(f"MIN_OTHER_RALLIES = {v16.MIN_OTHER_RALLIES}")
assert v16.MAX_OTHER_RALLIES == 22, "Cap must be 22 per Codex approval"
assert v16.PREFIX_CAP_K == 3, "Prefix cap must be 3 per Codex approval"
# -

# ## Launch the full 5-fold

# +
cmd = [
    "python", "-u", str(CODE_DIR / "train_v14.py"),
    "--feature-set", "v16match_v2",
    "--tag", "v14_seed2_v16match_v2",
    "--seed", "51966",
    "--folds", "5",
    "--n-boost", "3000",
    "--es", "200",
    "--test-path", str(DATA_DIR / "test_new.csv"),
]
print(f"Cmd: {' '.join(cmd)}")
env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"] = str(DATA_DIR / "test_new.csv")

t0 = time.time()
log_path = OUT_DIR / "logs" / "r032v2_full5fold.log"
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
import pandas as pd

oof_dir = OUT_DIR / "oof_predictions"
tag = "v14_seed2_v16match_v2"
for suffix in ["oof_act", "oof_pt", "oof_srv", "oof_mask",
               "test_act", "test_pt", "test_srv"]:
    fp = oof_dir / f"{tag}_{suffix}.npy"
    if fp.exists():
        arr = np.load(fp)
        finite = bool(np.isfinite(arr).all())
        print(f"  OK {fp.name}: shape={arr.shape}, finite={finite}")
    else:
        print(f"  MISSING {fp.name}")

# OOF mask coverage
mask = np.load(oof_dir / f"{tag}_oof_mask.npy")
print(f"\nOOF mask coverage: {mask.sum()}/{len(mask)} = {mask.sum()/len(mask):.3%}")
assert mask.sum() == len(mask), "Full 5-fold should cover 100% of train rows"
# -

# ## Final metrics summary

# +
# Extract from log
import re

with open(log_path) as f:
    log_txt = f.read()

print("=== Final OV metrics ===")
for m in re.finditer(r"FINAL OV \((base|opt)\):\s+(\d+\.\d+)", log_txt):
    print(f"  {m.group(0)}")

print("\n=== Per-fold OVs ===")
for m in re.finditer(r"FOLD OV=(\d+\.\d+)", log_txt):
    print(f"  {m.group(0)}")

print("\n=== Last 10 audit lines ===")
for line in log_txt.split("\n"):
    if "[v16match_v2" in line:
        print(f"  {line.strip()}")
# -

# ## Download instructions (for after run completes)
#
# Local PowerShell:
# ```powershell
# kaggle kernels output jabir95tsai/<slug> -p oof_predictions/
# python -u src/audit_all_parked_components.py --n-samples 200
# ```
#
# The new tag v14_seed2_v16match_v2 will auto-appear in the parked audit.
# Per Codex post-run policy: no direct LB upload — first run analyzer +
# blend-swap diagnostics + send metrics for Codex re-review.
