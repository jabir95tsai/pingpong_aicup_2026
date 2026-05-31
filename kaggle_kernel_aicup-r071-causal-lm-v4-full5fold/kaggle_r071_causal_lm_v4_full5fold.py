# # R-071 Causal LM v4 — FULL 5-FOLD (focal + class-balanced)
#
# Launched only if R-071 smoke gates pass (OV>=0.295, AUC>=0.65, push F1>=0.38).
# 8-10h T4 GPU. Produces 5-fold OOF + averaged test predictions for blend candidate.

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
sys.path.insert(0, str(CODE_DIR))
(OUT_DIR / "oof_predictions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
os.environ["PINGPONG_DATA_DIR"] = str(DATA_DIR)
print(f"DATA_DIR={DATA_DIR}")
print(f"CODE_DIR={CODE_DIR}")

import torch
print(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)}")
# -

# ## Launch full 5-fold

# +
cmd = [
    "python", "-u", str(CODE_DIR / "train_causal_lm_v4.py"),
    "--folds", "5",
    "--epochs", "40",
    "--batch-size", "32",
    "--lr", "1e-4",
    "--weight-decay", "1e-2",
    "--patience", "5",
    "--d-model", "192",
    "--n-heads", "4",
    "--n-layers", "4",
    "--dropout", "0.1",
    "--tag", "v22_causal_lm_v4_full",
    "--seed", "42",
    "--focal-gamma", "2.0",
    "--cb-beta", "0.999",
    "--include-old-test", str(DATA_DIR / "test.csv"),
    "--include-test-history", str(DATA_DIR / "test_history_pairs_new.parquet"),
    "--test-path", str(DATA_DIR / "test_new.csv"),
]
print(f"Cmd: {' '.join(cmd)}")
env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"] = str(DATA_DIR / "test_new.csv")

t0 = time.time()
log_path = OUT_DIR / "logs" / "r071_causal_lm_v4_full.log"
with open(log_path, "w") as f:
    proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
elapsed = (time.time() - t0) / 60
print(f"\nElapsed: {elapsed:.1f} min  exit={proc.returncode}")
assert proc.returncode == 0
# -

# ## Verify outputs

# +
import numpy as np
oof_dir = OUT_DIR / "oof_predictions"
tag = "v22_causal_lm_v4_full"
for suffix in ["oof_act", "oof_pt", "oof_srv", "oof_mask",
               "test_act", "test_pt", "test_srv", "test_rally_uid",
               "oof_y_act", "oof_y_pt", "oof_y_srv"]:
    fp = oof_dir / f"{tag}_{suffix}.npy"
    if fp.exists():
        arr = np.load(fp)
        finite = bool(np.isfinite(arr).all()) if arr.dtype.kind in "fc" else True
        print(f"  OK {fp.name}: shape={arr.shape}, finite={finite}")
    else:
        print(f"  MISSING {fp.name}")
mask = np.load(oof_dir / f"{tag}_oof_mask.npy")
print(f"\nOOF coverage (5-fold expected ~100%): "
      f"{mask.sum()}/{len(mask)} = {mask.sum()/len(mask):.3%}")
# -

# ## Final metrics from log

# +
import re
log_txt = open(log_path).read()
print("=== Final OV ===")
for m in re.finditer(r"FINAL OV.*?:\s+(\d+\.\d+).*?F1_a=(\d+\.\d+)\s+F1_p=(\d+\.\d+)\s+AUC=(\d+\.\d+)", log_txt):
    print(f"  OV={m.group(1)}  F1_a={m.group(2)}  F1_p={m.group(3)}  AUC={m.group(4)}")
# -
