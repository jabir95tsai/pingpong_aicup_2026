# # R-082 Phase 2 — V11 retrain with --save-checkpoint for embedding extraction
#
# Trains canonical V11 transformer (5-fold GroupKFold by match) with the new
# --save-checkpoint flag enabled. Output: 5 fold checkpoints to /kaggle/working/models/
# that can be pulled locally for offline embedding extraction.
#
# Authorized: Jabir 2026-05-26. STRATEGIC priority per GOAL_FUNCTION.md v0.4.
# Expected runtime: ~9 GPU-hours (5 folds × ~110 min each on T4).
# CPU fallback: ~5 hr/fold × 5 = 25 hr → exceeds 12hr Kaggle limit, but smoke
# completed in 13-22 min so most folds may early-stop sooner.

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
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
os.environ["PINGPONG_DATA_DIR"] = str(DATA_DIR)
print(f"DATA_DIR={DATA_DIR}")
print(f"CODE_DIR={CODE_DIR}")

# +
# Verify --save-checkpoint flag is present in the trainer
import io
with io.open(CODE_DIR / "train_v11_transformer.py", encoding="utf-8") as f:
    src = f.read()
assert "--save-checkpoint" in src, "Patched train_v11_transformer.py missing --save-checkpoint flag"
print("OK --save-checkpoint flag confirmed in dataset's train_v11_transformer.py")
# -

# +
import torch
print(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)}  total mem "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# -

# ## Launch v11 5-fold retrain with checkpoint save

# +
cmd = [
    "python", "-u", str(CODE_DIR / "train_v11_transformer.py"),
    "--folds", "5",
    "--epochs", "80",
    "--tag", "v11",
    "--save-checkpoint",
    "--test-path", str(DATA_DIR / "test_new.csv"),
]
print(f"Cmd: {' '.join(cmd)}")
env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"] = str(DATA_DIR / "test_new.csv")

t0 = time.time()
log_path = OUT_DIR / "logs" / "r082_v11_retrain.log"
with open(log_path, "w") as f:
    proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
elapsed = (time.time() - t0) / 60
print(f"\nElapsed: {elapsed:.1f} min  exit={proc.returncode}")
assert proc.returncode == 0
# -

# ## Verify checkpoint outputs (5 fold files)

# +
import numpy as np
models_dir = OUT_DIR / "models"
expected = [models_dir / f"v11_fold{f}.pt" for f in range(5)]
for fp in expected:
    if fp.exists():
        ckpt = torch.load(fp, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        n_params = sum(t.numel() for t in sd.values())
        print(f"  OK {fp.name}  best_ov={ckpt.get('best_ov')}  "
              f"params={n_params:,}  fold={ckpt.get('fold')}")
    else:
        print(f"  MISSING {fp.name}")
# -

# ## Print fold OVs from log

# +
import re
log_txt = open(log_path).read()
print("=== Per-fold OV ===")
for m in re.finditer(r"BEST FOLD: F1_a=(\d+\.\d+)\s+F1_p=(\d+\.\d+)\s+AUC=(\d+\.\d+)\s+OV=(\d+\.\d+)", log_txt):
    print(f"  F1_a={m.group(1)}  F1_p={m.group(2)}  AUC={m.group(3)}  OV={m.group(4)}")
print()
print("=== Final OOF ===")
for m in re.finditer(r"GLOBAL OOF.*?F1_a=(\d+\.\d+)\s+F1_p=(\d+\.\d+)\s+AUC=(\d+\.\d+)\s+OV=(\d+\.\d+)",
                       log_txt, re.DOTALL):
    print(f"  F1_a={m.group(1)}  F1_p={m.group(2)}  AUC={m.group(3)}  OV={m.group(4)}")
# -
