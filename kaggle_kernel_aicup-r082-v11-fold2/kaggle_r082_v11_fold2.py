# # R-082 Phase 2 split — V11 train fold 2 only (--fold-only 2)
#
# Per-fold kernel to fit Kaggle 12hr CPU limit. The original combined kernel
# (5 folds) hit the limit and only saved fold 0. This kernel trains ONLY
# fold 2, producing models/v11_fold2.pt.

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
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
os.environ["PINGPONG_DATA_DIR"] = str(DATA_DIR)

# Verify patched trainer
import io
with io.open(CODE_DIR / "train_v11_transformer.py", encoding="utf-8") as f:
    src = f.read()
assert "--fold-only" in src, "Patched train_v11_transformer.py missing --fold-only flag"
print("OK --fold-only flag confirmed in dataset's train_v11_transformer.py")

import torch
print(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)}")

cmd = [
    "python", "-u", str(CODE_DIR / "train_v11_transformer.py"),
    "--folds", "5",
    "--epochs", "80",
    "--tag", "v11",
    "--save-checkpoint",
    "--fold-only", "2",
    "--test-path", str(DATA_DIR / "test_new.csv"),
]
print(f"Cmd: {' '.join(cmd)}")
env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"] = str(DATA_DIR / "test_new.csv")

t0 = time.time()
log_path = OUT_DIR / "logs" / "r082_v11_fold2.log"
with open(log_path, "w") as f:
    proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
elapsed = (time.time() - t0) / 60
print(f"\nElapsed: {elapsed:.1f} min  exit={proc.returncode}")
assert proc.returncode == 0

models_dir = OUT_DIR / "models"
fp = models_dir / "v11_fold2.pt"
if fp.exists():
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    print(f"OK saved {fp.name} best_ov={ck.get('best_ov')}")
else:
    print(f"MISSING {fp.name}")
