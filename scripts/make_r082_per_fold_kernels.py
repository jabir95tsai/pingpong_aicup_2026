"""Generate per-fold kernel scaffolds for R-082 Phase 2 split (folds 1-4).

Fold 0 was already trained successfully (models/v11_fold0.pt pulled from
earlier kernel). This script creates 4 kernel directories, each running
v11 training for ONE specific fold via --fold-only N.

After running this script, push each kernel with:
    for f in 1 2 3 4; do
        kaggle kernels push -p kaggle_kernel_aicup-r082-v11-fold$f
    done
"""
from __future__ import annotations

import json
import os
import sys

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOTEBOOK_TEMPLATE = '''# # R-082 Phase 2 split — V11 train fold {fold} only (--fold-only {fold})
#
# Per-fold kernel to fit Kaggle 12hr CPU limit. The original combined kernel
# (5 folds) hit the limit and only saved fold 0. This kernel trains ONLY
# fold {fold}, producing models/v11_fold{fold}.pt.

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
print(f"PyTorch {{torch.__version__}}  CUDA={{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"  {{torch.cuda.get_device_name(0)}}")

cmd = [
    "python", "-u", str(CODE_DIR / "train_v11_transformer.py"),
    "--folds", "5",
    "--epochs", "80",
    "--tag", "v11",
    "--save-checkpoint",
    "--fold-only", "{fold}",
    "--test-path", str(DATA_DIR / "test_new.csv"),
]
print(f"Cmd: {{' '.join(cmd)}}")
env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"] = str(DATA_DIR / "test_new.csv")

t0 = time.time()
log_path = OUT_DIR / "logs" / "r082_v11_fold{fold}.log"
with open(log_path, "w") as f:
    proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
elapsed = (time.time() - t0) / 60
print(f"\\nElapsed: {{elapsed:.1f}} min  exit={{proc.returncode}}")
assert proc.returncode == 0

models_dir = OUT_DIR / "models"
fp = models_dir / "v11_fold{fold}.pt"
if fp.exists():
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    print(f"OK saved {{fp.name}} best_ov={{ck.get('best_ov')}}")
else:
    print(f"MISSING {{fp.name}}")
'''

METADATA_TEMPLATE = {
    "id": None,            # set per fold
    "title": None,         # set per fold
    "code_file": None,     # set per fold
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": ["jabir95tsai/aicup2026-pingpong-private"],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}


def main():
    for fold in [1, 2, 3, 4]:
        dir_name = f"kaggle_kernel_aicup-r082-v11-fold{fold}"
        out_dir = os.path.join(ROOT, dir_name)
        os.makedirs(out_dir, exist_ok=True)

        # Write .py notebook (will convert to ipynb via jupytext)
        py_filename = f"kaggle_r082_v11_fold{fold}.py"
        py_path = os.path.join(out_dir, py_filename)
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(NOTEBOOK_TEMPLATE.format(fold=fold))
        print(f"  wrote {py_path}")

        # Write kernel-metadata.json
        meta = dict(METADATA_TEMPLATE)
        meta["id"] = f"jabir95tsai/aicup-r-082-v11-fold{fold}"
        meta["title"] = f"AICUP R 082 V11 Fold {fold}"
        meta["code_file"] = f"kaggle_r082_v11_fold{fold}.ipynb"
        meta_path = os.path.join(out_dir, "kernel-metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"  wrote {meta_path}")

    print("\nNext steps:")
    print("  for f in 1 2 3 4; do")
    print("    cd kaggle_kernel_aicup-r082-v11-fold$f")
    print("    jupytext --to ipynb kaggle_r082_v11_fold$f.py")
    print("    # inject kernelspec (see prior kernel patches)")
    print("  done")
    print("  for f in 1 2 3 4; do kaggle kernels push -p kaggle_kernel_aicup-r082-v11-fold$f; done")


if __name__ == "__main__":
    main()
