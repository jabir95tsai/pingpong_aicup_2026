# # R-031b SoftF1 Phase-B-only (resume from CE checkpoint)
#
# The original R-031 fold-1 smoke completed Phase A (70 CE warmup epochs) +
# 7 of 10 Phase B epochs before hitting Kaggle's 12-hr timeout. The CE
# checkpoint (best_ce.pt) was saved to /kaggle/working/models and we
# downloaded + re-uploaded it as part of dataset v7 at:
#   /kaggle/input/datasets/jabir95tsai/aicup2026-pingpong-private/ce_checkpoint/best_ce_fold1.pt
#
# This notebook resumes from that checkpoint and runs Phase B only.
# Expected runtime: ~30 minutes on Kaggle GPU T4 (vs 12+ hr full).
#
# Original Phase A baseline (from R-031 run):
#   F1_a=0.3449, F1_p=0.2012, AUC=0.5467, OV=0.3278
# After 7/10 Phase B (R-031):
#   F1_a=0.3512 (+0.0063)
# Predicted final 10/10:
#   F1_a ~0.352 (within Codex predicted +0.005-0.015 range)

# ## Setup

# +
import os, sys, time
from pathlib import Path

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

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
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "runs").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

os.environ["PINGPONG_DATA_DIR"] = str(DATA_DIR)
print(f"PINGPONG_DATA_DIR={DATA_DIR}, PROJECT_ROOT={OUT_DIR}")

# Verify CE checkpoint present
CE_CKPT = DATA_DIR / "ce_checkpoint" / "best_ce_fold1.pt"
assert CE_CKPT.exists(), f"CE checkpoint missing: {CE_CKPT}"
print(f"CE checkpoint: {CE_CKPT} ({CE_CKPT.stat().st_size // 1024} KB)")
# -

# ## Verify aug parquet

# +
aug_parquet = DATA_DIR / "test_history_pairs_new.parquet"
print(f"aug_parquet: {aug_parquet} (exists: {aug_parquet.exists()})")
# -

# ## Run Phase B only

# +
import subprocess

cmd = [
    "python", "-u", str(CODE_DIR / "train_v11_mulminet_softf1.py"),
    "--tag", "v11_mulminet_aug_oldtest_softf1_phaseB",
    "--max-folds", "1",
    "--softf1-epochs", "10",
    "--alpha-start", "0.0",
    "--alpha-end", "0.3",
    "--freeze-encoder",
    "--softf1-lr", "1e-4",
    "--seed", "42",
    "--include-old-test", str(DATA_DIR / "test.csv"),
    "--test-path", str(DATA_DIR / "test_new.csv"),
    "--save-checkpoints",
    "--init-from-ce", str(CE_CKPT),   # SKIP Phase A entirely
]
if aug_parquet.exists():
    cmd += ["--aug-parquet", str(aug_parquet)]

env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"] = str(DATA_DIR / "test_new.csv")

t0 = time.time()
proc = subprocess.run(cmd, env=env)
print(f"\nElapsed: {(time.time()-t0)/60:.1f} min  exit={proc.returncode}")
assert proc.returncode == 0
# -

# ## Read final metrics

# +
import json
candidates = list(Path("/kaggle/working").rglob("*phaseB*metadata.json"))
if not candidates:
    print("No metadata.json found")
else:
    with open(candidates[0]) as f: meta = json.load(f)
    fold1 = meta["fold_metadata"][0]
    print(f"Final standalone_label: {meta.get('gate_verdict', meta.get('standalone_label'))}")
    print(f"Baseline (CE)   F1_a={fold1['baseline_f1_a']:.4f}  OV={fold1['baseline_ov']:.4f}")
    print(f"Post soft-F1    F1_a={fold1['post_f1_a']:.4f}  OV={fold1['post_ov']:.4f}")
    print(f"Deltas:         F1_a {fold1['post_f1_a']-fold1['baseline_f1_a']:+.4f}, OV {fold1['post_ov']-fold1['baseline_ov']:+.4f}")
# -
