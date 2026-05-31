# # R-066 Path B Causal LM smoke (Codex-pending APPROVE_WITH_FIXES)
#
# Per STRATEGY.md §9 Path B design:
#   - Causal Transformer decoder, d=192, 4 layers, 4 heads, FF=768, dropout=0.1
#   - Multi-position objective: predict every position from causal prefix
#   - LM pre-training on visible test action+point (P6 extension, SGP masked)
#
# **Scope**: Fold-1 ONLY smoke (~1 h T4 GPU). No LB upload. Per STRATEGY §9.6
# stop gates: report Fold-1 OV, OOF correlations with v11/v14, per-task F1.
#
# Authorization: user 2026-05-23 "do Path B causal LM smoke, run on kaggle"
# (teammate package_v8 confirmed SGP-leaked → unusable).
#
# **Post-run policy**: do NOT request full 30 h GPU commit without:
#   1. Smoke artifact added to R-066 in REVIEW_QUEUE.md
#   2. Codex review of artifact
#   3. Jabir explicit go-ahead for full commit

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
print(f"OUT_DIR={OUT_DIR} (writable)")
# -

# ## Sanity — required files

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

# ## GPU check

# +
import torch
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"  total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# -

# ## Launch Fold-1 smoke (~1 h target)

# +
cmd = [
    "python", "-u", str(CODE_DIR / "train_causal_lm_v1.py"),
    "--smoke",
    "--max-folds", "1",
    "--epochs", "25",
    "--batch-size", "32",
    "--lr", "1e-4",
    "--weight-decay", "1e-2",
    "--patience", "5",
    "--d-model", "192",
    "--n-heads", "4",
    "--n-layers", "4",
    "--dropout", "0.1",
    "--tag", "v22_causal_lm_v1_smoke",
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
log_path = OUT_DIR / "logs" / "r066_causal_lm_v1_smoke.log"
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
tag = "v22_causal_lm_v1_smoke"
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
print(f"\nOOF mask coverage (smoke = single fold): "
      f"{mask.sum()}/{len(mask)} = {mask.sum()/len(mask):.3%}")
# -

# ## Extract Fold-1 metrics from log

# +
import re

with open(log_path) as f:
    log_txt = f.read()

print("=== Fold-1 OV ===")
for m in re.finditer(r"FOLD 1 OV=(\d+\.\d+)", log_txt):
    print(f"  {m.group(0)}")
for m in re.finditer(r"FINAL OV.*?:\s+(\d+\.\d+)", log_txt):
    print(f"  {m.group(0)}")

print("\n=== Per-task F1 / AUC ===")
for m in re.finditer(r"F1_a=(\d+\.\d+)\s+F1_p=(\d+\.\d+)\s+AUC=(\d+\.\d+)", log_txt):
    print(f"  F1_a={m.group(1)}  F1_p={m.group(2)}  AUC={m.group(3)}")

print("\n=== Epoch loss progression ===")
for m in re.finditer(r"Epoch\s+(\d+)/(\d+)\s+tr_loss=(\d+\.\d+)\s+val_loss=(\d+\.\d+)", log_txt):
    print(f"  ep {m.group(1)}: tr {m.group(3)}  val {m.group(4)}")
# -

# ## OOF correlation with v11 / v14_seed2 (diversity gate)

# +
# Skip if reference OOF arrays not staged in this Kaggle dataset
ref_v11 = oof_dir / "v11_oof_act.npy"
ref_v14 = oof_dir / "v14_seed2_oof_act.npy"
if ref_v11.exists() and ref_v14.exists():
    new_act = np.load(oof_dir / f"{tag}_oof_act.npy")
    ref_v11_act = np.load(ref_v11)
    ref_v14_act = np.load(ref_v14)
    new_mask = np.load(oof_dir / f"{tag}_oof_mask.npy")
    # Compare on the smoke fold rows only (top-1 agreement)
    if new_mask.any():
        valid = new_mask & np.arange(len(new_mask)) < min(len(ref_v11_act), len(ref_v14_act))
        if valid.sum() > 0:
            new_pred = new_act[valid, :15].argmax(axis=1)
            v11_pred = ref_v11_act[valid][:, :15].argmax(axis=1)
            v14_pred = ref_v14_act[valid][:, :15].argmax(axis=1)
            corr_v11 = (new_pred == v11_pred).mean()
            corr_v14 = (new_pred == v14_pred).mean()
            print(f"  Action top-1 agreement vs v11: {corr_v11:.4f}")
            print(f"  Action top-1 agreement vs v14: {corr_v14:.4f}")
            print(f"  Target for diversity-only commit: < 0.85")
        else:
            print("  (no overlapping valid rows)")
else:
    print(f"  Reference OOF arrays not staged in dataset — skip correlation gate.")
# -

# ## Stop-gate decision (per STRATEGY.md §9.6)
#
# | Fold-1 OV | OOF corr w/ v11/v14 | Verdict |
# |---|---|---|
# | ≥ 0.314 (v11 baseline) | any | request full ~30 h commit |
# | 0.295 - 0.314 | < 0.85 | request commit for diversity-only zoo addition |
# | 0.295 - 0.314 | ≥ 0.95 | PARK (no diversity) |
# | < 0.295 | any | PARK (uncompetitive) |
#
# Manual decision step after this kernel finishes:
# 1. Extract Fold-1 OV from log above
# 2. Look up corresponding stop-gate row
# 3. Update R-066 in REVIEW_QUEUE.md with smoke artifact
# 4. Wait for Codex review + Jabir go-ahead before any further work

# ## Download / commit notes
#
# After the kernel runs:
# ```bash
# kaggle kernels output jabir95tsai/aicup-r066-causal-lm-smoke -p oof_predictions/
# # OOF arrays for blend-correlation audit will be at oof_predictions/v22_causal_lm_v1_smoke_*.npy
# ```
