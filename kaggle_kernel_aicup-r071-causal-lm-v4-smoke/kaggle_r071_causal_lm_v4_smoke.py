# # R-071 Path B Causal LM v4 — focal loss + class-balanced sampling (Fold-1 smoke)
#
# Successor to R-066 (Path B causal LM). v3 had AUC +0.066 vs v11 baseline but
# full-model OV failed gate by -0.0065. R-070 v15feat_e canary analysis showed
# push-family imbalance (action5/6/13) as root cause. v4 attacks this with:
#   - Focal CE (gamma=2.0) on action head
#   - Cui et al. 2019 class-balanced weights (beta=0.999) over action labels
#   - Point + server losses unchanged from v3
#   - All R-066 v3 fixes preserved (label shift, position-0 valid, BCE masking)
#
# Authorization: autonomous mode (Jabir 2026-05-25). No LB upload; OOF artifact
# only. Stop gates wired below — auto-mark ARTIFACT_READY_FOR_JABIR_UPLOAD_REVIEW
# only if all gates pass.

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

# ## Sanity — required files (must include the new v4 trainer)

# +
required = [
    "code/train_causal_lm_v4.py",   # NEW — must be in dataset v.latest
    "code/train_causal_lm_v1.py",   # kept for reference
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

# ## Launch Fold-1 smoke (~2-3 h target on T4)

# +
cmd = [
    "python", "-u", str(CODE_DIR / "train_causal_lm_v4.py"),
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
    "--tag", "v22_causal_lm_v4_smoke",
    "--seed", "42",
    # R-071 v4-specific
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
log_path = OUT_DIR / "logs" / "r071_causal_lm_v4_smoke.log"
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
tag = "v22_causal_lm_v4_smoke"
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

# ## Per-class F1 audit (push-family canary check)

# +
from sklearn.metrics import f1_score

# Compute per-class F1 for action and compare against R-066 v3's known issue
# (push-family classes 5, 6, 13 were regressing in v15feat_e analogue).
y_act = np.load(oof_dir / f"{tag}_oof_y_act.npy") if (oof_dir / f"{tag}_oof_y_act.npy").exists() else None
oof_act = np.load(oof_dir / f"{tag}_oof_act.npy")
mask = np.load(oof_dir / f"{tag}_oof_mask.npy")
if y_act is not None and mask.any():
    N_ACTION_TRAIN = 15
    valid = mask
    pred_a = oof_act[valid, :N_ACTION_TRAIN].argmax(axis=1)
    y_a_clip = np.where(y_act[valid] >= N_ACTION_TRAIN, 0, y_act[valid])
    f1_per_class = f1_score(y_a_clip, pred_a, labels=list(range(N_ACTION_TRAIN)),
                              average=None, zero_division=0)
    class_names = ["None", "Loop", "Cloop", "Smash", "Flip", "Pushfast", "Push",
                   "Flick", "Arch", "Knuckle", "Chop_r", "ShortStop", "Chop",
                   "Block", "Lob"]
    print("\nAction per-class F1 (R-071 v4 smoke OOF):")
    push_family = [5, 6, 13]
    for cls_id, (name, f1) in enumerate(zip(class_names, f1_per_class)):
        n = int((y_a_clip == cls_id).sum())
        flag = " [PUSH]" if cls_id in push_family else ""
        print(f"  action{cls_id:>2} {name:<10} n={n:>5} F1={f1:.4f}{flag}")
    push_f1 = [f1_per_class[c] for c in push_family]
    print(f"\n  Push family mean F1: {np.mean(push_f1):.4f}  "
          f"(target: >= R-066 v3 baseline ~0.40)")
# -

# ## Stop-gate decision

# +
# Extract metrics
ov_match = re.search(r"FINAL OV.*?:\s+(\d+\.\d+)", log_txt)
final_ov = float(ov_match.group(1)) if ov_match else None
final_auc = None
for m in re.finditer(r"F1_a=(\d+\.\d+)\s+F1_p=(\d+\.\d+)\s+AUC=(\d+\.\d+)", log_txt):
    final_auc = float(m.group(3))   # last match = final OOF AUC

print("\n=== R-071 v4 SMOKE STOP-GATE DECISION ===")
print(f"  Final OV:  {final_ov}")
print(f"  Final AUC: {final_auc}")
print(f"  Gate 1 (full-model OV >= 0.295):    "
      f"{'PASS' if final_ov is not None and final_ov >= 0.295 else 'FAIL'}")
print(f"  Gate 2 (server-head AUC >= 0.65):   "
      f"{'PASS' if final_auc is not None and final_auc >= 0.65 else 'FAIL'}")
if y_act is not None and mask.any():
    push_mean = float(np.mean(push_f1))
    print(f"  Gate 3 (push-family F1 >= 0.38):    "
          f"push_mean={push_mean:.4f}  "
          f"{'PASS' if push_mean >= 0.38 else 'FAIL'}")

print("\n=== POST-SMOKE ACTION (manual after pulling artifacts) ===")
print("If all 3 gates PASS -> autonomous mode will:")
print("  1. Pull artifacts via 'kaggle kernels output ...'")
print("  2. Auto-launch R-071 full 5-fold kernel")
print("If any gate FAILS -> mark R-071 as PARK; pivot to R-073 / R-074.")
# -
