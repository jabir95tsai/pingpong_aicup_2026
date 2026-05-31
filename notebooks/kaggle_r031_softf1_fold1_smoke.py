# # R-031 v1 Soft-F1 Fold-1 Smoke (Kaggle GPU T4)
#
# Per Codex APPROVE_WITH_FIXES (2026-05-21), revised scope:
# - From-scratch retrain (checkpoints don't exist for continuation)
# - Two-phase: CE warmup (70 ep) -> CE + soft-F1 fine-tune (10 ep) on action head
# - Alpha ramp 0.0 -> 0.3 across Phase B
# - Freeze encoder + point + SGP heads during Phase B (makes "point/SGP unchanged" TRUE)
# - Mask absent classes in soft-F1 (rare classes often missing from a batch)
# - Save fold checkpoints (enables future continuation experiments)
# - Fold-1 only; no full 5-fold, no analyzer, no LB until reviewed
#
# Goal: improve action macro F1 by +0.005 to +0.015 -> +0.002 to +0.006 OV
# Baseline (CE-only, fold-1): pinned mid-run from same model
#
# Time budget: ~45 min on T4 (Phase A 70 ep + Phase B 10 ep, action-head-only in B)

# ## Setup

# +
import os
import sys
import time
from pathlib import Path

IN_KAGGLE = Path("/kaggle").exists()
if IN_KAGGLE:
    DATA_DIR = Path("/kaggle/input/datasets/jabir95tsai/aicup2026-pingpong-private")
    if not DATA_DIR.exists():
        DATA_DIR = Path("/kaggle/input/aicup2026-pingpong-private")
    OUT_DIR = Path("/kaggle/working")
    # READ-ONLY source from dataset. We copy it into a writable location
    # so trainer's PROJECT_ROOT=parent(CODE_DIR)/oof_predictions can be written.
    RO_CODE_DIR = DATA_DIR / "code"
    CODE_DIR = OUT_DIR / "src"
    import shutil
    if not CODE_DIR.exists():
        shutil.copytree(RO_CODE_DIR, CODE_DIR)
        print(f"Copied source from {RO_CODE_DIR} to {CODE_DIR}")
else:
    DATA_DIR = Path("data")
    OUT_DIR = Path("kaggle_outputs")
    OUT_DIR.mkdir(exist_ok=True)
    CODE_DIR = Path("src")

print(f"DATA_DIR={DATA_DIR}")
print(f"CODE_DIR={CODE_DIR}")
print(f"OUT_DIR={OUT_DIR}")
sys.path.insert(0, str(CODE_DIR))

import os
os.environ["PINGPONG_DATA_DIR"] = str(DATA_DIR)
print(f"Set PINGPONG_DATA_DIR={DATA_DIR}")
print(f"PROJECT_ROOT for trainers will be: {OUT_DIR} (writable)")

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
# -

# ## Verify base deps (Kaggle base image already has lightgbm/xgboost/torch)

# +
import importlib
for mod in ["torch", "numpy", "pandas", "sklearn", "lightgbm"]:
    m = importlib.import_module(mod)
    print(f"  {mod:10s} version: {getattr(m, '__version__', '?')}")
# -

# ## Sanity check

# +
import pandas as pd

train = pd.read_csv(DATA_DIR / "train.csv")
test_new = pd.read_csv(DATA_DIR / "test_new.csv")
old_test = pd.read_csv(DATA_DIR / "test.csv")
print(f"train: {len(train):,} rows / {train.rally_uid.nunique():,} rallies")
print(f"test_new: {len(test_new):,} rows / {test_new.rally_uid.nunique():,} rallies")
print(f"old_test (legal aug): {len(old_test):,} rows / {old_test.rally_uid.nunique():,} rallies")
# -

# ## Verify aug parquet availability
#
# The R-031 plan uses test_history_pairs_new.parquet as aug data (matches
# v11_mulminet_aug_oldtest setup). If not in the Kaggle dataset, smoke runs
# without aug -- log says it -- but the comparison to baseline is still
# self-consistent because both phases use the same aug setting.

# +
aug_parquet = None
aug_path_candidates = [
    DATA_DIR / "test_history_pairs_new.parquet",
    DATA_DIR / "aug" / "test_history_pairs_new.parquet",
]
for c in aug_path_candidates:
    if c.exists():
        aug_parquet = c
        break
print(f"Aug parquet: {aug_parquet}  ({'WILL use' if aug_parquet else 'NOT available -- smoke runs without aug'})")
# -

# ## Verify R-031 module imports

# +
# Smoke-import: verify modified modules load
import train_v11_mulminet  # noqa
import train_v11_mulminet_softf1  # noqa
print("OK: R-031 trainer imported")
print(f"  softf1_loss: {train_v11_mulminet_softf1.softf1_loss.__doc__.split(chr(10))[0]}")
print(f"  freeze_non_action_params: {train_v11_mulminet_softf1.freeze_non_action_params.__doc__.split(chr(10))[0]}")
# -

# ## Run Fold-1 smoke

# +
import subprocess

cmd = [
    "python", "-u", str(CODE_DIR / "train_v11_mulminet_softf1.py"),
    "--tag", "v11_mulminet_aug_oldtest_softf1_smoke",
    "--max-folds", "1",
    "--ce-epochs", "70",
    "--softf1-epochs", "10",
    "--alpha-start", "0.0",
    "--alpha-end", "0.3",
    "--freeze-encoder",
    "--softf1-lr", "1e-4",
    "--seed", "42",
    "--include-old-test", str(DATA_DIR / "test.csv"),
    "--test-path", str(DATA_DIR / "test_new.csv"),
    "--save-checkpoints",
]
if aug_parquet is not None:
    cmd += ["--aug-parquet", str(aug_parquet)]

env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"] = str(DATA_DIR / "test_new.csv")

(OUT_DIR / "oof_predictions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "runs").mkdir(parents=True, exist_ok=True)

# Override env so the trainer writes outputs into /kaggle/working (downloadable)
env["PROJECT_ROOT_OVERRIDE"] = str(OUT_DIR)

t0 = time.time()
proc = subprocess.run(cmd, env=env)
print(f"\nElapsed: {(time.time()-t0)/60:.1f} min  exit={proc.returncode}")
assert proc.returncode == 0
# -

# ## Read gate verdict

# +
import json

meta_path = OUT_DIR / "runs" / "v11_mulminet_aug_oldtest_softf1_smoke_metadata.json"
if not meta_path.exists():
    # Try the global project root path (in case override didn't take)
    alt = Path("/kaggle/working") / "runs" / "v11_mulminet_aug_oldtest_softf1_smoke_metadata.json"
    if alt.exists():
        meta_path = alt
    else:
        # Search
        candidates = list(Path("/kaggle/working").rglob("v11_mulminet_aug_oldtest_softf1_smoke_metadata.json"))
        if candidates:
            meta_path = candidates[0]
print(f"Reading: {meta_path}")
with open(meta_path) as f:
    meta = json.load(f)

print(f"Gate verdict: {meta['gate_verdict']}")
fold1 = meta["fold_metadata"][0]
print(f"\nBaseline (CE)        F1_a={fold1['baseline_f1_a']:.4f}  F1_p={fold1['baseline_f1_p']:.4f}  AUC={fold1['baseline_auc']:.4f}  OV={fold1['baseline_ov']:.4f}")
print(f"Post soft-F1         F1_a={fold1['post_f1_a']:.4f}  F1_p={fold1['post_f1_p']:.4f}  AUC={fold1['post_auc']:.4f}  OV={fold1['post_ov']:.4f}")
print(f"Deltas               F1_a {fold1['post_f1_a'] - fold1['baseline_f1_a']:+.4f}  OV {fold1['post_ov'] - fold1['baseline_ov']:+.4f}")
print(f"\nPer-class action F1 deltas (cls 0-14):")
action_names = ["None","Loop","Cloop","Smash","Flip","Pushfast","Push","Flick",
                "Arch","Knuckle","Chop_r","ShortStop","Chop","Block","Lob"]
for c, name in enumerate(action_names):
    bf = fold1["baseline_per_class_action_f1"][c]
    pf = fold1["post_per_class_action_f1"][c]
    delta = pf - bf
    star = " *" if delta >= 0.02 else ""
    print(f"  cls {c:2d} {name:10s}: {bf:.4f} -> {pf:.4f}  ({delta:+.4f}){star}")
# -

# ## Download outputs locally (run from your laptop after kernel finishes)
#
# ```powershell
# kaggle kernels output jabir95tsai/<this-notebook-slug> -p oof_predictions/
# # OOF arrays land at oof_predictions/v11_mulminet_aug_oldtest_softf1_smoke_*.npy
# python -u src/audit_all_parked_components.py --n-samples 200
# # The new tag will be auto-detected by the blend audit.
# ```
