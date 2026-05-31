# # Kaggle batch retrain — Transformer Tier 1+2 (use ALL legal data)
#
# Trains the v11_aug + v11plus + v11 family with --include-old-test AND
# --aug-parquet (both legal data axes) on Kaggle GPU T4 x2.
#
# Per-variant time on Kaggle GPU T4:
#   v11plus_oldtest_aug_v2   ~2 hr     (transformer, biggest single upgrade)
#   v11_oldtest_aug          ~1.5 hr   (smaller model)
#
# Total: ~3.5 hr — well under 12-hr session.

# ## Setup

# +
import os, sys, time, subprocess, json
from pathlib import Path
from datetime import datetime

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Num GPUs: {torch.cuda.device_count()}")

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
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)

os.environ["PINGPONG_DATA_DIR"] = str(DATA_DIR)
print(f"PINGPONG_DATA_DIR={DATA_DIR}, PROJECT_ROOT={OUT_DIR}")
# -

# ## Verify dependencies

# +
import pandas as pd
train = pd.read_csv(DATA_DIR / "train.csv")
old_test = pd.read_csv(DATA_DIR / "test.csv")
test_new = pd.read_csv(DATA_DIR / "test_new.csv")
aug_parquet = DATA_DIR / "test_history_pairs_new.parquet"
print(f"train: {len(train):,} rows")
print(f"old_test: {len(old_test):,} rows / {old_test.rally_uid.nunique():,} rallies")
print(f"test_new: {len(test_new):,} rows / {test_new.rally_uid.nunique():,} rallies")
print(f"aug_parquet exists: {aug_parquet.exists()}")
if aug_parquet.exists():
    aug = pd.read_parquet(aug_parquet)
    print(f"  aug rows: {len(aug):,}  is_aug=1: {(aug['is_aug']==1).sum():,}")

for s in ["train_v11_transformer.py", "config.py", "data_cleaning.py"]:
    print(f"  trainer present: {s} - {(CODE_DIR / s).exists()}")
# -

# ## Batch definition

# +
TRAIN_PATH = str(DATA_DIR / "train.csv")
OLD_TEST_PATH = str(DATA_DIR / "test.csv")
TEST_NEW_PATH = str(DATA_DIR / "test_new.csv")
AUG_PARQUET = str(DATA_DIR / "test_history_pairs_new.parquet")

BATCH = [
    # v11 baseline with both legal data axes (default arch).
    {
        "tag": "v11_oldtest_aug",
        "trainer": "train_v11_transformer.py",
        "args": ["--epochs", "80", "--batch", "256", "--lr", "3e-4",
                 "--include-old-test", OLD_TEST_PATH,
                 "--aug-parquet", AUG_PARQUET],
    },
    # v11plus-equivalent (bigger d_model + 6 layers) with both data axes.
    {
        "tag": "v11plus_oldtest_aug_v2",
        "trainer": "train_v11_transformer.py",
        "args": ["--epochs", "80", "--batch", "256", "--lr", "3e-4",
                 "--d-model", "256", "--n-heads", "8", "--n-layers", "6",
                 "--include-old-test", OLD_TEST_PATH,
                 "--aug-parquet", AUG_PARQUET],
    },
]
print(f"Batch: {len(BATCH)} variants")
# -

# ## Run

# +
results = []
batch_start = time.time()
for i, item in enumerate(BATCH, 1):
    print()
    print("=" * 70)
    print(f"[{i}/{len(BATCH)}] {item['tag']}")
    print("=" * 70)
    t0 = time.time()
    cmd = [
        "python", "-u", str(CODE_DIR / item["trainer"]),
        "--tag", item["tag"],
        "--test-path", TEST_NEW_PATH,
    ] + item["args"]
    log_path = OUT_DIR / "logs" / f"{item['tag']}.log"
    env = os.environ.copy()
    env["TRAIN_PATH"] = TRAIN_PATH
    env["TEST_PATH"] = TEST_NEW_PATH

    print(f"Cmd: {' '.join(cmd)}")
    print(f"Log: {log_path}")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60
    status = "OK" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
    print(f"Done: {status}  elapsed {elapsed:.1f} min")
    results.append({"tag": item["tag"], "status": status,
                    "elapsed_min": elapsed, "exit_code": proc.returncode})

total_min = (time.time() - batch_start) / 60
print()
print(f"TRANSFORMER BATCH A complete in {total_min:.1f} min")
for r in results:
    marker = "OK" if r["exit_code"] == 0 else "FAIL"
    print(f"  [{marker:4}] {r['tag']:40s}  {r['elapsed_min']:.1f} min")

(OUT_DIR / "logs" / "batch_transformer_a_results.json").write_text(
    json.dumps({"batch": "transformer_a", "results": results,
                "total_min": total_min,
                "completed_at": datetime.now().isoformat()},
               indent=2), encoding="utf-8")
# -

# ## Verify outputs

# +
import numpy as np
oof_dir = OUT_DIR / "oof_predictions"
for item in BATCH:
    fp = oof_dir / f"{item['tag']}_oof_act.npy"
    if fp.exists():
        arr = np.load(fp)
        print(f"  OK {item['tag']}: shape={arr.shape}")
    else:
        print(f"  MISSING {item['tag']}")
# -
