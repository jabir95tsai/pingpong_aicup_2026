# # Kaggle batch retrain — Mulminet family (use ALL legal data)
#
# Trains v11_mulminet base + variants with both legal data axes.
# Already have v11_mulminet_aug_oldtest. Still missing standalone
# v11_mulminet_oldtest_aug (without aug-parquet at start).
#
# Per-variant Kaggle GPU T4 time:
#   v11_mulminet_oldtest_aug    ~2.5 hr
#   v11_mulminet_pretrained_aug_oldtest  ~2.5 hr (uses pretrain ckpt)
#
# Total: ~5 hr.

# ## Setup

# +
import os, sys, time, subprocess, json
from pathlib import Path
from datetime import datetime
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
(OUT_DIR / "submissions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)

os.environ["PINGPONG_DATA_DIR"] = str(DATA_DIR)
print(f"PINGPONG_DATA_DIR={DATA_DIR}, PROJECT_ROOT={OUT_DIR}")

import pandas as pd
print(f"train rows: {len(pd.read_csv(DATA_DIR / 'train.csv')):,}")
# -

# ## Verify trainer

# +
for s in ["train_v11_mulminet.py", "train_v11_mulminet_pretrained.py",
          "config.py", "data_cleaning.py"]:
    print(f"  {s}: {(CODE_DIR / s).exists()}")
# -

# ## Batch

# +
TRAIN_PATH = str(DATA_DIR / "train.csv")
OLD_TEST_PATH = str(DATA_DIR / "test.csv")
TEST_NEW_PATH = str(DATA_DIR / "test_new.csv")
AUG_PARQUET = str(DATA_DIR / "test_history_pairs_new.parquet")

BATCH = [
    # Mulminet base + oldtest + aug
    {
        "tag": "v11_mulminet_oldtest_aug",
        "trainer": "train_v11_mulminet.py",
        "args": ["--include-old-test", OLD_TEST_PATH,
                 "--aug-parquet", AUG_PARQUET,
                 "--aux-lambda", "0.2", "--seed", "42"],
    },
    # Pretrained + oldtest + aug (if pretrained checkpoint isn't shipped, skip)
    # Commented out by default — uncomment if you've added the pretrain ckpt
    # to the dataset.
    # {
    #     "tag": "v11_mulminet_pretrained_oldtest_aug",
    #     "trainer": "train_v11_mulminet_pretrained.py",
    #     "args": ["--include-old-test", OLD_TEST_PATH,
    #              "--aug-parquet", AUG_PARQUET, "--seed", "42"],
    # },
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
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60
    status = "OK" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
    print(f"Done: {status}  elapsed {elapsed:.1f} min")
    results.append({"tag": item["tag"], "status": status,
                    "elapsed_min": elapsed, "exit_code": proc.returncode})

print()
print(f"MULMINET BATCH complete in {(time.time()-batch_start)/60:.1f} min")
for r in results:
    print(f"  [{r['exit_code']}] {r['tag']}  {r['elapsed_min']:.1f} min")
(OUT_DIR / "logs" / "batch_mulminet_results.json").write_text(
    json.dumps({"batch": "mulminet", "results": results,
                "completed_at": datetime.now().isoformat()}, indent=2),
    encoding="utf-8")
# -
