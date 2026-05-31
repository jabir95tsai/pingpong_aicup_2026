# # R-203 — V14 focal CE + Cui CB weights, FULL 5-fold (gated on smoke GO)
#
# DO NOT PUSH until the fold-1 smoke (aicup-r-203-focal-fold1) returns GO.
# Produces the full 5-fold R-203 OOF + test predictions needed to build a
# single-component-swap blend candidate vs R-067cr.
#
# Runs: train_v14.py --folds 5 --r203-focal --tag v14_r203_full --seed 42
# CPU kernel (GBM needs no GPU). ETA ~3-4h (action+point+server × 5 folds).
# Dataset: jabir95tsai/aicup2026-r203-code (code + train.csv + test_new.csv,
#          NO teammate parquet).

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

DATA_DIR = Path("/kaggle/input/datasets/jabir95tsai/aicup2026-r203-code")
if not DATA_DIR.exists():
    DATA_DIR = Path("/kaggle/input/aicup2026-r203-code")
OUT_DIR = Path("/kaggle/working")
CODE_DIR = OUT_DIR / "src"


def _resolve_code_dir():
    """Locate the code/ tree (real dir) or unzip code.zip (--dir-mode zip)."""
    d = DATA_DIR / "code"
    if d.exists() and d.is_dir():
        return d
    import zipfile
    zp = DATA_DIR / "code.zip"
    if zp.exists():
        extract_root = OUT_DIR / "code_unzipped"
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(extract_root)
        cand = extract_root / "code"
        return cand if (cand.exists() and cand.is_dir()) else extract_root
    raise FileNotFoundError(
        f"Neither code/ nor code.zip under {DATA_DIR}. "
        f"Contents: {[p.name for p in DATA_DIR.iterdir()]}"
    )


RO_CODE_DIR = _resolve_code_dir()
print(f"Resolved code dir: {RO_CODE_DIR}")
if not CODE_DIR.exists():
    shutil.copytree(RO_CODE_DIR, CODE_DIR)
sys.path.insert(0, str(CODE_DIR))
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "oof_predictions").mkdir(parents=True, exist_ok=True)

# Stage writable data dir (config.py needs train+sample_submission+test present).
import pandas as pd  # noqa: E402
DATA_LOCAL = OUT_DIR / "data"
DATA_LOCAL.mkdir(parents=True, exist_ok=True)
for fn in ["train.csv", "test_new.csv"]:
    dst = DATA_LOCAL / fn
    if not dst.exists():
        shutil.copy(DATA_DIR / fn, dst)
ss = DATA_LOCAL / "sample_submission.csv"
if not ss.exists():
    _tdf = pd.read_csv(DATA_LOCAL / "test_new.csv")
    _uids = _tdf["rally_uid"].drop_duplicates().tolist()
    pd.DataFrame({"rally_uid": _uids, "actionId": 0, "pointId": 0,
                  "serverGetPoint": 0}).to_csv(ss, index=False)
os.environ["PINGPONG_DATA_DIR"] = str(DATA_LOCAL)
print(f"PINGPONG_DATA_DIR -> {DATA_LOCAL}  files: {[p.name for p in DATA_LOCAL.iterdir()]}")

import io
with io.open(CODE_DIR / "train_v14.py", encoding="utf-8") as f:
    src = f.read()
assert "--r203-focal" in src, "train_v14.py missing --r203-focal flag"
assert (CODE_DIR / "r203_focal_obj.py").exists(), "r203_focal_obj.py missing"
print("OK: R-203 code confirmed in dataset")

env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"] = str(DATA_DIR / "test_new.csv")

cmd = [
    "python", "-u", str(CODE_DIR / "train_v14.py"),
    "--folds", "5", "--seed", "42",
    "--r203-focal",
    "--tag", "v14_r203_full",
    "--test-path", str(DATA_DIR / "test_new.csv"),
]
print(f"Cmd: {' '.join(cmd)}")
t0 = time.time()
log_path = OUT_DIR / "logs" / "v14_r203_full.log"
with open(log_path, "w") as f:
    proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
print(f"exit={proc.returncode}  elapsed={(time.time()-t0)/60:.1f} min")
with io.open(log_path, encoding="utf-8", errors="replace") as f:
    tail = f.readlines()[-30:]
print("".join(tail))
assert proc.returncode == 0

# Confirm OOF + test arrays saved (these are the deliverable for the blend swap)
import numpy as np
oof_dir = OUT_DIR / "oof_predictions"
for name in ["v14_r203_full_oof_act.npy", "v14_r203_full_oof_pt.npy",
             "v14_r203_full_oof_srv.npy", "v14_r203_full_test_act.npy",
             "v14_r203_full_test_pt.npy", "v14_r203_full_test_srv.npy",
             "v14_r203_full_test_rally_uid.npy"]:
    p = oof_dir / name
    print(f"  {'OK' if p.exists() else 'MISSING'}  {name}"
          + (f"  shape={np.load(p).shape}" if p.exists() else ""))
print("DONE — pull oof_predictions/v14_r203_full_*.npy for the R-067cr swap candidate")
