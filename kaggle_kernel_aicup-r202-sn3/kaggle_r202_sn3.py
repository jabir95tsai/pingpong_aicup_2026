# # R-202 — Long-rally SN>=3 specialist (V11 full 5-fold, Kaggle GPU)
#
# Trains a V11 transformer ONLY on target shots with strikeNumber >= 3
# (longer rallies), via a per-fold TRAIN-index filter. VAL/OOF stays FULL so
# OOF remains globally aligned with R-067cr for the downstream blend, and TEST
# inference runs on ALL rows (no SN gating at inference — that would be LB-toxic).
#
# The trainer code is RUNTIME-PATCHED here to add --min-sn-train, so we do NOT
# re-push the Kaggle dataset (which would re-upload teammate parquets — barred
# by the project hard rules).

import os
import sys
import time
import subprocess
import shutil
import io
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
for d in ["oof_predictions", "models", "submissions", "logs"]:
    (OUT_DIR / d).mkdir(parents=True, exist_ok=True)
os.environ["PINGPONG_DATA_DIR"] = str(DATA_DIR)

# ---- Runtime-patch train_v11_transformer.py to add --min-sn-train (R-202) ----
tv = CODE_DIR / "train_v11_transformer.py"
src = io.open(tv, encoding="utf-8").read()

if "--min-sn-train" not in src:
    ARG_ANCHOR = '                             "Default -1 = all folds.")'
    ARG_ADD = ARG_ANCHOR + '''
    parser.add_argument("--min-sn-train", type=int, default=0,
                        help="(R-202) train only on target shots strikeNumber >= this; "
                             "filters per-fold TRAIN indices only, VAL/OOF stays full.")'''
    assert ARG_ANCHOR in src, "PATCH FAIL: --fold-only arg anchor not found"
    src = src.replace(ARG_ANCHOR, ARG_ADD, 1)

    FILT_ANCHOR = "        # P6: append all aug indices to this fold's training set. Aug rows are"
    FILT_ADD = '''        if args.min_sn_train > 0:
            _nb = len(tr_idx)
            tr_idx = tr_idx[nsn_all[tr_idx] >= args.min_sn_train]
            print(f"  [R-202 min-sn-train={args.min_sn_train}] train rows {_nb} -> {len(tr_idx)}")

''' + FILT_ANCHOR
    assert FILT_ANCHOR in src, "PATCH FAIL: P6 append anchor not found"
    src = src.replace(FILT_ANCHOR, FILT_ADD, 1)

    io.open(tv, "w", encoding="utf-8").write(src)

patched = io.open(tv, encoding="utf-8").read()
assert "--min-sn-train" in patched and "min_sn_train > 0" in patched, "PATCH VERIFY FAIL"
print("OK runtime-patched train_v11_transformer.py with --min-sn-train")

import torch
print(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))
else:
    print("  WARNING: no GPU allocated — 5-fold x 80ep may exceed the 12h limit.")

cmd = [
    "python", "-u", str(CODE_DIR / "train_v11_transformer.py"),
    "--folds", "5",
    "--epochs", "80",
    "--tag", "v11_r202_sn3",
    "--min-sn-train", "3",
    "--save-checkpoint",
    "--test-path", str(DATA_DIR / "test_new.csv"),
]
print("Cmd:", " ".join(cmd))
env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"] = str(DATA_DIR / "test_new.csv")

t0 = time.time()
log_path = OUT_DIR / "logs" / "r202_v11_sn3.log"
with open(log_path, "w") as f:
    proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
elapsed = (time.time() - t0) / 60
print(f"\nElapsed: {elapsed:.1f} min  exit={proc.returncode}")

tail = io.open(log_path, encoding="utf-8").read()[-4000:]
print("---- log tail ----")
print(tail)
assert proc.returncode == 0, f"trainer exited {proc.returncode}"

import numpy as np
print("\n---- output artifacts ----")
ok = True
for nm in ["oof_act", "oof_pt", "oof_srv", "oof_mask", "oof_nsn",
           "test_act", "test_pt", "test_srv", "test_rally_uid"]:
    p = OUT_DIR / "oof_predictions" / f"v11_r202_sn3_{nm}.npy"
    exists = p.exists()
    ok = ok and exists
    print(f"  {nm:16s} {'OK' if exists else 'MISSING'}")
assert ok, "some OOF/test arrays missing"
print("\nR-202 specialist training complete — OOF + test arrays saved.")
