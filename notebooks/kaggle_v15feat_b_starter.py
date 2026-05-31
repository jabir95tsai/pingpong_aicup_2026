# # AICUP 2026 — v14_seed2_v15feat_b (R-029b) Kaggle starter
#
# Trains v14 with v15feat_b feature set (v15feat + 33 transition priors).
# Same configuration as local R-034 build except:
#   - Dataset: from /kaggle/input/aicup2026-pingpong-private/
#   - Output:  /kaggle/working/  (downloadable via `kaggle kernels output`)
#
# **Run before**: clone our src/ files into /kaggle/working/src by adding them
# as a code-only Kaggle dataset, OR paste the contents of features_v15feat.py
# and features_v15feat_b.py into cells here. The notebook itself stays
# under your account — don't share.
#
# Compute: enable GPU only if you also want to run AutoGluon or transformer
# downstream. For LightGBM/XGBoost on this dataset, CPU is faster.

# ## 1. Environment

# +
import os, sys, time, shutil
from pathlib import Path

IN_KAGGLE = Path("/kaggle").exists()
if IN_KAGGLE:
    DATA_DIR = Path("/kaggle/input/aicup2026-pingpong-private")
    OUT_DIR = Path("/kaggle/working")
    SRC_DIR = Path("/kaggle/working/src")
else:
    # Allow local execution for parity testing
    DATA_DIR = Path("data")
    OUT_DIR = Path("kaggle_outputs")
    OUT_DIR.mkdir(exist_ok=True)
    SRC_DIR = Path("src")

print(f"DATA_DIR={DATA_DIR}, OUT_DIR={OUT_DIR}")
sys.path.insert(0, str(SRC_DIR))
# -

# ## 2. Install pinned deps (only on Kaggle)

# !pip install -q lightgbm==4.5.0 xgboost==2.1.1

# ## 3. Sanity check data attached

# +
import pandas as pd

train = pd.read_csv(DATA_DIR / "train.csv")
test_new = pd.read_csv(DATA_DIR / "test_new.csv")
print(f"train: {len(train):,} rows, {train.rally_uid.nunique():,} rallies, {train.match.nunique()} matches")
print(f"test_new: {len(test_new):,} rows, {test_new.rally_uid.nunique():,} rallies")
print("expected: train ~84707 / 14995 / 216; test_new ~5668 / 1845")
# -

# ## 4. Drive train_v14 with --feature-set v15feat_b
#
# On Kaggle we run the script directly to avoid copy-pasting the 1200-line train.

# +
import subprocess

cmd = [
    "python", "-u", str(SRC_DIR / "train_v14.py"),
    "--feature-set", "v15feat_b",
    "--tag", "v14_seed2_v15feat_b",
    "--seed", "51966",
    "--folds", "5",
    "--n-boost", "3000",
    "--es", "200",
    "--test-path", str(DATA_DIR / "test_new.csv"),
]

# Set env to point train_v14 at the Kaggle-mounted data
env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"]  = str(DATA_DIR / "test_new.csv")
env["OOF_DIR"]    = str(OUT_DIR / "oof_predictions")
env["SUBMISSION_DIR"] = str(OUT_DIR / "submissions")

(OUT_DIR / "oof_predictions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "submissions").mkdir(parents=True, exist_ok=True)

t0 = time.time()
proc = subprocess.run(cmd, env=env, capture_output=False)
elapsed = time.time() - t0
print(f"Exit code: {proc.returncode}  elapsed: {elapsed/60:.1f} min")
assert proc.returncode == 0, "training failed"
# -

# ## 5. Verify outputs

# +
import numpy as np

tag = "v14_seed2_v15feat_b"
oof_dir = OUT_DIR / "oof_predictions"
for suffix in ["oof_act", "oof_pt", "oof_srv", "test_act", "test_pt", "test_srv"]:
    fp = oof_dir / f"{tag}_{suffix}.npy"
    a = np.load(fp)
    print(f"{fp.name}: shape={a.shape}, dtype={a.dtype}, finite={np.isfinite(a).all()}")

# Submission CSV check
sub_fp = OUT_DIR / "submissions" / f"submission_{tag}.csv"
sub = pd.read_csv(sub_fp)
print(f"\nSubmission: {len(sub)} rows / {sub.rally_uid.nunique()} unique rallies")
print(sub.head())
# -

# ## 6. Compute OV against R-034 baseline (sanity)
#
# Loads R-034's OOF + builds the blend baseline + measures whether
# v15feat_b component delivers OOF lift in the v14_seed2_v15feat_a slot.

# +
import numpy as np
from sklearn.metrics import roc_auc_score

R034_TAGS = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]
N_ACTION = 19
N_POINT = 10
ACTION_EVAL = list(range(15))
POINT_EVAL = list(range(10))

def pad19(a):
    if a.shape[1] >= 19: return a.astype(np.float32, copy=False)
    out = np.zeros((a.shape[0], 19), dtype=np.float32); out[:, :a.shape[1]] = a; return out

def f1_macro(y, p, labels, n):
    cm = np.bincount(y.astype(np.int64)*n + p.astype(np.int64), minlength=n*n).reshape(n, n)
    cs = cm.sum(0); rs = cm.sum(1); d = np.diag(cm)
    f1s = []
    for c in labels:
        tp = d[c]; fp = cs[c]-tp; fn = rs[c]-tp
        f1s.append(0.0 if (2*tp+fp+fn)<=0 else (2*tp)/(2*tp+fp+fn))
    return float(np.mean(f1s))

# load each
def load(tag, oof_root):
    return {
        "a": pad19(np.load(oof_root / f"{tag}_oof_act.npy")),
        "p": np.load(oof_root / f"{tag}_oof_pt.npy").astype(np.float32),
        "s": np.load(oof_root / f"{tag}_oof_srv.npy").astype(np.float32),
    }

# Reference comes from the Kaggle dataset for R-034 components
ref_root = DATA_DIR / "oof"
y_a = np.load(ref_root / "v11_aug_oldtest_oof_y_act.npy")[:69712]
y_p = np.load(ref_root / "v11_aug_oldtest_oof_y_pt.npy")[:69712]
y_s = np.load(ref_root / "v11_aug_oldtest_oof_y_srv.npy")[:69712]

# Quick equal-weight blend with the new component swapped in
slot = "v14_seed2_v15feat_a"
comps = {t: load(t, ref_root if t != slot else oof_dir) for t in R034_TAGS}

# Handle 72065-row oldtest slice
for t in comps:
    for k in "aps":
        if comps[t][k].shape[0] != 69712:
            comps[t][k] = comps[t][k][:69712]

w = 1/5
ba = sum(w*comps[t]["a"] for t in R034_TAGS)
bp = sum(w*comps[t]["p"] for t in R034_TAGS)
bs = sum(w*comps[t]["s"] for t in R034_TAGS)

print(f"Equal-weight blend with v15feat_b swap into {slot}:")
print(f"  F1_a = {f1_macro(y_a, ba.argmax(1), ACTION_EVAL, N_ACTION):.4f}")
print(f"  F1_p = {f1_macro(y_p, bp.argmax(1), POINT_EVAL, N_POINT):.4f}")
print(f"  AUC  = {roc_auc_score(y_s, bs):.4f}")
# -

# ## 7. Ready to download
#
# In your local PowerShell:
#
# ```powershell
# kaggle kernels output jabir95tsai/<this-notebook-slug> -p oof_predictions/
# ```
#
# Then locally:
#
# ```powershell
# python -u src/audit_all_parked_components.py --n-samples 200
# ```
#
# v14_seed2_v15feat_b will now appear in the blend-swap audit as a new
# candidate. If dOV ≥ -0.002 vs R-034 PAIR, build it as an R-043 candidate
# CSV via src/build_low_risk_submissions.py.
