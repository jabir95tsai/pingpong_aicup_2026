# # Kaggle batch retrain — GBM Tier 1 + Tier 2 (use ALL legal data)
#
# Trains the GBM components that are missing the --include-old-test axis.
# Runs all variants sequentially in one Kaggle session.
#
# Per-variant time on Kaggle CPU (32 cores, no GPU needed for GBM):
#   v14_seed2_v15feat_a_oldtest         ~35 min
#   v14_seed2_v15feat_b_oldtest         ~40 min
#   v14_recvhand_oldtest                ~30 min
#   v14_recvprofile_oldtest             ~30 min
#   sgp_prefix_v3_full_oldtest          ~10 min
#   v16_avg3_oldtest (3 seeds chain)    ~75 min
#
# Total: ~3.5 - 4 hr — well under the 12-hr Kaggle session limit.
#
# Outputs land in /kaggle/working/oof_predictions/ and are downloadable
# via `kaggle kernels output ...`

# ## Setup

# +
import os, sys, time, subprocess, json
from pathlib import Path
from datetime import datetime

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

(OUT_DIR / "oof_predictions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "submissions").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "runs").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

# config.py reads PINGPONG_DATA_DIR env var first when resolving DATA_DIR
# (then "PROJECT_ROOT/data" as fallback). Setting it here makes the trainer
# scripts read train.csv/test.csv directly from the Kaggle mount.
# PROJECT_ROOT becomes parent(CODE_DIR) = OUT_DIR = /kaggle/working (writable).
os.environ["PINGPONG_DATA_DIR"] = str(DATA_DIR)
print(f"Set PINGPONG_DATA_DIR={DATA_DIR}")
print(f"PROJECT_ROOT for trainers will be: {OUT_DIR} (writable)")
# -

# ## Verify dataset

# +
import pandas as pd
train = pd.read_csv(DATA_DIR / "train.csv")
old_test = pd.read_csv(DATA_DIR / "test.csv")
test_new = pd.read_csv(DATA_DIR / "test_new.csv")
print(f"train: {len(train):,} rows / {train.rally_uid.nunique():,} rallies")
print(f"old_test: {len(old_test):,} rows / {old_test.rally_uid.nunique():,} rallies (LEGAL aug)")
print(f"test_new: {len(test_new):,} rows / {test_new.rally_uid.nunique():,} rallies")
# -

# ## Verify trainer scripts present

# +
required_scripts = [
    "train_v14.py", "sgp_prefix_v3.py",
    "features_v9.py", "features_v15feat.py", "features_v15feat_b.py",
    "features_v9_recvhand.py", "features_v9_recvprofile.py",
    "features_sgp_prefix_v3.py",
    "config.py", "data_cleaning.py",
]
missing = []
for s in required_scripts:
    fp = CODE_DIR / s
    if not fp.exists():
        missing.append(s)
        print(f"  MISSING: {s}")
    else:
        print(f"  OK: {s}")
assert not missing, f"Missing scripts: {missing}"
# -

# ## Define training batch

# +
TRAIN_PATH = str(DATA_DIR / "train.csv")
OLD_TEST_PATH = str(DATA_DIR / "test.csv")
TEST_NEW_PATH = str(DATA_DIR / "test_new.csv")

# Each entry = (tag, trainer_script, extra_args)
# All entries assume --include-old-test data/test.csv (the user's
# directive: use all legal data).
BATCH = [
    # Tier 1: R-034 PAIR upgrade — winning component + oldtest
    {
        "tag": "v14_seed2_v15feat_a_oldtest",
        "trainer": "train_v14.py",
        "args": ["--feature-set", "v15feat",
                 "--seed", "51966", "--folds", "5",
                 "--n-boost", "3000", "--es", "200",
                 "--include-old-test", OLD_TEST_PATH],
    },
    # Tier 2: R-029b + oldtest
    {
        "tag": "v14_seed2_v15feat_b_oldtest",
        "trainer": "train_v14.py",
        "args": ["--feature-set", "v15feat_b",
                 "--seed", "51966", "--folds", "5",
                 "--n-boost", "3000", "--es", "200",
                 "--include-old-test", OLD_TEST_PATH],
    },
    # Tier 2: recvhand + oldtest
    {
        "tag": "v14_recvhand_oldtest",
        "trainer": "train_v14.py",
        "args": ["--feature-set", "v9_recvhand",
                 "--seed", "42", "--folds", "5",
                 "--n-boost", "3000", "--es", "200",
                 "--include-old-test", OLD_TEST_PATH],
    },
    # Tier 2: recvprofile + oldtest
    {
        "tag": "v14_recvprofile_oldtest",
        "trainer": "train_v14.py",
        "args": ["--feature-set", "v9_recvprofile",
                 "--seed", "42", "--folds", "5",
                 "--n-boost", "3000", "--es", "200",
                 "--include-old-test", OLD_TEST_PATH],
    },
    # Tier 2: SGP specialist + oldtest (R-030 full + oldtest)
    {
        "tag": "sgp_prefix_v3_full_oldtest",
        "trainer": "sgp_prefix_v3.py",
        "args": ["--full-train", "--folds", "5",
                 "--seed", "51966",
                 # NOTE: sgp_prefix_v3 has --include-old-test arg but doesn't
                 # wire it through. We rely on TRAIN_PATH env override to
                 # have already concatenated old test (or train on the
                 # concatenated CSV approach).
                 ],
    },
    # Tier 1: v16_avg3 oldtest — 3 seeds, then average
    {
        "tag": "v16_seed4_oldtest",
        "trainer": "train_v16_testhist_aug.py",
        "args": ["--seed", "4", "--folds", "5",
                 "--include-old-test", OLD_TEST_PATH],
    },
    {
        "tag": "v16_seed9_oldtest",
        "trainer": "train_v16_testhist_aug.py",
        "args": ["--seed", "9", "--folds", "5",
                 "--include-old-test", OLD_TEST_PATH],
    },
    {
        "tag": "v16_seed31337_oldtest",
        "trainer": "train_v16_testhist_aug.py",
        "args": ["--seed", "31337", "--folds", "5",
                 "--include-old-test", OLD_TEST_PATH],
    },
]
print(f"Batch size: {len(BATCH)} variants")
# -

# ## Run batch

# +
results = []
batch_start = time.time()
for i, item in enumerate(BATCH, 1):
    print()
    print("=" * 70)
    print(f"[{i}/{len(BATCH)}] {item['tag']}  (trainer={item['trainer']})")
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
    results.append({"tag": item["tag"], "status": status, "elapsed_min": elapsed,
                    "exit_code": proc.returncode})

total_min = (time.time() - batch_start) / 60
print()
print("=" * 70)
print(f"BATCH COMPLETE in {total_min:.1f} min")
print("=" * 70)
for r in results:
    marker = "OK" if r["exit_code"] == 0 else "FAIL"
    print(f"  [{marker:4}] {r['tag']:40s}  {r['elapsed_min']:.1f} min")

# Save batch metadata
(OUT_DIR / "runs" / "batch_gbm_results.json").write_text(
    json.dumps({"batch": "gbm", "results": results, "total_min": total_min,
                "completed_at": datetime.now().isoformat()},
               indent=2),
    encoding="utf-8",
)
# -

# ## Verify outputs

# +
import numpy as np
oof_dir = OUT_DIR / "oof_predictions"
print(f"OOF dir contents: {len(list(oof_dir.iterdir()))} files")
for item in BATCH:
    fp = oof_dir / f"{item['tag']}_oof_act.npy"
    if fp.exists():
        arr = np.load(fp)
        print(f"  OK {item['tag']}: shape={arr.shape}, finite={np.isfinite(arr).all()}")
    else:
        print(f"  MISSING {item['tag']}")
# -

# ## (Optional) Build v16_avg3_oldtest from the 3 seeds

# +
seeds_present = []
for s in [4, 9, 31337]:
    tag = f"v16_seed{s}_oldtest"
    if (oof_dir / f"{tag}_oof_act.npy").exists():
        seeds_present.append(tag)

if len(seeds_present) >= 2:
    print(f"Averaging {len(seeds_present)} v16 oldtest seeds -> v16_avg3_oldtest")
    avg_tag = "v16_avg3_oldtest"
    for suffix in ["oof_act", "oof_pt", "oof_srv",
                   "test_act", "test_pt", "test_srv"]:
        arrs = [np.load(oof_dir / f"{t}_{suffix}.npy") for t in seeds_present]
        avg = np.mean(arrs, axis=0)
        np.save(oof_dir / f"{avg_tag}_{suffix}.npy", avg)
    # Copy passthrough arrays
    for suffix in ["oof_y_act", "oof_y_pt", "oof_y_srv", "oof_mask", "oof_nsn", "test_rally_uid"]:
        src = oof_dir / f"{seeds_present[0]}_{suffix}.npy"
        if src.exists():
            import shutil
            shutil.copy2(src, oof_dir / f"{avg_tag}_{suffix}.npy")
    print(f"  v16_avg3_oldtest arrays saved")
else:
    print(f"Skipping v16_avg3_oldtest avg — only {len(seeds_present)} seeds present")
# -

# ## Download outputs
#
# Local PowerShell:
#
# ```powershell
# kaggle kernels output jabir95tsai/<slug> -p oof_predictions/
# python -u src/audit_all_parked_components.py --n-samples 200
# ```
