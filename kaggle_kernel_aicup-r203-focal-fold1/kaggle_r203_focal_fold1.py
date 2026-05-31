# # R-203 — V14 focal CE + Cui CB weights, fold-1 smoke (baseline vs R-203)
#
# Runs the v14 GBM action pipeline twice on fold-1 of GroupKFold(5), seed 42:
#   1. BASELINE: standard multiclass CE + hand-tuned ACTION_CW
#   2. R-203:    focal CE (gamma=2) + Cui et al. Class-Balanced weights +
#                push/Loop boost (act 1,5,6,13 x1.5)
# Then compares fold-1 per-class action F1 (push family 5,6,13 + Loop 1) and
# writes a GO/NO-GO verdict. CPU kernel (GBM needs no GPU).
#
# Dataset: jabir95tsai/aicup2026-r203-code  (code + train.csv + test_new.csv,
#          NO teammate parquet).

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

# New private datasets mount under /kaggle/input/datasets/<owner>/<slug>/;
# older/attached ones sometimes mount at the flat /kaggle/input/<slug>/.
DATA_DIR = Path("/kaggle/input/datasets/jabir95tsai/aicup2026-r203-code")
if not DATA_DIR.exists():
    DATA_DIR = Path("/kaggle/input/aicup2026-r203-code")
OUT_DIR = Path("/kaggle/working")
CODE_DIR = OUT_DIR / "src"


def _resolve_code_dir():
    """Locate the code/ tree. The dataset may mount it as a real directory
    (code/) OR as a zip archive (code.zip, from --dir-mode zip). Handle both."""
    # Diagnostic: show what is actually mounted under /kaggle/input
    inp = Path("/kaggle/input")
    print(f"/kaggle/input exists={inp.exists()}")
    if inp.exists():
        for child in inp.iterdir():
            print(f"  mounted: {child}")
            if child.is_dir():
                for sub in list(child.iterdir())[:8]:
                    print(f"      - {sub.name}")
    # Resolve the dataset mount dynamically (slug may differ from the id).
    global DATA_DIR
    if not DATA_DIR.exists():
        if inp.exists():
            cands = [c for c in inp.iterdir() if "r203" in c.name.lower()]
            if cands:
                DATA_DIR = cands[0]
                print(f"Re-resolved DATA_DIR -> {DATA_DIR}")
    import zipfile
    search_bases = [DATA_DIR]
    if inp.exists():
        search_bases += list(inp.iterdir())
        nested = inp / "datasets"
        if nested.exists():
            for owner in nested.iterdir():
                if owner.is_dir():
                    search_bases += list(owner.iterdir())
    for base in search_bases:
        d = base / "code"
        if d.exists() and d.is_dir():
            return d
        zp = base / "code.zip"
        if zp.exists():
            extract_root = OUT_DIR / "code_unzipped"
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(extract_root)
            cand = extract_root / "code"
            return cand if (cand.exists() and cand.is_dir()) else extract_root
    raise FileNotFoundError(
        f"No code/ or code.zip found. /kaggle/input contents: "
        f"{[p.name for p in inp.iterdir()] if inp.exists() else 'MISSING'}"
    )


RO_CODE_DIR = _resolve_code_dir()
print(f"Resolved code dir: {RO_CODE_DIR}")
if not CODE_DIR.exists():
    shutil.copytree(RO_CODE_DIR, CODE_DIR)
sys.path.insert(0, str(CODE_DIR))
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "oof_predictions").mkdir(parents=True, exist_ok=True)

# config.py resolves data via PINGPONG_DATA_DIR but requires train.csv +
# sample_submission.csv + a test file all present in that dir. The dataset has
# no sample_submission.csv and /kaggle/input is read-only, so stage a writable
# data dir under /kaggle/working with all three files (sample_submission synth).
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
    print(f"Synthesized sample_submission.csv ({len(_uids)} rallies)")
os.environ["PINGPONG_DATA_DIR"] = str(DATA_LOCAL)
print(f"PINGPONG_DATA_DIR -> {DATA_LOCAL}  files: {[p.name for p in DATA_LOCAL.iterdir()]}")

# Verify the patched trainer + R-203 module are present
import io
with io.open(CODE_DIR / "train_v14.py", encoding="utf-8") as f:
    src = f.read()
assert "--r203-focal" in src, "train_v14.py missing --r203-focal flag"
assert (CODE_DIR / "r203_focal_obj.py").exists(), "r203_focal_obj.py missing"
print("OK: --r203-focal flag + r203_focal_obj.py confirmed in dataset code")

# Run the module self-tests first (fast, validates gradient correctness)
print("\n=== r203_focal_obj self-tests ===")
st = subprocess.run(["python", str(CODE_DIR / "r203_focal_obj.py")],
                    capture_output=True, text=True)
print(st.stdout[-2000:])
if st.returncode != 0:
    print("SELF-TEST STDERR:", st.stderr[-2000:])
assert st.returncode == 0, "r203_focal_obj self-tests failed"

env = os.environ.copy()
env["TRAIN_PATH"] = str(DATA_DIR / "train.csv")
env["TEST_PATH"] = str(DATA_DIR / "test_new.csv")


def run(tag, extra):
    cmd = [
        "python", "-u", str(CODE_DIR / "train_v14.py"),
        "--folds", "5", "--max-folds", "1", "--seed", "42",
        "--tag", tag,
        "--test-path", str(DATA_DIR / "test_new.csv"),
    ] + extra
    print(f"\n=== RUN {tag}: {' '.join(cmd)} ===")
    t0 = time.time()
    log_path = OUT_DIR / "logs" / f"{tag}.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"{tag} exit={proc.returncode}  elapsed={(time.time()-t0)/60:.1f} min")
    # Echo tail of log
    with io.open(log_path, encoding="utf-8", errors="replace") as f:
        tail = f.readlines()[-25:]
    print("".join(tail))
    return proc.returncode


rc_base = run("v14_baseline_fold1", [])
rc_r203 = run("v14_r203_fold1", ["--r203-focal"])

# ── Comparison verdict ──────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import f1_score

ACTION_NAMES = {0:"None",1:"Loop",2:"Cloop",3:"Smash",4:"Flip",5:"Pushfast",
                6:"Push",7:"Flick",8:"Arch",9:"Knuckle",10:"Chop_r",
                11:"ShortStop",12:"Chop",13:"Block",14:"Lob"}
PUSH_FAMILY = [5, 6, 13]
LABELS = list(range(15))
oof_dir = OUT_DIR / "oof_predictions"


def fold1_f1(tag):
    a = np.load(oof_dir / f"{tag}_oof_act.npy")
    y = np.load(oof_dir / f"{tag}_oof_y_act.npy")
    m = np.load(oof_dir / f"{tag}_oof_mask.npy")
    pred = np.argmax(a[m], axis=1)
    pc = f1_score(y[m], pred, labels=LABELS, average=None, zero_division=0)
    macro = f1_score(y[m], pred, labels=LABELS, average="macro", zero_division=0)
    return pc, macro, int(m.sum())

print("\n" + "=" * 60)
print("R-203 SMOKE VERDICT")
print("=" * 60)
if rc_base != 0 or rc_r203 != 0:
    print(f"INCOMPLETE: baseline rc={rc_base} r203 rc={rc_r203}")
else:
    base_pc, base_macro, n = fold1_f1("v14_baseline_fold1")
    r203_pc, r203_macro, _ = fold1_f1("v14_r203_fold1")
    print(f"{'cls':>3} {'name':<10} {'base':>8} {'r203':>8} {'delta':>8}")
    for c in LABELS:
        d = r203_pc[c] - base_pc[c]
        star = " *" if c in (PUSH_FAMILY + [1]) else ""
        print(f"{c:>3} {ACTION_NAMES[c]:<10} {base_pc[c]:>8.4f} {r203_pc[c]:>8.4f} {d:>+8.4f}{star}")
    push_delta = float(np.mean([r203_pc[c] - base_pc[c] for c in PUSH_FAMILY]))
    macro_delta = float(r203_macro - base_macro)
    est_ov_delta = 0.4 * macro_delta
    print(f"\npush-family (5,6,13) mean F1 delta: {push_delta:+.4f}  (crit >= +0.005)")
    print(f"Loop (act1) F1 delta:               {r203_pc[1]-base_pc[1]:+.4f}")
    print(f"action macro delta:                 {macro_delta:+.4f}")
    print(f"est OV delta (0.4*macro):           {est_ov_delta:+.4f}  (crit >= +0.003)")
    print(f"baseline macro={base_macro:.4f}  r203 macro={r203_macro:.4f}  n={n}")
    smoke_pass = (est_ov_delta >= 0.003) and (push_delta >= 0.005)
    print(f"\nRESULT: {'GO (smoke PASS)' if smoke_pass else 'NO-GO (smoke FAIL)'}")
    # Persist verdict as text artifact
    with open(OUT_DIR / "R203_smoke_verdict.txt", "w") as f:
        f.write(f"push_delta={push_delta:+.4f}\nmacro_delta={macro_delta:+.4f}\n"
                f"est_ov_delta={est_ov_delta:+.4f}\nsmoke_pass={smoke_pass}\n"
                f"baseline_macro={base_macro:.4f}\nr203_macro={r203_macro:.4f}\n")
print("DONE")
