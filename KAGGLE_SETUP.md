# Kaggle Setup — AICUP 2026 Parallel Training

Status: 2026-05-21 — bootstrapped while local R-029b trains.

**Rule reminder**: AICUP data + code is **competition-restricted** — sharing
across teams is banned. Everything on Kaggle goes **PRIVATE**.

---

## Step 1 — API token (1 minute)

1. Sign into https://www.kaggle.com
2. Open https://www.kaggle.com/settings
3. Scroll to **API** section → click **Create New Token**
4. A `kaggle.json` file downloads to your `Downloads` folder

The file looks like: `{"username":"jabir95tsai","key":"<32-char-hex>"}`

---

## Step 2 — Install local CLI + place token (PowerShell)

Open PowerShell in this project root and run:

```powershell
# Create config directory
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle" | Out-Null

# Move the token from Downloads
Move-Item -Force "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"

# Install CLI
pip install --user kaggle

# Verify (should list competitions you've joined)
kaggle competitions list 2>&1 | Select-Object -First 5
```

If the verify step prints competitions you can see, you're in.

---

## Step 3 — Push AICUP data as a PRIVATE Kaggle dataset

We package `data/train.csv`, `data/test.csv` (legal old test labels),
`data/test_new.csv`, plus the R-034 OOF arrays needed for blending. Run:

```powershell
# From project root
python -u src/kaggle_init_dataset.py --create
```

(Script written below; on first run it creates a private dataset
`jabir95tsai/aicup2026-pingpong-private`. Subsequent runs use `--update`
to push new file versions.)

---

## Step 4 — Create your first Kaggle Notebook (web UI)

1. Go to https://www.kaggle.com/code → **+ New Notebook**
2. In the right panel: **Add Data** → **Your Datasets** → attach
   `aicup2026-pingpong-private`
3. Top-right: **Notebook Settings** → set:
   - **Accelerator**: GPU T4 x2 (free, ~30 hrs/week) if doing AutoGluon or NN; OR
   - **None / CPU** (free, unlimited 12-hr sessions) for LightGBM/XGBoost runs
   - **Internet**: ON (needed for pip-install autogluon etc.)
4. **File** → **Upload** → pick `notebooks/kaggle_v15feat_b_starter.ipynb`
   (we'll create this below)
5. **Save & Run All** → wait

Outputs land in `/kaggle/working/` and you download them after the run via:
```powershell
kaggle kernels output <your-username>/<notebook-slug> -p kaggle_outputs/
```

---

## Step 5 — Pull OOF/test arrays back into our pipeline

After a Kaggle notebook produces `v14_seed2_v15feat_b_oof_act.npy` etc.,
download them and drop into `oof_predictions/` to slot into our blend-swap
analyzer:

```powershell
# Replace <slug> with the actual notebook short-id from URL
kaggle kernels output jabir95tsai/<slug> -p oof_predictions/

# Re-run blend audit
python -u src/audit_all_parked_components.py --n-samples 200
```

---

## What's safe to put on Kaggle

| Item | OK? |
|---|---|
| `data/train.csv`, `data/test.csv`, `data/test_new.csv` | ✅ as PRIVATE dataset |
| Our `src/*.py` training code | ✅ as PRIVATE notebook |
| Pretrained model `.pt`/`.pkl` checkpoints | ✅ as PRIVATE dataset |
| OOF arrays we generated | ✅ as PRIVATE dataset |
| Our LB submissions | ✅ private; do NOT publish |
| Teammate package | ❌ never upload — code-sharing ban |
| AICUP discussion screenshots | ❌ never upload |

## What NOT to do

- ❌ Set the dataset/notebook to **Public** — that constitutes external
  sharing and triggers AICUP disqualification under "嚴禁反向比對" rules
- ❌ Use Kaggle for *score* validation (still LB only — Kaggle CV is OOF)
- ❌ Bypass the daily 3-submission cap by uploading via Kaggle automation

---

## Tier of work, choose by accelerator

| Workload | Local | Kaggle CPU | Kaggle GPU T4 |
|---|---|---|---|
| v15feat_b (LightGBM 5-fold) | 3.3 hr (in progress) | ~1.5 hr | not needed |
| AutoGluon best_quality | not run | ~2 hr | ~50 min |
| Transformer v11plus retrain | slow | impractical | ~30 min |
| Meta-stack with LR head | <10 min | <5 min | <5 min |
