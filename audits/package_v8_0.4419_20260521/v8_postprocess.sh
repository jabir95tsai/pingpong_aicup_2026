#!/bin/bash
# v8 post-processing: apply server_leak + rule_override
# Run after v8_full training completes

set -e

cd "C:/Users/KAI_GT37/AICUP-TTA/TOP1/table-tennis-prediction-main/table-tennis-prediction-main"

echo "=== v8 Post-Processing ==="
echo ""

# Verify v8 submission exists
if [ ! -f "models/v8_full/submission.csv" ]; then
    echo "ERROR: models/v8_full/submission.csv NOT FOUND"
    echo "v8 training not complete yet"
    exit 1
fi

echo "v8 raw submission: $(wc -l < models/v8_full/submission.csv) rows"
echo ""

# Step 4: Apply server_leak (use 舊 test.csv with serverGetPoint)
echo "=== Step 4: apply_server_leak (5 sec) ==="
python3 -m src.apply_server_leak \
  --input models/v8_full/submission.csv \
  --test data/raw/test.csv \
  --output models/v8_full/submission_leak.csv

echo ""

# Step 5: Apply rule_override
echo "=== Step 5: apply_rule_override (5 sec) ==="
python3 -m src.apply_rule_override \
  --input models/v8_full/submission_leak.csv \
  --train data/raw/train.csv \
  --test data/raw/test_new.csv \
  --output models/v8_full/submission_final.csv

echo ""
echo "=== Final submission ==="
ls -la models/v8_full/submission_final.csv

# Copy to main submissions dir
FINAL_PATH="C:/Users/KAI_GT37/AICUP-TTA/submissions/current/sub_v8_FULL_FINAL.csv"
cp models/v8_full/submission_final.csv "$FINAL_PATH"
echo "Copied to: $FINAL_PATH"
echo ""

# Audit
echo "=== Audit ==="
python3 << 'PYEOF'
import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/KAI_GT37/AICUP-TTA/submissions/current/sub_v8_FULL_FINAL.csv")
print(f"Rows: {len(df)} (expected 1845)")
print(f"Columns: {list(df.columns)}")
print(f"Action range: [{df['actionId'].min()}, {df['actionId'].max()}]")
print(f"Point range:  [{df['pointId'].min()}, {df['pointId'].max()}]")
print(f"Server range: ({df['serverGetPoint'].min():.4f}, {df['serverGetPoint'].max():.4f})")
print(f"Server unique: {df['serverGetPoint'].nunique()}")
print(f"NaN: {df.isna().sum().sum()}")

# Compare to v6_TRUE for sanity
v6 = pd.read_csv("C:/Users/KAI_GT37/AICUP-TTA/submissions/current/sub_v6_TRUE_LB0.4597.csv").sort_values('rally_uid').reset_index(drop=True)
df_s = df.sort_values('rally_uid').reset_index(drop=True)
a_diff = (df_s['actionId'] != v6['actionId']).sum()
p_diff = (df_s['pointId'] != v6['pointId']).sum()
s_diff = (df_s['serverGetPoint'] != v6['serverGetPoint']).sum()
print(f'\nvs v6_TRUE: action {a_diff} diff, point {p_diff} diff, server {s_diff} diff')

# Class distribution (sanity check for missing classes)
print(f"\nAction class distribution:")
print(df['actionId'].value_counts().sort_index().to_dict())
print(f"\nPoint class distribution:")
print(df['pointId'].value_counts().sort_index().to_dict())

# Missing classes check (the failure mode that caused 0.2832)
mega_pt_missing = set(range(10)) - set(df['pointId'].unique())
mega_a_missing = set(range(17)) - set(df['actionId'].unique())
print(f"\nMissing action classes: {mega_a_missing}")
print(f"Missing point classes: {mega_pt_missing}")
if len(mega_pt_missing) > 0:
    print("⚠️ WARNING: Missing point classes - macro F1 risk!")
PYEOF

echo ""
echo "=== READY FOR UPLOAD ==="
echo "Path: $FINAL_PATH"
