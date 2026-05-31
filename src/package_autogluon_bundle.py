"""Package the AutoGluon meta-feature bundle for teammates.

Builds turnkey train (OOF) + test meta matrices from the 5 LB-best, cleanly
69712-aligned action/point components, plus copies raw arrays (incl. v22 SGP)
and the manifest. Output -> ~/Downloads/aicup_autogluon_bundle/.

Meta-matrix layout (one row per sample):
  train_meta.parquet : 69712 rows
    features: {tag}__act0..14 (15) + {tag}__pt0..9 (10) + {tag}__srv (1)
              for each of 5 components = 130 cols
    labels  : y_action (0-14), y_point (0-9), y_server (0/1)
  test_meta.parquet  : 1845 rows
    same 130 feature cols + rally_uid
"""
from __future__ import annotations
import os, shutil
import numpy as np
import pandas as pd

OOF = "oof_predictions"
N_REF = 69712

ACTPT = ["v11_aug_oldtest", "v11plus", "v13_oldtest", "v14_seed2_v15feat_a", "v16_avg3"]
V22 = "v22_causal_lm_v4_full"

OUT = os.path.join(os.path.expanduser("~"), "Downloads", "aicup_autogluon_bundle")


def load_act15(path):
    a = np.load(path)
    if a.shape[0] > N_REF:           # oldtest 72065 -> slice
        a = a[:N_REF]
    return a[:, :15]                 # eval classes 0-14


def load_pt(path):
    a = np.load(path)
    if a.shape[0] > N_REF:
        a = a[:N_REF]
    return a[:, :10]


def load_srv(path):
    a = np.load(path)
    if a.shape[0] > N_REF:
        a = a[:N_REF]
    return a


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "raw_arrays"), exist_ok=True)

    # ---- labels (from a 72065 oldtest tag, sliced; first N_REF verified canonical) ----
    y_act = np.load(f"{OOF}/v11_aug_oldtest_oof_y_act.npy")[:N_REF]
    y_pt  = np.load(f"{OOF}/v11_aug_oldtest_oof_y_pt.npy")[:N_REF]
    y_srv = np.load(f"{OOF}/v11_aug_oldtest_oof_y_srv.npy")[:N_REF]
    test_uid = np.load(f"{OOF}/v11_aug_oldtest_test_rally_uid.npy")

    train_cols, test_cols = {}, {}
    for tag in ACTPT:
        oa = load_act15(f"{OOF}/{tag}_oof_act.npy")
        op = load_pt(f"{OOF}/{tag}_oof_pt.npy")
        os_ = load_srv(f"{OOF}/{tag}_oof_srv.npy")
        ta = np.load(f"{OOF}/{tag}_test_act.npy")[:, :15]
        tp = np.load(f"{OOF}/{tag}_test_pt.npy")[:, :10]
        ts = np.load(f"{OOF}/{tag}_test_srv.npy")
        assert oa.shape[0] == N_REF and ta.shape[0] == len(test_uid), f"{tag} shape mismatch"
        for c in range(15):
            train_cols[f"{tag}__act{c}"] = oa[:, c]; test_cols[f"{tag}__act{c}"] = ta[:, c]
        for c in range(10):
            train_cols[f"{tag}__pt{c}"] = op[:, c];  test_cols[f"{tag}__pt{c}"] = tp[:, c]
        train_cols[f"{tag}__srv"] = os_;             test_cols[f"{tag}__srv"] = ts

    train_df = pd.DataFrame(train_cols)
    train_df["y_action"] = y_act.astype(int)
    train_df["y_point"]  = y_pt.astype(int)
    train_df["y_server"] = y_srv.astype(int)

    test_df = pd.DataFrame(test_cols)
    test_df["rally_uid"] = test_uid

    train_path = os.path.join(OUT, "train_meta.parquet")
    test_path  = os.path.join(OUT, "test_meta.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    # CSV mirrors for teammates without pyarrow handy
    train_df.to_csv(os.path.join(OUT, "train_meta.csv"), index=False)
    test_df.to_csv(os.path.join(OUT, "test_meta.csv"), index=False)

    print(f"train_meta {train_df.shape} -> {train_path}")
    print(f"test_meta  {test_df.shape} -> {test_path}")

    # ---- raw arrays (incl. v22 SGP) for full control ----
    raw_tags = ACTPT + [V22]
    suffixes = ["oof_act", "oof_pt", "oof_srv", "oof_mask",
                "test_act", "test_pt", "test_srv"]
    n_copied = 0
    for tag in raw_tags:
        for sfx in suffixes:
            src = f"{OOF}/{tag}_{sfx}.npy"
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(OUT, "raw_arrays", os.path.basename(src)))
                n_copied += 1
    for lab in ["oof_y_act", "oof_y_pt", "oof_y_srv", "test_rally_uid"]:
        src = f"{OOF}/v11_aug_oldtest_{lab}.npy"
        shutil.copy2(src, os.path.join(OUT, "raw_arrays", os.path.basename(src)))
        n_copied += 1
    print(f"copied {n_copied} raw arrays")

    # ---- manifest ----
    man = "audits/AUTOGLUON_COMPONENT_MANIFEST_2026-05-31.md"
    if os.path.exists(man):
        shutil.copy2(man, os.path.join(OUT, "COMPONENT_MANIFEST.md"))
    print("bundle dir:", OUT)


if __name__ == "__main__":
    main()
