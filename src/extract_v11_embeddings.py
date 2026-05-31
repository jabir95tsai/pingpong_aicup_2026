"""R-082 Phase 2 Step 2 — V11 embedding extraction from saved fold checkpoints.

OOF-safe protocol:
  - For each fold f, load `models/v11_fold{f}.pt`
  - Run inference on fold-f's VAL rows ONLY → those embeddings come strictly
    from the model that did NOT train on those rows
  - Run inference on the full TEST set with all 5 fold models → average across folds
  - Save `oof_predictions/v11_emb_{last|pool}_{oof|test}.npy`

The embedding shapes:
  - oof_emb_last:  (69712, d=192)   action+point head input embedding
  - oof_emb_pool:  (69712, d=192)   server head input embedding (mean-pool)
  - test_emb_last: (1845, d=192)    averaged across 5 fold models
  - test_emb_pool: (1845, d=192)

USAGE:
    python -u src/extract_v11_embeddings.py --tag v11
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT, TRAIN_PATH, TEST_PATH, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data
from train_v11_transformer import (
    build_samples, RallyDataset, RallyTransformer,
    N_ACTION_TRAIN, N_POINT, SEQ_CAT_CFG,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OOF_DIR = os.path.join(PROJECT_ROOT, "oof_predictions")


def collate_to_device(batch):
    """Same as V11's collate, but ready for forward pass."""
    cat = torch.stack([b["cat"]      for b in batch]).to(DEVICE)
    num = torch.stack([b["num"]      for b in batch]).to(DEVICE)
    ctx = torch.stack([b["context"]  for b in batch]).to(DEVICE)
    pid_s = torch.stack([b["pid_self"]  for b in batch]).to(DEVICE)
    pid_o = torch.stack([b["pid_other"] for b in batch]).to(DEVICE)
    pad   = torch.stack([b["pad_mask"]  for b in batch]).to(DEVICE)
    sl    = torch.stack([b["seq_len"]   for b in batch]).to(DEVICE)
    return cat, num, ctx, pid_s, pid_o, pad, sl


def extract_embeddings_batched(model, samples, batch_size=256):
    """Run forward with extract_embeddings=True; return (last_emb, pool_emb)."""
    model.eval()
    ds = RallyDataset(samples, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=lambda b: b)
    last_list, pool_list = [], []
    with torch.no_grad():
        for batch in loader:
            cat, num, ctx, pid_s, pid_o, pad, sl = collate_to_device(batch)
            _a, _p, _s, last_repr, pool_repr = model(
                cat, num, ctx, pid_s, pid_o, pad, sl,
                extract_embeddings=True,
            )
            last_list.append(last_repr.cpu().numpy())
            pool_list.append(pool_repr.cpu().numpy())
    return np.concatenate(last_list, axis=0), np.concatenate(pool_list, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v11", help="Checkpoint tag (default v11)")
    p.add_argument("--out-prefix", default="v11_emb",
                   help="Output prefix for OOF/test arrays")
    args = p.parse_args()

    print("=" * 80)
    print(f" R-082 Phase 2 Step 2 — extract embeddings from {args.tag} checkpoints")
    print("=" * 80)
    print(f" Device: {DEVICE}")

    # 1. Load + clean data identically to training (so sample order matches OOF arrays)
    print("\n Step 1: load and clean train + test (matching V11 training pipeline)")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    n_players = len(player_map)  # match trainer: build_samples expects the count, not the map
    print(f"   train rows: {len(train_df)}, test rows: {len(test_df)}, "
          f"n_players: {n_players}")

    # 2. Build samples (matches V11 training)
    print(" Step 2: build train + test samples")
    train_samples = build_samples(train_df, is_train=True, n_players=n_players)
    # serverGetPoint is a train-only label; the test branch of build_samples uses a
    # placeholder (y_server=0) and never reads it as an input feature. The real test
    # CSV omits the column, so add a dummy to satisfy the unconditional read at the
    # top of build_samples. This does NOT affect extracted embeddings.
    if "serverGetPoint" not in test_df.columns:
        test_df = test_df.copy()
        test_df["serverGetPoint"] = 0
    test_samples  = build_samples(test_df,  is_train=False, n_players=n_players)
    n_train = len(train_samples)
    n_test  = len(test_samples)
    print(f"   train samples: {n_train}, test samples: {n_test}")

    # 3. GroupKFold split (same seed/method as training)
    print(" Step 3: re-derive fold splits")
    sample_matches = np.array([s["match_id"] for s in train_samples])
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(np.arange(n_train), groups=sample_matches))
    print(f"   {N_FOLDS} folds")

    # 4. Allocate output arrays
    # Determine embedding dim from one checkpoint
    first_ckpt_path = os.path.join(MODELS_DIR, f"{args.tag}_fold0.pt")
    if not os.path.exists(first_ckpt_path):
        print(f" MISSING checkpoint: {first_ckpt_path}")
        print(" R-082 Phase 2 retrain has not delivered checkpoints yet. Abort.")
        sys.exit(1)
    ck0 = torch.load(first_ckpt_path, map_location="cpu", weights_only=False)
    arch = ck0.get("arch_config", {})
    d_model = arch.get("d_model", 192)
    print(f"   embedding dim (d_model) = {d_model}")

    oof_emb_last = np.zeros((n_train, d_model), dtype=np.float32)
    oof_emb_pool = np.zeros((n_train, d_model), dtype=np.float32)
    test_emb_last_acc = np.zeros((n_test, d_model), dtype=np.float32)
    test_emb_pool_acc = np.zeros((n_test, d_model), dtype=np.float32)
    oof_mask = np.zeros(n_train, dtype=bool)
    n_folds_used = 0

    # 5. Per-fold extraction
    for fold, (tr_idx, val_idx) in enumerate(splits):
        ckpt_path = os.path.join(MODELS_DIR, f"{args.tag}_fold{fold}.pt")
        if not os.path.exists(ckpt_path):
            print(f"\n FOLD {fold}: MISSING {ckpt_path} — SKIP")
            continue
        print(f"\n FOLD {fold}: loading {ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        arch = ck.get("arch_config", {})
        model = RallyTransformer(
            d_model=arch.get("d_model", 192),
            n_heads=arch.get("n_heads", 8),
            n_layers=arch.get("n_layers", 4),
            dropout=arch.get("dropout", 0.15),
            n_players=arch.get("n_players", n_players + 5),
            max_len=arch.get("max_len", 40),
        ).to(DEVICE)
        model.load_state_dict({k: v.to(DEVICE) for k, v in ck["state_dict"].items()})

        # Extract val embeddings (OOF — fold-safe)
        val_samps = [train_samples[i] for i in val_idx]
        t0 = time.time()
        val_last, val_pool = extract_embeddings_batched(model, val_samps)
        print(f"   val embeddings: last{val_last.shape} pool{val_pool.shape} "
              f"({time.time()-t0:.1f}s)")
        oof_emb_last[val_idx] = val_last
        oof_emb_pool[val_idx] = val_pool
        oof_mask[val_idx] = True

        # Extract test embeddings (accumulate for averaging)
        t1 = time.time()
        test_last, test_pool = extract_embeddings_batched(model, test_samples)
        print(f"   test embeddings: last{test_last.shape} pool{test_pool.shape} "
              f"({time.time()-t1:.1f}s)")
        test_emb_last_acc += test_last
        test_emb_pool_acc += test_pool
        n_folds_used += 1

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if n_folds_used == 0:
        print(" ERROR: zero fold checkpoints found. Abort.")
        sys.exit(1)
    test_emb_last = test_emb_last_acc / n_folds_used
    test_emb_pool = test_emb_pool_acc / n_folds_used
    print(f"\n averaged test embeddings across {n_folds_used} fold models")
    print(f" OOF mask coverage: {oof_mask.sum()}/{n_train} = "
          f"{100*oof_mask.sum()/n_train:.1f}%")

    # 6. Save
    np.save(os.path.join(OOF_DIR, f"{args.out_prefix}_last_oof.npy"), oof_emb_last)
    np.save(os.path.join(OOF_DIR, f"{args.out_prefix}_pool_oof.npy"), oof_emb_pool)
    np.save(os.path.join(OOF_DIR, f"{args.out_prefix}_last_test.npy"), test_emb_last)
    np.save(os.path.join(OOF_DIR, f"{args.out_prefix}_pool_test.npy"), test_emb_pool)
    np.save(os.path.join(OOF_DIR, f"{args.out_prefix}_oof_mask.npy"), oof_mask)
    print(f"\n Saved to {OOF_DIR}/{args.out_prefix}_*.npy")
    print(" Next: src/train_gbm_on_v11_embed_smoke.py for Fold-1 GBM smoke")


if __name__ == "__main__":
    main()
