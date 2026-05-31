"""train_pretrain_badminton — pretrain transformer encoder on ShuttleSet22 (R-021).

Per Codex APPROVE_WITH_FIXES (2026-05-12) plan §3:
- Pretrain a transformer encoder on badminton next-stroke prediction
- Save ENCODER weights only (transformer + positional embedding)
- Heads / input embeddings / context layers re-initialized at fine-tune time
- NO transfer of badminton-vocabulary label embeddings
- NO transfer of badminton player metadata

Architecture mirrors src/train_v11_transformer.py RallyTransformer body:
- Per-shot categorical embedding sum → shot_proj → d_model
- Learned positional encoding (max_len=32)
- Transformer encoder: 4 layers, 192-d, 8 heads, GELU, batch_first, norm_first
- Per-position output heads: badminton next-stroke type (19) + landing_area (11)

Loss: CE(next_type) + CE(next_landing) at each position predicting position+1.
Position 0 (BOS-equivalent) is the first observed shot; loss valid 0..T-2.

Pretrain output: models/v11_pretrained_badminton.pt — encoder weights only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_shuttleset22 import (
    BADMINTON_TYPES, build_rally_sequences, load_shuttleset22, split_train_val,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary sizes (per features_shuttleset22 encoding)
N_TYPE = 19          # 0=unknown, 1..18 = stroke types
N_LANDING = 11       # 0=unknown, 1..10 = grid + outside
N_HIT = 11           # 0=unknown, 1..10 = grid + outside
N_BACKHAND = 2       # 0/1
N_AROUNDHEAD = 2     # 0/1
N_SERVER = 2         # 0/1

# Feature columns + (vocab_size, embed_dim)
SHOT_FIELDS = [
    ("type",       N_TYPE,       24),
    ("landing",    N_LANDING,    16),
    ("hit",        N_HIT,        12),
    ("backhand",   N_BACKHAND,   4),
    ("aroundhead", N_AROUNDHEAD, 4),
    ("server",     N_SERVER,     4),
]
TOTAL_SHOT_DIM = sum(d for _, _, d in SHOT_FIELDS)


class RallyDataset(Dataset):
    """Per-rally batched dataset. shots: (T, 6) int8."""
    IGNORE = -100

    def __init__(self, sequences: list, max_len: int = 32):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        shots = seq["shots"]
        T = min(shots.shape[0], self.max_len)

        cat = np.zeros((self.max_len, 6), dtype=np.int64)
        cat[:T] = shots[:T].astype(np.int64)
        # Clip to vocab bounds
        cat[:, 0] = np.clip(cat[:, 0], 0, N_TYPE - 1)
        cat[:, 1] = np.clip(cat[:, 1], 0, N_LANDING - 1)
        cat[:, 2] = np.clip(cat[:, 2], 0, N_HIT - 1)
        cat[:, 3] = np.clip(cat[:, 3], 0, N_BACKHAND - 1)
        cat[:, 4] = np.clip(cat[:, 4], 0, N_AROUNDHEAD - 1)
        cat[:, 5] = np.clip(cat[:, 5], 0, N_SERVER - 1)

        pad_mask = np.ones(self.max_len, dtype=bool)
        pad_mask[:T] = False

        # Targets at position p = cat at position p+1 (for p in 0..T-2)
        y_type = np.full(self.max_len, self.IGNORE, dtype=np.int64)
        y_landing = np.full(self.max_len, self.IGNORE, dtype=np.int64)
        for p in range(T - 1):
            y_type[p] = int(cat[p + 1, 0])
            y_landing[p] = int(cat[p + 1, 1])

        return {
            "cat": torch.from_numpy(cat),
            "pad_mask": torch.from_numpy(pad_mask),
            "y_type": torch.from_numpy(y_type),
            "y_landing": torch.from_numpy(y_landing),
            "T": T,
        }


class BadmintonRallyEncoder(nn.Module):
    """Transformer encoder for per-rally next-stroke prediction.

    Mirrors V11 architecture as closely as possible — same d_model, layers,
    heads — so the encoder weights can be loaded into RallyTransformer for
    AI CUP fine-tune.

    Saved state_dict subset (ENCODER ONLY):
      - shot_proj.* (shared shot input projection)
      - pos_emb (positional embedding)
      - transformer.* (full encoder body)
    """
    def __init__(self, d_model=192, n_heads=8, n_layers=4, dropout=0.1, max_len=32):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Per-field embeddings
        self.field_embeds = nn.ModuleList([
            nn.Embedding(vocab, dim) for _, vocab, dim in SHOT_FIELDS
        ])

        # Combined shot projection: sum_dim → d_model
        self.shot_proj = nn.Sequential(
            nn.Linear(TOTAL_SHOT_DIM, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional embedding
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer encoder (matches V11 / v11_mulminet structure;
        # causal mask applied in forward() to prevent target leakage in
        # next-stroke pretraining objective).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        # Cache causal mask: position p attends to positions ≤ p only.
        causal = torch.triu(torch.full((max_len, max_len), float("-inf")),
                             diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

        # Per-position output heads (badminton vocabulary; NOT transferred)
        self.type_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, N_TYPE))
        self.landing_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, N_LANDING))

    def forward(self, cat, pad_mask):
        B, L, _ = cat.shape
        # Embed each field
        embs = [self.field_embeds[i](cat[:, :, i]) for i in range(len(SHOT_FIELDS))]
        emb_cat = torch.cat(embs, dim=-1)
        x = self.shot_proj(emb_cat)
        # Positional
        pos = torch.arange(L, device=cat.device).unsqueeze(0)
        x = x + self.pos_emb(pos)
        # Transformer with CAUSAL MASK (position p attends to ≤ p only) —
        # required because pretraining objective predicts shot p+1 from
        # encoder output at position p; without causal mask, position p
        # attends to position p+1 (the answer) and learns trivial copy.
        x = self.transformer(x, mask=self.causal_mask[:L, :L],
                             src_key_padding_mask=pad_mask)
        # Per-position heads
        type_logits = self.type_head(x)
        landing_logits = self.landing_head(x)
        return type_logits, landing_logits


def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total_t = total_l = 0.0
    n = 0
    for batch in loader:
        cat = batch["cat"].to(DEVICE, non_blocking=True)
        pm = batch["pad_mask"].to(DEVICE, non_blocking=True)
        y_t = batch["y_type"].to(DEVICE, non_blocking=True)
        y_l = batch["y_landing"].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            t_l, l_l = model(cat, pm)
            loss_t = F.cross_entropy(t_l.reshape(-1, N_TYPE), y_t.reshape(-1),
                                     ignore_index=-100)
            loss_l = F.cross_entropy(l_l.reshape(-1, N_LANDING), y_l.reshape(-1),
                                     ignore_index=-100)
            loss = 0.5 * loss_t + 0.5 * loss_l
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_t += float(loss_t.item()); total_l += float(loss_l.item())
        n += 1
    return total_t / max(n, 1), total_l / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_t = total_l = 0.0
    n = 0
    for batch in loader:
        cat = batch["cat"].to(DEVICE, non_blocking=True)
        pm = batch["pad_mask"].to(DEVICE, non_blocking=True)
        y_t = batch["y_type"].to(DEVICE, non_blocking=True)
        y_l = batch["y_landing"].to(DEVICE, non_blocking=True)
        with autocast():
            t_l, l_l = model(cat, pm)
            loss_t = F.cross_entropy(t_l.reshape(-1, N_TYPE), y_t.reshape(-1),
                                     ignore_index=-100)
            loss_l = F.cross_entropy(l_l.reshape(-1, N_LANDING), y_l.reshape(-1),
                                     ignore_index=-100)
        total_t += float(loss_t.item()); total_l += float(loss_l.item())
        n += 1
    return total_t / max(n, 1), total_l / max(n, 1)


def save_encoder_only(model: BadmintonRallyEncoder, path: str):
    """Save ONLY the transferable layers per Codex P1.3.

    NOT saved:
      - field_embeds.* (badminton vocabulary embeddings — not transferable)
      - type_head.* (badminton stroke vocab head)
      - landing_head.* (badminton landing vocab head)
    """
    state = model.state_dict()
    transferable_keys = [k for k in state.keys() if k.startswith(("shot_proj.",
                                                                  "pos_emb",
                                                                  "transformer."))]
    saved = {k: state[k] for k in transferable_keys}
    torch.save({
        "transferable_state_dict": saved,
        "saved_keys": transferable_keys,
        "d_model": model.d_model,
        "max_len": model.max_len,
        "shot_input_dim": TOTAL_SHOT_DIM,
    }, path)
    print(f"  [save] {len(saved)}/{len(state)} keys saved to {path}")
    print(f"  [save] transferable param count: "
          f"{sum(v.numel() for v in saved.values())/1e6:.2f}M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str,
                    default="data/external/CoachAI-Projects/CoachAI-Challenge-IJCAI2023/ShuttleSet22")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=51966)
    ap.add_argument("--out", type=str, default="models/v11_pretrained_badminton.pt")
    ap.add_argument("--smoke", action="store_true",
                    help="3-epoch tiny pretrain validation (CPU/GPU)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 70)
    print(f"R-021 ShuttleSet22 PRETRAIN  device={DEVICE}")
    print(f"  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    print(f"  d_model={args.d_model}  n_layers={args.n_layers}  n_heads={args.n_heads}")
    print(f"  smoke={args.smoke}")
    print("=" * 70)

    t_start = time.time()

    # Load data
    df = load_shuttleset22(args.root)
    seqs = build_rally_sequences(df, max_len=args.max_len)
    tr_seqs, val_seqs = split_train_val(seqs, val_frac=0.1, seed=args.seed)
    print(f"  Train rallies: {len(tr_seqs)}  Val rallies: {len(val_seqs)}")

    if args.smoke:
        # Tiny pretrain: 3 epochs on subset
        tr_seqs = tr_seqs[:200]
        val_seqs = val_seqs[:50]
        n_epochs = 3
        print(f"  SMOKE: trimmed to {len(tr_seqs)} train / {len(val_seqs)} val, 3 epochs")
    else:
        n_epochs = args.epochs

    tr_ds = RallyDataset(tr_seqs, max_len=args.max_len)
    val_ds = RallyDataset(val_seqs, max_len=args.max_len)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                            num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = BadmintonRallyEncoder(d_model=args.d_model, n_heads=args.n_heads,
                                  n_layers=args.n_layers, max_len=args.max_len).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler()

    best_val = float("inf"); best_state = None
    for ep in range(1, n_epochs + 1):
        ep_t = time.time()
        tr_t, tr_l = train_epoch(model, tr_loader, optimizer, scaler)
        val_t, val_l = eval_epoch(model, val_loader)
        val_total = 0.5 * val_t + 0.5 * val_l
        marker = ""
        if val_total < best_val:
            best_val = val_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        print(f"  Ep{ep:3d}/{n_epochs}  "
              f"tr_type={tr_t:.4f} tr_land={tr_l:.4f}  "
              f"val_type={val_t:.4f} val_land={val_l:.4f}  "
              f"val_total={val_total:.4f}{marker}  [{time.time()-ep_t:.0f}s]")

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Save encoder weights only
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_encoder_only(model, args.out)

    print(f"\n  Pretrain wall: {(time.time()-t_start)/60:.1f} min")
    print(f"  Best val_total: {best_val:.4f}")
    print(f"  Saved to: {args.out}")


if __name__ == "__main__":
    main()
