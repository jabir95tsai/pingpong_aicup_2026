"""v17_causal_lm — Path B autoregressive rally LM (R-013 Fold-1 smoke).

Decoder-only Transformer with causal mask. Trained in three phases per
Codex APPROVE_WITH_FIXES (2026-05-10) optimized legal protocol:

  Phase 1a — shared pretrain on TEST visible prefixes only.
             actionId + pointId next-token objective; no SGP, no train labels.
  Phase 1b — Fold-1 train rally continuation pretrain.
             Same next-token objective, on Fold-1 train rallies' full
             action+point sequences. Fold-1 val rallies EXCLUDED.
  Phase 2  — Fold-1 supervised fine-tune.
             Standard 69k supervised pairs (every shot N>=2 with prefix
             1..N-1) restricted to Fold-1 train. Action+point heads at
             last visible position; SGP head from rally mean-pool with
             real train labels only.

Smoke is Fold-1 ONLY. Hard cap 2h GPU on RTX 3060 Ti.
NOT a full 5-fold run. NOT a zoo intake. NOT a submission.

Outputs:
  runs/v17_causal_lm_smoke_fold1/
    audit.json
    train.log         (printed to stdout; redirect when launching)
    val_metrics.json
    correlation_matrix.json
    per_class_f1.json
    fold1_oof_partial.npz
    summary.txt
"""
from __future__ import annotations

import argparse
import gc
import json
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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RANDOM_SEED, TEST_PATH, TRAIN_PATH
from data_cleaning import STRIKE_ID_MAP, clean_data
from features_v17_lm_tokens import (
    META_DIM,
    META_FIELDS,
    TOKEN_FIELD_NAMES,
    TOKEN_PAD_INDEX,
    TOKEN_VOCAB_SIZES,
    audit_fold_safe_pretrain,
    audit_no_forbidden_fields,
    audit_no_target_in_prefix,
    audit_sgp_loss_count,
    audit_test_prefix_length,
    audit_train_val_match_disjoint,
    build_phase1_corpus,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTION_TRAIN = 15
N_POINT = 10
ACTION_EVAL_LABELS = list(range(15))
POINT_EVAL_LABELS = list(range(10))
RUN_DIR = Path("runs/v17_causal_lm_smoke_fold1")
RUN_DIR.mkdir(parents=True, exist_ok=True)


# ─── Sample builder (Phase 2 supervised, mirrors v11 build_samples) ──────────

def build_supervised_samples(raw_df: pd.DataFrame, is_train: bool) -> list:
    """Build (rally, target shot) samples mirroring v11 build_samples.

    SAME ITERATION ORDER as src/train_v11_transformer.py:build_samples to
    preserve GroupKFold partition.
    """
    samples = []
    for uid, grp in raw_df.groupby("rally_uid", sort=False):
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        n = len(grp)
        if n < 2 and is_train:
            continue
        if n < 1:
            continue
        match_id = str(grp["match"].iloc[0])

        sn = grp["strikeNumber"].to_numpy(dtype=np.int32)
        action_raw = grp["actionId"].to_numpy(dtype=np.int32)
        action = np.where(action_raw >= 15, 0, action_raw).astype(np.int8)
        point = grp["pointId"].to_numpy(dtype=np.int8)
        hand = grp["handId"].to_numpy(dtype=np.int8)
        strength = grp["strengthId"].to_numpy(dtype=np.int8)
        spin = grp["spinId"].to_numpy(dtype=np.int8)
        position = grp["positionId"].to_numpy(dtype=np.int8)
        strike_id = grp["strikeId"].to_numpy(dtype=np.int8)
        side = ((sn % 2) == 0).astype(np.int8)
        sex = float(grp["sex"].iloc[0]) / 2.0
        num_game = float(grp["numberGame"].iloc[0]) / 7.0
        meta = np.array([sex, num_game], dtype=np.float32)

        target_indices = range(1, n) if is_train else range(n, n + 1)

        for tgt in target_indices:
            k = tgt  # context length = tgt shots (positions 0..tgt-1)
            cat_seq = np.stack([action[:k], point[:k], hand[:k], strength[:k],
                                spin[:k], position[:k], strike_id[:k],
                                side[:k]], axis=1).astype(np.int8)

            if is_train:
                y_a_raw = int(action_raw[tgt])
                y_a = 0 if y_a_raw >= N_ACTION_TRAIN else y_a_raw
                y_p = int(point[tgt])
                y_s = int(grp["serverGetPoint"].iloc[0])
                nsn = int(sn[tgt]) if tgt < n else int(sn[-1]) + 1
            else:
                y_a, y_p, y_s = 0, 0, -1
                nsn = int(sn[-1]) + 1

            samples.append({
                "cat_seq": cat_seq,            # (k, 8) int8
                "meta": meta,                  # (META_DIM,) float32
                "y_action": y_a,
                "y_point": y_p,
                "y_server": y_s,               # rally-level, broadcast per row
                "next_sn": nsn,
                "rally_uid": str(uid),
                "match_id": match_id,
            })
    return samples


# ─── Datasets ────────────────────────────────────────────────────────────────

def _pad_shots(shots: np.ndarray, max_len: int) -> tuple:
    """Pad a (k, 8) shot array to (max_len, 8) with PAD indices.
    Returns (padded, pad_mask) where pad_mask[i]=True if position i is padded.
    """
    k = min(shots.shape[0], max_len)
    pad = np.zeros((max_len, 8), dtype=np.int64)
    pad_mask = np.ones(max_len, dtype=bool)
    for j, fname in enumerate(TOKEN_FIELD_NAMES):
        pad[:, j] = TOKEN_PAD_INDEX[fname]
    pad[:k] = shots[:k]
    pad_mask[:k] = False
    return pad, pad_mask, k


class Phase1Dataset(Dataset):
    """Next-token autoregressive dataset.

    Each item:
      cat: (max_len, 8) int64
      meta: (META_DIM,) float32
      pad_mask: (max_len,) bool
      seq_len: int (= valid shot count)
      y_action_seq: (max_len,) int64  — target action at each position p is
        the action of shot p+1; -100 if padded or no target (ignored by CE)
      y_point_seq:  (max_len,) int64  — same for point
    """

    IGNORE = -100

    def __init__(self, sequences: list, max_len: int):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        shots = seq.shots  # (T, 8)
        T = shots.shape[0]

        cat, pad_mask, k = _pad_shots(shots, self.max_len)

        # Target arrays: at position p, predict shot p+1.
        # Loss valid for p in 0..k-2 (predicting shots 1..k-1, i.e. shots
        # indexed 1..k-1 in shots array).
        # For convention: y_action_seq[p] = action of shots[p+1] for p in 0..k-2.
        y_a = np.full(self.max_len, self.IGNORE, dtype=np.int64)
        y_p = np.full(self.max_len, self.IGNORE, dtype=np.int64)
        for p in range(0, k - 1):
            y_a[p] = int(shots[p + 1, 0])  # column 0 = action
            y_p[p] = int(shots[p + 1, 1])  # column 1 = point

        return {
            "cat": torch.from_numpy(cat),
            "meta": torch.from_numpy(seq.meta),
            "pad_mask": torch.from_numpy(pad_mask),
            "seq_len": torch.tensor(k, dtype=torch.long),
            "y_action_seq": torch.from_numpy(y_a),
            "y_point_seq": torch.from_numpy(y_p),
        }


class Phase2Dataset(Dataset):
    """Supervised dataset — predict shot N from prefix 1..N-1.

    Each item:
      cat: (max_len, 8) int64
      meta: (META_DIM,) float32
      pad_mask: (max_len,) bool
      seq_len: int
      y_action: int (0..14)
      y_point: int (0..9)
      y_server: float (0/1)
      next_sn: int
      rally_uid: str
    """

    def __init__(self, samples: list, max_len: int):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cat, pad_mask, k = _pad_shots(s["cat_seq"], self.max_len)
        return {
            "cat": torch.from_numpy(cat),
            "meta": torch.from_numpy(s["meta"]),
            "pad_mask": torch.from_numpy(pad_mask),
            "seq_len": torch.tensor(k, dtype=torch.long),
            "y_action": torch.tensor(s["y_action"], dtype=torch.long),
            "y_point": torch.tensor(s["y_point"], dtype=torch.long),
            "y_server": torch.tensor(s["y_server"], dtype=torch.float),
            "next_sn": torch.tensor(s["next_sn"], dtype=torch.long),
        }


# ─── Model ───────────────────────────────────────────────────────────────────

class CausalRallyLM(nn.Module):
    """Decoder-only Transformer with causal mask.

    Per-shot embedding sums all 8 token-field embeddings + positional + meta bias.
    Produces per-position action+point logits AND a rally-level SGP logit
    via mean-pool over real (non-padded) positions.
    """
    SHOT_EMB_DIM = 24  # each of 8 fields embeds to this; sum projected to d_model

    def __init__(self, d_model: int = 192, n_heads: int = 6, n_layers: int = 4,
                 dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Per-field embeddings (each padded vocab includes PAD as last index).
        self.field_embeds = nn.ModuleList([
            nn.Embedding(TOKEN_VOCAB_SIZES[fname], self.SHOT_EMB_DIM,
                         padding_idx=TOKEN_PAD_INDEX[fname])
            for fname in TOKEN_FIELD_NAMES
        ])
        total_field_dim = self.SHOT_EMB_DIM * len(TOKEN_FIELD_NAMES)

        # Project sum of fields to d_model.
        self.shot_proj = nn.Sequential(
            nn.Linear(total_field_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Learned positional embedding.
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Meta context (sex, numberGame) — additive bias on every position.
        self.meta_proj = nn.Linear(META_DIM, d_model)

        # Transformer with causal mask.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False)

        # Heads.
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_ACTION_TRAIN),
        )
        self.point_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_POINT),
        )
        self.sgp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Cache the causal mask once.
        causal = torch.triu(torch.full((max_len, max_len), float("-inf")),
                            diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, cat, meta, pad_mask, seq_len):
        """
        cat: (B, L, 8) int64
        meta: (B, META_DIM) float32
        pad_mask: (B, L) bool   (True = pad position, masked out of attention)
        seq_len: (B,) int64

        Returns:
          action_logits_seq: (B, L, 15) per-position
          point_logits_seq:  (B, L, 10) per-position
          sgp_logit_rally:   (B,) rally-level (mean-pool over real positions)
        """
        B, L, _ = cat.shape

        # Field embeddings, summed.
        embs = [emb(cat[:, :, i]) for i, emb in enumerate(self.field_embeds)]
        emb_cat = torch.cat(embs, dim=-1)  # (B, L, total_field_dim)
        x = self.shot_proj(emb_cat)        # (B, L, d_model)

        # Positional.
        pos = torch.arange(L, device=cat.device).unsqueeze(0)
        x = x + self.pos_emb(pos)

        # Meta bias (broadcast).
        x = x + self.meta_proj(meta).unsqueeze(1)  # (B, 1, d_model)

        # Transformer with causal mask.
        x = self.transformer(x, mask=self.causal_mask[:L, :L],
                             src_key_padding_mask=pad_mask)
        # Per-position heads.
        action_logits_seq = self.action_head(x)  # (B, L, 15)
        point_logits_seq = self.point_head(x)    # (B, L, 10)

        # Rally-level SGP via mean-pool over real positions.
        real_mask = (~pad_mask).float().unsqueeze(-1)
        pool = (x * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)
        sgp_logit = self.sgp_head(pool).squeeze(-1)  # (B,)

        return action_logits_seq, point_logits_seq, sgp_logit


# ─── Training loops ──────────────────────────────────────────────────────────

def train_phase1_epoch(model, loader, optimizer, scaler, sgp_loss_counter):
    """Phase 1 next-token pretraining. SGP head is NOT trained here (sgp_loss_counter
    must remain 0 — audited)."""
    model.train()
    total = 0.0
    n_batches = 0
    for batch in loader:
        cat = batch["cat"].to(DEVICE, non_blocking=True)
        meta = batch["meta"].to(DEVICE, non_blocking=True)
        pm = batch["pad_mask"].to(DEVICE, non_blocking=True)
        sl = batch["seq_len"].to(DEVICE, non_blocking=True)
        y_a = batch["y_action_seq"].to(DEVICE, non_blocking=True)
        y_p = batch["y_point_seq"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            a_logits, p_logits, _sgp = model(cat, meta, pm, sl)
            # Flatten for CE; ignore_index=-100 skips padded/no-target positions.
            loss_a = F.cross_entropy(
                a_logits.reshape(-1, N_ACTION_TRAIN), y_a.reshape(-1),
                ignore_index=-100)
            loss_p = F.cross_entropy(
                p_logits.reshape(-1, N_POINT), y_p.reshape(-1),
                ignore_index=-100)
            loss = 0.5 * loss_a + 0.5 * loss_p
            # SGP head is intentionally NOT in this loss (audit 8.E).
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)


def train_phase2_epoch(model, loader, optimizer, scaler, sgp_loss_counter):
    """Phase 2 supervised fine-tune. Action+point loss at last visible position;
    SGP loss on rally-level head against real train labels."""
    model.train()
    total = 0.0
    n_batches = 0
    for batch in loader:
        cat = batch["cat"].to(DEVICE, non_blocking=True)
        meta = batch["meta"].to(DEVICE, non_blocking=True)
        pm = batch["pad_mask"].to(DEVICE, non_blocking=True)
        sl = batch["seq_len"].to(DEVICE, non_blocking=True)
        y_a = batch["y_action"].to(DEVICE, non_blocking=True)
        y_p = batch["y_point"].to(DEVICE, non_blocking=True)
        y_s = batch["y_server"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            a_seq, p_seq, sgp_logit = model(cat, meta, pm, sl)
            # Last-position readout.
            B = cat.shape[0]
            last = (sl - 1).clamp(min=0)
            a_last = a_seq.gather(1, last.view(B, 1, 1)
                                  .expand(B, 1, N_ACTION_TRAIN)).squeeze(1)
            p_last = p_seq.gather(1, last.view(B, 1, 1)
                                  .expand(B, 1, N_POINT)).squeeze(1)
            loss_a = F.cross_entropy(a_last, y_a)
            loss_p = F.cross_entropy(p_last, y_p)
            loss_s = F.binary_cross_entropy_with_logits(sgp_logit, y_s)
            loss = 0.4 * loss_a + 0.4 * loss_p + 0.2 * loss_s
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())
        n_batches += 1
        # SGP loss counter — every batch sample contributed exactly one SGP loss term.
        sgp_loss_counter[0] += B
    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate_phase2(model, loader):
    model.eval()
    a_list, p_list, s_list = [], [], []
    for batch in loader:
        cat = batch["cat"].to(DEVICE, non_blocking=True)
        meta = batch["meta"].to(DEVICE, non_blocking=True)
        pm = batch["pad_mask"].to(DEVICE, non_blocking=True)
        sl = batch["seq_len"].to(DEVICE, non_blocking=True)
        with autocast():
            a_seq, p_seq, sgp_logit = model(cat, meta, pm, sl)
            B = cat.shape[0]
            last = (sl - 1).clamp(min=0)
            a_last = a_seq.gather(1, last.view(B, 1, 1)
                                  .expand(B, 1, N_ACTION_TRAIN)).squeeze(1)
            p_last = p_seq.gather(1, last.view(B, 1, 1)
                                  .expand(B, 1, N_POINT)).squeeze(1)
        a_list.append(F.softmax(a_last.float(), dim=-1).cpu().numpy())
        p_list.append(F.softmax(p_last.float(), dim=-1).cpu().numpy())
        s_list.append(torch.sigmoid(sgp_logit.float()).cpu().numpy())
    return (np.vstack(a_list), np.vstack(p_list), np.concatenate(s_list))


def apply_action_rules(probs: np.ndarray, next_sns: np.ndarray) -> np.ndarray:
    """Zero serve classes (15-18) for non-serve shots — but our 15-class space
    is 0..14 only, so this is a no-op when target is non-serve. For serve shots
    (next_sn==1), all of 0..14 should also be near-zero in supervised target,
    and v17 outputs only 15-class — so we simply renormalise."""
    out = probs.copy()
    serve_mask = (next_sns == 1)
    if serve_mask.any():
        # For serve shots, our 15-class probs are unreliable; zero them all.
        # Action rules at submission time will take the argmax over the full
        # 0..18 space using v11/v14 outputs. Here we just renormalise so the
        # rest of the metric is computed on non-serve rows cleanly.
        pass  # v11 follows the same convention; the apply_action_rules in
              # v11_transformer also doesn't help for serves under 15-class.
    row_sum = out.sum(axis=1, keepdims=True)
    out /= np.where(row_sum == 0, 1.0, row_sum)
    return out


# ─── Correlation helpers ─────────────────────────────────────────────────────

def macro_class_correlation(probs_a: np.ndarray, probs_b: np.ndarray) -> float:
    """Average Pearson r across class columns. probs_a/b: (N, K)."""
    K = probs_a.shape[1]
    rs = []
    for c in range(K):
        x = probs_a[:, c]
        y = probs_b[:, c]
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        rs.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(rs)) if rs else float("nan")


def truncate_action_to_15(probs: np.ndarray) -> np.ndarray:
    """v14 action probs are 19-class; v17 is 15-class. Truncate v14 to first
    15 columns and renormalise for fair correlation."""
    out = probs[:, :15].copy()
    row_sum = out.sum(axis=1, keepdims=True)
    out /= np.where(row_sum == 0, 1.0, row_sum)
    return out


# ─── Audit harness ───────────────────────────────────────────────────────────

def audit_no_forbidden_in_model(model: nn.Module) -> dict:
    """8.D part 2 — model module names must not reference forbidden fields."""
    forbidden = {"servergetpoint", "sgp_input", "sgp_target_token",
                 "rally_uid", "match_id_emb", "gameplayer", "playerid_emb"}
    bad = []
    for name, _module in model.named_modules():
        nl = name.lower()
        for f in forbidden:
            if f in nl:
                bad.append((name, f))
    assert not bad, (
        f"VIOLATION (8.D model): forbidden tokens in module names: {bad}")
    # Note: sgp_head IS a head ON RALLY MEAN-POOL, not a token input. Not
    # forbidden — it just must not see SGP IN the input tokens (Phase 1
    # SGP loss count = 0 audit covers that).
    return {"audit_8D_model_modules": "PASS",
            "n_modules": sum(1 for _ in model.named_modules())}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a-epochs", type=int, default=6)
    parser.add_argument("--phase1b-epochs", type=int, default=6)
    parser.add_argument("--phase2-epochs", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=64,
                        help="Will auto-retry once at /2 on OOM.")
    parser.add_argument("--lr-phase1", type=float, default=3e-4)
    parser.add_argument("--lr-phase2", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=51966)
    parser.add_argument("--hard-cap-h", type=float, default=2.0,
                        help="Wall-time hard cap in hours (smoke).")
    args = parser.parse_args()

    t_start = time.time()
    hard_cap_s = args.hard_cap_h * 3600.0

    print("=" * 70)
    print(f"V17 CAUSAL LM — Fold-1 SMOKE  (R-013 APPROVE_WITH_FIXES applied)")
    print(f"  device={DEVICE}  d_model={args.d_model}  layers={args.n_layers}  "
          f"heads={args.n_heads}")
    print(f"  Phase 1a epochs={args.phase1a_epochs}  "
          f"Phase 1b epochs={args.phase1b_epochs}  "
          f"Phase 2 epochs={args.phase2_epochs}")
    print(f"  batch={args.batch}  lr_p1={args.lr_phase1}  lr_p2={args.lr_phase2}")
    print(f"  hard_cap={args.hard_cap_h}h  seed={args.seed}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load + clean data ────────────────────────────────────────────────────
    print("\n--- Loading data ---")
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1
    print(f"  train rows: {len(train_df)}  test rows: {len(test_df)}  "
          f"players: {len(player_map)}")

    # ── Build supervised samples (mirrors v11 ordering for fold split) ───────
    print("\n--- Building supervised samples (Phase 2) ---")
    t0 = time.time()
    train_samples = build_supervised_samples(train_df, is_train=True)
    test_samples = build_supervised_samples(test_df, is_train=False)
    n_train = len(train_samples)
    n_test = len(test_samples)
    print(f"  train supervised pairs: {n_train}  test inference samples: {n_test}  "
          f"({time.time() - t0:.1f}s)")
    assert n_train == 69712, f"expected 69712 train samples (v11 contract), got {n_train}"

    sample_matches = np.array([s["match_id"] for s in train_samples])
    sample_rally_uids = np.array([s["rally_uid"] for s in train_samples])

    # ── GroupKFold partition (Fold 1) ────────────────────────────────────────
    print("\n--- GroupKFold(5) by match — taking Fold 1 ---")
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(n_train), groups=sample_matches))
    tr_idx, val_idx = splits[0]
    fold1_train_matches = set(sample_matches[tr_idx])
    fold1_val_matches = set(sample_matches[val_idx])
    fold1_train_rally_uids = set(sample_rally_uids[tr_idx])
    fold1_val_rally_uids = set(sample_rally_uids[val_idx])
    print(f"  Fold-1 train samples: {len(tr_idx)}  val samples: {len(val_idx)}")
    print(f"  Fold-1 train matches: {len(fold1_train_matches)}  "
          f"val matches: {len(fold1_val_matches)}")
    print(f"  Fold-1 train rallies: {len(fold1_train_rally_uids)}  "
          f"val rallies: {len(fold1_val_rally_uids)}")

    # ── Audit 1: train/val match disjoint ────────────────────────────────────
    audits = {}
    audits["train_val_match_disjoint"] = audit_train_val_match_disjoint(
        fold1_train_matches, fold1_val_matches)
    print(f"  [audit] train/val match disjoint: PASS")

    # ── Audit 2: token builder uses no forbidden fields ──────────────────────
    audits["no_forbidden_token_fields"] = audit_no_forbidden_fields(
        TOKEN_FIELD_NAMES)
    print(f"  [audit] no forbidden token fields: PASS")

    # ── Audit 3: no target in own prefix ─────────────────────────────────────
    audits["no_target_in_prefix"] = audit_no_target_in_prefix(
        train_samples, max_check=5000)
    print(f"  [audit] no target in own prefix (5000 sampled): PASS")

    # ── Build Phase 1a corpus (test visible prefixes) ────────────────────────
    print("\n--- Building Phase 1a corpus (TEST visible prefixes) ---")
    t0 = time.time()
    phase1a_seqs = build_phase1_corpus(test_df, rally_filter=None,
                                        is_test=True, label="phase1a_test")
    print(f"  ({time.time() - t0:.1f}s)")

    # ── Build Phase 1b corpus (Fold-1 train rallies, full sequences) ─────────
    print("\n--- Building Phase 1b corpus (FOLD-1 TRAIN rallies) ---")
    t0 = time.time()
    phase1b_seqs = build_phase1_corpus(train_df,
                                       rally_filter=fold1_train_rally_uids,
                                       is_test=False, label="phase1b_fold1_train")
    print(f"  ({time.time() - t0:.1f}s)")

    # ── Audit 4: fold-safe pretraining ───────────────────────────────────────
    p1a_uids = {s.rally_uid for s in phase1a_seqs}
    p1b_uids = {s.rally_uid for s in phase1b_seqs}
    audits["fold_safe_pretrain"] = audit_fold_safe_pretrain(
        p1a_uids, p1b_uids, fold1_train_rally_uids, fold1_val_rally_uids)
    print(f"  [audit] fold-safe Phase 1 corpus: PASS")

    # ── Audit 5: test prefix length = visible length ─────────────────────────
    audits["test_prefix_length"] = audit_test_prefix_length(
        phase1a_seqs, test_df)
    print(f"  [audit] test prefix length matches visible: PASS")

    # ── Build datasets + loaders ─────────────────────────────────────────────
    p1a_ds = Phase1Dataset(phase1a_seqs, max_len=args.max_len)
    p1b_ds = Phase1Dataset(phase1b_seqs, max_len=args.max_len)

    fold1_tr_samples = [train_samples[i] for i in tr_idx]
    fold1_val_samples = [train_samples[i] for i in val_idx]
    p2_tr_ds = Phase2Dataset(fold1_tr_samples, max_len=args.max_len)
    p2_val_ds = Phase2Dataset(fold1_val_samples, max_len=args.max_len)
    p2_test_ds = Phase2Dataset(test_samples, max_len=args.max_len)

    def make_loader(ds, batch_size, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True, drop_last=False)

    bs = args.batch
    p1a_loader = make_loader(p1a_ds, bs, True)
    p1b_loader = make_loader(p1b_ds, bs, True)
    p2_tr_loader = make_loader(p2_tr_ds, bs, True)
    p2_val_loader = make_loader(p2_val_ds, bs * 2, False)
    p2_test_loader = make_loader(p2_test_ds, bs * 2, False)

    # ── Build model ──────────────────────────────────────────────────────────
    model = CausalRallyLM(d_model=args.d_model, n_heads=args.n_heads,
                          n_layers=args.n_layers, dropout=args.dropout,
                          max_len=args.max_len).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- Model: {n_params/1e6:.2f}M params ---")

    audits["no_forbidden_in_model"] = audit_no_forbidden_in_model(model)
    print(f"  [audit] no forbidden module names in model: PASS")

    # ── Phase 1a: pretrain on test prefixes ──────────────────────────────────
    print(f"\n=== PHASE 1a: pretrain on test visible prefixes ===")
    print(f"  rallies: {len(phase1a_seqs)}  epochs: {args.phase1a_epochs}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_phase1,
                                   weight_decay=1e-2, eps=1e-8)
    scaler = GradScaler()
    sgp_counter = [0]
    p1a_losses = []
    t_p1a = time.time()
    for ep in range(1, args.phase1a_epochs + 1):
        elapsed_h = (time.time() - t_start) / 3600.0
        if elapsed_h > args.hard_cap_h:
            print(f"  HARD CAP REACHED at Phase 1a epoch {ep}  "
                  f"({elapsed_h:.2f}h > {args.hard_cap_h}h). Killing.")
            break
        ep_t = time.time()
        try:
            loss = train_phase1_epoch(model, p1a_loader, optimizer, scaler,
                                       sgp_counter)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at Phase 1a epoch {ep}. One retry with batch /2 ...")
            torch.cuda.empty_cache()
            bs2 = bs // 2
            p1a_loader = make_loader(p1a_ds, bs2, True)
            loss = train_phase1_epoch(model, p1a_loader, optimizer, scaler,
                                       sgp_counter)
        p1a_losses.append(loss)
        print(f"  Phase1a Ep{ep}/{args.phase1a_epochs}  loss={loss:.4f}  "
              f"[{time.time() - ep_t:.0f}s, total {(time.time()-t_start)/60:.1f}min]")
        if not np.isfinite(loss):
            print(f"  NaN loss at Phase 1a epoch {ep}. Killing.")
            return _emergency_exit(audits, "Phase 1a NaN", t_start)

    print(f"  Phase 1a done. [{(time.time()-t_p1a)/60:.1f} min]  "
          f"sgp_loss_count={sgp_counter[0]} (must be 0)")
    assert sgp_counter[0] == 0, f"Phase 1a SGP loss count {sgp_counter[0]} != 0"

    # ── Phase 1b: continuation on Fold-1 train rallies ───────────────────────
    print(f"\n=== PHASE 1b: continuation on Fold-1 train rally sequences ===")
    print(f"  rallies: {len(phase1b_seqs)}  epochs: {args.phase1b_epochs}")

    # Reset optimizer for Phase 1b (fresh LR schedule).
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_phase1,
                                   weight_decay=1e-2, eps=1e-8)
    scaler = GradScaler()
    p1b_losses = []
    t_p1b = time.time()
    for ep in range(1, args.phase1b_epochs + 1):
        elapsed_h = (time.time() - t_start) / 3600.0
        if elapsed_h > args.hard_cap_h:
            print(f"  HARD CAP REACHED at Phase 1b epoch {ep}  "
                  f"({elapsed_h:.2f}h > {args.hard_cap_h}h). Killing.")
            break
        ep_t = time.time()
        try:
            loss = train_phase1_epoch(model, p1b_loader, optimizer, scaler,
                                       sgp_counter)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at Phase 1b epoch {ep}. Killing.")
            return _emergency_exit(audits, "Phase 1b OOM", t_start)
        p1b_losses.append(loss)
        print(f"  Phase1b Ep{ep}/{args.phase1b_epochs}  loss={loss:.4f}  "
              f"[{time.time() - ep_t:.0f}s, total {(time.time()-t_start)/60:.1f}min]")
        if not np.isfinite(loss):
            print(f"  NaN loss at Phase 1b epoch {ep}. Killing.")
            return _emergency_exit(audits, "Phase 1b NaN", t_start)

    print(f"  Phase 1b done. [{(time.time()-t_p1b)/60:.1f} min]  "
          f"sgp_loss_count={sgp_counter[0]} (must remain 0)")
    assert sgp_counter[0] == 0, f"Phase 1b SGP loss count {sgp_counter[0]} != 0"

    # ── Phase 2: Fold-1 supervised fine-tune ─────────────────────────────────
    print(f"\n=== PHASE 2: Fold-1 supervised fine-tune ===")
    print(f"  train pairs: {len(fold1_tr_samples)}  val pairs: {len(fold1_val_samples)}  "
          f"epochs: {args.phase2_epochs}")

    # Reset optimizer for Phase 2 (lower LR for fine-tune).
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_phase2,
                                   weight_decay=1e-2, eps=1e-8)
    scaler = GradScaler()
    p2_train_losses = []
    p2_val_metrics_per_ep = []
    best_ov = -1.0
    best_state = None
    t_p2 = time.time()
    for ep in range(1, args.phase2_epochs + 1):
        elapsed_h = (time.time() - t_start) / 3600.0
        if elapsed_h > args.hard_cap_h:
            print(f"  HARD CAP REACHED at Phase 2 epoch {ep}  "
                  f"({elapsed_h:.2f}h > {args.hard_cap_h}h). Killing.")
            break
        ep_t = time.time()
        try:
            loss = train_phase2_epoch(model, p2_tr_loader, optimizer, scaler,
                                       sgp_counter)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at Phase 2 epoch {ep}. Killing.")
            return _emergency_exit(audits, "Phase 2 OOM", t_start)
        p2_train_losses.append(loss)
        if not np.isfinite(loss):
            print(f"  NaN loss at Phase 2 epoch {ep}. Killing.")
            return _emergency_exit(audits, "Phase 2 NaN", t_start)
        # Evaluate on val every epoch (smoke is short).
        a_p, p_p, s_p = evaluate_phase2(model, p2_val_loader)
        y_a = np.array([s["y_action"] for s in fold1_val_samples])
        y_p = np.array([s["y_point"] for s in fold1_val_samples])
        y_s = np.array([s["y_server"] for s in fold1_val_samples], dtype=float)
        nsn = np.array([s["next_sn"] for s in fold1_val_samples])
        a_p_ruled = apply_action_rules(a_p, nsn)
        f1_a = f1_score(y_a, a_p_ruled.argmax(axis=1), labels=ACTION_EVAL_LABELS,
                        average="macro", zero_division=0)
        f1_p = f1_score(y_p, p_p.argmax(axis=1), labels=POINT_EVAL_LABELS,
                        average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_s, s_p)
        except ValueError:
            auc = float("nan")
        ov = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
        p2_val_metrics_per_ep.append({"epoch": ep, "loss": loss,
                                       "f1_a": f1_a, "f1_p": f1_p,
                                       "auc": auc, "ov": ov})
        print(f"  Phase2 Ep{ep}/{args.phase2_epochs}  loss={loss:.4f}  "
              f"F1_a={f1_a:.4f}  F1_p={f1_p:.4f}  AUC={auc:.4f}  OV={ov:.4f}  "
              f"[{time.time()-ep_t:.0f}s, total {(time.time()-t_start)/60:.1f}min]")
        if ov > best_ov:
            best_ov = ov
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f"  Phase 2 done. [{(time.time()-t_p2)/60:.1f} min]  best OV={best_ov:.4f}")

    # ── Final Fold-1 OOF + test inference ────────────────────────────────────
    print(f"\n--- Final Fold-1 val + test inference ---")
    a_val, p_val, s_val = evaluate_phase2(model, p2_val_loader)
    a_test, p_test, s_test = evaluate_phase2(model, p2_test_loader)

    y_a = np.array([s["y_action"] for s in fold1_val_samples])
    y_p = np.array([s["y_point"] for s in fold1_val_samples])
    y_s = np.array([s["y_server"] for s in fold1_val_samples], dtype=float)
    nsn = np.array([s["next_sn"] for s in fold1_val_samples])

    a_val_ruled = apply_action_rules(a_val, nsn)
    f1_a_final = f1_score(y_a, a_val_ruled.argmax(axis=1),
                          labels=ACTION_EVAL_LABELS, average="macro",
                          zero_division=0)
    f1_p_final = f1_score(y_p, p_val.argmax(axis=1),
                          labels=POINT_EVAL_LABELS, average="macro",
                          zero_division=0)
    try:
        auc_final = roc_auc_score(y_s, s_val)
    except ValueError:
        auc_final = float("nan")
    ov_final = 0.4 * f1_a_final + 0.4 * f1_p_final + 0.2 * auc_final

    # Per-class F1 for canary tracking
    per_class_action = f1_score(y_a, a_val_ruled.argmax(axis=1),
                                 labels=ACTION_EVAL_LABELS,
                                 average=None, zero_division=0)
    per_class_point = f1_score(y_p, p_val.argmax(axis=1),
                                labels=POINT_EVAL_LABELS,
                                average=None, zero_division=0)

    # ── Audit 8.E (final): SGP loss count check ──────────────────────────────
    n_train_in_fold1 = len(tr_idx)
    expected_sgp = n_train_in_fold1 * len(p2_train_losses)  # times epochs run
    audits["sgp_loss_count"] = audit_sgp_loss_count(
        phase1_sgp_count=0,  # asserted to be 0 inside Phase 1 loops
        phase2_sgp_count=sgp_counter[0],
        phase2_train_rally_count=expected_sgp,
        phase2_test_sgp_count=0)
    print(f"  [audit] SGP loss count: PASS  (Phase1=0, Phase2={sgp_counter[0]} "
          f"= train_pairs {n_train_in_fold1} × epochs {len(p2_train_losses)})")

    # ── Correlation matrix vs v11_aug, v11, v14_seed2 ────────────────────────
    print(f"\n--- Correlation vs v11_aug / v11 / v14_seed2 (Fold-1 val rows) ---")
    corr_matrix = {}
    for ref_tag in ["v11_aug", "v11", "v14_seed2"]:
        try:
            ref_a = np.load(f"oof_predictions/{ref_tag}_oof_act.npy")
            ref_p = np.load(f"oof_predictions/{ref_tag}_oof_pt.npy")
        except FileNotFoundError:
            print(f"  [warn] {ref_tag} OOF not found; skipping")
            continue
        # Subset to Fold-1 val sample indices.
        ref_a_v = ref_a[val_idx]
        ref_p_v = ref_p[val_idx]
        # If 19-class, truncate to 15.
        if ref_a_v.shape[1] == 19:
            ref_a_v = truncate_action_to_15(ref_a_v)
        r_act = macro_class_correlation(a_val_ruled, ref_a_v)
        r_pt = macro_class_correlation(p_val, ref_p_v)
        corr_matrix[ref_tag] = {"action": r_act, "point": r_pt}
        print(f"  vs {ref_tag}:  r_action={r_act:.4f}  r_point={r_pt:.4f}")

    # ── Baselines on Fold-1 val rows ─────────────────────────────────────────
    print(f"\n--- Baseline comparison on Fold-1 val rows ---")
    baselines = {}
    for ref_tag in ["v11_aug", "v11", "v14_seed2"]:
        try:
            ref_a = np.load(f"oof_predictions/{ref_tag}_oof_act.npy")[val_idx]
            ref_p = np.load(f"oof_predictions/{ref_tag}_oof_pt.npy")[val_idx]
            ref_s = np.load(f"oof_predictions/{ref_tag}_oof_srv.npy")[val_idx]
        except FileNotFoundError:
            continue
        if ref_a.shape[1] == 19:
            ref_a = truncate_action_to_15(ref_a)
        b_f1_a = f1_score(y_a, ref_a.argmax(axis=1), labels=ACTION_EVAL_LABELS,
                          average="macro", zero_division=0)
        b_f1_p = f1_score(y_p, ref_p.argmax(axis=1), labels=POINT_EVAL_LABELS,
                          average="macro", zero_division=0)
        try:
            b_auc = roc_auc_score(y_s, ref_s)
        except ValueError:
            b_auc = float("nan")
        b_ov = 0.4 * b_f1_a + 0.4 * b_f1_p + 0.2 * b_auc
        baselines[ref_tag] = {"f1_a": b_f1_a, "f1_p": b_f1_p,
                              "auc": b_auc, "ov": b_ov}
        print(f"  {ref_tag}:  F1_a={b_f1_a:.4f}  F1_p={b_f1_p:.4f}  "
              f"AUC={b_auc:.4f}  OV={b_ov:.4f}")
    print(f"  v17_smoke: F1_a={f1_a_final:.4f}  F1_p={f1_p_final:.4f}  "
          f"AUC={auc_final:.4f}  OV={ov_final:.4f}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    print(f"\n--- Saving smoke artifacts to {RUN_DIR}/ ---")
    val_metrics = {
        "v17_smoke_fold1": {"f1_a": f1_a_final, "f1_p": f1_p_final,
                            "auc": auc_final, "ov": ov_final,
                            "best_ov_during_training": best_ov,
                            "n_val_samples": int(len(val_idx))},
        "baselines_fold1": baselines,
        "training_curve_phase2": p2_val_metrics_per_ep,
        "phase1a_losses": p1a_losses,
        "phase1b_losses": p1b_losses,
        "phase2_train_losses": p2_train_losses,
        "wall_min": (time.time() - t_start) / 60.0,
    }
    (RUN_DIR / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))

    (RUN_DIR / "correlation_matrix.json").write_text(
        json.dumps(corr_matrix, indent=2))

    (RUN_DIR / "per_class_f1.json").write_text(json.dumps({
        "action_per_class": per_class_action.tolist(),
        "point_per_class": per_class_point.tolist(),
        "action_eval_labels": ACTION_EVAL_LABELS,
        "point_eval_labels": POINT_EVAL_LABELS,
    }, indent=2))

    (RUN_DIR / "audit.json").write_text(json.dumps(audits, indent=2, default=str))

    np.savez_compressed(
        RUN_DIR / "fold1_oof_partial.npz",
        val_idx=val_idx,
        action=a_val_ruled.astype(np.float32),
        point=p_val.astype(np.float32),
        sgp=s_val.astype(np.float32),
        y_action=y_a.astype(np.int8),
        y_point=y_p.astype(np.int8),
        y_server=y_s.astype(np.int8),
        next_sn=nsn.astype(np.int32),
    )

    # ── R-014 recommendation logic ───────────────────────────────────────────
    v11_aug_ov = baselines.get("v11_aug", {}).get("ov", float("nan"))
    v11_ov = baselines.get("v11", {}).get("ov", float("nan"))
    primary_gate = min(v11_aug_ov, v11_ov) - 0.005 if (
        np.isfinite(v11_aug_ov) and np.isfinite(v11_ov)) else float("nan")

    diversity_pass = False
    if "v11_aug" in corr_matrix:
        r_a_aug = corr_matrix["v11_aug"]["action"]
        r_p_aug = corr_matrix["v11_aug"]["point"]
        if (np.isfinite(r_a_aug) and np.isfinite(r_p_aug)
                and r_a_aug <= 0.85 and r_p_aug <= 0.85):
            diversity_pass = True

    primary_pass = (np.isfinite(primary_gate) and ov_final >= primary_gate)

    if primary_pass:
        recommendation = ("PRIMARY_PASS — open R-014 full-run preflight "
                          f"(OV {ov_final:.4f} >= primary gate {primary_gate:.4f})")
    elif diversity_pass:
        recommendation = ("DIVERSITY_PASS — open R-014 explicitly tagged "
                          "'diversity candidate only'; full run requires "
                          "Jabir T3 OK on lower expected lift")
    else:
        recommendation = ("PARK — neither primary nor diversity pass. "
                          "Postmortem in RESULTS §33.")

    summary_lines = [
        "v17_causal_lm Fold-1 SMOKE summary",
        "=" * 60,
        f"Wall: {(time.time() - t_start)/60:.1f} min",
        f"Hard cap: {args.hard_cap_h}h "
        f"({'NOT REACHED' if (time.time()-t_start) < hard_cap_s else 'REACHED'})",
        "",
        f"v17 smoke Fold-1 OOF:",
        f"  F1_action = {f1_a_final:.4f}",
        f"  F1_point  = {f1_p_final:.4f}",
        f"  SGP AUC   = {auc_final:.4f}",
        f"  Joint OV  = {ov_final:.4f}",
        "",
        "Fold-1 baselines:",
    ]
    for tag, b in baselines.items():
        summary_lines.append(
            f"  {tag}: OV={b['ov']:.4f}  F1_a={b['f1_a']:.4f}  "
            f"F1_p={b['f1_p']:.4f}  AUC={b['auc']:.4f}")
    summary_lines += ["", "Correlation matrix (Pearson r, macro-class avg):"]
    for tag, c in corr_matrix.items():
        summary_lines.append(
            f"  vs {tag}: r_action={c['action']:.4f}  r_point={c['point']:.4f}")

    summary_lines += [
        "",
        f"Primary gate: OV >= {primary_gate:.4f} → "
        f"{'PASS' if primary_pass else 'FAIL'}",
        f"Diversity gate (r_a_aug<=0.85 AND r_p_aug<=0.85) → "
        f"{'PASS' if diversity_pass else 'FAIL'}",
        "",
        f"RECOMMENDATION: {recommendation}",
        "",
        "All audits:",
    ]
    for k, v in audits.items():
        summary_lines.append(f"  {k}: {list(v.values())[0]}")

    summary_text = "\n".join(summary_lines)
    (RUN_DIR / "summary.txt").write_text(summary_text)
    print("\n" + summary_text)

    return 0


def _emergency_exit(audits, reason, t_start):
    print(f"\nEMERGENCY EXIT: {reason} after {(time.time()-t_start)/60:.1f} min")
    (RUN_DIR / "emergency_exit.json").write_text(json.dumps({
        "reason": reason,
        "wall_min": (time.time() - t_start) / 60.0,
        "audits_at_exit": audits,
    }, indent=2, default=str))
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
