"""R-066 Path B Causal LM v1 — multi-position objective trainer.

Per STRATEGY.md §9 and R-066 design (Codex APPROVE_WITH_FIXES pending):

  Architecture: causal Transformer decoder, d=192, 4 layers, 4 heads,
                FF=768, dropout=0.1. Per-position output heads for
                action (19-class incl serves) + point (10-class) + server (binary).

  Loss: multi-position — sum CE+CE+BCE across positions 2..N per rally.
        Server BCE masked at aug positions (server_true == -1).
        Position 1 (the serve) has no causal context → no loss.

  Pre-training: visible test action+point shots can be added as
                supervised training samples (SGP masked).

  Inference: per rally, run causal forward on visible shots 1..N,
             extract the (N+1)-th position's predicted distribution.

USAGE (Fold-1 smoke):
    python -u src/train_causal_lm_v1.py --smoke --max-folds 1 --epochs 25 \\
        --tag v22_causal_lm_v1_smoke --seed 42 \\
        --include-old-test data/test.csv \\
        --include-test-history data/test_history_pairs_new.parquet \\
        --test-path data/test_new.csv

USAGE (full 5-fold, post-smoke):
    python -u src/train_causal_lm_v1.py --epochs 40 --tag v22_causal_lm_v1 \\
        --seed 42 --include-old-test data/test.csv \\
        --include-test-history data/test_history_pairs_new.parquet \\
        --test-path data/test_new.csv
"""
import argparse
import gc
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data, STRIKE_ID_MAP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_ACTION_FULL = 19    # includes serves (15-18); only valid at strikeNumber=1
N_ACTION_TRAIN = 15   # macro-F1 evaluation classes (0-14)
N_POINT = 10
ACTION_EVAL_LABELS = list(range(15))
POINT_EVAL_LABELS = list(range(10))

# Per-shot categoricals (name, n_classes, embed_dim)
SHOT_CAT_CFG = [
    ("strikeId",   5,  8),
    ("handId",     3,  6),
    ("strengthId", 4,  6),
    ("spinId",     6, 10),
    ("pointId",   10, 16),
    ("actionId",  19, 24),
    ("positionId", 4,  6),
]
SHOT_CAT_DIM = sum(e for _, _, e in SHOT_CAT_CFG)  # 76

# Per-shot numericals (4 features)
N_NUM_FEATURES = 4

# Sequence cap (covers 99% of train rallies; max train rally = 52)
MAX_SEQ_LEN = 40


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_rally_samples(raw_df: pd.DataFrame, is_aug: bool = False,
                         n_players: int = 200) -> list:
    """Build one sample per RALLY (not per target). Each sample carries the
    full sequence of N shots and labels for positions 2..N (predict each
    non-serve shot from its causal prefix).

    For aug rows (test_history aug parquet), serverGetPoint is -1 sentinel.
    """
    samples = []
    rallies = raw_df.groupby("rally_uid", sort=False)
    for uid, grp in rallies:
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        n = len(grp)
        if n < 2:
            continue   # need at least 2 shots: one prefix + one target

        match_id = grp["match"].iloc[0]
        strike_id  = grp["strikeId"].values.astype(np.int8)
        hand_id    = grp["handId"].values.astype(np.int8)
        strength   = grp["strengthId"].values.astype(np.int8)
        spin       = grp["spinId"].values.astype(np.int8)
        point_id   = grp["pointId"].values.astype(np.int8)
        action_id  = grp["actionId"].values.astype(np.int8)
        pos_id     = grp["positionId"].values.astype(np.int8)
        sn         = grp["strikeNumber"].values.astype(np.float32)
        score_s    = grp["scoreSelf"].values.astype(np.float32)
        score_o    = grp["scoreOther"].values.astype(np.float32)
        server_gp  = grp["serverGetPoint"].values.astype(np.int8)
        pid_self   = int(grp["gamePlayerId"].iloc[0])
        pid_other  = int(grp["gamePlayerOtherId"].iloc[0])
        sex        = int(grp["sex"].iloc[0])
        num_game   = int(grp["numberGame"].iloc[0])

        # Truncate to MAX_SEQ_LEN
        k = min(n, MAX_SEQ_LEN)
        cat_seq = np.stack([
            strike_id[:k], hand_id[:k], strength[:k], spin[:k],
            point_id[:k], action_id[:k], pos_id[:k]
        ], axis=1).astype(np.int64)  # (k, 7)

        num_seq = np.stack([
            sn[:k] / 40.0,
            score_s[:k] / 11.0,
            score_o[:k] / 11.0,
            (score_s[:k] - score_o[:k]) / 22.0,
        ], axis=1).astype(np.float32)  # (k, 4)

        # Targets at each position t in 1..k-1 (predict from prefix 0..t-1)
        # Position 0 (the serve) has no causal context — no loss.
        # For training rallies: positions 1..k-1 have valid targets.
        # action_id is the action AT THAT POSITION; we'll predict it at position t-1.
        # For multi-position loss: at decoder output position t (0-indexed), predict
        # action[t+1] from prefix 0..t. So output positions 0..k-2 produce predictions
        # for shots 1..k-1.

        # Label arrays for positions 0..k-1; position t's label is action_id[t].
        # Multi-position loss masks position 0 (no prediction needed for serve from itself)
        y_action_seq = np.where(
            action_id[:k] >= N_ACTION_FULL, 0, action_id[:k]
        ).astype(np.int64)  # (k,)
        y_point_seq = point_id[:k].astype(np.int64)  # (k,)

        # Server label: rally-level constant in real train, -1 in aug
        y_server_seq = server_gp[:k].astype(np.int64)  # (k,)
        # If aug, set to -1 sentinel for masking
        if is_aug:
            y_server_seq = np.full_like(y_server_seq, -1)

        samples.append({
            "cat_seq":   cat_seq,
            "num_seq":   num_seq,
            "y_action":  y_action_seq,
            "y_point":   y_point_seq,
            "y_server":  y_server_seq,
            "pid_self":  min(pid_self,  n_players - 1),
            "pid_other": min(pid_other, n_players - 1),
            "sex":       sex,
            "num_game":  num_game,
            "n_shots":   k,
            "rally_uid": uid,
            "match_id":  match_id,
            "is_aug":    int(is_aug),
        })
    return samples


def build_test_samples(test_df: pd.DataFrame, n_players: int = 200) -> list:
    """Test rally → 1 sample. We use ALL visible shots as context and extract
    predictions for the (N+1)-th virtual position. Labels are placeholders.
    """
    samples = []
    rallies = test_df.groupby("rally_uid", sort=False)
    for uid, grp in rallies:
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        n = len(grp)
        if n < 1:
            continue
        match_id = grp["match"].iloc[0]
        strike_id  = grp["strikeId"].values.astype(np.int8)
        hand_id    = grp["handId"].values.astype(np.int8)
        strength   = grp["strengthId"].values.astype(np.int8)
        spin       = grp["spinId"].values.astype(np.int8)
        point_id   = grp["pointId"].values.astype(np.int8)
        action_id  = grp["actionId"].values.astype(np.int8)
        pos_id     = grp["positionId"].values.astype(np.int8)
        sn         = grp["strikeNumber"].values.astype(np.float32)
        score_s    = grp["scoreSelf"].values.astype(np.float32)
        score_o    = grp["scoreOther"].values.astype(np.float32)
        pid_self   = int(grp["gamePlayerId"].iloc[0])
        pid_other  = int(grp["gamePlayerOtherId"].iloc[0])
        sex        = int(grp["sex"].iloc[0])
        num_game   = int(grp["numberGame"].iloc[0])

        k = min(n, MAX_SEQ_LEN)
        cat_seq = np.stack([
            strike_id[:k], hand_id[:k], strength[:k], spin[:k],
            point_id[:k], action_id[:k], pos_id[:k]
        ], axis=1).astype(np.int64)
        num_seq = np.stack([
            sn[:k] / 40.0,
            score_s[:k] / 11.0,
            score_o[:k] / 11.0,
            (score_s[:k] - score_o[:k]) / 22.0,
        ], axis=1).astype(np.float32)

        samples.append({
            "cat_seq":   cat_seq,
            "num_seq":   num_seq,
            "y_action":  np.zeros(k, dtype=np.int64),
            "y_point":   np.zeros(k, dtype=np.int64),
            "y_server":  np.full(k, -1, dtype=np.int64),  # sentinel: never read
            "pid_self":  min(pid_self,  n_players - 1),
            "pid_other": min(pid_other, n_players - 1),
            "sex":       sex,
            "num_game":  num_game,
            "n_shots":   k,
            "rally_uid": uid,
            "match_id":  match_id,
            "is_aug":    0,
        })
    return samples


class RallySeqDataset(Dataset):
    def __init__(self, samples: list, max_len: int = MAX_SEQ_LEN):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ml = self.max_len
        k = min(s["n_shots"], ml)

        cat = np.zeros((ml, 7), dtype=np.int64)
        num = np.zeros((ml, N_NUM_FEATURES), dtype=np.float32)
        y_a = np.full(ml, -100, dtype=np.int64)   # -100 = ignore index for CE
        y_p = np.full(ml, -100, dtype=np.int64)
        y_s = np.full(ml, -1,   dtype=np.int64)
        pad_mask = np.ones(ml, dtype=bool)        # True = padded

        cat[:k] = s["cat_seq"][:k]
        num[:k] = s["num_seq"][:k]
        y_a[:k] = s["y_action"][:k]
        y_p[:k] = s["y_point"][:k]
        y_s[:k] = s["y_server"][:k]
        pad_mask[:k] = False

        return {
            "cat":      torch.from_numpy(cat),
            "num":      torch.from_numpy(num),
            "y_action": torch.from_numpy(y_a),
            "y_point":  torch.from_numpy(y_p),
            "y_server": torch.from_numpy(y_s),
            "pad_mask": torch.from_numpy(pad_mask),
            "n_shots":  torch.tensor(k, dtype=torch.long),
            "is_aug":   torch.tensor(s["is_aug"], dtype=torch.long),
            "sex":      torch.tensor(s["sex"], dtype=torch.long),
            "num_game": torch.tensor(s["num_game"], dtype=torch.long),
            "pid_self":  torch.tensor(s["pid_self"],  dtype=torch.long),
            "pid_other": torch.tensor(s["pid_other"], dtype=torch.long),
            "rally_uid": s["rally_uid"],
        }


# ─── Model ────────────────────────────────────────────────────────────────────

def sinusoidal_pos_embedding(max_len: int, d: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding."""
    pe = torch.zeros(max_len, d)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe   # (max_len, d)


class CausalRallyLM(nn.Module):
    """Causal Transformer decoder for rally sequences.

    For each position t in the sequence, outputs:
      - action logits (19-class)
      - point logits (10-class)
      - server logit (binary)
    Trained with multi-position loss summed across valid positions.
    """
    def __init__(self, d_model=192, n_heads=4, n_layers=4,
                 dropout=0.1, n_players=200, max_len=MAX_SEQ_LEN,
                 use_pid_emb=False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.use_pid_emb = use_pid_emb

        # Per-shot categorical embeddings
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(n_cls, e_dim) for _, n_cls, e_dim in SHOT_CAT_CFG
        ])
        # Per-shot numerical projection (4 nums → 16)
        self.num_proj = nn.Sequential(nn.Linear(N_NUM_FEATURES, 16), nn.GELU())
        # Combined → d_model
        self.shot_proj = nn.Sequential(
            nn.Linear(SHOT_CAT_DIM + 16, d_model),
            nn.LayerNorm(d_model),
        )

        # Sinusoidal positional encoding (registered as buffer, not parameter)
        self.register_buffer(
            "pos_emb", sinusoidal_pos_embedding(max_len, d_model),
            persistent=False,
        )

        # Rally-level context: sex, numberGame embeddings → projected to d_model
        self.sex_emb = nn.Embedding(3, 8)
        self.game_emb = nn.Embedding(8, 8)
        self.ctx_proj = nn.Linear(16, d_model)

        # Optional player embeddings (DISABLED for smoke per Codex sanity check #1)
        if self.use_pid_emb:
            self.player_emb = nn.Embedding(n_players + 5, 16)
            self.player_proj = nn.Linear(32, d_model)

        # Causal Transformer decoder layer (we use TransformerEncoderLayer with
        # causal mask — semantically the same as a self-attention-only decoder).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False,
        )

        # Per-position output heads
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_ACTION_FULL),
        )
        self.point_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, N_POINT),
        )
        self.server_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _build_causal_mask(self, L: int, device) -> torch.Tensor:
        # True = masked (cannot attend)
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device),
                           diagonal=1)

    def forward(self, cat, num, sex, num_game, pid_self, pid_other, pad_mask):
        """
        cat       : (B, L, 7) int64
        num       : (B, L, 4) float32
        sex       : (B,)      int64
        num_game  : (B,)      int64
        pid_self  : (B,)      int64
        pid_other : (B,)      int64
        pad_mask  : (B, L)    bool   True=padding

        Returns:
            action_logits : (B, L, 19)
            point_logits  : (B, L, 10)
            server_logits : (B, L)   (squeezed)
        """
        B, L, _ = cat.shape

        # Per-shot embeddings
        embeds = [emb(cat[:, :, i]) for i, emb in enumerate(self.cat_embeds)]
        cat_emb = torch.cat(embeds, dim=-1)        # (B, L, 76)
        num_emb = self.num_proj(num)                # (B, L, 16)
        x = self.shot_proj(torch.cat([cat_emb, num_emb], dim=-1))   # (B, L, d)

        # Positional encoding (broadcast)
        x = x + self.pos_emb[:L].unsqueeze(0)

        # Rally-level context (broadcast over positions)
        ctx = torch.cat([self.sex_emb(sex), self.game_emb(num_game)], dim=-1)
        ctx = self.ctx_proj(ctx).unsqueeze(1)        # (B, 1, d)
        x = x + ctx

        if self.use_pid_emb:
            pids = torch.cat([self.player_emb(pid_self),
                              self.player_emb(pid_other)], dim=-1)
            pids = self.player_proj(pids).unsqueeze(1)  # (B, 1, d)
            x = x + pids

        # Causal mask
        causal_mask = self._build_causal_mask(L, x.device)
        # Transformer forward (note: TransformerEncoder supports both attn_mask + key_padding_mask)
        out = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
        )   # (B, L, d)

        # Per-position outputs
        action_logits = self.action_head(out)        # (B, L, 19)
        point_logits = self.point_head(out)          # (B, L, 10)
        server_logits = self.server_head(out).squeeze(-1)   # (B, L)

        return action_logits, point_logits, server_logits


def multi_position_loss(action_logits, point_logits, server_logits,
                        y_action, y_point, y_server, pad_mask, is_aug,
                        alpha=0.4, beta=0.4, gamma=0.2):
    """Multi-position autoregressive next-shot prediction loss.

    FIX (R-066 v3, 2026-05-23): standard causal-LM label shift.
    output[t] is trained to predict shot[t+1] (not shot[t] which would be a
    trivial copy from the input embedding at position t under causal mask).

    Mechanically:
      - Use action_logits[:, :-1] (positions 0..L-2 — predictions FROM each prefix)
      - Compare against y_action[:, 1:] (positions 1..L-1 — NEXT shot's labels)
      - Drops the last position output (no next-shot label exists for it)
      - Position 0's output IS valid now: predict shot[1] from shot[0] (the serve)

    Server head: SGP is rally-constant; shifting still produces the correct
    prediction target (same value at every position). We shift for consistency.

    Validity mask aligns with the SHIFTED label positions (positions 1..L-1):
      - pad_mask shifted: positions whose NEXT POSITION is real (not padding)
      - y != -100 / -1 sentinels still apply
      - Aug rows (is_aug=1) excluded from server BCE only.

    Returns mean per-rally loss (averaged over valid positions, then over batch).
    """
    B, L, _ = action_logits.shape

    # Shift: predict label[t+1] from output[t]
    # logits[:, :-1] aligned with y[:, 1:] gives length L-1 sequences
    act_logits_shift = action_logits[:, :-1, :].contiguous()        # (B, L-1, N_ACTION_FULL)
    pt_logits_shift = point_logits[:, :-1, :].contiguous()           # (B, L-1, N_POINT)
    srv_logits_shift = server_logits[:, :-1].contiguous()             # (B, L-1)
    y_action_shift = y_action[:, 1:].contiguous()                     # (B, L-1)
    y_point_shift = y_point[:, 1:].contiguous()                       # (B, L-1)
    y_server_shift = y_server[:, 1:].contiguous()                     # (B, L-1)
    # A position t's NEXT POSITION (t+1) is padding if pad_mask[t+1] is True
    next_pad = pad_mask[:, 1:].contiguous()                           # (B, L-1)

    valid_mask = (~next_pad)                                          # (B, L-1)
    valid_act = valid_mask & (y_action_shift != -100)
    valid_pt = valid_mask & (y_point_shift != -100)
    # Server: also exclude aug rows and -1 sentinels (rally-level constant)
    aug_mask = is_aug.unsqueeze(1).expand(-1, L - 1).bool()           # (B, L-1)
    valid_srv = valid_mask & (y_server_shift != -1) & (~aug_mask)

    # CE for action / point — ignore_index handled by mask
    act_loss = F.cross_entropy(
        act_logits_shift.reshape(-1, N_ACTION_FULL),
        y_action_shift.reshape(-1),
        ignore_index=-100, reduction="none",
    ).reshape(B, L - 1)
    pt_loss = F.cross_entropy(
        pt_logits_shift.reshape(-1, N_POINT),
        y_point_shift.reshape(-1),
        ignore_index=-100, reduction="none",
    ).reshape(B, L - 1)

    act_loss = (act_loss * valid_act.float()).sum() / valid_act.float().sum().clamp_min(1.0)
    pt_loss = (pt_loss * valid_pt.float()).sum() / valid_pt.float().sum().clamp_min(1.0)

    # BCE for server (masked, shifted to align with action/point)
    if valid_srv.any():
        srv_pos = srv_logits_shift[valid_srv]
        srv_tgt = y_server_shift[valid_srv].float()
        srv_loss = F.binary_cross_entropy_with_logits(srv_pos, srv_tgt)
    else:
        srv_loss = torch.tensor(0.0, device=action_logits.device)

    total = alpha * act_loss + beta * pt_loss + gamma * srv_loss
    return total, act_loss.item(), pt_loss.item(), srv_loss.item()


# ─── Train / eval ─────────────────────────────────────────────────────────────

def collate_batch(batch):
    return {
        "cat":       torch.stack([b["cat"] for b in batch]),
        "num":       torch.stack([b["num"] for b in batch]),
        "y_action":  torch.stack([b["y_action"] for b in batch]),
        "y_point":   torch.stack([b["y_point"] for b in batch]),
        "y_server":  torch.stack([b["y_server"] for b in batch]),
        "pad_mask":  torch.stack([b["pad_mask"] for b in batch]),
        "n_shots":   torch.stack([b["n_shots"] for b in batch]),
        "is_aug":    torch.stack([b["is_aug"] for b in batch]),
        "sex":       torch.stack([b["sex"] for b in batch]),
        "num_game":  torch.stack([b["num_game"] for b in batch]),
        "pid_self":  torch.stack([b["pid_self"] for b in batch]),
        "pid_other": torch.stack([b["pid_other"] for b in batch]),
        "rally_uid": [b["rally_uid"] for b in batch],
    }


def evaluate_oof(model, loader, device):
    """Return per-position predictions; we extract the LAST valid position
    per rally for OOF metric computation."""
    model.eval()
    oof_action = []
    oof_point = []
    oof_srv = []
    oof_rally_uid = []
    oof_y_action = []
    oof_y_point = []
    oof_y_srv = []
    oof_nsn = []

    with torch.no_grad():
        for batch in loader:
            for k in ("cat", "num", "y_action", "y_point", "y_server",
                       "pad_mask", "is_aug", "sex", "num_game", "pid_self", "pid_other"):
                batch[k] = batch[k].to(device)
            act_logits, pt_logits, srv_logits = model(
                batch["cat"], batch["num"], batch["sex"], batch["num_game"],
                batch["pid_self"], batch["pid_other"], batch["pad_mask"],
            )
            act_probs = F.softmax(act_logits, dim=-1)
            pt_probs = F.softmax(pt_logits, dim=-1)
            srv_probs = torch.sigmoid(srv_logits)

            n_shots = batch["n_shots"]
            B = act_probs.size(0)
            for b in range(B):
                k = int(n_shots[b].item())
                # We treat each (rally, position) as an OOF sample: extract per-position
                # predictions at positions 1..k-1 (i.e., the predictions FROM prefix 0..t-1
                # for shot t). Position index in the model output: predicted_shot_t = output[t-1].
                # For the rally's OOF metric we use the LAST predicted position (t = k-1)
                # because that mirrors the test-time inference pattern.
                if k < 2:
                    continue
                last_t = k - 1
                # Decoder output position `last_t - 1` predicts shot `last_t` from prefix
                # Actually under standard causal LM, position t's output predicts shot t+1.
                # But here we used position t = shot index (model sees shots 0..t-1 to predict shot t),
                # we just take the output AT position last_t (which has seen shots 0..last_t,
                # and we want predictions for the NEXT shot... but our labels are co-indexed).
                # For simplicity: extract output at position last_t which encodes
                # "having seen up to shot last_t" → can predict shot last_t+1 (virtual).
                # For OOF metric: take output at position last_t which sees prefix 0..last_t-1
                # (Wait — under causal mask, position t sees positions 0..t INCLUSIVE.
                # So output[last_t] uses prefix 0..last_t INCLUSIVE which would include the target.
                # To predict shot t from prefix 0..t-1 we need output[t-1].)
                # Let's take output[last_t - 1] (the prediction for shot last_t given prefix 0..last_t-1).
                pred_pos = max(last_t - 1, 0)
                oof_action.append(act_probs[b, pred_pos].cpu().numpy())
                oof_point.append(pt_probs[b, pred_pos].cpu().numpy())
                oof_srv.append(float(srv_probs[b, pred_pos].cpu().numpy()))
                oof_rally_uid.append(batch["rally_uid"][b])
                oof_y_action.append(int(batch["y_action"][b, last_t].item()))
                oof_y_point.append(int(batch["y_point"][b, last_t].item()))
                oof_y_srv.append(int(batch["y_server"][b, last_t].item()))
                oof_nsn.append(last_t + 1)   # 1-indexed strikeNumber of target
    return {
        "oof_act":  np.stack(oof_action) if oof_action else np.zeros((0, N_ACTION_FULL)),
        "oof_pt":   np.stack(oof_point) if oof_point else np.zeros((0, N_POINT)),
        "oof_srv":  np.array(oof_srv, dtype=np.float32),
        "rally_uid": np.array(oof_rally_uid),
        "y_act":    np.array(oof_y_action, dtype=np.int64),
        "y_pt":     np.array(oof_y_point, dtype=np.int64),
        "y_srv":    np.array(oof_y_srv, dtype=np.int64),
        "nsn":      np.array(oof_nsn, dtype=np.int32),
    }


def predict_test(model, samples, device, max_len=MAX_SEQ_LEN, batch_size=32):
    """For each test rally, output the (N+1)-th position prediction.
    Implementation: take model output at the LAST visible position; that
    represents "given prefix 0..N-1, predict next"."""
    ds = RallySeqDataset(samples, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=0, collate_fn=collate_batch)
    test_act = []
    test_pt = []
    test_srv = []
    test_uid = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            for k in ("cat", "num", "pad_mask", "is_aug", "sex",
                       "num_game", "pid_self", "pid_other"):
                batch[k] = batch[k].to(device)
            act_logits, pt_logits, srv_logits = model(
                batch["cat"], batch["num"], batch["sex"], batch["num_game"],
                batch["pid_self"], batch["pid_other"], batch["pad_mask"],
            )
            act_probs = F.softmax(act_logits, dim=-1)
            pt_probs = F.softmax(pt_logits, dim=-1)
            srv_probs = torch.sigmoid(srv_logits)
            n_shots = batch["n_shots"]
            B = act_probs.size(0)
            for b in range(B):
                k_b = int(n_shots[b].item())
                # Take output at last visible position to predict next-shot
                pred_pos = max(k_b - 1, 0)
                test_act.append(act_probs[b, pred_pos].cpu().numpy())
                test_pt.append(pt_probs[b, pred_pos].cpu().numpy())
                test_srv.append(float(srv_probs[b, pred_pos].cpu().numpy()))
                test_uid.append(batch["rally_uid"][b])
    return (np.stack(test_act), np.stack(test_pt),
            np.array(test_srv, dtype=np.float32), np.array(test_uid))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: 1 fold, fewer epochs")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--max-folds", type=int, default=0,
                        help="If >0, run only this many folds with full epochs (smoke pattern)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience on val loss")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-pid-emb", action="store_true",
                        help="Enable pid_self/pid_other embeddings (DISABLED by default per R-066 Codex sanity check #1)")
    parser.add_argument("--tag", type=str, default="v22_causal_lm_v1")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--include-old-test", type=str, default=None,
                        help="Path to OLD test.csv to add as supervised training data")
    parser.add_argument("--include-test-history", type=str, default=None,
                        help="Path to test_history_pairs parquet for LM pre-training-style supervision")
    args = parser.parse_args()

    n_folds = 1 if args.smoke else args.folds
    set_seed(args.seed)

    test_path = args.test_path or TEST_PATH
    print("=" * 70)
    print(f" CAUSAL LM v1 — {'SMOKE' if args.smoke else 'FULL'}  seed={args.seed}")
    print(f"  d_model={args.d_model}  n_heads={args.n_heads}  n_layers={args.n_layers}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  device={DEVICE}  use_pid_emb={args.use_pid_emb}")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(test_path)
    if args.include_old_test:
        old_test = pd.read_csv(args.include_old_test)
        n_before = len(raw_train)
        required_cols = list(raw_train.columns)
        missing = [c for c in required_cols if c not in old_test.columns]
        if missing:
            raise ValueError(f"old test missing columns: {missing}")
        raw_train = pd.concat([raw_train, old_test[required_cols]], ignore_index=True)
        print(f"  [include-old-test] +{len(raw_train) - n_before} rows from {args.include_old_test}")

    train_df, test_df, _ = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1

    # Optionally append test-history aug parquet rows for LM supervision
    aug_df = None
    if args.include_test_history:
        aug_raw = pd.read_parquet(args.include_test_history)
        assert (aug_raw["serverGetPoint"] == -1).all(), "aug parquet SGP not -1 sentinel"
        aug_raw = aug_raw.copy()
        # Apply same cleaning
        aug_raw["strikeId"] = aug_raw["strikeId"].map(STRIKE_ID_MAP).fillna(0).astype(int)
        # Player remap: use train_df's encoding (we re-derive from clean_data)
        train_pids = pd.concat([
            train_df["gamePlayerId"], train_df["gamePlayerOtherId"]
        ]).unique()
        pid_map = {int(p): int(p) for p in train_pids}   # identity if already remapped
        aug_raw["gamePlayerId"] = aug_raw["gamePlayerId"].map(pid_map).fillna(-1).astype(int)
        aug_raw["gamePlayerOtherId"] = aug_raw["gamePlayerOtherId"].map(pid_map).fillna(-1).astype(int)
        aug_raw["numberGame"] = aug_raw["numberGame"].clip(upper=7)
        aug_df = aug_raw
        print(f"  [include-test-history] {len(aug_df)} aug rows from {args.include_test_history}")

    # Build samples
    print("Building rally samples...")
    t0 = time.time()
    train_samples = build_rally_samples(train_df, is_aug=False)
    print(f"  Train rallies (sup): {len(train_samples)}  [{time.time()-t0:.1f}s]")
    aug_samples = []
    if aug_df is not None:
        aug_samples = build_rally_samples(aug_df, is_aug=True)
        print(f"  Aug rallies (LM sup): {len(aug_samples)}")
    test_samples = build_test_samples(test_df)
    print(f"  Test rallies: {len(test_samples)}")

    # Build per-rally fold assignment for ORIGINAL train rallies only
    rally_to_match = {s["rally_uid"]: s["match_id"] for s in train_samples}
    rally_uids = np.array([s["rally_uid"] for s in train_samples])
    matches = np.array([rally_to_match[u] for u in rally_uids])
    gkf = GroupKFold(n_splits=max(n_folds, 2))
    splits = list(gkf.split(rally_uids, groups=matches))
    if args.smoke:
        splits = splits[:1]
    elif args.max_folds and args.max_folds > 0:
        splits = splits[:args.max_folds]

    # OOF buffers — per-rally OOF (one row per train rally)
    oof_act_all = np.zeros((len(train_samples), N_ACTION_FULL), dtype=np.float32)
    oof_pt_all = np.zeros((len(train_samples), N_POINT), dtype=np.float32)
    oof_srv_all = np.zeros(len(train_samples), dtype=np.float32)
    oof_mask = np.zeros(len(train_samples), dtype=bool)
    rally_uid_to_idx = {u: i for i, u in enumerate(rally_uids)}

    test_act_acc = np.zeros((len(test_samples), N_ACTION_FULL), dtype=np.float32)
    test_pt_acc = np.zeros((len(test_samples), N_POINT), dtype=np.float32)
    test_srv_acc = np.zeros(len(test_samples), dtype=np.float32)
    n_folds_run = 0

    for fold, (tr_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}\n  FOLD {fold+1}/{len(splits)}\n{'='*60}")
        t_fold = time.time()

        # Training pool: tr_idx of train_samples + ALL aug_samples (LM pre-training mix)
        tr_samples_fold = [train_samples[i] for i in tr_idx]
        if aug_samples:
            tr_samples_fold = tr_samples_fold + aug_samples
        val_samples_fold = [train_samples[i] for i in val_idx]
        print(f"  train: {len(tr_samples_fold)}  val: {len(val_samples_fold)}")

        tr_ds = RallySeqDataset(tr_samples_fold)
        val_ds = RallySeqDataset(val_samples_fold)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, collate_fn=collate_batch)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, collate_fn=collate_batch)

        model = CausalRallyLM(
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            dropout=args.dropout, use_pid_emb=args.use_pid_emb,
        ).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model params: {n_params:,}")

        best_val_loss = float("inf")
        bad_epochs = 0
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            tr_loss_acc = 0.0
            n_b = 0
            t_ep = time.time()
            for batch in tr_loader:
                for k in ("cat", "num", "y_action", "y_point", "y_server",
                           "pad_mask", "is_aug", "sex", "num_game", "pid_self", "pid_other"):
                    batch[k] = batch[k].to(DEVICE)
                act_logits, pt_logits, srv_logits = model(
                    batch["cat"], batch["num"], batch["sex"], batch["num_game"],
                    batch["pid_self"], batch["pid_other"], batch["pad_mask"],
                )
                loss, _, _, _ = multi_position_loss(
                    act_logits, pt_logits, srv_logits,
                    batch["y_action"], batch["y_point"], batch["y_server"],
                    batch["pad_mask"], batch["is_aug"],
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_loss_acc += float(loss.item())
                n_b += 1

            # Val loss
            model.eval()
            val_loss_acc = 0.0
            n_vb = 0
            val_act_loss = 0.0
            val_pt_loss = 0.0
            val_srv_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    for k in ("cat", "num", "y_action", "y_point", "y_server",
                               "pad_mask", "is_aug", "sex", "num_game", "pid_self", "pid_other"):
                        batch[k] = batch[k].to(DEVICE)
                    act_logits, pt_logits, srv_logits = model(
                        batch["cat"], batch["num"], batch["sex"], batch["num_game"],
                        batch["pid_self"], batch["pid_other"], batch["pad_mask"],
                    )
                    loss, al, pl, sl = multi_position_loss(
                        act_logits, pt_logits, srv_logits,
                        batch["y_action"], batch["y_point"], batch["y_server"],
                        batch["pad_mask"], batch["is_aug"],
                    )
                    val_loss_acc += float(loss.item())
                    val_act_loss += al
                    val_pt_loss += pl
                    val_srv_loss += sl
                    n_vb += 1
            val_loss = val_loss_acc / max(n_vb, 1)
            tr_loss = tr_loss_acc / max(n_b, 1)
            print(f"  Epoch {epoch+1:>2}/{args.epochs}  "
                  f"tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"(act {val_act_loss/n_vb:.3f} pt {val_pt_loss/n_vb:.3f} srv {val_srv_loss/n_vb:.3f})  "
                  f"[{time.time()-t_ep:.1f}s]")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                bad_epochs = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"  Early stop at epoch {epoch+1} (no val improvement in {args.patience} epochs)")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # OOF on val
        oof = evaluate_oof(model, val_loader, DEVICE)
        for i, uid in enumerate(oof["rally_uid"]):
            if uid in rally_uid_to_idx:
                idx = rally_uid_to_idx[uid]
                oof_act_all[idx] = oof["oof_act"][i]
                oof_pt_all[idx] = oof["oof_pt"][i]
                oof_srv_all[idx] = oof["oof_srv"][i]
                oof_mask[idx] = True

        # Fold OV (val rallies only, applied action-rule)
        ya_v = np.array([train_samples[i]["y_action"][train_samples[i]["n_shots"]-1]
                          for i in val_idx
                          if train_samples[i]["rally_uid"] in [u for u in oof["rally_uid"]]],
                         dtype=np.int64)
        ya_v = np.where(ya_v >= N_ACTION_TRAIN, 0, ya_v)
        pred_a = oof["oof_act"][:, :N_ACTION_TRAIN].argmax(axis=1)
        f1_a = f1_score(oof["y_act"], pred_a, labels=ACTION_EVAL_LABELS,
                         average="macro", zero_division=0)
        pred_p = oof["oof_pt"].argmax(axis=1)
        f1_p = f1_score(oof["y_pt"], pred_p, labels=POINT_EVAL_LABELS,
                         average="macro", zero_division=0)
        srv_mask = oof["y_srv"] >= 0
        if srv_mask.sum() > 0 and len(np.unique(oof["y_srv"][srv_mask])) > 1:
            auc = roc_auc_score(oof["y_srv"][srv_mask], oof["oof_srv"][srv_mask])
        else:
            auc = 0.5
        fold_ov = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
        print(f"  FOLD {fold+1} OV={fold_ov:.4f}  F1_a={f1_a:.4f}  F1_p={f1_p:.4f}  AUC={auc:.4f}  [{time.time()-t_fold:.1f}s]")

        # Predict test (accumulate across folds)
        tp_act, tp_pt, tp_srv, tp_uid = predict_test(model, test_samples, DEVICE,
                                                       batch_size=args.batch_size)
        test_act_acc += tp_act
        test_pt_acc += tp_pt
        test_srv_acc += tp_srv
        n_folds_run += 1

        del model, best_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average test predictions across folds
    if n_folds_run > 0:
        test_act_acc /= n_folds_run
        test_pt_acc /= n_folds_run
        test_srv_acc /= n_folds_run

    # Final OOF metrics
    if oof_mask.any():
        y_a_all = np.array([s["y_action"][s["n_shots"]-1] for s in train_samples])
        y_a_all = np.where(y_a_all >= N_ACTION_TRAIN, 0, y_a_all)
        y_p_all = np.array([s["y_point"][s["n_shots"]-1] for s in train_samples])
        y_s_all = np.array([s["y_server"][s["n_shots"]-1] for s in train_samples])
        pred_a_all = oof_act_all[:, :N_ACTION_TRAIN].argmax(axis=1)
        pred_p_all = oof_pt_all.argmax(axis=1)
        m = oof_mask
        f1_a = f1_score(y_a_all[m], pred_a_all[m], labels=ACTION_EVAL_LABELS,
                         average="macro", zero_division=0)
        f1_p = f1_score(y_p_all[m], pred_p_all[m], labels=POINT_EVAL_LABELS,
                         average="macro", zero_division=0)
        sm = m & (y_s_all >= 0)
        if sm.sum() > 0 and len(np.unique(y_s_all[sm])) > 1:
            auc = roc_auc_score(y_s_all[sm], oof_srv_all[sm])
        else:
            auc = 0.5
        ov = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
        print(f"\nFINAL OV (base): {ov:.4f}  F1_a={f1_a:.4f}  F1_p={f1_p:.4f}  AUC={auc:.4f}")
        print(f"OOF mask coverage: {m.sum()}/{len(m)} = {m.sum()/len(m):.3%}")

    # Save OOF arrays
    out_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag
    np.save(os.path.join(out_dir, f"{tag}_oof_act.npy"), oof_act_all)
    np.save(os.path.join(out_dir, f"{tag}_oof_pt.npy"), oof_pt_all)
    np.save(os.path.join(out_dir, f"{tag}_oof_srv.npy"), oof_srv_all)
    np.save(os.path.join(out_dir, f"{tag}_oof_mask.npy"), oof_mask)
    if oof_mask.any():
        np.save(os.path.join(out_dir, f"{tag}_oof_y_act.npy"), y_a_all)
        np.save(os.path.join(out_dir, f"{tag}_oof_y_pt.npy"), y_p_all)
        np.save(os.path.join(out_dir, f"{tag}_oof_y_srv.npy"), y_s_all)
    np.save(os.path.join(out_dir, f"{tag}_test_act.npy"), test_act_acc)
    np.save(os.path.join(out_dir, f"{tag}_test_pt.npy"), test_pt_acc)
    np.save(os.path.join(out_dir, f"{tag}_test_srv.npy"), test_srv_acc)
    np.save(os.path.join(out_dir, f"{tag}_test_rally_uid.npy"),
             np.array([s["rally_uid"] for s in test_samples]))
    print(f"\nSaved OOF + test arrays -> {out_dir}/{tag}_*.npy")


if __name__ == "__main__":
    main()
