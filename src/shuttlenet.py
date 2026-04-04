"""ShuttleNet-style autoregressive model for ping pong prediction.

Key difference from V1/V2: this is an autoregressive (causal) model that
predicts the next strike at EVERY position, giving much richer training signal.
Uses causal masking so position k can only attend to positions 0..k-1.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Embedding configs
N_ACTION = 19
N_POINT = 10
EMBED_CONFIGS = {
    "actionId": (N_ACTION, 32),
    "pointId": (N_POINT, 24),
    "strikeId": (5, 16),
    "handId": (3, 8),
    "strengthId": (4, 8),
    "spinId": (6, 16),
    "positionId": (4, 8),
}
SEQ_CAT_COLS = ["strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"]


class ShuttleNetModel(nn.Module):
    """Autoregressive Transformer for ping pong sequence prediction.

    At each position k, predicts actionId[k+1] and pointId[k+1] using
    causal attention (can only see positions 0..k).
    """

    def __init__(self, d_model=128, nhead=8, n_layers=2, dropout=0.1, n_players=200):
        super().__init__()
        self.d_model = d_model

        # Strike embeddings
        self.embeddings = nn.ModuleDict()
        total_dim = 0
        for col, (n_cls, e_dim) in EMBED_CONFIGS.items():
            self.embeddings[col] = nn.Embedding(n_cls, e_dim, padding_idx=0)
            total_dim += e_dim

        # Numerical input
        self.num_proj = nn.Linear(1, 16)  # strikeNumber
        total_dim += 16

        self.input_proj = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding (learnable)
        self.pos_embed = nn.Embedding(60, d_model)

        # Player embeddings
        self.player_embed = nn.Embedding(n_players + 1, 32)
        self.player_proj = nn.Linear(64, d_model)

        # Context (sex, numberGame, scores)
        self.ctx_proj = nn.Linear(4, d_model)

        # Causal Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Prediction heads
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, N_ACTION),
        )
        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, N_POINT),
        )
        self.server_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _generate_causal_mask(self, sz, device):
        """Generate causal mask: position i can attend to positions 0..i."""
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

    def forward(self, cat_seq, num_seq, context, player_ids, seq_mask=None):
        """
        cat_seq: (B, T, 7) categorical features
        num_seq: (B, T, 1) numerical features
        context: (B, 4) match context
        player_ids: (B, 2)
        seq_mask: (B, T) True=padding

        Returns action_logits, point_logits, server_logits
        All of shape (B, T, N_classes) for autoregressive prediction
        """
        B, T, _ = cat_seq.shape

        # Embed all strike features
        embeds = []
        for i, col in enumerate(SEQ_CAT_COLS):
            embeds.append(self.embeddings[col](cat_seq[:, :, i]))
        embeds.append(self.num_proj(num_seq))
        x = torch.cat(embeds, dim=-1)
        x = self.input_proj(x)  # (B, T, d_model)

        # Positional encoding
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos)

        # Memory: context + player info
        ctx = self.ctx_proj(context).unsqueeze(1)  # (B, 1, d_model)
        p1 = self.player_embed(player_ids[:, 0])
        p2 = self.player_embed(player_ids[:, 1])
        player = self.player_proj(torch.cat([p1, p2], dim=-1)).unsqueeze(1)
        memory = torch.cat([ctx, player], dim=1)  # (B, 2, d_model)

        # Causal mask
        causal_mask = self._generate_causal_mask(T, x.device)

        # Transformer decoder with causal masking
        x = self.transformer(
            x, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=seq_mask,
        )  # (B, T, d_model)

        # Predictions at each position
        action_logits = self.action_head(x)  # (B, T, N_ACTION)
        point_logits = self.point_head(x)    # (B, T, N_POINT)

        # Server prediction uses mean pooling over non-padded positions
        if seq_mask is not None:
            mask_expand = (~seq_mask).unsqueeze(-1).float()
            pooled = (x * mask_expand).sum(dim=1) / mask_expand.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        server_logits = self.server_head(pooled).squeeze(-1)  # (B,)

        return action_logits, point_logits, server_logits


class ShuttleNetDataset(torch.utils.data.Dataset):
    """Dataset for autoregressive training.

    For each rally, the input is the full sequence.
    Targets: action[k+1] and point[k+1] for each position k.
    """
    def __init__(self, rallies, max_seq_len=50):
        self.data = rallies
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        T = min(len(d["cat_seq"]), self.max_seq_len)
        pad_len = self.max_seq_len - T

        cat_seq = d["cat_seq"][-T:]
        num_seq = d["num_seq"][-T:]
        cat_seq = np.pad(cat_seq, ((0, pad_len), (0, 0)), constant_values=0)
        num_seq = np.pad(num_seq, ((0, pad_len), (0, 0)), constant_values=0)
        mask = np.array([False]*T + [True]*pad_len)

        result = {
            "cat_seq": torch.LongTensor(cat_seq),
            "num_seq": torch.FloatTensor(num_seq),
            "context": torch.FloatTensor(d["context"]),
            "player_ids": torch.LongTensor(d["player_ids"]),
            "mask": torch.BoolTensor(mask),
            "seq_len": T,
            "rally_uid": d["rally_uid"],
        }

        if "y_actions" in d:
            # Shifted targets: y_actions[k] = actionId at position k+1
            ya = np.full(self.max_seq_len, -1, dtype=np.int64)
            yp = np.full(self.max_seq_len, -1, dtype=np.int64)
            n_targets = min(len(d["y_actions"]), self.max_seq_len)
            ya[:n_targets] = d["y_actions"][-n_targets:] if n_targets <= len(d["y_actions"]) else np.pad(d["y_actions"], (self.max_seq_len - len(d["y_actions"]), 0), constant_values=-1)[-n_targets:]
            yp[:n_targets] = d["y_points"][-n_targets:] if n_targets <= len(d["y_points"]) else np.pad(d["y_points"], (self.max_seq_len - len(d["y_points"]), 0), constant_values=-1)[-n_targets:]

            # Fix: properly align
            ya = np.full(self.max_seq_len, -1, dtype=np.int64)
            yp = np.full(self.max_seq_len, -1, dtype=np.int64)
            targets_a = d["y_actions"]
            targets_p = d["y_points"]
            n = min(len(targets_a), T)
            ya[:n] = targets_a[:n]
            yp[:n] = targets_p[:n]

            result["y_actions"] = torch.LongTensor(ya)
            result["y_points"] = torch.LongTensor(yp)
            result["y_server"] = d["y_server"]

        return result


def prepare_autoregressive_data(df, is_train=True):
    """Prepare rally-level data for autoregressive training.

    For each rally:
    - Input: full sequence of strikes
    - Target at position k: actionId and pointId of strike k+1
    """
    rallies = df.groupby("rally_uid", sort=False)
    samples = []

    for rally_uid, group in rallies:
        group = group.sort_values("strikeNumber")

        cat_seq = group[SEQ_CAT_COLS].values.astype(np.int64)
        num_seq = (group[["strikeNumber"]].values / 20.0).astype(np.float32)

        first_row = group.iloc[0]
        context = np.array([
            first_row["sex"] / 2.0,
            first_row["numberGame"] / 7.0,
            first_row["scoreSelf"] / 15.0,
            first_row["scoreOther"] / 15.0,
        ], dtype=np.float32)

        player_ids = np.array([
            int(first_row["gamePlayerId"]),
            int(first_row["gamePlayerOtherId"]),
        ], dtype=np.int64)

        sample = {
            "cat_seq": cat_seq,
            "num_seq": num_seq,
            "context": context,
            "player_ids": player_ids,
            "rally_uid": int(rally_uid),
        }

        if is_train and len(group) >= 2:
            # Target at position k = strike k+1's action and point
            # Position 0 predicts strike 2, position 1 predicts strike 3, etc.
            actions = group["actionId"].values.astype(np.int64)
            points = group["pointId"].values.astype(np.int64)
            sample["y_actions"] = actions[1:]  # shifted by 1
            sample["y_points"] = points[1:]
            sample["y_server"] = int(first_row["serverGetPoint"])
            samples.append(sample)
        elif not is_train:
            samples.append(sample)

    return samples
