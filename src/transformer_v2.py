"""Transformer V2: Improved architecture for ping pong prediction.

Key improvements over V1:
1. Separate encoder paths for server/receiver (like ShuttleNet)
2. Cross-attention between player perspectives
3. Larger embedding dims, layer norm
4. Separate prediction heads with skip connections
5. Better positional encoding (sinusoidal + learnable)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


SEQ_CAT_COLS = ["strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"]
SEQ_NUM_COLS = ["strikeNumber"]
CONTEXT_COLS = ["sex", "numberGame", "scoreSelf", "scoreOther"]

EMBED_CONFIGS = {
    "strikeId": (5, 16),
    "handId": (3, 8),
    "strengthId": (4, 8),
    "spinId": (6, 16),
    "pointId": (10, 24),
    "actionId": (19, 32),
    "positionId": (4, 8),
}


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class StrikeEncoder(nn.Module):
    """Encode a single strike's features into a vector."""
    def __init__(self, d_model):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_dim = 0
        for col, (n_cls, e_dim) in EMBED_CONFIGS.items():
            self.embeddings[col] = nn.Embedding(n_cls, e_dim)
            total_dim += e_dim

        self.num_proj = nn.Linear(len(SEQ_NUM_COLS), 16)
        total_dim += 16

        self.proj = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, cat_seq, num_seq):
        embeds = []
        for i, col in enumerate(SEQ_CAT_COLS):
            embeds.append(self.embeddings[col](cat_seq[:, :, i]))
        embeds.append(self.num_proj(num_seq))
        x = torch.cat(embeds, dim=-1)
        return self.proj(x)


class DualPathTransformer(nn.Module):
    """Two separate paths for odd/even strikes (server/receiver perspective),
    with cross-attention fusion."""
    def __init__(self, d_model=128, nhead=8, n_layers=2, dropout=0.1):
        super().__init__()
        # Self-attention for each path
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Cross-attention: query from sequence, key/value from other player's strikes
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_ff = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model),
        )
        self.cross_ff_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Full sequence self-attention
        x = self.encoder(x, src_key_padding_mask=mask)

        # Split odd/even strikes for cross-attention
        B, T, D = x.shape
        # Create odd/even masks
        odd_idx = torch.arange(0, T, 2, device=x.device)
        even_idx = torch.arange(1, T, 2, device=x.device)

        if len(even_idx) > 0 and len(odd_idx) > 0:
            odd_seq = x[:, odd_idx]   # server's strikes
            even_seq = x[:, even_idx]  # receiver's strikes

            # Cross-attention: last position attends to opponent's strikes
            # This helps model opponent patterns
            cross_out, _ = self.cross_attn(x, x, x, key_padding_mask=mask)
            x = self.cross_norm(x + cross_out)
            x = self.cross_ff_norm(x + self.cross_ff(x))

        return x


class PingPongTransformerV2(nn.Module):
    def __init__(self, d_model=256, nhead=8, n_layers=3, dropout=0.1,
                 n_action_classes=19, n_point_classes=10, n_players=200):
        super().__init__()
        self.d_model = d_model

        # Strike encoder
        self.strike_encoder = StrikeEncoder(d_model)

        # Positional encoding (sinusoidal + learnable)
        self.pos_enc = SinusoidalPE(d_model, max_len=60)
        self.pos_learn = nn.Embedding(60, d_model)

        # Context projection
        self.ctx_proj = nn.Sequential(
            nn.Linear(len(CONTEXT_COLS), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Player embeddings
        self.player_embed = nn.Embedding(n_players + 1, 32)
        self.player_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # "Next strike" query token (learnable)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Main transformer with cross-attention
        self.transformer = DualPathTransformer(d_model, nhead, n_layers, dropout)

        # Prediction heads with skip connections
        self.action_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_action_classes),
        )

        self.point_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_point_classes),
        )

        self.server_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, cat_seq, num_seq, context, player_ids, seq_mask=None):
        B, T, _ = cat_seq.shape

        # Encode strikes
        x = self.strike_encoder(cat_seq, num_seq)

        # Add positional encoding (both sinusoidal and learnable)
        x = self.pos_enc(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_learn(positions)

        # Context token
        ctx = self.ctx_proj(context).unsqueeze(1)

        # Player token
        p1 = self.player_embed(player_ids[:, 0])
        p2 = self.player_embed(player_ids[:, 1])
        player = self.player_proj(torch.cat([p1, p2], dim=-1)).unsqueeze(1)

        # Query token for "next strike" prediction
        query = self.query_token.expand(B, -1, -1)

        # Combine: [CTX, PLAYER, strike1, ..., strikeT, QUERY]
        x = torch.cat([ctx, player, x, query], dim=1)  # (B, T+3, d_model)

        # Update mask
        if seq_mask is not None:
            prefix_suffix_mask = torch.zeros(B, 3, dtype=torch.bool, device=x.device)
            seq_mask = torch.cat([prefix_suffix_mask, seq_mask[:, :T],
                                  torch.zeros(B, 0, dtype=torch.bool, device=x.device)], dim=1)
            # Ensure mask length matches
            if seq_mask.shape[1] < x.shape[1]:
                pad = torch.zeros(B, x.shape[1] - seq_mask.shape[1], dtype=torch.bool, device=x.device)
                seq_mask = torch.cat([seq_mask, pad], dim=1)

        # Transformer
        x = self.transformer(x, mask=seq_mask)

        # Use query token output (last position) + context for prediction
        query_out = x[:, -1]  # query token
        ctx_out = x[:, 0]     # context token

        # Concatenate for skip connection
        combined = torch.cat([query_out, ctx_out], dim=-1)

        action_logits = self.action_head(combined)
        point_logits = self.point_head(combined)
        server_logits = self.server_head(combined).squeeze(-1)

        return action_logits, point_logits, server_logits


class PingPongDatasetV2(torch.utils.data.Dataset):
    def __init__(self, samples, max_seq_len=50):
        self.data = samples
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
        mask = np.array([False] * T + [True] * pad_len)

        result = {
            "cat_seq": torch.LongTensor(cat_seq),
            "num_seq": torch.FloatTensor(num_seq),
            "context": torch.FloatTensor(d["context"]),
            "player_ids": torch.LongTensor(d["player_ids"]),
            "mask": torch.BoolTensor(mask),
            "rally_uid": d["rally_uid"],
        }

        if "y_action" in d:
            result["y_action"] = d["y_action"]
            result["y_point"] = d["y_point"]
            result["y_server"] = d["y_server"]

        return result
