"""Transformer-based sequence model for ping pong prediction (ShuttleNet-style)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Embedding dimensions for each categorical feature
EMBED_CONFIGS = {
    # (num_classes, embed_dim)
    "strikeId": (5, 8),       # {0,1,2,3,4} after remap
    "handId": (3, 4),         # {0,1,2}
    "strengthId": (4, 4),     # {0,1,2,3}
    "spinId": (6, 8),         # {0..5}
    "pointId": (10, 16),      # {0..9}
    "actionId": (19, 16),     # {0..18}
    "positionId": (4, 4),     # {0..3}
}

# Feature columns in order (must match dataset)
SEQ_CAT_COLS = ["strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"]
SEQ_NUM_COLS = ["strikeNumber"]  # normalized
CONTEXT_COLS = ["sex", "numberGame", "scoreSelf", "scoreOther"]


class PingPongTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, n_layers=2, dropout=0.1,
                 n_action_classes=19, n_point_classes=10, n_players=200):
        super().__init__()
        self.d_model = d_model

        # Embeddings for categorical features in each strike
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        for col, (n_cls, e_dim) in EMBED_CONFIGS.items():
            self.embeddings[col] = nn.Embedding(n_cls, e_dim)
            total_embed_dim += e_dim

        # Numerical features projection
        n_num = len(SEQ_NUM_COLS)
        self.num_proj = nn.Linear(n_num, 8)
        total_embed_dim += 8

        # Project concatenated embeddings to d_model
        self.input_proj = nn.Linear(total_embed_dim, d_model)

        # Positional encoding (learnable, max 60 strikes)
        self.pos_embed = nn.Embedding(60, d_model)

        # Context features (sex, numberGame, scores)
        self.context_proj = nn.Linear(len(CONTEXT_COLS), d_model)

        # Player embedding
        self.player_embed = nn.Embedding(n_players + 1, 16)  # +1 for unknown
        self.player_proj = nn.Linear(32, d_model)  # 2 players * 16

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_action_classes),
        )
        self.point_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_point_classes),
        )
        self.server_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, cat_seq, num_seq, context, player_ids, seq_mask=None):
        """
        cat_seq: (B, T, n_cat_cols) - categorical features per strike
        num_seq: (B, T, n_num_cols) - numerical features per strike
        context: (B, n_context) - match-level context
        player_ids: (B, 2) - [gamePlayerId, gamePlayerOtherId]
        seq_mask: (B, T) - True for padded positions
        """
        B, T, _ = cat_seq.shape

        # Embed categorical features
        embeds = []
        for i, col in enumerate(SEQ_CAT_COLS):
            embeds.append(self.embeddings[col](cat_seq[:, :, i]))
        embeds.append(self.num_proj(num_seq))
        x = torch.cat(embeds, dim=-1)  # (B, T, total_embed_dim)

        # Project to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Add positional encoding
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions)

        # Add context as a "CLS" token at position 0
        ctx = self.context_proj(context).unsqueeze(1)  # (B, 1, d_model)

        # Add player embedding
        p1 = self.player_embed(player_ids[:, 0])
        p2 = self.player_embed(player_ids[:, 1])
        player_feat = self.player_proj(torch.cat([p1, p2], dim=-1)).unsqueeze(1)  # (B, 1, d_model)

        # Prepend context and player tokens: [CTX, PLAYER, strike1, strike2, ...]
        x = torch.cat([ctx, player_feat, x], dim=1)  # (B, T+2, d_model)

        # Update mask for prepended tokens
        if seq_mask is not None:
            prefix_mask = torch.zeros(B, 2, dtype=torch.bool, device=x.device)
            seq_mask = torch.cat([prefix_mask, seq_mask], dim=1)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=seq_mask)  # (B, T+2, d_model)

        # Use the last non-padded position for prediction
        # Find last valid position index
        if seq_mask is not None:
            # seq_lengths = T+2 - seq_mask.sum(dim=1)
            lengths = (~seq_mask).sum(dim=1) - 1  # last valid index
        else:
            lengths = torch.full((B,), T + 1, dtype=torch.long, device=x.device)

        # Gather last valid hidden state
        last_hidden = x[torch.arange(B, device=x.device), lengths]  # (B, d_model)

        # Prediction heads
        action_logits = self.action_head(last_hidden)
        point_logits = self.point_head(last_hidden)
        server_logits = self.server_head(last_hidden).squeeze(-1)

        return action_logits, point_logits, server_logits


class PingPongDataset(torch.utils.data.Dataset):
    """Dataset that converts rally data into sequences for the transformer."""

    def __init__(self, rally_data, max_seq_len=50):
        """
        rally_data: list of dicts, each with:
            - cat_seq: (T, n_cat) array
            - num_seq: (T, n_num) array
            - context: (n_ctx,) array
            - player_ids: (2,) array
            - y_action: int (optional)
            - y_point: int (optional)
            - y_server: int (optional)
            - rally_uid: int
        """
        self.data = rally_data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        T = min(len(d["cat_seq"]), self.max_seq_len)
        pad_len = self.max_seq_len - T

        # Truncate if needed, then pad
        cat_seq = d["cat_seq"][-T:]  # keep last T strikes
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


def prepare_sequences(df, is_train=True):
    """Convert raw DataFrame into sequence data for the transformer."""
    rallies = df.groupby("rally_uid", sort=False)
    samples = []

    for rally_uid, group in rallies:
        group = group.sort_values("strikeNumber")

        if is_train:
            # Generate samples for each target position
            for target_idx in range(1, len(group)):
                context_rows = group.iloc[:target_idx]
                target_row = group.iloc[target_idx]
                sample = _make_sample(rally_uid, context_rows, target_row)
                samples.append(sample)
        else:
            context_rows = group
            sample = _make_sample(rally_uid, context_rows, None)
            samples.append(sample)

    return samples


def _make_sample(rally_uid, context_rows, target_row):
    cat_seq = context_rows[SEQ_CAT_COLS].values.astype(np.int64)
    num_seq = context_rows[SEQ_NUM_COLS].values.astype(np.float32)
    # Normalize strikeNumber
    num_seq[:, 0] = num_seq[:, 0] / 20.0

    first_row = context_rows.iloc[0]
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

    if target_row is not None:
        sample["y_action"] = int(target_row["actionId"])
        sample["y_point"] = int(target_row["pointId"])
        sample["y_server"] = int(target_row["serverGetPoint"])

    return sample
