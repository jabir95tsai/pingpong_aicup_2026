"""V11 Transformer — Clean GPU sequence model for ping-pong prediction.

Architecture:
  - Per-shot embedding: learned categoricals + numerical projection → d_model
  - Bidirectional Transformer encoder (no causal mask; all context shots
    are in the PAST relative to the target shot, so full attention is fine)
  - Last-position representation → ActionId head (15-class) + PointId head (10-class)
  - Mean-pool representation  → ServerGetPoint head (binary)
  - Multi-task Focal loss with class weights

Key fixes (vs existing transformer files):
  - Correct 15-class action macro F1 (serve classes 15-18 never appear as targets)
  - y_action clipped to 0-14
  - Hand-flip sequence augmentation (mirrors FH↔BH in context shots)

Output:
  - OOF predictions saved as .npy files for blending with V10 GBM
  - Submission CSV: submissions/submission_v11_transformer.csv
"""

import sys, os, time, warnings, argparse, gc
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_ACTION_TRAIN = 15    # classes 0-14 only as next-shot targets
N_POINT        = 10
ACTION_EVAL_LABELS = list(range(15))
POINT_EVAL_LABELS  = list(range(10))

# Flip map: FH↔BH in sequence features
# For context shots: handId 1↔2, positionId 1↔3, pointId 1↔3, 4↔6, 7↔9
POINT_FLIP  = {1: 3, 3: 1, 4: 6, 6: 4, 7: 9, 9: 7}
HAND_FLIP   = {1: 2, 2: 1}
POS_FLIP    = {1: 3, 3: 1}

# Class weights for loss (Focal + weighted CE)
ACTION_W = np.array([
    1.5, 0.6, 0.9, 1.5, 1.2, 1.0, 0.8, 1.8,
    14.0, 8.0, 0.6, 1.2, 0.9, 0.7, 10.0
], dtype=np.float32)   # 15 classes

POINT_W = np.array([0.5, 12.0, 2.5, 22.0, 2.0, 0.9, 1.5, 0.8, 0.7, 0.6],
                    dtype=np.float32)  # 10 classes

# ─── Categorical embedding sizes ─────────────────────────────────────────────
# (n_categories, embed_dim) for each feature in SEQ_CAT order
SEQ_CAT_CFG = [
    ("strikeId",   5,  8),
    ("handId",     3,  6),
    ("strengthId", 4,  6),
    ("spinId",     6, 10),
    ("pointId",   10, 16),
    ("actionId",  19, 24),
    ("positionId", 4,  6),
]
SEQ_CAT_IDX = {name: i for i, (name, _, _) in enumerate(SEQ_CAT_CFG)}


# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_samples(raw_df: pd.DataFrame, is_train: bool, n_players: int = 200) -> list:
    """Build one sample per (rally, target-shot) pair.

    For training: every shot k+1 in every rally → N-1 samples per rally.
    For test:     only the last (visible) shot per rally → 1 sample per rally.

    Each sample is a dict:
      cat_seq  : (k, 7)   int8   — categorical features per context shot
      num_seq  : (k, 4)   f32    — normalised numericals per context shot
      context  : (3,)     f32    — [sex/2, numberGame/7, score_diff/22]
      pid_self : int             — gamePlayerId (remapped)
      pid_other: int             — gamePlayerOtherId (remapped)
      y_action : int             — 0-14 (clipped)
      y_point  : int             — 0-9
      y_server : int             — 0/1
      next_sn  : int             — strikeNumber of target shot
      rally_uid: any             — rally identifier
      match_id : any             — match identifier
    """
    samples = []

    rallies = raw_df.groupby("rally_uid", sort=False)
    for uid, grp in rallies:
        grp = grp.sort_values("strikeNumber").reset_index(drop=True)
        n = len(grp)
        if n < 2 and is_train:
            continue
        if n < 1:
            continue

        match_id = grp["match"].iloc[0]

        # Extract arrays once
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
        sex        = int(grp["sex"].iloc[0])
        num_game   = int(grp["numberGame"].iloc[0])
        server_gp  = grp["serverGetPoint"].values.astype(np.int8)
        pid_self   = int(grp["gamePlayerId"].iloc[0])
        pid_other  = int(grp["gamePlayerOtherId"].iloc[0])

        if is_train:
            target_indices = range(1, n)    # predict each shot

            for tgt in target_indices:
                k = tgt   # context length = tgt shots (indices 0..tgt-1)

                cat_seq = np.stack([
                    strike_id[:k], hand_id[:k], strength[:k],
                    spin[:k], point_id[:k], action_id[:k], pos_id[:k]
                ], axis=1).astype(np.int64)   # (k, 7)

                num_seq = np.stack([
                    sn[:k] / 40.0,
                    score_s[:k] / 11.0,
                    score_o[:k] / 11.0,
                    (score_s[:k] - score_o[:k]) / 22.0,
                ], axis=1).astype(np.float32)   # (k, 4)

                last_score_s = float(score_s[tgt - 1])
                last_score_o = float(score_o[tgt - 1])
                context = np.array([
                    sex / 2.0,
                    num_game / 7.0,
                    (last_score_s - last_score_o) / 22.0,
                ], dtype=np.float32)  # (3,)

                y_a = int(action_id[tgt])
                y_a = min(y_a, N_ACTION_TRAIN - 1)   # clip serve classes → 0..14
                y_p = int(point_id[tgt])
                y_s = int(server_gp[tgt])
                nsn = int(sn[tgt])

                samples.append({
                    "cat_seq":   cat_seq,
                    "num_seq":   num_seq,
                    "context":   context,
                    "pid_self":  min(pid_self,  n_players - 1),
                    "pid_other": min(pid_other, n_players - 1),
                    "y_action":  y_a,
                    "y_point":   y_p,
                    "y_server":  y_s,
                    "next_sn":   nsn,
                    "rally_uid": uid,
                    "match_id":  match_id,
                })

        else:
            # Test: use ALL n visible shots as context; predict the (n+1)-th shot.
            # tgt = n is a virtual index past the end — no ground-truth labels needed.
            k = n

            cat_seq = np.stack([
                strike_id[:k], hand_id[:k], strength[:k],
                spin[:k], point_id[:k], action_id[:k], pos_id[:k]
            ], axis=1).astype(np.int64)   # (k, 7)

            num_seq = np.stack([
                sn[:k] / 40.0,
                score_s[:k] / 11.0,
                score_o[:k] / 11.0,
                (score_s[:k] - score_o[:k]) / 22.0,
            ], axis=1).astype(np.float32)   # (k, 4)

            last_score_s = float(score_s[k - 1])
            last_score_o = float(score_o[k - 1])
            context = np.array([
                sex / 2.0,
                num_game / 7.0,
                (last_score_s - last_score_o) / 22.0,
            ], dtype=np.float32)  # (3,)

            # next_sn = last visible strikeNumber + 1 (the shot being predicted)
            nsn = int(sn[-1]) + 1

            samples.append({
                "cat_seq":   cat_seq,
                "num_seq":   num_seq,
                "context":   context,
                "pid_self":  min(pid_self,  n_players - 1),
                "pid_other": min(pid_other, n_players - 1),
                "y_action":  0,   # placeholder — unknown at test time
                "y_point":   0,   # placeholder
                "y_server":  0,   # placeholder
                "next_sn":   nsn,
                "rally_uid": uid,
                "match_id":  match_id,
            })

    return samples


def flip_sample(s: dict) -> dict:
    """Return a left-right mirrored copy of a sample.

    Flips handId (1↔2), positionId (1↔3), pointId (1↔3,4↔6,7↔9) in cat_seq.
    Also flips y_point.
    """
    cs = s["cat_seq"].copy()  # (k, 7): [strike,hand,strength,spin,point,action,pos]
    # hand col = index 1
    cs[:, 1] = np.vectorize(lambda v: HAND_FLIP.get(int(v), int(v)))(cs[:, 1])
    # pos col = index 6
    cs[:, 6] = np.vectorize(lambda v: POS_FLIP.get(int(v), int(v)))(cs[:, 6])
    # point col = index 4
    cs[:, 4] = np.vectorize(lambda v: POINT_FLIP.get(int(v), int(v)))(cs[:, 4])

    flipped = dict(s)
    flipped["cat_seq"]  = cs
    flipped["y_point"]  = POINT_FLIP.get(int(s["y_point"]), int(s["y_point"]))
    return flipped


class RallyDataset(Dataset):
    def __init__(self, samples: list, max_len: int = 40, augment: bool = False):
        self.samples   = samples
        self.max_len   = max_len
        self.augment   = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.augment and np.random.rand() < 0.5:
            s = flip_sample(s)

        k = min(len(s["cat_seq"]), self.max_len)
        ml = self.max_len

        cat = np.zeros((ml, 7), dtype=np.int64)
        num = np.zeros((ml, 4), dtype=np.float32)
        pad = np.ones(ml, dtype=bool)   # True = padded position

        cat[:k] = s["cat_seq"][:k]
        num[:k] = s["num_seq"][:k]
        pad[:k] = False

        return {
            "cat":      torch.from_numpy(cat),
            "num":      torch.from_numpy(num),
            "context":  torch.from_numpy(s["context"]),
            "pid_self":  torch.tensor(s["pid_self"],  dtype=torch.long),
            "pid_other": torch.tensor(s["pid_other"], dtype=torch.long),
            "pad_mask": torch.from_numpy(pad),
            "seq_len":  torch.tensor(k, dtype=torch.long),
            "y_action": torch.tensor(s["y_action"], dtype=torch.long),
            "y_point":  torch.tensor(s["y_point"],  dtype=torch.long),
            "y_server": torch.tensor(s["y_server"], dtype=torch.float),
        }


# ─── Model ───────────────────────────────────────────────────────────────────

class RallyTransformer(nn.Module):
    def __init__(self, d_model=192, n_heads=8, n_layers=4, dropout=0.1,
                 n_players=200, max_len=40):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Per-shot categorical embeddings
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(n_cat, e_dim)
            for _, n_cat, e_dim in SEQ_CAT_CFG
        ])
        total_cat_dim = sum(e for _, _, e in SEQ_CAT_CFG)  # 76

        # Per-shot numerical projection (4 nums → 16)
        self.num_proj = nn.Sequential(
            nn.Linear(4, 16), nn.GELU()
        )

        # Combined shot projection: 76+16 → d_model
        self.shot_proj = nn.Sequential(
            nn.Linear(total_cat_dim + 16, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding (learned)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Context features: [sex/2, game/7, score_diff/22] → d_model
        self.ctx_proj = nn.Linear(3, d_model)

        # Player embeddings (self + other, joined)
        self.player_emb = nn.Embedding(n_players + 5, 16)
        self.player_proj = nn.Linear(32, d_model)

        # Transformer encoder (bidirectional — no causal mask)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)

        # Task heads
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
        self.server_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, cat, num, context, pid_self, pid_other, pad_mask, seq_len):
        """
        cat      : (B, L, 7)    int64
        num      : (B, L, 4)    float32
        context  : (B, 3)       float32
        pid_self : (B,)         int64
        pid_other: (B,)         int64
        pad_mask : (B, L)       bool   True = padding
        seq_len  : (B,)         int64
        Returns action_logits (B,15), point_logits (B,10), server_logit (B,)
        """
        B, L = cat.shape[:2]

        # Embed each categorical feature
        embeds = [emb(cat[:, :, i]) for i, emb in enumerate(self.cat_embeds)]
        cat_emb = torch.cat(embeds, dim=-1)              # (B, L, 76)

        # Project numericals
        num_emb = self.num_proj(num)                     # (B, L, 16)

        # Combine and project to d_model
        x = self.shot_proj(torch.cat([cat_emb, num_emb], dim=-1))  # (B, L, d)

        # Add positional encoding
        pos = torch.arange(L, device=cat.device).unsqueeze(0)   # (1, L)
        x = x + self.pos_emb(pos)                                # (B, L, d)

        # Add context features as a bias (same for all positions)
        ctx_bias = self.ctx_proj(context).unsqueeze(1)           # (B, 1, d)
        x = x + ctx_bias

        # Add player embeddings
        p_emb = torch.cat([
            self.player_emb(pid_self),
            self.player_emb(pid_other),
        ], dim=-1)                                               # (B, 32)
        player_bias = self.player_proj(p_emb).unsqueeze(1)       # (B, 1, d)
        x = x + player_bias

        # Transformer encoder (pad_mask=True means ignore that position)
        x = self.transformer(x, src_key_padding_mask=pad_mask)   # (B, L, d)

        # Action / Point: use representation at the LAST real position
        # seq_len[i] = k, so last real position is index k-1
        last_idx = (seq_len - 1).clamp(min=0)                    # (B,)
        # Gather last position for each sample in batch
        last_idx_expanded = last_idx.view(B, 1, 1).expand(B, 1, self.d_model)
        last_repr = x.gather(1, last_idx_expanded).squeeze(1)    # (B, d)

        action_logits = self.action_head(last_repr)               # (B, 15)
        point_logits  = self.point_head(last_repr)                # (B, 10)

        # Server: mean-pool over real (non-padded) positions
        # Mask padded positions before pooling
        real_mask = (~pad_mask).float().unsqueeze(-1)             # (B, L, 1)
        pool_repr = (x * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)
        server_logit = self.server_head(pool_repr).squeeze(-1)    # (B,)

        return action_logits, point_logits, server_logit


# ─── Focal Loss ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p    = torch.exp(-ce)
        loss = ((1.0 - p) ** self.gamma) * ce
        return loss.mean()


# ─── Metrics ─────────────────────────────────────────────────────────────────

def action_macro_f1(y_true, probs):
    return f1_score(y_true, probs.argmax(axis=1),
                    labels=ACTION_EVAL_LABELS, average="macro", zero_division=0)

def point_macro_f1(y_true, probs):
    return f1_score(y_true, probs.argmax(axis=1),
                    labels=POINT_EVAL_LABELS, average="macro", zero_division=0)

def apply_action_rules(probs, next_sns):
    """Zero serve classes for non-serve shots."""
    out = probs.copy()
    # extend from 15 to 19 for rules check
    full = np.zeros((len(probs), 19), dtype=np.float32)
    full[:, :N_ACTION_TRAIN] = probs
    serve_mask = (next_sns == 1)
    full[serve_mask,  :15] = 0.0
    full[~serve_mask, 15:] = 0.0
    row_sums = full.sum(axis=1, keepdims=True)
    full /= np.where(row_sums == 0, 1.0, row_sums)
    return full[:, :N_ACTION_TRAIN]


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, act_loss_fn, pt_loss_fn):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        cat  = batch["cat"].to(DEVICE)
        num  = batch["num"].to(DEVICE)
        ctx  = batch["context"].to(DEVICE)
        ps   = batch["pid_self"].to(DEVICE)
        po   = batch["pid_other"].to(DEVICE)
        mask = batch["pad_mask"].to(DEVICE)
        slen = batch["seq_len"].to(DEVICE)
        ya   = batch["y_action"].to(DEVICE)
        yp   = batch["y_point"].to(DEVICE)
        ys   = batch["y_server"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            a_logits, p_logits, s_logit = model(cat, num, ctx, ps, po, mask, slen)
            loss_a = act_loss_fn(a_logits, ya)
            loss_p = pt_loss_fn(p_logits, yp)
            loss_s = F.binary_cross_entropy_with_logits(s_logit, ys)
            loss   = 0.4 * loss_a + 0.4 * loss_p + 0.2 * loss_s

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    """Run inference and return (act_probs, pt_probs, srv_probs) as numpy arrays."""
    model.eval()
    act_list, pt_list, srv_list = [], [], []
    for batch in loader:
        cat  = batch["cat"].to(device)
        num  = batch["num"].to(device)
        ctx  = batch["context"].to(device)
        ps   = batch["pid_self"].to(device)
        po   = batch["pid_other"].to(device)
        mask = batch["pad_mask"].to(device)
        slen = batch["seq_len"].to(device)

        with autocast():
            a_logits, p_logits, s_logit = model(cat, num, ctx, ps, po, mask, slen)

        act_list.append(F.softmax(a_logits.float(), dim=-1).cpu().numpy())
        pt_list.append( F.softmax(p_logits.float(), dim=-1).cpu().numpy())
        srv_list.append(torch.sigmoid(s_logit.float()).cpu().numpy())

    return (np.vstack(act_list),
            np.vstack(pt_list),
            np.concatenate(srv_list))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",       action="store_true")
    parser.add_argument("--folds",       type=int, default=N_FOLDS)
    parser.add_argument("--epochs",      type=int, default=80)
    parser.add_argument("--d-model",     type=int, default=192)
    parser.add_argument("--n-layers",    type=int, default=4)
    parser.add_argument("--batch",       type=int, default=256)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--no-aug",      action="store_true")
    parser.add_argument("--blend-gbm",   type=str, default="",
                        help="Path to V10 GBM submission CSV for final blend")
    args = parser.parse_args()

    is_smoke = args.smoke
    n_folds  = 1 if is_smoke else args.folds
    n_epochs = 5 if is_smoke else args.epochs
    use_aug  = not args.no_aug
    bs       = args.batch
    lr       = args.lr
    d_model  = args.d_model
    n_layers = args.n_layers

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t_start = time.time()
    print("=" * 70)
    print(f"V11 TRANSFORMER {'(SMOKE)' if is_smoke else ''}")
    print(f"  device={DEVICE}  d_model={d_model}  n_layers={n_layers}")
    print(f"  folds={n_folds}  epochs={n_epochs}  batch={bs}  lr={lr}")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)
    test_df["serverGetPoint"] = -1
    n_players = len(player_map)
    print(f"\n  Players: {n_players}")

    # ── Build samples ─────────────────────────────────────────────────────────
    print("\n--- Building samples ---")
    t0 = time.time()
    all_samples = build_samples(train_df, is_train=True, n_players=n_players)
    test_samples = build_samples(test_df, is_train=False, n_players=n_players)
    print(f"  Train samples: {len(all_samples)}  Test samples: {len(test_samples)}  ({time.time()-t0:.1f}s)")

    # Rally→match mapping for GroupKFold
    rally_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    sample_rallies  = np.array([s["rally_uid"] for s in all_samples])
    sample_matches  = np.array([rally_to_match.get(r, -1) for r in sample_rallies])

    # True labels for OOF evaluation
    y_a_all = np.array([s["y_action"] for s in all_samples])
    y_p_all = np.array([s["y_point"]  for s in all_samples])
    y_s_all = np.array([s["y_server"] for s in all_samples])
    nsn_all = np.array([s["next_sn"]  for s in all_samples])

    # OOF containers
    n_samples = len(all_samples)
    oof_act = np.zeros((n_samples, N_ACTION_TRAIN))
    oof_pt  = np.zeros((n_samples, N_POINT))
    oof_srv = np.zeros(n_samples)
    oof_mask_arr = np.zeros(n_samples, dtype=bool)

    # Test accumulators
    test_act_acc = np.zeros((len(test_samples), N_ACTION_TRAIN))
    test_pt_acc  = np.zeros((len(test_samples), N_POINT))
    test_srv_acc = np.zeros(len(test_samples))

    # Next-sn for test (needed for action rules)
    nsn_test = np.array([s["next_sn"] for s in test_samples])
    rally_uid_test = [s["rally_uid"] for s in test_samples]

    # Loss weights (on device)
    act_w = torch.tensor(ACTION_W, device=DEVICE)
    pt_w  = torch.tensor(POINT_W,  device=DEVICE)
    act_loss_fn = FocalLoss(act_w, gamma=2.0)
    pt_loss_fn  = FocalLoss(pt_w,  gamma=2.0)

    gkf    = GroupKFold(n_splits=max(n_folds, 2))
    splits = list(gkf.split(np.arange(n_samples), groups=sample_matches))
    if is_smoke:
        splits = splits[:1]

    test_ds     = RallyDataset(test_samples, augment=False)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                              num_workers=0, pin_memory=True)

    for fold, (tr_idx, val_idx) in enumerate(splits):
        t_fold = time.time()
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{len(splits)}")
        print(f"{'='*60}")

        tr_samps  = [all_samples[i] for i in tr_idx]
        val_samps = [all_samples[i] for i in val_idx]

        tr_ds  = RallyDataset(tr_samps,  augment=use_aug)
        val_ds = RallyDataset(val_samps, augment=False)

        tr_loader  = DataLoader(tr_ds,  batch_size=bs, shuffle=True,
                                 num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                                 num_workers=0, pin_memory=True)

        model = RallyTransformer(
            d_model=d_model, n_heads=8, n_layers=n_layers,
            dropout=0.15, n_players=n_players + 5, max_len=40
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                       weight_decay=1e-2, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=n_epochs, eta_min=lr / 20)
        scaler = GradScaler()

        best_ov   = -1.0
        best_state = None
        patience  = 12
        wait      = 0

        for epoch in range(1, n_epochs + 1):
            tr_loss = train_epoch(model, tr_loader, optimizer, scaler,
                                  act_loss_fn, pt_loss_fn)
            scheduler.step()

            # Validate every 5 epochs (saves time)
            if epoch % 5 == 0 or epoch == n_epochs:
                a_p, p_p, s_p = evaluate(model, val_loader)
                y_a_val = np.array([s["y_action"] for s in val_samps])
                y_p_val = np.array([s["y_point"]  for s in val_samps])
                y_s_val = np.array([s["y_server"] for s in val_samps])
                nsn_val = np.array([s["next_sn"]  for s in val_samps])

                a_p_ruled = apply_action_rules(a_p, nsn_val)
                f1_a = action_macro_f1(y_a_val, a_p_ruled)
                f1_p = point_macro_f1(y_p_val, p_p)
                auc  = roc_auc_score(y_s_val, s_p)
                ov   = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc

                print(f"  Ep{epoch:3d}/{n_epochs}  loss={tr_loss:.4f}  "
                      f"F1_a={f1_a:.4f}  F1_p={f1_p:.4f}  AUC={auc:.4f}  OV={ov:.4f}")

                if ov > best_ov:
                    best_ov    = ov
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience and not is_smoke:
                        print(f"  Early stopping at epoch {epoch} (patience={patience})")
                        break

        # Restore best checkpoint
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        model.eval()

        # Final val evaluation
        a_p, p_p, s_p = evaluate(model, val_loader)
        y_a_val = np.array([s["y_action"] for s in val_samps])
        y_p_val = np.array([s["y_point"]  for s in val_samps])
        y_s_val = np.array([s["y_server"] for s in val_samps])
        nsn_val = np.array([s["next_sn"]  for s in val_samps])

        a_p_ruled = apply_action_rules(a_p, nsn_val)
        f1_a = action_macro_f1(y_a_val, a_p_ruled)
        f1_p = point_macro_f1(y_p_val, p_p)
        auc  = roc_auc_score(y_s_val, s_p)
        ov   = 0.4 * f1_a + 0.4 * f1_p + 0.2 * auc
        print(f"\n  BEST FOLD: F1_a={f1_a:.4f}  F1_p={f1_p:.4f}  AUC={auc:.4f}  OV={ov:.4f}  [{time.time()-t_fold:.0f}s]")

        # Store OOF
        oof_act[val_idx]      = a_p
        oof_pt[val_idx]       = p_p
        oof_srv[val_idx]      = s_p
        oof_mask_arr[val_idx] = True

        # Accumulate test predictions
        at, pt, st = evaluate(model, test_loader)
        test_act_acc += at / len(splits)
        test_pt_acc  += pt / len(splits)
        test_srv_acc += st / len(splits)

        del model; gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ─── Global OOF evaluation ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GLOBAL OOF RESULTS")
    n_oof = oof_mask_arr.sum()
    print(f"  OOF: {n_oof}/{n_samples} ({100*n_oof/n_samples:.0f}%)")

    oa = apply_action_rules(oof_act[oof_mask_arr], nsn_all[oof_mask_arr])
    f1_a_oof = action_macro_f1(y_a_all[oof_mask_arr], oa)
    f1_p_oof = point_macro_f1(y_p_all[oof_mask_arr], oof_pt[oof_mask_arr])
    auc_oof  = roc_auc_score(y_s_all[oof_mask_arr], oof_srv[oof_mask_arr])
    ov_oof   = 0.4 * f1_a_oof + 0.4 * f1_p_oof + 0.2 * auc_oof
    print(f"  action={f1_a_oof:.4f}  point={f1_p_oof:.4f}  AUC={auc_oof:.4f}  OV={ov_oof:.4f}")

    print("\n  PointId per-class F1:")
    pf1s = f1_score(y_p_all[oof_mask_arr],
                    oof_pt[oof_mask_arr].argmax(axis=1),
                    labels=POINT_EVAL_LABELS, average=None, zero_division=0)
    zone_names = ["miss","FH_short","mid_short","BH_short","FH_half",
                  "mid_half","BH_half","FH_long","mid_long","BH_long"]
    for i, (nm, f) in enumerate(zip(zone_names, pf1s)):
        print(f"    {nm:12s}(cls{i}): F1={f:.4f}  n={(y_p_all[oof_mask_arr]==i).sum()}")

    print("\n  ActionId per-class F1:")
    af1s = f1_score(y_a_all[oof_mask_arr],
                    oa.argmax(axis=1),
                    labels=ACTION_EVAL_LABELS, average=None, zero_division=0)
    action_names = ["None","Loop","Cloop","Smash","Flip","Pushfast","Push","Flick",
                    "Arch","Knuckle","Chop_r","ShortStop","Chop","Block","Lob"]
    for i, (nm, f) in enumerate(zip(action_names, af1s)):
        print(f"    {nm:10s}(cls{i:2d}): F1={f:.4f}  n={(y_a_all[oof_mask_arr]==i).sum()}")

    # ─── Save OOF predictions (for later blending with V10 GBM) ───────────────
    oof_save_dir = os.path.join(os.path.dirname(SUBMISSION_DIR), "oof_predictions")
    os.makedirs(oof_save_dir, exist_ok=True)
    np.save(os.path.join(oof_save_dir, "v11_oof_act.npy"),  oof_act)
    np.save(os.path.join(oof_save_dir, "v11_oof_pt.npy"),   oof_pt)
    np.save(os.path.join(oof_save_dir, "v11_oof_srv.npy"),  oof_srv)
    np.save(os.path.join(oof_save_dir, "v11_oof_mask.npy"), oof_mask_arr)
    np.save(os.path.join(oof_save_dir, "v11_test_act.npy"), test_act_acc)
    np.save(os.path.join(oof_save_dir, "v11_test_pt.npy"),  test_pt_acc)
    np.save(os.path.join(oof_save_dir, "v11_test_srv.npy"), test_srv_acc)
    print(f"\n  OOF + test predictions saved to {oof_save_dir}")

    # ─── Generate submission ──────────────────────────────────────────────────
    print("\n--- Generating submission ---")

    # Apply action rules to test predictions
    test_act_ruled = apply_action_rules(test_act_acc, nsn_test)

    pred_act = test_act_ruled.argmax(axis=1)
    pred_pt  = test_pt_acc.argmax(axis=1)
    pred_srv = test_srv_acc   # CONTINUOUS probabilities for better AUC

    sub = pd.DataFrame({
        "rally_uid":      rally_uid_test,
        "actionId":       pred_act,
        "pointId":        pred_pt,
        "serverGetPoint": pred_srv,   # continuous [0,1]
    })

    out_path = os.path.join(SUBMISSION_DIR, "submission_v11_transformer.csv")
    sub.to_csv(out_path, index=False)
    print(f"  actionId dist: {dict(pd.Series(pred_act).value_counts().sort_index())}")
    print(f"  pointId  dist: {dict(pd.Series(pred_pt).value_counts().sort_index())}")
    print(f"  SGP mean: {pred_srv.mean():.4f}  std: {pred_srv.std():.4f}")
    print(f"  Saved: {out_path}")

    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal time: {elapsed:.1f} min")
    print(f"\n{'='*70}")
    print(f"FINAL OV: {ov_oof:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
