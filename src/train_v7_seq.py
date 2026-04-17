"""V7 Sequence Baseline: GRU prefix model for actionId / pointId / serverGetPoint.

Architecture:
- Per-field embeddings for categorical features + linear projection for continuous
- Bidirectional GRU encoder over the prefix sequence
- Multi-head output: actionId (19-class), pointId (10-class), serverGetPoint (binary)
- Each training sample = (prefix of rally up to strike t) -> predict strike t+1

Key design decisions:
- GroupKFold(by match) validation
- No future information leakage (strict prefix)
- Focal loss for class imbalance
- Conservative: small model, early stopping, gradient clipping
"""
import sys, os, time, warnings, argparse, gc
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH, TEST_PATH, MODEL_DIR, SUBMISSION_DIR, N_FOLDS, RANDOM_SEED
from data_cleaning import clean_data

N_ACTION, N_POINT = 19, 10
SERVE_FORBIDDEN = {15, 16, 17, 18}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================= DATASET ============================================

# Categorical field definitions: (column_name, num_classes)
CAT_FIELDS = [
    ("strikeId", 5),      # after remap: 0-4
    ("handId", 3),        # 0-2
    ("strengthId", 4),    # 0-3
    ("spinId", 6),        # 0-5
    ("pointId", 10),      # 0-9
    ("actionId", 19),     # 0-18
    ("positionId", 4),    # 0-3
]

# Continuous fields
CONT_FIELDS = [
    "scoreSelf", "scoreOther", "strikeNumber", "numberGame", "sex",
]


class RallyDataset(Dataset):
    """Dataset that produces prefix sequences for each prediction target."""

    def __init__(self, rallies, max_len=30):
        """
        rallies: list of dicts, each with:
            - 'cat': np.array shape (rally_len, n_cat_fields) int
            - 'cont': np.array shape (rally_len, n_cont_fields) float
            - 'targets': list of (actionId, pointId, serverGetPoint) for each strike from index 1
            - 'next_sn': list of next strikeNumber values
            - 'rally_uid': int
            - 'match': int
        """
        self.samples = []
        self.max_len = max_len

        for rally in rallies:
            cat = rally["cat"]
            cont = rally["cont"]
            rally_len = len(cat)

            for t in range(1, rally_len):
                # Prefix = strikes 0..t-1, target = strike t
                prefix_len = min(t, max_len)
                start = max(0, t - max_len)

                self.samples.append({
                    "cat": cat[start:t],
                    "cont": cont[start:t],
                    "prefix_len": prefix_len,
                    "y_action": int(rally["targets"][t - 1][0]),
                    "y_point": int(rally["targets"][t - 1][1]),
                    "y_sgp": int(rally["targets"][t - 1][2]),
                    "next_sn": int(rally["next_sn"][t - 1]),
                    "rally_uid": rally["rally_uid"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s


class TestRallyDataset(Dataset):
    """Dataset for test: one sample per rally (full prefix)."""

    def __init__(self, rallies, max_len=30):
        self.samples = []
        self.max_len = max_len

        for rally in rallies:
            cat = rally["cat"]
            cont = rally["cont"]
            rally_len = len(cat)
            prefix_len = min(rally_len, max_len)
            start = max(0, rally_len - max_len)

            self.samples.append({
                "cat": cat[start:],
                "cont": cont[start:],
                "prefix_len": prefix_len,
                "next_sn": int(rally["next_sn"]),
                "rally_uid": rally["rally_uid"],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad sequences to same length in batch."""
    max_len = max(s["prefix_len"] for s in batch)
    n_cat = batch[0]["cat"].shape[1]
    n_cont = batch[0]["cont"].shape[1]
    batch_size = len(batch)

    cat_padded = np.zeros((batch_size, max_len, n_cat), dtype=np.int64)
    cont_padded = np.zeros((batch_size, max_len, n_cont), dtype=np.float32)
    lengths = np.array([s["prefix_len"] for s in batch], dtype=np.int64)

    for i, s in enumerate(batch):
        L = s["prefix_len"]
        cat_padded[i, :L] = s["cat"]
        cont_padded[i, :L] = s["cont"]

    result = {
        "cat": torch.from_numpy(cat_padded),
        "cont": torch.from_numpy(cont_padded),
        "lengths": torch.from_numpy(lengths),
    }

    if "y_action" in batch[0]:
        result["y_action"] = torch.tensor([s["y_action"] for s in batch], dtype=torch.long)
        result["y_point"] = torch.tensor([s["y_point"] for s in batch], dtype=torch.long)
        result["y_sgp"] = torch.tensor([s["y_sgp"] for s in batch], dtype=torch.float32)
        result["next_sn"] = torch.tensor([s["next_sn"] for s in batch], dtype=torch.long)
    else:
        result["next_sn"] = torch.tensor([s["next_sn"] for s in batch], dtype=torch.long)

    return result


# ======================= MODEL ==============================================

class GRUSeqModel(nn.Module):
    """GRU-based sequence model with per-field embeddings and multi-head output."""

    def __init__(self, cat_fields, n_cont, emb_dim=16, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()

        # Per-field embeddings
        self.embeddings = nn.ModuleList()
        self.cat_sizes = []
        for name, n_classes in cat_fields:
            self.embeddings.append(nn.Embedding(n_classes + 1, emb_dim, padding_idx=0))
            # +1 for padding/unknown token
            self.cat_sizes.append(n_classes)

        total_emb = len(cat_fields) * emb_dim
        self.cont_proj = nn.Linear(n_cont, emb_dim)
        input_dim = total_emb + emb_dim

        # GRU encoder
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,  # causal: no future info
            dropout=dropout if n_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output heads
        self.head_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, N_ACTION),
        )
        self.head_point = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, N_POINT),
        )
        self.head_sgp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, cat, cont, lengths):
        """
        cat: (B, T, n_cat) int tensor
        cont: (B, T, n_cont) float tensor
        lengths: (B,) int tensor
        """
        # Embed categoricals
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            # Clamp to valid range
            field_vals = cat[:, :, i].clamp(0, self.cat_sizes[i])
            emb_list.append(emb_layer(field_vals))
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, T, total_emb)

        cont_emb = self.cont_proj(cont)  # (B, T, emb_dim)
        x = torch.cat([cat_emb, cont_emb], dim=-1)  # (B, T, input_dim)

        # Pack for variable-length GRU
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        # Take the last valid hidden state for each sequence
        batch_idx = torch.arange(len(lengths), device=lengths.device)
        last_hidden = gru_out[batch_idx, (lengths - 1).clamp(min=0)]  # (B, hidden_dim)

        last_hidden = self.layer_norm(self.dropout(last_hidden))

        # Output heads
        logits_action = self.head_action(last_hidden)  # (B, 19)
        logits_point = self.head_point(last_hidden)     # (B, 10)
        logits_sgp = self.head_sgp(last_hidden).squeeze(-1)  # (B,)

        return logits_action, logits_point, logits_sgp


# ======================= LOSS ===============================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ======================= DATA PREP ==========================================

def prepare_rallies(df, is_train=True):
    """Convert raw DataFrame to list of rally dicts for the dataset."""
    rallies = []
    n_cat = len(CAT_FIELDS)
    n_cont = len(CONT_FIELDS)

    for rally_uid, group in df.groupby("rally_uid", sort=False):
        group = group.sort_values("strikeNumber")

        # Build cat array.
        # Shift all values by +1 so that the real class 0 becomes 1.
        # This keeps padding_idx=0 (used in collate_fn) distinct from every
        # legitimate token, preventing class-0 embeddings from being frozen.
        cat = np.zeros((len(group), n_cat), dtype=np.int64)
        for i, (col, _) in enumerate(CAT_FIELDS):
            cat[:, i] = group[col].values.astype(int) + 1  # 0-pad safe shift

        # Build cont array
        cont = np.zeros((len(group), n_cont), dtype=np.float32)
        for i, col in enumerate(CONT_FIELDS):
            cont[:, i] = group[col].values.astype(float)

        # Normalize continuous features
        cont[:, 0] /= 11.0   # scoreSelf
        cont[:, 1] /= 11.0   # scoreOther
        cont[:, 2] /= 30.0   # strikeNumber
        cont[:, 3] /= 7.0    # numberGame
        cont[:, 4] /= 2.0    # sex

        rally_dict = {
            "cat": cat,
            "cont": cont,
            "rally_uid": int(rally_uid),
            "match": int(group["match"].iloc[0]),
        }

        if is_train:
            targets = []
            next_sns = []
            for t in range(1, len(group)):
                row = group.iloc[t]
                targets.append((int(row["actionId"]), int(row["pointId"]), int(row["serverGetPoint"])))
                next_sns.append(int(row["strikeNumber"]))
            rally_dict["targets"] = targets
            rally_dict["next_sn"] = next_sns
        else:
            # For test: predict next strike after the full prefix
            last_sn = int(group["strikeNumber"].iloc[-1])
            rally_dict["next_sn"] = last_sn + 1

        rallies.append(rally_dict)

    return rallies


# ======================= TRAINING ===========================================

def train_one_epoch(model, loader, optimizer, action_loss_fn, point_loss_fn, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        cat = batch["cat"].to(device)
        cont = batch["cont"].to(device)
        lengths = batch["lengths"].to(device)
        y_act = batch["y_action"].to(device)
        y_pt = batch["y_point"].to(device)
        y_sgp = batch["y_sgp"].to(device)

        optimizer.zero_grad()
        logits_act, logits_pt, logits_sgp = model(cat, cont, lengths)

        loss_act = action_loss_fn(logits_act, y_act)
        loss_pt = point_loss_fn(logits_pt, y_pt)
        loss_sgp = F.binary_cross_entropy_with_logits(logits_sgp, y_sgp)

        # Weighted multi-task loss (same as competition weights)
        loss = 0.4 * loss_act + 0.4 * loss_pt + 0.2 * loss_sgp

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_act_probs, all_pt_probs, all_sgp_probs = [], [], []
    all_y_act, all_y_pt, all_y_sgp, all_next_sn = [], [], [], []

    for batch in loader:
        cat = batch["cat"].to(device)
        cont = batch["cont"].to(device)
        lengths = batch["lengths"].to(device)

        logits_act, logits_pt, logits_sgp = model(cat, cont, lengths)

        all_act_probs.append(F.softmax(logits_act, dim=-1).cpu().numpy())
        all_pt_probs.append(F.softmax(logits_pt, dim=-1).cpu().numpy())
        all_sgp_probs.append(torch.sigmoid(logits_sgp).cpu().numpy())

        if "y_action" in batch:
            all_y_act.append(batch["y_action"].numpy())
            all_y_pt.append(batch["y_point"].numpy())
            all_y_sgp.append(batch["y_sgp"].numpy())
        all_next_sn.append(batch["next_sn"].numpy())

    act_probs = np.concatenate(all_act_probs)
    pt_probs = np.concatenate(all_pt_probs)
    sgp_probs = np.concatenate(all_sgp_probs)
    next_sn = np.concatenate(all_next_sn)

    if all_y_act:
        y_act = np.concatenate(all_y_act)
        y_pt = np.concatenate(all_y_pt)
        y_sgp = np.concatenate(all_y_sgp)
        return act_probs, pt_probs, sgp_probs, next_sn, y_act, y_pt, y_sgp
    return act_probs, pt_probs, sgp_probs, next_sn


def apply_action_rules(probs, next_sns):
    preds = probs.copy()
    for i in range(len(preds)):
        if next_sns[i] == 2:
            for a in SERVE_FORBIDDEN:
                if a < preds.shape[1]:
                    preds[i, a] = 0.0
        total = preds[i].sum()
        if total > 0:
            preds[i] /= total
        else:
            preds[i] = np.ones(preds.shape[1]) / preds.shape[1]
    return preds


# ======================= MAIN ===============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=30)
    args = parser.parse_args()

    is_smoke = args.smoke
    n_folds = 1 if is_smoke else args.folds
    n_epochs = 5 if is_smoke else args.epochs
    patience = 3 if is_smoke else 8

    t_start = time.time()
    print("=" * 70)
    print(f"V7 GRU SEQUENCE BASELINE {'(SMOKE)' if is_smoke else ''}")
    print(f"  - GRU({args.hidden}) x 2 layers, emb={args.emb_dim}")
    print(f"  - Multi-head: actionId + pointId + serverGetPoint")
    print(f"  - GroupKFold(match), {n_folds} folds, {n_epochs} epochs")
    print(f"  - Device: {DEVICE}")
    print("=" * 70)

    # --- Load & clean ---
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    train_df, test_df, player_map = clean_data(raw_train, raw_test)

    # --- Prepare rally data ---
    print("\nPreparing rally data...")
    t0 = time.time()
    train_rallies = prepare_rallies(train_df, is_train=True)
    test_rallies = prepare_rallies(test_df, is_train=False)
    print(f"  Train: {len(train_rallies)} rallies, Test: {len(test_rallies)} rallies ({time.time()-t0:.1f}s)")

    # Count samples
    total_train_samples = sum(len(r["targets"]) for r in train_rallies)
    print(f"  Total train samples: {total_train_samples}")

    # --- Class weights for focal loss ---
    all_actions = []
    all_points = []
    for r in train_rallies:
        for t in r["targets"]:
            all_actions.append(t[0])
            all_points.append(t[1])
    action_counts = np.bincount(all_actions, minlength=N_ACTION)
    point_counts = np.bincount(all_points, minlength=N_POINT)

    # Inverse frequency weights, capped
    action_weights = 1.0 / (action_counts + 1)
    action_weights = action_weights / action_weights.sum() * N_ACTION
    action_weights = np.clip(action_weights, 0.1, 10.0)
    point_weights = 1.0 / (point_counts + 1)
    point_weights = point_weights / point_weights.sum() * N_POINT
    point_weights = np.clip(point_weights, 0.1, 10.0)

    action_weights_t = torch.tensor(action_weights, dtype=torch.float32).to(DEVICE)
    point_weights_t = torch.tensor(point_weights, dtype=torch.float32).to(DEVICE)

    # --- GroupKFold ---
    rally_matches = np.array([r["match"] for r in train_rallies])
    rally_uids = np.array([r["rally_uid"] for r in train_rallies])

    gkf = GroupKFold(n_splits=max(n_folds, 2))
    all_splits = list(gkf.split(np.arange(len(train_rallies)), groups=rally_matches))
    if is_smoke:
        all_splits = all_splits[:1]

    # OOF storage (per-sample)
    # We need to map rally-level splits to sample-level OOF
    # Build sample -> rally index mapping
    sample_rally_idx = []
    sample_within_idx = []
    for ri, r in enumerate(train_rallies):
        for ti in range(len(r["targets"])):
            sample_rally_idx.append(ri)
            sample_within_idx.append(ti)
    sample_rally_idx = np.array(sample_rally_idx)

    n_total_samples = len(sample_rally_idx)
    oof_act = np.zeros((n_total_samples, N_ACTION))
    oof_pt = np.zeros((n_total_samples, N_POINT))
    oof_sgp = np.zeros(n_total_samples)
    oof_y_act = np.array([train_rallies[sample_rally_idx[i]]["targets"][sample_within_idx[i]][0]
                          for i in range(n_total_samples)])
    oof_y_pt = np.array([train_rallies[sample_rally_idx[i]]["targets"][sample_within_idx[i]][1]
                         for i in range(n_total_samples)])
    oof_y_sgp = np.array([train_rallies[sample_rally_idx[i]]["targets"][sample_within_idx[i]][2]
                          for i in range(n_total_samples)])
    oof_next_sn = np.array([train_rallies[sample_rally_idx[i]]["next_sn"][sample_within_idx[i]]
                            for i in range(n_total_samples)])

    # ========================================
    # CV LOOP
    # ========================================
    fold_best_epochs = []  # collect per-fold best epoch for final training schedule
    for fold, (tr_rally_idx, val_rally_idx) in enumerate(all_splits):
        t_fold = time.time()
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{len(all_splits)}")
        print(f"{'='*60}")

        tr_rallies_fold = [train_rallies[i] for i in tr_rally_idx]
        val_rallies_fold = [train_rallies[i] for i in val_rally_idx]

        tr_dataset = RallyDataset(tr_rallies_fold, max_len=args.max_len)
        val_dataset = RallyDataset(val_rallies_fold, max_len=args.max_len)

        print(f"  Train: {len(tr_dataset)} samples from {len(tr_rallies_fold)} rallies")
        print(f"  Val:   {len(val_dataset)} samples from {len(val_rallies_fold)} rallies")

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False,
                                collate_fn=collate_fn, num_workers=0, pin_memory=True)

        # Model
        model = GRUSeqModel(
            cat_fields=CAT_FIELDS,
            n_cont=len(CONT_FIELDS),
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden,
            n_layers=2,
            dropout=0.3,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model params: {n_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

        action_loss_fn = FocalLoss(gamma=2.0, weight=action_weights_t)
        point_loss_fn = FocalLoss(gamma=2.0, weight=point_weights_t)

        best_ov = -1
        best_epoch = 0
        best_state = None
        no_improve = 0

        for epoch in range(n_epochs):
            t_ep = time.time()
            train_loss = train_one_epoch(model, tr_loader, optimizer, action_loss_fn, point_loss_fn, DEVICE)
            scheduler.step()

            # Evaluate
            act_probs, pt_probs, sgp_probs, next_sn, y_act, y_pt, y_sgp = evaluate(model, val_loader, DEVICE)
            act_ruled = apply_action_rules(act_probs, next_sn)

            f1a = f1_score(y_act, np.argmax(act_ruled, axis=1),
                           labels=list(range(N_ACTION)), average="macro", zero_division=0)
            f1p = f1_score(y_pt, np.argmax(pt_probs, axis=1),
                           labels=list(range(N_POINT)), average="macro", zero_division=0)
            try:
                auc = roc_auc_score(y_sgp, sgp_probs)
            except ValueError:
                auc = 0.5
            ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    Ep{epoch+1:02d}: loss={train_loss:.4f} F1a={f1a:.4f} F1p={f1p:.4f} "
                  f"AUC={auc:.4f} OV={ov:.4f} lr={lr_now:.5f} ({time.time()-t_ep:.0f}s)")

            if ov > best_ov:
                best_ov = ov
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch+1} (best={best_epoch})")
                break

        # Load best model and get OOF predictions
        model.load_state_dict(best_state)
        act_probs, pt_probs, sgp_probs, next_sn, _, _, _ = evaluate(model, val_loader, DEVICE)

        # Map val predictions back to global OOF indices
        val_rally_set = set(val_rally_idx)
        val_sample_mask = np.array([sample_rally_idx[i] in val_rally_set for i in range(n_total_samples)])
        val_sample_indices = np.where(val_sample_mask)[0]

        if len(val_sample_indices) == len(act_probs):
            oof_act[val_sample_indices] = act_probs
            oof_pt[val_sample_indices] = pt_probs
            oof_sgp[val_sample_indices] = sgp_probs

        print(f"\n  Fold {fold+1} best: OV={best_ov:.4f} @ epoch {best_epoch} ({(time.time()-t_fold)/60:.1f} min)")
        fold_best_epochs.append(best_epoch)

        # Save best model
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(best_state, os.path.join(MODEL_DIR, f"gru_fold{fold}.pt"))

    # ========================================
    # OVERALL OOF
    # ========================================
    if is_smoke:
        eval_mask = np.where(np.array([sample_rally_idx[i] in set(all_splits[0][1])
                                        for i in range(n_total_samples)]))[0]
    else:
        eval_mask = np.arange(n_total_samples)

    print(f"\n{'='*60}")
    print(f"OOF RESULTS ({len(eval_mask)} samples)")
    print(f"{'='*60}")

    act_ruled = apply_action_rules(oof_act[eval_mask], oof_next_sn[eval_mask])
    f1a = f1_score(oof_y_act[eval_mask], np.argmax(act_ruled, axis=1),
                   labels=list(range(N_ACTION)), average="macro", zero_division=0)
    f1p = f1_score(oof_y_pt[eval_mask], np.argmax(oof_pt[eval_mask], axis=1),
                   labels=list(range(N_POINT)), average="macro", zero_division=0)
    try:
        auc = roc_auc_score(oof_y_sgp[eval_mask], oof_sgp[eval_mask])
    except ValueError:
        auc = 0.5
    ov = 0.4 * f1a + 0.4 * f1p + 0.2 * auc
    print(f"  GRU OOF: F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f}")

    # Save OOF
    np.savez(os.path.join(MODEL_DIR, "oof_v7_seq.npz"),
             act=oof_act, pt=oof_pt, sgp=oof_sgp,
             y_act=oof_y_act, y_pt=oof_y_pt, y_sgp=oof_y_sgp, next_sn=oof_next_sn)

    # ========================================
    # FINAL: Train on all data, predict test
    # ========================================
    print(f"\n{'='*60}")
    print("FINAL: Train on all data, predict test")
    print(f"{'='*60}")

    full_dataset = RallyDataset(train_rallies, max_len=args.max_len)
    test_dataset = TestRallyDataset(test_rallies, max_len=args.max_len)

    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False,
                             collate_fn=collate_fn, num_workers=0, pin_memory=True)

    model = GRUSeqModel(
        cat_fields=CAT_FIELDS,
        n_cont=len(CONT_FIELDS),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden,
        n_layers=2,
        dropout=0.3,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Use the mean best epoch across all CV folds so the schedule is not
    # skewed by whichever fold happened to run last.
    mean_best_epoch = int(round(float(np.mean(fold_best_epochs)))) if fold_best_epochs else n_epochs // 2
    final_epochs = max(mean_best_epoch + 2, n_epochs // 2) if not is_smoke else 5
    print(f"\n  Final training: {final_epochs} epochs (mean CV best={mean_best_epoch}, folds={fold_best_epochs})")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs, eta_min=1e-5)

    action_loss_fn = FocalLoss(gamma=2.0, weight=action_weights_t)
    point_loss_fn = FocalLoss(gamma=2.0, weight=point_weights_t)

    for epoch in range(final_epochs):
        t_ep = time.time()
        train_loss = train_one_epoch(model, full_loader, optimizer, action_loss_fn, point_loss_fn, DEVICE)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == final_epochs - 1:
            print(f"    Ep{epoch+1:02d}: loss={train_loss:.4f} ({time.time()-t_ep:.0f}s)")

    # Predict test
    act_probs, pt_probs, sgp_probs, next_sn = evaluate(model, test_loader, DEVICE)
    act_ruled = apply_action_rules(act_probs, next_sn)

    # Build submission
    test_rally_uids = [r["rally_uid"] for r in test_rallies]
    submission = pd.DataFrame({
        "rally_uid": test_rally_uids,
        "actionId": np.argmax(act_ruled, axis=1).astype(int),
        "pointId": np.argmax(pt_probs, axis=1).astype(int),
        "serverGetPoint": (sgp_probs >= 0.5).astype(int),
    })

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    suffix = "_smoke" if is_smoke else ""
    out = os.path.join(SUBMISSION_DIR, f"submission_v7_seq{suffix}.csv")
    submission.to_csv(out, index=False, lineterminator="\n", encoding="utf-8")
    print(f"\n  Saved: {out}")
    print(f"  actionId: {submission.actionId.value_counts().sort_index().to_dict()}")
    print(f"  pointId:  {submission.pointId.value_counts().sort_index().to_dict()}")
    print(f"  SGP:      {submission.serverGetPoint.value_counts().to_dict()}")
    print(f"\n  Total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    main()
