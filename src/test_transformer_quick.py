"""Quick 1-fold Transformer test with smaller model for CPU."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TRAIN_PATH
from data_cleaning import clean_data
from transformer_model import prepare_sequences, PingPongTransformer, PingPongDataset

import torch
import torch.nn.functional as F

N_ACTION, N_POINT = 19, 10

def macro_f1(y_true, y_probs, n_classes):
    y_pred = np.argmax(y_probs, axis=1)
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)

def main():
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(os.path.join(os.path.dirname(TRAIN_PATH), "test.csv"))
    train_df, _, player_map = clean_data(raw_train, raw_test)
    n_players = len(player_map)

    print("Building sequences...")
    samples = prepare_sequences(train_df, is_train=True)
    print(f"  {len(samples)} samples")

    rally_to_match = train_df.groupby("rally_uid")["match"].first()
    groups = np.array([rally_to_match.get(s["rally_uid"], 0) for s in samples])

    device = torch.device("cpu")
    print(f"Device: {device}")

    # Only 1 fold for quick test
    gkf = GroupKFold(n_splits=5)
    tr_idx, val_idx = next(iter(gkf.split(np.arange(len(samples)), groups=groups)))
    tr_samples = [samples[i] for i in tr_idx]
    val_samples = [samples[i] for i in val_idx]
    print(f"Train: {len(tr_samples)}, Val: {len(val_samples)}")

    max_seq = 50
    train_ds = PingPongDataset(tr_samples, max_seq_len=max_seq)
    val_ds = PingPongDataset(val_samples, max_seq_len=max_seq)
    # Smaller batch for CPU
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=512, num_workers=0)

    # Smaller model for CPU: d_model=64, nhead=4, n_layers=2
    model = PingPongTransformer(d_model=64, nhead=4, n_layers=2, n_players=n_players).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # Class weights
    act_counts = np.bincount([s["y_action"] for s in tr_samples], minlength=N_ACTION).astype(float)
    act_w = torch.FloatTensor((1.0/(act_counts+1)) / (1.0/(act_counts+1)).sum() * N_ACTION).to(device)
    pt_counts = np.bincount([s["y_point"] for s in tr_samples], minlength=N_POINT).astype(float)
    pt_w = torch.FloatTensor((1.0/(pt_counts+1)) / (1.0/(pt_counts+1)).sum() * N_POINT).to(device)

    best_score = -1
    patience, wait = 4, 0

    for epoch in range(15):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_batch = 0
        for batch in train_loader:
            al, pl, sl = model(
                batch["cat_seq"].to(device), batch["num_seq"].to(device),
                batch["context"].to(device), batch["player_ids"].to(device),
                batch["mask"].to(device))
            y_a = batch["y_action"].long().to(device)
            y_p = batch["y_point"].long().to(device)
            y_s = batch["y_server"].float().to(device)

            loss = 0.4*F.cross_entropy(al, y_a, weight=act_w) + \
                   0.4*F.cross_entropy(pl, y_p, weight=pt_w) + \
                   0.2*F.binary_cross_entropy_with_logits(sl, y_s)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        scheduler.step()

        # Eval
        model.eval()
        a_probs, p_probs, s_probs, ya, yp, ys = [], [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                al, pl, sl = model(
                    batch["cat_seq"].to(device), batch["num_seq"].to(device),
                    batch["context"].to(device), batch["player_ids"].to(device),
                    batch["mask"].to(device))
                a_probs.append(F.softmax(al, dim=-1).numpy())
                p_probs.append(F.softmax(pl, dim=-1).numpy())
                s_probs.append(torch.sigmoid(sl).numpy())
                ya.extend(batch["y_action"].numpy())
                yp.extend(batch["y_point"].numpy())
                ys.extend(batch["y_server"].numpy())

        a_probs = np.concatenate(a_probs)
        p_probs = np.concatenate(p_probs)
        s_probs = np.concatenate(s_probs)
        f1a = macro_f1(np.array(ya), a_probs, N_ACTION)
        f1p = macro_f1(np.array(yp), p_probs, N_POINT)
        auc = roc_auc_score(np.array(ys), s_probs)
        ov = 0.4*f1a + 0.4*f1p + 0.2*auc

        elapsed = time.time() - t0
        print(f"Ep {epoch+1:2d}: loss={total_loss/n_batch:.4f} F1a={f1a:.4f} F1p={f1p:.4f} AUC={auc:.4f} OV={ov:.4f} ({elapsed:.0f}s)")

        if ov > best_score:
            best_score = ov
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    print(f"\nBest Transformer OV (1 fold, d=64): {best_score:.4f}")
    print(f"\nFor reference - GBDT results from Fold 1:")
    print(f"  LightGBM:  OV=0.3103")
    print(f"  XGBoost:   OV=0.3141")
    print(f"  CatBoost:  OV=0.3223")


if __name__ == "__main__":
    main()
